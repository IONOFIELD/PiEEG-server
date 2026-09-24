"""
Timing-critical DRDY reader for the PiEEG-8 and PiEEG-16, run as its own
process.

Why a separate process: at 250 SPS a sample stays readable for only ~3.6 ms
after DRDY falls. Inside the server or the Scope, the reading thread has to
win back Python's GIL after every wake-up, and the server's own work held it
for 2-7 ms often enough to skip ~0.3% of samples even at realtime priority.
This process has its own GIL and does nothing but wait for the edge and read
27 bytes, so its only competitor is the OS scheduler, and SCHED_FIFO removes
that (0 skips in 2 x 60 s on the bench, 2026-09-16).

Standard library only, so `python -I -S drdy_reader.py ...` starts in ~40 ms.
The parent (acquisition.AcquisitionLoop) owns the SPI device and configures
the chip. This process gets the spidev and gpiochip file descriptors,
requests the DRDY falling-edge events itself (so no stale edges queue up
while it starts), and writes one fixed-size record per DRDY edge:

    RECORD = <Q edge timestamp, ns, CLOCK_MONOTONIC> <B kind> <27s raw frame>

kind:
    FRAME  raw holds the frame read after this edge
    LATE   edge skipped: a newer edge was already queued (this sample's data
           was overwritten), or too little time was left before the next one
    TORN   the read overlapped the next edge; its bytes are discarded
    READY  first record; raw[0] is 1 if SCHED_FIFO was granted

It exits when its control pipe closes (the parent stopped it, or died) or
when its output pipe breaks.

PiEEG-16 (read_loop16): the second ADS1299 runs on its own oscillator, so
its conversions slide against chip 1's (~0.4 ms/s on the bench, a full
period every ~10 s). Each chip 1 frame is paired with the chip 2 conversion
NEAREST in time to chip 1's edge, read as it is (no interpolation), so
channels 9-16 are within half a period (±2 ms at 250 SPS) of 1-8, and the
drift shows up as one chip 2 sample used twice (or skipped) per wrap. Each
record carries chip 2's own edge time, so the parent counts every repeat
and skip and knows the skew:

    RECORD16 = <Q chip 1 edge ns> <Q chip 2 edge ns> <B kind> <B flags>
               <54s chip 1 frame + chip 2 frame>

flags: FILLED  chip 2's edge event was lost by the kernel (it happens ~1 in
               200 while both chips are read); its time is put back one
               period after the previous one, chip 2's clock being steady
       REREAD  chip 2 updated during its read; it was read again at once
"""

import os
import struct
import sys
import time

RECORD = struct.Struct("<QB27s")
RECORD16 = struct.Struct("<QQBB54s")
FRAME, LATE, TORN, READY = 0, 1, 2, 3
FILLED, REREAD = 1, 2
BYTES_PER_READ = 27
# Don't start a chip 2 read this close to its next conversion (the read
# takes ~0.15 ms; a conversion landing in it tears the bytes).
CHIP2_MARGIN_NS = 300_000
# Hysteresis on the nearest-conversion choice: the pairing keeps stepping one
# chip 2 period per frame until the skew passes half a period by this much,
# so edge jitter at the boundary can't flip it (a repeat then a skip).
CHIP2_HYST_NS = 250_000

# Linux GPIO chardev v1 (include/uapi/linux/gpio.h); hardware.py uses these too.
GPIO_GET_LINEEVENT = 0xC030B404          # _IOWR(0xB4, 0x04, 48)
GPIOHANDLE_REQUEST_INPUT = 1 << 0
GPIOEVENT_REQUEST_FALLING_EDGE = 1 << 1  # ADS1299 DRDY asserts LOW = data ready
EVENT_REQUEST_SIZE = 48                  # sizeof(struct gpioevent_request)
EVENT_DATA_SIZE = 16                     # u64 timestamp + u32 id
GPIOHANDLE_SET_LINE_VALUES = 0xC040B409  # _IOWR(0xB4, 0x09, 64)
HANDLE_DATA_SIZE = 64                    # sizeof(struct gpiohandle_data)

_TIMESTAMP = struct.Struct("Q")


def request_falling_edge_events(chip_fd, pin, consumer=b"pieeg_evt"):
    """Request a GPIO line as a falling-edge event source; returns its fd.

    The fd becomes readable on each edge; reading EVENT_DATA_SIZE bytes gives
    one struct gpioevent_data (u64 kernel timestamp, u32 id).
    """
    import fcntl

    # struct gpioevent_request: lineoffset u32, handleflags u32, eventflags
    # u32, consumer_label char[32], fd i32 (filled in by the kernel).
    buf = bytearray(EVENT_REQUEST_SIZE)
    struct.pack_into("I", buf, 0, pin)
    struct.pack_into("I", buf, 4, GPIOHANDLE_REQUEST_INPUT)
    struct.pack_into("I", buf, 8, GPIOEVENT_REQUEST_FALLING_EDGE)
    label = consumer[:32]
    buf[12:12 + len(label)] = label
    fcntl.ioctl(chip_fd, GPIO_GET_LINEEVENT, buf)
    return struct.unpack_from("i", buf, 44)[0]


def make_realtime(priority):
    """SCHED_FIFO for this process; False if not permitted (no rtprio limit)."""
    if priority <= 0:
        return False
    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(priority))
    except OSError:
        return False
    return True


def read_loop(evt_fd, spi_fd, fs, ctrl_fd, out_fd, realtime=False):
    """Wait for DRDY edges and read frames until ctrl_fd closes.

    Mirrors AcquisitionLoop's in-thread interrupt loop, but only sends
    records; decoding, accounting and fan-out stay in the parent.
    """
    import select

    nominal_ns = 1_000_000_000 / fs
    # Latest start for a read: 0.4 ms before the next conversion. A read (27
    # bytes at 2 MHz plus overhead) takes ~0.2 ms; the torn-read check below
    # still discards any read the next edge lands in.
    read_deadline_ns = max(nominal_ns / 2, nominal_ns - 400_000)
    pack, write, read, now = RECORD.pack, os.write, os.read, time.monotonic_ns
    empty = bytes(BYTES_PER_READ)

    waiting = select.poll()
    waiting.register(evt_fd, select.POLLIN)
    waiting.register(ctrl_fd, select.POLLIN)
    queued = select.poll()
    queued.register(evt_fd, select.POLLIN)

    def edge():
        return _TIMESTAMP.unpack_from(read(evt_fd, EVENT_DATA_SIZE))[0]

    def queued_edge():
        return edge() if queued.poll(0) else None

    try:
        write(out_fd, pack(0, READY, bytes([1 if realtime else 0]) + empty[1:]))
        pending = None
        while True:
            if pending is None:
                ready = waiting.poll(1000)
                if any(fd == ctrl_fd for fd, _ in ready):
                    return 0                 # parent closed the control pipe
                if not ready:
                    continue
                ts = edge()
            else:
                ts, pending = pending, None

            newer = queued_edge()
            while newer is not None:         # this edge's data was overwritten
                write(out_fd, pack(ts, LATE, empty))
                ts, newer = newer, queued_edge()

            if now() - ts > read_deadline_ns:
                write(out_fd, pack(ts, LATE, empty))
                continue

            raw = read(spi_fd, BYTES_PER_READ)
            read_end = now()
            nxt = queued_edge()
            if nxt is not None:
                pending = nxt
                if nxt <= read_end:          # next conversion landed mid-read
                    write(out_fd, pack(ts, TORN, empty))
                    continue
            write(out_fd, pack(ts, FRAME, raw))
    except BrokenPipeError:
        return 0


def pick_chip2(t1, last2, now, period, prev=None):
    """Which chip 2 conversion goes with the chip 1 edge at t1.

    last2 is chip 2's newest known edge (<= now), filled in period by period
    if the kernel dropped some. prev is the (chip 1, chip 2) edge pair used
    last: the pairing carries on from it (one chip 2 period per chip 1
    period) and only moves by a period once the skew passes half a period
    plus CHIP2_HYST_NS, i.e. once per drift wrap. Returns (last2, filled,
    wait): wait is True when the conversion to use is the next one,
    last2 + period, or when that is due too soon to read last2 safely; the
    caller then waits for that edge and reads it.
    """
    filled = 0
    while now - last2 >= period:
        last2 += period
        filled += 1
    nxt = last2 + period
    if prev is None:
        target = t1
    else:
        target = prev[1] + round((t1 - prev[0]) / period) * period
        if target - t1 > period // 2 + CHIP2_HYST_NS:
            target -= period
        elif t1 - target > period // 2 + CHIP2_HYST_NS:
            target += period
    wait = (abs(nxt - target) < abs(target - last2)
            or nxt - now < CHIP2_MARGIN_NS)
    return last2, filled, wait


def read_loop16(evt_fd, evt2_fd, spi_fd, spi2_fd, cs_fd, fs, ctrl_fd, out_fd,
                realtime=False):
    """read_loop for the PiEEG-16: chip 1 on its DRDY edge as before, then
    the chip 2 sample nearest in time to that edge (see the module doc)."""
    import fcntl
    import select

    period = int(round(1_000_000_000 / fs))
    read_deadline_ns = max(period / 2, period - 400_000)
    pack, write, read, now = RECORD16.pack, os.write, os.read, time.monotonic_ns
    empty = bytes(2 * BYTES_PER_READ)
    cs_low, cs_high = bytearray(HANDLE_DATA_SIZE), bytearray(HANDLE_DATA_SIZE)
    cs_high[0] = 1

    waiting = select.poll()
    waiting.register(evt_fd, select.POLLIN)
    waiting.register(ctrl_fd, select.POLLIN)
    queued = select.poll()
    queued.register(evt_fd, select.POLLIN)
    queued2 = select.poll()
    queued2.register(evt2_fd, select.POLLIN)

    def edge(fd):
        return _TIMESTAMP.unpack_from(read(fd, EVENT_DATA_SIZE))[0]

    def queued_edge():
        return edge(evt_fd) if queued.poll(0) else None

    def edge2(timeout_ms=0):
        return edge(evt2_fd) if queued2.poll(timeout_ms) else None

    def read2():
        fcntl.ioctl(cs_fd, GPIOHANDLE_SET_LINE_VALUES, cs_low)
        try:
            return read(spi2_fd, BYTES_PER_READ)
        finally:
            fcntl.ioctl(cs_fd, GPIOHANDLE_SET_LINE_VALUES, cs_high)

    last2 = 0
    prev = None                              # (chip 1, chip 2) edges last paired
    # The chip hands a conversion out once: a second read of it returns
    # zeros. A repeat (chip 2 behind by a sample at the wrap) reuses these.
    got2, got2_raw = None, None
    try:
        write(out_fd, pack(0, 0, READY, 0,
                           bytes([1 if realtime else 0]) + empty[1:]))
        pending = None
        while True:
            if pending is None:
                ready = waiting.poll(1000)
                if any(fd == ctrl_fd for fd, _ in ready):
                    return 0                 # parent closed the control pipe
                if not ready:
                    continue
                ts = edge(evt_fd)
            else:
                ts, pending = pending, None

            newer = queued_edge()
            while newer is not None:         # this edge's data was overwritten
                write(out_fd, pack(ts, 0, LATE, 0, empty))
                ts, newer = newer, queued_edge()

            if now() - ts > read_deadline_ns:
                write(out_fd, pack(ts, 0, LATE, 0, empty))
                continue

            raw1 = read(spi_fd, BYTES_PER_READ)
            read1_end = now()

            # chip 2: newest edge seen, then the nearest conversion to ts
            t2 = edge2()
            while t2 is not None:
                last2, t2 = t2, edge2()
            if not last2:                    # first frame: wait for one
                last2 = edge2(int(period / 1e6) + 1) or now()
            flags = 0
            last2, filled, wait = pick_chip2(ts, last2, now(), period, prev)
            if filled:
                flags |= FILLED
            if wait:
                due_ms = (last2 + period - now()) / 1e6
                t2 = edge2(max(0, int(due_ms + 0.999)) + 1)
                if t2 is None:
                    last2 += period
                    flags |= FILLED
                else:
                    last2 = t2
            if got2 is not None and abs(last2 - got2) < period // 2:
                raw2 = got2_raw              # already read: same conversion
            else:
                start2 = now()
                raw2 = read2()
                t2 = edge2()
                if t2 is not None:
                    if t2 >= start2:         # chip 2 updated during the read
                        last2, raw2 = t2, read2()
                        flags |= REREAD
                    else:                    # a late event for what was read
                        last2 = t2
                got2, got2_raw = last2, raw2

            prev = (ts, last2)
            nxt = queued_edge()
            if nxt is not None:
                pending = nxt
                if nxt <= read1_end:         # chip 1 updated mid-read
                    write(out_fd, pack(ts, last2, TORN, flags, empty))
                    continue
            write(out_fd, pack(ts, last2, FRAME, flags, raw1 + raw2))
    except BrokenPipeError:
        return 0


def main(argv):
    spi_fd, chip_fd, pin = (int(a) for a in argv[1:4])
    fs = float(argv[4])
    priority, ctrl_fd, out_fd = (int(a) for a in argv[5:8])
    # PiEEG-16 adds: chip 2 spidev fd, chip-select line handle fd, DRDY2 pin
    extra = [int(a) for a in argv[8:11]]
    realtime = make_realtime(priority)
    evt_fd = request_falling_edge_events(chip_fd, pin, b"pieeg_drdy_reader")
    evt2_fd = -1
    try:
        if extra:
            spi2_fd, cs_fd, pin2 = extra
            evt2_fd = request_falling_edge_events(chip_fd, pin2,
                                                  b"pieeg_drdy2_reader")
            return read_loop16(evt_fd, evt2_fd, spi_fd, spi2_fd, cs_fd, fs,
                               ctrl_fd, out_fd, realtime)
        return read_loop(evt_fd, spi_fd, fs, ctrl_fd, out_fd, realtime)
    finally:
        os.close(evt_fd)
        if evt2_fd >= 0:
            os.close(evt2_fd)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
