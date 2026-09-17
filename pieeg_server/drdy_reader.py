"""
Timing-critical DRDY reader for the PiEEG-8, run as its own process.

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
"""

import os
import struct
import sys
import time

RECORD = struct.Struct("<QB27s")
FRAME, LATE, TORN, READY = 0, 1, 2, 3
BYTES_PER_READ = 27

# Linux GPIO chardev v1 (include/uapi/linux/gpio.h); hardware.py uses these too.
GPIO_GET_LINEEVENT = 0xC030B404          # _IOWR(0xB4, 0x04, 48)
GPIOHANDLE_REQUEST_INPUT = 1 << 0
GPIOEVENT_REQUEST_FALLING_EDGE = 1 << 1  # ADS1299 DRDY asserts LOW = data ready
EVENT_REQUEST_SIZE = 48                  # sizeof(struct gpioevent_request)
EVENT_DATA_SIZE = 16                     # u64 timestamp + u32 id

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


def main(argv):
    spi_fd, chip_fd, pin = (int(a) for a in argv[1:4])
    fs = float(argv[4])
    priority, ctrl_fd, out_fd = (int(a) for a in argv[5:8])
    realtime = make_realtime(priority)
    evt_fd = request_falling_edge_events(chip_fd, pin, b"pieeg_drdy_reader")
    try:
        return read_loop(evt_fd, spi_fd, fs, ctrl_fd, out_fd, realtime)
    finally:
        os.close(evt_fd)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
