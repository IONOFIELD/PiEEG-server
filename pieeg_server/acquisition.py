"""
Threaded data acquisition loop for PiEEG.

Reads EEG samples at 250 Hz from the hardware layer
and pushes timestamped frames into an asyncio-safe queue
for downstream consumers (WebSocket server, file writer, etc.).
"""

import asyncio
import logging
import os
import select
import subprocess
import sys
import threading
import time

from . import drdy_reader
from .decimate import Decimator
from .hardware import VREF_UV
from .spike_filter import HampelFilter

logger = logging.getLogger("pieeg.acquisition")

SAMPLE_RATE = 250  # Hz
SAMPLE_INTERVAL = 1.0 / SAMPLE_RATE  # 4 ms

# SCHED_FIFO priority requested for the SPI reading thread (0 disables).
# The operating system otherwise stalls the thread for 2-6 ms every few
# seconds, longer than the ~3.6 ms a sample stays readable at 250 SPS. Needs
# an rtprio limit: /etc/security/limits.d (login sessions) or LimitRTPRIO=
# (systemd). Without it the request fails quietly and nothing else changes.
RT_PRIORITY = int(os.environ.get("PIEEG_RT_PRIORITY", "50"))

# Read an 8-channel PiEEG from a separate process (drdy_reader) instead of a
# thread that competes for the GIL. PIEEG_READER_PROCESS=0 turns it off.
READER_PROCESS = os.environ.get("PIEEG_READER_PROCESS", "1") != "0"

# Number of frames to discard after a register config change.
# At 250 Hz, 25 frames = 100 ms — enough for SPI + ADC to settle.
_SETTLE_FRAMES = 25


class AcquisitionLoop:
    """Runs the SPI read loop in a background thread, feeds async queues."""

    def __init__(self, hardware, loop: asyncio.AbstractEventLoop,
                 mock: bool = False, ble: bool = False, serial: bool = False,
                 interrupt: bool = False):
        self._hw = hardware
        self._loop = loop
        self._mock = mock
        self._ble = ble
        self._serial = serial
        # interrupt=True -> block on a DRDY falling-edge event per sample
        # instead of busy-polling the DRDY level (lower CPU, no missed edges).
        self._interrupt = interrupt
        self._subscribers: list[asyncio.Queue] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._sample_count = 0
        self._settle_remaining = 0
        # Drop-detection instrumentation (populated by _run_hardware_interrupt).
        self._drdy_events = 0        # DRDY edges seen
        self._frames_read = 0        # frames actually decoded + enqueued
        self._dropped_frames = 0     # samples missed (inferred from timing gaps)
        self._gap_count = 0          # number of inter-edge gaps > 1.5x nominal
        self._first_event_ns = None
        self._last_event_ns = None
        self._max_interval_ns = 0    # largest gap between consecutive DRDY edges
        self._late_skips = 0         # edges skipped: too late to read cleanly
        self._torn_reads = 0         # reads discarded: next sample landed mid-read
        self._bad_frames = 0         # reads rejected by the hardware (sync/spike)
        self.realtime = False        # reading thread/process got SCHED_FIFO
        self._rt_warned = False
        self._reader_mode = None     # "process" or "thread" once running
        # PiEEG-16 chip 2 timing (reader process only; see _handle_record16)
        self._chip2_prev = None      # (chip 1 edge ns, chip 2 edge ns)
        self._last_emitted = None    # (sample, t, ts_ns): what a hold repeats
        self._chip2_repeats = 0      # chip 2 samples used for two frames
        self._chip2_skips = 0        # chip 2 samples never used
        self._chip2_skew_max_ns = 0  # largest |chip 2 - chip 1| edge time
        self._chip2_filled = 0       # chip 2 edges the kernel dropped
        self._chip2_rereads = 0      # chip 2 reads redone (updated mid-read)
        self._reader_announced = False
        self._nominal_ns = 1_000_000_000 / SAMPLE_RATE
        self._prev_edge_ns = None
        # Oversampling (hw.oversample > 1): chip samples in, decimated out.
        self._decimator: Decimator | None = None
        # Device-agnostic Hampel spike filter (runs in acquisition thread)
        self._hampel = HampelFilter(num_channels=hardware.num_channels)
        # Default both spike filters to OFF (user can enable via dashboard)
        self._hampel.enabled = False
        self._hw.spike_threshold = -1
        # Default subscriber for backward compat (.queue property)
        self._default_queue = self.subscribe()

    @property
    def num_channels(self) -> int:
        """Number of channels provided by the underlying hardware."""
        return self._hw.num_channels

    @property
    def pga_gain(self):
        """PGA gain read back from the hardware (None if it doesn't report one).

        Passed to the recorder so the sidecar calibration is derived from the
        gain actually programmed on the chip, never a hard-coded guess.
        """
        return getattr(self._hw, "pga_gain", None)

    @property
    def vref_uv(self) -> float:
        """ADC reference in µV (the ADS1299's 4.5 V unless the board says)."""
        return getattr(self._hw, "vref_uv", VREF_UV)

    @property
    def prefilter(self) -> str | None:
        """What the samples went through before they were handed on, for
        recording headers: the decimation FIR when oversampling, else None
        (raw chip output)."""
        d = self._decimator
        return d.describe() if d is not None else None

    @property
    def hampel(self) -> HampelFilter:
        """Access the Hampel spike filter for configuration."""
        return self._hampel

    def subscribe(self, maxsize: int = 2048) -> asyncio.Queue:
        """Create and return a new queue that receives every frame."""
        q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        """Remove a subscriber queue."""
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass

    @property
    def queue(self) -> asyncio.Queue:
        """Backward-compatible default queue."""
        return self._default_queue

    @property
    def sample_count(self) -> int:
        return self._sample_count

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="pieeg-acquisition", daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def restart_with_config(self, reg_map: dict[int, int]):
        """Stop acquisition, write registers, restart the thread.

        Drops ~10-20 ms of data during the transition (acceptable for config changes).
        After restart, the first SETTLE_FRAMES frames are discarded to let the
        SPI bus and ADC settle (avoids corrupted frames after RDATAC+START).
        """
        self.stop()
        self._hw.configure_registers(reg_map)
        self._hampel.reset()
        # settling is counted in chip frames: same time with oversampling
        self._settle_remaining = _SETTLE_FRAMES * getattr(self._hw,
                                                          "oversample", 1)
        self.start()

    def _make_realtime(self):
        """Put the calling (reading) thread on SCHED_FIFO if allowed.

        Only for the in-thread interrupt loop: the reader process asks for
        realtime priority itself.
        """
        if RT_PRIORITY <= 0 or not hasattr(os, "sched_setscheduler"):
            return
        try:
            # pid 0 = the calling thread on Linux.
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(RT_PRIORITY))
        except OSError as e:
            self.realtime = False
            if not self._rt_warned:
                self._rt_warned = True
                logger.info("realtime priority not available (%s); the reading "
                            "thread may be stalled now and then and skip a "
                            "sample", e.strerror or e)
            return
        if not self.realtime:
            logger.info("reading thread on SCHED_FIFO priority %d", RT_PRIORITY)
        self.realtime = True

    def _run(self):
        if self._mock:
            self._run_mock()
        elif self._ble:
            self._run_ble()
        elif self._serial:
            self._run_serial()
        elif self._interrupt:
            self._run_hardware_interrupt()
        elif getattr(self._hw, "oversample", 1) > 1:
            # only the DRDY-interrupt path decimates
            logger.error("oversampling needs interrupt acquisition; not "
                         "streaming (unset PIEEG_OVERSAMPLE)")
        else:
            self._run_hardware()

    def _run_mock(self):
        """Generate synthetic data for testing without hardware.

        Uses the hardware's advertised ``sample_rate`` if available, else
        falls back to the default 250 Hz. This lets the IronBCI-32 mock
        run at 500 Hz × 32 ch to faithfully reproduce its data rate.
        """
        interval = 1.0 / getattr(self._hw, "sample_rate", SAMPLE_RATE)
        while not self._stop_event.is_set():
            sample = self._hw.read_sample()
            sample = self._hampel.apply(sample)
            self._sample_count += 1
            frame = {
                "t": round(time.time(), 6),
                "n": self._sample_count,
                "channels": sample,
            }
            self._loop.call_soon_threadsafe(self._enqueue, frame)
            time.sleep(interval)

    def _run_hardware(self):
        """
        Tight loop: poll DRDY, read sample, push to async queue.

        The original PiEEG code uses a polling state machine:
        - Wait for DRDY pin to go HIGH (armed)
        - Wait for DRDY pin to go LOW (data ready)
        - Read SPI bytes

        We replicate the reference PiEEG script's tight busy-poll
        for lowest possible jitter at the cost of higher CPU.
        """
        armed = False

        while not self._stop_event.is_set():
            drdy = self._hw._drdy_get()

            # Arm: wait for DRDY to go high
            if not armed:
                if drdy == 1:
                    armed = True
                continue

            # Trigger: DRDY goes low → data ready
            if drdy != 0:
                continue

            armed = False
            sample = self._hw.read_sample()
            if sample is None:
                continue

            # Discard settling frames after register config change —
            # SPI bus often produces corrupted data right after RDATAC+START
            if self._settle_remaining > 0:
                self._settle_remaining -= 1
                continue

            sample = self._hampel.apply(sample)
            self._sample_count += 1
            timestamp = time.time()

            frame = {
                "t": round(timestamp, 6),
                "n": self._sample_count,
                "channels": sample,
            }

            # Non-blocking put into the asyncio queue from this thread
            self._loop.call_soon_threadsafe(self._enqueue, frame)

    def _run_hardware_interrupt(self):
        """Interrupt-driven acquisition: one frame per DRDY falling edge.

        Blocks on a GPIO edge event (no busy-poll). The kernel timestamps each
        edge, so any missed edge shows up as a larger-than-nominal interval and
        is counted as a dropped sample. Decoding + journaling are unchanged:
        the existing gain-aware decoder runs here, and frames go to the same
        subscriber queues.

        Reads must finish before the chip's next conversion overwrites its
        output. A late read can straddle that update and return a torn frame:
        on the bench that gave tens-of-mV single-sample spikes and all-zero
        frames. So: if newer edges are already queued, the older ones' data is
        gone — skip to the newest; don't start a read too close to the next
        conversion; and discard a read if the next edge landed while it ran.
        Skipped samples are counted as dropped, never passed on as data.

        On an 8-channel PiEEG the wait and the SPI read run in a separate
        process (drdy_reader), because inside a busy server this thread waits
        for the GIL long enough to skip ~0.3% of samples. Other boards, and a
        reader that can't start or dies, use the same loop in this thread.
        """
        # DRDY edges come at the chip's conversion rate, which is higher than
        # sample_rate when oversampling.
        fs = (getattr(self._hw, "chip_rate", None)
              or getattr(self._hw, "sample_rate", SAMPLE_RATE) or SAMPLE_RATE)
        self._nominal_ns = 1_000_000_000 / fs
        self._setup_decimator(fs)
        # None on the first edge of every run, including after
        # restart_with_config(): the pause for the register write is not an
        # interval, and not a drop.
        self._prev_edge_ns = None
        self._chip2_prev = None
        self._last_emitted = None           # no hold across a restart

        handles = self._reader_handles()
        if handles is not None:
            finished = True
            try:
                finished = self._run_reader_process(handles, fs)
            finally:
                if finished:
                    self._finish_streaming()
            if finished:
                return
        self._reader_mode = "thread"
        # Only this loop gets realtime priority: it sleeps between samples.
        # The polling loop never blocks and would hog a core.
        self._make_realtime()
        self._run_interrupt_thread(fs)

    def _account_edge(self, ts_ns):
        self._drdy_events += 1
        if self._first_event_ns is None:
            self._first_event_ns = ts_ns
        prev_ns = self._prev_edge_ns
        if prev_ns is not None:
            interval = ts_ns - prev_ns
            if interval > self._max_interval_ns:
                self._max_interval_ns = interval
            if interval > 1.5 * self._nominal_ns:
                # A DRDY edge is only truly MISSED when a full extra period
                # elapsed. round(interval/period)-1 gives the count; a
                # late-but-present edge (~1.5x) rounds to 0 via round(x-1), so
                # pure jitter is not miscounted as a drop.
                missed = round(interval / self._nominal_ns - 1.0)
                if missed > 0:
                    self._lost(missed)
                    self._gap_count += 1
                    logger.warning("DRDY gap: %.2f ms (~%d missed)",
                                   interval / 1e6, missed)
        self._last_event_ns = ts_ns
        self._prev_edge_ns = ts_ns

    def _setup_decimator(self, chip_rate):
        """A fresh (or reset) decimator for this run, or None without
        oversampling."""
        k = getattr(self._hw, "oversample", 1)
        if k <= 1:
            self._decimator = None
            return
        d = self._decimator
        if d is None or d.k != k or d.chip_rate != float(chip_rate):
            gain = getattr(self._hw, "pga_gain", None)
            self._decimator = Decimator(
                k, chip_rate, self._hw.num_channels,
                limit_uv=self.vref_uv / gain if gain else None)
            logger.info("oversampling: chip %d SPS, FIR %d taps, decimated "
                        "x%d to %d SPS (delay %.0f ms, taken off timestamps)",
                        chip_rate, len(self._decimator.taps), k,
                        chip_rate // k, self._decimator.delay_s * 1000)
        else:
            d.reset()

    def _lost(self, n):
        """n chip samples that never arrived (late, torn, rejected or a
        missed edge). Without oversampling they are simply gone; with it the
        decimator holds the last sample in their place, so the output stays
        on its 250 SPS time grid (counted in capture_stats "held_samples")."""
        self._dropped_frames += n
        if self._decimator is not None:
            for sample, t in self._decimator.hold(n):
                last = self._last_emitted
                ts_ns = (last[2] + round(self._nominal_ns * self._decimator.k)
                         if last is not None and last[2] is not None else None)
                self._emit(sample, t, ts_ns=ts_ns, held=True)
            return
        # Keep the time grid: a lost sample becomes a copy of the last one,
        # flagged "held" so recordings mark it (a dropped row would shift
        # everything after it by a sample period).
        last = self._last_emitted
        if last is None or self._settle_remaining > 0:
            return
        sample, t, ts_ns = last
        period_s = self._nominal_ns / 1e9
        for j in range(1, int(n) + 1):
            self._emit(sample, t + j * period_s,
                       ts_ns=(None if ts_ns is None
                              else ts_ns + round(j * self._nominal_ns)),
                       held=True)

    def _deliver(self, sample, t, ts_ns=None, t2_ns=None):
        """Pass one read on: count a rejected read, drop settling frames,
        otherwise decimate (when oversampling), filter, number, enqueue.
        ts_ns is the sample's DRDY edge (kernel CLOCK_MONOTONIC ns) and, on a
        PiEEG-16, t2_ns the edge of the chip 2 conversion in it."""
        if sample is None:
            self._bad_frames += 1
            self._lost(1)
            return
        # Discard settling frames after a register-config restart.
        if self._settle_remaining > 0:
            self._settle_remaining -= 1
            return
        self._frames_read += 1
        if self._decimator is not None:
            out = self._decimator.push(sample, t)
            if out is None:
                return
            # the output stands for the moment its FIR is centred on: the
            # edge that completed it, less the filter delay
            delay_ns = round((t - out[1]) * 1e9)
            sample, t = out
            ts_ns = None if ts_ns is None else ts_ns - delay_ns
            t2_ns = None
        self._emit(sample, t, ts_ns=ts_ns, t2_ns=t2_ns)

    def _emit(self, sample, t, ts_ns=None, t2_ns=None, held=False):
        sample = self._hampel.apply(sample)
        self._sample_count += 1
        frame = {
            "t": round(t, 6),
            "n": self._sample_count,
            "channels": sample,
        }
        # Timing for recordings (journal .timing file): the chip's own edge
        # times, and whether this row stands in for a lost sample.
        if ts_ns is not None:
            frame["ts_ns"] = ts_ns
        if t2_ns is not None:
            frame["t2_ns"] = t2_ns
        if held:
            frame["held"] = True
        # the next hold repeats this sample one period after this row
        self._last_emitted = (sample, t, ts_ns)
        self._loop.call_soon_threadsafe(self._enqueue, frame)

    def _finish_streaming(self):
        # Clean stop: halt streaming, then restore the DRDY level handle.
        try:
            self._hw.stop_streaming()
        finally:
            self._hw.disable_drdy_events()

    def _run_interrupt_thread(self, fs):
        """The DRDY wait and SPI read in this thread (see drdy_reader for the
        same loop in its own process)."""
        nominal_ns = 1_000_000_000 / fs
        # Latest start for a read: 0.4 ms before the next conversion. A read
        # (27 bytes at 2 MHz plus overhead) takes ~0.2 ms, and the torn-read
        # check below still discards any read the next edge lands in.
        read_deadline_ns = max(nominal_ns / 2, nominal_ns - 400_000)
        pending = None

        self._hw.enable_drdy_events()
        try:
            while not self._stop_event.is_set():
                if pending is not None:
                    ts_ns, pending = pending, None
                else:
                    ts_ns = self._hw.wait_drdy_event(timeout=0.5)
                    if ts_ns is None:
                        continue           # no edge yet — re-check the stop flag
                self._account_edge(ts_ns)

                # Newer edges already queued: this edge's data was overwritten.
                newer = self._hw.wait_drdy_event(timeout=0)
                while newer is not None:
                    self._lost(1)
                    self._late_skips += 1
                    self._account_edge(newer)
                    ts_ns = newer
                    newer = self._hw.wait_drdy_event(timeout=0)

                if time.monotonic_ns() - ts_ns > read_deadline_ns:
                    self._lost(1)
                    self._late_skips += 1
                    continue

                sample = self._hw.read_sample()
                read_end_ns = time.monotonic_ns()
                nxt = self._hw.wait_drdy_event(timeout=0)
                if nxt is not None:
                    pending = nxt
                    if nxt <= read_end_ns:
                        # The next conversion landed during the read.
                        self._lost(1)
                        self._torn_reads += 1
                        continue
                t2 = getattr(self._hw, "_drdy2_read_ns", 0) or None
                self._deliver(sample, time.time(), ts_ns=ts_ns,
                              t2_ns=t2 if self.num_channels == 16 else None)
        finally:
            self._finish_streaming()

    # --- separate reader process (8-channel PiEEG) ---

    def _reader_handles(self):
        if not READER_PROCESS:
            return None
        get = getattr(self._hw, "reader_handles", None)
        return get() if callable(get) else None

    def _spawn_reader(self, handles, fs):
        """Start drdy_reader; returns (process, control write fd, data read fd)."""
        chip_fd, pin, spi_fd = handles[:3]
        extra = tuple(handles[3:])          # PiEEG-16: spi2 fd, cs fd, pin2
        ctrl_r, ctrl_w = os.pipe()
        data_r, data_w = os.pipe()
        try:
            proc = subprocess.Popen(
                [sys.executable, "-I", "-S", drdy_reader.__file__,
                 str(spi_fd), str(chip_fd), str(pin), str(fs),
                 str(RT_PRIORITY), str(ctrl_r), str(data_w)]
                + [str(a) for a in extra],
                pass_fds=(spi_fd, chip_fd, ctrl_r, data_w) + extra[:2],
                stdin=subprocess.DEVNULL)
        except BaseException:
            for fd in (ctrl_r, ctrl_w, data_r, data_w):
                os.close(fd)
            raise
        os.close(ctrl_r)
        os.close(data_w)
        return proc, ctrl_w, data_r

    def _run_reader_process(self, handles, fs) -> bool:
        """Read records from drdy_reader until stop() is requested.

        Returns False (the caller falls back to the in-thread loop) when the
        reader can't start or exits on its own; True otherwise.
        """
        self._hw.release_drdy_level()       # the reader requests the event line
        try:
            proc, ctrl_w, data_r = self._spawn_reader(handles, fs)
        except OSError as e:
            logger.warning("DRDY reader process didn't start (%s); reading "
                           "in-thread", e)
            return False
        self._reader_mode = "process"
        if len(handles) > 3:                # PiEEG-16: chip 2 rides along
            record, handle = drdy_reader.RECORD16, self._handle_record16
        else:
            record, handle = drdy_reader.RECORD, self._handle_record
        size = record.size
        unpack = record.unpack_from
        buf = bytearray()
        exited = False
        try:
            while not self._stop_event.is_set():
                ready, _, _ = select.select([data_r], [], [], 0.2)
                if not ready:
                    if proc.poll() is not None:
                        exited = True
                        break
                    continue
                chunk = os.read(data_r, size * 256)
                if not chunk:
                    exited = True
                    break
                buf += chunk
                whole = len(buf) - len(buf) % size
                for off in range(0, whole, size):
                    handle(*unpack(buf, off))
                del buf[:whole]
        finally:
            os.close(ctrl_w)                # tells the reader to exit
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            os.close(data_r)
        if exited and not self._stop_event.is_set():
            logger.error("DRDY reader process exited (code %s); reading "
                         "in-thread", proc.returncode)
            return False
        return True

    def _handle_record(self, ts_ns, kind, raw):
        if kind == drdy_reader.READY:
            realtime = bool(raw[0])
            if realtime != self.realtime or not self._reader_announced:
                self._reader_announced = True
                if realtime:
                    logger.info("DRDY reader process on SCHED_FIFO priority %d",
                                RT_PRIORITY)
                else:
                    logger.info("DRDY reader process without realtime "
                                "priority (no rtprio limit); it may be "
                                "stalled now and then and skip a sample")
            self.realtime = realtime
            return
        self._account_edge(ts_ns)
        if kind == drdy_reader.LATE:
            self._lost(1)
            self._late_skips += 1
        elif kind == drdy_reader.TORN:
            self._lost(1)
            self._torn_reads += 1
        else:
            # Wall-clock time of the edge itself, not of this (later) decode.
            t = time.time() - (time.monotonic_ns() - ts_ns) / 1e9
            self._deliver(self._hw.decode_frame(list(raw)), t, ts_ns=ts_ns)

    def _handle_record16(self, ts_ns, ts2_ns, kind, flags, raw):
        """A PiEEG-16 record: the 8-ch bookkeeping, plus chip 2's timing.

        Chip 2 runs on its own clock; the reader pairs each chip 1 frame with
        the chip 2 conversion nearest in time. Between two delivered frames
        chip 2 should have advanced as many periods as chip 1 did; one less
        is a chip 2 sample used twice, one more a chip 2 sample skipped.
        """
        if kind == drdy_reader.READY:
            self._handle_record(ts_ns, kind, raw[:drdy_reader.BYTES_PER_READ])
            return
        if flags & drdy_reader.FILLED:
            self._chip2_filled += 1
        if flags & drdy_reader.REREAD:
            self._chip2_rereads += 1
        if kind != drdy_reader.FRAME:
            self._handle_record(ts_ns, kind, b"")
            return
        self._account_edge(ts_ns)
        skew = ts2_ns - ts_ns
        self._chip2_skew_max_ns = max(self._chip2_skew_max_ns, abs(skew))
        prev = self._chip2_prev
        if prev is not None:
            p = self._nominal_ns
            slip = (round((ts2_ns - prev[1]) / p)
                    - round((ts_ns - prev[0]) / p))
            if slip < 0:
                self._chip2_repeats -= slip
            else:
                self._chip2_skips += slip
        self._chip2_prev = (ts_ns, ts2_ns)
        t = time.time() - (time.monotonic_ns() - ts_ns) / 1e9
        n = drdy_reader.BYTES_PER_READ
        self._deliver(self._hw.decode_frame16(list(raw[:n]), list(raw[n:])),
                      t, ts_ns=ts_ns, t2_ns=ts2_ns)

    def capture_stats(self) -> dict:
        """Drop-detection summary for the interrupt loop.

        effective_rate_hz is derived from the kernel edge timestamps
        ((N-1) intervals over the measured span), independent of wall-clock.
        """
        span_s = 0.0
        if self._first_event_ns is not None and self._last_event_ns is not None:
            span_s = (self._last_event_ns - self._first_event_ns) / 1e9
        rate = (self._drdy_events - 1) / span_s if span_s > 0 else 0.0
        return {
            "drdy_events": self._drdy_events,
            "frames_read": self._frames_read,
            "dropped_frames": self._dropped_frames,
            "gap_count": self._gap_count,
            "span_seconds": round(span_s, 3),
            "effective_rate_hz": round(rate, 3),
            "max_interval_ms": round(self._max_interval_ns / 1e6, 3),
            "late_skips": self._late_skips,
            "torn_reads": self._torn_reads,
            "bad_frames": self._bad_frames,
            "reader": self._reader_mode,
            "realtime": self.realtime,
            "oversample": getattr(self._hw, "oversample", 1),
            "held_samples": (self._decimator.held
                             if self._decimator is not None else 0),
            **({"chip2_repeats": self._chip2_repeats,
                "chip2_skips": self._chip2_skips,
                "chip2_skew_max_ms": round(self._chip2_skew_max_ns / 1e6, 3),
                "chip2_filled_edges": self._chip2_filled,
                "chip2_rereads": self._chip2_rereads}
               if self._chip2_prev is not None else {}),
        }

    def _enqueue(self, frame: dict):
        for q in self._subscribers:
            try:
                q.put_nowait(frame)
            except asyncio.QueueFull:
                # Drop oldest frame to keep up with real-time
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    q.put_nowait(frame)
                except asyncio.QueueFull:
                    pass

    def _run_ble(self):
        """BLE acquisition: connect, then poll the notification buffer at 250 Hz.

        The IronBCIHardware receives data via BLE notification callbacks which
        fill an internal buffer. This loop drains that buffer at the sample rate
        and pushes frames into the async queues, matching the same timing
        contract as _run_hardware() and _run_mock().
        """
        import asyncio as _aio

        # Run scan_and_connect on the main event loop
        future = _aio.run_coroutine_threadsafe(
            self._hw.scan_and_connect(self._loop), self._loop
        )
        try:
            future.result(timeout=30.0)
        except Exception as e:
            import logging
            logging.getLogger("pieeg.acquisition").error(
                "BLE connection failed: %s", e
            )
            return

        interval = 1.0 / getattr(self._hw, "sample_rate", SAMPLE_RATE)
        next_t = time.monotonic()
        while not self._stop_event.is_set():
            sample = self._hw.read_sample()
            if sample is None:
                time.sleep(interval)
                next_t = time.monotonic()
                continue

            sample = self._hampel.apply(sample)
            self._sample_count += 1
            frame = {
                "t": round(time.time(), 6),
                "n": self._sample_count,
                "channels": sample,
            }
            self._loop.call_soon_threadsafe(self._enqueue, frame)

            next_t += interval
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -interval * 50:
                next_t = time.monotonic()

    def _run_serial(self):
        """Serial acquisition (IronBCI-32 / FreeEEG32-style USB-CDC boards).

        The hardware driver runs its own reader thread that decodes frames
        from the wire as fast as they arrive (~500 SPS) and stuffs samples
        into an internal deque. That driver thread is the real rate limiter;
        our job here is simply to forward whatever is queued downstream as
        promptly as possible.

        Why we don't pace per-sample:
          - On Windows, `time.sleep()` rounds up to the OS timer tick
            (~15.6 ms by default). A 2 ms-per-sample target is impossible
            to hit, and any cap on batch size that's smaller than what the
            wire delivers per sleep-tick (≈8 samples for 500 SPS) causes
            the deque to grow without bound, producing several seconds of
            latency in the dashboard.
          - The driver's deque already smooths bursts; downstream queues
            handle their own back-pressure.

        We therefore drain everything currently queued each iteration, then
        sleep one short tick (~5 ms) when we're caught up. That keeps the
        loop responsive to `stop_event` without throttling throughput.
        """
        sample_rate = getattr(self._hw, "sample_rate", SAMPLE_RATE)
        # Idle wait when the deque is empty. Short enough to keep visible
        # latency well under one frame (~16 ms on a 60 Hz display) but long
        # enough to avoid busy-spinning when the driver is between USB-CDC
        # chunks (which arrive every 8–16 ms).
        idle_sleep = min(0.005, 1.0 / sample_rate)
        while not self._stop_event.is_set():
            sample = self._hw.read_sample()
            if sample is None:
                time.sleep(idle_sleep)
                continue
            sample = self._hampel.apply(sample)
            self._sample_count += 1
            self._loop.call_soon_threadsafe(self._enqueue, {
                "t": round(time.time(), 6),
                "n": self._sample_count,
                "channels": sample,
            })
