"""
Threaded data acquisition loop for PiEEG.

Reads EEG samples at 250 Hz from the hardware layer
and pushes timestamped frames into an asyncio-safe queue
for downstream consumers (WebSocket server, file writer, etc.).
"""

import asyncio
import logging
import threading
import time

from .spike_filter import HampelFilter

logger = logging.getLogger("pieeg.acquisition")

SAMPLE_RATE = 250  # Hz
SAMPLE_INTERVAL = 1.0 / SAMPLE_RATE  # 4 ms

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
        self._settle_remaining = _SETTLE_FRAMES
        self.start()

    def _run(self):
        if self._mock:
            self._run_mock()
        elif self._ble:
            self._run_ble()
        elif self._serial:
            self._run_serial()
        elif self._interrupt:
            self._run_hardware_interrupt()
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
        read_sample() uses the existing gain-aware decoder, and frames go to the
        same subscriber queues.

        Reads must finish before the chip's next conversion overwrites its
        output. When this thread wakes late (the Scope's Tk viewer shares the
        GIL), a read can straddle that update and return a torn frame: on the
        bench that gave tens-of-mV single-sample spikes and all-zero frames.
        So: if newer edges are already queued, the older ones' data is gone —
        skip to the newest; don't start a read too close to the next
        conversion; and discard a read if the next edge landed while it ran.
        Skipped samples are counted as dropped, never passed on as data.
        """
        fs = getattr(self._hw, "sample_rate", SAMPLE_RATE) or SAMPLE_RATE
        nominal_ns = 1_000_000_000 / fs
        gap_ns = 1.5 * nominal_ns          # interval beyond this = missed sample(s)
        # Latest start for a read: 0.4 ms before the next conversion. A read
        # (27 bytes at 2 MHz plus overhead) takes ~0.2 ms, and the torn-read
        # check below still discards any read the next edge lands in.
        read_deadline_ns = max(nominal_ns / 2, nominal_ns - 400_000)
        prev_ns = None
        pending = None

        def account(ts_ns):
            nonlocal prev_ns
            self._drdy_events += 1
            if self._first_event_ns is None:
                self._first_event_ns = ts_ns
            if prev_ns is not None:
                # prev_ns is None on the first edge of every run, including
                # after restart_with_config(): the deliberate pause for the
                # register write is not an interval, and not a drop.
                interval = ts_ns - prev_ns
                if interval > self._max_interval_ns:
                    self._max_interval_ns = interval
                if interval > gap_ns:
                    # A DRDY edge is only truly MISSED when a full extra
                    # period elapsed. round(interval/period)-1 gives the
                    # count; a late-but-present edge (~1.5x) rounds to 0 via
                    # round(x-1), so pure jitter is not miscounted as a drop.
                    missed = round(interval / nominal_ns - 1.0)
                    if missed > 0:
                        self._dropped_frames += missed
                        self._gap_count += 1
                        logger.warning("DRDY gap: %.2f ms (~%d missed)",
                                       interval / 1e6, missed)
            self._last_event_ns = ts_ns
            prev_ns = ts_ns

        self._hw.enable_drdy_events()
        try:
            while not self._stop_event.is_set():
                if pending is not None:
                    ts_ns, pending = pending, None
                else:
                    ts_ns = self._hw.wait_drdy_event(timeout=0.5)
                    if ts_ns is None:
                        continue           # no edge yet — re-check the stop flag
                account(ts_ns)

                # Newer edges already queued: this edge's data was overwritten.
                newer = self._hw.wait_drdy_event(timeout=0)
                while newer is not None:
                    self._dropped_frames += 1
                    self._late_skips += 1
                    account(newer)
                    ts_ns = newer
                    newer = self._hw.wait_drdy_event(timeout=0)

                if time.monotonic_ns() - ts_ns > read_deadline_ns:
                    self._dropped_frames += 1
                    self._late_skips += 1
                    continue

                sample = self._hw.read_sample()
                read_end_ns = time.monotonic_ns()
                nxt = self._hw.wait_drdy_event(timeout=0)
                if nxt is not None:
                    pending = nxt
                    if nxt <= read_end_ns:
                        # The next conversion landed during the read.
                        self._dropped_frames += 1
                        self._torn_reads += 1
                        continue
                if sample is None:
                    self._dropped_frames += 1
                    self._bad_frames += 1
                    continue
                # Discard settling frames after a register-config restart.
                if self._settle_remaining > 0:
                    self._settle_remaining -= 1
                    continue

                sample = self._hampel.apply(sample)
                self._sample_count += 1
                self._frames_read += 1
                frame = {
                    "t": round(time.time(), 6),
                    "n": self._sample_count,
                    "channels": sample,
                }
                self._loop.call_soon_threadsafe(self._enqueue, frame)
        finally:
            # Clean stop: halt streaming, then restore the DRDY level handle.
            try:
                self._hw.stop_streaming()
            finally:
                self._hw.disable_drdy_events()

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
