"""
CSV recorder for PiEEG EEG data.

Subscribes to the acquisition loop and writes every frame to a CSV file.
Can be used standalone (``pieeg-server record``) or alongside the
WebSocket server (``pieeg-server --record data.csv``).

CSV format (matches the official PiEEG Save_data format):
    timestamp, ch1, ch2, ..., chN
"""

import asyncio
import csv
import logging
import os
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("pieeg.recorder")


def _make_csv_header(num_channels: int = 16) -> list[str]:
    return ["timestamp", *(f"ch{i}" for i in range(1, num_channels + 1))]


class Recorder:
    """Async consumer that writes EEG frames to a CSV file."""

    def __init__(self, acquisition, output: str | Path,
                 duration: float | None = None,
                 num_channels: int = 16):
        """
        Parameters
        ----------
        acquisition : AcquisitionLoop
        output : path to the CSV file (created or overwritten)
        duration : optional recording duration in seconds (None = until stopped)
        """
        self._acq = acquisition
        # raw chip samples, like the journal (see AcquisitionLoop.subscribe_raw)
        subscribe = getattr(acquisition, "subscribe_raw", acquisition.subscribe)
        self._queue = subscribe(maxsize=4096)
        self._output = Path(output)
        self._duration = duration
        self._num_channels = num_channels
        self._frames_written = 0
        self._start_time: float | None = None

    @property
    def frames_written(self) -> int:
        return self._frames_written

    async def run(self):
        """Record until duration expires or the task is cancelled."""
        self._output.parent.mkdir(parents=True, exist_ok=True)

        with open(self._output, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(_make_csv_header(self._num_channels))

            self._start_time = time.time()
            logger.info("Recording to %s%s",
                        self._output,
                        f" for {self._duration}s" if self._duration else "")

            try:
                while True:
                    # Check duration limit
                    if (self._duration is not None and
                            time.time() - self._start_time >= self._duration):
                        break

                    frame = await self._queue.get()
                    channels = frame.get("channels", [])
                    writer.writerow([frame["t"], *channels])
                    self._frames_written += 1

                    # Flush every 250 frames (~1 second at 250 Hz)
                    if self._frames_written % 250 == 0:
                        fh.flush()

            finally:
                fh.flush()
                self._acq.unsubscribe(self._queue)
                logger.info(
                    "Recording complete: %d frames → %s",
                    self._frames_written, self._output,
                )
