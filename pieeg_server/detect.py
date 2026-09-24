"""Find which EEG board is attached, so one Scope launch works for any of them.

Looked for in this order:

1. IronBCI-32 on USB: a serial port (/dev/ttyACM*, /dev/ttyUSB*) that is
   actually streaming its 0xA0 … 0xC0 frames. The board streams as soon as
   it has USB power, so listening briefly is enough; nothing is written.
2. PiEEG shield on SPI: chip 1's ADS1299 ID register answering means a
   PiEEG-8; chip 2 answering too (spidev0.1, CS on GPIO19) means a PiEEG-16.
   GPIO19 and GPIO13 are unused on the 8-channel shield, so probing chip 2
   there is harmless. Only register reads (plus RESET/SDATAC) are sent.

IronBCI-8 (Bluetooth) is never guessed: a BLE scan is slow and could pick up
someone else's board nearby, so it stays an explicit --device ironbci8.
"""

from __future__ import annotations

import glob
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger("pieeg.detect")

SERIAL_LISTEN_S = 1.5       # IronBCI-32 frames arrive every ~2 ms
SERIAL_GLOBS = ("/dev/ttyACM*", "/dev/ttyUSB*")
ID_TRIES = 3                # RESET + re-read attempts per ADS1299

# Human-readable board names, shown in the Scope title and the logs.
BOARD_NAMES = {"pieeg8": "PiEEG-8", "pieeg16": "PiEEG-16",
               "ironbci8": "IronBCI-8", "ironbci32": "IronBCI-32"}


@dataclass
class Detection:
    device: str | None              # "pieeg8" / "pieeg16" / "ironbci32" / None
    serial_port: str | None = None
    tried: list[str] = field(default_factory=list)   # what was looked at

    @property
    def name(self) -> str:
        return BOARD_NAMES.get(self.device, "no board")


def find_ironbci32(ports=None, listen_s=SERIAL_LISTEN_S, tried=None):
    """The serial port an IronBCI-32 is streaming on, or None."""
    try:
        import serial  # pyserial
    except ImportError:
        if tried is not None:
            tried.append("USB serial: pyserial not installed")
        return None
    from .ironbci_32 import DEFAULT_BAUDRATE, _detect_frame_size

    if ports is None:
        ports = sorted(p for g in SERIAL_GLOBS for p in glob.glob(g))
    if not ports and tried is not None:
        tried.append("USB serial: no /dev/ttyACM* or /dev/ttyUSB* port")
    for port in ports:
        buf = bytearray()
        try:
            with serial.Serial(port, DEFAULT_BAUDRATE, timeout=0.1) as s:
                end = time.monotonic() + listen_s
                while time.monotonic() < end:
                    buf += s.read(4096)
                    if _detect_frame_size(bytes(buf)):
                        logger.info("IronBCI-32 frames on %s", port)
                        return port
        except (OSError, serial.SerialException) as e:
            if tried is not None:
                tried.append(f"USB serial {port}: {e}")
            continue
        if tried is not None:
            tried.append(f"USB serial {port}: {len(buf)} bytes, "
                         f"no IronBCI-32 frames")
    return None


def _ads1299_answers(hw, chip_num) -> bool:
    """True if this chip's ID register reads a valid ADS1299 value."""
    from .hardware import CMD_RESET, CMD_SDATAC, WHO_I_AM
    for _ in range(ID_TRIES):
        hw._send_command(chip_num, CMD_SDATAC)
        time.sleep(2e-3)
        dev_id = hw._rreg(chip_num, WHO_I_AM)
        if (dev_id & 0x1F) == 0x1E:
            logger.info("SPI chip %d ID 0x%02X", chip_num, dev_id)
            return True
        hw._send_command(chip_num, CMD_RESET)
        time.sleep(0.05)
    return False


def probe_pieeg(gpio_chip, profile, tried=None):
    """Channel count of the PiEEG shield on SPI (8 or 16), or None."""
    try:
        from .hardware import PiEEGHardware, spidev
    except ImportError as e:
        if tried is not None:
            tried.append(f"SPI: {e}")
        return None
    if spidev is None:
        if tried is not None:
            tried.append("SPI: spidev not installed")
        return None
    hw = PiEEGHardware(gpio_chip=gpio_chip, num_channels=16, profile=profile)
    try:
        hw._init_gpio()
        hw._init_spi()
        if not _ads1299_answers(hw, 1):
            if tried is not None:
                tried.append("SPI: no ADS1299 answered on chip 1")
            return None
        return 16 if _ads1299_answers(hw, 2) else 8
    except OSError as e:
        if tried is not None:
            tried.append(f"SPI/GPIO: {e}")
        return None
    finally:
        try:
            hw.close()
        except Exception:                   # noqa: BLE001 - best-effort
            pass


def detect(gpio_chip="/dev/gpiochip4", profile="pi5") -> Detection:
    """Which board is attached (see the module docstring for the order)."""
    tried: list[str] = []
    port = find_ironbci32(tried=tried)
    if port:
        return Detection("ironbci32", serial_port=port, tried=tried)
    n = probe_pieeg(gpio_chip, profile, tried=tried)
    if n:
        return Detection(f"pieeg{n}", tried=tried)
    return Detection(None, tried=tried)
