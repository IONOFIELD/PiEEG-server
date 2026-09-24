"""Board auto-detection (pieeg_server.detect) and the IronBCI-32 recording
scale — no hardware: the serial port and the SPI chips are faked."""

import asyncio
import json

import pytest

from pieeg_server import detect
from pieeg_server.ironbci_32 import (DEFAULT_FRAME_BYTES, END_BYTE,
                                     IronBCI32Hardware, SCALE_UV, START_BYTE)
from pieeg_server.journal import JournalWriter


def _frames(n=10):
    body = bytes(DEFAULT_FRAME_BYTES - 2)
    return b"".join(bytes([START_BYTE]) + body + bytes([END_BYTE])
                    for _ in range(n))


class _FakeSerial:
    streams = {}

    def __init__(self, port, baud, timeout=None):
        if port not in self.streams:
            raise OSError(f"could not open {port}")
        self._data = self.streams[port]

    def read(self, n):
        chunk, self._data = self._data[:n], self._data[n:]
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def fake_serial(monkeypatch):
    serial = pytest.importorskip("serial")
    monkeypatch.setattr(serial, "Serial", _FakeSerial)
    _FakeSerial.streams = {}
    return _FakeSerial.streams


def test_ironbci32_found_on_the_port_streaming_frames(fake_serial):
    fake_serial["/dev/ttyACM0"] = b"\x00\x11 some other device \x22" * 20
    fake_serial["/dev/ttyACM1"] = _frames()
    tried = []
    port = detect.find_ironbci32(ports=["/dev/ttyACM0", "/dev/ttyACM1"],
                                 listen_s=0.2, tried=tried)
    assert port == "/dev/ttyACM1"
    assert any("ttyACM0" in t and "no IronBCI-32 frames" in t for t in tried)


def test_no_serial_port_means_no_ironbci32(fake_serial):
    tried = []
    assert detect.find_ironbci32(ports=[], listen_s=0.1, tried=tried) is None
    assert tried and "no /dev/ttyACM*" in tried[0]


class _FakeChips:
    """PiEEGHardware stand-in: `ids` is what each chip's ID register reads."""
    ids = {1: 0x3E, 2: 0x3E}
    closed = False

    def __init__(self, gpio_chip=None, num_channels=16, profile=None):
        pass

    def _init_gpio(self):
        pass

    def _init_spi(self):
        pass

    def _send_command(self, chip_num, command):
        pass

    def _rreg(self, chip_num, register):
        return self.ids[chip_num]

    def close(self):
        type(self).closed = True


@pytest.fixture
def fake_spi(monkeypatch):
    from pieeg_server import hardware
    monkeypatch.setattr(hardware, "PiEEGHardware", _FakeChips)
    monkeypatch.setattr(hardware, "spidev", object())
    monkeypatch.setattr(detect.time, "sleep", lambda s: None)
    _FakeChips.closed = False
    return _FakeChips


@pytest.mark.parametrize("ids, expect", [
    ({1: 0x3E, 2: 0x3E}, 16),      # both ADS1299s answer: PiEEG-16
    ({1: 0x3E, 2: 0xFF}, 8),       # chip 2 absent (floating MISO): PiEEG-8
    ({1: 0x3E, 2: 0x00}, 8),
    ({1: 0x00, 2: 0x00}, None),    # no shield
])
def test_pieeg_channel_count_from_id_registers(fake_spi, ids, expect):
    fake_spi.ids = ids
    assert detect.probe_pieeg("/dev/gpiochip4", "pi5") == expect
    assert fake_spi.closed                  # SPI/GPIO released for the Scope


def test_detect_prefers_a_streaming_ironbci32(monkeypatch):
    monkeypatch.setattr(detect, "find_ironbci32",
                        lambda tried=None: "/dev/ttyACM0")
    monkeypatch.setattr(detect, "probe_pieeg",
                        lambda *a, **k: pytest.fail("SPI probed"))
    found = detect.detect()
    assert (found.device, found.serial_port) == ("ironbci32", "/dev/ttyACM0")
    assert found.name == "IronBCI-32"


def test_detect_falls_back_to_the_pieeg_then_nothing(monkeypatch):
    monkeypatch.setattr(detect, "find_ironbci32", lambda tried=None: None)
    monkeypatch.setattr(detect, "probe_pieeg", lambda *a, **k: 16)
    assert detect.detect().device == "pieeg16"
    monkeypatch.setattr(detect, "probe_pieeg", lambda *a, **k: None)
    assert detect.detect().device is None


def test_scope_says_so_when_no_board_is_found(monkeypatch):
    from pieeg_server import scope_console
    monkeypatch.setattr(detect, "detect", lambda *a, **k: detect.Detection(
        None, tried=["USB serial: nothing", "SPI: nothing"]))
    shown = {}
    monkeypatch.setattr(scope_console, "_startup_error",
                        lambda args, head, detail: shown.update(
                            head=head, detail=detail) or 1)
    assert scope_console.main([]) == 1
    assert shown["head"] == "No EEG board found"
    assert "SPI: nothing" in shown["detail"]


class _FakeAcq:
    num_channels = 32

    def subscribe(self, maxsize=2048):
        return asyncio.Queue(maxsize=maxsize)

    def unsubscribe(self, q):
        pass


def test_ironbci32_recording_uses_its_own_scale(tmp_path):
    hw = IronBCI32Hardware.__new__(IronBCI32Hardware)
    jw = JournalWriter(_FakeAcq(), out_dir=tmp_path, session_name="ib",
                       num_channels=32, sample_rate=500, gain=hw.pga_gain,
                       vref_uv=hw.vref_uv, reference="as wired")
    jw._start_time = 1751662800.0
    jw._write_sidecar()
    side = json.load(open(jw.sidecar_path))
    # one journal count == one AD7771 count, so decoded µV round-trip
    assert side["lsb_uv"] == pytest.approx(SCALE_UV)
    assert side["vref_uv"] == 2.5e6 and side["gain"] == 8
    assert side["reference"] == "as wired"
    # BDF range ±Vref/gain = ±312.5 mV, not the ADS1299's ±187.5 mV
    assert (2**23) * side["lsb_uv"] == pytest.approx(312500, rel=1e-5)
