"""Lock in the physical calibration: sidecar lsb_uv is derived from gain.

Guards the fix where the recorder ran at gain x24 but stamped a gain-1/2^24
lsb_uv. Register gain and sidecar lsb_uv must stay tied together.
"""

import asyncio
import json

import pytest

from pieeg_server.journal import (JournalWriter, physical_lsb_uv,
                                   DECODE_LSB_UV, FULL_SCALE_23)


class _FakeAcq:
    num_channels = 8

    def subscribe(self, maxsize=2048):
        return asyncio.Queue(maxsize=maxsize)

    def unsubscribe(self, q):
        pass


def test_physical_lsb_uv_datasheet():
    # uV/count = Vref / (gain * (2^23 - 1))
    assert physical_lsb_uv(24) == pytest.approx(4.5e6 / (24 * (2**23 - 1)))
    assert physical_lsb_uv(1) == pytest.approx(4.5e6 / (2**23 - 1))
    # gain 24 is exactly 24x finer than gain 1.
    assert physical_lsb_uv(1) == pytest.approx(physical_lsb_uv(24) * 24)


def test_transport_scale_is_not_the_physical_scale():
    # The decode/transport scale (used to keep counts 1:1) must NOT equal the
    # physical calibration -- conflating them was the original bug.
    assert DECODE_LSB_UV != pytest.approx(physical_lsb_uv(24))


@pytest.mark.parametrize("gain", [1, 8, 24])
def test_sidecar_lsb_uv_derived_from_gain(tmp_path, gain):
    jw = JournalWriter(_FakeAcq(), out_dir=tmp_path, session_name=f"g{gain}",
                       num_channels=8, gain=gain)
    jw._start_time = 1751662800.0
    jw._write_sidecar()
    side = json.load(open(jw.sidecar_path))
    # gain and lsb_uv come from the same source and stay consistent.
    assert side["gain"] == gain
    assert side["lsb_uv"] == pytest.approx(physical_lsb_uv(gain))
    assert side["full_scale"] == FULL_SCALE_23
    assert side["physical_dimension"] == "uV"


def test_gain24_lsb_is_about_0p0224(tmp_path):
    jw = JournalWriter(_FakeAcq(), out_dir=tmp_path, session_name="g24",
                       num_channels=8, gain=24)
    assert jw._lsb_uv == pytest.approx(0.02235, abs=1e-4)
