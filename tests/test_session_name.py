"""Session names: the default "M-D-YY - NN" and typed-name cleaning."""
import asyncio
from datetime import datetime
from types import SimpleNamespace

import pytest

from pieeg_server.server import (PiEEGServer, clean_session_name,
                                 default_session_name)


DAY = datetime(2026, 9, 30, 14, 5)


def test_default_is_date_and_first_number(tmp_path):
    assert default_session_name(tmp_path, DAY) == "9-30-26 - 01"


def test_default_counts_up_from_the_days_highest(tmp_path):
    for n in ("9-30-26 - 01", "9-30-26 - 07", "9-29-26 - 12", "baseline",
              "9-30-26 - 03 extra"):
        (tmp_path / n).mkdir()
    assert default_session_name(tmp_path, DAY) == "9-30-26 - 08"


def test_default_zero_pads_and_handles_missing_dir(tmp_path):
    assert default_session_name(tmp_path / "nope", DAY) == "9-30-26 - 01"
    assert default_session_name(tmp_path, datetime(2027, 1, 2)) == "1-2-27 - 01"


@pytest.mark.parametrize("typed, clean", [
    ("  Smith  baseline ", "Smith baseline"),
    ("../../etc/passwd", "etcpasswd"),
    ("a/b\\c:d*e?f[g]", "abcdefg"),
    ("..hidden", "hidden"),
    ("tab\there", "tab here"),
    ("", None), ("   ", None), ("///", None), (None, None),
    ("9-30-26 - 01", "9-30-26 - 01"),
])
def test_clean_session_name(typed, clean):
    assert clean_session_name(typed) == clean


def test_clean_session_name_is_capped():
    assert len(clean_session_name("x" * 200)) == 64


def test_start_recording_refuses_a_used_name(tmp_path):
    (tmp_path / "Smith").mkdir()
    srv = SimpleNamespace(_recorder_task=None, _impedance_active=False,
                          _acq=SimpleNamespace(_hw=None),
                          _recordings_dir=tmp_path)
    with pytest.raises(ValueError, match="already exists"):
        asyncio.run(PiEEGServer._start_recording(srv, "Smith"))
