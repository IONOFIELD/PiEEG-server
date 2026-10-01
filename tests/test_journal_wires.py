"""Recording sidecars carry the lead-wire colours of both boards."""
import json

from pieeg_server.journal import JournalWriter
from pieeg_server.scope_console import _wires


class _Acq:
    num_channels = 32

    def subscribe(self, maxsize=0):
        return None


def test_sidecar_has_wire_colours(tmp_path):
    w = JournalWriter(_Acq(), tmp_path, session_name="s", num_channels=32,
                      sample_rate=512, wires=_wires("ironbci32", 32))
    w._write_sidecar()
    meta = json.loads((tmp_path / "s.json").read_text())
    assert meta["channel_wire_colours"][:8] == [
        "yellow", "orange", "red", "brown", "green", "blue", "purple", "grey"]
    assert meta["channel_wire_colours"][8:] == [None] * 24
    assert (meta["ref_wire_colour"], meta["bias_wire_colour"]) == (
        "white", "black")


def test_pieeg_wires_and_none_without():
    assert _wires("pieeg8", 8)["channels"] == [
        "grey", "purple", "blue", "green", "yellow", "orange", "red", "brown"]


def test_no_wires_no_fields(tmp_path):
    w = JournalWriter(_Acq(), tmp_path, session_name="s", num_channels=4,
                      sample_rate=250)
    w._write_sidecar()
    assert "channel_wire_colours" not in json.loads(
        (tmp_path / "s.json").read_text())
