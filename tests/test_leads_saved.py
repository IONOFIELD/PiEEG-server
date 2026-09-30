"""Choose leads is remembered between launches, per board layout."""
import json

from pieeg_server.acq_viewer import DEFAULT_ELECTRODES, MontageStore, ViewerModel
from pieeg_server.scope_console import _electrodes


def _model(path, device="ironbci32", n=32):
    return ViewerModel(n, 512, _electrodes(device, n),
                       store=MontageStore(path))


def test_selection_survives_a_relaunch(tmp_path):
    path = tmp_path / "montages.json"
    m = _model(path)
    off = m.electrodes[8:]                      # bundles 2-4 not wired
    m.set_wired(off, False)
    again = _model(path)
    assert again.unwired == set(off)
    assert json.loads(path.read_text())["leads"][m.leads_key] == off


def test_each_board_layout_keeps_its_own(tmp_path):
    path = tmp_path / "montages.json"
    _model(path).set_wired(_model(path).electrodes[8:], False)
    pieeg = ViewerModel(8, 250, DEFAULT_ELECTRODES, store=MontageStore(path))
    assert pieeg.unwired == set()


def test_switching_back_on_is_saved_too(tmp_path):
    path = tmp_path / "montages.json"
    m = _model(path)
    m.set_wired(m.electrodes[8:], False)
    m.set_wired(m.electrodes[8:16], True)
    assert _model(path).unwired == set(m.electrodes[16:])


def test_montages_and_filters_are_kept(tmp_path):
    path = tmp_path / "montages.json"
    store = MontageStore(path)
    store.put("Mine", [["Fp1", "F3", "x", True]], {"lff": "1 Hz"})
    m = ViewerModel(32, 512, _electrodes("ironbci32", 32),
                    store=MontageStore(path))
    m.set_wired(m.electrodes[:1], False)
    raw = json.loads(path.read_text())
    assert raw["montages"]["Mine"] and raw["filters"]["Mine"]


def test_no_store_means_no_saving():
    m = ViewerModel(8, 250, DEFAULT_ELECTRODES)
    m.set_wired(m.electrodes[:2], False)
    assert len(m.unwired) == 2
