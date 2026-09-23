"""Display filters are saved per montage, alongside its rows."""
import json

from pieeg_server import acq_viewer as av


def _model(path):
    return av.ViewerModel(8, 250, av.DEFAULT_ELECTRODES,
                          store=av.MontageStore(path))


def test_new_montage_uses_default_filters(tmp_path):
    m = _model(tmp_path / "s.json")
    assert m.montage_filters() == m.default_filters()
    assert not m.dirty()


def test_filter_change_marks_dirty_and_save_restores_it(tmp_path):
    path = tmp_path / "s.json"
    m = _model(path)
    name = m.current
    m.set_montage_filters("0.3 Hz", "35 Hz", "60 Hz")
    assert m.dirty()
    assert m.save_current()
    assert not m.dirty()
    data = json.loads(path.read_text())
    assert data["filters"][name] == {"lff": "0.3 Hz", "hff": "35 Hz",
                                     "notch": "60 Hz"}
    assert name not in data["montages"]          # rows untouched: factory
    m2 = _model(path)                            # next launch
    assert m2.current == name
    assert m2.montage_filters() == ("0.3 Hz", "35 Hz", "60 Hz")
    assert not m2.dirty()


def test_each_montage_keeps_its_own_filters(tmp_path):
    m = _model(tmp_path / "s.json")
    first = m.current
    other = next(n for n in av.MONTAGE_NAMES if n != first)
    m.set_montage_filters("0.3 Hz", "15 Hz", "Off")
    m.load_montage(other)
    assert m.montage_filters() == m.default_filters()
    m.load_montage(first)
    assert m.montage_filters() == ("0.3 Hz", "15 Hz", "Off")


def test_reset_restores_default_filters_and_saving_it_drops_the_entry(tmp_path):
    path = tmp_path / "s.json"
    m = _model(path)
    m.set_montage_filters("0.3 Hz", "35 Hz", "Off")
    m.save_current()
    m.reset_current_to_preset()
    assert m.montage_filters() == m.default_filters()
    assert m.dirty()
    m.save_current()
    assert json.loads(path.read_text()) == {"montages": {}, "filters": {}}


def test_old_store_without_filters_and_stale_labels_load(tmp_path):
    path = tmp_path / "s.json"
    name = av.DEFAULT_MONTAGE
    path.write_text(json.dumps({"montages": {}}))            # pre-3.7 file
    assert _model(path).montage_filters() == av.ViewerModel.default_filters()
    path.write_text(json.dumps({"montages": {}, "filters": {
        name: {"lff": "0.3 Hz", "hff": "999 Hz", "notch": "Off"}}}))
    lff, hff, notch = _model(path).montage_filters()
    assert (lff, hff, notch) == ("0.3 Hz", av.DEFAULT_HFF, "Off")


def test_rows_and_filters_saved_together(tmp_path):
    path = tmp_path / "s.json"
    m = _model(path)
    m.sessions[m.current][0]["on"] = False
    m.set_montage_filters("1 Hz", "70 Hz", "Off")
    m.save_current()
    m2 = _model(path)
    assert m2.sessions[m2.current][0]["on"] is False
    assert m2.montage_filters() == ("1 Hz", "70 Hz", "Off")


def test_set_row_pair_and_insert_row_edit_the_current_montage(tmp_path):
    m = _model(tmp_path / "s.json")
    name = m.current
    rows = m.rows()
    n = len(rows)
    a, b, c = m.electrodes[0], m.electrodes[1], m.electrodes[6]
    assert not m.set_row_pair(rows[0], a, a)            # self-pair refused
    assert m.set_row_pair(rows[0], a, c)
    assert rows[0]["pair"] == (a, c) and rows[0]["name"] == f"{a}-{c}"
    new = m.insert_row(1, b, c, label="EMG 1")
    assert m.current == name                            # stays on this montage
    assert m.rows()[1] is new and new["label"] == "EMG 1"
    assert len(m.rows()) == n + 1
    assert m.insert_row(None, a, "nope") is None
    assert m.insert_row(99, b, a)["pair"] == (b, a)     # clamped to the end
    assert m.rows()[-1]["pair"] == (b, a)
    assert m.dirty()
    m.save_current()
    m2 = _model(tmp_path / "s.json")
    assert [r["pair"] for r in m2.rows()] == [r["pair"] for r in m.rows()]
    assert m2.rows()[1]["label"] == "EMG 1"
