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
    assert json.loads(path.read_text()) == {"montages": {}, "filters": {},
                                            "leads": {}}


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


def test_editing_a_preset_moves_its_leads_into_custom(tmp_path):
    m = _model(tmp_path / "s.json")
    preset = m.current
    factory = [dict(r) for r in m.rows()]
    n = len(factory)
    a, b, c = m.electrodes[0], m.electrodes[1], m.electrodes[6]
    assert not m.set_row_pair(m.rows()[0], a, a)        # self-pair refused
    assert m.current == preset                          # nothing happened
    row = m.edit_row(m.rows()[0], a, c, "Fp1 test")
    assert m.current == av.CUSTOM_MONTAGE and m.forked_from == preset
    assert m.rows()[0] is row and row["pair"] == (a, c)
    assert row["label"] == "Fp1 test"
    assert m.sessions[preset] == factory                # the preset is untouched
    assert len(m.rows()) == n                           # all its leads, in order
    new = m.insert_row(1, b, c, label="EMG 1")
    assert m.current == av.CUSTOM_MONTAGE and m.rows()[1] is new
    assert m.insert_row(None, a, "nope") is None
    assert m.insert_row(99, b, a)["pair"] == (b, a)     # clamped to the end
    assert m.dirty()
    m.save_current()
    m2 = _model(tmp_path / "s.json")
    assert m2.current == av.DEFAULT_MONTAGE             # presets stay factory
    m2.load_montage(av.CUSTOM_MONTAGE)
    assert [r["pair"] for r in m2.rows()] == [r["pair"] for r in m.rows()]
    assert m2.rows()[1]["label"] == "EMG 1"


def test_each_kind_of_edit_moves_to_custom(tmp_path):
    for edit in (lambda m: m.toggle_row(0), lambda m: m.move_row(0, 2),
                 lambda m: m.remove_row(1),
                 lambda m: m.insert_row(0, m.electrodes[0], m.electrodes[3])):
        m = _model(tmp_path / "s.json")
        m.load_montage("Transverse")
        before = [dict(r) for r in m.rows()]
        edit(m)
        assert m.current == av.CUSTOM_MONTAGE
        assert m.sessions["Transverse"] == before


def test_edits_on_custom_stay_on_custom(tmp_path):
    m = _model(tmp_path / "s.json")
    m.toggle_row(0)
    m.forked_from = None
    m.move_row(0, 1)
    assert m.current == av.CUSTOM_MONTAGE and m.forked_from is None


def test_custom_takes_the_presets_filters(tmp_path):
    m = _model(tmp_path / "s.json")
    m.set_montage_filters("0.3 Hz", av.DEFAULT_HFF, "Off")
    m.toggle_row(0)
    assert m.montage_filters() == ("0.3 Hz", av.DEFAULT_HFF, "Off")


def test_choosing_leads_is_not_an_edit(tmp_path):
    m = _model(tmp_path / "s.json")
    m.set_wired(m.electrodes[:2], False)
    assert m.current == av.DEFAULT_MONTAGE and not m.dirty()
