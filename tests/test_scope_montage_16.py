"""The montages follow the board: Adaptive (the default) is the double
banana at the board's size, the other presets use every 10-20 site the board
has, and saved edits stay with their board size."""
import json

from pieeg_server import acq_viewer as av
from pieeg_server.scope_console import _electrodes


def _model(n, path):
    device = {8: "pieeg8", 16: "pieeg16", 32: "ironbci32"}[n]
    return av.ViewerModel(n, 250, _electrodes(device, n),
                          store=av.MontageStore(path))


def _inputs(m):
    return {m.site_index[s] + 1 for r in m.rows() for s in r["pair"]}


def test_adaptive_is_the_default_double_banana_at_board_size(tmp_path):
    for n, rows in ((8, 8), (16, 16), (32, 18)):
        m = _model(n, tmp_path / "s.json")
        assert m.current == av.ADAPTIVE_MONTAGE
        adaptive = [r["pair"] for r in m.rows()]
        assert len(adaptive) == rows
        m.load_montage("Double banana")
        assert [r["pair"] for r in m.rows()] == adaptive


def test_presets_use_every_input_of_a_16ch_board(tmp_path):
    m = _model(16, tmp_path / "s.json")
    for name in ("Double banana", "Transverse"):
        m.load_montage(name)
        assert [r["pair"] for r in m.rows()] == av.MONTAGE_PRESETS_16[name]
        assert _inputs(m) == set(range(1, 17))
    m.load_montage("Circumferential")
    assert len(m.rows()) == 10          # the ring: Fp, F7/8, T, O sites


def test_8ch_board_keeps_the_8_site_presets(tmp_path):
    m = _model(8, tmp_path / "s.json")
    for name, pairs in av.MONTAGE_PRESETS.items():
        m.load_montage(name)
        assert [r["pair"] for r in m.rows()] == pairs


def test_saved_8ch_rows_do_not_replace_the_16ch_montage(tmp_path):
    path = tmp_path / "s.json"
    m8 = _model(8, path)
    m8.rows()[0]["on"] = False
    m8.set_montage_filters("1 Hz", "35 Hz", "60 Hz")
    assert m8.save_current()

    m16 = _model(16, path)
    assert len(m16.rows()) == 16 and all(r["on"] for r in m16.rows())
    # filters are shared by montage name
    assert m16.montage_filters() == ("1 Hz", "35 Hz", "60 Hz")

    m16.rows()[3]["on"] = False
    assert m16.save_current()
    data = json.loads(path.read_text())["montages"]
    assert set(data) == {"Adaptive", "16ch/Adaptive"}
    assert len(_model(8, path).rows()) == 8
    assert not _model(16, path).rows()[3]["on"]


def test_ironbci32_sites_follow_its_electrode_map():
    sites = _electrodes("ironbci32", 32)
    assert len(sites) == len(set(sites)) == 32
    # spot checks against the board's electrode location drawing
    assert (sites[0], sites[11], sites[14], sites[24], sites[31]) == (
        "F7", "Fp1", "Cz", "Fp2", "E32")
    tenty = [x for x in sites if not x.startswith("E")]
    assert len(tenty) == 21                 # all 19 + Fpz, Oz
    assert {"T3", "T4", "T5", "T6"} <= set(tenty)
    # the rest go by their input number
    assert all(x == f"E{i + 1}" for i, x in enumerate(sites)
               if x.startswith("E"))


def test_ironbci32_gets_the_full_acns_presets(tmp_path):
    m = _model(32, tmp_path / "s.json")
    for name, pairs in av.MONTAGE_PRESETS_32.items():
        m.load_montage(name)
        assert [r["pair"] for r in m.rows()] == pairs
    m.load_montage("Double banana")
    assert m.rows()[-2]["pair"] == ("Fz", "Cz")      # midline chain last
    assert m.rows_prefix == "32ch/"


def test_ironbci32_recording_labels_are_its_sites():
    from pieeg_server.journal import referential_labels
    labels = referential_labels(_electrodes("ironbci32", 32))
    assert labels[0] == "EEG F7-REF" and labels[-1] == "EEG E32-REF"


def test_sides_of_the_head():
    assert [av.site_side(x) for x in ("Fp1", "T4", "Cz", "Fpz", "Oz",
                                      "E17")] == ["left", "right", "mid",
                                                  "mid", "mid", None]
    assert av.row_side(("Fp1", "F7")) == "left"
    assert av.row_side(("Fz", "Cz")) == "mid"
    assert av.row_side(("Fp1", "Fp2")) is None


def test_show_only_one_side_on_the_ironbci(tmp_path):
    m = _model(32, tmp_path / "s.json")
    m.show_only("left")
    shown = [r["pair"] for r in m.rows() if r["on"]]
    assert len(shown) == 8 and all(av.row_side(p) == "left" for p in shown)
    assert m.dirty()
    m.show_only("mid")
    assert [r["pair"] for r in m.rows() if r["on"]] == [("Fz", "Cz"),
                                                          ("Cz", "Pz")]
    m.show_only("all")
    assert all(r["on"] for r in m.rows())


def test_unwired_electrodes_leave_every_montage(tmp_path):
    m = _model(32, tmp_path / "s.json")
    wired = {"Fp1", "F3", "C3", "P3", "O1", "Fz", "Cz", "Pz"}
    m.set_wired(m.electrodes, False)
    m.set_wired(wired, True)
    assert [r["pair"] for r in m.visible_rows()] == [
        ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
        ("Fz", "Cz"), ("Cz", "Pz")]
    assert not m.dirty()                  # a session choice, not an edit
    assert set(m.montage_inputs()) == {m.site_index[s] + 1 for s in
                                       wired - {"Fz"} | {"Fz"}}
    m.load_montage("Transverse")
    assert {p for r in m.visible_rows() for p in r["pair"]} <= wired
    m.set_wired(m.electrodes, True)
    assert len(m.visible_rows()) == len(m.rows())


def test_input_text_like_the_pieeg(tmp_path):
    m = _model(32, tmp_path / "s.json")
    assert m.input_text("F7") == "E1 F7" and m.input_text("E2") == "E2"
    m8 = _model(8, tmp_path / "s8.json")
    assert m8.input_text("Fp1") == "E1 Fp1"


def _dual(path):
    pg = [f"X{i}" for i in range(1, 9)]
    return av.ViewerModel(
        40, 512, _electrodes("ironbci32", 32) + pg,
        store=av.MontageStore(path),
        input_labels={k: f"E{i}" for i, k in enumerate(pg, start=1)},
        boards=[("IronBCI-32", _electrodes("ironbci32", 32)),
                ("PiEEG-8", pg)],
        extra_rows=[("X1", "X2", "EKG"), ("X3", "X4", "EMG 1"),
                    ("X5", "X6", "EMG 2"), ("X7", "X8", "EMG 3")])


def test_two_boards_eeg_on_top_then_ekg_then_emg(tmp_path):
    m = _dual(tmp_path / "s.json")
    labels = [m.row_label(r) for r in m.rows()]
    assert labels[:18] == [f"{a}-{b}" for a, b in
                           av.MONTAGE_PRESETS_32["Double banana"]]
    assert labels[18:] == ["EKG", "EMG 1", "EMG 2", "EMG 3"]
    assert [av.row_colour(x) for x in labels[17:20]] == [
        av.GEIST["trace_eeg"], av.GEIST["trace_ekg"], av.GEIST["trace_emg"]]
    # the PiEEG's inputs keep their own E-numbers
    assert m.elabel("X1") == "E1" and m.input_text("X8") == "E8"
    assert m.epair_name(("X1", "X2")) == "E1-E2"
    assert m.elabel("F7") == "E1"               # the IronBCI's E1
    assert av.site_side("X3") is None           # not a scalp site
    m.load_montage("Transverse")
    assert [m.row_label(r) for r in m.rows()][-4:] == ["EKG", "EMG 1",
                                                       "EMG 2", "EMG 3"]
    assert m.rows_prefix == "32ch/+pg/"


def test_unwiring_a_pieeg_input_hides_its_row_only(tmp_path):
    m = _dual(tmp_path / "s.json")
    m.set_wired(["X5"], False)
    labels = [m.row_label(r) for r in m.visible_rows()]
    assert "EMG 2" not in labels and "EKG" in labels and len(labels) == 21


def test_adaptive_reduced_cuts_the_eeg_to_16_leads(tmp_path):
    m = _dual(tmp_path / "s.json")
    assert av.ADAPTIVE_REDUCED in m.montage_names()
    m.load_montage(av.ADAPTIVE_REDUCED)
    labels = [m.row_label(r) for r in m.rows()]
    assert labels[:16] == [f"{a}-{b}" for a, b in
                           av.MONTAGE_PRESETS_16["Double banana"]]
    assert labels[16:] == ["EKG", "EMG 1", "EMG 2", "EMG 3"]
    assert _model(32, tmp_path / "t.json").montage_names()[1] == \
        av.ADAPTIVE_REDUCED
    for n in (8, 16):                       # nothing to cut: not offered
        assert av.ADAPTIVE_REDUCED not in _model(n, tmp_path / f"{n}.json"
                                                 ).montage_names()
