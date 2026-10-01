"""Trace colour follows the row type: EEG blue, EKG red, EMG white."""
from pieeg_server import acq_viewer as av


def test_row_kind_from_name():
    for name in ("EKG", "ECG", "ekg 1", "L-R ECG", "E1-E2 EKG"):
        assert av.row_kind(name) == "ekg", name
    for name in ("EMG 1", "chin EMG", "EMG-L tib", "emg"):
        assert av.row_kind(name) == "emg", name
    for name in ("Fp1-F7", "Cz-Pz", "E3-E4", "EMGX", "BECG", ""):
        assert av.row_kind(name) == "eeg", name


def test_colours():
    assert av.row_colour("Fp1-F7") == av.GEIST["trace_eeg"]
    assert av.row_colour("EKG") == av.GEIST["trace_ekg"] == "#ee4444"
    assert av.row_colour("EMG 2") == av.GEIST["trace_emg"] == "#ffffff"


def test_default_sensitivity_is_20():
    assert av.DEFAULT_SENS == 20


def test_picked_kind_beats_the_name():
    assert av.row_kind("EMG 1", "eeg") == "eeg"
    assert av.row_kind("Fp1-REF", "ekg") == "ekg"
    assert av.row_kind("EKG", "bogus") == "ekg"
    assert av.row_colour("EMG 2", "eeg") == av.GEIST["trace_eeg"]


def test_row_kind_saved_and_forks_preset(tmp_path):
    from pieeg_server.acq_viewer import ViewerModel, MontageStore
    store = MontageStore(tmp_path / "m.json")
    m = ViewerModel(4, 250, ["Fp1", "Fp2", "E3", "E4"], store=store)
    m.load_montage(av.CUSTOM_MONTAGE)
    row = m.insert_row(None, "E3", "E4", label="EMG 1")
    assert m.row_kind(row) == "emg" and "kind" not in row
    row = m.edit_row(row, "E3", "E4", "EMG 1", "eeg")
    assert m.row_kind(row) == "eeg"
    assert m.row_colour(row) == av.GEIST["trace_eeg"]
    # a pick that matches the name isn't pinned
    row = m.edit_row(row, "E3", "E4", "EMG 1", "emg")
    assert "kind" not in row
    row = m.edit_row(row, "E3", "E4", "chin", "emg")
    assert row["kind"] == "emg"
    m.save_current()
    m2 = ViewerModel(4, 250, ["Fp1", "Fp2", "E3", "E4"],
                     store=MontageStore(tmp_path / "m.json"))
    m2.load_montage(av.CUSTOM_MONTAGE)
    r2 = [r for r in m2.rows() if r["pair"] == ("E3", "E4")][0]
    assert m2.row_kind(r2) == "emg" and m2.row_label(r2) == "chin"
