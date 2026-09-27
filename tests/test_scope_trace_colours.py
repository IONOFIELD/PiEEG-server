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
