"""IronBCI drift stage: a steady electrode ramp must not leave a standing
offset on the display (a settling O1 at 349 µV/s pinned its row edge)."""
import numpy as np

from pieeg_server.acq_viewer import StreamingFilter, ViewerModel

FS = 512


def _ramp_offset(drift_inputs, lff):
    t = np.arange(30 * FS) / FS
    x = np.zeros((len(t), 2)) + 20000.0
    x[:, 0] += 349.0 * t
    x[:, 1] += 349.0 * t
    f = StreamingFilter(2, FS, drift_inputs=drift_inputs)
    f.set_cutoffs(lff, 70.0, 60.0)
    out = np.vstack([f.process(x[i:i + 25]) for i in range(0, len(x), 25)])
    return out[-FS:].mean(axis=0)


def test_ramp_cancelled_on_drift_inputs_only():
    off = _ramp_offset(1, 1.0)
    assert abs(off[0]) < 1.0                    # IronBCI input: centred
    assert 50.0 < off[1] < 60.0                 # other input: s * TC as before


def test_lff_off_keeps_true_dc():
    off = _ramp_offset(2, None)
    assert off[0] > 20000.0


def test_eeg_band_unchanged():
    t = np.arange(120 * FS) / FS
    s = np.sin(2 * np.pi * 0.5 * t)[:, None]
    a = StreamingFilter(1, FS, drift_inputs=1)
    b = StreamingFilter(1, FS)
    for f in (a, b):
        f.set_cutoffs(0.1, None, None)
    ra = np.std(a.process(s)[-30 * FS:])
    rb = np.std(b.process(s)[-30 * FS:])
    assert abs(ra / rb - 1) < 0.005


def test_model_filters_carry_drift_inputs():
    m = ViewerModel(4, FS, ["F7", "T3", "T5", "O1"], store=None,
                    drift_inputs=4)
    assert m.new_filter()._ndrift == 4


def test_ironbci_bank1_is_the_site_cap():
    from pieeg_server.scope_console import _IRONBCI32_ELECTRODES as E
    assert E[:8] == ["Fp1", "Fp2", "Fz", "Pz", "C3", "C4", "O1", "O2"]
    assert len(set(E)) == 32


def test_wired_sites_skip_switched_off():
    m = ViewerModel(4, FS, ["Fp1", "Fp2", "E3", "E4"], store=None)
    m.set_wired(["E3", "E4"], False)
    assert m.wired_sites() == ["Fp1", "Fp2"]
