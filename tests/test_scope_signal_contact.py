"""The live contact estimate for boards without lead-off detection (IronBCI-32):
each input's mains pickup against the median of the board's wired inputs."""

import numpy as np

from pieeg_server import acq_viewer as av

FS = 512


def _block(mains_uv, seconds=2.0, line=60.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * FS)) / FS
    cols = []
    for k, a in enumerate(mains_uv):
        x = (a * np.sin(2 * np.pi * line * t + k)        # mains, any phase
             + 30 * np.sin(2 * np.pi * 10 * t + k)       # alpha
             + 200 * t                                  # electrode drift
             + rng.normal(0, 2, t.size) + 3000)          # noise + DC offset
        cols.append(x)
    return np.stack(cols, axis=1)


def test_mains_pickup_measures_the_line_through_drift_and_alpha():
    amp = av.mains_pickup(_block([2.0, 20.0, 0.0]), FS, 60.0)
    assert abs(amp[0] - 2.0) < 0.3
    assert abs(amp[1] - 20.0) < 0.5
    assert amp[2] < 0.3


def test_grades_are_relative_to_the_boards_median():
    x = _block([2, 2.5, 3, 2, 12, 40, 2, 2])
    g = av.grade_signal_contact(x, FS, 60.0, 312500.0, [True] * 8)
    assert g[:4] == ["green"] * 4 and g[6:] == ["green"] * 2
    assert g[4] == "amber"          # ~5x its neighbours
    assert g[5] == "red"            # ~18x


def test_low_pickup_is_always_fine_even_against_a_quiet_board():
    # on battery the whole board picks up ~0.3 µV: 2.5 µV is still fine
    x = _block([0.3, 0.3, 0.3, 2.5])
    g = av.grade_signal_contact(x, FS, 60.0, 312500.0, [True] * 4)
    assert g == ["green"] * 4


def test_flat_railed_and_unwired_inputs():
    x = _block([2, 2, 2, 2, 2])
    x[:, 1] = 0.0                                   # reads nothing
    x[:, 2] = 312000.0                              # at the rail
    g = av.grade_signal_contact(x, FS, 60.0, 312500.0,
                                [True, True, True, True, False])
    assert g == ["green", "red", "red", "green", None]


def test_unwired_inputs_dont_set_the_median():
    # three unwired (floating) inputs full of mains must not make the
    # wired lead with 15 µV look normal
    x = _block([2, 2, 15, 80, 80, 80])
    g = av.grade_signal_contact(x, FS, 60.0, 312500.0,
                                [True, True, True, False, False, False])
    assert g[:3] == ["green", "green", "amber"] and g[3:] == [None] * 3


def test_a_colour_shows_only_once_it_lasts():
    sc = av.SignalContact(2, hold=3)
    sc.update(["green", "red"])
    assert sc.electrode(0) is None                  # not enough estimates
    sc.update(["green", "red"])
    sc.update(["green", "green"])
    assert sc.electrode(0) == "green"
    assert sc.electrode(1) == "green"               # one clean estimate wins
    for _ in range(3):
        sc.update(["green", "red"])
    assert sc.electrode(1) == "red"
    sc.update([None, "red"])                        # unwired: forget it
    assert sc.electrode(0) is None and sc.electrode(1) == "red"


def test_model_estimates_only_its_signal_contact_inputs():
    elec = av.DEFAULT_ELECTRODES[:4] + ["X1", "X2"]
    m = av.ViewerModel(6, FS, elec, signal_contact_inputs=4,
                       input_labels={"X1": "E1", "X2": "E2"})
    assert not m.update_signal_contact(312500.0)   # nothing buffered yet
    x = _block([2, 2, 2, 50, 0, 0], seconds=3.0)
    m.push(x)
    m.unwired.add(elec[1])
    for _ in range(3):
        assert m.update_signal_contact(312500.0)
    assert m.site_contact(elec[0]) == "green"
    assert m.site_contact(elec[1]) is None          # not wired this session
    assert m.site_contact(elec[3]) == "red"
    # the second board's inputs keep the lead-off tracker (no readout yet)
    assert m.site_contact("X1") is None
