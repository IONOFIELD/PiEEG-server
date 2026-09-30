"""Saline bath check (IronBCI-32 bundle): grading and the guided lift test,
on synthetic data shaped like the 2026-09-30 bath (shared noise ~5 µV,
own noise ~0.5 µV, a few µV of mains, mV offsets)."""
import numpy as np
import pytest

from pieeg_server.saline_check import (BUNDLE, LEAD_COLOURS, SalineCheck,
                                       bundle_metrics, grade)

FS = 512
FULL = 312500.0


class Bath:
    """8 leads in saline; state changes as leads are lifted."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.dc = np.array([6.3, 6.4, 6.6, 5.9, 7.0, 5.6, 6.4, 6.9]) * 1000
        self.out = set()            # lifted colour leads (bundle index)
        self.ref_out = False
        self.bias_out = False
        self.crackly = set()
        self.hum = np.full(BUNDLE, 1.0)
        self.drift_uv_s = 0.0

    def take(self, seconds):
        n = int(seconds * FS)
        t = (self.t + np.arange(n)) / FS
        self.t += n
        common = self.rng.normal(0, 5.0, n)
        x = (self.dc + (-5000 if self.bias_out else 0)
             + common[:, None] + self.rng.normal(0, 0.5, (n, BUNDLE))
             + np.outer(np.sin(2 * np.pi * 60 * t), self.hum)
             + (t * self.drift_uv_s)[:, None])
        for i in self.crackly:
            steps = self.rng.random(n) < 0.02
            x[:, i] += np.cumsum(np.where(steps, self.rng.choice(
                [-1500, 1500], n), 0)) % 3000
        for i in self.out:
            x[:, i] = -FULL
        if self.ref_out:
            x[:] = FULL
        return x


def run(chk, bath, seconds):
    x = bath.take(seconds)
    for i in range(0, len(x), 26):
        chk.feed(x[i:i + 26])


def full_session(chk, bath, swap=None):
    """Baseline, each lead lifted and returned, white, black, final."""
    run(chk, bath, 16)
    for i in range(BUNDLE):
        run(chk, bath, 2)
        bath.out = {swap.get(i, i) if swap else i}
        run(chk, bath, 4)
        bath.out = set()
        run(chk, bath, 4)
    run(chk, bath, 2)
    bath.ref_out = True
    run(chk, bath, 4)
    bath.ref_out = False
    run(chk, bath, 4)
    bath.bias_out = True
    run(chk, bath, 5)
    bath.bias_out = False
    run(chk, bath, 22)


def test_good_bundle_passes_and_maps_every_lead():
    chk, bath = SalineCheck(FS, FULL, bundle_first=9), Bath()
    full_session(chk, bath)
    r = chk.result()
    assert chk.phase == "done" and r["completed"] and r["passed"]
    assert [e["channel"] for e in r["leads"]] == list(range(9, 17))
    assert all(e["mapping"] == "ok" and e["grade"] == "pass"
               for e in r["leads"])
    assert r["ref"]["mapping"] == "ok" and r["bias"]["mapping"] == "ok"
    assert "-5.0 mV" in r["bias"]["mapping_text"]
    assert r["bundle"] == "CH 9-16"
    # shared part reported: 5 µV white over 0-256 Hz is ~1.95 µV in 1-40 Hz
    assert 1.6 < r["common_uv_final"] < 2.3


def test_swapped_header_is_named():
    chk, bath = SalineCheck(FS, FULL), Bath(1)
    full_session(chk, bath, swap={0: 2, 2: 0})       # yellow <-> red pins
    r = chk.result()
    yellow, red = r["leads"][0], r["leads"][2]
    assert yellow["mapping"] == "fail" and "CH3" in yellow["mapping_text"]
    assert "header" in yellow["mapping_text"]
    assert red["mapping"] == "fail" and not r["passed"]


def test_crackly_electrode_fails_the_grade():
    chk, bath = SalineCheck(FS, FULL), Bath(2)
    bath.crackly = {1}
    full_session(chk, bath)
    orange = chk.result()["leads"][1]
    assert orange["grade"] == "fail"
    assert any("crackly" in n for n in orange["notes"])


def test_no_lift_times_out_and_moves_on():
    chk, bath = SalineCheck(FS, FULL, lift_timeout_s=5), Bath(3)
    run(chk, bath, 16)
    assert chk.status()["title"] == "LIFT YELLOW"
    run(chk, bath, 7)
    assert chk.lifts["yellow"]["text"] == "no change seen"
    assert chk.status()["title"] == "LIFT ORANGE"


def test_skip_moves_to_the_next_lead():
    chk, bath = SalineCheck(FS, FULL), Bath(4)
    run(chk, bath, 16)
    chk.skip()
    assert chk.lifts["yellow"]["result"] == "skipped"
    assert chk.status()["title"] == "LIFT ORANGE"


def test_lead_railed_at_baseline_is_not_lifted():
    chk, bath = SalineCheck(FS, FULL), Bath(5)
    bath.out = {0}
    run(chk, bath, 16)
    assert chk.lifts["yellow"]["text"] == "not connected at baseline"
    assert chk.status()["title"] == "LIFT ORANGE"


def test_slow_drift_is_not_black():
    # every lead drifting the same way (settling) must not pass as the
    # black lead's step: black is judged against the last few seconds
    chk, bath = SalineCheck(FS, FULL, lift_timeout_s=20), Bath(6)
    run(chk, bath, 16)
    for _ in range(BUNDLE + 1):
        chk.skip()
    assert chk.status()["title"] == "LIFT BLACK"
    bath.drift_uv_s = 150.0                          # 9 mV a minute
    run(chk, bath, 15)
    assert "black" not in chk.lifts


def test_other_lead_disturbed_during_a_lift_is_blamed():
    chk, bath = SalineCheck(FS, FULL), Bath(7)
    run(chk, bath, 16)
    run(chk, bath, 2)
    bath.out = {0}
    bath.hum[3] = 200.0                              # brown picks up hum
    run(chk, bath, 4)
    assert chk.lifts["yellow"]["result"] == "ok"
    assert "CH4 disturbed" in chk.lifts["yellow"]["text"]
    assert chk.side_notes[3]


def test_grade_flags_noise_and_mains():
    x = Bath(8).take(15)
    x[:, 4] += np.random.default_rng(1).normal(0, 20, len(x))   # bad contact
    x[:, 7] += 12 * np.sin(2 * np.pi * 60 * np.arange(len(x)) / FS)
    leads, bundle = grade(bundle_metrics(x, FS, FULL))
    assert leads[4][0] == "fail" and "noisy contact" in leads[4][1][0]
    assert leads[7][0] == "warn" and "60 Hz" in leads[7][1][0]
    assert all(g == "pass" for g, _ in leads[:4])
    assert bundle == []


def test_everything_railed_blames_ref():
    x = np.full((FS * 15, BUNDLE), FULL)
    leads, bundle = grade(bundle_metrics(x, FS, FULL))
    assert all(g == "fail" for g, _ in leads)
    assert "white" in bundle[0]


def test_cancel_reports_incomplete():
    chk, bath = SalineCheck(FS, FULL), Bath(9)
    run(chk, bath, 5)
    chk.cancel()
    r = chk.result()
    assert chk.phase == "done" and r["cancelled"] and not r["passed"]
    assert not r["completed"]


@pytest.mark.parametrize("colour", LEAD_COLOURS)
def test_status_names_each_lead(colour):
    chk, bath = SalineCheck(FS, FULL), Bath(10)
    run(chk, bath, 16)
    for _ in range(LEAD_COLOURS.index(colour)):
        chk.skip()
    assert chk.status()["title"] == f"LIFT {colour.upper()}"
    assert chk.status()["can_skip"]


def test_only_the_chosen_leads_are_lifted_and_graded():
    # leads 3 and 8 not chosen (not wired): one floats noisily, one rails
    chk, bath = SalineCheck(FS, FULL, leads=[0, 1, 3, 4, 5, 6]), Bath(11)
    run(chk, bath, 16)
    assert chk.status()["title"] == "LIFT YELLOW"
    titles = [s for s in chk.status()["progress"]]
    assert [c for c, _ in titles] == ["yellow", "orange", "brown", "green",
                                      "blue", "purple", "white", "black"]


def test_unchosen_leads_stay_out_of_the_report_and_the_verdict():
    chk, bath = SalineCheck(FS, FULL, leads=[0, 1, 3, 4, 5, 6]), Bath(12)
    bath.crackly = {2}                  # red: not chosen, would fail
    orig = bath.take

    def take(seconds):
        x = orig(seconds)
        x[:, 7] = -FULL                 # grey: not chosen, railed
        return x
    bath.take = take
    full_session(chk, bath)
    r = chk.result()
    assert r["passed"], r
    assert r["leads"][2]["mapping"] == "not selected"
    assert r["leads"][2]["grade"] is None and r["leads"][7]["grade"] is None
    assert r["selected_channels"] == [1, 2, 4, 5, 6, 7]
