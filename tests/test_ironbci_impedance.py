"""IronBCI-32 impedance with a test current from the Pi's GPIOs: plan,
measurement, calibration and the check against a fake stream."""

import asyncio
import math

import numpy as np
import pytest

from pieeg_server import ironbci_impedance as ii

FS = 512.0


def _plan(ref=True):
    return ii.make_plan([(5, 1), (6, 3), (12, 7)], ref_gpio=16 if ref else None)


def _signal(plan, ohms, ref_ohms=5000.0, zin=math.inf, seconds=5.0, seed=1,
            freq_err=0.0):
    """What the board records with the test leads driven: each tested
    input gets I*Z (Z shunted by the input's own zin) at its frequency, and
    the REF tone shows (inverted) on every input; plus EEG, mains, drift and
    noise."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    t = np.arange(n) / FS
    x = np.zeros((n, 32))
    for c in range(32):
        x[:, c] = (40 * np.sin(2 * np.pi * 10 * t + c)
                   + 3 * np.sin(2 * np.pi * 60 * t + c)
                   + 150 * t + 2000 + rng.normal(0, 1.0, n))
    def sq(f):
        # the square wave as the board's filter passes it: odd harmonics
        # below Nyquist only (a sampled sign() would alias its edges)
        f = f * (1 + freq_err)
        return sum(4 / np.pi / k * np.sin(2 * np.pi * k * f * t + 0.3 * k)
                   for k in range(1, int(FS / 2 / f), 2))
    for tl in plan.leads:
        z = ohms[tl.input]
        zeff = z / (1 + z / zin)
        # square wave of 0..3.3 V through 10 MΩ: ±0.165 µA -> µV = A*Ω*1e6
        i_sq = ii.GPIO_VOLTS / 2 / plan.series_ohms
        x[:, tl.input - 1] += i_sq * zeff * 1e6 * sq(tl.freq)
    if plan.ref is not None:
        i_sq = ii.GPIO_VOLTS / 2 / plan.series_ohms
        x -= (i_sq * ref_ohms * 1e6 * sq(plan.ref.freq))[:, None]
    return x


def test_plan_assigns_frequencies_and_refuses_bad_pins():
    p = _plan()
    assert p.ref.freq == ii.SLOTS_HZ[0]
    assert [t.freq for t in p.leads] == list(ii.SLOTS_HZ[1:4])
    assert [t.key for t in p.all_leads()] == ["E1", "E3", "E7", "REF"]
    with pytest.raises(ValueError, match="taken"):
        ii.make_plan([(26, 1)])                     # PiEEG DRDY
    with pytest.raises(ValueError, match="own GPIO"):
        ii.make_plan([(5, 1), (5, 2)])
    with pytest.raises(ValueError, match="distinct"):
        ii.make_plan([(5, 1), (6, 1)])
    back = ii.Plan.from_dict(p.to_dict())
    assert back == p


def test_slots_avoid_mains_and_square_harmonics():
    for f in ii.SLOTS_HZ:
        for line in (50.0, 60.0):
            assert abs(f - line) > 2
        # odd harmonics of every slot sit above the whole grid
        assert 3 * f > max(ii.SLOTS_HZ) + 2
    assert 31.25 not in ii.SLOTS_HZ                 # the PiEEG's own carrier


def test_uncalibrated_estimate_from_the_nominal_current():
    plan = _plan()
    x = _signal(plan, {1: 5000.0, 3: 20000.0, 7: 2000.0})
    r = ii.analyze(x[-2048:], FS, plan, cal={})
    by = {lead["name"]: lead for lead in r["leads"] if lead["status"] != ii.UNTESTED}
    assert set(by) == {"E1", "E3", "E7"}
    for key, z in (("E1", 5000.0), ("E3", 20000.0), ("E7", 2000.0)):
        assert by[key]["status"] == ii.UNCALIBRATED
        assert by[key]["estimate_ohms"] == pytest.approx(z, rel=0.03)
        assert by[key]["text"].startswith("≈")
    assert r["ref_lead"]["estimate_ohms"] == pytest.approx(5000.0, rel=0.03)
    assert r["leads"][1]["status"] == ii.UNTESTED
    assert all(d["crosstalk_uv"] < 1.0 for d in r["diag"].values())


def test_calibrated_readings_through_an_input_shunt():
    """Calibrate with 0/10k/47k on an input network that shunts (zin 400k),
    then read unknown electrodes: the shunt model recovers them."""
    plan = _plan()
    zin = 400e3
    steps = {}
    for z in (0.0, 10000.0, 47000.0):
        r = ii.analyze(_signal(plan, {1: z, 3: z, 7: z}, ref_ohms=z, zin=zin,
                               seed=int(z) % 7)[-2048:], FS, plan, cal={})
        for lead in [lead for lead in r["leads"] if lead["status"] != ii.UNTESTED] \
                + [r["ref_lead"]]:
            steps.setdefault(lead["name"], []).append((z, lead["carrier_uv"]))
    cal = {k: ii.fit_lead(v) for k, v in steps.items()}
    assert cal["E1"].zin == pytest.approx(zin, rel=0.15)
    truth = {1: 7000.0, 3: 30000.0, 7: 120000.0}
    r = ii.analyze(_signal(plan, truth, ref_ohms=3000.0, zin=zin,
                           seed=9)[-2048:], FS, plan, cal=cal)
    e1, e3, e7 = (r["leads"][i - 1] for i in (1, 3, 7))
    assert e1["status"] == ii.OK and e1["ohms"] == pytest.approx(7000, rel=0.05)
    assert e1["band"] == "green"
    assert e3["ohms"] == pytest.approx(30000, rel=0.05) and e3["band"] == "amber"
    assert e7["status"] == ii.ABOVE and e7["text"] == ">47.0 kΩ"
    assert e7["band"] == "red"
    # REF isn't shunted the same way in this model; just in range and near
    assert r["ref_lead"]["status"] == ii.OK
    assert r["ref_lead"]["ohms"] == pytest.approx(3000, rel=0.15)


def test_a_floating_electrode_reads_off_and_a_railed_input_off():
    plan = _plan(ref=False)
    cal = {k: ii.LeadCal(v0=0.0, k=0.21, zin=math.inf, max_ohms=47000.0)
           for k in ("E1", "E3", "E7")}
    x = _signal(plan, {1: 5000.0, 3: 1.5e6, 7: 5000.0})
    x[:, 6] = 312_400.0
    r = ii.analyze(x[-2048:], FS, plan, cal=cal)
    assert r["leads"][0]["status"] == ii.OK
    assert r["leads"][2]["status"] == ii.OFF and r["leads"][2]["band"] == "red"
    assert r["leads"][6]["status"] == ii.RAILED


def test_the_carrier_is_found_when_the_pwm_runs_slightly_off():
    plan = _plan(ref=False)
    x = _signal(plan, {1: 5000.0, 3: 5000.0, 7: 5000.0}, freq_err=2e-3)
    r = ii.analyze(x[-2048:], FS, plan, cal={})
    for i in (1, 3, 7):
        assert r["leads"][i - 1]["estimate_ohms"] == pytest.approx(5000, rel=0.03)


def test_fit_needs_zero_and_two_resistors():
    with pytest.raises(ValueError):
        ii.fit_lead([(0, 1.0), (10000, 50.0)])
    lc = ii.fit_lead([(0, 2.0), (10000, 2102.0), (47000, 9872.0)])
    assert math.isinf(lc.zin) or lc.zin > 1e8
    assert lc.ohms(2.0 + 0.21 * 20000) == pytest.approx(20000, rel=1e-3)


class _Injector:
    def __init__(self):
        self.calls = []

    def start(self, drives):
        self.calls.append(("start", list(drives)))

    def stop(self):
        self.calls.append(("stop",))


class _Acq:
    """Feeds _signal() frames to subscribers once the injector started."""
    vref_uv, pga_gain = 2.5e6, 8

    class _hw:
        sample_rate = 512

    def __init__(self, loop, block, inj):
        self._loop, self._block, self._inj = loop, block, inj
        self.subs = []

    def subscribe(self, maxsize=0):
        q = asyncio.Queue()
        self.subs.append(q)
        for k, row in enumerate(self._block):
            q.put_nowait({"n": k, "channels": list(row)})
        return q

    def unsubscribe(self, q):
        self.subs.remove(q)


def test_check_drives_then_always_releases_the_pins():
    plan = _plan()
    block = _signal(plan, {1: 5000.0, 3: 8000.0, 7: 2000.0}, seconds=6.0)
    inj = _Injector()
    loop = asyncio.new_event_loop()
    acq = _Acq(loop, block, inj)
    check = ii.IronBCIImpedanceCheck(acq, plan, calibration={}, injector=inj)
    r = loop.run_until_complete(check.run())
    loop.close()
    assert inj.calls[0] == ("start", plan.drives()) and inj.calls[-1] == ("stop",)
    assert acq.subs == []
    assert r["leads"][2]["estimate_ohms"] == pytest.approx(8000, rel=0.03)


def test_combine_places_each_board_and_withholds_per_board():
    eeg = ii.analyze(_signal(_plan(), {1: 5000.0, 3: 5000.0, 7: 5000.0})[-2048:],
                     FS, _plan(), cal={})
    pg = {"leads": [{"name": f"E{i}", "ohms": 4000.0, "status": "ok",
                     "text": "4.0 kΩ", "band": "green"} for i in range(1, 9)],
          "ref": "green", "gnd": "red", "problem": "GND (BIO) isn't connected"}
    r = ii.combine([("EEG", 1, eeg), ("PG", 33, pg)])
    assert len(r["leads"]) == 40 and r["first_input"] == 1
    assert r["leads"][0]["name"] == "E1" and r["leads"][32]["status"] == "withheld"
    assert r["notes"] == ["PG: GND (BIO) isn't connected"]
    texts = [e["text"] for e in r["extra_lines"]]
    assert texts[0].startswith("EEG REF ≈") and texts[1] == "PG REF ok  GND OFF"
    r2 = ii.combine([("EEG", 1, eeg), ("PG", 33, "lead-off busy")])
    assert r2["notes"] == ["PG: lead-off busy"] and len(r2["leads"]) == 32
