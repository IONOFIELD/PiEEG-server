"""Saline bath check for one IronBCI-32 cable bundle (8 leads + REF + BIAS).

The IronBCI-32 can't measure electrode impedance, so a bundle is proved in a
saline bath instead, the way it was first done by hand (2026-09-30):

1. Baseline, hands off: every lead graded on its own noise, mains pickup,
   sudden jumps, DC offset and rail.
2. Guided lift, hands free: "lift YELLOW" -> exactly its channel must break
   (rail, or float off with a big offset / hum), and no other; "put it back"
   -> it must recover. Then orange … grey, then white (REF: every channel
   breaks) and black (BIAS: every channel's DC steps together by a few mV).
3. Final baseline: graded again, since reseating a lead can disturb another.

Pure numpy, no Tk: the viewer feeds it the bundle's raw µV columns as they
arrive (feed) and draws `status()`; `result()` is the saved report.

Numbers it was tuned on (bundle CH 1-8, Ag/AgCl, two batteries): own noise
(common removed) 0.45-0.5 µV, shared noise 5.5 µV with BIAS in (1.5 out),
mains 0.2-3 µV bundled and 7-15 µV with a lead routed away, a bad electrode
832 jumps > 1 mV in 10 s, a lifted lead railed or floated to ~-109 mV.
"""

from __future__ import annotations

import datetime as _dt

import numpy as np

# Lead colour of each bundle channel, CH 1..8 of the bundle (the IronBCI
# strips: yellow 1, orange 2, red 3, brown 4, green 5, blue 6, purple 7,
# grey 8), then the two shared leads. Lift order = this order.
LEAD_COLOURS = ("yellow", "orange", "red", "brown", "green", "blue",
                "purple", "grey")
REF_COLOUR, BIAS_COLOUR = "white", "black"
BUNDLE = 8

EVAL_EVERY_S = 1.0          # the lift detector looks this often …
EVAL_WINDOW_S = 2.0         # … at the last this many seconds
BASELINE_S = 15.0
HOLD_EVALS = 2              # a lift / return must hold this many looks
LIFT_TIMEOUT_S = 30.0
RETURN_TIMEOUT_S = 45.0

RAIL = 0.99                 # of full scale: railed
JUMP_UV = 1000.0            # one-sample step this big = a crackly contact
JUMPS_FAIL = 3              # per baseline
NOISE_BAND = (1.0, 40.0)
OWN_NOISE_MIN_UV = 3.0      # own noise fails above max(this, REL x median)
OWN_NOISE_REL = 4.0
HUM_MIN_UV = 5.0            # mains warns above max(this, REL x median)
HUM_REL = 3.0
COMMON_WARN_UV = 10.0       # shared noise above this: REF/BIAS disturbed
DC_WARN_MV = 50.0

BROKEN_DC_MV = 20.0         # lifted lead: DC moved this far …
BROKEN_NOISE_UV = 30.0      # … or noise above max(this, REL x baseline)
BROKEN_NOISE_REL = 10.0
BROKEN_HUM_UV = 50.0        # … or mains above max(this, REL x baseline)
BROKEN_HUM_REL = 10.0
BIAS_STEP_MV = 2.0          # black lifted: every DC steps this far together
BIAS_BACK_MV = 1.5          # … and is back within this


def _band_rms(x, fs, lo, hi):
    """RMS (µV) of each column of x between lo and hi Hz (linear detrend +
    Hann, window-corrected)."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    t = np.arange(n, dtype=np.float64)
    t -= t.mean()
    xd = x - x.mean(axis=0)
    slope = (t @ xd) / (t @ t)
    xd = xd - np.outer(t, slope)
    w = np.hanning(n)
    spec = np.fft.rfft(xd * w[:, None], axis=0)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    band = (f >= lo) & (f <= hi)
    power = 2.0 * np.sum(np.abs(spec[band]) ** 2, axis=0) / n
    return np.sqrt(power / np.sum(w ** 2))


def _hum(x, fs, line):
    """Mains amplitude (µV) per column, one Hann-weighted bin at `line` Hz."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    w = np.hanning(n)
    ph = np.exp(-2j * np.pi * line * np.arange(n) / fs) * w
    return 2.0 * np.abs(ph @ (x - x.mean(axis=0))) / w.sum()


def bundle_metrics(x, fs, full_scale_uv, line=60.0, use=None):
    """Per-channel numbers for a block (N x 8 µV) of one bundle.

    own_noise_uv is each lead's noise with the part every lead shares taken
    out (median across the working leads), which is what says whether that
    electrode is good; common_uv is the shared part (REF / BIAS). use: bool
    per lead, the leads chosen for the check (default all); the others
    (probably not in the bath) never count toward the shared part."""
    x = np.asarray(x, dtype=np.float64)
    use = np.ones(x.shape[1], bool) if use is None else np.asarray(use, bool)
    railed = np.max(np.abs(x), axis=0) >= RAIL * full_scale_uv
    working = ~railed & use
    xd = x - x.mean(axis=0)
    if working.sum() >= 3:
        common = np.median(xd[:, working], axis=1)
    else:
        common = np.zeros(x.shape[0])
    lo, hi = NOISE_BAND
    jumps = (np.abs(np.diff(x, axis=0)) > JUMP_UV).sum(axis=0)
    jumps[railed] = 0
    return {
        "railed": railed,
        "dc_mv": x.mean(axis=0) / 1000.0,
        "noise_uv": _band_rms(xd, fs, lo, hi),
        "own_noise_uv": _band_rms(xd - common[:, None], fs, lo, hi),
        "hum_uv": _hum(x, fs, line),
        "jumps": jumps,
        "common_uv": float(_band_rms(common[:, None], fs, lo, hi)[0]),
        "use": use,
    }


def grade(m):
    """(grade, notes) per lead from bundle_metrics(): grade "pass", "warn"
    or "fail"; plus bundle-wide notes."""
    use = m.get("use", np.ones(len(m["railed"]), bool))
    working = ~m["railed"] & use
    med_own = (float(np.median(m["own_noise_uv"][working]))
               if working.any() else 0.0)
    med_hum = float(np.median(m["hum_uv"][working])) if working.any() else 0.0
    own_lim = max(OWN_NOISE_MIN_UV, OWN_NOISE_REL * med_own)
    hum_lim = max(HUM_MIN_UV, HUM_REL * med_hum)
    leads = []
    for i in range(len(m["railed"])):
        notes, g = [], "pass"
        if not use[i]:
            leads.append((None, ["not selected"]))
            continue
        if m["railed"][i]:
            leads.append(("fail", ["railed: not in the bath or not "
                                   "connected"]))
            continue
        if m["jumps"][i] >= JUMPS_FAIL:
            g = "fail"
            notes.append(f"crackly ({int(m['jumps'][i])} jumps): swap the "
                         "electrode, then the lead")
        if m["own_noise_uv"][i] > own_lim:
            g = "fail"
            notes.append(f"noisy contact ({m['own_noise_uv'][i]:.1f} µV): "
                         "bubble, not under, or loose snap")
        if abs(m["dc_mv"][i]) > DC_WARN_MV:
            g = "warn" if g == "pass" else g
            notes.append(f"offset {m['dc_mv'][i]:.0f} mV: electrode dry or "
                         "settling")
        if m["hum_uv"][i] > hum_lim:
            g = "warn" if g == "pass" else g
            notes.append(f"60 Hz {m['hum_uv'][i]:.1f} µV: keep the lead "
                         "with the bundle")
        leads.append((g, notes))
    bundle = []
    if not use.any():
        pass
    elif working.sum() == 0:
        bundle.append("every lead railed: is white (REF) in the bath?")
    elif m["common_uv"] > COMMON_WARN_UV:
        bundle.append(f"shared noise {m['common_uv']:.1f} µV: REF/BIAS moved "
                      "or still settling")
    return leads, bundle


class SalineCheck:
    """The guided check for one bundle. feed() raw µV blocks (M x 8, the
    bundle's CH 1..8); read status() to draw; result() once done."""

    def __init__(self, fs, full_scale_uv, line=60.0, bundle_first=1,
                 baseline_s=BASELINE_S, lift_timeout_s=LIFT_TIMEOUT_S,
                 return_timeout_s=RETURN_TIMEOUT_S, leads=None):
        self.fs = float(fs)
        self.full_scale_uv = float(full_scale_uv)
        self.line = float(line)
        self.bundle_first = int(bundle_first)       # board CH of lead 1
        # the bundle's leads chosen for the check (0-based; default all 8):
        # only these are lifted and graded
        self.use = np.zeros(BUNDLE, bool)
        self.use[list(range(BUNDLE)) if leads is None else list(leads)] = True
        self._n_base = int(round(baseline_s * self.fs))
        self._n_eval = int(round(EVAL_EVERY_S * self.fs))
        self._n_win = int(round(EVAL_WINDOW_S * self.fs))
        self._lift_timeout = int(round(lift_timeout_s * self.fs))
        self._return_timeout = int(round(return_timeout_s * self.fs))
        self._buf = np.zeros((0, BUNDLE))
        self._since_eval = 0
        self._samples = 0           # in the current step
        self.steps = ([("lead", i) for i in range(BUNDLE) if self.use[i]]
                      + [("ref", None), ("bias", None)])
        self.step_i = 0
        self.phase = "baseline"     # baseline, lift, return, final, done
        self.base = None            # metrics
        self.final = None
        self.lifts = {}             # step key -> {"result", "text"}
        self._seen = []             # last broken sets / bias verdicts
        # black's step is judged against the DC a few seconds before it
        # (not the baseline a minute ago: electrodes drift), frozen once seen
        self._dc_hist = []
        self._bias_ref = None
        # notes on OTHER leads seen while one was lifted ("CH2 disturbed")
        self.side_notes = {i: [] for i in range(BUNDLE)}
        self.started = _dt.datetime.now().astimezone()
        self.cancelled = False

    # ---- names ------------------------------------------------------------ #
    def channel(self, i):
        """Board CH number of bundle lead i (0-based)."""
        return self.bundle_first + i

    @property
    def name(self):
        return f"CH {self.bundle_first}-{self.bundle_first + BUNDLE - 1}"

    def _step_key(self, step):
        kind, i = step
        return LEAD_COLOURS[i] if kind == "lead" else (
            REF_COLOUR if kind == "ref" else BIAS_COLOUR)

    # ---- driving ---------------------------------------------------------- #
    def cancel(self):
        self.cancelled = True
        self.phase = "done"

    def skip(self):
        """Skip the lead being lifted (or the wait for it to come back)."""
        if self.phase == "lift":
            key = self._step_key(self.steps[self.step_i])
            self.lifts[key] = {"result": "skipped", "text": "skipped"}
            self._next_step()
        elif self.phase == "return":
            self._next_step()

    def feed(self, block):
        if self.phase == "done":
            return
        block = np.asarray(block, dtype=np.float64).reshape(-1, BUNDLE)
        if not len(block):
            return
        keep = max(self._n_base, self._n_win)
        self._buf = np.vstack([self._buf, block])[-keep:]
        self._samples += len(block)
        if self.phase in ("baseline", "final"):
            if self._samples >= self._n_base:
                m = bundle_metrics(self._buf[-self._n_base:], self.fs,
                                   self.full_scale_uv, self.line, self.use)
                if self.phase == "baseline":
                    self.base = m
                    self._begin_lifts()
                else:
                    self.final = m
                    self.phase = "done"
            return
        self._since_eval += len(block)
        while self._since_eval >= self._n_eval and self.phase != "done":
            self._since_eval -= self._n_eval
            self._evaluate()

    def _begin_lifts(self):
        self.step_i = -1
        self._next_step()

    def _next_step(self):
        self.step_i += 1
        self._seen, self._samples, self._since_eval = [], 0, 0
        self._bias_ref = None
        # a lead railed at baseline can't be seen lifting: it isn't in
        while (self.step_i < len(self.steps)
               and self.steps[self.step_i][0] == "lead"
               and self.base["railed"][self.steps[self.step_i][1]]):
            key = self._step_key(self.steps[self.step_i])
            self.lifts[key] = {"result": "fail",
                               "text": "not connected at baseline"}
            self.step_i += 1
        if self.step_i >= len(self.steps):
            self.phase = "final"
            self._buf = np.zeros((0, BUNDLE))
        else:
            self.phase = "lift"

    def _window(self):
        """Metrics of the last EVAL_WINDOW_S, with noise and mains taken from
        its quieter half: a lifted lead is broken in both halves, while the
        step of a lead going in or out (or black's few-mV step) spoils only
        one."""
        x = self._buf[-self._n_win:]
        args = (self.fs, self.full_scale_uv, self.line, self.use)
        m = bundle_metrics(x, *args)
        h = len(x) // 2
        a = bundle_metrics(x[:h], *args)
        b = bundle_metrics(x[h:], *args)
        m["noise_uv"] = np.minimum(a["noise_uv"], b["noise_uv"])
        m["hum_uv"] = np.minimum(a["hum_uv"], b["hum_uv"])
        m["railed"] = a["railed"] & b["railed"]
        m["dc_mv"] = b["dc_mv"]         # where it is now
        return m

    def _broken(self, m):
        b = self.base
        noise_lim = np.maximum(BROKEN_NOISE_UV,
                               BROKEN_NOISE_REL * b["noise_uv"])
        hum_lim = np.maximum(BROKEN_HUM_UV, BROKEN_HUM_REL * b["hum_uv"])
        broken = (m["railed"]
                  | (np.abs(m["dc_mv"] - b["dc_mv"]) > BROKEN_DC_MV)
                  | (m["noise_uv"] > noise_lim) | (m["hum_uv"] > hum_lim))
        return frozenset(int(i) for i in np.flatnonzero(
            broken & ~b["railed"] & self.use))

    def _bias_shift(self, m, broken, ref):
        """Median DC step (mV) of the working leads from `ref` when they all
        moved the same way, else 0."""
        live = [i for i in range(BUNDLE) if self.use[i]
                and not self.base["railed"][i] and i not in broken]
        if len(live) < 3:
            return 0.0
        d = m["dc_mv"][live] - ref[live]
        if np.all(d > 0) or np.all(d < 0):
            return float(np.median(d))
        return 0.0

    def _evaluate(self):
        m = self._window()
        broken = self._broken(m)
        self._dc_hist = (self._dc_hist + [m["dc_mv"]])[-8:]
        kind, i = self.steps[self.step_i]
        key = self._step_key(self.steps[self.step_i])
        live = frozenset(j for j in range(BUNDLE)
                         if self.use[j] and not self.base["railed"][j])
        if self.phase == "lift":
            if kind == "bias":
                if self._bias_ref is None:
                    hist = self._dc_hist
                    ref = hist[-3] if len(hist) >= 3 else (
                        hist[0] if hist else m["dc_mv"])
                else:
                    ref = self._bias_ref
                shift = self._bias_shift(m, broken, ref)
                stepped = abs(shift) >= BIAS_STEP_MV
                self._bias_ref = ref if stepped else None
                seen = ("broken", broken) if broken else (
                    ("shift", None) if stepped else None)
            else:
                seen = ("broken", broken) if broken else None
            self._seen.append(seen)
            last = self._seen[-HOLD_EVALS:]
            if (len(last) == HOLD_EVALS and last[0] is not None
                    and all(s == last[0] for s in last)):
                self.lifts[key] = self._judge(kind, i, last[0], live, m)
                self.phase = "return"
                self._seen, self._samples = [], 0
            elif self._samples >= self._lift_timeout:
                self.lifts[key] = {
                    "result": "fail",
                    "text": ("no change seen: is black on BIAS?"
                             if kind == "bias" else "no change seen")}
                self._next_step()
            return
        # return: the bundle looks like its baseline again
        back = not broken
        if kind == "bias" and self._bias_ref is not None:
            back = back and (abs(self._bias_shift(m, broken, self._bias_ref))
                             < BIAS_BACK_MV)
        self._seen.append(back)
        if len(self._seen) >= HOLD_EVALS and all(self._seen[-HOLD_EVALS:]):
            self._next_step()
        elif self._samples >= self._return_timeout:
            res = self.lifts.get(key, {})
            res.setdefault("notes", []).append(
                "didn't return to baseline after it was put back")
            self._next_step()

    def _judge(self, kind, i, seen, live, m):
        what, broken = seen
        chs = ", ".join(f"CH{self.channel(j)}" for j in sorted(broken or ()))
        if kind == "lead":
            if broken == {i}:
                return {"result": "ok", "text": f"= CH{self.channel(i)}"}
            if i in broken:
                others = sorted(broken - {i})
                oth = ", ".join(f"CH{self.channel(j)}" for j in others)
                if all(m["railed"][j] for j in others):
                    return {"result": "fail",
                            "text": f"also railed {oth}: leads shorted"}
                for j in others:
                    self.side_notes[j].append(
                        f"disturbed while {LEAD_COLOURS[i]} was out: "
                        "check its electrode")
                return {"result": "ok",
                        "text": f"= CH{self.channel(i)} ({oth} disturbed)"}
            return {"result": "fail",
                    "text": f"broke {chs}, not CH{self.channel(i)}: "
                            "check the header order"}
        if kind == "ref":
            if broken == live:
                return {"result": "ok", "text": "= REF (all leads broke)"}
            return {"result": "fail",
                    "text": f"only {chs} broke: is white on a REF pin?"}
        if what == "shift":
            step = self._bias_shift(m, frozenset(), self._bias_ref)
            return {"result": "ok",
                    "text": f"= BIAS (all leads stepped {step:+.1f} mV)"}
        return {"result": "fail",
                "text": f"lifting black broke {chs}: is black on a "
                        "channel pin?"}

    # ---- reading ---------------------------------------------------------- #
    def status(self):
        """What to show now: {"phase", "title", "detail", "progress":
        [(colour, "ok"/"fail"/"skipped"/None)], "can_skip"}."""
        fs = self.fs
        prog = [(self._step_key(s), self.lifts.get(self._step_key(s),
                                                   {}).get("result"))
                for s in self.steps]
        out = {"phase": self.phase, "progress": prog, "can_skip": False,
               "bundle": self.name}
        if self.phase in ("baseline", "final"):
            left = max(0.0, (self._n_base - self._samples) / fs)
            out["title"] = "HANDS OFF"
            out["detail"] = (f"{'baseline' if self.phase == 'baseline' else 'final check'}"
                             f"  {left:.0f} s")
        elif self.phase in ("lift", "return"):
            kind, i = self.steps[self.step_i]
            key = self._step_key(self.steps[self.step_i])
            role = (f"CH{self.channel(i)}" if kind == "lead" else
                    "REF" if kind == "ref" else "BIAS")
            if self.phase == "lift":
                left = max(0.0, (self._lift_timeout - self._samples) / fs)
                out["title"] = f"LIFT {key.upper()}"
                out["detail"] = f"{role} out of the bath  ·  {left:.0f} s"
            else:
                r = self.lifts.get(key, {})
                mark = "✓" if r.get("result") == "ok" else "✗"
                out["title"] = f"{mark} {key.upper()} {r.get('text', '')}"
                out["detail"] = "put it back in the bath"
            out["can_skip"] = True
        else:
            out["title"] = "DONE"
            out["detail"] = ""
        return out

    def result(self):
        """The report (JSON-safe): per lead its mapping and grade."""
        m = self.final if self.final is not None else self.base
        leads_out, bundle_notes = [], []
        if m is not None:
            grades, bundle_notes = grade(m)
        for i in range(BUNDLE):
            key = LEAD_COLOURS[i]
            lift = self.lifts.get(key, {"result": None, "text": "not tested"})
            if not self.use[i]:
                lift = {"result": "not selected", "text": "not selected"}
            entry = {"colour": key, "channel": self.channel(i),
                     "mapping": lift.get("result"),
                     "mapping_text": lift.get("text")}
            notes = list(lift.get("notes", [])) + self.side_notes[i]
            if m is not None:
                g, gnotes = grades[i]
                if self.side_notes[i] and g == "pass" and self.use[i]:
                    g = "warn"
                entry.update(
                    grade=g, dc_mv=round(float(m["dc_mv"][i]), 2),
                    own_noise_uv=round(float(m["own_noise_uv"][i]), 2),
                    noise_uv=round(float(m["noise_uv"][i]), 2),
                    hum_uv=round(float(m["hum_uv"][i]), 2),
                    jumps=int(m["jumps"][i]), railed=bool(m["railed"][i]))
                notes += gnotes
            entry["notes"] = notes
            leads_out.append(entry)
        shared = {}
        for key, role in ((REF_COLOUR, "ref"), (BIAS_COLOUR, "bias")):
            lift = self.lifts.get(key, {"result": None, "text": "not tested"})
            shared[role] = {"colour": key, "mapping": lift.get("result"),
                            "mapping_text": lift.get("text"),
                            "notes": list(lift.get("notes", []))}
        mappings = [e["mapping"] for e in leads_out
                    if e["mapping"] != "not selected"] + [
            s["mapping"] for s in shared.values()]
        passed = (not self.cancelled and self.final is not None
                  and all(x == "ok" for x in mappings)
                  and all(e.get("grade") != "fail" for e in leads_out))
        return {
            "kind": "ironbci_saline_check",
            "bundle": self.name, "first_channel": self.bundle_first,
            "started": self.started.isoformat(timespec="seconds"),
            "sample_rate": self.fs, "mains_hz": self.line,
            "cancelled": self.cancelled,
            "completed": self.final is not None,
            "passed": passed,
            "leads": leads_out, **shared,
            "selected_channels": [self.channel(i) for i in range(BUNDLE)
                                  if self.use[i]],
            "common_uv_baseline": (round(self.base["common_uv"], 2)
                                   if self.base is not None else None),
            "common_uv_final": (round(self.final["common_uv"], 2)
                                if self.final is not None else None),
            "notes": bundle_notes,
        }
