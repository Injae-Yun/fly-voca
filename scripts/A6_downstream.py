"""Where does the state live, if not in the loop? Scan the whole brain.

Chiu et al. 2024 (eLife 88598) imaged the cells docs 15-16 were trying to
make persist, and they do not: "neither aIPg nor pC1d neurons exhibit
persistent activity on the time scale necessary to sustain persistent
aggressive behavior". The loop's 250 ms collapse in the sim is therefore not
a defect to fix. The minutes-long state is carried "by other downstream
neurons that transform the output of aIPg or pC1d+e", and the same paper
gives a parameter-free pattern to test against:

    aIPg alone      -> persistent aggression (>= 10 min after 30 s)
    pC1d alone      -> time-locked only
    pC1d + pC1e     -> persistent

So the question changes from "why does pC1 not hold" to "who downstream
holds, and under what slow mechanism". Two passes over one design:

  gain 0     the fast graph alone. Who does each stimulus recruit, and does
             anyone at all outlast it. (Expected: recruitment yes,
             persistence no -- there is no slow element.)
  gain > 0   the monoaminergic edges switched on as a slow layer
             (`bodysnatch.monoamine`). The mechanism is brain-wide; if the
             cells that persist are specific to aIPg / pC1d+e and absent
             for pC1d alone, for sugar, and for 37 random cells, that is
             the anatomy speaking.

What is swept is what the data lacks -- receptor sign, gain, and tau. tau
runs at 20 s (inside Jung et al. 2020's 13-83 s) and at 2 s: the watch
window is 6 s, so at 20 s the layer's own charge cannot have decayed and
any persistence is the layer holding it; at 2 s the layer has decayed by
95% and anything still above sham is the network's doing. What is *not*
free: which cells carry the edges, and how many synapses.

Statistics, because a brain-wide scan with 8 trials will find something by
chance in 139,255 cells:
  * every driven trace is paired with a sham at the same seed (sim.py now
    keeps stimulus and background noise on separate streams, so the same
    seed means the same background);
  * a cell persists at 2 s if, in >= 3 of the 4 blocks 1-2 s after
    release, it is > 2 Hz above sham with Cohen's d > 2 under a Poisson
    variance floor, AND it was recruited during the stimulus by the same
    criterion;
  * a null arm (sham vs a second sham at another seed) is run per
    parameter set and its count is the threshold, not a literal number;
  * benchmarks run with the layer settled at the resting network's steady
    state, and self-ignition is read two-sided under the same gates.
"""
import json
import numpy as np
import pandas as pd

from flyvoca.graph import build
from flyvoca.paths import cache
from bodysnatch.sim import Brain, Params
from bodysnatch.monoamine import split_by_transmitter, SlowEdges

import os
SMOKE = bool(os.environ.get("A6_SMOKE"))
W_SYN, TRIALS, BLOCK = 0.20, (3 if SMOKE else 8), 250.0
ON, OFF = (8, 24) if not SMOKE else (2, 4)       # 2 s on, 6 s watch
Q = 1000.0 / BLOCK                               # Hz per spike per block
SIGN_SETS = {"+": {"OCT": +1, "SER": +1, "DA": +1},
             "mix": {"OCT": +1, "SER": +1, "DA": -1},
             "-": {"OCT": -1, "SER": -1, "DA": -1}}
#              signs  gain  tau
COMBOS = [("none", 0.0, 20.0), ("+", 3.0, 20.0), ("+", 6.0, 20.0), ("mix", 6.0, 20.0),
          ("-", 6.0, 20.0), ("+", 6.0, 2.0)]
if SMOKE:
    COMBOS = [("none", 0.0, 20.0), ("+", 6.0, 2.0)]
# Overrides for follow-up runs (e.g. Chiu's 30 s drive on one parameter set):
#   A6_ON=120  A6_COMBOS="none:0:20,+:6:20"  A6_STIMS="aIPg,pC1d+e,pC1d,sugar"
if os.environ.get("A6_ON"):
    ON = int(os.environ["A6_ON"])
if os.environ.get("A6_COMBOS"):
    COMBOS = [(a, float(b), float(c)) for a, b, c in
              (x.split(":") for x in os.environ["A6_COMBOS"].split(","))]
assert COMBOS[0][1] == 0.0
TAG = os.environ.get("A6_TAG", "")
# Half-saturation is normally stated relative to the stimulus length. For a
# long-drive follow-up keep the 2 s constant, so a longer stimulus charges
# *more* rather than being renormalised away.
CHARGE_S = float(os.environ.get("A6_CHARGE_S", ON * BLOCK / 1000))
SEED_TRACE, SEED_NULL, SEED_BENCH = 26000, 28000, 22001

meta, W, Ws, _ = build()
t = meta["cell_type"].astype(str)
add = meta["additional_type(s)"].astype(str)
ex = lambda *n: meta[t.isin(n)]["idx"].values
n = W.shape[0]
PC1D = ex("pC1d")
PC1DE = np.concatenate([ex("pC1d"), ex("pC1e")])
AIPG = meta[add.str.contains("aIPg")]["idx"].values
LOOP = np.unique(np.concatenate([PC1DE, AIPG]))
SUGAR = meta[t.str.startswith("LB3", na=False)]["idx"].values
MN9 = meta[meta.index == 720575940660219265]["idx"].values
LOOM = ex("LC4", "LPLC2")
MOTOR = meta[meta.super_class == "motor"]["idx"].values
DN = meta[meta.super_class == "descending"]["idx"].values
SPONT = meta[meta["super_class"].isin(["sensory", "sensory_ascending"])]["idx"].values
# size-matched control: 37 central-brain cells that are not aIPg, fixed seed
rng = np.random.default_rng(6)
pool = meta[(meta.super_class == "central") & ~add.str.contains("aIPg") & ~t.str.startswith("pC1")]["idx"].values
RAND37 = np.sort(rng.choice(pool, size=len(AIPG), replace=False))
STIMS = {"pC1d": PC1D, "pC1d+e": PC1DE, "aIPg": AIPG, "loop": LOOP,
         "sugar": SUGAR, "rand37": RAND37}
if os.environ.get("A6_STIMS"):
    STIMS = {k: STIMS[k] for k in os.environ["A6_STIMS"].split(",")}
WT = split_by_transmitter(Ws, meta["slow_class"].values)
print(f"ON {ON} blocks  OFF {OFF}  charge_s {CHARGE_S}  combos {COMBOS}")
print(f"pC1d {len(PC1D)}  pC1d+e {len(PC1DE)}  aIPg {len(AIPG)}  sugar {len(SUGAR)}  "
      f"rand37 {len(RAND37)}  DN {len(DN)}  slow edges "
      + ", ".join(f"{k} {M.nnz:,}" for k, M in WT.items()))

brain = Brain(W, Params(w_syn=W_SYN, sigma_v=2.0, r_spont=2.0), spont=SPONT)
rest = brain.run(t_run=1000.0, n_trials=TRIALS, seed=SEED_BENCH).mean(axis=1) / 1.0
print(f"resting network {rest.mean():.2f} Hz (for settling the slow layer)")


def make(gain, tau, signs):
    if gain <= 0:
        return None
    s = SlowEdges(WT, tau, gain, signs, n_trials=TRIALS, charge_s=CHARGE_S)
    s.settle(rest)
    return s


def trace(stim, slow, seed):
    """Block-wise run with the slow layer fed back. Returns (blocks, n, trials)
    of rates in Hz, and the layer's saturation per block."""
    state, out, sat = None, [], []
    for k in range(ON + OFF):
        c, state = brain.run(stim=stim if k < ON else (), t_run=BLOCK, n_trials=TRIALS,
                             r_poi=100.0, seed=seed + k, state=state,
                             v_offset=slow.v_offset() if slow else None, return_state=True)
        r = (c / (BLOCK / 1000)).astype(np.float32)
        if slow:
            slow.step(r, BLOCK / 1000)
            sat.append(slow.saturation())
        out.append(r)
    return np.stack(out), sat


def d_vs(a, s):
    """Cohen's d per cell for one block, a and s (n, trials), with a Poisson
    floor on the pooled sd: with 250 ms blocks the rate quantum is 4 Hz, so a
    cell firing exactly one spike in every trial has zero sample variance,
    and a constant floor would hand it d = 80."""
    sd = np.sqrt((a.std(axis=1, ddof=1) ** 2 + s.std(axis=1, ddof=1) ** 2) / 2)
    lam = (a.mean(axis=1) + s.mean(axis=1)) / 2
    sd = np.maximum(sd, np.sqrt(Q * np.maximum(lam, Q / 2)))
    return (a.mean(axis=1) - s.mean(axis=1)) / sd


def above(a, s):
    """Cells > 2 Hz above sham with d > 2 in this block (bool, n)."""
    return (d_vs(a, s) > 2) & (a.mean(axis=1) - s.mean(axis=1) > 2)


def stable(tr, sham, blocks, keep, need):
    """Cells above sham in >= `need` of `blocks`."""
    blocks = list(blocks)
    hits = sum(above(tr[k], sham[k]).astype(int) for k in blocks)
    return np.where(keep & (hits >= min(need, len(blocks))))[0]


def types(idx, k=8):
    return ", ".join(f"{a}:{b}" for a, b in
                     t.iloc[idx].value_counts().head(k).items()) if len(idx) else "-"


def analyse(tr, sham, stim):
    """Recruitment during the stimulus and persistence after it, same gates."""
    keep = np.ones(n, dtype=bool); keep[stim] = False
    rec = stable(tr, sham, range(ON), keep, 3)
    row = {"recruited": len(rec), "recruited_DN": int(np.isin(DN, rec).sum()),
           "recruited_types": types(rec)}
    p1 = stable(tr, sham, [ON], keep, 1)
    p2_any = stable(tr, sham, range(ON + min(4, OFF - 1), ON + min(8, OFF)), keep, 3)
    p2 = p2_any[np.isin(p2_any, rec)]
    late = stable(tr, sham, range(ON + OFF - 4, ON + OFF), keep, 3)
    late = late[np.isin(late, rec)]
    row.update({"persist_off1": len(p1), "persist_2s_any": len(p2_any),
                "persist_2s": len(p2), "persist_late": len(late),
                "persist_2s_DN": int(np.isin(DN, p2).sum()),
                "persist_2s_types": types(p2)})
    if len(p2):
        hz = tr[:, p2].mean(axis=(1, 2)); shz = sham[:, p2].mean(axis=(1, 2))
        row.update({"set_on_Hz": round(float(hz[:ON].mean()), 1),
                    "set_off1_Hz": round(float(hz[ON]), 2),
                    "set_2s_Hz": round(float(hz[ON + 4:ON + 8].mean()), 2),
                    "set_late_Hz": round(float(hz[-4:].mean()), 2),
                    "set_sham_Hz": round(float(shz[ON:].mean()), 2)})
        # decay of the persisting set's elevation over the off phase, in s
        el = hz[ON:] - shz[ON:]
        tt = np.arange(OFF) * BLOCK / 1000
        ok = el > 0.05 * max(el[0], 1e-6)
        if ok.sum() >= 4 and el[0] > 0:
            slope = np.polyfit(tt[ok], np.log(el[ok]), 1)[0]
            row["set_tau_s"] = round(float(-1.0 / slope), 1) if slope < 0 else np.inf
    for name, idx in (("pC1d+e", PC1DE), ("aIPg", AIPG), ("MN9", MN9)):
        d2 = np.mean([d_vs(tr[k], sham[k])[idx].mean() for k in range(ON + min(4, OFF - 1), ON + min(8, OFF))])
        row[f"{name}_d_2s"] = round(float(d2), 1)
    dn2 = np.mean([np.abs(d_vs(tr[k], sham[k])[DN]).max() for k in range(ON + min(4, OFF - 1), ON + min(8, OFF))])
    row["DN_maxd_2s"] = round(float(dn2), 1)
    return row, {"persist_2s": [int(meta.index[i]) for i in p2],
                 "persist_late": [int(meta.index[i]) for i in late]}


def bench(gain, tau, signs, seed=SEED_BENCH):
    """A0's benchmarks with the slow layer settled at the resting network."""
    slow = make(gain, tau, signs)
    off = slow.v_offset() if slow else None
    T = 800.0
    base = brain.run(t_run=T, n_trials=TRIALS, seed=seed, v_offset=off) / (T / 1000)
    bm, bs = base.mean(axis=1), base.std(axis=1, ddof=1)
    def effect(stim, target):
        r = brain.run(stim=stim, t_run=T, n_trials=TRIALS, r_poi=100.0, seed=seed,
                      v_offset=off) / (T / 1000)
        d = (r.mean(axis=1) - bm) / np.maximum(
            np.sqrt((r.std(axis=1, ddof=1) ** 2 + bs ** 2) / 2), 0.05)
        return float(np.abs(d[target]).max())
    out = {"rest_Hz": round(float(bm.mean()), 2),
           "sugar_MN9_d": round(effect(SUGAR, MN9), 1),
           "loom_motor_d": round(effect(LOOM, MOTOR), 1)}
    if slow:
        o = off.mean(axis=1)
        out.update({"off_mean_mV": round(float(o.mean()), 3),
                    "off_p99_mV": round(float(np.quantile(o, 0.99)), 2),
                    "off_min_mV": round(float(o.min()), 2)})
    return out


rows, ids = [], {}
sham0 = None
for sname, gain, tau in COMBOS:
    signs = SIGN_SETS[sname] if gain > 0 else SIGN_SETS["+"]
    print(f"\n=== signs {sname}  gain {gain} mV  tau {tau} s ===", flush=True)
    bm = bench(gain, tau, signs)
    print(f"  rest: {bm}", flush=True)
    sham, ssat = trace((), make(gain, tau, signs), seed=SEED_TRACE)
    null, _ = trace((), make(gain, tau, signs), seed=SEED_NULL)
    keep_all = np.ones(n, dtype=bool)
    nr, _ = analyse(null, sham, np.array([], dtype=np.int64))
    null_row = {"null_off1": nr["persist_off1"], "null_2s_any": nr["persist_2s_any"],
                "null_2s": nr["persist_2s"]}
    print(f"  null arm (sham vs sham'): {null_row}", flush=True)
    if sham0 is None:
        sham0 = sham
        ign = {}
    else:
        # self-ignition: the layer alone vs no layer, two-sided, same gates
        up = stable(sham, sham0, range(ON + OFF - 4, ON + OFF), keep_all, 3)
        dn = stable(sham0, sham, range(ON + OFF - 4, ON + OFF), keep_all, 3)
        ign = {"ignite_up": len(up), "ignite_down": len(dn),
               "ignite_up_types": types(up, 5), "ignite_down_types": types(dn, 5)}
        print(f"  self-ignition vs gain 0: up {len(up)} ({ign['ignite_up_types']})  "
              f"down {len(dn)} ({ign['ignite_down_types']})  saturation {ssat[-1]}", flush=True)
    for stim_name, stim in STIMS.items():
        tr, sat = trace(stim, make(gain, tau, signs), seed=SEED_TRACE)
        row, sets = analyse(tr, sham, stim)
        row = {"signs": sname, "gain": gain, "tau": tau, "stim": stim_name, **row,
               **null_row, **bm, **{k: v for k, v in ign.items() if "types" not in k}}
        rows.append(row); ids[f"{sname}|{gain}|{tau}|{stim_name}"] = sets
        print({k: v for k, v in row.items() if "types" not in k}, flush=True)
        print(f"    recruited: {row['recruited_types']}", flush=True)
        print(f"    persist 2s: {row['persist_2s_types']}", flush=True)
    pd.DataFrame(rows).to_csv(cache(f"downstream_scan{TAG}.csv"), index=False)
    with open(cache(f"downstream_sets{TAG}.json"), "w") as f:
        json.dump(ids, f)

df = pd.DataFrame(rows)
cols = ["signs", "gain", "tau", "stim", "recruited", "recruited_DN", "persist_off1",
        "persist_2s_any", "persist_2s", "persist_late", "null_2s", "set_tau_s",
        "pC1d+e_d_2s", "aIPg_d_2s", "MN9_d_2s", "DN_maxd_2s",
        "rest_Hz", "sugar_MN9_d", "loom_motor_d", "ignite_up", "ignite_down"]
print("\n" + df[[c for c in cols if c in df]].to_string(index=False))
print("\n=== Chiu 2024 pattern: aIPg persists, pC1d alone does not, pC1d+e does ===")
print("    (exploratory: 6 parameter sets x a 5-way conjunction, no correction)")
for (sname, gain, tau), g in df.groupby(["signs", "gain", "tau"]):
    p = {r.stim: int(r.persist_2s) for r in g.itertuples()}
    thr = max(int(g.null_2s.iloc[0]) * 2, 5)
    ok = (p.get("aIPg", 0) > thr and p.get("pC1d+e", 0) > thr and p.get("pC1d", 0) <= thr
          and p.get("sugar", 0) <= thr and p.get("rand37", 0) <= thr)
    print(f"  {sname:>4} {gain:.0f} mV tau {tau:>4.0f}  threshold {thr}  persist_2s {p}  "
          f"->  {'MATCH' if ok else 'no'}")
print("\ndone (downstream)")
