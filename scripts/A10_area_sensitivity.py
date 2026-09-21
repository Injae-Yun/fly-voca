"""Cell-specific input sensitivity from the one thing that is measured: size.

Doc 19: by the eigenvalue criterion the pC1d/e <-> aIPg loop is 2x short of
self-sustaining, and a 4x input sensitivity on pC1d/e alone would cross 1.
The question is whether anything measured gives pC1d/e that. Surface area
is measured (Codex cell_stats), and cable theory says input resistance --
hence PSP per synapse -- scales as 1/area. So s_j = median_area / area_j is
a per-cell sensitivity with no free parameter beyond the reference cell
(the median central-brain neuron, so the calibrated w_syn keeps its meaning
for a typical cell).

The linear prediction is already unfavourable (loop lambda x gamma falls
from 0.38-0.50 to 0.14-0.19: pC1d/e are large, s ~ 0.2). This script measures
what the whole network does with that heterogeneity, because the benchmarks
and the resting state are not linear predictions:

    uniform          every earlier document
    area-scaled      s_j on every synapse onto j, w_syn unchanged

for w_syn 0.2 and 0.4: resting rate, sugar -> MN9, looming -> motor, loop
arms and round-trip gain on pC1d/e <-> aIPg, and persistence after 1 s of
loop drive. Sensory afferents get s ~ 4 and motor neurons s ~ 0.3, so the
reflex benchmarks are the part most likely to move.
"""
import numpy as np
import pandas as pd

from flyvoca.graph import build
from flyvoca.paths import cache
from flyvoca.sim import Brain, Params

TRIALS, T = 8, 800.0
meta, W, _, _ = build()
t = meta["cell_type"].astype(str); add = meta["additional_type(s)"].astype(str)
ex = lambda *n: meta[t.isin(n)]["idx"].values
PC1DE = np.concatenate([ex("pC1d"), ex("pC1e")])
AIPG = meta[add.str.contains("aIPg")]["idx"].values
LOOP = np.unique(np.concatenate([PC1DE, AIPG]))
SUGAR = meta[t.str.startswith("LB3", na=False)]["idx"].values
MN9 = meta[meta.index == 720575940660219265]["idx"].values
LOOM = ex("LC4", "LPLC2")
MOTOR = meta[meta.super_class == "motor"]["idx"].values
sens = pd.read_csv(cache("input_sensitivity_area.csv")).set_index("root_id")["s_input"].reindex(meta.index).fillna(1.0).values
print(f"s: p5 {np.percentile(sens,5):.2f}  p50 {np.median(sens):.2f}  p95 {np.percentile(sens,95):.2f}   "
      f"pC1d/e {np.round(sens[PC1DE],2).tolist()}  aIPg median {np.median(sens[AIPG]):.2f}  "
      f"LB3 {np.median(sens[SUGAR]):.2f}  MN9 {sens[MN9][0]:.2f}  motor {np.median(sens[MOTOR]):.2f}")


def gain(b, seed=23000):
    out = {}
    for src, dst, arm in ((PC1DE, AIPG, "fwd"), (AIPG, PC1DE, "back")):
        pts = []
        for rate in (50.0, 150.0):
            r = b.run(stim=src, t_run=T, n_trials=TRIALS, r_poi=rate, seed=seed).mean(axis=1) / (T / 1000)
            pts.append((float(r[src].mean()), float(r[dst].mean())))
        (x1, y1), (x2, y2) = pts
        out[arm] = (y2 - y1) / max(x2 - x1, 1e-9)
    out["gain"] = out["fwd"] * out["back"]
    return out


def persist(b, seed=24000, on=4, off=8, block=250.0):
    state, tr = None, []
    for k in range(on + off):
        c, state = b.run(stim=LOOP if k < on else (), t_run=block, n_trials=TRIALS, r_poi=100.0,
                         seed=seed + k, state=state, return_state=True)
        tr.append(float((c / (block / 1000))[PC1DE].mean()))
    return {"on": round(np.mean(tr[:on]), 1), "off1": round(tr[on], 2), "off_late": round(np.mean(tr[-4:]), 2)}


def bench(b, seed=22001):
    base = b.run(t_run=T, n_trials=TRIALS, seed=seed) / (T / 1000)
    bm, bs = base.mean(axis=1), base.std(axis=1, ddof=1)
    def effect(stim, target):
        r = b.run(stim=stim, t_run=T, n_trials=TRIALS, r_poi=100.0, seed=seed) / (T / 1000)
        d = (r.mean(axis=1) - bm) / np.maximum(np.sqrt((r.std(axis=1, ddof=1) ** 2 + bs ** 2) / 2), 0.05)
        return float(np.abs(d[target]).max()), float(r.mean(axis=1)[target].mean())
    mn9_d, mn9_hz = effect(SUGAR, MN9); loom_d, _ = effect(LOOM, MOTOR)
    return {"rest_Hz": round(float(bm.mean()), 2), "rest_central_Hz": round(float(bm[meta.super_class.values == "central"].mean()), 2),
            "sugar_MN9_d": round(mn9_d, 1), "sugar_MN9_Hz": round(mn9_hz, 1), "loom_motor_d": round(loom_d, 1)}


rows = []
for w in (0.20, 0.40):
    for cond, sc in (("uniform", None), ("area-scaled", sens)):
        b = Brain.from_meta(W, meta, Params(w_syn=w, sigma_v=2.0, r_spont=2.0)) if sc is None else \
            Brain(W, Params(w_syn=w, sigma_v=2.0, r_spont=2.0), input_scale=sc,
                  spont=meta[meta["super_class"].isin(["sensory", "sensory_ascending"])]["idx"].values)
        g = gain(b); p = persist(b); bm = bench(b)
        row = {"w_syn": w, "cond": cond, "fwd": round(g["fwd"], 3), "back": round(g["back"], 3),
               "gain": round(g["gain"], 4), **p, **bm}
        rows.append(row); print(row, flush=True)
df = pd.DataFrame(rows); print("\n" + df.to_string(index=False))
df.to_csv(cache("area_sensitivity.csv"), index=False)
print("\ndone (area sensitivity)")
