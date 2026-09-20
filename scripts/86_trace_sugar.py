"""Stop guessing at causes; measure where the sugar signal dies.

Five candidates are now refuted -- a missed window, histamine, weak pairs,
the Kenyon mesh, and the Kenyon threshold. The last one also broke the
framing: raising the KC threshold by 6 mV halves Kenyon activity and leaves
the leg motor neurons at exactly 2.33 Hz. The mushroom body is a passenger,
not the engine, so "the brain is saturated and sugar is buried" was wrong.

This traces the chain instead. Sugar enters at LB3 and would have to cross
SEZ -> descending -> nerve cord -> leg motor neuron. Measuring the effect size
at each stage says which hop fails, and that is a fact rather than a candidate.

Looming is the positive control: it does reach the legs, so whatever the trace
shows for sugar has to be read against a signal that survives.
"""
import numpy as np
import pandas as pd

from flyvoca.banc_graph import build, select, leg_motor, spontaneous_groups
from bodysnatch.sim import Brain, Params

meta, W, _, st = build()
groups = spontaneous_groups(meta)
sel = lambda **kw: select(meta, **kw)["idx"].values

STAGES = {
    "stimulated": None,
    "SEZ/brain": meta[meta.super_class == "central_brain_intrinsic"]["idx"].values,
    "descending": meta[meta.super_class == "descending"]["idx"].values,
    "VNC": meta[meta.super_class == "ventral_nerve_cord_intrinsic"]["idx"].values,
    "leg motor": leg_motor(meta)["idx"].values,
}
CONDS = {
    "sugar": sel(prefix="LB3"),
    "looming": np.concatenate([sel(cell_type="LC4"), sel(cell_type="LPLC2")]),
    "co2": sel(cell_type="ORN_V"),
}

b = Brain(W, Params(w_syn=0.16, sigma_v=2.0, r_spont=0.0))
b.spont_groups = groups
base = b.run(t_run=600.0, n_trials=12, seed=11000) / 0.6
bm, bs = base.mean(axis=1), base.std(axis=1, ddof=1)

rows = []
for name, stim in CONDS.items():
    r = b.run(stim=stim, t_run=600.0, n_trials=12, r_poi=100.0, seed=11000) / 0.6
    d = (r.mean(axis=1) - bm) / np.maximum(
        np.sqrt((r.std(axis=1, ddof=1) ** 2 + bs ** 2) / 2), 0.05)
    for sname, idx in STAGES.items():
        ii = stim if idx is None else idx
        rows.append({"cond": name, "stage": sname, "n": len(ii),
                     "base_Hz": round(float(bm[ii].mean()), 2),
                     "stim_Hz": round(float(r.mean(axis=1)[ii].mean()), 2),
                     "max_|d|": round(float(np.abs(d[ii]).max()), 1),
                     "n_d>2": int((np.abs(d[ii]) > 2).sum()),
                     "frac_d>2": round(float((np.abs(d[ii]) > 2).mean()), 3)})
        print(rows[-1], flush=True)
    print()

df = pd.DataFrame(rows)
print(df.pivot(index="stage", columns="cond", values="n_d>2").reindex(STAGES).to_string())
print()
print(df.pivot(index="stage", columns="cond", values="max_|d|").reindex(STAGES).to_string())

print("\n=== does sugar even leave the SEZ? direct wiring check ===")
Wa = abs(W)
lb3 = sel(prefix="LB3")
dn = STAGES["descending"]
step = np.zeros(len(meta), dtype=np.float32); step[lb3] = 1.0
for h in range(1, 5):
    step = Wa.T @ step
    step[lb3] = 0
    n_hit = int((step[dn] > 0).sum())
    print(f"  hop {h}: reaches {n_hit:,} of {len(dn)} DNs, "
          f"weight {float(step[dn].sum()):,.0f}")
print("done (trace)")
