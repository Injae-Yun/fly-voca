"""Where does the steepness come from -- our graph, or the model itself?

Sweeping w_syn on our connectome, MN9 halves between 0.18 and 0.16 while the
trial-to-trial spread doubles. That is a network sitting near a bifurcation.
Calibrating w_syn to the benchmark would park the model right there.

Running the same sweep on the reference matrix separates the two explanations:

  both steep      criticality is a property of this model class, and the
                  published model lives near the same edge
  only ours steep our graph carries excess recurrent gain, and matching the
                  connectome's counting convention is the real fix

The diagnostic is the local log-log slope d ln(rate) / d ln(w_syn): how much
gain the network returns for a change in synaptic weight.
"""
import numpy as np
import pandas as pd

from flyvoca.graph import build
from bodysnatch.sim import Brain, Params
from bodysnatch import reference as ref

W_GRID = (0.12, 0.14, 0.16, 0.18, 0.20, 0.24, 0.275, 0.32)

meta, W_ours, _, _ = build()                      # threshold=False
ours = (meta.loc[[i for i in ref.SUGAR if i in meta.index], "idx"].values,
        int(meta.loc[ref.MN9, "idx"]), W_ours)

f2i, W_ref = ref.matrix()
theirs = (np.array([f2i[f] for f in ref.SUGAR if f in f2i]),
          f2i[ref.MN9], W_ref)

rows = []
for name, (stim, mn9, W) in (("ours", ours), ("reference", theirs)):
    for w in W_GRID:
        brain = Brain(W, Params(w_syn=w))
        c = brain.run(stim=stim, t_run=1000.0, n_trials=30, r_poi=100.0, seed=1)
        m, s = Brain.rate(c)
        rows.append({"graph": name, "w_syn": w,
                     "MN9": round(float(m[mn9]), 1),
                     "cv": round(float(s[mn9] / max(m[mn9], 1e-9)), 3),
                     "peak": round(float(m.max())),
                     "active": int((m > 0.5).sum())})
        print(rows[-1], flush=True)

df = pd.DataFrame(rows)
df["slope"] = np.nan
for name, g in df.groupby("graph"):
    ln_w, ln_r = np.log(g.w_syn.values), np.log(np.maximum(g.MN9.values, 1e-9))
    df.loc[g.index, "slope"] = np.gradient(ln_r, ln_w).round(1)

print()
for name, g in df.groupby("graph", sort=False):
    print(f"--- {name} ---")
    print(g.drop(columns="graph").to_string(index=False))
    print()

print("reference published: MN9 67.0 Hz, peak 114.3 Hz, active 357, at w_syn=0.275")
print("slope = d ln(MN9) / d ln(w_syn); 1.0 would be proportional, >5 is a cliff")
