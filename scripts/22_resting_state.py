"""Two different sources of spontaneous activity, and what the brain does at rest.

Membrane noise and spontaneous afferent firing are not the same thing:

  sigma_v   channel gating and miniature release -- ungated, everywhere, and
            it does not carry information
  r_spont   sensory neurons firing without a stimulus, as ORNs really do --
            actual spikes, entering through the front door and propagating
            down the pathways the connectome lays out

The second should structure the resting state; the first should only warm it.
"""
import numpy as np
import pandas as pd

from flyvoca.graph import build
from bodysnatch.sim import Brain, Params

meta, W, _, _ = build()
sens = meta[meta["super_class"].isin(["sensory", "sensory_ascending"])]["idx"].values
kc = meta[meta["klass"] == "Kenyon_Cell"]["idx"].values
groups = {k: g["idx"].values for k, g in meta.groupby("super_class") if len(g) > 50}

CONDS = {
    "silent":       Params(),
    "noise2.0":     Params(sigma_v=2.0),
    "spont5Hz":     Params(r_spont=5.0),
    "spont5+n1.5":  Params(r_spont=5.0, sigma_v=1.5),
    "spont2+n2.0":  Params(r_spont=2.0, sigma_v=2.0),
}

rows, profile = [], {}
for name, prm in CONDS.items():
    brain = Brain(W, prm)
    c = brain.run(t_run=1000.0, n_trials=10, seed=3,
                  spont=sens if prm.r_spont else ())
    m, _ = Brain.rate(c)
    rows.append({"cond": name, "net_kHz": round(float(m.sum()) / 1000, 1),
                 "frac>0.5Hz": round(float((m > 0.5).mean()), 3),
                 "KC>1Hz": round(float((m[kc] > 1).mean()), 4),
                 "KC_mean": round(float(m[kc].mean()), 3)})
    profile[name] = {k: round(float(m[v].mean()), 2) for k, v in groups.items()}
    print(rows[-1], flush=True)

print("\n" + pd.DataFrame(rows).to_string(index=False))
print("\n=== mean Hz by super_class at rest ===")
print(pd.DataFrame(profile).to_string())
