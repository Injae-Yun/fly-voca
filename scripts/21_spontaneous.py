"""Give the brain a resting state.

A silent network has nothing to modulate: multiply zero by any gain and it is
still zero. And no real neuron is silent -- ion channels gate stochastically,
vesicles release spontaneously, and sensory afferents fire without a stimulus.

Membrane noise enters as an exact Ornstein-Uhlenbeck step, so `sigma_v` is the
stationary sd in mV whatever dt is. The gap from rest to threshold is 7 mV.

Two things are watched that are *not* being fitted:

  Kenyon cells   the mushroom body codes sparsely; KCs are near-silent at rest
                 in the real animal. If noise makes them chatter, it is too high.
  runaway        this network sits near its transition (docs/02), so noise could
                 self-ignite it. A real brain does not seize when left alone.
"""
import numpy as np
import pandas as pd

from flyvoca.graph import build
from flyvoca.sim import Brain, Params

meta, W, _, _ = build()
central = meta[meta["super_class"] == "central"]["idx"].values
kc = meta[meta["klass"] == "Kenyon_Cell"]["idx"].values
sensory = meta[meta["super_class"].isin(["sensory", "sensory_ascending"])]["idx"].values
dn = meta[meta["super_class"] == "descending"]["idx"].values
print(f"central {len(central):,}  Kenyon {len(kc):,}  sensory {len(sensory):,}  DN {len(dn):,}")

rows = []
for sigma in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
    brain = Brain(W, Params(sigma_v=sigma))
    c = brain.run(t_run=1000.0, n_trials=10, seed=3)
    m, _ = Brain.rate(c)
    rows.append({
        "sigma_v": sigma,
        "net_kHz": round(float(m.sum()) / 1000, 1),
        "central_med": round(float(np.median(m[central])), 2),
        "central_mean": round(float(m[central].mean()), 2),
        "frac>0.5Hz": round(float((m > 0.5).mean()), 3),
        "KC_mean": round(float(m[kc].mean()), 3),
        "KC_frac>1Hz": round(float((m[kc] > 1).mean()), 4),
        "DN_mean": round(float(m[dn].mean()), 2),
    })
    print(rows[-1], flush=True)

print("\n" + pd.DataFrame(rows).to_string(index=False))
print("\nreal fly: KCs near-silent at rest (sparse coding); central neurons a few Hz")
