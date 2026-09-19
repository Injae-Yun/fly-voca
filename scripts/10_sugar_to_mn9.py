"""Does the simulated brain bridge sugar -> proboscis extension?

`LB3 -> MN9` is zero in the wiring diagram: proboscis extension is polysynaptic
via the SEZ. So this is a real test of the dynamics rather than of the graph.
Shiu et al. report that stimulating sugar-sensing neurons drives MN9; if our
implementation is sound, MN9 should come up here too.

v783 types the sugar neurons as LB3 (left 64 / right 58), which supersedes the
21 hard-coded root IDs in the reference notebook.
"""
import time

import numpy as np
import pandas as pd

from flyvoca.graph import build, select
from bodysnatch.sim import Brain, Params

MN9 = 720575940660219265

meta, W, W_slow, stats = build()
brain = Brain(W)
print(f"device={brain.device}  neurons={brain.n}  synapse_entries={W.nnz}")

lb3 = select(meta, cell_type="LB3")
print(f"LB3 sugar neurons: {len(lb3)}  ({lb3.groupby('side').size().to_dict()})")

mn9_idx = int(meta.loc[MN9, "idx"])
label = meta.set_index("idx")[["label", "side", "super_class"]]

for rate_hz in (150.0, 100.0):
    t0 = time.time()
    counts = brain.run(stim=lb3["idx"].values, t_run=1000.0, n_trials=30, r_poi=rate_hz)
    mean, std = Brain.rate(counts)
    dt = time.time() - t0

    active = int((mean > 0.5).sum())
    print(f"\n=== sugar @ {rate_hz:.0f} Hz ===  ({dt:.1f}s for 30 trials x 1s)")
    print(f"neurons above 0.5 Hz: {active}   total spikes/s: {mean.sum():,.0f}")
    print(f"MN9 ({MN9}): {mean[mn9_idx]:.1f} +/- {std[mn9_idx]:.1f} Hz")

    top = np.argsort(mean)[::-1][:25]
    df = label.loc[top].copy()
    df["Hz"] = mean[top].round(1)
    df["sd"] = std[top].round(1)
    print(df.to_string(index=False))
