"""T5 -- regression checks after T1/T4.

Nothing in flyvoca/*.py or bodysnatch/sim.py was touched by this debugging
pass (T4's build_dale() lives in its own script, 87_dale_test.py). These
checks confirm that is true operationally, not just by diff:

  1. BANC: looming still moves leg motor neurons the way it did before this
     session (the positive control the whole trace in 86_trace_sugar.py
     leans on).
  2. FAFB: the edge-level sugar -> MN9 benchmark (doc 02, bodysnatch/scripts/
     14_calibrate_wsyn.py) is unchanged -- same w_syn=0.275, same ref-20
     stim, same MN9 readout.
"""
import numpy as np
import pandas as pd

from flyvoca.banc_graph import build as banc_build, select as banc_select, \
    leg_motor, spontaneous_groups
from flyvoca.graph import build as fafb_build
from bodysnatch.sim import Brain, Params
from bodysnatch.reference import SUGAR as REF_SUGAR, MN9

print("=== 1. BANC: looming -> leg motor neurons (positive control) ===")
meta, W, _, st = banc_build()
print("graph:", st)
groups = spontaneous_groups(meta)
lm = leg_motor(meta)["idx"].values
sel = lambda **kw: banc_select(meta, **kw)["idx"].values
LOOM = np.concatenate([sel(cell_type="LC4"), sel(cell_type="LPLC2")])
print(f"leg motor {len(lm)}  looming stim {len(LOOM)}")

b = Brain(W, Params(w_syn=0.16, sigma_v=2.0, r_spont=0.0))
b.spont_groups = groups
TRIALS = 16
SEED = 5000
base = b.run(t_run=600.0, n_trials=TRIALS, seed=SEED) / 0.6
loom = b.run(stim=LOOM, t_run=600.0, n_trials=TRIALS, r_poi=100.0, seed=SEED) / 0.6
bm, bs = base.mean(axis=1), base.std(axis=1, ddof=1)
m = loom.mean(axis=1)
sd = np.sqrt((loom.std(axis=1, ddof=1) ** 2 + bs ** 2) / 2)
d = (m - bm) / np.maximum(sd, 0.05)
n_hit = int((np.abs(d[lm]) > 2).sum())
print(f"looming legMN_d>2 = {n_hit} (baseline for this bug report: 38)")
print(f"legMN base {float(bm[lm].mean()):.2f} Hz -> stim {float(m[lm].mean()):.2f} Hz, "
      f"max|d| {float(np.abs(d[lm]).max()):.1f}")

print("\n=== 2. FAFB: edge-level sugar -> MN9 benchmark (doc 02) ===")
meta_f, W_f, _, st_f = fafb_build()
print("graph:", st_f)
ref = [i for i in REF_SUGAR if i in meta_f.index]
stim = meta_f.loc[ref, "idx"].values
mn9 = int(meta_f.loc[MN9, "idx"])
brain_f = Brain(W_f, Params(w_syn=0.275, sigma_v=0.0, r_spont=0.0))
c = brain_f.run(stim=stim, t_run=1000.0, n_trials=30, r_poi=100.0, seed=1)
mf, sf = Brain.rate(c)
print(f"ref-20 @100Hz, w_syn=0.275: MN9 = {float(mf[mn9]):.1f} Hz "
      f"(sd {float(sf[mn9]):.1f}); doc-02 measured 71.9, published target 67.0")

print("\ndone (regression)")
