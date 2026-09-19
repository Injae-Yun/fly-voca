"""Calibrate the model's one free parameter against the reference benchmark.

Our connectome export uses a more permissive confidence cut: measured against
the reference matrix the synapse count per connection is a near-constant
1.25x, stable from single synapses up past 100 (see core/docs/02-calibration.md).
That is a threshold difference, not a different measurement, so it is absorbed
by `w_syn` -- the only free parameter in the model.

Algebra says w_syn = 0.275 / 1.25 ~ 0.22 mV. We measure instead, because our
graph also carries 6.3M weak pairs the reference drops, which the scalar does
not account for.

Benchmark: MN9 at 67.0 Hz under the reference's 20 sugar neurons at 100 Hz.
Peak rate (~114 Hz there) is reported as an independent check -- it is not
being fitted.
"""
import numpy as np
import pandas as pd

from flyvoca.graph import build
from bodysnatch.sim import Brain, Params

TARGET_MN9 = 67.0
REF_PEAK = 114.3
MN9 = 720575940660219265
REF_SUGAR = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]

meta, W, _, _ = build()          # threshold=False: every connection
stim = meta.loc[[i for i in REF_SUGAR if i in meta.index], "idx"].values
mn9 = int(meta.loc[MN9, "idx"])

rows = []
for w_syn in (0.275, 0.24, 0.22, 0.20, 0.18, 0.16):
    brain = Brain(W, Params(w_syn=w_syn))
    c = brain.run(stim=stim, t_run=1000.0, n_trials=30, r_poi=100.0, seed=1)
    m, s = Brain.rate(c)
    rows.append({"w_syn": w_syn, "MN9_Hz": round(float(m[mn9]), 1),
                 "sd": round(float(s[mn9]), 1),
                 "peak_Hz": round(float(m.max())),
                 "active": int((m > 0.5).sum())})
    print(rows[-1], flush=True)

df = pd.DataFrame(rows)
print("\n" + df.to_string(index=False))
print(f"\ntarget: MN9 = {TARGET_MN9} Hz, peak ~{REF_PEAK} Hz (not fitted)")
best = df.iloc[(df.MN9_Hz - TARGET_MN9).abs().argmin()]
print(f"closest: w_syn={best.w_syn}  MN9={best.MN9_Hz} Hz  peak={best.peak_Hz} Hz")
print(f"algebra predicted 0.275 / 1.2495 = {0.275/1.2495:.3f}")
