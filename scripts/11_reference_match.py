"""Compare against the published reference condition.

Shiu et al. stimulate 21 hand-listed sugar neurons (one side); 20 of them
survive into v783. Driving all 122 LB3 cells instead saturates the network --
150 Hz and 100 Hz then give nearly the same answer, which is a symptom, not a
result. Matching their condition restores the dose-response and makes the
numbers comparable to the paper.
"""
import numpy as np
import pandas as pd

from flyvoca.graph import build, select
from bodysnatch.sim import Brain

MN9 = 720575940660219265
REF_SUGAR = [  # from the reference notebook; one dropped in v783
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]

meta, W, _, _ = build()
brain = Brain(W)
mn9 = int(meta.loc[MN9, "idx"])

ref = [i for i in REF_SUGAR if i in meta.index]
ref_idx = meta.loc[ref, "idx"].values
all_lb3 = select(meta, cell_type="LB3")["idx"].values
print(f"reference set: {len(ref)}/21 alive   full LB3: {len(all_lb3)}")

rows = []
for name, stim in (("ref-20", ref_idx), ("LB3-122", all_lb3)):
    for hz in (150.0, 100.0, 50.0):
        c = brain.run(stim=stim, t_run=1000.0, n_trials=30, r_poi=hz, seed=1)
        m, s = Brain.rate(c)
        rows.append({"stim": name, "drive_Hz": hz, "MN9_Hz": round(float(m[mn9]), 1),
                     "MN9_sd": round(float(s[mn9]), 1),
                     "active>0.5Hz": int((m > 0.5).sum()),
                     "net_Hz": int(m.sum())})
        print(rows[-1])

print()
print(pd.DataFrame(rows).pivot(index="drive_Hz", columns="stim",
                               values=["MN9_Hz", "active>0.5Hz"]).to_string())

# The reference's lesion test: silence the 3 most active cells, watch MN9 fall.
c = brain.run(stim=ref_idx, t_run=1000.0, n_trials=30, r_poi=100.0, seed=1)
m, _ = Brain.rate(c)
base = float(m[mn9])
top3 = np.argsort(m)[::-1][:3]
lab = meta.set_index("idx")["label"]
print(f"\nlesion test (ref-20 @100Hz), baseline MN9 = {base:.1f} Hz")
for i in top3:
    c2 = brain.run(stim=ref_idx, t_run=1000.0, n_trials=30, r_poi=100.0,
                   silence=[int(i)], seed=1)
    m2, _ = Brain.rate(c2)
    print(f"  silence {lab.loc[i]:<12} ({m[i]:6.1f} Hz)  ->  MN9 {float(m2[mn9]):6.1f} Hz"
          f"  ({float(m2[mn9]) - base:+.1f})")
