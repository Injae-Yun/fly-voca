"""Recalibrate the operating point for BANC.

`w_syn = 0.20` was chosen against FAFB: its synapse-count convention, its E/I
ratio of 1.53, its 3.7M pairs. BANC differs on all three -- 1.334 and 12.4M --
so carrying the number over was never justified, and the symptom shows: leg
motor neurons idle at 3.76 Hz with a peak of 93.5 before any stimulus, which
is a network running hot.

Same criterion as doc 02: find the transition, then sit just above it, where
the network is responsive without saturating. Measured at rest, because a
resting brain that already burns is the thing to fix.

Kenyon cells come along as the unfitted check they were in doc 03 -- the
mushroom body codes sparsely, so they should stay near-silent whatever w_syn
does to the rest.
"""
import numpy as np
import pandas as pd

from flyvoca.banc_graph import build, select, leg_motor, spontaneous_groups
from flyvoca.sim import Brain, Params

meta, W, _, st = build()
print("graph:", st)
groups = spontaneous_groups(meta)

lm = leg_motor(meta)["idx"].values
dn = meta[meta.super_class == "descending"]["idx"].values
vnc = meta[meta.super_class == "ventral_nerve_cord_intrinsic"]["idx"].values
kc = meta[meta.cell_class.astype(str).str.contains("kenyon", case=False, na=False)]["idx"].values
brain_i = meta[meta.super_class == "central_brain_intrinsic"]["idx"].values
print(f"leg MN {len(lm)}  DN {len(dn)}  VNC {len(vnc):,}  Kenyon {len(kc):,}")

rows = []
for w in (0.05, 0.08, 0.11, 0.14, 0.17, 0.20):
    b = Brain(W, Params(w_syn=w, sigma_v=2.0, r_spont=0.0))
    b.spont_groups = groups
    c = b.run(t_run=600.0, n_trials=8, seed=6000)
    m = c.mean(axis=1) / 0.6
    rows.append({
        "w_syn": w,
        "net_kHz": round(float(m.sum()) / 1000, 1),
        "brain_Hz": round(float(m[brain_i].mean()), 2),
        "VNC_Hz": round(float(m[vnc].mean()), 2),
        "DN_Hz": round(float(m[dn].mean()), 2),
        "legMN_Hz": round(float(m[lm].mean()), 2),
        "legMN_max": round(float(m[lm].max()), 1),
        "KC>1Hz": round(float((m[kc] > 1).mean()), 4) if len(kc) else None,
        "frac>0.5": round(float((m > 0.5).mean()), 3),
    })
    print(rows[-1], flush=True)

df = pd.DataFrame(rows)
print("\n" + df.to_string(index=False))
print("\nwant: leg motor neurons near-silent at rest, Kenyon cells sparse,")
print("network alive but not saturated -- then stimuli have room to show.")
