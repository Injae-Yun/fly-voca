"""Is the 2.3x over-excitation caused by the synapse-count threshold?

Shiu et al. use every connection, down to single synapses (15.1M edges).
`connections_princeton.csv.gz` is pre-thresholded at >=5 synapses (3.7M edges).
Thresholding strips weak connections -- and inhibition is the more diffuse of
the two, so it strips more inhibition than excitation. In a recurrent network a
few percent of E/I imbalance is not a few percent of firing rate.
"""
import numpy as np

from flyvoca.graph import build
from bodysnatch.sim import Brain

MN9 = 720575940660219265
REF_SUGAR = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]
REFERENCE_MN9_100HZ = 67.0  # measured from results/example/sugarR_100Hz.parquet

for thr in (True, False):
    meta, W, _, st = build(threshold=thr)
    e, i = W[W > 0].sum(), -W[W < 0].sum()
    print(f"\n### threshold={thr}  edges={st['edges']:,}  pairs={W.nnz:,}")
    print(f"    exc={e:,.0f}  inh={i:,.0f}  E/I={e/i:.4f}")

    brain = Brain(W)
    idx = meta.loc[[x for x in REF_SUGAR if x in meta.index], "idx"].values
    mn9 = int(meta.loc[MN9, "idx"])
    for hz in (100.0, 150.0):
        c = brain.run(stim=idx, t_run=1000.0, n_trials=30, r_poi=hz, seed=1)
        m, s = Brain.rate(c)
        tag = f"  <- reference {REFERENCE_MN9_100HZ} Hz" if hz == 100.0 else ""
        print(f"    {hz:5.0f} Hz drive -> MN9 {m[mn9]:6.1f} +/- {s[mn9]:.1f} Hz"
              f"   active={int((m > 0.5).sum())}  peak={m.max():.0f} Hz{tag}")
