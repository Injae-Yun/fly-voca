"""Candidate 2, made specific: does BANC's weak-pair mass drive the runaway?

The fine sweep killed candidate 4 -- no window exists between 0.05 and 0.085 --
but the diagnostics located the heat. It is not the optic lobe (optic/brain
falls from 0.24 to 0.03 as w_syn rises, so vision cools relatively), which
clears candidate 3. It is the mushroom body: 68% of Kenyon cell output goes to
other Kenyon cells, 153,033 pairs carrying 197,072 synapses, against a single
APL neuron supplying most of their inhibition. KC input E/I is 11.08.

The number that matters is 1.29 synapses per KC-to-KC pair. In the animal
those contacts sit far below a Kenyon cell's threshold; in a uniform LIF where
every neuron has the same threshold and every synapse the same weight, 153k of
them sum into a positive-feedback mesh.

So this is the uniform-parameter limitation from doc 02 biting harder, and it
arrives through BANC's edgelist including every pair down to one synapse --
8.7M more pairs than FAFB. Earlier we chose to keep weak connections because
on FAFB thresholding barely moved E/I. This tests whether that still holds
here, rather than assuming either way.

Looming rides along as the signal that must survive: a threshold that fixes
Kenyon cells by deleting real circuitry is not a fix.
"""
import numpy as np
import pandas as pd

from flyvoca.banc_graph import build, select, leg_motor, spontaneous_groups
from flyvoca.sim import Brain, Params

print("=== how much of BANC is weak pairs? ===")
meta, W, _, st = build(min_syn=1)
cnt = np.abs(W.data)
for t in (1, 2, 3, 5, 10):
    k = cnt >= t
    print(f"  count >= {t:2}: {k.sum():10,} pairs ({k.mean()*100:5.1f}%)  "
          f"{cnt[k].sum():12,.0f} synapses ({cnt[k].sum()/cnt.sum()*100:5.1f}%)")

kc = meta[meta.cell_class.astype(str).str.contains("kenyon", case=False, na=False)]["idx"].values
lm = leg_motor(meta)["idx"].values
dn = meta[meta.super_class == "descending"]["idx"].values
groups = spontaneous_groups(meta)
sel = lambda **kw: select(meta, **kw)["idx"].values
LOOM = np.concatenate([sel(cell_type="LC4"), sel(cell_type="LPLC2")])
SUGAR = sel(prefix="LB3")

print("\n=== KC-to-KC mesh under each threshold ===")
for t in (1, 2, 3, 5):
    _, Wt, _, _ = build(min_syn=t)
    kk = abs(Wt)[np.ix_(kc, kc)]
    tot = np.asarray(abs(Wt)[kc, :].sum(axis=1)).ravel().sum()
    print(f"  >= {t}: KC->KC {kk.nnz:8,} pairs {kk.sum():10,.0f} syn  "
          f"= {kk.sum()/max(tot,1)*100:4.1f}% of KC output")

print("\n=== sweep threshold x w_syn, watching Kenyon sparseness and looming ===")
rows = []
for t in (1, 3, 5):
    _, Wt, _, stt = build(min_syn=t)
    for w in (0.10, 0.16, 0.22):
        b = Brain(Wt, Params(w_syn=w, sigma_v=2.0, r_spont=0.0))
        b.spont_groups = groups
        base = b.run(t_run=600.0, n_trials=8, seed=8000) / 0.6
        loom = b.run(stim=LOOM, t_run=600.0, n_trials=8, r_poi=100.0, seed=8000) / 0.6
        sug = b.run(stim=SUGAR, t_run=600.0, n_trials=8, r_poi=100.0, seed=8000) / 0.6
        bm = base.mean(axis=1)
        sd = lambda x: np.sqrt((x.std(axis=1, ddof=1) ** 2
                                + base.std(axis=1, ddof=1) ** 2) / 2)
        d_loom = (loom.mean(axis=1) - bm) / np.maximum(sd(loom), 0.05)
        d_sug = (sug.mean(axis=1) - bm) / np.maximum(sd(sug), 0.05)
        rows.append({
            "min_syn": t, "w_syn": w,
            "pairs_M": round(stt["fast_pairs"] / 1e6, 1),
            "KC>1Hz": round(float((bm[kc] > 1).mean()), 3),
            "legMN_rest": round(float(bm[lm].mean()), 2),
            "loom_legMN_d>2": int((np.abs(d_loom[lm]) > 2).sum()),
            "sugar_legMN_d>2": int((np.abs(d_sug[lm]) > 2).sum()),
            "loom_DN_d>2": int((np.abs(d_loom[dn]) > 2).sum()),
        })
        print(rows[-1], flush=True)

df = pd.DataFrame(rows)
print("\n" + df.to_string(index=False))
good = df[(df["KC>1Hz"] < 0.10) & (df["loom_legMN_d>2"] > 5)]
print("\n=== sparse Kenyon cells AND looming still reaching the legs ===")
print(good.to_string(index=False) if len(good) else "  none")
print("\ndone (threshold)")
