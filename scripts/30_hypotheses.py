"""Three open questions, tested.

H1c  The published spike data was produced on the *630* connectome -- their
     notebook's config points at 2023_03_23_connectivity_630_final.parquet --
     but it was benchmarked here against a run on their 783 matrix. If that
     version mismatch is the unexplained peak-rate gap (171 vs 114 Hz), the
     630 matrix should reproduce both MN9 and the peak.

H1a  Alternatively the exponential-Euler step, which holds g constant across
     dt, overestimates drive most in the highest-firing cells. Halving dt
     should then pull the peak down and leave MN9 alone.

H3a  The two FB6A driver lines disagree 16x on AstC. If a line is labelling
     off-target cells, the disagreement should be broad; if it is specific to
     AstC, something else is going on.
"""
import numpy as np
import pandas as pd
from scipy import sparse

from flyvoca.sim import Brain, Params
from voca import reference as ref
from flyvoca.paths import DATA_ROOT

PUB_MN9, PUB_PEAK = 67.0, 114.3
REF = DATA_ROOT / "reference"


def build_from(comp_csv, conn_pq):
    comp = pd.read_csv(REF / comp_csv, index_col=0)
    con = pd.read_parquet(REF / conn_pq)
    n = len(comp)
    W = sparse.csr_matrix(
        (con["Excitatory x Connectivity"].values.astype(np.float32),
         (con["Presynaptic_Index"].values, con["Postsynaptic_Index"].values)),
        shape=(n, n))
    return {f: i for i, f in enumerate(comp.index)}, W


print("=== H1c: which connectome version did the published run use? ===")
for tag, (c, q) in {"630": ("ref_comp630.csv", "ref_conn630.parquet"),
                    "783": ("ref_comp783.csv", "ref_conn783.parquet")}.items():
    f2i, W = build_from(c, q)
    stim = np.array([f2i[f] for f in ref.SUGAR if f in f2i])
    mn9 = f2i.get(ref.MN9)
    b = Brain(W, Params(w_syn=0.275, sigma_v=0.0, r_spont=0.0))  # the reference's own settings
    cnt = b.run(stim=stim, t_run=1000.0, n_trials=30, r_poi=100.0, seed=1)
    m, _ = Brain.rate(cnt)
    print(f"  {tag}: neurons={len(f2i):,} sugar={len(stim)}/21  "
          f"MN9={m[mn9]:6.1f} (pub {PUB_MN9})  peak={m.max():6.1f} (pub {PUB_PEAK})  "
          f"active={int((m > 0.5).sum())} (pub 357)", flush=True)

print("\n=== H1a: does a finer timestep move the peak? (783 matrix) ===")
f2i, W = build_from("ref_comp783.csv", "ref_conn783.parquet")
stim = np.array([f2i[f] for f in ref.SUGAR if f in f2i])
mn9 = f2i[ref.MN9]
for dt in (0.1, 0.05, 0.025):
    b = Brain(W, Params(dt=dt, w_syn=0.275, sigma_v=0.0, r_spont=0.0))
    cnt = b.run(stim=stim, t_run=1000.0, n_trials=20, r_poi=100.0, seed=1)
    m, _ = Brain.rate(cnt)
    print(f"  dt={dt:<6} MN9={m[mn9]:6.1f}  peak={m.max():6.1f}", flush=True)

print("\n=== H3a: do the two FB6A lines disagree broadly, or only on AstC? ===")
from voca.expression import load
cpm, _ = load()
a, b_ = cpm["SS54343"], cpm["SS57656"]
keep = (a + b_) > 10
la, lb = np.log10(a[keep] + 1), np.log10(b_[keep] + 1)
print(f"  genes compared: {int(keep.sum())}")
print(f"  Pearson r (log10 CPM): {np.corrcoef(la, lb)[0, 1]:.3f}")
ratio = (a[keep] + 1) / (b_[keep] + 1)
worst = ratio.apply(lambda x: max(x, 1 / x)).sort_values(ascending=False).head(8)
print("  largest disagreements (fold):")
for (cls, gene), f in worst.items():
    print(f"    {gene:<12} {cls:<24} {f:8.1f}x   SS54343={a[(cls,gene)]:9.0f}  SS57656={b_[(cls,gene)]:9.0f}")
