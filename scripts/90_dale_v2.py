"""T4, redesigned: does per-neuron (Dale) signing break a benchmark?

The first attempt compared edge-level against neuron-level signing on
`flyvoca.graph.build()` and called +28.7% on MN9 a refutation. Two things were
wrong with that.

First, the baseline was not a benchmark. 71.9 Hz came from the *reference's
own* connectivity matrix, a different code path; `flyvoca.graph.build()` on
the same stimulus gives 162.6 Hz. Comparing two schemes that are both 2.3x off
target says nothing about either.

Second, MN9 alone is the wrong readout. Dale signing moved `active>0.5Hz` from
571 to 5,262 -- a ninefold change in how much of the brain is awake -- while
MN9 moved 29%. A +-30% rule on one neuron cannot see that.

So this runs both schemes on the matrix where the benchmark is defined, and
scores them on the whole network rather than one cell:

  MN9              against the published 67.0 Hz
  active neurons   against the reference's own 357
  peak rate        against 114.3
  rank correlation of per-neuron rates against the published spike data

The reference matrix ships its own `Excitatory` column (per-edge), so the Dale
variant repaints it from each presynaptic neuron's majority sign -- the same
operation BANC forces on us, applied where we can check the answer.
"""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr

from flyvoca.paths import DATA_ROOT
from flyvoca.sim import Brain, Params
from voca import reference as ref

REF = DATA_ROOT / "reference"
PUB_MN9, PUB_PEAK, PUB_ACTIVE = 67.0, 114.3, 357

comp = pd.read_csv(REF / "ref_comp783.csv", index_col=0)
con = pd.read_parquet(REF / "ref_conn783.parquet")
n = len(comp)
f2i = {f: i for i, f in enumerate(comp.index)}
print(f"reference matrix: {n:,} neurons, {len(con):,} edges")

pre = con["Presynaptic_Index"].values
post = con["Postsynaptic_Index"].values
conn = con["Connectivity"].values.astype(np.float32)
exc = con["Excitatory"].values.astype(np.float32)      # per-edge, +1 / -1

# --- scheme A: per-edge signs, exactly as the reference ships them ---
W_edge = sparse.csr_matrix((exc * conn, (pre, post)), shape=(n, n))

# --- scheme B: Dale -- each neuron's majority sign paints all its outgoing ---
w_by_pre = pd.DataFrame({"pre": pre, "signed": exc * conn, "w": conn})
agg = w_by_pre.groupby("pre").agg(net=("signed", "sum"), tot=("w", "sum"))
neuron_sign = np.ones(n, dtype=np.float32)
neuron_sign[agg.index.values] = np.sign(agg["net"].values).astype(np.float32)
neuron_sign[neuron_sign == 0] = 1.0
W_dale = sparse.csr_matrix((neuron_sign[pre] * conn, (pre, post)), shape=(n, n))

flipped = int((np.sign(exc) != neuron_sign[pre]).sum())
print(f"edges whose sign changes under Dale: {flipped:,} of {len(con):,} "
      f"({flipped/len(con)*100:.1f}%)")
for name, W in (("edge", W_edge), ("dale", W_dale)):
    e, i = W[W > 0].sum(), -W[W < 0].sum()
    print(f"  {name:5} exc {e:>12,.0f}  inh {i:>12,.0f}  E/I {e/i:.3f}")

pub = ref.published_rates()
stim = np.array([f2i[f] for f in ref.SUGAR if f in f2i])
mn9 = f2i[ref.MN9]
print(f"\nstim {len(stim)}/21 sugar neurons, w_syn=0.275, 30 trials, 100 Hz")
print(f"published: MN9 {PUB_MN9} Hz, peak {PUB_PEAK}, active {PUB_ACTIVE}\n")

i2f = {i: f for f, i in f2i.items()}
rows = []
for name, W in (("edge-level (as shipped)", W_edge), ("neuron-level (Dale)", W_dale)):
    b = Brain(W, Params(w_syn=0.275, sigma_v=0.0, r_spont=0.0))
    m, sd = Brain.rate(b.run(stim=stim, t_run=1000.0, n_trials=30,
                             r_poi=100.0, seed=1))
    ours = pd.Series({i2f[i]: r for i, r in enumerate(m) if r > 0.5})
    both = pd.DataFrame({"ours": ours, "pub": pub}).dropna()
    rho = spearmanr(both.ours, both.pub).statistic if len(both) > 10 else np.nan
    rows.append({
        "scheme": name,
        "MN9_Hz": round(float(m[mn9]), 1),
        "MN9_err%": round((float(m[mn9]) - PUB_MN9) / PUB_MN9 * 100),
        "peak_Hz": round(float(m.max())),
        "peak_err%": round((float(m.max()) - PUB_PEAK) / PUB_PEAK * 100),
        "active": int((m > 0.5).sum()),
        "active_err%": round((int((m > 0.5).sum()) - PUB_ACTIVE) / PUB_ACTIVE * 100),
        "shared": len(both),
        "spearman_vs_published": round(float(rho), 3),
    })
    print(rows[-1], flush=True)

df = pd.DataFrame(rows)
print("\n" + df.to_string(index=False))
print("\nverdict is read off the whole row, not MN9 alone:")
print("  if Dale degrades active-count or the rank correlation materially,")
print("  the approximation is implicated for BANC; if both schemes track the")
print("  published data equally, it is cleared and BANC's problem is elsewhere.")
print("done (dale v2)")
