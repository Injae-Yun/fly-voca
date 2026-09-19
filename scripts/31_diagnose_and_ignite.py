"""Two things: find out *which* neurons the rate gap lives in, and see whether
the slow layer does anything at all once it is switched on.

Three guesses at the peak-rate gap have now been refuted -- timestep size,
connectome version, and input/reset ordering. Rather than guess a fourth,
compare the published rates neuron by neuron against ours.
"""
import numpy as np
import pandas as pd
from scipy import sparse

from flyvoca.graph import build, load_meta
from flyvoca.paths import DATA_ROOT
from bodysnatch.sim import Brain, Params
from bodysnatch import reference as ref
from bodysnatch.peptide import Field, Expression
from bodysnatch.expression import fill, releasers

REF = DATA_ROOT / "reference"

# ---------------------------------------------------------------- diagnosis
comp = pd.read_csv(REF / "ref_comp783.csv", index_col=0)
con = pd.read_parquet(REF / "ref_conn783.parquet")
n = len(comp)
W_ref = sparse.csr_matrix(
    (con["Excitatory x Connectivity"].values.astype(np.float32),
     (con["Presynaptic_Index"].values, con["Postsynaptic_Index"].values)), shape=(n, n))
f2i = {f: i for i, f in enumerate(comp.index)}
i2f = {i: f for f, i in f2i.items()}
stim = np.array([f2i[f] for f in ref.SUGAR if f in f2i])

b = Brain(W_ref, Params(w_syn=0.275, sigma_v=0.0, r_spont=0.0))
mine, _ = Brain.rate(b.run(stim=stim, t_run=1000.0, n_trials=30, r_poi=100.0, seed=1))
pub = ref.published_rates()

meta = load_meta()
lab = meta["cell_type"].to_dict()
ours = pd.Series({i2f[i]: r for i, r in enumerate(mine) if r > 0.5})
both = pd.DataFrame({"ours": ours, "pub": pub}).fillna(0.0)
both["label"] = [lab.get(f, "?") for f in both.index]
both["stim"] = [f in set(ref.SUGAR) for f in both.index]

print(f"active: ours {int((both.ours > 0.5).sum())}  published {int((both.pub > 0.5).sum())}  "
      f"shared {int(((both.ours > 0.5) & (both.pub > 0.5)).sum())}")
sh = both[(both.ours > 0.5) & (both.pub > 0.5)]
print(f"correlation on shared neurons: r = {np.corrcoef(sh.ours, sh.pub)[0, 1]:.3f}  "
      f"median ratio = {np.median(sh.ours / sh.pub):.2f}")
print("\ntop 12 by our rate:")
print(both.sort_values('ours', ascending=False).head(12).round(1).to_string())
print("\nonly we call active (top 8):")
print(both[both.pub < 0.5].sort_values('ours', ascending=False).head(8).round(1).to_string())
print("\nonly they call active (top 8):")
print(both[both.ours < 0.5].sort_values('pub', ascending=False).head(8).round(1).to_string())

# ------------------------------------------------------------------ ignition
print("\n\n=== slow layer: does hunger reach the sleep homeostat? ===")
meta, W, _, _ = build()
brain = Brain.from_meta(W, meta)
peps = sorted(set(releasers().peptide) | {"DILP", "DH44", "DH31", "CRZ", "DMS",
                                          "Hugin", "CAPA", "ITP"})
field = Field(meta, peptides=peps, tau=20.0, gain=1.0)
expr = Expression(len(meta), tuple(peps))
fill(expr, meta)

dh44 = meta[meta["cell_type"] == "m_NSC_DH44"]["idx"].values
er5 = meta[meta["cell_type"] == "ER5"]["idx"].values
exr1 = meta[meta["cell_type"] == "ExR1"]["idx"].values
print(f"driving m_NSC_DH44 ({len(dh44)}) -- reading ER5 ({len(er5)}) and ExR1 ({len(exr1)})")

state, rows = None, []
for epoch in range(8):
    counts, state = brain.run(stim=dh44, t_run=500.0, n_trials=6, r_poi=50.0,
                              seed=10 + epoch, state=state,
                              v_offset=field.v_offset(expr), return_state=True)
    m, _ = Brain.rate(counts, 500.0)
    c = field.step(m, dt=0.5)
    rows.append({"t_s": (epoch + 1) * 0.5,
                 "DH44_conc": round(float(c[peps.index("DH44")]), 3),
                 "ER5_Hz": round(float(m[er5].mean()), 2),
                 "ExR1_Hz": round(float(m[exr1].mean()), 2),
                 "v_off_ER5": round(float(field.v_offset(expr)[er5].mean()), 4)})
    print(rows[-1], flush=True)
print("\n" + pd.DataFrame(rows).to_string(index=False))
