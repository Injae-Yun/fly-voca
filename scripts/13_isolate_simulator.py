"""Run our simulator on the reference's own matrix.

If MN9 lands near 67 Hz, the simulator is faithful and our graph build differs.
If it stays near 160 Hz, the simulator differs from Brian2.
"""
import numpy as np, pandas as pd
from scipy import sparse
from bodysnatch.sim import Brain, Params

S = "."  # directory holding the reference repo files
comp = pd.read_csv(S + "/ref_comp783.csv", index_col=0)
con = pd.read_parquet(S + "/ref_conn783.parquet")
n = len(comp)
flyid2i = {f: i for i, f in enumerate(comp.index)}
print("neurons", n, "edges", len(con))

W = sparse.csr_matrix(
    (con["Excitatory x Connectivity"].values.astype(np.float32),
     (con["Presynaptic_Index"].values, con["Postsynaptic_Index"].values)),
    shape=(n, n))
print("pairs", W.nnz, "exc", W[W>0].sum(), "inh", -W[W<0].sum())

REF = [720575940624963786,720575940630233916,720575940637568838,720575940638202345,
720575940617000768,720575940630797113,720575940632889389,720575940621754367,
720575940621502051,720575940640649691,720575940639332736,720575940616885538,
720575940639198653,720575940620900446,720575940617937543,720575940632425919,
720575940633143833,720575940612670570,720575940628853239,720575940629176663,
720575940611875570]
stim = np.array([flyid2i[f] for f in REF if f in flyid2i])
mn9 = flyid2i[720575940660219265]
print("stim", len(stim), "of 21")

b = Brain(W)
for hz in (100.0, 150.0):
    c = b.run(stim=stim, t_run=1000.0, n_trials=30, r_poi=hz, seed=1)
    m, s = Brain.rate(c)
    print(f"{hz:5.0f} Hz -> MN9 {m[mn9]:6.1f} +/- {s[mn9]:.1f} Hz   "
          f"active={int((m>0.5).sum())}  peak={m.max():.0f} Hz  stim_mean={m[stim].mean():.1f}")
print("reference: MN9 @100Hz = 67.0 Hz, peak ~114 Hz")
