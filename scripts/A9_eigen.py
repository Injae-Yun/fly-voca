"""Is the loop an attractor by the criterion that matters?

Biswas, Stanoev, Romani & Fitzgerald (bioRxiv 2024) get an exact ring
attractor out of the connectome, and the condition is on the eigenstructure
of the recurrent weight matrix, not on a round-trip gain. Round-trip gain
(docs 16-17: 0.085) is one path; an eigenvalue of the full submatrix counts
every path, including the ones through inhibitory partners.

Linearise the LIF around its operating point: a presynaptic rate change of
1 Hz through one synapse changes the postsynaptic rate by gamma Hz. Then the
Jacobian of a set of cells is gamma * W_sub (signed synapse counts), and the
set can sustain itself iff the leading eigenvalue of gamma * W_sub reaches 1.

gamma is measured, not assumed: A1 drove pC1d/e and read aIPg (slope 0.306
Hz/Hz through 523/7 = 75 synapses per target) and the reverse (0.279 through
352/4 = 88), so gamma ~ 0.003-0.004 Hz/Hz/synapse at w_syn = 0.2. It is
reported as a range and the gamma needed for lambda = 1 is the result.

Sets, from the loop outward:
  loop          pC1d/e + aIPg (41)
  loop+partners cells exchanging >= 20 synapses with the loop in either
                direction (the SMP054 / AOTU064 / CB1544 inhibitors included)
  loop+2hop     as above plus their >= 20-synapse partners
"""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import eigs

from flyvoca.graph import build
from flyvoca.paths import cache

meta, W, _, _ = build()
t = meta["cell_type"].astype(str); add = meta["additional_type(s)"].astype(str)
n = W.shape[0]
PC1DE = meta[t.isin(["pC1d", "pC1e"])]["idx"].values
AIPG = meta[add.str.contains("aIPg")]["idx"].values
LOOP = np.unique(np.concatenate([PC1DE, AIPG]))
Wc = W.tocsr(); Wa = abs(Wc)
GAMMA = (0.003, 0.004)


def expand(idx, min_syn=20):
    v = np.zeros(n); v[idx] = 1
    out = np.asarray(Wa @ v).ravel() + np.asarray(Wa.T @ v).ravel()
    return np.unique(np.concatenate([idx, np.where(out >= min_syn)[0]]))


def leading(idx):
    S = Wc[np.ix_(idx, idx)].astype(np.float64)
    k = min(6, len(idx) - 2)
    vals, vecs = eigs(S, k=k, which="LR")
    i = np.argmax(vals.real)
    lam, vec = vals[i], vecs[:, i].real
    top = np.argsort(-np.abs(vec))[:8]
    return lam, [(t.iloc[idx[j]], round(float(vec[j]), 2)) for j in top]


rows = []
sets = {"loop": LOOP}
sets["loop+partners"] = expand(LOOP)
sets["loop+2hop"] = expand(sets["loop+partners"])
for name, idx in sets.items():
    lam, top = leading(idx)
    ex = (Wc[np.ix_(idx, idx)] > 0).sum(); inh = (Wc[np.ix_(idx, idx)] < 0).sum()
    row = {"set": name, "n_cells": len(idx), "edges_exc": int(ex), "edges_inh": int(inh),
           "lambda_max": round(lam.real, 1), "imag": round(lam.imag, 1),
           "lambda_x_gamma_lo": round(lam.real * GAMMA[0], 3),
           "lambda_x_gamma_hi": round(lam.real * GAMMA[1], 3),
           "gamma_needed": round(1.0 / lam.real, 4) if lam.real > 0 else np.inf,
           "fold_short": round((1.0 / lam.real) / GAMMA[1], 1) if lam.real > 0 else np.inf}
    rows.append(row); print(row, flush=True)
    print("   leading mode loads on:", top, flush=True)

# how much per-cell heterogeneity would it take? Scale each cell's INPUT by a
# factor s_j and ask for the leading eigenvalue of diag(s) W: the same
# question a cell-specific input resistance poses.
idx = sets["loop"]
S = Wc[np.ix_(idx, idx)].toarray().astype(np.float64)
for label, s in (("uniform", np.ones(len(idx))),
                 ("pC1d/e x4", np.where(np.isin(idx, PC1DE), 4.0, 1.0)),
                 ("pC1d/e x10", np.where(np.isin(idx, PC1DE), 10.0, 1.0)),
                 ("aIPg x4", np.where(np.isin(idx, AIPG), 4.0, 1.0))):
    lam = np.max(np.linalg.eigvals(np.diag(s) @ S).real)
    print(f"  loop with input scaling {label:10s}: lambda_max {lam:7.1f}  x gamma_hi {lam*GAMMA[1]:.3f}")

pd.DataFrame(rows).to_csv(cache("loop_eigen.csv"), index=False)
print("\ndone (eigen)")
