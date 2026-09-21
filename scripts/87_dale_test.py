"""T4 -- does the Dale approximation break the verified FAFB benchmark?

Hypothesis 1 (the one hypothesis still standing after T1-T3): BANC has no
per-edge transmitter, so every outgoing edge of a neuron is signed by that
neuron's own nt_type (Dale's principle). FAFB's export instead carries a
label on the connection itself (`connections_princeton*.csv.gz`'s `nt_type`
column), which `flyvoca.graph.build()` uses directly.

This script re-signs the *same* FAFB connectome by neuron instead of by edge
-- `nt_type` from `neurons.csv.gz`, painted onto every outgoing edge of that
neuron -- and re-runs the exact benchmark from doc 02 / voca/scripts/
14_calibrate_wsyn.py: ref-20 sugar neurons at 100 Hz, w_syn=0.275, sigma_v=0,
r_spont=0, 30 trials, MN9 readout. If the Dale-signed graph gives a
similar MN9 rate to the edge-signed one, hypothesis 1 is refuted and BANC's
problem is something else (unit convention, missing annotation, ...). If it
diverges sharply, Dale approximation is implicated.

`flyvoca/graph.py` is not modified -- `build_dale()` below is a standalone
copy-and-modify of `flyvoca.graph.build()`, reusing its `load_meta`,
`FAST_SIGN`, `SLOW_NT`, and `UNKNOWN_POLICY` for an apples-to-apples
comparison.
"""
import numpy as np
import pandas as pd
from scipy import sparse

from flyvoca.graph import (build, load_meta, select, FAST_SIGN, SLOW_NT,
                            UNKNOWN_POLICY)
from flyvoca.paths import dataset
from flyvoca.sim import Brain, Params
from voca.reference import SUGAR as REF_SUGAR, MN9

W_SYN = 0.275
SIGMA_V = 0.0
R_SPONT = 0.0
R_POI = 100.0
N_TRIALS = 30
T_RUN = 1000.0
SEED = 1


def build_dale(threshold: bool = False, unknown: str = UNKNOWN_POLICY):
    """Like flyvoca.graph.build(), but signs every outgoing edge of a neuron
    by that neuron's own `nt_type` (neurons.csv, a per-neuron prediction)
    instead of the per-edge `nt_type` column in the connections export.
    Dale's principle: one presynaptic identity, one sign, for every edge it
    makes. This is exactly what flyvoca.banc_graph.build() does for BANC.
    """
    meta = load_meta()  # indexed by root_id, has .idx and .nt_type already
    fname = ("connections_princeton.csv.gz" if threshold
              else "connections_princeton_no_threshold.csv.gz")
    con = pd.read_csv(dataset() / fname,
                       usecols=["pre_root_id", "post_root_id", "syn_count"])

    idx = meta["idx"]
    con["i"] = con["pre_root_id"].map(idx)
    con["j"] = con["post_root_id"].map(idx)
    dangling = con[["i", "j"]].isna().any(axis=1).sum()
    con = con.dropna(subset=["i", "j"])

    # per-neuron sign, indexed by idx -- the Dale-approximation step.
    nt_by_idx = meta.set_index("idx")["nt_type"]
    sign_by_idx = nt_by_idx.map(FAST_SIGN)
    if unknown == "excitatory":
        sign_by_idx = sign_by_idx.where(
            ~(sign_by_idx.isna() & ~nt_by_idx.isin(SLOW_NT)), 1.0)

    pre_i = con["i"].astype(int)
    sign = pre_i.map(sign_by_idx)
    nt_pre = pre_i.map(nt_by_idx)
    fast = con[sign.notna() & ~nt_pre.isin(SLOW_NT)]
    slow = con[nt_pre.isin(SLOW_NT)]

    n = len(meta)
    W_fast = sparse.csr_matrix(
        (sign.loc[fast.index].values * fast["syn_count"].values,
         (fast["i"].astype(int), fast["j"].astype(int))), shape=(n, n))
    W_slow = sparse.csr_matrix(
        (slow["syn_count"].values.astype(float),
         (slow["i"].astype(int), slow["j"].astype(int))), shape=(n, n))

    stats = {"edges": len(con), "dangling": int(dangling),
              "fast": len(fast), "slow": len(slow),
              "unknown_nt": int(nt_by_idx.isna().sum())}
    return meta, W_fast.tocsr(), W_slow.tocsr(), stats


print("=== building both graphs ===")
meta_e, W_edge, _, st_edge = build()               # edge-level sign (existing)
meta_n, W_dale, _, st_dale = build_dale()           # neuron-level (Dale) sign
print("edge-level  :", st_edge)
print("neuron-level:", st_dale)

# how many edges actually flip sign between the two schemes?
assert (meta_e.index == meta_n.index).all()
diff_edges = int((W_edge != W_dale).nnz) if hasattr(W_edge, "nnz") else None
same_shape = W_edge.shape == W_dale.shape
print(f"\nsame neuron indexing: {(meta_e['idx'] == meta_n['idx']).all()}  "
      f"same matrix shape: {same_shape}")
d = (W_edge - W_dale)
print(f"entries differing between edge-level and neuron-level W: {d.nnz:,} "
      f"of edge-level {W_edge.nnz:,} / dale {W_dale.nnz:,}")

ref = [i for i in REF_SUGAR if i in meta_e.index]
print(f"\nref-20 sugar neurons present in v783 meta: {len(ref)}/21")
stim_e = meta_e.loc[ref, "idx"].values
stim_n = meta_n.loc[ref, "idx"].values
assert (stim_e == stim_n).all(), "stim indices must match between builds"
mn9_e = int(meta_e.loc[MN9, "idx"])
mn9_n = int(meta_n.loc[MN9, "idx"])
assert mn9_e == mn9_n

params = Params(w_syn=W_SYN, sigma_v=SIGMA_V, r_spont=R_SPONT)
print(f"\nparams: {params}")
print(f"stim n={len(stim_e)}  r_poi={R_POI}  n_trials={N_TRIALS}  "
      f"t_run={T_RUN}  seed={SEED}")

rows = []
for label, W in (("edge-level (existing build)", W_edge),
                  ("neuron-level (Dale, build_dale)", W_dale)):
    brain = Brain(W, params)
    c = brain.run(stim=stim_e, t_run=T_RUN, n_trials=N_TRIALS, r_poi=R_POI,
                  seed=SEED)
    m, s = Brain.rate(c, t_run=T_RUN)
    rows.append({"scheme": label, "MN9_Hz": round(float(m[mn9_e]), 2),
                 "MN9_sd": round(float(s[mn9_e]), 2),
                 "peak_Hz": round(float(m.max()), 1),
                 "active>0.5Hz": int((m > 0.5).sum()),
                 "net_Hz": int(m.sum())})
    print(rows[-1], flush=True)

df = pd.DataFrame(rows)
print("\n" + df.to_string(index=False))

edge_hz = rows[0]["MN9_Hz"]
dale_hz = rows[1]["MN9_Hz"]
pct = (dale_hz - edge_hz) / edge_hz * 100 if edge_hz else float("nan")
print(f"\nedge-level MN9 = {edge_hz} Hz (reference target 67.0, "
      f"doc-02 measured 71.9)")
print(f"neuron-level (Dale) MN9 = {dale_hz} Hz")
print(f"relative change = {pct:+.1f}%")
verdict = "hypothesis 1 SUPPORTED (Dale sign implicated)" if abs(pct) > 30 \
    else "hypothesis 1 REFUTED (Dale sign is not the BANC problem)"
print(f"verdict (threshold +-30%): {verdict}")
print("\ndone (dale test)")
