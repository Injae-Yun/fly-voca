"""Is there a real source for the "longer timescale", or are we about to invent one?

Deutsch et al. 2020 describe the pC1d/e <-> aIPg loop as holding a female's
post-mating state for minutes, and say neuromodulation "supports the longer
timescale" -- without saying which modulator, or whether it acts on the loop
directly. A0/A1/98 already tried to get persistence out of the fast recurrent
wiring alone and it decays in seconds. The next move in this project would be
to attach a self-driven slow current to pC1d/e by hypothesis. Before doing
that, look for a real anatomical source: who actually contacts this loop with
a slow-acting signal (dopamine, serotonin, octopamine, or a neuropeptide), and
is that same source itself something the loop can drive -- which is the
precondition for a closed modulatory loop rather than an open one-way tap.

Two families, two different data sources, because the connectome only carries
one of them directly:

  monoamines   DA/SER/OCT edges are routed to W_slow (doc 13 / graph.py). Real
               synapse counts, onto real cells, straight from the connectome.

  peptides     never in the connectome -- releasers() (bodysnatch/expression.py)
               reports cell types measured by RNA profiling (GSE271123) to
               express a neuropeptide gene. Their anatomical contacts are
               still classical (ACh/GABA/Glu co-release), so contact counts
               come from W_fast, not a peptide-specific graph.

For both, "driven by the loop" is read on W_fast only, counting positive
(excitatory) paths at 1 and 2 hops -- a candidate that only receives
inhibition from the loop is not a feedback path, so it is scored 0, not
negative.
"""
import numpy as np
import pandas as pd

from flyvoca.graph import build
from flyvoca.paths import cache
from bodysnatch.expression import releasers

MIN_SYN = 3
BACK_THRESHOLD = 10  # "excitatory synapses back from the loop" to call a closed loop

meta, W_fast, W_slow, st = build()
print("graph:", st)

t = meta["cell_type"].astype(str)
add = meta["additional_type(s)"].astype(str)
PC1DE = meta[t.isin(["pC1d", "pC1e"])]
AIPG = meta[add.str.contains("aIPg", na=False)]
LOOP_IDX = np.unique(np.concatenate([PC1DE["idx"].values, AIPG["idx"].values]))
print(f"pC1d/e {len(PC1DE)}  aIPg {len(AIPG)}  loop {len(LOOP_IDX)}")

# ---- excitatory reach of the loop through the fast graph, 1 and 2 hops ----
# "positive sum": each edge is clipped to its excitatory part *before* summing,
# so an inhibitory loop member cannot cancel an excitatory one.
loop_rows = W_fast[LOOP_IDX, :].copy()
loop_rows.data = np.clip(loop_rows.data, 0, None)
exc1 = np.asarray(loop_rows.sum(axis=0)).ravel()          # loop -> X, 1 hop, positive only

Wpos = W_fast.copy()
Wpos.data = np.clip(Wpos.data, 0, None)
exc2 = np.asarray((exc1 @ Wpos)).ravel()                  # loop -> excited X -> *, 2 hops

Wf_abs = W_fast.copy()
Wf_abs.data = np.abs(Wf_abs.data)


def anatomy_table(idx, klass, nt_lookup):
    """syn onto pC1d/e / aIPg (unsigned) + 1/2-hop excitatory drive back from the loop."""
    syn_pc1de = np.asarray(Wf_abs[np.ix_(idx, PC1DE["idx"].values)].sum(axis=1)).ravel() \
        if klass != "monoamine" else np.asarray(W_slow[np.ix_(idx, PC1DE["idx"].values)].sum(axis=1)).ravel()
    syn_aipg = np.asarray(Wf_abs[np.ix_(idx, AIPG["idx"].values)].sum(axis=1)).ravel() \
        if klass != "monoamine" else np.asarray(W_slow[np.ix_(idx, AIPG["idx"].values)].sum(axis=1)).ravel()
    df = pd.DataFrame({
        "root_id": meta.index[idx], "cell_type": t.values[idx], "nt_type": nt_lookup[idx],
        "syn_pc1de": syn_pc1de.round(1), "syn_aipg": syn_aipg.round(1),
        "hop1_exc_from_loop": exc1[idx].round(1), "hop2_exc_from_loop": exc2[idx].round(1),
    })
    df["syn_total"] = df["syn_pc1de"] + df["syn_aipg"]
    return df[df["syn_total"] >= MIN_SYN].sort_values("syn_total", ascending=False).reset_index(drop=True)


nt = meta["nt_type"].astype(str).values

# ---- table 1: monoamine inputs (W_slow presynaptic neurons) ----
slow_pre = np.unique(W_slow.nonzero()[0])
t1 = anatomy_table(slow_pre, "monoamine", nt)
print("\n=== table 1: monoaminergic cells contacting pC1d/e or aIPg (>=3 syn) ===")
print(t1.head(10).to_string(index=False))

# ---- table 2: peptidergic releaser cells ----
rel = releasers()
pep_cell_types = set(rel["cell_type"])
pep_idx = meta[t.isin(pep_cell_types)]["idx"].values
t2_all = anatomy_table(pep_idx, "peptide", nt)
t2 = t2_all.merge(rel[["cell_type", "peptide", "confidence"]], on="cell_type", how="left")
t2 = t2.sort_values("syn_total", ascending=False).reset_index(drop=True)
print("\n=== table 2: peptidergic releaser cells contacting pC1d/e or aIPg (>=3 syn) ===")
print(t2.head(10).to_string(index=False))

# ---- table 3: does the loop itself carry a monoamine or peptide label? ----
pep_by_type = rel.groupby("cell_type")["peptide"].apply(lambda s: ",".join(sorted(set(s))))
loop_meta = meta.iloc[LOOP_IDX]
t3 = (loop_meta.groupby(["cell_type", "additional_type(s)"])
      .agg(n=("idx", "size"), nt_type=("nt_type", lambda s: ",".join(sorted(set(s.astype(str))))))
      .reset_index())
t3["peptide"] = t3["cell_type"].map(pep_by_type).fillna("")
print("\n=== table 3: transmitter/peptide label of the loop itself ===")
print(t3.to_string(index=False))

# ---- table 4: summary counts ----
summary_rows = []
for cls, tbl in (("DA", t1[t1.nt_type == "DA"]), ("SER", t1[t1.nt_type == "SER"]),
                 ("OCT", t1[t1.nt_type == "OCT"]),
                 ("peptide", t2.drop_duplicates("root_id"))):
    n = len(tbl)
    total_syn = float(tbl["syn_total"].sum())
    closed = int((tbl["hop1_exc_from_loop"] >= BACK_THRESHOLD).sum())
    summary_rows.append({"class": cls, "n_cells": n, "total_syn_onto_loop": round(total_syn, 1),
                          f"n_with_hop1>={BACK_THRESHOLD}exc_back": closed})
t4 = pd.DataFrame(summary_rows)
print(f"\n=== table 4: summary (candidate closed loop = hop1 excitatory synapses back >= {BACK_THRESHOLD}) ===")
print(t4.to_string(index=False))

# ---- save tables 1 and 2 ----
out = pd.concat([t1.assign(**{"class": "monoamine", "peptide": ""}),
                 t2.assign(**{"class": "peptide"})], ignore_index=True, sort=False)
out.to_csv(cache("modulation_anatomy.csv"), index=False)
print(f"\nsaved {len(out)} rows to {cache('modulation_anatomy.csv')}")
print("\ndone (modulation anatomy)")
