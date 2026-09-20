"""Is there a wired path from DNa02 to the leg motor neurons?

FAFB v783 ends at the neck, so the body connection in doc 11 had to be
hand-made: PFL3 left-right became a CPG amplitude because nothing in the data
said what descending neurons do once they descend.

BANC contains the rest -- same sex, same animal, brain through nerve cord.
This asks whether the link we invented exists in the wiring, and what it looks
like. Three things are checked, in order of how much they would change:

  reach      can DNa02 get to leg motor neurons at all, and in how many hops
  laterality does left DNa02 preferentially reach one side's legs -- the
             steering signal only means something if it breaks symmetry
  directness the MANC literature reports direct DN->MN contacts are rare and
             most control runs through premotor interneurons; if that holds
             here, the hand-made "one number drives a gait controller" was
             closer to the truth than a single synapse would have been
"""
import numpy as np
import pandas as pd
from scipy import sparse

from flyvoca.paths import RAW

D = RAW / "banc_v888"
meta = pd.read_feather(D / "banc_888_meta.feather")
edges = pd.read_feather(D / "banc_888_edgelist_simple_v3.feather")
print(f"meta {len(meta):,} neurons | edges {len(edges):,} pairs")
print("edge columns:", list(edges.columns))

meta = meta.reset_index(drop=True)
idx = pd.Series(meta.index.values, index=meta["banc_888_id"].values)
edges["i"] = edges["pre"].map(idx)
edges["j"] = edges["post"].map(idx)
drop = edges[["i", "j"]].isna().any(axis=1).sum()
edges = edges.dropna(subset=["i", "j"])
n = len(meta)
W = sparse.csr_matrix((edges["count"].values.astype(np.float32),
                       (edges["i"].astype(int), edges["j"].astype(int))),
                      shape=(n, n))
print(f"matrix {n:,} x {n:,}, {W.nnz:,} pairs ({drop:,} edges dropped)")

LEG_NERVES = [c for c in meta["nerve"].dropna().unique() if "leg_nerve" in str(c)]
legmn = meta[(meta.super_class == "motor") & (meta.nerve.isin(LEG_NERVES))]
print(f"\nleg motor neurons: {len(legmn)} across {len(LEG_NERVES)} nerves")
print(legmn.groupby("nerve").size().to_string())


def group(ct, side=None):
    m = meta[(meta.cell_type == ct) | (meta.fafb_cell_type == ct)]
    if side:
        m = m[m["side"] == side] if "side" in m.columns else m
    return m.index.values


def reach(src, dst, max_hops=4, thresh=5):
    """Synaptic weight reaching dst from src within k hops."""
    v = np.zeros(n, dtype=np.float32)
    v[src] = 1.0
    A = W.copy()
    A.data = (A.data >= thresh).astype(np.float32) * A.data
    A.eliminate_zeros()
    rows = []
    for h in range(1, max_hops + 1):
        v = A.T @ v
        v[src] = 0.0
        hit = v[dst]
        rows.append({"hops": h, "reached_MNs": int((hit > 0).sum()),
                     "total_weight": float(hit.sum())})
    return pd.DataFrame(rows)


for ct in ("DNa02", "DNa03", "DNp01", "DNg67"):
    src = group(ct)
    if not len(src):
        print(f"\n{ct}: not found"); continue
    print(f"\n=== {ct} ({len(src)} neurons) -> leg motor neurons ===")
    print(reach(src, legmn.index.values).to_string(index=False))

print("\n=== laterality: does each side's DNa02 favour one side's legs? ===")
dna02 = meta[(meta.cell_type == "DNa02") | (meta.fafb_cell_type == "DNa02")]
sidecol = "side" if "side" in dna02.columns else None
print(dna02[[c for c in ("banc_888_id", "cell_type", "side", "region") if c in dna02.columns]].to_string(index=False))
for _, r in dna02.iterrows():
    v = np.zeros(n, dtype=np.float32); v[r.name] = 1.0
    for _ in range(3):
        v = W.T @ v; v[r.name] = 0.0
    left = legmn[legmn.nerve.str.startswith("left")].index.values
    right = legmn[legmn.nerve.str.startswith("right")].index.values
    L, R = float(v[left].sum()), float(v[right].sum())
    s = r.get("side", "?")
    print(f"  DNa02 side={s}: reaches left legs {L:,.0f}  right legs {R:,.0f}  "
          f"L/R ratio {L/max(R,1e-9):.2f}")

print("\n=== directness: DN -> MN in one hop? ===")
dns = meta[meta.super_class == "descending"].index.values
direct = W[np.ix_(dns, legmn.index.values)]
print(f"  all {len(dns)} DNs -> {len(legmn)} leg MNs directly: "
      f"{direct.nnz:,} pairs, {direct.sum():,.0f} synapses")
print(f"  DNs with any direct leg-MN contact: "
      f"{int((np.asarray(direct.sum(axis=1)).ravel() > 0).sum())}")
print("done (banc path)")
