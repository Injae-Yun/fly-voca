"""Does the state vector actually move when something happens to the fly?

All eight axes resolve in BANC, so they can be *read*. That is not the same as
their being informative. Doc 09 found hunger has no descending output at all
(max |d| 0.41, below the sham floor), so an axis can be perfectly readable and
still say nothing about behaviour.

This asks the prerequisite for any combined stimulus/behaviour/state display:
under a stimulus, does the state vector change in a way that distinguishes
conditions? Three checks, in increasing strictness:

  moves        any axis past the sham floor
  separates    do conditions produce *different* state vectors, or does
               everything just rise together
  not trivial  the driven axis obviously lights up (sugar drives taste), so
               the interesting question is whether the OTHER axes respond --
               a stimulus changing an untouched internal state is what would
               make this a readout of state rather than an echo of input

Sham is baseline on a different seed, and its spread across axes is the floor
everything else must clear.
"""
import numpy as np
import pandas as pd

from flyvoca.banc_graph import build, select, spontaneous_groups
from flyvoca.paths import cache
from bodysnatch.sim import Brain, Params

W_SYN, TRIALS, T_RUN, SEED = 0.16, 16, 600.0, 13000

meta, W, _, st = build()
groups = spontaneous_groups(meta)
t = meta.cell_type.astype(str)


def axis(*pats):
    m = np.zeros(len(meta), dtype=bool)
    for p in pats:
        m |= t.str.fullmatch(p, na=False).values
        m |= t.str.startswith(p, na=False).values
    return meta.index.values[m]


AXES = {
    "hunger": axis("m_NSC_DILP", "m_NSC_DH44", "l_NSC_DH31", "SEZ_NSC_Hugin",
                   "m_NSC_DMS", "SEZ_NSC_CAPA"),
    "thirst": axis("ISN"),
    "sleep": axis("ER5", "FB6", "FB7"),
    "clock": axis("s-LNv", "l-LNv", "LPN", "DN2", "LNd", "DN1"),
    "courtship": axis("pC1"),
    "defense": axis("LC4", "LPLC2", "LC6", "DNp01", "DNp11"),
    "taste": axis("LB3"),
    "steering": axis("PFL3", "PFL2", "DNa02", "DNa03"),
}
for k, v in AXES.items():
    print(f"  {k:10} {len(v):4} neurons")

sel = lambda **kw: select(meta, **kw)["idx"].values
CONDS = {
    "sham": np.array([], dtype=np.int64),
    "sugar": sel(prefix="LB3"),
    "looming": np.concatenate([sel(cell_type="LC4"), sel(cell_type="LPLC2")]),
    "co2": sel(cell_type="ORN_V"),
    "vinegar": sel(cell_type="ORN_DM1"),
}

b = Brain(W, Params(w_syn=W_SYN, sigma_v=2.0, r_spont=0.0))
b.spont_groups = groups
base = b.run(t_run=T_RUN, n_trials=TRIALS, seed=SEED) / (T_RUN / 1000)
bm, bs = base.mean(axis=1), base.std(axis=1, ddof=1)

print(f"\nw_syn={W_SYN}, {TRIALS} trials, {T_RUN:.0f} ms")
print("driven axis marked * -- that one is an echo of the input, not a readout\n")

DRIVEN = {"sugar": "taste", "looming": "defense", "co2": None, "vinegar": None}
rows = []
for cname, stim in CONDS.items():
    r = b.run(stim=stim, t_run=T_RUN, n_trials=TRIALS, r_poi=100.0,
              seed=SEED + (1 if cname == "sham" else 0)) / (T_RUN / 1000)
    rm = r.mean(axis=1)
    d = (rm - bm) / np.maximum(
        np.sqrt((r.std(axis=1, ddof=1) ** 2 + bs ** 2) / 2), 0.05)
    row = {"cond": cname}
    for aname, idx in AXES.items():
        row[aname] = round(float(rm[idx].mean()), 2)
        row[aname + "_d"] = round(float(np.abs(d[idx]).max()), 1)
    rows.append(row)

df = pd.DataFrame(rows)
hz = df[["cond"] + list(AXES)]
dd = df[["cond"] + [a + "_d" for a in AXES]]
dd.columns = ["cond"] + list(AXES)

print("=== axis mean firing (Hz) ===")
print(hz.to_string(index=False))
print("\n=== max |d| within each axis ===")
print(dd.to_string(index=False))

floor = float(dd[dd.cond == "sham"][list(AXES)].values.max())
print(f"\nsham floor (largest |d| anywhere with no stimulus): {floor:.2f}")

print("\n=== do non-driven axes respond? ===")
for _, r in dd[dd.cond != "sham"].iterrows():
    drv = DRIVEN.get(r["cond"])
    others = {a: r[a] for a in AXES if a != drv and r[a] > floor}
    tag = f"(driven: {drv})" if drv else "(no axis driven directly)"
    print(f"  {r['cond']:8} {tag:28} "
          f"non-driven axes past floor: {others if others else 'none'}")

print("\n=== do conditions separate? pairwise distance in axis space ===")
V = hz.set_index("cond")[list(AXES)]
Vz = (V - V.loc["sham"]) / V.loc["sham"].replace(0, np.nan).abs().fillna(1)
for a in V.index:
    for bb in V.index:
        if a < bb and "sham" not in (a, bb):
            print(f"  {a:8} vs {bb:8} L2 = "
                  f"{float(np.linalg.norm(Vz.loc[a] - Vz.loc[bb])):.2f}")

hz.to_csv(cache("state_under_stimulus.csv"), index=False)
print("\ndone (state)")
