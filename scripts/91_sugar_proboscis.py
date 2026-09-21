"""Was the question wrong? Sugar should drive the proboscis, not the legs.

Six hypotheses died trying to explain why sugar fails to move leg motor
neurons in BANC. The trace then showed sugar reaches the descending neurons
fine (17 past |d|>2, max 30.5) and dies crossing into the nerve cord, while
looming amplifies there (82 -> 261).

But that may not be a bug. In FAFB the validated sugar benchmark targeted
**MN9, a proboscis motor neuron** -- the animal extends its mouthparts, it
does not walk. Escape has a dedicated giant-fibre express route to the legs;
feeding has no reason to.

BANC names MN9 directly, and carries the mouthpart nerves (maxillary-labial,
pharyngeal) alongside the six leg nerves. So this asks the question that was
missing: does sugar reach the *proboscis* motor pool?

  reaches it   the circuit works, the leg result was never a bug, and six
               hypotheses were chasing a question that had no answer
  does not     then something is genuinely broken downstream of the DNs

Looming and CO2 ride along as contrasts, and the leg pool stays in the table
so both targets are read side by side rather than one at a time.
"""
import numpy as np
import pandas as pd

from flyvoca.banc_graph import build, select, leg_motor, spontaneous_groups
from flyvoca.sim import Brain, Params

W_SYN, TRIALS, T_RUN, SEED = 0.16, 16, 600.0, 12000

meta, W, _, st = build()
groups = spontaneous_groups(meta)
mo = meta[meta.super_class == "motor"]
nerve = mo.nerve.astype(str)

TARGETS = {
    "MN9": meta[meta.cell_type == "MN9"]["idx"].values,
    "proboscis pool": mo[nerve.str.contains("maxillary-labial|pharyngeal",
                                            case=False, na=False)]["idx"].values,
    "leg pool": leg_motor(meta)["idx"].values,
    "all motor": mo["idx"].values,
}
for k, v in TARGETS.items():
    print(f"  {k:16} {len(v):4} neurons")

sel = lambda **kw: select(meta, **kw)["idx"].values
CONDS = {
    "sugar": sel(prefix="LB3"),
    "looming": np.concatenate([sel(cell_type="LC4"), sel(cell_type="LPLC2")]),
    "co2": sel(cell_type="ORN_V"),
}

b = Brain(W, Params(w_syn=W_SYN, sigma_v=2.0, r_spont=0.0))
b.spont_groups = groups
base = b.run(t_run=T_RUN, n_trials=TRIALS, seed=SEED) / (T_RUN / 1000)
bm, bs = base.mean(axis=1), base.std(axis=1, ddof=1)

print(f"\nw_syn={W_SYN}, {TRIALS} trials, {T_RUN:.0f} ms\n")
rows = []
for cname, stim in CONDS.items():
    r = b.run(stim=stim, t_run=T_RUN, n_trials=TRIALS, r_poi=100.0,
              seed=SEED) / (T_RUN / 1000)
    d = (r.mean(axis=1) - bm) / np.maximum(
        np.sqrt((r.std(axis=1, ddof=1) ** 2 + bs ** 2) / 2), 0.05)
    for tname, idx in TARGETS.items():
        rows.append({"cond": cname, "target": tname, "n": len(idx),
                     "base_Hz": round(float(bm[idx].mean()), 2),
                     "stim_Hz": round(float(r.mean(axis=1)[idx].mean()), 2),
                     "max_|d|": round(float(np.abs(d[idx]).max()), 1),
                     "n_d>2": int((np.abs(d[idx]) > 2).sum())})
        print(rows[-1], flush=True)
    print()

df = pd.DataFrame(rows)
print("n_d>2 --------------------------------------------")
print(df.pivot(index="target", columns="cond", values="n_d>2")
        .reindex(TARGETS).to_string())
print("\nmax |d| ------------------------------------------")
print(df.pivot(index="target", columns="cond", values="max_|d|")
        .reindex(TARGETS).to_string())

print("\n=== wiring, independent of dynamics: sugar -> each motor pool ===")
Wa = abs(W)
lb3 = sel(prefix="LB3")
for tname, idx in TARGETS.items():
    v = np.zeros(len(meta), dtype=np.float32)
    v[lb3] = 1.0
    line = []
    for h in range(1, 4):
        v = Wa.T @ v
        v[lb3] = 0.0
        line.append(f"hop{h} {int((v[idx] > 0).sum()):4}/{len(idx)}")
    print(f"  {tname:16} " + "  ".join(line))
print("\ndone (proboscis)")
