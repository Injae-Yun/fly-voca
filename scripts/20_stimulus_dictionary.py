"""A stimulus -> response dictionary: what does this brain do, and while doing
it, what state is it in?

Each condition drives one identified population and reads back three things:

  state vector   mean rate of each of the eight internal-state axes
  action         the most active descending neurons -- the brain's output to
                 the body, which is where behaviour actually leaves the head
  steering       left/right DNa02, whose difference is the turn command

Absolute rates run high (see core/docs/02-calibration.md); every number here is
meant to be read against the unstimulated baseline in the first column.
"""
import numpy as np
import pandas as pd

from flyvoca.graph import build, select
from bodysnatch.sim import Brain
from bodysnatch.state import AXES, read as read_state

CONDITIONS = {
    "baseline":   dict(),
    "sugar":      dict(cell_type="LB3"),
    "looming":    dict(cell_type=["LC4", "LPLC2"]),
    "courtship":  dict(prefix="pC1"),
    "thirst":     dict(cell_type="ISN"),
    "clock_am":   dict(cell_type="s-LNv"),
    "sleep":      dict(cell_type="ER5"),
    "insulin":    dict(cell_type="m_NSC_DILP"),
}

meta, W, _, _ = build()
brain = Brain(W)
lab = meta.set_index("idx")[["label", "side", "super_class"]]
dn = meta[meta["super_class"] == "descending"]

states, actions, dna02 = {}, {}, {}
for name, sel in CONDITIONS.items():
    stim = select(meta, **sel)["idx"].values if sel else np.array([], dtype=np.int64)
    counts = brain.run(stim=stim, t_run=1000.0, n_trials=20, seed=2)
    m, _ = Brain.rate(counts, 1000.0)

    states[name] = read_state(meta, m)
    states[name]["_stim_n"] = len(stim)
    states[name]["_net_Hz"] = int(m.sum())

    d = m[dn["idx"].values]
    top = np.argsort(d)[::-1][:6]
    actions[name] = [(dn.iloc[i]["label"], dn.iloc[i]["side"], round(float(d[i]), 1))
                     for i in top]
    dna02[name] = {sd: float(m[g["idx"].values].mean())
                   for sd, g in meta[meta["cell_type"] == "DNa02"].groupby("side")}
    print(f"  ran {name} (n={len(stim)})", flush=True)

print("\n=== internal state vector (mean Hz per axis) ===")
S = pd.DataFrame(states)
print(S.round(2).to_string())

print("\n=== delta from baseline ===")
axis_names = [a.name for a in AXES]
D = S.loc[axis_names].sub(S.loc[axis_names, "baseline"], axis=0).drop(columns="baseline")
print(D.round(2).to_string())

print("\n=== most active descending neurons (the action readout) ===")
for name, rows in actions.items():
    s = "  ".join(f"{l}/{sd[0]} {hz}" for l, sd, hz in rows)
    print(f"{name:<10} {s}")

print("\n=== steering: DNa02 left vs right ===")
a02 = meta[meta["cell_type"] == "DNa02"]
for name, sel in CONDITIONS.items():
    pass
