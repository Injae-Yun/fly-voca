"""What is actually persisting in the "defense" axis?

Two problems with how that number was produced.

The axis was built with prefix matching on "LC4", which swept in LC40, LC41,
LC43, LC44, LC45, LC46 and LC46a -- 101 unrelated lobula columnar types. The
FAFB state.py deliberately separated `exact` from `prefix` to prevent exactly
this; the BANC script reintroduced it.

Worse, the stimulated cells are *inside* the axis. Of 529 members, 295 are
the LC4+LPLC2 being driven, leaving only 234 genuinely downstream. A retained
signal could be ringing in the stimulated population itself rather than a
state held by anything downstream.

So the axis is split into parts that answer different questions:

  stimulated     LC4 + LPLC2 -- what we drove. Persistence here is afterglow.
  contaminant    LC40..LC46 -- swept in by the prefix bug, should not respond
                 at all. If they do, the whole grouping is suspect.
  downstream     LC6, DNp01, DNp11 -- not driven, genuinely postsynaptic.
                 This is where a real defensive state would live.

Run at tau 5 (no slow current) and tau 500 to see which part the slow current
was holding.
"""
import numpy as np
import pandas as pd

from flyvoca.banc_graph import build, select, spontaneous_groups
from flyvoca.paths import cache
from bodysnatch.sim import Brain, Params

W_SYN, TRIALS, BLOCK = 0.16, 10, 250.0
ON, OFF = 4, 10

meta, W, _, st = build()
groups = spontaneous_groups(meta)
t = meta.cell_type.astype(str)


def exact(*names):
    return meta.index.values[t.isin(names).values]


STIM_CELLS = exact("LC4", "LPLC2")
PARTS = {
    "stimulated (LC4,LPLC2)": STIM_CELLS,
    "contaminant (LC40-46)": exact("LC40", "LC41", "LC43", "LC44", "LC45",
                                   "LC46", "LC46a"),
    "downstream LC6": exact("LC6"),
    "downstream DNp01": exact("DNp01"),
    "downstream DNp11": exact("DNp11"),
    "all descending": meta[meta.super_class == "descending"]["idx"].values,
}
for k, v in PARTS.items():
    print(f"  {k:26} {len(v):5}")

# what the slow current is applied to: the old, contaminated axis
OLD_AXIS = np.unique(np.concatenate([v for k, v in PARTS.items()
                                     if "descending" not in k]))
print(f"\nold 'defense' axis = {len(OLD_AXIS)} cells "
      f"({len(STIM_CELLS)} of them stimulated)")

rows = []
for tau in (5.0, 500.0):
    for label, stim in (("looming", STIM_CELLS),
                        ("sham", np.array([], dtype=np.int64))):
        b = Brain(W, Params(w_syn=W_SYN, sigma_v=2.0, r_spont=0.0))
        b.spont_groups = groups
        if tau > 5.0:
            b.tau_per_neuron = (OLD_AXIS, tau)
        state, trace = None, []
        for blk in range(ON + OFF):
            on = blk < ON
            c, state = b.run(stim=stim if on else (), t_run=BLOCK,
                             n_trials=TRIALS, r_poi=100.0, seed=18000 + blk,
                             state=state, return_state=True)
            m = c.mean(axis=1) / (BLOCK / 1000)
            trace.append({"blk": blk, "phase": "on" if on else "off",
                          **{k: float(m[v].mean()) for k, v in PARTS.items()}})
        d = pd.DataFrame(trace)
        for part in PARTS:
            rows.append({"tau": tau, "cond": label, "part": part,
                         "on": round(d[d.phase == "on"][part].mean(), 2),
                         "off_late": round(d[d.phase == "off"][part].tail(4).mean(), 2)})
        print(f"  ran tau={tau} {label}", flush=True)

df = pd.DataFrame(rows)
piv = df.pivot_table(index="part", columns=["tau", "cond"],
                     values="off_late").reindex(PARTS)
print("\n=== firing after release (off-late) ===")
print(piv.to_string())

print("\n=== retention, measured against sham at the same tau ===")
out = []
for tau in (5.0, 500.0):
    for part in PARTS:
        g = df[(df.tau == tau) & (df.part == part)]
        loom = g[g.cond == "looming"].iloc[0]
        sh = g[g.cond == "sham"].iloc[0]
        denom = loom["on"] - sh["off_late"]
        ret = (loom["off_late"] - sh["off_late"]) / denom if abs(denom) > 0.05 else np.nan
        out.append({"tau": tau, "part": part,
                    "on": loom["on"], "off_late": loom["off_late"],
                    "sham": sh["off_late"],
                    "retained": round(float(ret), 3) if ret == ret else None})
O = pd.DataFrame(out)
print(O.to_string(index=False))
O.to_csv(cache("defense_decomposed.csv"), index=False)
print("\ndone (decompose)")
