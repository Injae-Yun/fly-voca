"""(가) Give only the state cells a slow current, and see if anything holds.

Doc 05 raised tau globally and the compass still would not hold a bump. But
globally is the wrong place for it: in the animal a slow recurrent current
belongs to particular circuits, not to every synapse. That variant was never
tried.

Doc 15 rules out the obvious carrier -- looming delivers a median of one
synapse per octopaminergic cell against the 44 needed at this operating
point, so arousal cannot be reached through octopamine here. What does fire
is the state population itself: pC1 reaches 102 Hz under drive and defense
59 Hz, then both collapse to sham within one 250 ms block once released.

So tau is raised only on those cells. Two controls decide whether any
persistence is real rather than a leak:

  slow + no stimulus    if activity climbs on its own, the slow current is
                        destabilising the network, not storing anything
  slow on the reflex    sugar must still collapse. A change that makes
                        everything persist has produced a leak, not a state

Persistence is scored against sham's own drift over the same blocks, because
the network wanders a little regardless.
"""
import numpy as np
import pandas as pd

from flyvoca.banc_graph import build, select, spontaneous_groups
from flyvoca.paths import cache
from flyvoca.sim import Brain, Params

W_SYN, TRIALS, BLOCK = 0.16, 10, 250.0
ON, OFF = 4, 10
TAUS = (5.0, 50.0, 200.0, 500.0)

meta, W, _, st = build()
groups = spontaneous_groups(meta)
t = meta.cell_type.astype(str)


def ct(*pats):
    m = np.zeros(len(meta), dtype=bool)
    for p in pats:
        m |= t.str.fullmatch(p, na=False).values | t.str.startswith(p, na=False).values
    return meta.index.values[m]


PC1 = ct("pC1")
DEFENSE = ct("LC4", "LPLC2", "LC6", "DNp01", "DNp11")
TASTE = ct("LB3")
READ = {"pC1": PC1, "defense": DEFENSE, "taste": TASTE,
        "DN": meta[meta.super_class == "descending"]["idx"].values}
LOOM = np.concatenate([select(meta, cell_type="LC4")["idx"].values,
                       select(meta, cell_type="LPLC2")["idx"].values])
print(f"pC1 {len(PC1)}  defense {len(DEFENSE)}  taste {len(TASTE)}")


def run(stim, slow_idx, tau_slow):
    b = Brain(W, Params(w_syn=W_SYN, sigma_v=2.0, r_spont=0.0))
    b.spont_groups = groups
    if slow_idx is not None and tau_slow > 5.0:
        b.tau_per_neuron = (np.asarray(slow_idx), float(tau_slow))
    state, rows = None, []
    for blk in range(ON + OFF):
        on = blk < ON
        c, state = b.run(stim=stim if on else (), t_run=BLOCK, n_trials=TRIALS,
                         r_poi=100.0, seed=17000 + blk, state=state,
                         return_state=True)
        m = c.mean(axis=1) / (BLOCK / 1000)
        rows.append({"blk": blk, "phase": "on" if on else "off",
                     **{k: round(float(m[v].mean()), 3) for k, v in READ.items()}})
    return pd.DataFrame(rows)


def retention(d, sham, key):
    on = d[d.phase == "on"][key].mean()
    late = d[d.phase == "off"][key].tail(4).mean()
    base = sham[sham.phase == "off"][key].tail(4).mean()
    denom = on - base
    if abs(denom) < 0.05:
        return np.nan
    return float((late - base) / denom)


CASES = [
    ("pC1 drive", PC1, PC1, "pC1"),
    ("looming", LOOM, DEFENSE, "defense"),
    ("sugar (reflex control)", select(meta, prefix="LB3")["idx"].values, TASTE, "taste"),
]

print(f"\n{ON} on / {OFF} off blocks of {BLOCK:.0f} ms, {TRIALS} trials\n")
out = []
for tau in TAUS:
    sham_slow = run(np.array([], dtype=np.int64), PC1, tau)
    for label, stim, slow, key in CASES:
        d = run(stim, slow, tau)
        on = d[d.phase == "on"][key].mean()
        e = d[d.phase == "off"][key].head(1).values[0]
        late = d[d.phase == "off"][key].tail(4).mean()
        out.append({"tau_ms": tau, "case": label, "watch": key,
                    "on": round(on, 2), "off_1st": round(float(e), 2),
                    "off_late": round(late, 2),
                    "retained": round(retention(d, sham_slow, key), 3)
                    if not np.isnan(retention(d, sham_slow, key)) else None})
        print(out[-1], flush=True)
    # control: slow current with nothing driven
    c = {"tau_ms": tau, "case": "CONTROL slow, no stimulus", "watch": "pC1",
         "on": round(sham_slow[sham_slow.phase == "on"].pC1.mean(), 2),
         "off_1st": round(float(sham_slow[sham_slow.phase == "off"].pC1.values[0]), 2),
         "off_late": round(sham_slow[sham_slow.phase == "off"].pC1.tail(4).mean(), 2),
         "retained": None}
    out.append(c)
    print(c, flush=True)
    print()

df = pd.DataFrame(out)
print(df.to_string(index=False))
df.to_csv(cache("selective_slow.csv"), index=False)
print("\nretained ~0 = collapsed with the stimulus; ~1 = held.")
print("the sugar row must stay near 0 at every tau, or the effect is a leak.")
print("done (selective slow)")
