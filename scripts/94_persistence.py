"""Does the brain hold a state after the cause is gone?

Everything measured so far was recorded *during* stimulation, which cannot
distinguish a state from an echo of the input. That is why the state vector
looked like nothing but the driven axis: of course taste reads 100 Hz while
sugar neurons are being driven at 100 Hz.

The defining property of an internal state is that it outlasts its cause. A
real fly stays agitated for seconds after a looming threat passes, and pC1
activity persists for minutes after courtship contact. A taste reflex does
not -- remove the sugar and the proboscis stops.

So: drive, release, and watch. Three stimuli chosen because they should differ:

  looming    should persist (defensive arousal)
  pC1        should persist (courtship state)
  sugar      should NOT persist (a reflex)

Sham is the same protocol with nothing driven, so persistence is measured
against how much the network drifts on its own.

The monoamine graph -- 1.17M pairs of dopamine, octopamine, serotonin and
tyramine, built back in doc 13 and never used -- is read out alongside,
because arousal in insects is carried by octopamine and if anything holds a
state it should be visible there.
"""
import numpy as np
import pandas as pd

from flyvoca.banc_graph import build, select, spontaneous_groups
from flyvoca.paths import cache
from bodysnatch.sim import Brain, Params

W_SYN, TRIALS, BLOCK = 0.16, 10, 250.0
ON, OFF = 4, 12          # blocks driven, then blocks released

meta, W, Wslow, st = build()
groups = spontaneous_groups(meta)
t = meta.cell_type.astype(str)
nt = meta.neurotransmitter_predicted.astype(str)
ntv = meta.neurotransmitter_verified.astype(str)


def ct(*pats):
    m = np.zeros(len(meta), dtype=bool)
    for p in pats:
        m |= t.str.fullmatch(p, na=False).values | t.str.startswith(p, na=False).values
    return meta.index.values[m]


def amine(name):
    return meta[(nt == name) | ntv.str.contains(name, na=False)]["idx"].values


READ = {
    "octopamine": amine("octopamine"),
    "dopamine": amine("dopamine"),
    "serotonin": amine("serotonin"),
    "pC1": ct("pC1"),
    "aSP": ct("aSP"),
    "defense": ct("LC4", "LPLC2", "LC6", "DNp01", "DNp11"),
    "taste": ct("LB3"),
    "DN": meta[meta.super_class == "descending"]["idx"].values,
}
for k, v in READ.items():
    print(f"  {k:11} {len(v):5}")

sel = lambda **kw: select(meta, **kw)["idx"].values
STIM = {
    "sham": np.array([], dtype=np.int64),
    "looming": np.concatenate([sel(cell_type="LC4"), sel(cell_type="LPLC2")]),
    "pC1": ct("pC1"),
    "sugar": sel(prefix="LB3"),
}

print(f"\n{ON} blocks driven then {OFF} released, {BLOCK:.0f} ms each, "
      f"{TRIALS} trials, w_syn={W_SYN}\n")

traces = {}
for name, stim in STIM.items():
    b = Brain(W, Params(w_syn=W_SYN, sigma_v=2.0, r_spont=0.0))
    b.spont_groups = groups
    state, rows = None, []
    for blk in range(ON + OFF):
        driving = blk < ON
        counts, state = b.run(stim=stim if driving else (), t_run=BLOCK,
                              n_trials=TRIALS, r_poi=100.0, seed=15000 + blk,
                              state=state, return_state=True)
        m = counts.mean(axis=1) / (BLOCK / 1000)
        row = {"blk": blk, "t_ms": int((blk + 1) * BLOCK),
               "phase": "on" if driving else "off"}
        row.update({k: round(float(m[v].mean()), 3) for k, v in READ.items()})
        rows.append(row)
    traces[name] = pd.DataFrame(rows)
    d = traces[name]
    print(f"--- {name} ---")
    print(d[["t_ms", "phase", "octopamine", "dopamine", "pC1", "defense",
             "taste", "DN"]].to_string(index=False), flush=True)
    print()

print("=== persistence: activity after release, relative to during ===")
print("   (last 4 off-blocks vs the 4 on-blocks, and vs sham's own drift)")
sham = traces["sham"]
rows = []
for name, d in traces.items():
    if name == "sham":
        continue
    on = d[d.phase == "on"]
    tail = d[d.phase == "off"].tail(4)
    early = d[d.phase == "off"].head(2)
    r = {"stim": name}
    for k in ("octopamine", "dopamine", "serotonin", "pC1", "defense", "taste", "DN"):
        base = float(sham[k].tail(4).mean())
        on_v, off_v = float(on[k].mean()), float(tail[k].mean())
        r[k] = f"{on_v:.2f}->{off_v:.2f} (sham {base:.2f})"
        r[k + "_ret"] = round((off_v - base) / max(on_v - base, 1e-6), 3)
    rows.append(r)
P = pd.DataFrame(rows)
print(P[["stim"] + [c for c in P.columns if c.endswith("_ret")]].to_string(index=False))
print("\n  _ret = (after release - sham) / (during - sham)")
print("  near 0 = the state collapsed with the stimulus (a reflex)")
print("  near 1 = it held (a state)")

for name, d in traces.items():
    d.to_csv(cache(f"persist_{name}.csv"), index=False)
print("\ndone (persistence)")
