"""Why does the loop hold in the fly and not here? Two candidates, measured.

The paper drove pC1d/e optogenetically for **5 minutes** and saw activity
persist for minutes afterwards. I drove for **1 second** (4 x 250 ms) and
watched for 3. A recurrent loop has to charge, and 300x less drive is the
most obvious difference between the two experiments.

The second candidate is loop gain. A recurrent circuit sustains itself only
if the round trip returns at least as much drive as it received. That is a
number this graph can produce: hold pC1 at a known rate, read what aIPg does,
then hold aIPg and read pC1. The product of the two slopes is the loop gain,
and if it is below 1 no amount of charging will help -- which would be a real
finding rather than a modelling excuse.

Worth stating plainly: the paper presents no computational model. Nobody has
shown this loop sustains in simulation. "It works in the fly" is not the same
as "it works in a leaky integrate-and-fire model of the fly", and if the gain
comes out below 1 the gap between those two statements is the result.
"""
import numpy as np
import pandas as pd

from flyvoca.graph import build
from flyvoca.paths import cache
from flyvoca.sim import Brain, Params

W_SYN, TRIALS = 0.20, 10

meta, W, _, st = build()
t = meta["cell_type"].astype(str)
ex = lambda *n: meta[t.isin(n)]["idx"].values
PC1 = ex("pC1a", "pC1b", "pC1c", "pC1d", "pC1e")
AIPG = ex("CB2131")
LOOP = np.unique(np.concatenate([PC1, AIPG]))
READ = {"pC1": PC1, "aIPg": AIPG}
print(f"pC1 {len(PC1)}  aIPg(CB2131) {len(AIPG)}")

b = Brain.from_meta(W, meta, Params(w_syn=W_SYN, sigma_v=2.0, r_spont=2.0))

# ---------------------------------------------------------------- loop gain
print("\n=== loop gain: drive one half, read the other ===")
rows = []
for rate in (25.0, 50.0, 100.0, 200.0):
    for src, dst, label in ((PC1, AIPG, "pC1 -> aIPg"), (AIPG, PC1, "aIPg -> pC1")):
        c = b.run(stim=src, t_run=1000.0, n_trials=TRIALS, r_poi=rate, seed=20000)
        m = c.mean(axis=1)
        rows.append({"arm": label, "drive_Hz": rate,
                     "src_Hz": round(float(m[src].mean()), 1),
                     "dst_Hz": round(float(m[dst].mean()), 2)})
        print(rows[-1], flush=True)
G = pd.DataFrame(rows)
slopes = {}
for arm, g in G.groupby("arm"):
    s = np.polyfit(g.src_Hz.values, g.dst_Hz.values, 1)[0]
    slopes[arm] = float(s)
    print(f"  slope {arm}: {s:.4f} Hz out per Hz in")
gain = slopes["pC1 -> aIPg"] * slopes["aIPg -> pC1"]
print(f"\n  round-trip loop gain = {gain:.4f}")
print("  >= 1 can sustain; < 1 decays no matter how long it is charged")

# ------------------------------------------------------------ charging time
print("\n=== does longer drive change anything? ===")
BLOCK = 250.0
OFF = 8
rows = []
for on_s in (1.0, 5.0, 20.0, 60.0):
    on_blocks = int(on_s * 1000 / BLOCK)
    bb = Brain.from_meta(W, meta, Params(w_syn=W_SYN, sigma_v=2.0, r_spont=2.0))
    state, trace = None, []
    for blk in range(on_blocks + OFF):
        on = blk < on_blocks
        c, state = bb.run(stim=LOOP if on else (), t_run=BLOCK, n_trials=TRIALS,
                          r_poi=100.0, seed=21000 + blk, state=state,
                          return_state=True)
        m = c.mean(axis=1) / (BLOCK / 1000)
        trace.append({"blk": blk, "phase": "on" if on else "off",
                      **{k: float(m[v].mean()) for k, v in READ.items()}})
    d = pd.DataFrame(trace)
    on_v = d[d.phase == "on"].tail(4).pC1.mean()
    off1 = d[d.phase == "off"].pC1.values[0]
    off_l = d[d.phase == "off"].pC1.tail(4).mean()
    rows.append({"drive_s": on_s, "on_blocks": on_blocks,
                 "pC1_on": round(on_v, 1), "pC1_off_1st": round(float(off1), 3),
                 "pC1_off_late": round(off_l, 3)})
    print(rows[-1], flush=True)
C = pd.DataFrame(rows)

# --------------------------------------------- what gain would be required
print("\n=== what would it take? ===")
need = 7.0 / W_SYN
Wa = abs(W)
a2p = np.asarray(Wa[np.ix_(AIPG, PC1)].sum(axis=0)).ravel()
p2a = np.asarray(Wa[np.ix_(PC1, AIPG)].sum(axis=0)).ravel()
print(f"  threshold needs {need:.0f} coincident synapses at w_syn={W_SYN}")
print(f"  aIPg->pC1 per target: {[int(x) for x in a2p]}")
print(f"  pC1->aIPg per target: {[int(x) for x in p2a]}")
print(f"  but those arrive over a {Params().tau:.0f} ms window -- a presynaptic")
print(f"  cell at {100:.0f} Hz delivers one spike every 10 ms, so a 100-synapse")
print(f"  connection from ONE cell contributes ~{100*0.01*100*W_SYN:.1f} mV/tau, not 100x0.2")

G.to_csv(cache("loop_gain.csv"), index=False)
C.to_csv(cache("loop_charging.csv"), index=False)
print("\ndone (loop gain)")
