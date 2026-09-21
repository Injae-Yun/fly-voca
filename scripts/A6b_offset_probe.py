"""What did the loop cells actually receive from the slow layer?

A6 found no persistence at gain 6 even with a 10 s drive. The arithmetic
predicted ~3 mV on pC1d/e from NPFL1-I alone, and more from OA-VUMa1 if it
was recruited. Before reading "no persistence" as a result, check that the
offset arrived: replay aIPg and pC1d+e at gain 6 / tau 20 / 10 s, and print
per block the v_offset and rate of pC1d/e, aIPg, and the modulator cells A5c
identified, plus how many cells brain-wide sit within 1 mV of the cap.
"""
import os
import numpy as np
import pandas as pd

from flyvoca.graph import build
from flyvoca.paths import cache
from bodysnatch.sim import Brain, Params
from bodysnatch.monoamine import split_by_transmitter, SlowEdges

W_SYN, TRIALS, BLOCK = 0.20, 8, 250.0
ON, OFF, TAU, GAIN = 40, 16, 20.0, 6.0
SIGNS = {"OCT": +1, "SER": +1, "DA": +1}

meta, W, Ws, _ = build()
t = meta["cell_type"].astype(str)
add = meta["additional_type(s)"].astype(str)
ex = lambda *n: meta[t.isin(n)]["idx"].values
PC1DE = np.concatenate([ex("pC1d"), ex("pC1e")])
AIPG = meta[add.str.contains("aIPg")]["idx"].values
SPONT = meta[meta["super_class"].isin(["sensory", "sensory_ascending"])]["idx"].values
WATCH = {"pC1d/e": PC1DE, "aIPg": AIPG, "NPFL1-I": ex("NPFL1-I"),
         "OA-VUMa1": ex("OA-VUMa1"), "OA-VUMa8": ex("OA-VUMa8"),
         "SMP385": ex("SMP385"), "DNp27": ex("DNp27")}
WT = split_by_transmitter(Ws, meta["slow_class"].values)
brain = Brain(W, Params(w_syn=W_SYN, sigma_v=2.0, r_spont=2.0), spont=SPONT)
rest = brain.run(t_run=1000.0, n_trials=TRIALS, seed=22001).mean(axis=1)

for name, stim in (("aIPg", AIPG), ("pC1d+e", PC1DE)):
    slow = SlowEdges(WT, TAU, GAIN, SIGNS, n_trials=TRIALS, charge_s=2.0)
    slow.settle(rest)
    state, rows = None, []
    print(f"\n=== {name} 10 s at 100 Hz, gain {GAIN}, tau {TAU} ===")
    for k in range(ON + OFF):
        off = slow.v_offset()
        c, state = brain.run(stim=stim if k < ON else (), t_run=BLOCK, n_trials=TRIALS,
                             r_poi=100.0, seed=26000 + k, state=state,
                             v_offset=off, return_state=True)
        r = (c / (BLOCK / 1000)).astype(np.float32)
        slow.step(r, BLOCK / 1000)
        if k % 4 == 3 or k == ON:
            om = off.mean(axis=1)
            row = {"t_s": round((k + 1) * BLOCK / 1000, 2), "phase": "on" if k < ON else "off",
                   "n_at_cap": int((om > GAIN - 1).sum()), "n_gt_3mV": int((om > 3).sum())}
            for w, idx in WATCH.items():
                row[f"{w}_mV"] = round(float(om[idx].mean()), 2)
                row[f"{w}_Hz"] = round(float(r[idx].mean()), 1)
            rows.append(row); print(row, flush=True)
    pd.DataFrame(rows).to_csv(cache(f"offset_probe_{name.replace('/', '')}.csv"), index=False)
print("\ndone (offset probe)")
