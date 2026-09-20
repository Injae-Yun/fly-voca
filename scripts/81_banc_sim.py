"""Run the LIF simulator on the whole central nervous system.

Every previous experiment stopped at the descending neurons because FAFB ends
at the neck. On BANC the same stimulus travels sensory -> brain -> descending
-> nerve cord -> leg motor neuron with nothing hand-wired in between, so for
the first time we can ask what a stimulus does to the *legs*.

Scale check first: 188k neurons and 12.4M pairs against FAFB's 139k and 3.7M,
so roughly 3x the edges. The same GPU kernels should carry it.

Conditions kept deliberately parallel to doc 09's descending scan so the two
can be compared: sugar, CO2, vinegar, looming, and a sham that repeats
baseline under a different seed to set the false-positive floor.
"""
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from flyvoca.banc_graph import build, select, leg_motor, spontaneous_groups
from flyvoca.paths import cache
from bodysnatch.sim import Brain, Params

TRIALS = 16
T_RUN = 600.0
SEED = 5000

meta, W, Wslow, st = build()
print("graph:", st)
brain = Brain(W, Params(sigma_v=2.0, r_spont=0.0))
brain.spont_groups = spontaneous_groups(meta)
print("spontaneous drive per sensory class:",
      [(len(i), hz) for i, hz in brain.spont_groups])
print(f"device {brain.device}  neurons {brain.n:,}")

lm = leg_motor(meta)
dn = meta[meta.super_class == "descending"]
vnc = meta[meta.super_class == "ventral_nerve_cord_intrinsic"]
print(f"leg motor {len(lm)}  descending {len(dn)}  VNC intrinsic {len(vnc):,}")

sel = lambda **kw: select(meta, **kw)["idx"].values
CONDS = {
    "sham":    np.array([], dtype=np.int64),
    "sugar":   sel(prefix="LB3"),
    "co2":     sel(cell_type="ORN_V"),
    "vinegar": sel(cell_type="ORN_DM1"),
    "looming": np.concatenate([sel(cell_type="LC4"), sel(cell_type="LPLC2")]),
}
for k, v in CONDS.items():
    print(f"  {k:8} {len(v):5} neurons")

base = None
rows, rates = [], {}
for name, stim in CONDS.items():
    t0 = time.time()
    c = brain.run(stim=stim, t_run=T_RUN, n_trials=TRIALS, r_poi=100.0,
                  seed=SEED + (1 if name == "sham" else 0))
    r = c / (T_RUN / 1000.0)
    rates[name] = r
    if name == "sham":
        base = r
    m = r.mean(axis=1)
    bm = base.mean(axis=1)
    sd = np.sqrt((r.std(axis=1, ddof=1) ** 2 + base.std(axis=1, ddof=1) ** 2) / 2)
    d = (m - bm) / np.maximum(sd, 0.05)
    rows.append({
        "cond": name,
        "secs": round(time.time() - t0),
        "net_kHz": round(float(m.sum()) / 1000, 1),
        "DN_Hz": round(float(m[dn["idx"].values].mean()), 2),
        "VNC_Hz": round(float(m[vnc["idx"].values].mean()), 2),
        "legMN_Hz": round(float(m[lm["idx"].values].mean()), 2),
        "legMN_max": round(float(m[lm["idx"].values].max()), 1),
        "legMN_d>2": int((np.abs(d[lm["idx"].values]) > 2).sum()),
    })
    print(rows[-1], flush=True)

df = pd.DataFrame(rows)
print("\n" + df.to_string(index=False))

print("\n=== which leg motor neurons move, and on which legs ===")
bm = base.mean(axis=1)
for name in ("sugar", "co2", "looming"):
    m = rates[name].mean(axis=1)
    sd = np.sqrt((rates[name].std(axis=1, ddof=1) ** 2
                  + base.std(axis=1, ddof=1) ** 2) / 2)
    d = (m - bm) / np.maximum(sd, 0.05)
    hit = lm.assign(d=d[lm["idx"].values], hz=m[lm["idx"].values],
                    base=bm[lm["idx"].values])
    hit = hit[np.abs(hit.d) > 2]
    print(f"\n  {name}: {len(hit)} leg MNs past |d|>2")
    if len(hit):
        by = hit.groupby("nerve").size().to_dict()
        print("   ", by)
        top = hit.reindex(hit.d.abs().sort_values(ascending=False).index).head(5)
        for _, r in top.iterrows():
            print(f"    {str(r.cell_type)[:18]:18} {r.nerve[:28]:28} "
                  f"{r.base:6.2f} -> {r.hz:7.2f} Hz  d={r.d:+6.1f}")

np.savez(cache("banc_scan.npz"),
         **{k: v.mean(axis=1) for k, v in rates.items()},
         legmn_idx=lm["idx"].values, dn_idx=dn["idx"].values)
print("\ndone (banc sim)")
