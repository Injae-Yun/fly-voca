"""Is vinegar's channel common-mode -- forward drive rather than turn?

The scan called vinegar null at |d| > 2, which was too blunt. It sits at
max |d| = 1.89 against a sham floor of 0.83, with 33 neurons past |d| > 1 and
a top rate change of +36 Hz. Not nothing; below the bar.

And its candidates look structured rather than scattered: DNa10 rises on both
sides by nearly the same amount (d = 1.77 and 1.61). A turn command is
antisymmetric across the midline; something that moves both sides together is
not steering. That is exactly the channel three previous experiments implied
without being able to name.

Two tests:

  dose      drive the vinegar ORNs harder. A real channel should grow with
            stimulus; noise should not order itself by dose.
  symmetry  for every responsive DN type with both sides present, compare the
            common-mode change (L+R)/2 against the differential (L-R). A
            speed channel is all common mode; a steering channel is all
            differential. Sugar, CO2 and looming come along as comparisons,
            since looming *should* contain a strong bilateral escape component.
"""
import numpy as np
import pandas as pd

from flyvoca.graph import build
from flyvoca.paths import cache
from flyvoca.sim import Brain, Params

TRIALS = 24
T_RUN = 800.0
SEED = 3000

meta, W, _, _ = build()
sel = lambda ct: meta[meta.cell_type == ct]["idx"].values
brain = Brain.from_meta(W, meta, Params(sigma_v=2.0, r_spont=2.0))

dn = meta[meta.super_class == "descending"]
dn_idx = dn["idx"].values
info = dn[["label", "side"]].reset_index(drop=True)

base = brain.run(stim=(), t_run=T_RUN, n_trials=TRIALS, r_poi=100.0, seed=SEED)
base_dn = base[dn_idx] / (T_RUN / 1000.0)
bm = base_dn.mean(axis=1)
bs = base_dn.std(axis=1, ddof=1)

print("=== dose-response: does the vinegar channel grow with drive? ===")
dm1 = sel("ORN_DM1")
dose_rows = []
for rate in (25.0, 50.0, 100.0, 200.0, 400.0):
    c = brain.run(stim=dm1, t_run=T_RUN, n_trials=TRIALS, r_poi=rate, seed=SEED)
    r = c[dn_idx] / (T_RUN / 1000.0)
    d = (r.mean(axis=1) - bm) / np.maximum(np.sqrt((r.std(axis=1, ddof=1) ** 2 + bs ** 2) / 2), 0.05)
    dose_rows.append({"rate_Hz": rate, "DN_mean": round(float(r.mean()), 2),
                      "max_d": round(float(np.abs(d).max()), 2),
                      "n_d>1": int((np.abs(d) > 1).sum()),
                      "n_d>2": int((np.abs(d) > 2).sum()),
                      "n_d>3": int((np.abs(d) > 3).sum())})
    print(dose_rows[-1], flush=True)

CONDS = {
    "vinegar": dict(stim=dm1, rate=400.0),
    "co2":     dict(stim=sel("ORN_V"), rate=100.0),
    "sugar":   dict(stim=sel("LB3"), rate=100.0),
    "looming": dict(stim=np.concatenate([sel("LC4"), sel("LPLC2")]), rate=100.0),
}

print("\n=== common mode vs differential, per condition ===")
summary = []
for name, cfg in CONDS.items():
    c = brain.run(stim=cfg["stim"], t_run=T_RUN, n_trials=TRIALS,
                  r_poi=cfg["rate"], seed=SEED)
    r = c[dn_idx] / (T_RUN / 1000.0)
    delta = r.mean(axis=1) - bm
    t = info.copy()
    t["delta"] = delta
    # types with exactly one left and one right
    pv = t[t.side.isin(["left", "right"])].pivot_table(
        index="label", columns="side", values="delta", aggfunc="mean")
    pv = pv.dropna()
    common = (pv["left"] + pv["right"]) / 2
    diff = (pv["left"] - pv["right"]) / 2
    resp = (np.abs(common) + np.abs(diff)) > 1.0       # responsive types only
    cc, dd = common[resp], diff[resp]
    frac = float(np.abs(cc).sum() / max(np.abs(cc).sum() + np.abs(dd).sum(), 1e-9))
    summary.append({"cond": name, "types_paired": int(resp.sum()),
                    "common_mag": round(float(np.abs(cc).mean()), 2),
                    "diff_mag": round(float(np.abs(dd).mean()), 2),
                    "common_fraction": round(frac, 3)})
    print(summary[-1], flush=True)
    top = pv[resp].assign(common=cc, diff=dd).reindex(
        np.abs(cc).sort_values(ascending=False).index).head(6)
    print(top.round(2).to_string())
    print()

print(pd.DataFrame(summary).to_string(index=False))
print("\ncommon_fraction near 1 = the condition moves both sides together")
print("(a turn command would sit near 0)")
print("done (cm)")
