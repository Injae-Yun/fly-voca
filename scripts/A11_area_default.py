"""Make area-scaled sensitivity the default: find its w_syn and check what moves.

Doc 19 / A10 showed that giving every cell the input sensitivity its surface
area implies (flyvoca.morph) makes the reflex benchmarks stronger and the
central brain quieter than the uniform model -- at w_syn 0.4. Before it
becomes the default, the calibration has to be chosen the way doc 02 chose
0.2 for the uniform model (regime, not a single number), and everything the
earlier documents rest on has to be re-read under it:

    resting state      rate by super_class (a resting fly sits at ~1-2 Hz)
    reflexes           sugar -> MN9 (doc 01/02), looming -> leg motor (doc 09)
    steering path      PFL3 left -> DNa02 right, contralateral (doc 01/06)
    state readout      the eight axes under sugar / looming vs sham (doc 14)

Uniform 0.2 is the row everything else is compared with. Same seeds
throughout.
"""
import numpy as np
import pandas as pd

from flyvoca.graph import build
from flyvoca.paths import cache
from flyvoca.sim import Brain, Params
from voca.state import AXES, read as read_state

TRIALS, T = 8, 800.0
meta, W, _, _ = build()
t = meta["cell_type"].astype(str)
ex = lambda *n: meta[t.isin(n)]["idx"].values
SUGAR = meta[t.str.startswith("LB3", na=False)]["idx"].values
MN9 = meta[meta.index == 720575940660219265]["idx"].values
LOOM = ex("LC4", "LPLC2")
MOTOR = meta[meta.super_class == "motor"]["idx"].values
PFL3 = {s: g["idx"].values for s, g in meta[t == "PFL3"].groupby("side")}
DNA02 = {s: g["idx"].values for s, g in meta[t == "DNa02"].groupby("side")}
CLASSES = ["sensory", "central", "visual_projection", "descending", "motor"]


def d_of(r, base):
    bm, bs = base.mean(axis=1), base.std(axis=1, ddof=1)
    return (r.mean(axis=1) - bm) / np.maximum(np.sqrt((r.std(axis=1, ddof=1) ** 2 + bs ** 2) / 2), 0.05)


def measure(b, label):
    base = b.run(t_run=T, n_trials=TRIALS, seed=22001) / (T / 1000)
    bm = base.mean(axis=1)
    row = {"model": label, "rest_all_Hz": round(float(bm.mean()), 2)}
    for c in CLASSES:
        row[f"rest_{c}"] = round(float(bm[meta.super_class.values == c].mean()), 2)
    row["silent_frac"] = round(float((bm < 0.05).mean()), 3)
    sug = b.run(stim=SUGAR, t_run=T, n_trials=TRIALS, r_poi=100.0, seed=22001) / (T / 1000)
    loo = b.run(stim=LOOM, t_run=T, n_trials=TRIALS, r_poi=100.0, seed=22001) / (T / 1000)
    row["sugar_MN9_d"] = round(float(np.abs(d_of(sug, base)[MN9]).max()), 1)
    row["sugar_MN9_Hz"] = round(float(sug.mean(axis=1)[MN9].mean()), 1)
    row["loom_motor_d"] = round(float(np.abs(d_of(loo, base)[MOTOR]).max()), 1)
    row["loom_motor_n_d>2"] = int((np.abs(d_of(loo, base)[MOTOR]) > 2).sum())
    # steering: PFL3 left drives DNa02 right (contralateral), not left
    pf = b.run(stim=PFL3["left"], t_run=T, n_trials=TRIALS, r_poi=100.0, seed=22001) / (T / 1000)
    d = d_of(pf, base)
    row["PFL3L_to_DNa02_R_d"] = round(float(d[DNA02["right"]].mean()), 1)
    row["PFL3L_to_DNa02_L_d"] = round(float(d[DNA02["left"]].mean()), 1)
    # state axes: how many axes move under each stimulus, and the driven axis
    for name, r in (("sugar", sug), ("loom", loo)):
        ax_base = read_state(meta, bm); ax = read_state(meta, r.mean(axis=1))
        moved = [k for k in ax if k in ax_base and abs(ax[k] - ax_base[k]) > 1.0]
        row[f"{name}_axes_moved"] = ",".join(moved) if moved else "-"
    print(row, flush=True)
    return row


rows = [measure(Brain.from_meta(W, meta, Params(w_syn=0.20, sigma_v=2.0, r_spont=2.0)), "uniform 0.2")]
for w in (0.30, 0.40, 0.50):
    rows.append(measure(Brain.from_meta(W, meta, Params(w_syn=w, sigma_v=2.0, r_spont=2.0), area_scaled=True),
                        f"area {w:.1f}"))
df = pd.DataFrame(rows)
print("\n" + df.to_string(index=False))
df.to_csv(cache("area_default.csv"), index=False)
print("\ndone (area default)")
