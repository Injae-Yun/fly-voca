"""Does the full aIPg population, not just CB2131, change the loop's gain?

A1 treated aIPg as the 7 CB2131 cells, because that primary type is the only
one FlyWire's `additional_type(s)` field tags with "aIPg" *and* `cell_type ==
"CB2131"` in the same breath. But the v783 consolidated types annotate aIPg
more broadly: 37 cells across 9 primary types carry "aIPg" somewhere in
`additional_type(s)`, CB2131 being only the largest of the nine.

H1: the paper's loop is pC1d/e <-> *all* aIPg, and population size alone --
five times as many cells converging on the same pC1d/e dendrites -- might be
what raises the gain past 1, independent of any single-synapse weight change.

Prediction, falsifiable: if H1 is right, total signed aIPg->pC1d/e input
should scale roughly with cell count, and the round-trip gain measured on
AIPG37 should sit visibly closer to 1 than AIPG7's. If aIPg1/CB2131 already
carries most of the anatomical weight and the other eight types are minor
contributors, input rises only ~1-3x and gain stays below 1 regardless of
which population is driven -- population size was never the missing factor.

Two loop populations (AIPG7, AIPG37) x two w_syn x two conditions (embedded,
isolated -- everything outside pC1d/e + aIPg muted, as in A1) are swept
together so the anatomy and the dynamics answer the same question in one
table.
"""
import numpy as np
import pandas as pd
from scipy import sparse

from flyvoca.graph import build
from flyvoca.paths import cache
from flyvoca.sim import Brain, Params

TRIALS, T = 8, 800.0
meta, W, _, _ = build()
t = meta["cell_type"].astype(str)
add = meta["additional_type(s)"].astype(str)
ex = lambda *n: meta[t.isin(n)]["idx"].values

PC1DE = np.concatenate([ex("pC1d"), ex("pC1e")])
AIPG7 = ex("CB2131")
AIPG37 = meta[add.str.contains("aIPg")]["idx"].values
n = W.shape[0]
Wc = W.tocsr()
SPONT = meta[meta["super_class"].isin(["sensory", "sensory_ascending"])]["idx"].values

SUGAR = meta[t.str.startswith("LB3", na=False)]["idx"].values
MN9 = meta[meta.index == 720575940660219265]["idx"].values
LOOM = ex("LC4", "LPLC2")
MOTOR = meta[meta.super_class == "motor"]["idx"].values

print("=== who is aIPg in v783 (additional_type(s) contains 'aIPg')? ===")
m37 = meta[add.str.contains("aIPg")]
print(m37.groupby(["cell_type", "nt_type"]).size().to_string())
print(f"\nAIPG7 (CB2131 only): {len(AIPG7)}   AIPG37 (all 9 types): {len(AIPG37)}")

print("\n=== signed input onto each pC1d/e cell ===")
def block(rows, cols):
    return np.asarray(Wc[np.ix_(rows, cols)].todense())
lab = lambda i: f"{t.iloc[i]}[{meta.index[i] % 1000}]"
inp = pd.DataFrame({"AIPG7_input": block(AIPG7, PC1DE).sum(axis=0).astype(int),
                    "AIPG37_input": block(AIPG37, PC1DE).sum(axis=0).astype(int)},
                   index=[lab(i) for i in PC1DE])
print(inp.to_string())
print(f"\ntotal: AIPG7 {inp.AIPG7_input.sum():+.0f}  AIPG37 {inp.AIPG37_input.sum():+.0f}  "
      f"ratio {inp.AIPG37_input.sum() / max(inp.AIPG7_input.sum(), 1):.2f}x")

# ------------------------------------------------------------ dynamics
def slope_matrix(b, AIPG, seed, silence=()):
    """Drive PC1DE and AIPG each at 25/100 Hz; off-diagonal slopes to the
    other population. Diagonal (a population's slope onto itself) is not a
    round-trip gain term and is left at 0, as instructed."""
    pops = {"PC1DE": PC1DE, "AIPG": AIPG}
    M = {y: {x: 0.0 for x in pops} for y in pops}
    for xname, xidx in pops.items():
        r_lo, r_hi = (b.run(stim=xidx, t_run=T, n_trials=TRIALS, r_poi=rate,
                            seed=seed, silence=silence).mean(axis=1) / (T / 1000)
                      for rate in (25.0, 100.0))
        x_lo, x_hi = float(r_lo[xidx].mean()), float(r_hi[xidx].mean())
        for yname, yidx in pops.items():
            if yname == xname:
                continue
            y_lo, y_hi = float(r_lo[yidx].mean()), float(r_hi[yidx].mean())
            M[yname][xname] = (y_hi - y_lo) / max(x_hi - x_lo, 1e-9)
    return M

def persist(b, LOOP, AIPG, silence=(), seed=24000, on=4, off=8, blk=250.0):
    state, tr = None, []
    for k in range(on + off):
        c, state = b.run(stim=LOOP if k < on else (), t_run=blk, n_trials=TRIALS,
                         r_poi=100.0, seed=seed + k, state=state,
                         return_state=True, silence=silence)
        r = c.mean(axis=1) / (blk / 1000)
        tr.append((float(r[PC1DE].mean()), float(r[AIPG].mean())))
    on_v = [np.mean([x[i] for x in tr[:on]]) for i in (0, 1)]
    late = [np.mean([x[i] for x in tr[-4:]]) for i in (0, 1)]
    return {"PC1DE_on": round(on_v[0], 1), "PC1DE_off1": round(tr[on][0], 2),
            "PC1DE_off_late": round(float(late[0]), 2), "AIPG_on": round(on_v[1], 1),
            "AIPG_off1": round(tr[on][1], 2), "AIPG_off_late": round(float(late[1]), 2)}

def bench(b, seed=22001):
    """Same computation as A0: 100 Hz drive, 800 ms, paired seed, ddof=1, floor 0.05."""
    base = b.run(t_run=T, n_trials=TRIALS, seed=seed) / (T / 1000)
    bm, bs = base.mean(axis=1), base.std(axis=1, ddof=1)
    def effect(stim, target):
        r = b.run(stim=stim, t_run=T, n_trials=TRIALS, r_poi=100.0, seed=seed) / (T / 1000)
        d = (r.mean(axis=1) - bm) / np.maximum(
            np.sqrt((r.std(axis=1, ddof=1) ** 2 + bs ** 2) / 2), 0.05)
        return float(np.abs(d[target]).max()), float(r.mean(axis=1)[target].mean())
    mn9_d, mn9_hz = effect(SUGAR, MN9) if len(MN9) else (np.nan, np.nan)
    loom_d, _ = effect(LOOM, MOTOR)
    return {"sugar_MN9_d": round(mn9_d, 1), "sugar_MN9_Hz": round(mn9_hz, 1),
            "loom_motor_d": round(loom_d, 1)}

rows = []
for pop_name, AIPG in (("AIPG7", AIPG7), ("AIPG37", AIPG37)):
    LOOP = np.unique(np.concatenate([PC1DE, AIPG]))
    keep = np.zeros(n, dtype=bool)
    keep[LOOP] = True
    for w in (0.20, 0.40):
        params = Params(w_syn=w, sigma_v=2.0, r_spont=2.0)
        b_emb = Brain.from_meta(W, meta, params)

        # Build the isolated (muted) matrix once per (population, w_syn),
        # instead of paying Brain._muted's rebuild on every run() call.
        W_iso = Wc.copy()
        row_of_nnz = np.repeat(np.arange(n), np.diff(W_iso.indptr))
        W_iso.data[~keep[row_of_nnz]] = 0.0   # zero rows of non-loop presynaptic neurons
        W_iso.eliminate_zeros()
        nz_rows = np.where(np.diff(W_iso.indptr) > 0)[0]
        assert np.all(keep[nz_rows]), "muted matrix leaked synapses outside the loop"
        b_iso = Brain(W_iso, params, spont=SPONT)

        for cond, b in (("embedded", b_emb), ("isolated", b_iso)):
            M = slope_matrix(b, AIPG, seed=23000)
            fwd, back = M["AIPG"]["PC1DE"], M["PC1DE"]["AIPG"]
            gain = fwd * back
            row = {"aIPg_pop": pop_name, "n_aIPg": len(AIPG), "w_syn": w, "cond": cond,
                   "fwd": round(fwd, 3), "back": round(back, 3), "gain": round(gain, 4),
                   "spectral_radius": round(float(np.sqrt(abs(gain))), 4),
                   **persist(b, LOOP, AIPG)}
            row.update(bench(b) if cond == "embedded" else
                       {"sugar_MN9_d": np.nan, "sugar_MN9_Hz": np.nan, "loom_motor_d": np.nan})
            rows.append(row)
            print(row, flush=True)

df = pd.DataFrame(rows)
print("\n" + df.to_string(index=False))
df.to_csv(cache("loop_full_aipg.csv"), index=False)
print("\ndone (full aIPg)")
