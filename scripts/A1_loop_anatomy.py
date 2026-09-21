"""Where does the loop gain go? Isolate the loop from the brain and re-measure.

A0 showed the round-trip gain never reaches 1 at any uniform w_syn, and --
more telling -- that it *falls* as w_syn rises (0.045 at 0.4, 0.002 at 1.6).
A pair of excitatory synapses cannot lose gain when you make them stronger.
Something else in the network is being recruited, and the obvious candidate
is inhibition that aIPg (or pC1) drives onto the loop's other half.

Two measurements separate the loop's own strength from what the brain does
to it:

  embedded   the loop as it sits in the connectome (what A0 measured)
  isolated   every neuron outside pC1d/e + aIPg has its outgoing synapses
             muted, so the loop hears only itself

If the isolated gain reaches 1 the wiring can sustain and the network quenches
it; if it does not, the synapses themselves are too weak and no amount of
disinhibition helps. Either way the number says which.

Two corrections to A0/99 along the way. The paper's loop is pC1d/e, not the
pC1 family: averaging over pC1a-c, which receive no aIPg input, dilutes the
back arm. And aIPg *is* annotated -- the FlyWire consolidated types carry it
in `additional_type(s)` as "aIPg1, aIPgb/c" under primary type CB2131.
"""
import numpy as np
import pandas as pd
from scipy import sparse

from flyvoca.graph import build
from flyvoca.paths import cache
from bodysnatch.sim import Brain, Params

TRIALS, T = 8, 800.0
meta, W, _, _ = build()
t = meta["cell_type"].astype(str)
add = meta["additional_type(s)"].astype(str)
ex = lambda *n: meta[t.isin(n)]["idx"].values

sub = {k: ex(k) for k in ("pC1a", "pC1b", "pC1c", "pC1d", "pC1e")}
PC1DE = np.concatenate([sub["pC1d"], sub["pC1e"]])
PC1ALL = np.concatenate(list(sub.values()))
AIPG = ex("CB2131")
LOOP = np.unique(np.concatenate([PC1DE, AIPG]))

print("=== who is aIPg in v783? ===")
m = meta[add.str.contains("aIPg") | (t == "CB2131")]
print(m.groupby(["cell_type", "additional_type(s)"]).size().to_string())
print(f"\nloop: pC1d {len(sub['pC1d'])}  pC1e {len(sub['pC1e'])}  aIPg(CB2131) {len(AIPG)}")

# ------------------------------------------------------------- anatomy
Wc = W.tocsr()
def block(rows, cols):
    return np.asarray(Wc[np.ix_(rows, cols)].todense())

print("\n=== signed synapses, per cell pair ===")
lab = lambda i: f"{t.iloc[i]}[{meta.index[i] % 1000}]"
print("aIPg -> pC1 (rows aIPg, cols pC1a..e)")
df = pd.DataFrame(block(AIPG, PC1ALL), index=[lab(i) for i in AIPG],
                  columns=[lab(i) for i in PC1ALL]).astype(int)
print(df.to_string())
print("\npC1d/e -> aIPg (rows pC1d/e, cols aIPg)")
df = pd.DataFrame(block(PC1DE, AIPG), index=[lab(i) for i in PC1DE],
                  columns=[lab(i) for i in AIPG]).astype(int)
print(df.to_string())

print("\n=== disynaptic inhibition recruited by the loop ===")
n = W.shape[0]
for src, dst, name in ((AIPG, PC1DE, "aIPg -> X -> pC1d/e"),
                       (PC1DE, AIPG, "pC1d/e -> X -> aIPg")):
    out = np.asarray(Wc[src].sum(axis=0)).ravel()          # what src drives
    inn = np.asarray(Wc[:, dst].sum(axis=1)).ravel()       # what reaches dst
    mono = float(block(src, dst).sum())
    score = np.where(out > 0, out * inn, 0.0)              # excited X -> dst
    di_inh = float(score[score < 0].sum())
    di_exc = float(score[score > 0].sum())
    print(f"\n{name}")
    print(f"  monosynaptic signed total        {mono:+.0f}")
    print(f"  disynaptic via excited X: +{di_exc:.0f} excitatory, {di_inh:.0f} inhibitory (syn x syn)")
    top = np.argsort(score)[:12]
    rows = [{"X": t.iloc[i], "nt": meta["nt_type"].iloc[i], "from_src": int(out[i]),
             "to_dst": int(inn[i]), "score": int(score[i])} for i in top if score[i] < 0]
    print(pd.DataFrame(rows).to_string(index=False))

# ------------------------------------------------------------ dynamics
def gain(b, silence=(), seed=23000):
    """Slope of each arm at 50 -> 150 Hz drive, read on the paper's cells."""
    out = {}
    for src, dst, arm in ((PC1DE, AIPG, "fwd"), (AIPG, PC1DE, "back")):
        pts = []
        for rate in (50.0, 150.0):
            c = b.run(stim=src, t_run=T, n_trials=TRIALS, r_poi=rate,
                      seed=seed, silence=silence)
            r = c.mean(axis=1) / (T / 1000)
            pts.append((float(r[src].mean()), float(r[dst].mean()), r))
        (x1, y1, r1), (x2, y2, r2) = pts
        out[arm] = (y2 - y1) / max(x2 - x1, 1e-9)
        out[arm + "_cells@150"] = [round(float(r2[i]), 1) for i in dst]
    out["gain"] = out["fwd"] * out["back"]
    return out

def persist(b, silence=(), seed=24000, on=4, off=8, block=250.0):
    state, tr = None, []
    for k in range(on + off):
        c, state = b.run(stim=LOOP if k < on else (), t_run=block, n_trials=TRIALS,
                         r_poi=100.0, seed=seed + k, state=state,
                         return_state=True, silence=silence)
        r = c.mean(axis=1) / (block / 1000)
        tr.append((float(r[PC1DE].mean()), float(r[AIPG].mean())))
    on_v = np.mean([x[0] for x in tr[:on]])
    return {"on": round(on_v, 1), "off1": round(tr[on][0], 2),
            "off_late": round(float(np.mean([x[0] for x in tr[-4:]])), 2),
            "aIPg_off_late": round(float(np.mean([x[1] for x in tr[-4:]])), 2)}

others = np.setdiff1d(np.arange(n), LOOP)
rows = []
for w in (0.20, 0.40, 0.80):
    b = Brain.from_meta(W, meta, Params(w_syn=w, sigma_v=2.0, r_spont=2.0))
    for cond, sil in (("embedded", ()), ("isolated", others)):
        g = gain(b, sil)
        p = persist(b, sil)
        row = {"w_syn": w, "cond": cond, "fwd": round(g["fwd"], 3),
               "back": round(g["back"], 3), "gain": round(g["gain"], 4), **p}
        rows.append(row)
        print(row, flush=True)
        print(f"    pC1d/e @ aIPg150: {g['back_cells@150']}   "
              f"aIPg @ pC1de150: {g['fwd_cells@150']}", flush=True)

df = pd.DataFrame(rows)
print("\n" + df.to_string(index=False))
df.to_csv(cache("loop_isolated.csv"), index=False)
print("\ndone (loop anatomy)")
