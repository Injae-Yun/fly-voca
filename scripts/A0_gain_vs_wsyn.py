"""Is there any w_syn where the loop sustains AND the benchmarks survive?

The 35x figure was arithmetic, not measurement. This measures it: sweep
w_syn, compute the round-trip loop gain at each value, and at the same time
run the two things that already work -- sugar reaching the proboscis pool and
looming reaching the legs -- so the cost of any gain increase is visible in
the same table.

Physiology narrows what is allowed. Drosophila nAChR mEPSCs decay with
tau = 1.4 +- 0.1 ms (rise 0.43 ms, 22.1 pA), with no slow cholinergic
component reported, so the model's 5 ms is already *generous* relative to the
real fast synapse. Raising tau to buy gain therefore moves away from the
measurement, not toward it -- which is worth knowing before spending the
knob. mAChR-A is inhibitory in Kenyon cells, so it is not the slow excitatory
component either.

Three outcomes and what each means:

  gain reaches 1 with benchmarks intact   the weights were simply too low and
                                          this is fixable
  gain reaches 1 but benchmarks break     one global number cannot serve both
                                          circuits; per-synapse weights are
                                          required, and that is a data problem
  gain never reaches 1                    the loop cannot sustain at any
                                          uniform weight, and the missing
                                          ingredient is not synaptic strength
"""
import numpy as np
import pandas as pd

from flyvoca.graph import build
from flyvoca.paths import cache
from bodysnatch.sim import Brain, Params

TRIALS = 8
meta, W, _, st = build()
t = meta["cell_type"].astype(str)
ex = lambda *n: meta[t.isin(n)]["idx"].values

PC1 = ex("pC1a", "pC1b", "pC1c", "pC1d", "pC1e")
AIPG = ex("CB2131")
SUGAR = meta[t.str.startswith("LB3", na=False)]["idx"].values
MN9 = meta[meta.index == 720575940660219265]["idx"].values
LOOM = ex("LC4", "LPLC2")
MOTOR = meta[meta.super_class == "motor"]["idx"].values
print(f"pC1 {len(PC1)}  aIPg {len(AIPG)}  sugar {len(SUGAR)}  "
      f"MN9 {len(MN9)}  looming {len(LOOM)}  motor {len(MOTOR)}")

rows = []
for w in (0.20, 0.40, 0.80, 1.60, 3.20, 6.40):
    b = Brain.from_meta(W, meta, Params(w_syn=w, sigma_v=2.0, r_spont=2.0))

    # --- loop gain: drive one arm at two rates, take the slope ---
    slopes = {}
    for src, dst, arm in ((PC1, AIPG, "fwd"), (AIPG, PC1, "back")):
        pts = []
        for rate in (50.0, 150.0):
            c = b.run(stim=src, t_run=800.0, n_trials=TRIALS, r_poi=rate,
                      seed=22000)
            m = c.mean(axis=1) / 0.8
            pts.append((float(m[src].mean()), float(m[dst].mean())))
        (x1, y1), (x2, y2) = pts
        slopes[arm] = (y2 - y1) / max(x2 - x1, 1e-9)
    gain = slopes["fwd"] * slopes["back"]

    # --- benchmarks that already work ---
    base = b.run(t_run=800.0, n_trials=TRIALS, seed=22001) / 0.8
    bm, bs = base.mean(axis=1), base.std(axis=1, ddof=1)

    def effect(stim, target):
        r = b.run(stim=stim, t_run=800.0, n_trials=TRIALS, r_poi=100.0,
                  seed=22001) / 0.8
        d = (r.mean(axis=1) - bm) / np.maximum(
            np.sqrt((r.std(axis=1, ddof=1) ** 2 + bs ** 2) / 2), 0.05)
        return float(np.abs(d[target]).max()), float(r.mean(axis=1)[target].mean())

    mn9_d, mn9_hz = effect(SUGAR, MN9) if len(MN9) else (np.nan, np.nan)
    loom_d, _ = effect(LOOM, MOTOR)

    rows.append({"w_syn": w,
                 "fwd": round(slopes["fwd"], 3),
                 "back": round(slopes["back"], 3),
                 "loop_gain": round(gain, 4),
                 "net_Hz": round(float(bm.mean()), 2),
                 "sugar_MN9_d": round(mn9_d, 1),
                 "sugar_MN9_Hz": round(mn9_hz, 1),
                 "loom_motor_d": round(loom_d, 1)})
    print(rows[-1], flush=True)

df = pd.DataFrame(rows)
print("\n" + df.to_string(index=False))
df.to_csv(cache("gain_vs_wsyn.csv"), index=False)

ok = df[df.loop_gain >= 1.0]
print("\n=== does gain ever reach 1? ===")
if len(ok):
    print(ok.to_string(index=False))
    print("\n  and at that w_syn, the resting network sits at "
          f"{ok.iloc[0].net_Hz} Hz (a resting fly is ~1-2 Hz)")
else:
    g = df.loop_gain.max()
    print(f"  no. best is {g:.4f} at w_syn={df.loc[df.loop_gain.idxmax(),'w_syn']}")
    print(f"  extrapolating, gain scales ~w^2, so 1.0 needs about "
          f"{df.iloc[0].w_syn * (1.0/df.iloc[0].loop_gain)**0.5:.1f} mV per synapse")
print("\ndone (gain vs wsyn)")
