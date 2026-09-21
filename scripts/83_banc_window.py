"""Candidate 4 first: is there an operating window between 0.05 and 0.08?

The coarse grid left a gap. Where Kenyon cells are sparse (w_syn 0.05, 2.5%
above 1 Hz) the leg motor neurons peak at 0.8 Hz and the network is dead;
one step up at 0.08 the mushroom body is already 59% on. If a window exists
it is in between, and that is the cheapest of the four candidates to test.

Two diagnostics ride along so that a null result points somewhere rather than
just failing:

  where does the heat come from   per-super_class firing at each w_syn. If
      the optic lobe runs away while the rest is quiet, candidate 3
      (histamine sign, 7411 neurons, mostly photoreceptors) is implicated.
  how unbalanced is the drive     per-neuron excitatory minus inhibitory input
      weight. Under Dale's principle a neuron's sign paints all its edges, so
      if BANC's inhibitory cells are systematically lower-degree than FAFB's
      the balance tilts -- candidate 1.

Kenyon cells stay the unfitted referee: the mushroom body codes sparsely, so
a healthy operating point keeps them near-silent whatever else is true.
"""
import numpy as np
import pandas as pd

from flyvoca.banc_graph import build, select, leg_motor, spontaneous_groups
from flyvoca.paths import cache
from flyvoca.sim import Brain, Params

meta, W, _, st = build()
print("graph:", st)
groups = spontaneous_groups(meta)

G = {
    "optic": meta[meta.super_class == "optic_lobe_intrinsic"]["idx"].values,
    "brain": meta[meta.super_class == "central_brain_intrinsic"]["idx"].values,
    "VNC": meta[meta.super_class == "ventral_nerve_cord_intrinsic"]["idx"].values,
    "DN": meta[meta.super_class == "descending"]["idx"].values,
    "AN": meta[meta.super_class == "ascending"]["idx"].values,
}
kc = meta[meta.cell_class.astype(str).str.contains("kenyon", case=False, na=False)]["idx"].values
lm = leg_motor(meta)["idx"].values
print({k: len(v) for k, v in G.items()}, "| Kenyon", len(kc), "| legMN", len(lm))

# ---- diagnostic: E/I balance per neuron, and who is inhibitory ----
pos = W.copy(); pos.data = np.maximum(pos.data, 0)
neg = W.copy(); neg.data = np.minimum(neg.data, 0)
exc_in = np.asarray(pos.sum(axis=0)).ravel()
inh_in = -np.asarray(neg.sum(axis=0)).ravel()
print("\n=== who provides inhibition, by super_class ===")
sign = meta["nt_sign"].fillna(1.0).values
out_w = np.asarray(abs(W).sum(axis=1)).ravel()
rows = []
for name, idx in list(G.items()) + [("Kenyon", kc)]:
    s = sign[idx]
    rows.append({"group": name, "n": len(idx),
                 "frac_inhib": round(float((s < 0).mean()), 3),
                 "out_weight_exc": int(out_w[idx][s > 0].sum()),
                 "out_weight_inh": int(out_w[idx][s < 0].sum())})
print(pd.DataFrame(rows).to_string(index=False))
print(f"\nwhole graph: {float((sign<0).mean()):.3f} of neurons inhibitory, "
      f"E/I weight {exc_in.sum()/inh_in.sum():.3f}")

# ---- the sweep ----
print("\n=== fine sweep 0.050 - 0.085 ===")
rows = []
for w in (0.050, 0.056, 0.062, 0.068, 0.074, 0.080, 0.085):
    b = Brain(W, Params(w_syn=w, sigma_v=2.0, r_spont=0.0))
    b.spont_groups = groups
    c = b.run(t_run=600.0, n_trials=8, seed=7000)
    m = c.mean(axis=1) / 0.6
    r = {"w_syn": w,
         "KC>1Hz": round(float((m[kc] > 1).mean()), 4),
         "legMN_Hz": round(float(m[lm].mean()), 2),
         "legMN_max": round(float(m[lm].max()), 1),
         "net_kHz": round(float(m.sum()) / 1000, 1)}
    for name, idx in G.items():
        r[name] = round(float(m[idx].mean()), 2)
    rows.append(r)
    print(r, flush=True)

df = pd.DataFrame(rows)
print("\n" + df.to_string(index=False))

ok = df[(df["KC>1Hz"] < 0.10) & (df["legMN_max"] > 5.0)]
print("\n=== windows where Kenyon cells stay sparse AND legs can fire ===")
print(ok.to_string(index=False) if len(ok) else "  none in this range")

print("\n=== where does the activity sit? (ratio to central brain) ===")
for _, r in df.iterrows():
    base = max(r["brain"], 1e-9)
    print(f"  w={r['w_syn']:.3f}  optic/brain {r['optic']/base:5.2f}  "
          f"VNC/brain {r['VNC']/base:5.2f}  DN/brain {r['DN']/base:5.2f}")
np.savez(cache("banc_window.npz"), **{c: df[c].values for c in df.columns})
print("\ndone (window)")
