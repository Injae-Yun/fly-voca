"""What is actually keeping the mushroom body on?

Three candidates are dead. Candidate 4 (a missed window) -- no window between
0.05 and 0.085. Candidate 3 (histamine) -- the optic lobe cools relatively as
w_syn rises, so vision is not the heat source. Candidate 2 (weak pairs) -- an
oddly clean refutation: raising the synapse-count threshold cuts the
KC-to-KC mesh from 68% to 24% of Kenyon output while KC activity only moves
71% to 63%. Quartering the mesh barely touches the firing, so the mesh is not
what sustains it.

That leaves the input side. Kenyon cells take 434k excitatory synapses against
39k inhibitory -- E/I of 11 -- almost all of it from the antennal lobe
projection neurons upstream. In the animal that drive meets the highest
spiking threshold in the brain, which is how ~5% of Kenyon cells respond to any
odour. Our LIF gives every neuron the same -45 mV.

So this tests a cell-type-specific threshold rather than a global one. Two
things have to hold for it to count as an explanation rather than a knob:
the sparseness has to appear at a threshold shift that is physiologically
ordinary (a few mV, not tens), and looming still has to reach the legs.
"""
import numpy as np
import pandas as pd

from flyvoca.banc_graph import build, select, leg_motor, spontaneous_groups
from bodysnatch.sim import Brain, Params

meta, W, _, st = build()
kc = meta[meta.cell_class.astype(str).str.contains("kenyon", case=False, na=False)]["idx"].values
lm = leg_motor(meta)["idx"].values
dn = meta[meta.super_class == "descending"]["idx"].values
groups = spontaneous_groups(meta)
sel = lambda **kw: select(meta, **kw)["idx"].values
LOOM = np.concatenate([sel(cell_type="LC4"), sel(cell_type="LPLC2")])
SUGAR = sel(prefix="LB3")
print(f"Kenyon {len(kc):,}  legMN {len(lm)}  DN {len(dn)}")

print("\n=== where does Kenyon excitation come from? ===")
pos = W.copy(); pos.data = np.maximum(pos.data, 0)
src = np.asarray(pos[:, kc].sum(axis=1)).ravel()
order = np.argsort(src)[::-1][:6]
tot = src.sum()
for i in order:
    r = meta.iloc[i]
    print(f"  {str(r.cell_type)[:20]:20} {str(r.super_class)[:24]:24} "
          f"{src[i]:>9,.0f}  ({src[i]/tot*100:.1f}%)")
by = pd.Series(src).groupby(meta.super_class.values).sum().sort_values(ascending=False)
print("  by super_class:", {k: int(v) for k, v in by.head(4).items()})

print("\n=== raise only the Kenyon cell threshold ===")
rows = []
for dv in (0.0, 1.0, 2.0, 3.0, 4.0, 6.0):
    off = np.zeros(len(meta), dtype=np.float32)
    off[kc] = -dv          # v_offset lowers resting potential = harder to fire
    for w in (0.16, 0.22):
        b = Brain(W, Params(w_syn=w, sigma_v=2.0, r_spont=0.0))
        b.spont_groups = groups
        base = b.run(t_run=600.0, n_trials=8, seed=9000, v_offset=off) / 0.6
        loom = b.run(stim=LOOM, t_run=600.0, n_trials=8, r_poi=100.0,
                     seed=9000, v_offset=off) / 0.6
        sug = b.run(stim=SUGAR, t_run=600.0, n_trials=8, r_poi=100.0,
                    seed=9000, v_offset=off) / 0.6
        bm = base.mean(axis=1)
        sdev = lambda x: np.sqrt((x.std(axis=1, ddof=1) ** 2
                                  + base.std(axis=1, ddof=1) ** 2) / 2)
        dl = (loom.mean(axis=1) - bm) / np.maximum(sdev(loom), 0.05)
        ds = (sug.mean(axis=1) - bm) / np.maximum(sdev(sug), 0.05)
        rows.append({
            "KC_shift_mV": dv, "w_syn": w,
            "KC>1Hz": round(float((bm[kc] > 1).mean()), 3),
            "KC_Hz": round(float(bm[kc].mean()), 2),
            "brain_Hz": round(float(bm[meta[meta.super_class == "central_brain_intrinsic"]["idx"].values].mean()), 2),
            "legMN_rest": round(float(bm[lm].mean()), 2),
            "loom_leg_d>2": int((np.abs(dl[lm]) > 2).sum()),
            "sugar_leg_d>2": int((np.abs(ds[lm]) > 2).sum()),
            "loom_DN_d>2": int((np.abs(dl[dn]) > 2).sum()),
        })
        print(rows[-1], flush=True)

df = pd.DataFrame(rows)
print("\n" + df.to_string(index=False))
ok = df[(df["KC>1Hz"] < 0.10) & (df["loom_leg_d>2"] > 5)]
print("\n=== sparse Kenyon cells AND looming reaching the legs ===")
print(ok.to_string(index=False) if len(ok) else "  none")
print("\ndone (kc threshold)")
