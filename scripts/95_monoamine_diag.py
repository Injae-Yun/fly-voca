"""Why is octopamine silent?

Looming reaches 1,312 of 2,316 octopaminergic neurons within two hops, yet
their firing does not move at all under stimulation (0.68 Hz, identical to
sham). Arousal in insects is carried by octopamine, so if anything were to
hold a state it should show here -- and it does not.

Three things could produce that, and they are distinguishable:

  they never fire     octopaminergic neurons might sit below threshold at
                      this operating point regardless of input. Check their
                      resting rate and input weight against the rest.
  input is balanced   two hops of reach says nothing about sign. If the
                      excitation arriving from looming is matched by
                      inhibition, the net is zero by construction.
  we deleted them     monoamine edges were routed to the slow graph and
                      removed from the fast one (doc 13). Octopaminergic
                      cells therefore have no outgoing fast edges -- but
                      their *inputs* should be unaffected. Verify that,
                      because if the build also stripped their inputs the
                      silence is a bug of mine, not biology.

The third is the one that would be my fault, so it is checked first.
"""
import numpy as np
import pandas as pd

from flyvoca.banc_graph import build, select, spontaneous_groups, FAST_SIGN, SLOW
from flyvoca.sim import Brain, Params

meta, W, Wslow, st = build()
nt = meta.neurotransmitter_predicted.astype(str)
ntv = meta.neurotransmitter_verified.astype(str)
t = meta.cell_type.astype(str)


def amine(name):
    return meta[(nt == name) | ntv.str.contains(name, na=False)]["idx"].values


OA = amine("octopamine")
DA = amine("dopamine")
ALL = meta["idx"].values
print(f"octopaminergic {len(OA):,}  dopaminergic {len(DA):,}  total {len(ALL):,}")

# --- check 3 first: did the build strip their inputs too? ---
Wa = abs(W)
in_w = np.asarray(Wa.sum(axis=0)).ravel()
out_w = np.asarray(Wa.sum(axis=1)).ravel()
print("\n=== do monoamine neurons still receive fast input? ===")
for name, idx in (("octopamine", OA), ("dopamine", DA), ("all neurons", ALL)):
    print(f"  {name:12} in-weight median {np.median(in_w[idx]):8.0f}  "
          f"out-weight median {np.median(out_w[idx]):8.0f}  "
          f"zero-in {int((in_w[idx] == 0).sum()):5}")

# --- check 2: is the input from looming signed against itself? ---
print("\n=== sign of the drive looming delivers to octopaminergic cells ===")
loom = np.concatenate([select(meta, cell_type="LC4")["idx"].values,
                       select(meta, cell_type="LPLC2")["idx"].values])
pos = W.copy(); pos.data = np.maximum(pos.data, 0)
neg = W.copy(); neg.data = np.minimum(neg.data, 0)
v = np.zeros(len(meta), dtype=np.float32); v[loom] = 1.0
for hop in (1, 2, 3):
    e = np.asarray(pos.T @ v).ravel()
    i = -np.asarray(neg.T @ v).ravel()
    tot = e + i
    reached = tot[OA] > 0
    print(f"  hop {hop}: {int(reached.sum()):5}/{len(OA)} reached   "
          f"exc {e[OA].sum():12,.0f}  inh {i[OA].sum():12,.0f}  "
          f"E/I {e[OA].sum()/max(i[OA].sum(),1):.3f}")
    v = np.asarray(pos.T @ v).ravel() - np.asarray(-(neg.T @ v)).ravel()
    v[loom] = 0.0
    v = np.maximum(v, 0)

# --- check 1: do they fire at all, at any drive? ---
print("\n=== do they fire under direct drive? ===")
groups = spontaneous_groups(meta)
b = Brain(W, Params(w_syn=0.16, sigma_v=2.0, r_spont=0.0))
b.spont_groups = groups
rows = []
for label, stim, rate in (("rest", (), 100.0),
                          ("looming 100Hz", loom, 100.0),
                          ("looming 400Hz", loom, 400.0),
                          ("OA driven directly", OA, 100.0)):
    c = b.run(stim=stim, t_run=600.0, n_trials=8, r_poi=rate, seed=16000) / 0.6
    m = c.mean(axis=1)
    rows.append({"cond": label,
                 "OA_Hz": round(float(m[OA].mean()), 3),
                 "OA_max": round(float(m[OA].max()), 1),
                 "OA>1Hz": int((m[OA] > 1).sum()),
                 "DA_Hz": round(float(m[DA].mean()), 3),
                 "all_Hz": round(float(m.mean()), 3)})
    print(rows[-1], flush=True)

print("\n=== how much drive would it take? threshold arithmetic ===")
print(f"  gap to threshold 7 mV at w_syn=0.16 -> "
      f"{7/0.16:.0f} coincident synapses needed")
sub = Wa[np.ix_(loom, OA)]
per = np.asarray(sub.sum(axis=0)).ravel()
print(f"  direct looming->OA synapses: total {sub.sum():,.0f} over "
      f"{int((per>0).sum())} cells, median per reached cell {np.median(per[per>0]):.0f}")
print("\ndone (monoamine)")
