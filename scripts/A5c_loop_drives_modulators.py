"""Does the loop switch on its own modulators?

A5b found 65 monoaminergic cells synapsing onto pC1d/e or aIPg, with real
counts -- OA-VUMa8 alone puts 177 synapses on pC1d/e. Deutsch et al. drove
pC1d/e optogenetically and saw the state persist, so any modulatory source
of that persistence has to be *downstream* of pC1d/e. The wiring says some
are (NPFL1-I gets 88 excitatory synapses back from the loop, CL344 36-40,
SMP143 21-23); whether that is enough to fire them under the LIF arithmetic
(peak PSP = 0.157 x g; ~225 coincident synapses per spike from rest) is a
simulation question, and a cheap one: drive the loop at 100 Hz and read them.

A cell that lights up here is a data-grounded candidate for the slow
excitability shift the model lacks. One that stays silent is not, however
many synapses it sends.
"""
import numpy as np
import pandas as pd

from flyvoca.graph import build
from flyvoca.paths import cache
from bodysnatch.sim import Brain, Params

TRIALS, T = 8, 800.0
meta, W, Ws, _ = build()
t = meta["cell_type"].astype(str)
add = meta["additional_type(s)"].astype(str)
ex = lambda *n: meta[t.isin(n)]["idx"].values
PC1DE = np.concatenate([ex("pC1d"), ex("pC1e")])
AIPG = meta[add.str.contains("aIPg")]["idx"].values
LOOP = np.unique(np.concatenate([PC1DE, AIPG]))

mod = pd.read_csv(cache("modulation_anatomy.csv"))
mod = mod[mod["class"] == "monoamine"] if "class" in mod else mod
cells = mod.sort_values("syn_total", ascending=False)
idx = meta.loc[cells.root_id.values, "idx"].values
print(f"reading {len(idx)} monoaminergic cells that contact the loop")

b = Brain.from_meta(W, meta, Params(w_syn=0.20, sigma_v=2.0, r_spont=2.0))
base = b.run(t_run=T, n_trials=TRIALS, seed=25000) / (T / 1000)
bm, bs = base.mean(axis=1), base.std(axis=1, ddof=1)
rows = []
for name, stim in (("pC1d/e", PC1DE), ("loop", LOOP)):
    r = b.run(stim=stim, t_run=T, n_trials=TRIALS, r_poi=100.0, seed=25000) / (T / 1000)
    rm, rs = r.mean(axis=1), r.std(axis=1, ddof=1)
    d = (rm - bm) / np.maximum(np.sqrt((rs ** 2 + bs ** 2) / 2), 0.05)
    for k, i in enumerate(idx):
        rows.append({"drive": name, "cell": cells.cell_type.iloc[k], "nt": cells.nt_type.iloc[k],
                     "syn_pC1de": int(cells.syn_pc1de.iloc[k]), "syn_aIPg": int(cells.syn_aipg.iloc[k]),
                     "exc_back_hop1": int(cells.hop1_exc_from_loop.iloc[k]),
                     "base_Hz": round(float(bm[i]), 2), "drive_Hz": round(float(rm[i]), 2),
                     "d": round(float(d[i]), 1)})
df = pd.DataFrame(rows)
for name, g in df.groupby("drive", sort=False):
    print(f"\n=== driving {name} at 100 Hz: modulator cells, sorted by response ===")
    print(g.sort_values("d", ascending=False).head(20).drop(columns="drive").to_string(index=False))
    hit = g[(g.d > 2) & (g.drive_Hz > 5)]
    print(f"  recruited (d>2 and >5 Hz): {len(hit)} of {len(g)}  "
          f"-> {sorted(set(hit.cell))}")
df.to_csv(cache("loop_drives_modulators.csv"), index=False)
print("\ndone (loop drives modulators)")
