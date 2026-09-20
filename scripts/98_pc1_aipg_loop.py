"""The persistence test, run on the loop the literature actually describes.

Every persistence attempt so far drove pC1 alone and watched pC1 alone. The
female-persistence paper (eLife 59502) says the state is held by a *recurrent*
circuit: pC1d/e reciprocally wired with aIPg, cholinergic, sustaining activity
for at least ten minutes after the stimulus ends. A loop needs both halves,
and only one was ever in the experiment.

aIPg is absent from every annotation we hold -- not in FAFB v783 cell types,
not in BANC's, not in BANC's fafb_cell_type column. It was found by its
connectivity signature instead: the paper reports pC1d <-> aIPg-b at 38:39
synapses, and CB2131 in v783 sits at 38:38, 46:39, 41:39 across its members.
The provisional CB#### name is why the search by name failed.

So this drives the loop and reads the loop, on FAFB where the paper's numbers
were measured. Controls, since the last persistence result turned out to be a
raised baseline rather than a held state:

  sham at every condition   persistence is scored against no-stimulus runs
                            under identical settings
  pC1-only drive            the original experiment, for comparison
  loop drive                pC1 + CB2131 together
  CB2131-only               does the partner alone hold anything
"""
import numpy as np
import pandas as pd

from flyvoca.graph import build, load_meta
from flyvoca.paths import cache
from bodysnatch.sim import Brain, Params

W_SYN, TRIALS, BLOCK = 0.20, 12, 250.0
ON, OFF = 4, 12

meta, W, _, st = build()
print("graph:", st)
t = meta["cell_type"].astype(str)
ex = lambda *n: meta[t.isin(n)]["idx"].values

PC1D = ex("pC1d"); PC1E = ex("pC1e")
PC1 = ex("pC1a", "pC1b", "pC1c", "pC1d", "pC1e")
AIPG = ex("CB2131")
LOOP = np.unique(np.concatenate([PC1, AIPG]))
print(f"pC1 {len(PC1)}  CB2131(aIPg) {len(AIPG)}  loop {len(LOOP)}")

Wa = abs(W)
print("\n=== the loop, in signed synapses ===")
w = lambda a, b: float(W[np.ix_(a, b)].sum())
print(f"  pC1d -> aIPg {w(PC1D, AIPG):7.0f}   aIPg -> pC1d {w(AIPG, PC1D):7.0f}")
print(f"  pC1  -> aIPg {w(PC1, AIPG):7.0f}   aIPg -> pC1  {w(AIPG, PC1):7.0f}")
print(f"  aIPg -> aIPg {w(AIPG, AIPG):7.0f}   pC1 -> pC1   {w(PC1, PC1):7.0f}")
per = np.asarray(Wa[np.ix_(AIPG, PC1)].sum(axis=0)).ravel()
print(f"  aIPg->pC1 per target cell: {[int(x) for x in per]}  "
      f"(threshold needs {7/W_SYN:.0f} coincident)")

READ = {"pC1": PC1, "aIPg": AIPG, "loop": LOOP,
        "DN": meta[meta.super_class == "descending"]["idx"].values}

CASES = {
    "pC1 only": PC1,
    "aIPg only": AIPG,
    "loop (pC1+aIPg)": LOOP,
    "sham": np.array([], dtype=np.int64),
}

print(f"\n{ON} on / {OFF} off blocks of {BLOCK:.0f} ms, w_syn={W_SYN}\n")
traces = {}
for label, stim in CASES.items():
    b = Brain.from_meta(W, meta, Params(w_syn=W_SYN, sigma_v=2.0, r_spont=2.0))
    state, rows = None, []
    for blk in range(ON + OFF):
        on = blk < ON
        c, state = b.run(stim=stim if on else (), t_run=BLOCK, n_trials=TRIALS,
                         r_poi=100.0, seed=19000 + blk, state=state,
                         return_state=True)
        m = c.mean(axis=1) / (BLOCK / 1000)
        rows.append({"blk": blk, "phase": "on" if on else "off",
                     **{k: round(float(m[v].mean()), 3) for k, v in READ.items()}})
    traces[label] = pd.DataFrame(rows)
    d = traces[label]
    print(f"--- {label} ---")
    print(d.to_string(index=False), flush=True)
    print()

sham = traces["sham"]
print("=== retention against sham ===")
rows = []
for label, d in traces.items():
    if label == "sham":
        continue
    r = {"case": label}
    for k in READ:
        on = d[d.phase == "on"][k].mean()
        late = d[d.phase == "off"][k].tail(4).mean()
        base = sham[sham.phase == "off"][k].tail(4).mean()
        denom = on - base
        r[k] = round(float((late - base) / denom), 3) if abs(denom) > 0.05 else None
        r[k + "_hz"] = f"{on:.1f}->{late:.2f} (sham {base:.2f})"
    rows.append(r)
R = pd.DataFrame(rows)
print(R[["case"] + list(READ)].to_string(index=False))
print()
print(R[["case"] + [k + "_hz" for k in READ]].to_string(index=False))
for label, d in traces.items():
    d.to_csv(cache(f"loop_{label.replace(' ', '_').replace('(', '').replace(')', '').replace('+', '_')}.csv"),
             index=False)
print("\ndone (loop)")
