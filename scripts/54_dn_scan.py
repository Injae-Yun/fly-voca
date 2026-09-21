"""Which descending channel carries motivation? Scan all 1305 and find out.

Three experiments -- vinegar, hunger-to-steering, hunger-to-compass -- each
looked at turn direction and each found nothing. Rather than guess the fourth
place to look, this measures every descending neuron under every condition and
lets the data name the channel.

Guards against fooling ourselves, since a 1305-way scan will always produce a
biggest number:

  paired trials   identical seeds across conditions, so a neuron's difference
                  is measured against its own baseline run, not across noise
  effect size     Cohen's d from per-trial variance, not raw Hz -- a 0.3 Hz
                  change on a silent neuron is not a discovery
  null condition  "sham" repeats baseline with a different seed; whatever it
                  turns up is the false-positive floor everything else must beat
"""
import numpy as np
import pandas as pd

from flyvoca.graph import build
from flyvoca.paths import cache
from flyvoca.sim import Brain, Params
from voca.peptide import Field, Expression
from voca.expression import fill, releasers

TRIALS = 24
T_RUN = 800.0

meta, W, _, _ = build()
sel = lambda ct, side=None: meta[(meta.cell_type == ct) &
                                 ((meta.side == side) if side else True)]["idx"].values
brain = Brain.from_meta(W, meta, Params(sigma_v=2.0, r_spont=2.0))

# --- build the hungry offset once ---
peps = sorted(set(releasers().peptide) | {"DILP", "DH44", "DH31", "CRZ", "DMS",
                                          "Hugin", "CAPA", "ITP"})
field = Field(meta, peptides=peps, tau=20.0, gain=3.0)
expr = Expression(len(meta), tuple(peps))
fill(expr, meta)
dh44 = sel("m_NSC_DH44")
state = None
for e in range(10):
    c, state = brain.run(stim=dh44, t_run=500.0, n_trials=6, r_poi=50.0,
                         seed=1400 + e, state=state,
                         v_offset=field.v_offset(expr), return_state=True)
    m, _ = Brain.rate(c, 500.0)
    field.step(m, dt=0.5)
off_hungry = field.v_offset(expr).astype(np.float32)
print(f"hungry offset built on {int((np.abs(off_hungry) > 0.01).sum())} neurons")

SEED = 2000
CONDS = {
    "sham":      dict(stim=(), off=None, seed=SEED + 1),   # null: baseline, other seed
    "hungry":    dict(stim=(), off=off_hungry, seed=SEED),
    "vinegar":   dict(stim=sel("ORN_DM1"), off=None, seed=SEED),
    "co2":       dict(stim=sel("ORN_V"), off=None, seed=SEED),
    "sugar":     dict(stim=sel("LB3"), off=None, seed=SEED),
    "looming":   dict(stim=np.concatenate([sel("LC4"), sel("LPLC2")]), off=None, seed=SEED),
    "hungry_vinegar": dict(stim=sel("ORN_DM1"), off=off_hungry, seed=SEED),
}

dn = meta[meta.super_class == "descending"]
dn_idx = dn["idx"].values
dn_lab = dn["label"].values
dn_side = dn["side"].values
print(f"scanning {len(dn_idx)} descending neurons x {len(CONDS)} conditions")

base = brain.run(stim=(), t_run=T_RUN, n_trials=TRIALS, r_poi=100.0, seed=SEED)
base_dn = base[dn_idx] / (T_RUN / 1000.0)          # (n_dn, trials)

res = {}
for name, cfg in CONDS.items():
    c = brain.run(stim=cfg["stim"], t_run=T_RUN, n_trials=TRIALS, r_poi=100.0,
                  seed=cfg["seed"], v_offset=cfg["off"])
    res[name] = c[dn_idx] / (T_RUN / 1000.0)
    print(f"  {name}: DN mean {res[name].mean():.2f} Hz", flush=True)


def cohen(a, b):
    """Effect size of a vs b, per neuron."""
    ma, mb = a.mean(axis=1), b.mean(axis=1)
    sa, sb = a.std(axis=1, ddof=1), b.std(axis=1, ddof=1)
    pooled = np.sqrt((sa ** 2 + sb ** 2) / 2)
    return (ma - mb) / np.maximum(pooled, 0.05), ma - mb


rows = []
for name, r in res.items():
    d, diff = cohen(r, base_dn)
    for k in range(len(dn_idx)):
        rows.append({"cond": name, "label": dn_lab[k], "side": dn_side[k],
                     "base_Hz": base_dn[k].mean(), "cond_Hz": r[k].mean(),
                     "delta": diff[k], "d": d[k]})
df = pd.DataFrame(rows)

sham = df[df.cond == "sham"]
floor = float(np.abs(sham.d).max())
n_sham = int((np.abs(sham.d) > 2).sum())
print(f"\nnull floor: sham max |d| = {floor:.2f}, "
      f"{n_sham} of {len(dn_idx)} neurons exceed |d| > 2 by chance")
THR = max(2.0, floor)
print(f"using |d| > {THR:.2f} as the bar\n")

for name in CONDS:
    if name == "sham":
        continue
    s = df[df.cond == name].copy()
    hits = s[np.abs(s.d) > THR].sort_values("d", key=np.abs, ascending=False)
    up = int((hits.d > 0).sum()); dn_ = int((hits.d < 0).sum())
    print(f"=== {name}: {len(hits)} DNs past the bar ({up} up, {dn_} down) ===")
    for _, r in hits.head(8).iterrows():
        print(f"    {r.label:<12} {r.side:<5} {r.base_Hz:6.2f} -> {r.cond_Hz:6.2f} Hz"
              f"   d = {r.d:+7.2f}")
    print()

df.to_csv(cache("dn_scan.csv"), index=False)
print(f"saved -> {cache('dn_scan.csv')}")

# does hunger change what vinegar does?
v = df[df.cond == "vinegar"].set_index(["label", "side"]).d
hv = df[df.cond == "hungry_vinegar"].set_index(["label", "side"]).d
both = pd.DataFrame({"vinegar": v, "hungry_vinegar": hv}).dropna()
both["shift"] = both.hungry_vinegar - both.vinegar
print("\n=== does hunger change the vinegar response? ===")
print(f"correlation of the two response profiles: r = "
      f"{np.corrcoef(both.vinegar, both.hungry_vinegar)[0,1]:.3f}")
top = both.reindex(both.shift.abs().sort_values(ascending=False).index).head(8)
print(top.round(2).to_string())
print("done (scan)")
