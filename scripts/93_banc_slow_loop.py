"""Run the slow layer on BANC and watch an internal state build over time.

Doc 04 showed the machinery works on FAFB: driving the DH44 neurosecretory
cells raised ER5 tenfold along a route the connectome does not contain. But
coverage was 77 neurons, and the emitters were inferred from cell-type names.

BANC annotates 7,025 emitters across 34 peptides, and independently confirms
the two links doc 04 derived from RNA-seq (ER5 -> Dh31, hDeltaK -> AstA/AstC/
NPF). Receptors are still only GSE271123's six central-complex cell types, so
the field is now well-sourced and the listeners are not.

Two things are measured, because they are different claims:

  does the field build     peptide concentrations over time, driven by real
                           firing rather than a clamp. Hunger is not a
                           stimulus; it is what happens when nothing is eaten.
  does state follow        the eight axes, per epoch, with the peptide offset
                           fed back into the next block

The gain-zero run is the control. Doc 04 established that direction survives
the whole plausible gain range, but on a new graph that has to be re-shown,
not assumed.
"""
import numpy as np
import pandas as pd

from flyvoca.banc_graph import build, select, spontaneous_groups
from flyvoca.banc_peptide import BancField, emitter_index
from flyvoca.paths import cache
from bodysnatch.sim import Brain, Params
from bodysnatch.expression import receptor_to_peptide, load as load_expr, LINES, CONFIDENCE

W_SYN, TRIALS, BLOCK, EPOCHS = 0.16, 6, 500.0, 12

meta, W, _, st = build()
groups = spontaneous_groups(meta)
t = meta.cell_type.astype(str)
emit = emitter_index(meta)
field = BancField(meta, tau=20.0, gain=3.0)
print(f"peptides tracked: {len(field.peptides)}  emitters: "
      f"{sum(len(v) for v in emit.values()):,}")

# ---- receptors: GSE271123 cell types mapped onto BANC ----
cpm, called, frac = load_expr()
r2p = receptor_to_peptide()
from bodysnatch.peptide import COUPLING

pep_i = {p: j for j, p in enumerate(field.peptides)}
Wr = np.zeros((len(meta), len(field.peptides)), dtype=np.float32)
known = np.zeros(len(meta), dtype=bool)
links = []
for ss, cts in LINES.items():
    idx = meta[t.isin(cts)]["idx"].values
    if not len(idx):
        print(f"  {ss} {cts}: absent from BANC"); continue
    known[idx] = True
    for gene, on in called.loc["neuropeptide_receptors"][ss].items():
        pep = r2p.get(gene.lower())
        if not on or pep not in pep_i:
            continue
        sign, conf, _ = COUPLING.get(gene, (None, None, None))
        if not sign:
            continue
        Wr[idx, pep_i[pep]] = float(sign)
        links.append({"cell_type": ",".join(cts), "receptor": gene, "peptide": pep,
                      "sign": "+" if sign > 0 else "-", "n": len(idx),
                      "emitters": len(emit.get(pep, [])),
                      "conf": CONFIDENCE.get(cts[0], "single_line")})
L = pd.DataFrame(links)
print(f"\nreceptor links: {len(L)} across {int(known.sum())} neurons "
      f"({known.mean()*100:.3f}% of BANC)")
print(L.to_string(index=False) if len(L) else "  none")

hungry_src = emit.get("DH44", np.array([], dtype=np.int64))
print(f"\nDH44 emitters in BANC: {len(hungry_src)} "
      f"(FAFB had 6 m_NSC_DH44)")


def axis(*pats):
    m = np.zeros(len(meta), dtype=bool)
    for p in pats:
        m |= t.str.fullmatch(p, na=False).values
        m |= t.str.startswith(p, na=False).values
    return meta.index.values[m]


AXES = {
    "hunger": axis("m_NSC_DILP", "m_NSC_DH44", "l_NSC_DH31", "SEZ_NSC_Hugin",
                   "m_NSC_DMS", "SEZ_NSC_CAPA"),
    "thirst": axis("ISN"), "sleep": axis("ER5", "FB6", "FB7"),
    "clock": axis("s-LNv", "l-LNv", "LPN", "DN2", "LNd", "DN1"),
    "courtship": axis("pC1"), "taste": axis("LB3"),
    "steering": axis("PFL3", "PFL2", "DNa02", "DNa03"),
}
WATCH = {"ER5": axis("ER5"), "hDeltaK": axis("hDeltaK"),
         "FB6A": axis("FB6A"), "ExR1": axis("ExR1")}

for gain in (0.0, 3.0):
    field = BancField(meta, tau=20.0, gain=gain)
    b = Brain(W, Params(w_syn=W_SYN, sigma_v=2.0, r_spont=0.0))
    b.spont_groups = groups
    state, rows = None, []
    print(f"\n=== gain = {gain} mV at saturation ===")
    for ep in range(EPOCHS):
        off = field.v_offset(Wr)
        counts, state = b.run(stim=hungry_src, t_run=BLOCK, n_trials=TRIALS,
                              r_poi=50.0, seed=14000 + ep, state=state,
                              v_offset=off, return_state=True)
        m = counts.mean(axis=1) / (BLOCK / 1000)
        field.step(m, dt=BLOCK / 1000)
        row = {"t_s": round((ep + 1) * BLOCK / 1000, 1),
               "DH44_c": round(float(field.c[pep_i["DH44"]]), 3)}
        row.update({k: round(float(m[v].mean()), 2) for k, v in WATCH.items()})
        row.update({k: round(float(m[v].mean()), 2) for k, v in AXES.items()})
        rows.append(row)
        print(row, flush=True)
    df = pd.DataFrame(rows)
    print(f"  ER5 {df.ER5.iloc[0]:.2f} -> {df.ER5.iloc[-1]:.2f} Hz   "
          f"hDeltaK {df.hDeltaK.iloc[0]:.2f} -> {df.hDeltaK.iloc[-1]:.2f}")
    df.to_csv(cache(f"banc_slow_gain{gain:g}.csv"), index=False)

print("\ndone (slow loop)")
