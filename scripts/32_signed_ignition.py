"""Re-run the slow layer now that the receptors have signs.

Three things changed since the first ignition:
  * receptors carry the sign of their G protein coupling, so AstA, AstC, NPF
    and myosuppressin inhibit rather than excite
  * the detection cut comes from the data (CPM 8) instead of a guess (50),
    which roughly triples the recovered links
  * concentration saturates, so `gain` is the millivolt shift at saturation --
    a quantity with a literature range rather than a free multiplier

The gain sweep is the point: if the direction of the effect survives the whole
plausible range, the conclusion does not rest on the one number we cannot
measure.
"""
import numpy as np
import pandas as pd

from flyvoca.graph import build
from bodysnatch.sim import Brain
from bodysnatch.peptide import Field, Expression
from bodysnatch.expression import fill, releasers

meta, W, _, _ = build()
brain = Brain.from_meta(W, meta)

peps = sorted(set(releasers().peptide) | {"DILP", "DH44", "DH31", "CRZ", "DMS",
                                          "Hugin", "CAPA", "ITP"})
read = {n: meta[meta["cell_type"] == n]["idx"].values
        for n in ("ER5", "ExR1", "hDeltaK", "FB7A")}
dh44 = meta[meta["cell_type"] == "m_NSC_DH44"]["idx"].values
print(f"peptides {len(peps)}  driving m_NSC_DH44 ({len(dh44)})")

for gain in (0.0, 1.0, 3.0, 6.0):
    field = Field(meta, peptides=peps, tau=20.0, gain=gain)
    expr = Expression(len(meta), tuple(peps))
    fill(expr, meta)

    state, rows = None, []
    for epoch in range(8):
        counts, state = brain.run(stim=dh44, t_run=500.0, n_trials=6, r_poi=50.0,
                                  seed=10 + epoch, state=state,
                                  v_offset=field.v_offset(expr), return_state=True)
        m, _ = Brain.rate(counts, 500.0)
        field.step(m, dt=0.5)
        rows.append({"t_s": (epoch + 1) * 0.5,
                     **{k: round(float(m[v].mean()), 2) for k, v in read.items()},
                     "v_ER5": round(float(field.v_offset(expr)[read["ER5"]].mean()), 2)})
    df = pd.DataFrame(rows)
    print(f"\n--- gain = {gain} mV at saturation ---")
    print(df.to_string(index=False))
    print(f"    ER5 {df.ER5.iloc[0]:.2f} -> {df.ER5.iloc[-1]:.2f} Hz")
