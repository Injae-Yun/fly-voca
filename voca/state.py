"""Internal state axes -- the "emotion" readout.

A fly does not have emotions, but it does have persistent internal states that
bias what every sensory input means: hunger, thirst, arousal, sexual state,
defensive arousal. Each axis here is a set of real, verified v783 cell types
whose firing we read out as one number.

Two honest limits:

  * The connectome carries who *emits* which peptide (80 endocrine cells typed
    by peptide) but not who *listens* -- receptor expression lives in
    transcriptomic data we have not joined yet. So these axes are read as
    firing rates of identified populations, not as peptide concentrations.
  * FAFB v783 is a female brain. Male-specific courtship circuitry (P1, pIP10)
    does not exist in it; pC1 is the female counterpart.

`exact` matches a cell type verbatim, `prefix` matches a family. The two are
kept separate because e.g. prefix "LC4" would also swallow LC40..LC46.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Axis:
    name: str
    why: str
    exact: tuple = ()
    prefix: tuple = ()

    def members(self, meta):
        t = meta["cell_type"].astype("string")
        m = t.isin(self.exact)
        for p in self.prefix:
            m |= t.str.startswith(p, na=False)
        return meta[m]


AXES = (
    Axis("hunger", "neurosecretory cells governing feeding and energy state",
         exact=("m_NSC_DILP", "m_NSC_DH44", "l_NSC_DH31", "SEZ_NSC_Hugin",
                "m_NSC_DMS", "SEZ_NSC_CAPA")),
    Axis("thirst", "interoceptive SEZ neurons",
         exact=("ISN",)),
    Axis("sleep_pressure", "dorsal fan-shaped body and the R5 homeostat",
         exact=("ER5",), prefix=("FB6", "FB7")),
    Axis("clock", "circadian network",
         exact=("s-LNv", "l-LNv", "LPN", "DN2"), prefix=("LNd", "DN1")),
    Axis("courtship", "female receptivity (pC1); FAFB has no male P1",
         prefix=("pC1",)),
    Axis("defense", "looming detectors and the giant fibre escape path",
         exact=("LC4", "LPLC2", "LC6", "DNp01", "DNp11")),
    Axis("taste_reward", "labellar sugar sensilla",
         exact=("LB3",)),
    Axis("steering", "goal-to-turn comparator and its descending output",
         exact=("PFL3", "PFL2", "DNa02", "DNa03")),
)

BY_NAME = {a.name: a for a in AXES}


def registry(meta):
    """What each axis actually resolves to in this dataset."""
    rows = []
    for a in AXES:
        m = a.members(meta)
        rows.append({"axis": a.name, "n": len(m),
                     "types": len(m["cell_type"].dropna().unique()),
                     "why": a.why})
    import pandas as pd
    return pd.DataFrame(rows)


def read(meta, rates, axes=AXES):
    """Firing rates -> one mean Hz per axis. This is the state vector."""
    out = {}
    for a in axes:
        idx = a.members(meta)["idx"].values
        out[a.name] = float(rates[idx].mean()) if len(idx) else float("nan")
    return out


def lateral(meta, rates, cell_type):
    """Left/right mean rate for one type -- the steering signal is a difference."""
    t = meta[meta["cell_type"] == cell_type]
    return {s: float(rates[g["idx"].values].mean())
            for s, g in t.groupby("side")}
