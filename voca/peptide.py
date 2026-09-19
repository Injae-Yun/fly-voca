"""The slow layer: peptides, and the state they carry.

Peptides do not travel along the connectome. They are released into the
surrounding tissue or into the haemolymph and reach whoever carries the
receptor, which is why a wiring diagram cannot show them. Three consequences
shape this module:

  1. Transmission is a *field*, not an edge. A peptide has a concentration;
     neurons expressing its receptor feel that concentration.
  2. It is slow. Concentrations move over seconds to minutes, against the
     millisecond spiking of the fast graph -- so the two are stepped at
     different rates and met in the middle.
  3. It is modulatory. The effect is a shift in how excitable a cell is,
     not a spike delivered to it: `Brain.run(v_offset=...)`.

Who releases what is known: the 80 endocrine cells in v783 are typed by
peptide. Who *listens* is not in the connectome, and brain-wide receptor
expression does not exist in clean machine-readable form -- mapping
single-cell transcriptome clusters onto connectome cell types is itself an
open problem. Two regions do have real data (central complex EASI-FISH across
80+ cell types and 17 peptides; the neurosecretory network's own receptors),
and those happen to be exactly where this project works.

`Expression` therefore distinguishes "no receptor" from "not measured", and
refuses to let the second quietly become the first.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: Established Drosophila peptide -> receptor pairs. Verify against FlyBase
#: before treating any single row as authoritative.
LIGAND_RECEPTOR = {
    "DILP":   ["InR"],
    "AKH":    ["AkhR"],
    "DH44":   ["Dh44-R1", "Dh44-R2"],
    "DH31":   ["Dh31-R"],
    "CRZ":    ["CrzR"],
    "Hugin":  ["PK2-R1", "PK2-R2"],
    "CAPA":   ["CapaR"],
    "DMS":    ["MsR1", "MsR2"],
    "sNPF":   ["sNPF-R"],
    "NPF":    ["NPFR"],
    "AstA":   ["AstA-R1", "AstA-R2"],
    "AstC":   ["AstC-R1", "AstC-R2"],
    "Tk":     ["TkR86C", "TkR99D"],
    "CNMa":   ["CNMaR"],
    "RYa":    ["RYa-R"],
    "ETH":    ["ETHR"],
    "Nplp1":  [],          # receptor not firmly assigned
    "LK":     ["Lkr"],
    "PDF":    ["Pdfr"],
    "SIFa":   ["SIFaR"],
    "Proc":   ["Proc-R"],
    "CCAP":   ["CCAP-R"],
    "CCHa1":  ["CCHa1-R"],
    "CCHa2":  ["CCHa2-R"],
    "Mip":    ["SPR"],
    "FMRFa":  ["FR"],
    "Dsk":    ["CCKLR-17D1", "CCKLR-17D3"],
    "ITP":    [],          # receptor not firmly assigned
}

#: v783 endocrine cell types -> the peptide they release. These are the
#: emitters, and they are the half the connectome does give us.
RELEASERS = {
    "m_NSC_DILP": "DILP",
    "m_NSC_DH44": "DH44",
    "l_NSC_DH31": "DH31",
    "l_NSC_CRZ": "CRZ",
    "m_NSC_DMS": "DMS",
    "SEZ_NSC_Hugin": "Hugin",
    "SEZ_NSC_CAPA": "CAPA",
    "l_NSC_ITP": "ITP",
}

#: Neurosecretory cells of the pars intercerebralis and pars lateralis release
#: into the haemolymph, so their signal is systemic rather than local.
SYSTEMIC = {"DILP", "DH44", "DH31", "CRZ", "DMS", "ITP", "AKH"}


@dataclass
class Expression:
    """Which neurons carry which receptor, and which we simply have not measured.

    `known` marks the neurons whose receptor complement has actually been
    determined. Everything outside it is unmeasured, and reading it as absence
    is how a model quietly invents results.
    """
    n: int
    peptides: tuple
    weight: np.ndarray = None          # (n, n_peptides), receptor sensitivity
    known: np.ndarray = None           # (n,) bool, was this neuron measured
    source: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.weight is None:
            self.weight = np.zeros((self.n, len(self.peptides)), dtype=np.float32)
        if self.known is None:
            self.known = np.zeros(self.n, dtype=bool)

    def set(self, idx, peptide: str, value: float = 1.0, source: str = ""):
        j = self.peptides.index(peptide)
        idx = np.asarray(idx, dtype=np.int64)
        self.weight[idx, j] = value
        self.known[idx] = True
        if source:
            self.source[peptide] = source

    def coverage(self) -> pd.Series:
        return pd.Series({"measured": int(self.known.sum()),
                          "unmeasured": int((~self.known).sum()),
                          "fraction": round(float(self.known.mean()), 4)})


class Field:
    """Peptide concentrations, driven by their releasers and decaying slowly.

        dc/dt = -c / tau + k * (mean firing rate of the cells that release it)

    Concentration is in arbitrary units; `gain` converts it to millivolts of
    resting-potential shift, and is the knob that has to be calibrated against
    something measured before any of this means anything.
    """

    def __init__(self, meta, peptides=None, tau: float = 20.0, gain: float = 1.0):
        self.peptides = tuple(peptides or sorted(set(RELEASERS.values())))
        self.tau = tau            # seconds
        self.gain = gain          # mV per unit concentration
        self.releaser_idx = {}
        for ct, pep in RELEASERS.items():
            if pep not in self.peptides:
                continue
            idx = meta[meta["cell_type"] == ct]["idx"].values
            if len(idx):
                self.releaser_idx.setdefault(pep, []).append(idx)
        self.releaser_idx = {p: np.concatenate(v) for p, v in self.releaser_idx.items()}
        self.c = np.zeros(len(self.peptides), dtype=np.float32)

    def step(self, rates: np.ndarray, dt: float, k: float = 0.01):
        """Advance concentrations by `dt` seconds given current firing rates."""
        decay = np.exp(-dt / self.tau)
        for j, pep in enumerate(self.peptides):
            idx = self.releaser_idx.get(pep)
            drive = float(rates[idx].mean()) if idx is not None and len(idx) else 0.0
            self.c[j] = self.c[j] * decay + k * drive * (1.0 - decay) * self.tau
        return self.c.copy()

    def v_offset(self, expr: Expression) -> np.ndarray:
        """Concentrations -> per-neuron shift in resting potential (mV)."""
        return (expr.weight @ self.c) * self.gain

    def state(self) -> dict:
        return dict(zip(self.peptides, np.round(self.c, 4)))
