"""Measured neuropeptide and receptor expression for central complex cell types.

Source: GSE271123, cell-type-specific RNA profiling behind Wolff et al.'s CX
split-GAL4 collection. Seven driver lines covering six cell types, each sorted
and sequenced in 3-9 replicates:

    ER5  hDeltaK  FB6A (two independent lines)  FB7A  FB2I_a/b  ExR1

Narrow, but real, and it lands on the sleep axis (ER5, FB6A, FB7A) and inside
the central complex, which is where this project works. FB6A being driven by
two unrelated lines gives a reproducibility check that costs nothing.

The paper's EASI-FISH survey covers 80+ cell types; that lives in its
supplementary tables rather than in GEO and is not loaded here.
"""
import gzip

import numpy as np
import pandas as pd

from flyvoca.paths import RAW

COUNTS = RAW / "transcriptome" / "GSE271123_counts.csv.gz"

#: driver line -> v783 cell type(s)
LINES = {
    "SS00070": ["ER5"],
    "SS02748": ["hDeltaK"],
    "SS54343": ["FB6A"],
    "SS55888": ["FB7A"],
    "SS56319": ["FB2I_a", "FB2I_b"],
    "SS56684": ["ExR1"],
    "SS57656": ["FB6A"],
}

CLASSES = ("neuropeptides", "neuropeptide_receptors")


def load(cpm_threshold: float = 50.0):
    """Return (cpm, calls) -- expression per driver line, and a boolean call.

    Counts are converted to CPM against the full gene set before thresholding,
    since library size varies several-fold between lines.
    """
    d = pd.read_csv(COUNTS, low_memory=False)
    reps = {ss: [c for c in d.columns
                 if c.startswith(ss + "_r") and c[len(ss) + 2:].isdigit()]
            for ss in LINES}

    total = {ss: d[cols].sum().sum() for ss, cols in reps.items()}
    sub = d[d.gene_class.isin(CLASSES)].copy()
    cpm = pd.DataFrame({ss: sub[cols].sum(axis=1) / total[ss] * 1e6
                        for ss, cols in reps.items()})
    cpm.index = pd.MultiIndex.from_arrays(
        [sub.gene_class.values, sub.gene_symbol.values], names=["class", "gene"])
    return cpm, cpm > cpm_threshold


def receptor_to_peptide():
    """Invert the ligand-receptor table: receptor gene -> peptide.

    Matching is case-folded: FlyBase writes `Dh44-R1` where the peptide is
    conventionally `DH44`, and a silent case mismatch drops real links.
    """
    from .peptide import LIGAND_RECEPTOR
    return {r.lower(): p for p, rs in LIGAND_RECEPTOR.items() for r in rs}


#: peptide gene symbol in the expression data -> peptide name used here
PEPTIDE_ALIAS = {"Dh31": "DH31", "Dh44": "DH44", "Crz": "CRZ", "Ms": "DMS",
                 "Capa": "CAPA", "Hug": "Hugin", "Ilp2": "DILP", "ITP": "ITP"}


def releasers(cpm_threshold: float = 50.0):
    """Cell types measured to *express* a peptide -- releasers the connectome
    cannot show. The central complex cells are themselves peptidergic, so
    limiting emitters to the endocrine cells would miss most of the network."""
    cpm, called = load(cpm_threshold)
    pep = called.loc["neuropeptides"]
    out = []
    for ss, cell_types in LINES.items():
        for gene, on in pep[ss].items():
            if on:
                name = PEPTIDE_ALIAS.get(gene, gene)
                for ct in cell_types:
                    out.append({"cell_type": ct, "peptide": name,
                                "cpm": round(float(cpm.loc[("neuropeptides", gene), ss]))})
    return pd.DataFrame(out)


def fill(expr, meta, cpm_threshold: float = 50.0):
    """Write measured receptor expression into an `Expression`.

    Only the cell types actually sequenced are marked as measured; every other
    neuron stays unmeasured rather than silently becoming receptor-negative.
    """
    cpm, called = load(cpm_threshold)
    r2p = receptor_to_peptide()
    rec = called.loc["neuropeptide_receptors"]

    filled = []
    for ss, cell_types in LINES.items():
        idx = meta[meta["cell_type"].isin(cell_types)]["idx"].values
        if not len(idx):
            continue
        expr.known[idx] = True
        for gene, on in rec[ss].items():
            pep = r2p.get(gene.lower())
            if on and pep in expr.peptides:
                expr.set(idx, pep, 1.0, source=f"GSE271123:{ss}")
                filled.append((",".join(cell_types), gene, pep, len(idx)))
    return pd.DataFrame(filled, columns=["cell_type", "receptor", "peptide", "n"])
