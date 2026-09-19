"""Measured neuropeptide and receptor expression for central complex cell types.

Source: GSE271123, the cell-type-specific RNA profiling behind the Wolff et al.
central complex split-GAL4 collection. Seven driver lines, six cell types,
3-9 replicates each:

    ER5  hDeltaK  FB6A (two independent lines)  FB7A  FB2I_a/b  ExR1

Narrow, and weaker than it looks. Three things are done about that:

  * Detection is decided by replicate consistency, not by a threshold alone.
    A gene counts as expressed when it clears the cut in most replicates of a
    line, so one outlier replicate cannot carry a call.

  * The cut comes from the data. The pooled log10(CPM) distribution is
    trimodal -- background near 1, ordinary expression near 12, and a handful
    of very high peptides near 10^4 -- putting the off/on boundary at CPM ~8.
    `sensitivity()` reports how the recovered links move with the cut, because
    that boundary is soft and pretending otherwise would be dishonest.

  * Every call carries a confidence. FB6A is the only cell type with a
    replicate driver line, and the two disagree across the whole profile
    (r = 0.475 on log CPM), so it is marked `low`. The others have no second
    line, which does not make them right -- only unchecked.
"""
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

#: cell types measured by more than one line, and whether the lines agreed
CONFIDENCE = {"FB6A": "low"}          # two lines, r = 0.475 -- do not trust
DEFAULT_CONFIDENCE = "single_line"    # unchecked, not validated

CLASSES = ("neuropeptides", "neuropeptide_receptors")
CPM_CUT = 8.0          # off/on antimode of the pooled distribution
MIN_FRACTION = 0.6     # share of a line's replicates that must clear the cut


def _replicates(d):
    return {ss: [c for c in d.columns
                 if c.startswith(ss + "_r") and c[len(ss) + 2:].isdigit()]
            for ss in LINES}


def load(cpm_cut: float = CPM_CUT, min_fraction: float = MIN_FRACTION):
    """Return (cpm, called, detected_fraction).

    `cpm` is the mean CPM over a line's replicates; `detected_fraction` is the
    share of those replicates in which the gene clears the cut; `called`
    requires both.
    """
    d = pd.read_csv(COUNTS, low_memory=False)
    reps = _replicates(d)
    lib = {ss: d[cols].sum() for ss, cols in reps.items()}      # per replicate
    sub = d[d.gene_class.isin(CLASSES)].copy()
    idx = pd.MultiIndex.from_arrays(
        [sub.gene_class.values, sub.gene_symbol.values], names=["class", "gene"])

    cpm, frac = {}, {}
    for ss, cols in reps.items():
        per_rep = sub[cols].div(lib[ss], axis=1) * 1e6          # CPM per replicate
        cpm[ss] = per_rep.mean(axis=1).values
        frac[ss] = (per_rep > cpm_cut).mean(axis=1).values
    cpm = pd.DataFrame(cpm, index=idx)
    frac = pd.DataFrame(frac, index=idx)
    called = (cpm > cpm_cut) & (frac >= min_fraction)
    return cpm, called, frac


def receptor_to_peptide():
    """Invert the ligand-receptor table: receptor gene -> peptide.

    Case-folded: FlyBase writes `Dh44-R1` where the peptide is conventionally
    `DH44`, and a silent case mismatch drops real links.
    """
    from .peptide import LIGAND_RECEPTOR
    return {r.lower(): p for p, rs in LIGAND_RECEPTOR.items() for r in rs}


#: peptide gene symbol in the expression data -> peptide name used here
PEPTIDE_ALIAS = {"Dh31": "DH31", "Dh44": "DH44", "Crz": "CRZ", "Ms": "DMS",
                 "Capa": "CAPA", "Hug": "Hugin", "Ilp2": "DILP", "ITP": "ITP"}


def releasers(**kw):
    """Cell types measured to release a peptide -- emitters the connectome
    cannot show. Central complex cells are themselves peptidergic, so limiting
    emitters to the endocrine cells would miss most of the network."""
    cpm, called, frac = load(**kw)
    pep = called.loc["neuropeptides"]
    out = []
    for ss, cell_types in LINES.items():
        for gene, on in pep[ss].items():
            if not on:
                continue
            for ct in cell_types:
                out.append({"cell_type": ct,
                            "peptide": PEPTIDE_ALIAS.get(gene, gene),
                            "cpm": round(float(cpm.loc[("neuropeptides", gene), ss])),
                            "rep_frac": round(float(frac.loc[("neuropeptides", gene), ss]), 2),
                            "confidence": CONFIDENCE.get(ct, DEFAULT_CONFIDENCE)})
    return pd.DataFrame(out)


def fill(expr, meta, **kw):
    """Write measured receptor expression, with its sign, into `Expression`.

    The sign comes from the receptor's G protein coupling: Gs and Gq raise
    excitability, Gi/o lowers it. Receptors with no coupling entry are skipped
    rather than assumed excitatory.
    """
    from .peptide import COUPLING
    cpm, called, frac = load(**kw)
    r2p = receptor_to_peptide()
    rec = called.loc["neuropeptide_receptors"]

    rows = []
    for ss, cell_types in LINES.items():
        idx = meta[meta["cell_type"].isin(cell_types)]["idx"].values
        if not len(idx):
            continue
        expr.known[idx] = True
        for gene, on in rec[ss].items():
            pep = r2p.get(gene.lower())
            if not on or pep not in expr.peptides:
                continue
            sign, conf, basis = COUPLING.get(gene, (None, None, None))
            if not sign:                    # unknown or non-GPCR: skip, do not guess
                continue
            expr.set(idx, pep, float(sign), source="GSE271123:" + ss)
            rows.append({"cell_type": ",".join(cell_types), "receptor": gene,
                         "peptide": pep, "sign": "+" if sign > 0 else "-",
                         "coupling": conf,
                         "cpm": round(float(cpm.loc[("neuropeptide_receptors", gene), ss])),
                         "rep_frac": round(float(frac.loc[("neuropeptide_receptors", gene), ss]), 2),
                         "data_conf": CONFIDENCE.get(cell_types[0], DEFAULT_CONFIDENCE),
                         "n": len(idx)})
    return pd.DataFrame(rows)


def sensitivity(meta, cuts=(4.0, 8.0, 20.0, 50.0, 200.0)):
    """How many links survive at each cut. The boundary is soft, so show it."""
    from .peptide import Expression
    out = []
    for c in cuts:
        peps = sorted(set(releasers(cpm_cut=c).peptide) |
                      {"DILP", "DH44", "DH31", "CRZ", "DMS", "Hugin", "CAPA", "ITP"})
        e = Expression(len(meta), tuple(peps))
        links = fill(e, meta, cpm_cut=c)
        exc = int((links.sign == "+").sum()) if len(links) else 0
        inh = int((links.sign == "-").sum()) if len(links) else 0
        out.append({"cpm_cut": c, "links": len(links), "excitatory": exc,
                    "inhibitory": inh, "neurons_measured": int(e.known.sum())})
    return pd.DataFrame(out)
