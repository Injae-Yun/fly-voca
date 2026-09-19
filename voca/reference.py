"""The published Shiu et al. 2024 model, for benchmarking against.

Files live in <data>/reference/ (shared, never committed), fetched from
github.com/philshiu/Drosophila_brain_model:

    ref_conn783.parquet   their connectivity matrix (15.1M pairs, 54.5M synapses)
    ref_comp783.csv       their neuron list (138,639)
    ref_100.parquet       spike times from their published sugar @100Hz run
"""
import numpy as np
import pandas as pd
from scipy import sparse

from flyvoca.paths import DATA_ROOT

REF = DATA_ROOT / "reference"

#: 21 sugar-sensing neurons hand-listed in their notebook; 20 survive in v783,
#: where they are typed LB3.
SUGAR = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]
MN9 = 720575940660219265


def matrix():
    """Their signed connectivity as (index map, W)."""
    comp = pd.read_csv(REF / "ref_comp783.csv", index_col=0)
    con = pd.read_parquet(REF / "ref_conn783.parquet")
    n = len(comp)
    W = sparse.csr_matrix(
        (con["Excitatory x Connectivity"].values.astype(np.float32),
         (con["Presynaptic_Index"].values, con["Postsynaptic_Index"].values)),
        shape=(n, n))
    return {f: i for i, f in enumerate(comp.index)}, W


def published_rates(t_run: float = 1.0):
    """Firing rates from their released spike data (sugar @ 100 Hz, 30 trials)."""
    d = pd.read_parquet(REF / "ref_100.parquet")
    n_trial = d.trial.nunique()
    return d.groupby("flywire_id").size() / (n_trial * t_run)
