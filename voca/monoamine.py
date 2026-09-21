"""Monoaminergic edges as a slow layer.

Doc 03 kept DA / 5-HT / OA out of the LIF matrix on purpose: they act
through G-protein-coupled receptors over seconds, so putting them in as
millisecond conductances would be wrong in kind, not just in degree. Every
persistence test through doc 16 therefore ran without them.

This puts them back as what they are. Each monoaminergic synapse in W_slow
drives a per-target concentration that decays with `tau` seconds,

    dc_j/dt = -c_j / tau + sum_i W_slow[i, j] * rate_i

and the saturated concentration shifts target j's resting potential by
`gain * sign` millivolts -- the `Field` arithmetic of doc 04, applied edge by
edge because release here is synaptic rather than humoral. The three
transmitter channels add, and the sum is clipped to +-`gain`, so `gain` is
the largest shift any cell can receive.

Concentrations are kept per trial. The trials of a run differ in membrane
noise and Poisson drive; if they shared one concentration, the offset would
be a single realisation whose effect never enters the trial-to-trial
variance, and every effect-size against sham would be inflated.

What the data does not say, and this class therefore takes as arguments:

  signs   which receptor a target expresses. OA: OAMB / Octbeta (Gq / Gs, +)
          or Octalpha2R (Gi, -). 5-HT: 5-HT7 / 2A (+) or 1A / 1B (-).
          DA: Dop1R1 (+) or Dop2R (-). No pC1 / aIPg receptor data exists.
  gain    mV shift at saturation. The gap to threshold is 7 mV.
  tau     seconds. Jung et al. 2020 measured 13 s (pCd driven directly)
          and 83 s (via P1) for the male integrator.
"""
import numpy as np
from scipy import sparse

CLASSES = ("OCT", "SER", "DA")


def split_by_transmitter(W_slow, slow_class):
    """W_slow -> {class: W.T (targets x sources), unsigned counts}.

    `slow_class` is the per-neuron transmitter derived from the edge table
    (`meta["slow_class"]`), not the neuron-level prediction, which is missing
    for 19,658 cells. Every slow edge must land in exactly one class."""
    n = W_slow.shape[0]
    Wa = abs(W_slow).tocsr().astype(np.float32)
    row_of = np.repeat(np.arange(n), np.diff(Wa.indptr))
    pre = np.asarray(slow_class).astype(str)
    out, kept = {}, 0
    for cls in CLASSES:
        M = Wa.copy()
        M.data[pre[row_of] != cls] = 0.0
        M.eliminate_zeros()
        kept += M.nnz
        out[cls] = M.T.tocsr()
    if kept != Wa.nnz:
        raise ValueError(f"{Wa.nnz - kept:,} of {Wa.nnz:,} slow edges have no "
                         f"transmitter class; pass meta['slow_class']")
    return out


class SlowEdges:
    """Per-target, per-trial monoamine concentration, one channel per transmitter."""

    def __init__(self, WT_by_class, tau: float, gain: float, signs: dict,
                 n_trials: int = 1, mute=(), half_sat_syn: float = 30.0,
                 charge_s: float = 2.0):
        self.WT = WT_by_class
        self.n = next(iter(WT_by_class.values())).shape[0]
        self.B = int(n_trials)
        self.tau, self.gain, self.signs = float(tau), float(gain), dict(signs)
        self.c = {k: np.zeros((self.n, self.B), dtype=np.float32) for k in self.WT}
        # sources whose release is switched off (a control, not a model)
        self.mask = np.ones(self.n, dtype=np.float32)
        self.mask[np.asarray(mute, dtype=np.int64)] = 0.0
        # Half-saturation. Concentration units are synapses x Hz x s, and
        # nothing measured fixes their scale, so it is set by a statement
        # rather than a number: one partner firing at 100 Hz through
        # `half_sat_syn` synapses for the whole stimulus (`charge_s`
        # seconds) half-saturates its target. Everything weaker or briefer
        # does proportionally less.
        self.half = 100.0 * half_sat_syn * self.tau * (1.0 - np.exp(-charge_s / self.tau))

    def step(self, rates: np.ndarray, dt: float):
        """`rates` is (n, trials) Hz, or (n,) to drive every trial alike."""
        r = np.asarray(rates, dtype=np.float32)
        if r.ndim == 1:
            r = np.repeat(r[:, None], self.B, axis=1)
        r = r * self.mask[:, None]
        decay = float(np.exp(-dt / self.tau))
        for k, M in self.WT.items():
            self.c[k] = self.c[k] * decay + (M @ r) * (1.0 - decay) * self.tau

    def settle(self, rest_rates: np.ndarray):
        """Bring the layer to its steady state for a resting network, so a
        run starts at the operating point instead of ramping from zero."""
        self.step(rest_rates, 5.0 * self.tau)

    def v_offset(self) -> np.ndarray:
        """(n, trials) shift in resting potential, mV, clipped to +-gain."""
        off = np.zeros((self.n, self.B), dtype=np.float32)
        for k, c in self.c.items():
            off += self.signs.get(k, 0.0) * (c / (c + self.half))
        return np.clip(off, -1.0, 1.0) * self.gain

    def saturation(self) -> dict:
        """Fraction of targets above half-saturation, per channel (trial mean)."""
        return {k: round(float((c > self.half).mean()), 4) for k, c in self.c.items()}
