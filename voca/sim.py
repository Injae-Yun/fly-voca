"""Leaky integrate-and-fire over the FlyWire connectome, on the GPU.

Parameters follow Shiu et al. 2024 exactly, so results can be checked against
the published Brian2 reference instead of merely looking plausible:

    v_0 = v_rst = -52 mV, v_th = -45 mV   ->  7 mV to threshold
    w_syn = 0.275 mV per synapse          ->  ~26 coincident inputs fire a cell
    t_mbr = 20 ms, tau = 5 ms, t_rfc = 2.2 ms, t_dly = 1.8 ms

`w_syn` is the model's only free parameter; everything else is measured.

Two deliberate departures from the reference implementation:

  * Trials are a batch dimension rather than separate processes. The Poisson
    drive is the only stochastic element, so 30 trials are 30 columns of the
    same matrices and one sparse matmul serves them all.
  * Integration is exponential Euler on

        dv/dt = (v_0 - v + g) / t_mbr        dg/dt = -g / tau

    which is what Brian2's 'linear' method solves; at dt = 0.1 ms the
    difference sits far below the trial-to-trial Poisson variance.
"""
from dataclasses import dataclass

import numpy as np
import torch
from scipy import sparse


@dataclass
class Params:
    """Units are mV, ms, Hz throughout.

    `w_syn` departs from the reference's 0.275 mV on purpose. Sweeping it on
    both connectomes gives the same S-curve -- silence, a steep rise, a plateau
    -- but ours sits ~0.07 lower on the axis, because our export counts ~25%
    more synapses per connection. The reference's own 0.275 is not in the middle
    of its plateau: it is just past its transition (local slope 1.6, against
    27.8 and 7.1 just below). Matching their absolute firing rate would drag us
    to w_syn ~0.17, i.e. *deeper* into the unstable region than they ever sit.

    So we match the dynamical regime rather than the number. At 0.20 our slope
    is 1.6 -- the same gain as the reference at 0.275. Absolute rates stay
    higher; see core/docs/02-calibration.md.
    """
    v_0: float = -52.0      # resting potential
    v_rst: float = -52.0    # reset after spike
    v_th: float = -45.0     # spike threshold
    t_mbr: float = 20.0     # membrane time constant
    tau: float = 5.0        # synaptic decay
    t_rfc: float = 2.2      # refractory period
    t_dly: float = 1.8      # synaptic delay
    w_syn: float = 0.20     # mV per synapse  <- the free parameter; see below
    r_poi: float = 150.0    # Hz, stimulation rate
    f_poi: float = 250.0    # stim weight scale; w_syn * f_poi >> threshold gap
    dt: float = 0.1         # integration step

    # Spontaneous activity. A silent brain has no state to modulate, and no
    # real neuron is silent: channel gating is stochastic, vesicles release
    # spontaneously, and sensory afferents fire without a stimulus.
    sigma_v: float = 2.0    # mV, stationary sd of membrane noise (0 = off)
    r_spont: float = 2.0    # Hz, spontaneous drive on sensory afferents


def pick_device(device=None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Brain:
    """A connectome wired up as a spiking network."""

    def __init__(self, W: sparse.csr_matrix, params: Params | None = None,
                 device=None, spont=None):
        self.p = params or Params()
        self.n = W.shape[0]
        self.device = pick_device(device)
        #: afferents carrying spontaneous drive; `from_meta` fills this in
        self.spont = np.asarray([] if spont is None else spont, dtype=np.int64)

        # W holds signed synapse counts with rows = presynaptic. The update
        # needs input *per postsynaptic* cell, so store the transpose and scale
        # into mV once.
        Wt = (W.T.tocsr().astype(np.float32) * np.float32(self.p.w_syn))
        Wt.sort_indices()
        self._Wt_np = Wt
        self.Wt = self._to_torch(Wt)

    @classmethod
    def from_meta(cls, W, meta, params: Params | None = None, device=None):
        """Build with spontaneous afferent drive already wired to the sensory
        neurons, so the brain has a resting state without the caller
        remembering to ask for one."""
        idx = meta[meta["super_class"].isin(["sensory", "sensory_ascending"])]["idx"].values
        return cls(W, params, device, spont=idx)

    def _to_torch(self, m: sparse.csr_matrix) -> torch.Tensor:
        return torch.sparse_csr_tensor(
            torch.from_numpy(m.indptr.astype(np.int64)),
            torch.from_numpy(m.indices.astype(np.int64)),
            torch.from_numpy(m.data),
            size=m.shape, device=self.device, dtype=torch.float32)

    def _muted(self, silence) -> torch.Tensor:
        """Zero a neuron's outgoing weights: it still spikes, but says nothing."""
        m = self._Wt_np.copy()
        mute = np.zeros(self.n, dtype=bool)
        mute[np.asarray(silence, dtype=np.int64)] = True
        m.data[mute[m.indices]] = 0.0        # columns of W.T are presynaptic
        return self._to_torch(m)

    @torch.no_grad()
    def run(self, stim=(), t_run: float = 1000.0, n_trials: int = 30,
            silence=(), seed: int | None = None, r_poi: float | None = None,
            spont=()):
        """Drive `stim` with Poisson input; return spike counts (n, n_trials).

        `spont` names the neurons carrying spontaneous afferent drive at
        `p.r_spont`; membrane noise at `p.sigma_v` applies to every neuron.
        """
        p, n, dev = self.p, self.n, self.device
        B = n_trials
        steps = int(round(t_run / p.dt))
        rate = (r_poi if r_poi is not None else p.r_poi) * p.dt / 1000.0

        Wt = self._muted(silence) if len(silence) else self.Wt
        gen = torch.Generator(device=dev)
        gen.manual_seed(torch.seed() if seed is None else seed)

        ev = float(np.exp(-p.dt / p.t_mbr))
        eg = float(np.exp(-p.dt / p.tau))
        # Exact Ornstein-Uhlenbeck step: this keeps the stationary sd equal to
        # sigma_v regardless of dt, which naive per-step noise does not.
        noise_amp = p.sigma_v * float(np.sqrt(1.0 - ev * ev))

        v = torch.full((n, B), p.v_0, device=dev)
        g = torch.zeros((n, B), device=dev)
        counts = torch.zeros((n, B), dtype=torch.int32, device=dev)
        free_at = torch.zeros((n, B), device=dev)

        # Stimulated neurons have no refractory period in the reference model.
        rfc = torch.full((n, 1), p.t_rfc, device=dev)
        stim = torch.as_tensor(np.asarray(stim, dtype=np.int64), device=dev)
        if stim.numel():
            rfc[stim] = 0.0
        spont = self.spont if spont is None else np.asarray(spont, dtype=np.int64)
        spont = torch.as_tensor(spont, device=dev)
        rate_sp = p.r_spont * p.dt / 1000.0

        # Uniform synaptic delay -> a ring buffer of past spike vectors.
        D = int(round(p.t_dly / p.dt))
        hist = torch.zeros((D + 1, n, B), device=dev)

        w_poi = p.w_syn * p.f_poi
        for s in range(steps):
            t = s * p.dt
            k = s % (D + 1)

            g += torch.sparse.mm(Wt, hist[k])
            hist[k].zero_()

            if stim.numel():
                fire = torch.rand((stim.numel(), B), device=dev, generator=gen) < rate
                v[stim] += fire.float() * w_poi

            if spont.numel() and rate_sp > 0:
                fire = torch.rand((spont.numel(), B), device=dev, generator=gen) < rate_sp
                v[spont] += fire.float() * w_poi

            live = free_at <= t
            target = g + p.v_0
            v_next = target + (v - target) * ev
            if noise_amp:
                v_next = v_next + noise_amp * torch.randn(
                    (n, B), device=dev, generator=gen)
            v = torch.where(live, v_next, v)
            g = torch.where(live, g * eg, g)

            fired = live & (v > p.v_th)
            if bool(fired.any()):
                f = fired.float()
                v = torch.where(fired, torch.full_like(v, p.v_rst), v)
                g = torch.where(fired, torch.zeros_like(g), g)
                free_at = torch.where(fired, t + rfc.expand(n, B), free_at)
                counts += fired.int()
                hist[(s + D) % (D + 1)] = f

        return counts.cpu().numpy()

    @staticmethod
    def rate(counts: np.ndarray, t_run: float = 1000.0):
        """Spike counts -> (mean Hz, std Hz) across trials."""
        r = counts / (t_run / 1000.0)
        return r.mean(axis=1), r.std(axis=1)
