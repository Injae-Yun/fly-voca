"""Cross-check: where does the 71.9 Hz MN9 benchmark actually come from?

core/docs/02-calibration.md says "the reference's own connectivity matrix,
run through our simulator, gives MN9 = 71.9 Hz" -- a different code path
from flyvoca.graph.build() (our own connectome export signed edge-by-edge).
T4 (87_dale_test.py) measured flyvoca.graph.build() at w_syn=0.275 giving
162.6 Hz, not 71.9. This reproduces the reference-matrix path directly to
settle which number is which, so the T4 report cites the right one.
"""
from voca.reference import matrix, MN9
from flyvoca.sim import Brain, Params

idx, W = matrix()
mn9 = idx[MN9]
brain = Brain(W, Params(w_syn=0.275, sigma_v=0.0, r_spont=0.0))
# same ref-20 ids used in 87_dale_test.py / 14_calibrate_wsyn.py
from voca.reference import SUGAR as REF_SUGAR
ref = [i for i in REF_SUGAR if i in idx]
stim = [idx[i] for i in ref]
print(f"ref-20 present in reference matrix: {len(ref)}/21")
c = brain.run(stim=stim, t_run=1000.0, n_trials=30, r_poi=100.0, seed=1)
m, s = Brain.rate(c)
print(f"reference's OWN connectivity matrix + our simulator: "
      f"MN9 = {float(m[mn9]):.1f} Hz (sd {float(s[mn9]):.1f})")
print("(this is the number core/docs/02-calibration.md calls 71.9 Hz / "
      "reference target 67.0 Hz -- a distinct code path from "
      "flyvoca.graph.build())")
print("\ndone (ref matrix check)")
