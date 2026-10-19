"""CARA (Mahadevan & Mathioudakis, Cost-aware retraining for machine learning, Knowledge-Based Systems
2024), priced in energy.

Staleness as the paper defines it: the stale model's loss on new data minus its loss on its own training
data near that data (an RBF kernel over inputs). Both are the learner's training loss (`item_losses`):

  Psi_t = L(adapter, newest window q) - sum_i w_i L(adapter, x_i) / sum_i w_i,
  w_i = exp(-|f(x_i) - f(q)|^2 / (2 s^2)),  s^2 = median_i |f(x_i) - f(q)|^2

over the items x_i of the buffer the adapter was last trained on; f is an item's frozen-prefix activations
averaged to one vector, and the bandwidth is the median heuristic. (Revised 2026-10-12: the first versions
used the uniform mean, which a single hard window in a short training buffer made too high, so CARA-CT's
signed sum stayed negative after its first update.)

The reference L(adapter, training buffer) is taken right after each update, on the FIFO buffer as it was at
that update (no replay). The pretrained model's training data is the base data, so its reference is its
loss on a buffer-sized sample of held-out base data (the retention set), measured at its first decision.
(Revised 2026-10-12: the first version used the first stream buffer, and a single hard or easy first buffer
made the signed sum of CARA-CT never or always fire.) All passes are forward-only and metered as probe energy; the policy charges
their joules through ctx["charge"] (the reference pass, inside after_update, uses the ctx kept from that
step's should_update). The paper's retraining cost kappa is set in staleness units and its threshold
fitted offline; here the threshold is lam x update joules, the same energy price as VoU, and lam is
chosen on the dev streams.

  variant threshold   (CARA-T):   update iff Psi_t >= lam * C
  variant cumulative  (CARA-CT):  update iff sum of Psi since the last update >= lam * C (signed, as in
                                  the paper; reset to 0 after each update)
CARA-P (periodic) is our periodic family.
"""
import numpy as np

from methods.base import Priced


class CARA(Priced):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.ref, self.cum, self.ctx = None, 0.0, None
        self.ref_losses = self.ref_feats = None

    def _charged(self, which):
        losses, feats, joules = self.ctx["learner"].item_losses(which)
        self.ctx["charge"](joules)
        return losses, feats

    def _set_reference(self, which):
        self.ref_losses, self.ref_feats = self._charged(which)

    def _staleness(self):
        (loss,), (f,) = self._charged("newest")
        d2 = ((self.ref_feats - f) ** 2).sum(1)
        s2 = np.median(d2)
        w = np.exp(-d2 / (2 * s2)) if s2 > 0 else np.ones_like(d2)
        self.ref = float((w * self.ref_losses).sum() / w.sum())
        return float(loss) - self.ref

    def should_update(self, t, ctx):
        self.ctx = ctx
        price = self.price(ctx["cost_est"])
        if price is None:
            return False
        if self.ref_losses is None:
            self._set_reference("base")
        stale = self._staleness()
        self.cum += stale
        if t < self.cfg["min_steps"]:
            return False
        s = self.cum if self.cfg["variant"] == "cumulative" else stale
        return s >= self.cfg["lam"] * price

    def after_update(self, t, joules):
        self._set_reference("buffer")
        self.cum = 0.0
