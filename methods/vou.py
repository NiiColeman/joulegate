"""Value-of-update (VoU) gating with a gradient-alignment benefit estimate (proposed).

Each step the learner probes the adapter gradient g_i on the newest window only (a
forward/backward through the adapted suffix on C samples; its energy is charged).
Since the last update we keep S = sum g_i and Q = sum ||g_i||^2 over s steps, and form
the unbiased estimate of the squared norm of the expected gradient

    U = (||S||^2 - Q) / (s (s - 1))  =  mean_{i != j} <g_i, g_j>

Pure noise averages to U ~ 0 however large the individual gradients are; a systematic
shift of the data makes the gradients agree and U grows. A gradient step of size eta
lowers the loss on future data by about eta * U per step (first-order Taylor), so

    benefit B = horizon * max(0, U)        cost C = EMA of measured update joules
    update iff B >= lam * C

Accumulators reset after every update, since earlier gradients describe the old adapter.

lam is a price in benefit per joule, tuned against analytic joules. Measured joules are on a
different scale per device, so a measured run sets `price_scale` = measured / analytic joules
of the same full-buffer update (engine/measured.py) and the price becomes C / price_scale.
"""
import torch

from methods.base import Policy


class VoU(Policy):
    needs_cost = True

    def __init__(self, cfg):
        super().__init__(cfg)
        self.S, self.Q, self.s = None, 0.0, 0
        self.price_scale = None  # set by measured runs; None leaves prices in their own joules

    def _reset(self):
        self.S, self.Q, self.s = None, 0.0, 0

    def alignment(self):
        if self.s < 2:
            return 0.0
        return float((self.S @ self.S - self.Q) / (self.s * (self.s - 1)))

    def should_update(self, t, ctx):
        g, joules = ctx["learner"].probe_grad()
        ctx["charge"](joules)
        self.accumulate(g)
        if t < self.cfg["min_steps"] or self.s < 2:
            return False
        return self.cfg["horizon"] * max(0.0, self.alignment()) >= self.cfg["lam"] * self.price(ctx["cost_est"])

    def accumulate(self, g):
        self.S = g.clone() if self.S is None else self.S + g
        self.Q += float(g @ g)
        self.s += 1

    def price(self, cost_est):
        return cost_est if self.price_scale is None else cost_est / self.price_scale

    def after_update(self, t, joules):
        self._reset()
