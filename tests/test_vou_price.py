import numpy as np
import torch

from methods.vou import VoU


class GradSource:
    """Stands in for the learner: a fixed sequence of probe gradients."""

    def __init__(self, seed=0):
        self.g = torch.from_numpy(np.random.default_rng(seed).normal(0.3, 1.0, (200, 16)))
        self.i = 0

    def probe_grad(self):
        self.i += 1
        return self.g[self.i - 1], 0.0


def decisions(scale, price_scale=None, lam=1.0):
    vou, src, out = VoU({"lam": lam, "horizon": 10, "min_steps": 4}), GradSource(), []
    vou.price_scale = price_scale
    for t in range(150):
        ctx = {"learner": src, "cost_est": scale * (0.1 + 0.01 * np.sin(t)), "charge": lambda j: None}
        d = vou.should_update(t, ctx)
        if d:
            vou.after_update(t, ctx["cost_est"])
        out.append(d)
    return out


def test_price_scale_makes_decisions_match_the_analytic_scale():
    """Measured joules are k x analytic; dividing by price_scale = k restores the decisions lam was tuned for."""
    analytic = decisions(1.0)
    assert any(analytic)
    assert decisions(12.0, price_scale=12.0) == analytic


def test_without_price_scale_the_joule_scale_changes_decisions():
    assert decisions(12.0) != decisions(1.0)
