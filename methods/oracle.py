"""Greedy hindsight oracle (not realizable): previews the update, peeks at the next
target, and commits only if the update lowers next-step error by at least `min_gain`
MASE. Its preview energy is not charged. It bounds how much a perfectly informed
selective policy could gain over blind triggers.
"""
from methods.base import Policy
from metrics.forecasting import mase as _mase


class Oracle(Policy):
    realizable = False

    def should_update(self, t, ctx):
        nxt = ctx["peek"]()
        if nxt is None:
            return False
        learner, (x_next, y_next), scale = ctx["learner"], nxt, ctx["scale"]
        before = _mase(learner._forecast(x_next)[0], y_next, scale)
        snap = learner.snapshot()
        learner.update()
        after = _mase(learner._forecast(x_next)[0], y_next, scale)
        learner.restore(snap)
        return before - after >= self.cfg["min_gain"]
