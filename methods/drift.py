"""Drift-triggered updates (the standard baseline in time series CL, e.g. CLeaR-style
triggers or error-magnitude triggers): update when the short-term error EMA exceeds
the long-term error EMA by a relative margin tau.
"""
from methods.base import Policy


class Drift(Policy):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.short = self.long = None

    def observe(self, t, err):
        a_s, a_l = self.cfg["alpha_short"], self.cfg["alpha_long"]
        self.short = err if self.short is None else (1 - a_s) * self.short + a_s * err
        self.long = err if self.long is None else (1 - a_l) * self.long + a_l * err

    def should_update(self, t, ctx):
        if t < self.cfg["min_steps"]:
            return False
        return self.short > (1 + self.cfg["tau"]) * self.long
