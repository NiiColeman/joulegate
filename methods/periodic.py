"""Update every `period` steps. period = 1 is the always-update energy ceiling.

Sweeping the period gives the naive energy knob: a fixed duty cycle.
"""
from methods.base import Policy


class Periodic(Policy):
    def should_update(self, t, ctx):
        return t % self.cfg["period"] == self.cfg["period"] - 1
