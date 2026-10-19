"""Detect-then-adapt trigger (D3A; Addressing concept shift in online time series forecasting:
Detect-then-Adapt, arXiv 2403.14949, 2024), its drift detector as the update trigger.

B holds the errors since the last update. With mu and sd over B and m the mean of the latest l_w errors:
    alarm iff |B| > l_w and (m - mu) / (sd / sqrt|B|) > alpha     (one-sided: rises only)
or, when restart_period is set, t % restart_period == 0. On an alarm the adapter is updated on the FIFO
buffer (our update stands for D3A's full tuning on its memory) and B is cleared.
Omitted: D3A's one-step updates on every window between alarms (they put D3A at or above always-update
energy, which our always-update point covers) and its Gaussian-noise augmentation from an older memory.
"""
import numpy as np

from methods.base import Policy


class D3A(Policy):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.B = []

    def observe(self, t, err):
        self.B.append(err)

    def should_update(self, t, ctx):
        period = self.cfg.get("restart_period")
        if period and t > 0 and t % period == 0:
            return True
        lw = self.cfg["window"]
        if len(self.B) <= lw:
            return False
        b = np.asarray(self.B, dtype=np.float64)
        sd = b.std(ddof=1)
        if sd == 0:
            return False
        return bool((b[-lw:].mean() - b.mean()) / (sd / np.sqrt(len(b))) > self.cfg["alpha"])

    def after_update(self, t, joules):
        self.B = []
