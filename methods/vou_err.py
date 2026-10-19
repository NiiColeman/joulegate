"""Error-excess value-of-update (ablation of VoU; the first design, superseded).

Update only when the predicted error reduction is worth its energy:

    benefit B_t = g * max(0, s_t - ref_t) * horizon        [MASE x steps]
    cost    C_t = EMA of measured update joules             [J]
    update  iff  B_t >= lam * C_t

s_t    short-term error EMA (what the model is doing now)
ref_t  the error level updates have been able to reach (EMA of post-update errors),
       capped by the long-term EMA
g      learned update efficacy: after each update, the fraction of the excess error
       that the update actually removed. Irreducible error (noise bursts) drives g
       down, so the policy learns when not to update; g_min keeps it able to recover.

Differences from the drift trigger: absolute excess instead of a ratio, a learned
efficacy term, and an explicit, measured energy price.
"""
from methods.base import Policy


class VoUErr(Policy):
    needs_cost = True

    def __init__(self, cfg):
        super().__init__(cfg)
        self.short = self.long = self.floor = None
        self.g = cfg["g_init"]
        self.pending = None  # (pre_error, pre_excess, errors seen since update)

    def _ema(self, old, new, a):
        return new if old is None else (1 - a) * old + a * new

    def _ref(self):
        return self.long if self.floor is None else min(self.long, self.floor)

    def observe(self, t, err):
        c = self.cfg
        if self.pending is not None:
            pre, pre_excess, seen = self.pending
            seen.append(err)
            if len(seen) >= c["probe_steps"]:
                realized = pre - sum(seen) / len(seen)
                g_hat = min(1.0, max(0.0, realized / max(pre_excess, 1e-8)))
                self.g = min(1.0, max(c["g_min"], self._ema(self.g, g_hat, c["alpha_g"])))
                self.floor = self._ema(self.floor, sum(seen) / len(seen), c["alpha_floor"])
                self.pending = None
        self.short = self._ema(self.short, err, c["alpha_short"])
        self.long = self._ema(self.long, err, c["alpha_long"])

    def benefit(self):
        return self.g * max(0.0, self.short - self._ref()) * self.cfg["horizon"]

    def should_update(self, t, ctx):
        if t < self.cfg["min_steps"] or self.pending is not None:
            return False
        return self.benefit() >= self.cfg["lam"] * ctx["cost_est"]

    def after_update(self, t, joules):
        excess = max(0.0, self.short - self._ref())
        self.pending = (self.short, excess, [])
