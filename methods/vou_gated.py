"""Shift-gated VoU: a free shift detector decides when VoU pays for gradient probes.

Plain VoU probes every step. Here a detector watches a signal the loop already has and, when
it reports a shift, arms VoU for `arm_steps` steps: VoU probes and updates if the gradients
agree, and every update extends the window by `arm_steps`, because re-adapting after a shift
takes a run of updates, not one. After `arm_steps` steps without an update VoU is disarmed
(no probes, no updates) and cannot re-arm for `cooldown` steps.

gate: input  the context window's per-channel mean and log spread (RevIN statistics) against
             their exponentially weighted mean and variance (alpha_in); a shift is a z-score
             above kappa on any channel, after 1/alpha_in warm-up steps
gate: error  the drift rule: short error EMA above (1 + tau) x long error EMA
"""
import torch

from methods.vou import VoU


class VoUGated(VoU):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.armed_until = -10 ** 9               # last step of the current or last armed window
        self.m = self.v = self.l = self.w = None  # EW mean/variance of window mean and log spread
        self.n = 0
        self.short = self.long = None             # error EMAs (error gate)

    def observe(self, t, err):
        a_s, a_l = self.cfg["alpha_short"], self.cfg["alpha_long"]
        self.short = err if self.short is None else (1 - a_s) * self.short + a_s * err
        self.long = err if self.long is None else (1 - a_l) * self.long + a_l * err

    def _input_z(self, learner):
        mu = learner.last_mu.reshape(-1).double().cpu()
        ls = torch.log(learner.last_sd.reshape(-1).double().cpu())
        if self.m is None:
            self.m, self.l = mu.clone(), ls.clone()
            self.v, self.w = torch.zeros_like(mu), torch.zeros_like(ls)
            return 0.0
        eps = 1e-12
        z = max(float(((mu - self.m).abs() / (self.v + eps).sqrt()).max()),
                float(((ls - self.l).abs() / (self.w + eps).sqrt()).max()))
        a = self.cfg["alpha_in"]
        for x, mean, var in ((mu, self.m, self.v), (ls, self.l, self.w)):
            d = x - mean
            mean += a * d
            var.mul_(1 - a).add_((1 - a) * a * d * d)
        return z

    def shift_detected(self, t, learner):
        """True while the detector reports a shift."""
        if self.cfg["gate"] == "input":
            z = self._input_z(learner)
            self.n += 1
            now = self.n > 1 / self.cfg["alpha_in"] and z > self.cfg["kappa"]
        elif self.cfg["gate"] == "error":
            now = t >= self.cfg["min_steps"] and self.short > (1 + self.cfg["tau"]) * self.long
        else:
            raise ValueError(f"unknown gate {self.cfg['gate']!r}")
        return now

    def should_update(self, t, ctx):
        cooldown = self.cfg.get("cooldown", self.cfg["arm_steps"])
        if self.shift_detected(t, ctx["learner"]) and t > self.armed_until + cooldown:
            self.armed_until = t + self.cfg["arm_steps"] - 1
            self._reset()                         # fresh evidence for this shift only
        if t > self.armed_until:
            return False
        return super().should_update(t, ctx)

    def after_update(self, t, joules):
        super().after_update(t, joules)
        self.armed_until = t + self.cfg["arm_steps"]  # keep adapting while the gradients agree
