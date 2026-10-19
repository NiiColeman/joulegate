"""Residual VoU: value-of-update gating without a gradient probe.

Same unbiased agreement estimate as VoU, U = mean_{i != j} <r_i, r_j>, but over the newest
forecast errors r_i (RevIN-normalised, channels x horizon) instead of adapter gradients. The
loop computes these errors anyway, so the decision costs no extra model pass. A systematic
shift makes consecutive errors point the same way (U ~ squared bias per element); noise makes
them cancel (U ~ 0). Errors are scaled by 1/sqrt(n) so U is per element and comparable across
streams with different channel counts. Its lam is on a different scale from VoU's.

Negative result (dev sweep, 2026-10-02): on these seasonal streams consecutive errors do not
agree even under a systematic shift. The stream advances a quarter period per step, so the
error pattern changes phase and sign from one forecast to the next (U < 0 throughout on
synthetic_dev, no updates at any lam). Gradients agree where errors do not, because the
model maps phase-shifted errors onto the same parameter direction. Kept for reproducibility;
not in the evaluation grids.
"""
from methods.vou import VoU


class VoUResid(VoU):
    def accumulate(self, r):
        super().accumulate(r / r.numel() ** 0.5)

    def should_update(self, t, ctx):
        self.accumulate(ctx["learner"].last_residual().double())
        if t < self.cfg["min_steps"] or self.s < 2:
            return False
        return self.cfg["horizon"] * max(0.0, self.alignment()) >= self.cfg["lam"] * self.price(ctx["cost_est"])
