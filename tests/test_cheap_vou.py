"""Lower-cost VoU variants: residual agreement (no probe) and shift-gated probing."""
import numpy as np
import pytest
import torch

from methods import build_policy
from methods.vou_gated import VoUGated
from methods.vou_resid import VoUResid
from utils.config import build


class FakeLearner:
    """Supplies the signals the policies read and counts gradient probes."""

    def __init__(self, residuals, mus=None, sds=None, grads=None):
        self.residuals, self.mus, self.sds, self.grads = residuals, mus, sds, grads
        self.t, self.probes = 0, 0

    def at(self, t):
        self.t = t
        if self.mus is not None:
            self.last_mu = torch.tensor(self.mus[t]).reshape(-1, 1)
            self.last_sd = torch.tensor(self.sds[t]).reshape(-1, 1)
        return self

    def last_residual(self):
        return torch.tensor(self.residuals[self.t])

    def probe_grad(self):
        self.probes += 1
        return torch.tensor(self.grads[self.t]), 0.0


def run(policy, learner, steps, cost=1.0):
    out = []
    for t in range(steps):
        ctx = {"learner": learner.at(t), "cost_est": cost, "charge": lambda j: None}
        policy.observe(t, float(np.abs(learner.residuals[t]).mean()))
        d = policy.should_update(t, ctx)
        if d:
            policy.after_update(t, cost)
        out.append(d)
    return out


RES = {"lam": 1.0, "horizon": 10, "min_steps": 4}


def test_residual_agreement_is_near_zero_for_noise_and_large_for_a_bias():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1.0, (60, 48))
    biased = noise + 0.8
    p_noise, p_bias = VoUResid(RES), VoUResid(RES)
    for t in range(60):
        for p, r in ((p_noise, noise), (p_bias, biased)):
            p.accumulate(torch.tensor(r[t]))
    assert abs(p_noise.alignment()) < 0.05 and p_bias.alignment() == pytest.approx(0.64, rel=0.25)


def test_residual_vou_updates_on_a_bias_but_not_on_noise_and_never_probes():
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 1.0, (100, 48))
    quiet = FakeLearner(noise)
    shifted = FakeLearner(noise + np.where(np.arange(100)[:, None] >= 20, 0.8, 0.0))
    d_quiet = run(VoUResid(dict(RES, lam=2.0)), quiet, 100)
    d_shift = run(VoUResid(dict(RES, lam=2.0)), shifted, 100)
    assert not any(d_quiet) and any(d_shift[20:])
    assert quiet.probes == shifted.probes == 0


GATED = {"lam": 1.0, "horizon": 10, "min_steps": 4, "arm_steps": 8, "kappa": 4.0, "alpha_in": 0.05,
         "tau": 0.3, "alpha_short": 0.5, "alpha_long": 0.05}


def stationary_inputs(steps, shift_at=None, mean_jump=0.0, sd_scale=1.0, seed=2):
    rng = np.random.default_rng(seed)
    mus = rng.normal(0, 0.05, (steps, 3))
    sds = np.exp(rng.normal(0, 0.02, (steps, 3)))
    if shift_at is not None:
        mus[shift_at:] += mean_jump
        sds[shift_at:] *= sd_scale
    return mus, sds


def test_input_gate_fires_on_mean_or_spread_shifts_only():
    for jump, scale, expect in ((0.0, 1.0, False), (1.0, 1.0, True), (0.0, 2.0, True)):
        mus, sds = stationary_inputs(80, 50, jump, scale)
        g, fired = VoUGated(dict(GATED, gate="input")), []
        for t in range(80):
            fired.append(g.shift_detected(t, FakeLearner(None, mus, sds).at(t)))
        assert any(fired[50:55]) == expect and not any(fired[10:50]), (jump, scale)


def test_gated_vou_probes_only_in_a_window_after_a_shift_then_cools_down():
    mus, sds = stationary_inputs(80, 50, 1.0)
    grads = np.random.default_rng(3).normal(0, 0.05, (80, 16))  # weak noise: VoU never updates
    learner = FakeLearner(np.zeros((80, 4)), mus, sds, grads)
    g, probes = VoUGated(dict(GATED, gate="input")), []
    for t in range(80):
        assert not g.should_update(t, {"learner": learner.at(t), "cost_est": 1.0, "charge": lambda j: None})
        probes.append(learner.probes)
    a = GATED["arm_steps"]
    assert probes[49] == 0 and probes[50 + a - 1] == a          # one window of arm_steps probes
    assert probes[50 + 2 * a - 1] == a                          # none during the cool-down


def test_each_update_extends_the_window_so_vou_keeps_adapting_after_a_shift():
    """One update rarely re-adapts the model after a shift; VoU keeps going while gradients agree."""
    mus, sds = stationary_inputs(80, 50, 1.0)
    grads = np.full((80, 16), 1.0)                                # agreeing gradients: VoU updates
    learner = FakeLearner(np.zeros((80, 4)), mus, sds, grads)
    d = run(VoUGated(dict(GATED, gate="input", lam=1e-3)), learner, 80)
    updates = [t for t, x in enumerate(d) if x]
    assert updates[0] >= 50 and len(updates) > 5
    assert updates[-1] > 50 + GATED["arm_steps"]                  # beyond the first window


def test_error_gate_uses_the_drift_rule():
    errs = np.r_[np.full(40, 1.0), np.full(40, 2.0)]
    learner = FakeLearner(errs[:, None] * np.ones((80, 4)), grads=np.zeros((80, 16)))
    g, fired = VoUGated(dict(GATED, gate="error")), []
    for t in range(80):
        g.observe(t, errs[t])
        fired.append(g.shift_detected(t, learner.at(t)))
    assert not any(fired[:40]) and any(fired[40:45])


def test_registry_builds_both_from_their_configs():
    assert isinstance(build_policy(build("ettm1", "vou_resid")["method"]), VoUResid)
    assert isinstance(build_policy(build("ettm1", "vou_gated")["method"]), VoUGated)
