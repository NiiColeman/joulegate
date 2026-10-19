import numpy as np
import pytest
import torch

from datasets.streams import build_stream
from engine.energy import AnalyticMeter
from engine.learner import Learner
from engine.online import run_stream
from methods import build_policy
from methods.base import Policy
from methods.vou import VoU
from models.mixer import PatchMixer
from utils.config import build


class SlowSettleMeter(AnalyticMeter):
    """Analytic values that only 'settle' after `lag` settled() calls, like a slow counter."""

    def __init__(self, *a, lag=3, **k):
        super().__init__(*a, **k)
        self.lag, self.calls = lag, {}

    def settled(self, i):
        self.calls[i] = self.calls.get(i, 0) + 1
        return self.values[i] if self.calls[i] >= self.lag else None

    def wait_settled(self, i, timeout=5.0):
        while (v := self.settled(i)) is None:
            pass
        return v


class SpyVoU(VoU):
    seen = []

    def should_update(self, t, ctx):
        SpyVoU.seen.append(ctx["cost_est"])
        return super().should_update(t, ctx)


def _setup(method, sets, meter_cls=AnalyticMeter, device="cpu"):
    cfg = build("ettm1", method, "analytic_45nm", sets + ["seed=0", "dataset.n_steps=60"])
    torch.manual_seed(0)
    model = PatchMixer(**cfg["model"])
    model.load_state_dict(torch.load(cfg["checkpoint"], map_location="cpu"))
    learner = Learner(model, cfg["learner"], device=device)
    learner.meter = meter_cls(model, cfg["energy"])
    stream = build_stream(cfg["dataset"], cfg["model"]["context"], cfg["model"]["horizon"])
    return cfg, learner, stream


def test_price_uses_only_settled_values():
    cfg, learner, stream = _setup("vou", ["method.lam=0.01"], meter_cls=SlowSettleMeter)
    SpyVoU.seen = []
    summary, _ = run_stream(stream, learner, SpyVoU(cfg["method"]), cfg["loop"])
    assert summary["n_updates"] > 0
    assert all(c is not None and c > 0 for c in SpyVoU.seen)
    assert summary["price_lag_mean"] == pytest.approx(2.0)


def test_buffer_eval_logs_losses_on_update_steps_only():
    cfg, learner, stream = _setup("periodic", ["method.period=4", "loop.buffer_eval=true"])
    _, trace = run_stream(stream, learner, build_policy(cfg["method"]), cfg["loop"])
    upd = np.asarray(trace["update"], dtype=bool)
    assert np.isfinite(trace["buf_before"][upd]).all() and np.isfinite(trace["buf_after"][upd]).all()
    assert np.isnan(trace["buf_before"][~upd]).all()


def test_buffer_eval_does_not_change_results():
    cfg, learner, stream = _setup("periodic", ["method.period=4"])
    a, _ = run_stream(stream, learner, build_policy(cfg["method"]), cfg["loop"])
    cfg, learner, stream = _setup("periodic", ["method.period=4", "loop.buffer_eval=true"])
    b, _ = run_stream(stream, learner, build_policy(cfg["method"]), cfg["loop"])
    assert a["mase"] == b["mase"] and a["update_j"] == b["update_j"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_cuda_learner_matches_cpu():
    _, cpu, stream = _setup("periodic", ["method.period=4"])
    _, gpu, _ = _setup("periodic", ["method.period=4"], device="cuda")
    ctx, _ = stream[0]
    np.testing.assert_allclose(cpu._forecast(ctx)[0], gpu._forecast(ctx)[0], rtol=1e-4, atol=1e-4)


def test_learner_exposes_window_statistics_and_normalised_residual():
    """Cheap VoU variants read signals the forecast already computes; nothing extra is run."""
    cfg, learner, stream = _setup("periodic", ["method.period=4"])
    ctx, y = stream[0]
    pred, _, _ = learner.predict(ctx)
    learner.push(ctx, y)
    C = ctx.shape[0]
    assert tuple(learner.last_mu.shape) == (C, 1) and tuple(learner.last_sd.shape) == (C, 1)
    np.testing.assert_allclose(learner.last_mu.cpu().numpy()[:, 0], ctx.mean(1), rtol=1e-4, atol=1e-4)
    r = learner.last_residual()
    assert tuple(r.shape) == (C * cfg["model"]["horizon"],)
    mu, sd = learner.last_mu.cpu().numpy(), learner.last_sd.cpu().numpy()
    expected = np.clip((pred - mu) / sd, -1e9, 1e9) - np.clip((y - mu) / sd, -10, 10)
    np.testing.assert_allclose(r.cpu().numpy(), expected.reshape(-1), rtol=1e-3, atol=1e-3)


def test_relative_budget_caps_updates_at_the_income_rate():
    """Income of 10 % of an update per step, bucket starting empty: every-step updating is cut to ~1 in 10."""
    cfg, learner, stream = _setup("periodic", ["method.period=1",
                                               "loop.budget={income_frac: 0.1, capacity_updates: 1}"])
    summary, trace = run_stream(stream, learner, build_policy(cfg["method"]), cfg["loop"])
    n_paid = len(stream) - learner.buffer.maxlen + 1   # steps that earn income after calibration
    assert 0 < summary["n_updates"] <= int(0.1 * n_paid) + 1
    b = trace["bucket"][np.isfinite(trace["bucket"])]   # -inf before calibration
    assert b.min() > -summary["j_per_update"]           # at most one update of debt while the price settles


class LateBurst(Policy):
    """Wants an update at every step from step 40 on: only a bucket that saved up can pay for a burst."""

    def should_update(self, t, ctx):
        return t >= 40


def test_relative_budget_capacity_lets_a_policy_save_up():
    n = {}
    for cap in (1, 10):
        cfg, learner, stream = _setup("periodic", [f"loop.budget={{income_frac: 0.1, capacity_updates: {cap}}}"])
        s, _ = run_stream(stream, learner, LateBurst(cfg["method"]), cfg["loop"])
        n[cap] = s["n_updates"]
    assert n[10] > n[1] >= 1


def test_relative_budget_rejects_absolute_and_relative_together():
    cfg, learner, stream = _setup("periodic", ["method.period=1",
                                               "loop.budget={income_frac: 0.1, capacity_updates: 1, income_j: 1.0}"])
    with pytest.raises(ValueError, match="budget"):
        run_stream(stream, learner, build_policy(cfg["method"]), cfg["loop"])
