import numpy as np
import pytest
import torch

from engine.energy import AnalyticMeter
from engine.online import run_stream
from engine.vision_learner import VisionLearner
from methods import build_policy
from models.vit_tiny import VIT_TARGETS, TinyViT
from utils.config import load_yaml

LCFG = {"adapt_blocks": 2, "rank": 4, "alpha": 8, "lr": 1e-3, "steps_per_update": 2, "buffer_steps": 4,
        "cache_prefix": True, "lora_targets": list(VIT_TARGETS), "train_head": True}
ECFG = load_yaml("configs/energy/analytic_45nm.yaml")


class TinyStream:
    """Random normalised images with labels; enough to drive the loop."""

    def __init__(self, n=8, batch=4, seed=0):
        rng = np.random.default_rng(seed)
        self.x = rng.uniform(-1, 1, (n, batch, 3, 96, 96)).astype(np.float32)
        self.y = rng.integers(0, 10, (n, batch)).astype(np.int64)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, t):
        return self.x[t], self.y[t]

    def error(self, logits, y):
        return float((np.asarray(logits).argmax(1) != y).mean())


def _learner(device="cpu", **kw):
    torch.manual_seed(0)
    m = TinyViT(pretrained=False)
    lr = VisionLearner(m, dict(LCFG, **kw), device=device)
    lr.meter = AnalyticMeter(m, ECFG)
    return lr


def _fill(lr, stream, n=3):
    for t in range(n):
        x, y = stream[t]
        lr.predict(x)
        lr.push(x, y)


def test_predict_returns_logits_and_batch_statistics():
    lr, s = _learner(), TinyStream()
    logits, j, sec = lr.predict(s[0][0])
    assert logits.shape == (4, 10) and j > 0
    assert tuple(lr.last_mu.shape) == (3, 1) and tuple(lr.last_sd.shape) == (3, 1)


def test_update_changes_only_adapter_parameters_and_probe_changes_nothing():
    lr, s = _learner(), TinyStream()
    _fill(lr, s)
    frozen = {n: p.detach().clone() for n, p in lr.model.named_parameters() if not p.requires_grad}
    before = [p.detach().clone() for p in lr.params]
    g, _ = lr.probe_grad()
    assert all(torch.equal(a, b) for a, b in zip(before, lr.params))
    assert g.numel() == lr.n_params
    lr.update()
    assert any(not torch.equal(a, b) for a, b in zip(before, lr.params))
    assert all(torch.equal(frozen[n], p) for n, p in lr.model.named_parameters() if not p.requires_grad)


def test_prefix_caching_gives_the_same_update():
    s = TinyStream()
    a, b = _learner(), _learner(cache_prefix=False)
    _fill(a, s), _fill(b, s)
    a.update(), b.update()
    for p, q in zip(a.params, b.params):
        torch.testing.assert_close(p, q, rtol=1e-4, atol=1e-5)


def test_vision_learner_runs_every_policy_family_through_the_loop():
    loop = {"cost_ema": 0.3, "min_buffer": 2, "buffer_eval": True, "budget": None}
    for method in ({"policy": "periodic", "period": 2}, {"policy": "drift", "tau": 0.1, "alpha_short": 0.5,
                   "alpha_long": 0.05, "min_steps": 2}, {"policy": "vou", "lam": 0.01, "horizon": 10, "min_steps": 2},
                   {"policy": "vou_gated", "gate": "input", "kappa": 3, "alpha_in": 0.2, "tau": 0.2,
                    "alpha_short": 0.5, "alpha_long": 0.05, "arm_steps": 4, "lam": 0.01, "horizon": 10,
                    "min_steps": 2}):
        s, tr = run_stream(TinyStream(), _learner(), build_policy(method), loop)
        assert 0.0 <= s["error"] <= 1.0 and "mase" not in s, method["policy"]


def test_retention_is_accuracy_on_a_stored_set(tmp_path):
    lr = _learner()
    rng = np.random.default_rng(1)
    np.savez(tmp_path / "ret.npz", images=rng.integers(0, 256, (6, 96, 96, 3), dtype=np.uint8),
             labels=rng.integers(0, 10, 6))
    acc = lr.retention(str(tmp_path / "ret.npz"), mean=[0.5] * 3, std=[0.5] * 3)
    assert 0.0 <= acc <= 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_gated_vision_run_on_cuda():
    loop = {"cost_ema": 0.3, "min_buffer": 2, "buffer_eval": False, "budget": None}
    method = {"policy": "vou_gated", "gate": "input", "kappa": 3, "alpha_in": 0.2, "tau": 0.2, "alpha_short": 0.5,
              "alpha_long": 0.05, "arm_steps": 4, "lam": 0.01, "horizon": 10, "min_steps": 2}
    s, _ = run_stream(TinyStream(), _learner(device="cuda"), build_policy(method), loop)
    assert 0.0 <= s["error"] <= 1.0
