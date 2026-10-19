import json

import numpy as np
import torch

from analysis.analyze import load
from datasets.streams import build_stream
from engine.energy import AnalyticMeter
from engine.learner import Learner
from engine.online import run_stream
from methods import build_policy
from metrics.forecasting import mase
from models.mixer import PatchMixer
from utils.config import build


class ConstantErrorStream:
    """Wraps a forecasting stream but scores every step 0.25 and has no MASE scale."""

    def __init__(self, inner):
        self.inner = inner

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, t):
        return self.inner[t]

    def error(self, pred, y):
        return 0.25


def _setup():
    cfg = build("ettm1", "periodic", "analytic_45nm", ["seed=0", "dataset.n_steps=20", "method.period=4"])
    torch.manual_seed(0)
    model = PatchMixer(**cfg["model"])
    model.load_state_dict(torch.load(cfg["checkpoint"], map_location="cpu"))
    learner = Learner(model, cfg["learner"])
    learner.meter = AnalyticMeter(model, cfg["energy"])
    return cfg, learner, build_stream(cfg["dataset"], cfg["model"]["context"], cfg["model"]["horizon"])


def test_forecasting_stream_scores_with_mase():
    cfg, learner, stream = _setup()
    ctx, y = stream[0]
    pred = np.zeros_like(y)
    assert stream.error(pred, y) == mase(pred, y, stream.scale)


def test_loop_uses_the_streams_error_and_needs_no_scale():
    cfg, learner, stream = _setup()
    s, tr = run_stream(ConstantErrorStream(stream), learner, build_policy(cfg["method"]), cfg["loop"])
    assert s["error"] == 0.25 and "mase" not in s
    assert np.all(tr["err"] == 0.25)


def test_forecasting_summary_keeps_mase_equal_to_error():
    cfg, learner, stream = _setup()
    s, _ = run_stream(stream, learner, build_policy(cfg["method"]), cfg["loop"])
    assert s["mase"] == s["error"]


def test_analysis_loads_vision_summaries(tmp_path):
    d = tmp_path / "core50" / "frozen" / "seed0"
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps({"error": 0.3, "n_updates": 0, "adapt_j": 0.0}))
    row = load(str(tmp_path)).iloc[0]
    assert row.mase == 0.3 and np.isnan(row.retention_mse)
