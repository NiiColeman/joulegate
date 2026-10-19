import numpy as np
import pytest
import torch

from utils.config import build


def test_old_backbone_config_is_unchanged():
    cfg = build("ettm1", "vou", "analytic_45nm")
    assert cfg.get("backbone", "patchmixer") == "patchmixer"
    assert cfg["model"]["context"] == 256 and cfg["model"]["horizon"] == 24 and cfg["dataset"]["stride"] == 24
    assert cfg["dataset"]["freq_token"] == 5


@pytest.mark.parametrize("backbone,name", [("ttm", "ttm_r2"), ("chronos", "chronos_bolt")])
def test_new_backbones_use_512_48_with_stride_48(backbone, name):
    for d in ("synthetic", "synthetic_dev", "ettm1", "ettm1_dev", "electricity", "solar"):
        cfg = build(d, "frozen", "analytic_45nm", backbone=backbone)
        assert cfg["backbone"] == backbone and cfg["model"]["name"] == name
        assert cfg["model"]["context"] == 512 and cfg["model"]["horizon"] == 48 and cfg["dataset"]["stride"] == 48
    assert build("synthetic", "frozen", "analytic_45nm", backbone=backbone)["dataset"]["synthetic"]["length"] == 42000
    assert build("electricity", "frozen", "analytic_45nm", backbone=backbone)["dataset"]["start"] == 2000


def test_stride_shorter_than_horizon_is_refused():
    from run import run
    cfg = build("ettm1", "frozen", "analytic_45nm", ["dataset.stride=12"])
    with pytest.raises(ValueError, match="stride"):
        run(cfg, "unused", verbose=False)


def _series(n=3, length=560, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    return torch.tensor(np.sin(2 * np.pi * t / 96)[None] * rng.uniform(1, 3, (n, 1)) + rng.normal(0, 0.1, (n, length)),
                        dtype=torch.float32)


def test_ttm_wrapper_matches_the_package_forecast():
    from models.ttm_r2 import TTMR2
    from models.vendor.ttm.modeling_tinytimemixer import TinyTimeMixerForPrediction
    w = TTMR2("ibm-granite/granite-timeseries-ttm-r2", "512-48-ft-r2.1", freq_token=5).eval()
    ref = TinyTimeMixerForPrediction.from_pretrained("ibm-granite/granite-timeseries-ttm-r2",
                                                     revision="512-48-ft-r2.1").eval()
    x = _series()[:, :512]
    with torch.no_grad():
        xn, mu, sd = w.revin(x)
        ours = w.forward_norm(xn) * sd + mu
        theirs = ref(past_values=x.unsqueeze(-1), freq_token=torch.full((3,), 5)).prediction_outputs[..., 0]
    assert ours.shape == (3, 48)
    assert torch.allclose(ours, theirs, rtol=1e-3, atol=1e-3 * x.abs().max())


def test_ttm_lora_prefix_cache_and_frozen_head():
    from engine.energy import build_meter
    from engine.learner import Learner
    from models.ttm_r2 import TTMR2
    cfg = build("ettm1", "periodic", "analytic_45nm", backbone="ttm")
    w = TTMR2(cfg["model"]["repo"], cfg["model"]["revision"], freq_token=5)
    lr = Learner(w, cfg["learner"])
    lr.meter = build_meter(cfg["energy"], w, "cpu")
    assert lr.first == 0 and lr.n_params > 0 and not any(p.requires_grad for p in w.head.parameters())
    x = _series(n=4).numpy()
    lr.predict(x[:, :512])
    h_cached = lr._last_prefix
    with torch.no_grad():
        assert torch.equal(h_cached, w.embed(w.revin(torch.from_numpy(x[:, :512]))[0]))


def test_chronos_wrapper_matches_the_package_forecast():
    from chronos import BaseChronosPipeline
    from models.chronos_bolt import ChronosBolt
    w = ChronosBolt("amazon/chronos-bolt-tiny").eval()
    ref = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-tiny", device_map="cpu",
                                              torch_dtype=torch.float32).model.eval()
    x = _series()[:, :512]
    with torch.no_grad():
        xn, mu, sd = w.revin(x)
        ours = w.forward_norm(xn) * sd + mu
        theirs = ref(context=x).quantile_preds[:, 4, :48]
    assert ours.shape == (3, 48)
    assert torch.allclose(ours, theirs, rtol=1e-3, atol=1e-3 * x.abs().max())


def test_quantile_learner_uses_pinball_loss_and_median():
    from engine.energy import build_meter
    from engine.quantile_learner import QuantileLearner
    from models.chronos_bolt import ChronosBolt
    cfg = build("ettm1", "periodic", "analytic_45nm", backbone="chronos")
    w = ChronosBolt(cfg["model"]["repo"])
    lr = QuantileLearner(w, cfg["learner"])
    lr.meter = build_meter(cfg["energy"], w, "cpu")
    assert lr.first == 1 and lr.n_params > 0
    q = torch.zeros(2, 9, 48)
    tgt = torch.ones(2, 48)
    levels = torch.linspace(0.1, 0.9, 9)
    assert torch.isclose(lr._loss(q, tgt), (2 * levels).sum())   # under-forecast by 1 at every level
    x = _series(n=4).numpy()
    pred, _, _ = lr.predict(x[:, :512])
    assert pred.shape == (4, 48)


@pytest.mark.parametrize("backbone", ["ttm", "chronos"])
def test_new_backbone_runs_end_to_end(tmp_path, backbone):
    from run import run
    sets = ["dataset.n_steps=6", "learner.buffer_steps=3", "loop.min_buffer=2", "method.period=2", "tag=t",
            "dataset.channels=2"]
    s = run(build("ettm1", "periodic", "analytic_45nm", sets, backbone=backbone), str(tmp_path), verbose=False)
    assert s["n_steps"] == 6 and s["n_updates"] > 0 and s["adapt_j"] > 0 and np.isfinite(s["mase"])
    assert np.isfinite(s["retention_mse"])


def test_electricity_at_stride_48_records_its_true_length():
    from datasets.streams import build_stream
    cfg = build("electricity", "frozen", "analytic_45nm", backbone="ttm")
    n = len(build_stream(cfg["dataset"], 512, 48))
    assert 400 < n < 800


def test_report_budget_rows_per_backbone():
    import pandas as pd
    from analysis.report_backbones import budget_rows
    m = pd.DataFrame([
        dict(dataset="ettm1", tag="frozen", family="frozen", mase=1.0, adapt_j=0.0),
        dict(dataset="ettm1", tag="periodic_p1", family="periodic", mase=0.5, adapt_j=100.0),
        dict(dataset="ettm1", tag="periodic_p16", family="periodic", mase=0.9, adapt_j=6.0),
        dict(dataset="ettm1", tag="vou_l1", family="vou", mase=0.8, adapt_j=9.0),
        dict(dataset="ettm1", tag="vou_l0.1", family="vou", mase=0.6, adapt_j=30.0),
    ])
    rows = budget_rows({"A": m}, budgets=(0.05, 0.10, 0.25), datasets=("ettm1",), fams=("periodic", "vou"))
    assert np.allclose(rows["A"]["ettm1"]["periodic"], [np.nan, -10.0, -10.0], equal_nan=True)
    assert np.allclose(rows["A"]["ettm1"]["vou"], [np.nan, -20.0, -20.0], equal_nan=True)
    assert rows["A"]["ettm1"]["always"] == -50.0


def test_budget_table_uses_each_streams_true_length():
    import pandas as pd
    from analysis import analyze
    m = pd.DataFrame([
        dict(dataset="electricity", tag="frozen", family="frozen", mase=1.0, adapt_j=0.0, j_per_update=0.0,
             infer_j=4.95, n_steps=495),
        dict(dataset="electricity", tag="periodic_p1", family="periodic", mase=0.9, adapt_j=100.0, j_per_update=0.2,
             infer_j=4.95, n_steps=495),
    ])
    row = analyze.budget_table(m).set_index("dataset").loc["electricity"]
    assert np.isclose(row["update_over_infer"], 0.2 / (4.95 / 495))


# ---------------------------------------------------------------- minor fixes after the final review

def test_report_per_call_rows_use_each_runs_true_length(tmp_path, monkeypatch):
    import json
    import os
    from analysis import report_measured
    for d in ("synthetic", "ettm1", "electricity", "solar"):
        p = tmp_path / "results/main" / d / "vou_l1" / "seed0"
        p.mkdir(parents=True)
        json.dump(dict(mase=1.0, adapt_j=1.0, n_updates=1, j_per_update=0.5, infer_j=4.0, s_per_update=0.1,
                       probe_j=2.0, n_steps=400, retention_mse=0.1), open(p / "summary.json", "w"))
    monkeypatch.chdir(tmp_path)
    row = report_measured.per_call_rows()["Estimated (45\\,nm model)"]["electricity"]
    assert np.isclose(row["infer"], 4.0 / 400) and np.isclose(row["probe"], 2.0 / 400)


def test_chronos_median_index_is_checked():
    from models.chronos_bolt import MEDIAN, ChronosBolt
    w = ChronosBolt("amazon/chronos-bolt-tiny")
    assert w.m.chronos_config.quantiles[MEDIAN] == 0.5
    w.m.chronos_config.quantiles = [0.1, 0.5, 0.9]
    with pytest.raises(AssertionError, match="median"):
        w._check_median()


def test_quantile_learner_buffer_loss_is_normalised_mse_of_the_median():
    from engine.energy import build_meter
    from engine.quantile_learner import QuantileLearner
    from models.chronos_bolt import MEDIAN, ChronosBolt
    cfg = build("ettm1", "periodic", "analytic_45nm", backbone="chronos")
    w = ChronosBolt(cfg["model"]["repo"])
    lr = QuantileLearner(w, cfg["learner"])
    lr.meter = build_meter(cfg["energy"], w, "cpu")
    x = _series(n=4).numpy()
    for _ in range(3):
        lr.predict(x[:, :512])
        lr.push(x[:, :512], x[:, 512:])
    inp, tgt = lr._batch()
    with torch.no_grad():
        med = w.head_out(w.run_blocks(inp, lr.first))[:, MEDIAN]
    assert np.isclose(lr.buffer_loss(), float(torch.mean((med - tgt) ** 2)), rtol=1e-5)


def test_chronos_decoder_builds_no_cache():
    from models.chronos_bolt import ChronosBolt
    w = ChronosBolt("amazon/chronos-bolt-tiny")
    seen = {}
    orig = w.m.decoder.forward
    def spy(*a, **k):
        seen.update(k)
        return orig(*a, **k)
    w.m.decoder.forward = spy
    with torch.no_grad():
        w.forward_norm(w.revin(_series()[:, :512])[0])
    assert seen.get("use_cache") is False


@pytest.mark.parametrize("backbone", ["ttm", "chronos"])
def test_prefix_cache_gives_the_same_update(backbone):
    from engine.energy import build_meter
    from run import build_model
    cfg = build("ettm1", "periodic", "analytic_45nm", backbone=backbone)
    x = _series(n=4).numpy()
    states = []
    for cache in (True, False):
        torch.manual_seed(0)
        model, cls = build_model(cfg)
        lr = cls(model, {**cfg["learner"], "cache_prefix": cache})
        lr.meter = build_meter(cfg["energy"], model, "cpu")
        for _ in range(3):
            lr.predict(x[:, :512])
            lr.push(x[:, :512], x[:, 512:])
        lr.update()
        states.append([p.detach().clone() for p in lr.params])
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(*states))   # no-grad vs grad prefix: ~1e-6 float noise


def test_t5_cross_attention_backward_prices_only_the_products_it_needs():
    import os
    from transformers import T5Config
    from transformers.models.t5.modeling_t5 import T5Attention
    from engine.energy import AnalyticMeter
    from utils.config import ROOT, load_yaml
    cfg = load_yaml(os.path.join(ROOT, "energy", "analytic_45nm.yaml"))
    att = T5Attention(T5Config(d_model=8, d_kv=4, num_heads=2), has_relative_attention_bias=False)
    for p in att.parameters():
        p.requires_grad_(False)
    att.q.weight.requires_grad_(True)              # only q trains: dP and dQ, no dK or dV
    meter = AnalyticMeter(att, cfg)
    x, kv = torch.randn(3, 5, 8), torch.randn(3, 7, 8)   # neither input needs a gradient
    meter.active = True
    out = att(x, key_value_states=kv, query_length=5)[0]
    before = meter.macs
    out.sum().backward()
    rows_q, rows_o = 3 * 5, 3 * 5
    linear = rows_o * 8 * 8 + rows_q * 8 * 8            # o: grad to its input only; q: weight gradient only
    assert meter.macs - before == linear + 2 * 3 * 2 * 5 * 7 * 4


def test_forecasting_report_ignores_vision_runs_in_the_same_results_tree():
    import pandas as pd
    from analysis import analyze
    df = pd.DataFrame([dict(dataset="ettm1", tag="frozen"), dict(dataset="core50", tag="frozen"),
                       dict(dataset="solar", tag="vou_l1")])
    assert list(analyze.forecasting_runs(df).dataset) == ["ettm1", "solar"]
