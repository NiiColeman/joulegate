"""Baselines from prior work: CARA, UPF, D3A, and replay."""
import filecmp
import os
import subprocess
import sys
from collections import deque

import numpy as np
import pandas as pd
import pytest
import torch

from engine.learner import Learner
from engine.measured import scale_for
from methods import build_policy
from run import run
from utils.config import build


class FakeLearner:
    """Input statistics per step, for the policies that read them (UPF)."""

    def __init__(self, mus=None):
        self.mus = mus
        self.t = 0

    def at(self, t):
        self.t = t
        mu = 0.0 if self.mus is None else self.mus[t]
        self.last_mu = torch.tensor([[mu]])
        self.last_sd = torch.tensor([[1.0]])
        return self


def drive(policy, errs, cost=1.0, learner=None, start=0):
    """Run the hook sequence of engine/online.run_stream; return the update steps."""
    learner = learner or FakeLearner()
    ups = []
    for t, e in enumerate(errs):
        policy.observe(t, e)
        if t < start:
            continue
        if policy.should_update(t, {"learner": learner.at(t), "cost_est": cost, "charge": lambda j: None}):
            policy.after_update(t, cost)
            ups.append(t)
    return ups


class ScriptedLoss(FakeLearner):
    """item_losses returns scripted values: `newest[t]` for the newest window, and the next entry of
    `bufs` for the training buffer or the base data (one per call); an entry is a loss (one item at
    feature 0) or a (losses, features) pair. Every call costs 0.5 J."""

    def __init__(self, newest, bufs):
        super().__init__()
        self.newest, self.bufs, self.calls = newest, list(bufs), []

    def item_losses(self, which):
        self.calls.append({"newest": True, "buffer": False, "base": "base"}[which])
        if which == "newest":
            v = self.newest[self.t]
            return (np.array([v[0]]), np.array([v[1]], float), 0.5) if isinstance(v, tuple) else \
                (np.array([v]), np.zeros((1, 1)), 0.5)
        v = self.bufs.pop(0)
        return (np.array(v[0], float), np.array(v[1], float), 0.5) if isinstance(v, tuple) else \
            (np.array([v]), np.zeros((1, 1)), 0.5)


def cara(variant, lam, **kw):
    cfg = build("ettm1", "cara", "analytic_45nm", [f"method.variant={variant}", f"method.lam={lam}"])["method"]
    cfg.update(kw)
    return build_policy(cfg)


def run_cara(policy, newest, bufs, cost=1.0, charged=None):
    """Hook order of run_stream: should_update, then after_update on an update (probe charged after)."""
    learner = ScriptedLoss(newest, bufs)
    charge = (lambda j: charged.append(j)) if charged is not None else (lambda j: None)
    ups = []
    for t in range(len(newest)):
        policy.observe(t, 0.0)
        if policy.should_update(t, {"learner": learner.at(t), "cost_est": cost, "charge": charge}):
            policy.after_update(t, cost)
            ups.append(t)
    return ups


def test_cara_threshold_updates_when_the_newest_window_is_stale_by_more_than_the_price():
    newest = [1.0] * 10 + [1.6] + [1.0] * 5
    assert run_cara(cara("threshold", 0.5), newest, [1.0, 1.0]) == [10]     # ref 1.0, staleness 0.6
    assert run_cara(cara("threshold", 0.7), newest, [1.0]) == []


def test_cara_cumulative_sums_staleness_since_the_last_update_and_resets():
    newest = [1.0] * 6 + [1.2] * 10                  # staleness 0.2 per step from step 6
    ups = run_cara(cara("cumulative", 0.5), newest, [1.0] * 10)
    assert ups[0] == 8                               # 0.2 + 0.2 + 0.2 >= 0.5 at the third stale step
    assert all(b - a >= 3 for a, b in zip(ups, ups[1:]))      # the sum restarted at 0 after each update


def test_cara_cumulative_is_signed():
    newest = [1.0] * 6 + [0.5] * 4 + [1.5] * 4       # -0.5 x 4 then +0.5 x 4: the sum goes back to 0
    assert run_cara(cara("cumulative", 1.0), newest, [1.0] * 5) == []


def test_cara_does_not_lock_out_after_an_update_made_mid_shift():
    # the old reference would be the post-update error 3.0 (shift still in progress); here the reference is
    # the adapter's loss on the fresh training buffer (1.0), so a later rise to 2.0 still fires.
    newest = [1.0] * 5 + [3.0] + [2.0] * 3
    ups = run_cara(cara("threshold", 0.5), newest, [1.0] * 10)
    assert ups[0] == 5 and len(ups) >= 2


def test_cara_reference_is_the_base_data_loss_for_the_pretrained_model_then_the_training_buffer():
    p = cara("threshold", 0.5)                                      # min_steps 4
    learner = ScriptedLoss([1.0] * 4 + [9.0, 1.0], [2.0, 0.7])
    ctx = lambda t: {"learner": learner.at(t), "cost_est": 1.0, "charge": lambda j: None}
    assert p.should_update(0, ctx(0)) is False
    assert learner.calls == ["base", True] and p.ref == 2.0       # pretrained: loss on its own base data
    assert not any(p.should_update(t, ctx(t)) for t in range(1, 4))
    assert p.should_update(4, ctx(4)) is True
    p.after_update(4, 1.0)
    assert learner.calls[-1] is False                             # after an update: the buffer it trained on
    p.should_update(5, ctx(5))
    assert p.ref == 0.7
    learner = ScriptedLoss([1.0], [])
    p2 = cara("threshold", 0.5)
    p2.should_update(0, {"learner": learner.at(0), "cost_est": None, "charge": lambda j: None})
    assert learner.calls == [] and p2.ref is None                  # no price: no probe, no decision


def test_cara_price_scales_with_update_joules_and_measured_scale():
    newest = [1.0] * 10 + [1.6] + [1.0] * 5
    assert run_cara(cara("threshold", 0.5), newest, [1.0], cost=2.0) == []      # 0.6 < 0.5 x 2
    p = cara("threshold", 0.5)
    p.price_scale = 4.0                                                          # price 0.5 x 2 / 4 = 0.25
    assert run_cara(p, newest, [1.0, 1.0], cost=2.0) == [10]


def test_cara_charges_every_item_losses_call():
    charged = []
    newest = [1.0] * 5 + [3.0] + [1.0] * 2
    learner_calls = []
    p = cara("threshold", 0.5)
    learner = ScriptedLoss(newest, [1.0, 1.0])
    for t in range(len(newest)):
        p.observe(t, 0.0)
        if p.should_update(t, {"learner": learner.at(t), "cost_est": 1.0, "charge": charged.append}):
            p.after_update(t, 1.0)
    assert len(charged) == len(learner.calls) and len(charged) > 0
    assert sum(charged) == pytest.approx(0.5 * len(learner.calls))


def test_cara_reference_weights_training_items_near_the_newest_window():
    # training buffer: one hard outlier far away (loss 5) and two typical windows near the query (0.5)
    buf = ([5.0, 0.5, 0.5], [[10.0], [0.0], [0.2]])
    newest = [(0.6, [0.1])] * 6
    p = cara("cumulative", 100.0)          # never fires here, so the sum keeps growing
    run_cara(p, newest, [buf] * 3)
    assert p.ref == pytest.approx(0.5)     # the uniform mean would be 2.0 and lock the signed sum out
    assert p.cum > 0


def test_cara_reference_is_uniform_when_all_items_are_equidistant():
    buf = ([1.0, 3.0], [[1.0], [-1.0]])
    p = cara("threshold", 0.5)
    run_cara(p, [(2.0, [0.0])] * 2, [buf])
    assert p.ref == pytest.approx(2.0)


def test_priced_policies_do_not_decide_before_the_cost_is_known():
    p = cara("threshold", 0.0)
    for t in range(8):
        p.observe(t, 1.0 + t)
        assert p.should_update(t, {"learner": ScriptedLoss([5.0] * 8, [1.0]).at(t), "cost_est": None,
                                   "charge": lambda j: None}) is False


def test_adapter_loss_is_finite_metered_and_leaves_the_parameters_alone():
    from tests.test_online import _setup
    _, learner, stream = _setup("periodic", ["method.period=4"])
    for t in range(6):
        ctx, y = stream[t]
        learner.predict(ctx)
        learner.push(ctx, y)
    before = [p.detach().clone() for p in learner.params]
    for newest in (True, False):
        loss, joules = learner.adapter_loss(newest)
        assert np.isfinite(loss) and joules > 0
    assert all(torch.equal(a, b) for a, b in zip(before, learner.params))


def test_base_loss_scores_a_fixed_sample_of_the_pretraining_windows_the_size_of_the_buffer():
    from tests.test_online import _setup
    cfg, learner, stream = _setup("periodic", ["method.period=4"])
    windows = np.load(cfg["retention_windows"])
    learner.set_base(lambda: windows, cfg["model"]["horizon"])
    for t in range(6):
        ctx, y = stream[t]
        learner.predict(ctx)
        learner.push(ctx, y)
    before = [p.detach().clone() for p in learner.params]
    a, ja = learner.adapter_loss(base=True)
    b, jb = learner.adapter_loss(base=True)
    assert np.isfinite(a) and a == b and ja > learner.adapter_loss(newest=True)[1] > 0
    assert all(torch.equal(x, y) for x, y in zip(before, learner.params))
    rows = learner._base_rows()
    assert len(rows) == min(len(windows), cfg["learner"]["buffer_steps"] * learner.buffer[-1][0].shape[0])


def test_item_losses_average_to_the_buffer_loss_and_carry_one_feature_per_item():
    from tests.test_online import _setup
    cfg, learner, stream = _setup("periodic", ["method.period=4"])
    learner.set_base(lambda: np.load(cfg["retention_windows"]), cfg["model"]["horizon"])
    for t in range(6):
        ctx, y = stream[t]
        learner.predict(ctx)
        learner.push(ctx, y)
    losses, feats, joules = learner.item_losses("buffer")
    assert len(losses) == len(learner.buffer) == len(feats) and feats.ndim == 2 and joules > 0
    assert losses.mean() == pytest.approx(learner.adapter_loss(False)[0], rel=1e-5)
    nl, nf, _ = learner.item_losses("newest")
    assert len(nl) == 1 and nf.shape[1] == feats.shape[1]
    bl, bf, _ = learner.item_losses("base")
    assert len(bl) == cfg["learner"]["buffer_steps"] and np.isfinite(bl).all()


def test_base_loss_without_base_data_is_an_error():
    from tests.test_online import _setup
    _, learner, _ = _setup("periodic", ["method.period=4"])
    with pytest.raises(RuntimeError):
        learner.adapter_loss(base=True)


def upf(lam, **kw):
    cfg = build("ettm1", "upf", "analytic_45nm", [f"method.lam={lam}"])["method"]
    cfg.update(kw)
    return build_policy(cfg)


def test_upf_warms_up_with_periodic_updates_before_it_has_a_history():
    ups = drive(upf(1e9), [1.0] * 40, start=3)
    assert ups == [8, 16, 24]                       # 3 warm-up models, then lam 1e9 never pays


def test_upf_updates_when_error_grows_with_adapter_age_and_not_on_a_flat_stream():
    rng = np.random.default_rng(0)
    flat = list(1.0 + 0.01 * rng.standard_normal(200))
    assert drive(upf(1.0), flat, start=3)[3:] == []                # nothing after warm-up
    p, errs, age = upf(1.0), [], 0
    learner = FakeLearner()
    for t in range(200):                            # error = 1 + 0.05 x steps since the last update
        errs.append(1.0 + 0.05 * age)
        p.observe(t, errs[-1])
        age += 1
        if t >= 3 and p.should_update(t, {"learner": learner.at(t), "cost_est": 1.0,
                                          "charge": lambda j: None}):
            p.after_update(t, 1.0)
            age = 0
    assert p.n_models > 6                           # keeps updating after warm-up


def test_upf_fit_is_finite_with_few_and_collinear_observations():
    p = upf(1.0, warmup_models=0)
    p.observe(3, 1.0)
    assert p.should_update(3, {"learner": FakeLearner().at(3), "cost_est": 1.0,
                               "charge": lambda j: None}) in (True, False)
    assert np.all(np.isfinite(p.w))


def test_upf_input_shift_feature_raises_the_cost_of_keeping():
    mus = [0.0] * 60 + [5.0] * 40
    learner = FakeLearner(mus)
    rng = np.random.default_rng(1)
    errs = [1.0 + 0.01 * rng.standard_normal() + (0.5 if t >= 60 else 0) for t in range(100)]
    ups = drive(upf(0.5), errs, learner=learner, start=3)
    assert any(60 <= u < 75 for u in ups)


def d3a(window, alpha, period=None):
    cfg = build("ettm1", "d3a", "analytic_45nm", [f"method.window={window}", f"method.alpha={alpha}"])["method"]
    cfg["restart_period"] = period
    return build_policy(cfg)


def test_d3a_alarms_on_a_rise_in_error_only():
    rng = np.random.default_rng(0)
    base = list(1.0 + 0.05 * rng.standard_normal(60))
    up = base + [2.0] * 20
    down = base + [0.2] * 20
    assert any(u >= 60 for u in drive(d3a(4, 3.0), up))
    assert [u for u in drive(d3a(4, 3.0), down) if u >= 60] == []


def test_d3a_needs_more_than_window_errors_since_the_last_update():
    ups = drive(d3a(16, -1e9), [1.0 + 0.01 * (t % 3) for t in range(80)])
    assert ups[0] == 16 and all(b - a == 17 for a, b in zip(ups, ups[1:]))


def test_d3a_constant_errors_never_alarm():
    assert drive(d3a(4, 0.0), [1.0] * 50) == []


def test_d3a_restart_period_forces_updates():
    assert drive(d3a(16, 1e9, period=10), [1.0] * 35) == [10, 20, 30]


def bare_learner(replay, buffer=4):
    """A Learner with only the replay state (no model): enough for _remember and _items."""
    lr = object.__new__(Learner)
    lr.replay = replay
    lr.buffer = deque(maxlen=buffer)
    lr.memory, lr.seen = [], 0
    lr.rng = np.random.default_rng(0)
    return lr


def feed(lr, n):
    for k in range(n):
        lr.buffer.append(k)
        lr._remember(k, 8)


def test_replay_off_uses_the_whole_fifo():
    lr = bare_learner(None)
    feed(lr, 10)
    assert lr._items() == [6, 7, 8, 9]


def test_replay_keeps_the_batch_size_with_newest_fifo_and_memory_draws():
    lr = bare_learner({"size": 100, "frac": 0.5})
    feed(lr, 10)
    items = lr._items()
    assert len(items) == 4 and items[:2] == [8, 9] and set(items[2:]) <= set(range(10))


def test_replay_with_a_small_memory_fills_from_the_fifo():
    lr = bare_learner({"size": 100, "frac": 0.5})
    lr.buffer.extend([0, 1, 2, 3])
    lr._remember(3, 8)                               # memory holds one item
    items = lr._items()
    assert len(items) == 4 and items[:3] == [1, 2, 3] and items[3] == 3


def test_reservoir_is_uniform_over_the_stream_and_charges_only_kept_writes():
    lr = bare_learner({"size": 50, "frac": 0.5})
    written = 0
    kept = 0
    for k in range(2000):
        before = list(lr.memory)
        written += lr._remember(k, 8)
        kept += lr.memory != before
    assert len(lr.memory) == 50 and np.mean(lr.memory) > 600     # not just the newest items
    assert written == 8 * kept and 50 < kept < 2000              # rejected items are not charged


def test_buffer_eval_does_not_consume_replay_draws():
    a, b = bare_learner({"size": 20, "frac": 0.5}), bare_learner({"size": 20, "frac": 0.5})
    feed(a, 30)
    feed(b, 30)
    b._items(replay=False)
    assert a._items() == b._items()


@pytest.mark.parametrize("method,sets", [("cara", ["method.variant=cumulative", "method.lam=0.01"]),
                                         ("cara", ["method.variant=threshold", "method.lam=0.01"]),
                                         ("upf", ["method.lam=0.01"]),
                                         ("d3a", ["method.window=4", "method.alpha=1"])])
def test_each_baseline_runs_a_forecasting_stream(tmp_path, method, sets):
    cfg = build("ettm1", method, "analytic_45nm", sets + ["dataset.n_steps=60", "seed=0", "tag=t"])
    summary = run(cfg, out_root=str(tmp_path), verbose=False)
    assert 0 < summary["n_updates"] < 60
    assert summary["probe_j"] > 0 if method == "cara" else summary["probe_j"] == 0.0


@pytest.mark.parametrize("method,sets", [("cara", ["method.lam=0.01"]), ("upf", ["method.lam=0.01"]),
                                         ("d3a", ["method.window=2", "method.alpha=0"]),
                                         ("periodic", ["method.period=1", "learner.replay={size: 4, frac: 0.5}"])])
def test_each_baseline_runs_a_vision_stream(tmp_path, method, sets):
    from tests.test_vision_run import _tiny_setup
    cfg = build("core50", method, "analytic_45nm", _tiny_setup(tmp_path) + sets)
    summary = run(cfg, out_root=str(tmp_path / "out"), verbose=False)
    # vision summaries carry no n_steps (run.py sets it for forecasting only), so count the saved trace rows
    trace = np.load(next((tmp_path / "out").rglob("trace.npz")))
    assert len(trace["err"]) == 6 and np.isfinite(summary["error"])


def test_replay_runs_reproduce_per_seed(tmp_path):
    from tests.test_vision_run import _tiny_setup
    sets = _tiny_setup(tmp_path) + ["method.period=1", "learner.replay={size: 4, frac: 0.5}"]
    a = run(build("core50", "periodic", "analytic_45nm", sets), out_root=str(tmp_path / "a"), verbose=False)
    b = run(build("core50", "periodic", "analytic_45nm", sets), out_root=str(tmp_path / "b"), verbose=False)
    assert a["error"] == b["error"] and a["adapt_j"] == b["adapt_j"]


def test_priced_baselines_require_a_measurable_price_and_d3a_does_not():
    for name in ("cara", "upf"):
        p = build_policy(build("ettm1", name, "analytic_45nm")["method"])
        with pytest.raises(RuntimeError):
            scale_for(p, 0.0, 1.0)
        assert scale_for(p, 2.0, 1.0) == 2.0
    assert scale_for(build_policy(build("ettm1", "d3a", "analytic_45nm")["method"]), 0.0, 1.0) is None


def test_family_styles_cover_every_baseline_with_validated_colours():
    from analysis.report_paper_vision import COLOR, LABEL, MARK
    for f in ("carat", "caract", "upf", "d3a", "replay"):
        assert f in COLOR and f in LABEL and f in MARK
    assert COLOR["caract"] == "#008300" and COLOR["upf"] == "#4a3aa7" and COLOR["d3a"] == "#e34948"
    assert COLOR["replay"] == COLOR["periodic"]


def test_budget_rows_take_the_family_list():
    from analysis.report_backbones import budget_rows
    rows = []
    for tag, fam, e, mase in [("frozen", "frozen", 0, 1.0), ("periodic_p1", "periodic", 100, 0.9),
                              ("upf_l1", "upf", 5, 0.8)]:
        rows.append(dict(dataset="ettm1", tag=tag, family=fam, adapt_j=e, mase=mase))
    out = budget_rows({"B": pd.DataFrame(rows)}, datasets=("ettm1",), fams=("upf",))
    assert out["B"]["ettm1"]["upf"] == [pytest.approx(-20.0)]


def test_core50_picks_are_fixed_and_upf_has_none():
    from analysis.report_paper_vision import picks
    for name in ("core50", "core50_b3"):
        p = picks(name)
        assert (p["carat"], p["caract"], p["d3a"], p["replay"], p["upf"]) == (
            "carat_l0.2", "caract_l0.3", "d3a_w4_a1", "replay_p16", None)


def test_existing_report_outputs_do_not_change_with_baseline_runs(tmp_path):
    """The baselines live in their own roots, so the main reports are byte-identical."""
    if not os.path.isdir("paper/tables") or not os.path.isdir("results/main"):
        pytest.skip("needs the local results and paper outputs")
    out = tmp_path / "new"
    out.mkdir()
    subprocess.run([sys.executable, "-m", "analysis.report_backbones", "--out", str(out)], check=True, capture_output=True)
    for f in ("tab_backbones_budget.tex", "tab_backbones_gpu.tex", "tab_backbones_percall.tex"):
        assert filecmp.cmp(out / "tables" / f, f"paper/tables/{f}", shallow=False), f


def test_quantile_buffer_loss_scores_the_fifo_buffer_without_replay():
    from engine.quantile_learner import QuantileLearner
    seen = []

    def batch(replay=True):
        seen.append(replay)
        raise StopIteration

    lr = object.__new__(QuantileLearner)
    lr._batch = batch
    with pytest.raises(StopIteration):
        lr.buffer_loss()
    assert seen == [False]
