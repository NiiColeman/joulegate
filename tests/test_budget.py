import json
import os

import pytest

import scripts.budget_sweeps as bs
import analysis.report_budget as rb
from analysis.analyze import load, seed_mean


def _summary(root, d, tag, seed, mase, n=10):
    p = os.path.join(root, d, tag, f"seed{seed}")
    os.makedirs(p, exist_ok=True)
    json.dump(dict(mase=mase, error=mase, adapt_j=1.0, n_updates=n, j_per_update=0.1, infer_j=1.0,
                   s_per_update=0.01, n_steps=100), open(os.path.join(p, "summary.json"), "w"))


def test_tags_round_trip_and_greedy_is_its_own_family():
    assert bs.suffix(0.06, 1) == "__b6c1" and bs.suffix(0.25, 10) == "__b25c10"
    assert bs.split_tag("vou_l10__b6c1") == ("vou_l10", "b6c1")
    assert bs.split_tag("frozen") == ("frozen", None)
    assert bs.family("periodic_p1") == "greedy" and bs.family("periodic_p16") == "periodic"


def test_budgeted_run_adds_the_relative_bucket_without_touching_the_source():
    src = {"method": "vou", "tag": "vou_l10", "set": {"method.lam": 10}}
    r = bs.budgeted(src, 0.03, 10)
    assert r["tag"] == "vou_l10__b3c10"
    assert r["set"] == {"method.lam": 10, "loop.budget": {"income_frac": 0.03, "capacity_updates": 10}}
    assert src["set"] == {"method.lam": 10}


def test_dev_specs_have_unique_tags_and_one_unbudgeted_frozen_run():
    for b in bs.BACKBONES:
        spec = bs.dev_spec(b)
        tags = [r["tag"] for r in spec["runs"]]
        assert len(tags) == len(set(tags))
        assert [t for t in tags if "__" not in t] == ["frozen"]
        assert spec["seeds"] == [0] and all(d.endswith("_dev") for d in spec["datasets"])
        assert sum(t.startswith("periodic_p1__") for t in tags) == len(bs.budgets())


def test_picks_use_mean_dev_error_per_budget_and_family(tmp_path):
    root = str(tmp_path)
    for d in ("a_dev", "b_dev"):
        _summary(root, d, "frozen", 0, 1.0)
        _summary(root, d, "vou_l1__b6c1", 0, 0.8 if d == "a_dev" else 0.9)    # mean 0.85
        _summary(root, d, "vou_l10__b6c1", 0, 0.7 if d == "a_dev" else 0.95)  # mean 0.825: picked
        _summary(root, d, "vou_l1__b1c1", 0, 0.9)
        _summary(root, d, "periodic_p1__b6c1", 0, 0.99)
    _summary(root, "a_dev", "drift_t1__b6c1", 0, 0.1)   # only on one dev stream: never picked
    picks = bs.picks_for(seed_mean(load(root)))
    assert picks["b6c1"]["vou"] == "vou_l10__b6c1"
    assert picks["b1c1"]["vou"] == "vou_l1__b1c1"
    assert picks["b6c1"]["greedy"] == "periodic_p1__b6c1"
    assert picks["b6c1"]["drift"] is None


def test_report_cells_and_winners(tmp_path):
    root = str(tmp_path / "r")
    for s in (3, 4, 5):
        for d in ("ettm1", "solar", "synthetic"):
            _summary(root, d, "frozen", s, 1.0)
            _summary(root, d, "vou_l1__b6c1", s, 0.9 if d != "solar" else 0.5)
            _summary(root, d, "periodic_p16__b6c1", s, 0.8)
    agg, pairs = rb.cells({"x": root})
    assert agg[("real", "b6c1", "periodic")]["value"] == pytest.approx(-20.0)
    assert agg[("real", "b6c1", "vou")]["value"] == pytest.approx(-30.0)   # mean of -10 and -50
    assert agg[("synthetic", "b6c1", "vou")]["n"] == 1
    w = rb.winners(pairs, "real")
    assert w["b6c1"] == {"periodic": 1, "vou": 1}
    rb.table(agg, str(tmp_path / "t.tex"))
    tex = open(tmp_path / "t.tex").read()
    assert r"$\mathbf{-30.0}$" in tex and "Greedy" not in tex
