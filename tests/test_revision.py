import yaml
import pytest

GROUPS = [(4, 1), (8, 2), (16, 4), (32, 8)]


def runs(name):
    return yaml.safe_load(open(f"configs/sweeps/{name}.yaml"))


@pytest.mark.parametrize("name", ["granularity", "granularity_measured"])
def test_granularity_matches_steps_and_exposure(name):
    spec = runs(name)
    assert spec["seeds"] == [0, 1, 2]
    assert spec["datasets"] == ["synthetic", "ettm1", "electricity", "solar"]
    lrs = {0.002, 0.0003} if name == "granularity" else {0.002}
    seen = set()
    for r in spec["runs"]:
        s = r["set"]
        assert s["learner.buffer_steps"] == 32 and s["learner.lr"] in lrs
        if r["method"] == "periodic" and not r["tag"].startswith("always"):
            p, k = s["method.period"], s["learner.steps_per_update"]
            assert (p, k) in GROUPS and 800 // p * k == 200 and 32 // p * k == 8
            seen.add((s["learner.lr"], p, k))
    assert seen == {(lr, p, k) for lr in lrs for p, k in GROUPS}
    tags = {r["tag"] for r in spec["runs"]}
    for lr in lrs:
        assert f"frozen_lr{lr}" in tags and f"always_lr{lr}" in tags


@pytest.mark.parametrize("name", ["conservative", "depth"])
def test_secondary_studies_have_three_seeds(name):
    assert runs(name)["seeds"] == [0, 1, 2]


# ---- E3: held-out confirmation -------------------------------------------------------------------

def test_confirm_pick_uses_the_declared_rule():
    import pandas as pd
    from scripts.confirm_picks import pick
    rows = []
    for d in ("a_dev", "b_dev"):
        rows += [dict(dataset=d, tag="periodic_p1", family="periodic", adapt_j=100.0, mase=1.0),
                 dict(dataset=d, tag="vou_l1", family="vou", adapt_j=9.0, mase=0.8),
                 dict(dataset=d, tag="vou_l2", family="vou", adapt_j=5.0, mase=0.9),
                 dict(dataset=d, tag="vou_l3", family="vou", adapt_j=20.0, mase=0.5)]
    m = pd.DataFrame(rows)
    assert pick(m, "vou") == "vou_l1"                     # <=10 % on every dev stream, lowest error
    m.loc[(m.dataset == "b_dev") & (m.tag == "vou_l1"), "adapt_j"] = 11.0
    assert pick(m, "vou") == "vou_l2"
    assert pick(m, "upf") is None


def test_confirm_pick_falls_back_to_25_percent_then_none():
    import pandas as pd
    from scripts.confirm_picks import pick
    rows = []
    for d in ("a_dev", "b_dev"):
        rows += [dict(dataset=d, tag="periodic_p1", family="periodic", adapt_j=100.0, mase=1.0),
                 dict(dataset=d, tag="upf_l1", family="upf", adapt_j=20.0, mase=0.7),
                 dict(dataset=d, tag="upf_l2", family="upf", adapt_j=24.0, mase=0.6),
                 dict(dataset=d, tag="d3a_w4_a1", family="d3a", adapt_j=40.0, mase=0.6)]
    m = pd.DataFrame(rows)
    assert pick(m, "upf") == "upf_l2"                     # nothing <=10 %: lowest error among <=25 %
    assert pick(m, "d3a") is None                         # nothing <=25 %


def test_confirm_pick_needs_every_dev_stream():
    import pandas as pd
    from scripts.confirm_picks import pick
    rows = [dict(dataset=d, tag="periodic_p1", family="periodic", adapt_j=100.0, mase=1.0) for d in ("a_dev", "b_dev")]
    rows.append(dict(dataset="a_dev", tag="vou_l1", family="vou", adapt_j=1.0, mase=0.1))   # absent on b_dev
    assert pick(pd.DataFrame(rows), "vou") is None


def test_confirm_frame_refuses_evaluation_streams():
    from scripts.confirm_picks import frame
    with pytest.raises(ValueError):
        frame("results/main", ["ettm1"])
    with pytest.raises(ValueError):
        frame(("results/dev",), ["synthetic_dev", "ettm1"])


def test_confirm_tag_table_rejects_conflicting_sets(tmp_path):
    from scripts.confirm_picks import tag_table
    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    a.write_text("runs:\n- {method: periodic, tag: t, set: {method.period: 2}}\n")
    b.write_text("runs:\n- {method: periodic, tag: t, set: {method.period: 4}}\n")
    with pytest.raises(ValueError):
        tag_table([str(a), str(b)])
    b.write_text("runs:\n- {method: periodic, tag: t, set: {method.period: 2}}\n")
    assert tag_table([str(a), str(b)])["t"]["set"] == {"method.period": 2}


def test_write_confirm_copies_runs_exactly(tmp_path):
    from scripts.confirm_picks import write_confirm
    dev = tmp_path / "dev.yaml"
    dev.write_text("backbone: ttm\nruns:\n"
                   "- {method: frozen, tag: frozen, set: {device: cuda}}\n"
                   "- {method: periodic, tag: periodic_p1, set: {method.period: 1, device: cuda}}\n"
                   "- {method: vou, tag: vou_l2, set: {method.lam: 2, device: cuda}}\n")
    out = tmp_path / "c.yaml"
    write_confirm({"frozen": "frozen", "always": "periodic_p1", "vou": "vou_l2", "upf": None},
                  [str(dev)], "results/x", ["ettm1"], {}, str(out), header="rule")
    import yaml as y
    s = y.safe_load(open(out))
    assert s["seeds"] == [3, 4, 5] and s["out"] == "results/x" and s["backbone"] == "ttm"
    assert [r["tag"] for r in s["runs"]] == ["frozen", "periodic_p1", "vou_l2"]
    assert s["runs"][2]["set"] == {"method.lam": 2, "device": "cuda"}


@pytest.mark.parametrize("name,out", [("confirm", "results/confirm"), ("confirm_ttm", "results/confirm_ttm"),
                                      ("confirm_chronos", "results/confirm_chronos"),
                                      ("confirm_core50", "results/confirm_core50")])
def test_confirm_specs_use_new_seeds_and_roots(name, out):
    spec = runs(name)
    assert spec["seeds"] == [3, 4, 5] and spec["out"] == out
    tags = {r["tag"] for r in spec["runs"]}
    assert {"frozen", "periodic_p1"} <= tags
    assert len({t.split("_")[0] for t in tags}) == len(tags) - 1   # one setting per family (+ always)
    assert all(not t.endswith("_dev") for t in spec["datasets"])


def test_confirm_shards_cover_every_spec_and_dataset():
    lines = [l.split() for l in open("slurm/shards/confirm.txt").read().splitlines() if l.strip()]
    assert len(lines) == 13
    for name, n in [("confirm", 4), ("confirm_ttm", 4), ("confirm_chronos", 4), ("confirm_core50", 1)]:
        ds = [d for s, d in lines if s == f"configs/sweeps/{name}.yaml"]
        assert ds == runs(name)["datasets"] and len(ds) == n
