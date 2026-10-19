import pandas as pd
import pytest
import json
import yaml

from sweep import expand


def test_expand_uses_out_override_for_skip_check(tmp_path):
    spec = {"out": "results/never", "datasets": ["ettm1"], "seeds": [0],
            "runs": [{"method": "frozen", "tag": "frozen"}]}
    done = tmp_path / "ettm1" / "frozen" / "seed0"
    done.mkdir(parents=True)
    (done / "summary.json").write_text("{}")
    assert expand(spec, out=str(tmp_path)) == []
    assert len(expand(spec)) == 1


def test_measured_sweep_grid():
    spec = yaml.safe_load(open("configs/sweeps/measured.yaml"))
    tags = [r["tag"] for r in spec["runs"]]
    assert spec["seeds"] == [0] and len(spec["datasets"]) == 4
    assert len(tags) == 11 and not any(t.startswith("oracle") for t in tags)


def test_measured_vou_runs_use_the_same_lambdas_as_the_analytic_sweep():
    """Measured runs rescale prices by a measured price scale; lam values stay the analytic ones."""
    spec = yaml.safe_load(open("configs/sweeps/measured.yaml"))
    vou = [r for r in spec["runs"] if r["method"] == "vou"]
    assert vou and all(set(r["set"]) == {"method.lam"} for r in vou)


def test_full_measured_grid_is_the_main_grid_without_unsupported_policies():
    """Oracle previews would be charged as decisions; vou_err prices in raw joules. Everything else matches main."""
    main = yaml.safe_load(open("configs/sweeps/main.yaml"))
    full = yaml.safe_load(open("configs/sweeps/measured_full.yaml"))
    keep = [r for r in main["runs"] if r["method"] not in ("oracle", "vou_err")]
    assert full["runs"] == keep and full["seeds"] == [0, 1, 2] and full["datasets"] == main["datasets"]


def test_per_call_grid_is_one_frozen_run_per_stream():
    spec = yaml.safe_load(open("configs/sweeps/measured_percall.yaml"))
    assert [r["tag"] for r in spec["runs"]] == ["frozen"] and len(spec["datasets"]) == 4


def test_gated_vou_dev_grid_runs_only_on_dev_streams_with_unique_tags():
    from methods import POLICIES
    spec = yaml.safe_load(open("configs/sweeps/dev_cheap.yaml"))
    assert set(spec["datasets"]) == {"synthetic_dev", "ettm1_dev"}
    tags = [r["tag"] for r in spec["runs"]]
    assert len(tags) == len(set(tags)) == 30
    assert all(r["method"] == "vou_gated" for r in spec["runs"])
    assert all(r["method"] in POLICIES for r in spec["runs"])


def test_gated_evaluation_grid_uses_dev_chosen_values_on_held_out_streams():
    spec = yaml.safe_load(open("configs/sweeps/gated.yaml"))
    dev = {r["tag"] for r in yaml.safe_load(open("configs/sweeps/dev_cheap.yaml"))["runs"]}
    assert spec["datasets"] == ["synthetic", "ettm1", "electricity", "solar"] and spec["seeds"] == [0, 1, 2]
    tags = [r["tag"] for r in spec["runs"]]
    assert len(tags) == 20 and set(tags) <= dev                     # nothing untried on dev streams


def test_measured_gated_grid_is_the_evaluation_grid_plus_references():
    g = yaml.safe_load(open("configs/sweeps/gated.yaml"))
    m = yaml.safe_load(open("configs/sweeps/measured_gated.yaml"))
    assert m["runs"][2:] == g["runs"] and [r["tag"] for r in m["runs"][:2]] == ["frozen", "periodic_p1"]
    assert m["seeds"] == [0] and m["datasets"] == g["datasets"]


def test_core50_dev_grid_runs_only_on_the_dev_stream():
    spec = yaml.safe_load(open("configs/sweeps/core50_dev.yaml"))
    assert spec["datasets"] == ["core50_dev"] and spec["out"] == "results/dev"
    fams = {r["method"] for r in spec["runs"]}
    assert fams == {"frozen", "periodic", "drift", "vou", "vou_gated"}
    assert len({r["tag"] for r in spec["runs"]}) == len(spec["runs"])


def test_core50_evaluation_grid_uses_only_dev_tried_configurations():
    dev = {r["tag"] for r in yaml.safe_load(open("configs/sweeps/core50_dev.yaml"))["runs"]}
    ev = yaml.safe_load(open("configs/sweeps/core50.yaml"))
    assert ev["datasets"] == ["core50"] and ev["seeds"] == [0, 1, 2]
    assert {r["tag"] for r in ev["runs"]} <= dev


def test_core50_measured_grids_match_the_evaluation_grid():
    ev = yaml.safe_load(open("configs/sweeps/core50.yaml"))
    me = yaml.safe_load(open("configs/sweeps/core50_measured.yaml"))
    assert {r["tag"] for r in ev["runs"]} | {"periodic_p1"} == {r["tag"] for r in me["runs"]}
    pc = yaml.safe_load(open("configs/sweeps/core50_percall.yaml"))
    assert [r["tag"] for r in pc["runs"]] == ["frozen"]


def test_expand_can_restrict_seeds_for_sharding(tmp_path):
    spec = {"out": str(tmp_path), "datasets": ["ettm1"], "seeds": [0, 1, 2],
            "runs": [{"method": "frozen", "tag": "frozen"}]}
    assert [j[2] for j in expand(spec, seeds=[1, 2])] == [1, 2]


@pytest.mark.parametrize("b", ["ttm", "chronos"])
def test_new_backbone_eval_grids_use_only_their_dev_configurations(b):
    dev = {(r["method"], json.dumps(r.get("set", {}), sort_keys=True))
           for r in yaml.safe_load(open(f"configs/sweeps/dev_{b}.yaml"))["runs"]}
    for name in (f"main_{b}", f"gated_{b}"):
        spec = yaml.safe_load(open(f"configs/sweeps/{name}.yaml"))
        assert spec["backbone"] == b and spec["datasets"] == ["synthetic", "ettm1", "electricity", "solar"]
        for r in spec["runs"]:
            assert (r["method"], json.dumps(r.get("set", {}), sort_keys=True)) in dev, r["tag"]


BASELINE_DEV = {"patchmixer": ("dev_baselines", ["synthetic_dev", "ettm1_dev"], "results/dev"),
                "ttm": ("dev_baselines_ttm", ["synthetic_dev", "ettm1_dev"], "results/dev_ttm"),
                "chronos": ("dev_baselines_chronos", ["synthetic_dev", "ettm1_dev"], "results/dev_chronos"),
                "core50": ("core50_dev_baselines", ["core50_dev"], "results/dev")}


@pytest.mark.parametrize("b", list(BASELINE_DEV))
def test_baseline_dev_grids_run_only_on_dev_streams_with_unique_family_tags(b):
    name, datasets, out = BASELINE_DEV[b]
    spec = yaml.safe_load(open(f"configs/sweeps/{name}.yaml"))
    assert spec["datasets"] == datasets and spec["out"] == out
    tags = [r["tag"] for r in spec["runs"]]
    assert len(tags) == len(set(tags)) >= 32
    assert {t.split("_")[0] for t in tags} == {"carat", "caract", "upf", "d3a"}
    assert spec.get("backbone", "patchmixer") == ("patchmixer" if b == "core50" else b)


def test_pick_keeps_the_energy_band_and_drops_near_duplicate_neighbours():
    from scripts.pick_grid import pick
    rows = [dict(dataset=d, tag="periodic_p1", family="periodic", adapt_j=100.0, n_updates=100)
            for d in ("a", "b")]
    for tag, e, n in [("x_l1", 50, 50), ("x_l2", 49, 49), ("x_l3", 10, 10), ("x_l4", 1, 1), ("x_l5", 2.9, 3)]:
        rows += [dict(dataset=d, tag=tag, family="x", adapt_j=e, n_updates=n) for d in ("a", "b")]
    assert pick(pd.DataFrame(rows), "x") == ["x_l3", "x_l2"]   # l4, l5 under 3 %; l1 duplicates l2


def _rows(datasets, tags, periodic=True):
    rows = []
    for d in datasets:
        if periodic:
            rows.append(dict(dataset=d, tag="periodic_p1", family="periodic", adapt_j=100.0, n_updates=100))
        rows += [dict(dataset=d, tag=t, family="x", adapt_j=10.0, n_updates=10) for t in tags]
    return pd.DataFrame(rows)


def test_pick_fails_clearly_when_periodic_p1_is_missing_on_a_dataset():
    from scripts.pick_grid import pick
    m = pd.concat([_rows(["a"], ["x_l1"]), _rows(["b"], ["x_l1"], periodic=False)])
    with pytest.raises(ValueError, match="b"):
        pick(m, "x")


def test_pick_fails_clearly_when_a_tag_is_missing_on_some_datasets():
    from scripts.pick_grid import pick
    m = pd.concat([_rows(["a", "b"], ["x_l1"]), _rows(["a"], ["x_l2"], periodic=False)])
    with pytest.raises(ValueError, match=r"x_l2 missing on \['b'\]"):
        pick(m, "x")


def test_pick_grid_write_requires_out_and_eval_datasets(monkeypatch, capsys):
    from scripts import pick_grid
    monkeypatch.setattr("sys.argv", ["pick_grid.py", "r", "s.yaml", "--datasets", "a", "--write", "w.yaml"])
    with pytest.raises(SystemExit) as e:
        pick_grid.main()
    assert e.value.code == 2 and "--out" in capsys.readouterr().err


def test_write_eval_copies_only_kept_runs_and_sets_the_eval_fields(tmp_path):
    from scripts.pick_grid import write_eval
    runs = [{"method": "cara", "tag": f"x_l{i}", "set": {"method.lam": i}} for i in (1, 2, 3)]
    dev = tmp_path / "dev.yaml"
    dev.write_text(yaml.safe_dump({"out": "r_dev", "backbone": "ttm", "datasets": ["a_dev"], "seeds": [0], "runs": runs}))
    out = tmp_path / "ev.yaml"
    write_eval(str(dev), ["x_l1", "x_l3"], "r_eval", ["a", "b"], [0, 1, 2], str(out), "line one\nline two")
    text = out.read_text().splitlines()
    assert text[:2] == ["# line one", "# line two"]
    spec = yaml.safe_load(out.read_text())
    assert spec["runs"] == [runs[0], runs[2]]
    assert (spec["out"], spec["datasets"], spec["seeds"], spec["backbone"]) == ("r_eval", ["a", "b"], [0, 1, 2], "ttm")
    dev.write_text(yaml.safe_dump({"out": "r_dev", "datasets": ["a_dev"], "seeds": [0], "runs": runs}))
    write_eval(str(dev), ["x_l2"], "r_eval", ["a"], [0], str(out), "h")
    assert "backbone" not in yaml.safe_load(out.read_text())


BASELINE_EVAL = {"baselines": ("dev_baselines", "results/baselines", ["synthetic", "ettm1", "electricity", "solar"]),
                 "baselines_ttm": ("dev_baselines_ttm", "results/baselines_ttm", ["synthetic", "ettm1", "electricity", "solar"]),
                 "baselines_chronos": ("dev_baselines_chronos", "results/baselines_chronos", ["synthetic", "ettm1", "electricity", "solar"]),
                 "core50_baselines": ("core50_dev_baselines", "results/baselines", ["core50"])}


@pytest.mark.parametrize("name", list(BASELINE_EVAL))
def test_baseline_eval_grids_use_only_their_dev_configurations(name):
    dev_name, out, datasets = BASELINE_EVAL[name]
    key = lambda r: (r["method"], json.dumps(r.get("set", {}), sort_keys=True))
    dev = {key(r) for r in yaml.safe_load(open(f"configs/sweeps/{dev_name}.yaml"))["runs"]}
    spec = yaml.safe_load(open(f"configs/sweeps/{name}.yaml"))
    assert spec["out"] == out and spec["datasets"] == datasets and spec["seeds"] == [0, 1, 2]
    for r in spec["runs"]:
        if r["tag"].startswith("replay_"):
            continue
        assert key(r) in dev, r["tag"]
    for fam in ("carat", "caract", "upf", "d3a"):
        assert any(r["tag"].startswith(fam + "_") for r in spec["runs"]), fam


def test_replay_uses_the_core50_periodic_periods_with_a_fixed_memory():
    periods = {r["set"]["method.period"] for r in yaml.safe_load(open("configs/sweeps/core50.yaml"))["runs"]
               if r["method"] == "periodic"}
    rep = [r for r in yaml.safe_load(open("configs/sweeps/core50_baselines.yaml"))["runs"]
           if r["tag"].startswith("replay_")]
    assert {r["set"]["method.period"] for r in rep} == periods
    assert all(r["set"]["learner.replay"] == {"size": 256, "frac": 0.5} for r in rep)


@pytest.mark.parametrize("name,ds", [("pacs_baselines", "pacs"), ("officehome_baselines", "officehome")])
def test_multidomain_baselines_run_the_core50_baseline_grid_unchanged(name, ds):
    core = {r["tag"]: r for r in yaml.safe_load(open("configs/sweeps/core50_baselines.yaml"))["runs"]}
    spec = yaml.safe_load(open(f"configs/sweeps/{name}.yaml"))
    assert spec["datasets"] == [ds] and spec["out"] == "results/domains"
    assert {r["tag"] for r in spec["runs"]} == set(core)
    for r in spec["runs"]:
        s = dict(r["set"])
        assert s.pop("domain_eval") == {} and s.pop("device") == "cuda"
        assert s == core[r["tag"]].get("set", {})


@pytest.mark.parametrize("ds,extra", [
    ("core50", {}),
    ("core50_b3", {"checkpoint": "checkpoints/vit_tiny_core50_b3.pt",
                   "retention": "checkpoints/vit_tiny_retention_b3.npz"})])
def test_core50_domain_baselines_use_one_evaluated_setting_per_family(ds, extra):
    core = {r["tag"]: r for r in yaml.safe_load(open("configs/sweeps/core50_baselines.yaml"))["runs"]}
    spec = yaml.safe_load(open(f"configs/sweeps/{ds}_domains_baselines.yaml"))
    assert spec["datasets"] == [ds] and spec["out"] == "results/domains" and spec["seeds"] == [0, 1, 2]
    fams = [r["tag"].split("_")[0] for r in spec["runs"]]
    assert sorted(fams) == ["caract", "carat", "d3a", "replay"]   # UPF has no setting within 25 %
    for r in spec["runs"]:
        s = dict(r["set"])
        assert s.pop("domain_eval") == {"n": 640} and s.pop("device") == "cuda"
        for k, v in extra.items():
            assert s.pop(k) == v
        assert r["method"] == core[r["tag"]]["method"] and s == core[r["tag"]].get("set", {})
