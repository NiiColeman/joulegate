"""Tests for report_revision.py: pure helpers, and the section builders on tiny synthetic result trees."""
import json
import os

import numpy as np
import pytest
import yaml

import analysis.report_revision as rr

DS = ["synthetic", "ettm1", "electricity", "solar"]


def put(root, dataset, tag, seed, mase, adapt_j=0.0, total_j=None, n_updates=10):
    d = os.path.join(root, dataset, tag, f"seed{seed}")
    os.makedirs(d)
    s = {"mase": mase, "n_updates": n_updates, "adapt_j": adapt_j, "j_per_update": adapt_j / max(n_updates, 1),
         "infer_j": 1.0, "s_per_update": 0.1, "retention_mse": 0.0}
    if total_j is not None:
        s["energy"] = {"system": {"inference": {"raw": 1, "incr": 1.0}, "adaptation": {"raw": 1, "incr": adapt_j},
                                  "other": {"raw": 1, "incr": total_j - adapt_j - 1.0}}}
    json.dump(s, open(os.path.join(d, "summary.json"), "w"))


@pytest.fixture(autouse=True)
def opts(monkeypatch):
    monkeypatch.setitem(rr.OPTS, "strict", True)
    monkeypatch.setitem(rr.OPTS, "gran_seeds", (0, 1))
    monkeypatch.setitem(rr.OPTS, "confirm_seeds", (3, 4))
    monkeypatch.setitem(rr.OPTS, "seeds_n", 2)
    del rr.NOTES[:]


# ----------------------------------------------------------------------------- helpers

def test_mean_sd_cell_formats_and_handles_missing():
    assert rr.cell(-12.345, 1.26) == "$-12.3 \\pm 1.3$"
    assert rr.cell(float("nan"), 0.0) == "--"
    assert rr.cell(-3.0, float("nan")) == "$-3.0$"          # one seed: no spread


def test_recovery_window_means_after_each_switch():
    err = np.array([0.0] * 10 + [1.0] * 5 + [0.0] * 5)
    assert rr.after_switch(err, [10], 5) == pytest.approx(1.0)
    assert rr.after_switch(err, [10], 10) == pytest.approx(0.5)
    assert rr.after_switch(err, [10], 10, ends=[12]) == pytest.approx(1.0)   # window clipped at the next switch


def test_relative_change():
    assert rr.rel(0.9, 1.0) == pytest.approx(-10.0)


def test_domain_switch_steps_from_test_masks(tmp_path):
    root = tmp_path / "data"
    os.makedirs(root)
    for dom, n_test in (("a", 10), ("b", 20), ("c", 0)):      # 100 images each, batch 10
        m = np.zeros(100, dtype=bool)
        m[:n_test] = True
        np.save(root / f"{dom}_test.npy", m)
    cfgs = tmp_path / "cfg"
    os.makedirs(cfgs)
    yaml.safe_dump({"root": "unused", "domains": ["a", "b", "c"], "batch": 10}, open(cfgs / "toy.yaml", "w"))
    starts, total = rr.domain_switch_steps("toy", data_root=str(root), cfg_dir=str(cfgs))
    assert list(starts) == [9, 17] and total == 27
    assert rr.domain_switch_steps("toy", data_root=str(tmp_path / "missing"), cfg_dir=str(cfgs)) is None


@pytest.mark.skipif(not os.path.exists("data/pacs/sketch_test.npy"), reason="PACS arrays not on this machine")
def test_pacs_switches_match_the_recorded_trace_length():
    trace = "results/domains/pacs/periodic_p1/seed0/trace.npz"
    if not os.path.exists(trace):
        pytest.skip("needs the local PACS results")
    starts, total = rr.domain_switch_steps("pacs")
    t = np.load(trace)["err"]
    assert total == len(t) and len(starts) == 2


# ----------------------------------------------------------------------------- granularity

def build_gran(root, skip=None):
    """frozen 1.0; periodic groups improve with larger k; always best. 2 seeds, 4 streams, 2 lrs."""
    for lr in rr.LRS:
        for d in DS:
            for seed in (0, 1):
                put(root, d, f"frozen_lr{lr}", seed, 1.0)
                put(root, d, f"always_lr{lr}", seed, 0.8 + 0.01 * seed, adapt_j=100.0, n_updates=800)
                for i, (p, k) in enumerate(rr.GROUPS):
                    if (lr, d, p) == skip and seed == 1:
                        continue
                    put(root, d, f"gran_lr{lr}_p{p}_k{k}", seed, 0.9 - 0.01 * i + 0.02 * seed, adapt_j=25.0, n_updates=800 // p)


def test_granularity_cells_are_per_seed_changes_against_frozen(tmp_path):
    root = str(tmp_path / "g")
    build_gran(root)
    cells = rr.gran_cells(rr.gran_runs(root))
    mu, sd = cells[(0.002, "ettm1", (4, 1))]["change"]       # mase 0.90 and 0.92 against frozen 1.0
    assert mu == pytest.approx(-9.0) and sd == pytest.approx(np.std([-10, -8], ddof=1))
    assert cells[(0.002, "ettm1", (4, 1))]["pct"] == pytest.approx(25.0)
    assert cells[(0.0003, "solar", "always")]["pct"] == pytest.approx(100.0)
    assert cells[(0.002, "solar", "frozen")]["change"] == (0.0, 0.0)


def test_granularity_section_writes_table_and_figure_with_measured_columns(tmp_path):
    root, meas = str(tmp_path / "g"), str(tmp_path / "gm")
    build_gran(root)
    for dev, scale in (("cpu", 10.0), ("gpu", 1.0)):
        for d in DS:
            for seed in (0, 1):
                put(f"{meas}/{dev}", d, "frozen_lr0.002", seed, 1.0, adapt_j=0.0, total_j=50 * scale)
                put(f"{meas}/{dev}", d, "gran_lr0.002_p8_k2", seed, 0.9, adapt_j=20 * scale, total_j=70 * scale)
    out = tmp_path / "out"
    os.makedirs(out)
    js = {}
    rr.section_granularity(str(out), {"gran": root, "gran_measured": meas}, None, js)
    tex = (out / "tables" / "tab_granularity.tex").read_text()
    assert (out / "figures" / "fig_granularity.pdf").exists()
    assert tex.count("\\\\") >= 4 * 6
    row = [l for l in tex.splitlines() if "100$\\times$2" in l][0]
    assert "200" in row and "20.0" in row and "70.0" in row         # measured CPU adapt 200, total 700; GPU 20, 70
    assert js["granularity"]["seeds"] == [0, 1]
    assert "gpu|lr0.002|synthetic|(8, 2)" in js["granularity"]["measured"]


def test_granularity_incomplete_tree_is_skipped_with_a_note(tmp_path, capsys):
    root = str(tmp_path / "g")
    build_gran(root, skip=(0.002, "solar", 16))
    out = tmp_path / "out"
    os.makedirs(out)
    rr.section_granularity(str(out), {"gran": root, "gran_measured": str(tmp_path / "none")}, None, {})
    assert not (out / "tables" / "tab_granularity.tex").exists()
    assert "incomplete" in capsys.readouterr().out
    rr.OPTS["strict"] = False
    rr.section_granularity(str(out), {"gran": root, "gran_measured": str(tmp_path / "none")}, None, {})
    assert (out / "tables" / "tab_granularity.tex").exists()


def test_missing_roots_skip_cleanly(tmp_path, capsys):
    out = str(tmp_path)
    none = {k: str(tmp_path / "no" / k) for k in rr.ROOTS}
    js = {}
    rr.section_granularity(out, none, None, js)
    rr.section_confirm(out, none, js)
    rr.section_protocol(out, none, js)
    rr.section_seeds(out, none, js)
    text = capsys.readouterr().out
    assert text.count("[skip]") >= 6 and js == {}
    assert os.listdir(out) == []


# ----------------------------------------------------------------------------- confirmation

def build_confirm(root, vision=False):
    for d in (["core50"] if vision else DS):
        for seed in (3, 4):
            put(root, d, "frozen", seed, 0.50 if vision else 1.0)
            put(root, d, "periodic_p1", seed, 0.30 if vision else 0.8, adapt_j=100.0, n_updates=800)
            put(root, d, "periodic_p16", seed, 0.44 if vision else 0.95 + 0.01 * seed, adapt_j=6.0)
            put(root, d, "vou_l1", seed, 0.40 if vision else 0.90 - 0.01 * seed, adapt_j=8.0)
            put(root, d, "drift_t0.2", seed, 0.45 if vision else 0.85, adapt_j=40.0)    # beyond 10 %: never marked best
            if vision:
                put(root, d, "replay_p16", seed, 0.42, adapt_j=9.0)


def test_confirm_cells_use_per_seed_changes_and_energy_share(tmp_path):
    root = str(tmp_path / "c")
    build_confirm(root)
    from analysis.analyze import load
    cells = rr.confirm_cells(load(root))
    c = cells["periodic"]["ettm1"]
    assert c["change"][0] == pytest.approx(-1.5) and c["pct"] == pytest.approx(6.0) and c["seeds"] == [3, 4]
    assert cells["always"]["ettm1"]["pct"] == pytest.approx(100.0)
    assert "upf" not in cells


def test_confirm_table_marks_best_realizable_and_renders_missing_families(tmp_path):
    out = tmp_path / "out"
    os.makedirs(out)
    build_confirm(str(tmp_path / "c"))
    build_confirm(str(tmp_path / "v"), vision=True)
    roots = {k: str(tmp_path / "none") for k in rr.ROOTS}
    roots.update(confirm=str(tmp_path / "c"), confirm_core50=str(tmp_path / "v"))
    js = {}
    rr.section_confirm(str(out), roots, js)
    tex = (out / "tables" / "tab_confirm.tex").read_text()
    vou = [l for l in tex.splitlines() if l.strip().startswith("VoU") or " VoU " in l][0]
    per = [l for l in tex.splitlines() if "Periodic &" in l and "replay" not in l][0]
    assert "\\mathbf" in vou and "\\mathbf" not in per            # vou -13.5 beats periodic -1.5; drift is >10 % energy
    drift = [l for l in tex.splitlines() if "Drift" in l][0]
    assert "\\mathbf" not in drift
    assert "accuracy gain" in tex and "Periodic + replay" in tex
    # a family missing in a block, and the whole TTM block, are absent without error
    assert "UPF" not in tex and "TTM" not in tex
    gain = js["confirm"]["ViT-Tiny"]["vou"]["core50"]["gain_pts_mean"]
    assert gain == pytest.approx(10.0)
    assert any("results/confirm_ttm" in n or "none" in n for n in rr.NOTES)


def test_confirm_incomplete_seeds_skip(tmp_path):
    out = tmp_path / "out"
    os.makedirs(out)
    build_confirm(str(tmp_path / "c"))
    import shutil
    shutil.rmtree(tmp_path / "c" / "solar" / "vou_l1" / "seed4")
    roots = {k: str(tmp_path / "none") for k in rr.ROOTS}
    roots["confirm"] = str(tmp_path / "c")
    rr.section_confirm(str(out), roots, {})
    assert not (out / "tables" / "tab_confirm.tex").exists() and any("incomplete" in n for n in rr.NOTES)


# ----------------------------------------------------------------------------- seeds, protocol

def test_seeds_table_mean_sd_over_seeds(tmp_path):
    out = tmp_path / "out"
    os.makedirs(out)
    cons, depth, main = (str(tmp_path / n) for n in ("cons", "depth", "main"))
    for d in DS:
        for seed in (0, 1):
            put(cons, d, "frozen", seed, 1.0)
            put(cons, d, "periodic_p1", seed, 0.8, adapt_j=100.0)
            put(cons, d, "periodic_p4", seed, 0.9 + 0.02 * seed, adapt_j=25.0)
            put(cons, d, "periodic_p16", seed, 0.95, adapt_j=6.0)
            put(cons, d, "vou_l1", seed, 0.93, adapt_j=8.0)
            put(cons, d, "drift_t0.2", seed, 0.97, adapt_j=50.0)
            put(main, d, "frozen", seed, 1.0)
    for d in ("ettm1", "electricity"):
        for seed in (0, 1):
            for k in (0, 2):
                put(depth, d, f"depth_k{k}_cache", seed, 0.9 - 0.01 * k + 0.02 * seed, adapt_j=10.0)
    roots = {k: str(tmp_path / "none") for k in rr.ROOTS}
    roots.update(conservative=cons, depth=depth, main=main)
    js = {}
    rr.section_seeds(str(out), roots, js)
    tex = (out / "tables" / "tab_seeds.tex").read_text()
    p4 = [l for l in tex.splitlines() if l.startswith("Periodic, every 4")][0]
    assert "$-9.0 \\pm 1.4$" in p4                                   # changes -10 and -8
    assert "Drift trigger" in tex and "--" in [l for l in tex.splitlines() if l.startswith("Drift")][0]   # >10 % energy
    assert "2 adapted blocks" in tex and "Head only" in tex
    assert js["seeds"]["conservative"]["ettm1|periodic_p4"]["n"] == 2


def test_seeds_with_one_seed_show_means_without_spread(tmp_path):
    out = tmp_path / "out"
    os.makedirs(out)
    cons = str(tmp_path / "cons")
    for d in DS:
        put(cons, d, "frozen", 0, 1.0)
        put(cons, d, "periodic_p1", 0, 0.8, adapt_j=100.0)
    roots = {k: str(tmp_path / "none") for k in rr.ROOTS}
    roots["conservative"] = cons
    rr.OPTS["seeds_n"] = 1
    rr.section_seeds(str(out), roots, {})
    tex = (out / "tables" / "tab_seeds.tex").read_text()
    row = [l for l in tex.splitlines() if l.startswith("Update every step")][0]
    assert "$-20.0$" in row and "\\pm" not in row


def test_protocol_stats_and_table(tmp_path):
    root = tmp_path / "m"
    for dev, doms in (("cpu", ["cpu_pkg"]), ("gpu", ["gpu", "cpu_pkg"])):
        d = root / dev / "synthetic" / "frozen" / "seed0"
        os.makedirs(d)
        e = {"p_idle_pre_w": {k: 10.0 for k in doms}, "p_idle_post_w": {k: 11.0 for k in doms}, "idle_drift": 0.02,
             "idle_drift_flag": False, "closure": 1e-16, "span_s": 12.0,
             "system": {c: {"raw": 100.0, "incr": 10.0} for c in ("inference", "decision", "adaptation", "other")}}
        json.dump({"energy": e}, open(d / "summary.json", "w"))
        yaml.safe_dump({"energy": {"idle_s": 30, "counters": [{"name": "cpu_pkg", "type": "rapl", "sample_s": 0.0005}]}},
                       open(d / "config.yaml", "w"))
    s = rr.protocol_stats(str(root / "gpu"))
    assert s["runs"] == 1 and s["idle_pre"] == [20.0] and s["cats"]["adaptation"] == [100.0, 10.0]
    out = tmp_path / "out"
    os.makedirs(out)
    rr.protocol_table({"cpu": rr.protocol_stats(str(root / "cpu")), "gpu": s},
                      {"python": "3.x", "torch": "2.x", "cuda": "12", "driver": "n/a"}, str(out / "p.tex"))
    tex = (out / "p.tex").read_text()
    assert "10.0 / 10.0" not in tex and "0.10 / 0.01, 10\\%" in tex and "driver n/a" in tex


def test_main_spread_summarises_sd_of_each_familys_pick_within_budget(tmp_path):
    root = str(tmp_path / "r")
    for s, (fz, p1, p4, p16) in enumerate([(1.0, 0.8, 0.70, 0.90), (1.0, 0.8, 0.74, 0.92)]):
        put(root, "ettm1", "frozen", s, fz)
        put(root, "ettm1", "periodic_p1", s, p1, adapt_j=100.0)
        put(root, "ettm1", "periodic_p4", s, p4, adapt_j=25.0)      # over 10 %: not picked
        put(root, "ettm1", "periodic_p16", s, p16, adapt_j=6.0)     # picked: changes -10, -8
    out = rr.main_spread({"B": root}, fams=("periodic", "drift"))
    assert out["B"]["cells"] == {"ettm1|periodic": pytest.approx([-9.0, np.std([-10.0, -8.0], ddof=1)])}
    assert out["B"]["sd_max"] == pytest.approx(np.std([-10.0, -8.0], ddof=1))
