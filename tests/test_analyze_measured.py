import json

import numpy as np
import pandas as pd
import pytest

from analysis.analyze_measured import event_energy, idle_table, improvement_per_joule, nonpositive


def _m():
    return pd.DataFrame([
        dict(dataset="ettm1", tag="frozen", family="frozen", mase=1.0, adapt_j=0.0, n_updates=0),
        dict(dataset="ettm1", tag="periodic_p1", family="periodic", mase=0.8, adapt_j=20.0, n_updates=800),
        dict(dataset="ettm1", tag="vou_l3", family="vou", mase=0.99, adapt_j=-0.2, n_updates=3),
    ])


def test_improvement_per_joule_skips_nonpositive_energy():
    t = improvement_per_joule(_m()).set_index("tag")
    assert t.loc["periodic_p1", "gain_per_j"] == pytest.approx(0.2 / 20.0)
    assert np.isnan(t.loc["vou_l3", "gain_per_j"])


def test_nonpositive_lists_adaptive_runs_only():
    assert list(nonpositive(_m()).tag) == ["vou_l3"]


def _write_run(root, tag, kinds, incr, p_idle):
    d = root / "ettm1" / tag / "seed0"
    d.mkdir(parents=True)
    np.savez(d / "events.npz", kind=np.array(kinds), parent=np.full(len(kinds), -1),
             incr_cpu_pkg=np.array(incr), incr_cpu_dram=np.zeros(len(kinds)))
    (d / "summary.json").write_text(json.dumps({"energy": {"p_idle_w": p_idle, "idle_drift_flag": False}}))


def test_event_energy_and_idle_table(tmp_path):
    _write_run(tmp_path, "periodic_p1", ["infer", "infer", "update"], [0.1, 0.3, 5.0], {"cpu_pkg": 100.0})
    e = event_energy(str(tmp_path)).set_index("kind")
    assert e.loc["infer", "mean_j"] == pytest.approx(0.2) and e.loc["update", "n"] == 1
    i = idle_table(str(tmp_path))
    assert i.iloc[0]["idle_w"] == 100.0


def test_positive_only_keeps_frozen_and_drops_noise_configs():
    """Frontier and budget table must use the same frame: a -0.2 J config is noise, not free."""
    from analysis.analyze_measured import positive_only
    assert list(positive_only(_m()).tag) == ["frozen", "periodic_p1"]


def _write_summary(root, tag, adapt_j, total_incr):
    d = root / "ettm1" / tag / "seed0"
    d.mkdir(parents=True)
    system = {"inference": {"incr": total_incr}, "decision": {"incr": 0.0}, "adaptation": {"incr": 0.0},
              "other": {"incr": 0.0}}
    (d / "summary.json").write_text(json.dumps({"adapt_j": adapt_j, "energy": {"system": system}}))


def test_frozen_difference_flags_window_attribution_that_disagrees_with_span_totals(tmp_path):
    """Span totals are exact even with a 100 ms counter; per-window attribution is not."""
    from analysis.analyze_measured import frozen_difference
    _write_summary(tmp_path, "frozen", 0.0, 100.0)
    _write_summary(tmp_path, "periodic_p1", 25.0, 130.0)   # span says 30 J extra: consistent
    _write_summary(tmp_path, "vou_l3", 10.0, 101.0)        # span says 1 J extra: windows smeared
    t = frozen_difference(str(tmp_path)).set_index("tag")
    assert t.loc["periodic_p1", "span_diff_j"] == pytest.approx(30.0)
    assert not t.loc["periodic_p1", "flag"] and t.loc["vou_l3", "flag"]
    assert "frozen" not in t.index


def test_per_call_table_reads_the_warm_up_windows(tmp_path):
    from analysis.analyze_measured import per_call_table
    d = tmp_path / "ettm1" / "frozen" / "seed0"
    d.mkdir(parents=True)
    pc = {k: {"j": v, "calls": 10, "window_s": 2.0} for k, v in (("infer", 0.01), ("probe", 0.03), ("update", 0.3))}
    (d / "summary.json").write_text(json.dumps({"energy": {"per_call": pc}}))
    t = per_call_table(str(tmp_path)).set_index("kind")
    assert t.loc["probe", "j"] == 0.03 and t.loc["update", "calls"] == 10
    assert t.loc["probe", "per_update"] == pytest.approx(0.1)


def _write_percall(root, dataset, update, probe):
    d = root / dataset / "frozen" / "seed0"
    d.mkdir(parents=True)
    pc = {"update": {"j": update, "calls": 20, "window_s": 2.0}, "probe": {"j": probe, "calls": 200, "window_s": 2.0},
          "infer": {"j": 0.03, "calls": 200, "window_s": 2.0}}
    (d / "summary.json").write_text(json.dumps({"energy": {"per_call": pc}}))


def test_projected_energy_prices_exact_counts_with_measured_per_call_costs(tmp_path):
    """Sparse GPU runs sit below the whole-run noise floor; counts x measured per-call costs do not."""
    from analysis.analyze_measured import per_call_costs, projected_runs
    _write_percall(tmp_path / "pc", "ettm1", update=0.3, probe=0.03)
    costs = per_call_costs(str(tmp_path / "pc"))
    assert costs == {"ettm1": {"update": 0.3, "probe": 0.03, "infer": 0.03}}
    d = tmp_path / "res" / "ettm1" / "vougerr_t0.2_l1" / "seed0"
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps({"mase": 0.9, "n_updates": 10, "calib_j": 0.02}))
    np.savez(d / "trace.npz", probe_j=np.r_[np.full(40, 0.001), np.zeros(60)])
    t = projected_runs(str(tmp_path / "res"), costs)
    r = t.iloc[0]
    assert (r.updates, r.probes, r.calib) == (10, 40, 1)
    assert r.adapt_j == pytest.approx((10 + 1) * 0.3 + 40 * 0.03)


def test_projected_runs_count_probes_from_events_in_measured_runs(tmp_path):
    """Measured traces hold provisional probe joules, which can be 0 or negative for a 1 ms probe;
    the event log is the reliable count."""
    from analysis.analyze_measured import projected_runs
    d = tmp_path / "ettm1" / "vou_l1" / "seed0"
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps({"mase": 0.9, "n_updates": 2, "calib_j": 0.0}))
    np.savez(d / "trace.npz", probe_j=np.r_[np.full(3, 0.01), np.zeros(5), np.full(2, -0.01)])
    # 500 warm-up probes (step -1, the per-call window) must not count; 10 in-loop probes must
    kind = np.array(["probe"] * 500 + ["probe"] * 10 + ["update"] * 2)
    step = np.r_[np.full(500, -1), np.arange(10), [3, 7]]
    np.savez(d / "events.npz", kind=kind, step=step, parent=np.r_[np.zeros(510), -np.ones(2)])
    r = projected_runs(str(tmp_path), {"ettm1": {"update": 1.0, "probe": 0.1, "infer": 0.01}}).iloc[0]
    assert r.probes == 10 and r.adapt_j == pytest.approx(2 * 1.0 + 10 * 0.1)


def test_projected_runs_read_vision_summaries(tmp_path):
    """Vision summaries report `error` (misclassification rate), not `mase`."""
    from analysis.analyze_measured import projected_runs
    d = tmp_path / "core50" / "periodic_p4" / "seed0"
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps({"error": 0.08, "n_updates": 3, "calib_j": 0.0}))
    np.savez(d / "trace.npz", probe_j=np.zeros(10))
    r = projected_runs(str(tmp_path), {"core50": {"update": 2.0, "probe": 0.1, "infer": 0.5}}).iloc[0]
    assert r.mase == 0.08 and r.adapt_j == pytest.approx(6.0)
