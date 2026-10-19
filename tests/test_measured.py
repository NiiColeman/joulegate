import json
import os
from multiprocessing import shared_memory

import numpy as np
import pytest

import engine.measured as measured
from engine.measured import price_scale, rolling_mase
from run import run
from utils.config import build


def test_rolling_mase_windows():
    err = np.arange(10, dtype=float)
    before, after = rolling_mase(err, np.array([0, 4, 9]), k=3)
    np.testing.assert_allclose(before, [0.0, 3.0, 8.0])
    np.testing.assert_allclose(after[:2], [2.0, 6.0])
    assert np.isnan(after[2])


def _cfg(method, sets=()):
    return build("ettm1", method, "fake", list(sets) + ["seed=0", "dataset.n_steps=30", "tag=m",
                                                        "loop.buffer_eval=true"])


def test_measured_run_writes_complete_outputs(tmp_path):
    # 3 s idle windows: the 6 J-quantized fake counter needs them for a ~1 % idle-power estimate
    s = run(_cfg("vou", ["method.lam=0.01", "energy.idle_s=3"]), out_root=str(tmp_path), verbose=False)
    d = tmp_path / "ettm1" / "m" / "seed0"
    for f in ("config.yaml", "summary.json", "trace.npz", "events.npz", "samples.npz"):
        assert (d / f).exists(), f
    ev = np.load(d / "events.npz")
    assert {"infer", "store", "decide", "probe", "update", "calib", "other"} <= set(ev["kind"])
    assert {"raw_fake_fast", "incr_fake_slow", "mase_before", "buf_after"} <= set(ev.files)
    e = s["energy"]
    assert e["closure"] < 1e-6
    assert set(e["system"]) == {"inference", "decision", "adaptation", "other"}
    assert s["adapt_j"] == pytest.approx(e["system"]["adaptation"]["incr"] + e["system"]["decision"]["incr"])
    # the fake counters add busy power for this process's CPU time, so work costs energy
    c = e["categories"]
    assert sum(c[k]["incr"]["fake_fast"] for k in ("adaptation", "decision")) > 0
    assert 0 < s["adapt_j"] < s["adapt_j_raw"]
    # the scale comes from one long window of back-to-back trial updates, not single 15 ms updates
    ecfg = build("ettm1", "frozen", "fake")["energy"]
    assert e["scale_window_s"] >= ecfg["scale_window_s"] and e["scale_updates"] >= ecfg["scale_min_updates"]
    assert e["price_scale"] == pytest.approx(e["scale_measured_j"] / e["scale_analytic_j"])
    assert e["price_scale"] > 0
    assert json.load(open(d / "summary.json"))["n_updates"] == s["n_updates"] > 0


def test_samples_cover_the_whole_post_run_idle_window(tmp_path):
    """Every counter must have refreshed after the idle window ends, or its tail reads as flat."""
    cfg = _cfg("periodic", ["method.period=4"])
    run(cfg, out_root=str(tmp_path), verbose=False)
    d = tmp_path / "ettm1" / "m" / "seed0"
    ev, smp = np.load(d / "events.npz"), np.load(d / "samples.npz")
    idle_end = ev["t1"].max() + cfg["energy"]["idle_s"]
    for dom in ("fake_fast", "fake_slow"):
        assert smp[f"{dom}_t"][-1] > idle_end, dom


def test_two_pinned_exclusive_runs_in_one_process(tmp_path):
    """A sweep worker runs many measured runs in one process; pinning must not break the next one."""
    rapl = "/sys/class/powercap/intel-rapl:0"
    before = os.sched_getaffinity(0)
    if not os.path.exists(rapl + "/energy_uj") or len(before) < os.cpu_count():
        pytest.skip("needs a readable RAPL counter and every CPU of the node")
    sets = ["energy.require_exclusive=true", "energy.pin={main: 2, sampler: null}",
            f"energy.counters=[{{name: pkg, type: rapl, path: {rapl}}}]", "seed=0", "dataset.n_steps=10"]
    for tag in ("a", "b"):
        run(build("ettm1", "frozen", "fake", sets + [f"tag={tag}"]), out_root=str(tmp_path), verbose=False)
    assert os.sched_getaffinity(0) == before


def test_setup_failure_starts_no_sampler(tmp_path, monkeypatch):
    """If the stream or policy cannot be built (e.g. a missing data file), no sampler may be left polling."""
    import multiprocessing
    import run as run_module

    def broken(cfg):
        raise FileNotFoundError("data/missing.csv")

    monkeypatch.setattr(run_module, "build_policy", broken)
    with pytest.raises(FileNotFoundError):
        run(_cfg("periodic"), out_root=str(tmp_path), verbose=False)
    assert multiprocessing.active_children() == []


def test_price_scale_is_measured_over_analytic():
    assert price_scale(1.0, 0.1) == pytest.approx(10.0)


def test_non_positive_price_scale_fails_loudly():
    """Updates measured within noise of idle cannot price anything; better to stop the run."""
    with pytest.raises(RuntimeError, match="trial updates"):
        price_scale(-0.1, 0.1)


def test_unpriceable_updates_only_stop_policies_that_price_them():
    """Frozen, periodic and drift never read prices: an unmeasurable scale must not fail their runs."""
    from engine.measured import scale_for
    from methods.periodic import Periodic
    from methods.vou import VoU
    assert scale_for(Periodic({}), -0.1, 0.1) is None
    assert scale_for(VoU({}), 1.0, 0.1) == pytest.approx(10.0)
    with pytest.raises(RuntimeError, match="trial updates"):
        scale_for(VoU({}), -0.1, 0.1)


def test_per_call_costs_are_measured_over_long_windows(tmp_path):
    """Forecast, probe and update each get joules per call from one long back-to-back window,
    which a 100 ms counter resolves even though single calls are a few ms."""
    s = run(_cfg("periodic", ["method.period=4"]), out_root=str(tmp_path), verbose=False)
    pc = s["energy"]["per_call"]
    assert set(pc) == {"infer", "probe", "update"}
    for kind, c in pc.items():
        assert c["calls"] >= 5 and c["window_s"] >= 0.5 and c["j"] > 0, kind
    assert pc["update"]["j"] > pc["probe"]["j"] > 0 and pc["update"]["j"] > pc["infer"]["j"]
    assert s["energy"]["scale_measured_j"] == pc["update"]["j"]


def test_budget_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="budget"):
        run(_cfg("periodic", ["loop.budget={capacity_j: 1.0, income_j: 0.01}"]), out_root=str(tmp_path), verbose=False)


def test_crash_cleans_up_and_saves_nothing(tmp_path, monkeypatch):
    seen = {}

    def boom(stream, learner, policy, cfg):
        seen["sampler"] = learner.meter.sampler
        raise RuntimeError("boom")

    monkeypatch.setattr(measured, "run_stream", boom)
    with pytest.raises(RuntimeError, match="boom"):
        run(_cfg("periodic"), out_root=str(tmp_path), verbose=False)
    s = seen["sampler"]
    assert not s.proc.is_alive()
    for n in s.shm_names:
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=n)
    assert not (tmp_path / "ettm1" / "m" / "seed0" / "summary.json").exists()


def test_non_exclusive_rapl_job_is_refused():
    ecfg = {"require_exclusive": True, "counters": [{"type": "rapl"}]}
    if len(os.sched_getaffinity(0)) == os.cpu_count():
        pytest.skip("this process holds every CPU")
    with pytest.raises(RuntimeError, match="exclusive"):
        measured.check_exclusive(ecfg)


def test_main_process_gets_one_core_per_thread_and_never_the_samplers():
    """Pinning an 8-thread run to one core put VoU's thread team on that core (7 s per update
    instead of 0.9 s on CORe50)."""
    assert measured.main_cpus(2, 3, 1) == {2}
    assert measured.main_cpus(2, 3, 8) == {2, 4, 5, 6, 7, 8, 9, 10}
    assert measured.main_cpus(2, None, 3) == {2, 3, 4}
