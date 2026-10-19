import time
from multiprocessing import shared_memory

import numpy as np
import pytest

from engine.power import FakeCounter, NvmlCounter, RaplCounter, Sampler, SysfsPowerCounter, make_counter


def _rapl_dir(tmp_path, value, max_uj=1000):
    (tmp_path / "max_energy_range_uj").write_text(str(max_uj))
    (tmp_path / "energy_uj").write_text(str(value))
    return tmp_path


def test_rapl_counter_survives_wraparound(tmp_path):
    d = _rapl_dir(tmp_path, 900)
    c = RaplCounter(str(d))
    assert c.read() == 0.0
    (d / "energy_uj").write_text("950")
    assert c.read() == pytest.approx(50e-6)
    (d / "energy_uj").write_text("50")          # wrapped: 950 -> 1000 -> 50
    assert c.read() == pytest.approx(150e-6)


def test_fake_counter_quantizes():
    c = FakeCounter(power_w=100.0, quantum_j=1.0)
    time.sleep(0.025)
    v = c.read()
    assert v == int(v) and v >= 2.0            # whole quanta only; sleep may overshoot


def test_sysfs_counter_integrates_power(tmp_path):
    f = tmp_path / "power"
    f.write_text("5000\n")                       # 5000 mW
    c = SysfsPowerCounter(str(f), unit_w=1e-3)
    c.read()
    time.sleep(0.2)
    assert c.read() == pytest.approx(1.0, rel=0.2)


def test_nvml_requires_a_single_visible_gpu():
    NvmlCounter.check_single(1, True)
    NvmlCounter.check_single(4, False)
    with pytest.raises(RuntimeError, match="4 GPUs"):
        NvmlCounter.check_single(4, True)


def test_make_counter_rejects_unknown_type():
    with pytest.raises(ValueError, match="unknown counter type"):
        make_counter({"name": "x", "type": "nope"})


def test_sampler_measures_fake_power_and_settles():
    s = Sampler([{"name": "a", "type": "fake", "power_w": 10.0, "sample_s": 0.001},
                 {"name": "b", "type": "fake", "power_w": 60.0, "quantum_j": 6.0, "sample_s": 0.01}],
                max_seconds=60)
    try:
        t0 = time.perf_counter()
        time.sleep(0.6)
        t1 = time.perf_counter()
        assert not s.settled_after(t1 + 10)
        time.sleep(0.25)                          # b refreshes every 100 ms
        assert s.settled_after(t1)
        e = s.energy(t0, t1)
        assert e["a"] == pytest.approx(10.0 * (t1 - t0), rel=0.05)
        assert e["b"] == pytest.approx(60.0 * (t1 - t0), rel=0.2)
        t, v = s.snapshot()["a"]
        assert np.all(np.diff(t) > 0) and np.all(np.diff(v) >= 0)
        s.check()
    finally:
        s.close()


def test_sampler_close_cleans_up():
    s = Sampler([{"name": "a", "type": "fake", "power_w": 1.0}], max_seconds=10)
    names = list(s.shm_names)
    s.close()
    s.close()                                     # idempotent
    assert not s.proc.is_alive()
    for n in names:
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=n)


def test_sampler_fails_loudly_when_capacity_runs_out():
    s = Sampler([{"name": "a", "type": "fake", "power_w": 1.0, "sample_s": 0.001}], max_seconds=0.05)
    try:
        time.sleep(0.3)
        with pytest.raises(RuntimeError, match="capacity"):
            s.check()
    finally:
        s.close()


def test_sampler_reports_a_bad_counter_at_startup():
    with pytest.raises(RuntimeError, match="no_such_dir"):
        Sampler([{"name": "a", "type": "rapl", "path": "/sys/class/powercap/no_such_dir"}], max_seconds=10)


def test_sampler_columns_are_contiguous():
    """np.interp copies a strided column on every call (O(n), ~1 ms at 500k points), and that
    cost lands inside open event windows; contiguous columns keep each lookup O(log n)."""
    s = Sampler([{"name": "a", "type": "fake", "power_w": 1.0}], max_seconds=10)
    try:
        time.sleep(0.05)
        t, v = s._columns(0)
        assert len(t) == len(v) > 1
        assert t.flags.c_contiguous and v.flags.c_contiguous
    finally:
        s.close()


def test_fake_counter_adds_busy_power_for_the_parents_cpu_time():
    """Constant power alone gives ~0 incremental energy; a busy term makes work cost energy, as on hardware."""
    s = Sampler([{"name": "a", "type": "fake", "power_w": 10.0, "busy_w": 100.0, "sample_s": 0.001}], max_seconds=60)
    try:
        t0 = time.perf_counter()
        time.sleep(0.3)
        t1 = time.perf_counter()
        x = 0
        while time.perf_counter() - t1 < 0.3:   # 0.3 s of CPU work in this (the parent) process
            x += 1
        t2 = time.perf_counter()
        time.sleep(0.05)
        idle, busy = s.energy(t0, t1)["a"], s.energy(t1, t2)["a"]
        assert idle == pytest.approx(3.0, rel=0.15)
        assert busy == pytest.approx(3.0 + 30.0, rel=0.25)
    finally:
        s.close()


def test_sampler_start_deadline_is_configurable():
    """A spawned sampler re-imports the parent's main module (run.py imports torch); on a loaded
    node that can exceed a fixed deadline, so it is a parameter."""
    with pytest.raises(RuntimeError, match="failed to start"):
        Sampler([{"name": "a", "type": "fake", "power_w": 1.0}], max_seconds=10, start_timeout_s=0.0)
    s = Sampler([{"name": "a", "type": "fake", "power_w": 1.0}], max_seconds=10, start_timeout_s=120)
    s.close()
