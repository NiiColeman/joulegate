import numpy as np
import pytest

from engine.attribution import CATEGORIES, attribute, counter_at, energy_between, idle_power


def linear(power, t_end=20.0, dt=0.001):
    t = np.arange(0.0, t_end, dt)
    return t, power * t


def quantized(power, period=0.1, lag=0.005, t_end=20.0):
    """NVML-like: the value changes every `period` in steps of power*period, observed `lag` late."""
    k = np.arange(0, int(t_end / period))
    return k * period + lag, power * k * period


def events(rows):
    kind, parent, t0, t1 = zip(*rows)
    return {"kind": np.array(kind), "parent": np.array(parent), "step": np.zeros(len(rows), dtype=int),
            "t0": np.array(t0, dtype=float), "t1": np.array(t1, dtype=float)}


def test_exact_for_linear_counter():
    s = {"cpu": linear(5.0)}
    assert energy_between(s, 1.0, 3.0)["cpu"] == pytest.approx(10.0)
    assert counter_at(*s["cpu"], 2.0) == pytest.approx(10.0)


def test_idle_power_over_two_windows():
    s = {"cpu": linear(7.0)}
    assert idle_power(s, [(1.0, 2.0), (5.0, 7.0)])["cpu"] == pytest.approx(7.0)


def test_quantized_counter_totals_are_accurate():
    s = {"gpu": quantized(60.0)}
    rng = np.random.default_rng(0)
    starts = np.sort(rng.choice(np.arange(10, 1800) * 0.01, size=100, replace=False))
    ev = events([("update", -1, a, a + 0.005) for a in starts])  # 5 ms each, far below the 100 ms period
    per, _, _ = attribute(s, ev, {"gpu": 0.0}, (0.0, 19.0))
    assert per["raw"]["gpu"].sum() == pytest.approx(60.0 * 0.5, rel=0.01)


def test_incremental_is_not_clipped():
    s = {"cpu": linear(5.0)}
    ev = events([("infer", -1, 1.0, 1.1)])
    per, totals, _ = attribute(s, ev, {"cpu": 8.0}, (0.5, 2.0))
    assert per["incr"]["cpu"][0] == pytest.approx(0.5 - 0.8)
    assert totals["inference"]["incr"]["cpu"] == pytest.approx(-0.3)


def test_categories_gaps_nesting_and_closure():
    s = {"cpu": linear(10.0)}
    ev = events([
        ("other", -1, 0.9, 1.0),
        ("infer", -1, 1.0, 1.1),
        ("decide", -1, 1.2, 1.3),
        ("probe", 2, 1.21, 1.25),   # nested in decide: counted in decision, not on its own
        ("update", -1, 1.4, 1.6),
    ])
    _, totals, closure = attribute(s, ev, {"cpu": 0.0}, (0.9, 2.0))
    assert totals["inference"]["raw"]["cpu"] == pytest.approx(1.0)
    assert totals["decision"]["raw"]["cpu"] == pytest.approx(1.0)
    assert totals["adaptation"]["raw"]["cpu"] == pytest.approx(2.0)
    assert totals["other"]["raw"]["cpu"] == pytest.approx(1.0 + 6.0)  # other window + 0.6 s of gaps
    assert closure < 1e-9


def test_overlapping_top_level_windows_raise():
    s = {"cpu": linear(1.0)}
    ev = events([("infer", -1, 1.0, 1.2), ("update", -1, 1.1, 1.3)])
    with pytest.raises(ValueError, match="overlap"):
        attribute(s, ev, {"cpu": 0.0}, (0.0, 2.0))


def test_window_outside_span_raises():
    s = {"cpu": linear(1.0)}
    ev = events([("infer", -1, 1.0, 3.0)])
    with pytest.raises(ValueError, match="outside"):
        attribute(s, ev, {"cpu": 0.0}, (0.0, 2.0))


def test_unknown_top_level_kind_raises():
    s = {"cpu": linear(1.0)}
    ev = events([("probe", -1, 1.0, 1.1)])
    with pytest.raises(ValueError, match="kind"):
        attribute(s, ev, {"cpu": 0.0}, (0.0, 2.0))


def test_category_map():
    assert set(CATEGORIES.values()) == {"inference", "decision", "adaptation", "other"}
