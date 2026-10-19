"""Forecast error and error-energy frontier metrics."""
import numpy as np


def mase(pred, y, scale):
    """Mean absolute scaled error for one step, averaged over channels. pred, y: (C, H)."""
    return float((np.abs(pred - y).mean(1) / scale).mean())


def pareto(energy, error):
    """Indices of the lower-left Pareto front (minimise both), sorted by energy."""
    order = np.argsort(energy)
    front, best = [], np.inf
    for i in order:
        if error[i] < best - 1e-12:
            front.append(i)
            best = error[i]
    return np.array(front, dtype=int)


def error_at_budget(energy, error, budget):
    """Best error achievable by any configuration using at most `budget` joules."""
    ok = energy <= budget
    return float(error[ok].min()) if ok.any() else np.nan


def frontier_area(energy, error, e_max, err_ref):
    """Area between a method's step-wise frontier and the frozen error on [0, e_max].

    Larger is better: it integrates the error reduction bought at each energy level,
    normalised by e_max so the result is an average gain in MASE.
    """
    grid = np.linspace(0, e_max, 400)
    best = np.array([min(err_ref, error_at_budget(energy, error, b)) if np.any(energy <= b) else err_ref
                     for b in grid])
    return float(np.trapezoid(err_ref - best, grid) / e_max)
