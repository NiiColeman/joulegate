"""Controlled regime-switching stream, used as a stress test with known change points.

Regimes differ in period, waveform and trend (an update can fix these), plus a
"noise burst" regime whose extra error is irreducible (an update cannot fix it).
A good update policy should adapt on real shifts and stay quiet during noise bursts.
"""
import numpy as np

REGIMES = ("base", "fast", "square", "noise")


def _regime(kind, t, phase, amp, rng):
    if kind == "base":
        return amp * np.sin(2 * np.pi * t / 96 + phase)
    if kind == "fast":
        return 1.4 * amp * np.sin(2 * np.pi * t / 40 + phase)
    if kind == "square":
        return amp * np.sign(np.sin(2 * np.pi * t / 96 + phase)) * 0.8 + 0.0015 * (t - t[0])
    if kind == "noise":
        return amp * np.sin(2 * np.pi * t / 96 + phase) + rng.normal(0, 0.9, len(t))
    raise ValueError(kind)


def regime_switching(length=24000, channels=8, seed=0, seg_min=2500, seg_max=4500,
                     schedule=("base", "fast", "base", "noise", "square", "base", "noise", "fast", "square")):
    rng = np.random.default_rng(seed)
    x = np.zeros((length, channels), dtype=np.float32)
    labels = np.zeros(length, dtype=np.int64)
    phase = rng.uniform(0, 2 * np.pi, channels)
    amp = rng.uniform(0.7, 1.3, channels)
    t0, k = 0, 0
    while t0 < length:
        kind = schedule[k % len(schedule)]
        t1 = min(length, t0 + int(rng.integers(seg_min, seg_max)))
        t = np.arange(t0, t1, dtype=np.float64)
        for c in range(channels):
            x[t0:t1, c] = _regime(kind, t, phase[c], amp[c], rng) + rng.normal(0, 0.1, len(t))
        labels[t0:t1] = REGIMES.index(kind)
        t0, k = t1, k + 1
    return x, labels
