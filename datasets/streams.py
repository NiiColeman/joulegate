"""Data loading: a pretraining corpus and held-out online streams.

Pretraining sources and stream sources are disjoint datasets, so every stream is a
deployment site the backbone has never seen. Models are channel independent
(univariate windows), as in TTM and PatchTST.
"""
import os

import numpy as np
import pandas as pd

from datasets.synthetic import regime_switching
from metrics.forecasting import mase


def load_matrix(name, root="data"):
    cache = os.path.join(root, "cache", name + ".npy")
    if os.path.exists(cache):
        return np.load(cache)
    path = os.path.join(root, name)
    if name.endswith(".csv"):
        x = pd.read_csv(path).iloc[:, 1:].to_numpy(np.float32)
    else:
        x = pd.read_csv(path, header=None).to_numpy(np.float32)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.save(cache, x)
    return x


def pick_channels(x, n, seed=0):
    if n is None or n >= x.shape[1]:
        return x
    rng = np.random.default_rng(seed)
    # Skip near-constant channels, they carry no forecasting signal.
    ok = np.where(x.std(0) > 1e-3 * (np.abs(x).mean(0) + 1e-6))[0]
    return x[:, np.sort(rng.choice(ok, size=n, replace=False))]


# ----------------------------------------------------------------------------- pretraining

def pretrain_series(cfg, root="data"):
    """Return (train, val) lists of 1D float arrays, split in time per channel."""
    train, val = [], []
    for src in cfg["sources"]:
        x = pick_channels(load_matrix(src["file"], root), src.get("channels"), seed=0)
        cut_tr = int(len(x) * cfg["train_frac"])
        cut_va = int(len(x) * (1 - cfg["val_frac"]))
        for c in range(x.shape[1]):
            train.append(x[:cut_tr, c])
            val.append(x[cut_va:, c])
    return train, val


def sample_windows(series, n, length, rng):
    """Draw n random windows of `length` from a list of 1D arrays (length-weighted)."""
    sizes = np.array([len(s) - length for s in series], dtype=np.float64)
    idx = rng.choice(len(series), size=n, p=sizes / sizes.sum())
    out = np.empty((n, length), dtype=np.float32)
    for i, k in enumerate(idx):
        t = rng.integers(0, len(series[k]) - length)
        out[i] = series[k][t:t + length]
    return out


# ----------------------------------------------------------------------------- online streams

class Stream:
    """Rolling-origin stream over a (T, C) matrix.

    Step t exposes the context ctx[t] = x[s : s+L] and, after the forecast is made,
    the target y[t] = x[s+L : s+L+H], with s = start + t * stride.
    """

    def __init__(self, x, context, horizon, stride, start, n_steps, season, warmup):
        need = start + (n_steps - 1) * stride + context + horizon
        if need > len(x):
            n_steps = (len(x) - start - context - horizon) // stride + 1
        self.x, self.L, self.H, self.stride = x, context, horizon, stride
        self.start, self.n_steps = start, n_steps
        w = x[max(0, start - warmup):start + context]
        err = np.abs(w[season:] - w[:-season]).mean(0)
        self.scale = np.maximum(err, 1e-6).astype(np.float32)  # MASE denominator per channel

    def __len__(self):
        return self.n_steps

    def __getitem__(self, t):
        s = self.start + t * self.stride
        ctx = self.x[s:s + self.L].T.copy()
        y = self.x[s + self.L:s + self.L + self.H].T.copy()
        return ctx, y

    def error(self, pred, y):
        """Per-step error the online loop records: MASE against the stream's seasonal scale."""
        return mase(pred, y, self.scale)


def build_stream(dcfg, context, horizon, root="data"):
    if dcfg["file"] == "synthetic":
        x, _ = regime_switching(**dcfg.get("synthetic", {}))
    else:
        x = pick_channels(load_matrix(dcfg["file"], root), dcfg.get("channels"), dcfg.get("channel_seed", 0))
    return Stream(x, context, horizon, dcfg["stride"], dcfg["start"], dcfg["n_steps"],
                  dcfg["season"], dcfg["warmup"])
