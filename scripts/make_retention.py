"""Retention windows for a given context + horizon, from the held-out split of the pretraining series.

The PatchMixer's windows (256 + 24) are written by pretrain.py; released backbones need their own length.
Usage: python -m scripts.make_retention --context 512 --horizon 48  ->  checkpoints/retention_windows_512_48.npy
"""
import argparse
import os

import numpy as np

from datasets.streams import pretrain_series, sample_windows
from utils.config import ROOT, load_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", type=int, required=True)
    ap.add_argument("--horizon", type=int, required=True)
    args = ap.parse_args()
    cfg = load_yaml(os.path.join(ROOT, "pretrain.yaml"))
    _, val = pretrain_series(cfg)
    w = sample_windows(val, cfg["retention_windows"], args.context + args.horizon, np.random.default_rng(123))
    path = f"checkpoints/retention_windows_{args.context}_{args.horizon}.npy"
    np.save(path, w)
    print(path, w.shape)


if __name__ == "__main__":
    main()
