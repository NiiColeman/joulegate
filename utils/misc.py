"""Seeding and result IO."""
import json
import os
import random

import numpy as np
import torch
import yaml


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_dir(root, cfg):
    tag = cfg.get("tag") or cfg["method"]["name"]
    path = os.path.join(root, cfg["dataset"]["name"], tag, f"seed{cfg['seed']}")
    os.makedirs(path, exist_ok=True)
    return path


def save_run(path, cfg, summary, trace, arrays=None):
    with open(os.path.join(path, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    np.savez_compressed(os.path.join(path, "trace.npz"), **{k: np.asarray(v) for k, v in trace.items()})
    for name, content in (arrays or {}).items():
        np.savez_compressed(os.path.join(path, name + ".npz"), **{k: np.asarray(v) for k, v in content.items()})
    # summary.json last: sweep.py treats its presence as "this run is complete"
    with open(os.path.join(path, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
