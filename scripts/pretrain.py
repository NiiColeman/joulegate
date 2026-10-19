"""Pretrain the frozen backbone on the pretraining corpus (disjoint from all streams).

Also saves a fixed set of held-out pretraining windows used to measure retention
(forgetting of the general forecasting skill) after online adaptation.
Usage: python -m scripts.pretrain [--set steps=2000]
"""
import argparse
import os
import time

import numpy as np
import torch
import yaml

from datasets.streams import pretrain_series, sample_windows
from models.mixer import CLIP, PatchMixer
from utils.config import ROOT, load_yaml, set_dotted
from utils.misc import seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_yaml(os.path.join(ROOT, "pretrain.yaml"))
    base = load_yaml(os.path.join(ROOT, "base.yaml"))
    for item in args.set:
        k, v = item.split("=", 1)
        set_dotted(cfg, k, yaml.safe_load(v))
    seed_everything(cfg["seed"])
    torch.set_num_threads(len(os.sched_getaffinity(0)))  # Slurm-allocated cores, not the whole node

    m = base["model"]
    L, H = m["context"], m["horizon"]
    train, val = pretrain_series(cfg)
    rng = np.random.default_rng(cfg["seed"])
    model = PatchMixer(**m)
    print(f"params {sum(p.numel() for p in model.parameters()):,}  series {len(train)}")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=cfg["steps"], pct_start=0.1)

    val_w = sample_windows(val, cfg["retention_windows"], L + H, np.random.default_rng(123))
    t0 = time.time()
    for step in range(1, cfg["steps"] + 1):
        w = torch.from_numpy(sample_windows(train, 2 * cfg["batch"], L + H, rng))
        xn, mu, sd = model.revin(w[:, :L])
        keep = (sd.squeeze(1) > 1e-3 * (mu.abs().squeeze(1) + 1e-3)).nonzero().squeeze(1)[:cfg["batch"]]
        xn, mu, sd, w = xn[keep], mu[keep], sd[keep], w[keep]  # drop near-constant contexts
        yn = ((w[:, L:] - mu) / sd).clamp(-CLIP, CLIP)
        loss = torch.mean((model.forward_norm(xn) - yn) ** 2)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == cfg["steps"]:
            model.eval()
            with torch.no_grad():
                v = torch.from_numpy(val_w)
                vx, vmu, vsd = model.revin(v[:, :L])
                vloss = torch.mean((model.forward_norm(vx) - ((v[:, L:] - vmu) / vsd).clamp(-CLIP, CLIP)) ** 2)
            model.train()
            print(f"step {step}  train {loss.item():.4f}  val {vloss.item():.4f}  {time.time() - t0:.0f}s", flush=True)

    os.makedirs(os.path.dirname(base["checkpoint"]), exist_ok=True)
    torch.save(model.state_dict(), base["checkpoint"])
    np.save(base["retention_windows"], val_w)
    print("saved", base["checkpoint"])


if __name__ == "__main__":
    main()
