"""Fine-tune ViT-Tiny (ImageNet weights, 96x96) on the base domain of a multi-domain dataset.

Trains on the base domain's non-test images; its test split becomes the retention set. Reports
accuracy on every domain's test split before any adaptation, to show the domain shift.
Usage: python -m scripts.pretrain_domains pacs [--steps 300] [--device cuda]
Outputs: the checkpoint and retention set named in configs/datasets/<name>.yaml, and
logs/pretrain_<name>.json.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from datasets.domains import heldout_set, load_domain
from models.vit_tiny import TinyViT
from scripts.pretrain_vision import accuracy, to_tensor
from utils.config import build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--steps", type=int, default=300, help="optimizer steps of 128 images (CORe50 used ~300)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    cfg = build(args.dataset, "frozen", "analytic_45nm")
    d = cfg["dataset"]
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    images, labels, test = load_domain(d["root"], d["base"])
    train_x, train_y = np.asarray(images[~test]), labels[~test]
    ret_x, ret_y = heldout_set(d["root"], d["base"])

    model = TinyViT(num_classes=cfg["model"]["num_classes"], pretrained=True).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=args.steps, pct_start=0.1)
    model.train()
    for step in range(args.steps):
        idx = rng.choice(len(train_y), min(128, len(train_y)), replace=False)
        x = to_tensor(train_x[idx], args.device)
        if rng.random() < 0.5:
            x = x.flip(3)
        loss = F.cross_entropy(model(x), torch.from_numpy(train_y[idx]).to(args.device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 50 == 0:
            print(f"step {step}/{args.steps} loss {loss.item():.3f}", flush=True)
    model.eval()

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.cpu().state_dict(), cfg["checkpoint"])
    np.savez(cfg["retention"], images=ret_x, labels=ret_y)
    model.to(args.device)
    report = {"base": d["base"], "train_images": int(len(train_y)), "steps": args.steps,
              "base_test_acc": accuracy(model, ret_x, ret_y, args.device), "target_test_acc": {}}
    for name in d["domains"]:
        x, y = heldout_set(d["root"], name)
        report["target_test_acc"][name] = accuracy(model, x, y, args.device)
    os.makedirs("logs", exist_ok=True)
    json.dump(report, open(f"logs/pretrain_{args.dataset}.json", "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
