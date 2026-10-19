"""Fine-tune ViT-Tiny (ImageNet weights, 96x96) on CORe50's source sessions (default 1 and 2).

Saves the checkpoint and a retention set (every 10th kept frame of each source video, never
trained on), and reports frozen accuracy on every target session's dev half, so we can see
before any adaptation whether the session shifts actually hurt.
Usage: python -m scripts.pretrain_vision [--epochs 3] [--device cuda] [--base 1 2 3 --suffix _b3]
Outputs: checkpoints/vit_tiny_core50<suffix>.pt, checkpoints/vit_tiny_retention<suffix>.npz,
logs/pretrain_vision<suffix>.json.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from datasets.core50 import category, half_mask, load_session
from models.vit_tiny import TinyViT

MEAN = STD = 0.5


def to_tensor(images, device):
    x = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float() / 255.0
    return ((x - MEAN) / STD).to(device)


def accuracy(model, images, labels, device, batch=512):
    correct = 0
    with torch.no_grad():
        for i in range(0, len(labels), batch):
            correct += int((model(to_tensor(images[i:i + batch], device)).argmax(1).cpu().numpy()
                            == labels[i:i + batch]).sum())
    return correct / len(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/core50")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--base", type=int, nargs="+", default=[1, 2], help="source sessions")
    ap.add_argument("--suffix", default="", help="appended to the output file names")
    args = ap.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    train_x, train_y, ret_x, ret_y = [], [], [], []
    for s in args.base:
        images, meta = load_session(args.root, s)
        labels = np.array([category(o) for o in meta["obj"]])
        ret = meta["pos"] % 10 == 0
        train_x.append(np.asarray(images[~ret])), train_y.append(labels[~ret])
        ret_x.append(np.asarray(images[ret])), ret_y.append(labels[ret])
    train_x, train_y = np.concatenate(train_x), np.concatenate(train_y)
    ret_x, ret_y = np.concatenate(ret_x), np.concatenate(ret_y)

    model = TinyViT(pretrained=True).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    steps = args.epochs * (len(train_y) // 128)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=steps, pct_start=0.1)
    model.train()
    for step in range(steps):
        idx = rng.choice(len(train_y), 128, replace=False)
        x = to_tensor(train_x[idx], args.device)
        if rng.random() < 0.5:
            x = x.flip(3)
        loss = F.cross_entropy(model(x), torch.from_numpy(train_y[idx]).to(args.device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 200 == 0:
            print(f"step {step}/{steps} loss {loss.item():.3f}", flush=True)
    model.eval()

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.cpu().state_dict(), f"checkpoints/vit_tiny_core50{args.suffix}.pt")
    np.savez(f"checkpoints/vit_tiny_retention{args.suffix}.npz", images=ret_x, labels=ret_y)
    model.to(args.device)
    report = {"base_sessions": args.base, "source_retention_acc": accuracy(model, ret_x, ret_y, args.device), "target_dev_acc": {}}
    for s in [s for s in range(1, 12) if s not in args.base]:
        images, meta = load_session(args.root, s)
        dev = half_mask(meta, "first")
        labels = np.array([category(o) for o in meta["obj"]])
        report["target_dev_acc"][s] = accuracy(model, np.asarray(images[dev]), labels[dev], args.device)
    os.makedirs("logs", exist_ok=True)
    json.dump(report, open(f"logs/pretrain_vision{args.suffix}.json", "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
