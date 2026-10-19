"""Run one (dataset, policy, seed) online continual fine-tuning experiment.

Usage:
  python run.py --dataset ettm1 --method vou --set method.lam=20 seed=1 tag=vou_l20
Writes <out, default results/main>/<dataset>/<tag>/seed<k>/{config.yaml, summary.json, trace.npz}.
"""
import argparse
import os

import numpy as np
import torch

from datasets.core50 import build_vision_stream, domain_test_set
from datasets.domains import build_domain_stream, heldout_set
from datasets.streams import build_stream
from engine.energy import build_meter
from engine.learner import Learner
from engine.measured import run_measured
from engine.online import run_stream
from engine.vision_learner import VisionLearner
from methods import build_policy
from models.mixer import PatchMixer
from models.vit_tiny import TinyViT
from utils.config import build
from utils.misc import run_dir, save_run, seed_everything


class DomainEval:
    """Accuracy on every domain (base = the retention set, then each streamed session) before the
    stream and after the last step of each session: row k = after domain k, column j = domain j.
    CORe50 test frames come from `domain_eval.half` (default: the half the stream does not train on);
    multi-domain datasets (kind: domains) use each domain's fixed test split."""

    def __init__(self, cfg, learner, stream):
        d, e = cfg["dataset"], cfg["domain_eval"]
        base = np.load(cfg["retention"])
        self.sets = [(base["images"], base["labels"])]
        if d.get("kind") == "domains":
            names = list(d["domains"])
            self.sets += [heldout_set(d["root"], n) for n in names]
        else:
            half = e.get("half", "first" if d["half"] == "second" else "second")
            names = [f"s{s}" for s in d["sessions"]]
            self.sets += [domain_test_set(d["root"], s, half, e.get("n", 640), e.get("seed", 0))[:2] for s in d["sessions"]]
        self.learner, self.mean, self.std = learner, d["mean"], d["std"]
        reg = stream.regimes
        name = (lambda r: str(r)) if d.get("kind") == "domains" else (lambda r: f"s{r}")
        self.ends = {t: name(reg[t]) for t in range(len(reg)) if t == len(reg) - 1 or reg[t + 1] != reg[t]}
        self.domains = ["base"] + names
        self.after, self.acc = ["start"], [self._row()]

    def _row(self):
        return [self.learner.accuracy(x, y, self.mean, self.std) for x, y in self.sets]

    def on_step(self, t):
        if t in self.ends:
            self.after.append(self.ends[t])
            self.acc.append(self._row())

    def result(self):
        return {"domains": self.domains, "after": self.after, "acc": self.acc}


def build_model(cfg):
    """The forecasting backbone and its learner class (PatchMixer, TTM-R2 or Chronos-Bolt)."""
    m = dict(cfg["model"])
    name = m.pop("name", None)
    if name == "ttm_r2":
        from models.ttm_r2 import TTMR2
        return TTMR2(**m, freq_token=cfg["dataset"]["freq_token"]), Learner
    if name == "chronos_bolt":
        from engine.quantile_learner import QuantileLearner
        from models.chronos_bolt import ChronosBolt
        return ChronosBolt(**m), QuantileLearner
    model = PatchMixer(**m)
    model.load_state_dict(torch.load(cfg["checkpoint"], map_location="cpu"))
    return model, Learner


def run(cfg, out_root="results/main", verbose=True):
    if cfg["dataset"].get("task") != "vision" and cfg["dataset"]["stride"] < cfg["model"]["horizon"]:
        raise ValueError(f"stride {cfg['dataset']['stride']} < horizon {cfg['model']['horizon']}: each target "
                         "would overlap the next step's target and be trained on before it is forecast")
    if cfg.get("domain_eval") is not None and cfg["energy"]["backend"] == "sampled":
        raise ValueError("domain_eval runs inside the measured span; use analytic energy")
    seed_everything(cfg["seed"])
    torch.set_num_threads(cfg["threads"])
    device = cfg.get("device", "cpu")
    if cfg["learner"].get("replay"):
        cfg["learner"]["replay"].setdefault("seed", cfg["seed"])  # each seed draws its own replay batches
    if cfg["dataset"].get("task") == "vision":
        model = TinyViT(**cfg["model"], pretrained=False)
        model.load_state_dict(torch.load(cfg["checkpoint"], map_location="cpu"))
        learner = VisionLearner(model, cfg["learner"], device=device)
        stream = (build_domain_stream if cfg["dataset"].get("kind") == "domains" else build_vision_stream)(cfg["dataset"])
    else:
        model, learner_cls = build_model(cfg)
        learner = learner_cls(model, cfg["learner"], device=device)
        stream = build_stream(cfg["dataset"], cfg["model"]["context"], cfg["model"]["horizon"])
    policy = build_policy(cfg["method"])
    if cfg["dataset"].get("task") == "vision":
        learner.set_base(lambda: np.load(cfg["retention"]), cfg["dataset"]["mean"], cfg["dataset"]["std"])
    else:
        learner.set_base(lambda: np.load(cfg["retention_windows"]), cfg["model"]["horizon"])
    # Last: after LoRA, so adapters are metered, and after everything that can fail during setup,
    # because a sampled meter starts its sampler process here and run_measured is what closes it.
    learner.meter = build_meter(cfg["energy"], model, device)

    if cfg["energy"]["backend"] == "sampled":
        summary, trace, arrays = run_measured(cfg, learner, stream, policy)
    else:
        domains = DomainEval(cfg, learner, stream) if cfg.get("domain_eval") is not None else None
        summary, trace = run_stream(stream, learner, policy, cfg["loop"], domains and domains.on_step)
        arrays = None
        if domains:
            summary["domain_acc"] = domains.result()
    if cfg["dataset"].get("task") == "vision":
        summary["retention_acc"] = learner.retention(cfg["retention"], cfg["dataset"]["mean"], cfg["dataset"]["std"])
    else:
        summary["n_steps"] = len(stream)  # electricity at stride 48 is shorter than n_steps
        retention = np.load(cfg["retention_windows"])
        summary["retention_mse"] = learner.eval_mse(retention, cfg["model"]["horizon"])
    summary["adapter_params"] = learner.n_params
    path = run_dir(out_root, cfg)
    save_run(path, cfg, summary, trace, arrays)
    if verbose:
        ret = summary.get("retention_mse", summary.get("retention_acc"))
        print(f"{cfg['dataset']['name']:12s} {cfg.get('tag') or cfg['method']['name']:14s} seed{cfg['seed']} "
              f"error {summary['error']:.4f}  updates {summary['n_updates']:4d}  "
              f"adapt {summary['adapt_j']:.3f} J  infer {summary['infer_j']:.3f} J  retention {ret:.4f}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset")
    ap.add_argument("--method")
    ap.add_argument("--energy")
    ap.add_argument("--out", default="results/main")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    run(build(args.dataset, args.method, args.energy, args.set), args.out)


if __name__ == "__main__":
    main()
