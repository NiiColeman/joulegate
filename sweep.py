"""Run a sweep file in parallel, skipping runs whose summary already exists.

A sweep YAML lists datasets, seeds and runs (and optionally a forecasting `backbone`: ttm | chronos);
each run is a method plus overrides:
  datasets: [ettm1]
  seeds: [0, 1]
  runs:
    - {method: periodic, tag: periodic_p4, set: {method.period: 4}}
Usage: python sweep.py configs/sweeps/main.yaml [--workers 2] [--datasets ettm1 solar]
                       [--energy measured_cpu] [--out results/measured/cpu] [--set device=cpu]
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import yaml

from utils.config import build


def _one(args):
    dataset, run, seed, out, energy, extra, backbone = args
    from run import run as run_one  # import inside the worker process
    overrides = [f"{k}={json.dumps(v)}" for k, v in run.get("set", {}).items()]
    overrides += list(extra) + [f"seed={seed}", f"tag={run['tag']}"]
    cfg = build(dataset, run["method"], energy or run.get("energy"), overrides, backbone=backbone)
    return run_one(cfg, out)


def expand(spec, datasets=None, out=None, energy=None, extra=(), seeds=None):
    out = out or spec.get("out", "results/main")
    jobs = []
    for d in datasets or spec["datasets"]:
        for r in spec["runs"]:
            for s in seeds or spec["seeds"]:
                if not os.path.exists(os.path.join(out, d, r["tag"], f"seed{s}", "summary.json")):
                    jobs.append((d, r, s, out, energy, tuple(extra), spec.get("backbone")))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0)))  # Slurm-allocated cores
    ap.add_argument("--datasets", nargs="*", help="run only these datasets of the spec (sharding)")
    ap.add_argument("--energy", help="energy config for every run (overrides the spec)")
    ap.add_argument("--out", help="results root (overrides the spec's out)")
    ap.add_argument("--set", nargs="*", default=[], help="overrides for every run, e.g. device=cuda")
    ap.add_argument("--seeds", nargs="*", type=int, help="run only these seeds of the spec (sharding)")
    args = ap.parse_args()
    with open(args.spec) as f:
        spec = yaml.safe_load(f)
    unknown = set(args.datasets or []) - set(spec["datasets"])
    if unknown:
        ap.error(f"datasets not in {args.spec}: {sorted(unknown)}")
    jobs = expand(spec, args.datasets, args.out, args.energy, args.set, args.seeds)
    print(f"{len(jobs)} runs to do", flush=True)
    # one fresh process per run: no CPU pin, CUDA state or sampler carries over between runs
    with ProcessPoolExecutor(args.workers, max_tasks_per_child=1) as pool:
        for fut in as_completed([pool.submit(_one, j) for j in jobs]):
            fut.result()


if __name__ == "__main__":
    main()
