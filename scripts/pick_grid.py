"""Apply the grid rule to a dev results tree; print, or write as an evaluation sweep, the kept settings.

Rule (as for configs/sweeps/main_ttm.yaml and core50.yaml): per family, keep settings whose adaptation
energy is between 3 % and 100 % of always-update energy on at least one dev stream; walking them in order
of mean energy share, drop a setting whose update count is within 10 % of the last kept one's on every
dev stream.
Usage: python -m scripts.pick_grid results/dev_ttm configs/sweeps/dev_baselines_ttm.yaml --datasets synthetic_dev
       ettm1_dev [--write configs/sweeps/baselines_ttm.yaml --out results/baselines_ttm
       --eval-datasets synthetic ettm1 electricity solar]
"""
import argparse

import numpy as np
import yaml

from analysis.analyze import load, seed_mean


def pick(m, family, lo=0.03, hi=1.0, same=0.10):
    shares, counts = {}, {}
    for d, sub in m.groupby("dataset"):
        if not (sub.tag == "periodic_p1").any():
            raise ValueError(f"periodic_p1 missing on {d}")
        e_max = sub[sub.tag == "periodic_p1"].adapt_j.iloc[0]
        for r in sub[sub.family == family].itertuples():
            shares.setdefault(r.tag, {})[d] = r.adapt_j / e_max
            counts.setdefault(r.tag, {})[d] = r.n_updates
    for t, s in shares.items():
        missing = set(m.dataset.unique()) - set(s)
        if missing:
            raise ValueError(f"{t} missing on {sorted(missing)}")
    ok = sorted((t for t, s in shares.items() if any(lo <= v <= hi for v in s.values())),
                key=lambda t: np.mean(list(shares[t].values())))
    kept = []
    for t in ok:
        if kept and all(abs(counts[t][d] - counts[kept[-1]][d]) <= same * max(counts[kept[-1]][d], 1)
                        for d in counts[t]):
            continue
        kept.append(t)
    return kept


def write_eval(dev_spec, tags, out, datasets, seeds, path, header):
    spec = yaml.safe_load(open(dev_spec))
    runs = [r for r in spec["runs"] if r["tag"] in set(tags)]
    new = {"out": out, **({"backbone": spec["backbone"]} if "backbone" in spec else {}),
           "datasets": datasets, "seeds": seeds, "runs": runs}
    with open(path, "w") as f:
        f.write("".join(f"# {line}\n" for line in header.splitlines()))
        yaml.safe_dump(new, f, sort_keys=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("dev_spec")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--families", nargs="+", default=["carat", "caract", "upf", "d3a"])
    ap.add_argument("--write")
    ap.add_argument("--out")
    ap.add_argument("--eval-datasets", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = ap.parse_args()
    if args.write and not (args.out and args.eval_datasets):
        ap.error("--write needs both --out and --eval-datasets")
    m = seed_mean(load(args.root))
    m = m[m.dataset.isin(args.datasets)]
    tags = []
    for f in args.families:
        kept = pick(m, f)
        print(f, kept)
        tags += kept
    if args.write:
        write_eval(args.dev_spec, tags, args.out, args.eval_datasets, args.seeds, args.write,
                   f"Evaluation grid for {', '.join(args.families)}, chosen on {args.root} ({' '.join(args.datasets)})\n"
                   "by pick_grid.py: 3-100 % of always-update energy on some dev stream, near-identical\n"
                   "neighbours (update counts within 10 % on every dev stream) dropped. Analytic energy.")


if __name__ == "__main__":
    main()
