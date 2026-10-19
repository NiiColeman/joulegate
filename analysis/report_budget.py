"""Hard energy budget (budget_sweeps.py): which policy is best when a token bucket caps adaptation energy?

Usage: python -m analysis.report_budget [--out paper]
Reads the held-out budget roots (seeds 3-5). Writes tab_hardbudget.tex, fig_hardbudget.pdf and
hardbudget.json (every number the paper text cites). A root that is missing or incomplete is skipped with a note.
"""
import argparse
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np

from analysis.analyze import DATASETS, INK2, MUTED, load  # applies rcParams
from scripts.budget_sweeps import CAPACITY, INCOME, PICK_FAMILIES, family, split_tag, suffix
from analysis.report_paper_vision import COLOR, LABEL, MARK
from analysis.report_revision import per_seed_change, write
from analysis.paths import paper_file

ROOTS = {"patchmixer": "results/budget", "ttm": "results/budget_ttm", "chronos": "results/budget_chronos",
         "core50": "results/budget_core50"}
SEEDS = (3, 4, 5)
REAL = [d for d in DATASETS if d != "synthetic"]
LABEL = {**LABEL, "greedy": "Greedy (when affordable)"}
COLOR = {**COLOR, "greedy": "#52514e"}
MARK = {**MARK, "greedy": "*"}
BLOCKS = [("real", "Real forecasting streams: mean $\\Delta$MASE (\\%) over 3 streams $\\times$ 3 backbones"),
          ("synthetic", "Synthetic stress test: mean $\\Delta$MASE (\\%) over 3 backbones"),
          ("core50", "CORe50 (ViT-Tiny): accuracy gain over frozen (points)")]


def cells(roots):
    """{(block, budget, fam): dict(value, sd, pairs)}, plus per-pair rows. value: mean over pairs of the seed-mean
    change (block real/synthetic: % MASE; core50: accuracy points); sd: median seed sd over the pairs."""
    pairs = []
    for name, root in roots.items():
        runs = load(root)
        ch = per_seed_change(runs)
        ch = ch[ch.tag.str.contains("__")]
        for (d, tag), g in ch.groupby(["dataset", "tag"]):
            base, bud = split_tag(tag)
            block = "core50" if d == "core50" else ("synthetic" if d == "synthetic" else "real")
            v = g.gain_pts if block == "core50" else g.change
            pairs.append(dict(backbone=name, dataset=d, block=block, budget=bud, fam=family(base), tag=base,
                              mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
                              rate=float(g.n_updates.mean() / g.n_steps.mean()) if "n_steps" in g else float("nan")))
    out = {}
    for p in pairs:
        out.setdefault((p["block"], p["budget"], p["fam"]), []).append(p)
    agg = {k: dict(value=float(np.mean([p["mean"] for p in v])), sd=float(np.nanmedian([p["sd"] for p in v])),
                   n=len(v)) for k, v in out.items()}
    return agg, pairs


def winners(pairs, block="real", maximize=False):
    """{budget: {fam: number of (backbone, stream) pairs where fam has the best seed-mean value}}."""
    best = {}
    for p in pairs:
        if p["block"] != block:
            continue
        k = (p["budget"], p["backbone"], p["dataset"])
        better = (lambda a, b: a > b) if maximize else (lambda a, b: a < b)
        if k not in best or better(p["mean"], best[k][1]):
            best[k] = (p["fam"], p["mean"])
    out = {}
    for (bud, _, _), (fam, _) in best.items():
        out.setdefault(bud, {}).setdefault(fam, 0)
        out[bud][fam] += 1
    return out


def budget_keys():
    return [suffix(i, c)[2:] for c in CAPACITY for i in INCOME]


def table(agg, path):
    keys = budget_keys()
    n_inc = len(INCOME)
    lines = [r"\begin{tabular}{l " + "r" * n_inc + " " + "r" * n_inc + "}", r"\toprule",
             rf" & \multicolumn{{{n_inc}}}{{c}}{{Capacity 1 update (no saving)}} & "
             rf"\multicolumn{{{n_inc}}}{{c}}{{Capacity 10 updates}} \\",
             rf"\cmidrule(lr){{2-{1 + n_inc}}} \cmidrule(lr){{{2 + n_inc}-{1 + 2 * n_inc}}}",
             "Income per step (\\% of an update) & " + " & ".join(f"{round(100 * i)}" for _ in CAPACITY for i in INCOME)
             + r" \\"]
    for block, title in BLOCKS:
        if not any(k[0] == block for k in agg):
            continue
        maximize = block == "core50"
        lines += [r"\midrule", rf"\multicolumn{{{1 + 2 * n_inc}}}{{l}}{{\emph{{{title}}}}} \\"]
        best = {}
        for b in keys:
            vals = [(agg[(block, b, f)]["value"], f) for f in PICK_FAMILIES if (block, b, f) in agg]
            if vals:
                best[b] = (max if maximize else min)(vals)[1]
        for f in PICK_FAMILIES:
            if not any((block, b, f) in agg for b in keys):
                continue
            row = [LABEL.get(f, f)]
            for b in keys:
                c = agg.get((block, b, f))
                if c is None:
                    row.append("--")
                    continue
                s = f"{c['value']:+.1f}"
                row.append(rf"$\mathbf{{{s}}}$" if best.get(b) == f else f"${s}$")
            lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(path, lines)


FIG_FAMS = ("greedy", "drift", "vou", "vougerr", "upf", "d3a")


def figure(agg, path, cap=1):
    blocks = [b for b in BLOCKS if any(k[0] == b[0] for k in agg)]
    fig, axes = plt.subplots(1, len(blocks), figsize=(3.6 * len(blocks), 2.9), squeeze=False)
    titles = {"real": "Real forecasting streams (9 pairs)", "synthetic": "Synthetic (3 backbones)",
              "core50": "CORe50 (ViT-Tiny)"}
    for ax, (block, _) in zip(axes[0], blocks):
        for f in FIG_FAMS:
            xs, ys = [], []
            for i in INCOME:
                c = agg.get((block, suffix(i, cap)[2:], f))
                if c is not None:
                    xs.append(100 * i)
                    ys.append(c["value"])
            if xs:
                ax.plot(xs, ys, color=COLOR.get(f, INK2), marker=MARK.get(f, "o"), ms=4, lw=1.4,
                        mec="white", mew=0.6, label=LABEL.get(f, f))
        ax.axhline(0, color=MUTED, lw=1, ls="--", zorder=1)
        ax.set_xscale("log")
        ax.set_xticks([100 * i for i in INCOME])
        ax.set_xticklabels([f"{round(100 * i)}" for i in INCOME])
        ax.set_xlabel("Income per step (% of one update)")
        ax.set_title(titles[block], fontsize=9.5, loc="left")
        ax.set_ylabel("Accuracy gain (points)" if block == "core50" else "MASE change vs frozen (%)")
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.08), fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def complete(root, seeds=SEEDS):
    if not os.path.isdir(root):
        return False
    runs = load(root)
    return not runs.empty and all(set(g.seed) == set(seeds) for _, g in runs.groupby(["dataset", "tag"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper")
    args = ap.parse_args()
    roots = {k: v for k, v in ROOTS.items() if complete(v)}
    for k in set(ROOTS) - set(roots):
        print(f"[skip] {ROOTS[k]} missing or incomplete")
    if not roots:
        return
    agg, pairs = cells(roots)
    table(agg, paper_file(args.out, "tab_hardbudget.tex"))
    figure(agg, paper_file(args.out, "fig_hardbudget.pdf"))
    js = {"cells": {"|".join(k): v for k, v in agg.items()},
          "wins_real": winners(pairs, "real"), "wins_synthetic": winners(pairs, "synthetic"),
          "wins_core50": winners(pairs, "core50", maximize=True),
          "pairs": pairs, "roots": roots}
    json.dump(js, open(paper_file(args.out, "hardbudget.json"), "w"), indent=1,
              default=lambda x: None if isinstance(x, float) and math.isnan(x) else x)
    for b in budget_keys():
        print(b, js["wins_real"].get(b))


if __name__ == "__main__":
    main()
