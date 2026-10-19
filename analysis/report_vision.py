"""Tables and figure for the vision (CORe50) results, in accuracy terms.

Usage: python -m analysis.report_vision [--out paper]
Estimated: results/core50 (3 seeds). Measured CPU: results/measured_vision/cpu (1 seed).
Projected CPU/GPU: exact counts from the analytic runs priced with per-call costs measured in
results/measured_vision_percall/{cpu,gpu}.
The task is online supervised domain adaptation: the model classifies each batch, then sees its
labels. Runs record online error (1 - accuracy); this report shows online accuracy and the gain
in accuracy points over the frozen model. 'Inference' is the forward pass on a new batch (the
engine's `infer` event; the time-series code calls it a forecast). Writes tab_vision.tex (budgets), tab_vision_percall.tex
(per-call joules), fig_vision_frontier.pdf, and vision_coverage.json (frozen / always-update
accuracy, accuracy after session switches, retention, pretraining).
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from analysis.analyze import load, seed_mean
from analysis.analyze_measured import per_call_costs, projected_runs
from metrics.forecasting import error_at_budget, pareto
from analysis.paths import paper_file

FAMS = ("periodic", "drift", "vou", "vougerr", "vougin")
NAME = {"periodic": "Periodic", "drift": "Drift", "vou": "VoU", "vougerr": "Gated VoU (error)",
        "vougin": "Gated VoU (input)"}
BUDGETS = (0.05, 0.10, 0.25)


def accuracy_gain_cells(m, d, fam, budgets):
    """Best accuracy gain over frozen, in percentage points, within each budget fraction of the
    always-update energy (m holds error in the `mase` column, as analyze.load provides)."""
    sub = m[m.dataset == d]
    ref = sub[sub.tag == "frozen"].mase.iloc[0]
    e_max = sub[sub.tag == "periodic_p1"].adapt_j.iloc[0]
    g = sub[sub.family == fam]
    out = []
    for b in budgets:
        e = error_at_budget(g.adapt_j.values, g.mase.values, b * e_max) if len(g) else np.nan
        out.append(np.nan if np.isnan(e) else 100 * (ref - e))
    return out


def fmt(x):
    return "--" if np.isnan(x) else f"${x:+.1f}$"


def frames():
    out = {"Estimated": seed_mean(load("results/main"))}
    if os.path.isdir("results/measured_vision/cpu"):
        out["Measured CPU"] = seed_mean(load("results/measured_vision/cpu"))
    for dev in ("cpu", "gpu"):
        root = f"results/measured_vision_percall/{dev}"
        if os.path.isdir(root):
            g = projected_runs("results/main", per_call_costs(root))
            g = g[g.dataset == "core50"]
            out[f"Projected {dev.upper()}"] = g.groupby(["dataset", "tag", "family"]).agg(
                mase=("mase", "mean"), adapt_j=("adapt_j", "mean")).reset_index()
    return {k: v[v.dataset == "core50"] for k, v in out.items()}


def switch_steps(dcfg):
    """First step of every session after the first, as SessionStream lays them out."""
    n, k = dcfg["n_steps"], len(dcfg["sessions"])
    per = [n // k + (i < n % k) for i in range(k)]
    return np.cumsum(per)[:-1]


def accuracy_after_switch(root, tag, dcfg, window=10):
    """Mean online accuracy over the `window` steps after each session switch, averaged over seeds."""
    starts = switch_steps(dcfg)
    vals = []
    for f in glob.glob(os.path.join(root, "core50", tag, "seed*", "trace.npz")):
        err = np.load(f)["err"]
        vals.append(1 - np.mean([err[s:s + window].mean() for s in starts]))
    return float(np.mean(vals)) if vals else float("nan")


def table_budgets(fr, path):
    lines = [r"\begin{tabular}{ll rrr}", r"\toprule",
             r"Energy & Policy & 5\% & 10\% & 25\% \\", r"\midrule"]
    for label, m in fr.items():
        for i, f in enumerate(FAMS):
            cells = accuracy_gain_cells(m, "core50", f, BUDGETS)
            lines.append((label if i == 0 else "") + f" & {NAME[f]} & " + " & ".join(fmt(c) for c in cells) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    open(path, "w").write("\n".join(lines) + "\n")


def table_percall(path):
    lines = [r"\begin{tabular}{l rrr rr}", r"\toprule",
             r"& Inference (mJ) & Probe (mJ) & Update (mJ) & Update / inference & Probe / update \\", r"\midrule"]
    for dev in ("cpu", "gpu"):
        root = f"results/measured_vision_percall/{dev}"
        if not os.path.isdir(root):
            continue
        c = per_call_costs(root)["core50"]
        lines.append(f"Measured {dev.upper()} & {1e3*c['infer']:.1f} & {1e3*c['probe']:.1f} & {1e3*c['update']:.0f} & "
                     f"{c['update']/c['infer']:.0f}$\\times$ & {c['probe']/c['update']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")


def frontier(fr, path):
    views = [v for v in ("Estimated", "Measured CPU", "Projected GPU") if v in fr]
    fig, axes = plt.subplots(1, len(views), figsize=(3.8 * len(views), 3.0), squeeze=False)
    for ax, v in zip(axes[0], views):
        m = fr[v]
        ref = 1 - m[m.tag == "frozen"].mase.iloc[0]
        for f in FAMS:
            g = m[(m.family == f) & (m.adapt_j > 0)].sort_values("adapt_j")
            if g.empty:
                continue
            acc = 100 * (1 - g.mase.values)
            idx = pareto(g.adapt_j.values, -acc)                 # lower-left front of (energy, -accuracy)
            ax.plot(g.adapt_j.values[idx], acc[idx], marker="o", ms=4, label=NAME[f])
        ax.axhline(100 * ref, color="0.6", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_title(v, loc="left", fontsize=9)
        ax.set_xlabel("Adaptation energy (J)")
    axes[0][0].set_ylabel("Online accuracy (%); dashed: frozen")
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper")
    args = ap.parse_args()
    fr = frames()
    table_budgets(fr, paper_file(args.out, "tab_vision.tex"))
    table_percall(paper_file(args.out, "tab_vision_percall.tex"))
    frontier(fr, paper_file(args.out, "fig_vision_frontier.pdf"))
    an = fr["Estimated"]
    dcfg = yaml.safe_load(open("configs/datasets/core50.yaml"))
    tags = ["frozen", "periodic_p1"] + sorted(set(an.tag) - {"frozen", "periodic_p1"})
    ret = {t: float(np.mean([json.load(open(f)).get("retention_acc", np.nan) for f in
                             glob.glob(os.path.join("results/main", "core50", t, "seed*", "summary.json"))]))
           for t in tags}
    cov = {"energy_views": list(fr), "frozen_accuracy": float(1 - an[an.tag == "frozen"].mase.iloc[0]),
           "always_update_accuracy": float(1 - an[an.tag == "periodic_p1"].mase.iloc[0]),
           "accuracy_10_steps_after_switch": {t: accuracy_after_switch("results/main", t, dcfg) for t in tags},
           "retention_accuracy": ret}
    pre = "logs/pretrain_vision.json"
    if os.path.exists(pre):
        cov["pretrain"] = json.load(open(pre))
    json.dump(cov, open(paper_file(args.out, "vision_coverage.json"), "w"), indent=2)
    print(json.dumps(cov, indent=2))
    for name in ("tab_vision.tex", "tab_vision_percall.tex"):
        print(open(paper_file(args.out, name)).read())


if __name__ == "__main__":
    main()
