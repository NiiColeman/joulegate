"""Forecasting results on three backbones: our PatchMixer, Granite TTM-R2 and Chronos-Bolt Tiny.

Usage: python -m analysis.report_backbones [--out paper]
Analytic runs: results/main (PatchMixer, context 256 / horizon 24), results/ttm and results/chronos (context
512 / horizon 48, stride 48). Measured per-call joules: results/measured_full/cpu and
results/measured_percall/gpu (PatchMixer), results/measured_percall_<b>/{cpu,gpu} (new backbones).
Writes tab_backbones_budget.tex (best MASE change within 10 % of always-update energy),
tab_backbones_percall.tex, tab_backbones_gpu.tex (projected GPU energy, 10 %) and
fig_backbones_frontier.pdf; with the prior-work baselines (results/baselines*) added, tab_baselines.tex,
tab_baselines_gpu.tex, fig_baselines_frontier.pdf and baselines.json.
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.analyze import DATASETS, INK, MUTED, TITLE, load, seed_mean  # also applies the report's rcParams
from analysis.analyze_measured import per_call_costs, projected_runs
from metrics.forecasting import error_at_budget, pareto
from analysis.report_paper_vision import COLOR, LABEL, MARK
from analysis.paths import paper_file

FAMS = ("periodic", "drift", "vou", "vougerr", "vougin")
BACKBONES = {"PatchMixer (ours, 0.6M)": ("results/main", "results/measured_full/cpu", "results/measured_percall/gpu"),
             "TTM-R2 (0.8M)": ("results/ttm", "results/measured_percall_ttm/cpu", "results/measured_percall_ttm/gpu"),
             "Chronos-Bolt Tiny (8.7M)": ("results/chronos", "results/measured_percall_chronos/cpu",
                                          "results/measured_percall_chronos/gpu")}


BASELINES = {"PatchMixer (ours, 0.6M)": "results/baselines", "TTM-R2 (0.8M)": "results/baselines_ttm",
             "Chronos-Bolt Tiny (8.7M)": "results/baselines_chronos"}
BASE_FAMS = ("periodic", "vou", "vougerr", "vougin", "carat", "caract", "upf", "d3a")
FIG_FAMS = ("periodic", "vou", "vougerr", "caract", "upf", "d3a")


def budget_rows(frames, budgets=(0.10,), datasets=DATASETS, fams=FAMS):
    """{backbone: {dataset: {family: [best MASE change % within each budget], "always": change %}}}."""
    out = {}
    for name, m in frames.items():
        for d in datasets:
            sub = m[m.dataset == d]
            if sub.empty or not (sub.tag == "periodic_p1").any():
                continue
            ref = sub[sub.tag == "frozen"].mase.iloc[0]
            e_max = sub[sub.tag == "periodic_p1"].adapt_j.iloc[0]
            row = {"always": 100 * (sub[sub.tag == "periodic_p1"].mase.iloc[0] - ref) / ref}
            for f in fams:
                g = sub[sub.family == f]
                cells = []
                for b in budgets:
                    e = error_at_budget(g.adapt_j.values, g.mase.values, b * e_max) if len(g) else np.nan
                    cells.append(np.nan if np.isnan(e) else 100 * (e - ref) / ref)
                row[f] = cells
            out.setdefault(name, {})[d] = row
    return out


def fmt(x):
    return "--" if np.isnan(x) else f"${x:+.1f}$"


def frames(with_baselines=False):
    out = {}
    for n, r in BACKBONES.items():
        if os.path.isdir(r[0]):
            fr = [load(r[0])]
            if with_baselines and os.path.isdir(BASELINES[n]):
                fr.append(load(BASELINES[n]))
            out[n] = seed_mean(pd.concat(fr, ignore_index=True))
    return out


def table_budget(fr, path, fams=FAMS):
    rows = budget_rows(fr, fams=fams)
    ds = [d for d in DATASETS if any(d in r for r in rows.values())]
    lines = [r"\begin{tabular}{ll " + "r" * len(ds) + "}", r"\toprule",
             "Backbone & Policy & " + " & ".join(TITLE[d].split(" (")[0] for d in ds) + r" \\", r"\midrule"]
    for name, by_d in rows.items():
        lines.append(f"{name} & Update every step & " + " & ".join(fmt(by_d[d]["always"]) for d in ds) + r" \\")
        for f in fams:
            lines.append(f" & {LABEL[f]} & " + " & ".join(fmt(by_d[d][f][0]) for d in ds) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    open(path, "w").write("\n".join(lines) + "\n")
    return rows


def table_percall(path):
    lines = [r"\begin{tabular}{ll rrr rr}", r"\toprule",
             r"Backbone & Device & Forecast (mJ) & Probe (mJ) & Update (mJ) & Update / forecast & Probe / update \\",
             r"\midrule"]
    out = {}
    for name, (_, cpu, gpu) in BACKBONES.items():
        for dev, root in (("CPU", cpu), ("GPU", gpu)):
            if not os.path.isdir(root):
                continue
            c = per_call_costs(root)
            v = {k: float(np.median([x[k] for x in c.values()])) for k in ("infer", "probe", "update")}
            out[f"{name} {dev}"] = v
            lines.append(f"{name} & {dev} & {1e3*v['infer']:.2f} & {1e3*v['probe']:.2f} & {1e3*v['update']:.0f} & "
                         f"{v['update']/v['infer']:.0f}$\\times$ & {v['probe']/v['update']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")
    return out


def table_gpu(path, fams=FAMS, with_baselines=False):
    fr = {}
    for name, (root, _, gpu) in BACKBONES.items():
        if os.path.isdir(root) and os.path.isdir(gpu):
            costs = per_call_costs(gpu)
            g = projected_runs(root, costs)
            if with_baselines and os.path.isdir(BASELINES[name]):
                g = pd.concat([g, projected_runs(BASELINES[name], costs)], ignore_index=True)
            fr[name] = g.groupby(["dataset", "tag", "family"]).agg(mase=("mase", "mean"),
                                                                   adapt_j=("adapt_j", "mean")).reset_index()
    return table_budget(fr, path, fams)


def figure(fr, path, fams=FAMS):
    names = list(fr)
    fig, axes = plt.subplots(len(names), 4, figsize=(11.5, 2.5 * len(names)), squeeze=False)
    for i, name in enumerate(names):
        m = fr[name]
        for j, d in enumerate(DATASETS):
            ax, sub = axes[i][j], m[m.dataset == d]
            if sub.empty:
                ax.set_visible(False)
                continue
            ref = sub[sub.tag == "frozen"].mase.iloc[0]
            e_max = sub[sub.tag == "periodic_p1"].adapt_j.iloc[0]
            for f in fams:
                g = sub[(sub.family == f) & (sub.adapt_j > 0.003 * e_max)].sort_values("adapt_j")
                if g.empty:
                    continue
                x, y = 100 * g.adapt_j.values / e_max, 100 * (g.mase.values - ref) / ref
                idx = pareto(x, y)
                ax.plot(x[idx], y[idx], color=COLOR[f], marker=MARK[f], ms=4, lw=1.5, label=LABEL[f],
                        markeredgecolor="white", markeredgewidth=0.5)
            ax.axhline(0, color=MUTED, ls="--", lw=1)
            ax.scatter([100], [100 * (sub[sub.tag == "periodic_p1"].mase.iloc[0] - ref) / ref], color=INK,
                       marker="*", s=55, zorder=5, label="Update every step")
            ax.set_xscale("log")
            ax.set_xlim(0.3, 130)
            ax.set_xticks([1, 3, 10, 30, 100], ["1", "3", "10", "30", "100"])
            if i == 0:
                ax.set_title(TITLE[d].split(" (")[0], loc="left", fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(f"{name}\nMASE change (%)", fontsize=8)
            if i == len(names) - 1:
                ax.set_xlabel("Adaptation energy (%)")
    h, lab = axes[0][0].get_legend_handles_labels()
    for ax in axes.ravel():
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in lab:
                h.append(hh)
                lab.append(ll)
    fig.legend(h, lab, loc="lower center", ncol=6, fontsize=7.5, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper")
    args = ap.parse_args()
    fr = frames()
    rows = table_budget(fr, paper_file(args.out, "tab_backbones_budget.tex"))
    costs = table_percall(paper_file(args.out, "tab_backbones_percall.tex"))
    gpu = table_gpu(paper_file(args.out, "tab_backbones_gpu.tex"))
    figure(fr, paper_file(args.out, "fig_backbones_frontier.pdf"))
    summary = {"budget_10": rows, "gpu_budget_10": gpu, "per_call": costs,
               "frozen_mase": {n: {d: float(m[(m.dataset == d) & (m.tag == "frozen")].mase.iloc[0])
                                   for d in DATASETS if ((m.dataset == d) & (m.tag == "frozen")).any()}
                               for n, m in fr.items()}}
    json.dump(summary, open(paper_file(args.out, "backbones.json"), "w"), indent=1, default=float)
    bfr = frames(True)
    brows = table_budget(bfr, paper_file(args.out, "tab_baselines.tex"), BASE_FAMS)
    bgpu = table_gpu(paper_file(args.out, "tab_baselines_gpu.tex"), BASE_FAMS, True)
    figure(bfr, paper_file(args.out, "fig_baselines_frontier.pdf"), FIG_FAMS)
    json.dump({"budget_10": brows, "gpu_budget_10": bgpu}, open(paper_file(args.out, "baselines.json"), "w"),
              indent=1, default=float)
    for f in ("tab_backbones_budget.tex", "tab_backbones_percall.tex", "tab_backbones_gpu.tex"):
        print(open(paper_file(args.out, f)).read())


if __name__ == "__main__":
    main()
