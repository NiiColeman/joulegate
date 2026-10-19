"""Paper figures and the cross-dataset table for the image-classification section.

Usage: python -m analysis.report_paper_vision [--out paper]
Needs the CORe50 runs (results/core50, analytic), the domain sweeps (results/domains/core50,
results/domains/core50_b3) and the PACS / Office-Home sweeps (results/domains/{pacs,officehome}).
Writes fig_vision_frontiers.pdf (online accuracy against adaptation energy, per dataset),
fig_vision_final.pdf (accuracy over all domains at the end against energy),
fig_vision_baselines.pdf (the prior-work baselines on the same axes) and
tab_domains_summary.tex.
"""
import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.analyze import GRID, INK, INK2, MUTED, load, seed_mean  # also applies the report's rcParams
from metrics.forecasting import pareto
from analysis.report_domains import metrics
from analysis.report_multidomain import best_within
from analysis.paths import paper_file

FAMS = ("periodic", "drift", "vou", "vougerr", "vougin")
BASE_FAMS = ("carat", "caract", "upf", "d3a", "replay")
VIS_FIG = ("periodic", "vou", "vougerr", "caract", "upf", "d3a", "replay")
COLOR = {"periodic": "#1baf7a", "drift": "#eb6834", "vou": "#2a78d6", "vougerr": "#eda100", "vougin": "#e87ba4",
         "caract": "#008300", "carat": "#008300", "upf": "#4a3aa7", "d3a": "#e34948", "replay": "#1baf7a"}
MARK = {"periodic": "o", "drift": "s", "vou": "D", "vougerr": "^", "vougin": "v",
        "caract": "X", "carat": "P", "upf": "h", "d3a": "<", "replay": "o"}
LINE = {"replay": "--"}
LABEL = {"periodic": "Periodic", "drift": "Drift trigger", "vou": "VoU", "vougerr": "Gated VoU (error)",
         "vougin": "Gated VoU (input)", "carat": "CARA-T", "caract": "CARA-CT", "upf": "UPF (Regol et al.)",
         "d3a": "D3A trigger", "replay": "Periodic + replay"}
SETS = [("CORe50", "results/main", "core50"), ("PACS", "results/domains", "pacs"),
        ("Office-Home", "results/domains", "officehome")]
BASE_SETS = [("CORe50", ("results/main", "results/baselines"), "core50"), ("PACS", "results/domains", "pacs"),
             ("Office-Home", "results/domains", "officehome")]
DOMAIN_SETS = [("CORe50, 2 base", "core50"), ("CORe50, 3 base", "core50_b3"), ("PACS", "pacs"),
               ("Office-Home", "officehome")]


def frame(roots, name):
    roots = (roots,) if isinstance(roots, str) else roots
    m = seed_mean(pd.concat([load(r) for r in roots], ignore_index=True))
    m = m[m.dataset == name].copy()
    e_max = m[m.tag == "periodic_p1"].adapt_j.iloc[0]
    m["pct"] = 100 * m.adapt_j / e_max
    m["acc"] = 100 * (1 - m.mase)
    return m


def frontiers(path):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1))
    for ax, (title, root, name) in zip(axes, SETS):
        m = frame(root, name)
        for f in FAMS:
            g = m[(m.family == f) & (m.pct >= 0.3)].sort_values("pct")  # below: almost no updates, ~frozen
            idx = pareto(g.pct.values, -g.acc.values)
            ax.plot(g.pct.values[idx], g.acc.values[idx], color=COLOR[f], marker=MARK[f], ms=4.5, lw=1.6,
                    label=LABEL[f], markeredgecolor="white", markeredgewidth=0.6)
        frozen = m[m.tag == "frozen"].acc.iloc[0]
        ax.axhline(frozen, color=MUTED, ls="--", lw=1)
        ax.annotate("frozen", (0.35, frozen), xytext=(0, 3), textcoords="offset points", fontsize=7, color=MUTED)
        ax.scatter([100], m[m.tag == "periodic_p1"].acc, color=INK, marker="*", s=60, zorder=5)
        ax.set_xscale("log")
        ax.set_xlim(0.3, 130)
        ax.set_xticks([0.3, 1, 3, 10, 30, 100], ["0.3", "1", "3", "10", "30", "100"])
        ax.set_title(title, loc="left", fontsize=9, color=INK)
        ax.set_xlabel("Adaptation energy (% of update every step)")
    axes[0].set_ylabel("Online accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=INK, marker="*", ls="", ms=8))
    labels.append("Update every step")
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=7.5, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path)


def baseline_frontiers(path):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    for ax, (title, roots, name) in zip(axes, BASE_SETS):
        m = frame(roots, name)
        for f in VIS_FIG:
            g = m[(m.family == f) & (m.pct >= 0.3)].sort_values("pct")
            if g.empty:
                continue
            idx = pareto(g.pct.values, -g.acc.values)
            ax.plot(g.pct.values[idx], g.acc.values[idx], color=COLOR[f], marker=MARK[f], ms=4.5, lw=1.6,
                    ls=LINE.get(f, "-"), label=LABEL[f], markeredgecolor=COLOR[f] if f == "replay" else "white",
                    markerfacecolor="none" if f == "replay" else COLOR[f], markeredgewidth=0.6)
        frozen = m[m.tag == "frozen"].acc.iloc[0]
        ax.axhline(frozen, color=MUTED, ls="--", lw=1)
        ax.annotate("frozen", (0.35, frozen), xytext=(0, 3), textcoords="offset points", fontsize=7, color=MUTED)
        ax.scatter([100], m[m.tag == "periodic_p1"].acc, color=INK, marker="*", s=60, zorder=5)
        ax.set_xscale("log")
        ax.set_xlim(0.3, 130)
        ax.set_xticks([0.3, 1, 3, 10, 30, 100], ["0.3", "1", "3", "10", "30", "100"])
        ax.set_title(title, loc="left", fontsize=9, color=INK)
        ax.set_xlabel("Adaptation energy (% of update every step)")
    axes[0].set_ylabel("Online accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes[1:]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    handles.append(plt.Line2D([], [], color=INK, marker="*", ls="", ms=8))
    labels.append("Update every step")
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(path)


def domain_runs(name, tag):
    root = "results/domains"
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(root, name, tag, "seed*", "summary.json")))]
    acc = np.mean([r["domain_acc"]["acc"] for r in runs], axis=0)
    return metrics(acc), float(np.mean([r["adapt_j"] for r in runs]))


def picks(name):
    """Frozen, update every step, and each family's best setting within 10 % (25 % if none), on the
    dataset's own online accuracy. CORe50's domain sweeps ran fixed tags chosen the same way."""
    if name.startswith("core50"):
        return {"frozen": "frozen", "always": "periodic_p1", "periodic": "periodic_p16", "drift": "drift_t0.4",
                "vou": "vou_l5", "vougerr": "vougerr_t0.05_l10", "vougin": "vougin_k5_l5",
                "carat": "carat_l0.2", "caract": "caract_l0.3", "d3a": "d3a_w4_a1", "replay": "replay_p16",
                "upf": None}  # no UPF setting within 25 % of always-update energy
    m = frame("results/domains", name)
    out = {"frozen": "frozen", "always": "periodic_p1"}
    for f in FAMS + BASE_FAMS:
        out[f] = best_within(m, f, 0.10) or best_within(m, f, 0.25)
    return out


def final_accuracy(path, table_path):
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.9), sharey=False)
    rows = {k: [] for k in ("frozen", "always", *FAMS, *BASE_FAMS)}
    for ax, (title, name) in zip(axes, DOMAIN_SETS):
        p = picks(name)
        res = {k: domain_runs(name, t) for k, t in p.items() if t}
        e_max = res["always"][1]
        frozen = 100 * res["frozen"][0]["final_all"]
        ax.axhline(frozen, color=MUTED, ls="--", lw=1)
        ax.annotate("frozen", (1.0, frozen), xytext=(0, 3), textcoords="offset points", fontsize=7, color=MUTED)
        for f in FAMS:
            mt, e = res[f]
            ax.scatter(100 * e / e_max, 100 * mt["final_all"], color=COLOR[f], marker=MARK[f], s=34,
                       edgecolor="white", linewidth=0.6, zorder=4, label=LABEL[f])
        ax.scatter(100, 100 * res["always"][0]["final_all"], color=INK, marker="*", s=70, zorder=5,
                   label="Update every step")
        ax.set_xscale("log")
        ax.set_xlim(0.8, 160)
        ax.set_xticks([1, 3, 10, 30, 100], ["1", "3", "10", "30", "100"])
        ax.set_title(title, loc="left", fontsize=9, color=INK)
        ax.set_xlabel("Adaptation energy (%)")
        for k in rows:
            if k not in res:
                rows[k].append(None)
                continue
            mt, e = res[k]
            rows[k].append((100 * mt["final_all"], 100 * mt["forgetting"], 100 * e / e_max))
    axes[0].set_ylabel("Accuracy over all domains\nat the end (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=7.5, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path)

    names = {"frozen": "Frozen", "always": "Update every step", **LABEL}
    lines = [r"\begin{tabular}{l " + "rrr " * len(DOMAIN_SETS) + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{t}}}" for t, _ in DOMAIN_SETS) + r" \\",
             " ".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(DOMAIN_SETS))),
             "Policy" + " & Acc. & Forg. & E\\%" * len(DOMAIN_SETS) + r" \\", r"\midrule"]
    for k, cells in rows.items():
        best = [max(r[i][0] for r in rows.values() if r[i]) for i in range(len(DOMAIN_SETS))]
        out = []
        for i, c in enumerate(cells):
            if c is None:
                out += ["--"] * 3
                continue
            a, fg, e = c
            out += [rf"\textbf{{{a:.1f}}}" if np.isclose(a, best[i]) else f"{a:.1f}", f"{fg:.1f}", f"{e:.0f}"]
        lines.append(names[k] + " & " + " & ".join(out) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(table_path, "w").write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper")
    args = ap.parse_args()
    frontiers(paper_file(args.out, "fig_vision_frontiers.pdf"))
    baseline_frontiers(paper_file(args.out, "fig_vision_baselines.pdf"))
    final_accuracy(paper_file(args.out, "fig_vision_final.pdf"), paper_file(args.out, "tab_domains_summary.tex"))
    print(open(paper_file(args.out, "tab_domains_summary.tex")).read())


if __name__ == "__main__":
    main()
