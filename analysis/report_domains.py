"""Per-domain accuracy on CORe50: the accuracy matrix, forgetting and transfer, per policy.

Usage: python -m analysis.report_domains [--root results/domains] [--dataset core50] [--out paper]
(--dataset core50_b3: 3 base domains; outputs get the suffix _b3.)
Reads summary["domain_acc"] from runs made with domain_eval (configs/sweeps/core50_domains.yaml):
acc[k][j] = accuracy on domain j after the stream has finished domain k (row 0: before the stream;
column 0: the base sessions 1-2). Averages over seeds. Writes fig_vision_domains.pdf (change in accuracy
since before the stream + average accuracy on the domains seen so far), tab_vision_domains.tex and
vision_domains.json.
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from analysis.paths import paper_file

NAME = {"frozen": "Frozen", "periodic_p1": "Update every step", "periodic_p16": "Periodic (every 16)",
        "drift_t0.4": "Drift", "vou_l5": "VoU", "vougerr_t0.05_l10": "Gated VoU (error)",
        "vougin_k5_l5": "Gated VoU (input)"}
HEATMAPS = ("frozen", "periodic_p1", "periodic_p16", "vou_l5")


def load(root, dataset="core50"):
    out = {}
    for tag in NAME:
        runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(root, dataset, tag, "seed*", "summary.json")))]
        if runs:
            d = runs[0]["domain_acc"]
            out[tag] = {"domains": d["domains"], "after": d["after"], "seeds": len(runs),
                        "acc": np.mean([r["domain_acc"]["acc"] for r in runs], axis=0),
                        "adapt_j": float(np.mean([r["adapt_j"] for r in runs])),
                        "online_acc": float(np.mean([1 - r["error"] for r in runs]))}
    return out


def metrics(acc):
    """acc is (K+1) x (K+1): rows start, after D1..DK; columns base, D1..DK."""
    K = acc.shape[0] - 1
    seen = [acc[k, :k + 1].mean() for k in range(K + 1)]               # base + domains learned so far
    forgetting = np.mean([acc[j:, j].max() - acc[K, j] for j in range(K)])  # best ever vs final, earlier domains
    just_learned = np.mean([acc[k, k] for k in range(1, K + 1)])
    before = np.mean([acc[k - 1, k] for k in range(1, K + 1)])           # on a domain just before learning it
    return {"final_all": float(acc[K].mean()), "final_base": float(acc[K, 0]), "seen_curve": [float(v) for v in seen],
            "forgetting": float(forgetting), "just_learned": float(just_learned), "before_learning": float(before)}


def figure(res, path, heatmaps=HEATMAPS, names=NAME):
    tags = [t for t in heatmaps if t in res]
    fig = plt.figure(figsize=(3.3 * len(tags), 6.6))
    gs = fig.add_gridspec(2, len(tags), height_ratios=[1.15, 1])
    for i, t in enumerate(tags):
        ax = fig.add_subplot(gs[0, i])
        a = 100 * (res[t]["acc"] - res[t]["acc"][0])                  # change since before the stream
        ax.imshow(a, cmap="RdBu", vmin=-15, vmax=15)
        for (r, c), v in np.ndenumerate(a):
            if r:
                ax.text(c, r, f"{v:+.0f}", ha="center", va="center", fontsize=5.5,
                        color="white" if abs(v) > 9 else "#1a1a1a")
        ax.set_xticks(range(a.shape[1]), res[t]["domains"], rotation=90, fontsize=6)
        ax.set_yticks(range(a.shape[0]), res[t]["after"], fontsize=6)
        ax.set_title(names[t], loc="left", fontsize=8)
        ax.set_xlabel("Change in accuracy on domain (points)", fontsize=7)
        if i == 0:
            ax.set_ylabel("After learning", fontsize=7)
    ax = fig.add_subplot(gs[1, :])
    for t in names:
        if t in res:
            m = res[t]["metrics"]["seen_curve"]
            ax.plot(range(len(m)), 100 * np.array(m), marker="o", ms=3, lw=1.5, label=names[t])
    first = next(iter(res.values()))
    ax.set_xticks(range(len(first["after"])), first["after"], fontsize=7)
    ax.set_ylabel("Mean accuracy on domains\nseen so far (%)", fontsize=7)
    ax.grid(axis="y", color="0.9")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=6, ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(path)


def table(res, path, e_max):
    lines = [r"\begin{tabular}{l rrrrrr}", r"\toprule",
             r"Policy & Energy (\%) & Final, all domains & Final, base & Just learned & Before learning & Forgetting \\",
             r"\midrule"]
    for t in NAME:
        if t in res:
            m = res[t]["metrics"]
            lines.append(f"{NAME[t]} & {100 * res[t]['adapt_j'] / e_max:.1f} & {100 * m['final_all']:.1f} & "
                         f"{100 * m['final_base']:.1f} & {100 * m['just_learned']:.1f} & "
                         f"{100 * m['before_learning']:.1f} & {100 * m['forgetting']:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/domains")
    ap.add_argument("--dataset", default="core50")
    ap.add_argument("--out", default="paper")
    args = ap.parse_args()
    res = load(args.root, args.dataset)
    sfx = args.dataset.replace("core50", "")
    for r in res.values():
        r["metrics"] = metrics(r["acc"])
    e_max = res["periodic_p1"]["adapt_j"]
    figure(res, paper_file(args.out, f"fig_vision_domains{sfx}.pdf"))
    table(res, paper_file(args.out, f"tab_vision_domains{sfx}.tex"), e_max)
    out = {t: {**r["metrics"], "acc": r["acc"].tolist(), "domains": r["domains"], "after": r["after"],
               "seeds": r["seeds"], "energy_pct": 100 * r["adapt_j"] / e_max, "online_acc": r["online_acc"]}
           for t, r in res.items()}
    json.dump(out, open(paper_file(args.out, f"vision_domains{sfx}.json"), "w"), indent=1)
    print(open(paper_file(args.out, f"tab_vision_domains{sfx}.tex")).read())


if __name__ == "__main__":
    main()
