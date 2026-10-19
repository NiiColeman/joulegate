"""Aggregate results into frontier figures and tables.

Usage: python -m analysis.analyze [--results results/main] [--out paper]
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metrics.forecasting import error_at_budget, frontier_area, pareto
from analysis.paths import paper_file

# Reference palette: categorical slots 1-3 for the three realizable families,
# neutral ink for references. Text stays in text colours.
C = {"vou": "#2a78d6", "drift": "#eb6834", "periodic": "#1baf7a"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
LABEL = {"vou": "VoU (ours)", "drift": "Drift trigger", "periodic": "Periodic"}
DATASETS = ["synthetic", "ettm1", "electricity", "solar"]
TITLE = {"synthetic": "Synthetic (regimes + noise)", "ettm1": "ETTm1", "electricity": "Electricity", "solar": "Solar"}
BUDGETS = (0.05, 0.10, 0.25)
ROOT_RESULTS = "results/main"  # fractions of always-update adaptation energy

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white",
})


def load(root):
    rows = []
    for f in glob.glob(os.path.join(root, "*", "*", "seed*", "summary.json")):
        dataset, tag, seed = f.split(os.sep)[-4:-1]
        s = json.load(open(f))
        s.setdefault("mase", s.get("error"))          # vision summaries report `error` only
        s.setdefault("retention_mse", float("nan"))   # vision reports retention accuracy instead
        family = tag.split("_")[0]
        m = re.search(r"_[a-z]+([0-9.]+)$", tag)
        rows.append(dict(dataset=dataset, tag=tag, family=family, seed=int(seed[4:]),
                         param=float(m.group(1)) if m else np.nan, **{k: v for k, v in s.items()
                                                                      if not isinstance(v, dict)}))
    return pd.DataFrame(rows)


def forecasting_runs(df):
    """Only the forecasting streams: results/main also holds the CORe50 vision runs (report_vision.py)."""
    return df[df.dataset.isin(DATASETS)].reset_index(drop=True)


def seed_mean(df):
    g = df.groupby(["dataset", "tag", "family", "param"], dropna=False)
    out = g.agg(mase=("mase", "mean"), mase_sd=("mase", "std"), adapt_j=("adapt_j", "mean"),
                n_updates=("n_updates", "mean"), retention=("retention_mse", "mean"),
                j_per_update=("j_per_update", "mean"), infer_j=("infer_j", "mean"),
                s_per_update=("s_per_update", "mean"), seeds=("seed", "nunique"),
                **({"n_steps": ("n_steps", "mean")} if "n_steps" in df else {})).reset_index()
    return out


def frontier_fig(m, out):
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 2.9))
    for ax, d in zip(axes, DATASETS):
        sub = m[m.dataset == d]
        if (sub.adapt_j > 0).sum() == 0:
            ax.set_visible(False)
            continue
        ref = sub[sub.family == "frozen"].mase.iloc[0]
        for fam in ("periodic", "drift", "vou"):
            s = sub[(sub.family == fam) & (sub.n_updates >= 1)].sort_values("adapt_j")
            if s.empty:
                continue
            rel = 100 * (s.mase - ref) / ref
            ax.plot(s.adapt_j, rel, color=C[fam], lw=1.6, alpha=0.35, zorder=2)
            f = pareto(s.adapt_j.values, rel.values)
            ax.plot(s.adapt_j.values[f], rel.values[f], color=C[fam], lw=2, marker="o", ms=4.5,
                    mec="white", mew=1, label=LABEL[fam], zorder=3)
        o = sub[(sub.family == "oracle") & (sub.adapt_j > 0)]
        ax.scatter(o.adapt_j, 100 * (o.mase - ref) / ref, s=34, facecolors="none", edgecolors=INK,
                   linewidths=1.1, label="Hindsight oracle", zorder=4)
        ax.axhline(0, color=MUTED, lw=1, ls="--", zorder=1)
        ax.text(0.98, 0.97, "frozen", transform=ax.transAxes, ha="right", va="top", color=INK2, fontsize=7.5)
        ax.set_xscale("log")
        ax.set_title(TITLE[d], color=INK, fontsize=9.5, loc="left")
        ax.set_xlabel("Adaptation energy over the stream (J)")
    axes[0].set_ylabel("MASE change vs frozen (%)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    fig.savefig(paper_file(out, "fig_frontier.pdf"))
    fig.savefig(paper_file(out, "fig_frontier.png"))


def steps_of(row):
    """Stream length of a run: recorded in summaries since the 48-step-stride streams (electricity has 495);
    older runs are all 800 steps."""
    n = row.get("n_steps", np.nan)
    return 800 if n is None or np.isnan(n) else n


def budget_table(m):
    rows = []
    for d in DATASETS:
        sub = m[m.dataset == d]
        if sub.empty:
            continue
        if (sub.tag == "periodic_p1").sum() == 0:
            continue
        ref = sub[sub.family == "frozen"].mase.iloc[0]
        always = sub[sub.tag == "periodic_p1"].iloc[0]
        row = {"dataset": d, "frozen": ref, "always_rel": 100 * (always.mase - ref) / ref, "always_j": always.adapt_j,
               "j_update_mj": 1e3 * always.j_per_update,
               "update_over_infer": always.j_per_update / (always.infer_j / steps_of(always))}
        e_max = always.adapt_j
        f = os.path.join(ROOT_RESULTS, d, "periodic_p1", "seed0", "summary.json")
        if os.path.exists(f):
            parts = json.load(open(f))["update_parts_j"]
            tot = sum(parts.values())
            for k in ("compute", "sram", "dram"):
                row[f"share_{k}"] = 100 * parts.get(k, 0.0) / tot
        for fam in ("periodic", "drift", "vou", "vouerr", "oracle"):
            s = sub[sub.family == fam]
            if s.empty:
                continue
            for b in BUDGETS:
                e = error_at_budget(s.adapt_j.values, s.mase.values, b * e_max)
                row[f"{fam}@{int(b * 100)}"] = np.nan if np.isnan(e) else 100 * (e - ref) / ref
            if fam != "oracle":
                row[f"{fam}_area"] = 100 * frontier_area(s.adapt_j.values, s.mase.values, 0.25 * e_max, ref) / ref
        rows.append(row)
    return pd.DataFrame(rows)


def latex_budget(bt, path):
    """Error change (%) vs frozen at 5/10/25% of always-update energy; best realizable in bold."""
    fams = [("periodic", "Per."), ("drift", "Drift"), ("vou", "VoU"), ("oracle", "Orc.$^\\dagger$")]
    lines = ["\\begin{tabular}{l r r " + " ".join(["rrrr"] * len(BUDGETS)) + "}", "\\toprule",
             "& & & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{{int(b * 100)}\\% budget}}" for b in BUDGETS) + " \\\\",
             " ".join(f"\\cmidrule(lr){{{4 + 4 * i}-{7 + 4 * i}}}" for i in range(len(BUDGETS))),
             "Stream & Frozen & Always & " + " & ".join(" & ".join(n for _, n in fams) for _ in BUDGETS) + " \\\\",
             "\\midrule"]
    for _, r in bt.iterrows():
        cells = [TITLE[r.dataset].split(" (")[0], f"{r.frozen:.3f}", f"{r.always_rel:+.1f}"]
        for b in BUDGETS:
            vals = {f: r.get(f"{f}@{int(b * 100)}", np.nan) for f, _ in fams}
            best = min(vals[f] for f in ("periodic", "drift", "vou"))
            for f, _ in fams:
                v = vals[f]
                s = "--" if np.isnan(v) else f"{v:+.1f}"
                cells.append(f"\\textbf{{{s}}}" if f != "oracle" and v == best else s)
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")


def depth_fig(m, out):
    sub = m[m.family == "depth"].copy()
    if sub.empty:
        return None
    sub["k"] = sub.tag.str.extract(r"_k(\d+)_").astype(int)
    sub["cache"] = sub.tag.str.endswith("cache")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7))
    ls = {"ettm1": "-", "electricity": ":"}
    for d in ("ettm1", "electricity"):
        s = sub[sub.dataset == d]
        ref = m[(m.dataset == d) & (m.family == "frozen")].mase.iloc[0]
        for cache, col, name in ((False, C["drift"], "raw windows"), (True, C["vou"], "cached prefix")):
            t = s[s.cache == cache].sort_values("k")
            lab = f"{name}, {TITLE[d]}"
            axes[0].plot(t.k, 1e3 * t.j_per_update, color=col, ls=ls[d], marker="o", ms=4, lw=1.8, label=lab)
            if cache:  # raw and cached updates are mathematically identical: one line per stream
                axes[1].plot(t.k, 100 * (t.mase - ref) / ref, color=INK2, ls=ls[d], marker="o", ms=4, lw=1.8)
                axes[1].annotate(TITLE[d], (t.k.iloc[-1], 100 * (t.mase.iloc[-1] - ref) / ref), xytext=(-4, 7),
                                 textcoords="offset points", ha="right", color=INK2, fontsize=7.5)
    axes[0].set_ylabel("Energy per update (mJ)")
    axes[1].set_ylabel("MASE change vs frozen (%)")
    for ax in axes:
        ax.set_xlabel("Adapted blocks (0 = head only)")
        ax.set_xticks([0, 1, 2, 4, 8])
    axes[0].set_yscale("log")
    axes[1].axhline(0, color=MUTED, lw=1, ls="--")
    axes[1].set_title("same for raw and cached updates", fontsize=7.5, color=INK2, loc="right")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.13), fontsize=7.5)
    fig.tight_layout()
    fig.savefig(paper_file(out, "fig_depth.pdf"))
    fig.savefig(paper_file(out, "fig_depth.png"))
    return sub


def trace_fig(m, root, out, dataset="synthetic"):
    """Error over time and update times for drift vs VoU at (roughly) matched energy."""
    from datasets.synthetic import regime_switching, REGIMES
    import yaml
    sub = m[m.dataset == dataset]
    if sub.empty:
        return None
    vou = sub[sub.family == "vou"]
    drift = sub[sub.family == "drift"]
    # Drift at its most active setting, and the VoU config closest to it in energy.
    dr = drift.loc[drift.adapt_j.idxmax()]
    v = vou.loc[(np.log(vou.adapt_j) - np.log(dr.adapt_j)).abs().idxmin()]
    cfg = yaml.safe_load(open(os.path.join(root, dataset, "frozen", "seed0", "config.yaml")))
    dc = cfg["dataset"]
    _, labels = regime_switching(**dc["synthetic"])
    L, H, stride, start = cfg["model"]["context"], cfg["model"]["horizon"], dc["stride"], dc["start"]
    steps = np.arange(len(np.load(os.path.join(root, dataset, "frozen", "seed0", "trace.npz"))["err"]))
    reg = labels[start + steps * stride + L]
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 3.6), sharex=True, gridspec_kw={"height_ratios": [3, 1.1]})
    ax = axes[0]
    k = 9
    for tag, col, name in (("frozen", MUTED, "Frozen"), (dr.tag, C["drift"], f"Drift trigger ({dr.adapt_j:.1f} J)"),
                           (v.tag, C["vou"], f"VoU ({v.adapt_j:.1f} J)")):
        tr = np.load(os.path.join(root, dataset, tag, "seed0", "trace.npz"))
        e = np.convolve(tr["err"], np.ones(k) / k, mode="same")
        ax.plot(steps, e, color=col, lw=1.4 if tag != "frozen" else 1.2, label=name)
        if tag != "frozen":
            row = 0 if tag == dr.tag else 1
            u = np.where(tr["update"] > 0)[0]
            axes[1].vlines(u, row + 0.1, row + 0.9, color=col, lw=0.8)
    for a in axes:
        on = reg == REGIMES.index("noise")
        edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
        for s0, s1 in zip(edges[::2], edges[1::2]):
            a.axvspan(s0, s1, color="#f1f0ec", zorder=0, lw=0)
        ch = np.flatnonzero(np.diff(reg)) + 1
        for c in ch:
            a.axvline(c, color=MUTED, lw=0.6, ls=":", zorder=1)
    ax.set_ylabel("MASE (9-step mean)")
    ax.legend(loc="upper left", ncol=3)
    ax.text(0.995, 0.97, "shaded: noise-burst regime (irreducible error)", transform=ax.transAxes,
            ha="right", va="top", color=INK2, fontsize=7.5)
    axes[1].set_yticks([0.5, 1.5], ["Drift", "VoU"])
    axes[1].set_ylim(0, 2)
    axes[1].grid(False)
    axes[1].set_xlabel("Stream step (1 step = 24 time points); dotted lines: regime changes")
    fig.tight_layout()
    fig.savefig(paper_file(out, "fig_trace.pdf"))
    fig.savefig(paper_file(out, "fig_trace.png"))
    # Share of updates spent inside noise bursts.
    share = {}
    for tag in (dr.tag, v.tag):
        tr = np.load(os.path.join(root, dataset, tag, "seed0", "trace.npz"))
        u = tr["update"] > 0
        share[tag] = float((u & (reg == REGIMES.index("noise"))).sum() / max(u.sum(), 1))
    return {"drift_tag": dr.tag, "vou_tag": v.tag, "drift_j": float(dr.adapt_j), "vou_j": float(v.adapt_j),
            "drift_mase": float(dr.mase), "vou_mase": float(v.mase), "noise_share": share,
            "noise_frac_steps": float((reg == REGIMES.index("noise")).mean())}


def noise_share_all(root, dataset="synthetic"):
    """Fraction of updates spent in noise bursts, per config, averaged over seeds."""
    from datasets.synthetic import regime_switching, REGIMES
    import yaml
    cfg = yaml.safe_load(open(os.path.join(root, dataset, "frozen", "seed0", "config.yaml")))
    dc = cfg["dataset"]
    _, labels = regime_switching(**dc["synthetic"])
    steps = np.arange(len(np.load(os.path.join(root, dataset, "frozen", "seed0", "trace.npz"))["err"]))
    noise = labels[dc["start"] + steps * dc["stride"] + cfg["model"]["context"]] == REGIMES.index("noise")
    rows = []
    for f in glob.glob(os.path.join(root, dataset, "*", "seed*", "trace.npz")):
        tag = f.split(os.sep)[-3]
        u = np.load(f)["update"] > 0
        if u.sum():
            rows.append(dict(tag=tag, family=tag.split("_")[0], share=(u & noise).sum() / u.sum(), n=u.sum()))
    return pd.DataFrame(rows).groupby(["family", "tag"]).mean(numeric_only=True).reset_index(), float(noise.mean())


def vou_gap_share(root, min_updates=10):
    """Share of gaps between consecutive VoU updates that equal 2 steps, the minimum the
    estimator allows (it needs two probes). Gaps are pooled over seeds; configs averaging
    fewer than `min_updates` updates per seed are dropped, too few gaps to mean anything."""
    rows = []
    for d in DATASETS[1:]:  # real streams only
        for tag_dir in glob.glob(os.path.join(root, d, "vou_l*")):
            gaps, n = [], []
            for f in glob.glob(os.path.join(tag_dir, "seed*", "trace.npz")):
                u = np.flatnonzero(np.load(f)["update"] > 0)
                gaps.append(np.diff(u))
                n.append(len(u))
            g = np.concatenate(gaps) if gaps else np.array([])
            if np.mean(n) >= min_updates and len(g):
                rows.append(dict(dataset=d, tag=os.path.basename(tag_dir), n_updates=float(np.mean(n)),
                                 min_gap_share=float((g == 2).mean())))
    return pd.DataFrame(rows).sort_values(["dataset", "tag"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/main")
    ap.add_argument("--out", default="paper")
    ap.add_argument("--depth", default="results/depth")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    global ROOT_RESULTS
    ROOT_RESULTS = args.results
    df = forecasting_runs(load(args.results))
    m = seed_mean(df)
    m.to_csv(paper_file(args.out, "summary_by_config.csv"), index=False)
    frontier_fig(m, args.out)
    bt = budget_table(m)
    bt.to_csv(paper_file(args.out, "budget_table.csv"), index=False)
    latex_budget(bt, paper_file(args.out, "tab_budget.tex"))
    depth = None
    if os.path.isdir(args.depth):
        md = seed_mean(load(args.depth))
        depth = depth_fig(pd.concat([m[m.family == "frozen"], md]), args.out)
    extra = {}
    if (m.dataset == "synthetic").any():
        extra["trace"] = trace_fig(m, args.results, args.out)
        ns, frac = noise_share_all(args.results)
        ns.to_csv(paper_file(args.out, "noise_share.csv"), index=False)
        extra["noise_frac_steps"] = frac
    gaps = vou_gap_share(args.results)
    if len(gaps):
        gaps.to_csv(paper_file(args.out, "vou_gaps.csv"), index=False)
        extra["vou_min_gap_share"] = {"min": float(gaps.min_gap_share.min()), "max": float(gaps.min_gap_share.max()),
                                      "configs": int(len(gaps)), "min_updates_per_seed": 10}
    # Cross-check of the analytic meter: rank agreement with measured CPU time per update.
    from scipy.stats import spearmanr
    both = df[df.n_updates > 0]
    if os.path.isdir(args.depth):
        both = pd.concat([both, load(args.depth)])
    rho, _ = spearmanr(both.j_per_update, both.s_per_update)
    extra["spearman_joules_vs_cpu_seconds_per_update"] = float(rho)
    extra["n_runs_for_spearman"] = int(len(both))
    json.dump(extra, open(paper_file(args.out, "extra.json"), "w"), indent=2)
    pd.set_option("display.width", 200)
    print(bt.round(2).T)
    if depth is not None:
        print(depth[["dataset", "tag", "mase", "j_per_update", "s_per_update"]].round(4))
    print(json.dumps(extra, indent=2))


if __name__ == "__main__":
    main()
