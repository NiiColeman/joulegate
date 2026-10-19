"""Tables and figures for the paper revision (E1-E7): granularity, held-out confirmation, measurement
protocol, hyperparameters, recovery after shifts, VoU conditions, seed spreads.

Usage: python -m analysis.report_revision [--out paper] [--png DIR] [--root name=path ...]
Reads result roots only; every section checks its inputs and skips with a printed note when they are
missing, so the script runs while jobs are still going. Existing report scripts are untouched.

Writes tab_granularity.tex, fig_granularity.pdf, tab_confirm.tex, tab_protocol.tex, tab_percall_spread.tex,
tab_hparams.tex, tab_recovery.tex (vision), tab_recovery_synth.tex, fig_trajectories.pdf,
tab_vou_conditions.tex, tab_seeds.tex and revision.json (every number the paper text cites).

Recovery on PACS / Office-Home: the traces carry no domain labels (their 'bucket' column is the energy
budget token bucket), but DomainStream lays domains out in order with floor(non-test images / batch) steps
each, so the switch steps are recomputed from the dataset config and the data/<name>/<domain>_test.npy
masks (checked against the trace length). Switches are those between streamed domains (not the shift
from the base domain at step 0), as report_vision.switch_steps does for CORe50; windows are clipped at
the next switch.
"""
import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from analysis.analyze import DATASETS, INK, INK2, MUTED, TITLE, load, noise_share_all, seed_mean  # applies rcParams
from analysis.analyze_measured import per_call_costs, per_call_table, projected_runs
from analysis.report_backbones import BACKBONES, BASELINES, budget_rows
from analysis.report_paper_vision import BASE_FAMS, COLOR, FAMS, LABEL, MARK, picks
from analysis.report_vision import switch_steps
from analysis.paths import paper_file

ROOTS = {
    "gran": "results/gran", "gran_measured": "results/gran_measured",
    "confirm": "results/confirm", "confirm_ttm": "results/confirm_ttm",
    "confirm_chronos": "results/confirm_chronos", "confirm_core50": "results/confirm_core50",
    "measured": "results/measured", "conservative": "results/conservative", "depth": "results/depth",
    "main": "results/main", "baselines": "results/baselines", "domains": "results/domains",
}
GROUPS = [(4, 1), (8, 2), (16, 4), (32, 8)]   # (period, steps per update): 200x1, 100x2, 50x4, 25x8
LRS = (0.002, 0.0003)
STREAM_LEN = 800
SHORT = {d: TITLE[d].split(" (")[0] for d in DATASETS}
CONFIRM_FAMS = ("periodic", "drift", "vou", "vougerr", "vougin", "carat", "caract", "upf", "d3a")
LABEL = {**LABEL, "frozen": "Frozen", "always": "Update every step"}
NOTES = []
OPTS = {"strict": True, "gran_seeds": (0, 1, 2), "confirm_seeds": (3, 4, 5), "seeds_n": 3}


def note(msg):
    print(f"[skip] {msg}")
    NOTES.append(msg)


def have(*paths):
    missing = [p for p in paths if not os.path.isdir(p)]
    return not missing, missing


# ----------------------------------------------------------------------------- helpers

def cell(mean, sd, digits=1, sign=True):
    """`mean +- sd` as a LaTeX cell; '--' for a missing mean, the mean alone when the sd is undefined (1 seed)."""
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "--"
    fmt = f"{{:{'+' if sign else ''}.{digits}f}}"
    if sd is None or (isinstance(sd, float) and math.isnan(sd)):
        return f"${fmt.format(mean)}$"
    return f"${fmt.format(mean)} \\pm {sd:.{digits}f}$"


def rel(x, ref):
    """Change of x against ref, in percent."""
    return 100.0 * (x - ref) / ref


def after_switch(err, starts, window, ends=None):
    """Mean of err[s:s+window] over the switch steps `starts`; `ends` (same length) clips each window,
    e.g. at the next switch."""
    err = np.asarray(err, dtype=float)
    vals = []
    for i, s in enumerate(starts):
        e = s + window if ends is None else min(s + window, ends[i])
        vals.append(err[s:e].mean())
    return float(np.mean(vals))


def mean_sd(x):
    x = np.asarray(list(x), dtype=float)
    x = x[~np.isnan(x)]
    if not len(x):
        return float("nan"), float("nan")
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else float("nan")


def per_seed_change(runs, frozen_tag="frozen"):
    """Rows of `runs` (analyze.load output) plus 'change': % MASE change against the frozen run of the same
    dataset and seed. For vision, mase holds the error rate."""
    fz = runs[runs.tag == frozen_tag].set_index(["dataset", "seed"]).mase
    out = runs[runs.tag != frozen_tag].copy()
    ref = [fz.get((d, s), np.nan) for d, s in zip(out.dataset, out.seed)]
    out["frozen"] = ref
    out["change"] = [rel(m, r) for m, r in zip(out.mase, ref)]
    out["gain_pts"] = [100 * (r - m) for m, r in zip(out.mase, ref)]
    return out


def energy_share(runs):
    """{(dataset, tag): adaptation energy as % of that dataset's periodic_p1} from seed means."""
    m = runs.groupby(["dataset", "tag"]).adapt_j.mean()
    out = {}
    for (d, t), j in m.items():
        e = m.get((d, "periodic_p1"), np.nan)
        out[(d, t)] = 100.0 * j / e if e and not np.isnan(e) and e > 0 else np.nan
    return out


def fmt_j(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:.0f}" if abs(x) >= 100 else f"{x:.1f}"


def write(path, lines):
    open(path, "w").write("\n".join(lines) + "\n")


def tex_escape(s):
    return str(s).replace("_", r"\_").replace("%", r"\%")


# ----------------------------------------------------------------------------- 1. granularity

TAG = re.compile(r"^(?:(frozen)_lr([0-9.e-]+)|(always)_lr([0-9.e-]+)|gran_lr([0-9.e-]+)_p(\d+)_k(\d+))$")


def parse_gran_tag(tag):
    m = TAG.match(tag)
    if not m:
        return None
    g = m.groups()
    if g[0]:
        return dict(kind="frozen", lr=float(g[1]), p=0, k=0)
    if g[2]:
        return dict(kind="always", lr=float(g[3]), p=1, k=1)
    return dict(kind="gran", lr=float(g[4]), p=int(g[5]), k=int(g[6]))


def gran_runs(root):
    df = load(root)
    info = [parse_gran_tag(t) for t in df.tag]
    keep = [i is not None for i in info]
    df = df[keep].reset_index(drop=True)
    return pd.concat([df, pd.DataFrame([i for i in info if i is not None])], axis=1)


def gran_cells(runs):
    """{(lr, dataset, key): dict(change=(mean, sd), pct=analytic adaptation energy % of always, adapt_j)}
    with key in 'frozen', 'always', (p, k). Change is against that lr's frozen run, per seed."""
    out = {}
    for lr in sorted(set(runs.lr)):
        for d in sorted(set(runs[runs.lr == lr].dataset)):
            sub = runs[(runs.lr == lr) & (runs.dataset == d)]
            fz = sub[sub.kind == "frozen"].set_index("seed").mase
            al = sub[sub.kind == "always"].adapt_j.mean()
            out[(lr, d, "frozen")] = dict(change=(0.0, 0.0), pct=0.0, adapt_j=0.0)
            for key, rows in [("always", sub[sub.kind == "always"])] + \
                             [((p, k), sub[(sub.kind == "gran") & (sub.p == p) & (sub.k == k)]) for p, k in GROUPS]:
                if rows.empty:
                    continue
                ch = [rel(r.mase, fz[r.seed]) for r in rows.itertuples() if r.seed in fz.index]
                out[(lr, d, key)] = dict(change=mean_sd(ch), pct=100.0 * rows.adapt_j.mean() / al if al > 0 else np.nan,
                                         adapt_j=float(rows.adapt_j.mean()))
    return out


def measured_totals(root):
    """Per run: incremental adaptation joules and total incremental joules (inference + decision +
    adaptation + other), straight from the summaries of a measured root."""
    rows = []
    for f in glob.glob(os.path.join(root, "*", "*", "seed*", "summary.json")):
        dataset, tag, seed = f.split(os.sep)[-4:-1]
        s = json.load(open(f))
        sysj = s.get("energy", {}).get("system", {})
        rows.append(dict(dataset=dataset, tag=tag, seed=int(seed[4:]), adapt_j=s["adapt_j"],
                         total_j=sum(c["incr"] for c in sysj.values()) if sysj else np.nan))
    return pd.DataFrame(rows)


def gran_measured(root):
    """{(dev, lr, dataset, key): (adapt_j mean, total_j mean)} from results/gran_measured/{cpu,gpu}."""
    out = {}
    for dev in ("cpu", "gpu"):
        r = os.path.join(root, dev)
        if not os.path.isdir(r):
            continue
        t = measured_totals(r)
        for row in t.itertuples():
            i = parse_gran_tag(row.tag)
            if i is None:
                continue
            key = "frozen" if i["kind"] == "frozen" else "always" if i["kind"] == "always" else (i["p"], i["k"])
            out.setdefault((dev, i["lr"], row.dataset, key), []).append((row.adapt_j, row.total_j))
    return {k: tuple(float(np.mean(c)) for c in zip(*v)) for k, v in out.items()}


def granularity_table(cells, meas, path):
    keys = ["frozen"] + GROUPS + ["always"]
    name = {"frozen": "Frozen", "always": r"800$\times$1 (every step)", **{g: rf"{800 // g[0]}$\times${g[1]}" for g in GROUPS}}
    streams = [d for d in DATASETS if any(k[1] == d for k in cells)]
    lines = [r"\begin{tabular}{ll rr rr rr rr}", r"\toprule",
             r"& & \multicolumn{2}{c}{lr $2\!\times\!10^{-3}$} & \multicolumn{2}{c}{lr $3\!\times\!10^{-4}$} & "
             r"\multicolumn{2}{c}{Measured CPU (J)} & \multicolumn{2}{c}{Measured GPU (J)} \\",
             r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8} \cmidrule(lr){9-10}",
             r"Stream & Events$\times$steps & $\Delta$MASE (\%) & E (\%) & $\Delta$MASE (\%) & E (\%) & Adapt. & Total & "
             r"Adapt. & Total \\", r"\midrule"]
    for d in streams:
        for i, key in enumerate(keys):
            row = [SHORT[d] if i == 0 else "", name[key]]
            for lr in LRS:
                c = cells.get((lr, d, key))
                row += [cell(*c["change"]) if c else "--",
                        f"{c['pct']:.1f}" if c and not math.isnan(c["pct"]) else "--"]
            for dev in ("cpu", "gpu"):
                m = meas.get((dev, LRS[0], d, key))
                row += [fmt_j(m[0]), fmt_j(m[1])] if m else ["--", "--"]
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    write(path, lines)


def granularity_fig(cells, path):
    streams = [d for d in DATASETS if any(k[1] == d for k in cells)]
    lrs = [lr for lr in LRS if any(k[0] == lr for k in cells)]
    fig, axes = plt.subplots(len(lrs), len(streams), figsize=(2.9 * len(streams), 2.3 * len(lrs) + 0.4),
                             squeeze=False, sharex=True)
    for i, lr in enumerate(lrs):
        for j, d in enumerate(streams):
            ax = axes[i][j]
            xs = [800 // p for p, _ in GROUPS if (lr, d, (p, _)) in cells]
            ys = [cells[(lr, d, g)]["change"] for g in GROUPS if (lr, d, g) in cells]
            ax.errorbar(xs, [y[0] for y in ys], yerr=[0 if math.isnan(y[1]) else y[1] for y in ys], color=COLOR["periodic"],
                        marker="o", ms=4.5, lw=1.6, capsize=2.5, markeredgecolor="white", markeredgewidth=0.6,
                        label="Fixed budget, regrouped")
            al = cells.get((lr, d, "always"))
            if al:
                ax.errorbar([800], [al["change"][0]], yerr=[0 if math.isnan(al["change"][1]) else al["change"][1]],
                            color=INK, marker="*", ms=8, ls="", capsize=2.5, label="Update every step (800$\\times$1)")
            ax.axhline(0, color=MUTED, ls="--", lw=1)
            ax.set_xscale("log")
            ax.set_xticks([25, 50, 100, 200, 800], ["25", "50", "100", "200", "800"], fontsize=6.5)
            ax.minorticks_off()
            if i == 0:
                ax.set_title(SHORT[d], loc="left", fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(f"lr {lr:g}\n$\\Delta$MASE vs frozen (%)", fontsize=8)
            if i == len(lrs) - 1:
                ax.set_xlabel("Update events (log)", fontsize=8)
    h, lab = axes[0][0].get_legend_handles_labels()
    for ax in axes.ravel():
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in lab:
                h.append(hh)
                lab.append(ll)
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path)
    return fig


def gran_missing(runs, seeds):
    """Cells (lr, dataset, key) without exactly the expected seeds, among the full grid."""
    bad = []
    for lr in LRS if len(set(runs.lr)) > 1 or LRS[0] in set(runs.lr) else sorted(set(runs.lr)):
        for d in DATASETS:
            for kind, p, k in [("frozen", 0, 0), ("always", 1, 1)] + [("gran", p, k) for p, k in GROUPS]:
                g = runs[(runs.lr == lr) & (runs.dataset == d) & (runs.kind == kind) & (runs.p == p) & (runs.k == k)]
                if set(g.seed) != set(seeds):
                    bad.append((lr, d, kind, p, k))
    return bad


def section_granularity(out, roots, png, js):
    ok, miss = have(roots["gran"])
    if not ok:
        return note(f"granularity: {roots['gran']} not present yet")
    runs = gran_runs(roots["gran"])
    if runs.empty:
        return note(f"granularity: {roots['gran']} holds no gran/frozen/always runs")
    bad = gran_missing(runs, OPTS["gran_seeds"])
    if bad and OPTS["strict"]:
        return note(f"granularity: {roots['gran']} incomplete ({len(bad)} lr/stream/grouping cells lack seeds "
                    f"{list(OPTS['gran_seeds'])}); rerun when the jobs finish or pass --allow-partial")
    cells = gran_cells(runs)
    meas = gran_measured(roots["gran_measured"]) if os.path.isdir(roots["gran_measured"]) else {}
    if not meas:
        print(f"[note] granularity: no measured runs under {roots['gran_measured']} yet; measured columns are '--'")
    granularity_table(cells, meas, paper_file(out, "tab_granularity.tex"))
    fig = granularity_fig(cells, paper_file(out, "fig_granularity.pdf"))
    if png:
        fig.savefig(os.path.join(png, "fig_granularity.png"))
    plt.close(fig)
    js["granularity"] = {
        "cells": {f"lr{k[0]:g}|{k[1]}|{k[2]}": {"change_mean": v["change"][0], "change_sd": v["change"][1],
                                                "energy_pct_of_always": v["pct"], "adapt_j": v["adapt_j"]}
                  for k, v in cells.items()},
        "measured": {f"{k[0]}|lr{k[1]:g}|{k[2]}|{k[3]}": {"adapt_j": v[0], "total_j": v[1]} for k, v in meas.items()},
        "seeds": sorted(set(int(s) for s in runs.seed))}


# ----------------------------------------------------------------------------- 2. confirmation

def confirm_cells(runs):
    """{family or 'always': {dataset: dict(change=(mean, sd), gain=(mean, sd), pct, tag)}} for a confirm root.
    Changes are per seed against the frozen run of the same seed; pct is the seed-mean adaptation energy
    over the dataset's periodic_p1."""
    ch = per_seed_change(runs)
    share = energy_share(runs)
    out = {}
    for (d, tag), g in ch.groupby(["dataset", "tag"]):
        fam = "always" if tag == "periodic_p1" else tag.split("_")[0]
        if fam in out and d in out[fam]:
            continue   # a spec holds one setting per family; keep the first if not
        out.setdefault(fam, {})[d] = dict(change=mean_sd(g.change), gain=mean_sd(g.gain_pts), pct=share.get((d, tag), np.nan),
                                          tag=tag, seeds=sorted(int(s) for s in g.seed))
    return out


def best_marks(blocks, key="change", maximize=False):
    """For each dataset the family with the best mean among realizable ones (energy <= 10 % of always)."""
    best = {}
    for fams in blocks:
        for fam, byd in fams.items():
            if fam == "always":
                continue
            for d, c in byd.items():
                if c["pct"] <= 10.0 and not math.isnan(c[key][0]):
                    cur = best.get(d)
                    if cur is None or (c[key][0] > cur[1] if maximize else c[key][0] < cur[1]):
                        best[d] = (fam, c[key][0])
    return {d: v[0] for d, v in best.items()}


def confirm_table(backbones, path):
    """backbones: ordered {display name: (cells, kind)}, kind 'forecast' (MASE change, 4 streams) or 'vision'
    (accuracy gain in points, one column group)."""
    lines = [r"\begin{tabular}{ll " + "rr " * len(DATASETS) + "}", r"\toprule",
             "Backbone & Policy & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{SHORT[d]}}}" for d in DATASETS) + r" \\",
             " ".join(rf"\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(DATASETS))),
             " & & " + " & ".join(r"$\Delta$MASE (\%) & E (\%)" for _ in DATASETS) + r" \\", r"\midrule"]
    for name, (cells, kind) in backbones.items():
        vis = kind == "vision"
        best = best_marks([cells], "gain" if vis else "change", maximize=vis)
        ds = ["core50"] if vis else DATASETS
        order = ["always"] + [f for f in CONFIRM_FAMS + ("replay",) if f in cells]
        if vis:
            lines.append(rf"\multicolumn{{{2 + 2 * len(DATASETS)}}}{{l}}{{\emph{{{name}: accuracy gain over frozen (points); column 1 = CORe50}}}} \\")
        for i, fam in enumerate(order):
            row = ["" if vis else (name if i == 0 else ""), LABEL.get(fam, fam)]
            for j, d in enumerate(DATASETS):
                dd = ds[0] if (vis and j == 0) else d
                c = cells.get(fam, {}).get(dd) if (not vis or j == 0) else None
                if c is None:
                    row += ["--", "--"]
                    continue
                v = c["gain"] if vis else c["change"]
                s = cell(*v)
                if best.get(dd) == fam:
                    s = r"$\mathbf{" + s.strip("$") + "}$"
                row += [s, "--" if math.isnan(c["pct"]) else f"{c['pct']:.0f}"]
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    write(path, lines)


def section_confirm(out, roots, js):
    spec = [("PatchMixer (ours, 0.6M)", roots["confirm"], "forecast"), ("TTM-R2 (0.8M)", roots["confirm_ttm"], "forecast"),
            ("Chronos-Bolt Tiny (8.7M)", roots["confirm_chronos"], "forecast"), ("ViT-Tiny", roots["confirm_core50"], "vision")]
    blocks, summary = {}, {}
    for name, root, kind in spec:
        if not os.path.isdir(root) or load(root).empty:
            note(f"confirmation: {root} not present yet")
            continue
        runs = load(root)
        want = ["core50"] if kind == "vision" else DATASETS
        bad = [(d, t) for (d, t), g in runs.groupby(["dataset", "tag"]) if set(g.seed) != set(OPTS["confirm_seeds"])]
        if (set(runs.dataset) != set(want) or bad) and OPTS["strict"]:
            note(f"confirmation: {root} incomplete (streams {sorted(set(runs.dataset))}, {len(bad)} runs lack seeds "
                 f"{list(OPTS['confirm_seeds'])}); pass --allow-partial to tabulate anyway")
            continue
        blocks[name] = (confirm_cells(runs), kind)
        summary[name] = {fam: {d: {"tag": c["tag"], "change_mean": c["change"][0], "change_sd": c["change"][1],
                                   "gain_pts_mean": c["gain"][0], "gain_pts_sd": c["gain"][1], "energy_pct": c["pct"],
                                   "seeds": c["seeds"]} for d, c in byd.items()} for fam, byd in blocks[name][0].items()}
    if not blocks:
        return
    confirm_table(blocks, paper_file(out, "tab_confirm.tex"))
    js["confirm"] = summary


# ----------------------------------------------------------------------------- 3. protocol

def _version(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[0]
    except Exception:
        return None


def software_versions():
    import torch
    drv = _version(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if drv is None:
        print("[note] protocol: nvidia-smi unavailable here (login node); driver version reported as n/a")
    return {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda or "n/a",
            "driver": drv or "n/a"}


def protocol_stats(root):
    """Idle power (sum over domains), drift, closure, span and raw / incremental joules per category, over
    the summaries of a measured root; plus the counters recorded in one run config."""
    idle_pre, idle_post, drift, closure, span, flags, n = [], [], [], [], [], [], 0
    cats = {}
    cfg = None
    for f in glob.glob(os.path.join(root, "*", "*", "seed*", "summary.json")):
        e = json.load(open(f)).get("energy", {})
        if "p_idle_pre_w" not in e:
            continue
        n += 1
        idle_pre.append(sum(e["p_idle_pre_w"].values()))
        idle_post.append(sum(e["p_idle_post_w"].values()))
        drift.append(e["idle_drift"])
        closure.append(abs(e["closure"]))
        flags.append(bool(e["idle_drift_flag"]))
        span.append(e["span_s"])
        for c, v in e["system"].items():
            a = cats.setdefault(c, [0.0, 0.0])
            a[0] += v["raw"]
            a[1] += v["incr"]
        if cfg is None:
            cfg = yaml.safe_load(open(f.replace("summary.json", "config.yaml"))).get("energy", {})
    if not n:
        return None
    return dict(runs=n, idle_pre=idle_pre, idle_post=idle_post, drift=drift, closure=closure, span=span, flags=flags, cats=cats, cfg=cfg)


def _rng(x, d=1):
    return f"{np.mean(x):.{d}f} ({np.min(x):.{d}f}--{np.max(x):.{d}f})"


def protocol_table(stats, versions, path):
    devs = [d for d in ("cpu", "gpu") if stats.get(d)]
    head = {"cpu": "CPU runs", "gpu": "GPU runs"}
    lines = [r"\begin{tabular}{l " + "l" * len(devs) + "}", r"\toprule",
             "& " + " & ".join(head[d] for d in devs) + r" \\", r"\midrule"]

    def row(label, f):
        lines.append(label + " & " + " & ".join(f(stats[d]) for d in devs) + r" \\")

    def counters(s):
        out = []
        for c in s["cfg"].get("counters", []):
            if c["type"] == "rapl":
                out.append(f"RAPL {c['name'].split('_')[-1]}")
            elif c["type"] == "nvml":
                out.append("NVML energy")
        return ", ".join(dict.fromkeys(out)) or "--"

    lines.append(r"Device & " + " & ".join({"cpu": "Xeon Platinum 8358", "gpu": "A100-SXM-64GB + host Xeon"}[d] for d in devs) + r" \\")
    row("Counters", counters)
    lines.append(r"Counter resolution & " + " & ".join(
        {"cpu": r"RAPL: $\approx$1 ms update", "gpu": r"NVML: 100 ms refresh; RAPL: $\approx$1 ms"}[d] for d in devs) + r" \\")
    row("Sampler period (ms)", lambda s: ", ".join(sorted({f"{1e3 * c['sample_s']:g}" for c in s["cfg"]["counters"]})))
    row("Idle window before and after (s)", lambda s: f"{s['cfg']['idle_s']:g}")
    row("Runs", lambda s: str(s["runs"]))
    row("Idle power before, W, all domains (min--max)", lambda s: _rng(s["idle_pre"]))
    row("Idle power after, W, all domains (min--max)", lambda s: _rng(s["idle_post"]))
    row(r"Idle drift, \% (mean, max)", lambda s: f"{100 * np.mean(s['drift']):.2f}, {100 * np.max(s['drift']):.2f}")
    row("Runs flagged for idle drift", lambda s: str(int(sum(s["flags"]))))
    row(r"Closure error, max $|\cdot|$", lambda s: f"{np.max(s['closure']):.1e}")
    row("Run span, s (min--max)", lambda s: _rng(s["span"], 0))
    lines.append(r"\midrule")
    for c in ("inference", "decision", "adaptation", "other"):
        row(f"{c.capitalize()}: raw / incremental (kJ), share",
            lambda s, c=c: (f"{s['cats'][c][0] / 1e3:.2f} / {s['cats'][c][1] / 1e3:.2f}, "
                            f"{100 * s['cats'][c][1] / s['cats'][c][0]:.0f}\\%") if c in s["cats"] else "--")
    lines.append(r"\midrule")
    lines.append(rf"Software & \multicolumn{{{len(devs)}}}{{l}}{{Python {versions['python']}, torch {versions['torch']}, "
                 rf"CUDA {versions['cuda']}, driver {versions['driver']}}} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(path, lines)


def spread_table(rows, path):
    """rows: [(label, {kind: windows array})]; median and IQR across windows, in mJ."""
    lines = [r"\begin{tabular}{l rrr r}", r"\toprule",
             r"& Forecast (mJ) & Probe (mJ) & Update (mJ) & Windows \\",
             r"& median [IQR] & median [IQR] & median [IQR] & \\", r"\midrule"]
    for label, byk in rows:
        cells = []
        for k in ("infer", "probe", "update"):
            w = 1e3 * np.asarray(byk[k])
            q1, med, q3 = np.percentile(w, [25, 50, 75])
            cells.append(f"{med:.{2 if med < 10 else 0}f} [{q1:.{2 if med < 10 else 0}f}, {q3:.{2 if med < 10 else 0}f}]")
        lines.append(f"{label} & " + " & ".join(cells) + f" & {len(byk['update'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(path, lines)


def section_protocol(out, roots, js):
    stats = {dev: protocol_stats(os.path.join(roots["measured"], dev)) for dev in ("cpu", "gpu")
             if os.path.isdir(os.path.join(roots["measured"], dev))}
    stats = {k: v for k, v in stats.items() if v}
    if not stats:
        return note(f"protocol: no measured summaries under {roots['measured']}")
    versions = software_versions()
    protocol_table(stats, versions, paper_file(out, "tab_protocol.tex"))
    js["protocol"] = {"versions": versions, **{dev: {
        "runs": s["runs"], "idle_pre_w": [float(np.mean(s["idle_pre"])), float(np.min(s["idle_pre"])), float(np.max(s["idle_pre"]))],
        "idle_post_w": [float(np.mean(s["idle_post"])), float(np.min(s["idle_post"])), float(np.max(s["idle_post"]))],
        "idle_drift_mean": float(np.mean(s["drift"])), "idle_drift_flags": int(sum(s["flags"])), "idle_drift_max": float(np.max(s["drift"])),
        "closure_max": float(np.max(s["closure"])), "span_s": [float(np.min(s["span"])), float(np.max(s["span"]))],
        "joules": {c: {"raw": v[0], "incr": v[1], "incr_share": v[1] / v[0]} for c, v in s["cats"].items()}}
        for dev, s in stats.items()}}
    # per-call spread across windows
    rows, spread = [], {}
    specs = [(n, dev, r[i]) for n, r in BACKBONES.items() for dev, i in (("CPU", 1), ("GPU", 2))] + [
        ("ViT-Tiny, CORe50", "CPU", "results/measured_vision_percall/cpu"),
        ("ViT-Tiny, CORe50", "GPU", "results/measured_vision_percall/gpu")]
    for name, dev, root in specs:
        if not os.path.isdir(root):
            continue
        t = per_call_table(root)
        if t.empty:
            continue
        byk = {k: t[t.kind == k].j.values for k in ("infer", "probe", "update")}
        rows.append((f"{name}, {dev}", byk))
        spread[f"{name} {dev}"] = {"windows": int(len(byk["update"])),
                                   **{k: [float(np.median(v)), *[float(x) for x in np.percentile(v, [25, 75])]]
                                      for k, v in byk.items()},
                                   "within_stream_iqr_max": {k: float(max(np.subtract(*np.percentile(g.j, [75, 25]))
                                                                            for _, g in t[t.kind == k].groupby("dataset")))
                                                             for k in ("infer", "probe", "update")}}
    if rows:
        spread_table(rows, paper_file(out, "tab_percall_spread.tex"))
        js["protocol"]["per_call_spread_j"] = spread


# ----------------------------------------------------------------------------- 4. hyperparameters

def _read_cfg(root, dataset, tag="frozen"):
    f = os.path.join(root, dataset, tag, "seed0", "config.yaml")
    return yaml.safe_load(open(f)) if os.path.exists(f) else None


def _adapter_params(root, dataset):
    for tag in ("periodic_p1", "frozen"):
        f = os.path.join(root, dataset, tag, "seed0", "summary.json")
        if os.path.exists(f) and "adapter_params" in json.load(open(f)):
            return json.load(open(f))["adapter_params"]
    return None


def _thousands(n):
    return "--" if n is None else f"{n:,}".replace(",", r"\,")


def hparam_columns(roots):
    """{column: dict of rows} read from the merged run configs of each backbone's evaluation runs."""
    cols = {}
    spec = [("PatchMixer", roots["main"], "ettm1", "0.6M", "5 public datasets (ours)", ("feat\\_mlp.fc1, fc2",)),
            ("TTM-R2", "results/ttm", "ettm1", "0.8M", "Granite TTM-R2 release (public corpora)", None),
            ("Chronos-Bolt", "results/chronos", "ettm1", "8.7M", "Chronos-Bolt release (public corpora)", None)]
    for name, root, ds, size, pre, targets in spec:
        c = _read_cfg(root, ds)
        if c is None:
            continue
        L = c["learner"]
        t = targets[0] if targets else f"{len(L['lora_targets'])} modules per adapted block"
        cols[name] = {"Input / output": f"{c['model']['context']} / {c['model']['horizon']} steps",
                      "Stride / batch": str(c["dataset"]["stride"]), "Parameters": size,
                      "Trainable (adapter)": _thousands(_adapter_params(root, ds)),
                      "Pretraining data": pre, "Adapted blocks": str(L["adapt_blocks"]), "LoRA targets": t,
                      "LoRA rank / alpha": f"{L['rank']} / {L['alpha']}", "Optimizer": "Adam", "Learning rate": f"{L['lr']:g}",
                      "Steps per update": str(L["steps_per_update"]), "Buffer (steps)": str(L["buffer_steps"]),
                      "Prefix cache": "yes" if L["cache_prefix"] else "no", "Stream length (steps)": str(c["dataset"]["n_steps"])}
    vis = [(roots["main"], "core50"), (roots["domains"], "pacs"), (roots["domains"], "officehome")]
    cfgs = [(_read_cfg(r, d), _adapter_params(r, d)) for r, d in vis]
    vis_len = []
    for r, d in vis:
        f = os.path.join(r, d, "periodic_p1", "seed0", "trace.npz")
        vis_len.append(len(np.load(f)["err"]) if os.path.exists(f) else "--")
    if cfgs[0][0] is not None:
        c = cfgs[0][0]
        L = c["learner"]
        cols["ViT-Tiny"] = {"Input / output": f"{c['model']['img_size']}$\\times${c['model']['img_size']} image / class",
                            "Stride / batch": f"batch {c['dataset']['batch']}", "Parameters": "5.7M",
                            "Trainable (adapter)": "/".join(_thousands(p) for _, p in cfgs),
                            "Pretraining data": "ImageNet (21k, fine-tuned 1k), then base domains", "Adapted blocks": str(L["adapt_blocks"]),
                            "LoRA targets": f"attn.qkv, attn.proj, mlp.fc1, fc2, head", "LoRA rank / alpha": f"{L['rank']} / {L['alpha']}",
                            "Optimizer": "Adam", "Learning rate": f"{L['lr']:g}", "Steps per update": str(L["steps_per_update"]),
                            "Buffer (steps)": str(L["buffer_steps"]), "Prefix cache": "yes" if L["cache_prefix"] else "no",
                            "Stream length (steps)": "/".join(str(n) for n in vis_len)}
    return cols


def hparam_table(cols, path):
    rows = list(next(iter(cols.values())))
    names = list(cols)
    lines = [r"\begin{tabular}{l " + "l" * len(names) + "}", r"\toprule", " & " + " & ".join(names) + r" \\", r"\midrule"]
    for r in rows:
        lines.append(r + " & " + " & ".join(cols[n].get(r, "--") for n in names) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(path, lines)


def section_hparams(out, roots, js):
    cols = hparam_columns(roots)
    if not cols:
        return note("hyperparameters: no evaluation run configs found")
    hparam_table(cols, paper_file(out, "tab_hparams.tex"))
    js["hparams"] = cols


# ----------------------------------------------------------------------------- 5. recovery

def domain_switch_steps(name, data_root=None, cfg_dir="configs/datasets"):
    """First step of every streamed domain after the first, as DomainStream lays them out: each domain
    contributes floor(non-test images / batch) steps. Returns (switch steps, total steps) or None."""
    d = yaml.safe_load(open(os.path.join(cfg_dir, f"{name}.yaml")))
    root = data_root or d["root"]
    n = []
    for dom in d["domains"]:
        f = os.path.join(root, f"{dom}_test.npy")
        if not os.path.exists(f):
            return None
        n.append(int((~np.load(f)).sum() // d["batch"]))
    return np.cumsum(n)[:-1], int(sum(n))


def trace_errs(roots, dataset, tag):
    """Per-seed online-error traces of a run, from the first root that holds it."""
    for r in ([roots] if isinstance(roots, str) else roots):
        fs = sorted(glob.glob(os.path.join(r, dataset, tag, "seed*", "trace.npz")))
        if fs:
            return [np.load(f)["err"] for f in fs]
    return []


def window_acc(errs, starts, window):
    ends = list(starts[1:]) + [len(errs[0])]
    return float(np.mean([1 - after_switch(e, starts, window, ends) for e in errs]))


def recovery_vision(roots):
    """{dataset: {policy: {10: acc, 50: acc}}} (accuracy in %), plus the traces for the figure."""
    sets = {"CORe50": ("core50", (roots["main"], roots["baselines"]), "core50"),
            "PACS": ("pacs", roots["domains"], "pacs"), "Office-Home": ("officehome", roots["domains"], "officehome")}
    res, traces, skipped = {}, {}, {}
    for title, (name, troot, pk) in sets.items():
        if name == "core50":
            if not os.path.isdir(os.path.join(roots["main"], "core50")):
                skipped[title] = "no core50 runs"
                continue
            starts = switch_steps(yaml.safe_load(open("configs/datasets/core50.yaml")))
        else:
            if not os.path.isdir(os.path.join(roots["domains"], name)):
                skipped[title] = f"no {name} runs"
                continue
            sw = domain_switch_steps(name)
            if sw is None:
                skipped[title] = "dataset files (test masks) not found, switch steps not recoverable"
                continue
            starts, total = sw
        try:
            p = picks(pk)
        except Exception as e:   # a missing root / tag set: leave the dataset out
            skipped[title] = f"picks failed: {e}"
            continue
        res[title], traces[title] = {}, {"starts": [int(s) for s in starts]}
        for pol, tag in p.items():
            errs = trace_errs(troot, name, tag) if tag else []
            if not errs:
                res[title][pol] = None
                continue
            if name != "core50" and len(errs[0]) != total:
                res[title][pol] = None
                skipped[title] = f"trace length {len(errs[0])} != recomputed stream length {total}"
                continue
            res[title][pol] = {"tag": tag, **{w: 100 * window_acc(errs, starts, w) for w in (10, 50)}}
            traces[title][pol] = np.mean(errs, axis=0)
    return res, traces, skipped


def recovery_table(res, path):
    sets = list(res)
    order = ["frozen", "always", "periodic", "drift", "vou", "vougerr", "vougin", "carat", "caract", "upf", "d3a", "replay"]
    lines = [r"\begin{tabular}{l " + "rr " * len(sets) + "}", r"\toprule",
             "& " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{t}}}" for t in sets) + r" \\",
             " ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(sets))),
             "Policy & " + " & ".join("10 steps & 50 steps" for _ in sets) + r" \\", r"\midrule"]
    for pol in order:
        if not any(res[t].get(pol) for t in sets):
            continue   # e.g. CARA while its rerun is pending
        row = [LABEL.get(pol, pol)]
        for t in sets:
            r = res[t].get(pol)
            row += [f"{r[10]:.1f}", f"{r[50]:.1f}"] if r else ["--", "--"]
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(path, lines)


def trajectories_fig(res, traces, path):
    pols = [p for p in ("frozen", "always", "periodic", "vou", "d3a", "replay")]
    sty = {"frozen": (MUTED, ":"), "always": (INK, "-"), "replay": ("#0b6b4a", "--")}
    sets = list(traces)
    fig, axes = plt.subplots(1, len(sets), figsize=(3.8 * len(sets), 3.0), squeeze=False)
    for ax, t in zip(axes[0], sets):
        for pol in pols:
            if pol not in traces[t]:
                continue
            acc = 100 * (1 - pd.Series(traces[t][pol]).rolling(25, min_periods=1).mean().values)
            col, ls = sty.get(pol, (COLOR.get(pol, INK2), "-"))
            ax.plot(acc, color=col, ls="--" if pol == "replay" else ls, lw=1.3, label=LABEL.get(pol, pol))
        for s in traces[t]["starts"]:
            ax.axvline(s, color=MUTED, lw=0.6, ls=":")
        ax.set_title(t, loc="left", fontsize=9, color=INK)
        ax.set_xlabel("Stream step (dotted: domain switch)")
    axes[0][0].set_ylabel("Online accuracy (%), 25-step mean")
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=6, fontsize=7.5, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path)
    return fig


def synth_regime_switches(root, dataset="synthetic"):
    """Step indices at which the synthetic stream changes regime (as analyze.trace_fig reads them)."""
    from datasets.synthetic import regime_switching
    cfg = yaml.safe_load(open(os.path.join(root, dataset, "frozen", "seed0", "config.yaml")))
    dc = cfg["dataset"]
    _, labels = regime_switching(**dc["synthetic"])
    n = len(np.load(os.path.join(root, dataset, "frozen", "seed0", "trace.npz"))["err"])
    reg = labels[dc["start"] + np.arange(n) * dc["stride"] + cfg["model"]["context"]]
    return np.flatnonzero(np.diff(reg)) + 1, n


def pick_within(m, fam, frac):
    """Tag of the family's lowest-error setting with adaptation energy <= frac of periodic_p1, or None."""
    e_max = m[m.tag == "periodic_p1"].adapt_j.iloc[0]
    g = m[(m.family == fam) & (m.adapt_j <= frac * e_max)]
    return None if g.empty else g.sort_values("mase").tag.iloc[0]


def recovery_synth(roots, window=20):
    rts = [r for r in (roots["main"], roots["baselines"]) if os.path.isdir(os.path.join(r, "synthetic"))]
    if not rts:
        return None
    m = seed_mean(pd.concat([load(r) for r in rts], ignore_index=True))
    m = m[m.dataset == "synthetic"]
    starts, n = synth_regime_switches(roots["main"])
    ends = list(starts[1:]) + [n]
    chosen = {"frozen": "frozen", "always": "periodic_p1"}
    for f in FAMS + BASE_FAMS:
        chosen[f] = pick_within(m, f, 0.10) or pick_within(m, f, 0.25)
    e_max = m[m.tag == "periodic_p1"].adapt_j.iloc[0]
    out = {}
    for pol, tag in chosen.items():
        errs = trace_errs(rts, "synthetic", tag) if tag else []
        if not errs:
            out[pol] = None
            continue
        out[pol] = {"tag": tag, "energy_pct": 100 * float(m[m.tag == tag].adapt_j.iloc[0]) / e_max,
                    "window": float(np.mean([after_switch(e, starts, window, ends) for e in errs])),
                    "overall": float(np.mean([e.mean() for e in errs]))}
    fz = out["frozen"]["window"]
    for v in out.values():
        if v:
            v["change"] = rel(v["window"], fz)
    return out, [int(s) for s in starts], window


def recovery_synth_table(res, path):
    lines = [r"\begin{tabular}{l l r rr}", r"\toprule",
             r"Policy & Setting & E (\%) & MASE, 20 steps after switch & Change vs frozen (\%) \\", r"\midrule"]
    for pol, v in res.items():
        if v is not None:
            lines.append(f"{LABEL.get(pol, pol)} & {tex_escape(v['tag'])} & {v['energy_pct']:.1f} & "
                         f"{v['window']:.3f} & {v['change']:+.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(path, lines)


def section_recovery(out, roots, png, js):
    if not os.path.isdir(roots["main"]):
        return note(f"recovery: {roots['main']} not present")
    rec = {}
    try:
        res, traces, skipped = recovery_vision(roots)
    except Exception as e:
        res, traces, skipped = {}, {}, {"vision": repr(e)}
    for k, v in skipped.items():
        print(f"[note] recovery (vision): {k}: {v}")
    if res:
        recovery_table(res, paper_file(out, "tab_recovery.tex"))
        fig = trajectories_fig(res, traces, paper_file(out, "fig_trajectories.pdf"))
        if png:
            fig.savefig(os.path.join(png, "fig_trajectories.png"))
        plt.close(fig)
        rec["vision"] = {t: {p: v for p, v in d.items()} for t, d in res.items()}
        rec["vision_switch_steps"] = {t: traces[t]["starts"] for t in traces}
    else:
        note("recovery (vision): no vision runs found")
    try:
        sy = recovery_synth(roots)
    except Exception as e:
        sy = None
        print(f"[note] recovery (synthetic): {e!r}")
    if sy:
        recovery_synth_table(sy[0], paper_file(out, "tab_recovery_synth.tex"))
        rec["synthetic"] = {"policies": sy[0], "switch_steps": sy[1], "window": sy[2]}
    else:
        note("recovery (synthetic): no synthetic runs found")
    if rec:
        js["recovery"] = rec


# ----------------------------------------------------------------------------- 6. VoU conditions

VOU_FAMS = ("vou", "vougerr", "vougin")


def vou_pick(m, frac=0.10):
    """(tag, family) of the lowest-error VoU-family setting within frac of always-update energy."""
    best = None
    for f in VOU_FAMS:
        t = pick_within(m, f, frac)
        if t:
            mase = float(m[m.tag == t].mase.iloc[0])
            if best is None or mase < best[2]:
                best = (t, f, mase)
    return best


def vou_conditions(backbones):
    """backbones: {name: (analytic root, cpu percall root, gpu percall root)} -> {name: {dataset: dict}}."""
    out = {}
    for name, (root, cpu, gpu) in backbones.items():
        if not os.path.isdir(root):
            continue
        m = seed_mean(load(root))
        rows = budget_rows({name: m}, budgets=(0.10,), fams=("periodic",) + VOU_FAMS).get(name, {})
        costs = {dev: per_call_costs(r) if os.path.isdir(r) else {} for dev, r in (("cpu", cpu), ("gpu", gpu))}
        proj = projected_runs(root, costs["gpu"]) if costs["gpu"] else None
        noise = None
        if os.path.exists(os.path.join(root, "synthetic", "frozen", "seed0", "trace.npz")):
            noise = noise_share_all(root)
        out[name] = {}
        for d, r in rows.items():
            sub = m[m.dataset == d]
            vals = {f: r[f][0] for f in VOU_FAMS if not np.isnan(r[f][0])}
            best = min(vals, key=vals.get) if vals else None
            row = dict(periodic=r["periodic"][0], vou_best=vals[best] if best else float("nan"), vou_family=best,
                       advantage=(vals[best] - r["periodic"][0]) if best else float("nan"),
                       noise_share=float(noise[1]) if (noise is not None and d == "synthetic") else float("nan"))
            for dev in ("cpu", "gpu"):
                c = costs[dev].get(d)
                row[f"probe_over_update_{dev}"] = c["probe"] / c["update"] if c else float("nan")
            row["gpu_probe_share"] = float("nan")
            pk = vou_pick(sub)
            if pk and proj is not None and costs["gpu"].get(d):
                p = proj[(proj.dataset == d) & (proj.tag == pk[0])]
                if len(p):
                    probe_j = (p.probes * costs["gpu"][d]["probe"]).mean()
                    row["gpu_probe_share"] = float(100 * probe_j / p.adapt_j.mean()) if p.adapt_j.mean() > 0 else float("nan")
                row["pick_tag"] = pk[0]
            if noise is not None and d == "synthetic" and pk:
                nsdf = noise[0]
                hit = nsdf[nsdf.tag == pk[0]]
                row["pick_update_noise_share"] = float(hit.share.iloc[0]) if len(hit) else float("nan")
            out[name][d] = row
    return out


def vou_table(cond, path):
    def f(x, spec):
        return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, spec)

    lines = [r"\begin{tabular}{ll r r rr r}", r"\toprule",
             r"& & VoU-family best $-$ periodic & Noise-burst & \multicolumn{2}{c}{Probe / update} & GPU probe share \\",
             r"Backbone & Stream & at 10\% (points) & share of steps (\%) & CPU & GPU & at 10\% (\%) \\", r"\midrule"]
    for name, byd in cond.items():
        for i, (d, r) in enumerate(byd.items()):
            lines.append(f"{name if i == 0 else ''} & {SHORT[d]} & {f(r['advantage'], '+.1f')} & "
                         f"{f(100 * r['noise_share'], '.0f')} & {f(r['probe_over_update_cpu'], '.3f')} & "
                         f"{f(r['probe_over_update_gpu'], '.3f')} & {f(r['gpu_probe_share'], '.0f')} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    write(path, lines)


def section_vou(out, roots, js):
    bb = {n: (roots["main"] if i == 0 else r[0], *r[1:]) for i, (n, r) in enumerate(BACKBONES.items())}
    cond = vou_conditions(bb)
    if not cond:
        return note("VoU conditions: no backbone roots present")
    vou_table(cond, paper_file(out, "tab_vou_conditions.tex"))
    js["vou_conditions"] = cond


# ----------------------------------------------------------------------------- 7. seeds

def seed_changes(runs, frozen_runs, tags):
    """{(dataset, tag): (mean, sd, n)} of the per-seed % MASE change against the frozen run of that seed."""
    fz = frozen_runs.set_index(["dataset", "seed"]).mase
    out = {}
    for (d, t), g in runs[runs.tag.isin(tags)].groupby(["dataset", "tag"]):
        ch = [rel(r.mase, fz[(d, r.seed)]) for r in g.itertuples() if (d, r.seed) in fz.index]
        mu, sd = mean_sd(ch)
        out[(d, t)] = (mu, sd, len(ch))
    return out


def main_spread(roots, fams=FAMS + ("carat", "caract", "upf", "d3a")):
    """Seed spread of the main budget tables: for each backbone (a root or a tuple of roots, pooled), stream and family, the per-seed % MASE
    change of the family's lowest-error setting within 10 % of always-update energy (seed-mean pick).
    -> {backbone: {"cells": {"dataset|family": [mean, sd]}, "sd_median", "sd_max"}}."""
    out = {}
    for name, root in roots.items():
        rs = [r for r in ((root,) if isinstance(root, str) else root) if os.path.isdir(r)]
        if not rs:
            continue
        runs = pd.concat([load(r) for r in rs], ignore_index=True)
        runs = runs[runs.dataset.isin(DATASETS)]       # forecasting streams only
        m = seed_mean(runs)
        cells = {}
        for d in sorted(set(m.dataset)):
            md = m[m.dataset == d]
            if not (md.tag == "periodic_p1").any() or not (md.tag == "frozen").any():
                continue
            for f in fams:
                t = pick_within(md, f, 0.10)
                if t:
                    mu, sd, _ = seed_changes(runs[runs.dataset == d], runs[runs.tag == "frozen"], {t})[(d, t)]
                    cells[f"{d}|{f}"] = [mu, sd]
        sds = [v[1] for v in cells.values() if not math.isnan(v[1])]
        out[name] = {"cells": cells, "sd_median": float(np.median(sds)) if sds else float("nan"),
                     "sd_max": float(max(sds)) if sds else float("nan")}
    return out


def seeds_table(cons, depth, rows_cons, rows_depth, path):
    """cons / depth: {(dataset, tag): (mean, sd, n)}; rows_*: [(label, {dataset: tag})]."""
    lines = [r"\begin{tabular}{l " + "r" * len(DATASETS) + "}", r"\toprule",
             "& " + " & ".join(SHORT[d] for d in DATASETS) + r" \\", r"\midrule",
             rf"\multicolumn{{{1 + len(DATASETS)}}}{{l}}{{\emph{{Conservative learner: MASE change vs frozen (\%), mean $\pm$ sd over seeds}}}} \\"]
    ns = []
    for label, tags in rows_cons:
        cs = []
        for d in DATASETS:
            v = cons.get((d, tags.get(d))) if tags.get(d) else None
            cs.append(cell(v[0], v[1]) if v else "--")
            if v:
                ns.append(v[2])
        lines.append(f"{label} & " + " & ".join(cs) + r" \\")
    if rows_depth:
        lines += [r"\midrule", rf"\multicolumn{{{1 + len(DATASETS)}}}{{l}}{{\emph{{Adapter depth (cached prefix): MASE change vs frozen (\%), mean $\pm$ sd over seeds}}}} \\"]
        for label, tags in rows_depth:
            cs = []
            for d in DATASETS:
                v = depth.get((d, tags.get(d))) if tags.get(d) else None
                cs.append(cell(v[0], v[1]) if v else "--")
                if v:
                    ns.append(v[2])
            lines.append(f"{label} & " + " & ".join(cs) + r" \\")
    lines += [r"\midrule", rf"Seeds per cell (min) & \multicolumn{{{len(DATASETS)}}}{{c}}{{{min(ns) if ns else '--'}}} \\",
              r"\bottomrule", r"\end{tabular}"]
    write(path, lines)


def section_seeds(out, roots, js):
    have_c, have_d = os.path.isdir(roots["conservative"]), os.path.isdir(roots["depth"])
    if not have_c:
        return note(f"seeds: {roots['conservative']} not present")
    rc = load(roots["conservative"])
    m = seed_mean(rc)
    rows_c = [("Update every step", {d: "periodic_p1" for d in DATASETS}),
              ("Periodic, every 4", {d: "periodic_p4" for d in DATASETS}),
              ("Periodic, every 16", {d: "periodic_p16" for d in DATASETS}),
              ("Drift trigger (best within 10\\%)", {d: pick_within(m[m.dataset == d], "drift", 0.10) for d in DATASETS if (m.dataset == d).any()}),
              ("VoU (best within 10\\%)", {d: pick_within(m[m.dataset == d], "vou", 0.10) for d in DATASETS if (m.dataset == d).any()})]
    cons = seed_changes(rc, rc[rc.tag == "frozen"], {t for _, tg in rows_c for t in tg.values() if t})
    depth, rows_d = {}, []
    if have_d:
        rd = load(roots["depth"])
        main = load(roots["main"])
        ks = sorted(set(int(re.search(r"_k(\d+)_", t).group(1)) for t in rd.tag))
        rows_d = [(f"{k} adapted block{'s' if k != 1 else ''}" if k else "Head only",
                   {d: f"depth_k{k}_cache" for d in ("ettm1", "electricity")}) for k in ks]
        depth = seed_changes(rd, main[main.tag == "frozen"], {f"depth_k{k}_cache" for k in ks})
    else:
        note(f"seeds: {roots['depth']} not present; depth rows omitted")
    ns = [v[2] for v in list(cons.values()) + list(depth.values())]
    if ns and min(ns) < OPTS["seeds_n"] and OPTS["strict"]:
        return note(f"seeds: some cells have only {min(ns)} of {OPTS['seeds_n']} seeds (jobs still running?); "
                    f"pass --allow-partial to tabulate anyway")
    seeds_table(cons, depth, rows_c, rows_d, paper_file(out, "tab_seeds.tex"))
    js["main_spread"] = main_spread({"PatchMixer (ours, 0.6M)": (roots["main"], roots["baselines"]),
                                     **{n: (BACKBONES[n][0], BASELINES[n]) for n in list(BACKBONES)[1:]}})
    js["seeds"] = {"conservative": {f"{d}|{t}": {"mean": v[0], "sd": v[1], "n": v[2]} for (d, t), v in cons.items()},
                   "depth": {f"{d}|{t}": {"mean": v[0], "sd": v[1], "n": v[2]} for (d, t), v in depth.items()}}


# ----------------------------------------------------------------------------- main

def run_all(out, roots=None, png=None, **opts):
    roots = {**ROOTS, **(roots or {})}
    OPTS.update(opts)
    del NOTES[:]
    os.makedirs(out, exist_ok=True)
    if png:
        os.makedirs(png, exist_ok=True)
    js = {}
    for name, fn in (("granularity", lambda: section_granularity(out, roots, png, js)),
                     ("confirmation", lambda: section_confirm(out, roots, js)),
                     ("protocol", lambda: section_protocol(out, roots, js)),
                     ("hyperparameters", lambda: section_hparams(out, roots, js)),
                     ("recovery", lambda: section_recovery(out, roots, png, js)),
                     ("vou conditions", lambda: section_vou(out, roots, js)),
                     ("seeds", lambda: section_seeds(out, roots, js))):
        print(f"== {name}")
        fn()
    js["skipped"] = NOTES
    json.dump(js, open(paper_file(out, "revision.json"), "w"), indent=1, default=float)
    return js


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper")
    ap.add_argument("--png", default=None, help="also save new figures as PNG into this directory")
    ap.add_argument("--root", action="append", default=[], metavar="NAME=PATH",
                    help=f"override a result root; names: {', '.join(ROOTS)}")
    ap.add_argument("--allow-partial", action="store_true",
                    help="tabulate sections whose inputs are only partly produced (default: skip them with a note)")
    args = ap.parse_args()
    run_all(args.out, dict(r.split("=", 1) for r in args.root), args.png, strict=not args.allow_partial)


if __name__ == "__main__":
    main()
