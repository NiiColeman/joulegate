"""Measured-energy results: frontiers and budget tables at measured joules, improvement
per joule, measured vs analytic energy, and CPU vs GPU.

Usage: python -m analysis.analyze_measured [--measured results/measured] [--analytic results/main] [--out results/analysis_measured]
Energies are incremental system joules (idle power subtracted, all domains summed).
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis import analyze
from analysis.analyze import budget_table, frontier_fig, latex_budget, load, seed_mean

DEVICES = ("cpu", "gpu")


def improvement_per_joule(m):
    """(MASE_frozen - MASE) / adaptation joules, per adaptive config. NaN where energy <= 0."""
    rows = []
    for d, sub in m.groupby("dataset"):
        frozen = sub[sub.family == "frozen"]
        if frozen.empty:
            continue
        ref = frozen.mase.iloc[0]
        for r in sub[sub.n_updates > 0].itertuples():
            rows.append(dict(dataset=d, tag=r.tag, adapt_j=r.adapt_j, mase=r.mase,
                             gain_per_j=(ref - r.mase) / r.adapt_j if r.adapt_j > 0 else np.nan))
    return pd.DataFrame(rows)


def nonpositive(m):
    """Adaptive configs whose incremental energy is <= 0: within noise of idle, not plottable on a log axis."""
    return m[(m.n_updates > 0) & (m.adapt_j <= 0)][["dataset", "tag", "adapt_j"]]


def positive_only(m):
    """Frozen plus adaptive configs with positive incremental energy. Used for both the frontier
    and the budget table, so a config within noise of idle is never counted as free."""
    return m[(m.family == "frozen") | (m.adapt_j > 0)]


def event_energy(root):
    """Mean incremental system joules per top-level event, by kind."""
    rows = []
    for f in glob.glob(os.path.join(root, "*", "*", "seed*", "events.npz")):
        dataset, tag, _ = f.split(os.sep)[-4:-1]
        ev = np.load(f)
        incr = sum(ev[k] for k in ev.files if k.startswith("incr_"))
        top = ev["parent"] < 0
        for kind in ("infer", "store", "decide", "update"):
            m = top & (ev["kind"] == kind)
            if m.any():
                rows.append(dict(dataset=dataset, tag=tag, kind=kind, n=int(m.sum()), mean_j=float(incr[m].mean())))
    return pd.DataFrame(rows)


def frozen_difference(root):
    """Cross-check of per-window attribution against whole-run totals.

    With a slow counter (NVML refreshes every ~100 ms) energy inside short windows is smeared
    over neighbouring events, but the total over the whole run is exact. span_diff_j is a run's
    total incremental energy minus the frozen run's on the same stream; it also contains buffer
    evaluation and loop overhead, so ratios a little above 1 are expected. flag marks configs
    whose window-attributed adapt_j is off by more than a factor of 2 either way.
    """
    rows = []
    for f in glob.glob(os.path.join(root, "*", "*", "seed*", "summary.json")):
        dataset, tag, seed = f.split(os.sep)[-4:-1]
        s = json.load(open(f))
        total = sum(c["incr"] for c in s["energy"]["system"].values())
        rows.append(dict(dataset=dataset, tag=tag, seed=seed, adapt_j=s["adapt_j"], total_j=total))
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    frozen = t[t.tag == "frozen"].set_index(["dataset", "seed"]).total_j
    t = t[t.tag != "frozen"].copy()
    t["span_diff_j"] = [r.total_j - frozen.get((r.dataset, r.seed), np.nan) for r in t.itertuples()]
    t["ratio"] = t.span_diff_j / t.adapt_j.where(t.adapt_j > 0)
    t["flag"] = ~t.ratio.between(0.5, 2.0)
    return t.reset_index(drop=True)


def per_call_table(root):
    """Exact incremental joules per forecast, probe and update (warm-up windows of back-to-back
    calls), and each as a fraction of an update. Robust on GPU, unlike per-event attribution."""
    rows = []
    for f in glob.glob(os.path.join(root, "*", "*", "seed*", "summary.json")):
        dataset, tag, seed = f.split(os.sep)[-4:-1]
        pc = json.load(open(f))["energy"].get("per_call")
        for kind, c in (pc or {}).items():
            rows.append(dict(dataset=dataset, tag=tag, seed=seed, kind=kind, j=c["j"], calls=c["calls"],
                             per_update=c["j"] / pc["update"]["j"]))
    return pd.DataFrame(rows)


def per_call_costs(root):
    """{dataset: {kind: median measured joules per call}} over all runs under root."""
    t = per_call_table(root)
    return {d: {k: float(g[g.kind == k].j.median()) for k in sorted(g.kind.unique())}
            for d, g in t.groupby("dataset")}


def count_probes(summary_path):
    """Gradient probes in the stream loop: from the event log when there is one (measured runs, whose
    trace holds provisional joules that can be 0 or negative), else steps with probe energy (analytic)."""
    ev = summary_path.replace("summary.json", "events.npz")
    if os.path.exists(ev):
        e = np.load(ev)
        return int(((e["kind"] == "probe") & (e["step"] >= 0)).sum())  # step -1: warm-up per-call windows
    return int((np.load(summary_path.replace("summary.json", "trace.npz"))["probe_j"] > 0).sum())


def projected_runs(root, costs):
    """Adaptation energy of every run under root, priced at measured per-call costs: (updates +
    calibration) x update cost + probes x probe cost. Exact counts times exact per-call joules;
    used where per-run measurement is below the noise floor (sparse runs on GPU)."""
    rows = []
    for f in glob.glob(os.path.join(root, "*", "*", "seed*", "summary.json")):
        dataset, tag, seed = f.split(os.sep)[-4:-1]
        if dataset not in costs:
            continue
        s = json.load(open(f))
        probes = count_probes(f)
        calib = int(s.get("calib_j", 0.0) > 0)
        c = costs[dataset]
        rows.append(dict(dataset=dataset, tag=tag, seed=int(seed[4:]), family=tag.split("_")[0], mase=s.get("mase", s.get("error")),
                         updates=s["n_updates"], probes=probes, calib=calib,
                         adapt_j=(s["n_updates"] + calib) * c["update"] + probes * c["probe"]))
    return pd.DataFrame(rows)


def idle_table(root):
    """Idle power per run and domain, with the drift flag."""
    rows = []
    for f in glob.glob(os.path.join(root, "*", "*", "seed*", "summary.json")):
        dataset, tag, _ = f.split(os.sep)[-4:-1]
        e = json.load(open(f))["energy"]
        for dom, w in e["p_idle_w"].items():
            rows.append(dict(dataset=dataset, tag=tag, domain=dom, idle_w=w, drift_flag=e["idle_drift_flag"]))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measured", default="results/measured")
    ap.add_argument("--analytic", default="results/main")
    ap.add_argument("--out", default="results/analysis_measured")
    args = ap.parse_args()
    extra, per_dev = {}, {}
    analytic = seed_mean(load(args.analytic)) if os.path.isdir(args.analytic) else None
    for dev in DEVICES:
        root = os.path.join(args.measured, dev)
        if not os.path.isdir(root):
            continue
        out = os.path.join(args.out, dev)
        os.makedirs(out, exist_ok=True)
        m = seed_mean(load(root))
        per_dev[dev] = m
        m.to_csv(os.path.join(out, "summary_by_config.csv"), index=False)
        bad = nonpositive(m)
        if len(bad):
            bad.to_csv(os.path.join(out, "nonpositive_energy.csv"), index=False)
            print(f"[{dev}] {len(bad)} adaptive configs with incremental energy <= 0 (not on the log-scale frontier):")
            print(bad.to_string(index=False))
        frontier_fig(positive_only(m), out)
        analyze.ROOT_RESULTS = root
        bt = budget_table(positive_only(m))  # excluded configs are listed in nonpositive_energy.csv
        bt.to_csv(os.path.join(out, "budget_table.csv"), index=False)
        latex_budget(bt, os.path.join(out, "tab_budget.tex"))
        improvement_per_joule(m).to_csv(os.path.join(out, "improvement_per_joule.csv"), index=False)
        event_energy(root).to_csv(os.path.join(out, "event_energy.csv"), index=False)
        fd = frozen_difference(root)
        fd.to_csv(os.path.join(out, "frozen_difference.csv"), index=False)
        per_call_table(root).to_csv(os.path.join(out, "per_call.csv"), index=False)
        idle = idle_table(root)
        idle.to_csv(os.path.join(out, "idle.csv"), index=False)
        info = {"runs": int(len(m)), "idle_drift_flags": int(idle.drift_flag.sum()) if len(idle) else 0,
                "nonpositive_configs": int(len(bad)),
                "attribution_flags": int(fd.flag.sum()) if len(fd) else 0}
        if analytic is not None:
            j = m.merge(analytic[["dataset", "tag", "j_per_update", "infer_j"]], on=["dataset", "tag"],
                        suffixes=("", "_analytic"))
            j[["dataset", "tag", "n_updates", "j_per_update", "j_per_update_analytic", "infer_j",
               "infer_j_analytic"]].to_csv(os.path.join(out, "measured_vs_analytic.csv"), index=False)
            j = j[j.n_updates > 0]
            if len(j) > 2:
                info["spearman_measured_vs_analytic_j_per_update"] = float(spearmanr(j.j_per_update, j.j_per_update_analytic)[0])
                info["n_runs_for_spearman"] = int(len(j))
        extra[dev] = info
    if len(per_dev) == 2:
        c = per_dev["cpu"].merge(per_dev["gpu"], on=["dataset", "tag"], suffixes=("_cpu", "_gpu"))
        c[["dataset", "tag", "mase_cpu", "mase_gpu", "adapt_j_cpu", "adapt_j_gpu", "j_per_update_cpu",
           "j_per_update_gpu", "infer_j_cpu", "infer_j_gpu"]].to_csv(os.path.join(args.out, "cpu_vs_gpu.csv"), index=False)
        a = c[(c.n_updates_cpu > 0) & (c.n_updates_gpu > 0)]
        if len(a) > 2:
            extra["spearman_adapt_j_cpu_vs_gpu"] = float(spearmanr(a.adapt_j_cpu, a.adapt_j_gpu)[0])
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "extra.json"), "w") as f:
        json.dump(extra, f, indent=2)
    print(json.dumps(extra, indent=2))


if __name__ == "__main__":
    main()
