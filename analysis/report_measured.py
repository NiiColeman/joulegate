"""Tables for the report's measured-energy section, from results on disk.

Usage: python -m analysis.report_measured [--out paper]
Reads results/ (analytic, incl. gated VoU), results/measured_full/cpu (measured CPU, 3 seeds),
results/measured_percall/gpu (exact GPU per-call joules). Streams missing from a source are
left out of that source's rows and listed in paper/data/measured_coverage.json.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from analysis.analyze import DATASETS, TITLE, load, seed_mean, steps_of
from analysis.analyze_measured import per_call_costs, projected_runs
from metrics.forecasting import error_at_budget
from analysis.paths import paper_file

FAMS = ("periodic", "drift", "vou", "vougerr", "vougin")
FAM_TEX = {"periodic": "Periodic", "drift": "Drift", "vou": "VoU", "vougerr": "Gated VoU (error)",
           "vougin": "Gated VoU (input)"}


def budget_cells(m, d, fam, budgets):
    """Best MASE change (%) vs frozen within each budget fraction of always-update energy."""
    sub = m[m.dataset == d]
    ref = sub[sub.tag == "frozen"].mase.iloc[0]
    e_max = sub[sub.tag == "periodic_p1"].adapt_j.iloc[0]
    g = sub[sub.family == fam]
    out = []
    for b in budgets:
        e = error_at_budget(g.adapt_j.values, g.mase.values, b * e_max) if len(g) else np.nan
        out.append(np.nan if np.isnan(e) else 100 * (e - ref) / ref)
    return out


def fmt(x):
    return "--" if np.isnan(x) else f"${x:+.1f}$"


def per_call_rows():
    an = seed_mean(load("results/main"))
    rows = {}
    for d in DATASETS:
        s = an[(an.dataset == d) & (an.tag == "vou_l1")].iloc[0]
        rows.setdefault("Estimated (45\\,nm model)", {})[d] = {"infer": s.infer_j / steps_of(s), "probe": np.nan,
                                                             "update": s.j_per_update}
    probe = {}
    for d in DATASETS:  # analytic probe joules per step, from the vou_l1 runs
        f = os.path.join("results/main", d, "vou_l1", "seed0", "summary.json")
        s = json.load(open(f))
        probe[d] = s["probe_j"] / s.get("n_steps", 800)
        rows["Estimated (45\\,nm model)"][d]["probe"] = probe[d]
    for label, root in (("Measured CPU (RAPL)", "results/measured_full/cpu"),
                        ("Measured GPU (NVML)", "results/measured_percall/gpu")):
        rows[label] = per_call_costs(root) if os.path.isdir(root) else {}
    return rows


def table_per_call(path):
    rows = per_call_rows()
    lines = [r"\begin{tabular}{l rrr rr}", r"\toprule",
             r"& Forecast & Probe & Update & Update / & Probe / \\",
             r"& (mJ) & (mJ) & (mJ) & forecast & update \\", r"\midrule"]
    for label, by_d in rows.items():
        if not by_d:
            continue
        v = {k: np.median([c[k] for c in by_d.values()]) for k in ("infer", "probe", "update")}
        lines.append(f"{label} & {1e3*v['infer']:.2f} & {1e3*v['probe']:.2f} & {1e3*v['update']:.0f} & "
                     f"{v['update']/v['infer']:.0f}$\\times$ & {v['probe']/v['update']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")
    return {k: sorted(v) for k, v in rows.items()}


def table_measured_budget(path):
    an = seed_mean(load("results/main"))
    me = seed_mean(load("results/measured_full/cpu"))
    have = [d for d in DATASETS if (me.dataset == d).any()]
    lines = [r"\begin{tabular}{l rrr rrr}", r"\toprule",
             r"& \multicolumn{3}{c}{Estimated} & \multicolumn{3}{c}{Measured CPU} \\",
             r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
             r"Stream & Periodic & Drift & VoU & Periodic & Drift & VoU \\", r"\midrule"]
    for d in have:
        cells = [budget_cells(m, d, f, (0.10,))[0] for m in (an, me) for f in ("periodic", "drift", "vou")]
        lines.append(TITLE[d].split(" (")[0] + " & " + " & ".join(fmt(c) for c in cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")
    return have


def table_gated(path):
    gpu = projected_runs("results/main", per_call_costs("results/measured_percall/gpu"))
    gpu = gpu[gpu.family.isin(FAMS + ("frozen",))]
    m = gpu.groupby(["dataset", "tag", "family"]).agg(mase=("mase", "mean"), adapt_j=("adapt_j", "mean")).reset_index()
    lines = [r"\begin{tabular}{l rrr r r r}", r"\toprule",
             r"& \multicolumn{3}{c}{Synthetic, budget} & ETTm1 & Electricity & Solar \\",
             r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
             r"Policy & 5\% & 10\% & 25\% & 10\% & 10\% & 10\% \\", r"\midrule"]
    for f in FAMS:
        cells = budget_cells(m, "synthetic", f, (0.05, 0.10, 0.25))
        cells += [budget_cells(m, d, f, (0.10,))[0] for d in ("ettm1", "electricity", "solar")]
        lines.append(FAM_TEX[f] + " & " + " & ".join(fmt(c) for c in cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")
    probes = gpu[gpu.family.isin(["vou", "vougerr", "vougin"])].groupby("family").probes.mean().round(0)
    return {k: float(v) for k, v in probes.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper")
    args = ap.parse_args()
    coverage = {"per_call": table_per_call(paper_file(args.out, "tab_percall.tex")),
                "measured_budget_streams": table_measured_budget(paper_file(args.out, "tab_measured_budget.tex")),
                "mean_probes_per_run_gpu": table_gated(paper_file(args.out, "tab_gated.tex"))}
    json.dump(coverage, open(paper_file(args.out, "measured_coverage.json"), "w"), indent=2)
    print(json.dumps(coverage, indent=2))
    for name in ("tab_percall.tex", "tab_measured_budget.tex", "tab_gated.tex"):
        print(f"--- {name}\n" + open(paper_file(args.out, name)).read())


if __name__ == "__main__":
    main()
