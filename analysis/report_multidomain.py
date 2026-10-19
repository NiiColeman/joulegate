"""PACS / Office-Home report: accuracy gains within energy budgets, and accuracy on every domain.

Usage: python -m analysis.report_multidomain pacs [--root results/domains] [--out paper]
Runs come from configs/sweeps/<name>.yaml (the CORe50 grid, untuned). For the per-domain view each
family is represented by its best setting within 10 % of always-update energy, as in CORe50's
domain sweep, or within 25 % when it has none that cheap. Writes tab_<name>.tex (budget gains), tab_<name>_domains.tex, fig_<name>_domains.pdf
and <name>_domains.json.
"""
import argparse
import glob
import json
import os

import numpy as np

from analysis.analyze import load, seed_mean
from analysis.report_domains import figure, metrics
from analysis.report_vision import BUDGETS, FAMS, NAME, accuracy_gain_cells, fmt
from analysis.paths import paper_file


def best_within(m, fam, budget_frac):
    """Tag of the family's lowest-error setting within the budget, or None."""
    e_max = m[m.tag == "periodic_p1"].adapt_j.iloc[0]
    g = m[(m.family == fam) & (m.adapt_j <= budget_frac * e_max)]
    return None if g.empty else g.sort_values("mase").tag.iloc[0]


def domain_results(root, name, tags):
    out = {}
    for t in tags:
        runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(root, name, t, "seed*", "summary.json")))]
        d = runs[0]["domain_acc"]
        out[t] = {"domains": d["domains"], "after": d["after"], "seeds": len(runs),
                  "acc": np.mean([r["domain_acc"]["acc"] for r in runs], axis=0),
                  "adapt_j": float(np.mean([r["adapt_j"] for r in runs]))}
        out[t]["metrics"] = metrics(out[t]["acc"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--root", default="results/domains")
    ap.add_argument("--out", default="paper")
    args = ap.parse_args()
    name = args.dataset
    m = seed_mean(load(args.root))
    m = m[m.dataset == name]
    e_max = m[m.tag == "periodic_p1"].adapt_j.iloc[0]

    lines = [r"\begin{tabular}{l rrr}", r"\toprule", r"Policy & 5\% & 10\% & 25\% \\", r"\midrule"]
    for f in FAMS:
        lines.append(f"{NAME[f]} & " + " & ".join(fmt(c) for c in accuracy_gain_cells(m, name, f, BUDGETS)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(paper_file(args.out, f"tab_{name}.tex"), "w").write("\n".join(lines) + "\n")

    picks = {"frozen": "Frozen", "periodic_p1": "Update every step"}
    for f in FAMS:  # best within 10 %; a family with nothing that cheap is shown at 25 % instead
        t = best_within(m, f, 0.10) or best_within(m, f, 0.25)
        if t is not None:
            picks[t] = f"{NAME[f]} ({t})"
    res = domain_results(args.root, name, picks)
    heat = ["frozen", "periodic_p1"] + [t for t in picks if t.startswith(("periodic_p", "vou_")) and t != "periodic_p1"][:2]
    figure(res, paper_file(args.out, f"fig_{name}_domains.pdf"), heat, picks)
    rows = [r"\begin{tabular}{l rrrrr}", r"\toprule",
            r"Policy & Energy (\%) & Final, all domains & Final, base & Before learning & Forgetting \\", r"\midrule"]
    for t, label in picks.items():
        k = res[t]["metrics"]
        rows.append(f"{label} & {100 * res[t]['adapt_j'] / e_max:.1f} & {100 * k['final_all']:.1f} & "
                    f"{100 * k['final_base']:.1f} & {100 * k['before_learning']:.1f} & {100 * k['forgetting']:.1f} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    open(paper_file(args.out, f"tab_{name}_domains.tex"), "w").write("\n".join(rows) + "\n")
    out = {"frozen_online_acc": float(1 - m[m.tag == "frozen"].mase.iloc[0]),
           "always_online_acc": float(1 - m[m.tag == "periodic_p1"].mase.iloc[0]),
           "budget_gains": {f: accuracy_gain_cells(m, name, f, BUDGETS) for f in FAMS},
           "per_domain": {t: {**r["metrics"], "label": picks[t], "acc": r["acc"].tolist(), "domains": r["domains"],
                              "after": r["after"], "energy_pct": 100 * r["adapt_j"] / e_max} for t, r in res.items()}}
    json.dump(out, open(paper_file(args.out, f"{name}_domains.json"), "w"), indent=1)
    print(open(paper_file(args.out, f"tab_{name}.tex")).read())
    print(open(paper_file(args.out, f"tab_{name}_domains.tex")).read())


if __name__ == "__main__":
    main()
