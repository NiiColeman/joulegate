"""Cross-regime analysis: how much does update timing matter, and fewer-larger vs
more-smaller updates at equal energy.

timing gap = (MASE of a policy config - MASE of the periodic curve at the same energy)
             / MASE of the periodic curve, with the periodic curve interpolated
             linearly in log-energy. Negative = better than a fixed duty cycle.
Usage: python -m analysis.compare_regimes [--out paper]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from analysis.analyze import DATASETS, TITLE, load, seed_mean
from analysis.paths import paper_file

REGIMES = {"aggressive": "results/main", "conservative": "results/conservative"}


def periodic_at(sub, energy):
    p = sub[(sub.family == "periodic") & (sub.n_updates >= 1)].sort_values("adapt_j")
    x, y = np.log(p.adapt_j.values), p.mase.values
    le = np.log(energy)
    if le < x[0] or le > x[-1]:
        return np.nan  # no extrapolation outside the swept duty cycles
    return float(np.interp(le, x, y))


def timing_gaps(m):
    rows = []
    for d in DATASETS:
        sub = m[m.dataset == d]
        for _, r in sub[sub.family.isin(["drift", "vou", "oracle"]) & (sub.n_updates >= 1)].iterrows():
            ref = periodic_at(sub, r.adapt_j)
            if not np.isnan(ref):
                rows.append(dict(dataset=d, family=r.family, tag=r.tag, gap=100 * (r.mase - ref) / ref))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper")
    args = ap.parse_args()
    ms = {k: seed_mean(load(v)) for k, v in REGIMES.items()}
    out = {}
    lines = ["\\begin{tabular}{l rrr rrr}", "\\toprule",
             "& \\multicolumn{3}{c}{Aggressive learner} & \\multicolumn{3}{c}{Conservative learner} \\\\",
             "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}",
             "Stream & Drift & VoU & Oracle$^\\dagger$ & Drift & VoU & Oracle$^\\dagger$ \\\\", "\\midrule"]
    gaps = {k: timing_gaps(m) for k, m in ms.items()}
    for d in DATASETS:
        cells = [TITLE[d].split(" (")[0]]
        for k in REGIMES:
            g = gaps[k]
            for fam in ("drift", "vou", "oracle"):
                v = g[(g.dataset == d) & (g.family == fam)].gap
                cells.append("--" if v.empty else f"{v.median():+.1f}")
                out[f"{k}/{d}/{fam}_median_gap"] = None if v.empty else float(v.median())
                out[f"{k}/{d}/{fam}_min_gap"] = None if v.empty else float(v.min())
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(paper_file(args.out, "tab_timing.tex"), "w").write("\n".join(lines) + "\n")

    # Equal energy, different update size: aggressive p8 vs conservative p4 (same joules).
    lines = ["\\begin{tabular}{l rr rr}", "\\toprule",
             "& \\multicolumn{2}{c}{200 small updates} & \\multicolumn{2}{c}{100 large updates} \\\\",
             "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}",
             "Stream & Energy (J) & $\\Delta$MASE & Energy (J) & $\\Delta$MASE \\\\", "\\midrule"]
    for d in DATASETS:
        a = ms["aggressive"]
        c = ms["conservative"]
        fa = a[(a.dataset == d) & (a.tag == "frozen")].mase.iloc[0]
        ra = a[(a.dataset == d) & (a.tag == "periodic_p8")].iloc[0]
        rc = c[(c.dataset == d) & (c.tag == "periodic_p4")].iloc[0]
        dc, da = 100 * (rc.mase - fa) / fa, 100 * (ra.mase - fa) / fa
        out[f"equal_energy/{d}"] = dict(small_j=float(rc.adapt_j), small=dc, large_j=float(ra.adapt_j), large=da)
        lines.append(f"{TITLE[d].split(' (')[0]} & {rc.adapt_j:.1f} & {dc:+.1f}\\% & {ra.adapt_j:.1f} & {da:+.1f}\\% \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(paper_file(args.out, "tab_update_size.tex"), "w").write("\n".join(lines) + "\n")
    json.dump(out, open(paper_file(args.out, "regimes.json"), "w"), indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
