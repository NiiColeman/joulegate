"""Hard energy budget: which policy is best when the device can only afford a fixed share of updates?

Every run enforces a token bucket (engine/online.py, relative budget): each step earns `income_frac` of one
update's calibrated cost, the bucket holds at most `capacity_updates` updates and starts empty, and any update or
probe the bucket cannot pay for is blocked. Budgets are hyperparameters: income 1, 3, 6, 10, 25 % of an update per
step, capacity 1 update (no saving) or 10 updates (saving for a burst). Because both are in units of the
device's own update cost, they carry over to any device whose costs scale together.

Selection rule, fixed before any seed 3-5 run: per backbone, budget and family, the setting with the lowest mean
dev error over the dev streams (the budget itself bounds the energy, so no energy tier is applied). `greedy` is
update-every-step under the bucket (update whenever affordable). The picks run on the evaluation streams with seeds
3-5; nothing is selected there.

Usage: python -m scripts.budget_sweeps dev     # writes configs/sweeps/budget_dev_*.yaml and slurm/shards/budget_dev.txt
       python -m scripts.budget_sweeps pick    # after the dev runs: writes configs/sweeps/budget_*.yaml,
                                       # slurm/shards/budget.txt and paper/data/budget_picks.json
"""
import json
import os
import sys

import numpy as np
import yaml

from analysis.analyze import load, seed_mean
from scripts.confirm_picks import BACKBONES, FAMILIES, SEEDS, SWEEPS, require_complete, tag_table

INCOME = (0.01, 0.03, 0.06, 0.10, 0.25)
CAPACITY = (1, 10)
GREEDY = "periodic_p1"
PICK_FAMILIES = ["greedy"] + FAMILIES


def suffix(income, cap):
    return f"__b{round(100 * income)}c{cap}"


def budgets():
    return [(i, c) for c in CAPACITY for i in INCOME]


def split_tag(tag):
    """'vou_l10__b6c1' -> ('vou_l10', 'b6c1'); unbudgeted tags -> (tag, None)."""
    base, _, b = tag.partition("__")
    return base, (b or None)


def family(base):
    return "greedy" if base == GREEDY else base.split("_")[0]


def budgeted(run, income, cap):
    r = dict(run)
    r["tag"] = run["tag"] + suffix(income, cap)
    r["set"] = {**run.get("set", {}), "loop.budget": {"income_frac": income, "capacity_updates": cap}}
    return r


def dev_spec(b):
    specs = [os.path.join(SWEEPS, s) for s in b["specs"]]
    table = tag_table(specs)
    first = yaml.safe_load(open(specs[0]))
    base = [table["frozen"]] if "frozen" in table else [{"method": "frozen", "tag": "frozen", "set": {}}]
    runs = base + [budgeted(table[t], i, c) for i, c in budgets()
                   for t in [GREEDY] + [t for t in table if t.split("_")[0] in FAMILIES and t != GREEDY]]
    extra = b.get("extra", {})
    if extra:
        runs = [{**r, "set": {**r.get("set", {}), **extra}} for r in runs]
    return {"out": out_root(b, dev=True), **({"backbone": first["backbone"]} if "backbone" in first else {}),
            "datasets": b["dev"], "seeds": [0], "runs": runs}


def out_root(b, dev):
    name = "" if b["name"] == "patchmixer" else "_" + b["name"]
    return f"results/budget{'_dev' if dev else ''}{name}"


def spec_path(b, dev):
    return os.path.join(SWEEPS, f"budget{'_dev' if dev else ''}_{b['name']}.yaml")


HEADER = __doc__.split("Usage:")[0].strip()


def write(spec, path):
    with open(path, "w") as f:
        f.write("".join(f"# {line}\n" for line in HEADER.splitlines()))
        yaml.safe_dump(spec, f, sort_keys=False)


def write_shards(specs, path):
    with open(path, "w") as f:
        for p, spec in specs:
            for d in spec["datasets"]:
                f.write(f"{p} {d}\n")


def picks_for(m):
    """m: seed-mean dev frame of budgeted runs. Returns {budget: {family: tag or None}}."""
    m = m.copy()
    m["base"], m["budget"] = zip(*m.tag.map(split_tag))
    m["fam"] = m.base.map(family)
    datasets = sorted(m.dataset.unique())
    out = {}
    for bud, sub in m[m.budget.notna()].groupby("budget"):
        out[bud] = {}
        for fam in PICK_FAMILIES:
            best = None
            for tag, s in sub[sub.fam == fam].groupby("tag"):
                if set(s.dataset) != set(datasets):
                    continue
                err = float(np.mean([s[s.dataset == d].mase.iloc[0] for d in datasets]))
                if best is None or err < best[1] - 1e-12:   # ties keep the earlier (alphabetical) tag
                    best = (tag, err)
            out[bud][fam] = best[0] if best else None
    return out


def main(stage):
    pairs = []
    if stage == "dev":
        for b in BACKBONES:
            spec, path = dev_spec(b), spec_path(b, dev=True)
            write(spec, path)
            pairs.append((path, spec))
            print(f"{path}: {len(spec['runs'])} runs x {len(spec['datasets'])} dev streams")
        write_shards(pairs, "slurm/shards/budget_dev.txt")
        return
    summary = {}
    for b in BACKBONES:
        dev = dev_spec(b)
        table = {r["tag"]: r for r in dev["runs"]}
        require_complete(dev["out"], [spec_path(b, dev=True)], b["dev"], list(table), [0])
        m = seed_mean(load(dev["out"]))
        m = m[m.dataset.isin(b["dev"]) & m.tag.isin(table)]
        picks = picks_for(m)
        runs = [table["frozen"]]
        for bud in sorted(picks):
            runs += [table[t] for t in picks[bud].values() if t]
        main_spec = os.path.join(SWEEPS, b["file"])  # the confirmation sweep has the unbudgeted always-update run
        always = [r for r in yaml.safe_load(open(main_spec))["runs"] if r["tag"] == GREEDY]
        spec = {"out": out_root(b, dev=False), **({"backbone": dev["backbone"]} if "backbone" in dev else {}),
                "datasets": b["eval"], "seeds": SEEDS, "runs": runs + always}
        path = spec_path(b, dev=False)
        write(spec, path)
        pairs.append((path, spec))
        summary[b["name"]] = picks
        print(f"== {b['name']}: {len(spec['runs'])} runs")
        for bud, p in picks.items():
            print(f"  {bud:6s} " + ", ".join(f"{f}={t.split('__')[0] if t else '-'}" for f, t in p.items()))
    write_shards(pairs, "slurm/shards/budget.txt")
    os.makedirs("paper/data", exist_ok=True)
    json.dump(summary, open("paper/data/budget_picks.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dev")
