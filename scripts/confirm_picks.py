"""Held-out confirmation (E3): pick one setting per family on the dev streams, write the confirmation sweeps.

Selection rule, fixed before any seed 3-5 run and applied to the dev results only: per family, the setting with
the lowest mean dev error (MASE; error rate for vision) among settings whose adaptation energy is at most 10 % of
update-every-step (periodic_p1) on every dev stream; if there is none, at most 25 %; if still none, no setting.
The picks, frozen and update-every-step are then run on the evaluation streams with new seeds 3, 4, 5.
Nothing is selected on those runs.

Usage: python -m scripts.confirm_picks            # prints the picks, writes configs/sweeps/confirm*.yaml and
                                          # paper/data/confirm_picks.json
"""
import json
import os

import numpy as np
import yaml

from analysis.analyze import load, seed_mean

FAMILIES = ["periodic", "drift", "vou", "vougerr", "vougin", "carat", "caract", "upf", "d3a"]
SEEDS = [3, 4, 5]
TIERS = (0.10, 0.25)
SWEEPS = "configs/sweeps"
RULE = ("Held-out confirmation picks (E3). Rule, fixed before any seed 3-5 run and applied to dev results only: per family,\n"
        "the setting with the lowest mean dev error among settings using at most 10 % of update-every-step\n"
        "(periodic_p1) adaptation energy on every dev stream, else at most 25 %, else none. Plus frozen and periodic_p1.\n"
        "Seeds 3-5, analytic energy; nothing is selected on these runs.")


def frame(root, datasets):
    """Seed-mean table of dev runs. Refuses any dataset that is not a `_dev` stream."""
    bad = [d for d in datasets if not d.endswith("_dev")]
    if bad:
        raise ValueError(f"confirmation picks may only read dev streams, got {bad}")
    import pandas as pd
    roots = [root] if isinstance(root, str) else list(root)
    m = pd.concat([seed_mean(load(r)) for r in roots], ignore_index=True)
    return m[m.dataset.isin(datasets)].reset_index(drop=True)


def shares(m):
    """tag -> {dataset: adaptation energy / periodic_p1 energy on that dataset}."""
    out = {}
    for d, sub in m.groupby("dataset"):
        ref = sub[sub.tag == "periodic_p1"]
        if ref.empty:
            raise ValueError(f"periodic_p1 missing on {d}")
        e_max = ref.adapt_j.iloc[0]
        for r in sub.itertuples():
            out.setdefault(r.tag, {})[d] = r.adapt_j / e_max
    return out


def pick_info(m, family):
    """(tag, tier, mean dev error, energy shares by dev stream) or None."""
    datasets = sorted(m.dataset.unique())
    sh = shares(m)
    cand = m[m.family == family]
    for tier in TIERS:
        best = None
        for tag, sub in cand.groupby("tag"):
            s = sh[tag]
            if set(sub.dataset) != set(datasets) or any(s[d] > tier + 1e-12 for d in datasets):
                continue
            err = float(np.mean([sub[sub.dataset == d].mase.iloc[0] for d in datasets]))
            if best is None or err < best[2] - 1e-12:      # ties keep the earlier (alphabetical) tag
                best = (tag, tier, err, {d: float(s[d]) for d in datasets})
        if best:
            return best
    return None


def pick(m, family):
    info = pick_info(m, family)
    return info[0] if info else None


def tag_table(specs):
    """tag -> run entry (method, set) over the dev specs; a tag with two different definitions is an error."""
    table = {}
    for path in specs:
        for r in yaml.safe_load(open(path))["runs"]:
            if r["tag"] in table and table[r["tag"]] != r:
                raise ValueError(f"tag {r['tag']} is defined differently in {path}")
            table[r["tag"]] = r
    return table


def write_confirm(picks, specs, out, datasets, extra_set, path, header=RULE, replay_from=None):
    """Write a sweep with the picked tags (family -> tag, None skipped), copied from the dev specs.

    extra_set is merged into each run's `set`. replay_from names the picked periodic tag whose period is reused
    for a replay_p{p} run (learner.replay size 256, frac 0.5)."""
    table = tag_table(specs)
    first = yaml.safe_load(open(specs[0]))
    runs = []
    for tag in (t for t in picks.values() if t):
        r = dict(table[tag])
        r["set"] = {**r.get("set", {}), **extra_set}
        runs.append(r)
    if replay_from:
        period = table[replay_from]["set"]["method.period"]
        runs.append({"method": "periodic", "tag": f"replay_p{period}",
                     "set": {"method.period": period, "learner.replay": {"size": 256, "frac": 0.5}, **extra_set}})
    new = {"out": out, **({"backbone": first["backbone"]} if "backbone" in first else {}),
           "datasets": datasets, "seeds": SEEDS, "runs": runs}
    with open(path, "w") as f:
        f.write("".join(f"# {line}\n" for line in header.splitlines()))
        yaml.safe_dump(new, f, sort_keys=False)


def require_complete(root, specs, datasets, tags, seeds):
    """Every needed dev run must have finished: a missing one would silently drop a setting from the pick."""
    table = tag_table(specs)
    missing = [(d, t, s) for t in tags if t in table for d in datasets for s in seeds
               if not os.path.exists(os.path.join(root, d, t, f"seed{s}", "summary.json"))]
    if missing:
        raise RuntimeError(f"{len(missing)} dev runs missing under {root}, e.g. {missing[:3]}")


BACKBONES = [
    # name, dev root, dev datasets, dev specs, main spec (header fields), eval datasets, out, spec file, device
    dict(name="patchmixer", root="results/dev", dev=["synthetic_dev", "ettm1_dev"],
         specs=["dev.yaml", "dev_cheap.yaml", "dev_baselines.yaml"], main=None,
         eval=["synthetic", "ettm1", "electricity", "solar"], out="results/confirm", file="confirm.yaml"),
    dict(name="ttm", root="results/dev_ttm", dev=["synthetic_dev", "ettm1_dev"],
         specs=["dev_ttm.yaml", "dev_baselines_ttm.yaml"], main="main_ttm.yaml",
         eval=["synthetic", "ettm1", "electricity", "solar"], out="results/confirm_ttm", file="confirm_ttm.yaml"),
    dict(name="chronos", root="results/dev_chronos", dev=["synthetic_dev", "ettm1_dev"],
         specs=["dev_chronos.yaml", "dev_baselines_chronos.yaml"], main="main_chronos.yaml",
         eval=["synthetic", "ettm1", "electricity", "solar"], out="results/confirm_chronos",
         file="confirm_chronos.yaml"),
    dict(name="core50", root="results/dev", dev=["core50_dev"], specs=["core50_dev.yaml", "core50_dev_baselines.yaml"],
         main="core50.yaml", eval=["core50"], out="results/confirm_core50", file="confirm_core50.yaml",
         extra={"device": "cuda"}, replay=True),
]


def main():
    summary = {}
    for b in BACKBONES:
        specs = [os.path.join(SWEEPS, s) for s in b["specs"]]
        table = tag_table(specs)
        needed = ["frozen", "periodic_p1"] + [t for t, r in table.items() if t.split("_")[0] in FAMILIES]
        seeds = sorted({s for p in specs for s in yaml.safe_load(open(p))["seeds"]})
        require_complete(b["root"], specs, b["dev"], needed, seeds)
        m = frame(b["root"], b["dev"])
        m = m[m.tag.isin(table)]
        picks, info = {"frozen": "frozen", "always": "periodic_p1"}, {}
        for fam in FAMILIES:
            r = pick_info(m, fam)
            picks[fam] = r[0] if r else None
            info[fam] = (dict(tag=r[0], tier=f"<={int(r[1] * 100)}%", dev_error=r[2], dev_energy_share=r[3])
                         if r else None)
        header = RULE + f"\nBackbone {b['name']}: dev streams {', '.join(b['dev'])} ({b['root']})."
        if b["main"]:   # the evaluation sweeps' own backbone / device conventions
            main_spec = yaml.safe_load(open(os.path.join(SWEEPS, b["main"])))
            assert main_spec.get("backbone") == yaml.safe_load(open(specs[0])).get("backbone")
        write_confirm(picks, specs, b["out"], b["eval"], b.get("extra", {}), os.path.join(SWEEPS, b["file"]), header,
                      replay_from=picks["periodic"] if b.get("replay") and picks["periodic"] else None)
        summary[b["name"]] = info
        print(f"== {b['name']} ({b['root']}, {', '.join(b['dev'])})")
        for fam, r in info.items():
            if r is None:
                print(f"  {fam:9s} none")
            else:
                sh = ", ".join(f"{d} {100 * v:.1f}%" for d, v in r["dev_energy_share"].items())
                print(f"  {fam:9s} {r['tag']:22s} {r['tier']:>5s}  dev error {r['dev_error']:.4f}  energy {sh}")
    os.makedirs("paper/data", exist_ok=True)
    json.dump(summary, open("paper/data/confirm_picks.json", "w"), indent=1)


if __name__ == "__main__":
    main()
