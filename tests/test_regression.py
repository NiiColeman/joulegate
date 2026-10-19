"""Analytic-meter runs must not change while measured-energy code is added.

Write the golden file once, before any engine change:
    python -m tests.test_regression --write
Check:
    python -m pytest tests/test_regression.py -q
"""
import json
import math
import os
import sys
import tempfile

from run import run
from utils.config import build

CASES = [
    ("ettm1", "vou", ["method.lam=1"]),
    ("synthetic", "drift", ["method.tau=0.1"]),
    ("ettm1", "periodic", ["method.period=4"]),
    ("ettm1", "oracle", []),
]
KEYS = ("mase", "n_updates", "update_j", "store_j", "probe_j", "infer_j", "adapt_j", "calib_j", "retention_mse")
GOLDEN = os.path.join(os.path.dirname(__file__), "golden_analytic.json")


def summaries(out_root):
    out = {}
    for dataset, method, sets in CASES:
        cfg = build(dataset, method, "analytic_45nm", sets + ["seed=0", "dataset.n_steps=40", f"tag=golden_{method}"])
        s = run(cfg, out_root=out_root, verbose=False)
        out[f"{dataset}/{method}"] = {k: s[k] for k in KEYS}
    return out


def test_analytic_unchanged(tmp_path):
    with open(GOLDEN) as f:
        want = json.load(f)
    got = summaries(str(tmp_path))
    for case, values in want.items():
        for key, value in values.items():
            assert math.isclose(got[case][key], value, rel_tol=1e-9, abs_tol=1e-12), (case, key, got[case][key], value)


if __name__ == "__main__" and "--write" in sys.argv:
    with tempfile.TemporaryDirectory() as tmp:
        with open(GOLDEN, "w") as f:
            json.dump(summaries(tmp), f, indent=2)
    print(f"wrote {GOLDEN}")
