import json

import numpy as np
import torch

from datasets.core50 import convert
from models.vit_tiny import TinyViT
from run import run
from tests.test_core50 import fake_core50
from utils.config import build


def test_forecasting_configs_still_build_on_base_yaml():
    cfg = build("ettm1", "vou", "analytic_45nm")
    assert cfg["model"]["context"] == 256 and "task" not in cfg["dataset"]


def test_core50_configs_build_on_base_vision_yaml():
    cfg = build("core50", "periodic", "analytic_45nm")
    assert cfg["dataset"]["task"] == "vision" and cfg["model"]["img_size"] == 96
    assert cfg["learner"]["train_head"] is True and cfg["dataset"]["sessions"][0] in range(3, 12)
    dev = build("core50_dev", "periodic", "analytic_45nm")
    assert dev["dataset"]["half"] == "first" and cfg["dataset"]["half"] == "second"
    assert sorted(dev["dataset"]["sessions"]) == sorted(cfg["dataset"]["sessions"]) == list(range(3, 12))
    assert dev["dataset"]["sessions"] != cfg["dataset"]["sessions"]


def _tiny_setup(tmp_path):
    src = fake_core50(tmp_path, sessions=(1, 3, 4), objects=(1, 6, 11, 16, 21, 26, 31, 36, 41, 46))
    convert(str(src), str(tmp_path / "data"))
    torch.manual_seed(0)
    torch.save(TinyViT(pretrained=False).state_dict(), tmp_path / "ckpt.pt")
    rng = np.random.default_rng(0)
    np.savez(tmp_path / "ret.npz", images=rng.integers(0, 256, (4, 96, 96, 3), dtype=np.uint8),
             labels=rng.integers(0, 10, 4))
    return [f"dataset.root={tmp_path / 'data'}", "dataset.sessions=[3, 4]", "dataset.n_steps=6",
            "dataset.batch=8", f"checkpoint={tmp_path / 'ckpt.pt'}", f"retention={tmp_path / 'ret.npz'}",
            "learner.buffer_steps=3", "loop.min_buffer=2", "seed=0", "tag=t"]


def test_vision_run_with_analytic_energy(tmp_path):
    sets = _tiny_setup(tmp_path)
    s = run(build("core50", "periodic", "analytic_45nm", sets + ["method.period=2"]), str(tmp_path / "res"),
            verbose=False)
    assert 0 <= s["error"] <= 1 and s["n_updates"] > 0 and s["adapt_j"] > 0
    assert 0 <= s["retention_acc"] <= 1
    saved = json.load(open(tmp_path / "res" / "core50" / "t" / "seed0" / "summary.json"))
    assert saved["error"] == s["error"]


def test_vision_run_with_fake_measured_energy(tmp_path):
    sets = _tiny_setup(tmp_path)
    s = run(build("core50", "vou_gated", "fake", sets + ["energy.idle_s=0.3", "energy.scale_window_s=0.3",
                                                         "dataset.n_steps=12", "dataset.batch=4"]),
            str(tmp_path / "res"), verbose=False)
    assert s["energy"]["closure"] < 1e-6 and s["energy"]["per_call"]["probe"]["j"] > 0


def test_domain_eval_gives_an_accuracy_matrix_without_touching_energy_or_the_stream(tmp_path):
    base = _tiny_setup(tmp_path)
    sets = base + ["method.period=2"]
    plain = run(build("core50", "periodic", "analytic_45nm", sets), str(tmp_path / "a"), verbose=False)
    s = run(build("core50", "periodic", "analytic_45nm", sets + ["domain_eval={n: 4, half: first}", "tag=d"]),
            str(tmp_path / "b"), verbose=False)
    d = s["domain_acc"]
    assert d["domains"] == ["base", "s3", "s4"] and d["after"] == ["start", "s3", "s4"]
    acc = np.array(d["acc"])
    assert acc.shape == (3, 3) and ((0 <= acc) & (acc <= 1)).all()
    assert s["adapt_j"] == plain["adapt_j"] and s["error"] == plain["error"]
    frozen = run(build("core50", "frozen", "analytic_45nm", base + ["domain_eval={n: 4}", "tag=f"]),
                 str(tmp_path / "c"), verbose=False)
    f = np.array(frozen["domain_acc"]["acc"])
    assert (f == f[0]).all()  # a frozen model scores the same after every domain


def test_domain_eval_is_refused_with_measured_energy():
    import pytest
    with pytest.raises(ValueError, match="domain_eval"):
        run(build("core50", "frozen", "fake", ["domain_eval={n: 4}"]), "unused", verbose=False)


def test_three_base_domain_stream_drops_session_3_and_keeps_the_order():
    b2, b3 = build("core50", "frozen", "analytic_45nm"), build("core50_b3", "frozen", "analytic_45nm")
    assert b3["dataset"]["sessions"] == [s for s in b2["dataset"]["sessions"] if s != 3]
    assert b3["dataset"]["half"] == "second" and b3["dataset"]["n_steps"] == b2["dataset"]["n_steps"]
