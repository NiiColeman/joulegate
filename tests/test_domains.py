import io
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

from datasets.domains import DomainStream, convert, load_domain, slug, heldout_set
from models.vit_tiny import TinyViT
from run import run
from utils.config import build


def fake_parquet(path, sizes=None, n_classes=3, seed=0):
    """A parquet in the Hugging Face layout (image {bytes, path}, domain, label); varied image sizes."""
    sizes = sizes or {"photo": 40, "art painting": 70, "sketch": 100}
    rng = np.random.default_rng(seed)
    rows = {"image": [], "domain": [], "label": []}
    for d, n in sizes.items():
        for i in range(n):
            w, h = int(rng.integers(100, 160)), int(rng.integers(100, 160))
            buf = io.BytesIO()
            Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)).save(buf, format="JPEG")
            rows["image"].append({"bytes": buf.getvalue(), "path": f"{d}_{i}.jpg"})
            rows["domain"].append(d)
            rows["label"].append(int(i % n_classes))
    pq.write_table(pa.table(rows), path)
    return sizes


def test_convert_writes_square_uint8_domains_with_a_fixed_test_split(tmp_path):
    sizes = fake_parquet(tmp_path / "a.parquet")
    counts = convert([str(tmp_path / "a.parquet")], str(tmp_path / "out"), size=96, test_frac=0.3, seed=0)
    assert counts == {slug(d): n for d, n in sizes.items()} and slug("Real World") == "real_world"
    images, labels, test = load_domain(str(tmp_path / "out"), "art_painting")
    assert images.shape == (70, 96, 96, 3) and images.dtype == np.uint8 and len(labels) == 70
    assert test.sum() == 21                                    # round(0.3 * 70)
    again = convert([str(tmp_path / "a.parquet")], str(tmp_path / "out2"), size=96, test_frac=0.3, seed=0)
    assert again == counts and np.array_equal(load_domain(str(tmp_path / "out2"), "art_painting")[2], test)
    x, y = heldout_set(str(tmp_path / "out"), "art_painting")
    assert len(x) == 21 and np.array_equal(y, labels[test])


def test_stream_uses_every_training_image_once_in_domain_order(tmp_path):
    fake_parquet(tmp_path / "a.parquet")
    convert([str(tmp_path / "a.parquet")], str(tmp_path / "out"), size=96, test_frac=0.3, seed=0)
    s = DomainStream(str(tmp_path / "out"), ["art_painting", "sketch"], batch=8, seed=0, mean=[0.5] * 3, std=[0.5] * 3)
    assert len(s) == 49 // 8 + 70 // 8                         # 49 and 70 training images
    assert list(s.regimes) == ["art_painting"] * 6 + ["sketch"] * 8
    x, y = s[0]
    assert x.shape == (8, 3, 96, 96) and x.dtype == np.float32 and -1.0 <= x.min() and x.max() <= 1.0
    _, _, test = load_domain(str(tmp_path / "out"), "art_painting")
    used = np.concatenate([s.index[t] for t in range(6)])
    assert len(set(used)) == len(used) and not test[used].any()
    assert s.error(np.eye(3)[y], y) == 0.0


def test_dataset_yaml_overrides_apply_before_command_line_ones():
    cfg = build("pacs", "frozen", "analytic_45nm")
    assert cfg["model"]["num_classes"] == 7 and cfg["checkpoint"].endswith("vit_tiny_pacs.pt")
    assert build("pacs", "frozen", "analytic_45nm", ["model.num_classes=3"])["model"]["num_classes"] == 3
    assert build("core50", "frozen", "analytic_45nm")["model"]["num_classes"] == 10


def test_domains_run_with_accuracy_matrix(tmp_path):
    fake_parquet(tmp_path / "a.parquet")
    convert([str(tmp_path / "a.parquet")], str(tmp_path / "data"), size=96, test_frac=0.3, seed=0)
    torch.manual_seed(0)
    torch.save(TinyViT(num_classes=3).state_dict(), tmp_path / "ckpt.pt")
    x, y = heldout_set(str(tmp_path / "data"), "photo")
    np.savez(tmp_path / "ret.npz", images=x, labels=y)
    sets = [f"dataset.root={tmp_path / 'data'}", "dataset.domains=[art_painting, sketch]", "dataset.batch=8",
            "model.num_classes=3", f"checkpoint={tmp_path / 'ckpt.pt'}", f"retention={tmp_path / 'ret.npz'}",
            "learner.buffer_steps=3", "loop.min_buffer=2", "method.period=2", "domain_eval={}", "tag=t"]
    s = run(build("pacs", "periodic", "analytic_45nm", sets), str(tmp_path / "res"), verbose=False)
    d = s["domain_acc"]
    assert d["domains"] == ["base", "art_painting", "sketch"] and d["after"] == ["start", "art_painting", "sketch"]
    assert np.array(d["acc"]).shape == (3, 3) and s["n_updates"] > 0
    saved = json.load(open(tmp_path / "res" / "pacs" / "t" / "seed0" / "summary.json"))
    assert saved["domain_acc"] == d
