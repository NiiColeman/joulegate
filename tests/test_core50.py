import numpy as np
import pytest
from PIL import Image

from datasets.core50 import SessionStream, category, convert, half_mask, load_session, parse_name


def fake_core50(tmp_path, sessions=(1, 2, 3), objects=(1, 6, 11), frames=None):
    """A tiny tree in CORe50's layout; videos can have unequal lengths."""
    frames = frames or {}
    src = tmp_path / "src"
    rng = np.random.default_rng(0)
    for s in sessions:
        for o in objects:
            d = src / f"s{s}" / f"o{o}"
            d.mkdir(parents=True)
            for f in range(frames.get((s, o), 20)):
                img = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
                Image.fromarray(img).save(d / f"C_{s:02d}_{o:02d}_{f:03d}.png")
    return src


def test_names_and_categories():
    assert parse_name("x/s3/o12/C_03_12_045.png") == (3, 12, 45)
    assert [category(o) for o in (1, 5, 6, 50)] == [0, 0, 1, 9]


def test_convert_keeps_every_other_frame_and_reports_counts(tmp_path):
    src = fake_core50(tmp_path, frames={(3, 6): 13})            # one shorter video
    counts = convert(str(src), str(tmp_path / "out"), keep_every=2, size=96)
    assert counts == {1: 30, 2: 30, 3: 27}                       # 10 + 10 + 10, and 10 + 7 + 10
    images, meta = load_session(str(tmp_path / "out"), 3)
    assert images.shape == (27, 96, 96, 3) and images.dtype == np.uint8
    assert set(meta["frame"] % 2) == {0}
    short = meta["obj"] == 6
    assert meta["length"][short].tolist() == [7] * 7


def test_halves_split_each_video_and_are_disjoint(tmp_path):
    src = fake_core50(tmp_path, frames={(3, 6): 13})
    convert(str(src), str(tmp_path / "out"))
    _, meta = load_session(str(tmp_path / "out"), 3)
    a, b = half_mask(meta, "first"), half_mask(meta, "second")
    assert not np.any(a & b) and np.all(a | b)
    for o in (1, 6, 11):
        v = meta["obj"] == o
        assert np.all(meta["frame"][v & a] < meta["frame"][v & b].min())


def _stream(tmp_path, **kw):
    if not (tmp_path / "out").exists():                          # build the fake data once per test
        src = fake_core50(tmp_path, sessions=(3, 4), objects=(1, 6, 11, 16, 21, 26, 31, 36, 41, 46),
                          frames={})
        convert(str(src), str(tmp_path / "out"))
    args = dict(root=str(tmp_path / "out"), sessions=[4, 3], half="first", n_steps=6, batch=8, seed=0,
                mean=[0.5] * 3, std=[0.5] * 3)
    args.update(kw)
    return SessionStream(**args)


def test_stream_visits_sessions_in_order_with_normalised_batches(tmp_path):
    s = _stream(tmp_path)
    assert len(s) == 6 and s.regimes.tolist() == [4, 4, 4, 3, 3, 3]
    x, y = s[0]
    assert x.shape == (8, 3, 96, 96) and x.dtype == np.float32 and y.dtype == np.int64
    assert -1.0 <= x.min() and x.max() <= 1.0 and abs(float(x.mean())) < 0.2   # normalised, not 0-255
    assert set(y.tolist()) <= set(range(10))


def test_stream_is_deterministic_and_steps_draw_without_replacement(tmp_path):
    a, b = _stream(tmp_path), _stream(tmp_path)
    np.testing.assert_array_equal(a[2][0], b[2][0])
    idx = np.concatenate([a.index[t] for t in range(3)])          # session 4's steps
    assert len(idx) == len(set(idx.tolist()))


def test_stream_refuses_a_pool_too_small_for_its_steps(tmp_path):
    with pytest.raises(ValueError, match="frames"):
        _stream(tmp_path, n_steps=60, batch=32)


def test_error_is_the_misclassification_rate(tmp_path):
    s = _stream(tmp_path)
    logits = np.eye(10)[[0, 1, 2, 3]]
    assert s.error(logits, np.array([0, 1, 9, 9])) == 0.5


def test_report_switch_steps_match_the_stream_layout(tmp_path):
    from analysis.report_vision import switch_steps
    s = _stream(tmp_path)
    starts = switch_steps({"n_steps": 6, "sessions": [4, 3]})
    assert starts.tolist() == [int(np.flatnonzero(np.diff(s.regimes))[0]) + 1]


def test_report_gives_accuracy_point_gains_over_frozen_within_a_budget():
    import pandas as pd
    from analysis.report_vision import accuracy_gain_cells
    m = pd.DataFrame([
        dict(dataset="core50", tag="frozen", family="frozen", mase=0.20, adapt_j=0.0),
        dict(dataset="core50", tag="periodic_p1", family="periodic", mase=0.05, adapt_j=100.0),
        dict(dataset="core50", tag="periodic_p8", family="periodic", mase=0.08, adapt_j=12.0),
        dict(dataset="core50", tag="periodic_p32", family="periodic", mase=0.12, adapt_j=4.0),
    ])
    cells = accuracy_gain_cells(m, "core50", "periodic", (0.03, 0.05, 0.25))
    assert np.isnan(cells[0])                                    # nothing fits 3 J
    assert cells[1] == pytest.approx(8.0) and cells[2] == pytest.approx(12.0)   # points: 80->88, 80->92


@pytest.mark.parametrize("sweep", ["configs/sweeps/core50.yaml", "configs/sweeps/core50_measured.yaml"])
def test_every_vou_family_has_a_dev_setting_under_the_5_percent_budget(sweep):
    """lam 10 sits at ~1-2 % of always-update energy on dev (lam 5 at 6-9 %); without it the 5 %
    column says '--' because nothing was sampled there, not because the policy cannot fit."""
    import yaml
    tags = {r["tag"] for r in yaml.safe_load(open(sweep))["runs"]}
    for fam in ("vou", "vougerr_t0.05", "vougerr_t0.2", "vougin_k3", "vougin_k5"):
        assert f"{fam}_l10" in tags, fam


def test_domain_test_set_samples_raw_frames_from_one_half_only(tmp_path):
    from datasets.core50 import domain_test_set
    src = fake_core50(tmp_path, sessions=(3,), objects=(1, 6, 11))
    convert(str(src), str(tmp_path / "data"))
    images, meta = load_session(str(tmp_path / "data"), 3)
    first = set(np.flatnonzero(half_mask(meta, "first")))
    x, y, idx = domain_test_set(str(tmp_path / "data"), 3, "first", 5, seed=0)
    assert x.dtype == np.uint8 and x.shape[0] == 5 and len(y) == 5
    assert set(idx) <= first and len(set(idx)) == 5
    assert np.array_equal(idx, domain_test_set(str(tmp_path / "data"), 3, "first", 5, seed=0)[2])
    assert np.array_equal(y, [category(o) for o in meta["obj"][idx]])


def test_domain_metrics_on_a_hand_matrix():
    from analysis.report_domains import metrics
    acc = np.array([[0.9, 0.5, 0.4],    # start: base, D1, D2
                    [0.8, 0.9, 0.6],    # after D1
                    [0.7, 0.7, 0.95]])  # after D2
    m = metrics(acc)
    assert np.isclose(m["final_all"], (0.7 + 0.7 + 0.95) / 3) and m["final_base"] == 0.7
    assert np.isclose(m["forgetting"], ((0.9 - 0.7) + (0.9 - 0.7)) / 2)
    assert np.isclose(m["just_learned"], (0.9 + 0.95) / 2) and np.isclose(m["before_learning"], (0.5 + 0.6) / 2)
    assert np.allclose(m["seen_curve"], [0.9, 0.85, (0.7 + 0.7 + 0.95) / 3])
