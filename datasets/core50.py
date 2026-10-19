"""CORe50 as an online stream with real domain shifts.

50 objects (10 categories of 5) filmed in 11 sessions, each with its own background and lighting.
convert() turns the 128x128 release (s<k>/o<j>/C_<ss>_<oo>_<fff>.png) into one image array and
one metadata file per session, keeping every `keep_every`-th frame (consecutive frames at 20 fps
are near-duplicates). A SessionStream visits target sessions in order; each step is a batch of
frames from the current session, drawn without replacement from one half of every video.
"""
import glob
import os
import re

import numpy as np
from PIL import Image

NAME = re.compile(r"C_(\d+)_(\d+)_(\d+)\.png$")


def parse_name(path):
    m = NAME.search(path)
    if not m:
        raise ValueError(f"not a CORe50 frame name: {path}")
    return tuple(int(g) for g in m.groups())


def category(obj):
    return (obj - 1) // 5


def convert(src, out, keep_every=2, size=96):
    os.makedirs(out, exist_ok=True)
    counts = {}
    for sdir in sorted(glob.glob(os.path.join(src, "s*")), key=lambda p: int(os.path.basename(p)[1:])):
        session = int(os.path.basename(sdir)[1:])
        images, obj, frame, pos, length = [], [], [], [], []
        for odir in sorted(glob.glob(os.path.join(sdir, "o*")), key=lambda p: int(os.path.basename(p)[1:])):
            files = sorted(glob.glob(os.path.join(odir, "C_*.png")), key=lambda p: parse_name(p)[2])
            kept = [f for f in files if parse_name(f)[2] % keep_every == 0]
            for i, f in enumerate(kept):
                _, o, fr = parse_name(f)
                images.append(np.asarray(Image.open(f).convert("RGB").resize((size, size), Image.BILINEAR)))
                obj.append(o), frame.append(fr), pos.append(i), length.append(len(kept))
        np.save(os.path.join(out, f"s{session}_images.npy"), np.stack(images).astype(np.uint8))
        np.savez(os.path.join(out, f"s{session}_meta.npz"), obj=np.array(obj), frame=np.array(frame),
                 pos=np.array(pos), length=np.array(length))
        counts[session] = len(images)
    return counts


def load_session(root, s):
    images = np.load(os.path.join(root, f"s{s}_images.npy"), mmap_mode="r")
    meta = dict(np.load(os.path.join(root, f"s{s}_meta.npz")))
    return images, meta


def half_mask(meta, half):
    first = meta["pos"] < meta["length"] // 2
    if half == "first":
        return first
    if half == "second":
        return ~first
    raise ValueError(f"half must be 'first' or 'second', got {half!r}")


class SessionStream:
    def __init__(self, root, sessions, half, n_steps, batch, seed, mean, std):
        rng = np.random.default_rng(seed)
        per = [n_steps // len(sessions) + (i < n_steps % len(sessions)) for i in range(len(sessions))]
        self.data, self.index, regimes = {}, [], []
        for s, k in zip(sessions, per):
            images, meta = load_session(root, s)
            pool = np.flatnonzero(half_mask(meta, half))
            if k * batch > len(pool):
                raise ValueError(f"session {s} ({half} half) has {len(pool)} frames, "
                                 f"{k} steps x {batch} need {k * batch}")
            chosen = rng.choice(pool, size=k * batch, replace=False)
            self.data[s] = (images, np.array([category(o) for o in meta["obj"]], dtype=np.int64))
            for i in range(k):
                self.index.append(chosen[i * batch:(i + 1) * batch])
                regimes.append(s)
        self.regimes = np.array(regimes)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 3, 1, 1)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, t):
        images, labels = self.data[self.regimes[t]]
        idx = np.sort(self.index[t])                      # sorted reads from the memory map
        x = images[idx].astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        return (x - self.mean) / self.std, labels[idx]

    def error(self, logits, y):
        return float((np.asarray(logits).argmax(1) != y).mean())


def domain_test_set(root, session, half, n, seed=0):
    """A fixed sample of n raw (uint8) frames and their categories from one half of a session's
    videos, for per-domain accuracy. Returns (images, labels, frame indices)."""
    images, meta = load_session(root, session)
    idx = np.sort(np.random.default_rng(seed).choice(np.flatnonzero(half_mask(meta, half)), size=n, replace=False))
    return np.asarray(images[idx]), np.array([category(o) for o in meta["obj"][idx]], dtype=np.int64), idx


def build_vision_stream(dcfg):
    return SessionStream(dcfg["root"], dcfg["sessions"], dcfg["half"], dcfg["n_steps"], dcfg["batch"],
                         dcfg["seed"], dcfg["mean"], dcfg["std"])
