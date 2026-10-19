"""Multi-domain image datasets (PACS, Office-Home) as online streams.

convert() turns the Hugging Face parquet files (columns image {bytes, path}, domain, label) into one
array per domain: data/<name>/<domain>_images.npy (n, size, size, 3) uint8, shorter side resized
to `size` and centre-cropped; <domain>_labels.npy; and <domain>_test.npy, a fixed random
`test_frac` of the domain held out for testing (never streamed, never used in fine-tuning).
DomainStream visits the given domains in order and streams every non-test image of each once,
in batches; a domain's leftover images (fewer than one batch) are dropped.
"""
import io
import os

import numpy as np
from PIL import Image


def slug(domain):
    return domain.lower().replace(" ", "_")


def _square(data, size):
    img = Image.open(io.BytesIO(data)).convert("RGB")
    w, h = img.size
    s = size / min(w, h)
    img = img.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    return np.asarray(img.crop((left, top, left + size, top + size)), dtype=np.uint8)


def convert(parquets, out, size=96, test_frac=0.3, seed=0):
    """Returns {domain: number of images}."""
    import pyarrow.parquet as pq
    images, labels = {}, {}
    for p in parquets:
        t = pq.read_table(p).to_pydict()
        for img, d, y in zip(t["image"], t["domain"], t["label"]):
            images.setdefault(slug(d), []).append(_square(img["bytes"], size))
            labels.setdefault(slug(d), []).append(y)
    os.makedirs(out, exist_ok=True)
    counts = {}
    for i, d in enumerate(sorted(images)):
        n = len(images[d])
        test = np.zeros(n, dtype=bool)
        test[np.random.default_rng([seed, i]).choice(n, size=round(test_frac * n), replace=False)] = True
        np.save(os.path.join(out, f"{d}_images.npy"), np.stack(images[d]))
        np.save(os.path.join(out, f"{d}_labels.npy"), np.array(labels[d], dtype=np.int64))
        np.save(os.path.join(out, f"{d}_test.npy"), test)
        counts[d] = n
    print({d: (n, int(round(test_frac * n))) for d, n in counts.items()}, "(images, test)")
    return counts


def load_domain(root, domain):
    """(images memory map, labels, test mask)."""
    return (np.load(os.path.join(root, f"{domain}_images.npy"), mmap_mode="r"),
            np.load(os.path.join(root, f"{domain}_labels.npy")),
            np.load(os.path.join(root, f"{domain}_test.npy")))


def heldout_set(root, domain):
    images, labels, test = load_domain(root, domain)
    idx = np.flatnonzero(test)
    return np.asarray(images[idx]), labels[idx]


class DomainStream:
    def __init__(self, root, domains, batch, seed, mean, std):
        rng = np.random.default_rng(seed)
        self.data, self.index, regimes = {}, [], []
        for d in domains:
            images, labels, test = load_domain(root, d)
            pool = rng.permutation(np.flatnonzero(~test))
            self.data[d] = (images, labels)
            for i in range(len(pool) // batch):
                self.index.append(pool[i * batch:(i + 1) * batch])
                regimes.append(d)
        self.regimes = np.array(regimes)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 3, 1, 1)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, t):
        images, labels = self.data[self.regimes[t]]
        idx = np.sort(self.index[t])
        x = images[idx].astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        return (x - self.mean) / self.std, labels[idx]

    def error(self, logits, y):
        return float((np.asarray(logits).argmax(1) != y).mean())


def build_domain_stream(dcfg):
    return DomainStream(dcfg["root"], dcfg["domains"], dcfg["batch"], dcfg["seed"], dcfg["mean"], dcfg["std"])
