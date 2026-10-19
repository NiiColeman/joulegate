"""Download the raw CSV/TXT files used for pretraining and for the continual streams.

All sources are public GitHub mirrors, so no Hugging Face access is needed.
Usage: python -m datasets.download            # time-series data
       python -m datasets.download vit        # cache the ViT-Tiny ImageNet weights (login node)
       python -m datasets.download core50     # CORe50 128x128: download, extract, convert
Downloads need the login node (compute nodes have no internet); once the zip is present,
`core50` only extracts and converts, so that part can run in a compute job.
"""
import gzip
import os
import shutil
import urllib.request

ETT = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{}.csv"
LAI = "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/{0}/{1}.txt.gz"

SOURCES = {
    "ETTh1.csv": ETT.format("ETTh1"),
    "ETTh2.csv": ETT.format("ETTh2"),
    "ETTm1.csv": ETT.format("ETTm1"),
    "ETTm2.csv": ETT.format("ETTm2"),
    "electricity.txt": LAI.format("electricity", "electricity"),
    "solar_AL.txt": LAI.format("solar-energy", "solar_AL"),
    "traffic.txt": LAI.format("traffic", "traffic"),
    "exchange_rate.txt": LAI.format("exchange_rate", "exchange_rate"),
}


def main(root="data"):
    os.makedirs(root, exist_ok=True)
    for name, url in SOURCES.items():
        path = os.path.join(root, name)
        if os.path.exists(path):
            continue
        print(f"downloading {name}")
        tmp = path + ".part"
        urllib.request.urlretrieve(url, tmp)
        if url.endswith(".gz"):
            with gzip.open(tmp, "rb") as src, open(path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            os.remove(tmp)
        else:
            os.rename(tmp, path)


CORE50_URL = "http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip"


def core50(root="data"):
    """Download the 128x128 CORe50 release (skipped if present), extract and convert it."""
    import zipfile

    from datasets.core50 import convert
    os.makedirs(root, exist_ok=True)
    zpath = os.path.join(root, "core50_128x128.zip")
    if not os.path.exists(zpath):
        print("downloading", CORE50_URL, flush=True)
        urllib.request.urlretrieve(CORE50_URL, zpath + ".part")
        os.rename(zpath + ".part", zpath)
    if os.environ.get("CORE50_DOWNLOAD_ONLY"):
        return None
    src = os.path.join(root, "core50_128x128")
    if not os.path.isdir(src):
        with zipfile.ZipFile(zpath) as z:
            z.extractall(root)
    counts = convert(src, os.path.join(root, "core50"), keep_every=2, size=96)
    print("kept frames per session:", counts)
    return counts


def vit_weights():
    """Cache the ImageNet ViT-Tiny weights in HF_HOME (login node only)."""
    from models.vit_tiny import TinyViT
    TinyViT(pretrained=True)
    print("cached vit_tiny weights")


HF_PARQUETS = {
    "pacs": ["flwrlabs/pacs", ["train-00000-of-00001"]],
    "officehome": ["flwrlabs/office-home", [f"train-0000{i}-of-00003" for i in range(3)]],
}


def multi_domain(name, root="data"):
    """Download a multi-domain dataset's parquet files from Hugging Face into data/raw/<name> (login
    node; skipped if present) and, unless DOWNLOAD_ONLY is set, convert them into data/<name>."""
    from datasets.domains import convert
    repo, files = HF_PARQUETS[name]
    raw = os.path.join(root, "raw", name)
    os.makedirs(raw, exist_ok=True)
    paths = []
    for i, f in enumerate(files):
        path = os.path.join(raw, f"train-{i}.parquet")
        if not os.path.exists(path):
            print(f"downloading {repo} {f}")
            urllib.request.urlretrieve(f"https://huggingface.co/datasets/{repo}/resolve/main/data/{f}.parquet",
                                       path + ".part")
            os.rename(path + ".part", path)
        paths.append(path)
    if not os.environ.get("DOWNLOAD_ONLY"):
        convert(paths, os.path.join(root, name))


if __name__ == "__main__":
    import sys
    {"core50": core50, "vit": vit_weights, "pacs": lambda: multi_domain("pacs"),
     "officehome": lambda: multi_domain("officehome")}.get(sys.argv[1] if len(sys.argv) > 1 else "", main)()
