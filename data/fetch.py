"""
Fetch data/amazon_reviews.parquet from the public Hugging Face mirror.

The 370 MB parquet is the one thing in this lab that does not live in git.
GitHub rejects any blob over 100 MB, and LFS is the wrong fix for a webinar
repo: the free tier grants 1 GB/month of LFS bandwidth, which two clones of a
370 MB file exhaust -- after which everyone else's clone fails. So the vectors
ship from a HF dataset repo (free, no bandwidth cliff, plain HTTPS) and this
script is the single step between `git clone` and the loaders.

Downloads to a .part file and renames only on a complete transfer, so an
interrupted fetch can never leave a truncated parquet that reads as valid.

    python data/fetch.py             # no-op if already present
    python data/fetch.py --force     # re-download over an existing copy
    python data/fetch.py --verify    # also check sha256 (~2s over 370 MB)
"""
import argparse
import hashlib
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PARQUET = os.path.join(HERE, "amazon_reviews.parquet")

REPO = "simonhearne/milvus-es-live-data"
URL = f"https://huggingface.co/datasets/{REPO}/resolve/main/amazon_reviews.parquet"
EXPECTED_BYTES = 388242488
EXPECTED_SHA256 = "ad1849e73e0d6112d5900a2b6f947bab39dd622a24baa16d929c998607533cb7"

MISSING = f"""data/amazon_reviews.parquet is missing.

It is not in git -- 370 MB is over GitHub's 100 MB blob limit -- so it is
fetched separately from https://huggingface.co/datasets/{REPO}

    python data/fetch.py                          # on the host
    docker compose exec demo python data/fetch.py # or inside the demo container
"""


def require_parquet(path=PARQUET):
    """Return `path`, or raise a FileNotFoundError that names the fix.

    Without this, a fresh clone fails four frames deep inside pyarrow with a
    message that says nothing about how to get the file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(MISSING)
    return path


def sha256(path, blocksize=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(blocksize):
            h.update(chunk)
    return h.hexdigest()


def download(path=PARQUET, force=False, verify=False):
    if os.path.exists(path) and not force:
        print(f"{path} already present ({os.path.getsize(path):,} bytes)")
    else:
        part = path + ".part"
        print(f"fetching {URL}")
        with urllib.request.urlopen(URL) as r:
            total = int(r.headers.get("Content-Length") or EXPECTED_BYTES)
            done = 0
            with open(part, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    print(f"  {done >> 20}/{total >> 20} MiB"
                          f" ({100 * done // total}%)", end="\r", file=sys.stderr)
        print(file=sys.stderr)
        if done != total:
            os.remove(part)
            raise OSError(f"truncated download: got {done:,} of {total:,} bytes")
        os.replace(part, path)
        print(f"wrote {path} ({done:,} bytes)")

    if verify:
        got = sha256(path)
        if got != EXPECTED_SHA256:
            raise OSError(f"sha256 mismatch\n  expected {EXPECTED_SHA256}\n"
                          f"  got      {got}\nre-run with --force")
        print("sha256 ok")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--force", action="store_true",
                    help="re-download even if the file is already there")
    ap.add_argument("--verify", action="store_true",
                    help="check sha256 after fetching")
    a = ap.parse_args()
    download(force=a.force, verify=a.verify)
