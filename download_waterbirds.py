"""
download_waterbirds.py -- Fetch and extract the Waterbirds dataset.

Run from the repository root. No Hugging Face token or any other credential is
required: Waterbirds is served from a public Stanford URL, with a CodaLab mirror
and the WILDS package as fallbacks.

    # download + extract + verify (~1.2 GB)
    python download_waterbirds.py

    # keep the tarball as well, in case you want to re-extract
    python download_waterbirds.py --keep-tar

    # just check a copy you already have
    python download_waterbirds.py --verify-only ../data/waterbird_complete95_forest2water2

The default destination is `../data`, i.e. a sibling of the repository, so the
dataset lands at `../data/waterbird_complete95_forest2water2` -- which is
exactly what `extract_features.py` reads by default. No paths need to be typed.

Where it goes, and why not inside the repo
------------------------------------------
The tarball is ~1.2 GB. It is deliberately placed OUTSIDE the repository:

  * GitHub rejects individual files over 100 MB and warns above 50 MB;
  * the data is derived from CUB-200-2011 and Places, so redistributing it is
    not yours to do;
  * for an anonymised code drop it adds nothing a reviewer needs -- they want
    to read the pipeline, not re-download a public benchmark.

The repository's `.gitignore` also excludes `data/`, `waterbird*/`, `*.tar.gz`
and `*.npz`, in case anything lands inside by accident.

Expected layout after extraction
--------------------------------
    <dest>/waterbird_complete95_forest2water2/
        metadata.csv
        001.Black_footed_Albatross/...
        002.Laysan_Albatross/...
        ...

`metadata.csv` carries the columns this pipeline needs: `img_filename`, `y`
(1 = waterbird), `place` (1 = water background) and `split`
(0 = train, 1 = val, 2 = test). The group index used by the theory is derived
from these as `g = (y != place)`, i.e. g = 0 when the background agrees with the
bird type (majority) and g = 1 when it disagrees (minority).
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
import urllib.request

# Primary source, then mirror. Both are public; neither needs credentials.
URLS = [
    "https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz",
    "https://worksheets.codalab.org/rest/bundles/"
    "0x505056d5cdea4e4eaa0e242cbfe2daa4/contents/blob/",
]
DIRNAME = "waterbird_complete95_forest2water2"
# Official split sizes, used as a sanity check rather than as a hard assertion,
# since minor re-releases could shift them.
EXPECTED = {"train": 4795, "val": 1199, "test": 5794}


def _progress(count: int, block: int, total: int) -> None:
    if total <= 0:
        sys.stdout.write(f"\r  {count * block / 1e6:,.0f} MB")
    else:
        pct = min(100.0, count * block * 100.0 / total)
        sys.stdout.write(f"\r  {pct:5.1f}%  ({count * block / 1e6:,.0f} / {total / 1e6:,.0f} MB)")
    sys.stdout.flush()


def download(dest: str, keep_tar: bool = False) -> str:
    os.makedirs(dest, exist_ok=True)
    target = os.path.join(dest, DIRNAME)
    if os.path.isdir(target) and os.path.exists(os.path.join(target, "metadata.csv")):
        print(f"already present: {target}")
        return target

    tar_path = os.path.join(dest, DIRNAME + ".tar.gz")
    if not os.path.exists(tar_path):
        last_err = None
        for url in URLS:
            try:
                print(f"downloading from {url.split('/')[2]} ...")
                urllib.request.urlretrieve(url, tar_path, reporthook=_progress)
                print()
                break
            except Exception as e:  # noqa: BLE001 - report and try the mirror
                last_err = e
                print(f"\n  failed: {e}")
                if os.path.exists(tar_path):
                    os.remove(tar_path)
        else:
            raise RuntimeError(
                f"all sources failed (last error: {last_err}). As a further "
                "fallback, `pip install wilds` then:\n"
                "    from wilds import get_dataset\n"
                "    get_dataset(dataset='waterbirds', download=True, root_dir='<dest>')"
            )
    else:
        print(f"using existing tarball: {tar_path}")

    print("extracting ...")
    with tarfile.open(tar_path, "r:gz") as tf:
        # `filter='data'` (Python >= 3.12) blocks absolute paths and traversal.
        try:
            tf.extractall(dest, filter="data")
        except TypeError:
            tf.extractall(dest)

    if not keep_tar:
        os.remove(tar_path)
        print("removed tarball")
    return target


def verify(root: str) -> bool:
    md = os.path.join(root, "metadata.csv")
    if not os.path.exists(md):
        print(f"FAIL: {md} not found")
        return False
    try:
        import pandas as pd
    except ImportError:
        print("pandas not installed; skipping the split check")
        return True

    df = pd.read_csv(md)
    need = {"img_filename", "y", "place", "split"}
    missing = need - set(df.columns)
    if missing:
        print(f"FAIL: metadata.csv is missing columns {missing}")
        return False

    ok = True
    print(f"metadata.csv: {len(df):,} rows")
    for name, code in (("train", 0), ("val", 1), ("test", 2)):
        n = int((df["split"] == code).sum())
        exp = EXPECTED[name]
        flag = "ok" if n == exp else f"expected {exp}"
        if n != exp:
            ok = False
        sub = df[df["split"] == code]
        eps = float((sub["y"] != sub["place"]).mean()) if len(sub) else float("nan")
        print(f"  {name:<5} n={n:<6,d} minority fraction eps={eps:.4f}   [{flag}]")
    if not ok:
        print("  (count mismatch is a warning, not necessarily an error -- "
              "check you have the complete95_forest2water2 variant)")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dest", default=os.path.join("..", "data"),
                    help="directory to download into (default: ../data, a "
                         "sibling of the repo). Keep this OUTSIDE the repository.")
    ap.add_argument("--keep-tar", action="store_true",
                    help="keep the .tar.gz after extracting")
    ap.add_argument("--verify-only", metavar="ROOT",
                    help="skip downloading; just check an existing directory")
    args = ap.parse_args()

    root = args.verify_only if args.verify_only else download(args.dest, args.keep_tar)
    print()
    if verify(root):
        print("\nready. Next:\n  python extract_features.py")


if __name__ == "__main__":
    main()
