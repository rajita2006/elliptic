"""Download the Elliptic transaction data (Elliptic++ mirror on GitHub) into data/raw/."""
import sys
import urllib.request
from pathlib import Path

BASE = "https://media.githubusercontent.com/media/git-disl/EllipticPlusPlus/main/Transactions%20Dataset/"
FILES = ["txs_classes.csv", "txs_edgelist.csv", "txs_features.csv"]   # features ≈ 695 MB
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        dst = RAW / f
        if dst.exists() and dst.stat().st_size > 1000:
            print(f"{f} already present"); continue
        print(f"downloading {f} ...", flush=True)
        urllib.request.urlretrieve(BASE + f, dst,
                                   lambda b, bs, t: sys.stdout.write(f"\r  {min(b * bs, t) / 1e6:,.0f} / {t / 1e6:,.0f} MB"))
        print()
    print("done ->", RAW)


if __name__ == "__main__":
    main()
