"""Write synthetic raw IH / ETF tick files in the vendor layouts to data/raw/."""
import argparse
from pathlib import Path

from basis_bt.synthetic import generate

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=ROOT / "data" / "raw")
    p.add_argument("--days", type=int, default=6)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    for name, path in generate(a.out, n_days=a.days, seed=a.seed).items():
        print(f"{name:9s} -> {path}")
