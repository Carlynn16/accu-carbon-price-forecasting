"""
Run the full data-cleaning pipeline on the real data and report results.

Usage:
    python scripts/run_cleaning.py
"""

import logging
import sys
from pathlib import Path

# Make src importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_cleaning import TARGET_COL, DATE_COL, run_pipeline

logging.basicConfig(
    level=logging.WARNING,   # suppress info spam; we print our own summary
    format="%(levelname)s %(name)s: %(message)s",
)

RAW_PATH = Path(__file__).parent.parent / "data" / "raw" / "raw_20241118.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "processed"


def fmt_date(ts):
    return ts.strftime("%Y-%m-%d")


def main():
    print("=" * 70)
    print("DATA CLEANING PIPELINE — REPORT")
    print("=" * 70)

    results = run_pipeline(RAW_PATH, OUTPUT_DIR)

    # ------------------------------------------------------------------ raw
    print(f"\nRaw shape:        {results['raw_shape']}")
    d0, d1 = results["date_range"]
    print(f"Parsed date range: {fmt_date(d0)} to {fmt_date(d1)}")

    # --------------------------------------------------------- dropped cols
    dm = results["dropped_high_missing"]
    print(f"\nColumns dropped for high missingness (>{70:.0f}%): {len(dm)}")
    for c in dm:
        print(f"  - {c}")

    dc = results["dropped_constant"]
    print(f"\nColumns dropped for near-constant (>99% identical): {len(dc)}")
    for c in dc:
        print(f"  - {c}")

    # ------------------------------------------------------- split summary
    train_n, val_n, test_n = (
        results["train_shape"][0],
        results["val_shape"][0],
        results["test_shape"][0],
    )
    total = train_n + val_n + test_n
    print(f"\nFinal shape: {total} rows × {results['final_cols']} cols")

    tr0, tr1 = results["train_date_range"]
    v0, v1 = results["val_date_range"]
    te0, te1 = results["test_date_range"]

    print(f"\nSplit row counts and date ranges:")
    print(f"  Train : {train_n:5d} rows   {fmt_date(tr0)} to {fmt_date(tr1)}")
    print(f"  Val   : {val_n:5d} rows   {fmt_date(v0)} to {fmt_date(v1)}")
    print(f"  Test  : {test_n:5d} rows   {fmt_date(te0)} to {fmt_date(te1)}")

    # --------------------------------------------------------- NaN check
    total_nan = results["total_nan_after_cleaning"]
    print(f"\nTotal NaN after cleaning: {total_nan}  ({'PASS' if total_nan == 0 else 'FAIL — NaN remain!'})")

    # ------------------------------------------------------- target stats
    print(f"\nTarget column '{TARGET_COL}' summary stats:")
    print(f"  {'Split':<8} {'count':>7} {'mean':>10} {'min':>10} {'max':>10}")
    print(f"  {'-'*50}")
    for label, split in [("Train", results["train"]), ("Val", results["val"]), ("Test", results["test"])]:
        s = split[TARGET_COL]
        print(
            f"  {label:<8} {int(s.count()):>7} {s.mean():>10.4f} {s.min():>10.4f} {s.max():>10.4f}"
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
