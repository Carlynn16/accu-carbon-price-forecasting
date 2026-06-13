"""
Run the full EDA pipeline on the cleaned splits and print a statistics summary.
Saves all five figures to figures/.

Usage:
    python scripts/run_eda.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.eda import (
    compute_stationarity_tests,
    plot_acf_pacf,
    plot_price_timeline,
    plot_returns,
    plot_target_distributions,
    plot_volatility,
    print_stats_summary,
)

PROCESSED = Path(__file__).parent.parent / "data" / "processed"
FIGURES = Path(__file__).parent.parent / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def main() -> dict:
    print("Loading cleaned splits...")
    train = pd.read_parquet(PROCESSED / "train.parquet")
    val   = pd.read_parquet(PROCESSED / "val.parquet")
    test  = pd.read_parquet(PROCESSED / "test.parquet")

    print("Generating figures...")
    paths = {
        "fig_price_timeline":      FIGURES / "fig_price_timeline.png",
        "fig_target_dist_by_split": FIGURES / "fig_target_dist_by_split.png",
        "fig_returns":             FIGURES / "fig_returns.png",
        "fig_volatility":          FIGURES / "fig_volatility.png",
        "fig_acf_pacf":            FIGURES / "fig_acf_pacf.png",
    }

    plot_price_timeline(train, val, test,   paths["fig_price_timeline"])
    print(f"  Saved: {paths['fig_price_timeline'].name}")

    plot_target_distributions(train, val, test, paths["fig_target_dist_by_split"])
    print(f"  Saved: {paths['fig_target_dist_by_split'].name}")

    plot_returns(train, val, test,          paths["fig_returns"])
    print(f"  Saved: {paths['fig_returns'].name}")

    plot_volatility(train, val, test,       paths["fig_volatility"])
    print(f"  Saved: {paths['fig_volatility'].name}")

    plot_acf_pacf(train,                    paths["fig_acf_pacf"])
    print(f"  Saved: {paths['fig_acf_pacf'].name}")

    print("\nComputing stationarity tests (train only)...")
    stats = compute_stationarity_tests(train)
    print_stats_summary(stats)

    return stats


if __name__ == "__main__":
    main()
