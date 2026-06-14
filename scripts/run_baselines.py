"""
Run baseline forecasters on val and test splits for all horizons.
Prints drop counts and the full results table.

Usage:
    python scripts/run_baselines.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.baselines import drift_predict, random_walk_predict
from src.evaluate import build_results_table, compute_metrics, prepare_horizon

PROCESSED = Path(__file__).parent.parent / "data" / "processed"
TARGET_COL = "ACCU spot price (Generic)"


def run_baselines(
    feat_val:   pd.DataFrame,
    feat_test:  pd.DataFrame,
    full_price: pd.Series,
    horizons:   list[int] = (1, 7, 30),
) -> pd.DataFrame:
    """Compute random-walk and drift metrics for all horizons on val and test."""
    records = []

    print("Per-horizon usable rows (after NaN drop):")
    for split_name, feat_df in [("val", feat_val), ("test", feat_test)]:
        for h in horizons:
            data = prepare_horizon(feat_df, h)
            print(f"  split={split_name:4s}  h={h:2d}: {data['n_rows']:4d} rows, "
                  f"  {data['n_dropped']:3d} dropped")

    print()
    for h in horizons:
        for split_name, feat_df in [("val", feat_val), ("test", feat_test)]:
            data = prepare_horizon(feat_df, h)

            # ── Random walk ──────────────────────────────────────────────────
            rw_pred    = random_walk_predict(data["n_rows"])
            rw_metrics = compute_metrics(rw_pred, data["y"], data["price_anchor"])
            rw_rmse    = rw_metrics["RMSE"]
            rw_mae     = rw_metrics["MAE"]
            rw_metrics.update({"RMSE_skill_%": 0.0, "MAE_skill_%": 0.0})
            records.append({"model": "random_walk", "horizon": h, "split": split_name,
                             **rw_metrics})

            # ── Drift ────────────────────────────────────────────────────────
            d_pred    = drift_predict(data["dates"], full_price, h)
            d_metrics = compute_metrics(
                d_pred, data["y"], data["price_anchor"],
                rw_rmse=rw_rmse, rw_mae=rw_mae,
            )
            records.append({"model": "drift", "horizon": h, "split": split_name,
                             **d_metrics})

    return build_results_table(records)


def main() -> pd.DataFrame:
    print("Loading parquets...")
    feat_val  = pd.read_parquet(PROCESSED / "feat_val.parquet")
    feat_test = pd.read_parquet(PROCESSED / "feat_test.parquet")

    train = pd.read_parquet(PROCESSED / "train.parquet")
    val   = pd.read_parquet(PROCESSED / "val.parquet")
    test  = pd.read_parquet(PROCESSED / "test.parquet")

    full_price = (
        pd.concat([train, val, test])
        .sort_values("Date")
        .set_index("Date")[TARGET_COL]
    )

    results = run_baselines(feat_val, feat_test, full_price)

    print("=== Baseline Results ===")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print(results.to_string(index=False))
    return results


if __name__ == "__main__":
    main()
