"""
Run all C2 models (RF, XGB, LGBM, SARIMAX) on val and test splits.
Prints a consolidated results table (baselines + models, all horizons).

Usage:
    python scripts/run_models.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.baselines import drift_predict, random_walk_predict
from src.evaluate import build_results_table, compute_metrics, prepare_horizon
from src.models import run_all_models

PROCESSED  = Path(__file__).parent.parent / "data" / "processed"
TARGET_COL = "ACCU spot price (Generic)"
HORIZONS   = (1, 7, 30)


def _compute_baselines(feat_val, feat_test, full_price):
    records = []
    for h in HORIZONS:
        for split_name, feat_df in [("val", feat_val), ("test", feat_test)]:
            data = prepare_horizon(feat_df, h)
            rw_pred    = random_walk_predict(data["n_rows"])
            rw_metrics = compute_metrics(rw_pred, data["y"], data["price_anchor"])
            rw_rmse    = rw_metrics["RMSE"]
            rw_mae     = rw_metrics["MAE"]
            rw_metrics.update({"RMSE_skill_%": 0.0, "MAE_skill_%": 0.0})
            records.append({"model": "random_walk", "horizon": h,
                             "split": split_name, **rw_metrics})

            d_pred    = drift_predict(data["dates"], full_price, h)
            d_metrics = compute_metrics(d_pred, data["y"], data["price_anchor"],
                                        rw_rmse=rw_rmse, rw_mae=rw_mae)
            records.append({"model": "drift", "horizon": h,
                             "split": split_name, **d_metrics})
    return build_results_table(records)


def main() -> pd.DataFrame:
    print("Loading parquets ...")
    feat_train = pd.read_parquet(PROCESSED / "feat_train.parquet")
    feat_val   = pd.read_parquet(PROCESSED / "feat_val.parquet")
    feat_test  = pd.read_parquet(PROCESSED / "feat_test.parquet")

    train = pd.read_parquet(PROCESSED / "train.parquet")
    val   = pd.read_parquet(PROCESSED / "val.parquet")
    test  = pd.read_parquet(PROCESSED / "test.parquet")

    full_price = (
        pd.concat([train, val, test])
        .sort_values("Date")
        .set_index("Date")[TARGET_COL]
    )

    print("Computing baselines ...")
    baseline_df = _compute_baselines(feat_val, feat_test, full_price)

    print("Running models ...")
    model_df, _ = run_all_models(
        feat_train, feat_val, feat_test, full_price,
        horizons=HORIZONS, baseline_df=baseline_df,
    )

    consolidated = pd.concat([baseline_df, model_df], ignore_index=True)
    consolidated = build_results_table(consolidated.to_dict("records"))

    print("\n=== Consolidated Results (val + test, h=1/7/30) ===")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    print(consolidated.to_string(index=False))

    # Save for report
    out = PROCESSED / "model_results.parquet"
    consolidated.to_parquet(out, index=False)
    print(f"\nSaved: {out}")

    return consolidated


if __name__ == "__main__":
    main()
