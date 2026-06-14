"""
Tree models (RF, XGB, LGBM) + SARIMAX benchmark for ACCU price-change forecasting.

Protocol:
  - Hyperparameter tuning: TimeSeriesSplit(n_splits=5) on TRAIN only via GridSearchCV
  - Evaluation: fit on TRAIN → eval VAL; refit on TRAIN+VAL → eval TEST once
  - Skill score normalised to random-walk RMSE (supplied via baseline_df)
"""
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

import xgboost as xgb
import lightgbm as lgb

from src.evaluate import build_results_table, compute_metrics, prepare_horizon

# ── Hyperparameter grids (kept small for reasonable runtime) ─────────────────

RF_GRID: dict = {
    "n_estimators":     [100, 300],
    "max_depth":        [5, None],
    "min_samples_leaf": [5, 10],
}

XGB_GRID: dict = {
    "n_estimators":  [100, 200],
    "max_depth":     [3, 5],
    "learning_rate": [0.05, 0.1],
}

LGBM_GRID: dict = {
    "n_estimators":  [100, 200],
    "num_leaves":    [31, 63],
    "learning_rate": [0.05, 0.1],
}

N_CV_SPLITS = 5

# ── Model catalogue ───────────────────────────────────────────────────────────
# Instantiated once; GridSearchCV clones internally, so these stay pristine.
_TREE_SPECS: list[tuple] = [
    (
        "random_forest",
        RandomForestRegressor(random_state=42, n_jobs=-1),
        RF_GRID,
    ),
    (
        "xgboost",
        xgb.XGBRegressor(random_state=42, verbosity=0, tree_method="hist"),
        XGB_GRID,
    ),
    (
        "lightgbm",
        lgb.LGBMRegressor(random_state=42, verbose=-1),
        LGBM_GRID,
    ),
]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _lookup_rw(
    baseline_df: pd.DataFrame | None,
    h: int,
    split: str,
) -> tuple[float | None, float | None]:
    """Return (rw_rmse, rw_mae) for the given horizon/split, or (None, None)."""
    if baseline_df is None:
        return None, None
    row = baseline_df[
        (baseline_df["model"] == "random_walk") &
        (baseline_df["horizon"] == h) &
        (baseline_df["split"] == split)
    ]
    if row.empty:
        return None, None
    return float(row["RMSE"].iloc[0]), float(row["MAE"].iloc[0])


def _tscv_tune(
    estimator,
    param_grid: dict,
    X: np.ndarray,
    y: np.ndarray,
) -> object:
    """
    GridSearchCV with TimeSeriesSplit.
    Returns best estimator already refit on the full (X, y).
    """
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    gs = GridSearchCV(
        estimator, param_grid,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    gs.fit(X, y)
    return gs.best_estimator_


# ── Tree models ───────────────────────────────────────────────────────────────

def run_tree_models(
    feat_train:  pd.DataFrame,
    feat_val:    pd.DataFrame,
    feat_test:   pd.DataFrame,
    horizons:    Sequence[int] = (1, 7, 30),
    baseline_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    RF, XGB, LGBM: tune on TRAIN (walk-forward CV), predict VAL, refit on
    TRAIN+VAL and predict TEST.  Returns tidy results DataFrame.
    """
    records: list[dict] = []

    for h in horizons:
        tr = prepare_horizon(feat_train, h)
        va = prepare_horizon(feat_val,   h)
        te = prepare_horizon(feat_test,  h)

        X_tr = tr["X"].to_numpy(dtype=float)
        y_tr = tr["y"]
        X_va = va["X"].to_numpy(dtype=float)
        y_va = va["y"]
        X_te = te["X"].to_numpy(dtype=float)
        y_te = te["y"]

        rw_rmse_val, rw_mae_val = _lookup_rw(baseline_df, h, "val")
        rw_rmse_te,  rw_mae_te  = _lookup_rw(baseline_df, h, "test")

        print(f"  h={h}: tuning tree models on {len(X_tr)} train rows ...")
        for name, estimator, grid in _TREE_SPECS:
            # ── tune on TRAIN only ────────────────────────────────────────────
            best = _tscv_tune(estimator, grid, X_tr, y_tr)

            # ── eval VAL (model fit on TRAIN) ─────────────────────────────────
            pred_val = best.predict(X_va)
            m_val = compute_metrics(
                pred_val, y_va, va["price_anchor"],
                rw_rmse=rw_rmse_val, rw_mae=rw_mae_val,
            )
            records.append({"model": name, "horizon": h, "split": "val", **m_val})

            # ── refit on TRAIN+VAL → eval TEST ────────────────────────────────
            X_trval = np.vstack([X_tr, X_va])
            y_trval = np.concatenate([y_tr, y_va])
            best.fit(X_trval, y_trval)
            pred_te = best.predict(X_te)
            m_te = compute_metrics(
                pred_te, y_te, te["price_anchor"],
                rw_rmse=rw_rmse_te, rw_mae=rw_mae_te,
            )
            records.append({"model": name, "horizon": h, "split": "test", **m_te})

    return build_results_table(records)


# ── SARIMAX ───────────────────────────────────────────────────────────────────

def _sarimax_fit_forecast(train_arr: np.ndarray, n_steps: int) -> np.ndarray:
    """
    Fit SARIMAX(1,1,1) on train_arr; return forecast array of length n_steps.
    Falls back to zeros on convergence failure.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = SARIMAX(
                train_arr,
                order=(1, 1, 1),
                trend="n",
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
            fc = np.asarray(result.forecast(n_steps), dtype=float)
    except Exception:
        fc = np.zeros(n_steps, dtype=float)

    return fc


def _apply_sarimax_forecast(
    fc:          np.ndarray,
    n_base:      int,
    date_to_pos: dict,
    feat_df:     pd.DataFrame,
    h:           int,
) -> tuple[np.ndarray, dict]:
    """
    Convert a pre-computed SARIMAX forecast array into predicted changes for
    all valid rows of feat_df at horizon h.

    fc[k] = predicted level for position (n_base + k) in the full price array.
    predicted_change_h(t) = fc[(pos(t) - n_base) + h] - price_anchor(t)
    """
    data  = prepare_horizon(feat_df, h)
    dates = pd.to_datetime(data["dates"])

    preds = np.empty(len(dates), dtype=float)
    for i, (d, p) in enumerate(zip(dates, data["price_anchor"])):
        pos = date_to_pos.get(d)
        if pos is None:
            preds[i] = 0.0
            continue
        idx = (pos - n_base) + h
        preds[i] = float(fc[idx]) - p if 0 <= idx < len(fc) else 0.0

    return preds, data


def run_sarimax(
    feat_val:    pd.DataFrame,
    feat_test:   pd.DataFrame,
    full_price:  pd.Series,           # Date-indexed, covers train+val+test
    horizons:    Sequence[int] = (1, 7, 30),
    baseline_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    SARIMAX(1,1,1) univariate benchmark.
      Val:  fit on train price only.
      Test: refit on train+val price.
    Two SARIMAX fits total (one per split); all horizons share each fit.
    """
    all_dates    = pd.to_datetime(full_price.index)
    date_to_pos  = {d: i for i, d in enumerate(all_dates)}
    full_arr     = full_price.to_numpy(dtype=float)

    val_first    = pd.to_datetime(feat_val["Date"].iloc[0])
    test_first   = pd.to_datetime(feat_test["Date"].iloc[0])
    n_train      = date_to_pos[val_first]
    n_trainval   = date_to_pos[test_first]
    h_max        = max(horizons)

    # ── Fit 1: train only (for val) ───────────────────────────────────────────
    print("  SARIMAX: fitting on train ...")
    n_fc_val = len(full_arr) - n_train + h_max + 5
    fc_val   = _sarimax_fit_forecast(full_arr[:n_train], n_fc_val)

    # ── Fit 2: train+val (for test) ───────────────────────────────────────────
    print("  SARIMAX: refitting on train+val ...")
    n_fc_te = len(full_arr) - n_trainval + h_max + 5
    fc_te   = _sarimax_fit_forecast(full_arr[:n_trainval], n_fc_te)

    records: list[dict] = []
    for h in horizons:
        for split, feat_df, fc, n_base in [
            ("val",  feat_val,  fc_val, n_train),
            ("test", feat_test, fc_te,  n_trainval),
        ]:
            rw_rmse, rw_mae = _lookup_rw(baseline_df, h, split)
            preds, data = _apply_sarimax_forecast(fc, n_base, date_to_pos, feat_df, h)
            m = compute_metrics(
                preds, data["y"], data["price_anchor"],
                rw_rmse=rw_rmse, rw_mae=rw_mae,
            )
            records.append({"model": "sarimax", "horizon": h, "split": split, **m})

    return build_results_table(records)


# ── Consolidated entry point ──────────────────────────────────────────────────

def run_all_models(
    feat_train:  pd.DataFrame,
    feat_val:    pd.DataFrame,
    feat_test:   pd.DataFrame,
    full_price:  pd.Series,
    horizons:    Sequence[int] = (1, 7, 30),
    baseline_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Run RF, XGB, LGBM and SARIMAX.  Returns a combined results DataFrame
    (same schema as build_results_table, without baselines).
    Concatenate with baseline_df in the caller for the full consolidated table.
    """
    print("Tree models ...")
    tree_df = run_tree_models(
        feat_train, feat_val, feat_test, horizons, baseline_df=baseline_df
    )
    print("SARIMAX ...")
    sarx_df = run_sarimax(
        feat_val, feat_test, full_price, horizons, baseline_df=baseline_df
    )
    combined = pd.concat([tree_df, sarx_df], ignore_index=True)
    return build_results_table(combined.to_dict("records"))
