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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    RF, XGB, LGBM: tune on TRAIN (walk-forward CV), predict VAL, refit on
    TRAIN+VAL and predict TEST.  Returns (results_df, preds_df).
    preds_df columns: model, horizon, split, date, pred_change, actual_change, price_anchor.
    """
    records: list[dict]      = []
    pred_records: list[dict] = []

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
            for d, anch, act, p in zip(va["dates"], va["price_anchor"], va["y"], pred_val):
                pred_records.append({
                    "model": name, "horizon": h, "split": "val",
                    "date": pd.Timestamp(d), "pred_change": float(p),
                    "actual_change": float(act), "price_anchor": float(anch),
                })

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
            for d, anch, act, p in zip(te["dates"], te["price_anchor"], te["y"], pred_te):
                pred_records.append({
                    "model": name, "horizon": h, "split": "test",
                    "date": pd.Timestamp(d), "pred_change": float(p),
                    "actual_change": float(act), "price_anchor": float(anch),
                })

    return build_results_table(records), pd.DataFrame(pred_records)


# ── SARIMAX ───────────────────────────────────────────────────────────────────

def _sarimax_fit(train_arr: np.ndarray):
    """Fit SARIMAX(1,1,1) on train_arr; return result or None on failure."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return SARIMAX(
                train_arr,
                order=(1, 1, 1),
                trend="n",
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
    except Exception:
        return None


def run_sarimax(
    feat_val:    pd.DataFrame,
    feat_test:   pd.DataFrame,
    full_price:  pd.Series,
    horizons:    Sequence[int] = (1, 7, 30),
    baseline_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    SARIMAX(1,1,1) univariate benchmark with rolling h-step-ahead forecasts.

    At each evaluation origin t the h-step predicted level is:
        E[y_{t+h} | y_0..y_t] = Z @ T^h @ alpha_{t|t}
    where alpha_{t|t} is the Kalman filtered state after observing y_t.
    One apply() call per split runs the Kalman filter in O(N); no rolling refit.
    ARIMA(0,1,0) through this harness gives skill = 0% exactly (verified).

    Val:  parameters fit on train; Kalman pass over train+val.
    Test: parameters refit on train+val; Kalman pass over full series.

    Returns (results_df, preds_df).
    """
    all_dates    = pd.to_datetime(full_price.index)
    date_to_pos  = {d: i for i, d in enumerate(all_dates)}
    full_arr     = full_price.to_numpy(dtype=float)

    val_first    = pd.to_datetime(feat_val["Date"].iloc[0])
    test_first   = pd.to_datetime(feat_test["Date"].iloc[0])
    n_train      = date_to_pos[val_first]
    n_trainval   = date_to_pos[test_first]

    print("  SARIMAX: fitting on train ...")
    result_val  = _sarimax_fit(full_arr[:n_train])
    print("  SARIMAX: refitting on train+val ...")
    result_test = _sarimax_fit(full_arr[:n_trainval])

    records: list[dict]      = []
    pred_records: list[dict] = []

    for split_name, feat_df, result, n_apply_end in [
        ("val",  feat_val,  result_val,  n_trainval),
        ("test", feat_test, result_test, len(full_arr)),
    ]:
        # One Kalman filter pass per split
        if result is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fr = result.apply(full_arr[:n_apply_end]).filter_results
            T    = fr.transition[:, :, 0]
            Z    = fr.design[0, :, 0]
            ZTh  = {h: Z @ np.linalg.matrix_power(T, h) for h in horizons}
            filt = fr.filtered_state          # (state_dim, n_apply_end)
        else:
            fr = None

        for h in horizons:
            rw_rmse, rw_mae = _lookup_rw(baseline_df, h, split_name)
            data  = prepare_horizon(feat_df, h)
            dates = np.array(data["dates"])

            if fr is not None:
                ZTh_h = ZTh[h]
                preds = np.empty(len(dates), dtype=float)
                for i, (d, anch) in enumerate(zip(dates, data["price_anchor"])):
                    pos = date_to_pos.get(pd.Timestamp(d))
                    if pos is None or pos >= n_apply_end:
                        preds[i] = 0.0
                    else:
                        preds[i] = float(ZTh_h @ filt[:, pos]) - float(anch)
            else:
                preds = np.zeros(len(dates), dtype=float)

            m = compute_metrics(
                preds, data["y"], data["price_anchor"],
                rw_rmse=rw_rmse, rw_mae=rw_mae,
            )
            records.append({"model": "sarimax", "horizon": h, "split": split_name, **m})
            for d, anch, act, p in zip(dates, data["price_anchor"], data["y"], preds):
                pred_records.append({
                    "model": "sarimax", "horizon": h, "split": split_name,
                    "date": pd.Timestamp(d), "pred_change": float(p),
                    "actual_change": float(act), "price_anchor": float(anch),
                })

    return build_results_table(records), pd.DataFrame(pred_records)


# ── Consolidated entry point ──────────────────────────────────────────────────

def run_all_models(
    feat_train:  pd.DataFrame,
    feat_val:    pd.DataFrame,
    feat_test:   pd.DataFrame,
    full_price:  pd.Series,
    horizons:    Sequence[int] = (1, 7, 30),
    baseline_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run RF, XGB, LGBM and SARIMAX.  Returns (results_df, preds_df).
    results_df: same schema as build_results_table, without baselines.
    preds_df:   long-form predictions (model, horizon, split, date, pred_change, ...).
    """
    print("Tree models ...")
    tree_df, tree_preds = run_tree_models(
        feat_train, feat_val, feat_test, horizons, baseline_df=baseline_df
    )
    print("SARIMAX ...")
    sarx_df, sarx_preds = run_sarimax(
        feat_val, feat_test, full_price, horizons, baseline_df=baseline_df
    )
    combined  = pd.concat([tree_df, sarx_df], ignore_index=True)
    all_preds = pd.concat([tree_preds, sarx_preds], ignore_index=True)
    return build_results_table(combined.to_dict("records")), all_preds
