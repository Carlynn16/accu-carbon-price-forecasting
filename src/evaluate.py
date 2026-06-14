"""
Evaluation harness for ACCU price-change forecasts.

Workflow:
  1. prepare_horizon(feat_df, h)          — drop NaN-target rows, split into X / y / meta
  2. compute_metrics(pred, actual, anchor) — RMSE, MAE, MAPE, dir_acc, skill scores
  3. build_results_table(records)          — tidy DataFrame (model × horizon × split)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Column-name constants (importable by Block C models) ─────────────────────
DATE_COL    = "Date"
ANCHOR_COL  = "price_anchor"
TARGET_COLS = frozenset({"target_1", "target_7", "target_30"})
META_COLS   = frozenset({DATE_COL, ANCHOR_COL}) | TARGET_COLS


# ── Per-horizon masking ───────────────────────────────────────────────────────

def prepare_horizon(feat_df: pd.DataFrame, h: int) -> dict:
    """
    For horizon h, drop rows where target_h is NaN, then return a dict with:
      X            — feature DataFrame (META_COLS excluded)
      y            — 1-D array of target_h values (changes)
      price_anchor — 1-D array of price(t) levels (for level reconstruction)
      dates        — 1-D array of Timestamps
      feat_cols    — list of feature column names
      n_dropped    — number of rows dropped (NaN targets)
      n_rows       — number of usable rows
    """
    col  = f"target_{h}"
    mask = feat_df[col].notna()
    sub  = feat_df[mask].reset_index(drop=True)

    feat_cols = [c for c in sub.columns if c not in META_COLS]

    return {
        "X":            sub[feat_cols],
        "y":            sub[col].to_numpy(dtype=float),
        "price_anchor": sub[ANCHOR_COL].to_numpy(dtype=float),
        "dates":        sub[DATE_COL].to_numpy(),
        "feat_cols":    feat_cols,
        "n_dropped":    int((~mask).sum()),
        "n_rows":       len(sub),
    }


# ── Metric helpers ────────────────────────────────────────────────────────────

def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def _mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def _mape(actual_level: np.ndarray, predicted_level: np.ndarray) -> float:
    """MAPE on reconstructed levels; skips rows where actual == 0."""
    nz = actual_level != 0.0
    if not nz.any():
        return float("nan")
    return float(100.0 * np.mean(
        np.abs((actual_level[nz] - predicted_level[nz]) / actual_level[nz])
    ))


def directional_accuracy(
    pred_change: np.ndarray,
    actual_change: np.ndarray,
) -> float:
    """
    Fraction of correctly signed predictions.
    Computed ONLY on genuine-move rows (actual_change != 0).
    Returns NaN when there are no genuine-move rows.
    """
    mask = actual_change != 0.0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.sign(pred_change[mask]) == np.sign(actual_change[mask])))


def skill_score(metric_model: float, metric_baseline: float) -> float:
    """
    Percentage improvement over baseline.
    Positive = better than baseline; 0 = tied; negative = worse.
    """
    return float(100.0 * (1.0 - metric_model / metric_baseline))


# ── Main evaluation function ──────────────────────────────────────────────────

def compute_metrics(
    pred_changes:   np.ndarray,
    actual_changes: np.ndarray,
    price_anchors:  np.ndarray,
    rw_rmse:        float | None = None,
    rw_mae:         float | None = None,
) -> dict:
    """
    Compute all metrics for one (model, horizon, split) combination.

    pred_changes:   predicted h-step price change
    actual_changes: true h-step price change  (target_h)
    price_anchors:  price(t) — to reconstruct levels for MAPE
    rw_rmse / rw_mae: random-walk RMSE/MAE for skill score computation
                       (pass None to leave skill score as NaN)

    Notes:
      RMSE and MAE on reconstructed levels equal RMSE/MAE on raw changes
      because price_anchor cancels in (actual_level - pred_level).
      MAPE uses the reconstructed level for interpretability.
    """
    actual_level = price_anchors + actual_changes
    pred_level   = price_anchors + pred_changes

    rmse_val = _rmse(actual_level, pred_level)   # = _rmse(actual_changes, pred_changes)
    mae_val  = _mae(actual_level, pred_level)
    mape_val = _mape(actual_level, pred_level)
    dir_acc  = directional_accuracy(pred_changes, actual_changes)

    return {
        "RMSE":         rmse_val,
        "MAE":          mae_val,
        "MAPE_%":       mape_val,
        "dir_acc_%":    dir_acc * 100.0 if not np.isnan(dir_acc) else float("nan"),
        "RMSE_skill_%": float("nan") if rw_rmse is None else skill_score(rmse_val, rw_rmse),
        "MAE_skill_%":  float("nan") if rw_mae  is None else skill_score(mae_val,  rw_mae),
    }


# ── Results table ─────────────────────────────────────────────────────────────

_ROUND = {
    "RMSE": 4, "MAE": 4, "MAPE_%": 3,
    "dir_acc_%": 1, "RMSE_skill_%": 2, "MAE_skill_%": 2,
}


def build_results_table(records: list[dict]) -> pd.DataFrame:
    """
    Convert a list of metric dicts (each containing 'model', 'horizon', 'split'
    plus metric keys) into a tidy, rounded DataFrame ordered by horizon then split.
    Rounding is applied here (display only) so compute_metrics stays full-precision.
    """
    df = pd.DataFrame(records)
    for col, decimals in _ROUND.items():
        if col in df.columns:
            df[col] = df[col].round(decimals)
    col_order = [
        "model", "horizon", "split",
        "RMSE", "MAE", "MAPE_%", "dir_acc_%",
        "RMSE_skill_%", "MAE_skill_%",
    ]
    present = [c for c in col_order if c in df.columns]
    return df[present].sort_values(["horizon", "split", "model"]).reset_index(drop=True)
