"""
Feature engineering for ACCU carbon-price forecasting.

All features are built on the FULL chronologically sorted series using only
shift() and strictly TRAILING rolling windows (never centered, never bfill).
The series is then re-split at the original train/val boundaries.

Warm-up rows (first WARMUP rows of train) are dropped to avoid NaN-heavy
rows produced by the 30-day rolling windows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_COL = "ACCU spot price (Generic)"
DATE_COL   = "Date"
HORIZONS   = [1, 7, 30]
WARMUP     = 30   # rows dropped from start of train after feature construction

# ── Source column references ───────────────────────────────────────────────────
_LGC     = "LGC spot price"
_STC     = "STC spot price"
_HIR     = "ACCU spot price (HIR)"
_SFM_NC  = "ACCU spot price (SFM) - No co-benefits"
_SFM_CB  = "ACCU spot price (SFM) - With co-benefits"
_ERF     = "ERF average price"
_VOL_G   = "Daily traded volume (Generic)"
_VOL_HIR = "Daily traded volume (HIR)"
_VOL_SFM = "Daily traded volume (SFM)"

# ── Feature exclusion list ─────────────────────────────────────────────────────
# Category 1: target level + all target-derived columns
# Category 2: sibling price LEVELS and premium columns (can reconstruct Generic)
# Category 3: pre-computed raw sibling change columns (we recompute from levels)
EXCLUDED_COLS: list[str] = [
    # --- Category 1: target and derivatives ---
    "ACCU spot price (Generic)",
    "$ Change (Generic)",
    "% Change (Generic)",
    "WoW change (Generic)",
    "WoW change % (Generic)",
    "YTD % change (Generic)",
    "7-day SMA (Generic)",
    "30-day SMA (Generic)",
    "50-day SMA (Generic)",
    "100-day SMA (Generic)",
    "7-day change (Generic)",
    "30-day change (Generic)",
    "50-day change (Generic)",
    "100-day change (Generic)",
    "7-day percent change (Generic)",
    "30-day percent change (Generic)",
    "50-day percent change (Generic)",
    "100-day percent change (Generic)",
    # --- Category 2: sibling levels and premiums ---
    "ACCU spot price (HIR)",
    "ACCU spot price (SFM) - No co-benefits",
    "ACCU spot price (SFM) - With co-benefits",
    "HIR calculated price (indexed)",
    "% premium HIR over Generic",
    "$ premium HIR over Generic",
    "$ premium SFM over Generic",
    "$ premium SFM over Generic (with co-benefits)",
    # --- Category 3: raw pre-computed sibling changes ---
    "$ change (HIR)",
    "$ WoW change (HIR)",
    "$ change (SFM)",
    "$ WoW change (SFM)",
]


# ── Helper ────────────────────────────────────────────────────────────────────

def _days_since_last_move(change: pd.Series) -> pd.Series:
    """O(n) iterative streak counter. 0 on a move, n on the n-th consecutive stale day."""
    result = np.zeros(len(change), dtype=float)
    streak = 0
    for i, c in enumerate(change):
        if pd.isna(c):
            result[i] = np.nan
            streak = 0
        elif c != 0.0:
            streak = 0
            result[i] = 0.0
        else:
            streak += 1
            result[i] = float(streak)
    return pd.Series(result, index=change.index)


# ── Core builder ─────────────────────────────────────────────────────────────

def build_feature_matrix(
    df: pd.DataFrame,
    horizons: list[int] = HORIZONS,
) -> pd.DataFrame:
    """
    Build all features + targets on a full sorted series.

    Returns a DataFrame with: Date, all feature columns, target_1/7/30.
    No EXCLUDED_COLS columns are present in the output.
    NaN rows (from warm-up and horizon look-ahead) are retained here —
    they are trimmed by the caller (build_features) after re-splitting.
    """
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    price  = df[TARGET_COL].copy()
    change = price.diff()                     # daily change; row 0 = NaN

    out = pd.DataFrame({DATE_COL: df[DATE_COL]})

    # ── Group A: momentum (change family) ──────────────────────────────────
    # chg_0 = s(t) = price(t)-price(t-1): most-recent realised change,
    # fully available at end of day t. chg_1..chg_5 are prior-day lags.
    out["chg_0"] = change.shift(0)   # s(t)
    out["chg_1"] = change.shift(1)   # s(t-1)
    out["chg_2"] = change.shift(2)   # s(t-2)
    out["chg_3"] = change.shift(3)   # s(t-3)
    out["chg_5"] = change.shift(5)   # s(t-5)

    # ── Group B: staleness ──────────────────────────────────────────────────
    moved = (change != 0.0).astype(float)
    moved[change.isna()] = np.nan
    out["price_moved"]        = moved.shift(1)          # past-only: did price move yesterday?
    out["days_since_last_move"] = _days_since_last_move(change).shift(1)
    out["moves_7d"]           = moved.shift(1).rolling(7,  min_periods=1).sum()
    out["moves_30d"]          = moved.shift(1).rolling(30, min_periods=1).sum()

    # ── Group C: volatility regime ──────────────────────────────────────────
    # shift(1) so the rolling window ends yesterday (pure trailing)
    out["vol_chg_7d"]  = change.shift(1).rolling(7,  min_periods=2).std()
    out["vol_chg_30d"] = change.shift(1).rolling(30, min_periods=2).std()

    # ── Group D: traded volume (Generic) ───────────────────────────────────
    if _VOL_G in df.columns:
        vol = df[_VOL_G].copy()
        out["vol_generic_raw"]    = vol.shift(1)
        out["vol_generic_log1p"]  = np.log1p(vol.shift(1).clip(lower=0))
        out["vol_generic_trail7"] = vol.shift(1).rolling(7, min_periods=1).mean()
        out["vol_generic_zero"]   = (vol.shift(1) == 0.0).astype(float)

    # ── Group E: calendar ───────────────────────────────────────────────────
    out["dow"]   = df[DATE_COL].dt.dayofweek
    out["month"] = df[DATE_COL].dt.month

    # ── Group F: exogenous diffs ────────────────────────────────────────────
    for src_col, feat_name in [
        (_LGC,    "lgc_chg"),
        (_STC,    "stc_chg"),
        (_HIR,    "hir_chg"),
        (_SFM_NC, "sfm_nc_chg"),
        (_SFM_CB, "sfm_cb_chg"),
        (_ERF,    "erf_chg"),
    ]:
        if src_col in df.columns:
            out[feat_name] = df[src_col].diff().shift(1)

    for src_col, feat_name in [
        (_VOL_HIR, "hir_vol_trail7"),
        (_VOL_SFM, "sfm_vol_trail7"),
    ]:
        if src_col in df.columns:
            out[feat_name] = df[src_col].shift(1).rolling(7, min_periods=1).mean()

    # ── Targets ─────────────────────────────────────────────────────────────
    for h in horizons:
        out[f"target_{h}"] = price.shift(-h) - price

    return out


# ── Public API ────────────────────────────────────────────────────────────────

def build_features(
    train: pd.DataFrame,
    val:   pd.DataFrame,
    test:  pd.DataFrame,
    horizons: list[int] = HORIZONS,
    warmup: int = WARMUP,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build and return (feat_train, feat_val, feat_test).

    Steps:
      1. Concatenate the three splits into one chronological series.
      2. Call build_feature_matrix on the full series (past-only construction).
      3. Re-split at original boundaries.
      4. Drop the first `warmup` rows from train only.
      5. Drop rows where any target is NaN (from horizon look-ahead at the tail).
    """
    n_train = len(train)
    n_val   = len(val)

    full = pd.concat([train, val, test], ignore_index=True)
    feat = build_feature_matrix(full, horizons=horizons)

    feat_train = feat.iloc[:n_train].copy()
    feat_val   = feat.iloc[n_train : n_train + n_val].copy()
    feat_test  = feat.iloc[n_train + n_val :].copy()

    # Drop warm-up from train
    feat_train = feat_train.iloc[warmup:].copy()

    # Drop rows where all targets are NaN (tail look-ahead; only affects test tail)
    target_cols = [f"target_{h}" for h in horizons]
    feat_train = feat_train.dropna(subset=target_cols, how="all").reset_index(drop=True)
    feat_val   = feat_val.dropna(subset=target_cols,   how="all").reset_index(drop=True)
    feat_test  = feat_test.dropna(subset=target_cols,  how="all").reset_index(drop=True)

    return feat_train, feat_val, feat_test


def save_features(
    feat_train: pd.DataFrame,
    feat_val:   pd.DataFrame,
    feat_test:  pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feat_train.to_parquet(output_dir / "feat_train.parquet", index=False)
    feat_val.to_parquet(output_dir   / "feat_val.parquet",   index=False)
    feat_test.to_parquet(output_dir  / "feat_test.parquet",  index=False)
