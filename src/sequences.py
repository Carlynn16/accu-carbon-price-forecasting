"""
Sliding-window sequence builder for LSTM / GRU models.

Pipeline
--------
1. Fit StandardScaler on TRAIN features only.
2. Scale all splits with the train-fit scaler (no val/test leakage).
3. Build L-step sliding windows over the full chronological feature matrix.
   For each anchor row t:
     input  = scaled_rows[t-L+1 .. t]  shape (L, n_features)
     target = target_h(t)              scalar h-step price change
   Row t is assigned to whichever split it came from.
4. Drop windows whose target_h is NaN (last h rows of the series).

Key leakage property
--------------------
A val-split window may look back into the training tail — this is past-only
and legitimate: the scaler is already fit, and those feature rows are historical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.evaluate import META_COLS

L_DEFAULT = 20   # sliding-window length


def _feature_cols(df: pd.DataFrame) -> list[str]:
    """All columns that are not meta/target columns."""
    return [c for c in df.columns if c not in META_COLS]


def build_sequences(
    feat_train: pd.DataFrame,
    feat_val:   pd.DataFrame,
    feat_test:  pd.DataFrame,
    h: int,
    L: int = L_DEFAULT,
) -> dict:
    """
    Build sliding-window sequences for horizon h.

    Parameters
    ----------
    feat_train / feat_val / feat_test :
        Feature DataFrames (Block B output). Each row is one trading day.
        Columns: META_COLS (Date, price_anchor, target_1/7/30) + feature cols.
    h : forecast horizon (1, 7, or 30).
    L : sliding-window length (default 20).

    Returns
    -------
    dict with keys 'train', 'val', 'test' — each a dict with:
        X            — float32 ndarray (N, L, n_features)
        y            — float64 ndarray (N,)  h-step price changes
        price_anchor — float64 ndarray (N,)  price(t) for level reconstruction
        dates        — ndarray (N,)           origin Timestamps
        n_features   — int
    Top-level keys also include:
        scaler       — fitted StandardScaler (train-only)
        feature_cols — list of feature column names
    """
    target_col = f"target_{h}"
    feat_cols  = _feature_cols(feat_train)   # compute BEFORE any concat

    # ── Build chronological label arrays before concat ─────────────────────
    n_tr, n_va = len(feat_train), len(feat_val)
    split_labels = (
        ["train"] * n_tr
        + ["val"]   * n_va
        + ["test"]  * len(feat_test)
    )

    all_df = pd.concat([feat_train, feat_val, feat_test], ignore_index=True)

    # ── Fit scaler on TRAIN features only ──────────────────────────────────
    train_X = feat_train[feat_cols].to_numpy(dtype=float)
    scaler  = StandardScaler()
    scaler.fit(train_X)

    # ── Scale the full feature matrix ──────────────────────────────────────
    full_X_raw = all_df[feat_cols].to_numpy(dtype=float)
    full_X     = scaler.transform(full_X_raw).astype(np.float32)   # (N_total, F)

    full_y      = all_df[target_col].to_numpy(dtype=float)
    full_anchor = all_df["price_anchor"].to_numpy(dtype=float)
    full_dates  = all_df["Date"].to_numpy()
    split_arr   = np.array(split_labels)

    n_total = len(all_df)
    n_feat  = full_X.shape[1]

    buckets: dict[str, list] = {"train": [], "val": [], "test": []}

    # ── Sliding windows: t runs from L-1 to N-1 ───────────────────────────
    for t in range(L - 1, n_total):
        target = full_y[t]
        if np.isnan(target):
            continue                             # skip tail rows with no target
        window = full_X[t - L + 1 : t + 1]     # (L, F)  past-only
        sp = split_arr[t]
        buckets[sp].append((window, target, full_anchor[t], full_dates[t]))

    # ── Pack into arrays ───────────────────────────────────────────────────
    result: dict = {"scaler": scaler, "feature_cols": feat_cols}
    for sp, items in buckets.items():
        if items:
            Xs, ys, anchs, dates = zip(*items)
            result[sp] = {
                "X":            np.stack(Xs),                          # (N, L, F)
                "y":            np.array(ys,     dtype=float),
                "price_anchor": np.array(anchs,  dtype=float),
                "dates":        np.array(dates),
                "n_features":   n_feat,
            }
        else:
            result[sp] = {
                "X":            np.empty((0, L, n_feat), dtype=np.float32),
                "y":            np.array([], dtype=float),
                "price_anchor": np.array([], dtype=float),
                "dates":        np.array([]),
                "n_features":   n_feat,
            }

    return result
