"""
Data cleaning pipeline for Australian carbon-market price data.

Pipeline order:
  1. load_raw          — skip placeholder row, normalise column names, parse Date
  2. strip_symbols     — remove $, %, thousands-commas; coerce to float
  3. assess_and_drop_high_missing — drop cols > 70% NaN
  4. ffill_series      — forward-fill on FULL sorted series (past-only, no bfill)
  5. split_chronological — 70 / 15 / 15 by row position
  6. fill_with_train_stats — fill residual leading NaN using TRAIN mean / mode
  7. drop_near_constant — drop cols >99% identical (computed on TRAIN)
  8. save to data/processed/ as parquet
"""

import re
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATE_COL = "Date"
DATE_FMT = "%d-%b-%y"
MISSING_THRESHOLD = 0.70
NEAR_CONSTANT_THRESHOLD = 0.99
TARGET_COL = "ACCU spot price (Generic)"


# ---------------------------------------------------------------------------
# Step 1 — load
# ---------------------------------------------------------------------------

def load_raw(file_path: str | Path) -> pd.DataFrame:
    """Load raw CSV (skip placeholder row 0), normalise column names, parse Date, sort asc."""
    df = pd.read_csv(file_path, header=1)

    # normalise column names: collapse any whitespace (incl. embedded newlines) to single space
    df.columns = [re.sub(r"\s+", " ", c).strip() for c in df.columns]

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], format=DATE_FMT, errors="coerce")

    n_before = len(df)
    df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)

    if len(df) < n_before:
        logger.info("Dropped %d rows with unparseable dates.", n_before - len(df))

    logger.info(
        "Loaded %d rows × %d cols; date range %s → %s",
        len(df), df.shape[1],
        df[DATE_COL].iloc[0].date(),
        df[DATE_COL].iloc[-1].date(),
    )
    return df


# ---------------------------------------------------------------------------
# Step 2 — symbol stripping
# ---------------------------------------------------------------------------

def strip_symbols(df: pd.DataFrame, exclude: list[str] | None = None) -> pd.DataFrame:
    """Strip $, %, thousands-commas from string/object columns and coerce to float."""
    exclude = set(exclude) if exclude else {DATE_COL}
    df = df.copy()
    for col in df.columns:
        if col in exclude:
            continue
        # pandas 3+ may store strings as StringDtype rather than object
        if not pd.api.types.is_numeric_dtype(df[col]):
            cleaned = (
                df[col]
                .astype(str)
                .str.replace(r"[\$%]", "", regex=True)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            # pd.to_numeric(..., errors='coerce') converts 'nan'/'None'/'' to NaN
            df[col] = pd.to_numeric(cleaned, errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Step 3 — missing-value assessment & drop
# ---------------------------------------------------------------------------

def assess_and_drop_high_missing(
    df: pd.DataFrame,
    threshold: float = MISSING_THRESHOLD,
    protect: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Report % missing per column; drop columns above threshold (never drop protected cols)."""
    protect = set(protect) if protect else {DATE_COL}

    missing_pct = df.isnull().mean() * 100
    logger.info(
        "Missing %% per column (top 20):\n%s",
        missing_pct.sort_values(ascending=False).head(20).to_string(),
    )

    to_drop = [
        c for c in missing_pct.index
        if missing_pct[c] > threshold * 100 and c not in protect
    ]

    if to_drop:
        logger.info(
            "Dropping %d columns with >%.0f%% missing:\n  %s",
            len(to_drop), threshold * 100, "\n  ".join(to_drop),
        )
        df = df.drop(columns=to_drop)
    else:
        logger.info("No columns exceed the %.0f%% missing threshold.", threshold * 100)

    return df, to_drop


# ---------------------------------------------------------------------------
# Step 4 — forward fill (past-only, full series)
# ---------------------------------------------------------------------------

def ffill_series(df: pd.DataFrame, date_col: str = DATE_COL) -> pd.DataFrame:
    """Forward-fill the FULL (sorted) series. Date column is untouched."""
    df = df.copy()
    value_cols = [c for c in df.columns if c != date_col]
    df[value_cols] = df[value_cols].ffill()
    return df


# ---------------------------------------------------------------------------
# Step 5 — chronological split
# ---------------------------------------------------------------------------

def split_chronological(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by row position (df must already be sorted chronologically)."""
    n = len(df)
    train_end = int(train_frac * n)
    val_end = int((train_frac + val_frac) * n)
    return (
        df.iloc[:train_end].copy(),
        df.iloc[train_end:val_end].copy(),
        df.iloc[val_end:].copy(),
    )


# ---------------------------------------------------------------------------
# Step 6 — fill residual NaN with TRAIN statistics
# ---------------------------------------------------------------------------

def fill_with_train_stats(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    date_col: str = DATE_COL,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fill any remaining NaN using TRAIN-set mean (numeric) or mode (categorical).
    Val/test statistics are never consulted."""
    train, val, test = train.copy(), val.copy(), test.copy()

    num_cols = train.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in train.columns if c not in num_cols and c != date_col]

    for col in num_cols:
        has_nan = any(s[col].isna().any() for s in (train, val, test))
        if has_nan:
            fill_val = train[col].mean()
            if pd.isna(fill_val):
                continue
            train[col] = train[col].fillna(fill_val)
            val[col] = val[col].fillna(fill_val)
            test[col] = test[col].fillna(fill_val)

    for col in cat_cols:
        has_nan = any(s[col].isna().any() for s in (train, val, test))
        if has_nan:
            mode_vals = train[col].mode()
            if mode_vals.empty:
                continue
            fill_val = mode_vals.iloc[0]
            train[col] = train[col].fillna(fill_val)
            val[col] = val[col].fillna(fill_val)
            test[col] = test[col].fillna(fill_val)

    return train, val, test


# ---------------------------------------------------------------------------
# Step 7 — drop near-constant columns
# ---------------------------------------------------------------------------

def drop_near_constant(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    threshold: float = NEAR_CONSTANT_THRESHOLD,
    protect: list[str] | None = None,
    date_col: str = DATE_COL,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Drop columns where >threshold of TRAIN values are identical. Applied to all splits."""
    protect = set(protect) if protect else set()
    protect.add(date_col)

    to_drop = []
    for col in train.columns:
        if col in protect:
            continue
        vc = train[col].value_counts(normalize=True, dropna=False)
        if not vc.empty and vc.iloc[0] > threshold:
            to_drop.append(col)

    if to_drop:
        logger.info(
            "Dropping %d near-constant columns (>%.0f%% identical):\n  %s",
            len(to_drop), threshold * 100, "\n  ".join(to_drop),
        )

    return (
        train.drop(columns=to_drop),
        val.drop(columns=to_drop),
        test.drop(columns=to_drop),
        to_drop,
    )


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    file_path: str | Path,
    output_dir: str | Path,
    target_col: str = TARGET_COL,
) -> dict:
    """Run the full cleaning pipeline and save parquet splits. Returns a summary dict."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load
    df = load_raw(file_path)
    raw_shape = df.shape
    date_range = (df[DATE_COL].iloc[0], df[DATE_COL].iloc[-1])

    # 2. Strip symbols
    df = strip_symbols(df)

    # 3. Drop high-missing
    df, dropped_high_missing = assess_and_drop_high_missing(
        df, protect=[DATE_COL, target_col]
    )

    # 4. ffill on full series (past-only)
    df = ffill_series(df)

    # 5. Chronological split
    train, val, test = split_chronological(df)

    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(train), len(val), len(test),
    )

    # 6. Fill residual NaN (leading NaN not reachable by ffill) with train stats
    train, val, test = fill_with_train_stats(train, val, test)

    # 7. Drop near-constant (computed on train)
    train, val, test, dropped_constant = drop_near_constant(
        train, val, test, protect=[target_col]
    )

    # 8. Save
    train.to_parquet(output_dir / "train.parquet", index=False)
    val.to_parquet(output_dir / "val.parquet", index=False)
    test.to_parquet(output_dir / "test.parquet", index=False)
    logger.info("Saved train/val/test parquet to %s", output_dir)

    total_nan = sum(
        s.isnull().sum().sum() for s in (train, val, test)
    )

    return {
        "raw_shape": raw_shape,
        "date_range": date_range,
        "dropped_high_missing": dropped_high_missing,
        "dropped_constant": dropped_constant,
        "final_cols": train.shape[1],
        "train_shape": train.shape,
        "val_shape": val.shape,
        "test_shape": test.shape,
        "train_date_range": (train[DATE_COL].iloc[0], train[DATE_COL].iloc[-1]),
        "val_date_range": (val[DATE_COL].iloc[0], val[DATE_COL].iloc[-1]),
        "test_date_range": (test[DATE_COL].iloc[0], test[DATE_COL].iloc[-1]),
        "total_nan_after_cleaning": total_nan,
        "train": train,
        "val": val,
        "test": test,
    }
