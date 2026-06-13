"""
Tests for src/data_cleaning.py — all use synthetic data; no real CSV required.
"""

import numpy as np
import pandas as pd
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_cleaning import (
    assess_and_drop_high_missing,
    drop_near_constant,
    ffill_series,
    fill_with_train_stats,
    split_chronological,
    strip_symbols,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n: int, cols: dict | None = None) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    data = {"Date": dates}
    if cols:
        data.update(cols)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# 1. Chronological split — sizes and ordering
# ---------------------------------------------------------------------------

class TestChronologicalSplit:
    def test_sizes_approximately_70_15_15(self):
        df = _make_df(100)
        train, val, test = split_chronological(df)
        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15

    def test_sizes_non_round_number(self):
        df = _make_df(1000)
        train, val, test = split_chronological(df)
        total = len(train) + len(val) + len(test)
        assert total == 1000
        assert abs(len(train) / 1000 - 0.70) < 0.02
        assert abs(len(val) / 1000 - 0.15) < 0.02

    def test_train_dates_before_val(self):
        df = _make_df(200, {"val": range(200)})
        train, val, _ = split_chronological(df)
        assert train["Date"].max() < val["Date"].min()

    def test_val_dates_before_test(self):
        df = _make_df(200, {"val": range(200)})
        _, val, test = split_chronological(df)
        assert val["Date"].max() < test["Date"].min()

    def test_no_overlap(self):
        df = _make_df(300)
        train, val, test = split_chronological(df)
        train_dates = set(train["Date"])
        val_dates = set(val["Date"])
        test_dates = set(test["Date"])
        assert train_dates.isdisjoint(val_dates)
        assert val_dates.isdisjoint(test_dates)
        assert train_dates.isdisjoint(test_dates)


# ---------------------------------------------------------------------------
# 2. Symbol stripping
# ---------------------------------------------------------------------------

class TestStripSymbols:
    def test_dollar_sign(self):
        df = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=1), "price": ["$30.50"]})
        result = strip_symbols(df)
        assert result["price"].iloc[0] == pytest.approx(30.50)

    def test_thousands_comma(self):
        df = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=1), "vol": ["10,000"]})
        result = strip_symbols(df)
        assert result["vol"].iloc[0] == pytest.approx(10000.0)

    def test_percent_sign(self):
        df = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=1), "pct": ["5%"]})
        result = strip_symbols(df)
        assert result["pct"].iloc[0] == pytest.approx(5.0)

    def test_combined_dollar_and_thousands(self):
        df = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=1), "x": ["$1,234.56"]})
        result = strip_symbols(df)
        assert result["x"].iloc[0] == pytest.approx(1234.56)

    def test_date_col_untouched(self):
        df = _make_df(3, {"price": ["$1.0", "$2.0", "$3.0"]})
        result = strip_symbols(df)
        assert pd.api.types.is_datetime64_any_dtype(result["Date"])

    def test_already_numeric_unchanged(self):
        df = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=2), "x": [1.5, 2.5]})
        result = strip_symbols(df)
        assert list(result["x"]) == [1.5, 2.5]


# ---------------------------------------------------------------------------
# 3. NO bfill — leading NaN must NOT be filled from future values
# ---------------------------------------------------------------------------

class TestNoBfill:
    def test_leading_nan_stays_nan_after_ffill(self):
        """First two rows are NaN with no prior values — they must remain NaN after ffill."""
        df = _make_df(5, {"val": [np.nan, np.nan, 3.0, 4.0, 5.0]})
        result = ffill_series(df)
        assert pd.isna(result["val"].iloc[0]), "Leading NaN should not be back-filled"
        assert pd.isna(result["val"].iloc[1]), "Leading NaN should not be back-filled"

    def test_non_leading_nan_is_filled(self):
        """An interior NaN IS forward-filled from a past value."""
        df = _make_df(5, {"val": [1.0, np.nan, np.nan, 4.0, 5.0]})
        result = ffill_series(df)
        assert result["val"].iloc[1] == pytest.approx(1.0)
        assert result["val"].iloc[2] == pytest.approx(1.0)

    def test_all_leading_nan_col_stays_entirely_nan(self):
        """A column that is NaN for its entire prefix (no previous observation) stays NaN."""
        df = _make_df(4, {"val": [np.nan, np.nan, np.nan, np.nan]})
        result = ffill_series(df)
        assert result["val"].isna().all()

    def test_date_col_untouched_by_ffill(self):
        df = _make_df(3, {"val": [np.nan, 2.0, 3.0]})
        original_dates = df["Date"].copy()
        result = ffill_series(df)
        pd.testing.assert_series_equal(result["Date"], original_dates)


# ---------------------------------------------------------------------------
# 4. Train-only fill — fill value comes from TRAIN mean, not global mean
# ---------------------------------------------------------------------------

class TestTrainOnlyFill:
    def _make_splits(self):
        # Train: [1.0, 2.0, NaN] — mean = 1.5
        # Val:   [NaN, 10.0]
        # Test:  [NaN]
        # Global mean of non-NaN values = (1+2+10)/3 ≈ 4.33 — very different from 1.5
        train = pd.DataFrame({
            "Date": pd.date_range("2020-01-01", periods=3),
            "val": [1.0, 2.0, np.nan],
        })
        val = pd.DataFrame({
            "Date": pd.date_range("2020-01-04", periods=2),
            "val": [np.nan, 10.0],
        })
        test = pd.DataFrame({
            "Date": pd.date_range("2020-01-06", periods=1),
            "val": [np.nan],
        })
        return train, val, test

    def test_train_nan_filled_with_train_mean(self):
        train, val, test = self._make_splits()
        t, v, s = fill_with_train_stats(train, val, test)
        assert t["val"].iloc[2] == pytest.approx(1.5)

    def test_val_nan_filled_with_train_mean_not_global(self):
        train, val, test = self._make_splits()
        t, v, s = fill_with_train_stats(train, val, test)
        # Should be 1.5 (train mean), NOT ~4.33 (global mean)
        assert v["val"].iloc[0] == pytest.approx(1.5)
        assert v["val"].iloc[0] != pytest.approx(4.333, abs=0.1)

    def test_test_nan_filled_with_train_mean(self):
        train, val, test = self._make_splits()
        t, v, s = fill_with_train_stats(train, val, test)
        assert s["val"].iloc[0] == pytest.approx(1.5)

    def test_original_splits_not_mutated(self):
        """fill_with_train_stats must return new objects, not modify in-place."""
        train, val, test = self._make_splits()
        fill_with_train_stats(train, val, test)
        assert pd.isna(train["val"].iloc[2]), "Original train should be unchanged"

    def test_different_train_val_means_uses_train(self):
        """Prove correctness when train and val have very different means."""
        train = pd.DataFrame({
            "Date": pd.date_range("2020-01-01", periods=4),
            "x": [10.0, 20.0, 30.0, np.nan],   # train mean = 20.0
        })
        val = pd.DataFrame({
            "Date": pd.date_range("2020-01-05", periods=3),
            "x": [100.0, 200.0, np.nan],         # val mean = 150.0
        })
        test = pd.DataFrame({
            "Date": pd.date_range("2020-01-08", periods=2),
            "x": [np.nan, 5.0],
        })
        _, v, s = fill_with_train_stats(train, val, test)
        assert v["x"].iloc[2] == pytest.approx(20.0)  # train mean, not 150
        assert s["x"].iloc[0] == pytest.approx(20.0)  # train mean, not val mean


# ---------------------------------------------------------------------------
# 5. High-missing drop — >70% dropped, <=70% kept
# ---------------------------------------------------------------------------

class TestHighMissingDrop:
    def _make_df_with_missing(self, n: int = 100) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        # col_high: 80% NaN (should be dropped)
        # col_keep: 60% NaN (should be kept)
        # col_fine: 10% NaN (should be kept)
        col_high = np.where(rng.random(n) < 0.80, np.nan, rng.random(n))
        col_keep = np.where(rng.random(n) < 0.60, np.nan, rng.random(n))
        col_fine = np.where(rng.random(n) < 0.10, np.nan, rng.random(n))
        return pd.DataFrame({
            "Date": pd.date_range("2020-01-01", periods=n),
            "col_high": col_high,
            "col_keep": col_keep,
            "col_fine": col_fine,
        })

    def test_high_missing_col_is_dropped(self):
        df = self._make_df_with_missing()
        result, dropped = assess_and_drop_high_missing(df, threshold=0.70)
        assert "col_high" in dropped
        assert "col_high" not in result.columns

    def test_sixty_pct_missing_col_is_kept(self):
        df = self._make_df_with_missing()
        result, dropped = assess_and_drop_high_missing(df, threshold=0.70)
        assert "col_keep" not in dropped
        assert "col_keep" in result.columns

    def test_dropped_list_matches_removed_columns(self):
        df = self._make_df_with_missing()
        result, dropped = assess_and_drop_high_missing(df, threshold=0.70)
        for col in dropped:
            assert col not in result.columns

    def test_protected_col_never_dropped_even_if_all_nan(self):
        df = pd.DataFrame({
            "Date": pd.date_range("2020-01-01", periods=5),
            "target": [np.nan] * 5,
        })
        result, dropped = assess_and_drop_high_missing(df, threshold=0.70, protect=["Date", "target"])
        assert "target" not in dropped
        assert "target" in result.columns

    def test_exactly_70_pct_boundary_kept(self):
        """A column at exactly 70% missing is NOT dropped (threshold is strictly >70%)."""
        n = 100
        col = [np.nan] * 70 + [1.0] * 30
        df = pd.DataFrame({
            "Date": pd.date_range("2020-01-01", periods=n),
            "boundary": col,
        })
        result, dropped = assess_and_drop_high_missing(df, threshold=0.70)
        assert "boundary" not in dropped


# ---------------------------------------------------------------------------
# 6. Near-constant drop (bonus: ensures drop uses TRAIN stats)
# ---------------------------------------------------------------------------

class TestNearConstantDrop:
    def test_constant_col_dropped_from_all_splits(self):
        train = pd.DataFrame({
            "Date": pd.date_range("2020-01-01", periods=10),
            "const": [1.0] * 10,
            "vary": range(10),
        })
        val = pd.DataFrame({
            "Date": pd.date_range("2020-01-11", periods=5),
            "const": [1.0, 2.0, 1.0, 1.0, 1.0],
            "vary": range(5),
        })
        test = pd.DataFrame({
            "Date": pd.date_range("2020-01-16", periods=5),
            "const": [99.0] * 5,
            "vary": range(5),
        })
        tr, v, te, dropped = drop_near_constant(train, val, test, threshold=0.99)
        # train has 100% identical 'const' → should be dropped
        assert "const" in dropped
        assert "const" not in tr.columns
        assert "const" not in v.columns
        assert "const" not in te.columns

    def test_varying_col_kept(self):
        train = _make_df(20, {"x": range(20)})
        val = _make_df(5, {"x": range(5)})
        test = _make_df(5, {"x": range(5)})
        _, _, _, dropped = drop_near_constant(train, val, test, threshold=0.99)
        assert "x" not in dropped

    def test_protected_col_never_dropped(self):
        train = pd.DataFrame({
            "Date": pd.date_range("2020-01-01", periods=10),
            "target": [5.0] * 10,
        })
        val = _make_df(3, {"target": [5.0] * 3})
        test = _make_df(3, {"target": [5.0] * 3})
        _, _, _, dropped = drop_near_constant(train, val, test, threshold=0.99, protect=["target"])
        assert "target" not in dropped
