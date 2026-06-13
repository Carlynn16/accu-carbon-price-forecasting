"""
Tests for src/features.py — 5 groups covering:
  1. Target arithmetic
  2. No future leakage
  3. price_moved and days_since_last_move correctness
  4. Rolling features are strictly trailing
  5. Every EXCLUDED_COL is absent from the output
"""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    EXCLUDED_COLS,
    TARGET_COL,
    DATE_COL,
    build_feature_matrix,
    build_features,
    _days_since_last_move,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_df(prices: list[float], start: str = "2020-01-01") -> pd.DataFrame:
    """Minimal DataFrame with Date and target price column."""
    dates = pd.date_range(start, periods=len(prices), freq="D")
    return pd.DataFrame({DATE_COL: dates, TARGET_COL: prices})


def _split(df: pd.DataFrame, n_train: int, n_val: int):
    return (
        df.iloc[:n_train].copy(),
        df.iloc[n_train : n_train + n_val].copy(),
        df.iloc[n_train + n_val :].copy(),
    )


# ── Group 1: Target arithmetic ────────────────────────────────────────────────

class TestTargetArithmetic:
    """target_h(t) must equal price(t+h) - price(t) exactly."""

    def _check_horizon(self, h: int) -> None:
        prices = [10.0, 11.0, 12.0, 13.5, 15.0, 14.0, 13.0, 12.5, 11.0, 10.5]
        df   = _make_df(prices)
        feat = build_feature_matrix(df, horizons=[h])
        col  = f"target_{h}"
        for i in range(len(prices) - h):
            expected = prices[i + h] - prices[i]
            obtained = feat[col].iloc[i]
            assert abs(obtained - expected) < 1e-9, (
                f"h={h}, row {i}: expected {expected:.6f}, got {obtained:.6f}"
            )

    def test_target_1(self):
        self._check_horizon(1)

    def test_target_7(self):
        self._check_horizon(7)

    def test_target_30(self):
        prices = list(range(1, 62))
        df = _make_df(prices)
        feat = build_feature_matrix(df, horizons=[30])
        for i in range(len(prices) - 30):
            expected = prices[i + 30] - prices[i]
            obtained = feat["target_30"].iloc[i]
            assert abs(obtained - expected) < 1e-9

    def test_target_tail_is_nan(self):
        """Last h rows of target_h must be NaN (no future price available)."""
        prices = list(range(1, 21))
        df = _make_df(prices)
        feat = build_feature_matrix(df, horizons=[7])
        assert feat["target_7"].iloc[-7:].isna().all()

    def test_target_non_nan_rows(self):
        prices = list(range(1, 21))
        df = _make_df(prices)
        feat = build_feature_matrix(df, horizons=[1])
        # All rows except the last should have a non-NaN target
        assert feat["target_1"].iloc[:-1].notna().all()


# ── Group 2: No future leakage ────────────────────────────────────────────────

class TestNoFutureLeakage:
    """Perturbing the last 10 rows of prices must not change features at earlier rows."""

    def test_features_unchanged_after_tail_perturbation(self):
        n = 60
        prices = list(range(1, n + 1))
        df_orig = _make_df(prices)
        feat_orig = build_feature_matrix(df_orig, horizons=[1])

        prices_perturbed = prices.copy()
        for i in range(n - 10, n):
            prices_perturbed[i] *= 100.0
        df_pert = _make_df(prices_perturbed)
        feat_pert = build_feature_matrix(df_pert, horizons=[1])

        safe_rows = n - 10 - 1   # rows 0..safe_rows-1 must be identical
        feat_cols = [c for c in feat_orig.columns
                     if c not in (DATE_COL, "target_1")]

        for col in feat_cols:
            orig_vals = feat_orig[col].iloc[:safe_rows].values
            pert_vals = feat_pert[col].iloc[:safe_rows].values
            mask = ~(np.isnan(orig_vals) & np.isnan(pert_vals))
            assert np.allclose(orig_vals[mask], pert_vals[mask], equal_nan=True), (
                f"Column '{col}' differs before the perturbed tail"
            )

    def test_chg_0_equals_price_diff(self):
        """chg_0 must equal price.diff() exactly at every row (including row 0 = NaN)."""
        prices = [10.0, 11.0, 12.0, 14.0, 13.0, 15.0]
        df   = _make_df(prices)
        feat = build_feature_matrix(df, horizons=[1])
        assert np.isnan(feat["chg_0"].iloc[0])
        for i in range(1, len(prices)):
            expected = prices[i] - prices[i - 1]
            assert abs(feat["chg_0"].iloc[i] - expected) < 1e-9, (
                f"chg_0 row {i}: expected {expected}, got {feat['chg_0'].iloc[i]}"
            )

    def test_chg_0_not_affected_by_future(self):
        """chg_0(t) = price(t)-price(t-1); perturbing price(t+1) must not change chg_0(t)."""
        prices = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
        df_orig = _make_df(prices)
        feat_orig = build_feature_matrix(df_orig, horizons=[1])

        prices_pert = prices.copy()
        prices_pert[-1] = 9999.0   # perturb only the last price
        df_pert = _make_df(prices_pert)
        feat_pert = build_feature_matrix(df_pert, horizons=[1])

        # chg_0 at rows 0..n-2 must be unchanged (only last row uses prices[-1])
        for i in range(len(prices) - 1):
            v1, v2 = feat_orig["chg_0"].iloc[i], feat_pert["chg_0"].iloc[i]
            if np.isnan(v1):
                assert np.isnan(v2)
            else:
                assert abs(v1 - v2) < 1e-9, f"chg_0 changed at row {i} after tail perturbation"

    def test_chg_1_equals_prior_day_change(self):
        """chg_1 at row i must equal price(i-1) - price(i-2), i.e. s(t-1)."""
        prices = [10.0, 11.0, 12.0, 14.0, 13.0, 15.0]
        df = _make_df(prices)
        feat = build_feature_matrix(df, horizons=[1])
        for i in range(2, len(prices)):
            expected = prices[i - 1] - prices[i - 2]
            obtained = feat["chg_1"].iloc[i]
            assert abs(obtained - expected) < 1e-9, (
                f"chg_1 row {i}: expected {expected}, got {obtained}"
            )


# ── Group 3: price_moved and days_since_last_move ─────────────────────────────

class TestStalenessFeatures:
    """
    Known price sequence:  [10, 10, 11, 11, 11, 12]
    Daily changes:         [NaN,  0,  1,  0,  0,  1]
    days_since_last_move:  [NaN,  1,  0,  1,  2,  0]
    price_moved (lag-1):   [NaN, NaN,  0,  1,  0,  0]
    """

    PRICES = [10.0, 10.0, 11.0, 11.0, 11.0, 12.0]

    def test_days_since_last_move_direct(self):
        prices = pd.Series(self.PRICES)
        change = prices.diff()
        dslm   = _days_since_last_move(change)
        # row 0: NaN (change is NaN)
        assert np.isnan(dslm.iloc[0])
        # row 1: change=0 → streak=1
        assert dslm.iloc[1] == 1.0
        # row 2: change=1 → streak resets to 0
        assert dslm.iloc[2] == 0.0
        # row 3: change=0 → streak=1
        assert dslm.iloc[3] == 1.0
        # row 4: change=0 → streak=2
        assert dslm.iloc[4] == 2.0
        # row 5: change=1 → streak resets to 0
        assert dslm.iloc[5] == 0.0

    def test_price_moved_in_feature_matrix(self):
        """price_moved at row i reflects whether price changed on row i-1."""
        df   = _make_df(self.PRICES)
        feat = build_feature_matrix(df, horizons=[1])
        # row 0: NaN — shift of row -1 (out of bounds)
        assert np.isnan(feat["price_moved"].iloc[0])
        # row 1: shift of row 0; row 0 change is NaN (first diff) → NaN
        assert np.isnan(feat["price_moved"].iloc[1])
        # row 2: shift of row 1; row 1 change = 0 (10→10) → price_moved = 0
        assert feat["price_moved"].iloc[2] == 0.0
        # row 3: shift of row 2; row 2 change = 1 (10→11) → price_moved = 1
        assert feat["price_moved"].iloc[3] == 1.0
        # row 4: shift of row 3; row 3 change = 0 (11→11) → price_moved = 0
        assert feat["price_moved"].iloc[4] == 0.0

    def test_days_since_last_move_in_feature_matrix(self):
        """days_since_last_move in feat matrix is the streak shifted by 1 (past-only)."""
        df   = _make_df(self.PRICES)
        feat = build_feature_matrix(df, horizons=[1])
        # row 0: NaN (shift of NaN)
        assert np.isnan(feat["days_since_last_move"].iloc[0])
        # row 2: yesterday dslm was 1 → feature = 1
        assert feat["days_since_last_move"].iloc[2] == 1.0
        # row 3: yesterday dslm was 0 (move occurred) → feature = 0
        assert feat["days_since_last_move"].iloc[3] == 0.0
        # row 4: yesterday dslm was 1 → feature = 1
        assert feat["days_since_last_move"].iloc[4] == 1.0
        # row 5: yesterday dslm was 2 → feature = 2
        assert feat["days_since_last_move"].iloc[5] == 2.0

    def test_dslm_all_zeros_constant_series(self):
        """Constant price → change always 0 → streak increments each row (except row 0)."""
        prices = [5.0] * 10
        change = pd.Series(prices).diff()
        dslm   = _days_since_last_move(change)
        assert np.isnan(dslm.iloc[0])
        for i in range(1, 10):
            assert dslm.iloc[i] == float(i)

    def test_dslm_all_moves(self):
        """Every day has a move → streak always resets to 0."""
        prices = list(range(1, 11))
        change = pd.Series(prices, dtype=float).diff()
        dslm   = _days_since_last_move(change)
        assert np.isnan(dslm.iloc[0])
        for i in range(1, 10):
            assert dslm.iloc[i] == 0.0


# ── Group 4: Rolling features are strictly trailing ───────────────────────────

class TestTrailingRolling:
    """A spike injected at row N must not affect features at rows before N."""

    def test_vol_chg_7d_is_trailing(self):
        prices = [10.0] * 20
        df_base = _make_df(prices)
        feat_base = build_feature_matrix(df_base, horizons=[1])

        prices_spike = prices.copy()
        prices_spike[-1] = 1000.0
        df_spike = _make_df(prices_spike)
        feat_spike = build_feature_matrix(df_spike, horizons=[1])

        # All rows before the last 8 (spike + 7-day window) must be identical
        safe_rows = len(prices) - 8
        base_vals  = feat_base["vol_chg_7d"].iloc[:safe_rows].values
        spike_vals = feat_spike["vol_chg_7d"].iloc[:safe_rows].values
        mask = ~(np.isnan(base_vals) & np.isnan(spike_vals))
        assert np.allclose(base_vals[mask], spike_vals[mask], equal_nan=True)

    def test_moves_30d_is_trailing(self):
        prices = [float(i) for i in range(1, 51)]
        df_base = _make_df(prices)
        feat_base = build_feature_matrix(df_base, horizons=[1])

        prices_flat = prices.copy()
        for i in range(40, 50):
            prices_flat[i] = prices_flat[39]   # flat tail = zero changes
        df_flat = _make_df(prices_flat)
        feat_flat = build_feature_matrix(df_flat, horizons=[1])

        # rows 0..38 (before the flat region) must be identical
        base_vals = feat_base["moves_30d"].iloc[:39].values
        flat_vals = feat_flat["moves_30d"].iloc[:39].values
        mask = ~(np.isnan(base_vals) & np.isnan(flat_vals))
        assert np.allclose(base_vals[mask], flat_vals[mask], equal_nan=True)

    def test_chg_0_and_chg_1_trailing_in_rolling_context(self):
        """chg_0 and chg_1 at row i are unaffected by prices at row i+k (k>=1)."""
        prices = [10.0, 11.0, 12.0, 13.0, 14.0]
        df = _make_df(prices)
        feat = build_feature_matrix(df, horizons=[1])

        prices_modified = prices.copy()
        prices_modified[4] = 9999.0
        df_mod = _make_df(prices_modified)
        feat_mod = build_feature_matrix(df_mod, horizons=[1])

        # rows 0..2: chg_0 unchanged (row 3 chg_0 = price[3]-price[2], unaffected)
        for col in ("chg_0", "chg_1"):
            for i in range(3):
                v1 = feat[col].iloc[i]
                v2 = feat_mod[col].iloc[i]
                if np.isnan(v1):
                    assert np.isnan(v2)
                else:
                    assert abs(v1 - v2) < 1e-9, f"{col} changed at row {i}"


# ── Group 5: Excluded columns absent from output ──────────────────────────────

class TestExcludedColumnsAbsent:
    """Every column in EXCLUDED_COLS must be absent from build_feature_matrix output."""

    def _make_full_df(self) -> pd.DataFrame:
        """Minimal df with target + a few excluded columns present."""
        n = 50
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        df = pd.DataFrame({
            DATE_COL:                          dates,
            TARGET_COL:                        np.linspace(10, 20, n),
            "$ Change (Generic)":              np.random.randn(n),
            "% Change (Generic)":              np.random.randn(n),
            "WoW change (Generic)":            np.random.randn(n),
            "7-day SMA (Generic)":             np.linspace(10, 20, n),
            "ACCU spot price (HIR)":           np.linspace(11, 21, n),
            "% premium HIR over Generic":      np.ones(n),
            "$ premium HIR over Generic":      np.ones(n),
            "ACCU spot price (SFM) - No co-benefits": np.linspace(10.5, 20.5, n),
            "$ change (HIR)":                  np.random.randn(n),
            "LGC spot price":                  np.linspace(5, 8, n),   # should appear as lgc_chg
        })
        return df

    def test_no_excluded_col_in_output(self):
        df   = self._make_full_df()
        feat = build_feature_matrix(df, horizons=[1])
        for excl in EXCLUDED_COLS:
            assert excl not in feat.columns, (
                f"Excluded column '{excl}' found in feature matrix output"
            )

    def test_target_col_not_in_features(self):
        df   = _make_df(list(range(1, 31)))
        feat = build_feature_matrix(df, horizons=[1])
        assert TARGET_COL not in feat.columns

    def test_build_features_no_excluded(self):
        """build_features (which calls build_feature_matrix) also excludes all cols."""
        n = 100
        dates  = pd.date_range("2020-01-01", periods=n, freq="D")
        prices = np.linspace(10, 20, n)
        df = pd.DataFrame({DATE_COL: dates, TARGET_COL: prices})
        train, val, test = _split(df, 70, 15)
        ft, fv, fte = build_features(train, val, test, horizons=[1, 7], warmup=5)
        for split_df in (ft, fv, fte):
            for excl in EXCLUDED_COLS:
                assert excl not in split_df.columns

    def test_date_col_present_target_absent(self):
        """Date column must be retained; TARGET_COL must be absent."""
        df   = _make_df(list(range(1, 31)))
        feat = build_feature_matrix(df, horizons=[1])
        assert DATE_COL in feat.columns
        assert TARGET_COL not in feat.columns

    def test_warmup_drops_correct_rows(self):
        """build_features with warmup=5 should drop exactly 5 rows from train head."""
        n = 100
        dates  = pd.date_range("2020-01-01", periods=n, freq="D")
        prices = np.linspace(10, 20, n)
        df = pd.DataFrame({DATE_COL: dates, TARGET_COL: prices})
        train, val, test = _split(df, 70, 15)
        ft_5, _, _  = build_features(train, val, test, horizons=[1], warmup=5)
        ft_10, _, _ = build_features(train, val, test, horizons=[1], warmup=10)
        assert len(ft_5) == len(ft_10) + 5
