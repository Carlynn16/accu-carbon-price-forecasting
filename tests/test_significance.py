"""
Tests for src/significance.py — Block D.

Groups:
  1. IdenticalForecasts  — DM on identical predictions handled gracefully (no /0)
  2. SyntheticForecasts  — clearly better/worse synthetic forecasts flagged correctly
  3. HACLag              — HAC lag = h-1; results finite for all supported horizons
  4. DirAccMovedays      — directional_accuracy_move_days excludes stale days
"""

import numpy as np
import pandas as pd
import pytest

from src.significance import (
    diebold_mariano,
    directional_accuracy_move_days,
    run_dm_tests,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _preds_df(actual, pred, model="m", h=1, split="test"):
    """Build a minimal preds_df for a single model + RW."""
    n = len(actual)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    rows = []
    for model_name, pch in [(model, pred), ("random_walk", np.zeros(n))]:
        for i in range(n):
            rows.append({
                "model":         model_name,
                "horizon":       h,
                "split":         split,
                "date":          dates[i],
                "pred_change":   float(pch[i]),
                "actual_change": float(actual[i]),
                "price_anchor":  10.0,
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Group 1: Identical forecasts → no division-by-zero
# ─────────────────────────────────────────────────────────────────────────────

class TestIdenticalForecasts:
    def test_both_same_as_rw_verdict_not_significant(self):
        """When model == RW (both zero), d_t = 0 for all t; result must be graceful."""
        rng    = np.random.default_rng(0)
        actual = rng.standard_normal(200)
        dm     = diebold_mariano(actual, np.zeros(200), np.zeros(200), h=1)
        assert dm["verdict"] == "not significant"

    def test_both_same_as_rw_pvalue_near_one(self):
        rng    = np.random.default_rng(1)
        actual = rng.standard_normal(200)
        dm     = diebold_mariano(actual, np.zeros(200), np.zeros(200), h=1)
        assert dm["p_value"] > 0.05

    def test_both_same_as_rw_dm_is_zero(self):
        rng    = np.random.default_rng(2)
        actual = rng.standard_normal(100)
        dm     = diebold_mariano(actual, np.zeros(100), np.zeros(100), h=1)
        assert dm["DM_stat"] == 0.0

    def test_identical_nonzero_forecasts(self):
        """Model predicts same series as RW (non-zero); d_t = 0 still."""
        rng    = np.random.default_rng(3)
        pred   = rng.standard_normal(150)
        actual = rng.standard_normal(150)
        dm     = diebold_mariano(actual, pred, pred, h=1)
        assert dm["DM_stat"] == 0.0
        assert dm["verdict"] == "not significant"


# ─────────────────────────────────────────────────────────────────────────────
# Group 2: Synthetic — clearly better/worse flagged correctly
# ─────────────────────────────────────────────────────────────────────────────

class TestSyntheticForecasts:
    def test_perfect_model_significantly_better(self):
        """Perfect forecast (zero error) vs RW: DM_stat < 0, p < 0.05."""
        rng    = np.random.default_rng(42)
        actual = rng.standard_normal(300)
        dm     = diebold_mariano(actual, actual, np.zeros(300), h=1)
        assert dm["DM_stat"] < 0, "Perfect model should yield negative DM stat"
        assert dm["p_value"] < 0.05
        assert dm["verdict"] == "significantly better"

    def test_terrible_model_significantly_worse(self):
        """Model predicting large errors: DM_stat > 0, p < 0.05."""
        rng    = np.random.default_rng(42)
        actual = rng.standard_normal(300)
        dm     = diebold_mariano(actual, -10.0 * actual, np.zeros(300), h=1)
        assert dm["DM_stat"] > 0
        assert dm["p_value"] < 0.05
        assert dm["verdict"] == "significantly worse"

    def test_negative_dm_matches_lower_model_loss(self):
        """DM_stat < 0 iff model's mean squared error < RW's mean squared error."""
        rng    = np.random.default_rng(7)
        actual = rng.standard_normal(250) + 5.0      # non-zero mean
        pred_m = np.full(250, actual.mean())          # predicts mean → beats zero-pred
        dm     = diebold_mariano(actual, pred_m, np.zeros(250), h=1)
        assert dm["mean_d"] < 0, "mean_d < 0 implies model loss < RW loss"
        assert dm["DM_stat"] < 0

    def test_run_dm_tests_tidy_output(self):
        """run_dm_tests returns a tidy DataFrame with expected columns."""
        rng    = np.random.default_rng(99)
        actual = rng.standard_normal(200) + 3.0
        pred   = np.full(200, actual.mean())
        df     = _preds_df(actual, pred, model="good_model")
        out    = run_dm_tests(df, horizons=(1,), splits=("test",))
        assert not out.empty
        for col in ["model", "horizon", "split", "DM_stat", "p_value", "verdict"]:
            assert col in out.columns
        assert len(out) == 1
        assert out.iloc[0]["model"] == "good_model"

    def test_run_dm_tests_excludes_rw_from_models(self):
        """random_walk should not appear as a tested model in the output."""
        rng    = np.random.default_rng(5)
        actual = rng.standard_normal(150)
        pred   = rng.standard_normal(150)
        df     = _preds_df(actual, pred, model="model_a")
        out    = run_dm_tests(df, horizons=(1,), splits=("test",))
        assert "random_walk" not in out["model"].values


# ─────────────────────────────────────────────────────────────────────────────
# Group 3: HAC lag = h-1 for all supported horizons
# ─────────────────────────────────────────────────────────────────────────────

class TestHACLag:
    def test_h1_produces_finite_result(self):
        """h=1 → lag M=0 (no HAC smoothing). Result should be finite."""
        rng  = np.random.default_rng(10)
        real = rng.standard_normal(200)
        pred = rng.standard_normal(200) * 0.5
        dm   = diebold_mariano(real, pred, np.zeros(200), h=1)
        assert np.isfinite(dm["DM_stat"])
        assert 0.0 <= dm["p_value"] <= 1.0

    def test_h7_produces_finite_result(self):
        """h=7 → lag M=6."""
        rng  = np.random.default_rng(11)
        real = rng.standard_normal(250)
        pred = rng.standard_normal(250) * 0.8
        dm   = diebold_mariano(real, pred, np.zeros(250), h=7)
        assert np.isfinite(dm["DM_stat"])
        assert 0.0 <= dm["p_value"] <= 1.0

    def test_h30_produces_finite_result(self):
        """h=30 → lag M=29."""
        rng  = np.random.default_rng(12)
        real = rng.standard_normal(250)
        pred = rng.standard_normal(250) * 0.5
        dm   = diebold_mariano(real, pred, np.zeros(250), h=30)
        assert np.isfinite(dm["DM_stat"])
        assert 0.0 <= dm["p_value"] <= 1.0

    def test_h1_vs_h7_stats_differ(self):
        """Using different lags (h=1 vs h=7) on same data must give different DM stats."""
        rng  = np.random.default_rng(20)
        real = rng.standard_normal(200)
        pred = rng.standard_normal(200) * 0.8
        dm1  = diebold_mariano(real, pred, np.zeros(200), h=1)
        dm7  = diebold_mariano(real, pred, np.zeros(200), h=7)
        assert dm1["DM_stat"] != dm7["DM_stat"], (
            "h=1 and h=7 use different HAC lags + HLN corrections; stats should differ"
        )

    def test_unknown_loss_raises(self):
        with pytest.raises(ValueError, match="Unknown loss"):
            diebold_mariano(np.ones(50), np.zeros(50), np.zeros(50), h=1, loss="linex")


# ─────────────────────────────────────────────────────────────────────────────
# Group 4: Directional accuracy on move-days
# ─────────────────────────────────────────────────────────────────────────────

def _dir_df(actual, pred, model="m", h=1, split="test"):
    n = len(actual)
    return pd.DataFrame({
        "model":         [model] * n,
        "horizon":       [h] * n,
        "split":         [split] * n,
        "date":          pd.date_range("2023-01-01", periods=n, freq="B"),
        "actual_change": actual.astype(float),
        "pred_change":   pred.astype(float),
        "price_anchor":  np.full(n, 10.0),
    })


class TestDirAccMovedays:
    def test_all_move_days_correct_gives_100(self):
        actual = np.array([1.0, -2.0, 3.0, -0.5, 0.0])
        pred   = np.array([0.5, -1.0, 2.0, -0.1, 0.0])   # same sign as actual
        df     = _dir_df(actual, pred)
        res    = directional_accuracy_move_days(df, h=1, split="test")
        assert len(res) == 1
        assert res.loc[0, "dir_acc_move_%"] == 100.0

    def test_stale_days_excluded_from_count(self):
        """actual=0 rows must not count toward n_move."""
        actual = np.array([0.0, 0.0, 1.0, -1.0, 0.0])
        pred   = np.array([1.0, -1.0, 0.5, -0.5, 0.5])
        df     = _dir_df(actual, pred)
        res    = directional_accuracy_move_days(df, h=1, split="test")
        assert res.loc[0, "n_move"] == 2

    def test_all_wrong_gives_zero(self):
        actual = np.array([1.0, 2.0, 3.0])
        pred   = np.array([-1.0, -2.0, -3.0])
        df     = _dir_df(actual, pred)
        res    = directional_accuracy_move_days(df, h=1, split="test")
        assert res.loc[0, "dir_acc_move_%"] == 0.0

    def test_horizon_filter_applied(self):
        """Should only return rows matching the requested horizon."""
        actual = np.array([1.0, -1.0])
        pred   = np.array([0.5, -0.5])
        df1    = _dir_df(actual, pred, h=1)
        df7    = _dir_df(actual, pred, h=7)
        combined = pd.concat([df1, df7], ignore_index=True)
        res    = directional_accuracy_move_days(combined, h=1, split="test")
        assert res.loc[0, "n_move"] == 2   # only h=1 rows counted
