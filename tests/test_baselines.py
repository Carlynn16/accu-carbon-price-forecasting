"""
Tests for src/baselines.py and src/evaluate.py — Block C1.

Groups:
  1. Random walk: predicts zero, level == anchor, RMSE == change RMSE, skill == 0%
  2. Skill score arithmetic
  3. Directional accuracy (correct values; flat rows ignored)
  4. Per-horizon NaN masking (prepare_horizon drop counts)
"""

import numpy as np
import pandas as pd
import pytest

from src.baselines import drift_predict, random_walk_predict
from src.evaluate import (
    build_results_table,
    compute_metrics,
    directional_accuracy,
    prepare_horizon,
    skill_score,
)

DATE_COL   = "Date"
ANCHOR_COL = "price_anchor"


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_feat_df(
    prices: list[float],
    horizons: tuple[int, ...] = (1, 7),
    start: str = "2020-01-01",
) -> pd.DataFrame:
    """Minimal feature DataFrame with price_anchor and targets for tests."""
    n     = len(prices)
    dates = pd.date_range(start, periods=n, freq="D")
    arr   = np.array(prices, dtype=float)

    df = pd.DataFrame({
        DATE_COL:   dates,
        ANCHOR_COL: arr,
        "feat_a":   np.ones(n),
    })
    for h in horizons:
        tgt = np.empty(n)
        tgt[:n - h] = arr[h:] - arr[:n - h]
        tgt[n - h:] = np.nan
        df[f"target_{h}"] = tgt
    return df


# ── Group 1: Random walk ──────────────────────────────────────────────────────

class TestRandomWalk:
    """random_walk predicts zero; level reconstruction; RMSE and skill invariants."""

    CHANGES = np.array([0.5, -0.3, 0.1, 0.0,  0.2, -0.4])
    ANCHORS = np.array([20.0, 20.5, 20.2, 20.3, 20.3, 20.5])

    def test_all_zeros(self):
        pred = random_walk_predict(len(self.CHANGES))
        assert np.all(pred == 0.0)
        assert pred.shape == (len(self.CHANGES),)

    def test_reconstructed_level_equals_anchor(self):
        """pred_level = anchor + 0 = anchor."""
        pred = random_walk_predict(len(self.CHANGES))
        pred_level = self.ANCHORS + pred
        np.testing.assert_array_equal(pred_level, self.ANCHORS)

    def test_rmse_equals_change_rmse(self):
        """RMSE(rw) = sqrt(mean(actual_change^2)), because anchor cancels."""
        pred    = random_walk_predict(len(self.CHANGES))
        metrics = compute_metrics(pred, self.CHANGES, self.ANCHORS)
        expected = float(np.sqrt(np.mean(self.CHANGES ** 2)))
        assert abs(metrics["RMSE"] - expected) < 1e-9

    def test_rmse_skill_is_zero(self):
        """random walk evaluated against itself has 0% skill."""
        pred    = random_walk_predict(len(self.CHANGES))
        rw_m    = compute_metrics(pred, self.CHANGES, self.ANCHORS)
        final_m = compute_metrics(
            pred, self.CHANGES, self.ANCHORS,
            rw_rmse=rw_m["RMSE"], rw_mae=rw_m["MAE"],
        )
        assert final_m["RMSE_skill_%"] == 0.0
        assert final_m["MAE_skill_%"]  == 0.0

    def test_skill_none_when_no_baseline_passed(self):
        pred    = random_walk_predict(3)
        metrics = compute_metrics(pred, np.array([0.1, 0.2, 0.3]), np.array([10.0] * 3))
        assert np.isnan(metrics["RMSE_skill_%"])
        assert np.isnan(metrics["MAE_skill_%"])

    def test_zero_input(self):
        pred = random_walk_predict(0)
        assert len(pred) == 0


# ── Group 2: Skill score arithmetic ──────────────────────────────────────────

class TestSkillScore:
    """skill_score(model, baseline) arithmetic."""

    def test_better_than_baseline(self):
        # model RMSE = 1.0, baseline RMSE = 2.0 → 50% improvement
        assert abs(skill_score(1.0, 2.0) - 50.0) < 1e-9

    def test_worse_than_baseline(self):
        # model RMSE = 3.0, baseline RMSE = 2.0 → -50% (worse)
        assert abs(skill_score(3.0, 2.0) - (-50.0)) < 1e-9

    def test_tied_with_baseline(self):
        assert abs(skill_score(1.5, 1.5) - 0.0) < 1e-9

    def test_perfect_model(self):
        # model RMSE = 0 → 100% skill
        assert abs(skill_score(0.0, 2.0) - 100.0) < 1e-9

    def test_skill_in_compute_metrics(self):
        """compute_metrics correctly wires skill scores."""
        pred_better = np.array([0.05, -0.05, 0.05])   # smaller errors than rw
        actual      = np.array([0.1,  -0.1,  0.1])
        anchors     = np.array([20.0,  20.0, 20.0])

        # RW metrics
        rw_pred    = random_walk_predict(3)
        rw_metrics = compute_metrics(rw_pred, actual, anchors)

        # Better-model metrics
        m = compute_metrics(
            pred_better, actual, anchors,
            rw_rmse=rw_metrics["RMSE"],
            rw_mae=rw_metrics["MAE"],
        )
        assert m["RMSE_skill_%"] > 0.0, "A better model should have positive skill"
        assert m["MAE_skill_%"]  > 0.0


# ── Group 3: Directional accuracy ────────────────────────────────────────────

class TestDirectionalAccuracy:
    """dir_acc ignores flat rows; correct sign-match logic."""

    def test_ignores_flat_rows(self):
        pred   = np.array([0.5,  -0.3,  0.1,  0.0])
        actual = np.array([0.3,   0.0, -0.2,  0.0])
        # genuine-move rows: 0 (sign match ✓) and 2 (sign mismatch ✗)
        # rows 1 and 3 have actual==0 → ignored
        acc = directional_accuracy(pred, actual)
        assert abs(acc - 0.5) < 1e-9

    def test_all_flat_returns_nan(self):
        acc = directional_accuracy(np.array([0.5, -0.3]), np.array([0.0, 0.0]))
        assert np.isnan(acc)

    def test_all_correct(self):
        pred   = np.array([0.3,  -0.5,  0.1])
        actual = np.array([0.2,  -0.1,  0.4])
        acc = directional_accuracy(pred, actual)
        assert abs(acc - 1.0) < 1e-9

    def test_all_wrong(self):
        pred   = np.array([ 0.3, -0.5])
        actual = np.array([-0.2,  0.1])
        acc = directional_accuracy(pred, actual)
        assert abs(acc - 0.0) < 1e-9

    def test_dir_acc_in_compute_metrics(self):
        """compute_metrics passes through directional accuracy correctly."""
        pred   = np.array([0.3, -0.5,  0.0])   # row 2 pred = 0 (sign ambiguous, but actual is 0.0 → ignored)
        actual = np.array([0.2, -0.1,  0.0])
        anchors = np.array([20.0, 20.0, 20.0])
        m = compute_metrics(pred, actual, anchors)
        # rows 0 and 1 are genuine; both match → 100%
        assert abs(m["dir_acc_%"] - 100.0) < 1e-6

    def test_random_walk_direction_on_genuine_moves(self):
        """Random walk predicts zero, so sign comparison is sign(0)==sign(actual_change).
        sign(0)==+1 by numpy convention? np.sign(0)=0, not matching ±1."""
        actual  = np.array([0.3, -0.5, 0.1])
        anchors = np.array([20.0] * 3)
        pred = random_walk_predict(3)
        m    = compute_metrics(pred, actual, anchors)
        # np.sign(0) = 0; sign(0) != sign(0.3) = 1 and sign(0) != sign(-0.5) = -1
        # so dir_acc should be 0% for a pure random walk (sign(0) matches no ±1 sign)
        assert m["dir_acc_%"] == 0.0


# ── Group 4: Per-horizon NaN masking ─────────────────────────────────────────

class TestPrepareHorizon:
    """prepare_horizon drops only NaN-target rows for the requested horizon."""

    def _df(self, n: int = 15) -> pd.DataFrame:
        return _make_feat_df(list(range(1, n + 1)), horizons=(1, 7))

    def test_h1_drops_exactly_1_row(self):
        df   = self._df(15)
        data = prepare_horizon(df, h=1)
        assert data["n_dropped"] == 1
        assert data["n_rows"]    == 14

    def test_h7_drops_exactly_7_rows(self):
        df   = self._df(15)
        data = prepare_horizon(df, h=7)
        assert data["n_dropped"] == 7
        assert data["n_rows"]    == 8

    def test_price_anchor_not_in_X(self):
        df   = self._df(15)
        data = prepare_horizon(df, h=1)
        assert ANCHOR_COL not in data["X"].columns
        assert DATE_COL   not in data["X"].columns
        assert "target_1" not in data["X"].columns

    def test_price_anchor_shape_matches_y(self):
        df   = self._df(15)
        data = prepare_horizon(df, h=1)
        assert len(data["price_anchor"]) == len(data["y"])
        assert len(data["y"])            == data["n_rows"]

    def test_y_is_correct_change(self):
        """y must equal price(t+h) - price(t) for each row."""
        prices = [10.0, 11.0, 12.0, 14.0, 13.0]
        df     = _make_feat_df(prices, horizons=(1,))
        data   = prepare_horizon(df, h=1)
        # After dropping last row, n_rows = 4
        assert data["n_rows"] == 4
        expected_y = np.array([1.0, 1.0, 2.0, -1.0])
        np.testing.assert_allclose(data["y"], expected_y, atol=1e-9)

    def test_no_nan_in_y_after_mask(self):
        df   = self._df(15)
        data = prepare_horizon(df, h=7)
        assert not np.any(np.isnan(data["y"]))

    def test_build_results_table_columns(self):
        records = [
            {"model": "rw",    "horizon": 1, "split": "val",
             "RMSE": 0.1, "MAE": 0.08, "MAPE_%": 0.5, "dir_acc_%": 40.0,
             "RMSE_skill_%": 0.0, "MAE_skill_%": 0.0},
            {"model": "drift", "horizon": 1, "split": "val",
             "RMSE": 0.09, "MAE": 0.07, "MAPE_%": 0.4, "dir_acc_%": 45.0,
             "RMSE_skill_%": 10.0, "MAE_skill_%": 12.5},
        ]
        tbl = build_results_table(records)
        assert "model" in tbl.columns
        assert "RMSE_skill_%" in tbl.columns
        assert len(tbl) == 2
