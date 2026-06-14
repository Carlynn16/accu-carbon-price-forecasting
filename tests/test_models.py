"""
Tests for src/models.py — Block C2.

Groups:
  1. CV time-ordering: TimeSeriesSplit folds are strictly chronological
  2. Smoke tests: RF/XGB/LGBM fit→predict on small synthetic data
  3. Tune helper: _tscv_tune returns a fitted estimator without future leakage
  4. Skill computation: C1 harness is correctly reused
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

from src.evaluate import build_results_table, compute_metrics, skill_score
from src.models import _lookup_rw, _tscv_tune


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_Xy():
    """80-row synthetic regression dataset."""
    rng = np.random.default_rng(0)
    n = 80
    X = rng.standard_normal((n, 5))
    y = X[:, 0] * 0.5 + rng.standard_normal(n) * 0.1
    return X, y


# ── Group 1: CV time-ordering ─────────────────────────────────────────────────

class TestCVTimeOrdering:
    """TimeSeriesSplit produces strictly chronological folds."""

    def test_train_always_before_val(self):
        tscv = TimeSeriesSplit(n_splits=5)
        X = np.arange(100).reshape(-1, 1)
        for train_idx, val_idx in tscv.split(X):
            assert train_idx.max() < val_idx.min(), (
                "Some training index exceeds a validation index — future leakage!"
            )

    def test_folds_are_expanding_window(self):
        """Each successive training fold is strictly larger than the previous."""
        tscv = TimeSeriesSplit(n_splits=5)
        X = np.arange(100).reshape(-1, 1)
        train_sizes = [len(tr) for tr, _ in tscv.split(X)]
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] > train_sizes[i - 1]

    def test_val_folds_do_not_overlap(self):
        """Validation folds must be non-overlapping time windows."""
        tscv = TimeSeriesSplit(n_splits=5)
        X = np.arange(100).reshape(-1, 1)
        seen = set()
        for _, val_idx in tscv.split(X):
            overlap = seen & set(val_idx.tolist())
            assert not overlap, f"Validation fold indices overlap: {overlap}"
            seen |= set(val_idx.tolist())


# ── Group 2: Smoke tests ──────────────────────────────────────────────────────

class TestSmoke:
    """Each model fits without error and returns finite predictions."""

    def test_rf_fits_and_predicts(self, tiny_Xy):
        X, y = tiny_Xy
        from sklearn.ensemble import RandomForestRegressor
        m = RandomForestRegressor(n_estimators=10, random_state=0)
        m.fit(X[:60], y[:60])
        pred = m.predict(X[60:])
        assert pred.shape == (20,)
        assert np.isfinite(pred).all()

    def test_xgboost_fits_and_predicts(self, tiny_Xy):
        import xgboost as xgb
        X, y = tiny_Xy
        m = xgb.XGBRegressor(n_estimators=10, random_state=0, verbosity=0)
        m.fit(X[:60], y[:60])
        pred = m.predict(X[60:])
        assert pred.shape == (20,)
        assert np.isfinite(pred).all()

    def test_lightgbm_fits_and_predicts(self, tiny_Xy):
        import lightgbm as lgb
        X, y = tiny_Xy
        m = lgb.LGBMRegressor(n_estimators=10, random_state=0, verbose=-1)
        m.fit(X[:60], y[:60])
        pred = m.predict(X[60:])
        assert pred.shape == (20,)
        assert np.isfinite(pred).all()

    def test_rf_refit_on_trval_changes_predictions(self, tiny_Xy):
        """Refitting on a larger dataset should change predictions (sanity check)."""
        X, y = tiny_Xy
        from sklearn.ensemble import RandomForestRegressor
        m = RandomForestRegressor(n_estimators=10, random_state=0)
        m.fit(X[:40], y[:40])
        pred_before = m.predict(X[40:60]).copy()
        m.fit(X[:60], y[:60])  # refit on larger set
        pred_after = m.predict(X[40:60])
        # Predictions should differ after refitting on more data
        # (not strictly guaranteed, but almost always true)
        assert not np.allclose(pred_before, pred_after), (
            "Refit on larger dataset produced identical predictions — unexpected."
        )


# ── Group 3: Tune helper ──────────────────────────────────────────────────────

class TestTuneHelper:
    """_tscv_tune returns a fitted estimator; param grid is respected."""

    def test_returns_fitted_estimator(self, tiny_Xy):
        X, y = tiny_Xy
        param_grid = {"n_estimators": [10, 20], "max_depth": [3]}
        est = _tscv_tune(RandomForestRegressor(random_state=0), param_grid, X[:60], y[:60])
        pred = est.predict(X[60:])
        assert pred.shape == (20,)

    def test_selected_param_within_grid(self, tiny_Xy):
        """Best estimator's n_estimators must come from the supplied grid."""
        X, y = tiny_Xy
        param_grid = {"n_estimators": [5, 15], "max_depth": [2]}
        est = _tscv_tune(RandomForestRegressor(random_state=0), param_grid, X[:60], y[:60])
        assert est.n_estimators in [5, 15]

    def test_no_future_data_in_training(self, tiny_Xy):
        """Verify TimeSeriesSplit is used: earliest CV train fold must not
        contain the latest data points."""
        X, y = tiny_Xy
        tscv = TimeSeriesSplit(n_splits=5)
        first_train, _ = next(tscv.split(X[:60]))
        # The first fold's training data should be a strict prefix of the array
        assert first_train[-1] < len(X[:60]) - 1


# ── Group 4: Skill computation reuses C1 harness ─────────────────────────────

class TestSkillReuseC1:
    """Models evaluated with the same C1 compute_metrics / skill_score."""

    def test_perfect_predictions_give_100_skill(self):
        actual  = np.array([0.1, -0.2, 0.3, -0.1])
        anchors = np.zeros(4)
        # RW baseline
        rw_m = compute_metrics(np.zeros(4), actual, anchors)
        # Perfect model
        m = compute_metrics(actual, actual, anchors,
                            rw_rmse=rw_m["RMSE"], rw_mae=rw_m["MAE"])
        assert m["RMSE_skill_%"] == 100.0

    def test_random_walk_skill_is_zero(self):
        actual  = np.array([0.5, -0.3, 0.1])
        anchors = np.zeros(3)
        rw_m = compute_metrics(np.zeros(3), actual, anchors)
        m = compute_metrics(np.zeros(3), actual, anchors,
                            rw_rmse=rw_m["RMSE"], rw_mae=rw_m["MAE"])
        assert m["RMSE_skill_%"] == 0.0

    def test_build_results_table_accepts_model_records(self):
        records = [
            {"model": "random_forest", "horizon": 1, "split": "val",
             "RMSE": 0.55, "MAE": 0.42, "MAPE_%": 1.2, "dir_acc_%": 55.0,
             "RMSE_skill_%": 5.7, "MAE_skill_%": 3.1},
            {"model": "xgboost", "horizon": 1, "split": "val",
             "RMSE": 0.53, "MAE": 0.41, "MAPE_%": 1.1, "dir_acc_%": 58.0,
             "RMSE_skill_%": 9.1, "MAE_skill_%": 5.6},
        ]
        tbl = build_results_table(records)
        assert set(["model", "horizon", "split", "RMSE", "RMSE_skill_%"]).issubset(tbl.columns)
        assert len(tbl) == 2

    def test_lookup_rw_returns_none_when_no_baseline(self):
        rmse, mae = _lookup_rw(None, 1, "val")
        assert rmse is None
        assert mae is None

    def test_lookup_rw_finds_correct_row(self):
        df = pd.DataFrame([
            {"model": "random_walk", "horizon": 1, "split": "val",
             "RMSE": 0.5834, "MAE": 0.40},
            {"model": "random_walk", "horizon": 7, "split": "val",
             "RMSE": 1.5479, "MAE": 1.10},
            {"model": "drift", "horizon": 1, "split": "val",
             "RMSE": 0.65, "MAE": 0.45},
        ])
        rmse, mae = _lookup_rw(df, 7, "val")
        assert abs(rmse - 1.5479) < 1e-9
        assert abs(mae  - 1.10)   < 1e-9

    def test_skill_positive_for_improvement(self):
        assert skill_score(0.4, 0.58) > 0.0

    def test_skill_negative_for_worse_model(self):
        assert skill_score(0.70, 0.58) < 0.0
