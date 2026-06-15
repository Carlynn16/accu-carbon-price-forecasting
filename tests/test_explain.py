"""
Smoke tests for src/explain.py — Block E.

Groups:
  1. SHAPShape      — shap_values array has shape (n_test, n_features)
  2. SHAPAdditivity — per-row sum + base_value ≈ model prediction (exact for trees)
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.explain import compute_shap_values


# ── Synthetic fixture ─────────────────────────────────────────────────────────

def _make_rf_and_data(n_train: int = 120, n_test: int = 30, n_feat: int = 6, seed: int = 0):
    """Fit a small RF on synthetic data; return (model, X_test)."""
    rng    = np.random.default_rng(seed)
    X_tr   = rng.standard_normal((n_train, n_feat)).astype(np.float32)
    y_tr   = rng.standard_normal(n_train)
    X_te   = rng.standard_normal((n_test,  n_feat)).astype(np.float32)

    model  = RandomForestRegressor(n_estimators=20, random_state=seed)
    model.fit(X_tr, y_tr)

    return model, X_te


# ── Group 1: shape ────────────────────────────────────────────────────────────

class TestSHAPShape:
    def test_shap_values_shape_matches_n_test_n_features(self):
        """shap_values must have shape (n_test, n_features)."""
        model, X_te = _make_rf_and_data(n_test=30, n_feat=6)
        sv, _ = compute_shap_values(model, X_te)
        assert sv.shape == X_te.shape, (
            f"Expected SHAP shape {X_te.shape}, got {sv.shape}"
        )

    def test_shap_values_are_finite(self):
        model, X_te = _make_rf_and_data(n_test=25, n_feat=5, seed=1)
        sv, _ = compute_shap_values(model, X_te)
        assert np.isfinite(sv).all(), "Non-finite SHAP values detected"

    def test_base_value_is_scalar(self):
        model, X_te = _make_rf_and_data(n_test=20, n_feat=4, seed=2)
        _, base = compute_shap_values(model, X_te)
        assert np.isscalar(base) or np.ndim(base) == 0

    def test_shap_values_vary_across_features(self):
        """Different features should have different mean |SHAP| (not all identical)."""
        model, X_te = _make_rf_and_data(n_test=50, n_feat=6, seed=3)
        sv, _ = compute_shap_values(model, X_te)
        mean_abs = np.abs(sv).mean(axis=0)
        assert mean_abs.std() > 1e-12, "All features have identical SHAP importance"

    @pytest.mark.parametrize("n_feat", [3, 8, 15])
    def test_shap_shape_for_various_feature_counts(self, n_feat):
        model, X_te = _make_rf_and_data(n_test=20, n_feat=n_feat, seed=4)
        sv, _ = compute_shap_values(model, X_te)
        assert sv.shape == (20, n_feat)


# ── Group 2: additivity ───────────────────────────────────────────────────────

class TestSHAPAdditivity:
    def test_row_sum_plus_base_equals_prediction(self):
        """
        For tree models, SHAP is exact:
            shap_values.sum(axis=1) + base_value  ==  model.predict(X_test)
        to within floating-point precision.
        """
        model, X_te = _make_rf_and_data(n_test=40, n_feat=6, seed=10)
        sv, base    = compute_shap_values(model, X_te)
        pred        = model.predict(X_te)
        reconstructed = sv.sum(axis=1) + base
        np.testing.assert_allclose(
            reconstructed, pred,
            atol=1e-4,
            err_msg="SHAP row sums + base do not match model predictions",
        )

    def test_additivity_holds_for_single_row(self):
        """Single-row check for additivity."""
        model, X_te = _make_rf_and_data(n_test=10, n_feat=5, seed=11)
        sv, base    = compute_shap_values(model, X_te)
        pred        = model.predict(X_te)
        # Check first row only
        assert abs(sv[0].sum() + base - pred[0]) < 1e-4

    def test_additivity_with_larger_forest(self):
        """Additivity must hold even with n_estimators=200."""
        rng   = np.random.default_rng(20)
        X_tr  = rng.standard_normal((200, 8))
        y_tr  = rng.standard_normal(200)
        X_te  = rng.standard_normal((30,  8))
        model = RandomForestRegressor(n_estimators=200, random_state=20)
        model.fit(X_tr, y_tr)

        sv, base = compute_shap_values(model, X_te)
        pred     = model.predict(X_te)
        np.testing.assert_allclose(sv.sum(axis=1) + base, pred, atol=1e-4)

    def test_base_value_near_training_mean(self):
        """
        base_value should be close to the mean prediction on training data,
        which for RF ≈ mean(y_train).
        """
        rng   = np.random.default_rng(30)
        X_tr  = rng.standard_normal((150, 5))
        y_tr  = rng.standard_normal(150) + 2.0   # mean ≈ 2
        X_te  = rng.standard_normal((20,  5))
        model = RandomForestRegressor(n_estimators=50, random_state=30)
        model.fit(X_tr, y_tr)

        _, base = compute_shap_values(model, X_te)
        # base_value is E[f(X)] ≈ mean(y_train) for RF on IID data
        assert abs(base - y_tr.mean()) < 0.5, (
            f"base_value {base:.3f} unexpectedly far from train mean {y_tr.mean():.3f}"
        )
