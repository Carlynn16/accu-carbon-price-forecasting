"""
Smoke tests for src/dl_models.py — Block C3.

Groups:
  1. Architecture: LSTM and GRU produce correct output shapes
  2. Training: _fit_phase1 converges without error; returns finite predictions
  3. Protocol: Phase 3 refit on larger dataset changes weights
  4. Determinism: same seed → same predictions
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src.dl_models import (
    BATCH_SIZE,
    GRURegressor,
    LSTMRegressor,
    PATIENCE,
    _fit_phase1,
    _fit_phase3,
    _predict,
    _to_tensors,
)
from src.sequences import build_sequences


# ── Fixture: tiny synthetic split ─────────────────────────────────────────────

def _tiny_splits(n_tr: int = 70, n_va: int = 20, n_te: int = 20,
                 h: int = 1, n_feat: int = 4, L: int = 10):
    """Return (seqs, n_features) for a small synthetic series."""
    rng = np.random.default_rng(99)
    n   = n_tr + n_va + n_te
    price = np.cumsum(rng.standard_normal(n)) + 30.0
    chg   = np.concatenate([price[1:] - price[:-1], [np.nan]])
    feat_dict = {f"f{i}": rng.standard_normal(n) for i in range(n_feat)}
    df = pd.DataFrame({
        "Date":         pd.date_range("2021-01-01", periods=n, freq="B"),
        "price_anchor": price,
        "target_1":     chg if h == 1  else np.full(n, np.nan),
        "target_7":     chg if h == 7  else np.full(n, np.nan),
        "target_30":    chg if h == 30 else np.full(n, np.nan),
        **feat_dict,
    })
    tr = df.iloc[:n_tr].reset_index(drop=True)
    va = df.iloc[n_tr:n_tr + n_va].reset_index(drop=True)
    te = df.iloc[n_tr + n_va:].reset_index(drop=True)
    seqs = build_sequences(tr, va, te, h=h, L=L)
    return seqs, n_feat


# ── Group 1: Architecture output shapes ──────────────────────────────────────

class TestArchitectures:
    @pytest.mark.parametrize("cls", [LSTMRegressor, GRURegressor])
    def test_output_shape(self, cls):
        batch, L, F = 8, 10, 5
        x = torch.randn(batch, L, F)
        m = cls(n_features=F)
        out = m(x)
        assert out.shape == (batch, 1), f"{cls.__name__} output shape wrong: {out.shape}"

    @pytest.mark.parametrize("cls", [LSTMRegressor, GRURegressor])
    def test_output_finite(self, cls):
        x = torch.randn(4, 10, 3)
        m = cls(n_features=3)
        out = m(x)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("cls", [LSTMRegressor, GRURegressor])
    def test_different_batch_sizes(self, cls):
        m = cls(n_features=6)
        for bs in [1, 4, 32]:
            x = torch.randn(bs, 15, 6)
            assert m(x).shape == (bs, 1)


# ── Group 2: Training convergence ────────────────────────────────────────────

class TestTraining:
    @pytest.mark.parametrize("cls", [LSTMRegressor, GRURegressor])
    def test_phase1_returns_finite_preds(self, cls):
        seqs, n_f = _tiny_splits()
        X_tr, y_tr = _to_tensors(seqs["train"])
        X_va, y_va = _to_tensors(seqs["val"])
        device = torch.device("cpu")
        model, epoch = _fit_phase1(cls, n_f, X_tr, y_tr, X_va, y_va, device)
        X_te, _ = _to_tensors(seqs["test"])
        preds = _predict(model, X_te, device)
        assert np.isfinite(preds).all(), "Non-finite predictions after phase 1"
        assert preds.shape == (len(seqs["test"]["X"]),)

    @pytest.mark.parametrize("cls", [LSTMRegressor, GRURegressor])
    def test_phase1_returns_positive_epoch(self, cls):
        seqs, n_f = _tiny_splits()
        X_tr, y_tr = _to_tensors(seqs["train"])
        X_va, y_va = _to_tensors(seqs["val"])
        _, epoch = _fit_phase1(cls, n_f, X_tr, y_tr, X_va, y_va, torch.device("cpu"))
        assert epoch >= 1

    @pytest.mark.parametrize("cls", [LSTMRegressor, GRURegressor])
    def test_phase3_produces_finite_preds(self, cls):
        seqs, n_f = _tiny_splits()
        X_tr, y_tr = _to_tensors(seqs["train"])
        X_va, y_va = _to_tensors(seqs["val"])
        X_trval = torch.cat([X_tr, X_va], dim=0)
        y_trval = torch.cat([y_tr, y_va], dim=0)
        model = _fit_phase3(cls, n_f, X_trval, y_trval, n_epochs=3, device=torch.device("cpu"))
        X_te, _ = _to_tensors(seqs["test"])
        preds = _predict(model, X_te, torch.device("cpu"))
        assert np.isfinite(preds).all()


# ── Group 3: Phase 3 uses more data ──────────────────────────────────────────

class TestProtocol:
    @pytest.mark.parametrize("cls", [LSTMRegressor, GRURegressor])
    def test_phase3_on_larger_data_changes_weights(self, cls):
        """
        A model trained on TRAIN+VAL (phase 3) should differ from one trained
        on TRAIN only (phase 1) — larger dataset should shift parameters.
        """
        seqs, n_f = _tiny_splits()
        X_tr, y_tr = _to_tensors(seqs["train"])
        X_va, y_va = _to_tensors(seqs["val"])
        device = torch.device("cpu")

        model_p1, _ = _fit_phase1(cls, n_f, X_tr, y_tr, X_va, y_va, device)
        w1 = list(model_p1.parameters())[0].data.clone()

        X_trval = torch.cat([X_tr, X_va], dim=0)
        y_trval = torch.cat([y_tr, y_va], dim=0)
        model_p3 = _fit_phase3(cls, n_f, X_trval, y_trval, n_epochs=5, device=device)
        w3 = list(model_p3.parameters())[0].data.clone()

        # Weights must differ (training on more data with same seed)
        assert not torch.allclose(w1, w3), "Phase 1 and phase 3 weights are identical"


# ── Group 4: Determinism ─────────────────────────────────────────────────────

class TestDeterminism:
    @pytest.mark.parametrize("cls", [LSTMRegressor, GRURegressor])
    def test_same_seed_same_predictions(self, cls):
        """Two runs with the same seed must produce identical predictions."""
        seqs, n_f = _tiny_splits()
        X_tr, y_tr = _to_tensors(seqs["train"])
        X_va, y_va = _to_tensors(seqs["val"])
        X_te, _    = _to_tensors(seqs["test"])
        device = torch.device("cpu")

        m1, _ = _fit_phase1(cls, n_f, X_tr, y_tr, X_va, y_va, device)
        p1 = _predict(m1, X_te, device)

        m2, _ = _fit_phase1(cls, n_f, X_tr, y_tr, X_va, y_va, device)
        p2 = _predict(m2, X_te, device)

        np.testing.assert_array_equal(p1, p2)
