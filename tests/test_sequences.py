"""
Tests for src/sequences.py — Block C3.

Groups:
  1. Scaler fit train-only
  2. Shape and dtype correctness
  3. Leakage-free windowing
  4. End-to-end smoke on a small synthetic series
"""

import numpy as np
import pandas as pd
import pytest

from src.sequences import L_DEFAULT, build_sequences


# ── Fixture helpers ───────────────────────────────────────────────────────────

def _make_splits(n_tr: int = 80, n_va: int = 20, n_te: int = 20,
                 h: int = 1, seed: int = 1):
    """Synthetic 3-split DataFrames with correct column structure."""
    rng   = np.random.default_rng(seed)
    n     = n_tr + n_va + n_te
    price = np.cumsum(rng.standard_normal(n)) + 30.0
    chg_h = np.where(
        np.arange(n) + h < n,
        price[np.minimum(np.arange(n) + h, n - 1)] - price,
        np.nan,
    )
    df = pd.DataFrame({
        "Date":          pd.date_range("2020-01-01", periods=n, freq="B"),
        "price_anchor":  price,
        "target_1":      chg_h if h == 1  else np.full(n, np.nan),
        "target_7":      chg_h if h == 7  else np.full(n, np.nan),
        "target_30":     chg_h if h == 30 else np.full(n, np.nan),
        "feat_a":        rng.standard_normal(n),
        "feat_b":        rng.standard_normal(n),
        "feat_c":        rng.standard_normal(n),
    })
    tr = df.iloc[:n_tr].reset_index(drop=True)
    va = df.iloc[n_tr:n_tr + n_va].reset_index(drop=True)
    te = df.iloc[n_tr + n_va:].reset_index(drop=True)
    return tr, va, te, h


@pytest.fixture
def splits_h1():
    return _make_splits(h=1)


# ── Group 1: Scaler fit train-only ────────────────────────────────────────────

class TestScalerTrainOnly:
    def test_mean_from_train(self, splits_h1):
        tr, va, te, h = splits_h1
        seqs = build_sequences(tr, va, te, h)
        expected = tr[["feat_a", "feat_b", "feat_c"]].to_numpy().mean(axis=0)
        np.testing.assert_allclose(seqs["scaler"].mean_, expected, rtol=1e-5)

    def test_std_from_train(self, splits_h1):
        tr, va, te, h = splits_h1
        seqs = build_sequences(tr, va, te, h)
        expected = tr[["feat_a", "feat_b", "feat_c"]].to_numpy().std(axis=0, ddof=0)
        np.testing.assert_allclose(seqs["scaler"].scale_, expected, rtol=1e-5)

    def test_val_shift_does_not_change_scaler(self, splits_h1):
        """Shifting val features by 100 must not move the scaler's mean."""
        tr, va, te, h = splits_h1
        seqs1 = build_sequences(tr, va, te, h)
        va2 = va.copy(); va2["feat_a"] += 100.0
        seqs2 = build_sequences(tr, va2, te, h)
        np.testing.assert_array_equal(seqs1["scaler"].mean_, seqs2["scaler"].mean_)

    def test_test_shift_does_not_change_scaler(self, splits_h1):
        tr, va, te, h = splits_h1
        seqs1 = build_sequences(tr, va, te, h)
        te2 = te.copy(); te2["feat_b"] += 500.0
        seqs2 = build_sequences(tr, te2, te, h)
        np.testing.assert_array_equal(seqs1["scaler"].mean_, seqs2["scaler"].mean_)


# ── Group 2: Shape and dtype ──────────────────────────────────────────────────

class TestShapes:
    def test_3d_shape(self, splits_h1):
        tr, va, te, h = splits_h1
        L = 10
        seqs = build_sequences(tr, va, te, h, L=L)
        for sp in ("train", "val", "test"):
            X = seqs[sp]["X"]
            assert X.ndim == 3,    f"{sp} X must be 3-D"
            assert X.shape[1] == L, f"{sp} X dim-1 must be {L}"
            assert X.shape[2] == 3, f"{sp} X dim-2 must equal n_features"

    def test_y_and_anchor_lengths_match_X(self, splits_h1):
        tr, va, te, h = splits_h1
        seqs = build_sequences(tr, va, te, h)
        for sp in ("train", "val", "test"):
            N = len(seqs[sp]["X"])
            assert len(seqs[sp]["y"])            == N
            assert len(seqs[sp]["price_anchor"]) == N
            assert len(seqs[sp]["dates"])        == N

    def test_no_nan_in_y(self, splits_h1):
        tr, va, te, h = splits_h1
        seqs = build_sequences(tr, va, te, h)
        for sp in ("train", "val", "test"):
            assert not np.any(np.isnan(seqs[sp]["y"])), f"NaN y in {sp}"

    def test_X_dtype_float32(self, splits_h1):
        tr, va, te, h = splits_h1
        seqs = build_sequences(tr, va, te, h)
        assert seqs["train"]["X"].dtype == np.float32

    def test_warmup_rows_are_skipped(self, splits_h1):
        """Train sequences start only after L-1 warm-up rows are available."""
        tr, va, te, h = splits_h1
        L = 15
        seqs = build_sequences(tr, va, te, h, L=L)
        # Train has 80 rows; NaN targets at rows [79] (h=1 shifts last row NaN)
        # Valid train rows: indices L-1 to 78 → 80 - (L-1) - 1 = 80 - L windows
        max_train = len(tr) - (L - 1) - 1   # approx upper bound
        assert len(seqs["train"]["X"]) <= max_train + 1


# ── Group 3: Leakage-free windowing ──────────────────────────────────────────

class TestLeakageFree:
    def test_val_row_does_not_appear_in_train_windows(self, splits_h1):
        """
        Changing a val feature value must not alter any train window.
        Val rows are at positions n_tr..n_tr+n_va-1; train windows only
        cover positions 0..n_tr-1 so no val row can appear in them.
        """
        tr, va, te, h = splits_h1
        L = 5
        seqs1 = build_sequences(tr, va, te, h, L=L)

        va2 = va.copy()
        va2["feat_a"] += 999.0    # large shift on val
        seqs2 = build_sequences(tr, va2, te, h, L=L)

        # All train windows must be identical — val data is not in any of them
        np.testing.assert_array_equal(
            seqs1["train"]["X"], seqs2["train"]["X"],
            err_msg="A val row appeared inside a train window (leakage).",
        )

    def test_first_val_window_looks_back_into_train(self, splits_h1):
        """
        The first val window's leading rows come from the training tail.
        Verified by reconstructing the expected window from the scaled matrix.
        """
        tr, va, te, h = splits_h1
        L = 8
        seqs = build_sequences(tr, va, te, h, L=L)
        sc = seqs["scaler"]
        feat_cols = seqs["feature_cols"]

        # Reconstruct scaled matrix
        all_feat = pd.concat(
            [tr[feat_cols], va[feat_cols], te[feat_cols]], ignore_index=True
        ).to_numpy(dtype=float)
        full_scaled = sc.transform(all_feat).astype(np.float32)

        n_tr = len(tr)
        expected_first_val_window = full_scaled[n_tr - L + 1 : n_tr + 1]

        if len(seqs["val"]["X"]) > 0:
            np.testing.assert_allclose(
                seqs["val"]["X"][0], expected_first_val_window, rtol=1e-5
            )

    def test_window_contains_only_past_rows(self, splits_h1):
        """
        Window ending at row t must equal scaled_matrix[t-L+1:t+1].
        Spot-check three windows in the training set.
        """
        tr, va, te, h = splits_h1
        L = 6
        seqs = build_sequences(tr, va, te, h, L=L)
        sc = seqs["scaler"]
        feat_cols = seqs["feature_cols"]

        all_feat = pd.concat(
            [tr[feat_cols], va[feat_cols], te[feat_cols]], ignore_index=True
        ).to_numpy(dtype=float)
        full_scaled = sc.transform(all_feat).astype(np.float32)

        # Check that the 5th train window matches rows [L-1+4, ..., L-1+4+L]
        idx_check = 4
        t = (L - 1) + idx_check
        expected = full_scaled[t - L + 1 : t + 1]
        np.testing.assert_allclose(seqs["train"]["X"][idx_check], expected, rtol=1e-5)


# ── Group 4: End-to-end smoke ─────────────────────────────────────────────────

class TestSmoke:
    def test_runs_on_minimal_series(self):
        n = 55
        rng = np.random.default_rng(7)
        price = np.cumsum(rng.standard_normal(n)) + 30.0
        chg1  = np.concatenate([price[1:] - price[:-1], [np.nan]])
        df = pd.DataFrame({
            "Date":         pd.date_range("2022-01-01", periods=n, freq="B"),
            "price_anchor": price,
            "target_1":     chg1,
            "target_7":     np.full(n, np.nan),
            "target_30":    np.full(n, np.nan),
            "f1":           rng.standard_normal(n),
            "f2":           rng.standard_normal(n),
        })
        tr = df.iloc[:35].reset_index(drop=True)
        va = df.iloc[35:45].reset_index(drop=True)
        te = df.iloc[45:].reset_index(drop=True)
        seqs = build_sequences(tr, va, te, h=1, L=10)
        for sp in ("train", "val", "test"):
            assert seqs[sp]["X"].shape[1] == 10
            assert seqs[sp]["X"].shape[2] == 2

    def test_larger_L_reduces_sample_count(self):
        tr, va, te, h = _make_splits(n_tr=60, n_va=15, n_te=15, h=1, seed=3)
        seqs_small = build_sequences(tr, va, te, h, L=5)
        seqs_large = build_sequences(tr, va, te, h, L=20)
        assert len(seqs_large["train"]["X"]) < len(seqs_small["train"]["X"])
