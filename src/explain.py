"""
Block E: SHAP explainability for the best tree model (Random Forest, h=1).

Refits RF on TRAIN+VAL using the identical protocol as Block C2, then computes
exact TreeExplainer SHAP values on the TEST set.

Public API:
    compute_shap_values(model, X_test) -> (shap_values, base_value)
    run_explain(feat_train, feat_val, feat_test, figures_dir) -> dict
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor

from src.evaluate import prepare_horizon
from src.models import RF_GRID, _tscv_tune

SEED    = 42
HORIZON = 1
TOP_N   = 8


# ── Core SHAP computation ─────────────────────────────────────────────────────

def compute_shap_values(
    model:  RandomForestRegressor,
    X_test: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Run shap.TreeExplainer on an already-fitted sklearn tree model.

    Returns (shap_values, base_value) where:
      shap_values : (n_test, n_features) — per-observation SHAP attributions
      base_value  : scalar — model's expected output over training data

    Additivity guarantee (exact for tree models):
        shap_values.sum(axis=1) + base_value  ≈  model.predict(X_test)
    """
    explainer = shap.TreeExplainer(model)
    sv        = explainer.shap_values(X_test)
    # expected_value may be a 1-D array of length 1 in SHAP >= 0.46
    ev        = explainer.expected_value
    base_val  = float(np.ravel(ev)[0]) if np.ndim(ev) > 0 else float(ev)
    return np.asarray(sv), base_val


# ── RF refit (C2-identical protocol) ─────────────────────────────────────────

def _refit_rf_h1(
    feat_train: pd.DataFrame,
    feat_val:   pd.DataFrame,
    feat_test:  pd.DataFrame,
) -> tuple[RandomForestRegressor, np.ndarray, np.ndarray, list[str]]:
    """
    Tune RF on TRAIN (TimeSeriesSplit), refit on TRAIN+VAL, return test arrays.
    Returns (model, X_test, y_test, feature_names).
    """
    tr = prepare_horizon(feat_train, HORIZON)
    va = prepare_horizon(feat_val,   HORIZON)
    te = prepare_horizon(feat_test,  HORIZON)

    feat_cols = list(tr["X"].columns)   # extract names before to_numpy()

    X_tr    = tr["X"].to_numpy(dtype=float)
    y_tr    = tr["y"]
    X_va    = va["X"].to_numpy(dtype=float)
    y_va    = va["y"]
    X_te    = te["X"].to_numpy(dtype=float)

    X_trval = np.vstack([X_tr, X_va])
    y_trval = np.concatenate([y_tr, y_va])

    print("  SHAP/RF h=1: tuning on TRAIN ...")
    rf_base = RandomForestRegressor(random_state=SEED, n_jobs=-1)
    best_rf = _tscv_tune(rf_base, RF_GRID, X_tr, y_tr)

    print("  SHAP/RF h=1: refitting on TRAIN+VAL ...")
    best_rf.fit(X_trval, y_trval)

    return best_rf, X_te, te["y"], feat_cols


# ── Public entry point ────────────────────────────────────────────────────────

def run_explain(
    feat_train:  pd.DataFrame,
    feat_val:    pd.DataFrame,
    feat_test:   pd.DataFrame,
    figures_dir: Path,
) -> dict:
    """
    Full SHAP pipeline for RF at h=1.

    Generates:
      fig_shap_summary.png          — mean |SHAP| bar chart
      fig_shap_dependence_chg1.png  — dependence for chg_1, coloured by chg_0

    Returns dict:
      shap_values, base_value, X_test, predictions, feature_names, top8
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    model, X_te, y_te, feat_cols = _refit_rf_h1(feat_train, feat_val, feat_test)

    print("  SHAP: computing TreeExplainer values on TEST ...")
    shap_values, base_value = compute_shap_values(model, X_te)
    predictions = model.predict(X_te)

    # ── Top-8 by mean |SHAP| ──────────────────────────────────────────────────
    mean_abs = np.abs(shap_values).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1]
    top8 = pd.DataFrame({
        "feature":     [feat_cols[i] for i in order[:TOP_N]],
        "mean_|SHAP|": [round(float(mean_abs[i]), 5) for i in order[:TOP_N]],
    })
    print("\nTop-8 features by mean |SHAP| (RF, h=1, TEST):")
    print(top8.to_string(index=False))

    # ── Figures ───────────────────────────────────────────────────────────────
    _plot_summary(
        shap_values, feat_cols,
        figures_dir / "fig_shap_summary.png",
    )

    if "chg_1" in feat_cols:
        _plot_dependence(
            shap_values, X_te, feat_cols, "chg_1",
            figures_dir / "fig_shap_dependence_chg1.png",
        )

    return {
        "shap_values":   shap_values,
        "base_value":    base_value,
        "X_test":        X_te,
        "predictions":   predictions,
        "feature_names": feat_cols,
        "top8":          top8,
    }


# ── Figure helpers ────────────────────────────────────────────────────────────

def _plot_summary(
    shap_values: np.ndarray,
    feat_cols:   list[str],
    outpath:     Path,
    top_k:       int = 15,
) -> None:
    """Horizontal bar chart of mean |SHAP| for all features, sorted descending."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1]
    top_k    = min(top_k, len(feat_cols))
    idx      = order[:top_k][::-1]   # reverse so highest is at top

    labels = [feat_cols[i] for i in idx]
    values = [float(mean_abs[i]) for i in idx]

    # Colour: momentum/change = red; staleness = green; other = blue
    def _color(lbl):
        if lbl.startswith("chg"):
            return "#e74c3c"
        if lbl in {"price_moved", "days_since_last_move", "moves_7d", "moves_30d"}:
            return "#27ae60"
        return "#2980b9"

    colors = [_color(lbl) for lbl in labels]

    fig, ax = plt.subplots(figsize=(8, 0.45 * top_k + 1.8))
    ax.barh(range(top_k), values, color=colors, alpha=0.85, edgecolor="white")
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Mean |SHAP value|  (A$/tonne impact on predicted change)")
    ax.set_title(
        "Random Forest Feature Importance (SHAP) — h = 1, TEST set\n"
        "Red = momentum lags  |  Green = staleness  |  Blue = vol/exog/calendar",
        fontsize=10,
    )

    red_p   = mpatches.Patch(color="#e74c3c", alpha=0.85, label="Momentum lags (chg_*)")
    green_p = mpatches.Patch(color="#27ae60", alpha=0.85, label="Staleness features")
    blue_p  = mpatches.Patch(color="#2980b9", alpha=0.85, label="Vol / exog / calendar")
    ax.legend(handles=[red_p, green_p, blue_p], fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_dependence(
    shap_values: np.ndarray,
    X_te:        np.ndarray,
    feat_cols:   list[str],
    target_feat: str,
    outpath:     Path,
) -> None:
    """
    Scatter: (feature value, SHAP value) for target_feat.
    Coloured by chg_0 to reveal the relationship with the most-recent change.
    """
    fidx = feat_cols.index(target_feat)
    x    = X_te[:, fidx]
    sv   = shap_values[:, fidx]

    color_feat = "chg_0"
    if color_feat in feat_cols:
        cidx   = feat_cols.index(color_feat)
        c_vals = X_te[:, cidx]
    else:
        c_vals = np.zeros(len(x))
        color_feat = None

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(
        x, sv,
        c=c_vals, cmap="RdYlBu_r",
        alpha=0.55, s=18, edgecolors="none",
        vmin=np.percentile(c_vals, 5),
        vmax=np.percentile(c_vals, 95),
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("chg_0 (most-recent daily change, A$/tonne)", fontsize=8)

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_xlabel(f"{target_feat}  (lagged price change t−2, A$/tonne)", fontsize=9)
    ax.set_ylabel(f"SHAP value for {target_feat}", fontsize=9)
    ax.set_title(
        f"SHAP Dependence Plot — {target_feat}  (RF, h = 1, TEST)\n"
        "Coloured by chg_0 (most-recent change). Large prior moves drive model "
        "attention,\npartly via the forward-fill staleness artifact.",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
