"""
Diebold-Mariano (1995) significance test with Harvey-Leybourne-Newbold (1997)
small-sample correction for comparing model forecasts against the random walk.

Public API:
    diebold_mariano(actual, pred_model, pred_rw, h, loss, alpha) -> dict
    run_dm_tests(preds_df, ...) -> pd.DataFrame
    directional_accuracy_move_days(preds_df, h, split) -> pd.DataFrame
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats


def diebold_mariano(
    actual:     np.ndarray,
    pred_model: np.ndarray,
    pred_rw:    np.ndarray,
    h:          int,
    loss:       str = "squared",
    alpha:      float = 0.05,
) -> dict:
    """
    DM test: compare model vs random-walk predictive accuracy.

    Loss differential: d_t = L(e_model_t) - L(e_rw_t).
    Negative mean_d and DM_stat < 0 → model has lower loss → model is better.

    HAC variance: Newey-West Bartlett kernel, M = max(h-1, 0) lags.
    Small-sample correction: Harvey-Leybourne-Newbold (1997), t(T-1) distribution.

    Parameters
    ----------
    actual     : true h-step changes, shape (T,)
    pred_model : model's predicted changes, shape (T,)
    pred_rw    : RW baseline predictions (zeros for random walk), shape (T,)
    h          : forecast horizon — determines HAC lag M = h-1
    loss       : "squared" (default) or "absolute"
    alpha      : significance level (default 0.05)

    Returns
    -------
    dict: DM_stat, p_value, verdict, n_obs, mean_d, loss
    """
    actual     = np.asarray(actual,     dtype=float)
    pred_model = np.asarray(pred_model, dtype=float)
    pred_rw    = np.asarray(pred_rw,    dtype=float)

    e_m = actual - pred_model
    e_r = actual - pred_rw

    if loss == "squared":
        d = e_m ** 2 - e_r ** 2
    elif loss == "absolute":
        d = np.abs(e_m) - np.abs(e_r)
    else:
        raise ValueError(f"Unknown loss: {loss!r}. Choose 'squared' or 'absolute'.")

    T    = len(d)
    dbar = d.mean()

    # ── Newey-West HAC variance (Bartlett kernel, M = h-1 lags) ───────────────
    M   = max(h - 1, 0)
    dev = d - dbar
    g0  = float(np.dot(dev, dev)) / T              # γ̂(0)

    hac_sum = 0.0
    for j in range(1, M + 1):
        w       = 1.0 - j / (M + 1)               # Bartlett weight
        gamma_j = float(np.dot(dev[j:], dev[:-j])) / T
        hac_sum += w * gamma_j

    V = g0 + 2.0 * hac_sum
    V = max(V, 1e-12)                              # numerical floor

    # ── Diebold-Mariano statistic ─────────────────────────────────────────────
    DM = dbar / np.sqrt(V / T)

    # ── Harvey-Leybourne-Newbold small-sample correction ─────────────────────
    # c = sqrt((T + 1 - 2h + h(h-1)/T) / T)
    inner = (T + 1 - 2 * h + h * (h - 1) / T) / T
    c     = np.sqrt(max(inner, 0.0))
    DM_hln = float(DM * c)

    # ── Two-sided p-value under t(T-1) ───────────────────────────────────────
    p_val = float(2.0 * stats.t.sf(abs(DM_hln), df=T - 1))

    # ── Verdict ───────────────────────────────────────────────────────────────
    if p_val < alpha:
        verdict = "significantly better" if DM_hln < 0 else "significantly worse"
    else:
        verdict = "not significant"

    return {
        "DM_stat":  round(DM_hln, 3),
        "p_value":  round(p_val, 4),
        "verdict":  verdict,
        "n_obs":    T,
        "mean_d":   round(float(dbar), 6),
        "loss":     loss,
    }


def run_dm_tests(
    preds_df:  pd.DataFrame,
    horizons:  Sequence[int] = (1, 7, 30),
    splits:    Sequence[str] = ("val", "test"),
    loss:      str = "squared",
    alpha:     float = 0.05,
) -> pd.DataFrame:
    """
    Run DM tests for every (model × horizon × split) vs the random walk.

    preds_df required columns: model, horizon, split, date, pred_change, actual_change
    Returns tidy DataFrame sorted by (horizon, split, model).
    """
    rw_name    = "random_walk"
    all_models = [m for m in preds_df["model"].unique() if m != rw_name]
    records: list[dict] = []

    for h in horizons:
        for split in splits:
            rw = preds_df[
                (preds_df["model"] == rw_name) &
                (preds_df["horizon"] == h) &
                (preds_df["split"] == split)
            ].sort_values("date")

            if rw.empty:
                continue

            for model in all_models:
                m_rows = preds_df[
                    (preds_df["model"] == model) &
                    (preds_df["horizon"] == h) &
                    (preds_df["split"] == split)
                ].sort_values("date")

                if m_rows.empty:
                    continue

                merged = rw[["date", "actual_change"]].merge(
                    m_rows[["date", "pred_change"]], on="date", how="inner"
                )
                if len(merged) < 5:
                    continue

                actual  = merged["actual_change"].to_numpy(dtype=float)
                pred_m  = merged["pred_change"].to_numpy(dtype=float)
                pred_rw = np.zeros_like(pred_m)

                dm = diebold_mariano(actual, pred_m, pred_rw, h, loss=loss, alpha=alpha)
                records.append({"model": model, "horizon": h, "split": split, **dm})

    if not records:
        return pd.DataFrame()

    out = pd.DataFrame(records)
    col_order = ["model", "horizon", "split", "n_obs", "DM_stat", "p_value", "verdict", "mean_d"]
    present = [c for c in col_order if c in out.columns]
    return (
        out[present]
        .sort_values(["horizon", "split", "model"])
        .reset_index(drop=True)
    )


def directional_accuracy_move_days(
    preds_df: pd.DataFrame,
    h:        int = 1,
    split:    str = "test",
) -> pd.DataFrame:
    """
    Directional accuracy restricted to genuine-move days (|actual_change| > 0).

    Returns DataFrame: model | n_move | dir_acc_move_%
    """
    sub  = preds_df[(preds_df["horizon"] == h) & (preds_df["split"] == split)].copy()
    move = sub[sub["actual_change"].abs() > 1e-9]

    records: list[dict] = []
    for model, grp in move.groupby("model"):
        actual  = grp["actual_change"].to_numpy()
        pred    = grp["pred_change"].to_numpy()
        n       = len(actual)
        if n == 0:
            continue
        correct = int(np.sum(np.sign(actual) == np.sign(pred)))
        records.append({
            "model":          model,
            "n_move":         n,
            "dir_acc_move_%": round(100.0 * correct / n, 1),
        })

    if not records:
        return pd.DataFrame(columns=["model", "n_move", "dir_acc_move_%"])

    return (
        pd.DataFrame(records)
        .sort_values("model")
        .reset_index(drop=True)
    )
