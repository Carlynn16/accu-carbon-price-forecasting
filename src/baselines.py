"""
Baseline forecasters for the ACCU carbon-price study.

All baselines predict the h-step CHANGE (not the level).
The caller reconstructs the level as price_anchor + predicted_change.

Baselines implemented:
  random_walk — predicted change = 0 for every row and horizon.
                This is THE primary benchmark. A model must beat it to claim skill.
  drift       — predicted change = h × trailing 30-day mean daily change.
                A slightly less naive reference that allows for a persistent trend.

Seasonal-naive is omitted: ACCU spot prices show no stable calendar seasonality
(the market is policy-driven and the staleness structure dominates any weekly or
monthly pattern). This is noted in the report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def random_walk_predict(n: int) -> np.ndarray:
    """Return an array of n zeros — the random-walk forecast for any horizon."""
    return np.zeros(n, dtype=float)


def drift_predict(
    feat_dates:        np.ndarray | pd.Series,
    full_price_series: pd.Series,
    h:                 int,
    window:            int = 30,
) -> np.ndarray:
    """
    Drift forecast: predicted_change(t, h) = h × trailing mean daily change.

    Parameters
    ----------
    feat_dates : dates for which to produce predictions (from prepare_horizon).
    full_price_series : pd.Series with DatetimeIndex covering the full
        chronological range (train + val + test). Used to compute the trailing
        rolling mean; the window always looks backwards (no look-ahead).
    h : forecast horizon in days.
    window : number of past days for the trailing mean (default 30).

    Returns
    -------
    1-D float array aligned with feat_dates.
    Rows whose date is not in full_price_series receive prediction 0.0
    (equivalent to random walk) — should not occur in normal usage.
    """
    daily_chg  = full_price_series.diff()
    rolling_mu = daily_chg.rolling(window, min_periods=1).mean()

    dates_s = pd.Series(pd.to_datetime(feat_dates))
    preds   = dates_s.map(rolling_mu) * h
    return preds.to_numpy(dtype=float, na_value=0.0)
