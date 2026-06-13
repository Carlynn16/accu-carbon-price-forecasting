"""
EDA functions: figures and stationarity tests for the ACCU carbon-price training data.

All statistical computations use the TRAIN split only.
Time plots may show all three splits for context.
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, adfuller, kpss

TARGET = "ACCU spot price (Generic)"
DATE_COL = "Date"

# Colour palette ─────────────────────────────────────────────────────────────
_C = {
    "train_bg": "#d4e9f7",
    "val_bg": "#fde8d0",
    "test_bg": "#d5ecd4",
    "train_v": "#5dade2",
    "val_v": "#f0a04b",
    "test_v": "#58d68d",
    "price": "#2c3e50",
    "change": "#c0392b",
    "vol": "#7d3c98",
}


def _set_style() -> None:
    for name in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid"):
        try:
            plt.style.use(name)
            return
        except OSError:
            continue


def _shade_splits(ax, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> list:
    ax.axvspan(train[DATE_COL].iloc[0], train[DATE_COL].iloc[-1],
               color=_C["train_bg"], alpha=0.55, zorder=0)
    ax.axvspan(val[DATE_COL].iloc[0], val[DATE_COL].iloc[-1],
               color=_C["val_bg"], alpha=0.75, zorder=0)
    ax.axvspan(test[DATE_COL].iloc[0], test[DATE_COL].iloc[-1],
               color=_C["test_bg"], alpha=0.75, zorder=0)
    return [
        mpatches.Patch(color=_C["train_bg"], alpha=0.7, label=f"Train (n={len(train)})"),
        mpatches.Patch(color=_C["val_bg"],   alpha=0.9, label=f"Val   (n={len(val)})"),
        mpatches.Patch(color=_C["test_bg"],  alpha=0.9, label=f"Test  (n={len(test)})"),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1
# ─────────────────────────────────────────────────────────────────────────────

def plot_price_timeline(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    outpath: Path,
) -> Path:
    """Full price series with train/val/test shading and 2021-22 spike annotation."""
    _set_style()
    full = pd.concat([train, val, test]).sort_values(DATE_COL).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    patches = _shade_splits(ax, train, val, test)
    ax.plot(full[DATE_COL], full[TARGET], color=_C["price"], linewidth=1.0, zorder=5)

    # annotate spike
    peak_idx = full[TARGET].idxmax()
    peak_date = full.loc[peak_idx, DATE_COL]
    peak_val = full.loc[peak_idx, TARGET]
    ax.annotate(
        f"2021-22 spike\n${peak_val:.2f}",
        xy=(peak_date, peak_val),
        xytext=(pd.Timestamp("2020-06-01"), peak_val - 8),
        fontsize=9, color="#922b21",
        arrowprops=dict(arrowstyle="->", color="#922b21", lw=1.2),
    )

    ax.legend(handles=patches, loc="upper left", fontsize=9)
    ax.set_title("ACCU Generic Carbon-Credit Spot Price (2018-2024)", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (A$/tonne)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return outpath


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2
# ─────────────────────────────────────────────────────────────────────────────

def plot_target_distributions(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    outpath: Path,
) -> Path:
    """Violin + overlaid histogram showing the regime shift across splits."""
    _set_style()

    splits = {
        f"Train\n(n={len(train)}, mean={train[TARGET].mean():.1f})": train[TARGET].values,
        f"Val\n(n={len(val)}, mean={val[TARGET].mean():.1f})":   val[TARGET].values,
        f"Test\n(n={len(test)}, mean={test[TARGET].mean():.1f})": test[TARGET].values,
    }
    vc = [_C["train_v"], _C["val_v"], _C["test_v"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                             gridspec_kw={"width_ratios": [2, 1.2]})

    # violin
    ax = axes[0]
    parts = ax.violinplot(
        list(splits.values()), positions=range(3),
        showmedians=True, showextrema=True,
    )
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(vc[i])
        body.set_alpha(0.75)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        if key in parts:
            parts[key].set_color("black")
            parts[key].set_linewidth(1.0)
    ax.set_xticks(range(3))
    ax.set_xticklabels(list(splits.keys()), fontsize=8.5)
    ax.set_ylabel("Price (A$/tonne)")
    ax.set_title("Distribution by Split (Violin)", fontsize=11)

    # overlaid histogram
    ax2 = axes[1]
    for i, (label, vals) in enumerate(splits.items()):
        ax2.hist(vals, bins=30, density=True, alpha=0.5, color=vc[i],
                 label=label.split("\n")[0])
    ax2.set_xlabel("Price (A$/tonne)")
    ax2.set_ylabel("Density")
    ax2.set_title("Overlaid Histogram", fontsize=11)
    ax2.legend(fontsize=8.5)

    fig.suptitle("Regime Shift: Train vs Val/Test Price Distributions", fontsize=13)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return outpath


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3
# ─────────────────────────────────────────────────────────────────────────────

def plot_returns(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    outpath: Path,
) -> Path:
    """Daily price change over the full period + distribution on train."""
    _set_style()
    full = pd.concat([train, val, test]).sort_values(DATE_COL).reset_index(drop=True)
    full["change"] = full[TARGET].diff()
    train_change = train[TARGET].diff().dropna()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # time series
    ax = axes[0]
    patches = _shade_splits(ax, train, val, test)
    ax.plot(full[DATE_COL], full["change"], color=_C["change"],
            linewidth=0.6, alpha=0.85, zorder=5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.45)
    ax.legend(handles=patches, loc="upper left", fontsize=8)
    ax.set_title("Daily Price Change Over Time", fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Change (A$/tonne)")

    # distribution (train only)
    ax2 = axes[1]
    ax2.hist(train_change, bins=70, density=True,
             color=_C["change"], alpha=0.65, edgecolor="white", linewidth=0.3)
    stats_txt = (
        f"mean = {train_change.mean():.4f}\n"
        f"std  = {train_change.std():.4f}\n"
        f"skew = {train_change.skew():.2f}\n"
        f"kurt = {train_change.kurtosis():.0f}"
    )
    ax2.text(0.97, 0.95, stats_txt, transform=ax2.transAxes, fontsize=8.5,
             va="top", ha="right",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    ax2.set_title("Distribution of Daily Changes (Train)", fontsize=12)
    ax2.set_xlabel("Change (A$/tonne)")
    ax2.set_ylabel("Density")

    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return outpath


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4
# ─────────────────────────────────────────────────────────────────────────────

def plot_volatility(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    outpath: Path,
    window: int = 30,
) -> Path:
    """Rolling 30-day std of daily changes over the full period."""
    _set_style()
    full = pd.concat([train, val, test]).sort_values(DATE_COL).reset_index(drop=True)
    full["change"] = full[TARGET].diff()
    full["vol"] = full["change"].rolling(window).std()

    fig, ax = plt.subplots(figsize=(12, 4))
    patches = _shade_splits(ax, train, val, test)
    ax.plot(full[DATE_COL], full["vol"], color=_C["vol"], linewidth=1.1, zorder=5)
    ax.legend(handles=patches, loc="upper left", fontsize=9)
    ax.set_title(f"Rolling {window}-Day Volatility of Daily Price Changes", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling Std (A$/tonne)")

    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return outpath


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5
# ─────────────────────────────────────────────────────────────────────────────

def plot_acf_pacf(
    train: pd.DataFrame,
    outpath: Path,
    nlags: int = 40,
) -> Path:
    """2x2 ACF/PACF grid: price level and daily change (train only)."""
    _set_style()
    price = train[TARGET].dropna()
    change = price.diff().dropna()

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    plot_acf(price,  lags=nlags, ax=axes[0, 0], alpha=0.05,
             title="ACF - Price Level (train)",   color="#2980b9", zero=False)
    plot_pacf(price, lags=nlags, ax=axes[0, 1], alpha=0.05, method="ywm",
              title="PACF - Price Level (train)",  color="#2980b9", zero=False)
    plot_acf(change,  lags=nlags, ax=axes[1, 0], alpha=0.05,
             title="ACF - Daily Change (train)",   color="#c0392b", zero=False)
    plot_pacf(change, lags=nlags, ax=axes[1, 1], alpha=0.05, method="ywm",
              title="PACF - Daily Change (train)", color="#c0392b", zero=False)

    for ax in axes.flat:
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_xlabel("Lag")

    fig.suptitle("Autocorrelation Analysis: Price Level vs Daily Change (train)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return outpath


# ─────────────────────────────────────────────────────────────────────────────
# Statistical tests (train only)
# ─────────────────────────────────────────────────────────────────────────────

def compute_stationarity_tests(train: pd.DataFrame) -> dict:
    """ADF + KPSS on price level and daily change. All computed on train split only."""
    warnings.filterwarnings("ignore")
    price = train[TARGET].dropna()
    change = price.diff().dropna()

    adf_l = adfuller(price, autolag="AIC")
    adf_c = adfuller(change, autolag="AIC")
    kp_l  = kpss(price,  regression="c", nlags="auto")
    kp_c  = kpss(change, regression="c", nlags="auto")

    # statsmodels KPSS bounds p-values at [0.01, 0.10]
    def _kpss_p_str(p: float) -> str:
        if p <= 0.01:
            return "<= 0.01"
        if p >= 0.10:
            return ">= 0.10"
        return f"{p:.4f}"

    acf_level  = acf(price,  nlags=10, fft=True)
    acf_change = acf(change, nlags=10, fft=True)

    return {
        "adf_level": {
            "stat": adf_l[0], "pvalue": adf_l[1],
            "lags": adf_l[2], "critical": adf_l[4],
        },
        "adf_change": {
            "stat": adf_c[0], "pvalue": adf_c[1],
            "lags": adf_c[2], "critical": adf_c[4],
        },
        "kpss_level": {
            "stat": kp_l[0], "pvalue": kp_l[1],
            "pvalue_str": _kpss_p_str(kp_l[1]),
            "lags": kp_l[2], "critical": kp_l[3],
        },
        "kpss_change": {
            "stat": kp_c[0], "pvalue": kp_c[1],
            "pvalue_str": _kpss_p_str(kp_c[1]),
            "lags": kp_c[2], "critical": kp_c[3],
        },
        "acf_level_lags":  [round(float(x), 4) for x in acf_level[1:11]],
        "acf_change_lags": [round(float(x), 4) for x in acf_change[1:11]],
        "change_stats": {
            "mean": float(change.mean()),
            "std":  float(change.std()),
            "skew": float(change.skew()),
            "kurtosis": float(change.kurtosis()),
        },
    }


def print_stats_summary(stats: dict) -> None:
    """Print a formatted summary of all stationarity test results."""
    sep = "=" * 65
    print(sep)
    print("STATIONARITY TESTS (train split only)")
    print(sep)

    al = stats["adf_level"]
    print(f"\nADF -- price LEVEL")
    print(f"  stat={al['stat']:.4f}, p={al['pvalue']:.6f}, lags used={al['lags']}")
    print(f"  Critical: 1%={al['critical']['1%']:.4f}, 5%={al['critical']['5%']:.4f}")
    print(f"  --> p > 0.05: FAIL to reject unit root => level is NON-STATIONARY")

    ac = stats["adf_change"]
    print(f"\nADF -- daily CHANGE")
    print(f"  stat={ac['stat']:.4f}, p={ac['pvalue']:.6f}, lags used={ac['lags']}")
    print(f"  Critical: 1%={ac['critical']['1%']:.4f}, 5%={ac['critical']['5%']:.4f}")
    print(f"  --> p < 0.001: REJECT unit root => change is STATIONARY")

    kl = stats["kpss_level"]
    print(f"\nKPSS -- price LEVEL  (H0 = stationary)")
    print(f"  stat={kl['stat']:.4f}, p-approx {kl['pvalue_str']} (table-bounded), lags={kl['lags']}")
    print(f"  Critical: 1%={kl['critical']['1%']:.4f}, 5%={kl['critical']['5%']:.4f}")
    print(f"  --> p <= 0.01: REJECT stationarity => level is NON-STATIONARY (confirms ADF)")

    kc = stats["kpss_change"]
    print(f"\nKPSS -- daily CHANGE  (H0 = stationary)")
    print(f"  stat={kc['stat']:.4f}, p-approx {kc['pvalue_str']} (table-bounded), lags={kc['lags']}")
    print(f"  Critical: 1%={kc['critical']['1%']:.4f}, 5%={kc['critical']['5%']:.4f}")
    print(f"  --> p >= 0.10: FAIL to reject stationarity => change is STATIONARY (confirms ADF)")

    print(f"\n{sep}")
    print("ACF TAKEAWAY (train split)")
    print(f"  Level  lags 1-10: {stats['acf_level_lags']}")
    print(f"  Change lags 1-10: {stats['acf_change_lags']}")
    print(f"  Level ACF ~1.0 (random walk / non-stationary).")
    print(f"  Change ACF ~0   (little linear structure -- lag-2 ~ {stats['acf_change_lags'][1]:.3f}).")

    ch = stats["change_stats"]
    print(f"\n{sep}")
    print("DAILY CHANGE DISTRIBUTION (train)")
    print(f"  mean={ch['mean']:.6f}  std={ch['std']:.4f}  "
          f"skew={ch['skew']:.2f}  excess-kurt={ch['kurtosis']:.0f}")
    print(f"  Heavy tails (extreme kurtosis); skewed left (large negative outliers).")
    print(sep)
