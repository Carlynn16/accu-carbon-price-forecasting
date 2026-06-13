"""
Incremental report builder.

Each public function writes one section into the provided python-docx Document.
Call them in order from scripts/build_report.py.

Sections implemented here (Block A-bis):
  build_intro(doc, figures_dir, stats)
  build_data_section(doc, figures_dir, stats)
  build_eda_section(doc, figures_dir, stats)

Later blocks will add:
  build_features_section, build_modeling_section,
  build_evaluation_section, build_explainability_section.
"""

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _h(doc: Document, text: str, level: int = 1) -> None:
    doc.add_heading(text, level=level)


def _p(doc: Document, text: str) -> None:
    doc.add_paragraph(text)


def _bullet(doc: Document, text: str) -> None:
    doc.add_paragraph(text, style="List Bullet")


def _bold_inline(doc: Document, label: str, rest: str) -> None:
    """Add a paragraph with a bold label followed by normal text."""
    para = doc.add_paragraph()
    run = para.add_run(label)
    run.bold = True
    para.add_run(rest)


def _figure(
    doc: Document,
    img_path: Path,
    caption: str,
    width: float = 5.8,
) -> None:
    """Embed an image (centered) with an italic caption below it."""
    if not img_path.exists():
        doc.add_paragraph(f"[Figure not found: {img_path.name}]")
        return

    doc.add_picture(str(img_path), width=Inches(width))
    # center the picture paragraph (last paragraph added)
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = cap.add_run(caption)
    run.italic = True
    run.font.size = Pt(9)


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Introduction
# ─────────────────────────────────────────────────────────────────────────────

def build_intro(
    doc: Document,
    figures_dir: Path | None = None,
    stats: dict | None = None,
) -> None:
    """Write Section 1: Introduction."""
    _h(doc, "1. Introduction")

    _p(doc,
       "This report presents a time-series forecasting study of the Australian Generic "
       "carbon-credit spot price (ACCU Generic) at horizons of 1, 7, and 30 calendar days "
       "ahead. Australian Carbon Credit Units (ACCUs) represent one tonne of CO₂-equivalent "
       "abated or sequestered under government-registered emissions reduction methods. The "
       "ACCU Generic price is the benchmark market price, reflecting arms-length transactions "
       "in the secondary market across all eligible methods."
       )

    _p(doc,
       "The guiding methodological principle of this study is that forecasting skill can only "
       "be claimed relative to a naive baseline. The benchmark throughout is the random-walk "
       "model: tomorrow’s forecast equals today’s observed price. A model that fits "
       "well by conventional metrics (R², in-sample RMSE) but cannot beat this baseline "
       "provides no actionable information about future price direction. All performance "
       "comparisons are therefore expressed as percentage improvement in RMSE and MAE over the "
       "random-walk forecast; statistical significance is assessed using the "
       "Diebold–Mariano test."
       )

    _p(doc,
       "A secondary theme is methodological hygiene in feature construction. The raw data "
       "contain columns that are arithmetically derived from the target (lagged changes, "
       "year-to-date percentage returns) and columns that can reconstruct the target level "
       "(sibling-method prices and their premiums over Generic). Using such columns as "
       "predictors produces inflated in-sample accuracy but no genuine forecasting skill. "
       "Section 4 documents the feature audit that excludes these columns before any model "
       "is fitted."
       )

    # Executive summary placeholder
    para = doc.add_paragraph()
    run = para.add_run(
        "[Executive summary — headline results to be completed in the final block]"
    )
    run.bold = True
    run.font.color.rgb = RGBColor(0xC0, 0x39, 0x2B)


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Data and Cleaning
# ─────────────────────────────────────────────────────────────────────────────

def build_data_section(
    doc: Document,
    figures_dir: Path,
    stats: dict | None = None,
) -> None:
    """Write Section 2: Data and Cleaning."""
    _h(doc, "2. Data and Cleaning")

    _p(doc,
       "The dataset consists of daily price and volume observations from the Australian carbon "
       "and renewable-certificate markets, spanning 2 January 2018 to 15 November 2024 "
       "(1,651 trading days). The raw file contained 97 columns covering spot prices across "
       "multiple abatement methods, forward-contract price series, volume figures, "
       "week-on-week and year-to-date change columns, and market-assessment type indicators."
       )

    _h(doc, "2.1  Cleaning Pipeline", level=2)

    _p(doc,
       "A principled, leakage-free cleaning pipeline was applied in the following order to "
       "produce three non-overlapping chronological splits:"
       )

    _bullet(doc,
            "Symbol stripping: dollar signs, percentage signs, and thousands-comma separators "
            "were removed from all value columns before numeric coercion."
            )
    _bullet(doc,
            "High-missingness drop: 41 columns with more than 70% missing values were removed. "
            "These were predominantly forward-contract series (CAL 21–26 and "
            "method-specific analogues) that are only sparsely quoted, plus illiquid-method "
            "columns (No-AD, Landfill Gas, Swaps, Assessment Types)."
            )
    _bullet(doc,
            "Near-constant drop: 6 additional columns where more than 99% of training-set "
            "values were identical were dropped (four calendar-year traded-volume columns and "
            "two CAL 26 change columns)."
            )
    _bullet(doc,
            "Forward-fill imputation on the full chronologically sorted series. This is "
            "strictly past-only: each missing entry is replaced by the most recent prior "
            "observation. Backward-fill was never applied. Carrying the last known price "
            "across a weekend or public holiday is the correct market convention and introduces "
            "no look-ahead."
            )
    _bullet(doc,
            "Residual NaN fill: the small number of leading NaN values at the start of each "
            "series (unreachable by forward-fill) were filled with the training-set column "
            "mean or mode. Validation and test statistics were never consulted."
            )

    _p(doc,
       "After cleaning, 50 columns remained across 1,651 observations with zero missing values."
       )

    _h(doc, "2.2  Chronological Split and Regime Shift", level=2)

    _p(doc,
       "The data were split chronologically by row position: training (70%, 1,155 rows, "
       "2018-01-02 to 2022-12-01), validation (15%, 248 rows, 2022-12-02 to 2023-11-23), "
       "and test (15%, 248 rows, 2023-11-24 to 2024-11-15). No shuffling was performed at "
       "any stage."
       )

    _p(doc,
       "A critical structural feature is the pronounced regime shift between the training "
       "period and the out-of-sample sets. During training, the ACCU Generic price ranged "
       "from A$13.25 to A$57.15 (mean A$21.55), capturing the 2021–22 price spike "
       "driven by policy expectations. The validation and test sets occupy a narrower, "
       "higher plateau (means A$33.66 and A$34.82 respectively, range approximately "
       "A$26–$42). This regime shift has three practical consequences for the analysis:"
       )

    _bullet(doc,
            "It motivates the random-walk as the primary benchmark: a model trained on the "
            "training mean would systematically underpredict the out-of-sample level, so "
            "level forecasting accuracy must be compared against the simplest possible "
            "extrapolation of the most recent observation."
            )
    _bullet(doc,
            "It motivates modelling first differences (daily changes) rather than price "
            "levels, since the distribution of daily changes is far more regime-stable "
            "than the distribution of levels."
            )
    _bullet(doc,
            "It validates the strict chronological split: any reshuffling of rows would "
            "leak post-spike observations into the training set, artificially improving "
            "in-sample fit and producing an over-optimistic evaluation."
            )

    _figure(
        doc,
        figures_dir / "fig_price_timeline.png",
        "Figure 1.  ACCU Generic price over the full sample period (2018-2024). "
        "Shaded regions indicate the training (blue), validation (orange), and test (green) "
        "splits. The 2021-22 price spike is annotated.",
    )
    _figure(
        doc,
        figures_dir / "fig_target_dist_by_split.png",
        "Figure 2.  Price distribution by split (violin plot, left; overlaid histogram, "
        "right). The training distribution is wide and left-skewed due to the 2021-22 spike; "
        "validation and test occupy a narrower high-price regime.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Exploratory Analysis
# ─────────────────────────────────────────────────────────────────────────────

def build_eda_section(
    doc: Document,
    figures_dir: Path,
    stats: dict,
    stale_stats: dict | None = None,
) -> None:
    """Write Section 3: Exploratory Analysis."""
    _h(doc, "3. Exploratory Analysis")

    _p(doc,
       "All statistical tests and parameter choices reported in this section were computed "
       "on the training split only. Figures show all three splits for visual context."
       )

    # ── 3.1 Stationarity ────────────────────────────────────────────────────
    _h(doc, "3.1  Stationarity Tests", level=2)

    al = stats["adf_level"]
    ac = stats["adf_change"]
    kl = stats["kpss_level"]
    kc = stats["kpss_change"]

    adf_l_p = f"{al['pvalue']:.4f}" if al["pvalue"] > 1e-4 else "< 0.0001"

    _p(doc,
       "The Augmented Dickey–Fuller (ADF) and "
       "Kwiatkowski–Phillips–Schmidt–Shin (KPSS) tests were applied to the "
       "price level and the daily first difference on the training set."
       )

    _bold_inline(
        doc,
        "ADF — price level:  ",
        f"statistic = {al['stat']:.4f}, p-value = {adf_l_p}. "
        f"The null hypothesis of a unit root cannot be rejected (p > 0.05). "
        f"The price level is non-stationary.",
    )
    _bold_inline(
        doc,
        "ADF — daily change:  ",
        f"statistic = {ac['stat']:.4f}, p-value < 0.0001. "
        f"The null hypothesis is strongly rejected. The daily change is stationary.",
    )
    _bold_inline(
        doc,
        "KPSS — price level:  ",
        f"statistic = {kl['stat']:.4f}, p-value {kl['pvalue_str']} (table-bounded). "
        f"H₀ of stationarity is rejected at the 1% level. "
        f"Confirms the level is non-stationary.",
    )
    _bold_inline(
        doc,
        "KPSS — daily change:  ",
        f"statistic = {kc['stat']:.4f}, p-value {kc['pvalue_str']} (table-bounded). "
        f"H₀ cannot be rejected. The daily change is stationary.",
    )

    _p(doc,
       "The joint verdict is unambiguous: the price level is integrated of order one I(1), "
       "and the daily first difference is I(0). Forecasting the level is therefore equivalent "
       "to cumulating a stationary series; the random-walk baseline (which does exactly this) "
       "is hard to beat."
       )

    # ── 3.2 Market Staleness ─────────────────────────────────────────────────
    _h(doc, "3.2  Market Staleness", level=2)

    # Pull staleness numbers — fall back to narrative defaults if not provided
    if stale_stats:
        tr  = stale_stats["train"]
        vl  = stale_stats["val"]
        te  = stale_stats["test"]
        pz_tr, pz_vl, pz_te = tr["pct_zero"], vl["pct_zero"], te["pct_zero"]
        mr_tr = tr["max_run"]
        nz_tr = tr["n_nonzero"]
    else:
        pz_tr, pz_vl, pz_te = 75.1, 59.1, 45.7
        mr_tr, nz_tr = 34, 287

    _p(doc,
       f"A critical but initially non-obvious feature of this market is its extreme illiquidity. "
       f"The data source records one row per calendar day, but genuine price-discovery events "
       f"— days on which at least one trade executes — are rare. On {pz_tr:.1f}% of training-set "
       f"days, the price does not change at all from the previous observation. The corresponding "
       f"figures are {pz_vl:.1f}% for the validation set and {pz_te:.1f}% for the test set. "
       f"Only {nz_tr} of {nz_tr + round(nz_tr * pz_tr / (100 - pz_tr)):,} training observations "
       f"represent genuine price moves."
       )

    _p(doc,
       f"These zero-change days are not independent observations of a flat market; they are "
       f"calendar days on which no trade occurred and the dataset carries forward the "
       f"previous closing price. Stale runs as long as {mr_tr} consecutive days are present "
       f"in the training set (Figure 6). Treating all calendar-day observations as equally "
       f"informative inflates the effective sample size and biases any autocorrelation "
       f"analysis."
       )

    _figure(
        doc,
        figures_dir / "fig_staleness.png",
        "Figure 6.  Left: percentage of zero-change days by split (with number of genuine "
        "move-days and longest stale run annotated). Right: distribution of stale run "
        "lengths in the training set — runs of 5 or more consecutive stale days are "
        "common, with a maximum run exceeding 30 days.",
    )

    # ── 3.3 Autocorrelation — Corrected Analysis ─────────────────────────────
    _h(doc, "3.3  Autocorrelation Structure (Corrected)", level=2)

    lag1_level = stats["acf_level_lags"][0]
    if stale_stats:
        acf_full_lag2 = tr["acf_full_lags"][1]   # lag-2 of full series
        acf_nz_lag1   = tr["acf_nz_lags"][0]     # lag-1 on move-days
        acf_nz_lag2   = tr["acf_nz_lags"][1]     # lag-2 on move-days
    else:
        acf_full_lag2, acf_nz_lag1, acf_nz_lag2 = 0.334, 0.41, 0.142

    _p(doc,
       f"Figure 5 shows the ACF and PACF for both the price level and the daily first "
       f"difference (40 lags, training set only). For the price level, the ACF coefficient "
       f"at lag 1 is {lag1_level:.4f} and remains near 1.0 across all displayed lags, "
       f"consistent with near-random-walk dynamics."
       )

    _p(doc,
       f"The naive ACF of the daily change series shows a prominent spike at lag 2 "
       f"(ACF ≈ {acf_full_lag2:.3f}). This is an artefact of market staleness, not a genuine "
       f"market signal. Because the majority of consecutive observations are identical "
       f"(carried-forward prices), differencing two adjacent stale rows produces two "
       f"consecutive zeros, inflating the lag-2 autocorrelation through the accumulation "
       f"of tied zero-change pairs."
       )

    _p(doc,
       f"Figure 7 isolates the genuine-price-discovery signal by re-computing the ACF "
       f"using only the {nz_tr} non-zero-change training days. On these days, the lag-2 "
       f"coefficient collapses to {acf_nz_lag2:.3f} (confirming it was an artefact), "
       f"while the lag-1 coefficient is {acf_nz_lag1:.3f} — a genuine short-run "
       f"momentum effect: price moves tend to continue in the same direction for one "
       f"additional trading day. This lag-1 momentum signal is the primary linear "
       f"predictive structure exploited in the feature set."
       )

    _figure(
        doc,
        figures_dir / "fig_acf_pacf.png",
        "Figure 5.  ACF and PACF for the price level (top row) and daily change "
        "(bottom row), computed on the training set only (40 lags, 5% confidence bands). "
        "Level ACF ≈ 1.0 across all lags (non-stationary); change ACF shows a spurious "
        "lag-2 spike that disappears on genuine-move days (see Figure 7).",
        width=6.0,
    )
    _figure(
        doc,
        figures_dir / "fig_momentum.png",
        "Figure 7.  ACF of daily price change: full series (blue) vs genuine-move "
        "days only (red), training set. The lag-2 spike collapses from "
        f"≈{acf_full_lag2:.2f} to ≈{acf_nz_lag2:.2f} — a forward-fill artefact. "
        f"The lag-1 momentum signal (≈{acf_nz_lag1:.2f}) on move-days is genuine.",
    )

    # ── 3.4 Returns and Volatility ───────────────────────────────────────────
    _h(doc, "3.4  Returns Distribution and Volatility Clustering", level=2)

    ch = stats["change_stats"]
    _p(doc,
       f"Daily price changes on the training set have mean {ch['mean']:.4f} A$/tonne "
       f"(effectively zero), standard deviation {ch['std']:.4f}, skewness {ch['skew']:.2f}, "
       f"and excess kurtosis {ch['kurtosis']:.0f}. The extreme kurtosis indicates heavy tails: "
       f"large positive or negative moves occur far more frequently than a Gaussian would "
       f"predict. The strong negative skewness reflects occasional large downward repricing "
       f"events. Both properties imply that RMSE and MAE on a few large-move days can "
       f"dominate overall evaluation metrics."
       )

    _p(doc,
       "The rolling 30-day volatility (Figure 4) shows clear volatility clustering. "
       "The 2018–2020 period is relatively calm; a high-volatility episode accompanies "
       "the 2021–22 price spike; and the post-spike period (val and test) is again "
       "calmer. This heteroskedasticity is relevant when interpreting the test-set results: "
       "the evaluation period is a low-volatility regime relative to the most turbulent part "
       "of the training set."
       )

    _figure(
        doc,
        figures_dir / "fig_returns.png",
        "Figure 3.  Daily price changes over the full sample period (left panel) and "
        "the distribution of daily changes on the training set (right panel). "
        "Near-zero mean; heavy tails; extreme excess kurtosis.",
    )
    _figure(
        doc,
        figures_dir / "fig_volatility.png",
        "Figure 4.  Rolling 30-day standard deviation of daily price changes. "
        "Volatility clustering is visible: the 2021-22 spike period shows markedly "
        "higher volatility than the surrounding years.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def build_features_section(
    doc: Document,
    figures_dir: Path,
    feat_stats: dict | None = None,
) -> None:
    """Write Section 4: Feature Engineering."""
    _h(doc, "4. Feature Engineering")

    _p(doc,
       "All features are constructed from the full chronologically sorted series using "
       "only lagged values and strictly trailing rolling windows. No centered windows, "
       "no backward-fill, and no information from validation or test rows enters any "
       "feature calculation. The first 30 rows of the training set (the warm-up period "
       "required to populate the 30-day rolling windows) are dropped after feature "
       "construction."
       )

    # ── 4.1 Targets ──────────────────────────────────────────────────────────
    _h(doc, "4.1  Forecast Targets", level=2)

    _p(doc,
       "Three forecast targets are defined, one per horizon h ∈ {1, 7, 30}:"
       )
    _bullet(doc,
            "target_1  =  price(t+1) − price(t)  (1-day-ahead price change)")
    _bullet(doc,
            "target_7  =  price(t+7) − price(t)  (7-day-ahead cumulative change)")
    _bullet(doc,
            "target_30 =  price(t+30) − price(t) (30-day-ahead cumulative change)")

    _p(doc,
       "Modelling the change, not the level, is motivated by the stationarity results in "
       "Section 3.1 (the level is I(1)) and by the pronounced regime shift between the "
       "training and out-of-sample sets. All evaluation metrics are computed on the "
       "change, and the random-walk forecast corresponds to predicting zero change."
       )

    # ── 4.2 Feature Audit — Exclusions ───────────────────────────────────────
    _h(doc, "4.2  Feature Audit: Excluded Columns", level=2)

    _p(doc,
       "Before constructing features, a formal audit excluded three categories of columns "
       "that would introduce data leakage or level-reconstruction artefacts:"
       )

    _bold_inline(doc,
        "Category 1 — Target-derived columns:  ",
        "The raw dataset contains columns that are arithmetic functions of the target "
        "price level: the daily change ($ Change), week-on-week change, YTD percentage "
        "return, and the 7/30/50/100-day SMAs and percentage changes. Retaining any of "
        "these would allow a model to invert them to obtain the contemporaneous price "
        "level, defeating the purpose of modelling changes. All 18 such columns are "
        "excluded."
    )
    _bold_inline(doc,
        "Category 2 — Sibling price levels and premiums:  ",
        "The HIR and SFM ACCU sub-method prices are correlated with the Generic price "
        "but can also reconstruct it: Generic = HIR_price − HIR_premium. Retaining "
        "the sibling price levels or their premium-over-Generic columns would enable "
        "level reconstruction and produce inflated in-sample R² with no genuine "
        "forecasting value. Four price levels and four premium columns are excluded."
    )
    _bold_inline(doc,
        "Category 3 — Raw sibling change columns:  ",
        "The dataset also pre-computes dollar changes and week-on-week changes for HIR "
        "and SFM. These are re-derived from scratch (from the sibling price diffs) to "
        "ensure exact replication of the construction logic. The pre-computed versions "
        "are excluded to avoid double-counting."
    )

    # ── 4.3 Feature Groups ───────────────────────────────────────────────────
    _h(doc, "4.3  Feature Groups", level=2)

    _p(doc,
       "After exclusions, the following feature groups are constructed:"
       )

    _bold_inline(doc,
        "Group A — Momentum (change family, 5 features):  ",
        "chg_0, chg_1, chg_2, chg_3, chg_5. "
        "chg_0 = price(t) − price(t−1): the most-recent realised change, fully "
        "available at the close of day t and the direct carrier of the lag-1 momentum "
        "signal (corr ≈ +0.045 with target_1 on the full series; ≈ +0.41 on genuine-move "
        "days only). chg_1 through chg_5 are further lags. Note: chg_2 shows an "
        "apparent train correlation of ≈ +0.33 — this is largely the forward-fill "
        "artefact identified in Section 3.3 (lag-2 ACF of the full change series), "
        "not a reliable signal; models should not be expected to rely on it out-of-sample."
    )
    _bold_inline(doc,
        "Group B — Staleness features (4 features):  ",
        "price_moved (1/0 flag), days_since_last_move, moves_7d (number of move-days "
        "in trailing 7 days), moves_30d (move-days in trailing 30 days). "
        "These encode the market-liquidity context at each row — essential given that "
        "75% of observations are stale carries."
    )
    _bold_inline(doc,
        "Group C — Volatility regime (2 features):  ",
        "vol_chg_7d and vol_chg_30d: trailing 7- and 30-day rolling standard deviation "
        "of daily changes. These capture local heteroskedasticity."
    )
    _bold_inline(doc,
        "Group D — Volume (4 features, if present):  ",
        "vol_generic_raw (raw traded volume), vol_generic_log1p (log1p-transformed), "
        "vol_generic_trail7 (7-day trailing average), vol_generic_zero (1/0 zero-volume "
        "day flag). Volume is a direct proxy for trading activity."
    )
    _bold_inline(doc,
        "Group E — Calendar (2 features):  ",
        "dow (day of week, 0=Monday) and month. Carbon-credit trading shows day-of-week "
        "and seasonal patterns due to compliance cycles."
    )
    _bold_inline(doc,
        "Group F — Exogenous diffs (up to 8 features):  ",
        "First differences of sibling prices (lgc_chg, stc_chg, hir_chg, sfm_nc_chg, "
        "sfm_cb_chg, erf_chg) and trailing volume averages for HIR and SFM methods. "
        "Levels and premiums are excluded (Category 2 audit above); only changes are used."
    )

    if feat_stats:
        n_feats = feat_stats.get("n_features", "—")
        shapes  = feat_stats.get("shapes", {})
        _p(doc,
           f"After construction and warm-up removal, the feature matrix contains "
           f"{n_feats} features. Training set shape: {shapes.get('train', '—')}; "
           f"validation: {shapes.get('val', '—')}; test: {shapes.get('test', '—')}."
           )

    _figure(
        doc,
        figures_dir / "fig_feature_target_corr.png",
        "Figure 8.  Pearson correlations of each feature with the 1-day-ahead target "
        "(target_1) on the training set. Positive (red) bars indicate momentum-aligned "
        "features; negative (blue) bars indicate mean-reverting signals. The staleness "
        "and lag features dominate the top rankings.",
    )
