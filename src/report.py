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

    # ── 3.2 Autocorrelation ─────────────────────────────────────────────────
    _h(doc, "3.2  Autocorrelation Structure", level=2)

    lag1_level  = stats["acf_level_lags"][0]
    lag2_change = stats["acf_change_lags"][1]

    _p(doc,
       f"Figure 5 shows the ACF and PACF for both the price level and the daily first "
       f"difference (40 lags, training set only). For the price level, the ACF coefficient "
       f"at lag 1 is {lag1_level:.4f} and remains near 1.0 across all displayed lags, "
       f"consistent with near-random-walk dynamics. The PACF drops sharply after lag 1, "
       f"indicating an AR(1)-like structure in levels — largely a manifestation of "
       f"non-stationarity rather than genuine mean-reversion."
       )

    _p(doc,
       f"For the daily change series, the ACF is near zero at almost all lags, confirming "
       f"that price changes carry very little linear predictive structure. The most notable "
       f"exception is lag 2 (ACF ≈ {lag2_change:.3f}), which may reflect "
       f"thin-trading or settlement-cycle effects, but it is not sustained enough to support "
       f"a robust linear forecasting model. This is the core empirical challenge: to "
       f"outperform the random walk, a model must identify non-linear or exogenous structure "
       f"that linear autocorrelation analysis does not capture."
       )

    # ── 3.3 Returns and Volatility ───────────────────────────────────────────
    _h(doc, "3.3  Returns Distribution and Volatility Clustering", level=2)

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
    _figure(
        doc,
        figures_dir / "fig_acf_pacf.png",
        "Figure 5.  ACF and PACF for the price level (top row) and daily change "
        "(bottom row), computed on the training set only (40 lags, 5% confidence bands). "
        "Level ACF ≈ 1.0 across all lags (non-stationary); "
        "change ACF ≈ 0 at most lags (near-absence of linear structure).",
        width=6.0,
    )
