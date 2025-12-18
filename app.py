import math
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from core.fetch import fetch_ticker_data
from core.metrics import compute_metrics
from profiles.investors import ALL_PROFILES, InvestorProfile


# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="Superinvestor Lab",
    page_icon="🧠",
    layout="wide",
)


# ------------------------------------------------------------
# Data loading / caching
# ------------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_sp500_universe() -> pd.DataFrame:
    """
    Load S&P 500 universe from a local CSV file.

    File: sp500_universe.csv
    Required columns: Ticker,Company,Sector,Industry
    """
    try:
        df = pd.read_csv("sp500_universe.csv")
    except FileNotFoundError:
        raise FileNotFoundError(
            "sp500_universe.csv not found. "
            "Place an S&P 500 CSV in the repo with columns: "
            "Ticker,Company,Sector,Industry."
        )

    required = ["Ticker", "Company", "Sector", "Industry"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"sp500_universe.csv is missing columns: {missing}")

    df["Ticker"] = (
        df["Ticker"].astype(str).str.upper().str.strip().str.replace(".", "-", regex=False)
    )
    return df[required].copy()


@st.cache_data(show_spinner=False)
def get_ticker_data_cached(ticker: str) -> Dict[str, Any]:
    return fetch_ticker_data(ticker)


@st.cache_data(show_spinner=False)
def get_metrics_cached(ticker: str) -> Dict[str, Any]:
    raw = get_ticker_data_cached(ticker)
    return compute_metrics(ticker, raw)


def _fmt_ratio(x: Any, pct: bool = False) -> str:
    """Nice formatting for numbers / percentages."""
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "—"
        if pct:
            return f"{x * 100:,.1f}%"
        return f"{x:,.2f}"
    except Exception:
        return "—"


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
st.sidebar.title("Superinvestor Lab 🧠")
st.sidebar.caption("Analyze stocks through the eyes of famous investors.")

profile_labels = [p.label for p in ALL_PROFILES]
profile_key_map = {p.label: p for p in ALL_PROFILES}

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tab1, tab2 = st.tabs(["🔎 Single Stock Analysis", "📊 S&P 500 Screener"])


# ------------------------------------------------------------
# TAB 1 – Single Stock Analysis
# ------------------------------------------------------------
with tab1:
    st.header("🔎 Single Stock – Superinvestor View")

    col_left, col_right = st.columns([2, 2])

    with col_left:
        ticker_input = st.text_input(
            "Ticker",
            value="AAPL",
            help="Enter a stock ticker (e.g. AAPL, MSFT, BRK-B).",
        )

    with col_right:
        selected_profiles_labels = st.multiselect(
            "Investor Profiles",
            options=profile_labels,
            default=profile_labels,
            help="Choose which investor styles to apply.",
        )

    analyze_btn = st.button("Analyze Stock")

    if analyze_btn:
        ticker = ticker_input.upper().strip()
        if not ticker:
            st.error("Please enter a valid ticker.")
        else:
            with st.spinner(f"Fetching data for {ticker}..."):
                try:
                    metrics = get_metrics_cached(ticker)
                except Exception as e:
                    st.error(f"Failed to fetch data for {ticker}: {e}")
                    st.stop()

            meta = metrics["meta"]
            valuation = metrics["valuation"]
            quality = metrics["quality"]
            growth = metrics["growth"]
            bs = metrics["balance_sheet"]

            # --- Top summary ---
            st.subheader(f"{meta.get('short_name') or ticker} ({ticker})")
            info_cols = st.columns(4)
            info_cols[0].metric("Price", _fmt_ratio(meta.get("price"), pct=False))
            mc = meta.get("market_cap")
            info_cols[1].metric(
                "Market Cap",
                f"{mc/1e9:,.2f}B" if mc else "—",
            )
            info_cols[2].metric("Sector", meta.get("sector") or "—")
            info_cols[3].metric("Industry", meta.get("industry") or "—")

            st.markdown("---")

            # --- Core metrics: valuation ---
            with st.expander("Core Metrics – Valuation", expanded=True):
                val_df = pd.DataFrame(
                    {
                        "Metric": [
                            "P/E",
                            "P/B",
                            "EV/EBITDA",
                            "EV/Sales",
                            "Earnings Yield",
                            "FCF Yield",
                            "PEG (P/E ÷ growth%)",
                        ],
                        "Value": [
                            _fmt_ratio(valuation["pe"]),
                            _fmt_ratio(valuation["pb"]),
                            _fmt_ratio(valuation["ev_ebitda"]),
                            _fmt_ratio(valuation["ev_sales"]),
                            _fmt_ratio(valuation["earnings_yield"], pct=True),
                            _fmt_ratio(valuation["fcf_yield"], pct=True),
                            _fmt_ratio(valuation["peg"], pct=False),
                        ],
                    }
                )
                st.table(val_df)

            # --- Core metrics: quality, growth, balance sheet ---
            with st.expander("Core Metrics – Quality, Growth & Balance Sheet", expanded=False):
                col_q, col_g, col_b = st.columns(3)

                with col_q:
                    st.markdown("**Quality**")
                    q_df = pd.DataFrame(
                        {
                            "Metric": [
                                "ROE",
                                "ROA",
                                "Gross Margin",
                                "Operating Margin",
                                "Net Margin",
                                "FCF / Net Income",
                            ],
                            "Value": [
                                _fmt_ratio(quality["roe"], pct=True),
                                _fmt_ratio(quality["roa"], pct=True),
                                _fmt_ratio(quality["gross_margin"], pct=True),
                                _fmt_ratio(quality["op_margin"], pct=True),
                                _fmt_ratio(quality["net_margin"], pct=True),
                                _fmt_ratio(quality["fcf_conversion"], pct=True),
                            ],
                        }
                    )
                    st.table(q_df)

                with col_g:
                    st.markdown("**Growth (approx)**")
                    g_df = pd.DataFrame(
                        {
                            "Metric": ["Revenue Growth", "Earnings Growth"],
                            "Value": [
                                _fmt_ratio(growth["revenue_growth"], pct=True),
                                _fmt_ratio(growth["earnings_growth"], pct=True),
                            ],
                        }
                    )
                    st.table(g_df)

                with col_b:
                    st.markdown("**Balance Sheet**")
                    b_df = pd.DataFrame(
                        {
                            "Metric": [
                                "Debt/Equity",
                                "Current Ratio",
                                "Quick Ratio",
                                "Interest Coverage",
                            ],
                            "Value": [
                                _fmt_ratio(bs["debt_to_equity"], pct=False),
                                _fmt_ratio(bs["current_ratio"], pct=False),
                                _fmt_ratio(bs["quick_ratio"], pct=False),
                                _fmt_ratio(bs["interest_coverage"], pct=False),
                            ],
                        }
                    )
                    st.table(b_df)

            st.markdown("---")

            # --- Superinvestor views ---
            if not selected_profiles_labels:
                st.warning("Select at least one investor profile to see their view.")
            else:
                st.subheader("Superinvestor Views")

                for label in selected_profiles_labels:
                    profile: InvestorProfile = profile_key_map[label]
                    result = profile.score_fn(metrics)
                    score = result.get("score", float("nan"))
                    verdict = result.get("verdict", "No verdict")
                    notes = result.get("notes", [])

                    with st.container():
                        st.markdown(f"### {profile.label}")
                        cols = st.columns([1, 3])
                        cols[0].metric("Score", f"{score:,.1f}/100")
                        cols[1].markdown(f"**Verdict:** {verdict}")
                        if notes:
                            st.markdown("**Rationale:**")
                            for n in notes:
                                st.markdown(f"- {n}")


# ------------------------------------------------------------
# TAB 2 – S&P 500 Screener
# ------------------------------------------------------------
with tab2:
    st.header("📊 S&P 500 Screener – Superinvestor Ranking")

    try:
        sp500_df = load_sp500_universe()
    except Exception as e:
        st.error(
            "Could not load S&P 500 universe. "
            "Ensure `sp500_universe.csv` exists in the repo with columns: "
            "`Ticker,Company,Sector,Industry`.\n\n"
            f"Details: {e}"
        )
        st.stop()

    st.markdown(f"**Universe size:** {len(sp500_df)} stocks")

    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        screener_profiles_labels = st.multiselect(
            "Investor Profiles to Rank By",
            options=profile_labels,
            default=[profile_labels[0], profile_labels[1]],  # e.g. Graham + Buffett
            help="Composite score will be the average of selected profiles.",
        )

    with col2:
        max_tickers = st.slider(
            "Max number of S&P 500 stocks to process",
            min_value=50,
            max_value=len(sp500_df),
            value=150,
            step=25,
            help="More stocks = slower. Start smaller if performance is an issue.",
        )

    with col3:
        run_screener = st.button("Run Screener")

    if run_screener:
        if not screener_profiles_labels:
            st.error("Select at least one investor profile to rank by.")
            st.stop()

        selected_profiles: List[InvestorProfile] = [
            profile_key_map[label] for label in screener_profiles_labels
        ]

        work_df = sp500_df.head(max_tickers).copy()
        results_rows: List[Dict[str, Any]] = []

        progress = st.progress(0, text="Scoring S&P 500 stocks...")
        total = len(work_df)

        for i, row in enumerate(work_df.itertuples(index=False), start=1):
            ticker = row.Ticker
            company = row.Company
            sector = row.Sector
            industry = row.Industry

            try:
                metrics = get_metrics_cached(ticker)
            except Exception:
                # Skip ticker if metrics can't be fetched
                continue

            profile_scores: Dict[str, float] = {}
            for p in selected_profiles:
                try:
                    res = p.score_fn(metrics)
                    s = float(res.get("score", float("nan")))
                except Exception:
                    s = float("nan")
                profile_scores[p.key] = s

            valid_scores = [s for s in profile_scores.values() if not math.isnan(s)]
            composite = float(np.mean(valid_scores)) if valid_scores else float("nan")

            row_dict: Dict[str, Any] = {
                "Ticker": ticker,
                "Company": company,
                "Sector": sector,
                "Industry": industry,
                "Composite Score": composite,
            }

            for p in selected_profiles:
                key_col = f"{p.key.capitalize()} Score"
                row_dict[key_col] = profile_scores.get(p.key, float("nan"))

            results_rows.append(row_dict)

            progress.progress(
                i / total,
                text=f"Scoring S&P 500 stocks... ({i}/{total})",
            )

        progress.empty()

        if not results_rows:
            st.error("No results could be computed for the selected set.")
            st.stop()

        results_df = pd.DataFrame(results_rows)

        results_df = results_df.sort_values(
            by="Composite Score", ascending=False, na_position="last"
        )

        score_cols = [c for c in results_df.columns if "Score" in c]
        for c in score_cols:
            results_df[c] = results_df[c].round(1)

        st.subheader("Ranked S&P 500 (Top Matches First)")
        st.dataframe(results_df, use_container_width=True)
