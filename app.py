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
@st.cache_data(show_spinner=False)
def get_ticker_data_cached(ticker: str) -> Dict[str, Any]:
    return fetch_ticker_data(ticker)


@st.cache_data(show_spinner=False)
def get_metrics_cached(ticker: str) -> Dict[str, Any]:
    raw = get_ticker_data_cached(ticker)
    return compute_metrics(ticker, raw)


@st.cache_data(show_spinner=True)
def load_sp500_universe() -> pd.DataFrame:
    """
    Load S&P 500 universe from sp500_universe.csv.

    Required columns: Ticker,Company,Sector,Industry
    """
    df = pd.read_csv("sp500_universe.csv")
    required = ["Ticker", "Company", "Sector", "Industry"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"sp500_universe.csv missing columns: {missing}")

    df["Ticker"] = (
        df["Ticker"].astype(str).str.upper().str.strip().str.replace(".", "-", regex=False)
    )
    return df[required].copy()


def _fmt_num(x: Any, pct: bool = False) -> str:
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
st.sidebar.caption("See a stock or an index through different legendary investors' checklists.")

profile_labels = [p.label for p in ALL_PROFILES]
profile_key_map = {p.label: p for p in ALL_PROFILES}

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tab1, tab2 = st.tabs(["🔎 Single Stock", "📊 S&P 500 Screener"])


# ------------------------------------------------------------
# TAB 1 – Single Stock
# ------------------------------------------------------------
with tab1:
    st.header("🔎 Single Stock – Superinvestor Checklists")

    col_left, col_right = st.columns([2, 2])

    with col_left:
        ticker_input = st.text_input(
            "Ticker",
            value="AAPL",
            help="Enter a stock ticker (e.g. AAPL, MSFT, JNJ).",
        )

    with col_right:
        selected_profiles_labels = st.multiselect(
            "Investor Profiles",
            options=profile_labels,
            default=profile_labels,
            help="Choose which investor styles to apply.",
        )

    analyze_btn = st.button("Analyze Stock")

    if not analyze_btn:
        st.info("Enter a ticker and click **Analyze Stock** to see investor checklists.")
    else:
        ticker = ticker_input.upper().strip()
        if not ticker:
            st.error("Please enter a valid ticker.")
        elif not selected_profiles_labels:
            st.error("Select at least one investor profile.")
        else:
            with st.spinner(f"Fetching data for {ticker}..."):
                try:
                    metrics = get_metrics_cached(ticker)
                except Exception as e:
                    st.error(f"Failed to fetch data for {ticker}: {e}")
                    st.stop()

            meta = metrics.get("meta", {})
            valuation = metrics.get("valuation", {})
            quality = metrics.get("quality", {})
            growth = metrics.get("growth", {})
            bs = metrics.get("balance_sheet", {})
            # NEW: be robust if dividends block is missing
            divs = metrics.get("dividends", {}) or {}
            if "dividend_yield" not in divs:
                divs["dividend_yield"] = float("nan")
            if "payout_ratio" not in divs:
                divs["payout_ratio"] = float("nan")

            # ----- Stock summary -----
            st.subheader(f"{meta.get('short_name') or ticker} ({ticker})")

            info_cols = st.columns(4)
            info_cols[0].metric("Price", _fmt_num(meta.get("price")))
            mc = meta.get("market_cap")
            info_cols[1].metric(
                "Market Cap",
                f"{mc/1e9:,.2f}B" if mc else "—",
            )
            info_cols[2].metric("Sector", meta.get("sector") or "—")
            info_cols[3].metric("Industry", meta.get("industry") or "—")

            # Dataroma link (13F / superinvestors)
            dataroma_url = f"https://www.dataroma.com/m/stock.php?s={ticker}"
            st.markdown(
                f"[🔗 View real superinvestor 13F holders on Dataroma]({dataroma_url})"
            )

            st.markdown("---")

            # ----- Core metrics snapshot -----
            with st.expander("Core Metrics Snapshot", expanded=True):
                c1, c2, c3 = st.columns(3)

                with c1:
                    st.markdown("**Valuation**")
                    val_df = pd.DataFrame(
                        {
                            "Metric": [
                                "P/E",
                                "P/B",
                                "EV/EBITDA",
                                "EV/Sales",
                                "Earnings Yield",
                                "FCF Yield",
                                "PEG",
                            ],
                            "Value": [
                                _fmt_num(valuation.get("pe")),
                                _fmt_num(valuation.get("pb")),
                                _fmt_num(valuation.get("ev_ebitda")),
                                _fmt_num(valuation.get("ev_sales")),
                                _fmt_num(valuation.get("earnings_yield"), pct=True),
                                _fmt_num(valuation.get("fcf_yield"), pct=True),
                                _fmt_num(valuation.get("peg")),
                            ],
                        }
                    )
                    st.table(val_df)

                with c2:
                    st.markdown("**Quality & Growth**")
                    q_df = pd.DataFrame(
                        {
                            "Metric": [
                                "ROE",
                                "ROA",
                                "Gross Margin",
                                "Operating Margin",
                                "Net Margin",
                                "FCF / Net Income",
                                "Revenue Growth",
                                "Earnings Growth",
                            ],
                            "Value": [
                                _fmt_num(quality.get("roe"), pct=True),
                                _fmt_num(quality.get("roa"), pct=True),
                                _fmt_num(quality.get("gross_margin"), pct=True),
                                _fmt_num(quality.get("op_margin"), pct=True),
                                _fmt_num(quality.get("net_margin"), pct=True),
                                _fmt_num(quality.get("fcf_conversion"), pct=True),
                                _fmt_num(growth.get("revenue_growth"), pct=True),
                                _fmt_num(growth.get("earnings_growth"), pct=True),
                            ],
                        }
                    )
                    st.table(q_df)

                with c3:
                    st.markdown("**Balance Sheet & Dividends**")
                    b_df = pd.DataFrame(
                        {
                            "Metric": [
                                "Debt/Equity",
                                "Current Ratio",
                                "Quick Ratio",
                                "Interest Coverage",
                                "Dividend Yield",
                                "Payout Ratio",
                            ],
                            "Value": [
                                _fmt_num(bs.get("debt_to_equity")),
                                _fmt_num(bs.get("current_ratio")),
                                _fmt_num(bs.get("quick_ratio")),
                                _fmt_num(bs.get("interest_coverage")),
                                _fmt_num(divs.get("dividend_yield"), pct=True),
                                _fmt_num(divs.get("payout_ratio"), pct=True),
                            ],
                        }
                    )
                    st.table(b_df)

            # Metric definitions / tooltips
            with st.expander("Metric definitions (what these mean)", expanded=False):
                st.markdown(
                    """
- **P/E** – Price / Earnings. Lower = cheaper.  
- **P/B** – Price / Book value. Below ~1.5 is classic deep value territory.  
- **EV/EBITDA** – Enterprise value / EBITDA. Common cash-flowish multiple.  
- **EV/Sales** – Enterprise value / Revenue. Useful for low-margin businesses.  
- **Earnings yield** – Approx. E / EV. Inverse of P/E; higher is cheaper.  
- **FCF yield** – Free cash flow / Market cap. Cash-based valuation.  
- **PEG** – P/E divided by growth (%). Around 1 is classic Lynch-style GARP.  

- **ROE / ROA** – Returns on equity / assets; quality & capital efficiency.  
- **Gross / Operating / Net margin** – Profitability at different levels.  
- **FCF / Net income** – How much accounting earnings turn into cash.  

- **Debt/Equity** – Leverage; lower = safer.  
- **Current / Quick ratio** – Short-term liquidity.  
- **Interest coverage** – Ability to pay interest (EBIT / interest expense).  

- **Dividend yield** – Cash yield on price.  
- **Payout ratio** – Dividends / earnings; higher means less room to reinvest.
"""
                )

            st.markdown("---")

            # ----- Superinvestor checklists -----
            status_emoji = {
                "pass": "✅",
                "warn": "⚠️",
                "fail": "❌",
                "na": "❔",
            }

            for label in selected_profiles_labels:
                profile: InvestorProfile = profile_key_map[label]
                result = profile.rules_fn(metrics)

                summary = result.get("summary", {})
                rules = result.get("rules", [])

                st.markdown(f"## {profile.label}")
                st.caption(profile.description)

                # Summary line
                passes = summary.get("passes", 0)
                warns = summary.get("warns", 0)
                fails = summary.get("fails", 0)
                headline = summary.get("headline", "")

                st.markdown(
                    f"**Checklist result:** {passes} ✅   {warns} ⚠️   {fails} ❌"
                    + (f"  —  {headline}" if headline else "")
                )

                # Detailed rules
                for r in rules:
                    status = r.get("status", "na")
                    emoji = status_emoji.get(status, "❔")
                    name = r.get("name", "")
                    cond = r.get("condition", "")
                    value = r.get("value", "—")
                    comment = r.get("comment", "")

                    text = f"{emoji} **{name}** — {cond}  |  **Value:** {value}"
                    if comment:
                        text += f"  \n• {comment}"

                    st.markdown(text)

                st.markdown("---")

            st.info(
                "This tool is a **rough educational checklist** based on Yahoo Finance "
                "snapshots – not a full replication of any investor’s actual process."
            )


# ------------------------------------------------------------
# TAB 2 – S&P 500 Screener (pass-count based)
# ------------------------------------------------------------
with tab2:
    st.header("📊 S&P 500 Checklist Screener")

    try:
        sp500_df = load_sp500_universe()
    except Exception as e:
        st.error(
            "Could not load S&P 500 universe. "
            "Ensure `sp500_universe.csv` exists with columns: "
            "`Ticker,Company,Sector,Industry`.\n\n"
            f"Details: {e}"
        )
        st.stop()

    st.markdown(f"**Universe size:** {len(sp500_df)} stocks")

    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        screener_profiles_labels = st.multiselect(
            "Investor Profiles to Apply",
            options=profile_labels,
            default=[profile_labels[0], profile_labels[1]],  # e.g. Graham + Buffett
            help="We count how many rules each stock passes for the selected profiles.",
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
            st.error("Select at least one investor profile to screen by.")
            st.stop()

        selected_profiles: List[InvestorProfile] = [
            profile_key_map[label] for label in screener_profiles_labels
        ]

        work_df = sp500_df.head(max_tickers).copy()
        results_rows: List[Dict[str, Any]] = []

        progress = st.progress(0, text="Running checklists across S&P 500...")
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

            total_passes = 0
            total_warns = 0
            total_fails = 0

            per_profile_passes: Dict[str, int] = {}

            for p in selected_profiles:
                try:
                    res = p.rules_fn(metrics)
                    summary = res.get("summary", {})
                    passes = int(summary.get("passes", 0))
                    warns = int(summary.get("warns", 0))
                    fails = int(summary.get("fails", 0))
                except Exception:
                    passes = warns = fails = 0

                total_passes += passes
                total_warns += warns
                total_fails += fails

                per_profile_passes[p.key] = passes

            total_rules = total_passes + total_warns + total_fails
            pass_rate = (
                float(total_passes) / total_rules if total_rules > 0 else float("nan")
            )

            row_dict: Dict[str, Any] = {
                "Ticker": ticker,
                "Company": company,
                "Sector": sector,
                "Industry": industry,
                "Total Passes": total_passes,
                "Total Fails": total_fails,
                "Total Warns": total_warns,
                "Total Rules (checked)": total_rules,
                "Pass Rate": pass_rate,
            }

            for p in selected_profiles:
                key_col = f"{p.key.capitalize()} Passes"
                row_dict[key_col] = per_profile_passes.get(p.key, 0)

            results_rows.append(row_dict)

            progress.progress(
                i / total,
                text=f"Running checklists across S&P 500... ({i}/{total})",
            )

        progress.empty()

        if not results_rows:
            st.error("No results could be computed for the selected set.")
            st.stop()

        results_df = pd.DataFrame(results_rows)

        results_df = results_df.sort_values(
            by=["Pass Rate", "Total Passes"],
            ascending=False,
            na_position="last",
        )

        results_df["Pass Rate"] = results_df["Pass Rate"].apply(
            lambda x: _fmt_num(x, pct=True)
        )

        st.subheader("Top Matches (by checklist pass rate)")

        st.dataframe(results_df, use_container_width=True, hide_index=True)

        st.caption(
            "Tip: Use this as a **shortlist generator** – then click into the Single "
            "Stock tab and run full checklists on interesting names."
        )
