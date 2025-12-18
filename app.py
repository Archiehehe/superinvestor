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
st.sidebar.caption("See a stock through different legendary investors' checklists.")

profile_labels = [p.label for p in ALL_PROFILES]
profile_key_map = {p.label: p for p in ALL_PROFILES}


# ------------------------------------------------------------
# Layout – Single-stock only for now
# ------------------------------------------------------------
st.title("🔎 Single Stock – Superinvestor Checklists")

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
    st.stop()

ticker = ticker_input.upper().strip()
if not ticker:
    st.error("Please enter a valid ticker.")
    st.stop()

if not selected_profiles_labels:
    st.error("Select at least one investor profile.")
    st.stop()

# ------------------------------------------------------------
# Fetch metrics
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Stock summary
# ------------------------------------------------------------
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

st.markdown("---")

# ------------------------------------------------------------
# Core metrics snapshot (so user sees the raw numbers)
# ------------------------------------------------------------
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
                    _fmt_num(valuation["pe"]),
                    _fmt_num(valuation["pb"]),
                    _fmt_num(valuation["ev_ebitda"]),
                    _fmt_num(valuation["ev_sales"]),
                    _fmt_num(valuation["earnings_yield"], pct=True),
                    _fmt_num(valuation["fcf_yield"], pct=True),
                    _fmt_num(valuation["peg"]),
                ],
            }
        )
        st.table(val_df)

    with c2:
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
                    _fmt_num(quality["roe"], pct=True),
                    _fmt_num(quality["roa"], pct=True),
                    _fmt_num(quality["gross_margin"], pct=True),
                    _fmt_num(quality["op_margin"], pct=True),
                    _fmt_num(quality["net_margin"], pct=True),
                    _fmt_num(quality["fcf_conversion"], pct=True),
                ],
            }
        )
        st.table(q_df)

    with c3:
        st.markdown("**Balance Sheet & Growth**")
        b_df = pd.DataFrame(
            {
                "Metric": [
                    "Debt/Equity",
                    "Current Ratio",
                    "Quick Ratio",
                    "Interest Coverage",
                    "Revenue Growth",
                    "Earnings Growth",
                ],
                "Value": [
                    _fmt_num(bs["debt_to_equity"]),
                    _fmt_num(bs["current_ratio"]),
                    _fmt_num(bs["quick_ratio"]),
                    _fmt_num(bs["interest_coverage"]),
                    _fmt_num(growth["revenue_growth"], pct=True),
                    _fmt_num(growth["earnings_growth"], pct=True),
                ],
            }
        )
        st.table(b_df)

st.markdown("---")

# ------------------------------------------------------------
# Superinvestor checklists
# ------------------------------------------------------------
status_emoji = {
    "pass": "✅",
    "warn": "⚠️",
    "fail": "❌",
    "na": "❔",
}

for label in selected_profiles_labels:
    profile: InvestorProfile = profile_key_map[label]
    result = profile.rules_fn(metrics)  # renamed behavior: returns summary + rules

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
    "This tool is a **rough educational checklist** based on reported Yahoo Finance "
    "snapshots – not a full replication of any investor’s actual process."
)
