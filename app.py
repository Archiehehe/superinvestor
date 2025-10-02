# app.py
import streamlit as st
import pandas as pd
import numpy as np

from metrics import fetch_core_financials, compute_common_multiples, compute_roic_variants
from lenses import burry_lens, greenblatt_lens, buffett_lens, klarman_lens, einhorn_lens

st.set_page_config(page_title="Superinvestor Screener", layout="centered")

st.title("🧠 Superinvestor Screener")
st.write(
    "Type 1 ticker and choose a lens. The app fetches TTM/MRQ numbers via yfinance and shows quick valuation stats. "
    "This is an idea filter only — numbers are approximate."
)

with st.sidebar:
    st.header("⚙️ Controls")
    ticker = st.text_input("Ticker", value="AAPL").strip().upper()
    lens_name = st.selectbox(
        "Investor Lens",
        [
            "Burry (EV/EBITDA)",
            "Greenblatt (Magic Formula)",
            "Buffett (Owner Earnings / ROIC)",
            "Klarman (Asset / MoS)",
            "Einhorn (Relative Value)",
        ],
        index=0,
    )
    run = st.button("Run Analysis", use_container_width=True)

@st.cache_data(show_spinner=False, ttl=3600)
def _cached_financials(t: str) -> dict:
    return fetch_core_financials(t)

def run_lens(name: str, fin: dict, mult: dict, roic: dict) -> dict:
    if name.startswith("Burry"):
        return burry_lens(fin, mult, roic)
    if name.startswith("Greenblatt"):
        return greenblatt_lens(fin, mult, roic)
    if name.startswith("Buffett"):
        return buffett_lens(fin, mult, roic)
    if name.startswith("Klarman"):
        return klarman_lens(fin, mult, roic)
    if name.startswith("Einhorn"):
        return einhorn_lens(fin, mult, roic)
    return {"verdict": "N/A", "score": np.nan, "checks": {}, "notes": ["Unknown lens"]}

if run:
    if not ticker:
        st.warning("Please enter a ticker.")
    else:
        with st.spinner("Fetching statements…"):
            fin = _cached_financials(ticker)

        st.subheader(f"Core Metrics — {ticker}")
        core_rows = [
            ("Market Cap", fin["Market_Cap"]),
            ("Enterprise Value", fin["EV"]),
            ("Revenue (TTM)", fin["Revenue_TTM"]),
            ("EBIT (TTM)", fin["EBIT_TTM"]),
            ("Depreciation (TTM)", fin["Dep_TTM"]),
            ("EBITDA (TTM)", fin["EBITDA_TTM"]),
            ("Operating Cash Flow (TTM)", fin["CFO_TTM"]),
            ("CapEx (TTM)", fin["CapEx_TTM"]),
            ("FCF (TTM)", fin["FCF_TTM"]),
            ("Debt (MRQ)", fin["Debt_MRQ"]),
            ("Cash (MRQ)", fin["Cash_MRQ"]),
            ("Equity (MRQ)", fin["Equity_MRQ"]),
            ("NWC (MRQ)", fin["NWC_MRQ"]),
            ("Net PPE (MRQ)", fin["NetPPE_MRQ"]),
            ("Tax Rate (est)", fin["Tax_Rate_est"]),
        ]
        st.dataframe(pd.DataFrame(core_rows, columns=["Metric", "Value"]).style.format({"Value": "{:,.0f}"}), use_container_width=True)

        mult = compute_common_multiples(fin)
        roic = compute_roic_variants(fin)

        st.subheader("Multiples / Yields")
        st.dataframe(pd.DataFrame(mult.items(), columns=["Multiple", "Value"]).style.format({"Value": "{:,.2f}"}), use_container_width=True)

        st.subheader("Returns on Capital")
        st.dataframe(pd.DataFrame(roic.items(), columns=["Metric", "Value"]).style.format({"Value": "{:,.2%}"}), use_container_width=True)

        st.subheader(f"Lens Verdict — {lens_name}")
        verdict = run_lens(lens_name, fin, mult, roic)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Verdict", verdict.get("verdict", "N/A"))
            score = verdict.get("score", np.nan)
            st.metric("Score", "—" if isinstance(score, float) and np.isnan(score) else int(score))
        with col2:
            checks_df = pd.DataFrame([{"Check": k, "Pass": "✅" if v else "❌"} for k, v in verdict.get("checks", {}).items()])
            if not checks_df.empty:
                st.dataframe(checks_df, use_container_width=True)

        if verdict.get("notes"):
            with st.expander("Notes"):
                for n in verdict["notes"]:
                    st.write("• " + n)

        st.caption(
            "Data via yfinance. TTM values are last four quarters; MRQ balances. "
            "Invested Capital ≈ Debt + Equity − Cash; Magic Formula capital = NWC + Net PPE. "
            "Educational; not investment advice."
        )
else:
    st.info("Enter a ticker (e.g., AAPL, MSFT, TSLA) and click **Run Analysis**.")
