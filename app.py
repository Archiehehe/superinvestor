# app.py
import math
import streamlit as st
import pandas as pd
import numpy as np

from metrics import fetch_core_financials, compute_common_multiples, compute_roic_variants
from lenses import burry_lens, greenblatt_lens, buffett_lens, klarman_lens, einhorn_lens

st.set_page_config(page_title="Superinvestor Screener", layout="centered")
st.title("🧠 Superinvestor Screener")
st.write("Type a ticker and choose a lens. Approximate numbers via yfinance.")

# -------- formatting helpers --------
def _isnan(x): return isinstance(x, float) and np.isnan(x)
def fmt_money_short(x):
    try:
        if x is None or _isnan(float(x)): return "—"
        neg = float(x) < 0
        v = abs(float(x))
        s = f"{v/1e12:.1f}T" if v>=1e12 else f"{v/1e9:.1f}B" if v>=1e9 else f"{v/1e6:.1f}M" if v>=1e6 else f"{v:,.0f}"
        return f"-{s}" if neg else s
    except Exception:
        return "—"
def fmt_ratio(x): return "—" if (x is None or _isnan(float(x))) else f"{float(x):.2f}"
# ------------------------------------

with st.sidebar:
    st.header("⚙️ Controls")
    ticker = st.text_input("Ticker", value="AAPL").strip().upper()
    lens_name = st.selectbox(
        "Investor Lens",
        ["Burry (EV/EBITDA)","Greenblatt (Magic Formula)","Buffett (Owner Earnings / ROIC)","Klarman (Asset / MoS)","Einhorn (Relative Value)"],
        index=0,
    )
    run = st.button("Run Analysis", use_container_width=True)

@st.cache_data(show_spinner=False, ttl=1800)
def _cached_financials(t: str) -> dict:
    return fetch_core_financials(t)

def _run_lens(name, fin, mult, roic):
    if name.startswith("Burry"): return burry_lens(fin, mult, roic)
    if name.startswith("Greenblatt"): return greenblatt_lens(fin, mult, roic)
    if name.startswith("Buffett"): return buffett_lens(fin, mult, roic)
    if name.startswith("Klarman"): return klarman_lens(fin, mult, roic)
    if name.startswith("Einhorn"): return einhorn_lens(fin, mult, roic)
    return {"verdict":"N/A","score":np.nan,"checks":{},"notes":["Unknown lens"]}

if run:
    if not ticker:
        st.warning("Please enter a ticker.")
    else:
        with st.spinner("Fetching statements…"):
            fin = _cached_financials(ticker)

        st.subheader(f"Core Metrics — {ticker}")
        core = [
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
        df = pd.DataFrame(core, columns=["Metric","Value"])
        df["Value"] = [
            (f"{v:.0%}" if k=="Tax Rate (est)" and not _isnan(v) else fmt_money_short(v) if k!="Tax Rate (est)" else "—")
            for k,v in zip(df["Metric"], df["Value"])
        ]
        st.dataframe(df, use_container_width=True)

        mult = compute_common_multiples(fin)
        roic = compute_roic_variants(fin)

        st.subheader("Multiples / Yields")
        mdf = pd.DataFrame(mult.items(), columns=["Multiple","Value"])
        mdf["Value"] = mdf["Value"].apply(fmt_ratio)
        st.dataframe(mdf, use_container_width=True)

        st.subheader("Returns on Capital")
        rdf = pd.DataFrame(
            [("ROIC (NOPAT / Invested Capital)", roic.get("ROIC")),
             ("ROC (Greenblatt: EBIT / (NWC + Net PPE))", roic.get("ROC_Greenblatt"))],
            columns=["Metric","Value"]
        )
        rdf["Value"] = rdf["Value"].apply(lambda x: "—" if x is None or _isnan(x) else f"{x:.2%}")
        st.dataframe(rdf, use_container_width=True)

        st.subheader(f"Lens Verdict — {lens_name}")
        verdict = _run_lens(lens_name, fin, mult, roic)
        c1, c2 = st.columns([1,2])
        with c1:
            st.metric("Verdict", verdict.get("verdict","N/A"))
            score = verdict.get("score", np.nan)
            st.metric("Score", "—" if isinstance(score, float) and np.isnan(score) else int(score))
        with c2:
            checks = pd.DataFrame([{"Check":k,"Pass":"✅" if v else "❌"} for k,v in verdict.get("checks",{}).items()])
            if not checks.empty: st.dataframe(checks, use_container_width=True)

        st.caption("TTM=last 4 quarters; MRQ=most recent quarter. Educational; not investment advice.")
else:
    st.info("Enter a ticker (e.g., AAPL, MSFT, TSLA) and click **Run Analysis**.")
