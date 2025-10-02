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

# ----- formatting helpers -----
def _isnan(x): 
    try: return isinstance(x, float) and np.isnan(x)
    except: return False

def fmt_money_short(x: float) -> str:
    try:
        if x is None or _isnan(float(x)): return "—"
        neg = float(x) < 0
        v = abs(float(x))
        s = (
            f"{v/1e12:.1f}T" if v >= 1e12 else
            f"{v/1e9:.1f}B"  if v >= 1e9  else
            f"{v/1e6:.1f}M"  if v >= 1e6  else
            f"{v:,.0f}"
        )
        return f"-{s}" if neg else s
    except Exception:
        return "—"

def fmt_ratio(x: float) -> str:
    try:
        return "—" if x is None or _isnan(float(x)) else f"{float(x):.2f}"
    except Exception:
        return "—"
# --------------------------------

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

@st.cache_data(show_spinner=False, ttl=1800)
def _cached_financials(t: str) -> dict:
    return fetch_core_financials(t)

def run_lens(name: str, fin: dict, mult: dict, roic: dict) -> dict:
    if name.startswith("Burry"): return burry_lens(fin, mult, roic)
    if name.startswith("Greenblatt"): return greenblatt_lens(fin, mult, roic)
    if name.startswith("Buffett"): return buffett_lens(fin, mult, roic)
    if name.startswith("Klarman"): return klarman_lens(fin, mult, roic)
    if name.startswith("Einhorn"): return einhorn_lens(fin, mult, roic)
    return {"verdict": "N/A", "score": np.nan, "checks": {}, "notes": ["Unknown lens"]}

if run:
    if not ticker:
        st.warning("Please enter a ticker.")
    else:
        with st.spinner("Fetching statements…"):
            fin = _cached_financials(ticker)

        mkt_ccy = fin.get("Currency_Market", "USD")
        fin_ccy = fin.get("Currency_Financial", mkt_ccy)
        fx = fin.get("FX_fin_to_market", 1.0)

        st.subheader(f"Core Metrics — {ticker}")
        core_rows = [
            ("Market currency", mkt_ccy),
            ("Financial currency", fin_ccy),
            ("FX (fin→market)", f"{fx:.4f}" if not _isnan(fx) else "—"),
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
        core_df = pd.DataFrame(core_rows, columns=["Metric", "Value"])
        # format money-like rows
        def _fmt_row(metric, val):
            if metric in ("Market currency", "Financial currency", "FX (fin→market)"):
                return val
            if metric == "Tax Rate (est)":
                return "—" if _isnan(val) else f"{val:.0%}"
            return fmt_money_short(val)
        core_df["Value"] = [ _fmt_row(m, v) for m,v in zip(core_df["Metric"], core_df["Value"]) ]
        st.dataframe(core_df, use_container_width=True)

        mult = compute_common_multiples(fin)
        roic = compute_roic_variants(fin)

        st.subheader("Multiples / Yields")
        mult_df = pd.DataFrame(mult.items(), columns=["Multiple", "Value"])
        mult_df["Value"] = mult_df["Value"].apply(fmt_ratio)
        st.dataframe(mult_df, use_container_width=True)

        st.subheader("Returns on Capital")
        roic_rows = [
            ("ROIC (NOPAT / Invested Capital)", roic.get("ROIC")),
            ("ROC (Greenblatt: EBIT / (NWC + Net PPE))", roic.get("ROC_Greenblatt")),
        ]
        roic_df = pd.DataFrame(roic_rows, columns=["Metric", "Value"])
        roic_df["Value"] = roic_df["Value"].apply(lambda x: "—" if _isnan(x) else f"{x:.2%}")
        st.dataframe(roic_df, use_container_width=True)

        st.subheader(f"Lens Verdict — {lens_n_
