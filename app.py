# app.py
import io
import math
import streamlit as st
import pandas as pd
import numpy as np

from metrics import (
    fetch_core_financials,
    compute_common_multiples,
    compute_roic_variants,
)
from lenses import (
    burry_lens,
    greenblatt_lens,
    buffett_lens,
    klarman_lens,
    einhorn_lens,
)

st.set_page_config(page_title="Superinvestor Valuation Engine", layout="centered")

st.title("🧠 Superinvestor Valuation Engine")
st.write("Type a ticker, pick an investor lens, and see an at-a-glance valuation view.")

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

def _format_percent(x):
    return "" if pd.isna(x) else f"{x*100:,.1f}%"

def _format_ratio(x):
    return "" if pd.isna(x) else f"{x:,.2f}×"

def _format_number(x):
    if pd.isna(x):
        return ""
    try:
        absx = abs(float(x))
        if absx >= 1e11:
            return f"{x/1e9:,.0f}B"
        if absx >= 1e9:
            return f"{x/1e9:,.1f}B"
        if absx >= 1e8:
            return f"{x/1e6:,.0f}M"
        if absx >= 1e6:
            return f"{x/1e6:,.1f}M"
        return f"{x:,.0f}"
    except Exception:
        return str(x)

lens_map = {
    "Burry (EV/EBITDA)": "burry",
    "Greenblatt (Magic Formula)": "greenblatt",
    "Buffett (Owner Earnings / ROIC)": "buffett",
    "Klarman (Asset / MoS)": "klarman",
    "Einhorn (Relative Value)": "einhorn",
}

if run and ticker:
    try:
        with st.status("Fetching data & computing metrics…", state="running"):
            fin = fetch_core_financials(ticker)
            multiples = compute_common_multiples(fin)
            returns = compute_roic_variants(fin)

        # Core metrics
        st.subheader(f"Core Metrics — {ticker}")
        core_rows = {
            "Market Cap": _format_number(fin.get("Market_Cap")),
            "Enterprise Value (EV)": _format_number(fin.get("EV")),
            "Revenue (TTM)": _format_number(fin.get("Revenue_TTM")),
            "EBIT (TTM)": _format_number(fin.get("EBIT_TTM")),
            "Depreciation (TTM)": _format_number(fin.get("Dep_TTM")),
            "EBITDA (TTM)": _format_number(fin.get("EBITDA_TTM")),
            "CFO (TTM)": _format_number(fin.get("CFO_TTM")),
            "CapEx (TTM)": _format_number(fin.get("CapEx_TTM")),
            "FCF (TTM)": _format_number(fin.get("FCF_TTM")),
            "Debt (MRQ)": _format_number(fin.get("Debt_MRQ")),
            "Cash (MRQ)": _format_number(fin.get("Cash_MRQ")),
            "Total Equity (MRQ)": _format_number(fin.get("Equity_MRQ")),
            "NWC (MRQ)": _format_number(fin.get("NWC_MRQ")),
            "Net PPE (MRQ)": _format_number(fin.get("NetPPE_MRQ")),
            "Tax Rate (est.)": _format_percent(fin.get("Tax_Rate_est")),
        }
        core_df = pd.DataFrame.from_dict(core_rows, orient="index", columns=["Value"])
        st.dataframe(core_df, use_container_width=True)

        # Multiples / yields
        st.subheader("Valuation Multiples / Yields")
        mult_rows = {
            "EV/EBITDA": _format_ratio(multiples.get("EV/EBITDA")),
            "EV/EBIT": _format_ratio(multiples.get("EV/EBIT")),
            "EV/Sales": _format_ratio(multiples.get("EV/Sales")),
            "EBIT/EV (Earnings Yield)": _format_percent(multiples.get("EBIT/EV")),
            "FCF Yield (to Equity)": _format_percent(multiples.get("FCF Yield (to Equity)")),
            "P/FCF": _format_ratio(multiples.get("P/FCF")),
        }
        mult_df = pd.DataFrame.from_dict(mult_rows, orient="index", columns=["Value"])
        st.dataframe(mult_df, use_container_width=True)

        # Returns on capital
        st.subheader("Returns on Capital")
        ret_rows = {
            "ROIC (NOPAT / Invested Capital)": _format_percent(returns.get("ROIC (NOPAT / Invested Capital)")),
            "ROC (Greenblatt, EBIT / (NWC + Net PPE))": _format_percent(returns.get("ROC (Greenblatt, EBIT / (NWC + Net PPE))")),
        }
        ret_df = pd.DataFrame.from_dict(ret_rows, orient="index", columns=["Value"])
        st.dataframe(ret_df, use_container_width=True)

        # Lens verdict
        st.subheader("Lens Verdict")
        lk = lens_map.get(lens_name, "burry")
        if lk == "burry":
            verdict = burry_lens(multiples)
        elif lk == "greenblatt":
            verdict = greenblatt_lens(multiples, returns)
        elif lk == "buffett":
            verdict = buffett_lens(multiples, returns)
        elif lk == "klarman":
            verdict = klarman_lens(fin, multiples)
        else:
            verdict = einhorn_lens(multiples)
        st.json(verdict)

        # Optional: PDF export
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            from reportlab.lib.units import inch

            def build_pdf() -> bytes:
                buf = io.BytesIO()
                c = canvas.Canvas(buf, pagesize=letter)
                width, height = letter
                x = 0.75*inch
                y = height - 0.75*inch

                def line(text, dy=14):
                    nonlocal y
                    c.drawString(x, y, text)
                    y -= dy

                c.setFont("Helvetica-Bold", 14)
                line(f"Superinvestor Valuation — {ticker}")
                c.setFont("Helvetica", 10)
                line(f"Lens: {verdict.get('Lens','N/A')}")

                line("")
                c.setFont("Helvetica-Bold", 11)
                line("Core Metrics")
                c.setFont("Helvetica", 9)
                for k, v in core_rows.items():
                    line(f"{k}: {v}")

                line("")
                c.setFont("Helvetica-Bold", 11)
                line("Multiples / Yields")
                c.setFont("Helvetica", 9)
                for k, v in mult_rows.items():
                    line(f"{k}: {v}")

                line("")
                c.setFont("Helvetica-Bold", 11)
                line("Returns")
                c.setFont("Helvetica", 9)
                for k, v in ret_rows.items():
                    line(f"{k}: {v}")

                line("")
                c.setFont("Helvetica-Bold", 11)
                line("Verdict")
                c.setFont("Helvetica", 9)
                line(f"Conclusion: {verdict.get('Verdict','')}")
                for n in verdict.get("Heuristics", []):
                    line(f"- {n}")

                line("")
                c.setFont("Helvetica-Oblique", 8)
                line("Notes: Data via yfinance; TTM = sum of last 4 quarters. Educational; not investment advice.")
                c.showPage()
                c.save()
                buf.seek(0)
                return buf.getvalue()

            pdf_bytes = build_pdf()
            st.download_button(
                "Download PDF report",
                data=pdf_bytes,
                file_name=f"{ticker}_{lk}_valuation.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except Exception:
            pass

        st.caption(
            "Data via yfinance. TTM values are approximated by summing last four quarters. "
            "Invested Capital ≈ Debt + Equity − Cash; Magic Formula capital = NWC + Net PPE. "
            "Tax rate is clamped 0–35% using Income Tax / |Pretax Income|. "
            "Educational; not investment advice."
        )
    except Exception as e:
        st.error(f"Error: {e!s}")
else:
    st.info("Enter a ticker (e.g., AAPL, MSFT, TSLA) and click **Run Analysis**.")
