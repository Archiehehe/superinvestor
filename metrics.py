# metrics.py
from __future__ import annotations

import math
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import yfinance as yf


def _coalesce_key(index_like: Iterable[str], *candidates: str) -> str | None:
    """Return the first matching key (case-insensitive, ignoring punctuation)."""
    import re
    norm = lambda s: re.sub(r"[^a-z0-9]+", "", s.lower()) if isinstance(s, str) else ""
    keys = {norm(k): k for k in index_like}
    for cand in candidates:
        k = norm(cand)
        if k in keys:
            return keys[k]
    return None


def _get_row(df: pd.DataFrame, *candidates: str) -> pd.Series | None:
    if df is None or df.empty:
        return None
    key = _coalesce_key(df.index, *candidates)
    return (df.loc[key] if key in df.index else None) if key else None


def _ttm_sum(df: pd.DataFrame, *candidates: str) -> float:
    """Sum the last four quarters of the matching row. Returns np.nan if not found."""
    try:
        row = _get_row(df, *candidates)
        if row is None:
            return float("nan")
        vals = pd.to_numeric(row[-4:], errors="coerce")
        return float(vals.sum())
    except Exception:
        return float("nan")


def _mrq_value(df: pd.DataFrame, *candidates: str) -> float:
    """Most-recent quarter value for the matching row."""
    try:
        row = _get_row(df, *candidates)
        if row is None:
            return float("nan")
        val = pd.to_numeric(row[-1:], errors="coerce")
        return float(val.iloc[0])
    except Exception:
        return float("nan")


def fetch_core_financials(ticker: str) -> Dict[str, float]:
    """Fetch core items and compute derived fields needed for the app."""
    t = yf.Ticker(ticker)

    # Price / market cap
    mc = np.nan
    try:
        fi = getattr(t, "fast_info", {}) or {}
        mc = float(fi.get("market_cap", np.nan))
    except Exception:
        mc = np.nan

    # Quarterly statements
    try:
        q_is = t.quarterly_income_stmt or t.quarterly_financials  # fallback alias
    except Exception:
        q_is = pd.DataFrame()
    try:
        q_bs = t.quarterly_balance_sheet
    except Exception:
        q_bs = pd.DataFrame()
    try:
        q_cf = t.quarterly_cashflow
    except Exception:
        q_cf = pd.DataFrame()

    # TTM calculations
    EBIT_ttm = _ttm_sum(q_is, "Operating Income", "EBIT", "Earnings Before Interest And Taxes")
    Dep_ttm = _ttm_sum(q_cf, "Depreciation And Amortization", "Depreciation", "Depreciation & Amortization")
    EBITDA_ttm = EBIT_ttm + Dep_ttm if not (math.isnan(EBIT_ttm) or math.isnan(Dep_ttm)) else float("nan")
    Revenue_ttm = _ttm_sum(q_is, "Total Revenue", "Revenue", "Sales")
    CFO_ttm = _ttm_sum(q_cf, "Total Cash From Operating Activities", "Operating Cash Flow", "Net Cash Provided By Operating Activities")
    CapEx_ttm = _ttm_sum(q_cf, "Capital Expenditures", "Investment In Property Plant And Equipment")
    FCF_ttm = CFO_ttm - CapEx_ttm if not (math.isnan(CFO_ttm) or math.isnan(CapEx_ttm)) else float("nan")

    # MRQ balance sheet items
    Debt_mrq = _mrq_value(q_bs, "Total Debt", "Short Long Term Debt", "Long Term Debt", "Short Term Debt")
    Cash_mrq = _mrq_value(q_bs, "Cash And Cash Equivalents", "Cash And Short Term Investments", "Cash Cash Equivalents And Short Term Investments")
    Equity_mrq = _mrq_value(q_bs, "Total Stockholder Equity", "Total Equity Gross Minority Interest", "Common Stock Equity")
    CA_mrq = _mrq_value(q_bs, "Total Current Assets", "Current Assets")
    CL_mrq = _mrq_value(q_bs, "Total Current Liabilities", "Current Liabilities")
    NetPPE_mrq = _mrq_value(q_bs, "Property Plant Equipment Net", "Net PPE", "Property Plant And Equipment Net")

    NWC_mrq = (CA_mrq - CL_mrq) if not (math.isnan(CA_mrq) or math.isnan(CL_mrq)) else float("nan")

    # Tax rate estimate
    tax_exp_ttm = _ttm_sum(q_is, "Income Tax Expense", "Provision For Income Taxes")
    pretax_ttm = _ttm_sum(q_is, "Income Before Tax", "Pretax Income")
    tax_rate = float("nan")
    try:
        if not (math.isnan(tax_exp_ttm) or math.isnan(pretax_ttm)) and pretax_ttm != 0:
            tax_rate = max(0.0, min(0.35, abs(tax_exp_ttm) / abs(pretax_ttm)))
        else:
            tax_rate = 0.21  # fallback
    except Exception:
        tax_rate = 0.21

    NOPAT_ttm = EBIT_ttm * (1.0 - tax_rate) if not math.isnan(EBIT_ttm) else float("nan")

    # Enterprise value & invested capital
    EV = (mc if not math.isnan(mc) else 0.0)          + (0.0 if math.isnan(Debt_mrq) else Debt_mrq)          - (0.0 if math.isnan(Cash_mrq) else Cash_mrq)

    Invested_Capital = (0.0 if math.isnan(Debt_mrq) else Debt_mrq)                        + (0.0 if math.isnan(Equity_mrq) else Equity_mrq)                        - (0.0 if math.isnan(Cash_mrq) else Cash_mrq)

    MF_capital = (0.0 if math.isnan(NWC_mrq) else NWC_mrq) + (0.0 if math.isnan(NetPPE_mrq) else NetPPE_mrq)

    return {
        "Market_Cap": mc,
        "EV": EV,
        "Revenue_TTM": Revenue_ttm,
        "EBIT_TTM": EBIT_ttm,
        "Dep_TTM": Dep_ttm,
        "EBITDA_TTM": EBITDA_ttm,
        "CFO_TTM": CFO_ttm,
        "CapEx_TTM": CapEx_ttm,
        "FCF_TTM": FCF_ttm,
        "Debt_MRQ": Debt_mrq,
        "Cash_MRQ": Cash_mrq,
        "Equity_MRQ": Equity_mrq,
        "NWC_MRQ": NWC_mrq,
        "NetPPE_MRQ": NetPPE_mrq,
        "Invested_Capital": Invested_Capital,
        "MF_capital": MF_capital,
        "Tax_Rate_est": tax_rate,
        "NOPAT_TTM": NOPAT_ttm,
    }


def compute_common_multiples(fin: Dict[str, float]) -> Dict[str, float]:
    EV = fin.get("EV", float("nan"))
    EBIT = fin.get("EBIT_TTM", float("nan"))
    EBITDA = fin.get("EBITDA_TTM", float("nan"))
    Sales = fin.get("Revenue_TTM", float("nan"))
    MC = fin.get("Market_Cap", float("nan"))
    FCF = fin.get("FCF_TTM", float("nan"))

    def safe_div(a, b):
        try:
            if b is None or (isinstance(b, float) and (math.isnan(b) or b == 0)):
                return float("nan")
            return float(a) / float(b)
        except Exception:
            return float("nan")

    return {
        "EV/EBITDA": safe_div(EV, EBITDA),
        "EV/EBIT": safe_div(EV, EBIT),
        "EV/Sales": safe_div(EV, Sales),
        "EBIT/EV": safe_div(EBIT, EV),
        "FCF Yield (to Equity)": safe_div(FCF, MC),
        "P/FCF": safe_div(MC, FCF),
    }


def compute_roic_variants(fin: Dict[str, float]) -> Dict[str, float]:
    EBIT = fin.get("EBIT_TTM", float("nan"))
    NOPAT = fin.get("NOPAT_TTM", float("nan"))
    IC = fin.get("Invested_Capital", float("nan"))
    MF_capital = fin.get("MF_capital", float("nan"))

    def safe_div(a, b):
        try:
            if b is None or (isinstance(b, float) and (math.isnan(b) or b == 0)):
                return float("nan")
            return float(a) / float(b)
        except Exception:
            return float("nan")

    roic_simple = safe_div(NOPAT, IC)
    roc_magic = safe_div(EBIT, MF_capital)

    return {
        "ROIC (NOPAT / Invested Capital)": roic_simple,
        "ROC (Greenblatt, EBIT / (NWC + Net PPE))": roc_magic,
    }
