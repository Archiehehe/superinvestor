# metrics.py
from __future__ import annotations
import math
from typing import Dict, Iterable, Optional
import numpy as np
import pandas as pd
import yfinance as yf

# ---------- helpers ----------
def _coalesce_key(index_like: Iterable[str], *candidates: str) -> Optional[str]:
    import re
    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", s.lower())
    idx = [norm(x) for x in index_like]
    for cand in candidates:
        n = norm(cand)
        for raw, nidx in zip(index_like, idx):
            if nidx == n:
                return str(raw)
    return None

def _get_df(t, names: Iterable[str]) -> pd.DataFrame:
    for n in names:
        try:
            df = getattr(t, n)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        except Exception:
            pass
    return pd.DataFrame()

def _ttm_sum(qdf: pd.DataFrame, *label_candidates: str) -> float:
    if qdf is None or qdf.empty:
        return float("nan")
    label = _coalesce_key(qdf.index, *label_candidates)
    if label is None:
        return float("nan")
    series = qdf.loc[label].dropna().astype(float)
    if series.empty:
        return float("nan")
    return float(series.iloc[:4].sum())

def _mrq_value(qdf: pd.DataFrame, *label_candidates: str) -> float:
    if qdf is None or qdf.empty:
        return float("nan")
    label = _coalesce_key(qdf.index, *label_candidates)
    if label is None:
        return float("nan")
    series = qdf.loc[label].dropna().astype(float)
    if series.empty:
        return float("nan")
    return float(series.iloc[0])

def _safe_div(n: float, d: float) -> float:
    try:
        if d == 0 or (isinstance(d, float) and math.isnan(d)):
            return float("nan")
        return float(n) / float(d)
    except Exception:
        return float("nan")
# -----------------------------

def _fast_info_get(fi, key):
    if fi is None:
        return np.nan
    try:
        if isinstance(fi, dict):
            return fi.get(key, np.nan)
        return getattr(fi, key, np.nan)
    except Exception:
        return np.nan

def _market_cap_robust(t: yf.Ticker) -> float:
    """Try several ways to get market cap; fallback to price*shares."""
    mc = _fast_info_get(getattr(t, "fast_info", None), "market_cap")
    if mc and not math.isnan(float(mc)):
        return float(mc)
    # info.marketCap
    try:
        info = getattr(t, "info", {}) or {}
        v = float(info.get("marketCap", np.nan))
        if not math.isnan(v):
            return v
    except Exception:
        pass
    # price * shares
    price = _fast_info_get(getattr(t, "fast_info", None), "last_price")
    if not price or math.isnan(float(price)):
        price = _fast_info_get(getattr(t, "fast_info", None), "regular_market_price")
    if not price or math.isnan(float(price)):
        try:
            hist = t.history(period="1d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
        except Exception:
            price = np.nan
    shares = np.nan
    try:
        info = getattr(t, "info", {}) or {}
        shares = float(info.get("sharesOutstanding", np.nan))
    except Exception:
        pass
    if math.isnan(shares):
        try:
            df = t.get_shares_full(start="2015-01-01")
            if isinstance(df, pd.DataFrame) and not df.empty:
                shares = float(df["SharesOutstanding"].dropna().iloc[-1])
        except Exception:
            shares = np.nan
    try:
        val = float(price) * float(shares)
        return val if not math.isnan(val) else float("nan")
    except Exception:
        return float("nan")

def fetch_core_financials(ticker: str) -> Dict[str, float]:
    t = yf.Ticker(ticker)

    # Market cap (robust)
    mc = _market_cap_robust(t)

    # Quarterly statements (allow for name drift)
    q_is = _get_df(t, ["quarterly_income_stmt", "quarterly_income_statement", "quarterly_income", "quarterly_financials"])
    q_bs = _get_df(t, ["quarterly_balance_sheet", "quarterly_balancesheet", "quarterly_balance", "quarterly_balance_sheet"])
    q_cf = _get_df(t, ["quarterly_cashflow", "quarterly_cash_flow", "quarterly_cashflow_statement"])

    # Income statement (TTM)
    revenue_ttm = _ttm_sum(q_is, "Total Revenue", "Revenue", "TotalRevenue")
    ebit_ttm = _ttm_sum(q_is, "EBIT", "Ebit", "Operating Income", "OperatingIncome")
    dep_ttm = _ttm_sum(q_is, "Depreciation And Amortization", "Depreciation", "Depreciation & Amortization", "Reconciled Depreciation")
    ebitda_ttm = _ttm_sum(q_is, "EBITDA", "Ebitda", "EBITDA (ttm)")

    # Backfill EBITDA/EBIT if one missing
    if math.isnan(ebit_ttm) and not math.isnan(ebitda_ttm) and not math.isnan(dep_ttm):
        ebit_ttm = float(ebitda_ttm - dep_ttm)
    if math.isnan(ebitda_ttm) and not math.isnan(ebit_ttm) and not math.isnan(dep_ttm):
        ebitda_ttm = float(ebit_ttm + dep_ttm)

    # Cash flow (TTM)
    cfo_ttm = _ttm_sum(
        q_cf,
        "Total Cash From Operating Activities",
        "Operating Cash Flow",
        "OperatingCashFlow",
        "NetCashProvidedByOperatingActivities",
        "Net Cash Provided By Operating Activities",
    )
    capex_ttm = _ttm_sum(
        q_cf,
        "Capital Expenditures",
        "Capital Expenditure",
        "Purchase Of Property And Equipment",
        "Purchases Of Property And Equipment",
        "Investment In Property Plant And Equipment",
        "Investments In Property Plant And Equipment",
        "Additions To Property Plant And Equipment",
    )
    if math.isnan(dep_ttm):
        dep_ttm = _ttm_sum(q_cf, "Depreciation", "Depreciation And Amortization")

    fcf_ttm = float("nan")
    if not math.isnan(cfo_ttm) and not math.isnan(capex_ttm):
        # CapEx is usually negative in statements; FCF = CFO - CapEx
        fcf_ttm = float(cfo_ttm - capex_ttm)

    # Balance sheet (MRQ)
    total_debt = _mrq_value(q_bs, "Total Debt", "TotalDebt")
    if math.isnan(total_debt):
        lt_debt = _mrq_value(q_bs, "Long Term Debt", "LongTermDebt", "Long Term Debt Noncurrent")
        st_debt = _mrq_value(q_bs, "Short Long Term Debt", "ShortTermDebt", "Current Debt", "Current Portion Of Long Term Debt")
        total_debt = float(0.0 if math.isnan(lt_debt) else lt_debt) + float(0.0 if math.isnan(st_debt) else st_debt)

    cash = _mrq_value(q_bs, "Cash And Cash Equivalents", "Cash", "CashAndCashEquivalents")
    equity = _mrq_value(q_bs, "Total Stockholder Equity", "Stockholders Equity", "Total Equity Gross Minority Interest", "totalStockholdersEquity")

    current_assets = _mrq_value(q_bs, "Total Current Assets", "Current Assets", "CurrentAssets")
    current_liabilities = _mrq_value(q_bs, "Total Current Liabilities", "Current Liabilities", "CurrentLiabilities")
    short_term_debt = _mrq_value(q_bs, "Short Long Term Debt", "ShortTermDebt", "Current Debt")
    net_ppe = _mrq_value(q_bs, "Property Plant Equipment Net", "Net PPE", "PPENet", "NetPropertyPlantAndEquipment")

    # Non-cash NWC
    ncwc = float("nan")
    if not math.isnan(current_assets) and not math.isnan(current_liabilities):
        cash0 = 0.0 if math.isnan(cash) else cash
        std0 = 0.0 if math.isnan(short_term_debt) else short_term_debt
        ncwc = (current_assets - cash0) - (current_liabilities - std0)

    # Enterprise Value (treat missing debt/cash as 0 so we still show EV when MC exists)
    ev = float("nan")
    if not math.isnan(mc):
        ev = float(mc + (0.0 if math.isnan(total_debt) else total_debt) - (0.0 if math.isnan(cash) else cash))

    # Tax rate estimate
    pretax = _ttm_sum(q_is, "Income Before Tax", "Pretax Income", "Income Before Income Taxes")
    tax = _ttm_sum(q_is, "Income Tax Expense", "Provision For Income Taxes")
    tax_rate_est = float("nan")
    try:
        base = abs(float(pretax))
        if base > 0 and not math.isnan(tax):
            tax_rate_est = max(0.0, min(0.35, float(tax) / base))
    except Exception:
        tax_rate_est = float("nan")

    return {
        "Market_Cap": mc,
        "EV": ev,
        "Revenue_TTM": revenue_ttm,
        "EBIT_TTM": ebit_ttm,
        "Dep_TTM": dep_ttm,
        "EBITDA_TTM": ebitda_ttm,
        "CFO_TTM": cfo_ttm,
        "CapEx_TTM": capex_ttm,
        "FCF_TTM": fcf_ttm,
        "Debt_MRQ": total_debt,
        "Cash_MRQ": cash,
        "Equity_MRQ": equity,
        "NWC_MRQ": ncwc,
        "NetPPE_MRQ": net_ppe,
        "Tax_Rate_est": tax_rate_est,
    }

def compute_common_multiples(fin: Dict[str, float]) -> Dict[str, float]:
    ev = fin.get("EV", float("nan"))
    ebitda = fin.get("EBITDA_TTM", float("nan"))
    ebit = fin.get("EBIT_TTM", float("nan"))
    sales = fin.get("Revenue_TTM", float("nan"))
    mc = fin.get("Market_Cap", float("nan"))
    fcf = fin.get("FCF_TTM", float("nan"))

    return {
        "EV/EBITDA": _safe_div(ev, ebitda),
        "EV/EBIT": _safe_div(ev, ebit),
        "EV/Sales": _safe_div(ev, sales),
        "EBIT/EV": _safe_div(ebit, ev),
        "FCF Yield (to Equity)": _safe_div(fcf, mc),
        "P/FCF": _safe_div(mc, fcf),
    }

def compute_roic_variants(fin: Dict[str, float]) -> Dict[str, float]:
    ebit = fin.get("EBIT_TTM", float("nan"))
    tax_rate = fin.get("Tax_Rate_est", float("nan"))
    if math.isnan(tax_rate):
        tax_rate = 0.21
    nopat = float("nan") if math.isnan(ebit) else float(ebit) * (1 - float(tax_rate))

    debt = fin.get("Debt_MRQ", float("nan"))
    equity = fin.get("Equity_MRQ", float("nan"))
    cash = fin.get("Cash_MRQ", float("nan"))
    invested_capital = float("nan")
    if not math.isnan(debt) or not math.isnan(equity) or not math.isnan(cash):
        invested_capital = (0.0 if math.isnan(debt) else debt) + (0.0 if math.isnan(equity) else equity) - (0.0 if math.isnan(cash) else cash)

    nwc = fin.get("NWC_MRQ", float("nan"))
    net_ppe = fin.get("NetPPE_MRQ", float("nan"))
    magic_formula_capital = float("nan")
    if not math.isnan(nwc) or not math.isnan(net_ppe):
        magic_formula_capital = (0.0 if math.isnan(nwc) else nwc) + (0.0 if math.isnan(net_ppe) else net_ppe)

    return {
        "ROIC": _safe_div(nopat, invested_capital),
        "ROC_Greenblatt": _safe_div(ebit, magic_formula_capital),
    }
