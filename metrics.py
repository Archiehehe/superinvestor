# metrics.py
from __future__ import annotations
import math
from typing import Dict, Iterable, Optional
import numpy as np
import pandas as pd
import yfinance as yf

# ---------------- helpers ----------------
def _coalesce_key(index_like: Iterable[str], *cands: str) -> Optional[str]:
    import re
    def norm(s: str) -> str: return re.sub(r"[^a-z0-9]+", "", s.lower())
    idx = [norm(str(x)) for x in index_like]
    for c in cands:
        n = norm(c)
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

def _ttm_sum(qdf: pd.DataFrame, *labels: str) -> float:
    if qdf is None or qdf.empty: return float("nan")
    lab = _coalesce_key(qdf.index, *labels)
    if lab is None: return float("nan")
    s = qdf.loc[lab].dropna().astype(float)
    if s.empty: return float("nan")
    return float(s.iloc[:4].sum())

def _mrq_value(qdf: pd.DataFrame, *labels: str) -> float:
    if qdf is None or qdf.empty: return float("nan")
    lab = _coalesce_key(qdf.index, *labels)
    if lab is None: return float("nan")
    s = qdf.loc[lab].dropna().astype(float)
    if s.empty: return float("nan")
    return float(s.iloc[0])

def _safe_div(n: float, d: float) -> float:
    try:
        if d == 0 or (isinstance(d, float) and math.isnan(d)): return float("nan")
        return float(n) / float(d)
    except Exception:
        return float("nan")

def _recent_close(t: yf.Ticker) -> float:
    # fast_info first
    for k in ("last_price", "regularMarketPrice", "regular_market_price"):
        try:
            v = getattr(t, "fast_info", None)
            if v:
                if isinstance(v, dict) and k in v and v[k] is not None:
                    return float(v[k])
                if hasattr(v, k) and getattr(v, k) is not None:
                    return float(getattr(v, k))
        except Exception:
            pass
    # history fallback
    try:
        h = t.history(period="5d")
        if not h.empty:
            return float(h["Close"].dropna().iloc[-1])
    except Exception:
        pass
    return float("nan")

def _shares_outstanding_robust(t: yf.Ticker, q_is: pd.DataFrame) -> float:
    # 1) info
    try:
        info = getattr(t, "info", {}) or {}
        so = info.get("sharesOutstanding", None)
        if so not in (None, "None"):
            so = float(so)
            if not math.isnan(so) and so > 0:
                return so
    except Exception:
        pass
    # 2) fast_info
    try:
        fi = getattr(t, "fast_info", None)
        for k in ("shares", "shares_outstanding"):
            if fi:
                v = fi.get(k) if isinstance(fi, dict) else getattr(fi, k, None)
                if v not in (None, "None"):
                    v = float(v)
                    if not math.isnan(v) and v > 0:
                        return v
    except Exception:
        pass
    # 3) get_shares_full (best historical source)
    try:
        df = t.get_shares_full(start="2015-01-01")
        if isinstance(df, pd.DataFrame) and not df.empty:
            s = df["SharesOutstanding"].dropna()
            if not s.empty:
                val = float(s.iloc[-1])
                if val > 0:
                    return val
    except Exception:
        pass
    # 4) As a last resort, use latest "Diluted/Basic Average Shares" from IS (approx)
    try:
        if q_is is not None and not q_is.empty:
            label = _coalesce_key(
                q_is.index,
                "Diluted Average Shares",
                "Weighted Average Shares Diluted",
                "Weighted Average Shs Out Dil",
                "Weighted Average Shares",
                "Basic Average Shares",
                "Weighted Average Shs Out"
            )
            if label:
                s = q_is.loc[label].dropna().astype(float)
                if not s.empty:
                    val = float(s.iloc[0])
                    if val > 0:
                        return val
    except Exception:
        pass
    return float("nan")

def _market_cap_robust(t: yf.Ticker, q_is: pd.DataFrame) -> float:
    # A) fast_info.market_cap
    try:
        fi = getattr(t, "fast_info", None)
        if fi:
            mc = fi["market_cap"] if isinstance(fi, dict) else getattr(fi, "market_cap", None)
            if mc not in (None, "None"):
                mc = float(mc)
                if not math.isnan(mc) and mc > 0:
                    return mc
    except Exception:
        pass
    # B) info.marketCap
    try:
        info = getattr(t, "info", {}) or {}
        mc = info.get("marketCap", None)
        if mc not in (None, "None"):
            mc = float(mc)
            if not math.isnan(mc) and mc > 0:
                return mc
    except Exception:
        pass
    # C) price * shares (robust)
    px = _recent_close(t)
    sh = _shares_outstanding_robust(t, q_is)
    try:
        mc = float(px) * float(sh)
        if mc > 0 and not math.isnan(mc):
            return mc
    except Exception:
        pass
    return float("nan")
# -------------- end helpers --------------

def fetch_core_financials(ticker: str) -> Dict[str, float]:
    t = yf.Ticker(ticker)

    # statements
    q_is = _get_df(t, ["quarterly_income_stmt", "quarterly_income_statement", "quarterly_income", "quarterly_financials"])
    q_bs = _get_df(t, ["quarterly_balance_sheet", "quarterly_balancesheet", "quarterly_balance", "quarterly_balance_sheet"])
    q_cf = _get_df(t, ["quarterly_cashflow", "quarterly_cash_flow", "quarterly_cashflow_statement"])

    # market cap (now very robust)
    mc = _market_cap_robust(t, q_is)

    # Income statement (TTM)
    revenue_ttm = _ttm_sum(q_is, "Total Revenue", "Revenue", "TotalRevenue")
    ebit_ttm    = _ttm_sum(q_is, "EBIT", "Ebit", "Operating Income", "OperatingIncome")
    dep_ttm     = _ttm_sum(q_is, "Depreciation And Amortization", "Depreciation", "Depreciation & Amortization", "Reconciled Depreciation")
    ebitda_ttm  = _ttm_sum(q_is, "EBITDA", "Ebitda", "EBITDA (ttm)")

    if math.isnan(ebit_ttm) and not math.isnan(ebitda_ttm) and not math.isnan(dep_ttm):
        ebit_ttm = float(ebitda_ttm - dep_ttm)
    if math.isnan(ebitda_ttm) and not math.isnan(ebit_ttm) and not math.isnan(dep_ttm):
        ebitda_ttm = float(ebit_ttm + dep_ttm)

    # Cash flow (TTM) – broaden CapEx keys so Apple et al work
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
        # CapEx is usually negative -> CFO - CapEx
        fcf_ttm = float(cfo_ttm - capex_ttm)

    # Balance sheet (MRQ)
    total_debt = _mrq_value(q_bs, "Total Debt", "TotalDebt")
    if math.isnan(total_debt):
        lt = _mrq_value(q_bs, "Long Term Debt", "LongTermDebt", "Long Term Debt Noncurrent")
        st = _mrq_value(q_bs, "Short Long Term Debt", "ShortTermDebt", "Current Debt", "Current Portion Of Long Term Debt")
        total_debt = (0.0 if math.isnan(lt) else lt) + (0.0 if math.isnan(st) else st)

    cash   = _mrq_value(q_bs, "Cash And Cash Equivalents", "Cash", "CashAndCashEquivalents")
    equity = _mrq_value(q_bs, "Total Stockholder Equity", "Stockholders Equity", "Total Equity Gross Minority Interest", "totalStockholdersEquity")

    cur_assets       = _mrq_value(q_bs, "Total Current Assets", "Current Assets", "CurrentAssets")
    cur_liabilities  = _mrq_value(q_bs, "Total Current Liabilities", "Current Liabilities", "CurrentLiabilities")
    short_term_debt  = _mrq_value(q_bs, "Short Long Term Debt", "ShortTermDebt", "Current Debt")
    net_ppe          = _mrq_value(q_bs, "Property Plant Equipment Net", "Net PPE", "PPENet", "NetPropertyPlantAndEquipment")

    # Non-cash NWC
    ncwc = float("nan")
    if not math.isnan(cur_assets) and not math.isnan(cur_liabilities):
        cash0 = 0.0 if math.isnan(cash) else cash
        std0  = 0.0 if math.isnan(short_term_debt) else short_term_debt
        ncwc  = (cur_assets - cash0) - (cur_liabilities - std0)

    # EV (treat missing debt/cash as 0 so EV still shows when MC exists)
    ev = float("nan")
    if not math.isnan(mc):
        ev = float(mc + (0.0 if math.isnan(total_debt) else total_debt) - (0.0 if math.isnan(cash) else cash))

    # Tax rate estimate
    pretax = _ttm_sum(q_is, "Income Before Tax", "Pretax Income", "Income Before Income Taxes")
    tax    = _ttm_sum(q_is, "Income Tax Expense", "Provision For Income Taxes")
    tax_rate_est = float("nan")
    try:
        base = abs(float(pretax))
        if base > 0 and not math.isnan(tax):
            tax_rate_est = max(0.0, min(0.35, float(tax) / base))
    except Exception:
        pass

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
    ev, ebitda, ebit, sales = (fin.get(k, float("nan")) for k in ("EV","EBITDA_TTM","EBIT_TTM","Revenue_TTM"))
    mc, fcf = (fin.get(k, float("nan")) for k in ("Market_Cap","FCF_TTM"))
    return {
        "EV/EBITDA": _safe_div(ev, ebitda),
        "EV/EBIT":   _safe_div(ev, ebit),
        "EV/Sales":  _safe_div(ev, sales),
        "EBIT/EV":   _safe_div(ebit, ev),
        "FCF Yield (to Equity)": _safe_div(fcf, mc),
        "P/FCF":     _safe_div(mc, fcf),
    }

def compute_roic_variants(fin: Dict[str, float]) -> Dict[str, float]:
    ebit = fin.get("EBIT_TTM", float("nan"))
    tr   = fin.get("Tax_Rate_est", float("nan"))
    if math.isnan(tr): tr = 0.21
    nopat = float("nan") if math.isnan(ebit) else float(ebit) * (1 - float(tr))

    debt, eq, cash = (fin.get(k, float("nan")) for k in ("Debt_MRQ","Equity_MRQ","Cash_MRQ"))
    invested_capital = float("nan")
    if not math.isnan(debt) or not math.isnan(eq) or not math.isnan(cash):
        invested_capital = (0.0 if math.isnan(debt) else debt) + (0.0 if math.isnan(eq) else eq) - (0.0 if math.isnan(cash) else cash)

    nwc, net_ppe = (fin.get(k, float("nan")) for k in ("NWC_MRQ","NetPPE_MRQ"))
    magic_cap = float("nan")
    if not math.isnan(nwc) or not math.isnan(net_ppe):
        magic_cap = (0.0 if math.isnan(nwc) else nwc) + (0.0 if math.isnan(net_ppe) else net_ppe)

    return {"ROIC": _safe_div(nopat, invested_capital), "ROC_Greenblatt": _safe_div(ebit, magic_cap)}
