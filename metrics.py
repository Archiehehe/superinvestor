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
    def norm(s: str) -> str: return re.sub(r"[^a-z0-9]+", "", str(s).lower())
    idx = [norm(x) for x in index_like]
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
    return float(s.iloc[:4].sum())  # last 4 quarters

def _mrq_value(qdf: pd.DataFrame, *labels: str) -> float:
    if qdf is None or qdf.empty: return float("nan")
    lab = _coalesce_key(qdf.index, *labels)
    if lab is None: return float("nan")
    s = qdf.loc[lab].dropna().astype(float)
    if s.empty: return float("nan")
    return float(s.iloc[0])  # most recent quarter

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
            fi = getattr(t, "fast_info", None)
            if fi:
                if isinstance(fi, dict) and k in fi and fi[k] is not None: return float(fi[k])
                if hasattr(fi, k) and getattr(fi, k) is not None: return float(getattr(fi, k))
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
    # info
    try:
        info = getattr(t, "info", {}) or {}
        so = info.get("sharesOutstanding", None)
        if so not in (None, "None"):
            so = float(so)
            if so > 0 and not math.isnan(so): return so
    except Exception: pass
    # fast_info
    try:
        fi = getattr(t, "fast_info", None)
        for k in ("shares", "shares_outstanding"):
            v = (fi.get(k) if isinstance(fi, dict) else getattr(fi, k, None)) if fi else None
            if v not in (None, "None"):
                v = float(v)
                if v > 0 and not math.isnan(v): return v
    except Exception: pass
    # get_shares_full
    try:
        df = t.get_shares_full(start="2015-01-01")
        if isinstance(df, pd.DataFrame) and not df.empty:
            s = df["SharesOutstanding"].dropna()
            if not s.empty:
                v = float(s.iloc[-1])
                if v > 0: return v
    except Exception: pass
    # fall back to IS average shares (approx)
    try:
        if q_is is not None and not q_is.empty:
            lab = _coalesce_key(
                q_is.index,
                "Diluted Average Shares", "Weighted Average Shares Diluted",
                "Weighted Average Shs Out Dil", "Weighted Average Shares",
                "Basic Average Shares", "Weighted Average Shs Out"
            )
            if lab:
                s = q_is.loc[lab].dropna().astype(float)
                if not s.empty:
                    v = float(s.iloc[0])
                    if v > 0: return v
    except Exception: pass
    return float("nan")

def _market_cap_robust(t: yf.Ticker, q_is: pd.DataFrame) -> float:
    # A) fast_info.market_cap
    try:
        fi = getattr(t, "fast_info", None)
        if fi:
            mc = fi["market_cap"] if isinstance(fi, dict) else getattr(fi, "market_cap", None)
            if mc not in (None, "None"):
                mc = float(mc)
                if mc > 0 and not math.isnan(mc): return mc
    except Exception: pass
    # B) info.marketCap
    try:
        info = getattr(t, "info", {}) or {}
        mc = info.get("marketCap", None)
        if mc not in (None, "None"):
            mc = float(mc)
            if mc > 0 and not math.isnan(mc): return mc
    except Exception: pass
    # C) price * shares
    px = _recent_close(t)
    sh = _shares_outstanding_robust(t, q_is)
    try:
        mc = float(px) * float(sh)
        if mc > 0 and not math.isnan(mc): return mc
    except Exception: pass
    return float("nan")

def _currencies(t: yf.Ticker) -> tuple[str, str]:
    """Return (market_price_currency, financial_statement_currency)."""
    market = "USD"; financial = "USD"
    try:
        fi = getattr(t, "fast_info", None)
        if fi:
            market = (fi.get("currency") if isinstance(fi, dict) else getattr(fi, "currency", None)) or market
    except Exception: pass
    try:
        info = getattr(t, "info", {}) or {}
        market = info.get("currency", market) or market
        financial = info.get("financialCurrency", financial) or financial
    except Exception: pass
    return market, financial

def _fx_rate(from_ccy: str, to_ccy: str) -> float:
    """Return FX to convert amounts FROM 'from_ccy' TO 'to_ccy'."""
    if not from_ccy or not to_ccy or from_ccy == to_ccy:
        return 1.0
    pair = f"{from_ccy}{to_ccy}=X"        # e.g., DKKUSD=X
    inv_pair = f"{to_ccy}{from_ccy}=X"    # e.g., USDDKK=X
    for sym, invert in ((pair, False), (inv_pair, True)):
        try:
            fx_t = yf.Ticker(sym)
            p = _recent_close(fx_t)
            if p and not math.isnan(p):
                return (1.0 / float(p)) if invert else float(p)
        except Exception:
            pass
    # If all else fails, assume 1 (better than crashing; UI will still show currencies)
    return 1.0

def _conv(x: float, fx: float) -> float:
    try:
        return float(x) * float(fx) if not math.isnan(float(x)) else float("nan")
    except Exception:
        return float("nan")
# -------------- end helpers --------------

def fetch_core_financials(ticker: str) -> Dict[str, float]:
    t = yf.Ticker(ticker)

    # statements
    q_is = _get_df(t, ["quarterly_income_stmt", "quarterly_income_statement", "quarterly_income", "quarterly_financials"])
    q_bs = _get_df(t, ["quarterly_balance_sheet", "quarterly_balancesheet", "quarterly_balance", "quarterly_balance_sheet"])
    q_cf = _get_df(t, ["quarterly_cashflow", "quarterly_cash_flow", "quarterly_cashflow_statement"])

    # currencies + FX (convert financials -> market currency)
    market_ccy, fin_ccy = _currencies(t)
    fx = _fx_rate(fin_ccy, market_ccy)

    # market cap (robust) in market currency
    mc = _market_cap_robust(t, q_is)

    # Income statement (TTM) in financial currency -> convert to market currency
    revenue_ttm_fc = _ttm_sum(q_is, "Total Revenue", "Revenue", "TotalRevenue")
    ebit_ttm_fc    = _ttm_sum(q_is, "EBIT", "Ebit", "Operating Income", "OperatingIncome")
    dep_ttm_fc     = _ttm_sum(q_is, "Depreciation And Amortization", "Depreciation", "Depreciation & Amortization", "Reconciled Depreciation")
    ebitda_ttm_fc  = _ttm_sum(q_is, "EBITDA", "Ebitda", "EBITDA (ttm)")

    if math.isnan(ebit_ttm_fc) and not math.isnan(ebitda_ttm_fc) and not math.isnan(dep_ttm_fc):
        ebit_ttm_fc = float(ebitda_ttm_fc - dep_ttm_fc)
    if math.isnan(ebitda_ttm_fc) and not math.isnan(ebit_ttm_fc) and not math.isnan(dep_ttm_fc):
        ebitda_ttm_fc = float(ebit_ttm_fc + dep_ttm_fc)

    # Cash flow (TTM) — broaden CapEx keys; CapEx is usually NEGATIVE -> FCF = CFO + CapEx
    cfo_ttm_fc = _ttm_sum(
        q_cf,
        "Total Cash From Operating Activities", "Operating Cash Flow",
        "OperatingCashFlow", "NetCashProvidedByOperatingActivities",
        "Net Cash Provided By Operating Activities",
    )
    capex_ttm_fc = _ttm_sum(
        q_cf,
        "Capital Expenditures", "Capital Expenditure",
        "Purchase Of Property And Equipment", "Purchases Of Property And Equipment",
        "Investment In Property Plant And Equipment", "Investments In Property Plant And Equipment",
        "Additions To Property Plant And Equipment",
    )
    # normalize capex to negative before combining
    if not math.isnan(capex_ttm_fc):
        capex_ttm_fc = -abs(capex_ttm_fc)

    # Balance sheet (MRQ) in financial currency
    total_debt_fc = _mrq_value(q_bs, "Total Debt", "TotalDebt")
    if math.isnan(total_debt_fc):
        lt = _mrq_value(q_bs, "Long Term Debt", "LongTermDebt", "Long Term Debt Noncurrent")
        st = _mrq_value(q_bs, "Short Long Term Debt", "ShortTermDebt", "Current Debt", "Current Portion Of Long Term Debt")
        total_debt_fc = (0.0 if math.isnan(lt) else lt) + (0.0 if math.isnan(st) else st)

    cash_fc   = _mrq_value(q_bs, "Cash And Cash Equivalents", "Cash", "CashAndCashEquivalents")
    equity_fc = _mrq_value(q_bs, "Total Stockholder Equity", "Stockholders Equity", "Total Equity Gross Minority Interest", "totalStockholdersEquity")

    cur_assets_fc      = _mrq_value(q_bs, "Total Current Assets", "Current Assets", "CurrentAssets")
    cur_liabilities_fc = _mrq_value(q_bs, "Total Current Liabilities", "Current Liabilities", "CurrentLiabilities")
    short_term_debt_fc = _mrq_value(q_bs, "Short Long Term Debt", "ShortTermDebt", "Current Debt")
    net_ppe_fc         = _mrq_value(q_bs, "Property Plant Equipment Net", "Net PPE", "PPENet", "NetPropertyPlantAndEquipment")

    # Convert all FC amounts into MARKET currency
    revenue_ttm = _conv(revenue_ttm_fc, fx)
    ebit_ttm    = _conv(ebit_ttm_fc, fx)
    dep_ttm     = _conv(dep_ttm_fc, fx)
    ebitda_ttm  = _conv(ebitda_ttm_fc, fx)
    cfo_ttm     = _conv(cfo_ttm_fc, fx)
    capex_ttm   = _conv(capex_ttm_fc, fx)
    total_debt  = _conv(total_debt_fc, fx)
    cash        = _conv(cash_fc, fx)
    equity      = _conv(equity_fc, fx)
    cur_assets  = _conv(cur_assets_fc, fx)
    cur_liab    = _conv(cur_liabilities_fc, fx)
    std         = _conv(short_term_debt_fc, fx)
    net_ppe     = _conv(net_ppe_fc, fx)

    # FCF (market currency) — CapEx already normalized to negative
    fcf_ttm = float("nan")
    if not math.isnan(cfo_ttm) and not math.isnan(capex_ttm):
        fcf_ttm = float(cfo_ttm + capex_ttm)

    # Non-cash NWC
    ncwc = float("nan")
    if not math.isnan(cur_assets) and not math.isnan(cur_liab):
        cash0 = 0.0 if math.isnan(cash) else cash
        std0  = 0.0 if math.isnan(std) else std
        ncwc = (cur_assets - cash0) - (cur_liab - std0)

    # EV — keep numbers in market currency; treat missing debt/cash as 0
    ev = float("nan")
    if not math.isnan(mc):
        ev = float(mc + (0.0 if math.isnan(total_debt) else total_debt) - (0.0 if math.isnan(cash) else cash))
    if math.isnan(ev):  # as a last resort, try info.enterpriseValue
        try:
            info = getattr(t, "info", {}) or {}
            ev_info = float(info.get("enterpriseValue", np.nan))
            if not math.isnan(ev_info): ev = ev_info
        except Exception:
            pass

    # Tax rate estimate (use FC values — rate is dimensionless)
    pretax_fc = _ttm_sum(q_is, "Income Before Tax", "Pretax Income", "Income Before Income Taxes")
    tax_fc    = _ttm_sum(q_is, "Income Tax Expense", "Provision For Income Taxes")
    tax_rate_est = float("nan")
    try:
        base = abs(float(pretax_fc))
        if base > 0 and not math.isnan(tax_fc):
            tax_rate_est = max(0.0, min(0.35, float(tax_fc) / base))
    except Exception:
        pass

    return {
        # currency metadata (for UI)
        "Currency_Market": market_ccy,
        "Currency_Financial": fin_ccy,
        "FX_fin_to_market": fx,

        # valuation base
        "Market_Cap": mc,
        "EV": ev,

        # TTM (market currency)
        "Revenue_TTM": revenue_ttm,
        "EBIT_TTM": ebit_ttm,
        "Dep_TTM": dep_ttm,
        "EBITDA_TTM": ebitda_ttm,
        "CFO_TTM": cfo_ttm,
        "CapEx_TTM": capex_ttm,
        "FCF_TTM": fcf_ttm,

        # MRQ (market currency)
        "Debt_MRQ": total_debt,
        "Cash_MRQ": cash,
        "Equity_MRQ": equity,
        "NWC_MRQ": ncwc,
        "NetPPE_MRQ": net_ppe,

        # misc
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
    invested
