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

    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(s).lower())

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
    if qdf is None or qdf.empty:
        return float("nan")
    lab = _coalesce_key(qdf.index, *labels)
    if lab is None:
        return float("nan")
    s = qdf.loc[lab].dropna().astype(float)
    if s.empty:
        return float("nan")
    return float(s.iloc[:4].sum())  # last 4 quarters


def _mrq_value(qdf: pd.DataFrame, *labels: str) -> float:
    if qdf is None or qdf.empty:
        return float("nan")
    lab = _coalesce_key(qdf.index, *labels)
    if lab is None:
        return float("nan")
    s = qdf.loc[lab].dropna().astype(float)
    if s.empty:
        return float("nan")
    return float(s.iloc[0])  # most recent quarter


def _safe_div(n: float, d: float) -> float:
    try:
        if d == 0 or (isinstance(d, float) and math.isnan(d)):
            return float("nan")
        return float(n) / float(d)
    except Exception:
        return float("nan")


def _recent_close(t: yf.Ticker) -> float:
    # fast_info first
    for k in ("last_price", "regularMarketPrice", "regular_market_price"):
        try:
            fi = getattr(t, "fast_info", None)
            if fi:
                if isinstance(fi, dict) and k in fi and fi[k] is not None:
                    return float(fi[k])
                if hasattr(fi, k) and getattr(fi, k) is not None:
                    return float(getattr(fi, k))
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
            if so > 0 and not math.isnan(so):
                return so
    except Exception:
        pass
    # fast_info
    try:
        fi = getattr(t, "fast_info", None)
        for k in ("shares", "shares_outstanding"):
            v = (fi.get(k) if isinstance(fi, dict) else getattr(fi, k, None)) if fi else None
            if v not in (None, "None"):
                v = float(v)
                if v > 0 and not math.isnan(v):
                    return v
    except Exception:
        pass
    # get_shares_full
    try:
        df = t.get_shares_full(start="2015-01-01")
        if isinstance(df, pd.DataFrame) and not df.empty:
            s = df["SharesOutstanding"].dropna()
            if not s.empty:
                v = float(s.iloc[-1])
                if v > 0:
                    return v
    except Exception:
        pass
    # fall back to IS average shares (approx)
    try:
        if q_is is not None and not q_is.empty:
            lab = _coalesce_key(
                q_is.index,
                "Diluted Average Shares",
                "Weighted Average Shares Diluted",
                "Weighted Average Shs Out Dil",
                "Weighted Average Shares",
                "Basic Average Shares",
                "Weighted Average Shs Out",
            )
            if lab:
                s = q_is.loc[lab].dropna().astype(float)
                if not s.empty:
                    v = float(s.iloc[0])
                    if v > 0:
                        return v
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
                if mc > 0 and not math.isnan(mc):
                    return mc
    except Exception:
        pass
    # B) info.marketCap
    try:
        info = getattr(t, "info", {}) or {}
        mc = info.get("marketCap", None)
        if mc not in (None, "None"):
            mc = float(mc)
            if mc > 0 and not math.isnan(mc):
                return mc
    except Exception:
        pass
    # C) price * shares
    px = _recent_close(t)
    sh = _shares_outstanding_robust(t, q_is)
    try:
        mc = float(px) * float(sh)
        if mc > 0 and not math.isnan(mc):
            return mc
    except Exception:
        pass
    return float("nan")


def _currencies(t: yf.Ticker) -> tuple[str, str]:
    """Return (market_price_currency, financial_statement_currency)."""
    market = "USD"
    financial = "USD"
    try:
        fi = getattr(t, "fast_info", None)
        if fi:
            market = (fi.get("currency") if isinstance(fi, dict) else getattr(fi, "currency", None)) or market
    except Exception:
        pass
    try:
        info = getattr(t, "info", {}) or {}
        market = info.get("currency", market) or market
        financial = info.get("financialCurrency", financial) or financial
    except Exception:
        pass
    return market, financial


def _fx_rate(from_ccy: str, to_ccy: str) -> float:
    """Return FX to convert amounts FROM 'from_ccy' TO 'to_ccy'."""
    if not from_ccy or not to_ccy or from_ccy == to_ccy:
        return 1.0
    pair = f"{from_ccy}{to_ccy}=X"     # e.g., DKKUSD=X
    inv = f"{to_ccy}{from_ccy}=X"      # e.g., USDDKK=X
    for sym, invert in ((pair, False), (inv, True)):
        try:
            fx_t = yf.Ticker(sym)
            p = _recent_close(fx_t)
            if p and not math.isnan(p):
                return (1.0 / float(p)) if invert else float(p)
        except Exception:
            pass
    return 1.0  # last-resort; keeps app usable


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

    # market cap (robust) in marke
