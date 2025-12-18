from typing import Any, Dict

import numpy as np
import pandas as pd


def _safe_div(a: Any, b: Any) -> float:
    """Safe division that returns NaN instead of blowing up."""
    try:
        if b in (0, None) or (isinstance(b, float) and np.isnan(b)):
            return float("nan")
        return float(a) / float(b)
    except Exception:
        return float("nan")


def compute_metrics(ticker: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Turn raw yfinance data into a standardized metrics dictionary.

    This is the ONE place where we derive:
      - valuation metrics
      - quality metrics
      - growth metrics
      - balance sheet metrics
      - dividend metrics
    """

    # ----- Raw pieces -----
    info: Dict[str, Any] = raw.get("info") or {}
    history = raw.get("history")
    if history is None:
        history = pd.DataFrame()
    # history is always a DataFrame (maybe empty)

    # ----- Basic price / size -----
    price = info.get("currentPrice")
    if price is None and not history.empty and "Close" in history.columns:
        try:
            price = float(history["Close"].iloc[-1])
        except Exception:
            price = None

    market_cap = info.get("marketCap")
    shares_out = info.get("sharesOutstanding")
    if market_cap is None and price is not None and shares_out:
        market_cap = price * shares_out

    enterprise_value = info.get("enterpriseValue")

    # Try to use direct PE first, then fall back to our own calc
    pe = info.get("trailingPE")
    if pe is None:
        net_income = (
            info.get("netIncomeToCommon")
            or info.get("netIncome")
            or info.get("profit")
        )
        if net_income:
            pe = _safe_div(market_cap, net_income)
    if pe is None:
        pe = float("nan")

    pb = info.get("priceToBook", float("nan"))

    ebitda = info.get("ebitda")
    total_revenue = info.get("totalRevenue")
    free_cash_flow = info.get("freeCashflow")

    ev_ebitda = _safe_div(enterprise_value, ebitda) if ebitda else float("nan")
    ev_sales = _safe_div(enterprise_value, total_revenue) if total_revenue else float(
        "nan"
    )

    # Earnings yield: prefer 1 / P/E if we have it
    if pe and not np.isnan(pe) and pe > 0:
        earnings_yield = 1.0 / float(pe)
    else:
        earnings_yield = float("nan")

    fcf_yield = (
        _safe_div(free_cash_flow, market_cap)
        if (free_cash_flow and market_cap)
        else float("nan")
    )

    # ----- Quality metrics -----
    roe = info.get("returnOnEquity", float("nan"))  # ratio, e.g. 0.15 = 15%
    roa = info.get("returnOnAssets", float("nan"))
    gross_margin = info.get("grossMargins", float("nan"))
    op_margin = info.get("operatingMargins", float("nan"))
    net_margin = info.get("profitMargins", float("nan"))

    net_income = (
        info.get("netIncomeToCommon")
        or info.get("netIncome")
        or info.get("profit")
    )

    fcf_conversion = (
        _safe_div(free_cash_flow, net_income)
        if (free_cash_flow and net_income)
        else float("nan")
    )

    # ----- Growth metrics (YoY-ish, from info) -----
    rev_growth = info.get("revenueGrowth", float("nan"))  # e.g. 0.08 = 8%
    earnings_growth = info.get("earningsGrowth", float("nan"))

    # For a rough PEG, use earnings growth if available, else revenue
    growth_for_peg = earnings_growth if not np.isnan(earnings_growth) else rev_growth
    peg_ratio = float("nan")
    if not np.isnan(pe) and not np.isnan(growth_for_peg) and growth_for_peg > 0:
        # growth_for_peg is in decimals (0.10 = 10%)
        peg_ratio = pe / (growth_for_peg * 100.0)

    # ----- Balance sheet / leverage metrics -----
    debt_to_equity = info.get("debtToEquity", float("nan"))  # often % style
    # Convert to ratio if it looks like a percentage (e.g. 80 = 0.8)
    if not np.isnan(debt_to_equity) and debt_to_equity > 10:
        debt_to_equity = debt_to_equity / 100.0

    current_ratio = info.get("currentRatio", float("nan"))
    quick_ratio = info.get("quickRatio", float("nan"))
    interest_cover = info.get("interestCoverage", float("nan"))

    # ----- Dividends -----
    dividend_yield = info.get("dividendYield", float("nan"))  # e.g. 0.025 = 2.5%
    if np.isnan(dividend_yield):
        # sometimes this one is filled instead
        trailing_yield = info.get("trailingAnnualDividendYield", float("nan"))
        if not np.isnan(trailing_yield):
            dividend_yield = trailing_yield
    payout_ratio = info.get("payoutRatio", float("nan"))  # e.g. 0.4 = 40%

    # ----- Pack into a nested metrics dict -----
    metrics: Dict[str, Any] = {
        "meta": {
            "ticker": ticker.upper(),
            "price": price,
            "market_cap": market_cap,
            "enterprise_value": enterprise_value,
            "currency": info.get("currency"),
            "short_name": info.get("shortName") or info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
        },
        "valuation": {
            "pe": float(pe),
            "pb": pb,
            "ev_ebitda": ev_ebitda,
            "ev_sales": ev_sales,
            "earnings_yield": earnings_yield,
            "fcf_yield": fcf_yield,
            "peg": peg_ratio,
        },
        "quality": {
            "roe": roe,
            "roa": roa,
            "gross_margin": gross_margin,
            "op_margin": op_margin,
            "net_margin": net_margin,
            "fcf_conversion": fcf_conversion,
        },
        "growth": {
            "revenue_growth": rev_growth,
            "earnings_growth": earnings_growth,
        },
        "balance_sheet": {
            "debt_to_equity": debt_to_equity,
            "current_ratio": current_ratio,
            "quick_ratio": quick_ratio,
            "interest_coverage": interest_cover,
        },
        "dividends": {
            "dividend_yield": dividend_yield,
            "payout_ratio": payout_ratio,
        },
    }

    return metrics
