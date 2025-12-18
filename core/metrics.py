from typing import Any, Dict

import numpy as np
import pandas as pd


def _safe_div(a: Any, b: Any) -> float:
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
    """

    info: Dict[str, Any] = raw.get("info", {}) or {}
    history: pd.DataFrame = raw.get("history") or pd.DataFrame()

    # --- Basic price / size ---
    price = info.get("currentPrice")
    if price is None and not history.empty:
        price = float(history["Close"].iloc[-1])

    market_cap = info.get("marketCap")
    shares_out = info.get("sharesOutstanding")
    if market_cap is None and price is not None and shares_out:
        market_cap = price * shares_out

    enterprise_value = info.get("enterpriseValue")
    ebitda = info.get("ebitda")
    total_revenue = info.get("totalRevenue")
    net_income = info.get("netIncomeToCommon") or info.get("netIncome") or info.get(
        "profit"
    )
    free_cash_flow = info.get("freeCashflow")

    # --- Valuation metrics ---
    pe = _safe_div(market_cap, net_income) if net_income else float("nan")
    pb = info.get("priceToBook", float("nan"))

    ev_ebitda = _safe_div(enterprise_value, ebitda) if ebitda else float("nan")
    ev_sales = _safe_div(enterprise_value, total_revenue) if total_revenue else float(
        "nan"
    )

    earnings_yield = (
        _safe_div(net_income, enterprise_value)
        if (net_income and enterprise_value)
        else float("nan")
    )
    fcf_yield = (
        _safe_div(free_cash_flow, market_cap)
        if (free_cash_flow and market_cap)
        else float("nan")
    )

    # --- Quality metrics ---
    roe = info.get("returnOnEquity", float("nan"))  # ratio, e.g. 0.15 = 15%
    roa = info.get("returnOnAssets", float("nan"))
    gross_margin = info.get("grossMargins", float("nan"))
    op_margin = info.get("operatingMargins", float("nan"))
    net_margin = info.get("profitMargins", float("nan"))

    # FCF / Net income "cash conversion"
    fcf_conversion = (
        _safe_div(free_cash_flow, net_income)
        if (free_cash_flow and net_income)
        else float("nan")
    )

    # --- Growth metrics (YoY, approximated) ---
    rev_growth = info.get("revenueGrowth", float("nan"))  # e.g. 0.08 = 8%
    earnings_growth = info.get("earningsGrowth", float("nan"))

    # For a rough PEG, use earnings growth if available, else revenue
    growth_for_peg = earnings_growth if not np.isnan(earnings_growth) else rev_growth
    peg_ratio = float("nan")
    if not np.isnan(pe) and not np.isnan(growth_for_peg) and growth_for_peg > 0:
        # growth_for_peg is in decimals (0.10 = 10%)
        peg_ratio = pe / (growth_for_peg * 100.0)

    # --- Balance sheet / leverage metrics ---
    debt_to_equity = info.get("debtToEquity", float("nan"))  # often % style
    if not np.isnan(debt_to_equity) and debt_to_equity > 10:
        debt_to_equity = debt_to_equity / 100.0

    current_ratio = info.get("currentRatio", float("nan"))
    quick_ratio = info.get("quickRatio", float("nan"))
    interest_cover = info.get("interestCoverage", float("nan"))

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
            "pe": pe,
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
    }

    return metrics
