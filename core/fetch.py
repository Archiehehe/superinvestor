from typing import Any, Dict

import pandas as pd
import yfinance as yf


def fetch_ticker_data(ticker: str) -> Dict[str, Any]:
    """
    Fetch raw data for a single ticker from Yahoo Finance via yfinance.

    Returns a dict with:
        - info: yfinance .info dict (fundamental snapshot)
        - history: price history (5y daily, auto-adjusted)
    """
    ticker = ticker.upper().strip()
    tk = yf.Ticker(ticker)

    try:
        info = tk.info or {}
    except Exception:
        info = {}

    try:
        history = tk.history(period="5y", interval="1d", auto_adjust=True)
    except Exception:
        history = pd.DataFrame()

    return {"info": info, "history": history}
