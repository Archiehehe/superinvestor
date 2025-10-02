# QuickStart (First Draft)

1) Create & activate a virtualenv (optional).  
2) `pip install -r requirements.txt` (ensure `streamlit`, `yfinance`, `pandas`, `reportlab` are present).  
3) Place these three files next to `requirements.txt`:
   - `app.py`
   - `metrics.py`
   - `lenses.py`
4) Run: `streamlit run app.py`

**Smoke test tickers:** `AAPL`, `MSFT`, `TSLA`, `META`, `BRK-B`, `XOM`.  
**What you'll see:** Core Metrics tables, Multiples/Yields, Returns, and a Lens Verdict JSON block.  
If the PDF export button appears, click it to download a one‑pager summary.

**Troubleshooting**
- If yfinance returns empty quarterlies, try again (API is flaky).  
- If EV/EBITDA shows blank, that usually means EBITDA TTM or EV is `NaN` (missing inputs).  
- If Streamlit fails to start, verify Python ≥3.10 and reinstall: `pip install --upgrade pip` then `pip install -r requirements.txt`.

This is an idea filter only—numbers are TTM from last 4 quarters and simple MRQ balances.
