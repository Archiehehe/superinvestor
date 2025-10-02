# lenses.py
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np


def _is_nan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)


def burry_lens(multiples: Dict[str, float]) -> Dict:
    """Michael Burry style: cheapness via EV/EBITDA, EV/EBIT; sanity with FCF yield."""
    ev_ebitda = multiples.get("EV/EBITDA", float("nan"))
    ev_ebit = multiples.get("EV/EBIT", float("nan"))
    fcf_yield = multiples.get("FCF Yield (to Equity)", float("nan"))

    notes: List[str] = []
    if not _is_nan(ev_ebitda):
        if ev_ebitda < 8:
            notes.append("EV/EBITDA < 8 → potentially cheap (Burry screen).")
        else:
            notes.append("EV/EBITDA ≥ 8 → not obviously cheap by Burry screen.")
    if not _is_nan(ev_ebit):
        if ev_ebit < 10:
            notes.append("EV/EBIT < 10 → supportive.")
        else:
            notes.append("EV/EBIT ≥ 10 → caution.")
    if not _is_nan(fcf_yield):
        if fcf_yield > 0.08:
            notes.append("FCF Yield > 8% → attractive cash yield.")
        elif fcf_yield > 0.0:
            notes.append("Positive FCF yield, but < 8%.")
        else:
            notes.append("Negative/zero FCF yield.")

    passes = sum([
        (not _is_nan(ev_ebitda) and ev_ebitda < 8),
        (not _is_nan(ev_ebit) and ev_ebit < 10),
        (not _is_nan(fcf_yield) and fcf_yield > 0.08),
    ])
    verdict = "Strong" if passes >= 2 else ("Mixed" if passes == 1 else "Weak")

    return {
        "Lens": "Burry (EV/EBITDA)",
        "Key Metrics": {
            "EV/EBITDA": ev_ebitda,
            "EV/EBIT": ev_ebit,
            "FCF Yield": fcf_yield,
        },
        "Heuristics": notes,
        "Verdict": verdict,
    }


def greenblatt_lens(multiples: Dict[str, float], returns: Dict[str, float]) -> Dict:
    """Magic Formula: Earnings yield (EBIT/EV) + ROC."""
    ey = multiples.get("EBIT/EV", float("nan"))
    roc = returns.get("ROC (Greenblatt, EBIT / (NWC + Net PPE))", float("nan"))

    notes: List[str] = []
    if not _is_nan(ey):
        notes.append(f"Earnings Yield: {ey*100:.1f}%")
    if not _is_nan(roc):
        notes.append(f"ROC: {roc*100:.1f}% (target > 20%)")

    verdict = "Weak"
    if (not _is_nan(ey) and ey > 0.08) and (not _is_nan(roc) and roc > 0.20):
        verdict = "Potentially Attractive"
    elif (not _is_nan(ey) and ey > 0.05) and (not _is_nan(roc) and roc > 0.10):
        verdict = "Mixed/OK"

    return {
        "Lens": "Greenblatt (Magic Formula)",
        "Key Metrics": {
            "Earnings Yield (EBIT/EV)": ey,
            "ROC": roc,
        },
        "Heuristics": notes,
        "Verdict": verdict,
    }


def buffett_lens(multiples: Dict[str, float], returns: Dict[str, float]) -> Dict:
    """Buffett style: FCF Yield + ability to reinvest at high ROIC."""
    fcf_yield = multiples.get("FCF Yield (to Equity)", float("nan"))
    roic = returns.get("ROIC (NOPAT / Invested Capital)", float("nan"))

    notes: List[str] = []
    if not _is_nan(fcf_yield):
        notes.append(f"FCF Yield: {fcf_yield*100:.1f}%")
    if not _is_nan(roic):
        notes.append(f"ROIC: {roic*100:.1f}%")

    if (not _is_nan(fcf_yield) and fcf_yield >= 0.05) and (not _is_nan(roic) and roic >= 0.15):
        verdict = "Quality at Reasonable Price"
    elif (not _is_nan(roic) and roic >= 0.15):
        verdict = "Great Business, Price TBD"
    else:
        verdict = "Pass / Wait"

    return {
        "Lens": "Buffett (Owner Earnings / ROIC)",
        "Key Metrics": {
            "FCF Yield": fcf_yield,
            "ROIC": roic,
        },
        "Heuristics": notes,
        "Verdict": verdict,
    }


def klarman_lens(fin: Dict[str, float], multiples: Dict[str, float]) -> Dict:
    """Klarman style: asset-based margin of safety."""
    equity = fin.get("Equity_MRQ", float("nan"))
    mc = fin.get("Market_Cap", float("nan"))
    cash = fin.get("Cash_MRQ", float("nan"))
    debt = fin.get("Debt_MRQ", float("nan"))
    nwc = fin.get("NWC_MRQ", float("nan"))

    # Price-to-book
    pb = float("nan")
    try:
        if not (math.isnan(mc) or math.isnan(equity) or equity == 0):
            pb = mc / equity
    except Exception:
        pb = float("nan")

    # NCAV (rough): NWC + Cash − Debt
    try:
        ncav_est = (0.0 if math.isnan(nwc) else nwc) + (0.0 if math.isnan(cash) else cash) - (0.0 if math.isnan(debt) else debt)
    except Exception:
        ncav_est = float("nan")

    notes: List[str] = []
    if not (isinstance(pb, float) and math.isnan(pb)):
        if pb < 1.2:
            notes.append("P/B < 1.2 → value territory.")
        else:
            notes.append("P/B ≥ 1.2 → not a classic asset bargain.")
    if not (isinstance(ncav_est, float) and math.isnan(ncav_est) or math.isnan(mc)):
        if ncav_est > mc:
            notes.append("NCAV > Market Cap → net-net territory (rare in large caps).")

    verdict = "Asset Play Possible" if (not math.isnan(pb) and pb < 1.2) else "Not an Asset Bargain"

    return {
        "Lens": "Klarman (Asset / Margin of Safety)",
        "Key Metrics": {
            "P/B (approx)": pb,
            "NCAV est.": ncav_est,
        },
        "Heuristics": notes,
        "Verdict": verdict,
    }


def einhorn_lens(multiples: Dict[str, float]) -> Dict:
    """Einhorn style: relative value via EV multiples; peer context ideal (not included)."""
    ev_ebitda = multiples.get("EV/EBITDA", float("nan"))
    ev_sales = multiples.get("EV/Sales", float("nan"))

    notes: List[str] = []
    if not _is_nan(ev_ebitda):
        notes.append(f"EV/EBITDA: {ev_ebitda:.2f} (screen < 8× often interesting).")
    if not _is_nan(ev_sales):
        notes.append(f"EV/Sales: {ev_sales:.2f} (useful for low-margin sectors).")

    verdict = "Needs Peer Context"
    if (not _is_nan(ev_ebitda) and ev_ebitda < 8):
        verdict = "Potentially Cheap (EV/EBITDA)"

    return {
        "Lens": "Einhorn (Relative Value)",
        "Key Metrics": {
            "EV/EBITDA": ev_ebitda,
            "EV/Sales": ev_sales,
        },
        "Heuristics": notes,
        "Verdict": verdict,
    }
