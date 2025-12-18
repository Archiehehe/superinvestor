from dataclasses import dataclass
from typing import Any, Callable, Dict, List
import math

import numpy as np


@dataclass
class InvestorProfile:
    key: str
    name: str
    label: str
    category: str
    description: str
    score_fn: Callable[[Dict[str, Any]], Dict[str, Any]]


def _get(metrics: Dict[str, Any], section: str, key: str) -> float:
    val = metrics.get(section, {}).get(key, float("nan"))
    try:
        return float(val)
    except Exception:
        return float("nan")


def _is_nan(x: Any) -> bool:
    try:
        return x is None or (isinstance(x, float) and math.isnan(x))
    except Exception:
        return False


# ---------- Graham (Deep Value) ----------


def graham_score(metrics: Dict[str, Any]) -> Dict[str, Any]:
    pe = _get(metrics, "valuation", "pe")
    pb = _get(metrics, "valuation", "pb")
    debt_to_equity = _get(metrics, "balance_sheet", "debt_to_equity")
    current_ratio = _get(metrics, "balance_sheet", "current_ratio")

    notes: List[str] = []
    score = 0.0

    if not _is_nan(pe):
        notes.append(f"P/E ≈ {pe:,.1f}")
        if pe <= 15:
            score += 30
            notes.append("P/E ≤ 15 ✅ (Graham-friendly)")
        else:
            notes.append("P/E > 15 ❌ (less Graham-style)")
    else:
        notes.append("P/E not available")

    if not _is_nan(pb):
        notes.append(f"P/B ≈ {pb:,.1f}")
        if pb <= 1.5:
            score += 30
            notes.append("P/B ≤ 1.5 ✅ (asset value support)")
        else:
            notes.append("P/B > 1.5 ❌")
    else:
        notes.append("P/B not available")

    if not _is_nan(pe) and not _is_nan(pb):
        prod = pe * pb
        notes.append(f"P/E × P/B ≈ {prod:,.1f}")
        if prod <= 22.5:
            score += 20
            notes.append("P/E × P/B ≤ 22.5 ✅ (classic Graham condition)")
        else:
            notes.append("P/E × P/B > 22.5 ❌")

    if not _is_nan(debt_to_equity):
        notes.append(f"Debt/Equity ≈ {debt_to_equity:,.2f}")
        if debt_to_equity <= 0.5:
            score += 10
            notes.append("Debt/Equity ≤ 0.5 ✅ (conservative)")
        elif debt_to_equity <= 1.0:
            score += 5
            notes.append("Debt/Equity ≤ 1.0 ⚠️ (okay-ish)")
        else:
            notes.append("Debt/Equity > 1.0 ❌")
    else:
        notes.append("Debt/Equity not available")

    if not _is_nan(current_ratio):
        notes.append(f"Current Ratio ≈ {current_ratio:,.2f}")
        if current_ratio >= 2.0:
            score += 10
            notes.append("Current ratio ≥ 2.0 ✅ (liquidity buffer)")
        elif current_ratio >= 1.5:
            score += 5
            notes.append("Current ratio ≥ 1.5 ⚠️")
        else:
            notes.append("Current ratio < 1.5 ❌")
    else:
        notes.append("Current ratio not available")

    verdict = "Speculative / Not classic Graham value"
    if score >= 70:
        verdict = "Classic Graham-style value candidate"
    elif score >= 50:
        verdict = "Value-leaning, partially Graham-friendly"

    return {
        "score": round(score, 1),
        "verdict": verdict,
        "notes": notes,
        "subscores": {
            "valuation": min(score, 80),
            "balance_sheet": max(0.0, score - 80),
        },
    }


# ---------- Buffett (Quality at a Fair Price) ----------


def buffett_score(metrics: Dict[str, Any]) -> Dict[str, Any]:
    roe = _get(metrics, "quality", "roe")
    gross_margin = _get(metrics, "quality", "gross_margin")
    op_margin = _get(metrics, "quality", "op_margin")
    fcf_conv = _get(metrics, "quality", "fcf_conversion")
    debt_to_equity = _get(metrics, "balance_sheet", "debt_to_equity")
    pe = _get(metrics, "valuation", "pe")

    notes: List[str] = []
    score = 0.0

    if not _is_nan(roe):
        notes.append(f"ROE ≈ {roe*100:,.1f}%")
        if roe >= 0.20:
            score += 25
            notes.append("ROE ≥ 20% ✅ (excellent business)")
        elif roe >= 0.15:
            score += 18
            notes.append("ROE ≥ 15% ✅ (good business)")
        elif roe >= 0.10:
            score += 10
            notes.append("ROE ≥ 10% ⚠️ (okay)")
        else:
            notes.append("ROE < 10% ❌")
    else:
        notes.append("ROE not available")

    if not _is_nan(gross_margin):
        notes.append(f"Gross margin ≈ {gross_margin*100:,.1f}%")
        if gross_margin >= 0.4:
            score += 10
            notes.append("Gross margin ≥ 40% ✅ (pricing power)")
    else:
        notes.append("Gross margin not available")

    if not _is_nan(op_margin):
        notes.append(f"Operating margin ≈ {op_margin*100:,.1f}%")
        if op_margin >= 0.20:
            score += 10
            notes.append("Operating margin ≥ 20% ✅")
    else:
        notes.append("Operating margin not available")

    if not _is_nan(fcf_conv):
        notes.append(f"FCF / Net income ≈ {fcf_conv*100:,.1f}%")
        if 0.8 <= fcf_conv <= 1.2:
            score += 10
            notes.append("Strong cash conversion ✅")
    else:
        notes.append("FCF / Net income not available")

    if not _is_nan(debt_to_equity):
        notes.append(f"Debt/Equity ≈ {debt_to_equity:,.2f}")
        if debt_to_equity <= 0.5:
            score += 15
            notes.append("Conservative balance sheet ✅")
        elif debt_to_equity <= 1.0:
            score += 8
            notes.append("Moderate leverage ⚠️")
        else:
            notes.append("High leverage ❌")
    else:
        notes.append("Debt/Equity not available")

    if not _is_nan(pe):
        notes.append(f"P/E ≈ {pe:,.1f}")
        if pe <= 20:
            score += 10
            notes.append("Price not extreme for a quality business ✅")
        elif pe <= 30:
            score += 5
            notes.append("Price somewhat rich ⚠️")
        else:
            notes.append("Very expensive relative to earnings ❌")
    else:
        notes.append("P/E not available")

    verdict = "Not obviously a Buffett-style compounder"
    if score >= 70:
        verdict = "Buffett-style high quality at reasonable price"
    elif score >= 50:
        verdict = "Decent quality, maybe watchlist material"

    return {
        "score": round(score, 1),
        "verdict": verdict,
        "notes": notes,
        "subscores": {},
    }


# ---------- Lynch (GARP / PEG) ----------


def lynch_score(metrics: Dict[str, Any]) -> Dict[str, Any]:
    pe = _get(metrics, "valuation", "pe")
    peg = _get(metrics, "valuation", "peg")
    rev_g = _get(metrics, "growth", "revenue_growth")
    earn_g = _get(metrics, "growth", "earnings_growth")
    debt_to_equity = _get(metrics, "balance_sheet", "debt_to_equity")

    notes: List[str] = []
    score = 0.0

    growth_display = earn_g if not _is_nan(earn_g) else rev_g
    if not _is_nan(growth_display):
        notes.append(f"Growth ≈ {growth_display*100:,.1f}%")
        if growth_display >= 0.15:
            score += 25
            notes.append("Strong growth ≥ 15% ✅")
        elif growth_display >= 0.10:
            score += 18
            notes.append("Solid growth ≥ 10% ✅")
        elif growth_display >= 0.05:
            score += 10
            notes.append("Mild growth ≥ 5% ⚠️")
        else:
            notes.append("Low growth ❌")
    else:
        notes.append("Growth not available")

    if not _is_nan(pe):
        notes.append(f"P/E ≈ {pe:,.1f}")
    else:
        notes.append("P/E not available")

    if not _is_nan(peg):
        notes.append(f"PEG ≈ {peg:,.2f}")
        if peg <= 1.0:
            score += 25
            notes.append("PEG ≤ 1.0 ✅ (classic Lynch)")
        elif peg <= 1.5:
            score += 15
            notes.append("PEG ≤ 1.5 ⚠️ (maybe okay)")
        else:
            notes.append("PEG > 1.5 ❌")
    else:
        notes.append("PEG not available")

    if not _is_nan(debt_to_equity):
        notes.append(f"Debt/Equity ≈ {debt_to_equity:,.2f}")
        if debt_to_equity <= 0.5:
            score += 10
            notes.append("Low leverage ✅")
        elif debt_to_equity <= 1.0:
            score += 5
            notes.append("Moderate leverage ⚠️")
        else:
            notes.append("High leverage ❌")
    else:
        notes.append("Debt/Equity not available")

    verdict = "Not obviously a Lynch-style GARP stock"
    if score >= 60:
        verdict = "Lynch-style GARP candidate (growth at reasonable price)"
    elif score >= 40:
        verdict = "Partially GARP-friendly, but not ideal"

    return {
        "score": round(score, 1),
        "verdict": verdict,
        "notes": notes,
        "subscores": {},
    }


# ---------- Greenblatt (Magic Formula) ----------


def greenblatt_score(metrics: Dict[str, Any]) -> Dict[str, Any]:
    earnings_yield = _get(metrics, "valuation", "earnings_yield")
    ev_ebitda = _get(metrics, "valuation", "ev_ebitda")
    roe = _get(metrics, "quality", "roe")

    notes: List[str] = []
    score = 0.0

    if not _is_nan(earnings_yield):
        ey_pct = earnings_yield * 100
        notes.append(f"Earnings yield ≈ {ey_pct:,.1f}%")
        if ey_pct >= 15:
            score += 30
            notes.append("Earnings yield ≥ 15% ✅ (very cheap)")
        elif ey_pct >= 8:
            score += 20
            notes.append("Earnings yield ≥ 8% ✅ (cheap-ish)")
        else:
            notes.append("Earnings yield < 8% ❌")
    elif not _is_nan(ev_ebitda):
        notes.append(f"EV/EBITDA ≈ {ev_ebitda:,.1f}")
        if ev_ebitda <= 8:
            score += 20
            notes.append("EV/EBITDA ≤ 8 ✅ (cheap-ish)")
        else:
            notes.append("EV/EBITDA > 8 ❌")
    else:
        notes.append("Earnings yield / EV multiples not available")

    if not _is_nan(roe):
        notes.append(f"ROE ≈ {roe*100:,.1f}%")
        if roe >= 0.20:
            score += 30
            notes.append("ROE ≥ 20% ✅ (high return on capital)")
        elif roe >= 0.15:
            score += 20
            notes.append("ROE ≥ 15% ✅ (solid)")
        else:
            notes.append("ROE < 15% ❌")
    else:
        notes.append("ROE (ROC proxy) not available")

    verdict = "Not clearly a Magic Formula standout"
    if score >= 60:
        verdict = "Magic-Formula style attractive (high EY & ROC)"
    elif score >= 40:
        verdict = "Partially Magic-Formula style, but not top tier"

    return {
        "score": round(score, 1),
        "verdict": verdict,
        "notes": notes,
        "subscores": {},
    }


# ---------- Burry (Deep Value + FCF) ----------


def burry_score(metrics: Dict[str, Any]) -> Dict[str, Any]:
    fcf_yield = _get(metrics, "valuation", "fcf_yield")
    ev_ebitda = _get(metrics, "valuation", "ev_ebitda")
    pe = _get(metrics, "valuation", "pe")
    debt_to_equity = _get(metrics, "balance_sheet", "debt_to_equity")

    notes: List[str] = []
    score = 0.0

    if not _is_nan(fcf_yield):
        fcf_yield_pct = fcf_yield * 100
        notes.append(f"FCF yield ≈ {fcf_yield_pct:,.1f}%")
        if fcf_yield_pct >= 10:
            score += 35
            notes.append("FCF yield ≥ 10% ✅ (very cheap cash flow)")
        elif fcf_yield_pct >= 6:
            score += 25
            notes.append("FCF yield ≥ 6% ✅ (cheap)")
        else:
            notes.append("FCF yield < 6% ❌")
    else:
        notes.append("FCF yield not available")

    if not _is_nan(ev_ebitda):
        notes.append(f"EV/EBITDA ≈ {ev_ebitda:,.1f}")
        if ev_ebitda <= 8:
            score += 15
            notes.append("EV/EBITDA ≤ 8 ✅ (cheap-ish)")
        elif ev_ebitda <= 10:
            score += 8
            notes.append("EV/EBITDA ≤ 10 ⚠️")
        else:
            notes.append("EV/EBITDA > 10 ❌")
    elif not _is_nan(pe):
        notes.append(f"P/E ≈ {pe:,.1f}")
        if pe <= 10:
            score += 10
            notes.append("P/E ≤ 10 ✅")
    else:
        notes.append("Valuation multiples unavailable")

    if not _is_nan(debt_to_equity):
        notes.append(f"Debt/Equity ≈ {debt_to_equity:,.2f}")
        if debt_to_equity <= 0.5:
            score += 20
            notes.append("Very conservative balance sheet ✅")
        elif debt_to_equity <= 1.0:
            score += 10
            notes.append("Moderate leverage ⚠️")
        else:
            notes.append("High leverage ❌")
    else:
        notes.append("Debt/Equity not available")

    verdict = "Not a classic Burry-style deep value"
    if score >= 65:
        verdict = "Burry-style deep value candidate (cheap + decent balance sheet)"
    elif score >= 45:
        verdict = "Some Burry-style elements, but not ideal"

    return {
        "score": round(score, 1),
        "verdict": verdict,
        "notes": notes,
        "subscores": {},
    }


# ---------- Registry ----------
GRAHAM = InvestorProfile(
    key="graham",
    name="Benjamin Graham",
    label="Graham – Deep Value",
    category="Deep Value",
    description="Classic Ben Graham: low P/E, low P/B, strong balance sheet.",
    score_fn=graham_score,
)

BUFFETT = InvestorProfile(
    key="buffett",
    name="Warren Buffett",
    label="Buffett – Quality at Fair Price",
    category="Quality",
    description="High-quality, high-ROE businesses with conservative leverage at sensible valuations.",
    score_fn=buffett_score,
)

LYNCH = InvestorProfile(
    key="lynch",
    name="Peter Lynch",
    label="Lynch – GARP (PEG)",
    category="GARP",
    description="Growth at a reasonable price; focus on growth and PEG around 1.",
    score_fn=lynch_score,
)

GREENBLATT = InvestorProfile(
    key="greenblatt",
    name="Joel Greenblatt",
    label="Greenblatt – Magic Formula",
    category="Deep Value / Quality",
    description="High earnings yield and high return on capital.",
    score_fn=greenblatt_score,
)

BURRY = InvestorProfile(
    key="burry",
    name="Michael Burry",
    label="Burry – Deep FCF Value",
    category="Deep Value",
    description="Cheap on cash flows with adequate balance sheet strength.",
    score_fn=burry_score,
)

ALL_PROFILES: List[InvestorProfile] = [
    GRAHAM,
    BUFFETT,
    LYNCH,
    GREENBLATT,
    BURRY,
]


def get_profile_by_key(key: str) -> InvestorProfile:
    for p in ALL_PROFILES:
        if p.key == key:
            return p
    raise KeyError(f"Unknown profile key: {key}")
