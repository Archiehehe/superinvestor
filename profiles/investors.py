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
    rules_fn: Callable[[Dict[str, Any]], Dict[str, Any]]  # returns summary & rules


# ---------- helpers ----------


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


def _fmt_pct(x: Any) -> str:
    try:
        if _is_nan(x):
            return "—"
        return f"{float(x) * 100:,.1f}%"
    except Exception:
        return "—"


def _fmt_num(x: Any) -> str:
    try:
        if _is_nan(x):
            return "—"
        return f"{float(x):,.2f}"
    except Exception:
        return "—"


def _rule(
    name: str,
    condition: str,
    value: Any,
    status: str,
    comment: str = "",
    as_pct: bool = False,
) -> Dict[str, Any]:
    return {
        "name": name,
        "condition": condition,
        "value": _fmt_pct(value) if as_pct else _fmt_num(value),
        "status": status,  # "pass", "warn", "fail", "na"
        "comment": comment,
    }


def _summary_from_rules(rules: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    passes = sum(1 for r in rules if r["status"] == "pass")
    warns = sum(1 for r in rules if r["status"] == "warn")
    fails = sum(1 for r in rules if r["status"] == "fail")

    headline = ""
    if fails == 0 and passes >= 4:
        headline = f"Very {label}-friendly profile."
    elif passes >= fails:
        headline = f"Mixed but somewhat {label}-compatible."
    else:
        headline = f"Not a classic {label}-style candidate."

    return {
        "passes": passes,
        "warns": warns,
        "fails": fails,
        "headline": headline,
    }


# ---------- Graham (Deep Value) ----------


def graham_rules(metrics: Dict[str, Any]) -> Dict[str, Any]:
    pe = _get(metrics, "valuation", "pe")
    pb = _get(metrics, "valuation", "pb")
    debt_to_equity = _get(metrics, "balance_sheet", "debt_to_equity")
    current_ratio = _get(metrics, "balance_sheet", "current_ratio")

    rules: List[Dict[str, Any]] = []

    # Rule 1: P/E ≤ 15
    if _is_nan(pe):
        rules.append(
            _rule(
                "P/E multiple",
                "P/E ≤ 15",
                pe,
                "na",
                "P/E not available from Yahoo Finance.",
            )
        )
    else:
        status = "pass" if pe <= 15 else "fail"
        comment = (
            "Classic Graham low multiple."
            if status == "pass"
            else "Above the classic Graham threshold."
        )
        rules.append(_rule("P/E multiple", "P/E ≤ 15", pe, status, comment))

    # Rule 2: P/B ≤ 1.5
    if _is_nan(pb):
        rules.append(
            _rule(
                "Price to book",
                "P/B ≤ 1.5",
                pb,
                "na",
                "Book value data missing.",
            )
        )
    else:
        status = "pass" if pb <= 1.5 else "fail"
        comment = (
            "Discount or near-discount to book."
            if status == "pass"
            else "Above classic Graham P/B."
        )
        rules.append(_rule("Price to book", "P/B ≤ 1.5", pb, status, comment))

    # Rule 3: P/E × P/B ≤ 22.5 (famous Graham product)
    if _is_nan(pe) or _is_nan(pb):
        rules.append(
            _rule(
                "Graham product",
                "P/E × P/B ≤ 22.5",
                float("nan"),
                "na",
                "Need both P/E and P/B to check this.",
            )
        )
    else:
        prod = pe * pb
        status = "pass" if prod <= 22.5 else "fail"
        comment = (
            "Within Graham's classic combined limit."
            if status == "pass"
            else "Above Graham's combined P/E×P/B limit."
        )
        rules.append(
            _rule(
                "Graham product",
                "P/E × P/B ≤ 22.5",
                prod,
                status,
            )
        )

    # Rule 4: Debt/Equity ≤ 0.5 (≤1.0 as warning)
    if _is_nan(debt_to_equity):
        rules.append(
            _rule(
                "Leverage",
                "Debt/Equity ≤ 0.5",
                debt_to_equity,
                "na",
                "Leverage data missing.",
            )
        )
    else:
        if debt_to_equity <= 0.5:
            status, comment = "pass", "Very conservative leverage."
        elif debt_to_equity <= 1.0:
            status, comment = "warn", "Moderate leverage."
        else:
            status, comment = "fail", "High leverage for Graham style."
        rules.append(
            _rule(
                "Leverage",
                "Debt/Equity ≤ 0.5",
                debt_to_equity,
                status,
                comment,
            )
        )

    # Rule 5: Current ratio ≥ 2 (≥1.5 warning)
    if _is_nan(current_ratio):
        rules.append(
            _rule(
                "Liquidity",
                "Current ratio ≥ 2.0",
                current_ratio,
                "na",
                "Liquidity data missing.",
            )
        )
    else:
        if current_ratio >= 2.0:
            status, comment = "pass", "Strong near-term liquidity."
        elif current_ratio >= 1.5:
            status, comment = "warn", "Acceptable but not ideal."
        else:
            status, comment = "fail", "Weak current ratio for Graham."
        rules.append(
            _rule(
                "Liquidity",
                "Current ratio ≥ 2.0",
                current_ratio,
                status,
                comment,
            )
        )

    summary = _summary_from_rules(rules, "Graham")
    return {"summary": summary, "rules": rules}


# ---------- Buffett (Quality at a Fair Price) ----------


def buffett_rules(metrics: Dict[str, Any]) -> Dict[str, Any]:
    roe = _get(metrics, "quality", "roe")
    gross_margin = _get(metrics, "quality", "gross_margin")
    op_margin = _get(metrics, "quality", "op_margin")
    fcf_conv = _get(metrics, "quality", "fcf_conversion")
    debt_to_equity = _get(metrics, "balance_sheet", "debt_to_equity")
    pe = _get(metrics, "valuation", "pe")

    rules: List[Dict[str, Any]] = []

    # ROE ≥ 15%
    if _is_nan(roe):
        rules.append(
            _rule(
                "Return on equity",
                "ROE ≥ 15%",
                roe,
                "na",
                "ROE not available.",
                as_pct=True,
            )
        )
    else:
        if roe >= 0.20:
            status, comment = "pass", "Excellent long-term profitability."
        elif roe >= 0.15:
            status, comment = "pass", "Good profitability."
        elif roe >= 0.10:
            status, comment = "warn", "Okay, but not standout."
        else:
            status, comment = "fail", "Low ROE for a Buffett compounder."
        rules.append(
            _rule(
                "Return on equity",
                "ROE ≥ 15%",
                roe,
                status,
                comment,
                as_pct=True,
            )
        )

    # Gross margin ≥ 40%
    if _is_nan(gross_margin):
        rules.append(
            _rule(
                "Gross margin",
                "Gross margin ≥ 40%",
                gross_margin,
                "na",
                "Margin data missing.",
                as_pct=True,
            )
        )
    else:
        status = "pass" if gross_margin >= 0.4 else "warn"
        comment = (
            "Indicates pricing power and moat."
            if status == "pass"
            else "Not obviously a high-moat margin."
        )
        rules.append(
            _rule(
                "Gross margin",
                "Gross margin ≥ 40%",
                gross_margin,
                status,
                comment,
                as_pct=True,
            )
        )

    # Operating margin ≥ 20%
    if _is_nan(op_margin):
        rules.append(
            _rule(
                "Operating margin",
                "Operating margin ≥ 20%",
                op_margin,
                "na",
                "Operating margin missing.",
                as_pct=True,
            )
        )
    else:
        if op_margin >= 0.20:
            status, comment = "pass", "Strong operating profitability."
        elif op_margin >= 0.12:
            status, comment = "warn", "Decent but not elite."
        else:
            status, comment = "fail", "Thin operating margin."
        rules.append(
            _rule(
                "Operating margin",
                "Operating margin ≥ 20%",
                op_margin,
                status,
                comment,
                as_pct=True,
            )
        )

    # FCF / Net income between 80% and 120%
    if _is_nan(fcf_conv):
        rules.append(
            _rule(
                "Cash conversion",
                "FCF / Net income ≈ 80–120%",
                fcf_conv,
                "na",
                "Cash-flow detail missing.",
                as_pct=True,
            )
        )
    else:
        if 0.8 <= fcf_conv <= 1.2:
            status, comment = "pass", "Earnings are backed by cash."
        elif 0.6 <= fcf_conv <= 1.4:
            status, comment = "warn", "Okay but a bit noisy."
        else:
            status, comment = "fail", "Earnings not reliably backed by cash."
        rules.append(
            _rule(
                "Cash conversion",
                "FCF / Net income ≈ 80–120%",
                fcf_conv,
                status,
                comment,
                as_pct=True,
            )
        )

    # Debt/Equity ≤ 0.5 (≤1.0 warning)
    if _is_nan(debt_to_equity):
        rules.append(
            _rule(
                "Leverage",
                "Debt/Equity ≤ 0.5",
                debt_to_equity,
                "na",
                "Leverage data missing.",
            )
        )
    else:
        if debt_to_equity <= 0.5:
            status, comment = "pass", "Very conservative balance sheet."
        elif debt_to_equity <= 1.0:
            status, comment = "warn", "Moderate leverage."
        else:
            status, comment = "fail", "Heavy leverage for Buffett style."
        rules.append(
            _rule("Leverage", "Debt/Equity ≤ 0.5", debt_to_equity, status, comment)
        )

    # P/E ≤ 20 (≤30 warning)
    if _is_nan(pe):
        rules.append(
            _rule(
                "Valuation",
                "P/E ≤ 20",
                pe,
                "na",
                "P/E not available.",
            )
        )
    else:
        if pe <= 20:
            status, comment = "pass", "Reasonable price for quality."
        elif pe <= 30:
            status, comment = "warn", "Somewhat rich valuation."
        else:
            status, comment = "fail", "Very expensive relative to earnings."
        rules.append(_rule("Valuation", "P/E ≤ 20", pe, status, comment))

    summary = _summary_from_rules(rules, "Buffett")
    return {"summary": summary, "rules": rules}


# ---------- Lynch (GARP / PEG) ----------


def lynch_rules(metrics: Dict[str, Any]) -> Dict[str, Any]:
    pe = _get(metrics, "valuation", "pe")
    peg = _get(metrics, "valuation", "peg")
    rev_g = _get(metrics, "growth", "revenue_growth")
    earn_g = _get(metrics, "growth", "earnings_growth")
    debt_to_equity = _get(metrics, "balance_sheet", "debt_to_equity")

    rules: List[Dict[str, Any]] = []

    growth = earn_g if not _is_nan(earn_g) else rev_g

    # Growth 10–20%+
    if _is_nan(growth):
        rules.append(
            _rule(
                "Growth rate",
                "Growth ≥ 10%",
                growth,
                "na",
                "Growth data missing.",
                as_pct=True,
            )
        )
    else:
        if growth >= 0.20:
            status, comment = "pass", "Very strong growth."
        elif growth >= 0.10:
            status, comment = "pass", "Solid, Lynch-style grower."
        elif growth >= 0.05:
            status, comment = "warn", "Mild growth."
        else:
            status, comment = "fail", "Low growth for Lynch-style idea."
        rules.append(
            _rule(
                "Growth rate",
                "Growth ≥ 10%",
                growth,
                status,
                comment,
                as_pct=True,
            )
        )

    # PEG around 1
    if _is_nan(peg):
        rules.append(
            _rule(
                "PEG ratio",
                "PEG ≈ 1.0",
                peg,
                "na",
                "PEG can't be computed reliably.",
            )
        )
    else:
        if peg <= 1.0:
            status, comment = "pass", "Classic Lynch PEG ≤ 1."
        elif peg <= 1.5:
            status, comment = "warn", "PEG a bit high but maybe okay."
        else:
            status, comment = "fail", "PEG too high for GARP."
        rules.append(_rule("PEG ratio", "PEG ≈ 1.0", peg, status, comment))

    # P/E sanity check (not crazy high)
    if _is_nan(pe):
        rules.append(
            _rule(
                "P/E guardrail",
                "P/E not extreme (≤ 30)",
                pe,
                "na",
                "P/E missing.",
            )
        )
    else:
        if pe <= 20:
            status, comment = "pass", "Reasonable earnings multiple."
        elif pe <= 30:
            status, comment = "warn", "Upper end of reasonable."
        else:
            status, comment = "fail", "Too expensive for Lynch-style GARP."
        rules.append(
            _rule("P/E guardrail", "P/E not extreme (≤ 30)", pe, status, comment)
        )

    # Debt/Equity guardrail
    if _is_nan(debt_to_equity):
        rules.append(
            _rule(
                "Leverage",
                "Debt/Equity ≤ 1.0",
                debt_to_equity,
                "na",
                "Leverage data missing.",
            )
        )
    else:
        if debt_to_equity <= 0.5:
            status, comment = "pass", "Comfortable leverage for a grower."
        elif debt_to_equity <= 1.0:
            status, comment = "warn", "Moderate leverage."
        else:
            status, comment = "fail", "High leverage for Lynch-style stock."
        rules.append(
            _rule("Leverage", "Debt/Equity ≤ 1.0", debt_to_equity, status, comment)
        )

    summary = _summary_from_rules(rules, "Lynch")
    return {"summary": summary, "rules": rules}


# ---------- Greenblatt (Magic Formula) ----------


def greenblatt_rules(metrics: Dict[str, Any]) -> Dict[str, Any]:
    earnings_yield = _get(metrics, "valuation", "earnings_yield")
    ev_ebitda = _get(metrics, "valuation", "ev_ebitda")
    roe = _get(metrics, "quality", "roe")

    rules: List[Dict[str, Any]] = []

    # Earnings yield (inverse of P/E)
    if _is_nan(earnings_yield):
        rules.append(
            _rule(
                "Earnings yield",
                "Earnings yield ≥ 8%",
                earnings_yield,
                "na",
                "Earnings yield can't be computed.",
                as_pct=True,
            )
        )
    else:
        ey = earnings_yield
        if ey >= 0.15:
            status, comment = "pass", "Very cheap on earnings."
        elif ey >= 0.08:
            status, comment = "pass", "Cheap-ish on earnings."
        else:
            status, comment = "fail", "Not cheap for Magic Formula."
        rules.append(
            _rule(
                "Earnings yield",
                "Earnings yield ≥ 8%",
                ey,
                status,
                comment,
                as_pct=True,
            )
        )

    # Return on equity as ROC proxy
    if _is_nan(roe):
        rules.append(
            _rule(
                "Return on capital (ROE proxy)",
                "ROE ≥ 15%",
                roe,
                "na",
                "ROE not available.",
                as_pct=True,
            )
        )
    else:
        if roe >= 0.20:
            status, comment = "pass", "Excellent return on capital."
        elif roe >= 0.15:
            status, comment = "pass", "Good return on capital."
        else:
            status, comment = "fail", "Weak ROC for Magic Formula."
        rules.append(
            _rule(
                "Return on capital (ROE proxy)",
                "ROE ≥ 15%",
                roe,
                status,
                comment,
                as_pct=True,
            )
        )

    # EV/EBITDA sanity
    if _is_nan(ev_ebitda):
        rules.append(
            _rule(
                "EV/EBITDA",
                "EV/EBITDA ≤ 10",
                ev_ebitda,
                "na",
                "EV/EBITDA missing.",
            )
        )
    else:
        if ev_ebitda <= 8:
            status, comment = "pass", "Multiple consistent with Magic Formula cheapness."
        elif ev_ebitda <= 10:
            status, comment = "warn", "Okay, not screaming cheap."
        else:
            status, comment = "fail", "Too expensive on EV/EBITDA."
        rules.append(
            _rule("EV/EBITDA", "EV/EBITDA ≤ 10", ev_ebitda, status, comment)
        )

    summary = _summary_from_rules(rules, "Greenblatt")
    return {"summary": summary, "rules": rules}


# ---------- Burry (Deep FCF Value) ----------


def burry_rules(metrics: Dict[str, Any]) -> Dict[str, Any]:
    fcf_yield = _get(metrics, "valuation", "fcf_yield")
    ev_ebitda = _get(metrics, "valuation", "ev_ebitda")
    pe = _get(metrics, "valuation", "pe")
    debt_to_equity = _get(metrics, "balance_sheet", "debt_to_equity")

    rules: List[Dict[str, Any]] = []

    # FCF yield
    if _is_nan(fcf_yield):
        rules.append(
            _rule(
                "FCF yield",
                "FCF yield ≥ 8–10%",
                fcf_yield,
                "na",
                "Free cash flow data missing.",
                as_pct=True,
            )
        )
    else:
        fy = fcf_yield
        if fy >= 0.10:
            status, comment = "pass", "Very cheap on cash flows."
        elif fy >= 0.06:
            status, comment = "warn", "Cheap-ish on cash flows."
        else:
            status, comment = "fail", "Not cheap on cash flows."
        rules.append(
            _rule(
                "FCF yield",
                "FCF yield ≥ 8–10%",
                fy,
                status,
                comment,
                as_pct=True,
            )
        )

    # EV/EBITDA or P/E as backup
    if _is_nan(ev_ebitda) and _is_nan(pe):
        rules.append(
            _rule(
                "Valuation multiples",
                "EV/EBITDA ≤ 10 or P/E ≤ 12",
                float("nan"),
                "na",
                "Valuation multiples missing.",
            )
        )
    else:
        if not _is_nan(ev_ebitda):
            if ev_ebitda <= 8:
                status, comment = "pass", "EV/EBITDA consistent with deep value."
            elif ev_ebitda <= 10:
                status, comment = "warn", "Okay but not extreme value."
            else:
                status, comment = "fail", "Rich on EV/EBITDA for Burry."
            rules.append(
                _rule(
                    "EV/EBITDA",
                    "EV/EBITDA ≤ 10",
                    ev_ebitda,
                    status,
                    comment,
                )
            )
        else:
            if pe <= 10:
                status, comment = "pass", "Low P/E as backup value signal."
            elif pe <= 14:
                status, comment = "warn", "Moderate P/E."
            else:
                status, comment = "fail", "High P/E for deep value."
            rules.append(_rule("P/E", "P/E ≤ 12", pe, status, comment))

    # Leverage
    if _is_nan(debt_to_equity):
        rules.append(
            _rule(
                "Leverage",
                "Debt/Equity ≤ 1.0",
                debt_to_equity,
                "na",
                "Leverage data missing.",
            )
        )
    else:
        if debt_to_equity <= 0.5:
            status, comment = "pass", "Very conservative balance sheet."
        elif debt_to_equity <= 1.0:
            status, comment = "warn", "Manageable leverage."
        else:
            status, comment = "fail", "High leverage for a deep value idea."
        rules.append(
            _rule("Leverage", "Debt/Equity ≤ 1.0", debt_to_equity, status, comment)
        )

    summary = _summary_from_rules(rules, "Burry")
    return {"summary": summary, "rules": rules}


# ---------- Registry ----------
GRAHAM = InvestorProfile(
    key="graham",
    name="Benjamin Graham",
    label="Graham – Deep Value",
    category="Deep Value",
    description="Low multiples, strong balance sheet, and classic Ben Graham safeguards.",
    rules_fn=graham_rules,
)

BUFFETT = InvestorProfile(
    key="buffett",
    name="Warren Buffett",
    label="Buffett – Quality at Fair Price",
    category="Quality",
    description="High-quality, high-ROE businesses with conservative leverage at sensible valuations.",
    rules_fn=buffett_rules,
)

LYNCH = InvestorProfile(
    key="lynch",
    name="Peter Lynch",
    label="Lynch – GARP (PEG)",
    category="GARP",
    description="Growth at a reasonable price; PEG around 1 with decent balance sheet.",
    rules_fn=lynch_rules,
)

GREENBLATT = InvestorProfile(
    key="greenblatt",
    name="Joel Greenblatt",
    label="Greenblatt – Magic Formula",
    category="Deep Value / Quality",
    description="High earnings yield and high return on capital, Magic Formula style.",
    rules_fn=greenblatt_rules,
)

BURRY = InvestorProfile(
    key="burry",
    name="Michael Burry",
    label="Burry – Deep FCF Value",
    category="Deep Value",
    description="Cheap on free cash flow with an acceptable balance sheet.",
    rules_fn=burry_rules,
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
