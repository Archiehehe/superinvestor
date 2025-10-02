# lenses.py
from __future__ import annotations
from typing import Dict, List
import math
import numpy as np

def _ok(x, thresh, mode="lt"):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return False
        return (x < thresh) if mode == "lt" else (x > thresh)
    except Exception:
        return False

def _score(passed_flags: List[bool]) -> int:
    return int(sum(1 for v in passed_flags if v))

def burry_lens(fin: Dict[str, float], mult: Dict[str, float], roic: Dict[str, float]) -> Dict:
    checks = {
        "EV/EBITDA < 10": _ok(mult.get("EV/EBITDA"), 10, "lt"),
        "P/FCF < 20 or FCF Yield > 5%": (
            _ok(mult.get("P/FCF"), 20, "lt") or _ok(mult.get("FCF Yield (to Equity)"), 0.05, "gt")
        ),
        "EBIT/EV > 5%": _ok(mult.get("EBIT/EV"), 0.05, "gt"),
    }
    passed = list(checks.values())
    return {"verdict": "PASS" if all(passed) else "MIXED" if any(passed) else "FAIL",
            "score": _score(passed), "checks": checks,
            "notes": ["Burry-style cheapness: focus on EV/EBITDA and cash yield."]}

def greenblatt_lens(fin: Dict[str, float], mult: Dict[str, float], roic: Dict[str, float]) -> Dict:
    checks = {
        "ROC (Greenblatt) > 15%": _ok(roic.get("ROC_Greenblatt"), 0.15, "gt"),
        "EV/EBIT < 12": _ok(mult.get("EV/EBIT"), 12, "lt"),
    }
    passed = list(checks.values())
    return {"verdict": "PASS" if all(passed) else "MIXED" if any(passed) else "FAIL",
            "score": _score(passed), "checks": checks,
            "notes": ["Greenblatt: rank by ROC and earnings yield."]}

def buffett_lens(fin: Dict[str, float], mult: Dict[str, float], roic: Dict[str, float]) -> Dict:
    checks = {
        "ROIC > 12%": _ok(roic.get("ROIC"), 0.12, "gt"),
        "FCF Yield > 3%": _ok(mult.get("FCF Yield (to Equity)"), 0.03, "gt"),
        "EV/Sales < 6": _ok(mult.get("EV/Sales"), 6, "lt"),
    }
    passed = list(checks.values())
    return {"verdict": "PASS" if all(passed) else "MIXED" if any(passed) else "FAIL",
            "score": _score(passed), "checks": checks,
            "notes": ["Buffett-ish: durable economics, cash generation at fair price."]}

def klarman_lens(fin: Dict[str, float], mult: Dict[str, float], roic: Dict[str, float]) -> Dict:
    equity = fin.get("Equity_MRQ")
    ev = fin.get("EV")
    ratio = (equity / ev) if (equity and ev and ev > 0) else np.nan
    checks = {
        "Equity >= 0.5 × EV": _ok(ratio, 0.5, "gt"),
        "EV/EBITDA < 8": _ok(mult.get("EV/EBITDA"), 8, "lt"),
    }
    passed = list(checks.values())
    return {"verdict": "PASS" if all(passed) else "MIXED" if any(passed) else "FAIL",
            "score": _score(passed), "checks": checks,
            "notes": ["Klarman-ish: asset backing and cheap on cash earnings."]}

def einhorn_lens(fin: Dict[str, float], mult: Dict[str, float], roic: Dict[str, float]) -> Dict:
    checks = {
        "EBIT/EV > 6%": _ok(mult.get("EBIT/EV"), 0.06, "gt"),
        "EV/EBIT < 16": _ok(mult.get("EV/EBIT"), 16, "lt"),
    }
    passed = list(checks.values())
    return {"verdict": "PASS" if all(passed) else "MIXED" if any(passed) else "FAIL",
            "score": _score(passed), "checks": checks,
            "notes": ["Einhorn-like: earnings yield compared to alternatives."]}
