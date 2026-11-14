"""Offline Engineering Reference Database & Physical Unit Verification Engine.

Runs locally inside the serve container with zero network calls and zero
hallucination. Provides deterministic lookups for AISC steel shapes,
ASCE 7 load combinations, NEC ampacities, geotechnical bearing factors,
and pint-based dimensional analysis.
"""

from __future__ import annotations

import json
import math
from typing import Any

try:
    import pint
    _UREG = pint.UnitRegistry()
except ImportError:
    _UREG = None


# Representative AISC W-shape database (A in^2, d in, bf in, tf in, tw in, Ix in^4, Sx in^3, Zx in^3, Iy in^4, ry in)
AISC_W_SHAPES: dict[str, dict[str, float]] = {
    "W8X18": {"A": 5.26, "d": 8.14, "bf": 5.25, "tf": 0.330, "tw": 0.230, "Ix": 61.9, "Sx": 15.2, "Zx": 17.0, "Iy": 7.97, "ry": 1.23},
    "W8X31": {"A": 9.13, "d": 8.00, "bf": 8.00, "tf": 0.435, "tw": 0.285, "Ix": 110.0, "Sx": 27.5, "Zx": 30.4, "Iy": 37.1, "ry": 2.02},
    "W10X33": {"A": 9.71, "d": 9.73, "bf": 7.96, "tf": 0.435, "tw": 0.290, "Ix": 171.0, "Sx": 35.0, "Zx": 38.8, "Iy": 36.6, "ry": 1.94},
    "W10X49": {"A": 14.4, "d": 9.98, "bf": 10.0, "tf": 0.560, "tw": 0.340, "Ix": 272.0, "Sx": 54.6, "Zx": 60.4, "Iy": 93.4, "ry": 2.54},
    "W12X26": {"A": 7.65, "d": 12.2, "bf": 6.49, "tf": 0.380, "tw": 0.230, "Ix": 204.0, "Sx": 33.4, "Zx": 37.2, "Iy": 17.3, "ry": 1.51},
    "W12X50": {"A": 14.6, "d": 12.2, "bf": 8.08, "tf": 0.640, "tw": 0.370, "Ix": 391.0, "Sx": 64.2, "Zx": 71.9, "Iy": 56.3, "ry": 1.96},
    "W12X65": {"A": 19.1, "d": 12.1, "bf": 12.0, "tf": 0.605, "tw": 0.390, "Ix": 533.0, "Sx": 87.9, "Zx": 96.8, "Iy": 174.0, "ry": 3.02},
    "W14X48": {"A": 14.1, "d": 13.8, "bf": 8.03, "tf": 0.595, "tw": 0.340, "Ix": 484.0, "Sx": 70.2, "Zx": 78.4, "Iy": 51.4, "ry": 1.91},
    "W14X90": {"A": 26.5, "d": 14.0, "bf": 14.5, "tf": 0.710, "tw": 0.440, "Ix": 999.0, "Sx": 143.0, "Zx": 157.0, "Iy": 362.0, "ry": 3.70},
    "W16X36": {"A": 10.6, "d": 15.9, "bf": 6.99, "tf": 0.430, "tw": 0.295, "Ix": 448.0, "Sx": 56.5, "Zx": 64.0, "Iy": 24.5, "ry": 1.52},
    "W18X50": {"A": 14.7, "d": 18.0, "bf": 7.50, "tf": 0.570, "tw": 0.355, "Ix": 800.0, "Sx": 88.9, "Zx": 101.0, "Iy": 40.1, "ry": 1.65},
    "W18X76": {"A": 22.3, "d": 18.2, "bf": 11.0, "tf": 0.680, "tw": 0.425, "Ix": 1330.0, "Sx": 146.0, "Zx": 163.0, "Iy": 152.0, "ry": 2.61},
    "W21X62": {"A": 18.3, "d": 21.0, "bf": 8.24, "tf": 0.615, "tw": 0.400, "Ix": 1330.0, "Sx": 127.0, "Zx": 144.0, "Iy": 57.5, "ry": 1.77},
    "W24X76": {"A": 22.4, "d": 23.9, "bf": 8.99, "tf": 0.680, "tw": 0.440, "Ix": 2100.0, "Sx": 176.0, "Zx": 200.0, "Iy": 82.5, "ry": 1.92},
    "W24X104": {"A": 30.6, "d": 24.1, "bf": 12.8, "tf": 0.750, "tw": 0.500, "Ix": 3100.0, "Sx": 258.0, "Zx": 289.0, "Iy": 259.0, "ry": 2.91},
    "W27X94": {"A": 27.7, "d": 26.9, "bf": 10.0, "tf": 0.745, "tw": 0.490, "Ix": 3270.0, "Sx": 243.0, "Zx": 278.0, "Iy": 124.0, "ry": 2.12},
    "W30X108": {"A": 31.7, "d": 29.8, "bf": 10.5, "tf": 0.760, "tw": 0.545, "Ix": 4470.0, "Sx": 299.0, "Zx": 346.0, "Iy": 146.0, "ry": 2.15},
    "W36X150": {"A": 44.2, "d": 35.9, "bf": 12.0, "tf": 0.940, "tw": 0.625, "Ix": 9040.0, "Sx": 504.0, "Zx": 581.0, "Iy": 270.0, "ry": 2.47},
}

# NEC Table 310.16 (75C Copper & Aluminum Ampacities in Amperes, <=3 current-carrying conductors)
NEC_AMPACITY_75C: dict[str, dict[str, int]] = {
    "14 AWG": {"Cu": 20, "Al": 0},
    "12 AWG": {"Cu": 25, "Al": 20},
    "10 AWG": {"Cu": 35, "Al": 30},
    "8 AWG": {"Cu": 50, "Al": 40},
    "6 AWG": {"Cu": 65, "Al": 50},
    "4 AWG": {"Cu": 85, "Al": 65},
    "3 AWG": {"Cu": 100, "Al": 75},
    "2 AWG": {"Cu": 115, "Al": 90},
    "1 AWG": {"Cu": 130, "Al": 100},
    "1/0 AWG": {"Cu": 150, "Al": 120},
    "2/0 AWG": {"Cu": 175, "Al": 135},
    "3/0 AWG": {"Cu": 200, "Al": 155},
    "4/0 AWG": {"Cu": 230, "Al": 180},
    "250 kcmil": {"Cu": 255, "Al": 205},
    "300 kcmil": {"Cu": 285, "Al": 230},
    "350 kcmil": {"Cu": 310, "Al": 250},
    "400 kcmil": {"Cu": 335, "Al": 270},
    "500 kcmil": {"Cu": 380, "Al": 310},
    "600 kcmil": {"Cu": 420, "Al": 340},
    "750 kcmil": {"Cu": 475, "Al": 385},
}

ASCE7_LRFD_COMBOS: list[str] = [
    "1.4*D",
    "1.2*D + 1.6*L + 0.5*max(Lr, S, R)",
    "1.2*D + 1.6*max(Lr, S, R) + max(L, 0.5*W)",
    "1.2*D + 1.0*W + L + 0.5*max(Lr, S, R)",
    "1.2*D + 1.0*E + L + 0.2*S",
    "0.9*D + 1.0*W",
    "0.9*D + 1.0*E",
]


def verify_units(expression: str, target_unit: str | None = None) -> dict[str, Any]:
    """Evaluates a physical quantity expression using pint and checks dimensional consistency.

    Args:
        expression: Python/pint expression e.g. '120 * lbf / (4 * ft**2)'.
        target_unit: Optional unit string to convert the result to (e.g. 'kPa').

    Returns:
        Dictionary with magnitude, unit, dimensionality, and optional converted value.
    """
    if _UREG is None:
        return {"ok": False, "error": "pint library is not installed"}
    try:
        qty = _UREG.Quantity(expression) if isinstance(expression, str) else expression
        if not hasattr(qty, "magnitude"):
            qty = _UREG(expression)
        res: dict[str, Any] = {
            "ok": True,
            "input": expression,
            "magnitude": float(qty.magnitude),
            "units": str(qty.units),
            "dimensionality": str(qty.dimensionality),
        }
        if target_unit:
            converted = qty.to(target_unit)
            res["converted_magnitude"] = float(converted.magnitude)
            res["converted_units"] = str(converted.units)
        return res
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def bearing_capacity_terzaghi(phi_deg: float, c: float, gamma: float, B: float, Df: float, shape: str = "square") -> dict[str, float]:
    """Computes Terzaghi ultimate bearing capacity factors and q_ult (consistent units)."""
    phi = math.radians(phi_deg)
    nq = math.exp(2 * (0.75 * math.pi - phi / 2) * math.tan(phi)) / (2 * (math.cos(math.radians(45 + phi_deg / 2))) ** 2)
    nc = (nq - 1) / math.tan(phi) if phi_deg > 0 else 5.7
    ngamma = 2 * (nq + 1) * math.tan(phi) / (1 + 0.4 * math.sin(4 * phi)) if phi_deg > 0 else 0.0
    q = gamma * Df
    if shape == "square":
        q_ult = 1.3 * c * nc + q * nq + 0.4 * gamma * B * ngamma
    elif shape == "circular":
        q_ult = 1.3 * c * nc + q * nq + 0.3 * gamma * B * ngamma
    else:  # strip
        q_ult = c * nc + q * nq + 0.5 * gamma * B * ngamma
    return {
        "Nc": round(nc, 3),
        "Nq": round(nq, 3),
        "Ngamma": round(ngamma, 3),
        "q_ult": round(q_ult, 4),
    }


def lookup(category: str, key: str = "") -> str:
    """Queries the offline engineering reference database."""
    cat = category.strip().lower()
    k = key.strip().upper()
    if cat in ("steel", "aisc", "w_shape", "beam"):
        if not k:
            return json.dumps({"available_shapes": list(AISC_W_SHAPES.keys())})
        if k in AISC_W_SHAPES:
            return json.dumps({"shape": k, "units": "in, in^2, in^3, in^4", **AISC_W_SHAPES[k]})
        matches = [s for s in AISC_W_SHAPES if k in s]
        return json.dumps({"error": f"Shape {k} not found", "matches": matches})
    if cat in ("nec", "ampacity", "wire"):
        if not key.strip():
            return json.dumps({"table": "NEC 310.16 (75C)", "sizes": list(NEC_AMPACITY_75C.keys())})
        for size, vals in NEC_AMPACITY_75C.items():
            if key.strip().lower() in size.lower():
                return json.dumps({"size": size, "temp_rating": "75C", "ampacity_A": vals})
        return json.dumps({"error": f"Wire size {key} not found"})
    if cat in ("asce7", "loads", "load_combos"):
        return json.dumps({"standard": "ASCE 7-16/22 LRFD", "combinations": ASCE7_LRFD_COMBOS})
    return json.dumps({"error": f"Unknown category '{category}'. Use: steel, nec, asce7."})
