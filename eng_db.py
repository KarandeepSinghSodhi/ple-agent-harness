"""Offline Engineering Reference Database & Deterministic Calculators.

Runs locally inside the serve container with zero network calls and zero
hallucination. Provides deterministic lookups and engineering calculations
across Structural, Civil/Geotech, Electrical, and Mechanical/Fire domains,
plus pint-based dimensional analysis.
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

# NEC Table 310.16 (75C Copper & Aluminum Ampacities) + NEC Ch 9 Table 9 (PVC conduit, Ohms/1000 ft)
NEC_CONDUCTORS: dict[str, dict[str, float]] = {
    "14 AWG": {"Cu_A": 20, "Al_A": 0, "Cu_R": 3.1, "Al_R": 5.1, "X_pvc": 0.058},
    "12 AWG": {"Cu_A": 25, "Al_A": 20, "Cu_R": 2.0, "Al_R": 3.2, "X_pvc": 0.054},
    "10 AWG": {"Cu_A": 35, "Al_A": 30, "Cu_R": 1.2, "Al_R": 2.0, "X_pvc": 0.050},
    "8 AWG": {"Cu_A": 50, "Al_A": 40, "Cu_R": 0.78, "Al_R": 1.3, "X_pvc": 0.052},
    "6 AWG": {"Cu_A": 65, "Al_A": 50, "Cu_R": 0.49, "Al_R": 0.81, "X_pvc": 0.051},
    "4 AWG": {"Cu_A": 85, "Al_A": 65, "Cu_R": 0.31, "Al_R": 0.51, "X_pvc": 0.048},
    "3 AWG": {"Cu_A": 100, "Al_A": 75, "Cu_R": 0.25, "Al_R": 0.40, "X_pvc": 0.047},
    "2 AWG": {"Cu_A": 115, "Al_A": 90, "Cu_R": 0.19, "Al_R": 0.32, "X_pvc": 0.045},
    "1 AWG": {"Cu_A": 130, "Al_A": 100, "Cu_R": 0.15, "Al_R": 0.25, "X_pvc": 0.046},
    "1/0 AWG": {"Cu_A": 150, "Al_A": 120, "Cu_R": 0.12, "Al_R": 0.20, "X_pvc": 0.044},
    "2/0 AWG": {"Cu_A": 175, "Al_A": 135, "Cu_R": 0.10, "Al_R": 0.16, "X_pvc": 0.043},
    "3/0 AWG": {"Cu_A": 200, "Al_A": 155, "Cu_R": 0.077, "Al_R": 0.13, "X_pvc": 0.042},
    "4/0 AWG": {"Cu_A": 230, "Al_A": 180, "Cu_R": 0.062, "Al_R": 0.10, "X_pvc": 0.041},
    "250 kcmil": {"Cu_A": 255, "Al_A": 205, "Cu_R": 0.052, "Al_R": 0.085, "X_pvc": 0.041},
    "300 kcmil": {"Cu_A": 285, "Al_A": 230, "Cu_R": 0.044, "Al_R": 0.071, "X_pvc": 0.041},
    "350 kcmil": {"Cu_A": 310, "Al_A": 250, "Cu_R": 0.038, "Al_R": 0.061, "X_pvc": 0.040},
    "400 kcmil": {"Cu_A": 335, "Al_A": 270, "Cu_R": 0.033, "Al_R": 0.054, "X_pvc": 0.040},
    "500 kcmil": {"Cu_A": 380, "Al_A": 310, "Cu_R": 0.027, "Al_R": 0.043, "X_pvc": 0.039},
    "600 kcmil": {"Cu_A": 420, "Al_A": 340, "Cu_R": 0.023, "Al_R": 0.036, "X_pvc": 0.039},
    "750 kcmil": {"Cu_A": 475, "Al_A": 385, "Cu_R": 0.019, "Al_R": 0.029, "X_pvc": 0.038},
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
    """Evaluates a physical quantity expression using pint and checks dimensional consistency."""
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


def asce7_load_combos(
    D: float = 0.0,
    L: float = 0.0,
    Lr: float = 0.0,
    S: float = 0.0,
    R: float = 0.0,
    W: float = 0.0,
    E: float = 0.0,
) -> dict[str, Any]:
    """Evaluates all ASCE 7-16/22 LRFD load combinations and returns the governing case."""
    roof = max(Lr, S, R)
    combos = {
        "LRFD-1 (1.4D)": 1.4 * D,
        "LRFD-2 (1.2D + 1.6L + 0.5(Lr/S/R))": 1.2 * D + 1.6 * L + 0.5 * roof,
        "LRFD-3 (1.2D + 1.6(Lr/S/R) + max(L,0.5W))": 1.2 * D + 1.6 * roof + max(L, 0.5 * abs(W)),
        "LRFD-4 (1.2D + 1.0W + L + 0.5(Lr/S/R))": 1.2 * D + 1.0 * W + L + 0.5 * roof,
        "LRFD-5 (1.2D + 1.0E + L + 0.2S)": 1.2 * D + 1.0 * E + L + 0.2 * S,
        "LRFD-6 (0.9D + 1.0W)": 0.9 * D + 1.0 * W,
        "LRFD-7 (0.9D + 1.0E)": 0.9 * D + 1.0 * E,
    }
    gov_name = max(combos, key=lambda k: abs(combos[k]))
    return {
        "combinations": {k: round(v, 4) for k, v in combos.items()},
        "governing_combo": gov_name,
        "governing_value": round(combos[gov_name], 4),
    }


def bearing_capacity_terzaghi(
    phi_deg: float, c: float, gamma: float, B: float, Df: float, shape: str = "square", fs: float = 3.0
) -> dict[str, float]:
    """Computes Terzaghi ultimate and allowable bearing capacity (consistent units)."""
    phi = math.radians(phi_deg)
    nq = math.exp(2 * (0.75 * math.pi - phi / 2) * math.tan(phi)) / (2 * (math.cos(math.radians(45 + phi_deg / 2))) ** 2)
    nc = (nq - 1) / math.tan(phi) if phi_deg > 0 else 5.7
    ngamma = 2 * (nq + 1) * math.tan(phi) / (1 + 0.4 * math.sin(4 * phi)) if phi_deg > 0 else 0.0
    q = gamma * Df
    if shape == "square":
        q_ult = 1.3 * c * nc + q * nq + 0.4 * gamma * B * ngamma
    elif shape == "circular":
        q_ult = 1.3 * c * nc + q * nq + 0.3 * gamma * B * ngamma
    else:
        q_ult = c * nc + q * nq + 0.5 * gamma * B * ngamma
    return {
        "Nc": round(nc, 3),
        "Nq": round(nq, 3),
        "Ngamma": round(ngamma, 3),
        "q_ult": round(q_ult, 4),
        "q_allow": round(q_ult / fs, 4),
        "FS": fs,
    }


def consolidation_settlement(
    H: float, e0: float, Cc: float, Cr: float, sigma0_eff: float, delta_sigma: float, sigma_p_eff: float | None = None
) -> dict[str, float | str]:
    """Computes 1D primary consolidation settlement (units of H, e.g. ft or m)."""
    pc = sigma_p_eff if sigma_p_eff is not None else sigma0_eff
    sigma_final = sigma0_eff + delta_sigma
    if sigma_final <= pc:
        sc = (Cr * H / (1.0 + e0)) * math.log10(sigma_final / sigma0_eff)
        regime = "overconsolidated (recompression only)"
    elif sigma0_eff < pc < sigma_final:
        sc = (Cr * H / (1.0 + e0)) * math.log10(pc / sigma0_eff) + (Cc * H / (1.0 + e0)) * math.log10(sigma_final / pc)
        regime = "overconsolidated -> virgin compression"
    else:
        sc = (Cc * H / (1.0 + e0)) * math.log10(sigma_final / sigma0_eff)
        regime = "normally consolidated"
    return {"settlement": round(sc, 6), "settlement_inches": round(sc * 12.0, 4), "regime": regime}


def voltage_drop_nec(
    size: str,
    material: str,
    current_a: float,
    length_ft: float,
    voltage_ll: float,
    phase: int = 3,
    pf: float = 0.85,
    parallel_sets: int = 1,
) -> dict[str, Any]:
    """Computes NEC Chapter 9 Table 9 voltage drop (Volts and %)."""
    cond = None
    for k, v in NEC_CONDUCTORS.items():
        if size.strip().lower() in k.lower():
            cond = (k, v)
            break
    if cond is None:
        return {"ok": False, "error": f"Conductor '{size}' not found"}
    name, props = cond
    mat = material.strip().lower()
    r = props["Cu_R"] if mat.startswith("cu") else props["Al_R"]
    x = props["X_pvc"]
    sin_theta = math.sqrt(max(0.0, 1.0 - pf * pf))
    zeff = r * pf + x * sin_theta
    factor = math.sqrt(3.0) if phase == 3 else 2.0
    vd = factor * current_a * (length_ft / 1000.0) * zeff / max(1, parallel_sets)
    vd_pct = (vd / voltage_ll) * 100.0
    return {
        "ok": True,
        "conductor": name,
        "material": "Cu" if mat.startswith("cu") else "Al",
        "R_ohm_per_kft": r,
        "X_ohm_per_kft": x,
        "Z_eff": round(zeff, 5),
        "voltage_drop_V": round(vd, 3),
        "voltage_drop_pct": round(vd_pct, 3),
    }


def fire_dstar(q_kw: float, dx_m: float, rho: float = 1.204, cp: float = 1.005, t_inf: float = 293.15, g: float = 9.81) -> dict[str, float | str]:
    """Computes FDS characteristic fire diameter D* and D*/dx mesh resolution ratio."""
    d_star = (q_kw / (rho * cp * t_inf * math.sqrt(g))) ** (2.0 / 5.0)
    ratio = d_star / dx_m
    quality = "fine (>=16)" if ratio >= 16 else ("moderate (10-16)" if ratio >= 10 else ("coarse (4-10)" if ratio >= 4 else "under-resolved (<4)"))
    return {
        "D_star_m": round(d_star, 4),
        "dx_m": dx_m,
        "D_star_over_dx": round(ratio, 2),
        "nureg_1824_assessment": quality,
    }


def calculate(calc_type: str, params: dict[str, Any]) -> str:
    """Dispatches deterministic engineering calculations."""
    ct = calc_type.strip().lower()
    try:
        if ct == "asce7_combos":
            return json.dumps(asce7_load_combos(**params), indent=2)
        if ct == "bearing_terzaghi":
            return json.dumps(bearing_capacity_terzaghi(**params), indent=2)
        if ct == "settlement":
            return json.dumps(consolidation_settlement(**params), indent=2)
        if ct == "voltage_drop":
            return json.dumps(voltage_drop_nec(**params), indent=2)
        if ct == "fire_dstar":
            return json.dumps(fire_dstar(**params), indent=2)
        return json.dumps({
            "error": f"Unknown calc_type '{calc_type}'. Available: asce7_combos, bearing_terzaghi, settlement, voltage_drop, fire_dstar"
        })
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


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
            return json.dumps({"table": "NEC 310.16 (75C) & Ch9 Table 9", "sizes": list(NEC_CONDUCTORS.keys())})
        for size, vals in NEC_CONDUCTORS.items():
            if key.strip().lower() in size.lower():
                return json.dumps({"size": size, "temp_rating": "75C", **vals})
        return json.dumps({"error": f"Wire size {key} not found"})
    if cat in ("asce7", "loads", "load_combos"):
        return json.dumps({"standard": "ASCE 7-16/22 LRFD", "combinations": ASCE7_LRFD_COMBOS})
    return json.dumps({"error": f"Unknown category '{category}'. Use: steel, nec, asce7."})
