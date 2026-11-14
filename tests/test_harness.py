"""Self-contained verification suite for PE-Ledger-Solver harness."""

import json
import os
import sys

os.environ.setdefault("MODEL_NAME", "claude-opus-5-5")
os.environ.setdefault("OPENAI_API_KEY", "dummy-local-key")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import eng_db
import ledger
import serve


def test_units():
    res = eng_db.verify_units("120 * kip / (4 * ft**2)", "kPa")
    assert res["ok"] is True
    assert abs(res["converted_magnitude"] - 1436.41) < 1.0


def test_lookups():
    steel = json.loads(eng_db.lookup("steel", "W14X90"))
    assert steel["Ix"] == 999.0 and steel["Zx"] == 157.0
    nec = json.loads(eng_db.lookup("nec", "4/0 AWG"))
    assert nec["Cu_A"] == 230 and nec["Al_A"] == 180
    asce = json.loads(eng_db.lookup("asce7"))
    assert len(asce["combinations"]) == 7


def test_calculators():
    combos = json.loads(eng_db.calculate("asce7_combos", {"D": 50, "L": 80, "S": 20}))
    assert combos["governing_combo"].startswith("LRFD-2")
    assert abs(combos["governing_value"] - (1.2 * 50 + 1.6 * 80 + 0.5 * 20)) < 1e-6

    bearing = json.loads(eng_db.calculate("bearing_terzaghi", {"phi_deg": 30, "c": 0, "gamma": 120, "B": 6, "Df": 3}))
    assert bearing["q_ult"] > 0 and bearing["q_allow"] == round(bearing["q_ult"] / 3.0, 4)

    settle = json.loads(eng_db.calculate("settlement", {
        "H": 10.0, "e0": 0.85, "Cc": 0.30, "Cr": 0.05, "sigma0_eff": 1000.0, "delta_sigma": 500.0
    }))
    assert settle["settlement"] > 0 and settle["regime"] == "normally consolidated"

    vd = json.loads(eng_db.calculate("voltage_drop", {
        "size": "4/0 AWG", "material": "Cu", "current_a": 180, "length_ft": 250, "voltage_ll": 480, "phase": 3, "pf": 0.85
    }))
    assert vd["ok"] is True and vd["voltage_drop_pct"] < 2.0

    fire = json.loads(eng_db.calculate("fire_dstar", {"q_kw": 2500.0, "dx_m": 0.1}))
    assert fire["D_star_over_dx"] >= 10.0


def test_ledger():
    lg = ledger.EngineeringLedger()
    out = lg.update(
        discipline="Structural",
        add_codes=["ASCE 7-16", "AISC 360-16"],
        set_invariants={"soil_allowable": "3000 psf"},
        set_deliverables={"/filesystem/submission.zip": "verified"},
    )
    assert "ASCE 7-16" in out
    assert "3000 psf" in out
    assert "/filesystem/submission.zip: verified" in out


def test_serve_tools_schema():
    names = {t["function"]["name"] for t in serve.BASE_TOOLS}
    assert {"bash", "read_file", "write_file", "eng_verify_units", "eng_lookup", "eng_calc", "ledger_update", "audit_deliverables"} <= names


if __name__ == "__main__":
    test_units()
    test_lookups()
    test_calculators()
    test_ledger()
    test_serve_tools_schema()
    print("ALL TESTS PASSED SUCCESSFULLY")
