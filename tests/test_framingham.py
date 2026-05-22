"""
Numerical tests for the D'Agostino 2008 Framingham 10-year CVD risk implementation.

Reference: D'Agostino R.B. et al., 2008, "General Cardiovascular Risk Profile
for Use in Primary Care," Circulation 117(6). The worked examples in the
paper give roughly:

    Male,   55y, TC=200, HDL=50, SBP=125 (untreated), no smoke, no diabetes
        -> ~10% 10-year CVD risk

    Female, 55y, TC=200, HDL=50, SBP=125 (untreated), no smoke, no diabetes
        -> ~5-6% 10-year CVD risk

These tests lock the implementation against those reference points with
~2-percentage-point bounds (the paper itself reports tier-level guidance, not
precise quantiles).
"""

import math

import numpy as np
import pandas as pd
import pytest

from framingham import compute_framingham


def _row(*, sex, age, tc, hdl, sbp, smoke=2, diab=2):
    """Build a one-row NHANES-style frame; sex 1=male/2=female, smoke/diab 1=yes/2=no."""
    return pd.DataFrame([{
        "SEQN": 1, "RIDAGEYR": age, "RIAGENDR": sex,
        "LBXTC": tc, "LBDHDD": hdl, "BPXSY1": sbp,
        "SMQ020": smoke, "DIQ010": diab,
    }])


def test_male_reference_case_matches_paper():
    df = _row(sex=1, age=55, tc=200, hdl=50, sbp=125)
    out = compute_framingham(df)
    risk = out["framingham_risk"].iloc[0]
    # Paper reports ~10%. Allow a 2 pp envelope.
    assert 0.08 <= risk <= 0.12, f"male 55y reference risk = {risk:.4f}"


def test_female_reference_case_matches_paper():
    df = _row(sex=2, age=55, tc=200, hdl=50, sbp=125)
    out = compute_framingham(df)
    risk = out["framingham_risk"].iloc[0]
    # Paper reports ~5-6%.
    assert 0.04 <= risk <= 0.08, f"female 55y reference risk = {risk:.4f}"


def test_risk_strictly_increases_with_age_holding_others_fixed():
    rows = [_row(sex=1, age=a, tc=220, hdl=45, sbp=135) for a in (35, 45, 55, 65)]
    risks = [compute_framingham(r)["framingham_risk"].iloc[0] for r in rows]
    assert all(b > a for a, b in zip(risks, risks[1:])), risks


def test_smoking_and_diabetes_each_raise_risk():
    baseline = _row(sex=1, age=55, tc=220, hdl=45, sbp=130).copy()
    smoker = baseline.copy(); smoker["SMQ020"] = 1
    diabetic = baseline.copy(); diabetic["DIQ010"] = 1
    r0 = compute_framingham(baseline)["framingham_risk"].iloc[0]
    rs = compute_framingham(smoker)["framingham_risk"].iloc[0]
    rd = compute_framingham(diabetic)["framingham_risk"].iloc[0]
    assert rs > r0, (r0, rs)
    assert rd > r0, (r0, rd)


def test_high_risk_flag_uses_10_percent_threshold():
    # A clearly high-risk profile: older male smoker with diabetes and bad lipids.
    high = _row(sex=1, age=68, tc=260, hdl=35, sbp=160, smoke=1, diab=1)
    out = compute_framingham(high)
    assert out["high_risk"].iloc[0] == 1
    assert out["framingham_risk"].iloc[0] >= 0.10


def test_age_filter_drops_under_30_and_over_74():
    df = pd.concat([
        _row(sex=1, age=25, tc=200, hdl=50, sbp=120),
        _row(sex=1, age=50, tc=200, hdl=50, sbp=120),
        _row(sex=1, age=80, tc=200, hdl=50, sbp=120),
    ], ignore_index=True)
    df["SEQN"] = [1, 2, 3]
    out = compute_framingham(df)
    # Only the age-50 row survives the validity filter.
    assert len(out) == 1
    assert out["RIDAGEYR"].iloc[0] == 50


def test_nan_inputs_produce_nan_risk_not_an_exception():
    df = _row(sex=1, age=55, tc=np.nan, hdl=50, sbp=125)
    out = compute_framingham(df)
    risk = out["framingham_risk"].iloc[0]
    assert math.isnan(risk)
