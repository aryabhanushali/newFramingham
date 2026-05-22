"""
Integrity tests for the trained model artifacts in models/.

These run without NHANES data — they only need the joblib + JSON sidecars
that ship with the repo. They verify:

  * Bundle structure (pipeline + isotonic + feature_cols + medians)
  * Round-trip prediction is finite and in [0, 1]
  * Risk increases monotonically with age (model-level sanity)
  * The Python-side feature order matches scaler.json (no drift)
  * The isotonic LUT is monotone non-decreasing
  * Pipeline scaler params match the Swift-readable scaler.json
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")


@pytest.fixture(scope="module")
def bundle():
    return joblib.load(os.path.join(MODELS, "cvd_risk_v1.joblib"))


@pytest.fixture(scope="module")
def scaler_sidecar():
    with open(os.path.join(MODELS, "scaler.json")) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def iso_sidecar():
    with open(os.path.join(MODELS, "isotonic_calibration.json")) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def metadata():
    with open(os.path.join(MODELS, "training_metadata.json")) as fh:
        return json.load(fh)


def _median_input(bundle):
    cols = bundle["feature_cols"]
    medians = bundle["feature_medians"]
    return pd.DataFrame([[medians[c] for c in cols]], columns=cols)


def _predict(bundle, X):
    raw = bundle["pipeline"].predict_proba(X)[:, 1]
    return bundle["isotonic"].predict(raw)


def test_bundle_has_expected_keys(bundle):
    for key in ("pipeline", "isotonic", "feature_cols", "feature_medians"):
        assert key in bundle, f"bundle missing key: {key}"


def test_prediction_at_cohort_median_is_finite_and_in_unit_interval(bundle):
    p = float(_predict(bundle, _median_input(bundle))[0])
    assert np.isfinite(p)
    assert 0.0 <= p <= 1.0


def test_calibrated_risk_increases_monotonically_with_age(bundle):
    base = _median_input(bundle)
    risks = []
    for age in (35, 45, 55, 65, 75):
        x = base.copy()
        x["age"] = age
        risks.append(float(_predict(bundle, x)[0]))
    assert all(b >= a for a, b in zip(risks, risks[1:])), risks


def test_feature_order_matches_between_joblib_and_scaler_json(bundle, scaler_sidecar):
    assert bundle["feature_cols"] == scaler_sidecar["feature_order"]


def test_isotonic_lut_is_monotone_non_decreasing(iso_sidecar):
    ys = np.asarray(iso_sidecar["y"])
    diffs = np.diff(ys)
    assert np.all(diffs >= -1e-12), f"min diff in LUT: {diffs.min():.3e}"


def test_isotonic_lut_outputs_are_probabilities(iso_sidecar):
    ys = np.asarray(iso_sidecar["y"])
    assert ys.min() >= 0.0
    assert ys.max() <= 1.0


def test_pipeline_scaler_matches_scaler_sidecar(bundle, scaler_sidecar):
    scaler = bundle["pipeline"].named_steps["scaler"]
    np.testing.assert_allclose(scaler.mean_, scaler_sidecar["mean"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(scaler.scale_, scaler_sidecar["scale"], rtol=0, atol=1e-9)


def test_metadata_reports_finite_auc_and_brier(metadata):
    m = metadata["metrics"]
    assert 0.5 < m["auc_cv5"] <= 1.0
    assert 0.0 <= m["brier_cv5_calibrated"] <= 0.25
    # Calibration should improve Brier or at worst match.
    assert m["brier_cv5_calibrated"] <= m["brier_cv5_uncalibrated"] + 1e-6


def test_median_imputer_present_in_pipeline(bundle):
    """Guards against accidentally dropping the imputer (which would re-introduce
    the CV leakage bug fixed in this branch)."""
    steps = dict(bundle["pipeline"].named_steps)
    assert "imputer" in steps
    assert steps["imputer"].strategy == "median"
