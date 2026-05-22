"""
Train the v1 HealthKit-input CVD risk model.

Pipeline (single fit, multiple deployment formats):

    SimpleImputer(median) -> StandardScaler -> GradientBoostingClassifier
                                                          │
                                                          ▼
                                                IsotonicRegression
                                                (fit on OOF predictions)

Why the imputer is *inside* the Pipeline: it must be fit on each cross-val
training fold, not on the full dataset, or test-fold information leaks into
the imputed values used during training.

The joblib bundle contains:
  - `pipeline`   — imputer + scaler + GBT (raw probability)
  - `isotonic`   — calibrator applied to pipeline output
  - `feature_cols`, `feature_medians`, and training metadata

The iOS Core ML export (`src/export_coreml.py`) consumes the *same* joblib so
the on-device tree ensemble + isotonic LUT match the Python prediction
bit-for-bit (modulo float32 rounding inside Core ML).

Outputs:
  models/cvd_risk_v1.joblib           — full Python bundle
  models/scaler.json                  — feature_order + scaler params + medians
  models/isotonic_calibration.json    — piecewise-linear calibration LUT
  models/training_metadata.json       — AUC, Brier, cohort info
  results/calibration.png             — ROC + reliability + score-distribution
"""

import json
import os
import sys
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (brier_score_loss, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from features_hk import build_model_dataset                  # noqa: E402
from framingham import compute_framingham                    # noqa: E402
from healthkit_schema import training_feature_names          # noqa: E402
from load_data import load_all                               # noqa: E402

ROOT = os.path.dirname(HERE)
MODELS_DIR = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# NHANES floating-point missingness sentinel: a tiny positive subnormal that
# survives a few merges and is not caught by isna(). Drop to NaN so the
# imputer sees it.
NHANES_NAN_SENTINEL = 5.397605e-79

# Tree-ensemble hyper-parameters: kept modest so the Core ML export stays
# small (≤100 KB) and inference is real-time on A14+. Increasing depth or
# n_estimators gave <0.005 AUC gains on this cohort and tripled model size.
GBT_PARAMS = dict(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.85,
    random_state=42,
)


def _prep_xy(dataset, feature_cols):
    X = dataset[feature_cols].copy().replace(NHANES_NAN_SENTINEL, np.nan)
    y = dataset["high_risk"].astype(int)
    return X, y


def _build_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("gbt", GradientBoostingClassifier(**GBT_PARAMS)),
    ])


def _isotonic_payload(iso, proba_cv):
    """Serialize the isotonic regressor as a 101-point LUT for Swift."""
    xs = np.linspace(0.0, 1.0, 101)
    ys = iso.predict(xs)
    return {
        "method": "isotonic",
        "x": xs.tolist(),
        "y": ys.tolist(),
        "domain_min": float(proba_cv.min()),
        "domain_max": float(proba_cv.max()),
    }


def _plot_diagnostics(y, proba, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    fpr, tpr, _ = roc_curve(y, proba)
    auc = roc_auc_score(y, proba)
    axes[0].plot(fpr, tpr, color="#FF3B30", linewidth=2)
    axes[0].plot([0, 1], [0, 1], color="#8E8E93", linestyle="--", linewidth=1)
    axes[0].set_title(f"ROC  •  AUC = {auc:.3f}")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].grid(alpha=0.3)

    frac_pos, mean_pred = calibration_curve(y, proba, n_bins=10, strategy="quantile")
    axes[1].plot([0, 1], [0, 1], color="#8E8E93", linestyle="--", linewidth=1)
    axes[1].plot(mean_pred, frac_pos, marker="o", color="#FF3B30", linewidth=2)
    axes[1].set_title(f"Calibration  •  Brier = {brier_score_loss(y, proba):.3f}")
    axes[1].set_xlabel("Predicted probability")
    axes[1].set_ylabel("Observed frequency")
    axes[1].grid(alpha=0.3)

    axes[2].hist(proba[y == 0], bins=40, alpha=0.6,
                 label="Low risk", color="#34C759")
    axes[2].hist(proba[y == 1], bins=40, alpha=0.6,
                 label="High risk", color="#FF3B30")
    axes[2].axvline(0.10, color="#8E8E93", linestyle="--",
                    label="10% clinical threshold")
    axes[2].set_title("Predicted 10-year CVD risk")
    axes[2].set_xlabel("Predicted probability")
    axes[2].set_ylabel("Participants")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    print("=== Loading NHANES ===")
    clinical, _, paxday, _ = load_all()

    print("\n=== Computing Framingham labels ===")
    scored = compute_framingham(clinical)

    print("\n=== Building HealthKit feature set ===")
    dataset, _ = build_model_dataset(scored, paxday)
    feature_cols = [c for c in training_feature_names() if c in dataset.columns]

    X, y = _prep_xy(dataset, feature_cols)
    print(f"\nFeature matrix: {X.shape}, label mean: {y.mean():.3f}")

    print("\n=== 5-fold OOF predictions (no leakage: imputer inside Pipeline) ===")
    pipeline = _build_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    proba_cv = cross_val_predict(pipeline, X, y, cv=cv,
                                 method="predict_proba")[:, 1]

    auc = roc_auc_score(y, proba_cv)
    brier_raw = brier_score_loss(y, proba_cv)
    print(f"5-fold AUC:               {auc:.3f}")
    print(f"5-fold Brier (uncalib):   {brier_raw:.3f}")

    print("\n=== Fitting isotonic calibrator on OOF predictions ===")
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(proba_cv, y.astype(float))
    proba_cal = iso.predict(proba_cv)
    brier_cal = brier_score_loss(y, proba_cal)
    print(f"5-fold Brier (calibrated): {brier_cal:.3f}")

    print("\n=== Final fit on full data ===")
    pipeline.fit(X, y)

    pred = (proba_cal >= 0.5).astype(int)
    print("\nClassification report (calibrated OOF predictions):")
    print(classification_report(y, pred, target_names=["Low risk", "High risk"]))
    print(f"Confusion matrix:\n{confusion_matrix(y, pred)}")

    diag_path = os.path.join(RESULTS_DIR, "calibration.png")
    _plot_diagnostics(y, proba_cal, diag_path)
    print(f"\nDiagnostics: {diag_path}")

    # ---- Persist artifacts ----

    # Compute medians on the imputed training matrix so the Streamlit/iOS
    # defaults match what the model actually sees post-imputation.
    X_imputed = pipeline.named_steps["imputer"].transform(X)
    medians = pd.DataFrame(X_imputed, columns=feature_cols).median().to_dict()

    bundle = {
        "pipeline": pipeline,
        "isotonic": iso,
        "feature_cols": feature_cols,
        "feature_medians": medians,
    }
    joblib.dump(bundle, os.path.join(MODELS_DIR, "cvd_risk_v1.joblib"))

    scaler = pipeline.named_steps["scaler"]
    with open(os.path.join(MODELS_DIR, "scaler.json"), "w") as fh:
        json.dump({
            "feature_order": feature_cols,
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "medians": {k: float(v) for k, v in medians.items()},
        }, fh, indent=2)

    with open(os.path.join(MODELS_DIR, "isotonic_calibration.json"), "w") as fh:
        json.dump(_isotonic_payload(iso, proba_cv), fh, indent=2)

    metadata = {
        "model_name": "CVDRiskModel",
        "model_version": "1.0.0",
        "framework": "scikit-learn GradientBoostingClassifier",
        "calibration": "isotonic (1-fit, OOF-trained)",
        "training_cohort": "NHANES 2011-2012, ages 30-74",
        "training_n": int(X.shape[0]),
        "high_risk_threshold": 0.10,
        "metrics": {
            "auc_cv5": float(auc),
            "brier_cv5_uncalibrated": float(brier_raw),
            "brier_cv5_calibrated": float(brier_cal),
            "high_risk_prevalence": float(y.mean()),
        },
        "hyperparameters": {
            **GBT_PARAMS,
            "imputer": "median",
            "scaler": "StandardScaler",
        },
        "feature_cols": feature_cols,
        "feature_medians": {k: float(v) for k, v in medians.items()},
    }
    with open(os.path.join(MODELS_DIR, "training_metadata.json"), "w") as fh:
        json.dump(metadata, fh, indent=2)

    print("Wrote: cvd_risk_v1.joblib, scaler.json, "
          "isotonic_calibration.json, training_metadata.json")
    return bundle, metadata


if __name__ == "__main__":
    main()
