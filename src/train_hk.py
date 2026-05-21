"""
Train the v1 HealthKit-input CVD risk model.

Pipeline:
  StandardScaler -> GradientBoostingClassifier  (sklearn, Core ML friendly)
  CalibratedClassifierCV (isotonic) wraps it for calibrated probabilities.

We use sklearn's GradientBoosting instead of XGBoost specifically because the
Core ML converter for native sklearn pipelines is the most stable and ships
the model as a single `.mlpackage` without external dependencies.

Outputs:
  models/cvd_risk_v1.joblib           — full sklearn pipeline (for Python)
  models/training_metadata.json       — feature names, AUC, calibration info
  results/calibration.png             — reliability + ROC plot
"""

import json
import os
import sys
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (brier_score_loss, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# allow `python src/train_hk.py` from project root
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


def _prep_xy(dataset, feature_cols):
    X = dataset[feature_cols].copy()
    # NHANES sentinel ≈ 0 floating point junk
    X = X.replace(5.397605e-79, np.nan)
    for col in X.columns:
        X[col] = X[col].fillna(X[col].median())
    y = dataset["high_risk"].astype(int)
    return X, y


def _build_pipeline():
    base = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        random_state=42,
    )
    # CalibratedClassifierCV gives us well-shaped probabilities, which matters
    # a lot for a Health-app-style "your 10yr risk is N%" UI.
    calibrated = CalibratedClassifierCV(
        estimator=base,
        method="isotonic",
        cv=5,
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", calibrated),
    ])


def _plot_diagnostics(y, proba, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # ROC
    fpr, tpr, _ = roc_curve(y, proba)
    auc = roc_auc_score(y, proba)
    axes[0].plot(fpr, tpr, color="#FF3B30", linewidth=2)
    axes[0].plot([0, 1], [0, 1], color="#8E8E93", linestyle="--", linewidth=1)
    axes[0].set_title(f"ROC  •  AUC = {auc:.3f}")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].grid(alpha=0.3)

    # Reliability / calibration curve
    frac_pos, mean_pred = calibration_curve(y, proba, n_bins=10, strategy="quantile")
    axes[1].plot([0, 1], [0, 1], color="#8E8E93", linestyle="--", linewidth=1)
    axes[1].plot(mean_pred, frac_pos, marker="o", color="#FF3B30", linewidth=2)
    axes[1].set_title(f"Calibration  •  Brier = {brier_score_loss(y, proba):.3f}")
    axes[1].set_xlabel("Predicted probability")
    axes[1].set_ylabel("Observed frequency")
    axes[1].grid(alpha=0.3)

    # Score distribution
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
    clinical, paxhd, paxday, paxhr = load_all()

    print("\n=== Computing Framingham labels ===")
    scored = compute_framingham(clinical)

    print("\n=== Building HealthKit feature set ===")
    dataset, feature_cols = build_model_dataset(scored, paxday)

    # Add demographics to feature list (already in dataset)
    all_train_cols = training_feature_names()
    feature_cols = [c for c in all_train_cols if c in dataset.columns]

    X, y = _prep_xy(dataset, feature_cols)
    print(f"\nFeature matrix: {X.shape}, label mean: {y.mean():.3f}")

    print("\n=== Cross-validation ===")
    pipeline = _build_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    proba_cv = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]

    auc = roc_auc_score(y, proba_cv)
    brier = brier_score_loss(y, proba_cv)
    print(f"5-fold AUC:  {auc:.3f}")
    print(f"5-fold Brier: {brier:.3f}")

    print("\n=== Final fit on full data ===")
    pipeline.fit(X, y)

    pred = (proba_cv >= 0.5).astype(int)
    print("\nClassification report (cross-val predictions):")
    print(classification_report(y, pred, target_names=["Low risk", "High risk"]))
    print(f"Confusion matrix:\n{confusion_matrix(y, pred)}")

    diag_path = os.path.join(RESULTS_DIR, "calibration.png")
    _plot_diagnostics(y, proba_cv, diag_path)
    print(f"\nDiagnostics written to {diag_path}")

    # Save pipeline
    model_path = os.path.join(MODELS_DIR, "cvd_risk_v1.joblib")
    joblib.dump({
        "pipeline": pipeline,
        "feature_cols": feature_cols,
        "feature_medians": X.median().to_dict(),
    }, model_path)
    print(f"Model written to {model_path}")

    # Metadata for the iOS app
    metadata = {
        "model_name": "CVDRiskModel",
        "model_version": "1.0.0",
        "framework": "scikit-learn GradientBoostingClassifier",
        "calibration": "isotonic, 5-fold",
        "training_cohort": "NHANES 2011-2012, ages 30-74",
        "training_n": int(X.shape[0]),
        "high_risk_threshold": 0.10,
        "metrics": {
            "auc_cv5": float(auc),
            "brier_cv5": float(brier),
            "high_risk_prevalence": float(y.mean()),
        },
        "feature_cols": feature_cols,
        "feature_medians": {k: float(v) for k, v in X.median().to_dict().items()},
    }
    meta_path = os.path.join(MODELS_DIR, "training_metadata.json")
    with open(meta_path, "w") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"Metadata written to {meta_path}")

    return pipeline, feature_cols, metadata


if __name__ == "__main__":
    main()
