"""
Export the v1 risk model as a Core ML tree ensemble + an isotonic LUT.

Why this design?
  CalibratedClassifierCV is not directly Core-ML convertible. So we ship the
  on-device inference as two cheap stages:

    1. CVDRiskTreeEnsemble.mlmodel
         Input  : MLMultiArray of length N_FEATURES (canonical order)
         Output : raw probability of high CVD risk from a GradientBoosting
                  tree ensemble.

    2. isotonic_calibration.json
         A monotone piecewise-linear lookup the iOS app applies to (1)
         before showing the probability to the user. Calibrated on the
         out-of-fold CV predictions of the *same* tree ensemble.

The pair behaves equivalently to the CalibratedClassifierCV pipeline in Python
but runs entirely on-device with no external dependencies.
"""

import json
import os
import sys
import warnings

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from features_hk import build_model_dataset                  # noqa: E402
from framingham import compute_framingham                    # noqa: E402
from healthkit_schema import training_feature_names          # noqa: E402
from load_data import load_all                               # noqa: E402

ROOT = os.path.dirname(HERE)
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def _build_features(dataset, cols):
    X = dataset[cols].copy().replace(5.397605e-79, np.nan)
    for c in X.columns:
        X[c] = X[c].fillna(X[c].median())
    return X, dataset["high_risk"].astype(int)


def _add_tree(builder, tree_id, tree_, learning_rate):
    """Walk an sklearn tree and add every node to the builder."""
    feat = tree_.feature
    thr = tree_.threshold
    left = tree_.children_left
    right = tree_.children_right
    value = tree_.value.squeeze(axis=(1, 2))  # regressor: shape (n_nodes,)

    for node_id in range(tree_.node_count):
        if left[node_id] == -1 and right[node_id] == -1:
            # leaf — contribution is learning_rate * value
            builder.add_leaf_node(
                tree_id, node_id,
                [float(learning_rate * value[node_id])],
            )
        else:
            builder.add_branch_node(
                tree_id, node_id,
                feature_index=int(feat[node_id]),
                feature_value=float(thr[node_id]),
                branch_mode="BranchOnValueLessThanEqual",
                true_child_id=int(left[node_id]),
                false_child_id=int(right[node_id]),
                missing_value_tracks_true_child=True,
            )


def _build_tree_ensemble_model(gbt, scaler, feature_cols):
    """Construct a Core ML TreeEnsembleClassifier from a fitted GBT + StandardScaler."""
    from coremltools.models import datatypes
    from coremltools.models.tree_ensemble import TreeEnsembleClassifier

    n_features = len(feature_cols)

    builder = TreeEnsembleClassifier(
        features=[("features", datatypes.Array(n_features))],
        class_labels=[0, 1],
        output_features=[("classLabel", datatypes.Int64()),
                         ("classProbability", datatypes.Dictionary(datatypes.Int64()))],
    )

    # init prediction is the log-odds of the positive class
    init_pred = float(gbt.init_.class_prior_[1])
    init_logit = float(np.log(init_pred / (1.0 - init_pred)))
    builder.set_default_prediction_value([init_logit])

    # The GBC stores estimators_ as shape (n_estimators, K). For binary K=1.
    learning_rate = gbt.learning_rate
    for tree_id, estimator in enumerate(gbt.estimators_[:, 0]):
        _add_tree(builder, tree_id, estimator.tree_, learning_rate)

    # The classifier post-eval transform turns log-odds into probabilities
    # Binary GBT: leaves output log-odds for class 1, class 0 reference = 0.
    # SoftMaxWithZeroClassReference applies softmax over [0, score_class1].
    builder.set_post_evaluation_transform("Classification_SoftMaxWithZeroClassReference")

    spec = builder.spec
    spec.description.metadata.shortDescription = (
        "10-year cardiovascular disease risk from passive Apple Watch signals. "
        "Trained on NHANES 2011-2012, ages 30-74. v1 inputs are activity, sleep, "
        "and demographics; heart-rate inputs are read on-device but not yet in "
        "the trained feature set."
    )
    spec.description.metadata.author = "newframingham"
    spec.description.metadata.versionString = "1.0.0"
    spec.description.metadata.userDefined["framework"] = "scikit-learn GradientBoosting"
    spec.description.metadata.userDefined["feature_order"] = ",".join(feature_cols)
    spec.description.metadata.userDefined["scaler_mean"] = \
        ",".join(f"{m:.8g}" for m in scaler.mean_.tolist())
    spec.description.metadata.userDefined["scaler_scale"] = \
        ",".join(f"{s:.8g}" for s in scaler.scale_.tolist())
    # We also bake the scaler params into a side-car JSON so Swift can load
    # them without parsing the spec's userDefined dict.
    return spec


def _fit_and_serialize_isotonic(y, proba_cv):
    """Fit IsotonicRegression and return its piecewise-linear knot points."""
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(proba_cv, y.astype(float))
    xs = np.linspace(0.0, 1.0, 101)
    ys = iso.predict(xs)
    return {
        "method": "isotonic",
        "x": xs.tolist(),
        "y": ys.tolist(),
        "domain_min": float(proba_cv.min()),
        "domain_max": float(proba_cv.max()),
    }


def main():
    warnings.filterwarnings("ignore")

    print("=== Loading + scoring ===")
    clinical, _, paxday, _ = load_all()
    scored = compute_framingham(clinical)
    dataset, feature_cols = build_model_dataset(scored, paxday)
    feature_cols = [c for c in training_feature_names() if c in dataset.columns]
    X, y = _build_features(dataset, feature_cols)

    print(f"n={len(X)}, d={X.shape[1]}, prevalence={y.mean():.3f}")

    print("\n=== Fitting scaler + uncalibrated GBT ===")
    scaler = StandardScaler().fit(X.values)
    Xs = scaler.transform(X.values)

    gbt = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        random_state=42,
    )

    print("=== OOF predictions for isotonic calibration ===")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    proba_cv = cross_val_predict(gbt, Xs, y, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, proba_cv)
    brier = brier_score_loss(y, proba_cv)
    print(f"  Tree ensemble OOF AUC:  {auc:.3f}")
    print(f"  Tree ensemble OOF Brier (pre-calib): {brier:.3f}")

    gbt.fit(Xs, y)

    print("\n=== Fitting isotonic calibrator on OOF predictions ===")
    iso_payload = _fit_and_serialize_isotonic(y, proba_cv)
    # estimate post-calibration brier
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(proba_cv, y.astype(float))
    post = iso.predict(proba_cv)
    print(f"  Post-calibration Brier: {brier_score_loss(y, post):.3f}")

    iso_path = os.path.join(MODELS_DIR, "isotonic_calibration.json")
    with open(iso_path, "w") as fh:
        json.dump(iso_payload, fh, indent=2)
    print(f"  Saved {iso_path}")

    print("\n=== Building Core ML tree ensemble ===")
    spec = _build_tree_ensemble_model(gbt, scaler, feature_cols)

    from coremltools.models import MLModel
    mlmodel = MLModel(spec)
    mlmodel_path = os.path.join(MODELS_DIR, "CVDRiskModel.mlmodel")
    mlmodel.save(mlmodel_path)
    print(f"  Saved {mlmodel_path}")

    # Scaler params side-car for Swift
    scaler_path = os.path.join(MODELS_DIR, "scaler.json")
    with open(scaler_path, "w") as fh:
        json.dump({
            "feature_order": feature_cols,
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "medians": X.median().to_dict(),
        }, fh, indent=2)
    print(f"  Saved {scaler_path}")


if __name__ == "__main__":
    main()
