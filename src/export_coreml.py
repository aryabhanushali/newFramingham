"""
Export the v1 risk model as a Core ML tree ensemble.

This script consumes the artifacts produced by `src/train_hk.py` — it does
NOT retrain. That's the whole point: one Python fit produces both the joblib
(used by the Streamlit web demo) and the Core ML bundle (used by the iOS
app). Train once, deploy twice.

The on-device inference path is:

    feature vector (HealthKit reads)
            │
            ▼  scaler.json (mean / scale arrays)
        z-scored vector
            │
            ▼  CVDRiskModel.mlmodel  (GradientBoosting tree ensemble)
        raw probability
            │
            ▼  isotonic_calibration.json  (101-point piecewise-linear LUT)
        calibrated 10-year CVD risk
"""

import os
import sys
import warnings

import joblib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ROOT = os.path.dirname(HERE)
MODELS_DIR = os.path.join(ROOT, "models")


def _add_tree(builder, tree_id, tree_, learning_rate):
    """Walk an sklearn tree and add every node to the Core ML builder."""
    feat = tree_.feature
    thr = tree_.threshold
    left = tree_.children_left
    right = tree_.children_right
    value = tree_.value.squeeze(axis=(1, 2))  # binary GBT: (n_nodes,)

    for node_id in range(tree_.node_count):
        if left[node_id] == -1 and right[node_id] == -1:
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


def _build_tree_ensemble(gbt, scaler, feature_cols):
    from coremltools.models import datatypes
    from coremltools.models.tree_ensemble import TreeEnsembleClassifier

    n_features = len(feature_cols)
    builder = TreeEnsembleClassifier(
        features=[("features", datatypes.Array(n_features))],
        class_labels=[0, 1],
        output_features=[
            ("classLabel", datatypes.Int64()),
            ("classProbability", datatypes.Dictionary(datatypes.Int64())),
        ],
    )

    init_pred = float(gbt.init_.class_prior_[1])
    init_logit = float(np.log(init_pred / (1.0 - init_pred)))
    builder.set_default_prediction_value([init_logit])

    learning_rate = gbt.learning_rate
    for tree_id, estimator in enumerate(gbt.estimators_[:, 0]):
        _add_tree(builder, tree_id, estimator.tree_, learning_rate)

    builder.set_post_evaluation_transform(
        "Classification_SoftMaxWithZeroClassReference"
    )

    spec = builder.spec
    spec.description.metadata.shortDescription = (
        "10-year cardiovascular disease risk from passive Apple Watch signals. "
        "Trained on NHANES 2011-2012, ages 30-74. Inputs are activity, sleep, "
        "and demographics; heart-rate inputs are read on-device but not yet in "
        "the trained feature set (v2)."
    )
    spec.description.metadata.author = "newframingham"
    spec.description.metadata.versionString = "1.0.0"
    spec.description.metadata.userDefined["framework"] = \
        "scikit-learn GradientBoosting"
    spec.description.metadata.userDefined["feature_order"] = \
        ",".join(feature_cols)
    spec.description.metadata.userDefined["scaler_mean"] = \
        ",".join(f"{m:.8g}" for m in scaler.mean_.tolist())
    spec.description.metadata.userDefined["scaler_scale"] = \
        ",".join(f"{s:.8g}" for s in scaler.scale_.tolist())
    return spec


def main():
    warnings.filterwarnings("ignore")

    bundle_path = os.path.join(MODELS_DIR, "cvd_risk_v1.joblib")
    print(f"=== Loading trained bundle: {bundle_path} ===")
    bundle = joblib.load(bundle_path)
    pipeline = bundle["pipeline"]
    feature_cols = bundle["feature_cols"]

    scaler = pipeline.named_steps["scaler"]
    gbt = pipeline.named_steps["gbt"]

    print(f"  trees:    {len(gbt.estimators_)}")
    print(f"  features: {len(feature_cols)}")

    print("\n=== Building Core ML tree ensemble ===")
    spec = _build_tree_ensemble(gbt, scaler, feature_cols)

    from coremltools.models import MLModel
    mlmodel = MLModel(spec)
    mlmodel_path = os.path.join(MODELS_DIR, "CVDRiskModel.mlmodel")
    mlmodel.save(mlmodel_path)
    size_kb = os.path.getsize(mlmodel_path) / 1024.0
    print(f"  Saved {mlmodel_path}  ({size_kb:.1f} KB)")

    print("\nReminder: scaler.json and isotonic_calibration.json are produced")
    print("by src/train_hk.py — copy all three into the iOS Resources folder.")


if __name__ == "__main__":
    main()
