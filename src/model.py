import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, classification_report,
                             confusion_matrix, brier_score_loss)
from sklearn.pipeline import Pipeline
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

FEATURE_COLS = [
    'RIDAGEYR', 'RIAGENDR',
    'mean_activity', 'std_activity', 'min_activity', 'max_activity',
    'activity_regularity', 'low_activity_days', 'high_activity_days',
    'mean_intensity', 'std_intensity',
    'mean_wake_mins', 'mean_sleep_mins', 'sleep_ratio',
    'mean_nonwear_mins', 'mean_lux', 'std_lux', 'n_valid_days'
]

def prepare_xy(dataset):
    available = [c for c in FEATURE_COLS if c in dataset.columns]
    X = dataset[available].copy()
    y = dataset['high_risk'].copy()

    # Replace near-zero floats (NHANES missing sentinel) with NaN then median
    X = X.replace(5.397605e-79, np.nan)
    for col in X.columns:
        X[col] = X[col].fillna(X[col].median())

    return X, y

def train_and_evaluate(dataset):
    X, y = prepare_xy(dataset)

    print(f"Training on {X.shape[0]} participants, {X.shape[1]} features")
    print(f"Class balance: {y.mean()*100:.1f}% high risk\n")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y==0).sum() / (y==1).sum(),
        eval_metric='auc',
        random_state=42,
        verbosity=0
    )

    # 5-fold cross validation (stratified — preserves class ratio)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')

    print("=== 5-Fold Cross-Validation Results ===")
    print(f"AUC scores: {[round(s,3) for s in auc_scores]}")
    print(f"Mean AUC:   {auc_scores.mean():.3f} (+/- {auc_scores.std():.3f})")

    # Train final model on full data
    model.fit(X, y)

    # In-sample predictions for confusion matrix
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    print(f"\nBrier score: {brier_score_loss(y, y_proba):.3f}")
    print(f"(0 = perfect, 0.25 = random)\n")
    print("Classification report:")
    print(classification_report(y, y_pred, target_names=['Low risk','High risk']))

    return model, X, y, y_proba

def plot_results(model, X, y, y_proba):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Feature importance
    importance = pd.Series(model.feature_importances_, index=X.columns)
    importance.sort_values().plot(kind='barh', ax=axes[0], color='steelblue')
    axes[0].set_title('Feature importance')
    axes[0].set_xlabel('XGBoost importance score')

    # 2. Confusion matrix
    cm = confusion_matrix(y, model.predict(X))
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[1],
                xticklabels=['Low','High'],
                yticklabels=['Low','High'],
                cmap='Blues')
    axes[1].set_title('Confusion matrix')
    axes[1].set_ylabel('Actual')
    axes[1].set_xlabel('Predicted')

    # 3. Risk score distribution by true label
    axes[2].hist(y_proba[y==0], bins=40, alpha=0.6,
                 label='Low risk', color='steelblue')
    axes[2].hist(y_proba[y==1], bins=40, alpha=0.6,
                 label='High risk', color='coral')
    axes[2].axvline(0.5, color='black', linestyle='--', label='Decision boundary')
    axes[2].set_title('Predicted probability by true label')
    axes[2].set_xlabel('Predicted probability of high risk')
    axes[2].set_ylabel('Count')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('../results/model_results.png', dpi=150)
    plt.show()

    return importance