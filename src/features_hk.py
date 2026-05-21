"""
Derive HealthKit-shaped daily features from NHANES PAXDAY (per-day accelerometer
summary) and merge with demographics + Framingham labels.

Mapping NHANES → HealthKit (justification):
  PAXAISMD (activity intensity sum/day)  -> proxy for active energy kcal/day
                                            and step count (counts scale linearly
                                            with steps for ambulatory adults).
  PAXMTSD  (mean intensity)              -> exercise-time proxy (minutes above
                                            brisk threshold per day).
  PAXSWMD  (sleep wear minutes)          -> sleep duration.
  PAXWWMD  (wake wear minutes)           -> awake / stand-time proxy.
  PAXNWMD  (non-wear)                    -> excluded from analysis (compliance).
  PAXLXSD  (lux sum)                     -> circadian light exposure.
  PAXVMD   (valid wear minutes)          -> day-validity filter (>= 600 min).

Scaling constants picked so values land in HealthKit-plausible ranges
(steps ~5-12k/day, kcal ~200-600/day) without overclaiming individual accuracy
— this is a population-level proxy, made explicit in the README.
"""

import numpy as np
import pandas as pd

from healthkit_schema import training_feature_names


# Display calibration — back-fit so population medians match published norms:
#   median PAXAISMD ≈ 2.8M counts/day  ->  ~6,500 steps/day  (CDC adult median)
#   median PAXAISMD ≈ 2.8M counts/day  ->  ~350 active kcal  (Apple Activity median)
#   median PAXMTSD  ≈ 12k counts/min   ->  ~15 exercise min  (Apple median is ~30)
# Scaling doesn't affect model output (StandardScaler normalizes) but makes the
# Health-app UI honest about what each number represents.
COUNTS_PER_STEP        = 430.0           # PAXAISMD counts -> step count
COUNTS_PER_KCAL        = 8000.0          # PAXAISMD counts -> active kcal
EXERCISE_BASELINE      = 8000.0          # PAXMTSD baseline for "moderate" activity
EXERCISE_PER_MIN       = 100.0           # PAXMTSD units above baseline per exercise min
EXERCISE_MAX_MIN       = 60.0
MIN_VALID_WEAR_MIN     = 600             # WHO/NHANES standard


def _safe_std(x):
    s = float(np.nanstd(x))
    return s if np.isfinite(s) else 0.0


def _person_features(person):
    """Compute one participant's HealthKit-shaped feature row."""
    activity = person["PAXAISMD"].dropna().astype(float).values
    intensity = person["PAXMTSD"].dropna().astype(float).values
    sleep_min = person["PAXSWMD"].dropna().astype(float).values
    wake_min = person["PAXWWMD"].dropna().astype(float).values
    lux = person["PAXLXSD"].dropna().astype(float).values

    if len(activity) == 0:
        return None

    # --- activity & energy ---
    avg_steps = float(np.mean(activity)) / COUNTS_PER_STEP
    avg_kcal = float(np.mean(activity)) / COUNTS_PER_KCAL
    peak_kcal = float(np.max(activity)) / COUNTS_PER_KCAL
    if len(intensity):
        per_day_min = np.clip(
            (intensity - EXERCISE_BASELINE) / EXERCISE_PER_MIN,
            0.0, EXERCISE_MAX_MIN,
        )
        exercise_min = float(np.mean(per_day_min))
    else:
        exercise_min = 0.0

    act_std = _safe_std(activity)
    regularity = 1.0 / (1.0 + act_std / (np.mean(activity) + 1.0))
    low_q = np.quantile(activity, 0.25) if len(activity) > 3 else activity.min()
    low_day_ratio = float(np.mean(activity < low_q))

    sedentary = max(0.0, float(np.mean(wake_min)) - exercise_min) \
        if len(wake_min) else 0.0

    # --- sleep ---
    avg_sleep_hours = float(np.mean(sleep_min)) / 60.0 if len(sleep_min) else 0.0
    sleep_reg = 1.0 / (1.0 + _safe_std(sleep_min) / 60.0)

    # --- circadian ---
    circadian_lux = float(np.mean(lux)) if len(lux) else 0.0

    return {
        "avg_daily_step_count":      avg_steps,
        "avg_daily_active_energy":   avg_kcal,
        "peak_intensity":            peak_kcal,
        "avg_daily_exercise_min":    exercise_min,
        "activity_regularity":       regularity,
        "low_activity_day_ratio":    low_day_ratio,
        "sedentary_minutes":         sedentary,
        "avg_sleep_hours":           avg_sleep_hours,
        "sleep_regularity":          sleep_reg,
        "circadian_light_exposure":  circadian_lux,
        "n_valid_days":              int(len(activity)),
    }


def build_hk_features(paxday):
    """Return one row per SEQN with HealthKit-shaped features."""
    df = paxday[paxday["PAXVMD"] >= MIN_VALID_WEAR_MIN].copy()
    rows = []
    for seqn, person in df.groupby("SEQN"):
        feats = _person_features(person)
        if feats is None:
            continue
        feats["SEQN"] = seqn
        rows.append(feats)
    return pd.DataFrame(rows)


def build_model_dataset(clinical_scored, paxday):
    """
    Merge demographics + Framingham label with HK-shaped features.

    Returns a dataframe with one row per participant ready for training:
      SEQN, age, biological_sex, <hk features>, framingham_risk, high_risk
    """
    hk = build_hk_features(paxday)

    demo = clinical_scored[["SEQN", "RIDAGEYR", "RIAGENDR",
                            "framingham_risk", "high_risk"]].dropna(
        subset=["framingham_risk"]
    ).copy()
    demo = demo.rename(columns={
        "RIDAGEYR": "age",
        "RIAGENDR": "biological_sex_raw",
    })
    # NHANES 1=male, 2=female  -> HealthKit-ish 1=male, 0=female
    demo["biological_sex"] = (demo["biological_sex_raw"] == 1).astype(int)
    demo = demo.drop(columns=["biological_sex_raw"])

    dataset = demo.merge(hk, on="SEQN", how="inner")

    train_cols = training_feature_names()
    available = [c for c in train_cols if c in dataset.columns]
    missing = [c for c in train_cols if c not in dataset.columns]
    if missing:
        print(f"[features_hk] training cols not produced: {missing}")

    print(f"[features_hk] {dataset.shape[0]} participants, "
          f"{len(available)} v1 features")
    print(f"[features_hk] high-risk prevalence: "
          f"{dataset['high_risk'].mean()*100:.1f}%")

    return dataset, available
