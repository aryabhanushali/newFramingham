"""
HealthKit-style schema for cardiovascular risk features.

Maps NHANES wearable signals to Apple HealthKit HKQuantityType / HKCategoryType
identifiers so that:
  (a) the Python pipeline is named in the same vocabulary as the iOS app, and
  (b) the on-device app can populate the same feature vector from a real
      Apple Watch using HKHealthStore queries.
"""

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# HealthKit identifiers we read on-device (iOS side mirrors this list).
# ---------------------------------------------------------------------------
HK_STEP_COUNT             = "HKQuantityTypeIdentifierStepCount"
HK_ACTIVE_ENERGY          = "HKQuantityTypeIdentifierActiveEnergyBurned"
HK_BASAL_ENERGY           = "HKQuantityTypeIdentifierBasalEnergyBurned"
HK_EXERCISE_TIME          = "HKQuantityTypeIdentifierAppleExerciseTime"
HK_STAND_TIME             = "HKQuantityTypeIdentifierAppleStandTime"
HK_DISTANCE_WALKING       = "HKQuantityTypeIdentifierDistanceWalkingRunning"

HK_RESTING_HR             = "HKQuantityTypeIdentifierRestingHeartRate"
HK_WALKING_HR             = "HKQuantityTypeIdentifierWalkingHeartRateAverage"
HK_HRV_SDNN               = "HKQuantityTypeIdentifierHeartRateVariabilitySDNN"
HK_VO2_MAX                = "HKQuantityTypeIdentifierVO2Max"

HK_SLEEP_ANALYSIS         = "HKCategoryTypeIdentifierSleepAnalysis"

HK_AGE                    = "HKCharacteristicTypeIdentifierDateOfBirth"
HK_SEX                    = "HKCharacteristicTypeIdentifierBiologicalSex"


@dataclass(frozen=True)
class FeatureSpec:
    """One model input feature, expressed in HealthKit-native terms."""
    name: str                # feature column name used by the model
    hk_identifier: str       # HealthKit type the iOS app queries
    unit: str                # HealthKit unit string (count, kcal, min, bpm, ...)
    aggregation: str         # daily/avg7d/avg30d/regularity
    source: str              # "v1_nhanes" or "v2_watch" (whether trained on this signal)
    description: str         # short clinician-readable description


# ---------------------------------------------------------------------------
# Feature catalogue. Order is the canonical model input order.
# Anything tagged source="v2_watch" is read on-device but not yet trained on;
# the iOS app still surfaces it in the UI for context.
# ---------------------------------------------------------------------------
FEATURES = [
    # --- demographics (HK characteristics) ---
    FeatureSpec("age",                 HK_AGE,  "year",  "static",     "v1_nhanes",
                "Age in years"),
    FeatureSpec("biological_sex",      HK_SEX,  "enum",  "static",     "v1_nhanes",
                "0 = female, 1 = male"),

    # --- activity & energy (HKQuantityType) ---
    FeatureSpec("avg_daily_active_energy",   HK_ACTIVE_ENERGY,    "kcal",    "avg7d",      "v1_nhanes",
                "Mean daily active energy burned"),
    FeatureSpec("avg_daily_step_count",      HK_STEP_COUNT,       "count",   "avg7d",      "v1_nhanes",
                "Mean daily step count (estimated from accelerometer counts)"),
    FeatureSpec("avg_daily_exercise_min",    HK_EXERCISE_TIME,    "min",     "avg7d",      "v1_nhanes",
                "Minutes per day above brisk-walking intensity"),
    FeatureSpec("activity_regularity",       HK_ACTIVE_ENERGY,    "ratio",   "regularity", "v1_nhanes",
                "1 / (1 + std of daily activity) — higher = more consistent"),
    FeatureSpec("low_activity_day_ratio",    HK_ACTIVE_ENERGY,    "ratio",   "regularity", "v1_nhanes",
                "Fraction of recorded days below personal 25th percentile"),
    FeatureSpec("peak_intensity",            HK_ACTIVE_ENERGY,    "kcal",    "max",        "v1_nhanes",
                "Peak daily activity intensity"),
    FeatureSpec("sedentary_minutes",         HK_STAND_TIME,       "min",     "avg7d",      "v1_nhanes",
                "Mean daily sedentary minutes (inverse of stand-time)"),

    # --- sleep (HKCategoryType) ---
    FeatureSpec("avg_sleep_hours",           HK_SLEEP_ANALYSIS,   "hour",    "avg7d",      "v1_nhanes",
                "Mean nightly sleep duration"),
    FeatureSpec("sleep_regularity",          HK_SLEEP_ANALYSIS,   "ratio",   "regularity", "v1_nhanes",
                "1 / (1 + std of nightly sleep duration)"),

    # --- circadian (derived from light exposure, HK proxy) ---
    FeatureSpec("circadian_light_exposure",  HK_ACTIVE_ENERGY,    "lux",     "avg7d",      "v1_nhanes",
                "Mean ambient light exposure during wake hours"),

    # --- heart (read on-device, not in NHANES v1 training) ---
    FeatureSpec("resting_heart_rate",        HK_RESTING_HR,       "bpm",     "avg7d",      "v2_watch",
                "Resting HR — read from Apple Watch, displayed only in v1"),
    FeatureSpec("hrv_sdnn",                  HK_HRV_SDNN,         "ms",      "avg7d",      "v2_watch",
                "HRV SDNN — read from Apple Watch, displayed only in v1"),
    FeatureSpec("vo2_max_estimate",          HK_VO2_MAX,          "ml/kg/min", "avg30d",   "v2_watch",
                "Cardio fitness — read from Apple Watch, displayed only in v1"),
]


def training_feature_names():
    """Feature columns the v1 model actually trains on (excludes v2_watch)."""
    return [f.name for f in FEATURES if f.source == "v1_nhanes"]


def display_feature_names():
    """All features the iOS app shows in the UI (training + watch-only)."""
    return [f.name for f in FEATURES]


def feature_by_name(name):
    for f in FEATURES:
        if f.name == name:
            return f
    raise KeyError(name)
