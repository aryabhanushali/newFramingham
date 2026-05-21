# Adaptive Cardiovascular Health Intelligence

**Predicting 10-year cardiovascular disease risk from passive Apple Watch-style
signals — entirely on-device, with no blood draws or clinical visits.**

A two-layer project:

1. **Python core** — ingests NHANES 2011-2012 accelerometer data, derives a
   HealthKit-shaped feature set, trains a calibrated tree-ensemble against
   Framingham-derived labels, and exports a Core ML model.
2. **iOS app** (`ios/`) — SwiftUI + HealthKit + Core ML. Reads Apple Watch
   metrics, runs the same model locally, and renders an Apple-Health-style
   risk card with a contextual insight.

---

## Why this matters for an on-device health story

The Framingham score requires cholesterol, blood pressure, smoking and diabetes
history — i.e. a clinic visit. The same 10-year risk can be approximated from
**activity, sleep, and demographics alone**, which are exactly the signals an
Apple Watch passively collects. That makes population-scale cardiovascular
triage possible without sending anyone to a lab.

The training story:

```
NHANES wearable signals  →  HealthKit-shaped features  →  CalibratedClassifierCV
        (passive)                  (on-device-ready)         (5-fold isotonic)
                ↓                                                    ↓
                                                          Framingham 10-yr risk
                                                            (clinical labels)
```

The model is trained to *predict the Framingham score* from passive inputs,
so the iOS app outputs a number that is directly comparable to the gold-standard
clinical estimate — without ever needing a needle.

---

## Results

|                          | Value |
|--------------------------|------:|
| Participants (NHANES 2011-2012, ages 30-74) | 2,967 |
| Tree-ensemble OOF AUC    | **0.92** |
| Brier score, post-calibration | **0.11** |
| Core ML model size       | 64 KB |
| Inference (Apple A14 +)  | sub-millisecond |

The calibration plot, ROC, and risk-score distributions are in
`results/calibration.png`. Sample rendered Health-card screenshots:
`results/health_card_low.png`, `health_card_mid.png`, `health_card_high.png`.

---

## HealthKit feature map

The Python features and the iOS HealthKit reads share one schema
(`src/healthkit_schema.py`), so the trained model takes exactly the same
feature vector regardless of whether the source is NHANES or a live Apple
Watch.

| Feature                       | HealthKit identifier                              | In v1 training? |
|-------------------------------|---------------------------------------------------|:--------------:|
| age                           | `HKCharacteristicTypeIdentifierDateOfBirth`       | ✅ |
| biological_sex                | `HKCharacteristicTypeIdentifierBiologicalSex`     | ✅ |
| avg_daily_step_count          | `HKQuantityTypeIdentifierStepCount`               | ✅ |
| avg_daily_active_energy       | `HKQuantityTypeIdentifierActiveEnergyBurned`      | ✅ |
| avg_daily_exercise_min        | `HKQuantityTypeIdentifierAppleExerciseTime`       | ✅ |
| sedentary_minutes             | `HKQuantityTypeIdentifierAppleStandTime` (inv.)   | ✅ |
| activity_regularity           | derived                                           | ✅ |
| low_activity_day_ratio        | derived                                           | ✅ |
| peak_intensity                | derived                                           | ✅ |
| avg_sleep_hours               | `HKCategoryTypeIdentifierSleepAnalysis`           | ✅ |
| sleep_regularity              | derived                                           | ✅ |
| circadian_light_exposure      | derived (ambient light)                           | ✅ |
| resting_heart_rate            | `HKQuantityTypeIdentifierRestingHeartRate`        | ⚪ shown only, **v2** |
| hrv_sdnn                      | `HKQuantityTypeIdentifierHeartRateVariabilitySDNN`| ⚪ shown only, **v2** |
| vo2_max_estimate              | `HKQuantityTypeIdentifierVO2Max`                  | ⚪ shown only, **v2** |

NHANES 2011-2012 does not include HR/HRV/VO2max, so v1 is trained on
accelerometer-only signals. The iOS app already reads those values from
HealthKit and displays them in the dashboard for context — adding them as
trained inputs in v2 is a drop-in change (the schema, scaler, and Core ML
pipeline all support it).

---

## Repository layout

```
src/
  healthkit_schema.py     ←  feature catalogue, HK identifiers, units
  features_hk.py          ←  NHANES → HK-shaped features
  framingham.py           ←  D'Agostino 2008 risk score (labels)
  load_data.py            ←  NHANES XPT loader
  train_hk.py             ←  CalibratedClassifierCV training run
  export_coreml.py        ←  Core ML tree ensemble + isotonic LUT
  health_report.py        ←  Apple-Health-style PNG report generator

models/
  CVDRiskModel.mlmodel        ←  on-device tree ensemble (64 KB)
  isotonic_calibration.json   ←  Swift-applied probability calibration
  scaler.json                 ←  feature order + StandardScaler params
  cvd_risk_v1.joblib          ←  Python reference pipeline
  training_metadata.json      ←  AUC/Brier/cohort info

ios/
  README.md                ←  how to open in Xcode
  CVDIntelligence/
    CVDIntelligenceApp.swift
    HealthKit/HealthStore.swift     ←  HKQuery, statistics, sleep, characteristics
    Models/HealthSnapshot.swift     ←  feature vector mirror of Python schema
    Models/RiskModel.swift          ←  Core ML inference + isotonic LUT
    Models/Insight.swift            ←  on-device rules engine
    Views/DashboardView.swift       ←  Apple-Health-style screen
    Views/RiskRingView.swift        ←  activity-ring risk indicator
    Views/MetricCard.swift          ←  glanceable metric tile
    Views/InsightCard.swift         ←  contextual recommendation
    Resources/                      ←  CVDRiskModel.mlmodel + JSON sidecars
    Info.plist                      ←  NSHealthShareUsageDescription
    CVDIntelligence.entitlements    ←  HealthKit capability
```

---

## Reproduce

```bash
# 1. install deps
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. download NHANES files into data/  (DEMO_G, TCHOL_G, HDL_G, BPX_G,
#    SMQ_G, GLU_G, GHB_G, DIQ_G, PAXHD_G, PAXDAY_G, PAXHR_G)
#    https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2011

# 3. train + export everything
python src/train_hk.py        # calibrated pipeline + diagnostics PNG
python src/export_coreml.py   # CVDRiskModel.mlmodel + scaler + isotonic LUT
python src/health_report.py   # sample Apple-Health-style PNGs into results/
```

To open the iOS app, see `ios/README.md`.

---

## Honest limitations

- **NHANES is a US-only cross-sectional sample.** The Framingham coefficients
  are themselves derived from a non-representative cohort. Use this for
  population-scale triage and self-tracking, not individual clinical decisions.
- **The "risk" output is a calibrated probability of *being labeled high-risk
  by Framingham*** — it is one degree removed from "ground truth" CVD events.
  A more rigorous v2 would train against actual NHANES-linked mortality.
- **Step / kcal calibration constants** were back-fit so that population
  medians match CDC adult norms; individual-level conversions from raw
  accelerometer counts are inherently noisy.
- **Heart-rate signals are read on-device but not in the v1 training set**
  because NHANES 2011-2012 doesn't include them. The schema and Swift code
  are wired so v2 can train on them without UI changes.

---

## Acknowledgments

- NHANES 2011-2012 (CDC) — open-access wearable + clinical data.
- D'Agostino, R.B. et al. 2008. "General Cardiovascular Risk Profile for Use in
  Primary Care." *Circulation*, 117 (6).
