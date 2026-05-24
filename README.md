# Adaptive Cardiovascular Health Intelligence


🔗 Live Demo: https://newframingham-zsc93c9i4xgh4kydr3j3rm.streamlit.app/
**Predicting 10-year cardiovascular disease risk from passive Apple Watch-style
signals — entirely on-device, with no blood draws or clinical visits.**

A three-layer project, all driven by the *same trained model*:

1. **Python core** — ingests NHANES 2011-2012 accelerometer data, derives a
   HealthKit-shaped feature set, and trains an isotonic-calibrated
   GradientBoosting tree ensemble against Framingham-derived labels.
   Training writes four artifacts in one pass (`cvd_risk_v1.joblib`,
   `scaler.json`, `isotonic_calibration.json`, `CVDRiskModel.mlmodel`) so the
   web and iOS deployments are guaranteed to give the same answer for the
   same inputs.
2. **Streamlit web demo** (`app.py`) — interactive risk calculator. Loads the
   same calibrated pipeline, exposes a HealthKit-labelled input form, and
   adds a local-sensitivity bar chart + what-if sweep so users can see what
   actually moves their estimated risk. Deployable to Streamlit Cloud as a
   live URL.
3. **iOS app** (`ios/`) — SwiftUI + HealthKit + Core ML. Reads Apple Watch
   metrics, runs the exported `.mlmodel` + isotonic LUT locally, and renders
   an Apple-Health-style risk card with a contextual insight.

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

Because all four artifacts come from one fit, the Streamlit app and the iOS
app are guaranteed to give the same answer for the same inputs.

---

# Results

|                                              | Value |
|----------------------------------------------|------:|
| Participants (NHANES 2011-2012, ages 30-74)  | 2,967 |
| Tree-ensemble AUC (5-fold OOF)               | **0.921** |
| Brier — uncalibrated (5-fold OOF)            | 0.111 |
| Brier — after isotonic calibration           | **0.108** |
| High-risk prevalence (Framingham ≥ 10%)      | 41.0% |
| Core ML model size                           | 62 KB |
| Inference (on-device, A14+)                  | real-time (single forward pass over a 200-tree, depth-3 ensemble) |

The calibration plot, ROC, and risk-score distributions are in
`results/calibration.png`. Sample rendered Health-card screenshots:
`results/health_card_low.png`, `health_card_mid.png`, `health_card_high.png`.

**Methodology notes.** The 5-fold cross-validation puts the
median-imputer *inside* the sklearn `Pipeline`, so test-fold values cannot
leak into the imputed training rows. Isotonic regression is fit on the
out-of-fold predictions, then the same fitted calibrator ships in both the
Python joblib and as a piecewise-linear LUT (`isotonic_calibration.json`)
applied on-device. AUC and Brier above are the OOF numbers, not in-sample.

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
app.py                       ←  Streamlit web demo (loads the joblib below)

src/
  healthkit_schema.py          # feature catalogue, HK identifiers, units
  features_hk.py               # NHANES → HK-shaped features
  framingham.py                # D'Agostino 2008 risk score (labels)
  load_data.py                 # NHANES XPT loader
  train_hk.py                  # one-shot training: joblib + sidecars
  export_coreml.py             # consumes joblib → CVDRiskModel.mlmodel
  health_report.py             # sample Apple-Health-style PNG renderer

models/
  CVDRiskModel.mlmodel         # on-device tree ensemble (62 KB)
  isotonic_calibration.json    # 101-point piecewise-linear LUT
  scaler.json                  # feature order, scaler params, medians
  cvd_risk_v1.joblib           # full Python bundle
  training_metadata.json       # AUC, Brier, cohort info

ios/CVDIntelligence/
  HealthKit/HealthStore.swift  # HKQuery, statistics, sleep samples
  Models/RiskModel.swift       # Core ML inference + isotonic LUT
  Views/                       # SwiftUI dashboard, ring, cards
  Resources/                   # bundled .mlmodel + JSON sidecars

tests/                         # 17 pytest tests (Framingham math,
                               # bundle integrity, Streamlit smoke test)

.github/workflows/ci.yml       # pytest on every push/PR (Python 3.11, 3.12)
```

---

# Running Locally

```bash
# 1. install deps
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. download NHANES 2011-2012 files into data/
#    (DEMO_G, TCHOL_G, HDL_G, BPX_G, SMQ_G, GLU_G, GHB_G, DIQ_G,
#     PAXHD_G, PAXDAY_G, PAXHR_G)
#    https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2011

# 3. train + export everything in one pass
python src/train_hk.py        # writes joblib + scaler + isotonic + metadata
python src/export_coreml.py   # consumes joblib → CVDRiskModel.mlmodel

# 4. launch the web demo
streamlit run app.py

# 5. run the test suite
pytest tests/ -v
```

To open the iOS app in Xcode, see `ios/README.md`.

## Tests + CI

`tests/` contains 17 unit/integration tests covering:

- D'Agostino 2008 Framingham math against the paper's worked examples (male
  55y → ~10%, female 55y → ~5-6%), age-monotonicity, and the smoking/diabetes
  covariate signs.
- Model-bundle integrity: joblib structure, scaler-vs-pipeline coherence,
  isotonic LUT monotonicity and range, and a regression guard against
  accidentally dropping the median imputer (the CV-leakage fix).
- Streamlit smoke test via `streamlit.testing.v1.AppTest` — if this passes,
  the app deploys cleanly on Streamlit Cloud.

CI runs the same suite on every push and PR against `main`, on Python 3.11
and 3.12 (`.github/workflows/ci.yml`).

---

## Honest limitations

- **NHANES is a US-only cross-sectional sample.** The Framingham coefficients
  are themselves derived from a non-representative cohort. Use this for
  population-scale triage and self-tracking, not individual clinical decisions.
- **The "risk" output is a calibrated probability of *being labelled high-risk
  by Framingham*** — it is one degree removed from "ground truth" CVD events.
  A more rigorous v2 would train against actual NHANES-linked mortality.
- **Smoking covariate uses NHANES SMQ020 ("ever smoked ≥100 cigarettes"), not
  current smoker.** D'Agostino 2008's smoking term is current smoker; the
  effect here is to nudge former-smoker risk scores up. v2 will combine
  SMQ020 with SMQ040 for a current-smoker definition.
- **Step / kcal calibration constants** were back-fit so that population
  medians match CDC adult norms; individual-level conversions from raw
  accelerometer counts are inherently noisy.
- **Ambient light is in the trained feature vector but is not exposed by
  HealthKit.** The iOS app passes `.nan` for `circadianLightExposure` and the
  inference layer substitutes the training-set median (mirroring the Python
  `SimpleImputer`). That fallback is acknowledged in code so the model's
  expectations stay aligned across deployments.
- **Heart-rate signals are read on-device but not in the v1 training set**
  because NHANES 2011-2012 doesn't include them. The schema and Swift code
  are wired so v2 can train on them without UI changes.

---

## Acknowledgments

- NHANES 2011-2012 (U.S. CDC) — open-access wearable + clinical dataset
- D'Agostino R.B. et al., 2008. "General Cardiovascular Risk Profile for Use
  in Primary Care." *Circulation* 117(6) — the labels this model is trained
  against
- Apple HealthKit and Core ML documentation
