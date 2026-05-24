# Adaptive Cardiovascular Health Intelligence

🔗 Live Demo: https://newframingham-zsc93c9i4xgh4kydr3j3rm.streamlit.app/

Predicting 10-year cardiovascular disease risk from passive wearable signals using on-device machine learning.

---

## Overview

This project explores whether Apple Watch-style health signals, such as activity, sleep, and exercise patterns, can approximate traditional cardiovascular risk estimates without blood tests or clinical visits.

Using NHANES wearable data and Framingham-derived risk labels, I built a full end-to-end pipeline that:

- Trains a calibrated machine learning model in Python
- Deploys the same model in a Streamlit web app
- Runs real-time inference locally in an iOS app using Core ML + HealthKit

The same trained model powers every platform to ensure consistent predictions across web and mobile deployments.

---

## Motivation

Traditional cardiovascular risk calculators require:

- Cholesterol tests
- Blood pressure measurements
- Smoking and diabetes history
- Clinical appointments

This project investigates whether passive wearable data alone can provide a meaningful estimate of long-term cardiovascular risk.

The goal is to make preventive health monitoring more accessible, privacy-preserving, and scalable through lightweight on-device AI.

---

## Model Pipeline

```text
NHANES wearable data
        ↓
HealthKit-style feature engineering
        ↓
Gradient Boosting classifier
        ↓
Isotonic probability calibration
        ↓
Core ML + Streamlit deployment
```

The pipeline exports:

- `cvd_risk_v1.joblib`
- `scaler.json`
- `isotonic_calibration.json`
- `CVDRiskModel.mlmodel`

allowing the Python, web, and iOS systems to use the exact same trained model.

---

## Results

| Metric | Result |
|---|---:|
| Participants | 2,967 |
| 5-Fold OOF AUC | **0.921** |
| Calibrated Brier Score | **0.108** |
| Model Size | 62 KB |
| Inference Speed | Real-time on-device |

The model was able to approximate Framingham cardiovascular risk using only passive wearable-derived signals.

---

## Features Used

### Activity Features
- Daily step count
- Exercise minutes
- Active energy burned
- Sedentary time
- Activity consistency
- Peak activity intensity

### Sleep Features
- Average sleep duration
- Sleep regularity

### Demographics
- Age
- Biological sex

### Planned v2 Features
- Resting heart rate
- HRV
- VO₂ max


---

## Methodology

To avoid data leakage:

- Median imputation was placed fully inside the sklearn pipeline
- Calibration was trained only on out-of-fold predictions
- Reported metrics are fully out-of-fold results

The deployed iOS app uses the same fitted calibration function as the Python training pipeline.

---

## Limitations

- NHANES is a US-only dataset and may not generalize globally
- The model predicts Framingham-derived risk rather than actual future cardiovascular events
- Some wearable features (HRV, VO₂ max) were unavailable in the training data
- This project is intended for research and educational purposes, not clinical diagnosis

---

## Future Work

Planned improvements include:

- Training on longitudinal cardiovascular outcomes
- Adding HRV and VO₂ max into model training
- Personalized temporal modeling
- Broader demographic validation
- Improved calibration across subgroups

---

## Tech Stack

### Machine Learning
- Python
- scikit-learn
- pandas / NumPy

### Deployment
- Streamlit
- Core ML
- SwiftUI
- HealthKit

### Infrastructure
- pytest
- GitHub Actions

---

## Repository Structure

```text
app.py                     # Streamlit application

src/
  healthkit_schema.py
  features_hk.py
  framingham.py
  train_hk.py
  export_coreml.py

models/
  CVDRiskModel.mlmodel
  isotonic_calibration.json
  cvd_risk_v1.joblib

ios/
  SwiftUI + HealthKit application
```

---

## Acknowledgments

- NHANES 2011–2012 (CDC)
- Framingham Risk Score methodology
- Apple HealthKit documentation
- Core ML framework
