# CVD Intelligence — iOS App

SwiftUI + HealthKit + Core ML companion to the Python pipeline at the repo root.

## How to open

1. Install **Xcode 15+** from the Mac App Store.
2. In Xcode: **File → New → Project… → iOS → App** → name it `CVDIntelligence`,
   interface **SwiftUI**, language **Swift**, organization identifier of your
   choice.
3. Save the new project somewhere convenient.
4. In Finder, **delete** the auto-generated `ContentView.swift` and the default
   `Assets.xcassets` inside the new project.
5. Drag the four folders below from `ios/CVDIntelligence/` into the Xcode
   project navigator (check **Copy items if needed**, **Create groups**):
     - `HealthKit/`
     - `Models/`
     - `Views/`
     - `Resources/`
6. Replace the generated `CVDIntelligenceApp.swift` with the one in this folder.
7. Replace the generated `Info.plist` (or merge in the `NSHealthShareUsageDescription`
   key from `Info.plist`).
8. Add `CVDIntelligence.entitlements` to the project and select it in
   **Target → Signing & Capabilities → All → + Capability → HealthKit**.
9. Drag `Resources/CVDRiskModel.mlmodel` into Xcode — it will recognise it as
   a Core ML model and auto-compile.

## How it works

```
HealthStore (HealthKit reads)
      │
      ▼   HealthSnapshot
RiskModel (CoreML inference + isotonic LUT)
      │
      ▼   RiskAssessment + Insight
DashboardView (SwiftUI, Apple-Health styling)
```

All compute is on-device. No network calls, no remote backends.

The model is the same `CVDRiskModel.mlmodel` exported by
`src/export_coreml.py`, with `scaler.json` and `isotonic_calibration.json`
loaded at startup.
