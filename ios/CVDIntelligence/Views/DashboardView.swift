import SwiftUI

struct DashboardView: View {
    @EnvironmentObject var store: HealthStore

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    header

                    if !store.authorizationGranted {
                        permissionCard
                    } else if store.loading {
                        ProgressView("Reading from Apple Watch…")
                            .frame(maxWidth: .infinity, minHeight: 160)
                    } else if let risk = store.risk, let snap = store.snapshot {
                        RiskRingView(risk: risk)
                            .frame(height: 280)

                        VStack(spacing: 12) {
                            HStack(spacing: 12) {
                                MetricCard(title: "Steps", accent: .green,
                                           value: snap.avgDailyStepCount.formatted(.number.precision(.fractionLength(0))),
                                           unit: "/ day", subtitle: "7-day avg")
                                MetricCard(title: "Active Energy", accent: .red,
                                           value: snap.avgDailyActiveEnergy.formatted(.number.precision(.fractionLength(0))),
                                           unit: "kcal", subtitle: "7-day avg")
                            }
                            HStack(spacing: 12) {
                                MetricCard(title: "Exercise", accent: .pink,
                                           value: snap.avgDailyExerciseMin.formatted(.number.precision(.fractionLength(0))),
                                           unit: "min", subtitle: "7-day avg")
                                MetricCard(title: "Sleep", accent: .purple,
                                           value: snap.avgSleepHours.formatted(.number.precision(.fractionLength(1))),
                                           unit: "h", subtitle: "7-day avg")
                            }
                            HStack(spacing: 12) {
                                MetricCard(title: "Resting HR", accent: .red,
                                           value: snap.restingHeartRate.formatted(.number.precision(.fractionLength(0))),
                                           unit: "bpm", subtitle: "from Watch")
                                MetricCard(title: "HRV (SDNN)", accent: .red,
                                           value: snap.hrvSDNN.formatted(.number.precision(.fractionLength(0))),
                                           unit: "ms", subtitle: "from Watch")
                            }
                        }

                        if let insight = store.insight {
                            InsightCard(insight: insight, accent: risk.tier.color)
                        }

                        footer
                    } else if let err = store.lastError {
                        Text(err).foregroundStyle(.secondary)
                    }
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 12)
            }
            .background(Color.black.ignoresSafeArea())
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        Task { await store.refresh() }
                    } label: { Image(systemName: "arrow.clockwise") }
                }
            }
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("Health")
                .font(.system(size: 32, weight: .bold))
                .foregroundStyle(.white)
            Text("Cardiovascular Intelligence  •  on-device")
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
    }

    private var permissionCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Image(systemName: "heart.text.square.fill")
                .font(.system(size: 36))
                .foregroundStyle(.red)
            Text("Allow access to your Health data")
                .font(.headline).foregroundStyle(.white)
            Text("All compute happens on this device. Your data never leaves your iPhone.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
            Button("Continue") { Task { await store.bootstrap() } }
                .buttonStyle(.borderedProminent)
                .tint(.red)
        }
        .padding(20)
        .background(Color(.secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }

    private var footer: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("Predicted on-device with CVDRiskModel v1.0")
            Text("Trained on NHANES 2011-2012 cohort  •  AUC 0.92  •  Brier 0.11")
        }
        .font(.caption2)
        .foregroundStyle(.secondary)
        .padding(.top, 8)
    }
}

extension RiskAssessment.RiskTier {
    var color: Color {
        switch self {
        case .low: return .green
        case .borderline: return .yellow
        case .elevated: return .orange
        case .high: return .red
        }
    }
}
