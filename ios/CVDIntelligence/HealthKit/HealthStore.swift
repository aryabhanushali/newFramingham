import Combine
import Foundation
import HealthKit
import SwiftUI

/// Single source of truth for HealthKit reads, feature assembly, on-device
/// inference, and the published values the SwiftUI views render.
@MainActor
final class HealthStore: ObservableObject {

    // MARK: published state
    @Published var authorizationGranted = false
    @Published var snapshot: HealthSnapshot?
    @Published var risk: RiskAssessment?
    @Published var insight: Insight?
    @Published var loading = false
    @Published var lastError: String?

    private let healthStore = HKHealthStore()
    private let model = RiskModel()

    /// Identifiers we read on-device. Mirrors src/healthkit_schema.py exactly.
    private var readTypes: Set<HKObjectType> {
        var s: Set<HKObjectType> = []
        let qIds: [HKQuantityTypeIdentifier] = [
            .stepCount, .activeEnergyBurned, .appleExerciseTime,
            .appleStandTime, .distanceWalkingRunning,
            .restingHeartRate, .walkingHeartRateAverage,
            .heartRateVariabilitySDNN, .vo2Max,
        ]
        for id in qIds { if let t = HKQuantityType.quantityType(forIdentifier: id) { s.insert(t) } }
        if let sleep = HKCategoryType.categoryType(forIdentifier: .sleepAnalysis) { s.insert(sleep) }
        s.insert(HKCharacteristicType.characteristicType(forIdentifier: .dateOfBirth)!)
        s.insert(HKCharacteristicType.characteristicType(forIdentifier: .biologicalSex)!)
        return s
    }

    func bootstrap() async {
        guard HKHealthStore.isHealthDataAvailable() else {
            lastError = "HealthKit not available on this device."
            return
        }
        do {
            try await healthStore.requestAuthorization(toShare: [], read: readTypes)
            authorizationGranted = true
            await refresh()
        } catch {
            lastError = "Authorization failed: \(error.localizedDescription)"
        }
    }

    func refresh() async {
        loading = true
        defer { loading = false }
        do {
            let snap = try await assembleSnapshot()
            self.snapshot = snap
            let r = try model.predict(snapshot: snap)
            self.risk = r
            self.insight = Insight.from(snapshot: snap, risk: r)
        } catch {
            lastError = error.localizedDescription
        }
    }

    // MARK: – HealthKit queries

    private func assembleSnapshot() async throws -> HealthSnapshot {
        async let age = readAge()
        async let sex = readBiologicalSex()
        async let steps = avgDaily(.stepCount, unit: .count(), days: 7)
        async let energy = avgDaily(.activeEnergyBurned, unit: .kilocalorie(), days: 7)
        async let exercise = avgDaily(.appleExerciseTime, unit: .minute(), days: 7)
        async let stand = avgDaily(.appleStandTime, unit: .minute(), days: 7)
        async let rhr = avgDaily(.restingHeartRate, unit: .count().unitDivided(by: .minute()), days: 7)
        async let hrv = avgDaily(.heartRateVariabilitySDNN, unit: HKUnit.secondUnit(with: .milli), days: 7)
        async let vo2 = avgDaily(.vo2Max, unit: HKUnit(from: "ml/(kg*min)"), days: 30)
        async let sleepStats = sleepStatistics(days: 7)
        async let activityStats = activityDaily(days: 14)

        // Resolve once so we don't await the same task multiple times
        let stats = try await activityStats
        let sleep = try await sleepStats

        // HealthKit does not expose ambient light as a queryable quantity, so
        // we pass .nan here; RiskModel.predict substitutes the training-set
        // median (matching the Python SimpleImputer) and the UI surfaces the
        // gap honestly rather than fabricating a value.
        return HealthSnapshot(
            age: try await age,
            biologicalSexIsMale: try await sex,
            avgDailyStepCount: try await steps,
            avgDailyActiveEnergy: try await energy,
            avgDailyExerciseMin: try await exercise,
            avgSedentaryMin: max(0, 16 * 60 - (try await stand)),
            activityRegularity: stats.regularity,
            lowActivityDayRatio: stats.lowDayRatio,
            peakIntensity: stats.peak,
            avgSleepHours: sleep.meanHours,
            sleepRegularity: sleep.regularity,
            circadianLightExposure: .nan,
            restingHeartRate: try await rhr,
            hrvSDNN: try await hrv,
            vo2Max: try await vo2
        )
    }

    private func readAge() async throws -> Double {
        let dob = try healthStore.dateOfBirthComponents()
        guard let date = Calendar.current.date(from: dob) else { return 50 }
        let years = Calendar.current.dateComponents([.year], from: date, to: Date()).year ?? 50
        return Double(years)
    }

    private func readBiologicalSex() async throws -> Double {
        let s = try healthStore.biologicalSex().biologicalSex
        return s == .male ? 1.0 : 0.0
    }

    private func avgDaily(_ id: HKQuantityTypeIdentifier,
                          unit: HKUnit, days: Int) async -> Double {
        guard let qType = HKQuantityType.quantityType(forIdentifier: id) else { return 0 }
        let end = Date()
        let start = Calendar.current.date(byAdding: .day, value: -days, to: end)!
        let predicate = HKQuery.predicateForSamples(withStart: start, end: end)

        return await withCheckedContinuation { cont in
            let q = HKStatisticsQuery(quantityType: qType,
                                      quantitySamplePredicate: predicate,
                                      options: id.isCumulative ? .cumulativeSum : .discreteAverage) { _, stats, _ in
                let value: Double
                if id.isCumulative {
                    value = (stats?.sumQuantity()?.doubleValue(for: unit) ?? 0) / Double(days)
                } else {
                    value = stats?.averageQuantity()?.doubleValue(for: unit) ?? 0
                }
                cont.resume(returning: value)
            }
            healthStore.execute(q)
        }
    }

    private struct SleepStats { let meanHours: Double; let regularity: Double }

    private func sleepStatistics(days: Int) async -> SleepStats {
        guard let sleepType = HKCategoryType.categoryType(forIdentifier: .sleepAnalysis) else {
            return .init(meanHours: .nan, regularity: .nan)
        }
        let end = Date()
        let start = Calendar.current.date(byAdding: .day, value: -days, to: end)!
        let predicate = HKQuery.predicateForSamples(withStart: start, end: end)

        return await withCheckedContinuation { cont in
            let q = HKSampleQuery(sampleType: sleepType, predicate: predicate,
                                  limit: HKObjectQueryNoLimit, sortDescriptors: nil) { _, samples, _ in
                let asleepValues: Set<Int> = [
                    HKCategoryValueSleepAnalysis.asleepCore.rawValue,
                    HKCategoryValueSleepAnalysis.asleepDeep.rawValue,
                    HKCategoryValueSleepAnalysis.asleepREM.rawValue,
                    HKCategoryValueSleepAnalysis.asleepUnspecified.rawValue,
                ]
                // Bucket asleep samples per calendar night (keyed on the
                // 6PM-cutoff date of the start time) so multi-segment nights
                // collapse into one duration before we measure variance.
                let calendar = Calendar.current
                var perNight: [Date: TimeInterval] = [:]
                for s in samples ?? [] {
                    guard let cat = s as? HKCategorySample,
                          asleepValues.contains(cat.value) else { continue }
                    let anchor = calendar.date(byAdding: .hour, value: -18, to: cat.startDate) ?? cat.startDate
                    let key = calendar.startOfDay(for: anchor)
                    perNight[key, default: 0] += cat.endDate.timeIntervalSince(cat.startDate)
                }
                guard !perNight.isEmpty else {
                    cont.resume(returning: .init(meanHours: .nan, regularity: .nan)); return
                }
                let hours = perNight.values.map { $0 / 3600.0 }
                let mean = hours.reduce(0, +) / Double(hours.count)
                let variance = hours.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(hours.count)
                let std = sqrt(variance)
                let regularity = 1.0 / (1.0 + std)
                cont.resume(returning: .init(meanHours: mean, regularity: regularity))
            }
            healthStore.execute(q)
        }
    }

    private struct ActivityStats { let regularity: Double; let lowDayRatio: Double; let peak: Double }

    private func activityDaily(days: Int) async -> ActivityStats {
        guard let qType = HKQuantityType.quantityType(forIdentifier: .activeEnergyBurned) else {
            return .init(regularity: 0.5, lowDayRatio: 0.25, peak: 0)
        }
        let end = Date()
        let start = Calendar.current.date(byAdding: .day, value: -days, to: end)!
        let predicate = HKQuery.predicateForSamples(withStart: start, end: end)
        let interval = DateComponents(day: 1)

        return await withCheckedContinuation { cont in
            let q = HKStatisticsCollectionQuery(
                quantityType: qType,
                quantitySamplePredicate: predicate,
                options: .cumulativeSum,
                anchorDate: Calendar.current.startOfDay(for: start),
                intervalComponents: interval
            )
            q.initialResultsHandler = { _, results, _ in
                var values: [Double] = []
                results?.enumerateStatistics(from: start, to: end) { stat, _ in
                    values.append(stat.sumQuantity()?.doubleValue(for: .kilocalorie()) ?? 0)
                }
                guard !values.isEmpty else {
                    cont.resume(returning: .init(regularity: 0.5, lowDayRatio: 0.25, peak: 0)); return
                }
                let mean = values.reduce(0, +) / Double(values.count)
                let variance = values.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(values.count)
                let std = sqrt(variance)
                let reg = 1.0 / (1.0 + std / (mean + 1.0))
                let p25 = values.sorted()[max(0, values.count / 4)]
                let lowRatio = Double(values.filter { $0 < p25 }.count) / Double(values.count)
                let peak = values.max() ?? 0
                cont.resume(returning: .init(regularity: reg, lowDayRatio: lowRatio, peak: peak))
            }
            healthStore.execute(q)
        }
    }
}

private extension HKQuantityTypeIdentifier {
    var isCumulative: Bool {
        switch self {
        case .stepCount, .activeEnergyBurned, .appleExerciseTime,
             .appleStandTime, .distanceWalkingRunning:
            return true
        default:
            return false
        }
    }
}
