import Foundation

/// Canonical feature snapshot the model takes as input.
///
/// Mirror of src/healthkit_schema.py — keep field order in sync with
/// `feature_order` in scaler.json.
struct HealthSnapshot: Equatable {
    var age: Double
    var biologicalSexIsMale: Double   // 0/1

    // Activity & energy (v1 features)
    var avgDailyStepCount: Double
    var avgDailyActiveEnergy: Double
    var avgDailyExerciseMin: Double
    var avgSedentaryMin: Double

    var activityRegularity: Double
    var lowActivityDayRatio: Double
    var peakIntensity: Double

    var avgSleepHours: Double
    var sleepRegularity: Double

    var circadianLightExposure: Double

    // Heart (read on-device, not in v1 trained feature set; shown for context)
    var restingHeartRate: Double
    var hrvSDNN: Double
    var vo2Max: Double

    /// Order MUST match feature_order in scaler.json.
    func featureVector(order: [String]) -> [Double] {
        order.map { key in
            switch key {
            case "age": return age
            case "biological_sex": return biologicalSexIsMale
            case "avg_daily_active_energy": return avgDailyActiveEnergy
            case "avg_daily_step_count": return avgDailyStepCount
            case "avg_daily_exercise_min": return avgDailyExerciseMin
            case "activity_regularity": return activityRegularity
            case "low_activity_day_ratio": return lowActivityDayRatio
            case "peak_intensity": return peakIntensity
            case "sedentary_minutes": return avgSedentaryMin
            case "avg_sleep_hours": return avgSleepHours
            case "sleep_regularity": return sleepRegularity
            case "circadian_light_exposure": return circadianLightExposure
            default: return 0
            }
        }
    }
}
