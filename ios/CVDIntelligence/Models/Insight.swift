import Foundation

/// Tiny rules-based insight engine. Deterministic, on-device, and explainable.
struct Insight: Equatable {
    let headline: String
    let detail: String
    let primaryDriver: Driver

    enum Driver: String {
        case sedentary, sleep, regularity, fitness, none
    }

    static func from(snapshot s: HealthSnapshot, risk r: RiskAssessment) -> Insight {
        var msgs: [(Driver, String)] = []

        if s.avgDailyStepCount < 5000 {
            msgs.append((.sedentary,
                "Your daily step count is averaging \(format(s.avgDailyStepCount, 0)), well below the 7-day target of 7,500."))
        } else if s.avgDailyStepCount > 9000 {
            msgs.append((.fitness,
                "Strong activity — averaging \(format(s.avgDailyStepCount, 0)) daily steps puts you in the top tier for your age band."))
        }

        if s.avgSleepHours < 6.5 {
            msgs.append((.sleep,
                "Sleep duration is consistently below 6.5 hours, which independently raises cardiovascular risk."))
        }

        if s.activityRegularity < 0.4 {
            msgs.append((.regularity,
                "Your activity pattern is highly variable day to day."))
        }

        if s.lowActivityDayRatio > 0.4 {
            msgs.append((.sedentary,
                "More than 40% of your recent days were below your personal activity baseline."))
        }

        if msgs.isEmpty {
            if r.calibratedProbability >= 0.15 {
                msgs.append((.fitness,
                    "Lifestyle signals look reasonable on their own, but demographic and baseline factors still place you in an elevated 10-year risk band — worth a clinician chat."))
            } else {
                msgs.append((.none,
                    "Your passive signals are consistent with a favourable cardiovascular profile."))
            }
        }

        let detail = msgs.prefix(2).map { $0.1 }.joined(separator: " ")
        let headline: String
        switch r.tier {
        case .low: headline = "On track"
        case .borderline: headline = "Watch this trend"
        case .elevated: headline = "Time to act"
        case .high: headline = "Talk to a clinician"
        }
        return Insight(headline: headline, detail: detail,
                       primaryDriver: msgs.first?.0 ?? .none)
    }

    private static func format(_ v: Double, _ d: Int) -> String {
        String(format: "%.\(d)f", v)
    }
}
