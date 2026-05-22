import Foundation
import CoreML

struct RiskAssessment: Equatable {
    let rawProbability: Double          // direct from tree ensemble
    let calibratedProbability: Double   // after isotonic LUT
    let tier: RiskTier

    enum RiskTier: String {
        case low = "Low"
        case borderline = "Borderline"
        case elevated = "Elevated"
        case high = "High"

        static func from(_ p: Double) -> RiskTier {
            switch p {
            case ..<0.075: return .low
            case ..<0.15: return .borderline
            case ..<0.25: return .elevated
            default: return .high
            }
        }
    }
}

/// Loads CVDRiskModel.mlmodel, runs inference on a feature vector, and
/// applies the isotonic calibration LUT exported from Python.
final class RiskModel {

    enum ModelError: Error { case missingResource(String); case predictionFailed }

    private let mlModel: MLModel
    private let scaler: ScalerSpec
    private let calibration: IsotonicLUT

    init() {
        let bundle = Bundle.main
        guard
            let modelURL = bundle.url(forResource: "CVDRiskModel", withExtension: "mlmodel"),
            let scalerURL = bundle.url(forResource: "scaler", withExtension: "json"),
            let calibURL = bundle.url(forResource: "isotonic_calibration", withExtension: "json")
        else {
            fatalError("CVDRiskModel resources missing from bundle.")
        }

        // Compile and load
        do {
            let compiled = try MLModel.compileModel(at: modelURL)
            self.mlModel = try MLModel(contentsOf: compiled)
        } catch {
            fatalError("Failed to load Core ML model: \(error)")
        }

        do {
            self.scaler = try JSONDecoder().decode(ScalerSpec.self,
                                                   from: Data(contentsOf: scalerURL))
            self.calibration = try JSONDecoder().decode(IsotonicLUT.self,
                                                        from: Data(contentsOf: calibURL))
        } catch {
            fatalError("Failed to load scaler/calibration JSON: \(error)")
        }
    }

    func predict(snapshot: HealthSnapshot) throws -> RiskAssessment {
        // 1. assemble feature vector in canonical order; any features the
        //    device cannot read (NaN / sentinel) are filled with the training
        //    median, mirroring the Python SimpleImputer step.
        let raw = snapshot.featureVector(order: scaler.featureOrder)
        let imputed = zip(raw, scaler.featureOrder).map { (v, name) -> Double in
            v.isFinite ? v : (scaler.medians[name] ?? 0)
        }

        // 2. standard-scale with mean/std from training
        let scaled = zip(imputed, zip(scaler.mean, scaler.scale)).map { (value, ms) -> Double in
            let (m, s) = ms
            return s == 0 ? 0 : (value - m) / s
        }

        // 3. pack into MLMultiArray
        let array = try MLMultiArray(shape: [NSNumber(value: scaled.count)],
                                     dataType: .double)
        for (i, v) in scaled.enumerated() {
            array[i] = NSNumber(value: v)
        }

        // 4. run Core ML
        let provider = try MLDictionaryFeatureProvider(dictionary: ["features": MLFeatureValue(multiArray: array)])
        let out = try mlModel.prediction(from: provider)

        // Core ML returns [NSNumber: Double] for Int64-labelled classes.
        // Try the friendlier cast first, then fall back to NSNumber lookup
        // so this is robust across iOS versions.
        let pPos: Double
        if let typed = out.featureValue(for: "classProbability")?.dictionaryValue as? [Int64: Double],
           let p = typed[1] {
            pPos = p
        } else if let raw = out.featureValue(for: "classProbability")?.dictionaryValue,
                  let p = raw[NSNumber(value: 1)] as? Double {
            pPos = p
        } else {
            throw ModelError.predictionFailed
        }

        // 5. isotonic calibration
        let calibrated = calibration.apply(pPos)
        return RiskAssessment(
            rawProbability: pPos,
            calibratedProbability: calibrated,
            tier: .from(calibrated)
        )
    }
}

// MARK: – sidecar resources

private struct ScalerSpec: Decodable {
    let featureOrder: [String]
    let mean: [Double]
    let scale: [Double]
    let medians: [String: Double]

    enum CodingKeys: String, CodingKey {
        case featureOrder = "feature_order"
        case mean, scale, medians
    }
}

private struct IsotonicLUT: Decodable {
    let x: [Double]
    let y: [Double]

    /// Piecewise-linear interpolation; clipped at the LUT ends.
    func apply(_ p: Double) -> Double {
        guard let first = x.first, let last = x.last else { return p }
        if p <= first { return y.first ?? p }
        if p >= last { return y.last ?? p }
        var lo = 0, hi = x.count - 1
        while hi - lo > 1 {
            let mid = (lo + hi) / 2
            if x[mid] <= p { lo = mid } else { hi = mid }
        }
        let t = (p - x[lo]) / (x[hi] - x[lo])
        return y[lo] + t * (y[hi] - y[lo])
    }
}
