import SwiftUI

struct RiskRingView: View {
    let risk: RiskAssessment

    var body: some View {
        VStack(spacing: 16) {
            ZStack {
                Circle()
                    .stroke(Color.gray.opacity(0.25), lineWidth: 22)
                Circle()
                    .trim(from: 0, to: min(max(risk.calibratedProbability, 0.005), 1))
                    .stroke(
                        AngularGradient(colors: [risk.tier.color.opacity(0.55),
                                                 risk.tier.color],
                                        center: .center),
                        style: StrokeStyle(lineWidth: 22, lineCap: .round)
                    )
                    .rotationEffect(.degrees(-90))
                    .animation(.easeOut(duration: 0.6), value: risk.calibratedProbability)
                VStack(spacing: 2) {
                    Text("\(Int((risk.calibratedProbability * 100).rounded()))%")
                        .font(.system(size: 56, weight: .bold, design: .rounded))
                        .foregroundStyle(.white)
                    Text("10-year CVD risk")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }
            .frame(width: 220, height: 220)

            Text(risk.tier.rawValue)
                .font(.headline)
                .foregroundStyle(risk.tier.color)
        }
    }
}
