import SwiftUI

struct InsightCard: View {
    let insight: Insight
    let accent: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Image(systemName: icon)
                    .foregroundStyle(accent)
                Text("INSIGHT")
                    .font(.caption.weight(.bold))
                    .foregroundStyle(accent)
            }
            Text(insight.headline)
                .font(.headline)
                .foregroundStyle(.white)
            Text(insight.detail)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }

    private var icon: String {
        switch insight.primaryDriver {
        case .sedentary: return "figure.walk"
        case .sleep: return "bed.double"
        case .regularity: return "waveform.path"
        case .fitness: return "heart.fill"
        case .none: return "checkmark.seal.fill"
        }
    }
}
