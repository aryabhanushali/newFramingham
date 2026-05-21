import SwiftUI

@main
struct CVDIntelligenceApp: App {
    @StateObject private var store = HealthStore()

    var body: some Scene {
        WindowGroup {
            DashboardView()
                .environmentObject(store)
                .preferredColorScheme(.dark)
                .task { await store.bootstrap() }
        }
    }
}
