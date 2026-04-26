// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "SlackAlert",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(name: "SlackAlert", targets: ["SlackAlert"])
    ],
    targets: [
        .executableTarget(
            name: "SlackAlert",
            linkerSettings: [
                .linkedLibrary("sqlite3")
            ]
        )
    ]
)
