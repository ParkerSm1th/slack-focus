import Foundation
import SQLite3
import SwiftUI

struct MessageLog: Identifiable, Hashable {
    let id: String
    let channel: String
    let text: String
    let score: Double?
    let level: String?
    let label: String?
    let createdAt: Double
    let sortTimestamp: Double
    let timestampText: String
}

struct Settings: Codable {
    var slack_app_token: String?
    var slack_user_token: String?
    var notify_min_level: String
    var thresholds: [String: Double]

    static let defaults = Settings(
        slack_app_token: nil,
        slack_user_token: nil,
        notify_min_level: "high",
        thresholds: ["low": 0.35, "medium": 0.55, "high": 0.75, "critical": 0.90]
    )
}

final class AppState: ObservableObject {
    @Published var appToken = ""
    @Published var userToken = ""
    @Published var notifyLevel = "high"
    @Published var lowThreshold = "0.35"
    @Published var mediumThreshold = "0.55"
    @Published var highThreshold = "0.75"
    @Published var criticalThreshold = "0.90"

    @Published var backfillDays = "14"
    @Published var backfillMaxConversations = "200"
    @Published var backfillTypes = "public_channel,private_channel,im,mpim"
    @Published var backfillChannels = ""

    @Published var manualExample = ""
    @Published var manualLabel = "critical"
    @Published var relabelLevel = "high"

    @Published var isListening = false
    @Published var isTraining = false
    @Published var isBackfilling = false
    @Published var status = "idle"
    @Published var processLog = ""
    @Published var messages: [MessageLog] = []
    @Published var selectedMessageID: String?

    let projectRoot: URL
    private let database: Database
    private var listenerProcess: Process?
    private var refreshTimer: Timer?

    init() {
        self.projectRoot = Self.resolveProjectRoot()
        self.database = Database(root: projectRoot)
        loadSettings()
        refreshMessages()
        startRefreshTimer()
        appendLog("Runtime folder: \(projectRoot.path)")
    }

    deinit {
        stopListening()
        refreshTimer?.invalidate()
    }

    var selectedMessage: MessageLog? {
        guard let selectedMessageID else { return nil }
        return messages.first { $0.id == selectedMessageID }
    }

    func loadSettings() {
        let settings = readSettings()
        appToken = settings.slack_app_token ?? ""
        userToken = settings.slack_user_token ?? ""
        notifyLevel = settings.notify_min_level
        lowThreshold = format(settings.thresholds["low"] ?? 0.35)
        mediumThreshold = format(settings.thresholds["medium"] ?? 0.55)
        highThreshold = format(settings.thresholds["high"] ?? 0.75)
        criticalThreshold = format(settings.thresholds["critical"] ?? 0.90)
    }

    func saveSettings() {
        guard let thresholds = currentThresholds() else {
            appendLog("Invalid thresholds. Need numbers ordered low <= medium <= high <= critical.")
            return
        }

        let settings = Settings(
            slack_app_token: appToken.trimmingCharacters(in: .whitespacesAndNewlines).nilIfEmpty,
            slack_user_token: userToken.trimmingCharacters(in: .whitespacesAndNewlines).nilIfEmpty,
            notify_min_level: notifyLevel,
            thresholds: thresholds
        )
        do {
            try writeSettings(settings)
            appendLog("Saved settings.")
            status = "settings saved"
        } catch {
            appendLog("Settings save failed: \(error.localizedDescription)")
        }
    }

    func startListening() {
        guard listenerProcess == nil else { return }
        saveSettings()
        appendLog("Starting listener...")
        let process = makeUVProcess(["run", "python", "-u", "-m", "slack_priority", "listen"])
        attachOutput(process)
        process.terminationHandler = { [weak self] _ in
            DispatchQueue.main.async {
                self?.appendLog("Listener exited code=\(process.terminationStatus)")
                self?.listenerProcess = nil
                self?.isListening = false
                self?.status = "listener stopped"
                self?.refreshMessages()
            }
        }

        do {
            try process.run()
            listenerProcess = process
            isListening = true
            status = "listening"
            appendLog("Listener process started pid=\(process.processIdentifier).")
        } catch {
            appendLog("Listener failed: \(error.localizedDescription)")
        }
    }

    func stopListening() {
        listenerProcess?.terminate()
        listenerProcess = nil
        isListening = false
        status = "stopped"
    }

    func toggleListening() {
        if isListening {
            stopListening()
        } else {
            startListening()
        }
    }

    func runBackfill() {
        guard !isBackfilling else { return }
        saveSettings()
        isBackfilling = true
        status = "backfilling"

        var args = ["run", "python", "-u", "-m", "slack_priority", "backfill", "--days", backfillDays]
        if !backfillTypes.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            args += ["--types", backfillTypes]
        }
        if !backfillMaxConversations.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            args += ["--max-conversations", backfillMaxConversations]
        }
        for channel in parsedChannels() {
            args += ["--channel", channel]
        }

        runWorker(args) { [weak self] in
            self?.isBackfilling = false
            self?.status = "backfill done"
            self?.refreshMessages()
        }
    }

    func trainAndRescore() {
        guard !isTraining else { return }
        isTraining = true
        status = "training"
        appendLog("Training model...")

        DispatchQueue.global(qos: .userInitiated).async {
            self.runBlocking(["run", "python", "-u", "-m", "slack_priority", "train", "--epochs", "12", "--local-weight", "20", "--public-limit", "3000"])
            self.runBlocking(["run", "python", "-u", "-m", "slack_priority", "rescore"])
            DispatchQueue.main.async {
                self.isTraining = false
                self.status = "trained"
                self.refreshMessages()
            }
        }
    }

    func addManualExample() {
        let text = manualExample.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        runWorker(["run", "python", "-u", "-m", "slack_priority", "label-text", text, "--label", manualLabel]) { [weak self] in
            self?.manualExample = ""
            self?.refreshMessages()
        }
    }

    func relabelSelected() {
        guard let selectedMessage else { return }
        do {
            try database.setLabel(messageID: selectedMessage.id, label: relabelLevel)
            appendLog("Relabeled \(selectedMessage.id) as \(relabelLevel).")
            refreshMessages()
        } catch {
            appendLog("Relabel failed: \(error.localizedDescription)")
        }
    }

    func refreshMessages() {
        DispatchQueue.global(qos: .userInitiated).async {
            let rows = self.database.recentMessages(limit: 1000)
            DispatchQueue.main.async {
                self.messages = rows
            }
        }
    }

    func openDataFolder() {
        let dataFolder = projectRoot.appendingPathComponent("data")
        try? FileManager.default.createDirectory(at: dataFolder, withIntermediateDirectories: true)
        NSWorkspace.shared.open(dataFolder)
    }

    private func startRefreshTimer() {
        refreshTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
            self?.refreshMessages()
        }
    }

    private func currentThresholds() -> [String: Double]? {
        guard
            let low = Double(lowThreshold),
            let medium = Double(mediumThreshold),
            let high = Double(highThreshold),
            let critical = Double(criticalThreshold),
            low <= medium,
            medium <= high,
            high <= critical
        else {
            return nil
        }
        return ["low": low, "medium": medium, "high": high, "critical": critical]
    }

    private func parsedChannels() -> [String] {
        backfillChannels
            .replacingOccurrences(of: ",", with: "\n")
            .split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    private func readSettings() -> Settings {
        let url = settingsURL
        guard let data = try? Data(contentsOf: url) else { return .defaults }
        return (try? JSONDecoder().decode(Settings.self, from: data)) ?? .defaults
    }

    private func writeSettings(_ settings: Settings) throws {
        let url = settingsURL
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let data = try JSONEncoder.pretty.encode(settings)
        try data.write(to: url, options: .atomic)
    }

    private var settingsURL: URL {
        projectRoot.appendingPathComponent("data/settings.json")
    }

    private func makeUVProcess(_ args: [String]) -> Process {
        let process = Process()
        let uv = uvPath()
        if uv.hasPrefix("/") {
            process.executableURL = URL(fileURLWithPath: uv)
            process.arguments = args
        } else {
            process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
            process.arguments = [uv] + args
        }
        process.currentDirectoryURL = projectRoot
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        process.environment = env
        return process
    }

    private func runWorker(_ args: [String], completion: @escaping () -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            self.runBlocking(args)
            DispatchQueue.main.async {
                completion()
            }
        }
    }

    private func runBlocking(_ args: [String]) {
        let process = makeUVProcess(args)
        attachOutput(process)
        do {
            try process.run()
            process.waitUntilExit()
        } catch {
            DispatchQueue.main.async {
                self.appendLog("Command failed: \(error.localizedDescription)")
            }
        }
    }

    private func attachOutput(_ process: Process) {
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            DispatchQueue.main.async {
                self?.appendLog(text.trimmingCharacters(in: .newlines))
            }
        }
    }

    private func appendLog(_ text: String) {
        guard !text.isEmpty else { return }
        let next = processLog + text + "\n"
        processLog = String(next.suffix(20_000))
    }

    private func uvPath() -> String {
        let homeUV = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".local/bin/uv").path
        for path in [homeUV, "/opt/homebrew/bin/uv", "/usr/local/bin/uv"] {
            if FileManager.default.isExecutableFile(atPath: path) {
                return path
            }
        }
        return "uv"
    }

    private static func resolveProjectRoot() -> URL {
        let fileManager = FileManager.default
        if let override = ProcessInfo.processInfo.environment["SLACK_FOCUS_ROOT"], !override.isEmpty {
            return URL(fileURLWithPath: override)
        }

        if
            let bundledRuntime = Bundle.main.resourceURL?.appendingPathComponent("SlackFocusRuntime"),
            fileManager.fileExists(atPath: bundledRuntime.appendingPathComponent("pyproject.toml").path),
            let supportRoot = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
        {
            let appSupport = supportRoot.appendingPathComponent("Slack Focus", isDirectory: true)
            let runtime = appSupport.appendingPathComponent("Runtime", isDirectory: true)
            do {
                try installBundledRuntime(from: bundledRuntime, to: runtime, fileManager: fileManager)
                return runtime
            } catch {
                return bundledRuntime
            }
        }

        return URL(fileURLWithPath: fileManager.currentDirectoryPath)
    }

    private static func installBundledRuntime(from bundled: URL, to runtime: URL, fileManager: FileManager) throws {
        try fileManager.createDirectory(at: runtime, withIntermediateDirectories: true)
        for name in ["slack_priority", "examples", "pyproject.toml", "uv.lock", ".python-version", "README.md"] {
            let source = bundled.appendingPathComponent(name)
            guard fileManager.fileExists(atPath: source.path) else { continue }
            let destination = runtime.appendingPathComponent(name)
            if fileManager.fileExists(atPath: destination.path) {
                try fileManager.removeItem(at: destination)
            }
            try fileManager.copyItem(at: source, to: destination)
        }
        try fileManager.createDirectory(at: runtime.appendingPathComponent("data"), withIntermediateDirectories: true)
        try fileManager.createDirectory(at: runtime.appendingPathComponent("models"), withIntermediateDirectories: true)
    }
}

struct ContentView: View {
    @StateObject private var app = AppState()

    var body: some View {
        HStack(spacing: 14) {
            controls
                .frame(width: 360)

            VStack(spacing: 12) {
                topBar
                messageList
                logPanel
                    .frame(height: 170)
            }
        }
        .padding(14)
        .background(AppTheme.background)
        .frame(minWidth: 1180, minHeight: 780)
    }

    private var controls: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Slack Focus")
                        .font(.system(size: 30, weight: .bold, design: .rounded))
                    Text("Local priority filter")
                        .foregroundStyle(.secondary)
                }
                .padding(.bottom, 2)

                section("Slack") {
                    VStack(alignment: .leading) {
                        SecureField("xapp- token", text: $app.appToken)
                        SecureField("xoxp- token", text: $app.userToken)
                        Button("Save Settings") { app.saveSettings() }
                            .buttonStyle(.borderedProminent)
                    }
                }

                section("Notify") {
                    VStack(alignment: .leading) {
                        Picker("Level", selection: $app.notifyLevel) {
                            ForEach(["low", "medium", "high", "critical"], id: \.self) { Text($0) }
                        }
                        Grid(alignment: .leading, horizontalSpacing: 8, verticalSpacing: 8) {
                            thresholdRow("Low", $app.lowThreshold)
                            thresholdRow("Medium", $app.mediumThreshold)
                            thresholdRow("High", $app.highThreshold)
                            thresholdRow("Critical", $app.criticalThreshold)
                        }
                    }
                }

                section("Backfill") {
                    VStack(alignment: .leading) {
                        HStack {
                            TextField("Days", text: $app.backfillDays)
                            TextField("Max convs", text: $app.backfillMaxConversations)
                        }
                        TextField("Types", text: $app.backfillTypes)
                        TextField("Channel IDs optional", text: $app.backfillChannels)
                        Button(app.isBackfilling ? "Backfilling..." : "Backfill") {
                            app.runBackfill()
                        }
                        .disabled(app.isBackfilling)
                    }
                }

                section("Training") {
                    VStack(alignment: .leading) {
                        Picker("Label", selection: $app.manualLabel) {
                            ForEach(["ignore", "low", "high", "critical"], id: \.self) { Text($0) }
                        }
                        TextEditor(text: $app.manualExample)
                            .frame(height: 80)
                            .scrollContentBackground(.hidden)
                            .background(AppTheme.fieldBackground, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                        HStack {
                            Button("Add Example") { app.addManualExample() }
                            Button(app.isTraining ? "Training..." : "Train + Rescore") {
                                app.trainAndRescore()
                            }
                            .disabled(app.isTraining)
                        }
                    }
                }

                Button("Open Data Folder") { app.openDataFolder() }
            }
            .padding(2)
        }
    }

    private var topBar: some View {
        HStack(spacing: 14) {
            LiveStatusBadge(isListening: app.isListening, status: app.status)
            Spacer()
            ListeningToggle(isListening: app.isListening) {
                app.toggleListening()
            }

            Button("Refresh") { app.refreshMessages() }
                .controlSize(.large)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .liquidPanel(cornerRadius: 22)
    }

    private func section<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title)
                .font(.headline)
                .foregroundStyle(.secondary)
            content()
        }
        .padding(14)
        .liquidPanel()
    }

    private func thresholdRow(_ title: String, _ binding: Binding<String>) -> some View {
        GridRow {
            Text(title).frame(width: 64, alignment: .leading)
            TextField(title, text: binding).frame(width: 80)
        }
    }

    private var messageList: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Messages + Scores")
                    .font(.title3.bold())
                Spacer()
                Text("\(app.messages.count) newest first")
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 16)
            .padding(.top, 14)

            ScrollView {
                LazyVStack(spacing: 0) {
                    ForEach(app.messages) { message in
                        Button {
                            app.selectedMessageID = message.id
                        } label: {
                            MessageRow(
                                message: message,
                                isSelected: app.selectedMessageID == message.id
                            )
                        }
                        .buttonStyle(.plain)
                        .accessibilityLabel(message.accessibilitySummary)
                    }
                }
                .padding(.horizontal, 10)
                .padding(.bottom, 8)
            }

            selectedMessagePanel
        }
        .liquidPanel(cornerRadius: 22)
    }

    private var selectedMessagePanel: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let message = app.selectedMessage {
                Text("\(message.timestampText) | \(message.channel)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text(message.text)
                    .font(.body)
                    .lineLimit(3)
                HStack {
                    Picker("Relabel", selection: $app.relabelLevel) {
                        ForEach(["ignore", "low", "high", "critical"], id: \.self) { Text($0) }
                    }
                    Button("Save Label") { app.relabelSelected() }
                    Spacer()
                    Text(message.id).foregroundStyle(.secondary).font(.caption)
                }
            } else {
                Text("Select a message to relabel.")
                    .foregroundStyle(.secondary)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(AppTheme.selectionBackground)
    }

    private var logPanel: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Process Log").font(.headline)
            ScrollViewReader { proxy in
                ScrollView {
                    Text(app.processLog.isEmpty ? "No process logs yet." : app.processLog)
                        .font(.system(.caption, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                    Color.clear
                        .frame(height: 1)
                        .id("process-log-bottom")
                }
                .onReceive(app.$processLog) { _ in
                    DispatchQueue.main.async {
                        proxy.scrollTo("process-log-bottom", anchor: .bottom)
                    }
                }
            }
        }
        .padding(12)
        .liquidPanel(cornerRadius: 18)
    }
}

struct MessageRow: View {
    let message: MessageLog
    let isSelected: Bool

    var body: some View {
        HStack(spacing: 10) {
            Text(scoreText)
                .font(.system(.caption, design: .monospaced))
                .frame(width: 50, alignment: .trailing)
            Text(message.level ?? "-")
                .frame(width: 64, alignment: .leading)
                .foregroundStyle(levelColor)
            Text(message.label ?? "")
                .frame(width: 54, alignment: .leading)
                .foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 2) {
                Text(message.channel)
                    .lineLimit(1)
                Text(message.timestampText)
                    .font(.system(.caption2, design: .monospaced))
                    .foregroundStyle(.tertiary)
            }
            .frame(width: 150, alignment: .leading)
            .foregroundStyle(.secondary)
            Text(message.text.replacingOccurrences(of: "\n", with: " "))
                .lineLimit(1)
            Spacer()
        }
        .font(.system(size: 13))
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .background(isSelected ? AppTheme.selectionBackground : Color.clear, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(alignment: .bottom) {
            Rectangle()
                .fill(Color.white.opacity(0.06))
                .frame(height: 1)
                .padding(.leading, 190)
        }
    }

    private var scoreText: String {
        guard let score = message.score else { return "-" }
        return String(format: "%.3f", score)
    }

    private var levelColor: Color {
        switch message.level {
        case "critical": return .red
        case "high": return .orange
        case "medium": return .yellow
        case "low": return .blue
        default: return .secondary
        }
    }
}

struct ListeningToggle: View {
    let isListening: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 10) {
                Image(systemName: isListening ? "pause.fill" : "play.fill")
                    .font(.system(size: 13, weight: .bold))
                Text(isListening ? "Disable Listening" : "Enable Listening")
                    .font(.system(size: 15, weight: .semibold))
            }
            .foregroundStyle(isListening ? Color.white : Color.black)
            .padding(.horizontal, 18)
            .frame(height: 36)
            .background(
                Capsule(style: .continuous)
                    .fill(isListening ? Color.red.opacity(0.92) : Color.green.opacity(0.92))
            )
            .overlay {
                Capsule(style: .continuous)
                    .stroke(Color.white.opacity(0.22), lineWidth: 1)
            }
            .shadow(color: (isListening ? Color.red : Color.green).opacity(0.25), radius: 10, y: 4)
        }
        .buttonStyle(.plain)
        .liquidControl(tint: isListening ? .red : .green)
        .accessibilityLabel(isListening ? "Disable listening" : "Enable listening")
    }
}

struct LiveStatusBadge: View {
    let isListening: Bool
    let status: String

    var body: some View {
        HStack(spacing: 12) {
            PulsingDot(isActive: isListening)
            VStack(alignment: .leading, spacing: 2) {
                Text(isListening ? "Listening enabled" : "Listening disabled")
                    .font(.headline)
                Text(status)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .accessibilityLabel(isListening ? "Listening enabled" : "Listening disabled")
    }
}

struct PulsingDot: View {
    let isActive: Bool
    @State private var pulse = false

    var body: some View {
        ZStack {
            if isActive {
                Circle()
                    .fill(Color.green.opacity(0.28))
                    .frame(width: 30, height: 30)
                    .scaleEffect(pulse ? 1.3 : 0.75)
                    .opacity(pulse ? 0.05 : 1.0)
            }
            Circle()
                .fill(isActive ? Color.green : Color.secondary)
                .frame(width: 12, height: 12)
                .shadow(color: isActive ? Color.green.opacity(0.8) : .clear, radius: 8)
        }
        .frame(width: 34, height: 34)
        .onAppear {
            pulse = false
            withAnimation(.easeInOut(duration: 1.05).repeatForever(autoreverses: false)) {
                pulse = true
            }
        }
    }
}

enum AppTheme {
    static let background = LinearGradient(
        colors: [
            Color(red: 0.060, green: 0.060, blue: 0.064),
            Color(red: 0.095, green: 0.095, blue: 0.102)
        ],
        startPoint: .topLeading,
        endPoint: .bottomTrailing
    )
    static let panelFill = Color.white.opacity(0.045)
    static let panelStroke = Color.white.opacity(0.075)
    static let fieldBackground = Color.black.opacity(0.20)
    static let selectionBackground = Color.white.opacity(0.070)
}

struct LiquidPanelModifier: ViewModifier {
    let cornerRadius: CGFloat

    func body(content: Content) -> some View {
        content
            .padding(0)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
            .background(AppTheme.panelFill, in: RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .stroke(AppTheme.panelStroke, lineWidth: 1)
            }
            .shadow(color: Color.black.opacity(0.20), radius: 16, y: 10)
    }
}

extension View {
    func liquidPanel(cornerRadius: CGFloat = 18) -> some View {
        modifier(LiquidPanelModifier(cornerRadius: cornerRadius))
    }

    func liquidControl(tint: Color) -> some View {
        modifier(LiquidControlModifier(tint: tint))
    }
}

struct LiquidControlModifier: ViewModifier {
    let tint: Color

    func body(content: Content) -> some View {
        content
    }
}

final class Database {
    private let path: String

    init(root: URL) {
        self.path = root.appendingPathComponent("data/slack_priority.sqlite3").path
    }

    func recentMessages(limit: Int) -> [MessageLog] {
        guard FileManager.default.fileExists(atPath: path) else { return [] }
        var db: OpaquePointer?
        guard sqlite3_open_v2(path, &db, SQLITE_OPEN_READONLY, nil) == SQLITE_OK else { return [] }
        defer { sqlite3_close(db) }

        let sql = """
        select
            m.message_id,
            coalesce(m.channel_name, m.channel_id, '') as channel,
            m.text,
            p.score,
            p.level,
            l.label,
            m.created_at,
            coalesce(cast(nullif(m.ts, '') as real), m.created_at) as sort_ts
        from messages m
        left join predictions p on p.message_id = m.message_id
        left join labels l on l.message_id = m.message_id
        order by
            sort_ts desc,
            m.created_at desc
        limit ?
        """
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &statement, nil) == SQLITE_OK else { return [] }
        defer { sqlite3_finalize(statement) }
        sqlite3_bind_int(statement, 1, Int32(limit))

        var rows: [MessageLog] = []
        while sqlite3_step(statement) == SQLITE_ROW {
            let sortTimestamp = sqlite3_column_double(statement, 7)
            rows.append(
                MessageLog(
                    id: columnText(statement, 0),
                    channel: columnText(statement, 1),
                    text: columnText(statement, 2),
                    score: sqlite3_column_type(statement, 3) == SQLITE_NULL ? nil : sqlite3_column_double(statement, 3),
                    level: optionalText(statement, 4),
                    label: optionalText(statement, 5),
                    createdAt: sqlite3_column_double(statement, 6),
                    sortTimestamp: sortTimestamp,
                    timestampText: formatTimestamp(sortTimestamp)
                )
            )
        }
        return rows
    }

    func setLabel(messageID: String, label: String) throws {
        guard let score = ["ignore": 0.0, "low": 0.35, "high": 0.80, "critical": 1.0][label] else {
            throw AppError("Unknown label \(label)")
        }
        var db: OpaquePointer?
        guard sqlite3_open(path, &db) == SQLITE_OK else { throw AppError("Could not open database") }
        defer { sqlite3_close(db) }

        let sql = """
        insert into labels (message_id, label, score, updated_at)
        values (?, ?, ?, ?)
        on conflict(message_id) do update set
            label = excluded.label,
            score = excluded.score,
            updated_at = excluded.updated_at
        """
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &statement, nil) == SQLITE_OK else {
            throw AppError("Could not prepare label update")
        }
        defer { sqlite3_finalize(statement) }

        sqlite3_bind_text(statement, 1, messageID, -1, SQLITE_TRANSIENT)
        sqlite3_bind_text(statement, 2, label, -1, SQLITE_TRANSIENT)
        sqlite3_bind_double(statement, 3, score)
        sqlite3_bind_double(statement, 4, Date().timeIntervalSince1970)
        guard sqlite3_step(statement) == SQLITE_DONE else {
            throw AppError("Could not write label")
        }
    }
}

struct AppError: LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}

@main
struct SlackAlertSwiftApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .windowStyle(.titleBar)
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
            NSApp.windows.forEach { window in
                window.title = "Slack Focus"
                window.makeKeyAndOrderFront(nil)
            }
        }
    }
}

private let SQLITE_TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

private func columnText(_ statement: OpaquePointer?, _ index: Int32) -> String {
    guard let cString = sqlite3_column_text(statement, index) else { return "" }
    return String(cString: cString)
}

private func optionalText(_ statement: OpaquePointer?, _ index: Int32) -> String? {
    guard sqlite3_column_type(statement, index) != SQLITE_NULL else { return nil }
    return columnText(statement, index)
}

private func format(_ value: Double) -> String {
    String(format: "%.2f", value)
}

private func formatTimestamp(_ value: Double) -> String {
    guard value > 0 else { return "-" }
    let date = Date(timeIntervalSince1970: value)
    return MessageTimestampFormatter.string(from: date)
}

private enum MessageTimestampFormatter {
    private static let queue = DispatchQueue(label: "slack-focus.timestamp-formatter")

    private static let today: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "h:mm a"
        return formatter
    }()

    private static let older: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "MMM d, h:mm a"
        return formatter
    }()

    static func string(from date: Date) -> String {
        queue.sync {
            let formatter = Calendar.current.isDateInToday(date) ? today : older
            return formatter.string(from: date)
        }
    }
}

private extension String {
    var nilIfEmpty: String? {
        isEmpty ? nil : self
    }
}

private extension JSONEncoder {
    static var pretty: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return encoder
    }
}

private extension MessageLog {
    var accessibilitySummary: String {
        let scoreText = score.map { String(format: "%.3f", $0) } ?? "no score"
        return "\(level ?? "unknown level"), score \(scoreText), \(channel), \(text)"
    }
}
