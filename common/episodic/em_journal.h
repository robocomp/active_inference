#pragma once
/*
 * em_journal.h — the controller's journal (2026-10-17): a bounded, serial-numbered log of mission / step / claim
 * events that the controller publishes so the episodic memory can reconstruct what was asked and what happened.
 * Header-only, Qt Core only, no DSR.
 *
 * Wire form: {"epoch":<ms>,"entries":[{...}]} compact JSON. `epoch` identifies the controller run (a changed epoch
 * means a restart). Entry keys are the field names; serial / t_ms / epoch are JSON numbers (doubles, exact below
 * 2^53). `promised` is ABSENT when unset (absent is not zero). `text` and `candidates` are carried as STRINGS that
 * contain JSON, verbatim — the consumer parses them (StepStart text = {"target":…,"grounding":{…}} per ruling R3;
 * ClaimStart text = {"affordance","producer","target","target_type","contract"}).
 *
 * The journal keeps the newest `capacity` entries; serials keep counting, so a gap tells the reader it missed some.
 *
 * Where it lives (final review, finding 4): the `em_journal` attribute of a DEDICATED controller-owned node — type
 * `intention`, name kJournalNodeName, `has` edge from the robot node — NOT the controller's agent node, which cortex's
 * 1 Hz AgentInfo heartbeat read-modify-writes (a stale copy reverted the journal, and re-asserting it re-broadcast
 * the whole wire). The controller creates it when missing, replaces a previous run's, deletes it on a graceful stop.
 */

#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonValue>
#include <QString>

namespace rc::em
{

inline constexpr const char* kJournalNodeName = "em_journal";   // the controller's journal node (type intention)
inline constexpr const char* kJournalNodeType = "intention";

struct JournalEntry
{
    std::uint64_t serial = 0, t_ms = 0;
    enum class Kind { MissionStart, MissionEnd, StepStart, StepEnd, ClaimStart, ClaimEnd, DriveMode } kind = Kind::DriveMode;
    std::string mission;        // controller mission serial as text ("m12"); "" outside a human mission
    std::string step_id;        // StepStart/End: IR step id ("2.1"); single commands use "1"
    std::string op;             // step op / affordance kind / drive mode name
    std::string source;         // MissionStart: chat|remote|panel
    std::string text;           // MissionStart: IR JSON or command text · StepStart: resolved grounding JSON
                                // ClaimStart: {"affordance","producer","target","target_type","contract"} JSON
    std::string outcome;        // *End: outcome word (claims: an aff_outcome word or "preempted")
    int attempt = 0, iteration = 0;
    std::optional<double> promised;     // ClaimStart: quoted epistemic_gain
    std::string candidates;             // ClaimStart: selector candidates JSON array
};

namespace journal_detail
{
inline constexpr std::pair<JournalEntry::Kind, const char*> kKinds[] = {
    {JournalEntry::Kind::MissionStart, "mission_start"}, {JournalEntry::Kind::MissionEnd, "mission_end"},
    {JournalEntry::Kind::StepStart, "step_start"},       {JournalEntry::Kind::StepEnd, "step_end"},
    {JournalEntry::Kind::ClaimStart, "claim_start"},     {JournalEntry::Kind::ClaimEnd, "claim_end"},
    {JournalEntry::Kind::DriveMode, "drive_mode"}};

inline QString q(const std::string& s) { return QString::fromUtf8(s.data(), qsizetype(s.size())); }
inline std::string s(const QJsonValue& v) { return v.toString().toStdString(); }

inline QJsonObject to_json(const JournalEntry& e)
{
    QJsonObject o;
    o["serial"] = double(e.serial);
    o["t_ms"] = double(e.t_ms);
    for (const auto& [k, n] : kKinds)
        if (k == e.kind) o["kind"] = QString::fromLatin1(n);
    o["mission"] = q(e.mission);
    o["step_id"] = q(e.step_id);
    o["op"] = q(e.op);
    o["source"] = q(e.source);
    o["text"] = q(e.text);
    o["outcome"] = q(e.outcome);
    o["attempt"] = e.attempt;
    o["iteration"] = e.iteration;
    if (e.promised) o["promised"] = *e.promised;
    o["candidates"] = q(e.candidates);
    return o;
}

inline std::optional<JournalEntry> entry_from_json(const QJsonObject& o)
{
    JournalEntry e;
    const std::string kind = s(o["kind"]);
    bool known = false;
    for (const auto& [k, n] : kKinds)
        if (kind == n) { e.kind = k; known = true; }
    if (not known) return std::nullopt;
    e.serial = std::uint64_t(o["serial"].toDouble());
    e.t_ms = std::uint64_t(o["t_ms"].toDouble());
    e.mission = s(o["mission"]);
    e.step_id = s(o["step_id"]);
    e.op = s(o["op"]);
    e.source = s(o["source"]);
    e.text = s(o["text"]);
    e.outcome = s(o["outcome"]);
    e.attempt = o["attempt"].toInt();
    e.iteration = o["iteration"].toInt();
    if (o.contains("promised") and o["promised"].isDouble()) e.promised = o["promised"].toDouble();
    e.candidates = s(o["candidates"]);
    return e;
}
}   // namespace journal_detail

class Journal
{
public:
    explicit Journal(std::uint64_t epoch_ms, std::size_t capacity = 256) : epoch_(epoch_ms), cap_(capacity) {}

    // Assigns serial = last+1, drops the oldest beyond capacity, marks dirty.
    std::uint64_t push(JournalEntry e)
    {
        e.serial = ++last_;
        entries_.push_back(std::move(e));
        while (entries_.size() > cap_) entries_.pop_front();
        dirty_ = true;
        return last_;
    }

    std::string to_wire() const
    {
        QJsonArray a;
        for (const JournalEntry& e : entries_) a.append(journal_detail::to_json(e));
        const QJsonObject o{{"epoch", double(epoch_)}, {"entries", a}};
        return QJsonDocument(o).toJson(QJsonDocument::Compact).toStdString();
    }

    bool dirty() const { return dirty_; }
    void clear_dirty() { dirty_ = false; }

private:
    std::uint64_t epoch_, last_ = 0;
    std::size_t cap_;
    std::deque<JournalEntry> entries_;
    bool dirty_ = false;
};

struct WireView { std::uint64_t epoch = 0; std::vector<JournalEntry> entries; };

// Strict: malformed JSON, a missing epoch/entries, or an unknown kind rejects the whole wire.
inline std::optional<WireView> journal_from_wire(std::string_view wire)
{
    QJsonParseError err;
    const QJsonDocument d = QJsonDocument::fromJson(QByteArray(wire.data(), qsizetype(wire.size())), &err);
    if (err.error != QJsonParseError::NoError or not d.isObject()) return std::nullopt;
    const QJsonObject o = d.object();
    if (not o["epoch"].isDouble() or not o["entries"].isArray()) return std::nullopt;
    WireView w;
    w.epoch = std::uint64_t(o["epoch"].toDouble());
    for (const QJsonValue& v : o["entries"].toArray())
    {
        auto e = journal_detail::entry_from_json(v.toObject());
        if (not e) return std::nullopt;
        w.entries.push_back(std::move(*e));
    }
    return w;
}

}   // namespace rc::em
