#pragma once
/*
 * em_builder.h — the episodic memory's record builder (2026-10-17): a pure state machine that turns the controller's
 * journal (em_journal.h), facts read from the live graph (through an abstract FactSource) and structural graph
 * events into commitment Records (em_schema.h). Header-only, Qt Core only, no DSR.
 *
 * Every call returns the records that OPENED or CLOSED during it, each a full copy (an open one has no t1); a record
 * that opens and closes in one call is returned twice, open then closed — the archive keeps the last line per id.
 *
 * Journal rules (spec §5–§6 + rulings R8–R10):
 *  - The FIRST wire only adopts its epoch and max serial; it creates no records (its history has no valid pre-facts).
 *  - A later wire with a different epoch (controller restarted): close every open record `lost`, then process ALL
 *    its entries. Same epoch: entries with serial <= last seen are ignored (idempotent polling); a gap (first new
 *    serial > last+1) closes every open record `lost` — any closing entry may have been dropped — then processes.
 *  - MissionStart opens a Human mission; StepStart a step whose parent is the innermost open step of that mission
 *    whose step_id is a strict dotted prefix ("2" for "2.1"), else the mission; ClaimStart a claim whose parent is the
 *    innermost open step, else the open human mission, else the open self mission, else a self mission opened now.
 *  - *End closes its match (mission key; (mission, step_id, attempt, iteration); the most recent open claim), with the
 *    journal's outcome. An *End with no open match is ignored. Closing a MISSION closes its open descendants first
 *    with the same outcome; closing a step does not touch its children (their own *End closes them).
 *  - Times: t0/t1 are the journal's stamps for journal-driven opens/closes; now_ms for what EM itself decides
 *    (`lost`, shutdown, a self mission opened/split at a check).
 *
 * Self mission: while drive mode is "affordances_only" (assumed until a DriveMode entry is seen, R9) and no human
 * mission is open, a Self mission is open. It closes `superseded` when a human mission starts or drive mode leaves
 * affordances_only, and `room_changed` (a new one opening) when current_room() differs from its room. Checked after
 * each journal poll and each on_events call; suspended after owner_lost until the next journal poll.
 *
 * Absent is not zero: a node the FactSource cannot read contributes no Fact; `delivered` exists only when both pre and
 * post carry cov_xx and cov_yy (ruling R2): 0.5*ln((cov_xx_pre*cov_yy_pre)/(cov_xx_post*cov_yy_post)).
 */

#include "em_journal.h"
#include "em_schema.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonValue>

namespace rc::em
{

struct FactSource                               // implemented by the agent over DSR; faked in tests
{
    virtual ~FactSource() = default;
    virtual std::optional<Fact> fact(std::string_view node) const = 0;   // nullopt = node absent
    virtual std::string current_room() const = 0;
};

namespace builder_detail
{
inline QJsonObject parse_object(const std::string& text)
{
    const QJsonDocument d = QJsonDocument::fromJson(QByteArray(text.data(), qsizetype(text.size())));
    return d.isObject() ? d.object() : QJsonObject{};
}

inline std::string compact(const QJsonValue& v)
{
    if (v.isObject()) return QJsonDocument(v.toObject()).toJson(QJsonDocument::Compact).toStdString();
    if (v.isArray()) return QJsonDocument(v.toArray()).toJson(QJsonDocument::Compact).toStdString();
    return v.toString().toStdString();
}

inline std::optional<double> det_xy(const std::vector<Fact>& fs)
{
    if (fs.empty()) return std::nullopt;
    const auto& v = fs.front().values;
    const auto xx = v.find("cov_xx"), yy = v.find("cov_yy");
    if (xx == v.end() or yy == v.end()) return std::nullopt;
    return xx->second * yy->second;
}

// "2" is a strict dotted prefix of "2.1" and "2.1.3", not of "2" or "21".
inline bool dotted_prefix(std::string_view parent, std::string_view child)
{
    return child.size() > parent.size() and child.starts_with(parent) and child[parent.size()] == '.';
}
}   // namespace builder_detail

class Builder
{
public:
    Builder(const FactSource& facts, std::function<std::string(std::uint64_t t0)> next_id)
        : facts_(facts), next_id_(std::move(next_id)) {}

    // One poll. Returns records that OPENED or CLOSED this call (to append to the archive / graph).
    std::vector<Record> on_journal(const WireView& w, std::uint64_t now_ms)
    {
        out_.clear();
        owner_present_ = true;
        const std::uint64_t max_serial =
            std::ranges::fold_left(w.entries, std::uint64_t{0}, [](std::uint64_t m, const JournalEntry& e)
                                   { return std::max(m, e.serial); });
        if (not primed_)                        // R8: adopt only
        {
            primed_ = true;
            epoch_ = w.epoch;
            last_seen_ = max_serial;
            return {};
        }
        if (w.epoch != epoch_)                  // controller restarted: everything it had open is lost
        {
            close_all("lost", now_ms);
            epoch_ = w.epoch;
            last_seen_ = 0;
        }
        else
        {
            const auto first_new = std::ranges::find_if(w.entries, [&](const JournalEntry& e) { return e.serial > last_seen_; });
            if (first_new != w.entries.end() and first_new->serial > last_seen_ + 1) close_all("lost", now_ms);
        }
        for (const JournalEntry& e : w.entries)
            if (e.serial > last_seen_) apply(e);
        last_seen_ = std::max(last_seen_, max_serial);
        check_self(now_ms);
        return std::exchange(out_, {});
    }

    // Attaches each event to every open record (dt from now_ms; the input dt_ms is ignored), then re-checks the self
    // mission (a current_moved event usually means the room changed).
    std::vector<Record> on_events(std::span<const Event> events, std::uint64_t now_ms)
    {
        out_.clear();
        for (Open& o : open_)
            for (const Event& e : events)
                o.rec.events.push_back(Event{.dt_ms = std::int64_t(now_ms) - std::int64_t(o.rec.t0), .kind = e.kind, .node = e.node});
        check_self(now_ms);
        return std::exchange(out_, {});
    }

    std::vector<Record> shutdown(std::uint64_t now_ms)      // close all open: "interrupted_by_em_shutdown"
    {
        out_.clear();
        close_all("interrupted_by_em_shutdown", now_ms);
        return std::exchange(out_, {});
    }

    std::vector<Record> owner_lost(std::uint64_t now_ms)    // controller node gone: close all open "lost"
    {
        out_.clear();
        close_all("lost", now_ms);
        owner_present_ = false;
        return std::exchange(out_, {});
    }

    std::vector<Record> open() const
    {
        return open_ | std::views::transform(&Open::rec) | std::ranges::to<std::vector>();
    }

private:
    struct Open
    {
        Record rec;
        std::string mission;        // controller mission key (human missions and their steps); "" otherwise
        bool self = false;          // the synthetic self mission
        std::string step_id;        // steps: the IR step id ("2.1")
    };

    const FactSource& facts_;
    std::function<std::string(std::uint64_t)> next_id_;
    std::vector<Open> open_;        // in open order: later = more inner
    std::vector<Record> out_;
    bool primed_ = false, owner_present_ = true;
    std::uint64_t epoch_ = 0, last_seen_ = 0;
    std::string drive_mode_ = "affordances_only";   // R9

    bool self_driving() const { return drive_mode_ == "affordances_only"; }

    Open* find_last(auto pred)
    {
        const auto it = std::ranges::find_if(open_ | std::views::reverse, pred);
        return it == (open_ | std::views::reverse).end() ? nullptr : &*it;
    }
    Open* human_mission(std::string_view key = {})
    {
        return find_last([&](const Open& o)
                         { return o.rec.level == Level::Mission and not o.self and (key.empty() or o.mission == key); });
    }
    Open* self_mission() { return find_last([](const Open& o) { return o.self; }); }

    std::vector<Fact> read(const std::string& node) const
    {
        if (node.empty()) return {};
        if (auto f = facts_.fact(node)) return {std::move(*f)};
        return {};
    }

    void open_record(Open o)
    {
        o.rec.id = next_id_(o.rec.t0);
        if (o.rec.room.empty()) o.rec.room = facts_.current_room();
        if (not o.rec.pre.empty()) o.rec.target_ltsm = o.rec.pre.front().ltsm_id;
        out_.push_back(o.rec);
        open_.push_back(std::move(o));
    }

    Open& open_self(std::uint64_t t)
    {
        Open o{.rec = {}, .mission = {}, .self = true, .step_id = {}};
        o.rec.level = Level::Mission;
        o.rec.author = Author::Self;
        o.rec.kind = "mission";
        o.rec.source = "self";
        o.rec.t0 = t;
        open_record(std::move(o));
        return open_.back();
    }

    // Closes the open record with this id; a mission takes its open descendants with it (innermost first).
    void close(const std::string& id, const std::string& outcome, std::uint64_t t)
    {
        const auto it = std::ranges::find(open_, id, [](const Open& o) { return o.rec.id; });
        if (it == open_.end()) return;
        if (it->rec.level == Level::Mission)
        {
            std::vector<std::string> family{id};
            std::vector<std::string> doomed;
            for (const Open& o : open_)        // open order: a parent precedes its children
                if (o.rec.id != id and std::ranges::contains(family, o.rec.parent))
                {
                    family.push_back(o.rec.id);
                    doomed.push_back(o.rec.id);
                }
            for (const std::string& d : doomed | std::views::reverse) finish(d, outcome, t);
        }
        finish(id, outcome, t);
    }

    void finish(const std::string& id, const std::string& outcome, std::uint64_t t)
    {
        const auto it = std::ranges::find(open_, id, [](const Open& o) { return o.rec.id; });
        if (it == open_.end()) return;
        Record r = std::move(it->rec);
        open_.erase(it);
        r.t1 = t;
        r.outcome = outcome;
        if (r.level != Level::Mission)
        {
            r.post = read(r.target);
            if (r.level == Level::Claim)
            {
                const auto pre = builder_detail::det_xy(r.pre), post = builder_detail::det_xy(r.post);
                if (pre and post)
                    if (const double d = 0.5 * std::log(*pre / *post); std::isfinite(d)) r.delivered = d;
            }
        }
        out_.push_back(std::move(r));
    }

    void close_all(const std::string& outcome, std::uint64_t t)
    {
        while (not open_.empty()) finish(open_.back().rec.id, outcome, t);
    }

    void apply(const JournalEntry& e)
    {
        using K = JournalEntry::Kind;
        switch (e.kind)
        {
        case K::MissionStart:
        {
            if (Open* s = self_mission()) close(s->rec.id, "superseded", e.t_ms);
            Open o{.rec = {}, .mission = e.mission, .self = false, .step_id = {}};
            o.rec.level = Level::Mission;
            o.rec.author = Author::Human;
            o.rec.kind = "mission";
            o.rec.action_json = e.text;
            o.rec.source = e.source;
            o.rec.t0 = e.t_ms;
            open_record(std::move(o));
            break;
        }
        case K::MissionEnd:
            if (Open* m = human_mission(e.mission); m and not e.mission.empty()) close(m->rec.id, e.outcome, e.t_ms);
            break;
        case K::StepStart:
        {
            Open* m = e.mission.empty() ? nullptr : human_mission(e.mission);
            if (not m) break;                   // its mission is not open (e.g. closed `lost` by a gap): no record
            const std::string mission_id = m->rec.id;
            const Author author = m->rec.author;
            const Open* parent = find_last([&](const Open& o)
                                           { return o.rec.level == Level::Step and o.mission == e.mission
                                                    and builder_detail::dotted_prefix(o.step_id, e.step_id); });
            Open o{.rec = {}, .mission = e.mission, .self = false, .step_id = {}};
            o.rec.level = Level::Step;
            o.rec.author = author;
            o.rec.parent = parent ? parent->rec.id : mission_id;
            o.rec.kind = e.op;
            o.rec.target = builder_detail::parse_object(e.text)["target"].toString().toStdString();
            o.rec.action_json = e.text;
            o.rec.t0 = e.t_ms;
            o.rec.attempt = e.attempt;
            o.rec.iteration = e.iteration;
            o.step_id = e.step_id;
            o.rec.pre = read(o.rec.target);
            open_record(std::move(o));
            break;
        }
        case K::StepEnd:
            if (Open* s = find_last([&](const Open& o)
                                    { return o.rec.level == Level::Step and o.mission == e.mission and o.step_id == e.step_id
                                             and o.rec.attempt == e.attempt and o.rec.iteration == e.iteration; }))
                close(s->rec.id, e.outcome, e.t_ms);
            break;
        case K::ClaimStart:
        {
            if (Open* c = find_last([](const Open& o) { return o.rec.level == Level::Claim; }))
                close(c->rec.id, "lost", e.t_ms);          // its ClaimEnd never reached us
            Open* parent = find_last([](const Open& o) { return o.rec.level == Level::Step; });
            if (not parent) parent = human_mission();
            if (not parent) parent = self_mission();
            if (not parent) parent = &open_self(e.t_ms);
            const QJsonObject text = builder_detail::parse_object(e.text);
            Open o{.rec = {}, .mission = {}, .self = false, .step_id = {}};
            o.rec.level = Level::Claim;
            o.rec.author = parent->rec.author;
            o.rec.parent = parent->rec.id;
            o.rec.kind = e.op;
            o.rec.target = text["target"].toString().toStdString();
            o.rec.action_json = builder_detail::compact(text["contract"]);
            o.rec.t0 = e.t_ms;
            o.rec.promised = e.promised;
            o.rec.alternatives_json = e.candidates;
            o.rec.pre = read(o.rec.target);
            open_record(std::move(o));
            break;
        }
        case K::ClaimEnd:
            if (Open* c = find_last([](const Open& o) { return o.rec.level == Level::Claim; }))
                close(c->rec.id, e.outcome, e.t_ms);
            break;
        case K::DriveMode:
            drive_mode_ = e.op;
            if (Open* s = self_mission(); s and not self_driving()) close(s->rec.id, "superseded", e.t_ms);
            break;
        }
    }

    void check_self(std::uint64_t now_ms)
    {
        if (not owner_present_) return;
        Open* s = self_mission();
        const bool wanted = self_driving() and not human_mission();
        if (s and not wanted) { close(s->rec.id, "superseded", now_ms); return; }
        if (not wanted) return;
        if (s and s->rec.room != facts_.current_room())
        {
            close(s->rec.id, "room_changed", now_ms);
            s = nullptr;
        }
        if (not s) open_self(now_ms);
    }
};

}   // namespace rc::em
