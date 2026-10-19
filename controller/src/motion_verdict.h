#pragma once
/*
 * motion_verdict.h — WHY IS THE ROBOT NOT MOVING (2026-10-19, user: "we need to know why the robot is not moving …
 * there has to be a way to see why it is blocked").
 *
 * Live try97 leg 2: the robot stood in open space for ~7 minutes sending NO base command. The door passage was at step
 * Cross waiting for 'aff_door_1_cross', which door_concept had WITHDRAWN (P(passable)=0.12, clear span 0.65 m < 0.70 m
 * of body), and the passage's own rules (reached_door ⇒ no fallback; opened ⇒ a re-offered `open` ignored) left it
 * waiting. Nothing on screen or in the log said so: a throttled `[ctrl-idle] fresh_lidar=false`, a residual-cells
 * line, and much earlier "IDLE in command mode — waiting for a target."
 *
 * ★ OBSERVABILITY ONLY. Nothing in this file is read by a driving decision. Each branch of the control pipeline that
 *   ends the cycle without motion (or limits it) NOTES its cause here, with the numbers it decided on; the worker adds
 *   what it alone knows (presence, the LiDAR watchdog, Run/Pause, the output arm); at the end of the cycle the ONE
 *   binding cause is selected and shown permanently in the toolbar, logged `[why]` on change, and appended to
 *   why_log.jsonl next to base_commands.jsonl (same steady clock, so the two files join on t_ms).
 *
 * ★ PRECEDENCE IS THE DECLARATION ORDER OF Cause. When several causes hold in one cycle, the one that would still stop
 *   the robot if all the later ones vanished binds: a disarmed output stops it whatever the planner says; a passage
 *   waiting on a withdrawn affordance explains the selector's "nothing eligible", not the other way round.
 *
 * ★ A ZERO COMMAND NOBODY EXPLAINED IS ITSELF A VERDICT: "UNEXPLAINED zero command at <fn:line>", class Fault. Hiding
 *   it behind a generic "idle" is exactly the failure this file exists to end.
 *
 * Header-only, no Qt, no DSR: tests/test_motion_verdict.cpp compiles it alone.
 */

#include <algorithm>
#include <cstdint>
#include <format>
#include <map>
#include <optional>
#include <source_location>
#include <string>
#include <string_view>
#include <vector>

namespace rc::why
{

/// The colour of the chip: who has to act for the robot to move.
enum class Klass : std::uint8_t
{
    Driving,          ///< a non-zero command is going out and nothing limits it
    Deliberate,       ///< standing still ON PURPOSE: dwell, orienting, lock-on, arrived, clearing a doorway
    Operator,         ///< Run not pressed, paused, output disarmed — a human has to act
    WaitingOnAgent,   ///< another agent must publish something (an affordance, the pose, the room)
    BlockedByWorld,   ///< the world refuses the body: no route, live returns, a wedge, an unreachable standpoint
    Fault,            ///< a stream stalled, presence lost, or a zero command no branch explained
};

/// Every way the controller ends up not moving. ★ DECLARATION ORDER = PRECEDENCE (earlier binds).
enum class Cause : std::uint8_t
{
    PresenceDegraded,   ///< a required peer is missing: the pipeline is not Operating
    LidarStalled,       ///< the LiDAR watchdog holds the pipeline
    Halted,             ///< Run not pressed / mission stopped / paused
    Disarmed,           ///< the base output is disarmed: commands are computed and discarded
    NoWorld,            ///< room / robot nodes or the room polygon are missing
    NoPose,             ///< no robot pose in the room frame
    RelativeMotion,     ///< a "turn/move" order owns the base
    Escape,             ///< the physical-stuck escape manoeuvre owns the base
    ReturnsBackoff,     ///< backing off live returns that hem the body in
    PassageWaiting,     ///< the door passage waits for an affordance that is not on offer
    PassageClearing,    ///< the passage drives on through the doorway on its anchor
    AnchorWaiting,      ///< the cross's interaction anchor is stale: wait for a fresh one
    AnchorHandBack,     ///< the anchored cross was handed back (timeout / not a crossing)
    MissionWaiting,     ///< a mission step holds the base (wait)
    Dwell,              ///< post-affordance dwell: looking at what was just acquired
    TargetRejected,     ///< this cycle's target was refused (outside room, unreachable, infeasible)
    NoRoute,            ///< the planner produced no route / the route could not be built or repaired
    Blockage,           ///< the follower hit a blockage and dropped the plan to re-route
    PathShort,          ///< the path ends short of the goal
    ReturnsHold,        ///< the live-returns net capped the command (to zero or below the ask)
    LockOn,             ///< the lock-on micro-search owns the base
    Orienting,          ///< an orient affordance turns in place
    Arrived,            ///< the goal was reached this cycle
    MissionDone,        ///< the route mission completed
    NoTarget,           ///< nothing to drive to: no click, no order, nothing eligible
    TrackerZero,        ///< the tracker itself asked for zero with a plan held
    Unexplained,        ///< nobody noted a cause for a zero command
    Driving,            ///< moving, nothing binding
};

[[nodiscard]] constexpr Klass klass_of(Cause c)
{
    switch (c)
    {
        case Cause::Driving: return Klass::Driving;
        case Cause::RelativeMotion: case Cause::PassageClearing: case Cause::Dwell: case Cause::LockOn:
        case Cause::Orienting: case Cause::Arrived: case Cause::MissionDone: return Klass::Deliberate;
        case Cause::Halted: case Cause::Disarmed: return Klass::Operator;
        case Cause::NoWorld: case Cause::NoPose: case Cause::PassageWaiting: case Cause::AnchorWaiting:
        case Cause::MissionWaiting: case Cause::NoTarget: return Klass::WaitingOnAgent;
        case Cause::Escape: case Cause::ReturnsBackoff: case Cause::AnchorHandBack: case Cause::TargetRejected:
        case Cause::NoRoute: case Cause::Blockage: case Cause::PathShort: case Cause::ReturnsHold:
            return Klass::BlockedByWorld;
        case Cause::PresenceDegraded: case Cause::LidarStalled: case Cause::TrackerZero: case Cause::Unexplained:
            return Klass::Fault;
    }
    return Klass::Fault;
}

[[nodiscard]] constexpr std::string_view klass_name(Klass k)
{
    switch (k)
    {
        case Klass::Driving: return "driving";
        case Klass::Deliberate: return "deliberate";
        case Klass::Operator: return "operator";
        case Klass::WaitingOnAgent: return "waiting-on-agent";
        case Klass::BlockedByWorld: return "blocked-by-world";
        case Klass::Fault: return "fault";
    }
    return "?";
}

/// Short upper-case word for the chip and the log.
[[nodiscard]] constexpr std::string_view cause_name(Cause c)
{
    switch (c)
    {
        case Cause::PresenceDegraded: return "PRESENCE";
        case Cause::LidarStalled: return "LIDAR STALLED";
        case Cause::Halted: return "HALTED";
        case Cause::Disarmed: return "DISARMED";
        case Cause::NoWorld: return "NO WORLD";
        case Cause::NoPose: return "NO POSE";
        case Cause::RelativeMotion: return "MOVE ORDER";
        case Cause::Escape: return "ESCAPING";
        case Cause::ReturnsBackoff: return "BACKING OFF";
        case Cause::PassageWaiting: return "WAITING";
        case Cause::PassageClearing: return "CLEARING DOORWAY";
        case Cause::AnchorWaiting: return "ANCHOR STALE";
        case Cause::AnchorHandBack: return "CROSS HANDED BACK";
        case Cause::MissionWaiting: return "MISSION WAIT";
        case Cause::Dwell: return "DWELL";
        case Cause::TargetRejected: return "TARGET REFUSED";
        case Cause::NoRoute: return "NO ROUTE";
        case Cause::Blockage: return "BLOCKED";
        case Cause::PathShort: return "PATH SHORT";
        case Cause::ReturnsHold: return "HELD BY RETURNS";
        case Cause::LockOn: return "LOCK-ON";
        case Cause::Orienting: return "ORIENTING";
        case Cause::Arrived: return "ARRIVED";
        case Cause::MissionDone: return "MISSION DONE";
        case Cause::NoTarget: return "NO TARGET";
        case Cause::TrackerZero: return "TRACKER ZERO";
        case Cause::Unexplained: return "UNEXPLAINED";
        case Cause::Driving: return "DRIVING";
    }
    return "?";
}

/// One cause noted by one branch in one cycle.
struct Note
{
    Cause cause = Cause::Unexplained;
    std::string detail;
    std::string site;   ///< "fn:line" of the branch that noted it
};

/// "fn:line" with only the bare function name: source_location::function_name() carries the whole signature.
[[nodiscard]] inline std::string site_of(std::string_view fn, std::uint32_t line)
{
    if (const auto paren = fn.find('('); paren != std::string_view::npos) fn = fn.substr(0, paren);
    if (const auto sp = fn.rfind(' '); sp != std::string_view::npos) fn = fn.substr(sp + 1);
    if (const auto cc = fn.rfind("::"); cc != std::string_view::npos) fn = fn.substr(cc + 2);
    return std::format("{}:{}", fn, line);
}
[[nodiscard]] inline std::string site_of(const std::source_location &loc) { return site_of(loc.function_name(), loc.line()); }

/// Everything noted during one control cycle. Owned by the control thread; reset by begin().
class Evidence
{
public:
    void begin()
    {
        notes_.clear();
        zero_site_.clear();
        context_.clear();
        driving_detail_.clear();
    }
    void note(Cause c, std::string detail, std::source_location loc = std::source_location::current())
    {
        notes_.push_back({c, std::move(detail), site_of(loc)});
    }
    void add(Note n) { notes_.push_back(std::move(n)); }
    /// What the drive is doing when nothing binds ("to 'aff_x' 0.30 m/s"), set where the command is sent.
    void set_driving_detail(std::string d) { driving_detail_ = std::move(d); }
    /// The branch that ended the cycle with a zero command (note_no_command / stop sites), for the UNEXPLAINED verdict.
    void note_zero_site(std::string site) { zero_site_ = std::move(site); }
    /// Appended to every verdict: e.g. the running mission step.
    void set_context(std::string c) { context_ = std::move(c); }

    [[nodiscard]] const std::vector<Note> &notes() const { return notes_; }
    [[nodiscard]] const std::string &zero_site() const { return zero_site_; }
    [[nodiscard]] const std::string &context() const { return context_; }
    [[nodiscard]] const std::string &driving_detail() const { return driving_detail_; }

private:
    std::vector<Note> notes_;
    std::string zero_site_;
    std::string context_;
    std::string driving_detail_;
};

/// The verdict of one cycle.
struct Verdict
{
    bool driving = false;   ///< a non-zero command is the one in force
    Cause cause = Cause::Unexplained;
    std::string detail;
    std::string site;       ///< where the binding cause was noted ("" for Driving / Unexplained)
    [[nodiscard]] Klass klass() const { return klass_of(cause); }
};

/// Pick the binding cause. `driving` = the command in force is non-zero. `fallback_zero_site` names the last stop the
/// motion commander saw, used when no branch noted one.
[[nodiscard]] inline Verdict select(const Evidence &ev, bool driving, const std::string &fallback_zero_site = {},
                                    const std::string &driving_detail = {})
{
    Verdict v;
    v.driving = driving;
    const Note *best = nullptr;
    for (const auto &n : ev.notes())   // stable: the first note of the winning cause keeps its detail
        if (best == nullptr or static_cast<int>(n.cause) < static_cast<int>(best->cause)) best = &n;
    if (best != nullptr)
    {
        v.cause = best->cause;
        v.detail = best->detail;
        v.site = best->site;
    }
    else if (driving)
    {
        v.cause = Cause::Driving;
        v.detail = not ev.driving_detail().empty() ? ev.driving_detail() : driving_detail;
    }
    else
    {
        v.cause = Cause::Unexplained;
        const std::string &site = not ev.zero_site().empty() ? ev.zero_site() : fallback_zero_site;
        v.detail = std::format("UNEXPLAINED zero command at {}", site.empty() ? std::string("<no stop site recorded>") : site);
        v.site = site;
    }
    if (not ev.context().empty())
        v.detail += (v.detail.empty() ? "" : " · ") + ev.context();
    return v;
}

[[nodiscard]] inline std::string format_duration(std::uint64_t ms)
{
    const std::uint64_t s = ms / 1000;
    if (s < 60) return std::format("{}.{} s", s, (ms % 1000) / 100);
    if (s < 3600) return std::format("{}m{:02}s", s / 60, s % 60);
    return std::format("{}h{:02}m", s / 3600, (s / 60) % 60);
}

/// What the toolbar shows: a copy, so the GUI never touches the control thread's state.
struct View
{
    bool valid = false;
    bool driving = false;
    Cause cause = Cause::Unexplained;
    Klass klass = Klass::Fault;
    std::string detail;
    std::uint64_t duration_ms = 0;
};

/// Since-when + change-only logging. Control thread only.
class Tracker
{
public:
    struct Emit
    {
        std::string log_line;   ///< "[why] …" — "" = nothing to log this cycle
        bool changed = false;   ///< a new cause (append to why_log.jsonl)
    };

    /// Feed one cycle's verdict. Logs on a change of (cause, driving), and every heartbeat_ms while unchanged and not
    /// driving (the "why was it stopped from t1 to t2" question needs the stretch to be visible, not one line at t1).
    Emit update(const Verdict &v, std::uint64_t now_ms)
    {
        Emit e;
        const bool changed = not has_ or v.cause != cur_.cause or v.driving != cur_.driving;
        if (changed)
        {
            const std::string was = has_ ? std::format(" (was {} for {})", cause_name(cur_.cause),
                                                       format_duration(now_ms - since_ms_))
                                         : std::string{};
            cur_ = v;
            has_ = true;
            since_ms_ = now_ms;
            last_log_ms_ = now_ms;
            e.changed = true;
            e.log_line = std::format("[why] {} {} — {}{}", v.driving ? "MOVING" : "STOPPED", cause_name(v.cause),
                                     v.detail, was);
            return e;
        }
        cur_ = v;   // fresh numbers in the detail, same cause
        if (not v.driving and now_ms - last_log_ms_ >= kHeartbeatMs)
        {
            last_log_ms_ = now_ms;
            e.log_line = std::format("[why] still STOPPED {} for {} — {}", cause_name(v.cause),
                                     format_duration(now_ms - since_ms_), v.detail);
        }
        return e;
    }

    [[nodiscard]] View view(std::uint64_t now_ms) const
    {
        if (not has_) return {};
        return {.valid = true, .driving = cur_.driving, .cause = cur_.cause, .klass = cur_.klass(),
                .detail = cur_.detail, .duration_ms = now_ms >= since_ms_ ? now_ms - since_ms_ : 0};
    }
    [[nodiscard]] std::uint64_t since_ms() const { return since_ms_; }

    static constexpr std::uint64_t kHeartbeatMs = 10000;   // the user's "once every 10 s while unchanged"

private:
    bool has_ = false;
    Verdict cur_;
    std::uint64_t since_ms_ = 0;
    std::uint64_t last_log_ms_ = 0;
};

/// The JSONL row for why_log.jsonl. Strings escaped for JSON (quotes, backslashes, control characters).
[[nodiscard]] inline std::string json_escape(std::string_view s)
{
    std::string out;
    out.reserve(s.size() + 8);
    for (const char ch : s)
    {
        switch (ch)
        {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(ch) < 0x20) out += std::format("\\u{:04x}", static_cast<int>(ch));
                else out += ch;
        }
    }
    return out;
}
[[nodiscard]] inline std::string json_row(const Verdict &v, std::uint64_t t_ms)
{
    return std::format(R"({{"t_ms":{},"driving":{},"cause":"{}","class":"{}","detail":"{}","site":"{}"}})", t_ms,
                       v.driving ? "true" : "false", cause_name(v.cause), klass_name(v.klass()), json_escape(v.detail),
                       json_escape(v.site));
}

// ── WHAT AN AFFORDANCE LAST SAID, KEPT AFTER ITS NODE IS GONE ──────────────────────────────────────────────────────
// A withdrawn affordance is DELETED from the graph, and with it everything that would explain the wait. The controller
// keeps, per name, the last sighting: its protocol state and the precondition as the controller could read it then
// (the producer's own `why` text is not on the wire — see the report — so the precondition is composed from what the
// parent node publishes, e.g. door_open_prob / door_phi_rad / width_m for a door).
struct Sighting
{
    std::string state;          ///< offered / executing / completed
    std::string outcome;        ///< aff_outcome ("" = none)
    std::string precondition;   ///< what the controller could read of the precondition, in words
    std::uint64_t t_ms = 0;
};

class AffordanceMemory
{
public:
    /// The node is in the graph this cycle.
    void seen(const std::string &name, Sighting s)
    {
        auto &e = map_[name];
        e.last = std::move(s);
        e.gone_since_ms.reset();
    }
    /// The node is NOT in the graph this cycle.
    void absent(const std::string &name, std::uint64_t now_ms)
    {
        auto &e = map_[name];
        if (e.last.has_value() and not e.gone_since_ms.has_value()) e.gone_since_ms = now_ms;
    }
    [[nodiscard]] std::optional<Sighting> last(const std::string &name) const
    {
        const auto it = map_.find(name);
        if (it == map_.end()) return std::nullopt;
        return it->second.last;
    }
    [[nodiscard]] std::optional<std::uint64_t> gone_since(const std::string &name) const
    {
        const auto it = map_.find(name);
        if (it == map_.end()) return std::nullopt;
        return it->second.gone_since_ms;
    }
    /// "'aff_door_1_cross' withdrawn by door_concept 412 s ago (last precondition: …)" or "… never offered".
    [[nodiscard]] std::string describe_absent(const std::string &name, const std::string &producer,
                                              std::uint64_t now_ms) const
    {
        const auto l = last(name);
        if (not l.has_value())
            return std::format("'{}' is not in the graph — {} has not offered it since this controller started", name,
                               producer);
        const auto g = gone_since(name);
        std::string s = std::format("'{}' withdrawn by {}{} (last seen {}{}", name, producer,
                                    g.has_value() ? std::format(" {} ago", format_duration(now_ms - *g)) : std::string{},
                                    l->state, l->outcome.empty() ? std::string{} : ", outcome " + l->outcome);
        if (not l->precondition.empty()) s += "; last precondition: " + l->precondition;
        return s + ")";
    }

private:
    struct Entry
    {
        std::optional<Sighting> last;
        std::optional<std::uint64_t> gone_since_ms;
    };
    std::map<std::string, Entry> map_;
};

// ── THE DOOR PASSAGE'S WAIT, IN WORDS ──────────────────────────────────────────────────────────────────────────────
/// What the session reads of the passage each cycle (controller_door_passage.h View + the graph).
struct PassageFacts
{
    std::string door;           ///< "door_1"
    std::string step;           ///< "approach" / "open" / "cross" / "close"
    std::string required;       ///< "aff_door_1_cross"
    std::string state;          ///< offered / executing / completed / not offered (= absent from the graph)
    bool reached_door = false;  ///< the passage will not fall back to earlier steps
    bool opened = false;        ///< `open` succeeded: a re-offered `open` is ignored
    bool clearing = false;      ///< going on through the doorway on the anchor
    int attempts = 0;
    std::string last_outcome;
    std::string door_now;       ///< the door's precondition numbers now ("" = door node gone)
    std::string producer = "door_concept";
};

/// The passage's note for this cycle, or nullopt when the passage is not what holds the robot (idle, or its step is
/// on offer / executing — then the drive itself, or the selector, is what the verdict must describe).
[[nodiscard]] inline std::optional<Note> passage_note(const PassageFacts &f, const AffordanceMemory &mem,
                                                      std::uint64_t now_ms,
                                                      std::source_location loc = std::source_location::current())
{
    if (f.door.empty() or f.required.empty()) return std::nullopt;
    if (f.clearing)
        return Note{Cause::PassageClearing,
                    std::format("door passage {}: through the doorway on the anchor until the servo reads `through`",
                                f.door),
                    site_of(loc)};
    if (f.state == "offered" or f.state == "executing") return std::nullopt;
    std::string d = std::format("door passage step {}: ", f.step);
    if (f.state == "completed")
        d += std::format("'{}' completed {} (attempt {}) — waiting for {} to offer it again", f.required,
                         f.last_outcome.empty() ? std::string("without success") : f.last_outcome, f.attempts,
                         f.producer);
    else
        d += mem.describe_absent(f.required, f.producer, now_ms);
    if (not f.door_now.empty()) d += std::format("; {} now: {}", f.door, f.door_now);
    // The two rules that keep it waiting (DoorPassage::update): no fallback once at the door, no back-to-open once opened.
    if (f.reached_door or f.opened)
    {
        std::string why;
        if (f.reached_door) why = "reached door";
        if (f.opened) why += std::string(why.empty() ? "" : ", ") + "open already done";
        d += std::format("; passage will not fall back ({})", why);
    }
    else
        d += "; passage falls back once an earlier step is on offer (none is)";
    return Note{Cause::PassageWaiting, std::move(d), site_of(loc)};
}

}   // namespace rc::why
