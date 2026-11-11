#pragma once
/*
 * mission_plan.h — the model writes a mission; these are the pure helpers around it (2026-10-23). Tested in
 * tests/test_mission_plan.cpp.
 *
 *   parse_planned   text or JSON → a validated Mission, or ONE line the planner can act on.
 *   unknown_name    the first node / passage the mission names that the graph does not have — checked before sending,
 *                   so a typo comes back to the model instead of failing on the robot.
 *   progress_lines  the fixed one-line messages for every step-state change between two snapshots.
 *   new_reports     the text of each `report` step that has just run, for the reply.
 *   planner_request   the planner's user message: the person's VERBATIM words, then the chat's request if it differs.
 *   strip_fence / planned_ask / check_planned   the planner reply → `ask:` question, or a card-ready mission.
 *   card_still_valid  the yes to a mission card must be the person's very next message.
 *   plan_still_wanted a planner reply is discarded once the person has spoken again.
 *   WakeQueue / *_wake  a robot event (report, failure, completion) for the model, held until it is idle.
 *   fix_still_open    how long a failure's wake may still stamp a planned mission as its fix.
 *   allowed_in_wake   the tools a robot event's turn may offer and run (look, stop, propose — never act).
 *   mission_refused_note  a mission the controller refused before its first step: the only line the person gets.
 */

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <deque>
#include <expected>
#include <format>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "../../common/mission/mission_ir.h"
#include "../../common/mission/mission_json.h"
#include "../../common/mission/mission_progress.h"
#include "../../common/mission/mission_text.h"
#include "command_gate.h"

namespace rc::chat
{

/// `{…}` is the JSON form, anything else the DSL; either way the mission is validated. The error is one line.
[[nodiscard]] inline std::expected<rc::mission::Mission, std::string> parse_planned(std::string_view text)
{
    const auto first = text.find_first_not_of(" \t\r\n");
    const bool json = first != std::string_view::npos and text[first] == '{';
    auto m = json ? rc::mission::decode_json(text) : rc::mission::parse_text(text);
    if (not m.has_value()) return std::unexpected(m.error().message());
    if (const auto bad = rc::mission::validate(*m); bad.has_value()) return std::unexpected(bad->message());
    return *std::move(m);
}

namespace mission_plan_detail
{
using rc::mission::Mission;
using rc::mission::Pred;
using rc::mission::Ref;
using rc::mission::Step;

inline const Step* find_step(const std::vector<Step>& list, std::string_view id)
{
    for (const auto& s : list)
    {
        if (s.id == id) return &s;
        if (const Step* c = find_step(s.steps, id)) return c;
    }
    return nullptr;
}

/// 1-based position of the top-level step that holds step `id` ("2.1" → 2).
inline int top_index(std::string_view id)
{
    int n = 0;
    for (const char c : id.substr(0, id.find('.')))
    {
        if (c < '0' or c > '9') return 0;
        n = n * 10 + (c - '0');
    }
    return n;
}

/// The line's name for a step: a `say` shows its text verbatim, anything else its label ("1:cross door_1").
inline std::string name_of(const Step& s)
{
    if (s.op == "say")
        if (const auto t = s.string("text")) return *t;
    return rc::mission::Interpreter::label(s);
}

inline std::string name_of(const Mission& m, std::string_view id)
{
    const Step* s = find_step(m.steps, id);
    return s ? name_of(*s) : std::string(id);
}

inline std::optional<std::string> unknown_in_ref(const Ref& r, const std::function<bool(std::string_view)>& exists)
{
    if ((r.kind == Ref::Kind::Node or r.kind == Ref::Kind::Passage) and not exists(r.name)) return r.name;
    // a passage's face is named by the room it shows in (passage:NAME@ROOM): that room is a name too
    if (r.kind == Ref::Kind::Passage and not r.from.empty() and not exists(r.from)) return r.from;
    return std::nullopt;
}

inline std::optional<std::string> unknown_in(const std::vector<Step>& list,
                                             const std::function<bool(std::string_view)>& exists)
{
    for (const auto& s : list)
    {
        // `expect=ROOM` is a String field, not a Ref, but it names a room all the same ($NAME is a bound room: skipped)
        if (const auto e = s.string("expect"); e.has_value() and not e->starts_with('$') and not exists(*e)) return *e;
        for (const auto& [key, v] : s.args)
        {
            if (const Ref* r = std::get_if<Ref>(&v))
            {
                if (auto u = unknown_in_ref(*r, exists)) return u;
            }
            else if (const Pred* p = std::get_if<Pred>(&v))
                for (const auto& [pk, pr] : p->refs)
                    if (auto u = unknown_in_ref(pr, exists)) return u;
        }
        if (auto u = unknown_in(s.steps, exists)) return u;
    }
    return std::nullopt;
}

}   // namespace mission_plan_detail

/// An object only the long-term memory has: its last pose, and the steps that bring it into view from here (built in
/// code by remembered_goto.h view_steps: turn to it and wait in the robot's own room, else go_to its pose).
struct RememberedPlace
{
    rc::mission::Pose pose;
    std::vector<std::string> view_steps;
};
/// nullopt for anything that is not an object only the long-term memory has.
using RememberedLookup = std::function<std::optional<RememberedPlace>(std::string_view)>;

namespace mission_plan_detail
{
/// What the run has done so far that can bring a remembered object into view: go_to poses (by room), a turn, a wait.
struct ViewSoFar
{
    std::vector<std::string> go_to_rooms;
    bool turned = false;   ///< a `move` with turn_deg
    bool waited = false;   ///< a `wait` AFTER that turn (the tracker needs frames of it to bring it back)
};

/// The first `approach` of a REMEMBERED object (one the live graph does not have: it offers no approach until it is seen
/// again) not reached the way the missions skill says: brought into view earlier in the run order — a `go_to pose:` in
/// its room, or a turn then a wait — and on_fail=skip so a chair that has gone does not abort the mission. The error
/// names the exact lines to write. `seen` accumulates in run order (a body runs where it stands).
inline std::optional<std::string> remembered_unreached(const std::vector<Step>& list, const RememberedLookup& remembered,
                                                       ViewSoFar& seen)
{
    for (const auto& s : list)
    {
        if (s.op == "go_to")
            if (const auto r = s.ref("to"); r.has_value() and r->kind == Ref::Kind::Pose) seen.go_to_rooms.push_back(r->pose.room);
        if (s.op == "move" and s.has("turn_deg")) { seen.turned = true; seen.waited = false; }
        if (s.op == "wait" and seen.turned) seen.waited = true;
        if (s.op == "approach")
            if (const auto r = s.ref("object"); r.has_value() and r->kind == Ref::Kind::Node)
                if (const auto p = remembered(r->name); p.has_value())
                {
                    const bool in_view = std::ranges::contains(seen.go_to_rooms, p->pose.room) or seen.waited;
                    if (not in_view or s.on_fail.kind != rc::mission::OnFail::Kind::Skip)
                    {
                        std::string lines;
                        for (const auto& v : p->view_steps) lines += (lines.empty() ? "`" : "`, then `") + v;
                        return std::format("'{0}' is remembered, NOT perceived now, so it offers no approach until it is "
                                           "seen again: put {1}` before it and write `approach {0} on_fail=skip`",
                                           r->name, lines.empty() ? std::string("`go_to its pose") : lines);
                    }
                }
        if (auto e = remembered_unreached(s.steps, remembered, seen)) return e;
    }
    return std::nullopt;
}

inline const rc::mission::Interpreter::StepView* view_of(const rc::mission::Progress& p, std::string_view id)
{
    const auto it = std::ranges::find(p.steps, id, &rc::mission::Interpreter::StepView::id);
    return it == p.steps.end() ? nullptr : &*it;
}

/// "paused|2" → "2"; empty when the mission is not paused.
inline std::string paused_at(const rc::mission::Progress& p)
{
    return p.status.starts_with("paused|") ? p.status.substr(7) : std::string{};
}
}   // namespace mission_plan_detail

/// The first Node or Passage name the mission uses that `exists` refuses (op args, predicate refs, a passage's
/// `@ROOM` and `expect=ROOM`, recursively). Var, Anchor and Pose refs are not names in the graph and are skipped.
[[nodiscard]] inline std::optional<std::string> unknown_name(const rc::mission::Mission& m,
                                                             const std::function<bool(std::string_view)>& exists)
{
    return mission_plan_detail::unknown_in(m.steps, exists);
}

/// The planner's reply without its code fence: models copy the skill's ``` examples (qwen3.8 fenced ~50% of its
/// replies, eval 2026-10-23). The inside of the FIRST fenced block (any language tag; to the end if it is never
/// closed), else the whole reply; trimmed either way. Same rule as tests/check_llm_missions.py's unfence(), so the
/// eval measures what runs: text outside a fence is the model chatting, never part of the mission.
[[nodiscard]] inline std::string_view strip_fence(std::string_view s)
{
    const auto trim = [](std::string_view t)
    {
        const auto a = t.find_first_not_of(" \t\r\n");
        if (a == std::string_view::npos) return std::string_view{};
        return t.substr(a, t.find_last_not_of(" \t\r\n") - a + 1);
    };
    const auto open = s.find("```");
    if (open == std::string_view::npos) return trim(s);
    std::string_view rest = s.substr(open + 3);
    const auto tag = rest.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_+-");
    if (tag == std::string_view::npos) return {};          // "```json" and nothing else
    if (rest[tag] == '\n') rest.remove_prefix(tag + 1);    // the fence line, with its language tag
    return trim(rest.substr(0, rest.find("```")));
}

/// The planner's user message. ★ The person's VERBATIM words lead (live 2026-10-24: the chat summarised "…, tell me
/// what objects are there, then cross back" without the report, and the mission had no `report` step); the chat's
/// plan_mission request follows only when it says something else, labelled neutrally: it may be a lossy summary of
/// the words, or the request itself when the words are the person's answer to the planner's `ask:` ("door_1"). A plan asked for in a robot event's turn (a failure's fix) has no words of the person behind it: the
/// last ones were about the old mission, so the request alone goes. tests/check_llm_missions.py sends the same layout.
[[nodiscard]] inline std::string planner_request(std::string_view said, std::string_view request, bool from_wake)
{
    const auto trim = [](std::string_view t)
    {
        const auto a = t.find_first_not_of(" \t\r\n");
        return a == std::string_view::npos ? std::string_view{} : t.substr(a, t.find_last_not_of(" \t\r\n") - a + 1);
    };
    if (from_wake or trim(said).empty()) return std::string(trim(request));
    // ★ The words go BARE, as they always went: wrapped as `The person said: "…"` gemma4:26b dropped an explore and
    //   invented a room name (mission eval 2026-10-24: 66/72 wrapped, 72/72 bare).
    std::string out(trim(said));
    if (not trim(request).empty() and ascii_lower_trim(std::string(said)) != ascii_lower_trim(std::string(request)))
        out += std::format("\n\n(the chat's request: \"{}\")", trim(request));
    return out;
}

/// `ask: <question>` (any case, fenced or not) → the question; nullopt for anything else.
[[nodiscard]] inline std::optional<std::string> planned_ask(std::string_view reply)
{
    const std::string_view s = strip_fence(reply);
    if (s.size() < 4) return std::nullopt;
    std::string head(s.substr(0, 4));
    std::ranges::transform(head, head.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    if (head != "ask:") return std::nullopt;
    const std::string_view q = s.substr(4);
    const auto a = q.find_first_not_of(" \t");
    return a == std::string_view::npos ? std::string{} : std::string(q.substr(a));
}

/// A planner reply → the mission language_agent may put on a card: fence stripped, parsed, validated, every name known,
/// and every remembered object reached the missions skill's way (`remembered`, when given: see remembered_unreached).
/// The error is ONE line the planner can act on (it is sent back to it to fix). Live 2026-11-10: the planner wrote a
/// bare `approach chair_34` for a remembered chair; the controller refused it before anything moved.
[[nodiscard]] inline std::expected<rc::mission::Mission, std::string> check_planned(
    std::string_view reply, const std::function<bool(std::string_view)>& exists, const RememberedLookup& remembered = {})
{
    auto m = parse_planned(strip_fence(reply));
    if (not m.has_value()) return m;
    if (const auto bad = unknown_name(*m, exists); bad.has_value())
        return std::unexpected(std::format("unknown name '{}': there is no such door, room or object in the live map now "
                                           "(a room not yet reached is named with cross ... bind=NAME, then $NAME)", *bad));
    if (remembered)
    {
        mission_plan_detail::ViewSoFar seen;
        if (auto e = mission_plan_detail::remembered_unreached(m->steps, remembered, seen); e.has_value())
            return std::unexpected(std::move(*e));
    }
    return m;
}

/// A mission card is answered by the person's NEXT message only: anything typed in between cancels it.
[[nodiscard]] constexpr bool card_still_valid(std::uint64_t card_turn, std::uint64_t user_turn)
{
    return user_turn == card_turn + 1;
}

/// A planner reply is used only while the person has not typed since plan_mission was called (`job_turn` = user_turn_
/// then): a message in between resolves the job as REJECTED, so a late reply never becomes a card.
[[nodiscard]] constexpr bool plan_still_wanted(std::uint64_t job_turn, std::uint64_t user_turn)
{
    return user_turn == job_turn;
}

/// One fixed line per step-state change between two snapshots, oldest first (the order of the steps).
[[nodiscard]] inline std::vector<std::string> progress_lines(const rc::mission::Progress& before,
                                                             const rc::mission::Progress& after,
                                                             const rc::mission::Mission& m)
{
    using namespace mission_plan_detail;
    using SS = rc::mission::Interpreter::StepState;
    const int total = int(m.steps.size());
    std::vector<std::string> out;

    for (const auto& v : after.steps)
    {
        const auto* old = view_of(before, v.id);
        const SS was = old ? old->state : SS::Pending;
        const Step* step = find_step(m.steps, v.id);
        const std::string label = step ? name_of(*step) : v.id;
        if (v.state == SS::Running and was == SS::Running and old and v.attempt > old->attempt)
        {
            const int max = step ? step->on_fail.retries : v.attempt - 1;
            out.push_back(std::format("⟳ retry {}/{} {}", v.attempt - 1, max, label));
        }
        else if (v.state == SS::Running and was != SS::Running)
            out.push_back(std::format("→ {}/{} {}", top_index(v.id), total, label));
        if (v.state == SS::Satisfied and was != SS::Satisfied)
            out.push_back(std::format("✓ {} ({:.0f} s)", label, v.t1 < v.t0 ? 0.0 : double(v.t1 - v.t0) / 1000.0));
        else if (v.state == SS::Failed and was != SS::Failed)
            out.push_back(std::format("✗ {}: {}", label, v.outcome));
        else if (v.state == SS::Skipped and was != SS::Skipped)   // on_fail=skip: it did not happen, and the person is told
            out.push_back(std::format("↷ skipped: {} ({})", label, v.outcome.empty() ? std::string("failed") : v.outcome));
    }

    const std::string was_paused = paused_at(before), now_paused = paused_at(after);
    if (was_paused.empty() and not now_paused.empty())
        out.push_back(std::format("⏸ paused at {}/{} {}", top_index(now_paused), total, name_of(m, now_paused)));
    else if (not was_paused.empty() and now_paused.empty() and after.status == "in_progress")
        out.push_back(std::format("▶ resumed: {}/{} {} (restarted)", top_index(was_paused), total, name_of(m, was_paused)));
    if (after.status.starts_with("aborted") and not before.status.starts_with("aborted"))
        out.push_back(after.status == "aborted|cancelled" ? std::string("■ mission cancelled")
                                                          : "■ mission aborted: " + after.status.substr(std::min<size_t>(8, after.status.size())));
    return out;
}

/// The `text` of each `report` step that became Running or Satisfied between the snapshots (an instant step is
/// usually seen once, as Satisfied; one seen Running then Satisfied is reported once).
[[nodiscard]] inline std::vector<std::string> new_reports(const rc::mission::Progress& before,
                                                          const rc::mission::Progress& after,
                                                          const rc::mission::Mission& m)
{
    using namespace mission_plan_detail;
    using SS = rc::mission::Interpreter::StepState;
    std::vector<std::string> out;
    for (const auto& v : after.steps)
    {
        if (v.op != "report" or (v.state != SS::Running and v.state != SS::Satisfied)) continue;
        const auto* old = view_of(before, v.id);
        if (old and (old->state == SS::Running or old->state == SS::Satisfied)) continue;
        if (const Step* s = find_step(m.steps, v.id))
            if (const auto t = s->string("text")) out.push_back(*t);
    }
    return out;
}

// ── pause / continue / cancel: the person's fast path, only where it means something (spec §5.6) ───────────────

struct MissionControl
{
    std::string word;    ///< the controller word to send; "" = nothing to send, only the reply
    std::string reply;   ///< the fixed text the chat answers with (no model)
};

/// The op of the innermost Running step ("" when none): what "continue" would act on.
[[nodiscard]] inline std::string running_op(const rc::mission::Progress& p)
{
    std::string op;
    for (const auto& v : p.steps)   // pre-order: the last Running one is the leaf
        if (v.state == rc::mission::Interpreter::StepState::Running) op = v.op;
    return op;
}

/// The person's words + the mission's status (+ its running step's op) → what to do, or nullopt = not about the
/// mission: the message goes on to the normal path. "sigue recto" or "espera, ¿qué ves?" with nothing to pause or
/// unpause is conversation, and the person's working language is Spanish.
///   pause     → `pause` only while in_progress
///   continue  → `unpause` while paused|…; `resume` only while in_progress AND the running step is a `wait`
///   cancel    → `cancel` while in_progress or paused; with nothing running, the fixed "no mission" reply (the
///               phrases are unambiguous)
[[nodiscard]] inline std::optional<MissionControl> mission_control_action(std::string_view text, std::string_view status,
                                                                          std::string_view running)
{
    const std::string t(text);
    const bool paused = status.starts_with("paused|");
    const bool in_progress = status == "in_progress";
    if (is_pause(t) and in_progress) return MissionControl{"pause", "Pausing the mission."};
    if (is_continue(t))
    {
        if (paused) return MissionControl{"unpause", "Continuing the mission: the interrupted step starts again."};
        if (in_progress and running == "wait") return MissionControl{"resume", "Continuing: the mission stops waiting."};
        return std::nullopt;
    }
    if (is_cancel_mission(t))
        return in_progress or paused ? MissionControl{"cancel", "Cancelling the mission. The robot stays where it is."}
                                     : MissionControl{"", "There is no mission to cancel."};
    return std::nullopt;
}

// ── wakes: the robot, not the person, has something to say (spec §5.1 steps 7-8, §5.5) ─────────────────────────

/// Robot events for the model, oldest first. A note waits while the model (or the planner) is busy and is never
/// dropped: the next idle tick takes it. A failure's note carries the key of the mission it failed ("m12") so the
/// mission planned in answer can say what it fixes (Mission::fixes).
struct WakeQueue
{
    void push(std::string note, std::string fixes = {}) { q_.push_back({std::move(note), std::move(fixes)}); }
    [[nodiscard]] std::optional<std::string> pop_if_idle(bool busy)
    {
        if (busy or q_.empty()) return std::nullopt;
        auto [note, fixes] = std::move(q_.front());
        q_.pop_front();
        last_fixes_ = std::move(fixes);
        return note;
    }
    [[nodiscard]] const std::string& last_fixes() const { return last_fixes_; }   ///< of the last popped note
    [[nodiscard]] bool empty() const { return q_.empty(); }
    void clear() { q_.clear(); last_fixes_.clear(); }   ///< a new conversation: its history has no place for them

private:
    std::deque<std::pair<std::string, std::string>> q_;
    std::string last_fixes_;
};

/// A wake goes to the model as a user-role message that says it is NOT the person speaking.
[[nodiscard]] inline std::string wake_message(std::string_view note)
{
    return "(robot event, not the person) " + std::string(note);
}

/// The `report` step ran: its topic and the room's objects AT THAT INSTANT (the robot keeps moving while the model
/// writes, so the answer is about the snapshot, never about the STATE of the reply's time; spec §5.3).
/// `skipped`: the steps skipped BEFORE it (skipped_steps), named so the model does not say they happened (chat_e2e2:
/// the go_to was skipped for no_route and the model said "I have arrived at the location where I last saw chair_34").
[[nodiscard]] inline std::string report_wake(std::string_view topic, std::string_view room, std::string_view objects,
                                             const std::vector<std::string>& skipped = {})
{
    std::string before;
    for (const auto& k : skipped) before += (before.empty() ? "" : "; ") + k;
    return std::format("The mission reached its report step: \"{}\". When it ran the robot was in {}, and this is what "
                       "find_objects answered then:\n{}\n{}Tell the person about it now, from this snapshot only (the robot "
                       "may have moved on since: the STATE is no longer about that room). The mission goes on by itself.",
                       topic, room.empty() ? std::string_view("an unknown room") : room, objects,
                       before.empty() ? std::string{}
                                      : std::format("Before it, these steps were SKIPPED and did not happen: {}. Do not say "
                                                    "the robot reached a place or did anything those steps were for.\n",
                                                    before));
}

/// The top-level steps not done (neither satisfied nor skipped), by their line names, in mission order.
[[nodiscard]] inline std::vector<std::string> undone_steps(const rc::mission::Progress& p, const rc::mission::Mission& m)
{
    using namespace mission_plan_detail;
    using SS = rc::mission::Interpreter::StepState;
    std::vector<std::string> out;
    for (const auto& s : m.steps)
    {
        const auto* v = view_of(p, s.id);
        if (v and (v->state == SS::Satisfied or v->state == SS::Skipped)) continue;
        out.push_back(name_of(s));
    }
    return out;
}

/// `failed|<step>|<outcome>` → what failed, how, what is left, and what the model is to do about it; nullopt when the
/// mission has not failed.
[[nodiscard]] inline std::optional<std::string> failure_wake(const rc::mission::Progress& p, const rc::mission::Mission& m)
{
    using namespace mission_plan_detail;
    if (not p.status.starts_with("failed|")) return std::nullopt;
    const std::string rest = p.status.substr(7);
    const auto bar = rest.find('|');
    const std::string step = rest.substr(0, bar);
    const std::string outcome = bar == std::string::npos ? std::string("no reason given") : rest.substr(bar + 1);
    std::string undone;
    for (const auto& u : undone_steps(p, m)) undone += (undone.empty() ? "" : "; ") + u;
    return std::format("The mission \"{}\" failed at step {}: {}. Steps not done: {}. Tell the person what happened in "
                       "one or two sentences, then either propose a fix with plan_mission (a new mission that finishes "
                       "what is left and brings the robot back where the person wanted it; it only shows a card, and runs "
                       "only on their yes) or explain why there is none. No step is retried by itself.",
                       p.title, name_of(m, step), outcome, undone.empty() ? std::string("none") : undone);
}

/// Every step (any depth) the mission SKIPPED (on_fail=skip after a failure), by line name with its outcome, in order.
[[nodiscard]] inline std::vector<std::string> skipped_steps(const rc::mission::Progress& p, const rc::mission::Mission& m)
{
    using namespace mission_plan_detail;
    std::vector<std::string> out;
    for (const auto& v : p.steps)
        if (v.state == rc::mission::Interpreter::StepState::Skipped)
            out.push_back(std::format("{} ({})", name_of(m, v.id), v.outcome.empty() ? std::string("failed") : v.outcome));
    return out;
}

/// `completed` → one closing sentence. ★ A completed mission may have SKIPPED steps (on_fail=skip): they are named, so
/// the model never tells the person "every step done" about a step that did not happen (anti-goal 1).
[[nodiscard]] inline std::string summary_wake(const rc::mission::Progress& p, const rc::mission::Mission& m)
{
    const double s = p.t1 > p.t0 ? double(p.t1 - p.t0) / 1000.0 : 0.0;
    std::string skipped;
    for (const auto& k : skipped_steps(p, m)) skipped += (skipped.empty() ? "" : "; ") + k;
    if (skipped.empty())
        return std::format("The mission \"{}\" completed ({:.0f} s, every step done). Close it for the person in one short "
                           "sentence; do not repeat what you already reported.", p.title, s);
    return std::format("The mission \"{}\" completed ({:.0f} s) but SKIPPED these steps, which did not happen: {}. Close it "
                       "for the person in one or two short sentences, saying what was skipped; do not repeat what you "
                       "already reported.", p.title, s, skipped);
}

/// A failure's wake was delivered at user turn `fix_turn`: a card planned in that same turn, or in answer to the
/// person's next message (an ASK: their answer), is the fix; anything later is a new subject.
[[nodiscard]] constexpr bool fix_still_open(std::uint64_t fix_turn, std::uint64_t user_turn)
{
    return user_turn == fix_turn or user_turn == fix_turn + 1;
}

// ── a robot event's turn: the model may look, stop and propose, never act (spec §4 anti-goal 2, final review C1) ──

/// The tool result of an acting call the model makes while answering a robot event.
inline constexpr std::string_view kWakeRefusal =
    "REFUSED: during a robot event only plan_mission/stop/cancel_passage may act; propose a fix with plan_mission";

/// The tools a robot event's turn (a report, a failure, a completion, an agent's late launch result) may offer AND run.
/// Nobody asked for anything in that turn, so nothing may move the robot or change the fleet: an acting call there
/// (cross_door after a failed cross, approach_object mid-mission, agent_control start after a failed launch) would
/// run without the person's yes or supersede the mission they approved. What is left: the read-only tools, `stop` and
/// `cancel_passage` (they only ever halt), and `plan_mission`, which only shows a card that runs on the person's yes.
/// ★ An ALLOW-list: a tool added later is refused in a wake until someone decides it is harmless there.
[[nodiscard]] inline bool allowed_in_wake(std::string_view tool)
{
    static constexpr std::string_view kAllowed[] = {
        "stop", "cancel_passage", "plan_mission",                                                  // halt / propose
        "look", "find_objects", "where_is", "known_rooms", "route_to", "can_see", "imagine_view", "imagine", "agents", "sensors",        // read the world
        "graph_summary", "inspect", "use_skill", "episodes", "episode", "outcome_stats", "last_episode",
        "show", "draw", "table", "hide_canvas",                                                     // the side canvas
    };
    return std::ranges::find(kAllowed, tool) != std::ranges::end(kAllowed);
}

/// The controller's report on a sent mission (`{…}` JSON) → the line for the person when it REFUSED it before its first
/// step (chat_inbox K::Rejected: state `failed|<reason>`, bt step `-`), else nullopt. A mission that ran and then
/// failed carries its step in the bt and is told by its progress lines instead; a refused one never has any, so
/// without this line the person sees "Sent to the controller" and then nothing (final review I2).
[[nodiscard]] inline std::optional<std::string> mission_refused_note(std::string_view state, std::string_view bt_step,
                                                                     std::string_view title)
{
    if (not state.starts_with("failed") or bt_step != "-") return std::nullopt;
    const auto bar = state.find('|');
    const std::string why = bar == std::string_view::npos ? std::string("no reason given") : std::string(state.substr(bar + 1));
    return std::format("The controller REFUSED the mission \"{}\" before its first step: {}. Nothing ran and the robot did "
                       "not move. Tell the person in one sentence; a fix is a new plan_mission, only if they want one.",
                       title, why);
}

}   // namespace rc::chat
