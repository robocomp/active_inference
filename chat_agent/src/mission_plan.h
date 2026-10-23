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
 */

#include <algorithm>
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
    return std::nullopt;
}

inline std::optional<std::string> unknown_in(const std::vector<Step>& list,
                                             const std::function<bool(std::string_view)>& exists)
{
    for (const auto& s : list)
    {
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

/// The first Node or Passage name the mission uses that `exists` refuses (op args and predicate refs, recursively).
/// Var, Anchor and Pose refs are not names in the graph and are skipped.
[[nodiscard]] inline std::optional<std::string> unknown_name(const rc::mission::Mission& m,
                                                             const std::function<bool(std::string_view)>& exists)
{
    return mission_plan_detail::unknown_in(m.steps, exists);
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
            const int n = step ? step->on_fail.retries + 1 : v.attempt;
            out.push_back(std::format("⟳ retry {}/{} {}", v.attempt, n, label));
        }
        else if (v.state == SS::Running and was != SS::Running)
            out.push_back(std::format("→ {}/{} {}", top_index(v.id), total, label));
        if (v.state == SS::Satisfied and was != SS::Satisfied)
            out.push_back(std::format("✓ {} ({:.0f} s)", label, double(v.t1 - v.t0) / 1000.0));
        else if (v.state == SS::Failed and was != SS::Failed)
            out.push_back(std::format("✗ {}: {}", label, v.outcome));
    }

    const std::string was_paused = paused_at(before), now_paused = paused_at(after);
    if (was_paused.empty() and not now_paused.empty())
        out.push_back(std::format("⏸ paused at {}/{} {}", top_index(now_paused), total, name_of(m, now_paused)));
    else if (not was_paused.empty() and now_paused.empty() and not after.status.starts_with("aborted"))
        out.push_back(std::format("▶ resumed: {}/{} {} (restarted)", top_index(was_paused), total, name_of(m, was_paused)));
    if (after.status.starts_with("aborted") and not before.status.starts_with("aborted"))
        out.push_back("■ mission cancelled");
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

}   // namespace rc::chat
