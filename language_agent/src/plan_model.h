#pragma once
/*
 * plan_model.h — what the plan view (plan_view.h) paints, without painting it (2026-10-17).
 *
 * One row per step of the mission IR, in pre-order — the same order the controller's mission_progress lists them —
 * each carrying its own line of the text form (mission_text.h), so the view reads as the DSL with live state laid
 * over it. apply() folds in a progress snapshot by step id; annotation() / headline() / fraction() are the words and
 * the bar the window draws for a row at time `now`. Pure (no Qt), tested in tests/test_plan_model.cpp.
 *
 * Times are the controller's (ms since the epoch); `now` is the chat's wall clock. Both run on this machine (or on
 * NTP-synced ones), and an elapsed time shown to a person needs no better than that.
 */

#include <algorithm>
#include <cstdint>
#include <format>
#include <optional>
#include <string>
#include <vector>

#include "../../common/mission/mission_interpreter.h"
#include "../../common/mission/mission_progress.h"
#include "../../common/mission/mission_text.h"

namespace rc::chat::plan
{

using rc::mission::Interpreter;

struct Row
{
    std::string id, op, line;   ///< line: this step's text form, without its children
    int depth = 0;
    bool composite = false;
    std::optional<double> bound_s;   ///< a wait's `s` or a budget_s: what a progress bar measures against
    int attempts = 1;                ///< retry=N → N+1
    std::optional<long> times;       ///< repeat N

    Interpreter::StepState state = Interpreter::StepState::Pending;
    std::string outcome;
    int attempt = 0, iteration = 0;
    std::uint64_t t0 = 0, t1 = 0;
};

struct Model
{
    std::string title, source, status, line;
    std::uint64_t serial = 0, t0 = 0, t1 = 0;
    std::vector<Row> rows;

    static Model from(const rc::mission::Mission& m)
    {
        Model md;
        md.title = m.title;
        const auto walk = [&md](this const auto& self, const std::vector<rc::mission::Step>& steps, int depth) -> void
        {
            for (const auto& s : steps)
            {
                rc::mission::Step alone = s;
                alone.steps.clear();
                std::string text;
                rc::mission::text_detail::print_steps(text, {alone}, 0);
                if (text.ends_with('\n')) text.pop_back();
                Row r;
                r.id = s.id;
                r.op = s.op;
                r.line = std::move(text);
                r.depth = depth;
                r.composite = not s.steps.empty() or s.op == "seq" or s.op == "repeat";
                if (const auto secs = s.number("s"); s.op == "wait" and secs.has_value()) r.bound_s = *secs;
                else if (const auto b = s.number("budget_s"); b.has_value()) r.bound_s = *b;
                if (s.on_fail.kind == rc::mission::OnFail::Kind::Retry) r.attempts = s.on_fail.retries + 1;
                if (const auto n = s.integer("times"); n.has_value()) r.times = *n;
                md.rows.push_back(std::move(r));
                self(s.steps, depth + 1);
            }
        };
        walk(m.steps, 0);
        return md;
    }

    /// Fold a snapshot in. Returns the rows whose state changed, in order.
    std::vector<std::size_t> apply(const rc::mission::Progress& p)
    {
        serial = p.serial;
        source = p.source;
        status = p.status;
        line = p.line;
        t0 = p.t0;
        t1 = p.t1;
        std::vector<std::size_t> changed;
        for (const auto& v : p.steps)
        {
            const auto it = std::ranges::find(rows, v.id, &Row::id);
            if (it == rows.end()) continue;
            if (it->state != v.state) changed.push_back(std::size_t(it - rows.begin()));
            it->state = v.state;
            it->outcome = v.outcome;
            it->attempt = v.attempt;
            it->iteration = v.iteration;
            it->t0 = v.t0;
            it->t1 = v.t1;
        }
        std::ranges::sort(changed);
        return changed;
    }

    /// The deepest running row (the leaf that drives), or -1.
    [[nodiscard]] int running_row() const
    {
        for (int i = int(rows.size()) - 1; i >= 0; --i)
            if (rows[std::size_t(i)].state == Interpreter::StepState::Running) return i;
        return -1;
    }

    [[nodiscard]] static std::string clock(std::uint64_t ms)
    {
        const auto s = ms / 1000;
        return std::format("{}:{:02}", s / 60, s % 60);
    }

    [[nodiscard]] std::string headline(std::uint64_t now) const
    {
        if (status.empty()) return "not started";
        if (status == "in_progress") return "in progress · " + clock(now > t0 ? now - t0 : 0);
        if (status == "completed") return "completed in " + clock(t1 > t0 ? t1 - t0 : 0);
        if (status.starts_with("failed|"))
        {
            const auto rest = status.substr(7);
            const auto bar = rest.find('|');
            return bar == std::string::npos ? "failed: " + rest
                                            : std::format("failed at {}: {}", rest.substr(0, bar), rest.substr(bar + 1));
        }
        if (status.starts_with("aborted|")) return "aborted: " + status.substr(8);
        return status;
    }

    /// How far a running, bounded leaf is through its bound (0..1); nullopt if it is not one.
    [[nodiscard]] std::optional<double> fraction(const Row& r, std::uint64_t now) const
    {
        if (r.state != Interpreter::StepState::Running or r.composite or not r.bound_s.has_value() or *r.bound_s <= 0)
            return std::nullopt;
        return std::clamp(elapsed_s(r, now) / *r.bound_s, 0.0, 1.0);
    }

    [[nodiscard]] std::string annotation(const Row& r, std::uint64_t now) const
    {
        using SS = Interpreter::StepState;
        const bool special = not r.outcome.empty() and r.outcome != "satisfied";
        switch (r.state)
        {
            case SS::Pending: return "";
            case SS::Running:
            {
                if (r.op == "repeat")
                    return r.times.has_value() ? std::format("iteration {}/{}", r.iteration + 1, *r.times)
                                               : std::format("iteration {}", r.iteration + 1);
                if (r.composite) return "";
                std::string out = r.attempts > 1 ? std::format("attempt {}/{} · ", r.attempt, r.attempts) : "";
                out += r.bound_s.has_value() ? std::format("{:.1f} / {:g} s", elapsed_s(r, now), *r.bound_s)
                                             : std::format("{:.1f} s", elapsed_s(r, now));
                return out;
            }
            case SS::Satisfied:
                return std::format("{:.1f} s", double(r.t1 > r.t0 ? r.t1 - r.t0 : 0) / 1000.0) +
                       (special ? " · " + r.outcome : "");
            case SS::Failed:
                return r.attempt > 1 ? std::format("{} after {} attempts", r.outcome, r.attempt) : r.outcome;
            case SS::Skipped: return "skipped: " + r.outcome;
            case SS::Cancelled: return r.outcome.empty() ? "cancelled" : "cancelled: " + r.outcome;
        }
        return "";
    }

private:
    static double elapsed_s(const Row& r, std::uint64_t now) { return double(now > r.t0 ? now - r.t0 : 0) / 1000.0; }
};

}   // namespace rc::chat::plan
