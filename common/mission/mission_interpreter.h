#pragma once
/*
 * mission_interpreter.h — the step machine that runs a grounded mission (mission_ir.h), 2026-10-17.
 * Design: chat_agent/docs/mission_language_2026-10-13.md §3.1 (what each primitive compiles to), §3.5 (failure),
 * §3.6 (progress), §7.5 (which requests replace a running mission).
 *
 * The interpreter owns the CONTROL FLOW every mission shares — seq, repeat, on_fail, `until`, budgets, anchors,
 * `expect` — and nothing about how a leaf moves the robot. That is the Host's: the controller implements it with
 * what it already has (DoorPassage for cross/open, a point target for go_to, the free selector for explore,
 * RelativeMotion for move, a held base for wait). So the whole machine is checked here without a robot
 * (mission_interpreter_test.cpp), and the controller's adapter is a thin mapping.
 *
 *   start(mission)   validate, then replace whatever runs (it reports aborted|superseded)
 *   tick(now)        one control cycle: start / poll / cancel leaves, evaluate predicates, fold outcomes
 *   cancel(outcome)  `cancel_mission` / `stop`: cancels what runs, state aborted|<outcome>
 *   resume()         ends the current `wait` only (§7.5)
 *
 * ★ ONE LEAF RUNS AT A TIME. Every Phase-1 leaf that takes time claims the base (mission_ir.h Resources), so a
 *   sequential machine is exact; a future `parallel` is where claims_of() is checked.
 * ★ PREDICATES ARE READ EVERY CYCLE WHILE A STEP DEPENDS ON THEM (L7): an `until` is checked before its step starts
 *   (already true = nothing to do, nothing started) and on every tick while it runs; a repeat's `until` at every
 *   iteration boundary. An unknown belief (the host cannot evaluate it now) never satisfies anything.
 * ★ BUDGETS ARE THE TERMINATION GUARANTEE, not a tuning knob: they are the user's numbers, measured from the start
 *   of the current attempt. An explore whose budget runs out with no `until` has done what was asked ("explore for
 *   30 s"); with an `until` still unmet it failed: budget_exhausted.
 * ★ A BODY THAT NEVER WAITS DOES NOT SPIN: an iteration that began and ended within one tick yields to the next.
 * ★ THE HOST DECIDES WHAT "HOLD" MEANS on failure or abort; the interpreter only cancels the leaf it started.
 *
 * Pure: no graph, no Qt. Called from one thread (the controller's control loop).
 */

#include <algorithm>
#include <cstdint>
#include <format>
#include <map>
#include <optional>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

#include "mission_ir.h"
#include "mission_text.h"

namespace rc::mission
{

enum class LeafState { Running, Succeeded, Failed };

struct LeafStatus
{
    LeafState state = LeafState::Running;
    std::string detail{};   ///< Failed: the outcome word (unreachable, door_wont_open, …)
};

/// What the interpreter needs from the robot. Leaves are the non-instant ops (cross, open, go_to, explore, wait, move);
/// each arrives with its anchors already resolved to poses. At most one leaf is started and not yet finished.
class Host
{
public:
    virtual ~Host() = default;
    /// Start a leaf. A refusal (the reason, an outcome word or a sentence) fails the step at once.
    virtual std::optional<std::string> start(const Step& leaf) = 0;
    /// How the started leaf is going. A `wait` or an `explore` simply keeps Running; the interpreter ends them.
    virtual LeafStatus poll(const Step& leaf) = 0;
    /// Stop a started leaf that has not finished (its until held, its budget ran out, the mission was cancelled).
    virtual void cancel(const Step& leaf) = 0;
    virtual void say(const std::string& text) = 0;
    /// The `stop` step: end whatever drives and hold the base. The mission continues.
    virtual void halt() = 0;
    /// The robot's pose now, for `anchor`. nullopt = not known.
    virtual std::optional<Pose> pose() = 0;
    /// A predicate's belief now: a probability, or 0 / 1 for a fact. nullopt = cannot be evaluated now.
    virtual std::optional<double> eval(const Pred& p) = 0;
};

class Interpreter
{
public:
    enum class State { Idle, Running, Completed, Failed, Aborted };
    enum class StepState { Pending, Running, Satisfied, Failed, Skipped, Cancelled };

    struct StepView
    {
        std::string id, op;
        StepState state = StepState::Pending;
        std::string outcome;
        std::uint64_t t0 = 0, t1 = 0;   ///< first start · end (ms, the caller's clock)
        int attempt = 0;                ///< 1 = the first try
        int iteration = 0;              ///< repeat: iterations finished
    };

    /// One step attempt (or one repeat iteration) starting (state Running) or ending (its final state), for the
    /// episodic journal (common/episodic/em_journal.h). steps() keeps one entry per step id and so cannot show a
    /// retry or an iteration; this does. `iteration`: a repeat's own iteration (0-based), and for any other step the
    /// iteration of its innermost enclosing repeat (0 outside one), so (id, attempt, iteration) names one try.
    struct StepTransition
    {
        std::string id, op;
        int attempt = 0, iteration = 0;
        StepState state = StepState::Running;
        std::string outcome;   ///< the end's outcome word; "" for Running
        std::uint64_t t = 0;   ///< ms, the caller's clock
    };

    explicit Interpreter(Host& host) : host_(host) {}
    Interpreter(const Interpreter&) = delete;
    Interpreter& operator=(const Interpreter&) = delete;

    /// Validate and start. A refused mission leaves the running one untouched; an accepted one supersedes it.
    std::optional<Error> start(Mission m, std::uint64_t now_ms)
    {
        if (auto e = validate(m); e.has_value()) return e;
        if (state_ == State::Running) cancel("superseded", now_ms);
        ++mission_;
        title_ = std::move(m.title);
        anchors_ = std::move(m.anchors);
        captured_.clear();
        root_step_ = Step{.id = "", .op = "seq", .steps = std::move(m.steps)};
        root_ = build(root_step_);
        state_ = State::Running;
        abort_outcome_.clear();
        tick_ = 0;
        event(std::format("mission '{}': started ({} steps)", title_, root_step_.steps.size()));
        return std::nullopt;
    }

    void tick(std::uint64_t now_ms)
    {
        if (state_ != State::Running) return;
        ++tick_;
        advance(root_, now_ms);
        if (root_.state == StepState::Satisfied)
        {
            state_ = State::Completed;
            event(std::format("mission '{}': completed", title_));
        }
        else if (root_.state == StepState::Failed)
        {
            state_ = State::Failed;
            event(std::format("mission '{}': failed at step {}: {}", title_, root_.origin, root_.outcome));
        }
    }

    void cancel(const std::string& outcome, std::uint64_t now_ms)
    {
        if (state_ != State::Running) return;
        cancel_subtree(root_, now_ms, outcome);
        state_ = State::Aborted;
        abort_outcome_ = outcome;
        event(std::format("mission '{}': aborted ({})", title_, outcome));
    }

    /// End the running `wait`, if that is what runs. True if one was ended.
    bool resume(std::uint64_t now_ms)
    {
        if (state_ != State::Running) return false;
        Run* leaf = running_leaf(root_);
        if (leaf == nullptr or leaf->step->op != "wait" or not leaf->started) return false;
        finish_leaf(*leaf, now_ms, "resumed");
        return true;
    }

    [[nodiscard]] State state() const { return state_; }
    [[nodiscard]] const std::string& title() const { return title_; }
    /// Which start() this is: 1, 2, … (0 = none yet). The mission serial everything the controller publishes carries.
    [[nodiscard]] std::uint64_t serial() const { return mission_; }
    /// The mission's first tick · its end (0 while it runs).
    [[nodiscard]] std::pair<std::uint64_t, std::uint64_t> span() const { return {root_.t0, root_.t1}; }

    /// The edge's `state`: in_progress · completed · failed|<step>|<outcome> · aborted|<outcome> (chat_inbox.h).
    [[nodiscard]] std::string status() const
    {
        switch (state_)
        {
            case State::Idle: return "";
            case State::Running: return "in_progress";
            case State::Completed: return "completed";
            case State::Failed: return std::format("failed|{}|{}", root_.origin, root_.outcome);
            case State::Aborted: return "aborted|" + abort_outcome_;
        }
        return "";
    }

    /// The step part of the edge's `bt_state`: "<id>:<op> <args>|running (attempt a/n, iteration i/N)|<last event>".
    [[nodiscard]] std::string progress() const
    {
        if (state_ != State::Running) return std::format("-|{}|{}", status(), last_event_);
        std::vector<const Run*> path;
        for (const Run* r = &root_; r != nullptr; r = running_child(*r)) path.push_back(r);
        const Run& leaf = *path.back();
        std::vector<std::string> notes;
        if (leaf.step->on_fail.kind == OnFail::Kind::Retry)
            notes.push_back(std::format("attempt {}/{}", leaf.attempt, leaf.step->on_fail.retries + 1));
        for (const Run* r : path | std::views::reverse)
            if (r->step->op == "repeat")
            {
                const auto times = r->step->integer("times");
                notes.push_back(times.has_value() ? std::format("iteration {}/{}", r->iteration + 1, *times)
                                                  : std::format("iteration {}", r->iteration + 1));
            }
        std::string joined;
        for (const auto& n : notes) joined += (joined.empty() ? " (" : ", ") + n;
        if (not joined.empty()) joined += ")";
        return std::format("{}|running{}|{}", label(*leaf.step), joined, last_event_);
    }

    /// Every step in pre-order (mission_progress).
    [[nodiscard]] std::vector<StepView> steps() const
    {
        std::vector<StepView> out;
        const auto walk = [&out](this const auto& self, const Run& r) -> void
        {
            for (const auto& k : r.kids)
            {
                out.push_back({.id = k.step->id, .op = k.step->op, .state = k.state, .outcome = k.outcome, .t0 = k.t0,
                               .t1 = k.t1, .attempt = k.attempt, .iteration = k.iteration});
                self(k);
            }
        };
        if (state_ != State::Idle) walk(root_);
        return out;
    }

    /// Transitions since the last call, for the log. The newest is also in progress().
    [[nodiscard]] std::vector<std::string> take_events() { return std::exchange(events_, {}); }
    /// Step transitions since the last call, oldest first. A superseding start() leaves the old mission's closing
    /// transitions here: drain before start() to keep them apart from the new mission's.
    [[nodiscard]] std::vector<StepTransition> take_transitions() { return std::exchange(transitions_, {}); }

    /// "1:cross door_27", "2:explore room_2": the id, the op and its positional arguments.
    [[nodiscard]] static std::string label(const Step& s)
    {
        std::string out = s.id + ":" + s.op;
        if (const OpSpec* spec = op_spec(s.op))
            for (const auto& f : spec->fields)
                if (const auto it = s.args.find(f.name); f.positional and it != s.args.end())
                    out += " " + text_detail::value_text(it->second, f.name);
        return out;
    }

private:
    struct Run
    {
        const Step* step = nullptr;
        StepState state = StepState::Pending;
        std::string outcome{}, origin{};   ///< origin: the id of the step whose failure this is
        std::uint64_t t0 = 0, t1 = 0, began = 0;
        int attempt = 0, iteration = 0;
        int outer = 0;          ///< the innermost enclosing repeat's iteration (set by the parent each tick)
        int open_iteration = -1;   ///< the iteration of the transition opened and not yet closed; -1 = none
        std::size_t cur = 0;
        std::uint64_t iteration_tick = 0;
        bool started = false;   ///< leaf: the host has it
        Step resolved{};        ///< leaf: what the host was given
        std::vector<Run> kids{};
    };

    Host& host_;
    State state_ = State::Idle;
    std::string title_, abort_outcome_, last_event_;
    std::map<std::string, Pose, std::less<>> anchors_, captured_;
    Step root_step_{};
    Run root_{};
    std::uint64_t tick_ = 0;
    std::vector<std::string> events_;
    std::vector<StepTransition> transitions_;
    std::uint64_t mission_ = 0;

    static Run build(const Step& s)
    {
        Run r{.step = &s};
        for (const auto& k : s.steps) r.kids.push_back(build(k));
        return r;
    }

    void event(std::string e)
    {
        last_event_ = e;
        events_.push_back(std::move(e));
    }

    /// An attempt (or, for a repeat, an iteration) begins: Running. The root is the mission, not a step.
    void open(Run& r, std::uint64_t now)
    {
        if (r.step == &root_step_) return;
        r.open_iteration = r.step->op == "repeat" ? r.iteration : r.outer;
        transitions_.push_back({.id = r.step->id, .op = r.step->op, .attempt = r.attempt, .iteration = r.open_iteration,
                                .state = StepState::Running, .outcome = {}, .t = now});
    }
    /// What open() began ends, in the step's state now. Once: a step already closed (a repeat's last iteration,
    /// closed at its boundary) emits nothing more.
    void close(Run& r, std::uint64_t now)
    {
        if (r.step == &root_step_ or r.open_iteration < 0) return;
        transitions_.push_back({.id = r.step->id, .op = r.step->op, .attempt = r.attempt, .iteration = r.open_iteration,
                                .state = r.state, .outcome = r.outcome, .t = now});
        r.open_iteration = -1;
    }

    static bool finished(StepState s) { return s != StepState::Pending and s != StepState::Running; }

    static void reset(Run& r)
    {
        r.state = StepState::Pending;
        r.outcome.clear();
        r.origin.clear();
        r.t0 = r.t1 = r.began = 0;
        r.attempt = r.iteration = 0;
        r.cur = 0;
        r.started = false;
        for (auto& k : r.kids) reset(k);
    }

    bool holds(const Pred& p)
    {
        const auto v = host_.eval(p);
        if (not v.has_value()) return false;
        if (not p.cmp.has_value()) return *v != 0.0;
        return p.cmp->op == Comparison::Op::Ge ? *v >= p.cmp->value : *v < p.cmp->value;
    }

    void satisfy(Run& r, std::uint64_t now, std::string why = {})
    {
        r.state = StepState::Satisfied;
        r.outcome = why.empty() ? "satisfied" : std::move(why);
        r.t1 = now;
        close(r, now);
        if (r.step != &root_step_) event(std::format("step {}: {}", label(*r.step), r.outcome));
    }

    void fail(Run& r, std::uint64_t now, std::string outcome, std::string origin)
    {
        r.state = StepState::Failed;
        r.outcome = std::move(outcome);
        r.origin = std::move(origin);
        r.t1 = now;
        r.started = false;
        if (r.step != &root_step_) event(std::format("step {}: failed ({})", label(*r.step), r.outcome));
    }

    void advance(Run& r, std::uint64_t now)
    {
        if (finished(r.state)) return;
        if (r.state == StepState::Pending)
        {
            ++r.attempt;
            if (r.attempt == 1) r.t0 = now;
            r.began = now;
            r.state = StepState::Running;
            open(r, now);
        }
        if (r.kids.empty()) run_leaf(r, now);
        else run_composite(r, now);
        if (r.state == StepState::Failed) on_fail(r, now);
    }

    void on_fail(Run& r, std::uint64_t now)
    {
        const OnFail& f = r.step->on_fail;
        if (f.kind == OnFail::Kind::Retry and r.attempt <= f.retries)
        {
            event(std::format("step {}: retry {}/{} after {}", label(*r.step), r.attempt + 1, f.retries + 1, r.outcome));
            close(r, now);   // this attempt ends Failed; the next one opens in advance()
            const int attempt = r.attempt;
            const std::uint64_t t0 = r.t0;
            reset(r);
            r.attempt = attempt;
            r.t0 = t0;
            advance(r, now);   // bounded: each pass spends an attempt
            return;
        }
        if (f.kind == OnFail::Kind::Skip)
        {
            r.state = StepState::Skipped;
            event(std::format("step {}: skipped after {}", label(*r.step), r.outcome));
        }
        close(r, now);
    }

    void run_composite(Run& r, std::uint64_t now)
    {
        const Step& s = *r.step;
        const bool is_repeat = s.op == "repeat";
        const auto until = s.pred("until");
        if (const auto budget = s.number("budget_s"); budget.has_value() and double(now - r.began) >= *budget * 1000.0)
        {
            cancel_subtree_kids(r, now, "cancelled");
            return fail(r, now, "budget_exhausted", s.id);
        }
        if (is_repeat and r.cur == 0 and r.kids.front().state == StepState::Pending and until.has_value() and holds(*until))
            return satisfy(r, now, "until");
        while (true)
        {
            if (r.cur == r.kids.size())
            {
                if (not is_repeat) return satisfy(r, now);
                ++r.iteration;
                if (const auto times = s.integer("times"); times.has_value() and r.iteration >= *times) return satisfy(r, now);
                for (auto& k : r.kids) reset(k);
                r.cur = 0;
                if (until.has_value() and holds(*until)) return satisfy(r, now, "until");
                // The iteration that just finished ends here; the next one opens when its first step starts.
                r.state = StepState::Satisfied;
                r.outcome = "satisfied";
                close(r, now);
                r.state = StepState::Running;
                r.outcome.clear();
                if (r.iteration_tick == tick_) return;   // began this tick: the next one starts next tick
            }
            if (r.cur == 0 and r.kids.front().state == StepState::Pending)
            {
                r.iteration_tick = tick_;
                if (r.open_iteration < 0) open(r, now);   // a repeat's next iteration (the first opened in advance())
            }
            Run& k = r.kids[r.cur];
            k.outer = is_repeat ? r.iteration : r.outer;
            advance(k, now);
            if (k.state == StepState::Satisfied or k.state == StepState::Skipped)
            {
                ++r.cur;
                continue;
            }
            if (k.state == StepState::Failed) return fail(r, now, k.outcome, k.origin);
            return;   // the child is running
        }
    }

    void run_leaf(Run& r, std::uint64_t now)
    {
        const Step& s = *r.step;
        if (s.op == "say")
        {
            host_.say(*s.string("text"));
            return satisfy(r, now);
        }
        if (s.op == "stop")
        {
            host_.halt();
            return satisfy(r, now);
        }
        if (s.op == "anchor")
        {
            const auto p = host_.pose();
            if (not p.has_value()) return fail(r, now, "no_pose", s.id);
            captured_[*s.string("name")] = *p;
            return satisfy(r, now);
        }
        const auto until = s.pred("until");
        if (not r.started)
        {
            if (until.has_value() and holds(*until)) return satisfy(r, now, "until");
            auto resolved = resolve(s);
            if (not resolved.has_value()) return fail(r, now, "no_anchor", s.id);
            r.resolved = std::move(*resolved);
            if (auto refused = host_.start(r.resolved); refused.has_value()) return fail(r, now, *refused, s.id);
            r.started = true;
            event(std::format("step {}: started", label(s)));
        }
        const double elapsed_s = double(now - r.began) / 1000.0;
        if (until.has_value() and holds(*until)) return finish_leaf(r, now, "until");
        if (const auto secs = s.number("s"); s.op == "wait" and secs.has_value() and elapsed_s >= *secs)
            return finish_leaf(r, now, "satisfied");
        if (const auto budget = s.number("budget_s"); budget.has_value() and elapsed_s >= *budget)
        {
            if (until.has_value())
            {
                host_.cancel(r.resolved);
                return fail(r, now, "budget_exhausted", s.id);
            }
            return finish_leaf(r, now, "budget");
        }
        const LeafStatus st = host_.poll(r.resolved);
        if (st.state == LeafState::Failed) return fail(r, now, st.detail.empty() ? "failed" : st.detail, s.id);
        if (st.state == LeafState::Succeeded)
        {
            r.started = false;
            if (const auto expect = s.string("expect"); expect.has_value())
            {
                const Pred in_room{.name = "in_room", .refs = {{"room", Ref{.kind = Ref::Kind::Node, .name = *expect}}}};
                if (not holds(in_room)) return fail(r, now, "wrong_room", s.id);
            }
            satisfy(r, now);
        }
    }

    /// The leaf ends on the interpreter's say-so (until, s, budget, resume): the host stops it.
    void finish_leaf(Run& r, std::uint64_t now, std::string why)
    {
        host_.cancel(r.resolved);
        r.started = false;
        satisfy(r, now, std::move(why));
    }

    /// A copy of the step with every anchor replaced by its pose: captured during this mission, else declared.
    std::optional<Step> resolve(const Step& s) const
    {
        Step out = s;
        for (auto& [_, v] : out.args)
            if (auto* ref = std::get_if<Ref>(&v); ref != nullptr and ref->kind == Ref::Kind::Anchor)
            {
                const auto c = captured_.find(ref->name);
                const auto d = anchors_.find(ref->name);
                if (c == captured_.end() and d == anchors_.end()) return std::nullopt;
                *ref = Ref{.kind = Ref::Kind::Pose, .pose = c != captured_.end() ? c->second : d->second};
            }
        return out;
    }

    void cancel_subtree(Run& r, std::uint64_t now, const std::string& outcome)
    {
        if (r.state != StepState::Running) return;
        if (r.started) host_.cancel(r.resolved);
        r.started = false;
        cancel_subtree_kids(r, now, outcome);
        r.state = StepState::Cancelled;
        r.outcome = outcome;
        r.t1 = now;
        close(r, now);
    }
    void cancel_subtree_kids(Run& r, std::uint64_t now, const std::string& outcome)
    {
        for (auto& k : r.kids) cancel_subtree(k, now, outcome);
    }

    template <class R>
    static R* running_child(R& r)
    {
        for (auto& k : r.kids)
            if (k.state == StepState::Running) return &k;
        return nullptr;
    }
    static Run* running_leaf(Run& r)
    {
        Run* cur = &r;
        while (Run* next = running_child(*cur)) cur = next;
        return cur == &r ? nullptr : cur;
    }
};

}   // namespace rc::mission
