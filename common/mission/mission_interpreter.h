#pragma once
/*
 * mission_interpreter.h — the step machine that runs a grounded mission (mission_ir.h), 2026-10-17.
 * Design: language_agent/docs/mission_language_2026-10-13.md §3.1 (what each primitive compiles to), §3.5 (failure),
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
#include <chrono>
#include <cstdint>
#include <format>
#include <map>
#include <optional>
#include <cctype>
#include <ranges>
#include <string>
#include <deque>
#include <set>
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
    std::string detail{};   ///< Failed: the outcome word (unreachable, door_wont_open, …). Running: why the step is
                            ///< WAITING, in words for the person ("no door I know of in room_1 leads to room_3 yet");
                            ///< "" = it is making progress. Shown in progress() and the snapshot (StepView::note).
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
    /// The room the robot is physically in now, a proto-room included, for `cross … bind=NAME`. nullopt = not known.
    /// (The controller answers with rc::room::physical_room, the same rule the chat reports by.)
    virtual std::optional<std::string> current_room() = 0;
    /// `bind=NAME` has just latched `room` (a log line for whoever runs the robot; the default says nothing).
    virtual void on_bind(const std::string& /*name*/, const std::string& /*room*/) {}
    /// A predicate's belief now: a probability, or 0 / 1 for a fact. nullopt = cannot be evaluated now.
    virtual std::optional<double> eval(const Pred& p) = 0;
    /// A SET predicate's members now: for `seen(what)`, the object nodes of that class (or that node) present in the
    /// graph. The interpreter turns "present now and not when the step started" into an appearance. Default: none
    /// (a host that does not answer it makes `seen` never hold; budgets still end the step).
    virtual std::vector<std::string> matches(const Pred& /*p*/) { return {}; }
    /// May this started leaf be cancelled now by a watch's reaction? `restart` = it would be started again afterwards
    /// (a moving `continue` reaction); false = it ends there (a `stop` watch). False while that would undo progress —
    /// a door passage past its approach restarts at the approach (room review I1, 2026-11-14), and one that has already
    /// succeeded must not be restarted at all (final review F1). The firing then waits until the leaf ends or becomes
    /// interruptible. Default: always.
    virtual bool interruptible(const Step& /*leaf*/, bool /*restart*/) { return true; }
    /// The wall clock now, epoch ms, for `wait at=` (the `now` of tick() is the caller's steady clock: durations only).
    virtual std::int64_t wall_ms()
    {
        return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    }
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
        std::string note;               ///< running and waiting: why (LeafStatus::detail); "" otherwise
        std::string after;              ///< a retry: the outcome the previous attempt failed with; "" on attempt 1
        std::vector<std::string> fired; ///< a watch: every node it fired on, in order (its size = the firing count)
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
        vars_.clear();
        root_step_ = Step{.id = "", .op = "seq", .steps = std::move(m.steps)};
        root_ = build(root_step_);
        state_ = State::Running;
        abort_outcome_.clear();
        paused_ = false;
        tick_ = 0;
        event(std::format("mission '{}': started ({} steps)", title_, root_step_.steps.size()));
        return std::nullopt;
    }

    void tick(std::uint64_t now_ms)
    {
        if (state_ != State::Running or paused_) return;
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
        paused_ = false;
        event(std::format("mission '{}': aborted ({})", title_, outcome));
    }

    /// Freeze the mission at any moment: the running leaf is cancelled (the host decides what holding means) and
    /// every clock stops. True if a running, not yet paused mission was paused.
    bool pause(std::uint64_t now_ms)
    {
        if (state_ != State::Running or paused_) return false;
        if (Run* leaf = running_leaf(root_); leaf != nullptr)
        {
            if (leaf->started) host_.cancel(leaf->resolved);
            leaf->started = false;
            paused_leaf_ = leaf->step->id;
            event(std::format("step {}: paused", label(*leaf->step)));
        }
        else paused_leaf_.clear();
        paused_ = true;
        paused_at_ = now_ms;
        return true;
    }

    /// Thaw it: the interrupted leaf restarts from its beginning with its whole budget, and no attempt is spent.
    /// Enclosing steps' budgets resume where they stopped (their clocks are shifted by the time paused).
    bool unpause(std::uint64_t now_ms)
    {
        if (state_ != State::Running or not paused_) return false;
        paused_ = false;
        for (Run* r = running_child(root_); r != nullptr; r = running_child(*r))
        {
            if (r->kids.empty())
            {
                r->began = now_ms;
                event(std::format("step {}: restarted", label(*r->step)));
            }
            else r->began += now_ms - paused_at_;
        }
        return true;
    }

    [[nodiscard]] bool paused() const { return paused_; }

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
            case State::Running: return paused_ ? "paused|" + paused_leaf_ : "in_progress";
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
        // paused: the same step path, so the paused step is the one reported (status() carries "paused|<id>")
        std::vector<const Run*> path;
        for (const Run* r = &root_; r != nullptr; r = running_child(*r)) path.push_back(r);
        const Run& leaf = *path.back();
        std::vector<std::string> notes;
        if (leaf.step->on_fail.kind == OnFail::Kind::Retry)
            notes.push_back(std::format("attempt {}/{}{}", leaf.attempt, leaf.step->on_fail.retries + 1,
                                        leaf.last_failure.empty() ? "" : " after " + leaf.last_failure));
        for (const Run* r : path | std::views::reverse)
            if (r->step->op == "repeat")
            {
                const auto times = r->step->integer("times");
                notes.push_back(times.has_value() ? std::format("iteration {}/{}", r->iteration + 1, *times)
                                                  : std::format("iteration {}", r->iteration + 1));
            }
        if (not leaf.note.empty()) notes.push_back("waiting: " + leaf.note);
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
                               .t1 = k.t1, .attempt = k.attempt, .iteration = k.iteration, .note = k.note,
                               .after = k.last_failure, .fired = k.fired});
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
        std::string note{};          ///< leaf, running: why it waits (LeafStatus::detail; '|' replaced, bt_state is '|'-split)
        std::string last_failure{};  ///< the outcome the previous attempt failed with (kept across a retry's reset)
        Step resolved{};        ///< leaf: what the host was given
        int branch = -1;        ///< if: the branch taken (0 then, 1 else); -1 = not decided yet
        bool seen_armed = false;                ///< a `seen` until: its baseline is taken
        std::set<std::string> seen_base{};      ///< the matches when it was taken
        // watch: the nodes it fired on (in order), those still to react to, whether its reaction body runs now, and
        // the scope's leaf it interrupted (restarted when the reaction ends)
        std::vector<std::string> fired{};
        std::deque<std::string> queue{};
        bool body_running = false;
        Run* interrupted = nullptr;
        std::vector<Run> kids{};
    };

    Host& host_;
    State state_ = State::Idle;
    std::string title_, abort_outcome_, last_event_;
    std::map<std::string, Pose, std::less<>> anchors_, captured_;
    std::map<std::string, std::string, std::less<>> vars_;   ///< `bind=NAME` → the room name; cleared in start()
    Step root_step_{};
    Run root_{};
    std::uint64_t tick_ = 0;
    bool paused_ = false;
    int in_reaction_ = 0;   ///< > 0 while a watch's reaction body advances (its failing steps are skipped: on_fail)
    std::uint64_t paused_at_ = 0;
    std::string paused_leaf_;
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
        r.note.clear();
        r.last_failure.clear();
        r.branch = -1;
        r.seen_armed = false;
        r.seen_base.clear();
        r.fired.clear();
        r.queue.clear();
        r.body_running = false;
        r.interrupted = nullptr;
        for (auto& k : r.kids) reset(k);
    }

    /// A ref with a `$NAME` replaced by the node it was bound to. False = the name is not bound (yet).
    bool substitute(Ref& ref) const
    {
        if (ref.kind != Ref::Kind::Var) return true;
        const auto v = vars_.find(ref.name);
        if (v == vars_.end()) return false;
        ref = Ref{.kind = Ref::Kind::Node, .name = v->second};
        return true;
    }
    bool substitute(Pred& p) const
    {
        for (auto& [_, ref] : p.refs)
            if (not substitute(ref)) return false;
        return true;
    }
    /// A text with every bound `$NAME` (a whole identifier after the `$`) replaced by its node; unbound names stay.
    std::string substitute_text(std::string_view text) const
    {
        std::string out;
        for (std::size_t i = 0; i < text.size();)
        {
            std::size_t j = i + 1;
            if (text[i] == '$')
                while (j < text.size() and (std::isalnum(static_cast<unsigned char>(text[j])) or text[j] == '_')) ++j;
            const auto v = text[i] == '$' ? vars_.find(text.substr(i + 1, j - i - 1)) : vars_.end();
            if (v != vars_.end()) out += v->second;
            else out += text.substr(i, j - i);
            i = j;
        }
        return out;
    }

    /// The decision a predicate gives now; nullopt = it cannot be evaluated (unbound name, host has no belief).
    std::optional<bool> decide(const Pred& pred)
    {
        Pred p = pred;
        if (not substitute(p)) return std::nullopt;   // an unbound name is a belief that cannot be evaluated
        if (p.name == "seen") return not host_.matches(p).empty();   // in an `if`: is one there now
        const auto v = host_.eval(p);
        if (not v.has_value()) return std::nullopt;
        if (not p.cmp.has_value()) return *v != 0.0;
        return p.cmp->op == Comparison::Op::Ge ? *v >= p.cmp->value : *v < p.cmp->value;
    }
    /// An `until`: true once it holds. `seen(what)` is an APPEARANCE: the first call on run `r` takes the baseline
    /// (and is false), later calls are true when a node is there that was not then.
    bool holds(const Pred& pred, Run* r = nullptr)
    {
        if (pred.name != "seen") return decide(pred).value_or(false);
        if (r == nullptr) return false;
        Pred p = pred;
        if (not substitute(p)) return false;
        const auto now = host_.matches(p);
        if (not r->seen_armed)
        {
            r->seen_armed = true;
            r->seen_base = {now.begin(), now.end()};
            return false;
        }
        return std::ranges::any_of(now, [r](const std::string& n) { return not r->seen_base.contains(n); });
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
            std::string why = r.outcome;
            reset(r);
            r.attempt = attempt;
            r.t0 = t0;
            r.last_failure = std::move(why);
            advance(r, now);   // bounded: each pass spends an attempt
            return;
        }
        // A reaction's step that fails is skipped unless it says otherwise: the watched steps carry on (final review F7:
        // one unreachable chair failed "approach every chair on the way" before the robot reached the room).
        if (f.kind == OnFail::Kind::Skip or (f.kind == OnFail::Kind::Abort and in_reaction_ > 0))
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
        if (s.op == "if") return run_if(r, now);
        if (not tend_watches(r, now)) return;   // a reaction runs (or a stop watch just ended the scope's steps)
        if (is_repeat and r.cur == 0 and r.kids.front().state == StepState::Pending and until.has_value() and holds(*until, &r))
            return satisfy(r, now, "until");
        while (true)
        {
            if (r.cur == r.kids.size())
            {
                // A firing held behind the last step (it could not be interrupted) runs now, before the watch closes
                // (final review F2: it was dropped). Back here once its reaction has ended.
                if (not tend_watches(r, now) or r.cur != r.kids.size()) return;
                close_watches(r, now);   // a watch lives as long as the steps after it in this sequence
                if (not is_repeat) return satisfy(r, now);
                ++r.iteration;
                if (const auto times = s.integer("times"); times.has_value() and r.iteration >= *times) return satisfy(r, now);
                for (auto& k : r.kids) reset(k);
                r.cur = 0;
                if (until.has_value() and holds(*until, &r)) return satisfy(r, now, "until");
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
            if (r.cur > 0 and not tend_watches(r, now)) return;   // a firing held while the step before ran acts now
            if (r.cur == r.kids.size()) continue;                  // a stop watch just ended the sequence
            Run& k = r.kids[r.cur];
            k.outer = is_repeat ? r.iteration : r.outer;
            if (k.step->op == "watch")
            {
                arm_watch(k, now);
                ++r.cur;
                continue;
            }
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

    // ── watch ──────────────────────────────────────────────────────────────────────────────────────────────────────
    /// Armed when its sequence reaches it: Running, with the matches now as its baseline; the sequence moves on.
    void arm_watch(Run& w, std::uint64_t now)
    {
        ++w.attempt;
        w.t0 = w.began = now;
        w.state = StepState::Running;
        open(w, now);
        Pred p = *w.step->pred("when");
        if (substitute(p))
            for (auto& n : host_.matches(p)) w.seen_base.insert(std::move(n));
        event(std::format("step {}: watching", label(*w.step)));
    }
    /// The nodes that appeared since the watch was armed join its queue, once each. A node of the baseline that
    /// leaves the graph stops being "known": if it comes back it is seen anew.
    void detect(Run& w)
    {
        Pred p = *w.step->pred("when");
        if (not substitute(p)) return;
        const auto now = host_.matches(p);
        std::erase_if(w.seen_base, [&now](const std::string& n) { return not std::ranges::contains(now, n); });
        for (const auto& n : now)
            if (not w.seen_base.contains(n) and not std::ranges::contains(w.fired, n))
            {
                w.fired.push_back(n);
                w.queue.push_back(n);
            }
    }
    /// The reaction body, as a seq over the watch's kids. True = it has ended (satisfied); false = still running or
    /// failed (w.state tells which).
    bool run_body(Run& w, std::uint64_t now)
    {
        while (w.cur < w.kids.size())
        {
            Run& k = w.kids[w.cur];
            k.outer = w.outer;
            ++in_reaction_;
            advance(k, now);
            --in_reaction_;
            if (k.state == StepState::Satisfied or k.state == StepState::Skipped)
            {
                ++w.cur;
                continue;
            }
            if (k.state == StepState::Failed) fail(w, now, k.outcome, k.origin);
            return false;
        }
        return true;
    }
    /// Every armed watch of `r` (innermost scope = these kids before r.cur), in order: detect, then run its
    /// reactions. False = the scope must not advance this tick (a reaction is running, the scope failed, or a stop
    /// watch ended it — then r.cur is past the end and the caller's next pass closes the sequence).
    bool tend_watches(Run& r, std::uint64_t now)
    {
        for (std::size_t i = 0; i < r.cur and i < r.kids.size(); ++i)
        {
            Run& w = r.kids[i];
            if (w.step->op != "watch" or w.state != StepState::Running) continue;
            detect(w);
            const bool stop = w.step->string("then") == "stop";
            while (true)
            {
                if (w.body_running)
                {
                    if (not run_body(w, now))
                    {
                        if (w.state == StepState::Failed) fail(r, now, w.outcome, w.origin);
                        return false;
                    }
                    w.body_running = false;
                    if (stop)
                    {
                        satisfy(w, now, "fired");
                        r.cur = r.kids.size();
                        return true;   // the loop closes the sequence (a repeat goes on to its next iteration)
                    }
                    if (w.interrupted != nullptr)
                    {
                        w.interrupted->began = now;   // the interrupted leaf restarts whole, no attempt spent
                        event(std::format("step {}: restarted", label(*w.interrupted->step)));
                        w.interrupted = nullptr;
                    }
                }
                if (w.queue.empty()) break;
                // A reaction that would cancel the scope's running leaf waits while that leaf must not be interrupted.
                const bool interrupts = stop or (claims_of(*w.step) & kBase) != 0;
                if (Run* l = scope_leaf(r);
                    interrupts and l != nullptr and l->started and not host_.interruptible(l->resolved, not stop))
                    break;
                // One driver at a time (final review F3): it also waits while another watch's reaction of this scope runs.
                const auto other_running = [&r, i](std::size_t j) { return j != i and r.kids[j].body_running; };
                if (interrupts and std::ranges::any_of(std::views::iota(std::size_t{0}, r.kids.size()), other_running)) break;
                const std::string name = w.queue.front();
                w.queue.pop_front();
                if (const auto each = w.step->string("each"); each.has_value()) vars_[*each] = name;
                for (auto& k : w.kids) reset(k);
                w.cur = 0;
                w.body_running = true;
                event(std::format("step {}: fired on {}", label(*w.step), name));
                if (stop)
                {
                    const std::string why = "watch:" + w.step->pred("when")->name;
                    for (std::size_t j = i + 1; j < r.kids.size(); ++j)
                    {
                        Run& k = r.kids[j];
                        if (k.state == StepState::Running) cancel_subtree(k, now, why);
                        else if (k.state == StepState::Pending)
                        {
                            k.state = StepState::Cancelled;
                            k.outcome = why;
                            k.t1 = now;
                        }
                    }
                }
                else if ((claims_of(*w.step) & kBase) != 0)
                    if (Run* l = scope_leaf(r); l != nullptr and l->started)
                    {
                        host_.cancel(l->resolved);
                        l->started = false;
                        event(std::format("step {}: interrupted by {}", label(*l->step), label(*w.step)));
                        w.interrupted = l;
                    }
            }
        }
        return true;
    }
    /// The leaf of r's current child that runs now (the child itself when it is a leaf); nullptr = none.
    static Run* scope_leaf(Run& r)
    {
        if (r.cur >= r.kids.size()) return nullptr;
        Run& k = r.kids[r.cur];
        if (k.kids.empty()) return k.state == StepState::Running ? &k : nullptr;
        return running_leaf(k);
    }
    void close_watches(Run& r, std::uint64_t now)
    {
        for (auto& w : r.kids)
            if (w.step->op == "watch" and w.state == StepState::Running) satisfy(w, now, "scope ended");
    }

    /// `if`: decided once on entry, then the chosen branch runs like a seq; the other is Skipped.
    void run_if(Run& r, std::uint64_t now)
    {
        if (r.branch < 0)
        {
            const Pred cond = *r.step->pred("cond");
            const auto yes = decide(cond);
            if (not yes.has_value()) return fail(r, now, "cannot_evaluate:" + cond.name, r.step->id);
            r.branch = *yes ? 0 : 1;
            for (std::size_t i = 0; i < r.kids.size(); ++i)
                if (int(i) != r.branch) r.kids[i].state = StepState::Skipped;
            if (std::size_t(r.branch) >= r.kids.size()) return satisfy(r, now, "else-none");
        }
        Run& k = r.kids[std::size_t(r.branch)];
        k.outer = r.outer;
        advance(k, now);
        if (k.state == StepState::Satisfied or k.state == StepState::Skipped) return satisfy(r, now);
        if (k.state == StepState::Failed) return fail(r, now, k.outcome, k.origin);
    }

    void run_leaf(Run& r, std::uint64_t now)
    {
        const Step& s = *r.step;
        if (s.op == "say")
        {
            host_.say(substitute_text(*s.string("text")));
            return satisfy(r, now);
        }
        if (s.op == "stop")
        {
            host_.halt();
            return satisfy(r, now);
        }
        if (s.op == "report") return satisfy(r, now);   // the text is the model's, read by the chat; nothing to do here
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
            if (until.has_value() and holds(*until, &r)) return satisfy(r, now, "until");
            auto resolved = resolve(s);
            if (not resolved.has_value()) return fail(r, now, resolve_failure_, s.id);
            r.resolved = std::move(*resolved);
            if (auto refused = host_.start(r.resolved); refused.has_value()) return fail(r, now, *refused, s.id);
            r.started = true;
            event(std::format("step {}: started", label(s)));
        }
        const double elapsed_s = double(now - r.began) / 1000.0;
        if (until.has_value() and holds(*until, &r)) return finish_leaf(r, now, "until");
        if (const auto secs = s.number("s"); s.op == "wait" and secs.has_value() and elapsed_s >= *secs)
            return finish_leaf(r, now, "satisfied");
        const auto at = s.op == "wait" ? s.string("at") : std::nullopt;
        const auto at_ms = at.has_value() ? parse_iso(*at) : std::nullopt;
        if (at_ms.has_value())
        {
            const std::int64_t left_ms = *at_ms - host_.wall_ms();
            if (left_ms <= 0) return finish_leaf(r, now, "satisfied");
            // the wait note (StepView::note), as a LeafStatus detail would be: "until 18:00 (in 42 min)"
            const std::int64_t min = (left_ms + 59'999) / 60'000;
            r.note = min < 120 ? std::format("until {} (in {} min)", iso_hhmm(*at), min)
                               : std::format("until {} on {} (in {} h)", iso_hhmm(*at), at->substr(0, 10), (min + 30) / 60);
            const LeafStatus st = host_.poll(r.resolved);
            if (st.state == LeafState::Failed) return fail(r, now, st.detail.empty() ? "failed" : st.detail, s.id);
            return;
        }
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
        r.note = st.state == LeafState::Running ? st.detail : std::string{};
        std::ranges::replace(r.note, '|', '/');
        if (st.state == LeafState::Failed) return fail(r, now, st.detail.empty() ? "failed" : st.detail, s.id);
        if (st.state == LeafState::Succeeded)
        {
            r.started = false;
            if (const auto expect = s.string("expect"); expect.has_value())
            {
                const Pred in_room{.name = "in_room", .refs = {{"room", Ref{.kind = Ref::Kind::Node, .name = *expect}}}};
                if (not holds(in_room)) return fail(r, now, "wrong_room", s.id);
            }
            if (const auto& name = s.bind; name.has_value())
            {
                const auto room = host_.current_room();
                if (not room.has_value()) return fail(r, now, "no_room", s.id);
                vars_[*name] = *room;
                host_.on_bind(*name, *room);
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
    /// and every `$NAME` by the room it was bound to. nullopt: resolve_failure_ says why (no_anchor | unbound).
    std::optional<Step> resolve(const Step& s)
    {
        Step out = s;
        for (auto& [_, v] : out.args)
        {
            if (auto* pred = std::get_if<Pred>(&v); pred != nullptr)
            {
                if (not substitute(*pred)) return unresolved("unbound");
                continue;
            }
            auto* ref = std::get_if<Ref>(&v);
            if (ref == nullptr) continue;
            if (ref->kind == Ref::Kind::Var)
            {
                if (not substitute(*ref)) return unresolved("unbound");
            }
            else if (ref->kind == Ref::Kind::Anchor)
            {
                const auto c = captured_.find(ref->name);
                const auto d = anchors_.find(ref->name);
                if (c == captured_.end() and d == anchors_.end()) return unresolved("no_anchor");
                *ref = Ref{.kind = Ref::Kind::Pose, .pose = c != captured_.end() ? c->second : d->second};
            }
        }
        return out;
    }
    std::nullopt_t unresolved(std::string why)
    {
        resolve_failure_ = std::move(why);
        return std::nullopt;
    }
    std::string resolve_failure_{};

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
        for (auto& k : r.kids)   // a watch whose reaction runs comes first: that is what drives now
            if (k.state == StepState::Running and k.step->op == "watch" and k.body_running) return &k;
        for (auto& k : r.kids)
            if (k.state == StepState::Running and k.step->op != "watch") return &k;
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
