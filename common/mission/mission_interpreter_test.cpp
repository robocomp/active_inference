// g++ -std=c++23 -O0 -g -Wall -Wextra -Wno-deprecated-literal-operator common/mission/mission_interpreter_test.cpp -o /tmp/mission_interpreter_test && /tmp/mission_interpreter_test
//
// The step machine (mission_interpreter.h) against a scripted host: what it starts, polls and cancels, in what order,
// and what it reports. The controller's adapter (DoorPassage, RelativeMotion, point target, selector) is not here —
// only the control flow every mission shares, so it can be checked without a robot. Missions are written in the DSL.
#include "mission_interpreter.h"
#include "mission_text.h"

#include <clocale>
#include <map>
#include <print>
#include <string>
#include <vector>

namespace
{
int failures = 0, checks = 0;

void check(bool ok, int line, std::string_view what)
{
    ++checks;
    if (not ok)
    {
        ++failures;
        std::println("FAIL line {}: {}", line, what);
    }
}
#define CHECK(x) check((x), __LINE__, #x)

using namespace rc::mission;

/// Records every call; the test scripts what each leaf and predicate reports.
struct FakeHost final : Host
{
    std::vector<std::string> log;                  // "start 1:cross door_27", "cancel 2:explore room_2", "say hi", "halt"
    std::vector<Step> started;                     // the steps start() received (anchors resolved)
    std::map<std::string, LeafStatus> leaf;        // by step id; absent = Running
    std::map<std::string, std::string> refuse;     // by step id: start() refuses with this reason
    std::map<std::string, double> belief;          // by predicate text (pred_text); absent = unknown
    std::optional<Pose> where = Pose{.room = "room_1", .x = 1, .y = 2, .yaw = 0.5, .sigma_xy = 0.1};

    std::optional<std::string> start(const Step& s) override
    {
        log.push_back("start " + s.id + ":" + s.op);
        started.push_back(s);
        if (const auto it = refuse.find(s.id); it != refuse.end()) return it->second;
        leaf.erase(s.id);   // a new attempt starts Running
        return std::nullopt;
    }
    LeafStatus poll(const Step& s) override
    {
        const auto it = leaf.find(s.id);
        return it == leaf.end() ? LeafStatus{} : it->second;
    }
    void cancel(const Step& s) override { log.push_back("cancel " + s.id + ":" + s.op); }
    void say(const std::string& text) override { log.push_back("say " + text); }
    void halt() override { log.push_back("halt"); }
    std::optional<Pose> pose() override { return where; }
    std::optional<double> eval(const Pred& p) override
    {
        const auto it = belief.find(text_detail::pred_text(Pred{.name = p.name, .refs = p.refs}));
        return it == belief.end() ? std::nullopt : std::optional<double>(it->second);
    }

    void done(const std::string& id) { leaf[id] = {.state = LeafState::Succeeded}; }
    void fail(const std::string& id, std::string why) { leaf[id] = {.state = LeafState::Failed, .detail = std::move(why)}; }
    int count(std::string_view entry) const { return int(std::ranges::count(log, std::string(entry))); }
};

Mission parse(std::string_view text)
{
    auto m = parse_text(text);
    if (not m.has_value()) std::println("  parse: {}", m.error().message());
    return m.value_or(Mission{});
}

using S = Interpreter::State;
}   // namespace

int main()
{
    std::setlocale(LC_ALL, "");

    // ── seq: one leaf at a time, in order; instant steps run in the same tick as the step before them ──────────────
    {
        FakeHost h;
        Interpreter in(h);
        CHECK(not in.start(parse("mission\n  cross door_27\n  say there\n  move distance_m=0.5\n"), 0).has_value());
        CHECK(in.state() == S::Running);
        in.tick(0);
        CHECK(h.log == std::vector<std::string>{"start 1:cross"});
        CHECK(in.status() == "in_progress");
        CHECK(in.progress().starts_with("1:cross door_27|running"));
        in.tick(100);
        CHECK(h.log.size() == 1);   // still running: nothing new
        h.done("1");
        in.tick(200);
        CHECK((h.log == std::vector<std::string>{"start 1:cross", "say there", "start 3:move"}));
        h.done("3");
        in.tick(300);
        CHECK(in.state() == S::Completed);
        CHECK(in.status() == "completed");
        const auto v = in.steps();
        CHECK(v.size() == 3 and v[0].state == Interpreter::StepState::Satisfied and v[0].t0 == 0 and v[0].t1 == 200);
        in.tick(400);
        CHECK(h.log.size() == 3);   // a finished mission does nothing more
    }

    // ── A refusal or a failure stops the mission at that step; later steps never start ────────────────────────────
    {
        FakeHost h;
        Interpreter in(h);
        h.refuse["1"] = "door 'door_9' is not in the graph";
        CHECK(not in.start(parse("mission\n  cross door_9\n  say never\n"), 0).has_value());
        in.tick(0);
        CHECK(in.state() == S::Failed);
        CHECK(in.status() == "failed|1|door 'door_9' is not in the graph");
        CHECK(h.count("say never") == 0);
    }
    {
        FakeHost h;
        Interpreter in(h);
        in.start(parse("mission\n  seq\n    open door_19\n    cross door_19\n"), 0);
        in.tick(0);
        h.fail("1.1", "door_wont_open");
        in.tick(10);
        CHECK(in.status() == "failed|1.1|door_wont_open");   // the step that failed, not its parent
        CHECK(h.count("start 1.2:cross") == 0);
    }

    // ── on_fail: retry restarts the step, skip carries on, abort (the default) propagates ──────────────────────────
    {
        FakeHost h;
        Interpreter in(h);
        in.start(parse("mission\n  cross door_27 retry=2\n  say through\n"), 0);
        in.tick(0);
        h.fail("1", "unreachable");
        in.tick(10);
        CHECK(h.count("start 1:cross") == 2);
        CHECK(in.steps()[0].attempt == 2);
        h.fail("1", "unreachable");
        in.tick(20);
        CHECK(h.count("start 1:cross") == 3);
        h.done("1");
        in.tick(30);
        CHECK(in.state() == S::Completed and h.count("say through") == 1);
    }
    {
        FakeHost h;
        Interpreter in(h);
        in.start(parse("mission\n  cross door_27 retry=1\n"), 0);
        in.tick(0);
        h.fail("1", "unreachable");
        in.tick(10);
        h.fail("1", "unreachable");
        in.tick(20);
        CHECK(in.status() == "failed|1|unreachable");
        CHECK(h.count("start 1:cross") == 2);   // the first try + 1 retry, no more
    }
    {
        FakeHost h;
        Interpreter in(h);
        in.start(parse("mission\n  open door_19 on_fail=skip\n  say anyway\n"), 0);
        in.tick(0);
        h.fail("1", "door_wont_open");
        in.tick(10);
        CHECK(in.state() == S::Completed and h.count("say anyway") == 1);
        CHECK(in.steps()[0].state == Interpreter::StepState::Skipped and in.steps()[0].outcome == "door_wont_open");
    }
    {   // a composite retries as a whole: its body starts again from the first step
        FakeHost h;
        Interpreter in(h);
        in.start(parse("mission\n  seq retry=1\n    open door_19\n    cross door_19\n"), 0);
        in.tick(0);
        h.done("1.1");
        in.tick(10);
        h.fail("1.2", "cross_unreachable");
        in.tick(20);
        CHECK(h.count("start 1.1:open") == 2);
    }

    // ── repeat times: the body N times, in order ───────────────────────────────────────────────────────────────────
    {
        FakeHost h;
        Interpreter in(h);
        in.start(parse("mission\n  repeat 2\n    cross door_1\n    cross door_2\n"), 0);
        for (std::uint64_t t = 0; t < 10 and in.state() == S::Running; ++t)
        {
            in.tick(t * 10);
            if (not h.started.empty()) h.done(h.started.back().id);
        }
        CHECK((h.log == std::vector<std::string>{"start 1.1:cross", "start 1.2:cross", "start 1.1:cross", "start 1.2:cross"}));
        CHECK(in.state() == S::Completed);
    }

    // ── repeat until: checked before every iteration; the budget ends it, cancelling what runs ─────────────────────
    {
        FakeHost h;
        Interpreter in(h);
        in.start(parse("mission\n  repeat until=in_room(room_2) budget_s=60s\n    cross door_1\n"), 0);
        h.belief["in_room(room_2)"] = 0;
        in.tick(0);
        h.done("1.1");
        in.tick(10);
        CHECK(h.count("start 1.1:cross") == 2);
        h.belief["in_room(room_2)"] = 1;
        h.done("1.1");
        in.tick(20);
        CHECK(in.state() == S::Completed and h.count("start 1.1:cross") == 2);
    }
    {
        FakeHost h;
        Interpreter in(h);
        in.start(parse("mission\n  repeat until=in_room(room_2) budget_s=60s\n    cross door_1\n"), 0);
        in.tick(0);
        in.tick(59'999);
        CHECK(in.state() == S::Running);
        in.tick(60'000);
        CHECK(in.status() == "failed|1|budget_exhausted");
        CHECK(h.count("cancel 1.1:cross") == 1);
    }
    {   // a body that never waits does not spin: one iteration per tick
        FakeHost h;
        Interpreter in(h);
        in.start(parse("mission\n  repeat until=in_room(room_2) budget_s=1s\n    say again\n"), 0);
        in.tick(0);
        CHECK(h.count("say again") == 1);
        in.tick(10);
        CHECK(h.count("say again") == 2);
    }

    // ── explore: until ends it early, the budget ends it; with an until unmet the budget is a failure ──────────────
    {
        FakeHost h;
        Interpreter in(h);
        in.start(parse("mission\n  explore room_2 budget_s=30s\n"), 0);
        in.tick(0);
        in.tick(29'000);
        CHECK(in.state() == S::Running);
        in.tick(30'000);
        CHECK(in.state() == S::Completed and h.count("cancel 1:explore") == 1);   // "explore for 30 s" is done
    }
    {
        FakeHost h;
        Interpreter in(h);
        in.start(parse("mission\n  explore room_2 until=explored(room_2)>=0.9 budget_s=30s\n"), 0);
        h.belief["explored(room_2)"] = 0.5;
        in.tick(0);
        h.belief["explored(room_2)"] = 0.9;
        in.tick(1000);
        CHECK(in.state() == S::Completed and h.count("cancel 1:explore") == 1);
    }
    {
        FakeHost h;
        Interpreter in(h);
        in.start(parse("mission\n  explore room_2 until=explored(room_2)>=0.9 budget_s=30s\n"), 0);
        in.tick(0);   // the belief is unknown: never satisfied
        in.tick(30'000);
        CHECK(in.status() == "failed|1|budget_exhausted");
    }
    {   // already true before it starts: nothing to do, and nothing is started
        FakeHost h;
        Interpreter in(h);
        h.belief["explored(room_2)"] = 0.95;
        in.start(parse("mission\n  explore room_2 until=explored(room_2)>=0.9 budget_s=30s\n"), 0);
        in.tick(0);
        CHECK(in.state() == S::Completed and h.count("start 1:explore") == 0);
    }

    // ── wait: s elapses, or until holds (a `<` boundary too), or the person says resume ────────────────────────────
    {
        FakeHost h;
        Interpreter in(h);
        in.start(parse("mission\n  wait s=5s\n"), 0);
        in.tick(0);
        CHECK(h.count("start 1:wait") == 1);   // the host holds the base, armed
        in.tick(4'999);
        CHECK(in.state() == S::Running);
        in.tick(5'000);
        CHECK(in.state() == S::Completed and h.count("cancel 1:wait") == 1);
    }
    {
        FakeHost h;
        Interpreter in(h);
        in.start(parse("mission\n  wait until=door_open(door_19)<0.5\n"), 0);
        h.belief["door_open(door_19)"] = 0.8;
        in.tick(0);
        in.tick(100);
        CHECK(in.state() == S::Running);
        h.belief["door_open(door_19)"] = 0.2;
        in.tick(200);
        CHECK(in.state() == S::Completed);
    }
    {
        FakeHost h;
        Interpreter in(h);
        in.start(parse("mission\n  wait until=in_room(room_2)\n  say thanks\n"), 0);
        CHECK(not in.resume(0));   // nothing is waiting yet: the first tick has not run
        in.tick(0);
        CHECK(in.resume(50));
        in.tick(100);
        CHECK(in.state() == S::Completed and h.count("say thanks") == 1);
        CHECK(in.steps()[0].outcome == "resumed");
    }
    {   // resume ends only a wait, never another step
        FakeHost h;
        Interpreter in(h);
        in.start(parse("mission\n  cross door_1\n"), 0);
        in.tick(0);
        CHECK(not in.resume(10));
        CHECK(in.state() == S::Running);
    }

    // ── anchors: captured at the step, resolved to a pose before the host sees them ───────────────────────────────
    {
        FakeHost h;
        Interpreter in(h);
        in.start(parse("mission\nanchor start = pose:room_1(0.4, -1.1, 1.5, 0.05)\n  anchor here\n  go_to anchor:here\n"
                       "  go_to anchor:start standoff_m=0.8m\n"), 0);
        in.tick(0);
        CHECK(h.started.size() == 1);
        const auto to = h.started[0].ref("to");
        CHECK(to.has_value() and to->kind == Ref::Kind::Pose and to->pose == *h.where);
        h.done("2");
        h.where->x = 9;   // the robot moved: the anchor keeps the pose it had at the step
        in.tick(10);
        const auto start = h.started.back().ref("to");
        CHECK(start.has_value() and start->kind == Ref::Kind::Pose and start->pose.x == 0.4 and start->pose.y == -1.1);
        CHECK(h.started.back().number("standoff_m") == 0.8);   // other fields pass through
    }
    {
        FakeHost h;
        Interpreter in(h);
        h.where.reset();
        in.start(parse("mission\n  anchor here\n  go_to anchor:here\n"), 0);
        in.tick(0);
        CHECK(in.status() == "failed|1|no_pose");
    }

    // ── cross expect: the passage finishing is not enough, the robot must be in that room ──────────────────────────
    {
        FakeHost h;
        Interpreter in(h);
        in.start(parse("mission\n  cross door_27 expect=room_2\n"), 0);
        in.tick(0);
        h.belief["in_room(room_2)"] = 0;
        h.done("1");
        in.tick(10);
        CHECK(in.status() == "failed|1|wrong_room");
    }
    {
        FakeHost h;
        Interpreter in(h);
        in.start(parse("mission\n  cross door_27 expect=room_2\n"), 0);
        in.tick(0);
        h.belief["in_room(room_2)"] = 1;
        h.done("1");
        in.tick(10);
        CHECK(in.state() == S::Completed);
    }

    // ── stop is a step that halts the base; the mission goes on ───────────────────────────────────────────────────
    {
        FakeHost h;
        Interpreter in(h);
        in.start(parse("mission\n  stop\n  say stopped\n"), 0);
        in.tick(0);
        CHECK((h.log == std::vector<std::string>{"halt", "say stopped"}) and in.state() == S::Completed);
    }

    // ── cancel: what runs is cancelled; the edge reports why; a new start supersedes ──────────────────────────────
    {
        FakeHost h;
        Interpreter in(h);
        in.start(parse("mission\n  seq\n    explore room_2 budget_s=60s\n"), 0);
        in.tick(0);
        in.cancel("stopped_by_user", 10);
        CHECK(in.state() == S::Aborted and in.status() == "aborted|stopped_by_user");
        CHECK(h.count("cancel 1.1:explore") == 1);
        CHECK(in.steps()[1].state == Interpreter::StepState::Cancelled);
        in.tick(20);
        CHECK(h.count("start 1.1:explore") == 1);
    }
    {
        FakeHost h;
        Interpreter in(h);
        in.start(parse("mission \"a\"\n  cross door_1\n"), 0);
        in.tick(0);
        CHECK(not in.start(parse("mission \"b\"\n  cross door_2\n"), 10).has_value());
        CHECK(h.count("cancel 1:cross") == 1);
        CHECK(in.title() == "b" and in.state() == S::Running);
        bool superseded = false;
        for (const auto& e : in.take_events()) superseded = superseded or e.find("superseded") != std::string::npos;
        CHECK(superseded);
    }

    // ── an invalid mission is refused before anything moves, and a running one is left alone ──────────────────────
    {
        FakeHost h;
        Interpreter in(h);
        in.start(parse("mission\n  cross door_1\n"), 0);
        in.tick(0);
        Mission bad = parse("mission\n  stop\n");
        bad.steps[0].op = "fly";
        const auto e = in.start(bad, 10);
        CHECK(e.has_value() and e->status() == "unsupported:fly");
        CHECK(in.state() == S::Running and h.count("cancel 1:cross") == 0);
    }

    // ── progress text: where it is, in what state, and the last thing that happened ───────────────────────────────
    {
        FakeHost h;
        Interpreter in(h);
        in.start(parse("mission\n  repeat 2\n    cross passage:passage_0@room_2 retry=1\n"), 0);
        in.tick(0);
        h.fail("1.1", "unreachable");
        in.tick(10);
        const std::string p = in.progress();
        CHECK(p.starts_with("1.1:cross passage:passage_0@room_2|running"));
        CHECK(p.find("attempt 2/2") != std::string::npos);
        CHECK(p.find("iteration 1/2") != std::string::npos);
        CHECK(not in.take_events().empty() and in.take_events().empty());   // drained once
    }

    std::println("mission_interpreter_test: {} checks, {} failures", checks, failures);
    return failures == 0 ? 0 : 1;
}
