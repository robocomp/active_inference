// mission_plan.h: parse/validate for the planner, name check, progress lines, reports.
#include "../src/mission_plan.h"

#include <clocale>
#include <print>

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

using namespace rc::chat;
using namespace rc::mission;
using SS = Interpreter::StepState;

const std::string kText = "mission \"t\"\n  cross door_1 bind=n\n  explore $n budget_s=5s\n  report \"objects here\"\n";

Progress snap(std::string status, std::vector<Interpreter::StepView> steps)
{
    Progress p;
    p.status = std::move(status);
    p.steps = std::move(steps);
    return p;
}
Interpreter::StepView view(std::string id, std::string op, SS st, std::uint64_t t0 = 0, std::uint64_t t1 = 0)
{
    return {.id = std::move(id), .op = std::move(op), .state = st, .t0 = t0, .t1 = t1, .attempt = 1};
}
}   // namespace

int main()
{
    std::setlocale(LC_ALL, "");

    // ── parse_planned ────────────────────────────────────────────────────────────────────────────────────────────
    const auto m = parse_planned(kText);
    CHECK(m.has_value());
    if (not m.has_value()) return 1;
    CHECK(m->steps.size() == 3);
    const auto from_json = parse_planned("  " + encode_json(*m));
    CHECK(from_json.has_value() and *from_json == *m);
    const auto unbound = parse_planned("mission\n  explore $x budget_s=5s\n");
    CHECK(not unbound.has_value() and unbound.error().contains("unbound"));
    CHECK(not parse_planned("mission\n  fly\n").has_value());

    // ── unknown_name ─────────────────────────────────────────────────────────────────────────────────────────────
    const auto exists = [](std::string_view n) { return n == "door_1"; };
    const auto bad = parse_planned("mission\n  explore room_9 budget_s=5s\n");
    CHECK(bad.has_value() and unknown_name(*bad, exists) == "room_9");
    CHECK(not unknown_name(*m, exists).has_value());
    const auto in_pred = parse_planned("mission\n  wait s=5s until=door_open(door_7)<0.5\n");
    CHECK(in_pred.has_value() and unknown_name(*in_pred, exists) == "door_7");
    const auto nested = parse_planned("mission\n  repeat 2\n    cross door_1\n    open door_8\n");
    CHECK(nested.has_value() and unknown_name(*nested, exists) == "door_8");

    // ── progress_lines ───────────────────────────────────────────────────────────────────────────────────────────
    const Progress idle = snap("in_progress", {view("1", "cross", SS::Pending), view("2", "explore", SS::Pending),
                                               view("3", "report", SS::Pending)});
    const Progress run1 = snap("in_progress", {view("1", "cross", SS::Running, 1000), view("2", "explore", SS::Pending),
                                               view("3", "report", SS::Pending)});
    const auto l1 = progress_lines(idle, run1, *m);
    CHECK(l1 == std::vector<std::string>{"→ 1/3 1:cross door_1"});
    const Progress done1 = snap("in_progress", {view("1", "cross", SS::Satisfied, 1000, 43000),
                                                view("2", "explore", SS::Pending), view("3", "report", SS::Pending)});
    const auto l2 = progress_lines(run1, done1, *m);
    CHECK(l2.size() == 1 and l2[0].starts_with("✓ 1:cross door_1 (42 s)"));
    const Progress paused = snap("paused|2", run1.steps);
    const auto l3 = progress_lines(run1, paused, *m);
    CHECK(l3.size() == 1 and l3[0].starts_with("⏸ paused at 2/3"));
    const auto l4 = progress_lines(paused, run1, *m);
    CHECK(l4.size() == 1 and l4[0].starts_with("▶ resumed: 2/3"));
    Progress failed = run1;
    failed.steps[0].state = SS::Failed;
    failed.steps[0].outcome = "door shut";
    CHECK(progress_lines(run1, failed, *m) == std::vector<std::string>{"✗ 1:cross door_1: door shut"});
    Progress retry = run1;
    retry.steps[0].attempt = 2;
    CHECK(progress_lines(run1, retry, *m) == std::vector<std::string>{"⟳ retry 2/1 1:cross door_1"});
    CHECK(progress_lines(run1, snap("aborted|cancelled", run1.steps), *m) ==
          std::vector<std::string>{"■ mission cancelled"});
    CHECK(progress_lines(run1, run1, *m).empty());

    // ── new_reports ──────────────────────────────────────────────────────────────────────────────────────────────
    const Progress rep = snap("in_progress", {view("1", "cross", SS::Satisfied, 1000, 2000),
                                              view("2", "explore", SS::Satisfied, 2000, 3000),
                                              view("3", "report", SS::Satisfied, 3000, 3000)});
    CHECK(new_reports(done1, rep, *m) == std::vector<std::string>{"objects here"});
    CHECK(new_reports(rep, rep, *m).empty());

    std::println("test_mission_plan: {} checks passed{}", checks - failures, failures ? " — FAILURES" : "");
    return failures ? 1 : 0;
}
