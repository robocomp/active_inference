// The plan view's model (src/plan_model.h): the rows the window paints, what each says while it runs and after.
#include "../src/plan_model.h"
#include "../../common/mission/mission_progress.h"
#include "../../common/mission/mission_text.h"

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

using namespace rc::mission;
using rc::chat::plan::Model;
using SS = Interpreter::StepState;

Mission parse(std::string_view t) { return parse_text(t).value(); }

Interpreter::StepView sv(std::string id, SS st, int attempt = 1, std::uint64_t t0 = 0, std::uint64_t t1 = 0,
                         std::string outcome = {}, int iteration = 0)
{
    return {.id = std::move(id), .op = "", .state = st, .outcome = std::move(outcome), .t0 = t0, .t1 = t1,
            .attempt = attempt, .iteration = iteration};
}
}   // namespace

int main()
{
    std::setlocale(LC_ALL, "");

    const Mission m = parse("mission \"tour\"\n  anchor start\n  repeat 3\n    cross door_1 retry=2\n    wait s=10s\n"
                            "  explore room_2 budget_s=30s\n");

    // ── rows: pre-order, one per step, its own DSL line, indented by depth ──────────────────────────────────────
    {
        const Model md = Model::from(m);
        CHECK(md.title == "tour");
        CHECK(md.rows.size() == 5);
        CHECK(md.rows.at(0).id == "1" and md.rows.at(0).depth == 0 and md.rows.at(0).line == "anchor start");
        CHECK(md.rows.at(1).id == "2" and md.rows.at(1).composite and md.rows.at(1).line == "repeat 3");
        CHECK(md.rows.at(2).id == "2.1" and md.rows.at(2).depth == 1 and md.rows.at(2).line == "cross door_1 retry=2");
        CHECK(md.rows.at(3).line == "wait s=10s" and md.rows.at(3).bound_s == 10.0);
        CHECK(md.rows.at(4).bound_s == 30.0 and not md.rows.at(2).bound_s.has_value());
        CHECK(md.rows.at(2).attempts == 3 and md.rows.at(1).times == 3);
        CHECK(md.headline(0) == "not started");
        for (const auto& r : md.rows) CHECK(r.state == SS::Pending and md.annotation(r, 0).empty());
    }

    // ── apply(): states by id; it returns the rows whose state changed (the ones that flash) ───────────────────
    {
        Model md = Model::from(m);
        Progress p{.serial = 4, .source = "chat", .title = "tour", .status = "in_progress", .t0 = 1000, .t1 = 0,
                   .line = "2.1:cross door_1|running (attempt 2/3, iteration 2/3)|…",
                   .steps = {sv("1", SS::Satisfied, 1, 1000, 1000), sv("2", SS::Running, 1, 1000, 0, "", 1),
                             sv("2.1", SS::Running, 2, 2000), sv("2.2", SS::Pending, 0), sv("3", SS::Pending, 0)}};
        auto changed = md.apply(p);
        CHECK((changed == std::vector<std::size_t>{0, 1, 2}));
        CHECK(md.apply(p).empty());   // the same snapshot again: nothing changed
        CHECK(md.serial == 4 and md.source == "chat" and md.line == p.line);
        CHECK(md.headline(61'000) == "in progress · 1:00");
        CHECK(md.annotation(md.rows.at(1), 61'000) == "iteration 2/3");
        CHECK(md.annotation(md.rows.at(2), 5'000) == "attempt 2/3 · 3.0 s");   // elapsed from the step's start
        CHECK(md.running_row() == 2);

        // a bounded leaf reports how far through its bound it is
        p.steps.at(2) = sv("2.1", SS::Satisfied, 2, 2000, 6000);
        p.steps.at(3) = sv("2.2", SS::Running, 1, 6000);
        md.apply(p);
        CHECK(md.annotation(md.rows.at(3), 8'500) == "2.5 / 10 s");
        CHECK(md.fraction(md.rows.at(3), 8'500) == 0.25);
        CHECK(md.fraction(md.rows.at(3), 60'000) == 1.0);   // clamped: the controller ends it, not the view
        CHECK(not md.fraction(md.rows.at(2), 8'500).has_value());   // finished
        CHECK(md.annotation(md.rows.at(2), 8'500) == "4.0 s");
        CHECK(md.running_row() == 3);
    }

    // ── finished states: the outcome word is what the row says ──────────────────────────────────────────────────
    {
        Model md = Model::from(m);
        Progress p{.serial = 5, .source = "remote", .title = "tour", .status = "failed|2.1|unreachable", .t0 = 0,
                   .t1 = 9000, .line = "",
                   .steps = {sv("1", SS::Satisfied, 1, 0, 0), sv("2", SS::Failed, 1, 0, 9000, "unreachable", 0),
                             sv("2.1", SS::Failed, 3, 0, 9000, "unreachable"), sv("2.2", SS::Pending, 0),
                             sv("3", SS::Skipped, 1, 9000, 9000, "door_wont_open")}};
        md.apply(p);
        CHECK(md.headline(20'000) == "failed at 2.1: unreachable");
        CHECK(md.annotation(md.rows.at(2), 20'000) == "unreachable after 3 attempts");
        CHECK(md.annotation(md.rows.at(4), 20'000) == "skipped: door_wont_open");
        CHECK(md.running_row() == -1);

        p.status = "completed";
        CHECK((md.apply(p), md.headline(20'000)) == "completed in 0:09");
        p.status = "aborted|stopped_by_user";
        p.steps.at(2) = sv("2.1", SS::Cancelled, 1, 0, 9000, "stopped_by_user");
        md.apply(p);
        CHECK(md.headline(20'000) == "aborted: stopped_by_user");
        CHECK(md.annotation(md.rows.at(2), 20'000) == "cancelled: stopped_by_user");

        // an until / budget ending says so
        p.steps.at(4) = sv("3", SS::Satisfied, 1, 0, 30'000, "budget");
        md.apply(p);
        CHECK(md.annotation(md.rows.at(4), 0) == "30.0 s · budget");
    }

    // ── a snapshot for another mission's steps leaves unknown ids alone ─────────────────────────────────────────
    {
        Model md = Model::from(m);
        Progress p;
        p.status = "in_progress";
        p.steps = {sv("9.9", SS::Running)};
        CHECK(md.apply(p).empty());
    }

    std::println("test_plan_model: {} checks, {} failures", checks, failures);
    return failures == 0 ? 0 : 1;
}
