// test_door_decision.cpp — Task 6: costed existence decision, door role, running-median cost estimator.
// Build + run (from door_concept/):
//   g++ -std=c++23 -O1 -w -I../common tests/test_door_decision.cpp -o /tmp/test_door_decision && /tmp/test_door_decision
#include <clocale>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <format>
#include <unistd.h>
#include "../src/core/door_decision.h"
using namespace rc::door::core;
static int fails = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static std::filesystem::path tmp_path(const char* tag)
{
    return std::filesystem::temp_directory_path() / std::format("test_door_decision_{}_{}.csv", tag, ::getpid());
}

static void route_door_survives()
{
    const Costs c{30.f, 120.f, 420.f};
    const Verdict o = decide(-2.0f, Role::Ordinary, c);
    CHECK(std::fabs(o.delete_line - (-std::log(120.f / 30.f))) < 1e-5f);   // -1.386
    CHECK(o.remove and not o.exists);
    const Verdict r = decide(-2.0f, Role::Route, c);
    CHECK(std::fabs(r.delete_line - (-std::log(420.f / 30.f))) < 1e-5f);   // -2.639
    CHECK(not r.remove and not r.exists);
}

static void being_crossed_never_removed()
{
    const Costs c{30.f, 120.f, 420.f};
    CHECK(not decide(-50.f, Role::BeingCrossed, c).remove);
    CHECK(decide(-50.f, Role::Ordinary, c).remove);
}

static void exists_at_zero()
{
    const Costs c{30.f, 120.f, 420.f};
    const Verdict p = decide(1e-4f, Role::Ordinary, c);
    CHECK(p.exists and not p.remove);
    const Verdict n = decide(-1e-4f, Role::Ordinary, c);
    CHECK(not n.exists and not n.remove);   // between the lines: neither created nor deleted
    CHECK(not decide(0.f, Role::Ordinary, c).exists);   // exists is L > 0, strictly
}

static void delete_line_above_zero_removes()
{
    const Costs c{100.f, 30.f, 330.f};   // line = -ln(30/100) = +1.2
    const Verdict o = decide(0.5f, Role::Ordinary, c);
    CHECK(o.remove and not o.exists);
    const Verdict b = decide(0.5f, Role::BeingCrossed, c);
    CHECK(not b.remove and b.exists);
}

static void median_and_persist()
{
    std::setlocale(LC_ALL, "");   // the agents run under es_ES; a comma-decimal parse must not corrupt the round trip
    CostEstimator e;
    for (float s : {50.f, 10.f, 30.f}) e.add(CostEstimator::Kind::Keep, s, "t:keep");
    for (float s : {0.25f, 100.5f, 7.125f, 9.0f}) e.add(CostEstimator::Kind::DeleteOrdinary, s, "t:del");
    CHECK(e.n(CostEstimator::Kind::Keep) == 3 and e.n(CostEstimator::Kind::DeleteOrdinary) == 4);
    Costs c = e.costs(300.f);
    CHECK(c.keep_s == 30.f);
    CHECK(std::fabs(c.delete_ordinary_s - 8.0625f) < 1e-5f);   // median of {0.25, 7.125, 9, 100.5}
    CHECK(std::fabs(c.delete_route_s - (8.0625f + 300.f)) < 1e-4f);
    const auto p = tmp_path("rt");
    e.save(p);
    CostEstimator g = CostEstimator::load(p);
    CHECK(g.n(CostEstimator::Kind::Keep) == 3 and g.n(CostEstimator::Kind::DeleteOrdinary) == 4);
    const Costs d = g.costs(300.f);
    CHECK(d.keep_s == c.keep_s and d.delete_ordinary_s == c.delete_ordinary_s and d.delete_route_s == c.delete_route_s);
    std::filesystem::remove(p);
}

static void malformed_rows_counted()
{
    const auto p = tmp_path("bad");
    { std::ofstream f(p); f << "kind,seconds,source\nkeep,12.5,a\nkeep,12.5x,b\nkeep,,c\nbogus,3,d\ndelete_ordinary,4,e\nkeep\n"; }
    CostEstimator g = CostEstimator::load(p);
    CHECK(g.n(CostEstimator::Kind::Keep) == 1 and g.n(CostEstimator::Kind::DeleteOrdinary) == 1);
    CHECK(g.malformed() == 4);
    CHECK(CostEstimator::load("/nonexistent/door_costs.csv").n(CostEstimator::Kind::Keep) == 0);
    std::filesystem::remove(p);
}

static void empty_estimator_has_no_costs_to_lie_about()
{
    const CostEstimator e;   // no samples: the median of nothing is the spec's reading, 30 s, not 0 (0 would divide by zero)
    const Costs c = e.costs(300.f);
    CHECK(c.keep_s == 30.f and c.delete_ordinary_s == 30.f);
    CHECK(decide(-1.f, Role::Ordinary, c).delete_line == 0.f);   // equal costs: delete when L < 0
}

static void role_from_route_faces()
{
    const std::vector<std::string> faces{"door_3", "door_7"};
    CHECK(door_role("door_3", faces, false) == Role::Route);
    CHECK(door_role("door_4", faces, false) == Role::Ordinary);
    CHECK(door_role("door_4", faces, true) == Role::BeingCrossed);
    CHECK(door_role("door_3", faces, true) == Role::BeingCrossed);   // crossing outranks the route
    CHECK(door_role("door_3", {}, false) == Role::Ordinary);
}

// A missing (or empty) etc/door_costs.csv falls back to the compiled LOGGED estimate (loud on stderr), never to equal costs.
static void missing_file_gives_logged_estimate()
{
    const auto e = CostEstimator::load_or_default(tmp_path("does_not_exist"));
    const Costs c = e.costs(300.f);
    CHECK(c.delete_ordinary_s > c.keep_s);
    CHECK(c.keep_s == 30.0f and c.delete_ordinary_s == 88.25f);
    CHECK(decide(0.0f, Role::Ordinary, c).delete_line < 0.0f);              // a door at L = 0 is kept
    const Costs d = CostEstimator{}.costs(300.f);                           // the empty estimator is the spec's equal reading (line 0) ...
    CHECK(d.delete_ordinary_s == d.keep_s);                                  // ... which is exactly why it is not a fallback
    CHECK(CostEstimator::logged_default().n(CostEstimator::Kind::DeleteOrdinary) == 1);
    // a real file wins
    const auto p = tmp_path("real");
    { CostEstimator f; f.add(CostEstimator::Kind::Keep, 12.f, "t"); f.add(CostEstimator::Kind::DeleteOrdinary, 40.f, "t"); f.save(p); }
    CHECK(CostEstimator::load_or_default(p).costs(300.f).delete_ordinary_s == 40.f);
    std::filesystem::remove(p);
}

// re-review N3 (ruling): a file with C_delete <= C_keep, or lacking a kind, is INVALID: the logged estimate, never its costs
static void invalid_file_gives_logged_estimate()
{
    const auto p = tmp_path("invalid");
    const auto with = [&](float keep, float del, bool has_del = true)
    {
        CostEstimator f; f.add(CostEstimator::Kind::Keep, keep, "t");
        if (has_del) f.add(CostEstimator::Kind::DeleteOrdinary, del, "t");
        f.save(p);
        return CostEstimator::load_or_default(p).costs(300.f);
    };
    for (const Costs c : {with(30.f, 30.f), with(30.f, 10.f), with(30.f, 0.f, false)})   // equal, inverted, no delete samples
        CHECK(c.keep_s == 30.0f and c.delete_ordinary_s == 88.25f);
    CHECK(with(30.f, 30.5f).delete_ordinary_s == 30.5f);                                   // deletion dearer, by any margin: valid
    std::filesystem::remove(p);
}

int main()
{
    missing_file_gives_logged_estimate();
    invalid_file_gives_logged_estimate();
    route_door_survives();
    being_crossed_never_removed();
    exists_at_zero();
    delete_line_above_zero_removes();
    median_and_persist();
    malformed_rows_counted();
    empty_estimator_has_no_costs_to_lie_about();
    role_from_route_faces();
    std::printf(fails == 0 ? "ALL PASS\n" : "%d FAILED\n", fails);
    return fails == 0 ? 0 : 1;
}
