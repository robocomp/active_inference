// ─────────────────────────────────────────────────────────────────────────────────────────────────
// EVERY SITUATION IN THE ROOM↔CONTROLLER PROTOCOL, AS A TEST — with ten producers.
//
// Run:  g++ -std=c++23 -O2 -o protocol_multi_bench protocol_multi_bench.cpp && ./protocol_multi_bench
//
// The situations are the ones enumerated in "The Standpoint Handshake": a cell that is standable and
// routable, one blocked at its centre but free inside the producer's cell, one nothing fits in, one
// with no route but a closer pose, one sealed in a pocket, one whose detection never arrives, one
// that wedges, a consumer that dies mid-claim, a producer that will not re-arm an unchanged cell,
// and ten agents competing for one body.
//
// Each scenario asserts the same four properties, because a protocol that satisfies three of them is
// still broken in a way that took a full day to find on the robot:
//
//   NO PHANTOM   observed ⊆ stood_at. A producer is never told about a standpoint the body never
//                occupied. (140 of 163 live arrivals violated this, at a median of 2.86 m.)
//   PROGRESS     observations still happening in the LAST THIRD of the run — not "at least one",
//                which one arrival followed by an hour of silence satisfies.
//   FAIRNESS     every producer holding a servable cell is served at least once. Unstatable with
//                one producer, and the property that scales worst to ten.
//   NO FREE CHURN  no long run of completions that cost the robot no motion. This is the 104/min
//                  oscillator — and the 9677 facts in 20 minutes measured live on 2026-08-19. A run
//                  of honest failures that each cost a drive is NOT this, and must not be flagged.
// ─────────────────────────────────────────────────────────────────────────────────────────────────
#include "protocol_multi.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace rc::affordance::multi;

namespace
{

struct Check
{
    std::string name;
    bool        passed = true;
    std::string why;
};

int failures = 0;

void report(const std::string &scenario, const Result &r, const std::vector<Check> &checks)
{
    bool ok = true;
    for (const auto &c : checks) ok = ok and c.passed;
    if (not ok) ++failures;

    std::printf("%-4s %-46s obs=%-4d compl=%-4d last3rd=%-3d barren=%-3d%s\n",
                ok ? "PASS" : "FAIL", scenario.c_str(), r.observations, r.completions,
                r.observations_last_third, r.longest_free_run,
                r.deadlocked ? "  DEADLOCKED" : "");
    for (const auto &c : checks)
        if (not c.passed) std::printf("       ✗ %s: %s\n", c.name.c_str(), c.why.c_str());
}

// The four properties, asked the same way of every scenario.
std::vector<Check> standard_checks(const Result &r, bool expect_progress = true,
                                   bool expect_fairness = true)
{
    std::vector<Check> v;
    v.push_back({"NO PHANTOM", not r.phantom,
                 r.phantom ? std::to_string(r.phantom_count) + " cells reported observed that the "
                             "robot never stood at" : ""});
    v.push_back({"BELIEF INTEGRITY", r.believed_unvisited == 0,
                 std::to_string(r.believed_unvisited) + " cells a producer believes explored were "
                 "never stood at — a belief formed from a protocol event, not from sensing"});
    if (expect_progress)
        v.push_back({"PROGRESS", r.observations_last_third > 0,
                     "no observation in the last third of the run — the pair went quiet"});
    if (expect_fairness)
    {
        std::string starved;
        for (const auto &s : r.starved) { if (not starved.empty()) starved += ", "; starved += s; }
        v.push_back({"FAIRNESS", r.starved.empty(),
                     starved.empty() ? "" : "never served: " + starved});
    }
    v.push_back({"NO DEADLOCK", not r.deadlocked, "the wire stopped changing entirely"});
    v.push_back({"NO RE-DECIDING", r.redecided == 0,
                 std::to_string(r.redecided) + " map-only verdicts re-decided from a pose that had "
                 "already produced them — each one is a completion that cost the robot nothing"});
    return v;
}

void report_must_break(const std::string &scenario, const Result &r, bool broke)
{
    if (not broke) ++failures;
    std::printf("%-4s %-46s obs=%-4d compl=%-4d %s\n", broke ? "PASS" : "FAIL", scenario.c_str(),
                r.observations, r.completions,
                broke ? "(broke, as it must)" : "★ THE DEFECT NO LONGER REPRODUCES — the model has "
                                                "stopped modelling it");
}

Cell mk(int id, float x, Standable s = Standable::Yes, Routable rt = Routable::Yes,
        bool detects = true, bool wedges = false)
{
    return Cell{.id = id, .x = x, .standable = s, .routable = rt, .detects = detects, .wedges = wedges};
}

// Ten agents, three cells each, spread over a 30-unit world — the fleet this has to survive.
struct Fleet
{
    World world;
    std::vector<Producer> producers;
};

Fleet make_fleet(ProducerPolicy pp, const std::vector<std::string> &names)
{
    Fleet f;
    int id = 0;
    for (std::size_t a = 0; a < names.size(); ++a)
    {
        std::vector<int> mine;
        for (int k = 0; k < 3; ++k)
        {
            const float x = static_cast<float>(a) * 3.f + static_cast<float>(k);
            f.world.cells.push_back(mk(id, x));
            mine.push_back(id);
            ++id;
        }
        f.producers.emplace_back(names[a], pp, mine);
    }
    return f;
}

const std::vector<std::string> kTen = {"room", "table", "chair", "bottle", "cabinet",
                                       "fridge", "hood", "door", "kitchen", "dining"};

}  // namespace

int main()
{
    std::printf("── the situation space, with ten producers ──\n\n");
    constexpr std::uint64_t kCycles = 4000;

    // ── 1. BASELINE: ten agents, everything servable ────────────────────────────────────────────
    {
        auto f = make_fleet(ProducerPolicy{}, kTen);
        Consumer cons{ConsumerPolicy{}};
        auto r = run(f.producers, cons, f.world, kCycles);
        report("1  ten agents, everything servable", r, standard_checks(r));
    }

    // ── 2. A CELL BLOCKED AT ITS CENTRE, free inside the producer's cell ────────────────────────
    {
        auto f = make_fleet(ProducerPolicy{}, kTen);
        for (auto &c : f.world.cells) if (c.id % 7 == 0) c.standable = Standable::OffCentre;
        Consumer cons{ConsumerPolicy{}};
        auto r = run(f.producers, cons, f.world, kCycles);
        report("2  centres blocked, cells still admit the body", r, standard_checks(r));
    }

    // ── 3. CELLS NOTHING FITS IN ────────────────────────────────────────────────────────────────
    {
        auto f = make_fleet(ProducerPolicy{}, kTen);
        for (auto &c : f.world.cells) if (c.id % 3 == 1) c.standable = Standable::Never;
        Consumer cons{ConsumerPolicy{}};
        auto r = run(f.producers, cons, f.world, kCycles);
        report("3  a third of all cells are infeasible", r, standard_checks(r));
    }

    // ── 3b. CELLS THE PRODUCER PLACED OUTSIDE ITS OWN ROOM ──────────────────────────────────────
    {
        auto f = make_fleet(ProducerPolicy{}, kTen);
        for (auto &c : f.world.cells) if (c.id % 8 == 5) c.outside_room = true;
        Consumer cons{ConsumerPolicy{}};
        auto r = run(f.producers, cons, f.world, kCycles);
        report("3b cells outside the room layout", r, standard_checks(r));
    }

    // ── 4. NO ROUTE, BUT A CLOSER POSE EXISTS → approach, report unreachable ────────────────────
    {
        auto f = make_fleet(ProducerPolicy{}, kTen);
        for (auto &c : f.world.cells) if (c.id % 4 == 2) c.routable = Routable::CloserExists;
        Consumer cons{ConsumerPolicy{}};
        auto r = run(f.producers, cons, f.world, kCycles);
        report("4  unroutable cells, approached honestly", r, standard_checks(r));
    }

    // ── 5. THE ROBOT IS SEALED IN A POCKET for part of the world ────────────────────────────────
    {
        auto f = make_fleet(ProducerPolicy{}, kTen);
        for (auto &c : f.world.cells) if (c.x > 15.f) c.routable = Routable::Sealed;
        Consumer cons{ConsumerPolicy{}};
        auto r = run(f.producers, cons, f.world, kCycles);
        // Half the fleet is behind the seal, so fairness cannot hold — the run must still be honest
        // about the half it can serve, and must never book an observation for the half it cannot.
        auto checks = standard_checks(r, /*progress=*/true, /*fairness=*/false);
        report("5  half the world sealed behind a pocket", r, checks);
    }

    // ── 6. THE ACQUISITION NEVER ARRIVES → bounded dwell, then Timeout ──────────────────────────
    {
        auto f = make_fleet(ProducerPolicy{}, kTen);
        for (auto &c : f.world.cells) if (c.id % 5 == 3) c.detects = false;
        Consumer cons{ConsumerPolicy{}};
        auto r = run(f.producers, cons, f.world, kCycles);
        report("6  detections that never come", r, standard_checks(r));
    }

    // ── 7. THE DRIVE WEDGES on some approaches ──────────────────────────────────────────────────
    {
        auto f = make_fleet(ProducerPolicy{}, kTen);
        for (auto &c : f.world.cells) if (c.id % 6 == 4) c.wedges = true;
        Consumer cons{ConsumerPolicy{}};
        auto r = run(f.producers, cons, f.world, kCycles);
        report("7  some approaches wedge", r, standard_checks(r));
    }

    // ── 8. THE CONSUMER DIES MID-CLAIM ──────────────────────────────────────────────────────────
    {
        auto f = make_fleet(ProducerPolicy{}, kTen);
        ConsumerPolicy cp; cp.die_after_claims = 5;
        Consumer cons{cp};
        auto r = run(f.producers, cons, f.world, kCycles);
        // Nothing can be observed after it dies; what MUST hold is that no producer is left holding
        // a claim for ever, and that nothing was booked that did not happen.
        std::vector<Check> checks;
        checks.push_back({"NO PHANTOM", not r.phantom, "booked an observation that never happened"});
        checks.push_back({"LEASE RECLAIMS", not r.deadlocked,
                          "the wire stopped changing: a dead consumer parked a node for ever"});
        report("8  consumer dies after five claims", r, checks);
    }

    // ── 9. THE PRODUCER WILL NOT RE-ARM AN UNCHANGED CELL ───────────────────────────────────────
    {
        ProducerPolicy pp; pp.rearm_unchanged = false;
        auto f = make_fleet(pp, kTen);
        Consumer cons{ConsumerPolicy{}};
        auto r = run(f.producers, cons, f.world, kCycles);
        report("9  rearm_unchanged = false", r, standard_checks(r));
    }

    // ── 10. BELIEF UPDATED FROM PROTOCOL EVENTS (rule-5 violation) ──────────────────────────────
    {
        ProducerPolicy pp; pp.belief_from_protocol_events = true;
        auto f = make_fleet(pp, kTen);
        for (auto &c : f.world.cells) if (c.id % 3 == 1) c.routable = Routable::Sealed;
        Consumer cons{ConsumerPolicy{}};
        auto r = run(f.producers, cons, f.world, kCycles);
        // ★A REGRESSION GUARD, NOT A BUG. This scenario turns rule 5 OFF, so it must break — if it
        // ever passes, the model has stopped modelling the failure and the other scenarios' passes
        // mean nothing. Four instruments reported success while producing nothing on 2026-08-19;
        // this is the check that a green board is green for a reason.
        report_must_break("10 belief from protocol events (must break)", r,
                          r.believed_unvisited > 0 or r.deadlocked or r.phantom);
    }

    // ── 11. THE PHANTOM: substitute the standpoint and claim satisfied ──────────────────────────
    {
        auto f = make_fleet(ProducerPolicy{}, kTen);
        for (auto &c : f.world.cells) if (c.x > 12.f) c.routable = Routable::Sealed;
        ConsumerPolicy cp; cp.substitute_and_claim_satisfied = true;
        Consumer cons{cp};
        auto r = run(f.producers, cons, f.world, kCycles);
        report_must_break("11 substitution as satisfied (must break)", r, r.phantom);
    }

    // ── 12. THE SILENT LOCAL VETO ───────────────────────────────────────────────────────────────
    {
        auto f = make_fleet(ProducerPolicy{}, kTen);
        for (auto &c : f.world.cells) if (c.id % 4 == 2) c.wedges = true;
        ConsumerPolicy cp; cp.silent_local_veto = true;
        Consumer cons{cp};
        auto r = run(f.producers, cons, f.world, kCycles);
        report("12 silent 120 s local veto", r, standard_checks(r));
    }

    // ── 12b. THE APPROACH THAT COVERS NO GROUND ─────────────────────────────────────────────────
    {
        auto f = make_fleet(ProducerPolicy{}, kTen);
        for (auto &c : f.world.cells) if (c.id % 3 == 1) c.routable = Routable::CloserWithinBand;
        Consumer cons{ConsumerPolicy{}};
        auto r = run(f.producers, cons, f.world, kCycles);
        report("12b approach pose inside the arrival band", r, standard_checks(r));
    }

    // ── 13. EVERYTHING AT ONCE — the world the fleet actually runs in ───────────────────────────
    {
        auto f = make_fleet(ProducerPolicy{}, kTen);
        for (auto &c : f.world.cells)
        {
            if (c.id % 7 == 0) c.standable = Standable::OffCentre;
            if (c.id % 11 == 3) c.standable = Standable::Never;
            if (c.id % 5 == 2) c.routable = Routable::CloserExists;
            if (c.id % 13 == 6) c.routable = Routable::Sealed;
            if (c.id % 19 == 7) c.routable = Routable::CloserWithinBand;
            if (c.id % 9 == 5) c.detects = false;
            if (c.id % 17 == 8) c.wedges = true;
        }
        Consumer cons{ConsumerPolicy{}};
        auto r = run(f.producers, cons, f.world, kCycles);
        report("13 all situations mixed, ten agents", r, standard_checks(r));
    }

    std::printf("\n%d scenario(s) FAILED\n", failures);
    return failures == 0 ? 0 : 1;
}
