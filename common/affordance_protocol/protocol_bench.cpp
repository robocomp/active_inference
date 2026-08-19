// ─────────────────────────────────────────────────────────────────────────────────────────────────
// PROTOCOL BENCH — every affordance failure of 2026-08-19, as a test that runs in microseconds.
//
// Each of these cost a ~20 minute robot run to diagnose, and four of the fixes were wrong and had to
// be reverted. All of them are decidable in a few hundred simulated cycles.
//
// ★THE PROPERTY IS REPEATED OBSERVATION, NOT "AT LEAST ONE". The live symptom at 17:00 was an
// affordance viewer reading "reached, 0.20 m to go of 8.44" — a perfect 8.4 m drive — and then
// nothing, for an hour. A system that observes once and stops satisfies every weaker property while
// being completely broken, which is precisely how "held 10 min, looking good" kept being wrong today.
//
// build: g++ -std=c++23 -O2 protocol_bench.cpp -o protocol_bench && ./protocol_bench
// ─────────────────────────────────────────────────────────────────────────────────────────────────
#include "protocol_core.h"
#include <print>
#include <string>
#include <vector>

using namespace rc::affordance::core;

namespace
{
int failures = 0;

struct Case
{
    std::string name;
    ProducerPolicy prod;
    ConsumerPolicy cons;
    std::set<int> reachable;
    int want_observations;      // minimum, over the run
    std::string why;            // what this case is really about
};

void check(const Case &c)
{
    Producer prod(c.prod, {0, 1, 2});
    Consumer cons(c.cons);
    World world{.reachable = c.reachable, .robot_at = -1};
    const auto r = run(prod, cons, world, 4000);
    const bool ok = r.observations >= c.want_observations and not r.deadlocked;
    if (not ok) ++failures;
    std::println("{}  {:<46}  obs={:<4} compl={:<5} {}{}",
                 ok ? "PASS" : "FAIL", c.name, r.observations, r.completions,
                 r.deadlocked ? "DEADLOCKED " : "",
                 ok ? "" : "  <-- " + c.why);
}
}  // namespace

int main()
{
    std::println("── protocol bench: the day's failures, as tests ──\n");

    // Cell 0 is the one the argmax favours and is NOT standable — the fridge-behind-the-wall case.
    const std::set<int> good{1, 2};

    check({.name = "baseline: everything reachable",
           .prod = {}, .cons = {}, .reachable = {0, 1, 2}, .want_observations = 20,
           .why = "the pair cannot even work when nothing is wrong"});

    check({.name = "favoured cell unreachable",
           .prod = {}, .cons = {}, .reachable = good, .want_observations = 20,
           .why = "one bad standpoint must not stop exploration"});

    // ── 15:00 live: producer declines to re-arm, consumer rejects anything not Offered ────────────
    check({.name = "rearm_unchanged=false + stuck argmax",
           .prod = {.rearm_unchanged = false, .refusal_moves_argmax = false},
           .cons = {}, .reachable = good, .want_observations = 20,
           .why = "DEADLOCK: 'unchanged is not news' + an argmax that never moves. "
                  "AffordanceSelection.tla flags this in all 8 combinations"});

    // ── the ~104 completions/min busy loop ───────────────────────────────────────────────────────
    check({.name = "no refusal hold (busy loop)",
           .prod = {}, .cons = {.refusal_hold_cycles = 0}, .reachable = good, .want_observations = 20,
           .why = "a refusal retried instantly cannot end differently; the pair spins"});

    // ── 9b3e5a4: an early arrival reported as a refusal ──────────────────────────────────────────
    check({.name = "instant arrival reported as REFUSAL",
           .prod = {}, .cons = {.instant_arrival_is_refusal = true}, .reachable = {0, 1, 2},
           .want_observations = 20,
           .why = "'already here' is not a statement about the CELL; reporting it to the producer "
                  "livelocks the pair"});

    // ── the 100.6 s and 191.6 s target gaps ──────────────────────────────────────────────────────
    check({.name = "offer never expires + consumer holds it",
           .prod = {.offer_timeout_cycles = 0},
           .cons = {.refusal_hold_cycles = 500}, .reachable = good, .want_observations = 20,
           .why = "producer waits on a consumer that already declined; nothing bounds the wait"});

    // ── the argmax that de-prioritisation cannot move ────────────────────────────────────────────
    check({.name = "refusal does not move the argmax",
           .prod = {.refusal_moves_argmax = false}, .cons = {}, .reachable = good,
           .want_observations = 20,
           .why = "gain fell 0.171->0.0004 and the cell still won: score is not dominated by gain"});

    // ── THE FAILURE MEASURED AT 19:37 AND CONFIRMED BY TLA+ AND BY INDEPENDENT REVIEW ────────────
    check({.name = "consumer claims then STOPS (no lease)",
           .prod = {.execution_lease_cycles = 0},
           .cons = {.die_after_claims = 3}, .reachable = good, .want_observations = 2,
           .why = "producer has no escape from a claim the consumer abandoned: node sits Executing "
                  "for ever. 12% of live records were exactly this"});

    check({.name = "consumer claims then STOPS (lease ON)",
           .prod = {}, .cons = {.die_after_claims = 3}, .reachable = good, .want_observations = 2,
           .why = "the lease must let the producer reclaim unilaterally"});

    // ── RULE 5: the belief loop must pass through sensing ─────────────────────────────────────────
    check({.name = "belief from protocol events + stuck argmax",
           .prod = {.belief_from_protocol_events = true, .refusal_moves_argmax = false},
           .cons = {}, .reachable = good, .want_observations = 20,
           .why = "a protocol event wired into the value function is the 104/min oscillator"});

    std::println("\n{}", failures == 0 ? "all scenarios PASS"
                                       : std::format("{} scenario(s) FAILED", failures));
    return failures == 0 ? 0 : 1;
}
