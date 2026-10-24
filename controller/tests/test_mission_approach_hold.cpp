// ApproachHold (src/mission_approach_hold.h): a hand-back of the approach holds; fails only on budget or withdrawal.
//   g++ -std=c++23 -O0 -g -o /tmp/tmah test_mission_approach_hold.cpp && /tmp/tmah
#include "../src/mission_approach_hold.h"

#include <print>

namespace
{
int failures = 0;
void check(bool ok, int line, const char *what) { if (not ok) { ++failures; std::println("FAIL line {}: {}", line, what); } }
#define CHECK(x) check((x), __LINE__, #x)
using V = rc::ApproachHold::Verdict;
}

int main()
{
    rc::ApproachHold h;
    CHECK(h.check(true, 0, true, false, 1000) == V::NotHeld);               // no hand-back yet
    CHECK(h.check(false, 1, true, false, 1000) == V::NotHeld);              // another step failing is not ours
    CHECK(h.check(true, 1, true, false, 1000) == V::Waiting);               // hand-back: hold, not fail
    CHECK(h.check(true, 1, true, false, 30'000) == V::Waiting);             // still within budget
    CHECK(h.check(true, 0, true, false, 31'000) == V::NotHeld);             // re-offered and the attempt counter cleared: released
    CHECK(h.check(true, 1, true, false, 40'000) == V::Waiting);             // a later hand-back restarts the budget
    CHECK(h.check(true, 1, true, false, 40'000 + rc::ApproachHold::kHoldBudgetMs) == V::Failed);   // budget ran out
    CHECK(h.check(true, 1, true, true, 200'000) == V::Waiting);       // re-claimed and driving: not a budget case
    CHECK(h.check(true, 1, true, false, 200'000 + 59'000) == V::Waiting);   // new hand-back: fresh budget
    {   // R4: the node absent for LESS than the grace (delete / re-create across a re-site) still waits; continuously absent fails
        rc::ApproachHold g;
        CHECK(g.check(true, 1, true,  false, 1'000) == V::Waiting);
        CHECK(g.check(true, 1, false, false, 2'000) == V::Waiting);                                   // flap starts
        CHECK(g.check(true, 1, false, false, 2'000 + rc::ApproachHold::kWithdrawGraceMs - 1) == V::Waiting);
        CHECK(g.check(true, 1, true,  false, 2'000 + rc::ApproachHold::kWithdrawGraceMs - 1) == V::Waiting);   // re-offered: resets
        CHECK(g.check(true, 1, false, false, 10'000) == V::Waiting);
        CHECK(g.check(true, 1, false, false, 10'000 + rc::ApproachHold::kWithdrawGraceMs) == V::Failed);       // withdrawn
    }
    {   // R5: the step's own budget_s (smaller than the default) bounds the hold; a larger one does not extend it
        rc::ApproachHold g;
        CHECK(g.check(true, 1, true, false, 1'000, 10'000) == V::Waiting);
        CHECK(g.check(true, 1, true, false, 11'000, 10'000) == V::Failed);
        CHECK(g.check(true, 1, true, false, 20'000, 500'000) == V::Waiting);
        CHECK(g.check(true, 1, true, false, 20'000 + rc::ApproachHold::kHoldBudgetMs, 500'000) == V::Failed);
    }
    std::println("{}", failures == 0 ? "ALL PASS" : "FAILURES");
    return failures == 0 ? 0 : 1;
}
