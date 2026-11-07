// test_carried_door.cpp — Task 9 of the door-existence model: a crossed door is born in the new room carrying its evidence plus
// the crossing, and a restore reads the stored belief (aged), not a seed. DSR-free.
// Build + run (from door_concept/):
//   g++ -std=c++23 -O1 -w -I/usr/include/eigen3 -I/usr/include/opencv4 -I../common tests/test_carried_door.cpp -o /tmp/test_carried_door && /tmp/test_carried_door
#include <clocale>
#include <cmath>
#include <print>
#include "../src/core/place_table.h"
#include "../src/core/door_decision.h"
#include "../src/door_leaf_visibility.h"
using namespace rc::door::core;
static int fails = 0;
#define CHECK(c) do { if (not (c)) { ++fails; std::print("FAIL {}:{}  {}\n", __FILE__, __LINE__, #c); } } while (0)
static bool near(float a, float b, float e = 1e-5f) { return std::fabs(a - b) <= e; }

int main()
{
    std::setlocale(LC_ALL, "");
    const float x = rc::door::crossing_logodds(0.5f, 0.05f);   // the compiled Existence.Cross* defaults: ln(10) = 2.30

    {   // carried_door_above_creation
        PlaceRow src; src.L_evidence = 1.5f;
        const PlaceBelief b = carried_belief(&src, x);
        std::print("carried: L {:.3f} = carried {:.2f} + crossing {:.3f}\n", b.L(), b.carried, b.crossing);
        CHECK(near(b.L(), 1.5f + x));
        CHECK(b.L() > 0.f);
        CHECK(b.prior.total() == 0.f and b.ledger.views().empty() and b.carried == 1.5f and b.crossing == x);
        // the source's ledger is NOT copied: views are per room
        PlaceRow withled = src; withled.ledger = ViewLedger{};
        CHECK(carried_belief(&withled, x).ledger.views().empty());
    }
    {   // a source that was believed against: the crossing still counts, the sum is what it is (no clamp)
        PlaceRow src; src.L_evidence = -1.0f;
        CHECK(near(carried_belief(&src, x).L(), -1.0f + x));
    }
    {   // carried_without_source: the new room's prior + the crossing
        PlacePrior pr; pr.band = -0.4f; pr.base_rate = std::log(0.4f / 0.6f); pr.size = -0.3f;
        const PlaceBelief b = carried_belief(nullptr, x, pr);
        CHECK(near(b.L(), pr.total() + x));
        CHECK(b.carried == 0.f and b.crossing == x and b.ledger.views().empty());
        CHECK(near(carried_belief(nullptr, x).L(), x));          // no prior known yet: the crossing alone
    }
    {   // the role keeps it: BeingCrossed is never removed, however the first views go
        const Costs c{30.f, 30.f, 630.f};
        const auto role = door_role("door_9", {}, true);
        CHECK(role == Role::BeingCrossed);
        PlaceRow src; src.L_evidence = -6.f;
        const auto v = decide(carried_belief(&src, x).L() - 50.f, role, c);
        CHECK(not v.remove);
        const auto v0 = decide(carried_belief(nullptr, x).L(), role, c);
        CHECK(v0.exists and not v0.remove);
    }
    {   // a restore reads the aged stored belief: ledger kept, evidence aged by the hazard, no seed
        PlaceTable T(false);
        PlaceRow r; r.L_evidence = 3.f; r.last_seen_wall_ms = 1'000'000;
        const auto fresh = T.restored_belief(r, 1'000'000);
        CHECK(near(fresh.L(), 3.f, 1e-5f));
        const auto aged = T.restored_belief(r, 1'000'000 + 600'000);
        CHECK(near(aged.L(), 3.f * std::exp(-1.f), 1e-4f));           // prior mean hazard 1/600 s
        T.note_revisit(false, 100.0);                                   // a revisit that found no door: the hazard rises
        CHECK(T.restored_belief(r, 1'000'000 + 600'000).L() < aged.L());
    }
    std::print("{}\n", fails == 0 ? "ALL PASS" : "FAILED");
    return fails != 0;
}
