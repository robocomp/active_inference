// test_door_restore_idonly.cpp — an identity-only remembered door under the one-model is restored with L_evidence = -prior (P = 0.5).
// (live regression 11-15, e18 door_21: restored at the new-door prior, 0.4 nat above its delete line, removed after a few low-q misses)
// Build + run (from door_concept/):
//   g++ -std=c++23 -O1 -w -I/usr/include/eigen3 -I/usr/include/opencv4 -I../common tests/test_door_restore_idonly.cpp -o /tmp/test_door_restore_idonly && /tmp/test_door_restore_idonly
#include <clocale>
#include <cmath>
#include <cstdio>
#include "../src/core/place_belief.h"
#include "../src/core/place_table.h"
#include "../src/core/door_decision.h"
using namespace rc::door::core;
using rc::door::IdentityRow;
static int fails = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)
#define CHECK_NEAR(a, b, tol) do { const double a_ = (a), b_ = (b); if (not (std::fabs(a_ - b_) <= (tol))) { std::printf("FAIL %s:%d  %s = %.5f vs %.5f (tol %.3f)\n", __FILE__, __LINE__, #a, a_, b_, double(tol)); ++fails; } } while (0)

static const DetectorModel M = [] { DetectorModel m = default_detectors().zed; m.hit_rate = 0.9f; m.false_alarm = 0.02f; return m; }();
static constexpr std::int64_t kNow = 1'000'000'000;
static PlacePrior new_door_prior()   // the e18 numbers: band -0.00, base -1.14, size -0.73, extent -0.28 (total -2.15)
{ PlacePrior p; p.band = 0.f; p.base_rate = -1.14f; p.size = -0.73f; p.extent = -0.28f; return p; }
static IdentityRow idrow() { IdentityRow r; r.room = "room_3"; r.name = "door_21"; r.xy = {3.44f, -1.86f}; return r; }
/// n views of the place at q = 0.05 (far, low detect probability): `hits` of them detected, the rest misses.
static void low_q_views(ViewLedger& L, int n, int hits)
{
    const Vec2f door(3.44f, -1.86f), nrm(0.f, 1.f);
    for (int i = 0; i < n; ++i)
    {
        const float a = 0.9f * float(i);
        const auto v = L.observe(door + 4.f * Vec2f(std::cos(a), std::sin(a)), door, nrm, 1.f, 0.f, i);
        L.add_bernoulli(v, Channel::Zed, 0.05f, M, i < hits ? 1.f : 0.f); L.recompute(v, false);
    }
}

int main()
{
    std::setlocale(LC_ALL, "");
    const auto prior = new_door_prior();
    {   // one-model, id-only, no age: L = 0, P = 0.5 once the prior is entered
        PlaceTable T(false, kLookalikeProbDefault, true);
        auto b = T.restored_belief(PlaceTable::row_of(idrow(), 0.1f, false, kLookalikeProbDefault, true), kNow);
        CHECK(b.identity_only_g.has_value());
        b.adopt_prior(prior);
        CHECK_NEAR(b.L(), 0.0, 1e-4);
        CHECK_NEAR(b.p(), 0.5, 1e-4);
        CHECK_NEAR(b.L_evidence(), 2.15, 1e-3);
        b.adopt_prior(prior);   // replaced, never accumulated: the second adoption changes nothing
        CHECK_NEAR(b.L(), 0.0, 1e-4);
        // the SAME holds when the prior later changes (the evidence is entered once; the prior is replaced)
        PlacePrior p2 = prior; p2.size = -0.2f; b.adopt_prior(p2);
        CHECK_NEAR(b.L(), 0.53, 1e-3);
    }
    {   // a row WITH a remembered L = +3: unchanged (evidence carried, prior added, aged)
        PlaceTable T(false, kLookalikeProbDefault, true);
        auto r = idrow(); r.exist_logodds = 3.f; r.exist_physical = 3.f; r.seen_wall_ms = kNow - 60'000;
        const auto row = PlaceTable::row_of(r, 0.1f, false, kLookalikeProbDefault, true);
        auto b = T.restored_belief(row, kNow);
        CHECK(not b.identity_only_g.has_value());
        b.adopt_prior(prior);
        CHECK_NEAR(b.L(), 3.0 * T.age_factor(row, kNow) + prior.total(), 1e-3);
    }
    {   // an id-only row WITH an age: the evidence -prior is aged as any remembered evidence (relaxes toward the prior)
        PlaceTable T(false, kLookalikeProbDefault, true);
        auto r = idrow(); r.seen_wall_ms = kNow - 600'000;
        const auto row = PlaceTable::row_of(r, 0.1f, false, kLookalikeProbDefault, true);
        auto b = T.restored_belief(row, kNow);
        b.adopt_prior(prior);
        const double g = T.age_factor(row, kNow);
        CHECK(g < 1.0 and g > 0.0);
        CHECK_NEAR(b.L(), (1.0 - g) * prior.total(), 1e-3);
    }
    {   // flag off: an id-only restore gives the same L as before (0, no prior entered, no hold)
        PlaceTable T(false);
        auto b = T.restored_belief(PlaceTable::row_of(idrow(), 0.1f, false), kNow);
        CHECK(not b.identity_only_g.has_value());
        b.adopt_prior(prior);
        CHECK_NEAR(b.L(), prior.total(), 1e-5);
        CHECK_NEAR(b.L_evidence(), 0.0, 1e-6);
    }
    {   // e18-like evidence: a few ZED hits at low q plus misses
        constexpr Costs kCosts{30.f, 388.f, 388.f};   // delete line -ln(388/30) = -2.56 (the log's)
        const auto line = decide(0.f, Role::Ordinary, kCosts).delete_line;
        CHECK_NEAR(line, -2.56, 0.01);
        PlaceTable T(false, kLookalikeProbDefault, true);
        auto fixed = T.restored_belief(PlaceTable::row_of(idrow(), 0.1f, false, kLookalikeProbDefault, true), kNow);
        fixed.adopt_prior(prior);
        PlaceBelief old = PlaceTable(false).restored_belief(PlaceTable::row_of(idrow(), 0.1f, false), kNow);
        old.prior = prior;   // the regression: evidence 0 + the new-door prior
        for (int n : {4, 8, 12, 20}) { PlaceBelief f2 = fixed, o2 = old; low_q_views(f2.ledger, n, 0); low_q_views(o2.ledger, n, 0); std::printf("  n=%d misses: fixed %.3f old %.3f\n", n, f2.L(), o2.L()); }
        low_q_views(fixed.ledger, 12, 0); low_q_views(old.ledger, 12, 0);
        std::printf("e18-like (12 low-q misses): fixed L %.3f, old L %.3f, delete line %.3f\n", fixed.L(), old.L(), line);
        CHECK(fixed.L() > line);
        CHECK(fixed.L() > old.L() + 1.5);   // by the prior it no longer pays twice
        CHECK(not decide(fixed.L(), Role::Ordinary, kCosts).remove);
    }
    std::printf(fails == 0 ? "ALL OK\n" : "%d FAILED\n", fails);
    return fails == 0 ? 0 : 1;
}
