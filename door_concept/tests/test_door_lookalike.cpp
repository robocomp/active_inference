// test_door_lookalike.cpp — "looks like a door but is not": the third hypothesis (spec §3.1b, 2026-11-08).
//   L = priors + carried + Sc + Sp − ln((1 − lambda) + lambda·exp(Sc))
// Build + run (from door_concept/):
//   g++ -std=c++23 -O1 -w -I/usr/include/eigen3 -I/usr/include/opencv4 -I../common tests/test_door_lookalike.cpp -o /tmp/test_door_lookalike && /tmp/test_door_lookalike
#include <clocale>
#include <cmath>
#include <cstdio>
#include <limits>
#include "../src/core/place_belief.h"
#include "../src/core/place_table.h"
#include "../src/core/door_decision.h"
using namespace rc::door::core;
using rc::door::IdentityRow;
static int fails = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)
#define CHECK_NEAR(a, b, tol) do { const double a_ = (a), b_ = (b); if (not (std::fabs(a_ - b_) <= (tol))) { std::printf("FAIL %s:%d  %s = %.5f vs %.5f (tol %.3f)\n", __FILE__, __LINE__, #a, a_, b_, double(tol)); ++fails; } } while (0)

static const DetectorModel M = [] { DetectorModel m = default_detectors().zed; m.hit_rate = 0.9f; m.false_alarm = 0.02f; return m; }();
static const Vec2f DOOR(2.f, 1.f), NRM(0.f, 1.f);
static Vec2f cam_at(float deg, float d = 3.f) { const float a = deg * float(M_PI) / 180.f; return DOOR + d * Vec2f(std::cos(a), std::sin(a)); }
/// n views from n distinct standpoints of a door the detector fires on (hit_w = 1).
static void hit_views(ViewLedger& L, int n, float deg0 = 0.f, int visit = 0)
{
    for (int i = 0; i < n; ++i) { const auto v = L.observe(cam_at(deg0 + 65.f * float(i)), DOOR, NRM, 1.f, 0.f, visit + i); L.add_bernoulli(v, Channel::Zed, 0.9f, M, 1.f); L.recompute(v, false); }
}
static constexpr Costs kCosts{30.f, 88.f, 88.f};   // delete line −ln(88/30) = −1.08

int main()
{
    std::setlocale(LC_ALL, "");
    const float priors = -0.4f;
    const auto prior_of_value = [](float p) { PlacePrior pr; pr.base_rate = p; return pr; };

    for (const float lam : {0.05f, 0.2f})
    {   // saturation: 25 agreeing independent views leave L near priors − ln(lambda), for two lambdas
        PlaceBelief b; b.lookalike = lam; b.prior = prior_of_value(priors);
        hit_views(b.ledger, 25);
        std::printf("saturate lam %.2f: Sc %.2f L %.3f bound %.3f views %zu\n", lam, b.Sc(), b.L(), priors - std::log(lam), b.ledger.views().size());
        CHECK(b.Sc() > 40.0);
        CHECK_NEAR(b.L(), priors - std::log(lam), 0.01);
        CHECK_NEAR(b.ledger.sum(), b.Sc(), 1e-3);   // the raw ledger still accumulates the channels (the trace reads it)
    }
    {   // a saturated phantom crosses the delete line within 2 failed opens at ActuationLlr 3
        PlaceBelief b; b.prior = prior_of_value(priors); hit_views(b.ledger, 25);
        const auto line = decide(0.f, Role::Ordinary, kCosts).delete_line;
        CHECK_NEAR(line, -1.08, 0.01);
        CHECK(not decide(b.L(), Role::Ordinary, kCosts).remove);
        for (int attempts = 1; attempts <= 2; ++attempts)
        {
            b.ledger.add_actuation(0, -3.f); b.ledger.recompute(0, false);
            std::printf("phantom after %d failed opens: L %.3f (Sc %.2f Sp %.2f term %.2f)\n", attempts, b.L(), b.Sc(), b.Sp(), b.saturation_term());
        }
        CHECK(decide(b.L(), Role::Ordinary, kCosts).remove);
        CHECK_NEAR(b.Sp(), -6.0, 1e-3);
    }
    {   // absence: negative Sc behaves as the old sum (difference < 0.06 at lambda 0.05)
        for (const float sc : {-0.5f, -3.f, -8.f, -20.f})
        {
            PlaceBelief b; b.prior = prior_of_value(priors); b.carried_sc = sc;
            const double old_L = double(priors) + sc;
            CHECK(std::fabs(double(b.L()) - old_L) < 0.06);
            CHECK(b.L() >= old_L);
        }
    }
    {   // a real door: saturated, then actuation success or a crossing lifts it above the saturation
        const auto saturated = [&] { PlaceBelief b; b.prior = prior_of_value(priors); hit_views(b.ledger, 25); return b; };
        PlaceBelief opened = saturated(); opened.ledger.add_actuation(0, 3.f); opened.ledger.recompute(0, false);
        CHECK_NEAR(opened.L(), saturated().L() + 3.f, 1e-3);
        PlaceBelief crossed = saturated(); crossed.crossing = 2.3f;
        CHECK_NEAR(crossed.L(), saturated().L() + 2.3f, 1e-3);
        CHECK(crossed.L() > priors - std::log(0.05f) + 2.f);
        std::printf("real door: saturated %.3f opened %.3f crossed %.3f\n", saturated().L(), opened.L(), crossed.L());
    }
    {   // continuity and no overflow at Sc = ±1000 (and beyond)
        for (const double s : {-1000.0, 1000.0, -1e30, 1e30})
        {
            CHECK(std::isfinite(saturated_vision(s, 0.05)));
        }
        CHECK_NEAR(saturated_vision(1000.0, 0.05), -std::log(0.05), 1e-9);
        CHECK_NEAR(saturated_vision(-1000.0, 0.05), -1000.0 - std::log(1.0 - 0.05), 1e-6);
        // smooth: no step between consecutive small increments anywhere on the curve
        double last = saturated_vision(-100.0, 0.05), max_step = 0.0;
        for (double s = -100.0 + 0.01; s <= 100.0; s += 0.01) { const double v = saturated_vision(s, 0.05); max_step = std::max(max_step, std::fabs(v - last)); last = v; }
        CHECK(max_step <= 0.0101);
        // the exact formula, away from the log-sum form's rounding
        for (double s : {-5.0, -1.0, 0.0, 1.0, 5.0, 30.0})
            CHECK_NEAR(saturated_vision(s, 0.05), s - std::log(0.95 + 0.05 * std::exp(s)), 1e-9);
        // the inverse really is one
        for (double s : {-300.0, -5.0, 0.0, 3.0, 30.0})
        {
            const double v = saturated_vision(s, 0.05);
            CHECK_NEAR(vision_of_saturated(v, 0.05), s, 1e-2);
        }
    }
    {   // a revisit (a restored belief) does not re-saturate above the bound from new views
        for (const float lam : {0.05f, 0.2f})
        {
            PlaceBelief b; b.lookalike = lam; hit_views(b.ledger, 25);
            PlaceRow r; r.L_evidence = b.L_evidence(); r.physical = float(b.Sp()); r.ledger = b.ledger; r.last_seen_wall_ms = 1000; r.visit = 1;
            PlaceTable T(false, lam);
            PlaceBelief restored = T.restored_belief(r, 1000);
            CHECK_NEAR(restored.L_evidence(), b.L_evidence(), 1e-3);       // no new views: the memory itself
            hit_views(restored.ledger, 25, 7.f, 100);                       // the next visit sees it from 25 fresh standpoints
            CHECK(restored.L_evidence() <= float(-std::log(lam)) + 0.01f);   // vision alone cannot lift it above the bound
            // …and an aged memory (the world may have changed) is not stuck below it: new views can bring it back to the bound
            PlaceBelief old = T.restored_belief(r, 1000 + 600 * 1000 * 100);  // a very long time later
            CHECK(old.L_evidence() < 0.2f);
            hit_views(old.ledger, 25, 7.f, 100);
            CHECK_NEAR(old.L_evidence(), -std::log(lam), 0.05);
        }
    }
    {   // the crossed door carried into the new room (empty ledger): its vision is carried inside the saturation, the crossing outside
        PlaceBelief src; hit_views(src.ledger, 25); src.ledger.add_actuation(0, 3.f); src.ledger.recompute(0, false);
        PlaceRow r; r.L_evidence = src.L_evidence(); r.physical = float(src.Sp());
        const PlaceBelief c = carried_belief(&r, 2.3f);
        CHECK_NEAR(c.L(), src.L_evidence() + 2.3f, 1e-3);
        PlaceBelief seen = c; hit_views(seen.ledger, 25);
        CHECK(seen.L() <= c.L() + 0.05f);                                    // the new room's views add nothing above the bound (saturated already)
        CHECK(seen.L() > 3.f + 3.f + 2.f);
        // a memory row of a phantom: it goes on being a phantom, on the vision saturation plus two failed opens
        PlaceBelief ph; hit_views(ph.ledger, 25); ph.ledger.add_actuation(0, -6.f); ph.ledger.recompute(0, false);
        PlaceRow pr; pr.L_evidence = ph.L_evidence(); pr.physical = float(ph.Sp()); pr.ledger = ph.ledger; pr.last_seen_wall_ms = 1000;
        PlaceTable T(false);
        PlaceBelief phb = T.restored_belief(pr, 1000); hit_views(phb.ledger, 25, 7.f, 100);
        CHECK(phb.L_evidence() <= pr.L_evidence + 0.05f);
        CHECK(phb.L_evidence() < -2.f);
    }
    {   // a row written before the model (no exist_physical): its L is the OLD unbounded sum, saturated once on load
        IdentityRow i; i.room = "r"; i.name = "door_4"; i.xy = {0.f, 0.f}; i.exist_logodds = 89.0f;
        PlaceRow r = PlaceTable::row_of(i, 0.1f, false);
        CHECK_NEAR(r.L_evidence, -std::log(0.05f), 1e-3);
        // the actuation in its ledger stays physical
        ViewLedger L; hit_views(L, 2); L.add_actuation(0, -6.f); L.recompute(0, false); i.ledger = L.serialise();
        PlaceRow r2 = PlaceTable::row_of(i, 0.1f, false);
        CHECK_NEAR(r2.physical, -6.0f, 1e-3);
        CHECK_NEAR(r2.L_evidence, -6.0f + (-std::log(0.05f)), 1e-3);
        // and with the column present nothing is rescaled
        i.exist_physical = -6.0f; i.exist_logodds = -3.0f;
        CHECK_NEAR(PlaceTable::row_of(i, 0.1f, false).L_evidence, -3.0f, 1e-3);
    }
    std::printf(fails ? "FAILED %d\n" : "PASS test_door_lookalike\n", fails);
    return fails ? 1 : 0;
}
