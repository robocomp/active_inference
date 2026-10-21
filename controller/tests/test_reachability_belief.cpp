// F1 (2026-10-21 doorway-release spec) + fix round 1: ReachabilityBelief — Unreachable as an accumulated, place-discounted
// belief. (What each tick observes — claims judged against their own rollout, counts graded by the best shortening an option
// offers — is the Navigator's, exercised by the sim2d scenarios.)
// Build+run (from controller/; one line):
//   g++ -std=c++23 -O2 -Wall -Wextra -I/usr/include/eigen3 tests/test_reachability_belief.cpp -o /tmp/trb && /tmp/trb
#include "../src/local_policy/reachability_belief.h"

#include <cmath>
#include <cstdio>

using namespace rc::policy;
using RB = ReachabilityBelief;
static int fails = 0;
#define CHECK(name, cond) do { const bool ok_ = (cond); std::printf("%s %s\n", ok_ ? "PASS" : "FAIL", name); if (not ok_) ++fails; } while (0)

static BaseLimits base()
{
    BaseLimits b; b.v_max = 0.5f; b.w_max = 1.5f; b.a_lin = 1.f; b.d_lin = 0.5f; b.a_rot = 2.f; b.d_rot = 2.f;
    b.v_min_eff = 0.03f; b.w_min_eff = 0.08f; return b;
}
static bool near(float a, float b, float tol = 1e-4f) { return std::abs(a - b) < tol; }

int main()
{
    constexpr float kCell = 0.05f, kR = 0.36f, kLat = 0.1f;
    const BaseLimits b = base();
    auto at = [](float x, float y) { Pose2 p; p.p = {x, y}; return p; };
    const Pose2 P0 = at(0.012f, 0.012f);
    const float kPrior0 = std::log(0.01f / 0.99f);

    // 1. PRIOR: the base rate logit(0.01), present from the reset on (C1; the route-price prior was dropped in fix round 2)
    {
        RB rb; rb.reset(kCell, kR);
        rb.tick(P0, 0.f, 0.f, b);
        CHECK("prior at reset = logit(0.01) (UnseenContact::kPriorBlocked, a route without doubt)", near(rb.log_odds(), kPrior0));
    }
    // 2. CLAIM likelihood: continuous, no clamp
    {
        CHECK("claim delivered as predicted (D 0.5 s, r 0.5 s, sigma 0.05): strongly AGAINST U", RB::claim(0.5f, 0.5f, 0.05f) < -40.f);
        CHECK("claim that stopped at the obstruction (r 0): strongly FOR U", RB::claim(0.5f, 0.f, 0.05f) > 40.f);
        CHECK("claim half delivered: exactly neutral", near(RB::claim(0.5f, 0.25f, 0.05f), 0.f, 1e-3f));
        CHECK("a predicted retreat (D < 0) says nothing — U allows retreats", near(RB::claim(-0.4f, -0.42f, 0.05f), 0.f));
        CHECK("a back-up-and-realign plan (D ~ 0) says ~nothing (I3)", std::abs(RB::claim(0.005f, 0.f, 0.05f)) < 0.01f);
        CHECK("the log-ratio formula: ((r-D)^2 - (r-mu_U)^2) / (2 sigma^2)", near(RB::claim(0.3f, 0.1f, 0.1f), ((0.1f - 0.3f) * (0.1f - 0.3f) - 0.01f) / 0.02f));
    }
    // 3. COUNT: no escape among n effective options, at the coast's exposure
    {
        CHECK("no escape among n: log(n+1)", near(RB::no_escape(20), std::log(21.f)));
        CHECK("nothing searched: 0", RB::no_escape(0) == 0.f);
        RB rb; rb.reset(kCell, kR);
        const float coast = 0.35f * kLat + 0.35f * 0.35f / (2.f * 0.5f);
        CHECK("coast = |v| latency + v^2/(2 d_lin)", near(rb.coast_m(0.35f, 0.f, kLat, b), coast));
        CHECK("a Stuck while moving at 0.35 m/s is exposed at < 1/10", rb.exposure(0.35f, 0.f, kLat, b) < 0.1f);
        CHECK("a turn's coast is the body's far point's arc", near(rb.coast_m(0.f, 0.5f, kLat, b), kR * (0.5f * kLat + 0.25f / 4.f)));
    }
    // 4. PLACE kernel: one cell away e^-1, recalled on return, heading not part of the place, cells beyond 10 cell lengths skipped
    {
        RB rb; rb.reset(kCell, kR);
        rb.tick(P0, 0.f, 0.f, b); rb.add(P0.p, 3.f, 2.f);
        rb.tick(at(P0.p.x() + kCell, P0.p.y()), 0.f, 0.f, b);
        CHECK("one cell away: prior + e^-1 x evidence", near(rb.log_odds(), kPrior0 + 3.f * std::exp(-1.f)));
        rb.tick(at(P0.p.x() + 11.f * kCell, P0.p.y()), 0.f, 0.f, b);
        CHECK("beyond 10 cell lengths: skipped (M1)", near(rb.log_odds(), kPrior0));
        rb.tick(P0, 0.f, 0.f, b);
        CHECK("back at the place: recalled", near(rb.log_odds(), kPrior0 + 3.f));
        Pose2 turned = P0; turned.yaw = 1.f;
        rb.tick(turned, 0.f, 0.f, b);
        CHECK("a turn in place keeps the place", near(rb.log_odds(), kPrior0 + 3.f));
    }
    // 5. FROZEN pose sample while moving: not an observation; at rest the same pose is fresh
    {
        RB rb; rb.reset(kCell, kR);
        CHECK("first sample fresh", rb.tick(P0, 0.35f, 0.f, b));
        CHECK("identical sample at 0.35 m/s: frozen", not rb.tick(P0, 0.35f, 0.f, b));
        CHECK("identical sample at rest: fresh", rb.tick(P0, 0.f, 0.f, b));
    }
    // 6. DECISION: L > log(T / dt_obs), dt_obs = the interval BETWEEN OBSERVATIONS (I1), T = the goal's remaining time (C1)
    {
        RB rb; rb.reset(kCell, kR);
        rb.tick(P0, 0.f, 0.f, b);
        CHECK("no observation yet: no bar (+inf)", std::isinf(rb.bar(2.f)));
        rb.observed(10.0);
        CHECK("first observation: still no bar (fix round 2)", std::isinf(rb.bar(2.f)));
        rb.observed(10.5);
        CHECK("then on the measured interval between observations (0.5 s), whatever the tick rate", near(rb.bar(2.f), std::log(2.f / 0.5f)));
        rb.interrupt(); rb.observed(100.0);
        CHECK("after an interruption the gap does not price the bar (none until the next measured interval)", std::isinf(rb.bar(2.f)));
        CHECK("a goal with no finite remaining time: never", not rb.hand_back(INFINITY));
        // C1: a Heading goal 90 deg from done has T = (pi/2) / w_max ~ 1.05 s, NOT ~0 — the prior alone (-4.6) never clears it
        CHECK("C1: a turn owed (T 1.05 s, plan 1 s) is far below the prior: no hand back from the prior", not rb.hand_back(1.05f));
    }
    // 7. FIX ROUND 2 (live rtD1c): the three ways a goal was handed back without evidence about the place
    {
        // (1) the prior alone cannot hand back: the belief starts at the base rate (no planner-price route prior — rtD1c handed
        //     goals back on tick 1 with "priors" of 22.95 / 81.65 / 5.15 nats from summed, correlated residual log-odds)
        RB rb; rb.reset(kCell, kR);
        rb.tick(P0, 0.f, 0.f, b);
        rb.observed(0.0); rb.observed(1.0);
        CHECK("round 2 (1): the prior is the base rate logit(0.01), whatever the route", near(rb.log_odds(), kPrior0));
        CHECK("round 2 (1): the prior alone does not hand back a 0.5 s goal", not rb.hand_back(0.5f));
        // (2) no decision before a MEASURED interval: the first observation has no bar (rtD1c: bar finite on tick 1)
        RB r2; r2.reset(kCell, kR);
        r2.tick(P0, 0.f, 0.f, b);
        r2.add(P0.p, 50.f, 2.f);
        r2.observed(0.0);
        CHECK("round 2 (2): one observation — no bar, no hand back, whatever L", std::isinf(r2.bar(2.f)) and not r2.hand_back(2.f));
        r2.observed(0.5);
        CHECK("round 2 (2): the second observation prices the bar on the measured interval", near(r2.bar(2.f), std::log(2.f / 0.5f)));
        // (3) evidence gathered at T = inf (an infinite field: a statement about the map) is not folded, so it cannot be released
        //     when the field turns finite (rtD1c: ~17.9 nats stored over 105 s of an infinite cross field, released in one tick)
        RB r3; r3.reset(kCell, kR);
        r3.tick(P0, 0.f, 0.f, b);
        r3.add(P0.p, 17.9f, INFINITY);
        CHECK("round 2 (3): evidence at T = inf is not folded", near(r3.log_odds(), kPrior0));
        r3.add(P0.p, 1.f, 2.f);
        CHECK("round 2 (3): evidence at a finite T is", near(r3.log_odds(), kPrior0 + 1.f));
    }
    // 8. FIX ROUND 3 (live rtD1d leg 4): ONE claim of +27.5 nats — a back-up decided at v0 0.33 m/s right after a one-tick Stuck
    //    hold (the base already braking) predicted 0.123 m of shortening, ALL of it the forward braking coast, and realized
    //    0.010 m. sigma held only the field resolution (+) drift (~1.5 cm). Replay, in time units at v_max 0.5:
    {
        BaseLimits lb = b; lb.d_lin = 0.5f;                       // WheelDecelMps2 0.5 (live)
        const float vmax = lb.v_max, v0 = 0.33f, lat = 0.07f;     // latency as in the analysis' coast (0.33 x 0.07)
        const float D = 0.123f / vmax, r = 0.010f / vmax;
        const float D_stop = (v0 * lat + v0 * v0 / (2.f * lb.d_lin)) / vmax;   // the stop rollout: the common coast
        const float sig_meas = 0.015f / vmax;
        const float old = RB::claim(D, r, sig_meas);
        std::printf("   rtD1d replay: round-2 grading %+.1f nats (live +27.5)\n", old);
        CHECK("round 3 replay: the round-2 grading reproduces the live jump (> 20 nats)", old > 20.f);
        // (1) the dynamics model's own prediction variance; (3) the hold in force 0.15 s before the claim (488.64 -> 488.79)
        const float sig_dyn = RB::dynamics_sigma_s(v0, 0.f, lat, lat + 0.15f, lb, true, false);
        // (2) only the part of D a CHOICE made: the stop rollout's coast is subtracted from both D and r
        const float ev = RB::claim_choice(D, r, D_stop, std::hypot(sig_meas, sig_dyn));
        std::printf("   rtD1d replay: round-3 grading %+.2f nats (sigma_dyn %.3f s = %.3f m)\n", ev, sig_dyn, sig_dyn * vmax);
        CHECK("round 3 replay: the coast misprediction is worth a couple of nats at most, not 27", std::abs(ev) < 3.f);
        CHECK("round 3: sigma_dyn > 0 for a moving base", sig_dyn > 0.f);
        // a GENUINE non-delivery from rest still counts: from rest, 0.30 m predicted, nothing realized, no coast, no hold
        const float sig_rest = std::hypot(sig_meas, RB::dynamics_sigma_s(0.f, 0.f, lat, lat, lb, true, false));
        const float ev_rest = RB::claim_choice(0.30f / vmax, 0.f, 0.f, sig_rest);
        std::printf("   from rest, 0.30 m predicted, 0 realized: %+.1f nats\n", ev_rest);
        CHECK("round 3: a genuine non-delivery from rest still accumulates strongly (> 20 nats)", ev_rest > 20.f);
        CHECK("round 3: and a delivery from rest is strongly against", RB::claim_choice(0.30f / vmax, 0.30f / vmax, 0.f, sig_rest) < -20.f);
        // the dynamics sigma grows with the unobserved braking time (a hold in force) and with v0
        CHECK("round 3: a hold in force widens sigma_dyn", RB::dynamics_sigma_s(v0, 0.f, lat, lat + 0.3f, lb, true, false) > RB::dynamics_sigma_s(v0, 0.f, lat, lat, lb, true, false));
        CHECK("round 3: the heading part only for a goal that asks for a heading", RB::dynamics_sigma_s(0.f, 1.f, lat, lat, lb, true, false) == 0.f
              and RB::dynamics_sigma_s(0.f, 1.f, lat, lat, lb, false, true) > 0.f);
    }
    std::printf("%s (%d failing)\n", fails == 0 ? "ALL PASS" : "FAILURES", fails);
    return fails == 0 ? 0 : 1;
}
