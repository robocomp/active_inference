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

    // 1. PRIOR from the route's surprisal R: P(U) = 1 - 0.99 e^-R; present from the reset on (C1)
    {
        RB rb; rb.reset(kCell, kR);
        rb.tick(P0, 0.f, 0.f, b);
        CHECK("prior at reset = logit(0.01) (UnseenContact::kPriorBlocked, a route without doubt)", near(rb.log_odds(), kPrior0));
        rb.set_route_surprisal(2.f);
        const float pu = 1.f - 0.99f * std::exp(-2.f);
        CHECK("prior with 2 nats of route doubt = logit(1 - 0.99 e^-2)", near(rb.log_odds(), std::log(pu / (1.f - pu)), 1e-3f));
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
        rb.tick(P0, 0.f, 0.f, b); rb.add(P0.p, 3.f);
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
        rb.observed(10.0, 1.0f);
        CHECK("first observation: priced on its plan's duration", near(rb.bar(2.f), std::log(2.f / 1.f)));
        rb.observed(10.5, 1.0f);
        CHECK("then on the measured interval between observations (0.5 s), whatever the tick rate", near(rb.bar(2.f), std::log(2.f / 0.5f)));
        rb.interrupt(); rb.observed(100.0, 0.8f);
        CHECK("after an interruption the gap does not price the bar", near(rb.dt_obs(), 0.8f));
        CHECK("a goal with no finite remaining time: never", not rb.hand_back(INFINITY));
        // C1: a Heading goal 90 deg from done has T = (pi/2) / w_max ~ 1.05 s, NOT ~0 — the prior alone (-4.6) never clears it
        CHECK("C1: a turn owed (T 1.05 s, plan 1 s) is far below the prior: no hand back from the prior", not rb.hand_back(1.05f));
    }
    std::printf("%s (%d failing)\n", fails == 0 ? "ALL PASS" : "FAILURES", fails);
    return fails == 0 ? 0 : 1;
}
