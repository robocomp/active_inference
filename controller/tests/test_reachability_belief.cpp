// F1 (2026-10-21 doorway-release spec): ReachabilityBelief — Unreachable as an accumulated, place-discounted belief.
// (What each tick observes — no-escape decisions at once, escape claims at their judgement — is the Navigator's, and is
//  exercised by the sim2d scenarios off_centre_fast_doorway / blocked_doorway / stop_is_the_only_release.)
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
static float logit(float q) { return std::log(q / (1.f - q)); }
static bool near(float a, float b, float tol = 1e-4f) { return std::abs(a - b) < tol; }

int main()
{
    constexpr float kCell = 0.05f, kR = 0.36f, kDt = 0.05f, kLat = 0.1f;
    constexpr int kN = 80;
    const float kNoEscape = std::log(float(kN) + 1.f);
    const BaseLimits b = base();
    auto at = [](float x, float y) { Pose2 p; p.p = {x, y}; return p; };
    const Pose2 P0 = at(0.012f, 0.012f);   // inside cell (0, 0)
    // one decision at pose p, speed v: no escape found (escape = false) or an escape delivered with P(contact) q
    auto decide = [&](RB& rb, Pose2 p, float v, double t, bool escape, float q = 0.f) {
        if (not rb.tick(p, v, 0.f, t, b, kDt)) return false;
        const float w = rb.exposure(v, 0.f, kLat, b);
        rb.add(p.p, w * (escape ? RB::delivered(q, kN) : RB::no_escape(kN)));
        return true; };

    // 1. a decision AT REST that found no escape among N candidates: the Bayes factor 1 / (1/(N+1)) = N+1
    {
        RB rb; rb.reset(kCell, kR);
        decide(rb, P0, 0.f, 0.0, false);
        std::printf("   no escape at rest, N %d: L %+.3f (log(N+1) %.3f)\n", kN, rb.log_odds(), kNoEscape);
        CHECK("Stuck at rest: L = log(N+1) > 0", near(rb.log_odds(), kNoEscape) and rb.p_unreachable() > 0.5f);
        CHECK("N = 0 searched: zero", RB::no_escape(0) == 0.f and RB::delivered(0.1f, 0) == 0.f);
    }
    // 2. a Stuck WHILE MOVING (0.35 m/s, latency 0.1 s, decel 0.5: coast 0.1575 m) barely moves it — the coast explains it
    {
        RB rb; rb.reset(kCell, kR);
        decide(rb, P0, 0.35f, 0.0, false);
        const float coast = 0.35f * kLat + 0.35f * 0.35f / (2.f * 0.5f);
        std::printf("   no escape moving 0.35 m/s: L %+.3f (exposure %.3f, coast %.4f m)\n", rb.log_odds(), rb.exposure(0.35f, 0.f, kLat, b), coast);
        CHECK("Stuck moving: coast = |v| latency + v^2 / (2 d_lin)", near(rb.coast_m(0.35f, 0.f, kLat, b), coast));
        CHECK("Stuck moving: < 1/10 of the same decision at rest", rb.log_odds() > 0.f and rb.log_odds() < 0.1f * kNoEscape);
        CHECK("a turn's coast is the body's far point's arc", near(rb.coast_m(0.f, 0.5f, kLat, b), kR * (0.5f * kLat + 0.25f / 4.f)));
    }
    // 3. a DELIVERED escape lowers it: logit(q), no finer than one in N+2
    {
        CHECK("delivered (q 0.05): logit(0.05)", near(RB::delivered(0.05f, kN), logit(0.05f)));
        CHECK("delivered (q 1e-6 < 1/(N+2)): -log(N+1), one decision's resolution", near(RB::delivered(1e-6f, kN), -kNoEscape));
        CHECK("delivered evidence is never FOR U (q read at most 0.5)", RB::delivered(0.9f, kN) <= 0.f);
        RB r; r.reset(kCell, kR);
        decide(r, P0, 0.f, 0.0, false); decide(r, P0, 0.f, 0.05, true, 0.f);
        CHECK("one delivered escape at rest cancels one no-escape at rest", near(r.log_odds(), 0.f));
    }
    // 4. the place kernel on the field's resolution: one cell away counts e^-1; coming BACK recalls the evidence
    {
        RB rb; rb.reset(kCell, kR);
        decide(rb, P0, 0.f, 0.0, false);
        rb.tick(at(P0.p.x() + kCell, P0.p.y()), 0.f, 0.f, 0.05, b, kDt);
        CHECK("one cell away: L = e^-1 x the evidence there", near(rb.log_odds(), kNoEscape * std::exp(-1.f)));
        rb.tick(at(P0.p.x() + 10.f * kCell, P0.p.y()), 0.f, 0.f, 0.10, b, kDt);
        CHECK("ten cells away: L ~ 0", std::abs(rb.log_odds()) < 1e-3f);
        rb.tick(P0, 0.f, 0.f, 0.15, b, kDt);
        CHECK("back at the place: its evidence is recalled (distance FROM the place, not travelled)", near(rb.log_odds(), kNoEscape));
        Pose2 turned = P0; turned.yaw = 1.f;
        rb.tick(turned, 0.f, 0.f, 0.20, b, kDt);
        CHECK("a turn in place keeps the place's evidence (heading is not part of the place)", near(rb.log_odds(), kNoEscape));
    }
    // 5. a FROZEN pose sample while moving is not an observation; at REST the same pose with a new sweep is
    {
        RB rb; rb.reset(kCell, kR);
        CHECK("first sample: fresh", decide(rb, P0, 0.35f, 0.0, false));
        const float L0 = rb.log_odds();
        CHECK("the same pose sample while moving at 0.35 m/s: frozen", not decide(rb, P0, 0.35f, 0.05, false));
        CHECK("frozen pose: no double count", near(rb.log_odds(), L0, 1e-6f));
        RB r2; r2.reset(kCell, kR);
        decide(r2, P0, 0.f, 0.0, false);
        CHECK("same pose at rest: fresh", decide(r2, P0, 0.f, 0.05, false));
        CHECK("same pose at rest: the new decision counts", near(r2.log_odds(), 2.f * kNoEscape));
    }
    // 5b. the PRIOR: the map's finite route out of every place, weighed as one delivered escape at rest
    {
        RB rb; rb.reset(kCell, kR);
        rb.set_prior(kN);
        rb.tick(P0, 0.f, 0.f, 0.0, b, kDt);
        CHECK("prior: L0 = -log(N+1) before any decision", near(rb.log_odds(), -kNoEscape));
        decide(rb, P0, 0.f, 0.05, false);
        CHECK("prior: ONE no-escape decision at rest leaves L = 0 (P(U) = 1/2), not a verdict", near(rb.log_odds(), 0.f));
        CHECK("prior: ... below any bar of a route longer than v_max dt", not rb.hand_back(0.05f, b.v_max));
        decide(rb, P0, 0.f, 0.10, false);
        CHECK("prior: the SECOND one at the same place clears a 1.5 m route's bar", rb.hand_back(1.5f, b.v_max));
    }
    // 6. the decision: hand back iff L > log(route / (v_max dt)) — the objective's value of the goal vs the cost of a tick
    {
        RB rb; rb.reset(kCell, kR);
        decide(rb, P0, 0.f, 0.0, true, 0.2f);
        decide(rb, P0, 0.f, 0.05, true, 0.2f);
        const float route = 1.5f, bar = std::log(route / (b.v_max * 0.05f));
        std::printf("   bar for a 1.5 m route at v_max 0.5, dt 0.05: %.3f nats (P(U) > %.3f)\n", rb.bar(route, b.v_max), 1.f / (1.f + std::exp(-bar)));
        CHECK("bar = log(route / (v_max dt))", near(rb.bar(route, b.v_max), bar));
        // from L0 = 2 logit(0.2) = -2.77, decisions at rest that find no escape: ceil((bar - L0) / log(N+1)) = 2
        const float L0 = rb.log_odds();
        int k = 0;
        while (not rb.hand_back(route, b.v_max) and k < 100) decide(rb, P0, 0.f, 0.10 + 0.05 * k++, false);
        const int model = int(std::ceil((bar - L0) / kNoEscape));
        std::printf("   persistent no-escape at rest from L %.2f: hand back after %d decisions (model ceil((%.3f - %.3f) / %.3f) = %d)\n",
                    L0, k, bar, L0, kNoEscape, model);
        CHECK("persistent no-escape at rest: hand back after ceil((bar - L0) / log(N+1)) decisions", k == model);
        CHECK("a 10 m route needs more evidence than a 1.5 m one", rb.bar(10.f, b.v_max) > rb.bar(route, b.v_max));
        CHECK("no finite route: never on this rule", not rb.hand_back(INFINITY, b.v_max));
        // an interruption: the gap is not a tick period (the next interval uses the fallback)
        rb.interrupt();
        rb.tick(P0, 0.f, 0.f, 100.0, b, kDt);
        CHECK("after an interruption a 90 s gap does not price the bar", near(rb.last_dt(), kDt));
        // the rtD1b sequence: delivered Drives at 0.35 m/s, ONE Stuck while moving, Drives again — never handed back
        RB live; live.reset(kCell, kR);
        Pose2 p = P0; double t = 0.0; bool handed = false;
        for (int i = 0; i < 40; ++i)
        {
            p.p.y() += 0.35f * 0.05f; t += 0.05;
            decide(live, p, 0.35f, t, i != 20, 0.004f);
            handed = handed or live.hand_back(0.6f, b.v_max);
        }
        CHECK("rtD1b: one Stuck while moving among Drives is never a hand back", not handed);
        // the F2 dither at one place (no-gain turn decisions from rest, stops not observed) closes: a trap is not held open
        RB dither; dither.reset(kCell, kR);
        int n = 0;
        while (not dither.hand_back(0.82f, b.v_max) and n < 100) decide(dither, P0, 0.f, 0.05 * n++, false);
        CHECK("a stop / no-gain-turn dither at one place is handed back", dither.hand_back(0.82f, b.v_max));
    }
    std::printf("%s (%d failing)\n", fails == 0 ? "ALL PASS" : "FAILURES", fails);
    return fails == 0 ? 0 : 1;
}
