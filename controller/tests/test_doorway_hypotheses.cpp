// Task 3e (rtD1h leg 5): the doorway goal line as the MAP of anchor-vs-gap, and the anchor's re-initialisation test
// (src/local_policy/doorway_hypotheses.h).
// Build+run (from controller/; relative_pose.h pulls rt_twist.h → dsr_api.h, hence Qt/fastdds on the line):
//   Q=/usr/include/x86_64-linux-gnu/qt5; g++ -std=c++23 -O1 -Wall -Wextra -I/usr/include/eigen3 -I../common -Isrc
//     -I$Q -I$Q/QtCore -fPIC tests/test_doorway_hypotheses.cpp -o /tmp/tdh -lfastdds -lfastcdr -lQt5Core && /tmp/tdh   (one line)
#include "../src/local_policy/doorway_hypotheses.h"

#include <cmath>
#include <cstdio>
#include <numbers>

using namespace rc::policy;
using rc::interaction::RelPose;

static int fails = 0;
#define CHECK(name, cond) do { const bool ok_ = (cond); std::printf("%s %s\n", ok_ ? "PASS" : "FAIL", name); if (not ok_) ++fails; } while (0)

static RelPose rel(float x, float y, float psi, float sxy, float spsi)
{
    RelPose r;
    r.x = {x, y, psi};
    r.P = Eigen::Vector3f(sxy * sxy, sxy * sxy, spsi * spsi).asDiagonal();
    return r;
}

int main()
{
    constexpr float kDoorW = 0.87f, kHalfBody = 0.25f;   // DOOR_1's clear width; Shadow's lateral half width (order of)
    const double lambda0 = kDoorwayAnchorLostPrior;

    // ---- (1a) rtD1h leg 5 attempt 1: a FRESH anchor on an alias ~1.9 m / 70° off, against a 0.99-score gap on the true door
    //           0.6 m ahead. The goal line is the gap's (the 3b rule — anchor always — took the alias).
    {
        const DoorwayLine anchor{rel(-0.70f, -1.27f, 1.18f, 0.03f, 0.03f), kDoorW, 1.0 - lambda0};
        const DoorwayLine gap{rel(-0.05f, 0.60f, 0.00f, 0.02f, 0.02f), kDoorW, 0.99};
        const auto c = choose_doorway_line(anchor, gap, kHalfBody);
        CHECK("(1a) 0.99 gap vs alias anchor: the GAP's line", c.gap);
        CHECK("(1a) the two cannot both be right (posterior both-right ~0)", c.p_both_right < 1e-6);
    }
    // ---- (1b) rtD1c leg 5: a 0.03-0.3-score gap locked on the open LEAF (0.45 m wide) against a right anchor: the ANCHOR's line
    for (const double s : {0.03, 0.3})
    {
        const DoorwayLine anchor{rel(0.f, 1.5f, 0.f, 0.03f, 0.03f), kDoorW, 1.0 - lambda0};
        const DoorwayLine gap{rel(0.35f, 1.4f, 0.3f, 0.02f, 0.02f), 0.45f, s};
        CHECK(s < 0.1 ? "(1b) leaf phantom s=0.03: the ANCHOR's line" : "(1b) leaf phantom s=0.30: the ANCHOR's line",
              not choose_doorway_line(anchor, gap, kHalfBody).gap);
    }
    // ---- (1c) an anchor whose P(lost) has risen to ½ (fix round 1: the recursive AnchorLostBelief, tested below): the gap wins
    //           above an even score, loses below
    {
        const double lam = 0.5;
        const DoorwayLine anchor{rel(-1.9f, 0.5f, 1.93f, 0.05f, 0.05f), kDoorW, 1.0 - lam};
        const DoorwayLine gap_hi{rel(0.f, 0.6f, 0.f, 0.02f, 0.02f), kDoorW, 0.6};
        const DoorwayLine gap_lo{rel(0.f, 0.6f, 0.f, 0.02f, 0.02f), kDoorW, 0.4};
        CHECK("(1c) re-initialised anchor vs 0.6 gap: the GAP", choose_doorway_line(anchor, gap_hi, kHalfBody).gap);
        CHECK("(1c) re-initialised anchor vs 0.4 gap: the ANCHOR", not choose_doorway_line(anchor, gap_lo, kHalfBody).gap);
    }
    // ---- (1d) the anchor's COVARIANCE: the same 0.7-score gap loses to a tight anchor and beats one known only to ±0.4 m
    {
        const DoorwayLine gap{rel(0.f, 1.0f, 0.f, 0.02f, 0.02f), kDoorW, 0.7};
        const DoorwayLine tight{rel(0.05f, 1.0f, 0.f, 0.02f, 0.02f), kDoorW, 1.0 - lambda0};
        const DoorwayLine broad{rel(0.05f, 1.0f, 0.f, 0.40f, 0.10f), kDoorW, 1.0 - lambda0};
        const auto ct = choose_doorway_line(tight, gap, kHalfBody), cb = choose_doorway_line(broad, gap, kHalfBody);
        CHECK("(1d) tight anchor (σ 2 cm) vs 0.7 gap: the ANCHOR", not ct.gap);
        CHECK("(1d) broad anchor (σ 40 cm) vs 0.7 gap: the GAP", cb.gap);
        CHECK("(1d) agreeing measurements: both-right posterior high", ct.p_both_right > 0.9);
        CHECK("(1d) coverage falls with sigma", cb.c_anchor < ct.c_anchor);
    }
    // ---- (1e) a gap narrower than the body has no clearance: never the door, whatever its score
    {
        const DoorwayLine anchor{rel(0.f, 1.0f, 0.f, 0.03f, 0.03f), kDoorW, 1.0 - lambda0};
        const DoorwayLine narrow{rel(0.f, 1.0f, 0.f, 0.01f, 0.01f), 0.45f, 0.999};
        CHECK("(1e) a gap narrower than the body: the ANCHOR", not choose_doorway_line(anchor, narrow, kHalfBody).gap);
    }
    // ---- (1f) no gap: the anchor; a tie keeps the anchor (the door's owner)
    {
        const DoorwayLine anchor{rel(0.f, 1.0f, 0.f, 0.03f, 0.03f), kDoorW, 1.0 - lambda0};
        CHECK("(1f) no gap: the ANCHOR", not choose_doorway_line(anchor, std::nullopt, kHalfBody).gap);
        const DoorwayLine twin{rel(0.f, 1.0f, 0.f, 0.03f, 0.03f), kDoorW, 1.0 - lambda0};
        CHECK("(1f) equal credibility: the ANCHOR", not choose_doorway_line(anchor, twin, kHalfBody).gap);
    }

    // ---- (2a) re-initialisation: rtD1h attempt 2 — the block driven on vs door_concept's re-initialised wait pose 2.17 m / 12.8°
    //           away (Mahalanobis² 88 by door_concept's own test) ⇒ P(re-init) ≈ 1
    {
        const Eigen::Matrix3d robot_cov = Eigen::Vector3d(0.02 * 0.02, 0.02 * 0.02, 0.02 * 0.02).asDiagonal();
        const auto driven = anchor_in_room(rel(-0.70f, -1.27f, 1.18f, 0.05f, 0.05f), {-0.44f, 4.03f}, -0.11f);
        // the same room point, re-expressed after the robot turned 0.5 rad and moved: a continuation
        const float th2 = 0.39f;
        const Eigen::Vector2f p2(-0.40f, 4.10f);
        const Eigen::Vector2d d = driven.x.head<2>() - p2.cast<double>();
        const Eigen::Vector2d local = Eigen::Rotation2Dd(-double(th2)) * d;
        const auto same = anchor_in_room(rel(float(local.x()), float(local.y()), float(driven.x.z()) - th2, 0.06f, 0.06f), p2, th2);
        const double p_same = p_reinitialised(driven, same, kDoorW, robot_cov, 1.5);
        CHECK("(2a) the same room point after a turn and a move: a CONTINUATION (P(re-init) < 1/2)", p_same < 0.5);
        const auto moved = anchor_in_room(rel(-0.70f + 2.10f, -1.27f + 0.55f, 1.18f + 0.223f, 0.08f, 0.08f), {-0.44f, 4.03f}, -0.11f);
        const double p_moved = p_reinitialised(driven, moved, kDoorW, robot_cov, 1.5);
        CHECK("(2a) 2.17 m / 12.8 deg away: RE-INITIALISED (P > 0.99)", p_moved > 0.99);
        std::printf("      P(re-init): continuation %.4f, 2.17 m jump %.6f\n", p_same, p_moved);
    }
    // ---- (2b) the localiser's own uncertainty explains a small shift: 10 cm under a 10 cm-sigma pose is no re-init
    {
        const Eigen::Matrix3d robot_cov = Eigen::Vector3d(0.10 * 0.10, 0.10 * 0.10, 0.03 * 0.03).asDiagonal();
        const auto a = anchor_in_room(rel(0.f, 1.5f, 0.f, 0.03f, 0.03f), {0.f, 0.f}, 0.f);
        const auto b = anchor_in_room(rel(0.10f, 1.5f, 0.02f, 0.03f, 0.03f), {0.f, 0.f}, 0.f);
        CHECK("(2b) a 10 cm shift under a 10 cm pose sigma: a CONTINUATION", p_reinitialised(a, b, kDoorW, robot_cov, 1.5) < 0.5);
    }

    // ==== FIX ROUND 1 (review of 142d8d3a) ==========================================================================================
    // ---- I-1: one missed block must not zero the anchor. Its credibility decays CONTINUOUSLY with the block's age: the robot's
    //      motion over the age (|v|·age, |w|·age with the range as lever arm) widens the block's covariance, and coverage falls.
    {
        const RelPose a0 = rel(0.f, 1.5f, 0.f, 0.03f, 0.03f);
        const RelPose a_old = aged_anchor(a0, 0.12, 0.30, 0.6);     // one missed refresh while moving
        const RelPose a_very = aged_anchor(a0, 3.0, 0.30, 0.6);     // 3 s of motion unseen
        const double c0 = lateral_coverage(lateral_sigma(a0), 0.5 * kDoorW - kHalfBody);
        const double c1 = lateral_coverage(lateral_sigma(a_old), 0.5 * kDoorW - kHalfBody);
        const double c3 = lateral_coverage(lateral_sigma(a_very), 0.5 * kDoorW - kHalfBody);
        CHECK("(I-1) coverage falls continuously with the block's age", c0 >= c1 and c1 > c3 and c1 > 0.5);
        CHECK("(I-1) a parked / stationary block keeps its covariance (no motion, no widening)",
              lateral_sigma(aged_anchor(a0, 0.9, 0.0, 0.0)) == lateral_sigma(a0));
        // the reviewer's case: a right anchor one block stale vs the rtD1c leaf gap (0.52 m, s 0.03) ⇒ the ANCHOR
        const DoorwayLine stale{a_old, kDoorW, 1.0 - lambda0};
        const DoorwayLine leaf{rel(0.35f, 1.4f, 0.3f, 0.02f, 0.02f), 0.52f, 0.03};
        CHECK("(I-1) one-block-stale right anchor vs s=0.03 leaf gap: the ANCHOR", not choose_doorway_line(stale, leaf, kHalfBody).gap);
    }
    // ---- I-2: INERTIA from accumulated evidence, no hysteresis band. A gap score jittering 0.78/0.82 around an anchor at 0.8 must
    //      not flip the line tick by tick; sustained evidence must still move it.
    {
        // The reviewer's case: per-tick π made s = 0.78 / 0.80 / 0.82 against an anchor at 0.80 flip the line every tick. A score
        // jittering by ±0.02 tick to tick around a mean a little below (or above) the anchor's credibility: the accumulated
        // evidence settles the line once and holds it. (Evidence that nets EXACTLY zero leaves P(gap) at ½ — then the two lines
        // are equally credible and the MAP is honestly undecided; that is the one case the belief cannot and should not hold.)
        const DoorwayLine anchor{rel(0.f, 1.5f, 0.f, 0.03f, 0.03f), kDoorW, 0.8};
        for (const double mean : {0.79, 0.81})
        {
            DoorwayBelief b;
            int flips = 0;
            bool last = b.gap();
            for (int k = 0; k < 400; ++k)   // 20 s at 20 Hz
            {
                const DoorwayLine gap{rel(-0.6f, 1.2f, 0.f, 0.03f, 0.03f), kDoorW, mean + ((k % 2) ? 0.02 : -0.02)};
                const auto c = choose_doorway_line(anchor, gap, kHalfBody);
                b.step(c.pi_anchor, c.pi_gap, 0.05);
                if (b.gap() != last) { ++flips; last = b.gap(); }
            }
            std::printf("      jitter %.2f±0.02 vs 0.80 over 400 ticks: %d flips, P(gap) %.3f\n", mean, flips, b.p_gap());
            CHECK(mean < 0.8 ? "(I-2) score 0.79±0.02 vs anchor 0.80: the line never flips in 20 s"
                             : "(I-2) score 0.81±0.02 vs anchor 0.80: the line flips at most once in 20 s",
                  flips <= (mean < 0.8 ? 0 : 1) and b.gap() == (mean > 0.8));
        }

        // rtD1h: the anchor turns into an alias (credibility unchanged — nothing told the controller) while a 0.99 gap keeps
        // measuring the door: the gap's line within a few seconds, and it STAYS
        DoorwayBelief r;
        const DoorwayLine alias{rel(-0.70f, -1.27f, 1.18f, 0.03f, 0.03f), kDoorW, 1.0 - lambda0};
        const DoorwayLine good{rel(-0.05f, 0.60f, 0.f, 0.02f, 0.02f), kDoorW, 0.99};
        double t_gap = -1.0;
        for (int k = 0; k < 200; ++k)
        {
            const auto c = choose_doorway_line(alias, good, kHalfBody);
            r.step(c.pi_anchor, c.pi_gap, 0.05);
            if (r.gap() and t_gap < 0.0) t_gap = 0.05 * (k + 1);
        }
        std::printf("      rtD1h: gap's line after %.2f s, P(gap) %.3f at 10 s\n", t_gap, r.p_gap());
        CHECK("(I-2) 0.99 gap vs alias anchor: the gap's line within 1 s and held at 10 s", t_gap > 0.0 and t_gap <= 1.0 and r.gap());
        // and an rtD1c leaf gap never takes it from a right anchor, however long it is seen
        DoorwayBelief q;
        bool ever = false;
        const DoorwayLine right{rel(0.f, 1.5f, 0.f, 0.03f, 0.03f), kDoorW, 1.0 - lambda0};
        const DoorwayLine leaf{rel(0.35f, 1.4f, 0.3f, 0.02f, 0.02f), 0.45f, 0.3};
        for (int k = 0; k < 1200; ++k) { const auto c = choose_doorway_line(right, leaf, kHalfBody); q.step(c.pi_anchor, c.pi_gap, 0.05); ever = ever or q.gap(); }
        CHECK("(I-2) a 0.3 leaf gap never takes the line from a right anchor in 60 s", not ever);
        // the belief cannot saturate past what the switching hazard allows: a reversal is never locked out
        DoorwayBelief z;
        for (int k = 0; k < 6000; ++k) { const auto c = choose_doorway_line(alias, good, kHalfBody); z.step(c.pi_anchor, c.pi_gap, 0.05); }
        double t_back = -1.0;
        for (int k = 0; k < 2000 and t_back < 0.0; ++k)
        { const auto c = choose_doorway_line(right, leaf, kHalfBody); z.step(c.pi_anchor, c.pi_gap, 0.05); if (not z.gap()) t_back = 0.05 * (k + 1); }
        std::printf("      after 5 min on the gap, a reversed scene returns the line to the anchor in %.2f s\n", t_back);
        CHECK("(I-2) the hazard bounds the inertia: a reversal returns within 15 s", t_back > 0.0 and t_back <= 15.0);
    }
    // ---- I-2 (λ): the anchor's lost belief is RECURSIVE — it falls on confirmed continuations and rises on a jump; the re-init
    //      test uses the CURRENT λ as its prior
    {
        const Eigen::Matrix3d robot_cov = Eigen::Vector3d(0.01 * 0.01, 0.01 * 0.01, 0.01 * 0.01).asDiagonal();
        AnchorLostBelief lam;
        const auto a = anchor_in_room(rel(0.f, 1.5f, 0.f, 0.03f, 0.03f), {0.f, 0.f}, 0.f);
        const auto b = anchor_in_room(rel(0.01f, 1.5f, 0.005f, 0.03f, 0.03f), {0.f, 0.f}, 0.f);
        for (int k = 0; k < 100; ++k) lam.observe(a, b, kDoorW, robot_cov, 1.5, 0.05);   // 5 s of continuations
        const double after_cont = lam.lambda;
        CHECK("(I-2 λ) 5 s of confirmed continuations: P(lost) FALLS below its prior", after_cont < lambda0);
        const auto j = anchor_in_room(rel(2.0f, 1.0f, 0.3f, 0.05f, 0.05f), {0.f, 0.f}, 0.f);
        const double p_jump = lam.observe(a, j, kDoorW, robot_cov, 1.5, 0.05);
        CHECK("(I-2 λ) a 2 m jump: re-initialisation is the MAP and P(lost) RISES", p_jump > 0.5 and lam.lambda > after_cont);
        std::printf("      λ: prior %.2f -> %.4f after continuations -> %.3f after a jump (P(re-init) %.4f)\n", lambda0, after_cont,
                    lam.lambda, p_jump);
        // the prior is the current λ: the same disagreement is less likely a re-init for a well-confirmed anchor
        const auto m = anchor_in_room(rel(0.12f, 1.5f, 0.04f, 0.03f, 0.03f), {0.f, 0.f}, 0.f);
        CHECK("(I-2 λ) p_reinitialised uses the prior it is given",
              p_reinitialised(a, m, kDoorW, robot_cov, 1.5, 0.01) < p_reinitialised(a, m, kDoorW, robot_cov, 1.5, 0.5));
        // the hazard pulls an old verdict back toward the prior (no belief is final)
        AnchorLostBelief h; h.lambda = 0.999; h.predict(60.0);
        CHECK("(I-2 λ) 60 s without evidence relaxes P(lost) toward the prior", h.lambda < 0.9 and h.lambda > lambda0);
    }

    std::printf("%s (%d failures)\n", fails == 0 ? "ALL PASS" : "FAILURES", fails);
    return fails == 0 ? 0 : 1;
}
