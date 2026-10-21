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
    // ---- (1c) a RE-INITIALISED anchor (λ → ½ after one sure re-init): the gap wins above an even score, loses below
    {
        const double lam = anchor_lost_after_reinit(lambda0, 1.0);
        CHECK("(1c) λ after a sure re-init is 1/2", std::abs(lam - 0.5) < 1e-12);
        CHECK("(1c) λ without a re-init is unchanged", std::abs(anchor_lost_after_reinit(lambda0, 0.0) - lambda0) < 1e-12);
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

    std::printf("%s (%d failures)\n", fails == 0 ? "ALL PASS" : "FAILURES", fails);
    return fails == 0 ? 0 : 1;
}
