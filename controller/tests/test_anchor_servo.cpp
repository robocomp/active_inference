// anchor_cross_command (controller_anchor_servo.h): crossing a door by servoing on its robot-frame anchor.
// Build+run (pure Eigen + std, but relative_pose.h pulls rt_twist.h → dsr_api.h, hence Qt/fastdds on the line):
//   Q=/usr/include/x86_64-linux-gnu/qt5; g++ -std=c++23 -O1 -I/usr/include/eigen3 -I../common -Isrc
//     -I$Q -I$Q/QtCore -I$Q/QtGui -I$Q/QtWidgets -fPIC tests/test_anchor_servo.cpp -o /tmp/tas
//     -lfastdds -lfastcdr -lQt5Core && /tmp/tas          (one line)
// (run from controller/)
#include "../src/controller_anchor_servo.h"
#include "../src/stall_judge.h"

#include <cmath>
#include <numbers>
#include <print>
#include <string_view>
#include <vector>

namespace
{
int failures = 0, checks = 0;
void check(bool ok, int line, std::string_view what)
{
    ++checks;
    if (not ok)
    {
        ++failures;
        std::println("FAIL line {}: {}", line, what);
    }
}
#define CHECK(x) check((x), __LINE__, #x)

using rc::interaction::RelPose;

constexpr float kWidth = 0.90f, kBody = 0.2938f, kStandoff = 2.0f, kVmax = 0.35f, kWmax = 1.0f;

// An anchor at (x, y, yaw) in the robot frame, isotropic position sigma, small yaw sigma.
RelPose anchor(float x, float y, float yaw, float sigma_xy)
{
    RelPose r;
    r.x = {x, y, yaw};
    r.P = Eigen::Matrix3f::Zero();
    r.P(0, 0) = r.P(1, 1) = sigma_xy * sigma_xy;
    r.P(2, 2) = 0.02f * 0.02f;
    return r;
}

// Rectangular bodies, CCW, robot frame (x right, y forward).
std::vector<Eigen::Vector2f> rect(float hw, float hl) { return {{hw, -hl}, {hw, hl}, {-hw, hl}, {-hw, -hl}}; }
const auto kLong = rect(kBody, 0.35f);       // the long test body of the original jamb test
const auto kShadow = rect(kBody, 0.23f);     // Shadow-like: 0.59 m wide, 0.46 m long

}  // namespace

// ── Task 12 fix A: the swung leaf ──────────────────────────────────────────────────────────────────────────────────
// try70's door: the leaf hinged on one jamb, swung 90° TOWARD the robot (its side, anchor −y). Anchor frame: the hinge on
// the +x jamb's robot-side face, the leaf running back along −y for its length.
constexpr float kWallHalfT = 0.05f, kMargin = 0.05f;
rc::interaction::AnchorLeaf leaf_toward(float phi = std::numbers::pi_v<float> / 2.f, float sigma_phi = 0.05f)
{ return rc::interaction::make_anchor_leaf(+0.5f * kWidth, -kWallHalfT, -1.f, -1.f, kWidth, phi, sigma_phi); }

// The TRUE leaf slab against the TRUE body (no margin): its centre line sampled at 5 mm, each sample a disc of the leaf's
// half-thickness — a contact is any disc reaching inside the body box.
int leaf_contacts(const RelPose& r, const rc::interaction::AnchorLeaf& l, const std::vector<Eigen::Vector2f>& body)
{
    const float psi = r.x.z(), c = std::cos(psi), s = std::sin(psi);
    int n = 0;
    for (int i = 0; i <= 200; ++i)
    {
        const Eigen::Vector2f qa = l.hinge + (l.tip - l.hinge) * (float(i) / 200.f);
        const Eigen::Vector2f q = r.x.head<2>() + Eigen::Vector2f(c * qa.x() - s * qa.y(), s * qa.x() + c * qa.y());
        if (rc::anchor_signed_distance(body, q) < l.half_thick) ++n;
    }
    return n;
}

// The session's chain on a differential base: the servo (with or without the leaf), then the jamb+leaf guard.
using Leaves = std::span<const rc::interaction::AnchorLeaf>;
Leaves one(const rc::interaction::AnchorLeaf* l) { return l != nullptr ? Leaves(l, 1) : Leaves(); }
void clear_gap_beside_the_leaf()
{
    const auto l90 = leaf_toward();
    const auto g90 = rc::anchor_clear_gap(kWidth, &l90);
    // Hinged at +w/2: the gap runs from −w/2 to the leaf's −x face (thickness 2·0.02 off the hinge side).
    CHECK(std::abs(g90.width - (kWidth - 2.f * float(rc::interaction::kLeafHalfThickM))) < 1e-4f);   // the slab is inside the opening
    CHECK(g90.centre < 0.f and g90.centre > -0.03f);
    const auto l60 = leaf_toward(std::numbers::pi_v<float> / 3.f, 0.f);   // ajar: projects L·cos 60° = 0.45 m into the opening
    const auto g60 = rc::anchor_clear_gap(kWidth, &l60);
    CHECK(g60.width < 0.5f * kWidth and g60.width > 0.5f * kWidth - 4.f * float(rc::interaction::kLeafHalfThickM));
    CHECK(g60.centre < -0.2f);
    const auto l120 = leaf_toward(2.f * std::numbers::pi_v<float> / 3.f, 0.f);   // past 90°: folded back over the wall
    CHECK(rc::anchor_clear_gap(kWidth, &l120).width > kWidth - 2.f * float(rc::interaction::kLeafHalfThickM) - 1e-3f);
    CHECK(rc::anchor_clear_gap(kWidth, nullptr).width == kWidth and rc::anchor_clear_gap(kWidth, nullptr).centre == 0.f);
    // The encoding round-trips: hinge, free edge, σ.
    auto l2 = leaf_toward(); l2.p = 0.3f; auto l1 = l90; l1.p = 0.7f;
    const auto [xs, ys] = rc::interaction::encode_anchor_leaves({l1, l2});
    const auto d = rc::interaction::decode_anchor_leaves(xs, ys, l90.phi);
    CHECK(d.size() == 2 and (d[0].hinge - l90.hinge).norm() < 1e-6f and (d[0].tip - l90.tip).norm() < 1e-5f
          and std::abs(d[0].sigma_phi - 0.05f) < 1e-4f and std::abs(d[0].p - 0.7f) < 1e-6f and std::abs(d[1].p - 0.3f) < 1e-6f);
    CHECK(rc::interaction::decode_anchor_leaves({0.f, 0.f}, {0.f, 0.f}, 0.f).empty());
    CHECK(rc::interaction::decode_anchor_leaves({xs[0], xs[1], xs[2]}, {ys[0], ys[1], ys[2]}, 0.f).size() == 1);   // the 3-vertex form
    std::println("  clear gap: 90 deg {:.3f} m at {:+.3f}; 60 deg {:.3f} m at {:+.3f}", g90.width, g90.centre, g60.width, g60.centre);
}

// Task 12 fix C: the approach standpoint, asked in the anchor's frame, ends ON the cross line at the producer's depth.
void approach_standpoint_on_the_cross_line()
{
    // The opening 2.2 m ahead, its normal rotated 0.2 rad; the room-frame standpoint the producer published lands 0.31 m off
    // the normal (try70 legs 3 and 5) at 1.9 m before the plane.
    const RelPose r = anchor(0.4f, 2.2f, 0.2f, 0.02f);
    const auto g0 = rc::anchor_cross_geometry(r);
    const Eigen::Vector2f offered = r.x.head<2>() + g0.a * (-0.31f) + g0.n * (-1.9f);
    for (const bool with_leaf : {false, true})
    {
        const auto l = leaf_toward();
        const auto* lp = with_leaf ? &l : nullptr;
        const Eigen::Vector2f q = rc::anchor_standpoint(r, kWidth, one(lp), kBody + kMargin, offered);
        // The robot standing at q (heading along the normal): the anchor seen from there.
        RelPose seen = r;
        seen.x = rc::interaction::move_mean(r.x, q.x(), q.y(), 0.f);   // translate the robot to q (robot-frame step)
        const auto g = rc::anchor_cross_geometry(seen);
        const float line = rc::anchor_cross_line(kWidth, one(lp), kBody + kMargin, rc::anchor_cross_geometry(r).sigma_lat);
        std::println("  approach standpoint{}: offered e {:+.3f} s {:+.3f} -> e {:+.3f} s {:+.3f} (the line at {:+.3f})",
                     with_leaf ? " (leaf at 90 deg)" : "", g0.a.dot(offered - r.x.head<2>()), g0.n.dot(offered - r.x.head<2>()),
                     g.e, g.s, line);
        CHECK(std::abs(g.e - line) < 0.005f);       // on the cross line
        CHECK(std::abs(g.s - (-1.9f)) < 0.005f);    // at the producer's depth
    }
    CHECK((rc::anchor_standpoint(RelPose{}, 0.f, Leaves(), kBody, offered) - offered).norm() == 0.f);   // no width: as published
    // I-5: fused by precision with the room frame — a room σ equal to the anchor's lands halfway between line and offer.
    {
        const float sa = rc::anchor_cross_geometry(r).sigma_lat;
        const Eigen::Vector2f q = rc::anchor_standpoint(r, kWidth, Leaves(), kBody, offered, sa);
        RelPose seen = r; seen.x = rc::interaction::move_mean(r.x, q.x(), q.y(), 0.f);
        const auto g = rc::anchor_cross_geometry(seen);
        std::println("  fused with an equally sure room frame: e {:+.3f} (offered -0.310, line 0)", g.e);
        CHECK(std::abs(g.e - (-0.155f)) < 0.005f);
    }
}

// Task 12 Fix E: the cross ends when the robot is THROUGH on the anchor, not at the room switch.
void cross_ends_through_not_at_the_room_switch()
{
    const RelPose at03 = anchor(0.f, -0.3f, 0.f, 0.02f), at20 = anchor(0.f, -2.0f, 0.f, 0.02f);   // s = +0.3, s = +2.0
    // `through` = the body's centre past the plane by the standoff (the cross protocol's own definition, rc::cross_progress)
    const bool t03 = rc::anchor_cross_geometry(at03).s >= kStandoff;
    const bool t20 = rc::anchor_cross_geometry(at20).s >= kStandoff;
    using E = rc::AnchorCrossEnd;
    CHECK(rc::anchor_cross_end(true, true, t03) == E::Continue);    // room switch at s +0.3 with an anchor: keep going
    CHECK(rc::anchor_cross_end(true, true, t20) == E::Through);     // s +2.0: through
    CHECK(rc::anchor_cross_end(true, false, t20) == E::Through);
    CHECK(rc::anchor_cross_end(false, true, t03) == E::RoomSwitch); // no anchor: the old room-switch rule
    CHECK(rc::anchor_cross_end(false, false, t03) == E::Continue);
    std::println("  cross end: switch at s +0.3 on the anchor -> continue (through {}), s +2.0 -> through ({})", t03, t20);
}

// Task 16 (try76 leg 2): a stale anchor past the plane, before any room switch, is driven on to `through`; before it, waited for.
void stale_past_the_plane_continues()
{
    CHECK(rc::anchor_drive_on(false, false, true, anchor(0.f, -0.36f, 0.f, 0.03f)));   // seen, then stale at s +0.36: drive
    CHECK(not rc::anchor_drive_on(false, false, true, anchor(0.f, 0.30f, 0.f, 0.02f))); // s −0.30: wait for a fresh one
    CHECK(rc::anchor_drive_on(true, false, false, anchor(0.f, 0.30f, 0.f, 0.02f)));    // fresh: drive
    CHECK(rc::anchor_drive_on(false, true, false, anchor(0.f, 0.30f, 0.f, 0.02f)));    // clearing: drive
    // Task 16 fix (try81 leg 3): a NEVER-SEEN anchor (the retry's room initialisation, carried) at "s +0.20", σ_lat 0.38: waits.
    CHECK(not rc::anchor_drive_on(false, false, false, anchor(0.f, -0.20f, 0.f, 0.38f)));
}

// Final review I-1: the anchor gone past the room switch with no held gap ends the cross (never a room route on old-room
// coordinates), and the passage then takes its room-switch end — no unbounded `clearing`.
void anchor_gone_while_clearing_ends()
{
    using A = rc::AnchorGoneAction;
    CHECK(rc::anchor_gone_action(true, true) == A::CarryOnGap);     // held gap + near side seen: carry to through
    CHECK(rc::anchor_gone_action(true, false) == A::CarryOnGap);
    CHECK(rc::anchor_gone_action(false, false) == A::RoomRoute);    // before the switch: the room route is this room's
    CHECK(rc::anchor_gone_action(false, true) == A::EndClearing);   // past it, nothing to carry: end
    // names_ltsm2 (2026-11-15): the door-route DRIVE-IN (past THROUGH, the body already wholly past the plane) lost its anchor
    // 1.1 m past the plane and fell back to the room route on the cross pose of the room just left ("goal outside the grid",
    // cross_unreachable, the leg failed with the robot in room_2). With no gap to carry it, a drive-in ENDS: never a room route.
    CHECK(rc::anchor_gone_action(false, false, /*driving_in=*/true) == A::EndDriveIn);
    CHECK(rc::anchor_gone_action(false, true, /*driving_in=*/true) == A::EndDriveIn);
    CHECK(rc::anchor_gone_action(true, false, /*driving_in=*/true) == A::CarryOnGap);   // a held gap still carries it in
    CHECK(rc::anchor_gone_action(false, false, /*driving_in=*/false) == A::RoomRoute);
    // The session drops `engaged` on EndClearing; the passage's decision then ends the cross by the room-switch rule.
    CHECK(rc::anchor_cross_end(/*anchored=*/false, /*clearing=*/true, /*through=*/false) == rc::AnchorCrossEnd::RoomSwitch);
    // Before the fix `engaged` stayed true: Continue for ever.
    CHECK(rc::anchor_cross_end(true, true, false) == rc::AnchorCrossEnd::Continue);
    // A gap that is held but whose near side was never observed does not carry (the retry-near-the-plane case).
    rc::CrossThroughGate gate;
    CHECK(not rc::cross_continues_without_anchor(true, gate));
    CHECK(rc::anchor_gone_action(rc::cross_continues_without_anchor(true, gate), true) == A::EndClearing);
}

// Task 8: the cross's COMPLETION, extracted from step_anchor_cross so the local policy's doorway drive and the servo share it:
// through only at s >= standoff AND observed on the near side in this claim; s >= standoff without that is not a crossing.
static void cross_progress_semantics()
{
    const float standoff = 2.0f, half_len = 0.3f;
    rc::interaction::RelPose r; r.x = Eigen::Vector3f(0.f, -2.1f, 0.f);   // centre 2.1 m BEHIND the robot: s = +2.1
    rc::CrossThroughGate gate;
    CHECK(rc::cross_progress(r, standoff, gate) == rc::CrossProgress::NotACrossing);   // never seen on the near side
    CHECK(not rc::cross_progress_complete(r, standoff, gate));
    rc::interaction::RelPose before; before.x = Eigen::Vector3f(0.f, 1.0f, 0.f);       // s = -1.0: wholly before the plane
    CHECK(rc::cross_progress(before, standoff, gate) == rc::CrossProgress::Going);
    CHECK(gate.observe(true, rc::anchor_cross_geometry(before).s, half_len));
    CHECK(rc::cross_progress(r, standoff, gate) == rc::CrossProgress::Through);
    CHECK(rc::cross_progress_complete(r, standoff, gate));
    rc::interaction::RelPose mid; mid.x = Eigen::Vector3f(0.f, -1.0f, 0.f);           // s = +1.0 < standoff
    CHECK(rc::cross_progress(mid, standoff, gate) == rc::CrossProgress::Going);
    rc::interaction::RelPose bad; bad.x = Eigen::Vector3f(NAN, 0.f, 0.f);
    CHECK(rc::cross_progress(bad, standoff, gate) == rc::CrossProgress::Going);         // nothing to judge: not through
    rc::interaction::RelPose badP = r; badP.P(0, 0) = NAN;                              // review M3: a non-finite covariance
    CHECK(rc::cross_progress(badP, standoff, gate) == rc::CrossProgress::Going);
}

// clean3 / tour21: the route's vantage is 1.43 m out, door_concept's live standpoint has moved to 0.58 m, the body stands at 1.31 m.
void test_at_vantage_uses_the_routes_vantage()
{
    const Eigen::Vector2f door(0.04f, 0.08f), robot(0.05f, 1.39f), route_vantage(0.03f, 1.51f), live_approach(0.04f, 0.66f);
    const float tol = 0.25f;
    CHECK(rc::passage_at_vantage(robot, door, route_vantage, tol));         // at the point the route drove to
    CHECK(not rc::passage_at_vantage(robot, door, live_approach, tol));     // the old comparison: the deadlock
    CHECK(not rc::passage_at_vantage(Eigen::Vector2f(0.05f, 5.0f), door, route_vantage, tol));  // pingpong17: 5 m out is not at the vantage
}

// clean8: door_2 is 1.26 m away through the corridor wall, the route is at goal 2 of 12 and its vantage is goal 7.
void test_route_reached_vantage()
{
    CHECK(not rc::route_reached_vantage(1, 7, false));   // close by distance, not at the vantage on the route
    CHECK(not rc::route_reached_vantage(6, 7, false));   // driving to the vantage goal itself
    CHECK(rc::route_reached_vantage(7, 7, false));       // the vantage goal is done
    CHECK(rc::route_reached_vantage(0, 1, true));        // no vantage: the body already stands in the doorway
}

int main()
{
    test_route_reached_vantage();
    test_at_vantage_uses_the_routes_vantage();
    clear_gap_beside_the_leaf();
    approach_standpoint_on_the_cross_line();
    cross_ends_through_not_at_the_room_switch();
    stale_past_the_plane_continues();
    anchor_gone_while_clearing_ends();
    cross_progress_semantics();
    std::println("{} / {} checks passed", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
