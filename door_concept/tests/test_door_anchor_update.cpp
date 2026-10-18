// Offline harness for door_anchor_update.h: the door's opening pose in the ROBOT frame corrected by the robot's own
// LiDAR rays (+ the camera's door-mask points). Synthetic 2-D ray casts in the style of test_door_phi_stability.cpp:
// a 0.11 m wall 2.5 m ahead of the robot with a 1.0 m opening, a far wall 4 m behind it, a 0.25 deg scan.
//
//   open_corrects_lateral_offset — prior 0.45 m to the side (σ 0.5): lateral error < 0.05 m, p_open > 0.9
//   no_camera_still_converges    — same with an empty mask (the user's case: an open door the camera misses)
//   leaf_rays_ignored            — a leaf at 90 deg on the robot's side: n_short > 0, posterior within 0.05 m of the above
//   shut_door                    — a flush leaf: p_open < 0.1, depth/yaw corrected, lateral σ ≥ 0.8 × prior's
//   misplaced_mask_downweighted  — mask centroid 0.6 m off the opening: posterior within 0.08 m, camera_nis > 9
//   no_data_is_prior             — no rays and no mask return the prior exactly
// Fix round 1 (review):
//   open/shut_prior_depth_{+0.35,+0.30,-0.30} — a depth prior beyond the old short-ray gate's reach: same bounds as
//                                  above, and the shut anchor does not run along the wall (|Δx| < 0.1 m)
//   single_frame_not_overconfident — depth and lateral σ ≥ 0.015 m after one frame (common mode marginalised)
//   second_opening_does_not_capture — a second 1 m opening 2.0 m along the same wall, prior 0.45 m toward it
// Task 7 (try64 live fixes):
//   turning_away_does_not_drag_anchor — door SHUT, a corner wall 1.5 m along at 90 deg, the robot turning in place at
//                                  0.35 rad/s (6 s out, 6 s back) from the try64 start pose with exact odometry and a
//                                  ±90 deg scanner: the posterior stays within 0.10 m / 0.05 rad of the truth; p_view
//                                  falls while the doorway is outside the sampled sector and rises when it returns
//   other_wall_outside_extent_ignored — the door's wall ends 0.3 m past its jamb (u_max); a 1 m gap and a collinear
//                                  wall beyond it; the prior 0.3 m toward them (door shut): the posterior is not pulled past the extent
//   shut_door_reads_shut           — the try64-like start pose, door shut, corner present: p_open < 0.1
//
// Build + run (from door_concept/; relative_pose.h pulls rt_twist.h ⇒ DSR/Qt/fastdds headers):
//   mkdir -p build; Q=/usr/include/x86_64-linux-gnu/qt5; g++ -std=c++23 -O1 -fPIC -I/usr/include/eigen3 -Isrc -I../common
//     -I$Q -I$Q/QtCore -I$Q/QtGui -I$Q/QtWidgets tests/test_door_anchor_update.cpp -o build/test_door_anchor_update
//     -lfastdds -lfastcdr -lQt5Core && ./build/test_door_anchor_update
#include <cmath>
#include <cstdio>
#include <format>
#include <print>
#include <random>
#include <string>
#include <vector>

#include "door_anchor_update.h"

using V2 = Eigen::Vector2f;
using V3 = Eigen::Vector3f;
using rc::interaction::RelPose;

static int g_fail = 0;
static void check(bool ok, const std::string& what)
{
    std::print("  [{}] {}\n", ok ? "PASS" : "FAIL", what);
    if (not ok) ++g_fail;
}

// ─── the scene, in the ROBOT frame (forward = +y, side = +x) ─────────────────────────────────────────
// The robot sits at the origin facing the wall. The wall's mid-plane is y = D (the TRUE anchor: x = 0, y = D,
// yaw = 0), thickness T, spanning x in [-4, 4] with the opening x in [-W/2, W/2]; a far wall at y = D + 4.
constexpr float D = 2.5f, T = 0.11f, W = 1.0f;
struct Seg { V2 a, b; };   // 2-D obstacle edges (boxes as 4 segments)
static void box(std::vector<Seg>& s, float x0, float x1, float y0, float y1)
{
    s.push_back({{x0, y0}, {x1, y0}}); s.push_back({{x1, y0}, {x1, y1}});
    s.push_back({{x1, y1}, {x0, y1}}); s.push_back({{x0, y1}, {x0, y0}});
}
enum class Leaf { none, side90, flush };
static std::vector<Seg> scene(Leaf leaf, bool second_opening = false)
{
    std::vector<Seg> s;
    box(s, -4.f, -0.5f * W, D - 0.5f * T, D + 0.5f * T);      // wall, left of the opening
    if (second_opening)                                       // a second 1 m opening centred 2.0 m along the wall
    {
        box(s, 0.5f * W, 1.5f, D - 0.5f * T, D + 0.5f * T);
        box(s, 2.5f, 4.f, D - 0.5f * T, D + 0.5f * T);
    }
    else
        box(s, 0.5f * W, 4.f, D - 0.5f * T, D + 0.5f * T);    // wall, right of the opening
    box(s, -4.f, 4.f, D + 4.f, D + 4.1f);                     // the far room's wall
    if (leaf == Leaf::side90)   // hinged at the right jamb's near face, swung 90 deg toward the robot
        box(s, 0.5f * W - 0.04f, 0.5f * W, D - 0.5f * T - W, D - 0.5f * T);
    if (leaf == Leaf::flush)    // shut, flush with the near face
        box(s, -0.5f * W, 0.5f * W, D - 0.5f * T, D - 0.5f * T + 0.04f);
    return s;
}
static float ray_seg(const V2& o, const V2& u, const Seg& sg)
{
    const V2 e = sg.b - sg.a;
    const float den = u.x() * e.y() - u.y() * e.x();
    if (std::abs(den) < 1e-9f) return -1.f;
    const V2 w = sg.a - o;
    const float t = (w.x() * e.y() - w.y() * e.x()) / den;
    const float s = (w.x() * u.y() - w.y() * u.x()) / den;
    return (t > 0.f and s >= 0.f and s <= 1.f) ? t : -1.f;
}
static std::vector<V3> cast(Leaf leaf, unsigned seed = 7, bool second_opening = false)
{
    const auto segs = scene(leaf, second_opening);
    std::mt19937 rng(seed);
    std::normal_distribution<float> noise(0.f, 0.02f);
    std::vector<V3> pts;
    for (int i = 0; i < 1440; ++i)
    {
        const float a = float(i) * 0.25f * float(M_PI) / 180.f;
        const V2 u(std::cos(a), std::sin(a));
        float best = 1e9f;
        for (const auto& sg : segs) if (const float t = ray_seg(V2::Zero(), u, sg); t > 0.f and t < best) best = t;
        if (best > 8.f) continue;                               // no return
        const V2 p = u * (best + noise(rng));
        pts.emplace_back(p.x(), p.y(), 1.0f);
    }
    return pts;
}
// Camera door-mask points: a cloud filling the opening (its along-wall centroid at `centre`), at the wall plane.
static std::vector<V3> mask(float centre, int n = 200, unsigned seed = 3)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> ux(-0.5f * W, 0.5f * W), uz(0.f, 2.0f);
    std::vector<V3> m;
    for (int i = 0; i < n; ++i) m.emplace_back(centre + ux(rng), D, uz(rng));
    return m;
}
static RelPose prior_offset(float dy = -0.10f)
{
    RelPose p;
    p.x = V3(0.45f, D + dy, 0.05f);                             // 0.45 m to the side, dy in depth (default 10 cm short), 3 deg off
    p.P = V3(0.5f * 0.5f, 0.2f * 0.2f, 0.1f * 0.1f).asDiagonal();
    p.stamp_ms = 1234;
    return p;
}
static const V3 kOrigin(0.f, 0.f, 1.0f);
// The original scene's door wall runs x in [-4, 4] (the anchor at x = 0): its extent in anchor coordinates.
constexpr float kUmin = -4.f, kUmax = 4.f;

// ─── Task 7: a WORLD scene (anchor coordinates: opening centre at the origin, wall along x, mid-plane y = 0), a robot
// pose in it, and a scanner with a limited sampled sector ──────────────────────────────────────────────────────────
struct Pose2 { V2 p; float psi; };   // the robot frame's origin and rotation in the world (forward = R(psi)·(0, 1))
static V2 to_robot(const Pose2& r, const V2& w)
{
    const float c = std::cos(r.psi), s = std::sin(r.psi);
    const V2 q = w - r.p;
    return {c * q.x() + s * q.y(), -s * q.x() + c * q.y()};
}
static V3 anchor_truth(const Pose2& r) { const V2 a = to_robot(r, V2::Zero()); return {a.x(), a.y(), rc::interaction::wrap_pi(-r.psi)}; }
// The try64 start: the opening at (1.93, 1.50) yaw -0.84 in the robot frame ⇒ the robot at (-0.17, -2.44), psi = 0.84.
static Pose2 try64_start() { return {V2(-0.172f, -2.440f), 0.84f}; }
// Door wall u in [u0, u1] (the door's own run) with the opening; a corner wall at u = u0 running toward the robot's
// side; optionally a 1 m gap after u1 and a collinear wall beyond it; a far wall 4 m behind.
struct WorldOpts { float u0 = -1.5f, u1 = 4.0f; bool corner = true, shut = true, collinear_beyond = false; };
static std::vector<Seg> world(const WorldOpts& o)
{
    std::vector<Seg> s;
    box(s, o.u0, -0.5f * W, -0.5f * T, 0.5f * T);
    box(s, 0.5f * W, o.u1, -0.5f * T, 0.5f * T);
    if (o.corner) box(s, o.u0 - T, o.u0, -4.f, 0.5f * T);
    if (o.collinear_beyond) box(s, o.u1 + 1.0f, o.u1 + 3.5f, -0.5f * T, 0.5f * T);
    if (o.shut) box(s, -0.5f * W, 0.5f * W, -0.5f * T, -0.5f * T + 0.04f);
    box(s, -6.f, 6.f, 4.f, 4.1f);
    return s;
}
// A scan from the robot at `r`: beams every 0.25 deg within ±half_sector of the robot's forward, from the sensor at
// `sensor_robot`; returned in the ROBOT frame (z = 0.5, the wall band).
static const V3 kSensor(0.f, 0.14f, 0.67f);
static std::vector<V3> scan(const std::vector<Seg>& segs, const Pose2& r, float half_sector_deg = 90.f, unsigned seed = 7)
{
    std::mt19937 rng(seed);
    std::normal_distribution<float> noise(0.f, 0.02f);
    const float c = std::cos(r.psi), s = std::sin(r.psi);
    const V2 o_w = r.p + V2(c * kSensor.x() - s * kSensor.y(), s * kSensor.x() + c * kSensor.y());
    std::vector<V3> pts;
    for (float a = -half_sector_deg; a <= half_sector_deg; a += 0.25f)
    {
        const float b = r.psi + float(M_PI_2) + a * float(M_PI) / 180.f;   // forward ± a
        const V2 u(std::cos(b), std::sin(b));
        float best = 1e9f;
        for (const auto& sg : segs) if (const float t = ray_seg(o_w, u, sg); t > 0.f and t < best) best = t;
        if (best > 8.f) continue;
        const V2 pr = to_robot(r, o_w + u * (best + noise(rng)));
        pts.emplace_back(pr.x(), pr.y(), 0.5f);
    }
    return pts;
}
// The creation prior (anchor_from_room): the truth as mean, σ_model in the anchor's axes.
static RelPose creation_prior(const Pose2& r)
{
    RelPose p;
    p.x = anchor_truth(r);
    p.P = rc::door::anchor_axes_to_robot(V3(0.30f * 0.30f, 0.20f * 0.20f, 0.10f * 0.10f).asDiagonal(), p.x.z());
    p.stamp_ms = 1;
    return p;
}

int main()
{
    std::setlocale(LC_ALL, "");
    using rc::door::update_door_anchor;

    std::print("open_corrects_lateral_offset\n");
    const auto open_rays = cast(Leaf::none);
    {
        const auto [post, ev] = update_door_anchor(prior_offset(), W, T, kUmin, kUmax, kOrigin, open_rays, mask(0.f));
        std::print("    x=({:.3f},{:.3f},{:.3f}) σ_lat={:.3f} through={} blocked={} short={} p_open={:.3f} nis={:.2f}\n",
                   post.x.x(), post.x.y(), post.x.z(), std::sqrt(post.P(0, 0)), ev.n_through, ev.n_blocked, ev.n_short,
                   ev.p_open, ev.camera_nis);
        check(std::abs(post.x.x()) < 0.05f, "lateral error < 0.05 m");
        check(ev.p_open > 0.9f, "p_open > 0.9");
    }

    std::print("no_camera_still_converges\n");
    RelPose open_nomask;
    {
        const auto [post, ev] = update_door_anchor(prior_offset(), W, T, kUmin, kUmax, kOrigin, open_rays, {});
        open_nomask = post;
        std::print("    x=({:.3f},{:.3f},{:.3f}) σ_lat={:.3f} through={} blocked={} short={} p_open={:.3f} nis={:.2f}\n",
                   post.x.x(), post.x.y(), post.x.z(), std::sqrt(post.P(0, 0)), ev.n_through, ev.n_blocked, ev.n_short,
                   ev.p_open, ev.camera_nis);
        check(std::abs(post.x.x()) < 0.05f, "lateral error < 0.05 m");
        check(std::abs(post.x.y() - D) < 0.02f, "depth on the opening's centre, not the near face (< 2 cm)");
        check(ev.p_open > 0.9f, "p_open > 0.9");
        check(ev.camera_nis == -1.f, "camera_nis = -1 (no mask)");
    }

    std::print("leaf_rays_ignored\n");
    {
        const auto [post, ev] = update_door_anchor(prior_offset(), W, T, kUmin, kUmax, kOrigin, cast(Leaf::side90), {});
        std::print("    x=({:.3f},{:.3f},{:.3f}) through={} blocked={} short={} p_open={:.3f}\n",
                   post.x.x(), post.x.y(), post.x.z(), ev.n_through, ev.n_blocked, ev.n_short, ev.p_open);
        check(ev.n_short > 0, "n_short > 0");
        check((post.x.head<2>() - open_nomask.x.head<2>()).norm() < 0.05f, "within 0.05 m of the no-leaf posterior");
    }

    std::print("shut_door\n");
    {
        const RelPose pr = prior_offset();
        const auto [post, ev] = update_door_anchor(pr, W, T, kUmin, kUmax, kOrigin, cast(Leaf::flush), {});
        std::print("    x=({:.3f},{:.3f},{:.3f}) σ=({:.3f},{:.3f},{:.3f}) through={} blocked={} short={} p_open={:.3f}\n",
                   post.x.x(), post.x.y(), post.x.z(), std::sqrt(post.P(0, 0)), std::sqrt(post.P(1, 1)),
                   std::sqrt(post.P(2, 2)), ev.n_through, ev.n_blocked, ev.n_short, ev.p_open);
        check(ev.p_open < 0.1f, "p_open < 0.1");
        check(std::abs(post.x.y() - D) < 0.02f, "depth corrected (< 2 cm, prior 0.10 m)");
        check(std::abs(post.x.z()) < 0.01f, "yaw corrected (< 0.01 rad, prior 0.05)");
        check(std::sqrt(post.P(1, 1)) < 0.25f * std::sqrt(pr.P(1, 1)) and std::sqrt(post.P(2, 2)) < 0.25f * std::sqrt(pr.P(2, 2)),
              "depth and yaw σ shrink (< 0.25 × prior's)");
        check(std::sqrt(post.P(0, 0)) >= 0.8f * std::sqrt(pr.P(0, 0)), "lateral σ ≥ 0.8 × prior's");
    }

    std::print("misplaced_mask_downweighted\n");
    {
        const auto [post, ev] = update_door_anchor(prior_offset(), W, T, kUmin, kUmax, kOrigin, open_rays, mask(0.6f));
        std::print("    x=({:.3f},{:.3f},{:.3f}) p_open={:.3f} nis={:.2f}\n", post.x.x(), post.x.y(), post.x.z(),
                   ev.p_open, ev.camera_nis);
        check((post.x.head<2>() - V2(0.f, D)).norm() < 0.08f, "posterior within 0.08 m of the truth");
        check(ev.camera_nis > 9.f, "camera_nis > 9");
    }

    // ─── fix round 1: a depth prior beyond the gate's old reach (was: every wall return dropped as "short") ─────
    for (const float dy : {+0.35f, +0.30f, -0.30f})
    {
        std::print("open_prior_depth_{:+.2f}\n", dy);
        {
            const auto [post, ev] = update_door_anchor(prior_offset(dy), W, T, kUmin, kUmax, kOrigin, open_rays, {});
            std::print("    x=({:.3f},{:.3f},{:.3f}) through={} blocked={} short={} p_open={:.3f}\n", post.x.x(), post.x.y(),
                       post.x.z(), ev.n_through, ev.n_blocked, ev.n_short, ev.p_open);
            check(std::abs(post.x.x()) < 0.05f, "lateral error < 0.05 m");
            check(std::abs(post.x.y() - D) < 0.02f, "depth error < 2 cm");
            check(std::abs(post.x.z()) < 0.01f, "yaw error < 0.01 rad");
            check(ev.p_open > 0.9f, "p_open > 0.9");
        }
        std::print("shut_prior_depth_{:+.2f}\n", dy);
        {
            const RelPose pr = prior_offset(dy);
            const auto [post, ev] = update_door_anchor(pr, W, T, kUmin, kUmax, kOrigin, cast(Leaf::flush), {});
            std::print("    x=({:.3f},{:.3f},{:.3f}) σ=({:.3f},{:.3f},{:.3f}) blocked={} short={} p_open={:.3f}\n", post.x.x(),
                       post.x.y(), post.x.z(), std::sqrt(post.P(0, 0)), std::sqrt(post.P(1, 1)), std::sqrt(post.P(2, 2)),
                       ev.n_blocked, ev.n_short, ev.p_open);
            check(ev.p_open < 0.1f, "p_open < 0.1");
            check(std::abs(post.x.y() - D) < 0.02f, "depth corrected (< 2 cm)");
            check(std::abs(post.x.z()) < 0.01f, "yaw corrected (< 0.01 rad)");
            check(std::sqrt(post.P(0, 0)) >= 0.8f * std::sqrt(pr.P(0, 0)), "lateral σ ≥ 0.8 × prior's");
            check(std::abs(post.x.x() - pr.x.x()) < 0.1f, "does not run along the wall (|Δx| < 0.1 m)");
        }
    }

    std::print("single_frame_not_overconfident\n");
    {
        const auto [post, ev] = update_door_anchor(prior_offset(), W, T, kUmin, kUmax, kOrigin, open_rays, {});
        std::print("    σ=({:.4f},{:.4f},{:.4f})\n", std::sqrt(post.P(0, 0)), std::sqrt(post.P(1, 1)), std::sqrt(post.P(2, 2)));
        check(std::sqrt(post.P(1, 1)) >= 0.015f, "depth σ ≥ 0.015 m");
        check(std::sqrt(post.P(0, 0)) >= 0.015f, "lateral σ ≥ 0.015 m");
    }

    std::print("second_opening_does_not_capture\n");
    {
        const auto [post, ev] = update_door_anchor(prior_offset(), W, T, kUmin, kUmax, kOrigin, cast(Leaf::none, 7, true), {});
        std::print("    x=({:.3f},{:.3f},{:.3f}) p_open={:.3f}\n", post.x.x(), post.x.y(), post.x.z(), ev.p_open);
        check((post.x.head<2>() - V2(0.f, D)).norm() < 0.05f, "posterior within 0.05 m of the true opening");
    }

    // ★ The caller's prior (specificworker_anchor.cpp::anchor_from_room) after fix round 1: door belief + projected
    //   localisation covariance + σ_model. try63 leg 2: σ_s 0.04 m on a door fitted 0.45 m off. The OLD prior (the
    //   belief's 0.04 alone on the lateral axis) must fail to correct it — that is what made the error uncorrectable.
    std::print("room_frame_error_is_correctable\n");
    {
        const Eigen::Matrix3f belief = V3(0.04f * 0.04f, 0.f, 0.f).asDiagonal();            // σ_s 0.04, anchor axes
        const Eigen::Matrix3f C_edge = V3(0.03f * 0.03f, 0.03f * 0.03f, 0.01f * 0.01f).asDiagonal();   // room in robot
        const V3 mean(0.45f, D, 0.f);                                                          // 0.45 m off along the wall
        const Eigen::Matrix3f robot_pose =
            rc::door::robot_pose_cov_on_anchor_from_room_in_robot(C_edge, V2(-1.0f, -2.0f), mean);
        auto run = [&](const Eigen::Matrix3f& P_anchor)
        {
            RelPose r; r.x = mean; r.P = rc::door::anchor_axes_to_robot(P_anchor, mean.z()); r.stamp_ms = 1;
            for (int k = 0; k < 3; ++k) r = update_door_anchor(r, W, T, kUmin, kUmax, kOrigin, open_rays, {}).first;
            return r;
        };
        const RelPose fixed = run(rc::door::anchor_prior_cov(belief, robot_pose, V3(0.3f, 0.2f, 0.1f)));
        const RelPose old   = run(V3(0.04f * 0.04f, 0.2f * 0.2f, 0.1f * 0.1f).asDiagonal());
        std::print("    new prior: x={:.3f} σ_lat={:.3f}   old prior (σ_s only): x={:.3f} σ_lat={:.3f}\n",
                   fixed.x.x(), std::sqrt(fixed.P(0, 0)), old.x.x(), std::sqrt(old.P(0, 0)));
        check(std::abs(fixed.x.x()) < 0.05f, "summed prior: lateral error < 0.05 m after ≤ 3 updates");
        check(std::abs(old.x.x()) >= 0.05f, "RED with the belief-only prior: lateral error stays ≥ 0.05 m");
        const Eigen::Matrix3f sum = rc::door::anchor_prior_cov(belief, robot_pose, V3(0.3f, 0.2f, 0.1f));
        check(std::abs(sum(0, 0) - (0.04f * 0.04f + robot_pose(0, 0) + 0.09f)) < 1e-6f, "the three sources ADD (no max, no floor)");
    }

    // ─── Task 7 ──────────────────────────────────────────────────────────────────────────────────────────────
    std::print("turning_away_does_not_drag_anchor\n");
    {
        const WorldOpts wo;                                    // shut, corner at u = -1.5
        const auto segs = world(wo);
        Pose2 robot = try64_start();
        RelPose belief = creation_prior(robot);
        constexpr float dt = 0.1f, w = 0.35f;
        float worst_pos = 0.f, worst_yaw = 0.f, pv_first = -1.f, pv_min_out = 1.f, pv_last = -1.f;
        int n_out = 0;
        for (int k = 0; k < 120; ++k)
        {
            const float omega = k < 60 ? w : -w;               // 6 s out, 6 s back
            rc::rt::Twist tw; tw.angular = V3(0.f, 0.f, omega);
            belief = rc::interaction::predict(belief, tw, V3(1e-4f, 1e-4f, 1e-4f), dt);
            robot.psi += omega * dt;
            const auto [post, ev] = update_door_anchor(belief, W, T, wo.u0, wo.u1, kSensor, scan(segs, robot, 90.f, 7 + k), {});
            belief = post;
            const V3 tr = anchor_truth(robot);
            worst_pos = std::max(worst_pos, (post.x.head<2>() - tr.head<2>()).norm());
            worst_yaw = std::max(worst_yaw, std::abs(rc::interaction::wrap_pi(post.x.z() - tr.z())));
            // The doorway (opening ± one width) wholly outside the ±90 deg sector: its bearing from forward > 90 + 25 deg.
            const float bearing = std::abs(std::atan2(tr.x(), tr.y())) * 180.f / float(M_PI);
            if (k == 0) pv_first = ev.p_view;
            if (bearing > 115.f) { pv_min_out = std::min(pv_min_out, ev.p_view); ++n_out; }
            pv_last = ev.p_view;
            if (k % 10 == 0 or k == 119)
                std::print("    k={:3d} bearing={:5.1f} err=({:.3f} m, {:.3f} rad) p_view={:.3f} p_open={:.3f}\n", k, bearing,
                           (post.x.head<2>() - tr.head<2>()).norm(), rc::interaction::wrap_pi(post.x.z() - tr.z()), ev.p_view, ev.p_open);
        }
        std::print("    worst err {:.3f} m {:.3f} rad; p_view first {:.3f}, max while out ({} frames) {:.3f}, last {:.3f}\n",
                   worst_pos, worst_yaw, pv_first, n_out, pv_min_out, pv_last);
        check(worst_pos < 0.10f, "posterior within 0.10 m of the truth throughout");
        check(worst_yaw < 0.05f, "posterior within 0.05 rad of the truth throughout");
        check(n_out > 0 and pv_min_out < 0.1f, "p_view falls (< 0.1) while the doorway is out of the sampled sector");
        check(pv_first > 0.9f and pv_last > 0.9f, "p_view high (> 0.9) before it leaves and after it returns");
    }

    std::print("other_wall_outside_extent_ignored\n");
    {
        // The door near the END of its wall run (u_max 0.3 m past its jamb), then a 1 m gap and a collinear wall:
        // the gap is an "opening" the old infinite-line model could slide into.
        WorldOpts wo; wo.u0 = -3.0f; wo.u1 = 0.5f * W + 0.3f; wo.corner = false; wo.collinear_beyond = true;
        const Pose2 robot{V2(0.f, -2.5f), 0.f};                // facing the wall squarely
        RelPose pr = creation_prior(robot);
        pr.x.x() += 0.3f;                                      // 0.3 m toward the gap and the wall beyond u_max
        RelPose r = pr;
        for (int k = 0; k < 3; ++k) r = update_door_anchor(r, W, T, wo.u0, wo.u1, kSensor, scan(world(wo), robot, 90.f, 11 + k), {}).first;
        std::print("    x=({:.3f},{:.3f},{:.3f}) prior x {:.3f}; the opening's far jamb at {:.3f} (u_max {:.2f})\n",
                   r.x.x(), r.x.y(), r.x.z(), pr.x.x(), r.x.x() + 0.5f * W, wo.u1);
        check(r.x.x() + 0.5f * W <= wo.u1 + 0.02f, "the opening is not pulled past the wall's extent");
        check(r.x.x() <= pr.x.x() + 0.05f, "not pulled toward the gap beyond u_max (Δx ≤ 0.05 m)");
    }

    std::print("shut_door_reads_shut\n");
    {
        const WorldOpts wo;
        const Pose2 robot = try64_start();
        const auto [post, ev] = update_door_anchor(creation_prior(robot), W, T, wo.u0, wo.u1, kSensor, scan(world(wo), robot), {});
        const V3 tr = anchor_truth(robot);
        std::print("    x=({:.3f},{:.3f},{:.3f}) truth ({:.3f},{:.3f},{:.3f}) through={} blocked={} p_open={:.3f} p_view={:.3f}\n",
                   post.x.x(), post.x.y(), post.x.z(), tr.x(), tr.y(), tr.z(), ev.n_through, ev.n_blocked, ev.p_open, ev.p_view);
        check(ev.p_open < 0.1f, "p_open < 0.1");
        check((post.x.head<2>() - tr.head<2>()).norm() < 0.10f, "and the pose stays within 0.10 m of the truth");
    }

    std::print("no_data_is_prior\n");
    {
        const RelPose pr = prior_offset();
        const auto [post, ev] = update_door_anchor(pr, W, T, kUmin, kUmax, kOrigin, {}, {});
        check(post.x == pr.x and post.P == pr.P and post.stamp_ms == pr.stamp_ms, "posterior == prior exactly");
        check(ev.n_through == 0 and ev.n_blocked == 0 and ev.n_short == 0 and ev.p_open == 0.5f and ev.camera_nis == -1.f
              and ev.p_view == 0.f,
              "evidence is empty");
    }

    std::print("{}\n", g_fail == 0 ? "ALL PASS" : std::format("{} FAILED", g_fail));
    return g_fail == 0 ? 0 : 1;
}
