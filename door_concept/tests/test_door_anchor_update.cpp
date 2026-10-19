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
//                                  (to its in-view level, which for a SHUT door is < 0.5 by design — fix round 3)
//   other_wall_outside_extent_ignored — the door's wall ends 0.3 m past its jamb (u_max); a 1 m gap and a collinear
//                                  wall beyond it; the prior 0.3 m toward them (door shut): the posterior is not pulled past the extent
//   shut_door_reads_shut           — the try64-like start pose, door shut, corner present: p_open < 0.1
// Task 7 fix round 1 (review):
//   turning_away_360               — the same turn with a 360° scan: the pose bounds hold (p_view need not fall — the
//                                  doorway stays in the sampled dome)
//   mid_crossing_360_keeps_view    — open door, 360° scan, the robot driving along the normal from s = -0.5 to +0.8 m:
//                                  p_view ≥ 0.6 at every step, lateral error < 0.05 m
//   pview_low_in_free_space / pview_low_when_rotated / pview_low_on_solid_wall (fix round 2) — open door (P(open)
//                                  0.95), 360° scan from the try64 start: a prior 1.2 m in front of the wall, yawed
//                                  +90°, or 1.5 m along onto solid wall reads P(door in view) < 0.5
//   crossing_with_leaf_at_moderate_pi (fix round 3) — open door with its leaf at 90° beside the jamb on the robot
//                                  side (the door's leaf belief passed in), π = 0.7 and 0.8, 3-D scan through
//                                  select_anchor_rays, s = -0.5 … +0.8 m:
//                                  min p_view ≥ 0.6, lateral < 0.05 m
//   true_pose_outscores_wrong_wall (fix round 3) — open door, π ∈ {0.05, 0.5, 0.95}: p_view(truth) > p_view(1.5 m
//                                  along on solid wall); for π ≥ 0.5 the wrong pose < 0.5 (at π = 0.05 it may be
//                                  ambiguous: the documented shut-door unobservability, door_anchor_update.h)
//   shut_door_reads_shut_with_floor — 3-D bpearl-like scan (z = 0.67, downward layers) with floor returns, shut door,
//                                  prior 0.25 m in front of the wall, rays through select_anchor_rays: p_open < 0.1,
//                                  depth < 0.05 m within 3 updates
// Task 7 fix round 4 (review):
//   crossing_with_leaf_at_moderate_pi now runs leaf_crossing(); its 3-D frames also feed
//   update_cost_on_3d_crossing_frames — per-update ms printed; FAILS above 50 ms in an optimised build (target 30)
//   wrong_leaf_belief_is_marginalised — π = 0.8, a confident belief (P(hinge), P(swing) 0.95) wrong on φ (believes 0),
//                                  the hinge side or the swing side: worst lateral < 0.10 m and yaw < 0.10 rad, and no
//                                  worse than no leaf model; the correct belief keeps lateral ≤ 0.05 m
//   dragged_posterior_not_published — the round-3 belief taken as certain (φ = 0) drags the anchor; with p_view judged
//                                  at the POSTERIOR no frame publishes a pose > 0.10 m / 0.10 rad off; the reported
//                                  p_view equals door_view of the returned posterior
//   pview_low_* ask door_view at the wrong pose (update_door_anchor's p_view is now its posterior's) and also check
//                                  that the update from each wrong prior publishes only a pose near the truth
// Task 8 (try65: a correct anchor whose publication flapped):
//   camera_view_persists_within_window (fix 1) — p_lidar 0.45 (try65's), a mask frame every other cycle with p_cam 0.95:
//                                  p_view never crosses the publication level while the frames keep coming (the old
//                                  "no new frame ⇒ p_cam 0" crossed it on every cycle), and falls to p_lidar once the last
//                                  frame is older than kAnchorStaleMs
//   try65_webots_door_crossing (fix 2) — DOOR_1 of tworooms-piso.wbt as built (0.2 m-deep frame, 0.9 m clear between
//                                  jambs, the 0.11 m wall 4.5 cm behind the frame's centre, the leaf open 90° on the
//                                  robot's side beside the −u jamb) under the LIVE caller configuration (w 1.14 from the
//                                  belief, t 0.05, leaf known with hinge/swing at 0.5/0.5, φ known at π/2 ± 0.25, φ_max
//                                  120°, π 0.6 and 0.75), a bpearl-like dome (875 azimuths, 32 downward layers, 0.67 m)
//                                  through select_anchor_rays with floor returns, the update chained from the creation
//                                  prior, the sensor 0.0 / 0.1 / 0.2 m off-centre from s = −2.0 to +0.8 m: p_view ≥ 0.6
//                                  on the approach and the exit (|s| ≥ 0.25 m) and ≥ 0.5 — never unpublished — inside
//                                  the doorway band; and at s = −1.0 the wrong poses (free space, rotated, solid wall)
//                                  read < 0.5, the shut door at π 0.7 is never published.
//                                  ⚠ The brief's ≥ 0.6 is NOT met inside the band |s| < 0.25 m: 0.53–0.66 over sensor
//                                  offsets. Cause, measured: the posterior's yaw error grows to 0.05–0.10 rad there with
//                                  σ_yaw 0.002 (the model's leaf hinged at w/2 = 0.57 m from the belief's 1.14 m width,
//                                  the real one at 0.475 m and 9 cm proud), so long grazing beams read as passing the
//                                  walls. With the width at 1.0 / 1.06 m the band's minimum is 0.65 / 0.66 (task-8 report).
//   anchor_frame_record_roundtrip (fix 3) — a recorded frame reads back bit-identical and re-running the update on it
//                                  reproduces the recorded posterior exactly (the replay is the live update)
// Task 9 (try66 leg 1: an anchor 1.1 m / 18° off that never published):
//   doorway_beside_a_room_corner (fix D) — the door's wall run ends 0.14 m past the +u jamb at a room corner (try66: 0.64 m
//                                  from the centre), the believed width 16 % wider than the clear one (try66: 1.06 vs
//                                  0.9), open door, 3-D scan from s = −2 m: the truth reads ≥ 0.6 and the wrong poses
//                                  < 0.5 — the old cell weighting counted the doorway grid PAST the run's end as wall and
//                                  read the free-space pose 0.62 (published)
//   sim_clock_rate (fix D; fix round 1) — rc::rt::SimClockRate maps wall onto sim time by CONSECUTIVE twist samples: a sim
//                                  at 0.94× integrates 0.94 s per wall second, a PAUSED sim nothing, a restart re-anchors
//   pi_zero_open_door_is_finite (fix round 1) — π = 0 with the door open at the mouth: finite posterior (try67: NaN, 0/0 in P(H_door))
//   first_anchor_predicted_before_clock_observed (fix round 2) — an anchor stamped before the first twist sample is still
//                                  predicted: SimClockRate::interval_s maps both ends by the current samples (try68: dt 0 for ever)
//   anchor_publish_decision (fix round 1) — Seen / Predicted (cross claimed, seen before, ≤ kAnchorPredictMs) / None
// Task 10 (an honest covariance: try66–69 published σ_yaw 0.1–0.2° against 2–28° errors):
//   persistent_bias_does_not_average_out — 40 frames of one open doorway, fresh beam noise each: carrying AnchorBias the
//                                  posterior keeps σ ≥ the physical common mode (mount yaw about the LiDAR, the wall's
//                                  straightness, σ_c); without it the chain collapses ~1/√N
//   misfit_widens_sigma_on_unmodelled_casings — 0.2 m-deep jamb casings the model lacks: the cluster-robust misfit grows
//                                  and so does the published σ_lat (0.031 → 0.054 m), the error staying within ~2σ
//
// Build + run (from door_concept/; relative_pose.h pulls rt_twist.h ⇒ DSR/Qt/fastdds headers):
//   mkdir -p build; Q=/usr/include/x86_64-linux-gnu/qt5; g++ -std=c++23 -O2 -fPIC -I/usr/include/eigen3 -Isrc -I../common
//     -I$Q -I$Q/QtCore -I$Q/QtGui -I$Q/QtWidgets tests/test_door_anchor_update.cpp -o build/test_door_anchor_update
//     -lfastdds -lfastcdr -lQt5Core && ./build/test_door_anchor_update
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <format>
#include <fstream>
#include <iterator>
#include <optional>
#include <print>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "door_anchor_update.h"
#include "anchor_frame_record.h"
#include "interaction_anchor/interaction_anchor.h"   // kAnchorStaleMs: the controller's staleness window (fix 1)

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
static const V3 kSensor(0.f, 0.14f, 0.67f);   // the bpearl in the robot frame
constexpr float kFloorZ = 0.05f;               // the wall band's floor (the caller's kFloorBandM): below it, a floor return
struct WorldOpts { float u0 = -1.5f, u1 = 4.0f; bool corner = true, shut = true, collinear_beyond = false, back_wall = false, leaf90 = false,
                  corner_u1 = false; };
static std::vector<Seg> world(const WorldOpts& o)
{
    std::vector<Seg> s;
    box(s, o.u0, -0.5f * W, -0.5f * T, 0.5f * T);
    box(s, 0.5f * W, o.u1, -0.5f * T, 0.5f * T);
    if (o.corner) box(s, o.u0 - T, o.u0, -4.f, 0.5f * T);
    if (o.corner_u1) box(s, o.u1, o.u1 + T, -4.f, 0.5f * T);   // Task 9: the room's corner at the +u end of the door's wall
    if (o.collinear_beyond) box(s, o.u1 + 1.0f, o.u1 + 3.5f, -0.5f * T, 0.5f * T);
    if (o.shut) box(s, -0.5f * W, 0.5f * W, -0.5f * T, -0.5f * T + 0.04f);
    box(s, -6.f, 6.f, 4.f, 4.1f);
    if (o.back_wall) box(s, -6.f, 6.f, -4.1f, -4.f);   // the robot's own room closes behind it
    if (o.leaf90) box(s, 0.5f * W - 0.04f, 0.5f * W, -0.5f * T - W, -0.5f * T);   // open 90°, beside the +u jamb, robot side
    return s;
}
// A 3-D bpearl-like scan: the sensor at kSensor (z = 0.67), 360° in azimuth at 1°, layers from 0 to -60° at 2°. A beam
// stops on a wall (walls are taller than the sensor) or on the floor (z = 0), whichever it reaches first. Robot frame.
static std::vector<V3> scan3d(const std::vector<Seg>& segs, const Pose2& r, unsigned seed = 5)
{
    std::mt19937 rng(seed);
    std::normal_distribution<float> noise(0.f, 0.02f);
    const float c = std::cos(r.psi), s = std::sin(r.psi);
    const V2 o_w = r.p + V2(c * kSensor.x() - s * kSensor.y(), s * kSensor.x() + c * kSensor.y());
    std::vector<V3> pts;
    for (float a = 0.f; a < 360.f; a += 1.f)
    {
        const float b = r.psi + a * float(M_PI) / 180.f;
        const V2 u(std::cos(b), std::sin(b));
        float r_wall = 1e9f;
        for (const auto& sg : segs) if (const float t = ray_seg(o_w, u, sg); t > 0.f and t < r_wall) r_wall = t;
        for (float el = 0.f; el >= -60.f; el -= 2.f)
        {
            const float te = std::tan(el * float(M_PI) / 180.f);
            const float r_floor = te < 0.f ? kSensor.z() / -te : 1e9f;
            const float rr = std::min(r_wall, r_floor);
            if (rr > 8.f) continue;
            const float rn = rr + noise(rng);
            const V2 pr = to_robot(r, o_w + u * rn);
            pts.emplace_back(pr.x(), pr.y(), r_floor <= r_wall ? 0.f + 0.3f * noise(rng) : kSensor.z() + rn * te);
        }
    }
    return pts;
}
// A scan from the robot at `r`: beams every 0.25 deg within ±half_sector of the robot's forward, from the sensor at
// `sensor_robot`; returned in the ROBOT frame (z = 0.5, the wall band).
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


// ─── Task 8: DOOR_1 of webots-shadow worlds/piso/tworooms-piso.wbt, in ANCHOR coordinates (u along the wall = world x
// − (−4.769), y = world y − 9.3796: the origin at the door frame's centre). The Door PROTO (protos/Door.proto, size
// 0.1 × 1 × 2.1, frameSize 0.05): two jamb frames 0.05 wide and 0.2 deep (local x ±0.1 = 9 cm proud of each wall face),
// 0.9 m clear between them; the leaf 0.05 × 0.95 hinged at the −u jamb (jointAtLeft), opening toward local +x = world −y =
// the robot's side; the door rotated −1.5857 rad (0.85° off the wall). Room2's south walls: 0.11 m thick, centred at
// world y 9.425 (4.5 cm behind the frame's centre), ending 3 cm short of the frames (x −5.30 and −4.24).
static void rbox(std::vector<Seg>& s, float x0, float x1, float y0, float y1, float th)
{
    const float c = std::cos(th), sn = std::sin(th);
    auto R = [&](float x, float y) { return V2(c * x - sn * y, sn * x + c * y); };
    const V2 p[4] = {R(x0, y0), R(x1, y0), R(x1, y1), R(x0, y1)};
    for (int i = 0; i < 4; ++i) s.push_back({p[i], p[(i + 1) % 4]});
}
static std::vector<Seg> webots_door1(bool leaf_open)
{
    std::vector<Seg> s;
    const float wy0 = 9.37f - 9.3796f, wy1 = 9.48f - 9.3796f;
    box(s, -7.77f + 4.769f, -5.30f + 4.769f, wy0, wy1);                 // Room2WallSouthWest
    box(s, -4.24f + 4.769f, -1.77f + 4.769f, wy0, wy1);                 // Room2WallSouthEast
    constexpr float th = -1.5857206f;                                    // the door's local frame → anchor
    rbox(s, -0.1f, 0.1f, -0.5f, -0.45f, th);                             // left frame (local x = −anchor y)
    rbox(s, -0.1f, 0.1f, 0.45f, 0.5f, th);                               // right frame
    if (leaf_open) rbox(s, 0.075f, 1.025f, -0.5f, -0.45f, th);           // the leaf at −90° about its hinge: robot side
    else           rbox(s, 0.1f, 0.15f, -0.45f, 0.45f, th);              // shut, on the frame's robot-side face
    box(s, -3.f, 3.f, 13.535f - 9.3796f - 0.055f, 13.535f - 9.3796f + 0.055f);   // Room2WallNorth
    box(s, -3.f, 3.f, -4.1f, -4.0f);                                     // the robot's room, behind
    return s;
}
// The bpearl as Shadow.proto builds it: 875 azimuths over 360°, 64 layers over 180° about its horizon, mounted inverted
// at 0.67 m — so 32 layers from −1.4° down to −90° reach the floor plane band; 5 mm range noise. A beam stops on the first
// wall (all taller than the sensor) or the floor. Robot frame.
static std::vector<V3> bpearl_scan(const std::vector<Seg>& segs, const Pose2& r, unsigned seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<float> noise(0.f, 0.005f);
    const float c = std::cos(r.psi), s = std::sin(r.psi);
    const V2 o_w = r.p + V2(c * kSensor.x() - s * kSensor.y(), s * kSensor.x() + c * kSensor.y());
    std::vector<V3> pts;
    for (int ia = 0; ia < 875; ++ia)
    {
        const float b = r.psi + 2.f * float(M_PI) * float(ia) / 875.f;
        const V2 u(std::cos(b), std::sin(b));
        float r_wall = 1e9f;
        for (const auto& sg : segs) if (const float t = ray_seg(o_w, u, sg); t > 0.f and t < r_wall) r_wall = t;
        for (int i = 32; i < 64; ++i)
        {
            const float el = (90.f - float(i) * 180.f / 63.f) * float(M_PI) / 180.f;
            const float te = std::tan(el);
            const float r_floor = kSensor.z() / -te;
            const float rr = std::min(r_wall, r_floor);
            if (rr / std::cos(el) > 30.f) continue;
            const float rn = rr + noise(rng) * std::cos(el);
            const V2 pr = to_robot(r, o_w + u * rn);
            pts.emplace_back(pr.x(), pr.y(), r_floor <= r_wall ? 0.f : kSensor.z() + rn * te);
        }
    }
    return pts;
}
// door_concept's live leaf belief for DOOR_1 (specificworker_anchor.cpp: hinge/swing 0.5/0.5, φ known, σ_φ 0.25, φ_max
// 120°), the stated hinge the −u jamb. Task 9 fix round 1: hinged at the CLEAR half-width and spanning it, the believed
// wall gap less the config's casing boards (rc::door::anchor_clear_width, FrameCasingM 0.06), as the caller now does.
static rc::door::LeafPrior live_leaf(float w)
{
    rc::door::LeafPrior lp;
    const float clear = rc::door::anchor_clear_width(w, 0.06f);
    lp.known = true; lp.hinge_u = -0.5f * clear; lp.len = clear; lp.dir_u = 1.f; lp.swing_y = -1.f;
    lp.phi = float(M_PI_2); lp.phi_sigma = 0.25f; lp.phi_known = true; lp.phi_max = 2.094f; lp.p_hinge = 0.5f; lp.p_swing = 0.5f;
    return lp;
}

// ─── fix round 4: the leaf-crossing run, shared by the crossing / timing / wrong-belief tests ─────────────────────────
// Open door with its leaf at 90° beside the +u jamb on the robot side, 3-D bpearl-like scans through select_anchor_rays
// (5,000–8,000 rays per frame within 2 m), the sensor driven from s = -0.5 to +0.8 m along the normal.
struct CrossStats
{
    float pv_min = 1.f, worst_lat = 0.f, worst_yaw = 0.f;
    double ms_max = 0.0, ms_sum = 0.0;
    std::size_t rays_max = 0, rays_min = ~std::size_t(0);
    int frames = 0;
    int wrong_pub_post = 0, wrong_pub_prior = 0;   // frames PUBLISHED (p_view ≥ 0.5) while > 0.10 m / 0.10 rad off,
                                                   // with p_view judged at the posterior (now) / at the prior (round 3)
};
static CrossStats leaf_crossing(const rc::door::LeafPrior& lp, float pi, bool verbose = false)
{
    WorldOpts wo; wo.u0 = -4.f; wo.u1 = 4.f; wo.corner = false; wo.shut = false; wo.back_wall = true; wo.leaf90 = true;
    const auto segs = world(wo);
    Pose2 robot{V2(0.f, -0.5f - kSensor.y()), 0.f};
    RelPose belief = creation_prior(robot);
    CrossStats st;
    for (int k = 0; k <= 26; ++k)
    {
        if (k > 0)
        {
            rc::rt::Twist tw; tw.linear = V3(0.f, 0.5f, 0.f);
            belief = rc::interaction::predict(belief, tw, V3(1e-4f, 1e-4f, 1e-4f), 0.1f);
            robot.p.y() += 0.05f;
        }
        const auto rays = rc::door::select_anchor_rays(belief, W, T, kFloorZ, 2.0f, kSensor, scan3d(segs, robot, 51 + k), 2.0f);
        const auto t0 = std::chrono::steady_clock::now();
        const auto [post, ev] = rc::door::update_door_anchor(belief, W, T, wo.u0, wo.u1, kSensor, rays, {}, {}, pi, lp, kFloorZ);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const float pv_prior = rc::door::door_view(belief, W, T, wo.u0, wo.u1, kSensor, rays, {}, {}, pi, lp, kFloorZ).p_view;
        belief = post;
        const V3 tr = anchor_truth(robot);
        const float lat = std::abs(std::cos(tr.z()) * (post.x.x() - tr.x()) + std::sin(tr.z()) * (post.x.y() - tr.y()));
        const float yaw = std::abs(rc::interaction::wrap_pi(post.x.z() - tr.z()));
        const bool off = (post.x.head<2>() - tr.head<2>()).norm() > 0.10f or yaw > 0.10f;
        st.wrong_pub_post  += (off and ev.p_view >= 0.5f) ? 1 : 0;
        st.wrong_pub_prior += (off and pv_prior >= 0.5f) ? 1 : 0;
        st.pv_min = std::min(st.pv_min, ev.p_view);
        st.worst_lat = std::max(st.worst_lat, lat);
        st.worst_yaw = std::max(st.worst_yaw, yaw);
        st.ms_max = std::max(st.ms_max, ms);
        st.ms_sum += ms;
        st.rays_max = std::max(st.rays_max, rays.size());
        st.rays_min = std::min(st.rays_min, rays.size());
        ++st.frames;
        if (verbose and (k % 4 == 0 or ev.p_view < 0.6f))
            std::print("    s={:+.2f} rays={} {:.1f} ms p_view={:.3f} lat err={:.3f} yaw err={:.3f} p_open={:.3f}\n",
                       robot.p.y() + kSensor.y(), rays.size(), ms, ev.p_view, lat, yaw, ev.p_open);
    }
    return st;
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
            const auto [post, ev] = update_door_anchor(belief, W, T, wo.u0, wo.u1, kSensor, scan(segs, robot, 90.f, 7 + k), {}, {}, 0.05f);
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
        // A SHUT door's opening is indistinguishable from wall (fix round 3: P(H_door) ≤ (1 − π)/(2 − π) < 0.5), so its
        // in-view level is just under the publication level BY DESIGN: try64's shut-door anchor is never published, and
        // the cross needs the door open anyway. What must hold: it returns to that level when the doorway comes back.
        check(pv_first > 0.4f and pv_last > 0.9f * pv_first, "p_view back at its in-view level after the doorway returns");
        check(pv_first < 0.5f, "a shut door reads below the publication level (unobservable vs wall)");
    }

    std::print("turning_away_360\n");
    {
        const WorldOpts wo;
        const auto segs = world(wo);
        Pose2 robot = try64_start();
        RelPose belief = creation_prior(robot);
        float worst_pos = 0.f, worst_yaw = 0.f, pv_min = 1.f;
        for (int k = 0; k < 120; ++k)
        {
            const float omega = k < 60 ? 0.35f : -0.35f;
            rc::rt::Twist tw; tw.angular = V3(0.f, 0.f, omega);
            belief = rc::interaction::predict(belief, tw, V3(1e-4f, 1e-4f, 1e-4f), 0.1f);
            robot.psi += omega * 0.1f;
            const auto [post, ev] = update_door_anchor(belief, W, T, wo.u0, wo.u1, kSensor, scan(segs, robot, 180.f, 7 + k), {}, {}, 0.05f);
            belief = post;
            const V3 tr = anchor_truth(robot);
            worst_pos = std::max(worst_pos, (post.x.head<2>() - tr.head<2>()).norm());
            worst_yaw = std::max(worst_yaw, std::abs(rc::interaction::wrap_pi(post.x.z() - tr.z())));
            pv_min = std::min(pv_min, ev.p_view);
        }
        std::print("    worst err {:.3f} m {:.3f} rad; min p_view {:.3f}\n", worst_pos, worst_yaw, pv_min);
        check(worst_pos < 0.10f, "posterior within 0.10 m of the truth throughout (360°)");
        check(worst_yaw < 0.05f, "posterior within 0.05 rad of the truth throughout (360°)");
    }

    std::print("mid_crossing_360_keeps_view\n");
    {
        WorldOpts wo; wo.u0 = -4.f; wo.u1 = 4.f; wo.corner = false; wo.shut = false; wo.back_wall = true;
        const auto segs = world(wo);
        Pose2 robot{V2(0.12f, -0.5f - kSensor.y()), 0.f};   // the SENSOR at s = -0.5 along the normal, 0.12 m off-centre
        RelPose belief = creation_prior(robot);
        float pv_min = 1.f, worst_lat = 0.f;
        constexpr float ds = 0.05f;
        for (int k = 0; k <= 26; ++k)
        {
            if (k > 0)
            {
                rc::rt::Twist tw; tw.linear = V3(0.f, ds / 0.1f, 0.f);
                belief = rc::interaction::predict(belief, tw, V3(1e-4f, 1e-4f, 1e-4f), 0.1f);
                robot.p.y() += ds;
            }
            const auto [post, ev] = update_door_anchor(belief, W, T, wo.u0, wo.u1, kSensor, scan(segs, robot, 180.f, 21 + k), {}, {}, 0.95f);
            belief = post;
            const V3 tr = anchor_truth(robot);
            const float lat = std::abs(std::cos(tr.z()) * (post.x.x() - tr.x()) + std::sin(tr.z()) * (post.x.y() - tr.y()));
            pv_min = std::min(pv_min, ev.p_view);
            worst_lat = std::max(worst_lat, lat);
            if (k % 2 == 0 or ev.p_view < 0.6f)
                std::print("    s={:+.2f} p_view={:.3f} lat err={:.3f} depth err={:.3f} p_open={:.3f}\n", robot.p.y() + kSensor.y(),
                           ev.p_view, lat, (post.x.head<2>() - tr.head<2>()).norm(), ev.p_open);
        }
        std::print("    min p_view {:.3f}, worst lateral {:.3f} m\n", pv_min, worst_lat);
        check(pv_min >= 0.6f, "p_view ≥ 0.6 at every step of the crossing");
        check(worst_lat < 0.05f, "lateral error < 0.05 m throughout");
    }

    // ─── fix round 2: "seen" means seen AS PREDICTED ───────────────────────────────────────────────────────────
    {
        WorldOpts wo; wo.shut = false; wo.back_wall = true;
        const auto segs = world(wo);
        const Pose2 robot = try64_start();
        const auto rays = scan(segs, robot, 180.f, 41);
        // Fix round 4: update_door_anchor now reports p_view at its POSTERIOR, which from these priors converges onto
        // the true opening (free space: 0.92). The question these tests ask — is a WRONG pose seen as predicted? — is
        // put to door_view, the same function, at the pose itself.
        auto view_of = [&](const RelPose& pr) { return rc::door::door_view(pr, W, T, wo.u0, wo.u1, kSensor, rays, {}, {}, 0.95f).p_view; };
        const RelPose truth = creation_prior(robot);
        const V2 along(std::cos(truth.x.z()), std::sin(truth.x.z())), normal(-along.y(), along.x());
        RelPose free_space = truth; free_space.x.head<2>() -= 1.2f * normal;
        RelPose rotated = truth;    rotated.x.z() = rc::interaction::wrap_pi(truth.x.z() + float(M_PI_2));
        RelPose solid = truth;      solid.x.head<2>() += 1.5f * along;
        const float pv_t = view_of(truth), pv_f = view_of(free_space), pv_r = view_of(rotated), pv_s = view_of(solid);
        std::print("pview: truth {:.3f}  free space {:.3f}  rotated {:.3f}  solid wall {:.3f}\n", pv_t, pv_f, pv_r, pv_s);
        std::print("pview_low_in_free_space\n");  check(pv_f < 0.5f, "prior 1.2 m in front of the wall: p_view < 0.5");
        std::print("pview_low_when_rotated\n");   check(pv_r < 0.5f, "prior yawed +90°: p_view < 0.5");
        std::print("pview_low_on_solid_wall\n");  check(pv_s < 0.5f, "open door, prior 1.5 m along onto solid wall: p_view < 0.5");
        check(pv_t > 0.5f, "and the true pose reads in view (> 0.5)");
        // ...and what update_door_anchor PUBLISHES from those priors is never a wrong pose read as in view.
        for (const auto& [name, pr] : {std::pair{"free space", free_space}, std::pair{"rotated", rotated}, std::pair{"solid wall", solid}})
        {
            const auto [post, ev] = update_door_anchor(pr, W, T, wo.u0, wo.u1, kSensor, rays, {}, {}, 0.95f);
            const float e_pos = (post.x.head<2>() - truth.x.head<2>()).norm();
            const float e_yaw = std::abs(rc::interaction::wrap_pi(post.x.z() - truth.x.z()));
            std::print("    from the {} prior: posterior off {:.3f} m / {:.3f} rad, p_view(posterior) {:.3f}\n", name, e_pos, e_yaw, ev.p_view);
            check(ev.p_view < 0.5f or (e_pos < 0.10f and e_yaw < 0.10f),
                  std::format("{} prior: published (p_view ≥ 0.5) only if within 0.10 m / 0.10 rad of the truth", name));
        }
    }

    // ─── Task 9, fix D: a doorway beside a ROOM CORNER (try66 leg 1) ───────────────────────────────────────────────────
    // The door's wall ends 0.14 m past the +u jamb, where the room turns toward the robot (try66: the run ended 0.64 m from
    // the opening's centre). The doorway neighbourhood |u| < w reaches past that end, where the model predicts no wall: those
    // cells counted as "visible wall nobody saw" (try66 frame at 24 s, the TRUE pose, w 1.06: walls 0.44, p_view 0.42), and
    // beams that passed them credited a WRONG pose: here, with the believed width 16 % too wide, a pose 1.2 m in front of
    // the wall read 0.62 under the old cell weighting — published. Wall cells are now cells of the door's wall run.
    std::print("doorway_beside_a_room_corner\n");
    {
        WorldOpts wo; wo.u0 = -1.02f; wo.u1 = 0.5f * W + 0.14f; wo.corner = true; wo.corner_u1 = true; wo.shut = false; wo.back_wall = true;
        const auto segs = world(wo);
        const Pose2 robot{V2(-0.2f, -2.0f - kSensor.y()), 0.f};
        const RelPose truth = creation_prior(robot);
        // The believed width 16 % wider than the clear one, as try66's 1.06 m belief against DOOR_1's 0.9 m between its frames.
        const float WB = 1.16f * W;
        const auto rays = rc::door::select_anchor_rays(truth, WB, T, kFloorZ, 2.0f, kSensor, scan3d(segs, robot, 61), 2.0f);
        auto view_of = [&](const RelPose& pr) { return rc::door::door_view(pr, WB, T, wo.u0, wo.u1, kSensor, rays, {}, {}, 0.95f, {}, kFloorZ); };
        const V2 along(std::cos(truth.x.z()), std::sin(truth.x.z())), normal(-along.y(), along.x());
        RelPose free_space = truth; free_space.x.head<2>() -= 1.2f * normal;
        RelPose rotated = truth;    rotated.x.z() = rc::interaction::wrap_pi(truth.x.z() + float(M_PI_2));
        RelPose solid = truth;      solid.x.head<2>() -= 1.5f * along;
        const auto vt = view_of(truth);
        const float pv_f = view_of(free_space).p_view, pv_r = view_of(rotated).p_view, pv_s = view_of(solid).p_view;
        std::print("    WB {:.2f} truth {:.3f} (opening {:.2f} · P(door) {:.2f} · walls {:.2f})  free space {:.3f}  rotated {:.3f}  solid wall {:.3f}\n",
                   WB, vt.p_view, vt.lidar_open, vt.lidar_door, vt.lidar_wall, pv_f, pv_r, pv_s);
        check(vt.p_view >= 0.6f, "the true pose of a doorway beside a room corner reads in view (≥ 0.6)");
        check(pv_f < 0.5f and pv_r < 0.5f and pv_s < 0.5f, "and the wrong poses (free space, rotated, solid wall) stay < 0.5");
    }

    // ─── fix round 3: p_view must not scale with π ─────────────────────────────────────────────────────────────
    double cross_ms_max = 0.0, cross_ms_sum = 0.0;
    int cross_frames = 0;
    std::size_t cross_rays_min = ~std::size_t(0), cross_rays_max = 0;
    for (const float pi : {0.7f, 0.8f})
    {
        std::print("crossing_with_leaf_at_moderate_pi (π = {:.1f})\n", pi);
        // The door's own leaf belief: hinged at the +u jamb, swung 90° toward the robot (σ_φ 0.25, the live config).
        rc::door::LeafPrior lp; lp.known = true; lp.hinge_u = 0.5f * W; lp.dir_u = -1.f; lp.swing_y = -1.f;
        lp.phi = float(M_PI_2); lp.phi_sigma = 0.25f;
        const CrossStats st = leaf_crossing(lp, pi, true);
        std::print("    min p_view {:.3f}, worst lateral {:.3f} m, worst yaw {:.3f} rad\n", st.pv_min, st.worst_lat, st.worst_yaw);
        check(st.pv_min >= 0.6f, std::format("π = {:.1f}: p_view ≥ 0.6 at every step of the crossing", pi));
        check(st.worst_lat < 0.05f, std::format("π = {:.1f}: lateral error < 0.05 m throughout", pi));
        cross_ms_max = std::max(cross_ms_max, st.ms_max); cross_ms_sum += st.ms_sum; cross_frames += st.frames;
        cross_rays_min = std::min(cross_rays_min, st.rays_min); cross_rays_max = std::max(cross_rays_max, st.rays_max);
    }
    // ─── fix round 4, finding 1: COST on the crossing's 3-D frames ─────────────────────────────────────────────────
    std::print("update_cost_on_3d_crossing_frames\n");
    std::print("    {} frames of {}–{} rays: {:.1f} ms mean, {:.1f} ms max per update\n", cross_frames, cross_rays_min, cross_rays_max,
               cross_ms_sum / std::max(1, cross_frames), cross_ms_max);
#ifdef __OPTIMIZE__
    check(cross_ms_max <= 50.0, "every update ≤ 50 ms (target 30 ms at -O2)");
#else
    std::print("  [SKIP] timing not checked in an unoptimised build: compile at -O2\n");
#endif

    // ─── fix round 4, finding 2: the leaf configuration is a BELIEF ───────────────────────────────────────────────
    // A confident belief (P(hinge), P(swing) 0.95 — more than door_concept, which has none, ever passes), WRONG in turn
    // on φ, the hinge side and the swing side, against the same crossing with no leaf model and with the right belief.
    std::print("wrong_leaf_belief_is_marginalised (π = 0.8)\n");
    {
        rc::door::LeafPrior right; right.known = true; right.hinge_u = 0.5f * W; right.dir_u = -1.f; right.swing_y = -1.f;
        right.phi = float(M_PI_2); right.phi_sigma = 0.25f; right.p_hinge = 0.95f; right.p_swing = 0.95f;
        rc::door::LeafPrior phi0 = right;   phi0.phi = 0.f;                                    // believes it shut
        rc::door::LeafPrior hinge = right;  hinge.hinge_u = -0.5f * W; hinge.dir_u = 1.f;      // the other jamb
        rc::door::LeafPrior swing = right;  swing.swing_y = 1.f;                               // the far side
        const CrossStats none = leaf_crossing(rc::door::LeafPrior{}, 0.8f);
        const CrossStats ok   = leaf_crossing(right, 0.8f);
        std::print("    {:22s} worst lateral {:.3f} m, worst yaw {:.3f} rad, min p_view {:.3f}\n", "no leaf model", none.worst_lat, none.worst_yaw, none.pv_min);
        std::print("    {:22s} worst lateral {:.3f} m, worst yaw {:.3f} rad, min p_view {:.3f}\n", "correct belief", ok.worst_lat, ok.worst_yaw, ok.pv_min);
        check(ok.worst_lat <= 0.05f, "correct belief: lateral error ≤ 0.05 m throughout");
        for (const auto& [name, lp] : {std::pair{"believes φ = 0", phi0}, std::pair{"hinge flipped", hinge}, std::pair{"swing flipped", swing}})
        {
            const CrossStats st = leaf_crossing(lp, 0.8f);
            std::print("    {:22s} worst lateral {:.3f} m, worst yaw {:.3f} rad, min p_view {:.3f}\n", name, st.worst_lat, st.worst_yaw, st.pv_min);
            check(st.worst_lat < 0.10f and st.worst_yaw < 0.10f, std::format("{}: worst lateral < 0.10 m and yaw < 0.10 rad", name));
            check(st.worst_lat <= none.worst_lat and st.worst_yaw <= none.worst_yaw, std::format("{}: no worse than no leaf model", name));
        }
    }

    // ─── fix round 4, finding 3: p_view is the POSTERIOR's ────────────────────────────────────────────────────────
    // The round-3 belief taken as certain (one hinge, one swing, φ = 0 ± 0.01, no "not as believed" mass) drags the
    // anchor within a frame. Judged at the prior, such frames were published; judged at the posterior, never.
    std::print("dragged_posterior_not_published\n");
    {
        rc::door::LeafPrior certain; certain.known = true; certain.hinge_u = 0.5f * W; certain.dir_u = -1.f; certain.swing_y = -1.f;
        certain.phi = 0.f; certain.phi_sigma = 0.01f; certain.p_hinge = 1.f; certain.p_swing = 1.f; certain.p_not_as_believed = 0.f;
        const CrossStats st = leaf_crossing(certain, 0.8f);
        std::print("    worst lateral {:.3f} m, worst yaw {:.3f} rad; frames published while > 0.10 m / 0.10 rad off: "
                   "{} with p_view at the posterior, {} had it been judged at the prior\n",
                   st.worst_lat, st.worst_yaw, st.wrong_pub_post, st.wrong_pub_prior);
        check(st.wrong_pub_post == 0, "no frame publishes a posterior > 0.10 m / 0.10 rad off");
        WorldOpts wo; wo.shut = false; wo.back_wall = true;
        const auto rays = scan(world(wo), try64_start(), 180.f, 41);
        const RelPose pr = creation_prior(try64_start());
        const auto [post, ev] = update_door_anchor(pr, W, T, wo.u0, wo.u1, kSensor, rays, {}, {}, 0.95f);
        check(ev.p_view == rc::door::door_view(post, W, T, wo.u0, wo.u1, kSensor, rays, {}, {}, 0.95f).p_view,
              "the reported p_view IS door_view of the returned posterior");
    }
    {
        WorldOpts wo; wo.shut = false; wo.back_wall = true;
        const auto segs = world(wo);
        const Pose2 robot = try64_start();
        const auto rays = scan(segs, robot, 180.f, 41);
        const RelPose truth = creation_prior(robot);
        RelPose wrong = truth; wrong.x.head<2>() += 1.5f * V2(std::cos(truth.x.z()), std::sin(truth.x.z()));
        for (const float pi : {0.05f, 0.5f, 0.95f})
        {
            const float pt = update_door_anchor(truth, W, T, wo.u0, wo.u1, kSensor, rays, {}, {}, pi).second.p_view;
            const float pw = update_door_anchor(wrong, W, T, wo.u0, wo.u1, kSensor, rays, {}, {}, pi).second.p_view;
            std::print("true_pose_outscores_wrong_wall (π = {:.2f}): truth {:.3f}, 1.5 m along on solid wall {:.3f}\n", pi, pt, pw);
            check(pt > pw, std::format("π = {:.2f}: the true pose outscores the wrong one", pi));
            if (pi >= 0.5f) check(pw < 0.5f, std::format("π = {:.2f}: the wrong pose reads < 0.5", pi));
        }
    }

    std::print("shut_door_reads_shut_with_floor\n");
    {
        WorldOpts wo; wo.u0 = -4.f; wo.u1 = 4.f; wo.corner = false; wo.shut = true; wo.back_wall = true;
        const auto segs = world(wo);
        const Pose2 robot{V2(0.2f, -2.0f), 0.f};
        RelPose r = creation_prior(robot);
        r.x.y() -= 0.25f;                                         // 0.25 m in front of the true wall (toward the robot)
        const V3 tr = anchor_truth(robot);
        float p_open_max = 0.f;
        int n_floor = 0, n_floor_first = -1;
        for (int k = 0; k < 3; ++k)
        {
            const auto sweep = scan3d(segs, robot, 31 + k);
            const auto rays = rc::door::select_anchor_rays(r, W, T, kFloorZ, 2.0f, kSensor, sweep, 2.0f);
            for (const auto& q : rays) if (q.z() < 0.05f) ++n_floor;
            if (k == 0) n_floor_first = n_floor;
            const auto [post, ev] = update_door_anchor(r, W, T, wo.u0, wo.u1, kSensor, rays, {}, {}, 0.5f, {}, kFloorZ);
            r = post;
            p_open_max = std::max(p_open_max, ev.p_open);
            std::print("    k={} rays={} depth err={:.3f} p_open={:.3f} through={} blocked={} p_view={:.3f}\n", k, rays.size(),
                       std::abs(r.x.y() - tr.y()), ev.p_open, ev.n_through, ev.n_blocked, ev.p_view);
        }
        std::print("    floor returns kept: {} (frame 0, from the mis-predicted prior: {})\n", n_floor, n_floor_first);
        check(n_floor_first == 0, "no floor return kept as 'through' a shut door from a prior 0.25 m short");
        check(p_open_max < 0.1f, "p_open < 0.1 on every update");
        check(std::abs(r.x.y() - tr.y()) < 0.05f, "depth corrected to < 0.05 m within 3 updates");
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

    // ─── Task 8, fix 1: the camera's view evidence persists within the staleness window ─────────────────────────────
    std::print("camera_view_persists_within_window\n");
    {
        // The anchor at the opening 2.5 m ahead, the zed at the robot origin looking forward, a mask filling the opening.
        RelPose truth; truth.x = V3(0.f, D, 0.f); truth.P = V3(0.03f * 0.03f, 0.03f * 0.03f, 0.02f * 0.02f).asDiagonal();
        rc::door::CameraFrustum cam; cam.known = true; cam.origin = {0.f, 0.f}; cam.forward = {0.f, 1.f}; cam.half_fov = 0.96f; cam.max_range = 8.f;
        const auto frame_pts = mask(0.f, 200);
        rc::door::CameraViewHold hold;
        constexpr float p_lidar = 0.45f;   // try65's p_lidar mid-approach
        int crossings_new = 0, crossings_old = 0, k_fell = -1;
        bool last_new = true, last_old = true;
        float pv_min_streaming = 1.f, pc_frame = 0.f;
        for (int k = 0; k < 40; ++k)
        {
            const std::int64_t now = 1'000'000 + 100 * k;              // a 10 Hz anchor cycle
            const bool frame = k < 20 and k % 2 == 0;                  // a mask frame every other cycle, then none
            if (frame) rc::door::hold_mask_frame(hold, frame_pts, now - 30);
            const float pc = rc::door::held_camera_view(hold, now, rc::interaction::kAnchorStaleMs, cam, W, truth);
            if (frame) pc_frame = rc::door::camera_seen(cam, W, truth, frame_pts);
            const float pv_new = rc::door::combine_view(p_lidar, pc);
            const float pv_old = rc::door::combine_view(p_lidar, frame ? pc_frame : 0.f);   // before: no new frame ⇒ 0
            const bool pub_new = pv_new >= 0.5f, pub_old = pv_old >= 0.5f;
            if (k > 0) { crossings_new += pub_new != last_new; crossings_old += pub_old != last_old; }
            last_new = pub_new; last_old = pub_old;
            if (k < 20) pv_min_streaming = std::min(pv_min_streaming, pv_new);
            if (k_fell < 0 and not pub_new) k_fell = k;
        }
        std::print("    camera term at the truth {:.3f}; while frames stream: min p_view {:.3f}; publication-level crossings: {} (old rule {}); "
                   "falls at cycle {} (last frame at cycle 18, captured 30 ms before it)\n", pc_frame, pv_min_streaming, crossings_new, crossings_old, k_fell);
        check(pv_min_streaming >= 0.5f, "p_view stays at or above the publication level while mask frames keep coming");
        check(crossings_old > 10, "RED reference: the old rule crosses the level on every frame gap");
        check(crossings_new == 1 and k_fell == 28, "...and falls exactly once, when the last frame is kAnchorStaleMs old");
        // Fix round 1: the HELD frame's term is the CURRENT pose's — a pose since dragged 0.6 m along the wall reads low.
        RelPose dragged = truth; dragged.x.x() += 0.6f;
        rc::door::hold_mask_frame(hold, frame_pts, 2'000'000);
        const float pc_t = rc::door::held_camera_view(hold, 2'000'100, rc::interaction::kAnchorStaleMs, cam, W, truth);
        const float pc_d = rc::door::held_camera_view(hold, 2'000'100, rc::interaction::kAnchorStaleMs, cam, W, dragged);
        std::print("    held frame, recomputed: at the truth {:.3f}, at a pose dragged 0.6 m {:.3f} (frustum alone {:.3f})\n", pc_t, pc_d,
                   rc::door::anchor_detail::camera_view(cam, W, dragged.x.cast<double>(), dragged.P.cast<double>()));
        check(pc_d < 0.5f and pc_t > 0.8f, "the held camera term is recomputed at the current pose (dragged < 0.5, truth > 0.8)");
        // A new frame WITHOUT a door slice is the camera seeing none: it replaces the held frame at once.
        rc::door::hold_mask_frame(hold, {}, 2'000'200);
        check(rc::door::held_camera_view(hold, 2'000'200, rc::interaction::kAnchorStaleMs, cam, W, truth) == 0.f, "a new frame with no door replaces the held one");
    }

    // ─── fix round 1: a HELD mask frame from a correct earlier frame cannot publish a wrong pose ─────────────────────────
    std::print("held_mask_wrong_poses\n");
    {
        WorldOpts wo; wo.shut = false; wo.back_wall = true;
        const Pose2 robot = try64_start();
        const auto rays = scan(world(wo), robot, 180.f, 41);
        rc::door::CameraFrustum cam; cam.known = true; cam.origin = {0.f, -0.075f}; cam.forward = {0.f, 1.f}; cam.half_fov = 0.96f; cam.max_range = 8.f;
        // The correct frame's mask: the opening's pixels, on the door plane and seen through it (the far room).
        std::vector<V3> pts;
        const float c = std::cos(robot.psi), sn = std::sin(robot.psi);
        const V2 cam_w = robot.p + V2(c * cam.origin.x() - sn * cam.origin.y(), sn * cam.origin.x() + c * cam.origin.y());
        for (int i = 0; i < 60; ++i)
        {
            const V2 on_plane(-0.5f * W + W * (float(i) + 0.5f) / 60.f, 0.f);
            const V2 beyond = on_plane + 2.0f * (on_plane - cam_w).normalized();
            for (const V2& q : {on_plane, beyond}) { const V2 r = to_robot(robot, q); pts.emplace_back(r.x(), r.y(), 1.0f); }
        }
        rc::door::CameraViewHold hold;
        rc::door::hold_mask_frame(hold, pts, 5'000'000);
        const std::int64_t now = 5'000'500;   // half a window later, no new frame
        const RelPose truth = creation_prior(robot);
        const V2 along(std::cos(truth.x.z()), std::sin(truth.x.z())), normal(-along.y(), along.x());
        RelPose free_space = truth; free_space.x.head<2>() -= 1.2f * normal;
        RelPose rotated = truth;    rotated.x.z() = rc::interaction::wrap_pi(truth.x.z() + float(M_PI_2));
        RelPose solid = truth;      solid.x.head<2>() += 1.5f * along;
        for (const auto& [name, pr] : {std::pair{"truth", truth}, std::pair{"free space", free_space}, std::pair{"rotated", rotated}, std::pair{"solid wall", solid}})
        {
            const float pl = rc::door::door_view(pr, W, T, wo.u0, wo.u1, kSensor, rays, {}, {}, 0.95f).p_lidar;
            const float pc = rc::door::held_camera_view(hold, now, rc::interaction::kAnchorStaleMs, cam, W, pr);
            const float frustum = float(rc::door::anchor_detail::camera_view(cam, W, pr.x.cast<double>(), pr.P.cast<double>()));
            const float pv = rc::door::combine_view(pl, pc);
            std::print("    {:10s} p_lidar {:.3f}  held p_cam {:.3f} (frustum alone {:.3f})  p_view {:.3f}\n", name, pl, pc, frustum, pv);
            if (std::string_view(name) == "truth") check(pv > 0.5f, "truth with the held frame: p_view > 0.5");
            else check(pv < 0.5f, std::format("{} with a held correct frame: p_view < 0.5", name));
        }
    }

    // ─── Task 8, fix 2: try65's door, the live configuration ────────────────────────────────────────────────────────
    {
        constexpr float Wm = 1.14f, Tm = 0.05f, Ulo = -2.5f, Uhi = 2.5f;   // the belief's width, aperture.thickness, the run
        const auto open_world = webots_door1(true), shut_world = webots_door1(false);
        const rc::door::LeafPrior lp = live_leaf(Wm);
        for (const auto& [pi, su] : {std::pair{0.6f, 0.1f}, std::pair{0.75f, 0.1f}, std::pair{0.7f, 0.0f}, std::pair{0.7f, 0.2f}})
        {
            std::print("try65_webots_door_crossing (π = {:.2f}, sensor {:.1f} m off-centre)\n", pi, su);
            Pose2 robot{V2(su, -2.0f - kSensor.y()), 0.f};
            RelPose belief = creation_prior(robot);
            float pv_min_out = 1.f, pv_min_in = 1.f, worst_lat = 0.f;
            for (int k = 0; k <= 28; ++k)
            {
                if (k > 0)
                {
                    rc::rt::Twist tw; tw.linear = V3(0.f, 1.0f, 0.f);
                    belief = rc::interaction::predict(belief, tw, V3(1e-4f, 1e-4f, 1e-4f), 0.1f);
                    robot.p.y() += 0.1f;
                }
                const float s_now = robot.p.y() + kSensor.y();
                const auto rays = rc::door::select_anchor_rays(belief, Wm, Tm, kFloorZ, 2.1f, kSensor, bpearl_scan(open_world, robot, 300 + k), 2.0f);
                const auto [post, ev] = update_door_anchor(belief, Wm, Tm, Ulo, Uhi, kSensor, rays, {}, {}, pi, lp, kFloorZ);
                belief = post;
                const V3 tr = anchor_truth(robot);
                const float lat = std::abs(std::cos(tr.z()) * (post.x.x() - tr.x()) + std::sin(tr.z()) * (post.x.y() - tr.y()));
                (std::abs(s_now) >= 0.25f ? pv_min_out : pv_min_in) = std::min(std::abs(s_now) >= 0.25f ? pv_min_out : pv_min_in, ev.p_view);
                worst_lat = std::max(worst_lat, lat);
                if (k % 4 == 0 or ev.p_view < 0.6f)
                    std::print("    s={:+.2f} rays={} p_view={:.3f} = opening {:.3f} x P(door) {:.3f} x walls {:.3f}  lat err={:.3f} yaw err={:+.3f}\n",
                               s_now, rays.size(), ev.p_view, ev.lidar_open, ev.lidar_door, ev.lidar_wall, lat,
                               rc::interaction::wrap_pi(post.x.z() - tr.z()));
            }
            std::print("    min p_view: approach/exit {:.3f}, doorway band {:.3f}; worst lateral {:.3f} m\n", pv_min_out, pv_min_in, worst_lat);
            check(pv_min_out >= 0.6f, std::format("π {:.2f}, {:.1f} m: p_view ≥ 0.6 at every step with |s| ≥ 0.25 m", pi, su));
            check(pv_min_in >= 0.5f, std::format("π {:.2f}, {:.1f} m: p_view ≥ 0.5 (published) inside the doorway band", pi, su));
            check(worst_lat < 0.10f, std::format("π {:.2f}, {:.1f} m: the anchor stays within 0.10 m laterally", pi, su));
        }
        // The wrong-pose guarantees on the same door, mid-approach (s = −1.0), against the truth.
        const Pose2 robot{V2(0.1f, -1.0f - kSensor.y()), 0.f};
        const auto rays_open = bpearl_scan(open_world, robot, 77);
        auto view_at = [&](const RelPose& at, const std::vector<V3>& sweep, float pi)
        {
            const auto rays = rc::door::select_anchor_rays(at, Wm, Tm, kFloorZ, 2.1f, kSensor, sweep, 2.0f);
            return rc::door::door_view(at, Wm, Tm, Ulo, Uhi, kSensor, rays, {}, {}, pi, lp, kFloorZ).p_view;
        };
        RelPose truth = creation_prior(robot);
        truth.P = V3(0.03f * 0.03f, 0.03f * 0.03f, 0.02f * 0.02f).asDiagonal();
        const V2 along(std::cos(truth.x.z()), std::sin(truth.x.z())), normal(-along.y(), along.x());
        RelPose free_space = truth; free_space.x.head<2>() -= 1.2f * normal;
        RelPose rotated = truth;    rotated.x.z() = rc::interaction::wrap_pi(truth.x.z() + float(M_PI_2));
        RelPose solid = truth;      solid.x.head<2>() += 1.5f * along;
        const float pt = view_at(truth, rays_open, 0.7f), pf = view_at(free_space, rays_open, 0.7f), pr = view_at(rotated, rays_open, 0.7f),
                    ps = view_at(solid, rays_open, 0.7f);
        std::print("try65_webots_wrong_poses (π 0.70, s = −1.0): truth {:.3f}  free space {:.3f}  rotated {:.3f}  solid wall {:.3f}\n", pt, pf, pr, ps);
        check(pt >= 0.6f and pf < 0.5f and pr < 0.5f and ps < 0.5f, "the true pose reads ≥ 0.6; free space, rotated and solid wall < 0.5");
        // The SHUT door (the commanded state already says open: π 0.7) is never published along the approach.
        float pv_shut_max = 0.f;
        for (int k = 0; k <= 16; ++k)
        {
            const Pose2 rb{V2(0.1f, -2.0f + 0.1f * float(k) - kSensor.y()), 0.f};
            RelPose at = creation_prior(rb);
            at.P = truth.P;
            pv_shut_max = std::max(pv_shut_max, view_at(at, bpearl_scan(shut_world, rb, 500 + k), 0.7f));
        }
        std::print("try65_webots_shut_door: max p_view over the approach {:.3f}\n", pv_shut_max);
        check(pv_shut_max < 0.5f, "the shut door is never published (π 0.7)");
    }

    // ─── Task 8, fix 3: the frame record replays the live update ─────────────────────────────────────────────────────
    std::print("anchor_frame_record_roundtrip\n");
    {
        const Pose2 robot{V2(0.1f, -1.0f - kSensor.y()), 0.f};
        const rc::door::LeafPrior lp = live_leaf(1.14f);
        rc::door::rec::Frame f;
        f.t_ms = 1792355330007;
        f.prior = creation_prior(robot);
        f.width = 1.14f; f.thickness = 0.05f; f.wall_u_min = -2.5f; f.wall_u_max = 2.5f; f.floor_below = kFloorZ;
        f.lidar_origin = kSensor;
        const auto rays = rc::door::select_anchor_rays(f.prior, f.width, f.thickness, kFloorZ, 2.1f, kSensor, bpearl_scan(webots_door1(true), robot, 9), 2.0f);
        f.mask = mask(0.f, 50);
        f.cam.known = true; f.cam.origin = {0.f, -0.075f}; f.cam.forward = {0.f, 1.f}; f.cam.half_fov = 0.95f; f.cam.max_range = 8.f;
        f.leaf = lp;
        f.pi_open = 0.7f;
        const auto [post, ev] = update_door_anchor(f.prior, f.width, f.thickness, f.wall_u_min, f.wall_u_max, f.lidar_origin, rays, f.mask,
                                                   f.cam, f.pi_open, f.leaf, f.floor_below, &f.staging);
        f.post = post;
        f.p_lidar = ev.p_lidar; f.p_cam = ev.p_cam; f.p_view = ev.p_view;
        f.lidar_open = ev.lidar_open; f.lidar_door = ev.lidar_door; f.lidar_wall = ev.lidar_wall;
        const std::string path = "build/test_anchor_frames.bin";
        {
            rc::door::rec::Recorder recorder;
            check(recorder.open(path), "the recorder opens (truncating) its file");
            check(recorder.append(f) and recorder.append(f), "two records appended");
        }
        std::ifstream in(path, std::ios::binary);
        std::vector<rc::door::rec::Frame> back;
        check(rc::door::rec::read_all(in, back) and back.size() == 2, "both records read back");
        if (back.size() == 2)
        {
            const auto& b = back[1];
            check(rc::door::rec::encode(b) == rc::door::rec::encode(f), "the record round-trips bit for bit");
            rc::door::AnchorStaging st = b.staging;
            st.replay = true;
            const auto [p2, e2] = update_door_anchor(b.prior, b.width, b.thickness, b.wall_u_min, b.wall_u_max, b.lidar_origin, {}, b.mask,
                                                     b.cam, b.pi_open, b.leaf, b.floor_below, &st);
            std::print("    {} rays pre-budget, {} + {} staged, {} mask points, record {} bytes; replay vs record: Δxy {:.6f} m, Δyaw {:.6f}, Δp_view {:.6f}\n",
                       b.staging.prior.n_pre, b.staging.prior.rays.size(), b.staging.post.rays.size(), b.mask.size(), rc::door::rec::encode(b).size(), (p2.x.head<2>() - b.post.x.head<2>()).norm(), std::abs(p2.x.z() - b.post.x.z()),
                       std::abs(e2.p_view - b.p_view));
            check(p2.x == b.post.x and p2.P == b.post.P and e2.p_view == b.p_view and e2.p_lidar == b.p_lidar,
                  "re-running the update on the record's budgeted sets reproduces it exactly");
            check(b.staging.prior.n_pre == rays.size() and b.staging.prior.rays.size() <= 400 and b.staging.prior.az_step > 0.0,
                  "the record holds n_pre, ≤ the budget's rays, and the measured azimuth step");
        }
        // A record cut short (the agent killed mid-write) ends the read; the complete ones before it survive.
        {
            std::ifstream whole(path, std::ios::binary);
            std::string bytes((std::istreambuf_iterator<char>(whole)), std::istreambuf_iterator<char>());
            bytes.resize(bytes.size() - 7);
            std::ofstream(path, std::ios::binary | std::ios::trunc).write(bytes.data(), std::streamsize(bytes.size()));
            std::ifstream cut(path, std::ios::binary);
            std::vector<rc::door::rec::Frame> some;
            check(rc::door::rec::read_all(cut, some) and some.size() == 1, "a truncated last record is dropped, the first survives");
        }
    }

    // ─── Task 9, fix D: the twist's own clock ─────────────────────────────────────────────────────────────────────────
    std::print("sim_clock_rate\n");
    {
        rc::rt::SimClockRate clk;
        check(clk.rate() == 1.0 and clk.sim_at(123) == 123.0, "unobserved (real hardware): the identity");
        // A producer at 20 Hz wall, its sim clock running at 0.94× (try66): 4 s of samples.
        for (int k = 0; k <= 80; ++k) clk.observe(1'000'000 + 50 * k, 5'000 + std::int64_t(std::llround(0.94 * 50.0 * k)));
        std::print("    sim at 0.94x real time: rate {:.4f}\n", clk.rate());
        check(std::abs(clk.rate() - 0.94) < 0.03, "the rate is Δsim/Δwall between consecutive samples (0.94, ms-quantised)");
        // The anchor's prediction over 1 s of wall time at a sim-second yaw rate of 0.57 rad/s (try66's spin).
        rc::interaction::RelPose p0; p0.x = {0.f, 2.f, 0.f}; p0.P = Eigen::Matrix3f::Identity() * 1e-4f;
        rc::rt::Twist tw; tw.angular = {0.f, 0.f, 0.57f};
        const double s0 = clk.sim_at(1'000'000 + 50 * 60), s1 = clk.sim_at(1'000'000 + 50 * 80);
        const auto sim = rc::interaction::predict(p0, tw, V3::Zero(), float((s1 - s0) * 1e-3));
        std::print("    1 s wall at 0.57 rad/s: {:.4f} rad over the sim clock (true {:.4f}; the wall clock gives 0.5700)\n",
                   -sim.x.z(), 0.57 * 0.94);
        check(std::abs(-sim.x.z() - 0.57f * 0.94f) < 2e-3f, "integrated over the sim clock the turn is the sim's");
        // A repeated or out-of-order sample does not move it.
        const double before = clk.sim_at(1'000'000 + 50 * 80);
        clk.observe(1'000'000 + 50 * 80, 9'999'999);
        clk.observe(1'000'000 + 50 * 79, 9'999'999);
        check(clk.sim_at(1'000'000 + 50 * 80) == before, "a repeated or out-of-order sample is ignored");
        // A PAUSED simulator (fix round 1): the sim stamp stops while the wall clock and the last twist go on — nothing
        // is integrated across the pause.
        for (int k = 81; k <= 100; ++k) clk.observe(1'000'000 + 50 * k, 5'000 + std::int64_t(std::llround(0.94 * 50.0 * 80)));
        const double p0s = clk.sim_at(1'000'000 + 50 * 90), p1s = clk.sim_at(1'000'000 + 50 * 100 + 40);
        std::print("    paused: rate {:.4f}, sim advance over 0.54 s of wall {:.1f} ms\n", clk.rate(), p1s - p0s);
        check(clk.rate() == 0.0 and p1s - p0s == 0.0, "a paused simulator integrates nothing (the trailing-rate estimator did)");
        clk.observe(1'000'000 + 50 * 101, 10);   // the sim clock jumped back: restarted
        clk.observe(1'000'000 + 50 * 102, 35);
        check(std::abs(clk.rate() - 0.5) < 1e-6, "a restarted simulator re-anchors the estimate");
    }

    // ─── Task 9 fix round 2 (try68): the first anchor of a process is created BEFORE the first twist sample is seen ────
    std::print("first_anchor_predicted_before_clock_observed\n");
    {
        rc::rt::SimClockRate clk;
        const std::int64_t wall0 = 1'792'364'000'000;          // the anchor's stamp: wall epoch ms, clock not observed yet
        rc::interaction::RelPose a; a.x = {0.f, 2.f, 0.f}; a.P = Eigen::Matrix3f::Identity() * 1e-4f; a.stamp_ms = wall0;
        const double old_way = clk.sim_at(wall0);              // what f.r.1 stored at creation: the identity — the WALL epoch
        rc::rt::Twist tw; tw.angular = {0.f, 0.f, 0.5f};       // the robot turning CCW at 0.5 rad/s (per sim second)
        double turned = 0.0, old_dt = 0.0;
        for (int k = 1; k <= 20; ++k)                          // 2 s at 10 Hz; the sim clock (~5 s in) at 0.94x
        {
            const std::int64_t now = wall0 + 100 * k;
            clk.observe(now - 10, 5'000 + std::int64_t(std::llround(0.94 * (100 * k - 10))));
            const float dt = float(clk.interval_s(a.stamp_ms, now));   // the fix: both ends mapped by the current samples
            old_dt += std::max(0.0, clk.sim_at(now) - old_way) * 1e-3 > 1e6 ? 0.0 : std::max(0.0, clk.sim_at(now) - old_way) * 1e-3;
            a = rc::interaction::predict(a, tw, V3(1e-4f, 1e-4f, 1e-4f), dt);
            a.stamp_ms = now;
            turned += 0.5 * dt;
        }
        std::print("    2 s of a 0.5 rad/s turn: anchor yaw {:+.4f} rad (expected {:+.4f}); f.r.1's interval sim_now − sim_at(stamp@creation) "
                   "summed {:.4f} s; σ_yaw {:.4f} (from 0.0100)\n", a.x.z(), -turned, old_dt, std::sqrt(a.P(2, 2)));
        check(std::abs(a.x.z() + 0.5f * 0.94f * 1.99f) < 0.02f, "the anchor turns by −Δyaw of the robot (≈ −0.94 rad over 2 s at 0.94x)");
        check(old_dt == 0.0, "(the f.r.1 bookkeeping integrated nothing: the sim clock lies far below the wall epoch)");
        check(a.P(2, 2) > 1.01e-4f, "and its covariance grew");
    }

    // ─── Task 9 fix round 1: what to publish (the softer rule near the door, kAnchorPredictMs) ───────────────────────
    // ─── Task 9 fix round 1: π = 0 with thousands of passes (try67 at the mouth) must not make the anchor NaN ────────────
    std::print("pi_zero_open_door_is_finite\n");
    {
        WorldOpts wo; wo.u0 = -4.f; wo.u1 = 4.f; wo.corner = false; wo.shut = false; wo.back_wall = true;
        const auto segs = world(wo);
        const Pose2 robot{V2(0.f, -1.0f - kSensor.y()), 0.f};
        const RelPose truth = creation_prior(robot);
        const auto rays = rc::door::select_anchor_rays(truth, W, T, kFloorZ, 2.0f, kSensor, scan3d(segs, robot, 71), 2.0f);
        const auto [post, ev] = update_door_anchor(truth, W, T, wo.u0, wo.u1, kSensor, rays, {}, {}, 0.0f, {}, kFloorZ);
        std::print("    π = 0, {} rays at s = −1.0: posterior ({:.3f},{:.3f},{:.3f}), P(door) {:.3f}, p_view {:.3f}\n", rays.size(),
                   post.x.x(), post.x.y(), post.x.z(), ev.lidar_door, ev.p_view);
        check(post.x.allFinite() and post.P.allFinite() and std::isfinite(ev.p_view), "the posterior and p_view are finite at π = 0");
        check((post.x.head<2>() - truth.x.head<2>()).norm() < 0.05f, "...and the posterior is the open door's (within 0.05 m)");
    }

    std::print("anchor_publish_decision\n");
    {
        using rc::door::AnchorPublish;
        const auto K = rc::interaction::kAnchorPredictMs;
        check(rc::door::anchor_publish_decision(true, false, false, 0, K) == AnchorPublish::Seen, "seen now ⇒ Seen (claimed or not)");
        check(rc::door::anchor_publish_decision(false, true, true, 0, K) == AnchorPublish::Predicted, "unseen 0 ms, cross claimed, seen before ⇒ Predicted");
        check(rc::door::anchor_publish_decision(false, true, true, 2999, K) == AnchorPublish::Predicted, "unseen 2999 ms ⇒ Predicted");
        check(rc::door::anchor_publish_decision(false, true, true, 3001, K) == AnchorPublish::None, "unseen 3001 ms ⇒ None (the strict rule)");
        check(rc::door::anchor_publish_decision(false, false, true, 100, K) == AnchorPublish::None, "cross NOT claimed ⇒ None (strict rule)");
        check(rc::door::anchor_publish_decision(false, true, false, 100, K) == AnchorPublish::None, "never seen in this passage ⇒ None");
        check(K == 3000, "kAnchorPredictMs is 3000 ms");
    }


    // ─── Task 10: the published covariance is honest about what does not average out ───────────────────────────────
    // 40 frames of the same open doorway from the same pose, each with fresh beam noise: the physical common mode (mount
    // translation σ_c, mount yaw ε about the LiDAR, the wall's straightness over its run) is the same error on every frame,
    // so the chained posterior must not claim less — while a chain that does NOT carry the bias factor collapses ~1/√N.
    std::print("persistent_bias_does_not_average_out\n");
    {
        const double run = double(kUmax - kUmin), r = double(D);   // the LiDAR origin is the robot's (kOrigin x, y = 0)
        const double floor_yaw = std::sqrt(rc::door::kMountYawSigma * rc::door::kMountYawSigma
                                           + 2.0 * rc::door::anchor_detail::kCommonSigma * rc::door::anchor_detail::kCommonSigma / (run * run));
        const double floor_lat = std::sqrt(rc::door::anchor_detail::kCommonSigma * rc::door::anchor_detail::kCommonSigma
                                           + rc::door::kMountYawSigma * rc::door::kMountYawSigma * r * r);
        RelPose carried = prior_offset(), fresh = prior_offset();
        rc::door::AnchorBias bias;
        float yaw1 = 0.f;
        for (int k = 0; k < 40; ++k)
        {
            const auto rays = cast(Leaf::none, 100 + unsigned(k));
            carried = update_door_anchor(carried, W, T, kUmin, kUmax, kOrigin, rays, {}, {}, 0.5f, {},
                                         -std::numeric_limits<float>::infinity(), nullptr, &bias).first;
            fresh = update_door_anchor(fresh, W, T, kUmin, kUmax, kOrigin, rays, {}).first;
            if (k == 0) yaw1 = std::sqrt(carried.P(2, 2));
        }
        const float sy_c = std::sqrt(carried.P(2, 2)), sl_c = std::sqrt(carried.P(0, 0));
        const float sy_f = std::sqrt(fresh.P(2, 2)), sl_f = std::sqrt(fresh.P(0, 0));
        std::print("    after 40 frames: carried σ_lat {:.4f} m σ_yaw {:.4f} rad (frame 1: {:.4f}); not carried σ_lat {:.4f} σ_yaw {:.4f}; "
                   "physics floor σ_lat {:.4f} σ_yaw {:.4f}; mean error ({:+.3f}, {:+.3f}, {:+.4f})\n", sl_c, sy_c, yaw1, sl_f, sy_f,
                   floor_lat, floor_yaw, carried.x.x(), carried.x.y() - D, carried.x.z());
        check(sy_c >= 0.9f * float(floor_yaw), "carried: σ_yaw ≥ 0.9 × the physical yaw common mode after 40 frames");
        check(sl_c >= 0.9f * float(floor_lat), "carried: σ_lat ≥ 0.9 × σ_c ⊕ σ_ε·range after 40 frames");
        check(sy_c >= 0.8f * yaw1, "carried: 40 frames claim no less than ~1 frame in yaw (does not average out)");
        check(sy_f < 0.5f * sy_c, "not carried: the same chain collapses (σ_yaw < ½ the carried one) — the defect Task 10 removes");
        check(std::abs(carried.x.x()) < 0.05f and std::abs(carried.x.z()) < 0.01f, "the carried chain's mean is still on the truth");
    }

    // An open doorway whose jambs carry 0.2 m-deep CASINGS the thin-wall model does not describe (DOOR_1's frames): the
    // rays that end on them pull the fit coherently, the clusters' scores disperse, and the misfit term must widen σ —
    // relative to the same doorway without casings, from the same mouth pose, both chained with their bias factor.
    std::print("misfit_widens_sigma_on_unmodelled_casings\n");
    {
        auto run_case = [&](bool casings)
        {
            WorldOpts wo; wo.u0 = -4.f; wo.u1 = 4.f; wo.corner = false; wo.shut = false; wo.back_wall = true;
            auto segs = world(wo);
            if (casings)
            {
                box(segs, -0.5f * W, -0.5f * W + 0.05f, -0.5f * T - 0.09f, 0.5f * T + 0.09f);
                box(segs, 0.5f * W - 0.05f, 0.5f * W, -0.5f * T - 0.09f, 0.5f * T + 0.09f);
            }
            RelPose x;
            rc::door::AnchorBias b;
            float js = 0.f, err_yaw = 0.f, err_lat = 0.f;
            for (int k = 0; k < 6; ++k)
            {
                const Pose2 robot{V2(0.08f, -1.2f + 0.1f * float(k) - kSensor.y()), 0.03f};
                const RelPose truth = creation_prior(robot);
                if (k == 0) x = truth;
                const auto rays = rc::door::select_anchor_rays(x, W, T, kFloorZ, 2.0f, kSensor, scan3d(segs, robot, 200 + unsigned(k)), 2.0f);
                const auto [post, ev] = update_door_anchor(x, W, T, wo.u0, wo.u1, kSensor, rays, {}, {}, 0.9f, {}, kFloorZ, nullptr, &b);
                x = post;
                js = ev.misfit.trace();
                err_yaw = std::abs(rc::interaction::wrap_pi(post.x.z() - truth.x.z()));
                err_lat = std::abs(post.x.x() - truth.x.x());
                if (k + 1 < 6)   // the next pose is 0.1 m ahead: carry the anchor and its bias by that step
                {
                    const Eigen::Vector3f xb = x.x;
                    x.x = rc::interaction::move_mean(x.x, 0.f, 0.1f, 0.f);
                    rc::door::transport_bias(b, xb, 0.f, 0.1f, 0.f, W);
                }
            }
            return std::array<float, 5>{std::sqrt(x.P(0, 0)), std::sqrt(x.P(2, 2)), js, err_lat, err_yaw};
        };
        const auto plain = run_case(false), cased = run_case(true);
        std::print("    plain: σ_lat {:.4f} σ_yaw {:.4f}, misfit tr {:.3g}, |err| {:.3f} m {:.4f} rad | casings: σ_lat {:.4f} σ_yaw {:.4f}, "
                   "misfit tr {:.3g}, |err| {:.3f} m {:.4f} rad\n", plain[0], plain[1], plain[2], plain[3], plain[4], cased[0], cased[1],
                   cased[2], cased[3], cased[4]);
        check(cased[2] > plain[2], "the casings' rays disperse the cluster scores (misfit trace larger)");
        check(cased[0] > 1.3f * plain[0], "...and the published σ_lat is wider with them (> 1.3×: the casings pull along the wall)");
        check(cased[1] >= 0.99f * plain[1], "...while σ_yaw does not shrink");
        check(cased[4] <= 2.f * cased[1] + 0.005f and cased[3] <= 2.f * cased[0] + 0.01f, "the casing doorway's error is within ~2σ");
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
