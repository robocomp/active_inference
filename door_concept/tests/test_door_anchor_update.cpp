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
static std::vector<Seg> scene(Leaf leaf)
{
    std::vector<Seg> s;
    box(s, -4.f, -0.5f * W, D - 0.5f * T, D + 0.5f * T);      // wall, left of the opening
    box(s, 0.5f * W, 4.f, D - 0.5f * T, D + 0.5f * T);        // wall, right of the opening
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
static std::vector<V3> cast(Leaf leaf, unsigned seed = 7)
{
    const auto segs = scene(leaf);
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
static RelPose prior_offset()
{
    RelPose p;
    p.x = V3(0.45f, D - 0.10f, 0.05f);                          // 0.45 m to the side, 10 cm short, 3 deg off
    p.P = V3(0.5f * 0.5f, 0.2f * 0.2f, 0.1f * 0.1f).asDiagonal();
    p.stamp_ms = 1234;
    return p;
}
static const V3 kOrigin(0.f, 0.f, 1.0f);

int main()
{
    std::setlocale(LC_ALL, "");
    using rc::door::update_door_anchor;

    std::print("open_corrects_lateral_offset\n");
    const auto open_rays = cast(Leaf::none);
    {
        const auto [post, ev] = update_door_anchor(prior_offset(), W, T, kOrigin, open_rays, mask(0.f));
        std::print("    x=({:.3f},{:.3f},{:.3f}) σ_lat={:.3f} through={} blocked={} short={} p_open={:.3f} nis={:.2f}\n",
                   post.x.x(), post.x.y(), post.x.z(), std::sqrt(post.P(0, 0)), ev.n_through, ev.n_blocked, ev.n_short,
                   ev.p_open, ev.camera_nis);
        check(std::abs(post.x.x()) < 0.05f, "lateral error < 0.05 m");
        check(ev.p_open > 0.9f, "p_open > 0.9");
    }

    std::print("no_camera_still_converges\n");
    RelPose open_nomask;
    {
        const auto [post, ev] = update_door_anchor(prior_offset(), W, T, kOrigin, open_rays, {});
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
        const auto [post, ev] = update_door_anchor(prior_offset(), W, T, kOrigin, cast(Leaf::side90), {});
        std::print("    x=({:.3f},{:.3f},{:.3f}) through={} blocked={} short={} p_open={:.3f}\n",
                   post.x.x(), post.x.y(), post.x.z(), ev.n_through, ev.n_blocked, ev.n_short, ev.p_open);
        check(ev.n_short > 0, "n_short > 0");
        check((post.x.head<2>() - open_nomask.x.head<2>()).norm() < 0.05f, "within 0.05 m of the no-leaf posterior");
    }

    std::print("shut_door\n");
    {
        const RelPose pr = prior_offset();
        const auto [post, ev] = update_door_anchor(pr, W, T, kOrigin, cast(Leaf::flush), {});
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
        const auto [post, ev] = update_door_anchor(prior_offset(), W, T, kOrigin, open_rays, mask(0.6f));
        std::print("    x=({:.3f},{:.3f},{:.3f}) p_open={:.3f} nis={:.2f}\n", post.x.x(), post.x.y(), post.x.z(),
                   ev.p_open, ev.camera_nis);
        check((post.x.head<2>() - V2(0.f, D)).norm() < 0.08f, "posterior within 0.08 m of the truth");
        check(ev.camera_nis > 9.f, "camera_nis > 9");
    }

    std::print("no_data_is_prior\n");
    {
        const RelPose pr = prior_offset();
        const auto [post, ev] = update_door_anchor(pr, W, T, kOrigin, {}, {});
        check(post.x == pr.x and post.P == pr.P and post.stamp_ms == pr.stamp_ms, "posterior == prior exactly");
        check(ev.n_through == 0 and ev.n_blocked == 0 and ev.n_short == 0 and ev.p_open == 0.5f and ev.camera_nis == -1.f,
              "evidence is empty");
    }

    std::print("{}\n", g_fail == 0 ? "ALL PASS" : std::format("{} FAILED", g_fail));
    return g_fail == 0 ? 0 : 1;
}
