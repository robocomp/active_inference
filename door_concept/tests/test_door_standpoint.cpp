// Offline harness for door_standpoint.h: wall association + approach/through poses on room_1's REMEMBERED
// outline (ltsm r1_room_1), for door_24 (Webots DOOR_0, in the y≈2.4 wall at the partition corner) and door_19
// (Webots DOOR_1, outer top wall). Reproduces the live failure (try25/28/30: aff_door_24_approach OUTSIDE the
// room) with the OLD rules and checks the NEW ones. Build + run:
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 tests/test_door_standpoint.cpp -o /tmp/test_door_standpoint && /tmp/test_door_standpoint
#include <cmath>
#include <cstdio>
#include <print>
#include <vector>
#include "../src/door_standpoint.h"

using V = Eigen::Vector2f;
using namespace rc::door;

static const std::vector<V> kRoom1 = {
    {-2.51f,2.35f},{-4.68f,2.37f},{-4.73f,-3.41f},{-4.7f,-3.41f},{-4.71f,-4.59f},{-0.95f,-4.63f},{-0.94f,-3.94f},
    {-0.4f,-3.94f},{-0.41f,-4.75f},{3.49f,-4.78f},{3.49f,-4.6f},{4.05f,-4.61f},{4.05f,-4.04f},{3.74f,-4.03f},
    {3.75f,-3.43f},{3.64f,-3.43f},{3.69f,2.36f},{4.14f,2.36f},{0.09f,2.4f},{0.07f,0.39f},{-0.03f,0.39f},
    {0.01f,4.49f},{-2.08f,4.51f},{-2.09f,3.39f},{-2.5f,3.4f}};

// door_concept defaults (etc/config.toml): ActuationReachM 2.0, RobotPassageWidthM 0.6, SwingClearanceM 0.2,
// CrossStandoffM 1.2. d_stand = middle of [w + 0.3 + 0.2, max(2.0, d_min + 0.3)].
constexpr float kW = 1.0f, kCross = 1.2f;
static float d_stand(float w) { const float dmin = w + 0.3f + 0.2f; return 0.5f * (dmin + std::max(2.0f, dmin + 0.3f)); }

// ── the OLD rules, verbatim in effect: nearest edge to the centroid, |cos| merge, side = half-plane sign ──
static WallRun old_nearest_wall(const V& q, std::span<const V> poly)
{
    const std::size_t n = poly.size(), bi = nearest_edge(q, poly);
    const auto dir = [&](std::size_t i) { return V((poly[(i + 1) % n] - poly[i]).normalized()); };
    const V d0 = dir(bi); std::size_t lo = bi, hi = bi;
    for (std::size_t k = 0; k + 1 < n; ++k) { const std::size_t p = (lo + n - 1) % n; if (std::abs(dir(p).dot(d0)) < 0.99863f) break; lo = p; }
    for (std::size_t k = 0; k + 1 < n; ++k) { const std::size_t x = (hi + 1) % n; if (x == lo or std::abs(dir(x).dot(d0)) < 0.99863f) break; hi = x; }
    WallRun w; w.O = poly[lo]; const V far = poly[(hi + 1) % n]; w.len = (far - w.O).norm(); w.u = (far - w.O) / w.len; w.ok = true;
    if (polygon_area2(poly) < 0.0) { w.O = far; w.u = -w.u; }
    return w;
}

static Aperture seed(const WallRun& wf, const V& centroid)
{
    Aperture a; a.wall_O = wf.O; a.wall_u = wf.u; a.w = kW;
    a.s = std::clamp((centroid - wf.O).dot(wf.u) - 0.5f * kW, 0.0f, std::max(0.0f, wf.len - kW));
    return a;
}

static int failures = 0;
static void check(bool ok, const char* what) { std::print("    [{}] {}\n", ok ? "PASS" : "FAIL", what); failures += ok ? 0 : 1; }

static void report_pose(const char* tag, const Pose2& p)
{
    std::print("    {:<10} ({:+.2f},{:+.2f}) yaw {:+.2f}  {}\n", tag, p.x, p.y, p.yaw,
               polygon_contains({p.x, p.y}, kRoom1) ? "inside room_1" : "OUTSIDE room_1");
}

struct Case { const char* name; std::vector<V> pts; V robot; };

static std::vector<V> leaf(V a, V b, int n = 40)   // plan-view samples of a leaf from a to b, ±1 cm depth jitter
{
    std::vector<V> v; const V d = (b - a), nn = V(-d.y(), d.x()).normalized();
    for (int i = 0; i < n; ++i) { const float t = (i + 0.5f) / n; v.push_back(a + t * d + 0.01f * ((i % 3) - 1) * nn); }
    return v;
}

int main()
{
    std::print("room_1: {} vertices, signed area {:+.2f} m² ({})\n", kRoom1.size(), 0.5 * polygon_area2(kRoom1),
               polygon_area2(kRoom1) > 0 ? "CCW" : "CW");
    const float ds = d_stand(kW);
    std::print("d_stand (w = {:.2f}) = {:.3f} m\n\n", kW, ds);

    // Hinge of DOOR_0 = (0.12,2.49) (Door.proto anchor, room frame = world + (4.18,-4.78)); leaf 1 m long.
    const std::vector<Case> cases = {
        {"door_24 leaf SWUNG 90deg away from the room (the live case), robot in the lower-right region",
         leaf({0.13f, 2.50f}, {0.13f, 3.48f}), {2.00f, 0.00f}},
        {"door_24 leaf CLOSED, robot in the lower-right region",
         leaf({0.12f, 2.50f}, {1.10f, 2.50f}), {2.00f, 0.00f}},
        {"door_24 leaf CLOSED, robot right below it",
         leaf({0.12f, 2.50f}, {1.10f, 2.50f}), {0.60f, 0.70f}},
        {"door_19 leaf CLOSED (outer top wall), robot in the left region",
         leaf({-1.09f, 4.62f}, {-0.09f, 4.62f}), {-1.00f, 1.50f}},
        {"door_19 leaf SWUNG 90deg into room_2, robot in the left region",
         leaf({-1.06f, 4.65f}, {-1.06f, 5.63f}), {-1.00f, 1.50f}},
    };

    for (const auto& c : cases)
    {
        std::print("== {}\n", c.name);
        V centroid = V::Zero(); for (const auto& p : c.pts) centroid += p; centroid /= static_cast<float>(c.pts.size());
        std::print("    mask centroid ({:+.2f},{:+.2f}), robot ({:+.2f},{:+.2f}) {}\n", centroid.x(), centroid.y(),
                   c.robot.x(), c.robot.y(), polygon_contains(c.robot, kRoom1) ? "inside" : "OUTSIDE");

        // OLD
        const WallRun wo = old_nearest_wall(centroid, kRoom1);
        const Aperture ao = seed(wo, centroid);
        const float side_o = ao.across_u().dot(c.robot - ao.centre_xy());
        std::print("  OLD: wall O({:+.2f},{:+.2f}) u({:+.2f},{:+.2f}) len {:.2f}; aperture centre ({:+.2f},{:+.2f}); robot_side {:+.2f}\n",
                   wo.O.x(), wo.O.y(), wo.u.x(), wo.u.y(), wo.len, ao.centre_xy().x(), ao.centre_xy().y(), side_o);
        report_pose("approach", front_pose(ao, side_o, ds));

        // NEW
        const WallRun wn = associate_wall(kRoom1, c.pts, c.robot);
        const Aperture an = seed(wn, centroid);
        const float side_n_plane = an.across_u().dot(c.robot - an.centre_xy());
        const float side_n = approach_side(an, c.robot, side_n_plane, ds, kRoom1);
        std::print("  NEW: wall O({:+.2f},{:+.2f}) u({:+.2f},{:+.2f}) len {:.2f} (edges {}..{}); aperture centre ({:+.2f},{:+.2f}); "
                   "robot_side {:+.2f} → approach side {:+.0f}\n",
                   wn.O.x(), wn.O.y(), wn.u.x(), wn.u.y(), wn.len, wn.lo, wn.hi, an.centre_xy().x(), an.centre_xy().y(),
                   side_n_plane, side_n);
        const Pose2 fp = front_pose(an, side_n, ds), tp = through_pose(an, side_n, kCross);
        report_pose("approach", fp);
        report_pose("through", tp);
        check(polygon_contains({fp.x, fp.y}, kRoom1), "approach standpoint inside room_1");
        check(not polygon_contains({tp.x, tp.y}, kRoom1), "through target beyond the door (not in room_1)");
        check(an.across_u().dot(V(fp.x, fp.y) - an.centre_xy()) > 0.0f, "approach on the wall's INWARD normal side");
        if (std::string_view(c.name).starts_with("door_24"))
        {
            check(std::abs(an.wall_u.y()) < 0.1f and std::abs(an.centre_xy().y() - 2.38f) < 0.1f, "door_24 on the y≈2.4 wall");
            check(std::abs(an.centre_xy().x() - 0.60f) < 0.15f, "door_24 aperture centred at x≈0.6 (DOOR_0)");
        }
        else
        {
            // On THIS outline the top wall is at y≈4.50, so the standpoint is 4.50 − 1.75 = 2.75; live, on the given
            // 32-vertex map whose top wall is at y≈4.60, the same rule put it at (-0.67,2.85).
            check(std::abs(an.wall_u.y()) < 0.1f and std::abs(an.centre_xy().y() - 4.50f) < 0.05f, "door_19 on the top wall");
            check(std::abs(fp.y - (an.centre_xy().y() - ds)) < 0.02f, "door_19 standpoint d_stand below the top wall");
            check(std::abs(fp.yaw - 1.5708f) < 0.05f, "door_19 standpoint faces +y");
            const Pose2 fo = front_pose(ao, side_o, ds);
            check(std::abs(fo.x - fp.x) < 1e-4f and std::abs(fo.y - fp.y) < 1e-4f and std::abs(fo.yaw - fp.yaw) < 1e-4f,
                  "door_19 standpoint UNCHANGED from the old rules");
        }
        std::print("\n");
    }

    // The belief as it actually CONVERGED live (try30: s=2.05 w=1.00 on the partition, centre (-0.01,2.93)) — the
    // side choice alone must already keep the standpoint inside the room, whatever wall the belief ended up on.
    {
        std::print("== replay of the converged try30 belief (partition wall, s=2.05), robot in the lower-right region\n");
        const WallRun wa = wall_run_of_edge(kRoom1, 20);   // (-0.03,0.39) → (0.01,4.49)
        Aperture a; a.wall_O = wa.O; a.wall_u = wa.u; a.s = 2.05f; a.w = 1.0f;
        const V robot{2.0f, 0.0f};
        const float plane = a.across_u().dot(robot - a.centre_xy());
        std::print("    centre ({:+.2f},{:+.2f}) robot_side {:+.2f}\n", a.centre_xy().x(), a.centre_xy().y(), plane);
        report_pose("OLD appr.", front_pose(a, plane, ds));
        const Pose2 fp = front_pose(a, approach_side(a, robot, plane, ds, kRoom1), ds);
        report_pose("NEW appr.", fp);
        check(polygon_contains({fp.x, fp.y}, kRoom1), "standpoint inside room_1 even on the wrong wall");
    }

    std::print("\n{}\n", failures == 0 ? "ALL PASS" : std::format("{} FAILURE(S)", failures));
    return failures == 0 ? 0 : 1;
}
