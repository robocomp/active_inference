/*
 * z_occlusion_test.cpp — the 3-D sight test. Standalone:
 *
 *     g++ -std=c++23 -O1 -I/usr/include/eigen3 z_occlusion_test.cpp -o z_occlusion_test && ./z_occlusion_test
 *
 * ★WHY THIS EXISTS AND WHY IT LEADS WITH A REGRESSION GUARD. rc::nbv::Obstacle was a floor footprint with no
 * top: every obstacle was infinitely tall. That over-occludes, so visible_fraction reported too little
 * visibility, so p_detect came out too LOW, so absence was charged too WEAKLY — and objects were HELD. That
 * is the safe direction, and giving obstacles a height moves the error to the UNSAFE side (more removal).
 *
 * So the first thing this file asserts is that an obstacle with NO declared height behaves exactly as before.
 * Everything else is opt-in: z1 <= z0 means unknown, and the collector does not populate it unless asked.
 *
 * The struct + primitive are DUPLICATED below because viewpoint_score.h pulls in the fleet's headers. Keep
 * them in step — if you change segment_blocked_3d, change it here and watch this fail first.
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <Eigen/Dense>

// ── mirror of rc::nbv::Obstacle + detail::segment_box_interval / segment_blocked_3d ────────────────
struct Obstacle
{
    float cx = 0, cy = 0, w = 0, h = 0, yaw = 0;
    float z0 = 0, z1 = 0;
    bool has_height() const { return z1 > z0; }
};

static bool segment_box_interval(const Eigen::Vector2f& a, const Eigen::Vector2f& b, const Obstacle& o,
                                 float& out_t0, float& out_t1)
{
    const float c = std::cos(-o.yaw), s = std::sin(-o.yaw);
    const Eigen::Vector2f d = b - a;
    const Eigen::Vector2f half(0.5f * o.w, 0.5f * o.h);
    const Eigen::Vector2f p0(c * (a.x() - o.cx) - s * (a.y() - o.cy),
                             s * (a.x() - o.cx) + c * (a.y() - o.cy));
    const Eigen::Vector2f dd(c * d.x() - s * d.y(), s * d.x() + c * d.y());
    float t0 = 0.0f, t1 = 1.0f;
    for (int k = 0; k < 2; ++k)
    {
        if (std::abs(dd(k)) < 1e-6f) { if (std::abs(p0(k)) > half(k)) return false; }
        else
        {
            float ta = (-half(k) - p0(k)) / dd(k), tb = (half(k) - p0(k)) / dd(k);
            if (ta > tb) std::swap(ta, tb);
            t0 = std::max(t0, ta); t1 = std::min(t1, tb);
        }
    }
    out_t0 = t0; out_t1 = t1;
    return t0 < t1 and t1 > 1e-3f and t0 < 1.0f - 1e-3f;
}
static bool segment_hits_box(const Eigen::Vector2f& a, const Eigen::Vector2f& b, const Obstacle& o)
{ float t0, t1; return segment_box_interval(a, b, o, t0, t1); }

static bool segment_blocked_3d(const Eigen::Vector3f& A, const Eigen::Vector3f& B, const Obstacle& o)
{
    if (not o.has_height()) return segment_hits_box(A.head<2>(), B.head<2>(), o);
    float t0, t1;
    if (not segment_box_interval(A.head<2>(), B.head<2>(), o, t0, t1)) return false;
    const float cl0 = std::clamp(t0, 0.0f, 1.0f), cl1 = std::clamp(t1, 0.0f, 1.0f);
    const float za = A.z() + cl0 * (B.z() - A.z());
    const float zb = A.z() + cl1 * (B.z() - A.z());
    return std::max(std::min(za, zb), o.z0) <= std::min(std::max(za, zb), o.z1);
}

static int fails = 0;
static void ck(bool c, const char* what) { if (not c) { std::printf("FAIL: %s\n", what); ++fails; } }

int main()
{
    // A worktop 2 m in front of the camera: 2 m wide, 0.6 m deep, top at 0.89 m.
    Obstacle worktop; worktop.cx = 0.0f; worktop.cy = 2.0f; worktop.w = 2.0f; worktop.h = 0.6f;
    Obstacle worktop_h = worktop; worktop_h.z0 = 0.0f; worktop_h.z1 = 0.89f;

    const Eigen::Vector3f cam(0.0f, 0.0f, 1.20f);          // ZED at 1.20 m

    // ── ★REGRESSION GUARD FIRST: no declared height ⇒ exactly the 2-D answer, whatever the heights are ──
    for (float target_z : {0.05f, 0.85f, 1.10f, 2.00f})
    {
        const Eigen::Vector3f p(0.0f, 4.0f, target_z);
        ck(segment_blocked_3d(cam, p, worktop) == segment_hits_box(cam.head<2>(), p.head<2>(), worktop),
           "an obstacle with NO height behaves exactly as the 2-D test, at every target height");
    }
    ck(segment_blocked_3d(cam, Eigen::Vector3f(0, 4, 1.10f), worktop),
       "...and that 2-D answer is BLOCKED here, so the guard above is not vacuous");

    // ── with a height, the same worktop stops blocking what it cannot reach ───────────────────────
    ck(not segment_blocked_3d(cam, Eigen::Vector3f(0.0f, 4.0f, 1.10f), worktop_h),
       "a 0.89 m worktop does NOT occlude a target at 1.10 m seen from a 1.20 m camera");
    ck(segment_blocked_3d(cam, Eigen::Vector3f(0.0f, 4.0f, 0.05f), worktop_h),
       "the same worktop DOES occlude a target at 0.05 m (the ray dives under its top)");

    // ── ★THE MICROWAVE / BOTTLE CASE: standing ON the support, inside its footprint ───────────────
    // A microwave at 0.89..1.19 sitting on that worktop, sampled at its base and its top.
    ck(not segment_blocked_3d(cam, Eigen::Vector3f(0.0f, 2.0f, 1.19f), worktop_h),
       "an object standing ON a worktop is not occluded BY it (top sample)");
    ck(not segment_blocked_3d(cam, Eigen::Vector3f(0.0f, 2.0f, 0.90f), worktop_h),
       "...nor at its base, 1 cm above the surface");
    // The 2-D model gets this wrong, which is why bottle carries a local work-around.
    ck(segment_hits_box(cam.head<2>(), Eigen::Vector2f(0.0f, 2.0f), worktop),
       "the 2-D model DOES report it occluded — the defect bottle works around locally");

    // ── a tall cabinet still blocks everything behind it ──────────────────────────────────────────
    Obstacle tall = worktop; tall.z0 = 0.0f; tall.z1 = 2.20f;
    for (float tz : {0.05f, 1.10f, 2.00f})
        ck(segment_blocked_3d(cam, Eigen::Vector3f(0, 4, tz), tall),
           "a 2.2 m cabinet occludes at every sampled height");

    // ── a HANGING obstacle (a hood at 1.99..2.28) must not occlude what is under it ───────────────
    Obstacle hood = worktop; hood.z0 = 1.99f; hood.z1 = 2.28f;
    ck(not segment_blocked_3d(cam, Eigen::Vector3f(0, 4, 1.10f), hood),
       "a hanging hood does not occlude a target below it");
    ck(segment_blocked_3d(Eigen::Vector3f(0, 0, 2.20f), Eigen::Vector3f(0, 4, 2.10f), hood),
       "...but does when both camera and target are up at its level");

    // ── ★THE CROSSING INTERVAL IS THE TEST, NOT THE ENDPOINTS ────────────────────────────────────
    // Camera high at 1.60 m, target low at 0.10 m and 6 m away. The ENDPOINTS straddle the 0..0.89 band,
    // so "does the segment's z-range meet the obstacle's z-range?" answers BLOCKED. But the ray only
    // crosses the footprint between y=1.7 and y=2.3, where it is still at z=1.175..1.025 — over the top.
    // It flies clean over the worktop and lands beyond it.
    {
        const Eigen::Vector3f A(0, 0, 1.60f), B(0, 6, 0.10f);
        ck(not segment_blocked_3d(A, B, worktop_h),
           "a ray that clears the worktop and descends BEYOND it is not blocked by it");
        const bool endpoint_test = std::max(std::min(A.z(), B.z()), worktop_h.z0)
                                <= std::min(std::max(A.z(), B.z()), worktop_h.z1);
        ck(endpoint_test,
           "...while judging by the segment's endpoints would call it blocked — hence the interval");
    }

    // ── degenerate inputs ────────────────────────────────────────────────────────────────────────
    ck(not segment_blocked_3d(cam, Eigen::Vector3f(0, -4, 1.0f), worktop_h),
       "an obstacle BEHIND the camera never blocks");
    Obstacle flat = worktop_h; flat.z1 = flat.z0;          // z1 == z0 ⇒ unknown, not zero-height
    ck(flat.has_height() == false and segment_blocked_3d(cam, Eigen::Vector3f(0, 4, 1.10f), flat),
       "z1 == z0 reads as UNKNOWN (infinitely tall), never as a zero-height object that blocks nothing");

    std::printf(fails ? "z_occlusion: %d FAILED\n" : "z_occlusion: all checks passed\n", fails);
    return fails ? 1 : 0;
}
