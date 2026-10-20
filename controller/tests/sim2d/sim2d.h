// sim2d.h — a 2-D closed-loop world for the look-ahead local policy (Task 6). Header-only, Eigen + std, no Qt/DSR.
//
//   World   : axis-aligned boxes and thin segments. Each obstacle is either IN THE MAP (the cost-to-go field knows it) or
//             not (a STALE map: the LiDAR and the contact test see it, the field does not — the case release exists for).
//   LiDAR   : 1440 beams from the robot centre over 360 deg, 10 m, Gaussian range noise (sigma 0.02 m), raycast against the
//             TRUE pose; the driver re-expresses the (bearing, range) pairs on the BELIEVED pose, as the controller does.
//   Base    : differential drive with Task-2 BaseLimits (ramps at a_lin/d_lin, a_rot/d_rot), a command latency queue
//             (60 ms) and the base's deadband (|v| < v_min_eff -> 0, |w| < w_min_eff -> 0: it does not execute them).
//   Contact : the TRUE bare hull (RobotFootprint::shadow(), margin 0) against the world's surfaces sampled every 5 mm; the
//             motion is integrated in sub-steps of <= 5 mm at the hull's farthest point. A sub-step that would put a surface
//             point inside the hull is a CONTACT: it is counted and the body stays where it was (the world pushes back).
//   Pose    : optional Ornstein-Uhlenbeck pose error (stationary sigma, time constant) plus scripted jumps. The policy
//             sees the noisy pose; contact is judged on the true one.
//
// The DOOR_1 boxes and the beam loop are COPIED from controller/tests/test_jamb_cross.cpp:74-146 (not included: that test
// pulls Qt/DSR through the old controller headers).
#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <numbers>
#include <random>
#include <vector>

#include "../../src/local_policy/policy_types.h"
#include "../../src/local_policy/contact.h"
#include "../../../common/robot_footprint/robot_footprint.h"

namespace sim2d
{
using V2 = Eigen::Vector2f;
using rc::policy::Pose2;
using rc::policy::BaseLimits;
constexpr float kPi = std::numbers::pi_v<float>;

inline float wrap(float a) { while (a > kPi) a -= 2.f * kPi; while (a <= -kPi) a += 2.f * kPi; return a; }

struct Box { float x0, y0, x1, y1; bool in_map = true; };
struct Segment { V2 a, b; bool in_map = true; };

struct World
{
    std::vector<Box> boxes;
    std::vector<Segment> segments;

    // Distance from p to the nearest obstacle (0 inside a box). map_only: only obstacles the map knows.
    float clearance(const V2& p, bool map_only) const
    {
        float best = std::numeric_limits<float>::infinity();
        for (const auto& b : boxes)
        {
            if (map_only and not b.in_map) continue;
            const float dx = std::max({b.x0 - p.x(), 0.f, p.x() - b.x1}), dy = std::max({b.y0 - p.y(), 0.f, p.y() - b.y1});
            best = std::min(best, std::hypot(dx, dy));
        }
        for (const auto& s : segments)
        {
            if (map_only and not s.in_map) continue;
            const V2 ab = s.b - s.a;
            const float t = std::clamp((p - s.a).dot(ab) / std::max(ab.squaredNorm(), 1e-12f), 0.f, 1.f);
            best = std::min(best, (s.a + t * ab - p).norm());
        }
        return best;
    }

    // Range along a unit ray from o (max_r if nothing): slab test for boxes (copied loop), segment intersection.
    float raycast(const V2& o, const V2& d, float max_r) const
    {
        float best = max_r;
        for (const auto& bx : boxes)
        {
            float t0 = 0.f, t1 = best;
            bool hit = true;
            for (int ax = 0; ax < 2 and hit; ++ax)
            {
                const float oo = o[ax], dd = d[ax], lo = ax == 0 ? bx.x0 : bx.y0, hi = ax == 0 ? bx.x1 : bx.y1;
                if (std::abs(dd) < 1e-9f) { if (oo < lo or oo > hi) hit = false; continue; }
                float ta = (lo - oo) / dd, tb = (hi - oo) / dd;
                if (ta > tb) std::swap(ta, tb);
                t0 = std::max(t0, ta); t1 = std::min(t1, tb);
                if (t0 > t1) hit = false;
            }
            if (hit and t0 > 1e-4f and t0 < best) best = t0;
        }
        for (const auto& s : segments)
        {
            const V2 e = s.b - s.a;
            const float den = d.x() * e.y() - d.y() * e.x();
            if (std::abs(den) < 1e-9f) continue;
            const V2 w = s.a - o;
            const float t = (w.x() * e.y() - w.y() * e.x()) / den, u = (w.x() * d.y() - w.y() * d.x()) / den;
            if (t > 1e-4f and t < best and u >= 0.f and u <= 1.f) best = t;
        }
        return best;
    }

    // Every surface point, every 5 mm (box perimeters and segments): the contact test's world.
    std::vector<V2> surface_points(float step = 0.005f) const
    {
        std::vector<V2> out;
        auto line = [&](const V2& a, const V2& b)
        {
            const int n = std::max(1, int(std::ceil((b - a).norm() / step)));
            for (int i = 0; i <= n; ++i) out.push_back(a + (b - a) * (float(i) / float(n)));
        };
        for (const auto& b : boxes)
        {
            const V2 p00(b.x0, b.y0), p10(b.x1, b.y0), p11(b.x1, b.y1), p01(b.x0, b.y1);
            line(p00, p10); line(p10, p11); line(p11, p01); line(p01, p00);
        }
        for (const auto& s : segments) line(s.a, s.b);
        return out;
    }
};

// One LiDAR beam in the robot frame: bearing (CCW from forward), measured range.
struct Beam { float bearing, range; };

// ★ 1440 beams, sigma 0.02 m, 10 m — the jamb test's raycaster, from the TRUE pose.
inline std::vector<Beam> scan(const World& w, const Pose2& truth, std::mt19937& rng, float sigma_r = 0.02f, int n = 1440, float max_r = 10.f)
{
    std::normal_distribution<float> nd(0.f, sigma_r);
    std::vector<Beam> out;
    out.reserve(n);
    for (int k = 0; k < n; ++k)
    {
        const float b = 2.f * kPi * float(k) / float(n);
        const float a = truth.yaw + b;
        const float r = w.raycast(truth.p, V2(std::cos(a), std::sin(a)), max_r);
        if (r < max_r) out.push_back({b, r + nd(rng)});
    }
    return out;
}

// The true contact test: any surface point inside the BARE hull at `pose`. Broad phase by the circumscribed radius.
struct ContactTester
{
    const rc::RobotFootprint* body;
    std::vector<V2> pts;
    float r_circ;
    ContactTester(const World& w, const rc::RobotFootprint& b) : body(&b), pts(w.surface_points()), r_circ(b.circumscribed_radius()) {}
    // min clearance from the bare hull to the world (negative = overlap), over points within reach
    float clearance(const Pose2& pose) const
    {
        float m = 1e9f;
        for (const auto& q : pts)
            if ((q - pose.p).squaredNorm() <= (r_circ + 0.5f) * (r_circ + 0.5f)) m = std::min(m, rc::policy::hull_distance(*body, pose, q));
        return m;
    }
    bool touches(const Pose2& pose) const
    {
        for (const auto& q : pts)
            if ((q - pose.p).squaredNorm() <= r_circ * r_circ and rc::policy::hull_distance(*body, pose, q) < 0.f) return true;
        return false;
    }
};

// ★ Differential base: commands reach the wheels `latency_s` after they are sent; each axis ramps to the command
//   (a when the speed grows, d when it shrinks, through zero on a sign change); a command inside the deadband is not
//   executed (the axis goes to 0, as the real base does when it ignores it).
struct Base
{
    BaseLimits lim;
    float latency_s = 0.06f;
    float v = 0.f, w = 0.f;        // actual
    float cmd_v = 0.f, cmd_w = 0.f; // the command currently in effect at the wheels
    struct Cmd { float t, v, w; };
    std::deque<Cmd> queue;

    void send(float t_now, float v_cmd, float w_cmd) { queue.push_back({t_now + latency_s, v_cmd, w_cmd}); }

    static float ramp(float x, float target, float a, float d, float dt)
    {
        // shrinking toward zero (or through it) uses d, growing uses a
        float rem = dt;
        while (rem > 1e-9f and x != target)
        {
            const bool through_zero = (x > 0.f and target < 0.f) or (x < 0.f and target > 0.f);
            const float goal = through_zero ? 0.f : target;
            const bool shrinking = std::abs(goal) < std::abs(x);
            const float rate = shrinking ? d : a;
            if (rate <= 0.f) { x = goal; continue; }
            const float need = std::abs(goal - x) / rate;
            if (need <= rem) { x = goal; rem -= need; }
            else { x += (goal > x ? 1.f : -1.f) * rate * rem; rem = 0.f; }
        }
        return x;
    }
    // advance the velocity state by dt at time t (applies due commands first)
    void step_velocity(float t, float dt)
    {
        while (not queue.empty() and queue.front().t <= t + 1e-6f) { cmd_v = queue.front().v; cmd_w = queue.front().w; queue.pop_front(); }
        const float tv = std::abs(cmd_v) < lim.v_min_eff ? 0.f : std::clamp(cmd_v, -lim.v_max, lim.v_max);
        const float tw = std::abs(cmd_w) < lim.w_min_eff ? 0.f : std::clamp(cmd_w, -lim.w_max, lim.w_max);
        v = ramp(v, tv, lim.a_lin, lim.d_lin, dt);
        w = ramp(w, tw, lim.a_rot, lim.d_rot, dt);
    }
};

// ★ Pose error as a RIGID transform of the room frame (a localiser that is wrong is wrong about the whole frame, so a
//   believed displacement is the true one rotated by the yaw error — a translation-only error would make the believed
//   heading disagree with the believed motion): believed.p = R(eyaw)(truth.p - pivot) + pivot + e, yaw + eyaw.
//   OU part: e' = -e/tau + sqrt(2/tau)*sigma*xi, stationary sigma per axis. "0.5 m / 13 deg over ~20 s" (the measured
//   process) = sigma_xy 0.5, sigma_yaw 13 deg, tau 20 s. Off by default. jump() adds a step to the error.
struct PoseError
{
    bool on = false;
    float sigma_xy = 0.5f, sigma_yaw = 13.f * kPi / 180.f, tau = 20.f;
    V2 pivot = V2::Zero();
    V2 e = V2::Zero(); float eyaw = 0.f;
    void step(float dt, std::mt19937& rng)
    {
        if (not on) return;
        std::normal_distribution<float> n(0.f, 1.f);
        const float k = std::exp(-dt / tau), s = std::sqrt(1.f - k * k);
        e = k * e + s * sigma_xy * V2(n(rng), n(rng));
        eyaw = k * eyaw + s * sigma_yaw * n(rng);
    }
    // A step in the error, pivoting about `at` (the robot's position when it happens): the believed pose of the robot
    // itself jumps by exactly (de, deyaw).
    void jump(const V2& at, const V2& de, float deyaw)
    {
        // keep the current believed map of every point except the step: re-pivot so the new rotation acts about `at`
        const Eigen::Rotation2Df R(eyaw);
        const V2 off = pivot + e - R * pivot;                   // believed = R*truth + off
        eyaw += deyaw;
        const Eigen::Rotation2Df R2(eyaw);
        const V2 b_at = R * at + off;                           // where `at` was believed before
        // new: believed = R2*(truth - at) + b_at + de
        pivot = at; e = b_at + de - at;
        (void)R2;
    }
    Pose2 believed(const Pose2& truth) const
    {
        const Eigen::Rotation2Df R(eyaw);
        return {R * (truth.p - pivot) + pivot + e, wrap(truth.yaw + eyaw)};
    }
    // the true point the believed point q corresponds to
    V2 to_truth(const V2& q) const { const Eigen::Rotation2Df R(eyaw); return R.inverse() * (q - pivot - e) + pivot; }
};

// Exact unicycle step (arc).
inline Pose2 integrate(const Pose2& p, float v, float w, float dt) { return rc::policy::advance(p, v, w, dt); }
}  // namespace sim2d
