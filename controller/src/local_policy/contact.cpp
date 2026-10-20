#include "contact.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace rc::policy
{
namespace
{
constexpr float kHalfPi = 1.57079633f;

// Integral of a speed profile "hold x for H, then decelerate linearly to 0 at rate d" up to time t.
// ★ Integrating the profile (not sampling it) is what makes the end pose exact whatever dt is.
float travelled(float x, float H, float d, float t)
{
    if (t <= H) return x * t;
    const float tb_total = d > 0.f ? std::abs(x) / d : 0.f;       // d <= 0: instant stop, no braking distance
    const float tb = std::min(t - H, tb_total);
    return x * H + std::copysign(std::abs(x) * tb - 0.5f * d * tb * tb, x);
}

// Unsigned distance from p to segment ab, and the signed side (>0 left of a->b) for the CCW inside test.
float seg_dist(const Eigen::Vector2f& p, const Eigen::Vector2f& a, const Eigen::Vector2f& b)
{
    const Eigen::Vector2f ab = b - a;
    const float l2 = ab.squaredNorm();
    const float t = l2 > 0.f ? std::clamp((p - a).dot(ab) / l2, 0.f, 1.f) : 0.f;
    return (p - (a + t * ab)).norm();
}

// Signed distance to the margin-grown polygon given the point already in the ROBOT frame.
float sd_local(const std::vector<Eigen::Vector2f>& poly, float margin, const Eigen::Vector2f& q)
{
    float dmin = 1e30f;
    bool inside = true;
    const std::size_t n = poly.size();
    for (std::size_t i = 0; i < n; ++i)
    {
        const auto& a = poly[i];
        const auto& b = poly[(i + 1) % n];
        dmin = std::min(dmin, seg_dist(q, a, b));
        const Eigen::Vector2f ab = b - a, aq = q - a;
        if (ab.x() * aq.y() - ab.y() * aq.x() < 0.f) inside = false;   // right of a CCW edge => outside
    }
    return (inside ? -dmin : dmin) - margin;
}

Eigen::Vector2f to_robot(const Pose2& pose, const Eigen::Vector2f& q)
{
    // ★ same quarter turn as RobotFootprint::contains_yaw: body forward is +y, room yaw forward is (cos, sin)
    const float th = pose.yaw - kHalfPi, c = std::cos(th), s = std::sin(th);
    const Eigen::Vector2f d = q - pose.p;
    return {c * d.x() + s * d.y(), -s * d.x() + c * d.y()};
}

}  // namespace

std::vector<Pose2> rollout(const Pose2& start, float v, float w, float hold_s, const BaseLimits& b, float dt)
{
    const float H = std::max(0.f, hold_s);
    const float Tv = b.d_lin > 0.f ? std::abs(v) / b.d_lin : 0.f;
    const float Tw = b.d_rot > 0.f ? std::abs(w) / b.d_rot : 0.f;
    const float T = H + std::max(Tv, Tw);
    std::vector<Pose2> out{start};
    Pose2 cur = start;
    float t = 0.f, sv = 0.f, sw = 0.f;       // sv, sw: distance / angle integrated so far
    while (t < T - 1e-6f)
    {
        t = std::min(t + dt, T);
        const float nv = travelled(v, H, b.d_lin, t), nw = travelled(w, H, b.d_rot, t);
        const float dist = nv - sv, dth = nw - sw;
        sv = nv; sw = nw;
        // exact arc: chord = dist * sinc(dth/2) along the mean heading; the straight limit falls out as dth -> 0
        const float h = 0.5f * dth;
        const float chord = std::abs(h) < 1e-5f ? dist : dist * std::sin(h) / h;
        const float mid = cur.yaw + h;
        cur.p += chord * Eigen::Vector2f(std::cos(mid), std::sin(mid));
        cur.yaw += dth;
        out.push_back(cur);
    }
    return out;
}

float signed_distance(const rc::RobotFootprint& body, const Pose2& pose, const Eigen::Vector2f& q)
{
    return sd_local(body.polygon(), body.safety_margin(), to_robot(pose, q));
}

float log_p_no_contact(const rc::RobotFootprint& body, std::span<const Pose2> sweep, std::span<const Return> r,
                       float sigma_pose_xy, float sigma_pose_yaw)
{
    if (sweep.empty()) return 0.f;
    const auto& poly = body.polygon();
    const float margin = body.safety_margin();
    float path = 0.f;
    for (std::size_t i = 1; i < sweep.size(); ++i) path += (sweep[i].p - sweep[i - 1].p).norm();
    // ★ broad phase: anything farther than (reach + margin + path + 3 sigma) from the start can never touch the body
    const float reach = body.circumscribed_radius() + margin + path;
    const Pose2& s0 = sweep.front();
    double lp = 0.0;
    for (const Return& q : r)
    {
        const float rho = (q.p - s0.p).norm();
        const float sig2 = q.sigma * q.sigma + sigma_pose_xy * sigma_pose_xy + rho * rho * sigma_pose_yaw * sigma_pose_yaw;
        const float sig = std::sqrt(sig2);
        if (rho > reach + 3.f * sig) continue;
        const float sd0 = sd_local(poly, margin, to_robot(s0, q.p));
        float sd = sd0;
        for (std::size_t i = 1; i < sweep.size(); ++i) sd = std::min(sd, sd_local(poly, margin, to_robot(sweep[i], q.p)));
        // ★ CONDITIONED ON NOT TOUCHING NOW. The pose/range error eps ~ N(0, sig^2) is common-mode over a 1 s sweep,
        //   so with true distance sd - eps the sweep touches iff eps > sd_min, the start pose iff eps > sd0. Where the
        //   robot stands is a given, not a consequence of the motion:
        //     P_j = p_exist * P(eps > sd_min | eps <= sd0) = p_exist * [Phi(-sd_min/s) - Phi(-sd0/s)] / [1 - Phi(-sd0/s)]
        //   (0 when sd_min >= sd0: a motion that never gets closer than now cannot produce contact). Continuous across
        //   the surface — it replaces the Task-2 step rule "inside at start counts only if deeper".
        if (sd >= sd0) continue;
        // stable form: P/p_exist = 1 - Phi(sd_min/s)/Phi(sd0/s) = 1 - erfc(-sd_min*k)/erfc(-sd0*k), k = 1/(s*sqrt2)
        const double k = 1.0 / (double(sig) * 1.4142135623730951);
        const double ea = std::erfc(-double(sd0) * k), eb = std::erfc(-double(sd) * k);
        // ★ numeric guard, not a model gate: deep inside, erfc(a) underflows; the smallest positive double keeps the
        //   ratio finite (=> P_j -> p_exist, the right limit for a motion that sinks further into a return).
        const double ratio = std::min(1.0, eb / std::max(ea, std::numeric_limits<double>::min()));
        const double pc = double(q.p_exist) * (1.0 - ratio);
        // log1p for the small-P regime; near P = 1 take the log of the complement directly (1 - pc rounds to 0 there)
        lp += pc < 0.5 ? std::log1p(-pc)
                       : std::log(std::max((1.0 - double(q.p_exist)) + double(q.p_exist) * ratio, std::numeric_limits<double>::min()));
    }
    return float(lp);
}
}  // namespace rc::policy
