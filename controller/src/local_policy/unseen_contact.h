// unseen_contact.h — evidence of an obstacle the LiDAR cannot see, from commanded motion that odometry denies.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include "policy_types.h"
#include "../../../common/robot_footprint/robot_footprint.h"

namespace rc::policy
{
// ★ When the base is commanded to move and odometry shows it did not, something is holding it (below the scan
//   band, glass, a wheel caught). Each tick adds the log-likelihood ratio of the measured displacement under
//   "blocked" N(0, s) against "free" N(cmd*dt, s):  llr = (c^2 - 2 d c) / (2 s^2),  c = cmd*dt, d = measured.
//   Moving as commanded gives -c^2/(2 s^2) < 0, so L falls: no timer, no counter, the evidence IS the integral.
//   Four accumulators (forward, backward, turn left, turn right): being free to go forward says nothing about
//   backward. Ticks below the base's deadband (v_min_eff / w_min_eff) add nothing: the base would not have moved
//   anyway, so the absence of motion discriminates nothing.
//   Translation s = sigma_odo_m; rotation s = sigma_odo_m / circumscribed radius (the same displacement error,
//   seen as an angle at the outermost corner).
//   The hypothesis is a Return at the BARE hull boundary in the blocked direction plus sigma_odo outward (forward
//   for v > 0, backward for v < 0), with sigma = sigma_odo_m and p_exist = logistic(L). For a pure rotation it sits
//   at the hull vertex that LEADS the turn: the farthest vertex from the centre (it sweeps the most, so it is the
//   one that meets an obstacle first), tie-broken toward the side moving forward in the commanded sense (the
//   right side for a left turn), then toward the front, since the front is where a body meets what it turns into.
//   A hypothesis exists while L > 0 (blocked more likely than free: the Bayes decision point, p_exist > 0.5, not a
//   tuned gate); below it the policy has nothing to avoid.
class UnseenContact
{
public:
    void update(float v_cmd, float w_cmd, float ds_meas, float dyaw_meas, float dt, const Pose2& pose,
                const BaseLimits& b, const rc::RobotFootprint& body, float sigma_odo_m)
    {
        pose_ = pose;
        sigma_ = sigma_odo_m;
        const float s = std::max(sigma_odo_m, 1e-6f);
        if (std::abs(v_cmd) >= b.v_min_eff and std::abs(v_cmd) > 0.f)
            accumulate(v_cmd > 0.f ? L_[0] : L_[1], std::abs(v_cmd) * dt, v_cmd > 0.f ? ds_meas : -ds_meas, s);
        if (std::abs(w_cmd) >= b.w_min_eff and std::abs(w_cmd) > 0.f)
        {
            const float sr = s / std::max(body.circumscribed_radius(), 1e-3f);
            accumulate(w_cmd > 0.f ? L_[2] : L_[3], std::abs(w_cmd) * dt, w_cmd > 0.f ? dyaw_meas : -dyaw_meas, sr);
        }
        body_ = body;
    }

    std::optional<Return> hypothesis() const
    {
        if (not body_) return std::nullopt;
        const int k = int(std::max_element(L_.begin(), L_.end()) - L_.begin());
        if (not (L_[k] > 0.f)) return std::nullopt;
        const float p = 1.f / (1.f + std::exp(-L_[k]));
        const Eigen::Vector2f fwd(std::cos(pose_.yaw), std::sin(pose_.yaw));
        Return r; r.sigma = sigma_; r.p_exist = p;
        if (k < 2)
        {
            const Eigen::Vector2f d = (k == 0 ? 1.f : -1.f) * fwd;
            // ★ support_radius_yaw includes the safety margin; the hypothesis sits on the bare hull.
            const float reach = body_->support_radius_yaw(pose_.yaw, d) - body_->safety_margin();
            r.p = pose_.p + d * (reach + sigma_);
            return r;
        }
        const float sgn = (k == 2) ? 1.f : -1.f;
        const float th = pose_.yaw - 1.57079633f, c = std::cos(th), sn = std::sin(th);
        float best = -1e9f; Eigen::Vector2f bw = Eigen::Vector2f::Zero(), bt = Eigen::Vector2f::Zero();
        for (const auto& q : body_->polygon())
        {
            const Eigen::Vector2f w(c * q.x() - sn * q.y(), sn * q.x() + c * q.y());
            const float rad = w.norm();
            if (rad < 1e-6f) continue;
            const Eigen::Vector2f tan = sgn * Eigen::Vector2f(-w.y(), w.x()) / rad;
            const float score = rad + 0.02f * tan.dot(fwd) + 0.005f * w.dot(fwd);
            if (score > best) { best = score; bw = w; bt = tan; }
        }
        r.p = pose_.p + bw + bt * sigma_;   // sigma_odo outward along the sweep
        return r;
    }

    void reset() { L_.fill(0.f); body_.reset(); }

private:
    // ★ L is clamped to +-kLClamp only so exp() and long stuck/free runs stay finite and recoverable (a free base
    //   must be able to unwind a long block within a few ticks); it is not a gate: 30 nats is p = 1 - 1e-13.
    static constexpr float kLClamp = 30.f;
    static void accumulate(float& L, float c, float d, float s)
    {
        L = std::clamp(L + (c * c - 2.f * d * c) / (2.f * s * s), -kLClamp, kLClamp);
    }
    std::array<float, 4> L_{0.f, 0.f, 0.f, 0.f};   // forward, backward, turn left, turn right
    Pose2 pose_;
    float sigma_ = 0.02f;
    const rc::RobotFootprint* body_.reset();   // ★ observed only; the caller's body outlives the evidence
};
}  // namespace rc::policy
