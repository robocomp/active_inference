// unseen_contact.h — evidence of an obstacle the LiDAR cannot see, from commanded motion that odometry denies.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <vector>
#include "policy_types.h"
#include "../../../common/robot_footprint/robot_footprint.h"

namespace rc::policy
{
// ★ When the base is commanded to move and odometry shows it did not, something is holding it (below the scan
//   band, glass, a wheel caught). Each tick adds the log-likelihood ratio of the measured displacement under
//   "blocked" N(0, s) against "free" N(cmd*dt, s):  llr = (c^2 - 2 d c) / (2 s^2),  c = cmd*dt, d = measured.
//   Moving as commanded gives -c^2/(2 s^2) < 0, so L falls (break-even at d = c/2): no timer, no counter.
//   Four accumulators (forward, backward, turn left, turn right): being free to go forward says nothing about
//   backward. Ticks below the base's deadband (v_min_eff / w_min_eff) add nothing: the base would not have moved
//   anyway, so the absence of motion discriminates nothing.
//   Translation s = sigma_odo_m; rotation s = sigma_odo_m / circumscribed radius.
//   hypotheses(): one Return per direction that has ever received evidence, p_exist = logistic(L_k). There is NO
//   switch: the contact model weights by p_exist, so a weak hypothesis counts weakly.
//   ★ A hypothesis marks a PLACE, so its room-frame position is frozen where the evidence accrued: it is the
//   evidence-weighted mean of the per-tick placements, updated only on ticks that ADD blocked evidence (llr > 0).
//   When the robot backs away the obstacle stays where it was. Before any blocked evidence the latest placement
//   is reported (with a tiny p_exist it hardly matters).
//   Placement: translation = the BARE hull support point in the motion direction plus sigma_odo outward.
//   Rotation = the leading vertex of the bare hull placed with RobotFootprint::at_pose_yaw (no hand-rolled quarter
//   turn): among vertices whose sweep moves forward in the commanded sense, the largest radius from the rotation
//   centre (it sweeps the most, so it meets an obstacle first); ★ exact ties go to the lowest vertex index (the
//   hull's own order), no weights.
class UnseenContact
{
public:
    void update(float v_cmd, float w_cmd, float ds_meas, float dyaw_meas, float dt, const Pose2& pose,
                const BaseLimits& b, const rc::RobotFootprint& body, float sigma_odo_m)
    {
        const float s = std::max(sigma_odo_m, 1e-6f);
        sigma_ = sigma_odo_m;
        if (std::abs(v_cmd) >= b.v_min_eff and std::abs(v_cmd) > 0.f)
        {
            const int k = v_cmd > 0.f ? 0 : 1;
            accumulate(k, std::abs(v_cmd) * dt, v_cmd > 0.f ? ds_meas : -ds_meas, s, place(k, pose, body));
        }
        if (std::abs(w_cmd) >= b.w_min_eff and std::abs(w_cmd) > 0.f)
        {
            const int k = w_cmd > 0.f ? 2 : 3;
            const float sr = s / std::max(body.circumscribed_radius(), 1e-3f);
            accumulate(k, std::abs(w_cmd) * dt, w_cmd > 0.f ? dyaw_meas : -dyaw_meas, sr, place(k, pose, body));
        }
    }

    std::vector<Return> hypotheses() const
    {
        std::vector<Return> out;
        for (const auto& d : dir_)
            if (d.seen) out.push_back({d.pos(), sigma_, 1.f / (1.f + std::exp(-d.L))});
        return out;
    }

    float log_odds(int k) const { return dir_[k].L; }   // 0 fwd, 1 back, 2 left, 3 right
    void reset() { dir_ = {}; }

private:
    // ★ L is clamped to +-kLClamp only so exp() and long free runs stay finite and recoverable; it is not a gate
    //   (30 nats is p = 1 - 1e-13).
    static constexpr float kLClamp = 30.f;
    struct Dir
    {
        float L = 0.f, W = 0.f;   // log-odds; total blocked-evidence weight
        Eigen::Vector2f sum = Eigen::Vector2f::Zero(), latest = Eigen::Vector2f::Zero();
        bool seen = false;
        Eigen::Vector2f pos() const { return W > 0.f ? Eigen::Vector2f(sum / W) : latest; }
    };

    void accumulate(int k, float c, float d, float s, const Eigen::Vector2f& at)
    {
        Dir& x = dir_[k];
        const float llr = (c * c - 2.f * d * c) / (2.f * s * s);
        x.seen = true;
        x.latest = at;
        if (llr > 0.f) { x.sum += llr * at; x.W += llr; }
        x.L = std::clamp(x.L + llr, -kLClamp, kLClamp);
    }

    Eigen::Vector2f place(int k, const Pose2& pose, const rc::RobotFootprint& body) const
    {
        const Eigen::Vector2f fwd(std::cos(pose.yaw), std::sin(pose.yaw));
        if (k < 2)
        {
            const Eigen::Vector2f d = (k == 0 ? 1.f : -1.f) * fwd;
            // ★ support_radius_yaw includes the safety margin; the hypothesis sits on the bare hull.
            return pose.p + d * (body.support_radius_yaw(pose.yaw, d) - body.safety_margin() + sigma_);
        }
        const float sgn = (k == 2) ? 1.f : -1.f;
        float best = -1.f; Eigen::Vector2f bp = pose.p;
        for (const auto& q : body.at_pose_yaw(pose.p, pose.yaw, false))
        {
            const Eigen::Vector2f w = q - pose.p;
            const float rad = w.norm();
            if (rad < 1e-6f) continue;
            const Eigen::Vector2f tan = sgn * Eigen::Vector2f(-w.y(), w.x()) / rad;
            if (tan.dot(fwd) < 0.f) continue;   // not leading
            if (rad > best) { best = rad; bp = q + tan * sigma_; }   // strict > : ties keep the lowest index
        }
        return bp;
    }

    std::array<Dir, 4> dir_{};   // forward, backward, turn left, turn right
    float sigma_ = 0.02f;
};
}  // namespace rc::policy
