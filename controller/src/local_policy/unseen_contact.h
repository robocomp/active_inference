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
//   "blocked" N(0, s) against "free" N(c, s):  llr = (c^2 - 2 d c) / (2 s^2),  d = measured.
//   ★ D3 (controller ruling, Task 6): c is the displacement the base WOULD ACTUALLY make if free — from the MEASURED
//     velocity at the start of the tick, ramping toward the command at a_lin/d_lin (a_rot/d_rot; through zero on a sign
//     change; a rate <= 0 is instantaneous) over dt — not cmd*dt. With cmd*dt every start-up ramp read as "blocked"
//     (Task-6 closed loop: hypotheses at p ~ 0.45 on the hull edge at t = 0.15 s, the robot held Stuck "returns inside
//     the body" in open space). A tick whose free displacement is not in the commanded sense (still coasting the other
//     way) tests nothing about an obstacle ahead and adds nothing.
//   ★ PRIOR: L starts at logit(kPriorBlocked), kPriorBlocked = 0.01 — the fraction of commanded motions blocked by
//     something the LiDAR cannot see (stated, not fitted). It was 0 (p 0.5): one ambiguous tick put a coin-flip
//     obstacle on the body edge.
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
    // v_meas / w_meas: the base's measured velocity at the START of the tick (odometry); ds/dyaw: measured over dt.
    void update(float v_cmd, float w_cmd, float v_meas, float w_meas, float ds_meas, float dyaw_meas, float dt, const Pose2& pose,
                const BaseLimits& b, const rc::RobotFootprint& body, float sigma_odo_m)
    {
        const float s = std::max(sigma_odo_m, 1e-6f);
        sigma_ = sigma_odo_m;
        if (std::abs(v_cmd) >= b.v_min_eff and std::abs(v_cmd) > 0.f)
        {
            const int k = v_cmd > 0.f ? 0 : 1;
            const float sg = v_cmd > 0.f ? 1.f : -1.f;
            const float c = sg * free_travel(v_meas, v_cmd, b.a_lin, b.d_lin, dt);
            if (c > 0.f) accumulate(k, c, sg * ds_meas, s, place(k, pose, body));
        }
        if (std::abs(w_cmd) >= b.w_min_eff and std::abs(w_cmd) > 0.f)
        {
            const int k = w_cmd > 0.f ? 2 : 3;
            const float sg = w_cmd > 0.f ? 1.f : -1.f;
            const float sr = s / std::max(body.circumscribed_radius(), 1e-3f);
            const float c = sg * free_travel(w_meas, w_cmd, b.a_rot, b.d_rot, dt);
            if (c > 0.f) accumulate(k, c, sg * dyaw_meas, sr, place(k, pose, body));
        }
    }

    // Distance one axis covers in dt starting at x0 and ramping toward `target` (a when |speed| grows, d when it shrinks,
    // through zero on a sign change; rate <= 0 = instantaneous) — the FREE base's prediction.
    static float free_travel(float x0, float target, float a, float d, float dt)
    {
        float x = x0, rem = dt, dist = 0.f;
        while (rem > 1e-9f and x != target)
        {
            const bool through_zero = (x > 0.f and target < 0.f) or (x < 0.f and target > 0.f);
            const float goal = through_zero ? 0.f : target;
            const float rate = std::abs(goal) < std::abs(x) ? d : a;
            if (rate <= 0.f) { x = goal; continue; }
            const float need = std::abs(goal - x) / rate, tt = std::min(need, rem);
            const float x1 = x + (goal > x ? 1.f : -1.f) * rate * tt;
            dist += 0.5f * (x + x1) * tt; x = x1; rem -= tt;
        }
        return dist + x * rem;
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
public:
    static constexpr float kPriorBlocked = 0.01f;
private:
    struct Dir
    {
        float L = std::log(kPriorBlocked / (1.f - kPriorBlocked)), W = 0.f;   // log-odds (from the prior); blocked-evidence weight
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
