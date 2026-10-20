// unseen_contact.h — evidence of an obstacle the LiDAR cannot see, from commanded motion that odometry denies.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <span>
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
//   hypotheses(): one Return per direction that has accumulated BLOCKED evidence (W > 0; round-3 ruling R1 — a
//   direction only ever seen moving freely has no place for an obstacle), p_exist = logistic(L_k). No switch on p: the
//   contact model weights by p_exist, so a weak hypothesis counts weakly. Each is flagged aggregate = false (its own
//   obstacle, never merged with LiDAR returns).
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
    // ★ REVIEW FIX ROUND 2 — `pose` is the ODOMETRY pose (continuous, never relocalised): hypotheses are stored in the
    //   odometry frame, so a localisation jump cannot move a frozen hypothesis into the body (measured: an OU belief
    //   step of 13 cm put one inside the hull and every candidate read P(contact) = 1). hypotheses(room, odom) converts
    //   them to the room frame each tick; hypotheses() returns them in the odometry frame.
    // ★ REVIEW FIX ROUND 3 — THE COMMANDS IN FORCE. The base executes each command latency_s after it was sent, so within
    //   one tick it may follow several (with jitter): predicting the free motion from the latest command alone read every
    //   rapid reversal (w = +-0.3 each tick under 60 +- 20 ms latency) as "blocked" turning and froze phantom corner
    //   hypotheses (2 Stuck seeds in the Task-6 dead end). The caller passes the commands actually in force during the
    //   tick as consecutive sub-intervals {duration, v, w} summing to dt (the controller reconstructs them from its send
    //   history shifted by its latency; the harness knows them). The free prediction chains the ramp through them from
    //   the measured velocity; a command inside the deadband is a target of 0. A direction is tested when the chained
    //   free displacement is in its sense and some sub-interval actually commanded that sense above the deadband.
    struct CmdSeg { float dt, v, w; };
    void update(std::span<const CmdSeg> in_force, float v_meas, float w_meas, float ds_meas, float dyaw_meas, const Pose2& pose,
                const BaseLimits& b, const rc::RobotFootprint& body, float sigma_odo_m)
    {
        const float s = std::max(sigma_odo_m, 1e-6f);
        sigma_ = sigma_odo_m;
        tested_ = false; blocked_ = false;
        float cv = 0.f, cw = 0.f, xv = v_meas, xw = w_meas;
        bool pos_v = false, neg_v = false, pos_w = false, neg_w = false;
        // ★ Live fix 2 ruling 1: the FREE base's prediction uses the same differential-drive dynamics as rc::policy::rollout —
        //   the two wheels (v -+ w L/2) ramp at a_lin / d_lin each. With independent axes a reversal the real (coupled) base
        //   executes more slowly read as "blocked" and froze phantom hypotheses behind the robot (harness, narrow doorway).
        const float L = b.track_m > 0.f ? b.track_m : (b.a_lin > 0.f and b.a_rot > 0.f ? 2.f * b.a_lin / b.a_rot : 0.f);
        const float hl = 0.5f * L;
        float xl = xv - xw * hl, xr = xv + xw * hl, dl = 0.f, dr = 0.f;
        for (const CmdSeg& g : in_force)
        {
            const float tv = std::abs(g.v) >= b.v_min_eff ? g.v : 0.f, tw = std::abs(g.w) >= b.w_min_eff ? g.w : 0.f;
            pos_v = pos_v or tv > 0.f; neg_v = neg_v or tv < 0.f; pos_w = pos_w or tw > 0.f; neg_w = neg_w or tw < 0.f;
            if (L > 0.f)
            {
                dl += ramp(xl, tv - tw * hl, b.a_lin, b.d_lin, g.dt, &xl);
                dr += ramp(xr, tv + tw * hl, b.a_lin, b.d_lin, g.dt, &xr);
            }
            else
            {
                cv += ramp(xv, tv, b.a_lin, b.d_lin, g.dt, &xv);
                cw += ramp(xw, tw, b.a_rot, b.d_rot, g.dt, &xw);
            }
        }
        if (L > 0.f) { cv = 0.5f * (dl + dr); cw = (dr - dl) / L; }
        if ((cv > 0.f and pos_v) or (cv < 0.f and neg_v))
        {
            const int k = cv > 0.f ? 0 : 1;
            const float sg = cv > 0.f ? 1.f : -1.f;
            accumulate(k, sg * cv, sg * ds_meas, s, place(k, pose, body));
        }
        if ((cw > 0.f and pos_w) or (cw < 0.f and neg_w))
        {
            const int k = cw > 0.f ? 2 : 3;
            const float sg = cw > 0.f ? 1.f : -1.f;
            const float sr = s / std::max(body.circumscribed_radius(), 1e-3f);
            accumulate(k, sg * cw, sg * dyaw_meas, sr, place(k, pose, body));
        }
        refute(pose, body);
    }
    // One command in force for the whole tick (the Task-5 form; tests and callers without a latency queue).
    void update(float v_cmd, float w_cmd, float v_meas, float w_meas, float ds_meas, float dyaw_meas, float dt, const Pose2& pose,
                const BaseLimits& b, const rc::RobotFootprint& body, float sigma_odo_m)
    {
        const CmdSeg one{dt, v_cmd, w_cmd};
        update(std::span<const CmdSeg>(&one, 1), v_meas, w_meas, ds_meas, dyaw_meas, pose, b, body, sigma_odo_m);
    }

    // Distance covered in dt starting at x0 and ramping toward `target`; *x_end receives the speed reached.
    static float ramp(float x0, float target, float a, float d, float dt, float* x_end)
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
        if (x_end) *x_end = x;
        return dist + x * rem;
    }

private:
    void refute(const Pose2& pose, const rc::RobotFootprint& body)
    {
        // ★ ROUND 6 (B2) — REFUTATION. On a tick whose measured motion matched the command (every tested direction added
        //   free evidence, llr <= 0), a hypothesis whose place the BARE hull now occupies cannot exist: the body is there
        //   and touched nothing. Its direction's evidence is reset (W = 0, L = the prior), so it is no longer reported.
        if (tested_ and not blocked_)
        {
            rc::RobotFootprint bare = body; bare.set_safety_margin(0.f);
            for (auto& d : dir_)
                if (d.W > 0.f and bare.contains_yaw(d.pos(), pose.p, pose.yaw)) d = Dir{};
        }
    }
public:

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

    // The hypotheses in the ROOM frame: q_room = room.p + R(room.yaw - odom.yaw) (q_odom - odom.p).
    std::vector<Return> hypotheses(const Pose2& room_pose, const Pose2& odom_pose) const
    {
        auto out = hypotheses();
        const float a = room_pose.yaw - odom_pose.yaw, c = std::cos(a), sn = std::sin(a);
        for (auto& h : out)
        {
            const Eigen::Vector2f d = h.p - odom_pose.p;
            h.p = room_pose.p + Eigen::Vector2f(c * d.x() - sn * d.y(), sn * d.x() + c * d.y());
        }
        return out;
    }
    std::vector<Return> hypotheses() const
    {
        std::vector<Return> out;
        for (const auto& d : dir_)
            // ★ R1 (round 3): only directions with BLOCKED evidence (W > 0) — a direction only ever seen moving freely
            //   has no place for an obstacle; reported, it rode the hull edge at p ~ prior and held the robot Stuck.
            //   aggregate = false: a different measurement source, never merged with LiDAR returns (policy_types.h).
            if (d.seen and d.W > 0.f) out.push_back({d.pos(), sigma_, 1.f / (1.f + std::exp(-d.L)), 0.f, 0.f, 0.f, false});
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
        tested_ = true; blocked_ = blocked_ or llr > 0.f;
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
    bool tested_ = false, blocked_ = false;   // this tick: some direction tested / some tested direction read blocked
};
}  // namespace rc::policy
