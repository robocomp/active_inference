/*
 * door_rt_publish.h — when door_concept rewrites a door's RT edge (DoorSceneGraph::write_rt_pose). Pure; tested by
 * tests/test_door_rt_publish.cpp.
 *
 * ★Task D-C (R2 review M-6): THE DEAD-BAND MEASURES HOW FAR THE APERTURE MOVED, NOT HOW FAR ITS CENTRE MOVED. The old rule
 * compared only the centre's translation with 5 cm, so a pure yaw re-fit about a fixed centre was never published — and the
 * restored DOOR_0, first initialised in the wrong wall (yaw 1.547 for −3.20), could be corrected by the fit and still leave
 * room_concept reading the 90°-off aperture. The aperture is a segment; its consumers read its two jambs. So the move is
 * the larger displacement of the two jambs, c ± (w/2)·(cos yaw, sin yaw), and it is compared with the SAME 5 cm the
 * translation used: one constant, in metres of the door's own geometry — a yaw change is published exactly when it moves
 * a jamb as far as a translation would have had to.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>
#include <string>

#include <Eigen/Dense>

namespace rc::door
{

/// The rewrite dead-band (m): below this jamb displacement the RT edge is left as it is (gradient oscillation churn).
inline constexpr float kRtMinJambMoveM = 0.05f;

/// The larger displacement of the aperture's two jambs between the written pose and the new one.
[[nodiscard]] inline float aperture_jamb_move(float old_cx, float old_cy, float old_yaw,
                                              float cx, float cy, float yaw, float half_w)
{
    if (not std::isfinite(old_cx) or not std::isfinite(old_cy) or not std::isfinite(old_yaw))
        return std::numeric_limits<float>::infinity();   // never written
    const Eigen::Vector2f c0(old_cx, old_cy), c1(cx, cy);
    const Eigen::Vector2f u0(std::cos(old_yaw), std::sin(old_yaw)), u1(std::cos(yaw), std::sin(yaw));
    const float h = std::max(0.0f, half_w);
    const float a = ((c1 + h * u1) - (c0 + h * u0)).norm();
    const float b = ((c1 - h * u1) - (c0 - h * u0)).norm();
    return std::max(a, b);
}

[[nodiscard]] inline bool rt_write_due(float old_cx, float old_cy, float old_yaw, float cx, float cy, float yaw, float half_w)
{
    return aperture_jamb_move(old_cx, old_cy, old_yaw, cx, cy, yaw, half_w) >= kRtMinJambMoveM;
}

/// A door's room-frame aperture pose as the GRAPH currently publishes it.
struct PublishedPose { float cx, cy, yaw; };

/// The inverse of door_in_wall: the door's local RT (lx, ly along/across the wall, lyaw) under a wall whose RT puts its
/// midpoint at `mid` with tangent `wall_yaw` (room frame; floor->room is the identity) ⇒ the room-frame pose.
[[nodiscard]] inline PublishedPose door_room_pose(const Eigen::Vector2f& mid, float wall_yaw, float lx, float ly, float lyaw)
{
    const float c = std::cos(wall_yaw), s = std::sin(wall_yaw);
    return {mid.x() + c * lx - s * ly, mid.y() + s * lx + c * ly,
            std::remainder(wall_yaw + lyaw, 2.0f * static_cast<float>(M_PI))};
}

/// ★ROOM-TRANSITION RULE 1 (2026-10-29): room_concept moves a wall IN PLACE when the layout changes, and the door's RT
/// (its offset along the wall) goes with it — so the pose the graph publishes can leave the door's belief without the
/// belief moving. The dead-band therefore compares the belief with what the graph PUBLISHES when that can be read, and
/// with what this agent last wrote only when it cannot (no edge yet). A slid door is re-seated on the next cycle.
[[nodiscard]] inline bool rt_write_due_published(const std::optional<PublishedPose>& published,
                                                 float last_cx, float last_cy, float last_yaw,
                                                 float cx, float cy, float yaw, float half_w)
{
    if (published.has_value())
        return rt_write_due(published->cx, published->cy, published->yaw, cx, cy, yaw, half_w);
    return rt_write_due(last_cx, last_cy, last_yaw, cx, cy, yaw, half_w);
}

/// A planar rigid transform: p' = R(yaw)·p + t. Room-transition case 2: ltsm's proposal carries T(P<-K) on its RT edge; the
/// adopted room's frame changes P -> K, so what door_concept holds in P's frame moves by T(K<-P) = inverse.
struct Rigid2 { float tx = 0.0f, ty = 0.0f, yaw = 0.0f; };
[[nodiscard]] inline Rigid2 rigid_inverse(const Rigid2& T)
{
    const float c = std::cos(T.yaw), s = std::sin(T.yaw);
    return {-(c * T.tx + s * T.ty), -(-s * T.tx + c * T.ty), -T.yaw};
}
[[nodiscard]] inline Eigen::Vector2f rigid_apply(const Rigid2& T, const Eigen::Vector2f& p)
{
    const float c = std::cos(T.yaw), s = std::sin(T.yaw);
    return {c * p.x() - s * p.y() + T.tx, s * p.x() + c * p.y() + T.ty};
}
/// A pose (position + heading) seen in frame B, given the frame change `B_from_A`: the door's pose in the new frame.
[[nodiscard]] inline PublishedPose rigid_apply(const Rigid2& B_from_A, const PublishedPose& p)
{
    const auto q = rigid_apply(B_from_A, Eigen::Vector2f(p.cx, p.cy));
    return {q.x(), q.y(), std::remainder(p.yaw + B_from_A.yaw, 2.0f * static_cast<float>(M_PI))};
}

/// A ∘ B (apply B, then A).
[[nodiscard]] inline Rigid2 rigid_compose(const Rigid2& A, const Rigid2& B)
{
    const auto t = rigid_apply(A, Eigen::Vector2f(B.tx, B.ty));
    return {t.x(), t.y(), std::remainder(A.yaw + B.yaw, 2.0f * static_cast<float>(M_PI))};
}
/// Δ^s on SE(2) (log, scale, exp): the same twist held for s times as long. s = 0 is the identity, s = 1 is Δ.
[[nodiscard]] inline Rigid2 rigid_power(const Rigid2& D, float s)
{
    const float th = std::remainder(D.yaw, 2.0f * static_cast<float>(M_PI));
    // V(θ) = [[a, -b], [b, a]], a = sin θ/θ, b = (1 - cos θ)/θ  (→ identity as θ → 0)
    const auto V = [](float t) -> Eigen::Matrix2f
    {
        const float a = std::abs(t) < 1e-4f ? 1.0f - t * t / 6.0f : std::sin(t) / t;
        const float b = std::abs(t) < 1e-4f ? 0.5f * t : (1.0f - std::cos(t)) / t;
        return (Eigen::Matrix2f() << a, -b, b, a).finished();
    };
    const Eigen::Vector2f rho = V(th).inverse() * Eigen::Vector2f(D.tx, D.ty);
    const Eigen::Vector2f t = V(s * th) * (s * rho);
    return {t.x(), t.y(), s * th};
}

/// The robot's pose in the followed room's frame as one RT block: the pose and the block's stamp (ms).
struct RobotFix { Rigid2 pose; std::uint64_t stamp_ms = 0; };

/// ★THE FRAME CHANGE OF AN ACCEPTED ADOPTION IS THE PROPOSAL EDGE'S, READ AT THE VERDICT (Task 20). room_concept computes
/// T(P<-K) from its own fit (robot_P ∘ robot_K⁻¹) and writes it on the RT edge P -> proposal BEFORE state = "accepted", so a
/// reader that sees "accepted" reads the ACCEPTED transform — never ltsm's prior, which recog_proto6 showed a half turn
/// off. One fact, three readers (room_concept's own edges, ltsm's carry-over, these doors): no wait, no robot pairing.
/// nullopt unless the proposal is accepted and its edge was readable.
[[nodiscard]] inline std::optional<Rigid2> accepted_K_from_P(std::string_view state, const std::optional<Rigid2>& edge_P_from_K)
{
    if (state != "accepted" or not edge_P_from_K.has_value()) return std::nullopt;
    return rigid_inverse(*edge_P_from_K);
}

/// How far two frame changes T(K<-P) disagree, at P's origin (metres) and in heading (degrees): the robot-paired one is
/// logged against the accepted one as a CROSS-CHECK only (it never moves a door).
struct FrameDiscrepancy { float d_xy = 0.f, d_yaw_deg = 0.f; };
[[nodiscard]] inline FrameDiscrepancy frame_discrepancy(const Rigid2& a, const Rigid2& b)
{
    return {std::hypot(a.tx - b.tx, a.ty - b.ty),
            std::abs(std::remainder(a.yaw - b.yaw, 2.0f * static_cast<float>(M_PI))) * 180.0f / static_cast<float>(M_PI)};
}

/// T(K<-P) FROM THE ROBOT — since Task 20 a CROSS-CHECK of the accepted transform, logged, never applied (Task 19, recog_proto6). ltsm's proposal edge carries T(P<-K) built from
/// ITS best pose mode; room_concept's adoption fit then relocalises the robot against K's outline, and what it publishes
/// in K is the FIT's pose — 179° / 3.12 m away from ltsm's mode in recog_proto6, so the proposal's frame change moved the
/// entry mirror door_43 6.5 m off DOOR_2's hole (a 180° turn about (0.21,-0.19) in K). The frame change every consumer
/// must agree with is the one the robot's published pose implies: the same physical robot, seen in P's frame just before
/// the switch and in K's frame just after, T(K<-P) = robot_K(t) ∘ robot_P(t)⁻¹. The two blocks are not simultaneous
/// (room_concept publishes no pose in the cycle that adopts), so the P pose is carried to the K block's stamp by the
/// body-frame motion of the last two P blocks (constant twist; a body-frame twist is frame-independent).
///
/// Which blocks are which frame: every block read while the room still wore P's name is P's (room_concept renames
/// first and publishes in K only on a later cycle). A block read after the rename may still be P's last one, so K's
/// first block is the first one STRICTLY NEWER than the newest block read in the cycle the rename was first seen.
class AdoptionFramePairing
{
public:
    /// One read per cycle of the followed room (id, current name) and the robot's newest block in it (nullopt: unreadable).
    void observe(std::uint64_t room, const std::string& name, const std::optional<RobotFix>& fix)
    {
        if (room != room_) { room_ = room; name_ = name; prev_.reset(); last_.reset(); fence_.reset(); }
        if (name == name_)
        {
            if (fix.has_value() and (not last_.has_value() or fix->stamp_ms > last_->stamp_ms)) { prev_ = last_; last_ = fix; }
            return;
        }
        if (not fence_.has_value() and fix.has_value()) fence_ = fix->stamp_ms;   // the rename's cycle: maybe still P's
    }
    /// Robot blocks were read under the previous name: the pairing can (eventually) answer.
    [[nodiscard]] bool has_previous() const { return last_.has_value(); }
    /// T(K<-P) once `now` is a block of the new frame; nullopt while it may still be P's (or nothing was read under P).
    [[nodiscard]] std::optional<Rigid2> K_from_P(const std::optional<RobotFix>& now) const
    {
        if (not last_.has_value() or not fence_.has_value() or not now.has_value() or now->stamp_ms <= *fence_)
            return std::nullopt;
        Rigid2 robot_P = last_->pose;
        if (prev_.has_value() and last_->stamp_ms > prev_->stamp_ms)
        {
            const float s = static_cast<float>(now->stamp_ms - last_->stamp_ms) / static_cast<float>(last_->stamp_ms - prev_->stamp_ms);
            robot_P = rigid_compose(last_->pose, rigid_power(rigid_compose(rigid_inverse(prev_->pose), last_->pose), s));
        }
        return rigid_compose(now->pose, rigid_inverse(robot_P));
    }
    /// The adoption was acted on: the current name is the frame from now on.
    void rebase(const std::string& name) { name_ = name; prev_.reset(); last_.reset(); fence_.reset(); }

private:
    std::uint64_t room_ = 0;
    std::string name_;
    std::optional<RobotFix> prev_, last_;
    std::optional<std::uint64_t> fence_;
};

}  // namespace rc::door
