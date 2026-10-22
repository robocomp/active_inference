// sensing.h — the DoorCore's views over the sensor part of a DoorCycleInput: the mask packet (MaskSource), the zed
// camera (CameraModel + the ego-motion it implies) and the low LiDAR sweep (LeafSweep). DSR-free (src/core/ rule).
//
// ★WHY VIEWS AND NOT THE INGESTORS. The live agent gets these from rc::MaskIngestor / rc::ConceptLidarIngestor /
// DSR::CameraAPI, which all pull the graph or the media plane. The core sees only what the adapter recorded into the
// input — and the input says "nullopt = nothing new, keep the last one" for the big media (door_cycle_io.h), which is
// exactly the ingestors' packet()/frame() semantics. So each view HOLDS the last value it was given and answers the
// questions the old code asked the ingestor, with the same arithmetic.
// ★THEY MUST SEE EVERY CYCLE. A packet staged on a cycle where the core's decision step does not run (no room yet)
// is still "the last packet" on the next cycle; DoorCore::observe_media() exists so the adapter (and the replay) can
// hand the media over on those cycles too — otherwise the core would decide on a staler packet than the ingestor holds.
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>
#include <Eigen/Dense>
#include "door_cycle_io.h"

namespace rc::door::core
{

/// The last mask packet the adapter delivered (MaskIngestor::packet()).
class MaskSource
{
public:
    void apply(const std::optional<MaskPacket>& p) { if (p.has_value()) packet_ = p.value(); }
    [[nodiscard]] const MaskPacket& packet() const { return packet_; }
    /// The room-frame support points of one slice, clamped to the array (the tracker's association reads them).
    [[nodiscard]] std::span<const Vec3f> support(const MaskSlice& s) const
    {
        const std::size_t n = packet_.support_points.size();
        const std::size_t b = std::min<std::size_t>(s.support_begin, n);
        const std::size_t e = std::min<std::size_t>(s.support_end, n);
        return e > b ? std::span<const Vec3f>(packet_.support_points.data() + b, e - b) : std::span<const Vec3f>{};
    }
private:
    MaskPacket packet_{};
};

/// The zed camera as the core sees it: the pose and the intrinsics recorded at the top of the cycle.
struct CameraModel
{
    /// room ← zed at ts=0 composed EXACTLY as DoorFitter::room_T_zed_matrix composes it: (room ← body) · (body ← zed),
    /// in double, from the two transforms the snapshot recorded. nullopt while either does not resolve.
    [[nodiscard]] static std::optional<Mat4d> room_T_zed(const RobotState& r)
    {
        if (not (r.room_T_body.has_value() and r.body_T_zed.has_value()))
            return std::nullopt;
        return Mat4d(r.room_T_body.value() * r.body_T_zed.value());
    }
};

/// Robot/camera ego-motion (room frame) from the camera pose — moved out of DoorFitter::update_ego_motion, line for
/// line, so the birth gate (frame_admissible) and the fit's confirm-only gate read ONE value. The only change: the
/// pose and the clock come from the cycle's input (recorded at the TOP of the cycle) instead of a cortex query and a
/// steady_clock read a few milliseconds later, mid-cycle. ★That is a real, named difference (Review Focus 1): dt is
/// measured between cycle tops rather than between the two mid-cycle reads, and the pose is the cycle-top snapshot.
struct EgoMotion
{
    float lin_mps   = 0.0f;
    float ang_radps = 0.0f;

    void update(const std::optional<Mat4d>& M, std::int64_t t_steady_ns)
    {
        if (not M.has_value())
        {
            have_prev_ = false;           // pose chain unavailable → can't judge motion; reset baseline
            return;
        }
        const Eigen::Vector3f pos(static_cast<float>(M->coeff(0, 3)),
                                  static_cast<float>(M->coeff(1, 3)),
                                  static_cast<float>(M->coeff(2, 3)));
        const Eigen::Vector3f fwd(static_cast<float>(M->coeff(0, 1)),   // zed +y is the depth/forward axis
                                  static_cast<float>(M->coeff(1, 1)),
                                  static_cast<float>(M->coeff(2, 1)));
        if (have_prev_)
        {
            // The same std::chrono conversion the old code made (steady time_point difference → float seconds).
            const float dt = std::max(1e-3f, std::chrono::duration<float>(
                                                 std::chrono::nanoseconds(t_steady_ns - prev_ns_)).count());
            lin_mps = (pos - prev_pos_).norm() / dt;
            const float fa = prev_fwd_.norm(), fb = fwd.norm();
            const float cang = (fa > 1e-6f and fb > 1e-6f)
                ? std::clamp(prev_fwd_.dot(fwd) / (fa * fb), -1.0f, 1.0f) : 1.0f;
            ang_radps = std::acos(cang) / dt;
        }
        prev_pos_ = pos; prev_fwd_ = fwd; prev_ns_ = t_steady_ns; have_prev_ = true;
    }

private:
    Eigen::Vector3f prev_pos_ = Eigen::Vector3f::Zero();
    Eigen::Vector3f prev_fwd_ = Eigen::Vector3f::UnitY();
    std::int64_t    prev_ns_ = 0;
    bool            have_prev_ = false;
};

/// The newest low-LiDAR (bpearl) sweep in the ROOM frame with its origin (ConceptLidarIngestor::sweep_bpearl_room /
/// origin_bpearl_room as DoorFitter::set_leaf_points received them). The fit's leaf-angle channel (door_fit.cpp
/// estimate_phi) reads it.
/// ★present(): the old fitter held a POINTER to the ingestor's sweep, null until the first room-frame sweep was staged,
/// and the leaf channel's `have_sensor` was exactly "that pointer is set" — an empty-but-staged sweep still counts.
class LeafSweep
{
public:
    void apply(const std::optional<BpearlSweep>& s)
    {
        if (s.has_value() and s->room_fresh) { pts_ = s->pts_room; origin_ = s->origin_room; present_ = true; }
    }
    [[nodiscard]] bool present() const { return present_; }
    [[nodiscard]] const std::vector<Vec3f>& points() const { return pts_; }
    [[nodiscard]] const Vec3f& origin() const { return origin_; }
private:
    std::vector<Vec3f> pts_;
    Vec3f origin_ = Vec3f::Zero();
    bool present_ = false;
};

}  // namespace rc::door::core
