/*
 * hood_lidar_range_channel.h — YOLO-independent LiDAR range channel for hood_concept (extracted from HoodFitter).
 *
 * Owns the staged per-cycle sweep (room frame + sensor origin) and selects the returns landing on ONE hood,
 * staging them on the frame's LiDAR channel for the first-hit range factor / free-space VACATE term:
 * set_sweep (stage this cycle's sweep + origin, from HoodFitter fed by HoodLidarIngestor), clear (drop the
 * staged sweep, called each cycle before set_sweep so a stale sweep never leaks), and feed (select this
 * hood's returns and populate frame.lidar). Holds the HoodConfig by reference. Plain class (no Q_OBJECT).
 */

#pragma once

#include <vector>

#include <Eigen/Dense>

#include "hood_config.h"     // rc::HoodConfig
#include "hood_instance.h"   // rc::HoodInstance
#include "hood_belief.h"     // rc::HoodFrame

namespace rc {

class HoodLidarRangeChannel
{
public:
    explicit HoodLidarRangeChannel(const HoodConfig& cfg) : cfg_(cfg) {}

    // Stage this cycle's primary (helios) sweep (room frame) + sensor origin for the range factor / VACATE.
    void set_sweep(const std::vector<Eigen::Vector3f>& sweep_room, const Eigen::Vector3f& origin_room)
    { lidar_sweep_room_ = sweep_room; lidar_origin_room_ = origin_room; lidar_have_sweep_ = true; }
    // Stage this cycle's low (bpearl) sweep + its OWN origin — a SEPARATE per-device ray-set (own occlusion-aware
    // first-hit). Range factor only (the VACATE term stays on the primary helios sweep).
    void set_sweep_bpearl(const std::vector<Eigen::Vector3f>& sweep_room, const Eigen::Vector3f& origin_room)
    { lidar_sweep_bpearl_ = sweep_room; lidar_origin_bpearl_ = origin_room; lidar_have_bpearl_ = true; }
    // clear() each cycle first so a stale sweep never leaks into a frame with no fresh LiDAR.
    void clear() { lidar_have_sweep_ = false; lidar_have_bpearl_ = false; }

    // Select this cycle's LiDAR returns landing on THIS hood and stage them on the frame's range channel(s).
    void feed(HoodInstance& inst, HoodFrame& frame) const;

private:
    // Select returns from `sweep` (sensor at `origin`) that land on this hood into `out`, with the two
    // informativeness down-weights (ray-count + angular coverage) applied to `precision_base`. Returns the
    // staged ray count; writes the angular-coverage weight to *cov_ang (if non-null). Shared by helios + bpearl.
    int select_into(const HoodInstance& inst, const Eigen::Vector3f& centroid,
                    const std::vector<Eigen::Vector3f>& sweep, const Eigen::Vector3f& origin,
                    float precision_base, rc::ai::LidarRays& out, float* cov_ang = nullptr) const;

    // Birth-footprint select box: rotation-agnostic circumscribed XY radius + floor-referenced z-band (spans the
    // legs + rim). Fixed from the BIRTH dims (never the fitted w/h, so a blown-up extent can't explode the region).
    struct SelectBox { float rxy, z_lo, z_hi; };
    // z-band follows the fitted top, clamped to the height prior's +/-2 sigma. Pass the current fitted H.
    SelectBox select_box(float fitted_z_top_m) const;

    const HoodConfig&             cfg_;
    // Staged LiDAR sweeps for the range factor (room frame). Refreshed each compute() cycle; have flag false
    // ⇒ no fresh sweep this cycle ⇒ that plane is a no-op (never reuses a stale sweep).
    std::vector<Eigen::Vector3f>   lidar_sweep_room_;
    Eigen::Vector3f                lidar_origin_room_ = Eigen::Vector3f::Zero();
    bool                           lidar_have_sweep_  = false;
    std::vector<Eigen::Vector3f>   lidar_sweep_bpearl_;
    Eigen::Vector3f                lidar_origin_bpearl_ = Eigen::Vector3f::Zero();
    bool                           lidar_have_bpearl_ = false;
};

}  // namespace rc
