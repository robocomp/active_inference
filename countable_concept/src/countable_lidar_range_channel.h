/*
 * countable_lidar_range_channel.h — YOLO-independent LiDAR range channel for countable_concept (extracted from CountableFitter).
 *
 * Owns the staged per-cycle sweep (room frame + sensor origin) and selects the returns landing on ONE bed,
 * staging them on the frame's LiDAR channel for the first-hit range factor / free-space VACATE term:
 * set_sweep (stage this cycle's sweep + origin, from CountableFitter fed by ConceptLidarIngestor), clear (drop the
 * staged sweep, called each cycle before set_sweep so a stale sweep never leaks), and feed (select this
 * bed's returns and populate frame.lidar). Holds the CountableConfig by reference. Plain class (no Q_OBJECT).
 */

#pragma once

#include <functional>
#include <span>
#include <vector>

#include <Eigen/Dense>

#include "countable_config.h"     // rc::CountableConfig
#include "countable_instance.h"   // rc::CountableInstance
#include "countable_belief.h"     // rc::CountableFrame

namespace rc {

class CountableLidarRangeChannel
{
public:
    explicit CountableLidarRangeChannel(const CountableConfig& cfg) : cfg_(cfg) {}

    // Stage this cycle's primary (helios) sweep (room frame) + sensor origin for the range factor / VACATE.
    void set_sweep(const std::vector<Eigen::Vector3f>& sweep_room, const Eigen::Vector3f& origin_room)
    { lidar_sweep_room_ = sweep_room; lidar_origin_room_ = origin_room; lidar_have_sweep_ = true; }
    // Stage this cycle's low (bpearl) sweep + its OWN origin — a SEPARATE per-device ray-set (own occlusion-aware
    // first-hit). Range factor only (the VACATE term stays on the primary helios sweep).
    void set_sweep_bpearl(const std::vector<Eigen::Vector3f>& sweep_room, const Eigen::Vector3f& origin_room)
    { lidar_sweep_bpearl_ = sweep_room; lidar_origin_bpearl_ = origin_room; lidar_have_bpearl_ = true; }
    // clear() each cycle first so a stale sweep never leaks into a frame with no fresh LiDAR.
    void clear() { lidar_have_sweep_ = false; lidar_have_bpearl_ = false; }

    // Select this cycle's LiDAR returns landing on THIS bed and stage them on the frame's range channel(s). The select box
    // is anchored on `anchor_cloud` (this cycle's mask cloud; on a LiDAR-only frame the camera points that are NOT fed to
    // the engine, spec §7b). An empty `membership` ⇒ select_into (today's selection, bit for bit); else every return
    // carries the camera's m(p) as a weight (select_into_weighted) and Σm goes to inst.dbg_lidar_msum.
    void feed(CountableInstance& inst, CountableFrame& frame, std::span<const Eigen::Vector3f> anchor_cloud,
              const std::function<float(const Eigen::Vector3f&)>& membership = {}) const;

private:
    // Select returns from `sweep` (sensor at `origin`) that land on this bed into `out`, with the two
    // informativeness down-weights (ray-count + angular coverage) applied to `precision_base`. Returns the
    // staged ray count; writes the angular-coverage weight to *cov_ang (if non-null). Shared by helios + bpearl.
    int select_into(const CountableInstance& inst, const Eigen::Vector3f& centroid,
                    const std::vector<Eigen::Vector3f>& sweep, const Eigen::Vector3f& origin,
                    float precision_base, rc::ai::LidarRays& out, float* cov_ang = nullptr,
                    const std::function<float(const Eigen::Vector3f&)>* membership = nullptr,
                    float* m_sum = nullptr) const;

    // Birth-footprint select box: rotation-agnostic circumscribed XY radius + floor-referenced z-band (spans the
    // legs + rim). Fixed from the BIRTH dims (never the fitted w/h, so a blown-up extent can't explode the region).
    struct SelectBox { float rxy, z_lo, z_hi; };
    SelectBox select_box() const;

    const CountableConfig&             cfg_;
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
