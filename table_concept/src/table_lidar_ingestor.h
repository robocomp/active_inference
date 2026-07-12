/*
 * table_lidar_ingestor.h  —  lidar3D media-plane consumer for the YOLO-independent range factor
 *
 * Port of bottle_concept/bottle_lidar_ingestor.h (the sanctioned pattern). Brings up the shared zero-copy
 * LidarFrame subscriber (same descriptor-driven factory every agent uses) and, once per compute() cycle,
 * stages the newest sweep transformed into the ROOM frame plus the sensor origin — exactly what the shared
 * LiDAR first-hit factor needs (see common/ai_belief/lidar_ray_factor.h). Because TableBelief exposes the
 * same sdf_prim / sdf_jacobian / n_prims hooks as the bottle, the factor itself is unchanged.
 *
 * Crash-safety (mirrors bottle_concept::BottleLidarIngestor / room_concept::LidarIngestor):
 *  - The subscriber is created LAZILY in pump() (called from the Operating compute/main thread), never in a
 *    ctor or a free-running thread, and only once the "lidar3D" node + media descriptor exist. Discovery is
 *    self-throttled to ~1 Hz.
 *  - pump() reads the DSR graph (inner_eigen room←lidar3D) — it MUST be called on the main thread.
 *  - The owner (SpecificWorker) must reset this ingestor BEFORE tearing the graph down, while G is alive.
 *  - It stays entirely dormant (no DDS participant) while cfg.lidar_precision == 0, so the feature is a true
 *    no-op when off.
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>

#include "table_config.h"

namespace rc::media { class LidarPlaneReader; }

namespace rc
{

class TableLidarIngestor
{
public:
    TableLidarIngestor(std::shared_ptr<DSR::DSRGraph> graph, DSR::InnerEigenAPI* inner_eigen,
                       const TableConfig& cfg);
    ~TableLidarIngestor();

    // Poll BOTH planes (helios primary, bpearl if LidarBpearlPrecision>0); on a NEW sweep, transform each into
    // the room frame at its OWN capture stamp and stage it. Returns true iff a fresh sweep was staged for EITHER
    // plane this call. Per-plane freshness via helios_fresh()/bpearl_fresh(). Main-thread only (reads the graph).
    bool pump();

    const std::vector<Eigen::Vector3f>& sweep_room()  const { return sweep_room_; }
    const Eigen::Vector3f&              origin_room() const { return origin_room_; }
    bool                                helios_fresh() const { return helios_fresh_; }
    // Low bpearl plane (separate per-device ray-set; own origin). Only pumped while LidarBpearlPrecision>0.
    const std::vector<Eigen::Vector3f>& sweep_bpearl_room()  const { return sweep_bpearl_room_; }
    const Eigen::Vector3f&              origin_bpearl_room() const { return origin_bpearl_room_; }
    bool                                bpearl_fresh() const { return bpearl_fresh_; }

private:
    std::shared_ptr<DSR::DSRGraph>               G_;
    DSR::InnerEigenAPI*                          inner_eigen_ = nullptr;
    const TableConfig*                           cfg_ = nullptr;
    // Per-device media-plane readers: the high "helios" 360 plane and the low "bpearl" plane, each a SINGLE
    // plane transformed straight to the ROOM frame — a single plane per reader keeps the first-hit ray factor's
    // origin well-defined (see lidar_ray_factor.h). They are consumed as SEPARATE ray-sets, never merged.
    std::unique_ptr<rc::media::LidarPlaneReader> reader_;
    std::unique_ptr<rc::media::LidarPlaneReader> reader_bpearl_;

    std::vector<Eigen::Vector3f> sweep_room_;                    // latest helios sweep, ROOM frame
    Eigen::Vector3f              origin_room_ = Eigen::Vector3f::Zero();  // helios sensor centre, ROOM frame
    bool                         helios_fresh_ = false;
    std::vector<Eigen::Vector3f> sweep_bpearl_room_;             // latest bpearl sweep, ROOM frame
    Eigen::Vector3f              origin_bpearl_room_ = Eigen::Vector3f::Zero();
    bool                         bpearl_fresh_ = false;
};

}  // namespace rc
