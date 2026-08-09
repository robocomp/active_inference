/*
 * bottle_lidar_ingestor.h  —  lidar3D media-plane consumer for the YOLO-independent range factor
 *
 * Brings up the shared zero-copy LidarFrame subscriber (same descriptor-driven factory every agent uses) and,
 * once per compute() cycle, stages the newest sweep transformed into the ROOM frame plus the sensor origin —
 * exactly what BottleBelief's LiDAR first-hit factor needs (see common/ai_belief/lidar_ray_factor.h).
 *
 * Crash-safety (mirrors room_concept::LidarIngestor, the sanctioned pattern):
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

#include "bottle_config.h"

namespace rc::media { class LidarPlaneReader; }

namespace rc
{

class BottleLidarIngestor
{
public:
    BottleLidarIngestor(std::shared_ptr<DSR::DSRGraph> graph, DSR::InnerEigenAPI* inner_eigen,
                        const BottleConfig& cfg);
    ~BottleLidarIngestor();

    // Poll the media plane; on a NEW sweep, transform it into the room frame and stage it. Returns true iff a
    // fresh sweep was staged this call. Main-thread only (reads the DSR graph).
    bool pump();

    const std::vector<Eigen::Vector3f>& sweep_room()  const { return sweep_room_; }
    const Eigen::Vector3f&              origin_room() const { return origin_room_; }

private:
    std::shared_ptr<DSR::DSRGraph>               G_;
    DSR::InnerEigenAPI*                          inner_eigen_ = nullptr;
    const BottleConfig*                          cfg_ = nullptr;
    // Shared media-plane reader (the same one every agent uses): the high "helios" plane (DEVICE
    // frame) transformed straight to the ROOM frame, with the fused "lidar3D" plane as fallback. A
    // single plane keeps the first-hit ray factor's single-origin assumption (see lidar_ray_factor.h).
    std::unique_ptr<rc::media::LidarPlaneReader> reader_;

    std::vector<Eigen::Vector3f> sweep_room_;                    // latest sweep, ROOM frame
    Eigen::Vector3f              origin_room_ = Eigen::Vector3f::Zero();  // sensor centre, ROOM frame
};

}  // namespace rc
