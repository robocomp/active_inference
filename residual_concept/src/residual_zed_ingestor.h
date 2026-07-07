/*
 * residual_zed_ingestor.h  —  ZED depth media-plane consumer for the dense-depth boost.
 *
 * Brings up the shared zero-copy ImageFrame subscriber on the "zed" node's "depth" stream (descriptor-driven,
 * same factory every agent uses) and, once per compute() cycle, stages the newest depth image (metres, camera
 * frame) + the camera intrinsics. The worker then projects each residual box into it and backprojects the
 * region (residual_zed_boost.h). Crash-safety mirrors the LiDAR ingestor: lazy subscriber on the main thread,
 * reset before graph teardown.
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <memory>

#include <dsr/api/dsr_api.h>

#include "residual_config.h"
#include "residual_zed_boost.h"   // DepthImage, CamIntrinsics

namespace rc::media { class MediaSubscriber; }

namespace rc
{

class ResidualZedIngestor
{
public:
    ResidualZedIngestor(std::shared_ptr<DSR::DSRGraph> graph, const ResidualConfig& cfg);
    ~ResidualZedIngestor();

    bool pump();   // drain the newest depth frame; true iff fresh this call. Main-thread only.

    const DepthImage&    depth()      const { return depth_; }
    const CamIntrinsics& intrinsics() const { return K_; }
    std::uint64_t        stamp_ms()   const { return static_cast<std::uint64_t>(last_ts_); }
    bool                 has_depth()  const { return depth_.width > 0 and K_.fx > 0.0f; }

private:
    bool ensure_subscriber();
    void read_intrinsics(int width, int height);

    std::shared_ptr<DSR::DSRGraph>              G_;
    const ResidualConfig*                       cfg_ = nullptr;
    std::unique_ptr<rc::media::MediaSubscriber> depth_sub_;

    DepthImage    depth_;
    CamIntrinsics K_;
    std::int64_t  last_ts_ = 0;
    std::chrono::steady_clock::time_point last_init_{};
};

}  // namespace rc
