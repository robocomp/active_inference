/*
 * skeleton_source.h
 *
 * Decoupled body-keypoint input for human_concept. The agent consumes BODY_18 3D
 * skeletons through this abstract interface, so the live source (ZED SDK / a media-
 * plane skeleton frame / a DSR producer) can be swapped later without touching the
 * fitter or scene graph. The first backend is a CSV replay (reuses the cpp/harness
 * format), so the full agent can be brought up and tested without a camera.
 */

#pragma once

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "body18.h"                  // rc::human::NUM_KP
#include "human_kinematic_model.h"   // rc::human::KpArray

namespace DSR { class DSRGraph; class InnerEigenAPI; }

namespace rc {

// One tracked body in a frame: a stable track id + its 18×3 keypoints (+ optional per-joint conf).
struct SkeletonBody
{
    int     id = 0;
    human::KpArray kp;                                        // (18,3); NaN rows = missing
    std::optional<std::array<float, human::NUM_KP>> conf;     // per-joint confidence in [0,100]
};

// Abstract source. poll() returns the bodies for the current step and advances internally.
class SkeletonSource
{
public:
    virtual ~SkeletonSource() = default;
    virtual std::vector<SkeletonBody> poll() = 0;
    virtual bool ok() const = 0;
};

// CSV replay backend. Row format (lines starting with '#' ignored):
//   frame[,id], kp0_x..kp17_z (54) [, c0..c17 (18)]
// id column is auto-detected from the column count; absent → id 0. Each row is one body/step.
class ReplaySkeletonSource : public SkeletonSource
{
public:
    ReplaySkeletonSource(std::string path, bool loop);
    std::vector<SkeletonBody> poll() override;
    bool ok() const override { return not rows_.empty(); }

private:
    std::vector<SkeletonBody> rows_;
    std::size_t cursor_ = 0;
    bool loop_ = true;
};

// Live DSR backend: reads the voxelizer's 'skeleton' node (BODY_18 keypoints in the CAMERA frame,
// metres) and returns one SkeletonBody per detected person. Keypoints are transformed CAMERA→world
// (room) via the live camera RT so the existing room-frame fitter places people correctly; the
// on-wire data stays localization-clean, leaving the interaction-time re-parent (person→robot for
// visual servoing) to a future path. Returns empty until a new frame id appears (no reprocessing).
class DsrSkeletonSource : public SkeletonSource
{
public:
    DsrSkeletonSource(std::shared_ptr<DSR::DSRGraph> graph,
                      DSR::InnerEigenAPI* inner_eigen,
                      std::string world_frame = "room",
                      std::string camera_frame = "zed");
    std::vector<SkeletonBody> poll() override;
    bool ok() const override { return static_cast<bool>(G_); }

private:
    std::shared_ptr<DSR::DSRGraph> G_;
    DSR::InnerEigenAPI* inner_eigen_ = nullptr;
    std::string world_frame_;
    std::string camera_frame_;
    int last_frame_id_ = -1;
};

// Factory from config. kind = "replay" (CSV) | "dsr"/"live" (voxelizer 'skeleton' node). The DSR
// backend needs the graph + inner-eigen API; if absent it falls back to replay.
std::unique_ptr<SkeletonSource> make_skeleton_source(const std::string& kind,
                                                     const std::string& replay_path,
                                                     bool replay_loop,
                                                     std::shared_ptr<DSR::DSRGraph> graph = nullptr,
                                                     DSR::InnerEigenAPI* inner_eigen = nullptr);

}  // namespace rc
