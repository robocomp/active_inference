/*
 * bottle_projection.h — DSR-side plumbing for the bottle's classifier-free contour channel.
 *
 * bottle_concept had no projection at all: its camera evidence ends at YOLO. Every other object agent
 * (chair, table, door, refrigerator, hood) also asks the image directly, with no network: project the
 * agent's OWN fitted model and score the outline against RGB edges and the depth plane. The geometry of a
 * cylinder's outline is in bottle_cylinder_contour.h (pure); this file supplies the camera pose pinned to the
 * frame's capture stamp and the ZED projector.
 */

#pragma once

#include <cstdint>
#include <optional>

#include <Eigen/Dense>
#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_camera_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>

#include "bottle_cylinder_contour.h"

namespace rc
{

// room_T_zed (camera → room) with the moving room→body hop pinned to `stamp_ms` (the frame's capture time)
// and the rigid body→zed mount at latest — the same split every other agent uses. nullopt if any hop is missing.
[[nodiscard]] std::optional<Eigen::Matrix4d> room_T_zed_at(DSR::DSRGraph& G, DSR::InnerEigenAPI& inner_eigen,
                                                           std::uint64_t stamp_ms);

// World point → pixel in the DELIVERED frame (frame_cols × frame_rows) plus the ZED depth-plane value
// (camera-frame FORWARD coordinate, not the Euclidean norm). Empty function if the camera has no size.
// ⚠ holds a reference to `cam`: use it within the lifetime of the CameraAPI.
[[nodiscard]] rc::edges::PointProjector make_zed_projector(const Eigen::Matrix4d& zed_T_room,
                                                           const DSR::CameraAPI& cam,
                                                           int frame_cols, int frame_rows);

}   // namespace rc
