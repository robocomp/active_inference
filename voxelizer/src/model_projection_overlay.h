#pragma once

/*
 * model_projection_overlay.{h,cpp} — projects the DSR scene onto the live ZED image.
 *
 * Draws, in the RGB canvas:
 *   - every model instance (object/table/cylinder/obstacle) as an oriented 3D bounding box
 *     (12 edges + name label), from the pre-built GraphObjectBox structs the scene supplies;
 *   - the room floor and ceiling as light translucent meshes (room polygon at z=0 / z=room_height);
 *   - every DSR wall node as a light translucent mesh labelled with its name.
 *
 * Deliberately independent of the YOLO/viewer pipeline so future projected geometry (skeletons,
 * per-instance covariance ellipsoids, …) can be added here without touching specificworker/compute().
 * Mirrors room_concept's CameraVisualizer::draw_projections, but consumes pre-built GraphObjectBox
 * structs (single source of truth = SceneProcessor::build_graph_object_box) for the boxes and draws
 * with OpenCV onto a cv::Mat instead of QPainter onto a QImage.
 */

#include <opencv2/core.hpp>

#include <cstdint>
#include <memory>
#include <span>
#include <string>

#include <Eigen/Geometry>
#include <dsr/api/dsr_eigen_defs.h>   // Mat::RTMat = Eigen::Transform<double,3,Affine>

#include "graph_object_box.h"

namespace DSR { class DSRGraph; class CameraAPI; }
using DSRGraph = DSR::DSRGraph;

namespace rc
{

// Projects the DSR scene (model boxes + room floor/ceiling/walls) onto the live ZED image.
//
// IMPORTANT: draw() does NO live graph traversal. All geometry is passed in from the caller's
// already-gathered per-frame data (room←zed transform + room polygon), which the agent collects on
// the main thread in one place. An earlier version re-read the graph here every frame (inner_eigen
// tree-walks + get_nodes_by_type) and raced the DDS reader threads inserting residual/obstacle nodes
// → heap corruption / SIGSEGV. The only graph touch left is a ONE-TIME lazy CameraAPI creation for
// the static zed intrinsics (cached afterwards).
class ModelProjectionOverlay
{
    public:
        ModelProjectionOverlay(std::shared_ptr<DSRGraph> graph,
                               std::string camera_node_name = "zed");
        ~ModelProjectionOverlay();   // out-of-line for unique_ptr<CameraAPI> of incomplete type

        // Draw onto `canvas` (RGB, in place): the model-instance boxes in `boxes`, plus the room floor
        // and ceiling (room polygon at z=0 / z=room_height) and one wall per polygon edge, as light
        // translucent meshes. `room_T_zed` is the zed pose in the room frame at the capture time
        // (maps zed→room); its inverse gives the room→camera basis. No-op until the polygon and the
        // cached zed intrinsics are available.
        void draw(cv::Mat& canvas, std::span<const GraphObjectBox> boxes,
                  const Mat::RTMat& room_T_zed,
                  std::span<const float> room_poly_x, std::span<const float> room_poly_y,
                  float room_height);

    private:
        bool ensure_camera_api();   // lazily create + cache the zed CameraAPI (one-time graph touch)

        std::shared_ptr<DSRGraph>          graph_;
        std::string                        camera_node_name_;
        std::unique_ptr<DSR::CameraAPI>    camera_api_;   // cached zed intrinsics (created once)
};

}  // namespace rc
