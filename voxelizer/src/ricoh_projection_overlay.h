#pragma once

/*
 * ricoh_projection_overlay.{h,cpp} — projects the DSR scene (model boxes + room floor/ceiling/walls)
 * onto the RGB-360 equirectangular panorama, the 360 counterpart of ModelProjectionOverlay.
 *
 * Projection goes through the SHARED CameraAPI: the ricoh node's cam_fov≈2π selects CameraAPI's
 * equirectangular model, and cam_equirect_azimuth_sign/offset (graph intrinsics) encode the panorama
 * column convention (mirror + seam zero). So a room point is transformed to the ricoh frame via
 * room_T_ricoh and handed to cam.project() — the SAME projection path the ZED overlay uses, no
 * private equations and no per-consumer azimuth knobs. The single source of truth for the ricoh
 * pose+intrinsics is the graph (body→ricoh RT edge + the ricoh node attributes).
 *
 * Draws WIREFRAMES only (edges as seam-split subdivided polylines): equirectangular fills across the
 * wrap-seam are error-prone, and a wireframe reads more clearly in the distorted panorama. IMPORTANT
 * — draw with cv::line + cv::putText ONLY, never cv::fillPoly/cv::polylines: a bare vector<cv::Point>
 * handed to those is read as one-contour-per-point and dereferences the coordinates as pointers →
 * wild out-of-bounds writes (the ModelProjectionOverlay SIGSEGV, 2026-07-06). cv::line takes two
 * cv::Points directly and clips internally. The canvas is BGR (the ricoh popup viewer converts
 * BGR→RGB), so cv::Scalar here is (B,G,R).
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

class RicohProjectionOverlay
{
    public:
        explicit RicohProjectionOverlay(std::shared_ptr<DSRGraph> graph, std::string ricoh_node_name = "ricoh");
        ~RicohProjectionOverlay();   // out-of-line for unique_ptr<CameraAPI> of incomplete type

        // Draw onto `pano_bgr` (equirectangular BGR panorama, in place): the model-instance boxes and
        // the room floor/ceiling/walls as wireframes. `room_T_ricoh` is the ricoh pose in the room
        // frame (from the graph); its inverse takes room points into the ricoh frame for CameraAPI.
        // No-op until the panorama and the cached ricoh CameraAPI are available.
        void draw(cv::Mat& pano_bgr, std::span<const GraphObjectBox> boxes,
                  const Mat::RTMat& room_T_ricoh,
                  std::span<const float> room_poly_x, std::span<const float> room_poly_y,
                  float room_height);

        // Draw the lidar cloud reprojected into the panorama as a sparse depth overlay (points coloured
        // by range), via the SAME CameraAPI projection as draw() — a live check of the ricoh
        // extrinsics/intrinsics (points should land on the matching RGB structure). `cloud_room` is the
        // room-frame lidar sweep. No-op until the cached ricoh CameraAPI is available.
        void draw_lidar_points(cv::Mat& pano_bgr, std::span<const Eigen::Vector3f> cloud_room,
                               const Mat::RTMat& room_T_ricoh);

    private:
        bool ensure_camera_api();   // lazily create + cache the ricoh CameraAPI (one-time graph touch)

        std::shared_ptr<DSRGraph>       graph_;
        std::string                     ricoh_node_name_;
        std::unique_ptr<DSR::CameraAPI> camera_api_;   // cached ricoh equirectangular intrinsics
};

}  // namespace rc
