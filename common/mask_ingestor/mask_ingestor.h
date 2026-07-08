/*
 * common/mask_ingestor/mask_ingestor.h
 *
 * Shared perception input layer for the concept agents (bottle/table/chair/…): reads the YOLO
 * "masks" DSR node (written by the voxelizer), parses it into per-instance mask slices + support
 * points + raw 2D silhouette pixels, and serves the nearest slice of a requested label to a query
 * centroid. Object-AGNOSTIC: the caller passes the query point + label, so this is a single shared
 * file (no <Obj>Instance dependency). Also reads the per-node candidate/residual point attributes.
 *
 * Owns the parsed MasksPacket (and the last-seen frame guard); consumers read it read-only via
 * packet(). Plain class (no Q_OBJECT) constructed by SpecificWorker once G is ready.
 *
 * FRAME CONTRACT (this class IS the single agreement point with the voxelizer producer):
 *   The producer dual-publishes mask support points, 1-to-1:
 *     - "mask_support_points"      → ROOM frame  (default path here)
 *     - "mask_support_points_cam"  → ZED/camera frame
 *   Frame is chosen PER CONSUMER, here, not at the producer:
 *     - default            → read room-frame points as-is  (table_concept, chair_concept)
 *     - enable_frame_transform(...) → read the _cam points, transform src→target via inner_eigen
 *       pinned to the capture stamp, and RECOMPUTE per-slice centroids/bboxes in the target frame
 *       (bottle_concept). No *_cam centroid/bbox attribute exists or is needed for this reason.
 *   Both producer arrays must keep existing; switching everyone to one frame is NOT a producer change.
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <Eigen/Dense>
#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>

namespace rc {

class MaskIngestor
{
public:
    struct MaskSlice
    {
        std::string label;
        float class_id = -1.0f;
        float confidence = 0.0f;
        Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
        Eigen::Vector3f bbox_min = Eigen::Vector3f::Zero();
        Eigen::Vector3f bbox_max = Eigen::Vector3f::Zero();
        std::size_t support_begin = 0;
        std::size_t support_end = 0;
        std::size_t pixel_begin = 0;   // raw 2D mask pixel range (into MasksPacket::mask_pixels)
        std::size_t pixel_end = 0;
        // Ego-motion capture-corruption channel (see ../MASK_MOTION_CORRUPTION.md). Per-mask, written by
        // the voxelizer producer; 0 when the producer predates the feature (backward-compatible).
        float motion_var      = 0.0f;  // variance to ADD to R (m²): exposure blur + timing jitter, ×peripheral
        float motion_bias     = 0.0f;  // systematic displacement from a known timing offset (m) → GATE if large
        float motion_dotd     = 0.0f;  // metric position-corruption speed Z·‖ṡ‖ (m/s), diagnostic
        float trunc_frac      = 0.0f;  // fraction of silhouette pixels on the image border → unobserved face
        float centroid_radius = 0.0f;  // normalised centroid radius from the principal point (periphery)
        float range           = 0.0f;  // mean camera→mask depth Z (m): STATIC range — consumer grows R + pose
                                       // common-mode with it so a distant view can't resolve pose/orientation
        // RGB-360 (ricoh) peripheral evidence (Part B, RICOH_360_PERIPHERAL_DETECTION.md). A no-depth slice
        // carries NO 3D points (support_begin==support_end, centroid/bbox are NaN) — only a room-frame
        // BEARING. Consumers MUST skip !has_depth slices until a bearing confirm/birth path exists (Part C).
        // Defaults match the zed contract, so a producer that predates the field reads back has_depth=true.
        bool  has_depth        = true;   // false ⇒ ricoh bearing-only slice
        float azimuth_room_rad = 0.0f;   // room-frame bearing; meaningful only when has_depth==false
        // Depth-uncertainty channel (common/depth_projection). σ_range² (m²) to ADD to R along the mask
        // ray, SAME currency as motion_var — sum them. 0 for dense-depth zed masks; the scored range
        // variance for ricoh masks depth-filled from reprojected lidar (grows as hits get sparse/scattered,
        // → the mask degrades back toward bearing-only). 0 when the producer predates the field.
        float depth_var        = 0.0f;
    };

    struct MasksPacket
    {
        bool valid = false;
        int frame_id = -1;
        std::uint64_t timestamp_ms = 0;   // capture stamp of the source RGBD frame (0 = producer didn't publish one)
        std::vector<MaskSlice> slices;
        std::vector<Eigen::Vector3f> support_points;
        std::vector<Eigen::Vector2f> mask_pixels;   // raw YOLO foreground (col,row), depth-independent
        // Per-frame ego-motion (optical frame), written by the producer; empty/0 if absent.
        std::array<float, 6> cam_twist{};   // [vx,vy,vz,wx,wy,wz]
        float                frame_dt_s = 0.0f;
    };

    explicit MaskIngestor(std::shared_ptr<DSR::DSRGraph> graph);

    // Opt-in (Part B): transform support points from the producer's SOURCE frame into the consumer's
    // TARGET frame via inner_eigen, pinned to the mask capture stamp. When enabled, points are read
    // from "mask_support_points_cam" (camera/source frame, dual-published by the voxelizer) and
    // transformed src→tgt; per-slice centroids/bboxes are recomputed in the target frame. When NOT
    // enabled (default), points come room-frame from "mask_support_points" (legacy — table/chair).
    void enable_frame_transform(DSR::InnerEigenAPI* inner_eigen,
                                std::string source_frame, std::string target_frame);

    // Re-read the "masks" node. Returns true only when a NEW frame was ingested.
    bool refresh();

    // The most recently ingested packet (read-only).
    const MasksPacket& packet() const { return masks_packet_; }

    // Nearest `label`-labelled slice to `query_centroid` (room frame). nullopt if none. The caller
    // builds the centroid from its model state, so this stays object-agnostic.
    std::optional<MaskSlice> select_nearest(const Eigen::Vector3f& query_centroid,
                                            std::string_view label) const;

    // Read a flat float3 point attribute (candidate_pts_att / residual_pts_att) off a node.
    std::vector<Eigen::Vector3f> read_pts_attrib(const DSR::Node& node, const std::string& att_name) const;

private:
    std::shared_ptr<DSR::DSRGraph> G_;
    MasksPacket                    masks_packet_;
    int                            last_masks_frame_seen_ = -1;

    // Opt-in source→target frame transform (see enable_frame_transform).
    DSR::InnerEigenAPI* inner_eigen_       = nullptr;
    std::string         src_frame_;
    std::string         tgt_frame_;
    bool                transform_enabled_ = false;
};

}  // namespace rc
