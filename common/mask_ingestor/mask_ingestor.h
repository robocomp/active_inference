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
 */

#pragma once

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
    };

    struct MasksPacket
    {
        bool valid = false;
        int frame_id = -1;
        std::uint64_t timestamp_ms = 0;   // capture stamp of the source RGBD frame (0 = producer didn't publish one)
        std::vector<MaskSlice> slices;
        std::vector<Eigen::Vector3f> support_points;
        std::vector<Eigen::Vector2f> mask_pixels;   // raw YOLO foreground (col,row), depth-independent
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
