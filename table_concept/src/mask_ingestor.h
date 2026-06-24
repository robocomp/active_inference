/*
 * mask_ingestor.h
 *
 * Perception input layer for table_concept: reads the YOLO "masks" DSR node
 * (written by the voxelizer), parses it into per-instance mask slices + support
 * points, and serves the nearest "table"-labelled slice to a given instance.
 * Also reads the per-table candidate/residual point attributes written on a node.
 *
 * Owns the parsed MasksPacket (and the last-seen frame guard); consumers read it
 * read-only via packet(). Mirrors bottle_concept/mask_ingestor.h. Plain class
 * (no Q_OBJECT) constructed by SpecificWorker once G is ready.
 */

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <dsr/api/dsr_api.h>

#include "table_instance.h"

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
        std::vector<MaskSlice> slices;
        std::vector<Eigen::Vector3f> support_points;
        std::vector<Eigen::Vector2f> mask_pixels;   // raw YOLO foreground (col,row), depth-independent
    };

    explicit MaskIngestor(std::shared_ptr<DSR::DSRGraph> graph);

    // Re-read the "masks" node. Returns true only when a NEW frame was ingested.
    bool refresh();

    // The most recently ingested packet (read-only).
    const MasksPacket& packet() const { return masks_packet_; }

    // Nearest "table"-labelled slice to the instance's current centroid (nullopt if none).
    std::optional<MaskSlice> select_for_table(const TableInstance& inst) const;

    // Read a flat float3 point attribute (candidate_pts_att / residual_pts_att) off a node.
    std::vector<Eigen::Vector3f> read_pts_attrib(const DSR::Node& node, const std::string& att_name) const;

private:
    std::shared_ptr<DSR::DSRGraph> G_;
    MasksPacket                    masks_packet_;
    int                            last_masks_frame_seen_ = -1;
};

}  // namespace rc
