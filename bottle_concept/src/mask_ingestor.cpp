/*
 * mask_ingestor.cpp — YOLO masks reading (verbatim port from table_concept).
 */

#include "mask_ingestor.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <utility>

namespace rc {

MaskIngestor::MaskIngestor(std::shared_ptr<DSR::DSRGraph> graph)
    : G_(std::move(graph))
{}

bool MaskIngestor::refresh()
{
    const auto masks_node_opt = G_->get_node("masks");
    if (not masks_node_opt.has_value())
    {
        masks_packet_ = {};
        return false;
    }

    const auto& masks_node = masks_node_opt.value();
    const auto& attrs = masks_node.attrs();

    const auto find_attr = [&](const std::string& key) -> const DSR::Attribute*
    {
        const auto it = attrs.find(key);
        return (it != attrs.end()) ? &it->second : nullptr;
    };

    const DSR::Attribute* frame_attr     = find_attr("mask_frame_id");
    const DSR::Attribute* count_attr     = find_attr("mask_count");
    const DSR::Attribute* labels_attr    = find_attr("mask_labels");
    const DSR::Attribute* label_ids_attr = find_attr("mask_label_ids");
    const DSR::Attribute* confs_attr     = find_attr("mask_confidences");
    const DSR::Attribute* offsets_attr   = find_attr("mask_support_offsets");
    const DSR::Attribute* points_attr    = find_attr("mask_support_points");
    const DSR::Attribute* centroids_attr = find_attr("mask_centroids_xyz");
    const DSR::Attribute* bbox_min_attr  = find_attr("mask_bbox_min_xyz");
    const DSR::Attribute* bbox_max_attr  = find_attr("mask_bbox_max_xyz");
    // Optional (newer voxelizer): raw 2D mask silhouette, depth-independent. Absent → diagnostic
    // falls back to projecting the support points, so don't gate the parse on these.
    const DSR::Attribute* px_attr        = find_attr("mask_pixels_xy");
    const DSR::Attribute* px_off_attr    = find_attr("mask_pixel_offsets");

    if (frame_attr == nullptr or count_attr == nullptr or labels_attr == nullptr or
        label_ids_attr == nullptr or confs_attr == nullptr or offsets_attr == nullptr or
        points_attr == nullptr or centroids_attr == nullptr or bbox_min_attr == nullptr or
        bbox_max_attr == nullptr)
    {
        masks_packet_ = {};
        return false;
    }

    const int frame_id = frame_attr->dec();
    if (frame_id <= last_masks_frame_seen_)
        return false;

    const int mask_count = std::max(0, count_attr->dec());
    const auto& labels = labels_attr->str();
    const auto& label_ids = label_ids_attr->float_vec();
    const auto& confidences = confs_attr->float_vec();
    const auto& offsets = offsets_attr->float_vec();
    const auto& support_flat = points_attr->float_vec();
    const auto& centroids_flat = centroids_attr->float_vec();
    const auto& bbox_min_flat = bbox_min_attr->float_vec();
    const auto& bbox_max_flat = bbox_max_attr->float_vec();

    const std::size_t support_count = support_flat.size() / 3;
    const std::size_t centroid_count = centroids_flat.size() / 3;
    const std::size_t bbox_min_count = bbox_min_flat.size() / 3;
    const std::size_t bbox_max_count = bbox_max_flat.size() / 3;

    std::vector<std::string> label_tokens;
    {
        std::stringstream ss(labels);
        std::string token;
        while (std::getline(ss, token, '|'))
            label_tokens.push_back(token);
    }

    MasksPacket packet;
    packet.valid = true;
    packet.frame_id = frame_id;
    packet.support_points.reserve(support_count);
    for (std::size_t i = 0; i < support_count; ++i)
        packet.support_points.emplace_back(support_flat[i*3], support_flat[i*3+1], support_flat[i*3+2]);

    static const std::vector<float> kEmptyF;
    const auto& px_flat = px_attr     ? px_attr->float_vec()     : kEmptyF;
    const auto& px_off  = px_off_attr ? px_off_attr->float_vec() : kEmptyF;
    const std::size_t pixel_count = px_flat.size() / 2;
    packet.mask_pixels.reserve(pixel_count);
    for (std::size_t i = 0; i < pixel_count; ++i)
        packet.mask_pixels.emplace_back(px_flat[i*2], px_flat[i*2+1]);

    packet.slices.reserve(static_cast<std::size_t>(mask_count));
    for (int i = 0; i < mask_count; ++i)
    {
        const std::size_t begin = (i < static_cast<int>(offsets.size())) ? static_cast<std::size_t>(std::max(0.0f, offsets[static_cast<std::size_t>(i)])) : 0;
        const std::size_t end = (i + 1 < static_cast<int>(offsets.size())) ? static_cast<std::size_t>(std::max(0.0f, offsets[static_cast<std::size_t>(i + 1)])) : begin;
        const std::size_t clamped_begin = std::min(begin, packet.support_points.size());
        const std::size_t clamped_end = std::min(end, packet.support_points.size());

        const auto fetch_vec3 = [&](const std::vector<float>& flat, std::size_t count) -> Eigen::Vector3f
        {
            if (static_cast<std::size_t>(i) >= count)
                return Eigen::Vector3f::Zero();
            return {flat[i*3], flat[i*3+1], flat[i*3+2]};
        };

        MaskSlice slice;
        if (static_cast<std::size_t>(i) < label_tokens.size())
            slice.label = label_tokens[static_cast<std::size_t>(i)];
        if (static_cast<std::size_t>(i) < label_ids.size())
            slice.class_id = label_ids[static_cast<std::size_t>(i)];
        if (static_cast<std::size_t>(i) < confidences.size())
            slice.confidence = confidences[static_cast<std::size_t>(i)];
        slice.support_begin = clamped_begin;
        slice.support_end = clamped_end;
        const std::size_t pbegin = (i < static_cast<int>(px_off.size())) ? static_cast<std::size_t>(std::max(0.0f, px_off[static_cast<std::size_t>(i)])) : 0;
        const std::size_t pend   = (i + 1 < static_cast<int>(px_off.size())) ? static_cast<std::size_t>(std::max(0.0f, px_off[static_cast<std::size_t>(i + 1)])) : pbegin;
        slice.pixel_begin = std::min(pbegin, packet.mask_pixels.size());
        slice.pixel_end   = std::min(pend,   packet.mask_pixels.size());
        slice.centroid = fetch_vec3(centroids_flat, centroid_count);
        slice.bbox_min = fetch_vec3(bbox_min_flat, bbox_min_count);
        slice.bbox_max = fetch_vec3(bbox_max_flat, bbox_max_count);
        packet.slices.push_back(slice);
    }

    masks_packet_ = std::move(packet);
    last_masks_frame_seen_ = frame_id;
    return true;
}

std::optional<MaskIngestor::MaskSlice>
MaskIngestor::select_for_bottle(const BottleInstance& inst) const
{
    if (not masks_packet_.valid or masks_packet_.slices.empty())
        return std::nullopt;

    const auto& state = inst.model.state();
    const Eigen::Vector3f bottle_centroid(state.cx, state.cy, state.cz);

    float best_cost = std::numeric_limits<float>::max();
    std::optional<MaskSlice> best;
    for (const auto& slice : masks_packet_.slices)
    {
        if (slice.label != "bottle")
            continue;

        const float dx = slice.centroid.x() - bottle_centroid.x();
        const float dy = slice.centroid.y() - bottle_centroid.y();
        const float dz = slice.centroid.z() - bottle_centroid.z();
        const float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (dist < best_cost)
        {
            best_cost = dist;
            best = slice;
        }
    }

    return best;
}

std::vector<Eigen::Vector3f>
MaskIngestor::read_pts_attrib(const DSR::Node& node, const std::string& att_name) const
{
    std::vector<Eigen::Vector3f> pts;

    std::optional<std::reference_wrapper<const std::vector<float>>> opt;
    if (att_name == "candidate_pts_att")
        opt = G_->get_attrib_by_name<candidate_pts_att>(node);
    else if (att_name == "residual_pts_att")
        opt = G_->get_attrib_by_name<residual_pts_att>(node);
    else
        return pts;

    if (not opt.has_value())
        return pts;

    const auto& data = opt.value().get();
    const std::size_t n = data.size() / 3;
    pts.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        pts.emplace_back(data[i*3], data[i*3+1], data[i*3+2]);

    return pts;
}

}  // namespace rc
