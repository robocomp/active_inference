/*
 * common/mask_ingestor/mask_ingestor.cpp — shared YOLO "masks" DSR node reading.
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

void MaskIngestor::enable_frame_transform(DSR::InnerEigenAPI* inner_eigen,
                                          std::string source_frame, std::string target_frame)
{
    inner_eigen_       = inner_eigen;
    src_frame_         = std::move(source_frame);
    tgt_frame_         = std::move(target_frame);
    transform_enabled_ = (inner_eigen_ != nullptr) and not src_frame_.empty() and not tgt_frame_.empty();
}

bool MaskIngestor::refresh()
{
    const auto masks_node_opt = G_->get_node("masks");
    if (!masks_node_opt.has_value())
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

    const DSR::Attribute* frame_attr = find_attr("mask_frame_id");
    const DSR::Attribute* ts_attr = find_attr("mask_timestamp_ms");   // optional (newer producers only)
    const DSR::Attribute* count_attr = find_attr("mask_count");
    const DSR::Attribute* labels_attr = find_attr("mask_labels");
    const DSR::Attribute* label_ids_attr = find_attr("mask_label_ids");
    const DSR::Attribute* confs_attr = find_attr("mask_confidences");
    const DSR::Attribute* offsets_attr = find_attr("mask_support_offsets");
    const DSR::Attribute* points_attr = find_attr("mask_support_points");
    const DSR::Attribute* centroids_attr = find_attr("mask_centroids_xyz");
    const DSR::Attribute* bbox_min_attr = find_attr("mask_bbox_min_xyz");
    const DSR::Attribute* bbox_max_attr = find_attr("mask_bbox_max_xyz");
    const DSR::Attribute* pixels_attr = find_attr("mask_pixels_xy");
    const DSR::Attribute* pixel_offsets_attr = find_attr("mask_pixel_offsets");
    // Ego-motion capture-corruption channel (optional; newer producers only) — see MASK_MOTION_CORRUPTION.md
    const DSR::Attribute* motion_var_attr      = find_attr("mask_motion_var");
    const DSR::Attribute* motion_bias_attr     = find_attr("mask_motion_bias");
    const DSR::Attribute* motion_dotd_attr     = find_attr("mask_motion_dotd");
    const DSR::Attribute* trunc_frac_attr      = find_attr("mask_trunc_frac");
    const DSR::Attribute* centroid_radius_attr = find_attr("mask_centroid_radius");
    const DSR::Attribute* range_attr           = find_attr("mask_range");
    const DSR::Attribute* cam_twist_attr       = find_attr("mask_cam_twist");
    const DSR::Attribute* frame_dt_attr        = find_attr("mask_frame_dt_s");

    if (frame_attr == nullptr || count_attr == nullptr || labels_attr == nullptr ||
        label_ids_attr == nullptr || confs_attr == nullptr || offsets_attr == nullptr ||
        points_attr == nullptr || centroids_attr == nullptr || bbox_min_attr == nullptr ||
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
    static const std::vector<float> empty_flat;
    const auto& pixels_flat  = pixels_attr        ? pixels_attr->float_vec()        : empty_flat;
    const auto& pixel_offsets = pixel_offsets_attr ? pixel_offsets_attr->float_vec() : empty_flat;
    const auto& motion_var_v      = motion_var_attr      ? motion_var_attr->float_vec()      : empty_flat;
    const auto& motion_bias_v     = motion_bias_attr     ? motion_bias_attr->float_vec()     : empty_flat;
    const auto& motion_dotd_v     = motion_dotd_attr     ? motion_dotd_attr->float_vec()     : empty_flat;
    const auto& trunc_frac_v      = trunc_frac_attr      ? trunc_frac_attr->float_vec()      : empty_flat;
    const auto& centroid_radius_v = centroid_radius_attr ? centroid_radius_attr->float_vec() : empty_flat;
    const auto& range_v           = range_attr           ? range_attr->float_vec()           : empty_flat;

    // Part B: with frame-transform enabled, source the camera-frame support array and transform it to
    // the target frame below; otherwise use the legacy room-frame array as-is.
    const DSR::Attribute* points_cam_attr = find_attr("mask_support_points_cam");
    const bool use_cam = transform_enabled_ and points_cam_attr != nullptr
                         and not points_cam_attr->float_vec().empty();
    const auto& source_flat = use_cam ? points_cam_attr->float_vec() : support_flat;

    const std::size_t support_count = source_flat.size() / 3;
    const std::size_t pixel_count = pixels_flat.size() / 2;
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
    packet.timestamp_ms = ts_attr ? ts_attr->uint64() : 0;   // 0 → consumer falls back to latest pose
    if (cam_twist_attr and cam_twist_attr->float_vec().size() >= 6)
        for (int k = 0; k < 6; ++k) packet.cam_twist[k] = cam_twist_attr->float_vec()[k];
    packet.frame_dt_s = frame_dt_attr ? frame_dt_attr->fl() : 0.0f;

    // src→tgt transform (Part B): one matrix for the whole frame, pinned to the capture stamp. Built
    // element-wise to dodge the Eigen-alignment ABI trap. Identity fallback if the chain isn't ready
    // (points stay in source frame this cycle — better than dropping the frame).
    Eigen::Matrix4d T_tgt_src = Eigen::Matrix4d::Identity();
    if (use_cam)
        if (const auto Topt = inner_eigen_->get_transformation_matrix(tgt_frame_, src_frame_, packet.timestamp_ms);
            Topt.has_value())
        {
            const auto& s = Topt.value().matrix();
            for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) T_tgt_src(r, c) = s(r, c);
        }

    packet.support_points.reserve(support_count);
    for (std::size_t i = 0; i < support_count; ++i)
    {
        if (use_cam)
        {
            const Eigen::Vector4d q = T_tgt_src * Eigen::Vector4d(source_flat[i*3], source_flat[i*3+1], source_flat[i*3+2], 1.0);
            packet.support_points.emplace_back(static_cast<float>(q.x()), static_cast<float>(q.y()), static_cast<float>(q.z()));
        }
        else
            packet.support_points.emplace_back(source_flat[i*3], source_flat[i*3+1], source_flat[i*3+2]);
    }

    packet.mask_pixels.reserve(pixel_count);
    for (std::size_t i = 0; i < pixel_count; ++i)
        packet.mask_pixels.emplace_back(pixels_flat[i*2], pixels_flat[i*2+1]);

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
        const auto fetch1 = [&](const std::vector<float>& v) -> float
        { return static_cast<std::size_t>(i) < v.size() ? v[static_cast<std::size_t>(i)] : 0.0f; };
        slice.motion_var      = fetch1(motion_var_v);
        slice.motion_bias     = fetch1(motion_bias_v);
        slice.motion_dotd     = fetch1(motion_dotd_v);
        slice.trunc_frac      = fetch1(trunc_frac_v);
        slice.centroid_radius = fetch1(centroid_radius_v);
        slice.range           = fetch1(range_v);
        slice.support_begin = clamped_begin;
        slice.support_end = clamped_end;
        {
            const std::size_t pbegin = (i < static_cast<int>(pixel_offsets.size())) ? static_cast<std::size_t>(std::max(0.0f, pixel_offsets[static_cast<std::size_t>(i)])) : 0;
            const std::size_t pend = (i + 1 < static_cast<int>(pixel_offsets.size())) ? static_cast<std::size_t>(std::max(0.0f, pixel_offsets[static_cast<std::size_t>(i + 1)])) : pbegin;
            slice.pixel_begin = std::min(pbegin, packet.mask_pixels.size());
            slice.pixel_end   = std::min(pend,   packet.mask_pixels.size());
        }
        if (use_cam and clamped_end > clamped_begin)
        {
            // Recompute centroid + bbox in the TARGET frame from the transformed support points (the
            // published room-frame centroids/bboxes are stale once we re-frame the points).
            Eigen::Vector3f c = Eigen::Vector3f::Zero();
            Eigen::Vector3f mn = Eigen::Vector3f::Constant(std::numeric_limits<float>::max());
            Eigen::Vector3f mx = Eigen::Vector3f::Constant(std::numeric_limits<float>::lowest());
            for (std::size_t k = clamped_begin; k < clamped_end; ++k)
            {
                const auto& p = packet.support_points[k];
                c += p; mn = mn.cwiseMin(p); mx = mx.cwiseMax(p);
            }
            slice.centroid = c / static_cast<float>(clamped_end - clamped_begin);
            slice.bbox_min = mn;
            slice.bbox_max = mx;
        }
        else
        {
            slice.centroid = fetch_vec3(centroids_flat, centroid_count);
            slice.bbox_min = fetch_vec3(bbox_min_flat, bbox_min_count);
            slice.bbox_max = fetch_vec3(bbox_max_flat, bbox_max_count);
        }
        packet.slices.push_back(slice);
    }

    masks_packet_ = std::move(packet);
    last_masks_frame_seen_ = frame_id;
    return true;
}

std::optional<MaskIngestor::MaskSlice>
MaskIngestor::select_nearest(const Eigen::Vector3f& query_centroid, std::string_view label) const
{
    if (!masks_packet_.valid || masks_packet_.slices.empty())
        return std::nullopt;

    float best_cost = std::numeric_limits<float>::max();
    std::optional<MaskSlice> best;
    for (const auto& slice : masks_packet_.slices)
    {
        if (slice.label != label)
            continue;

        const float dist = (slice.centroid - query_centroid).norm();
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
