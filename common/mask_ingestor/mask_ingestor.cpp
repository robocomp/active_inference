/*
 * common/mask_ingestor/mask_ingestor.cpp  —  shared YOLO "masks" DSR node reader (implementation)
 *
 * SHARED across the concept agents: parses the voxelizer-written "masks" node into a MasksPacket (slices +
 * support points + raw silhouette pixels + the per-mask corruption/range/bearing channels), optionally
 * re-frames the support points src→target (pinned to the capture stamp, recomputing centroids/bboxes there),
 * and serves the nearest slice of a requested label. Frame-agnostic; the caller chooses room vs camera frame.
 */

#include <optional>
#include "mask_ingestor.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <print>
#include <sstream>
#include <utility>

#include <QDateTime>   // wall-clock ms source for the producer-liveness probes (kept out of the header)

namespace rc {

// ─── Construction / configuration ────────────────────────────────────────────────────────────────

MaskIngestor::MaskIngestor(std::shared_ptr<DSR::DSRGraph> graph)
    : G_(std::move(graph))
{}

// Opt in (Part B) to reading camera-frame support points and transforming them src→target per capture stamp.
void MaskIngestor::enable_frame_transform(DSR::InnerEigenAPI* inner_eigen,
                                          std::string source_frame, std::string target_frame)
{
    inner_eigen_       = inner_eigen;
    src_frame_         = std::move(source_frame);
    tgt_frame_         = std::move(target_frame);
    transform_enabled_ = (inner_eigen_ != nullptr) and not src_frame_.empty() and not tgt_frame_.empty();
}

// ─── Ingest one masks frame ──────────────────────────────────────────────────────────────────────

// Re-read the "masks" node; returns true only when a NEW frame (higher mask_frame_id) was ingested.
bool MaskIngestor::refresh()
{
    const auto masks_node_opt = G_->get_node("masks");
    if (!masks_node_opt.has_value())
    {
        masks_packet_ = {};
        return false;
    }

    const auto& masks_node = masks_node_opt.value();

    // TYPE-ATTRIBUTED reads (CLAUDE.md): compile-checked against dsr_attr_name.h. Held in named optionals so
    // the reference_wrapper payloads stay alive for the rest of the function (they reference masks_node, which
    // is in scope throughout). The vec() helper gives the previous "attr ? ->float_vec() : empty" semantics.
    static const std::vector<float> empty_flat;
    using VecOpt = std::optional<std::reference_wrapper<const std::vector<float>>>;
    const auto vref = [](const VecOpt& o) -> const std::vector<float>& { return o ? o->get() : empty_flat; };

    // REQUIRED core attributes — all must be present or this is not a valid mask packet.
    const auto frame_opt     = G_->get_attrib_by_name<mask_frame_id_att>(masks_node);
    const auto count_opt     = G_->get_attrib_by_name<mask_count_att>(masks_node);
    const VecOpt label_ids_opt = G_->get_attrib_by_name<mask_label_ids_att>(masks_node);
    const VecOpt confs_opt     = G_->get_attrib_by_name<mask_confidences_att>(masks_node);
    const VecOpt offsets_opt   = G_->get_attrib_by_name<mask_support_offsets_att>(masks_node);
    const VecOpt points_opt    = G_->get_attrib_by_name<mask_support_points_att>(masks_node);
    const VecOpt centroids_opt = G_->get_attrib_by_name<mask_centroids_xyz_att>(masks_node);
    const VecOpt bbox_min_opt  = G_->get_attrib_by_name<mask_bbox_min_xyz_att>(masks_node);
    const VecOpt bbox_max_opt  = G_->get_attrib_by_name<mask_bbox_max_xyz_att>(masks_node);
    const auto labels_opt      = G_->get_attrib_by_name<mask_labels_att>(masks_node);   // ref_wrapper<const string>

    if (not frame_opt or not count_opt or not labels_opt or not label_ids_opt or not confs_opt
        or not offsets_opt or not points_opt or not centroids_opt or not bbox_min_opt or not bbox_max_opt)
    {
        masks_packet_ = {};
        return false;
    }

    const int frame_id = frame_opt.value();
    if (frame_id == last_masks_frame_seen_)
        return false;                          // same frame republished (producer frozen) — not a new ingest
    if (frame_id < last_masks_frame_seen_)
        // mask_frame_id is a per-PROCESS publish counter; a backward jump means the producer (voxelizer)
        // RESTARTED and reset it. Adopt the new stream (fall through to ingest) — otherwise refresh() would
        // reject every frame until the fresh counter climbs past the stale pre-restart value, silently
        // starving ALL mask consumers after a producer restart.
        std::println("[MaskIngestor] mask_frame_id {} < last-seen {} — producer restarted; adopting new stream",
                     frame_id, last_masks_frame_seen_);

    // OPTIONAL attributes (newer-producer channels): timestamp, pixels, ego-motion corruption, RGB-360 bearing.
    const auto   ts_opt          = G_->get_attrib_by_name<mask_timestamp_ms_att>(masks_node);   // uint64, optional
    const VecOpt pixels_opt      = G_->get_attrib_by_name<mask_pixels_xy_att>(masks_node);
    const VecOpt pixel_offs_opt  = G_->get_attrib_by_name<mask_pixel_offsets_att>(masks_node);
    const VecOpt motion_var_opt  = G_->get_attrib_by_name<mask_motion_var_att>(masks_node);
    const VecOpt motion_bias_opt = G_->get_attrib_by_name<mask_motion_bias_att>(masks_node);
    const VecOpt motion_dotd_opt = G_->get_attrib_by_name<mask_motion_dotd_att>(masks_node);
    const VecOpt trunc_frac_opt  = G_->get_attrib_by_name<mask_trunc_frac_att>(masks_node);
    const VecOpt cent_rad_opt    = G_->get_attrib_by_name<mask_centroid_radius_att>(masks_node);
    const VecOpt range_opt       = G_->get_attrib_by_name<mask_range_att>(masks_node);
    const VecOpt has_depth_opt   = G_->get_attrib_by_name<mask_has_depth_att>(masks_node);
    const VecOpt azimuth_opt     = G_->get_attrib_by_name<mask_azimuth_att>(masks_node);
    const VecOpt depth_var_opt   = G_->get_attrib_by_name<mask_depth_var_att>(masks_node);
    const VecOpt color_rgb_opt   = G_->get_attrib_by_name<mask_color_rgb_att>(masks_node);
    const VecOpt color_var_opt   = G_->get_attrib_by_name<mask_color_var_att>(masks_node);
    const VecOpt color_neff_opt  = G_->get_attrib_by_name<mask_color_neff_att>(masks_node);
    const auto   cam_twist_opt   = G_->get_attrib_by_name<mask_cam_twist_att>(masks_node);
    const auto   frame_dt_opt    = G_->get_attrib_by_name<mask_frame_dt_s_att>(masks_node);

    const int mask_count = std::max(0, count_opt.value());
    const auto& labels = labels_opt.value().get();
    const auto& label_ids = label_ids_opt.value().get();
    const auto& confidences = confs_opt.value().get();
    const auto& offsets = offsets_opt.value().get();
    const auto& support_flat = points_opt.value().get();
    const auto& centroids_flat = centroids_opt.value().get();
    const auto& bbox_min_flat = bbox_min_opt.value().get();
    const auto& bbox_max_flat = bbox_max_opt.value().get();
    const auto& pixels_flat  = vref(pixels_opt);
    const auto& pixel_offsets = vref(pixel_offs_opt);
    const auto& motion_var_v      = vref(motion_var_opt);
    const auto& motion_bias_v     = vref(motion_bias_opt);
    const auto& motion_dotd_v     = vref(motion_dotd_opt);
    const auto& trunc_frac_v      = vref(trunc_frac_opt);
    const auto& centroid_radius_v = vref(cent_rad_opt);
    const auto& range_v           = vref(range_opt);
    const auto& has_depth_v       = vref(has_depth_opt);
    const auto& azimuth_v         = vref(azimuth_opt);
    const auto& depth_var_v       = vref(depth_var_opt);
    const auto& color_rgb_v       = vref(color_rgb_opt);
    const auto& color_var_v       = vref(color_var_opt);
    const auto& color_neff_v      = vref(color_neff_opt);

    // Part B: with frame-transform enabled, source the camera-frame support array and transform it to
    // the target frame below; otherwise use the legacy room-frame array as-is.
    const VecOpt points_cam_opt = G_->get_attrib_by_name<mask_support_points_cam_att>(masks_node);
    const bool use_cam = transform_enabled_ and points_cam_opt and not points_cam_opt->get().empty();
    const auto& source_flat = use_cam ? points_cam_opt->get() : support_flat;

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
    packet.timestamp_ms = ts_opt ? ts_opt.value() : 0;   // 0 → consumer falls back to latest pose
    if (cam_twist_opt and cam_twist_opt->get().size() >= 6)
        for (int k = 0; k < 6; ++k) packet.cam_twist[k] = cam_twist_opt->get()[k];
    packet.frame_dt_s = frame_dt_opt ? frame_dt_opt.value() : 0.0f;

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

    // RAW camera-frame support points (untransformed), when the producer dual-published them — 1-to-1
    // with support_points' indexing. Pose-independent; consumed by object-anchor z_o. Loaded regardless
    // of use_cam (which only governs the room/target-frame support_points path above).
    if (points_cam_opt and points_cam_opt->get().size() == source_flat.size() and not use_cam)
    {
        const auto& cam_flat = points_cam_opt->get();
        packet.support_points_cam.reserve(support_count);
        for (std::size_t i = 0; i < support_count; ++i)
            packet.support_points_cam.emplace_back(cam_flat[i*3], cam_flat[i*3+1], cam_flat[i*3+2]);
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
        // Bearing-only (ricoh) channel: default has_depth=true when the producer predates the field.
        slice.has_depth        = (static_cast<std::size_t>(i) < has_depth_v.size())
                                 ? (has_depth_v[static_cast<std::size_t>(i)] != 0.0f) : true;
        slice.azimuth_room_rad = fetch1(azimuth_v);
        slice.depth_var        = fetch1(depth_var_v);
        // Appearance: 3 floats per slice for chroma/var, 1 for n_eff. An absent attr (older producer) or a
        // short array leaves all three at zero, i.e. "no colour information" — the same state a ricoh
        // bearing slice publishes explicitly, so consumers need only one code path.
        slice.color_neff = fetch1(color_neff_v);
        {
            const auto fetch3 = [&](const std::vector<float>& v) -> Eigen::Vector3f
            {
                const std::size_t base = static_cast<std::size_t>(i) * 3;
                return (base + 2 < v.size()) ? Eigen::Vector3f{v[base], v[base + 1], v[base + 2]}
                                             : Eigen::Vector3f::Zero();
            };
            slice.color_chroma = fetch3(color_rgb_v);
            slice.color_var    = fetch3(color_var_v);
        }
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
    // Stamp liveness ONLY on the fresh-frame success path (wall clock, never the source stamp) — a producer
    // republishing an old mask_frame_id must read as NOT live, which is exactly the stall we want to detect.
    last_fresh_wall_ms_ = QDateTime::currentMSecsSinceEpoch();
    return true;
}

// ─── Producer-liveness probes ──────────────────────────────────────────────────────────────────────

std::int64_t MaskIngestor::ms_since_last_frame() const noexcept
{
    if (last_fresh_wall_ms_ == 0) return -1;   // nothing ever ingested
    return QDateTime::currentMSecsSinceEpoch() - last_fresh_wall_ms_;
}

bool MaskIngestor::stream_ready(std::string* detail) const
{
    if (not G_) { if (detail) *detail = "no DSR graph"; return false; }
    const auto n = G_->get_node("masks");
    if (not n.has_value()) { if (detail) *detail = "no 'masks' node (voxelizer not up?)"; return false; }
    if (not G_->get_attrib_by_name<mask_frame_id_att>(n.value()).has_value())
    { if (detail) *detail = "'masks' node has no mask_frame_id"; return false; }
    if (detail) *detail = "masks";
    return true;
}

// ─── Queries ─────────────────────────────────────────────────────────────────────────────────────

// Nearest depth-bearing slice of `label` to `query_centroid` (room frame); nullopt if none.
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
        if (not slice.has_depth)   // bearing-only (ricoh) slices have no 3D centroid to match against
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

// Read a flat float3 point attribute (candidate_pts_att / residual_pts_att) off a node into Vector3f's.
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
