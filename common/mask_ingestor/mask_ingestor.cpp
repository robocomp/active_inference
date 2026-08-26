/*
 * common/mask_ingestor/mask_ingestor.cpp  —  shared YOLO "masks" DSR node reader (implementation)
 *
 * SHARED across the concept agents: parses the retina-written "masks" node into a MasksPacket (slices +
 * support points + raw silhouette pixels + the per-mask corruption/range/bearing channels), re-frames the
 * support points src→target (pinned to the capture stamp, recomputing centroids/bboxes there), and serves
 * the nearest slice of a requested label. The producer publishes CAMERA-frame points; the room transform
 * lives here, so raw perception never depends on localization. See the FRAME CONTRACT in the header.
 */

#include <optional>
#include "mask_ingestor.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <print>
#include <sstream>
#include <utility>

#include <QDateTime>   // wall-clock ms source for the producer-liveness probes (kept out of the header)

#include "../media_transport/rt_extrapolate.h"   // efference-copy forward-extrapolation over the RT lag

namespace rc {

// ─── Construction / configuration ────────────────────────────────────────────────────────────────

MaskIngestor::MaskIngestor(std::shared_ptr<DSR::DSRGraph> graph)
    : G_(std::move(graph))
{
    // Own the transform by default (see FRAME CONTRACT): the producer publishes camera-frame points and
    // this class converts them to the room at the capture stamp. get_inner_eigen_api() hands back a FRESH
    // instance per call, so this one is never shared with the consumer's own — which is what keeps the
    // ts==0 cache per-instance. We only ever query it with ts!=0 anyway, so no cache is touched.
    if (G_)
        owned_inner_eigen_ = G_->get_inner_eigen_api();
    inner_eigen_       = owned_inner_eigen_.get();
    transform_enabled_ = (inner_eigen_ != nullptr);
    // A/B escape hatch: force the legacy room-frame array so the two paths can be diffed on the same
    // producer. They must agree to float precision — same matrix, same timestamp, one process later.
    if (const char* v = std::getenv("MASK_INGESTOR_LEGACY_ROOM"); v != nullptr and v[0] == '1')
    {
        force_legacy_room_ = true;
        std::println("[MaskIngestor] MASK_INGESTOR_LEGACY_ROOM=1 — reading producer ROOM-frame points as-is");
    }
}

// Override the default zed→room transform. A null inner_eigen or an empty frame name reverts to the
// legacy path (read the producer's ROOM-frame array verbatim).
void MaskIngestor::enable_frame_transform(DSR::InnerEigenAPI* inner_eigen,
                                          std::string source_frame, std::string target_frame)
{
    inner_eigen_       = inner_eigen;
    src_frame_         = std::move(source_frame);
    tgt_frame_         = std::move(target_frame);
    transform_enabled_ = (inner_eigen_ != nullptr) and not src_frame_.empty() and not tgt_frame_.empty();
}

void MaskIngestor::set_frames(std::string source_frame, std::string target_frame)
{
    src_frame_         = std::move(source_frame);
    tgt_frame_         = std::move(target_frame);
    transform_enabled_ = (inner_eigen_ != nullptr) and not src_frame_.empty() and not tgt_frame_.empty();
}

void MaskIngestor::set_pose_extrapolation(bool enabled, float max_dt_s)
{
    pose_extrapolate_     = enabled;
    pose_extrap_max_dt_s_ = max_dt_s;
}

// ─── Transform resolution ─────────────────────────────────────────────────────────────────────────

// ★THE RT LAG IS WHY THIS IS NOT ONE get_transformation_matrix CALL.
// DSR's InterpolatedRT CLAMPS a query at the newest RT block — it never extrapolates velocity — so a mask
// whose capture stamp is AHEAD of room_concept's latest published pose (~90 ms of localization pipeline
// lag, measured) resolves against a STALE robot pose. The retina used to correct for this internally
// (room_T_zed_extrapolated + forward_extrapolate_room_T_robot) before baking the transform into the
// points; moving the transform here without that correction would have silently reintroduced the lag
// exactly when it matters — while the robot is MOVING — and the regression would have looked like a
// tracker/association problem, not a frame problem. So the chain is split and the same efference-copy
// prediction is applied here, reusing the shared helper the LiDAR consumers already use.
//   tgt←src  =  extrapolate(tgt←robot @ stamp)  ·  robot←src @ 0 (static camera mount, never changes)
std::optional<Eigen::Matrix4d> MaskIngestor::resolve_transform(std::uint64_t stamp) const
{
    if (inner_eigen_ == nullptr)
        return std::nullopt;

    const auto to_m4 = [](const Mat::RTMat& t)
    {
        Eigen::Matrix4d m = Eigen::Matrix4d::Identity();
        const auto& s = t.matrix();          // element-wise: dodges the Eigen-alignment ABI trap
        for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) m(r, c) = s(r, c);
        return m;
    };

    // ★ASK THE GRAPH BEFORE ASKING CORTEX. get_transformation_matrix answers a missing endpoint with a
    // qWarning — `origen or dest nodes do not exist` — and NOTHING throttles it, so a frame that has left
    // the graph (a room_concept restart) turns into one line PER INGESTED MASK FRAME for the life of the
    // run. The caller already reports an unresolvable chain properly: 1 line in 100, naming both frames
    // and the running total. Bailing here routes the condition into that counter instead of into the log.
    // Costs two `name_map` lookups under a shared_lock; get_node would have copied both whole nodes.
    if (G_ != nullptr)
        if (not G_->get_id_from_name(tgt_frame_).has_value()
            or not G_->get_id_from_name(src_frame_).has_value())
            return std::nullopt;

    // Direct lookup: correct whenever the chain can't be split (no robot node, src not under the robot),
    // and the only path when extrapolation is off. Still capture-stamp pinned.
    const auto direct = [&]() -> std::optional<Eigen::Matrix4d>
    {
        const auto T = inner_eigen_->get_transformation_matrix(tgt_frame_, src_frame_, stamp);
        return T.has_value() ? std::optional{to_m4(T.value())} : std::nullopt;
    };

    if (not pose_extrapolate_ or stamp == 0 or not G_)
        return direct();

    // ★THE MEMO IS DROPPED WHEN THE NODE BEHIND IT LEAVES THE GRAPH. It was write-once, so after the
    // robot node went (or was renamed — P3Bot vs Shadow) this handed back a dead name for ever and the
    // three inner_eigen calls below each answered with cortex's warning, per mask frame. Re-discovery is
    // the same `get_nodes_by_type` call and it is what lets a re-created robot be picked up without
    // restarting the agent. Clearing it is also SAFE rather than merely quiet: with no robot the chain
    // simply cannot be split, and `direct()` below is the correct answer, not a degraded one.
    if (not robot_name_.empty() and not G_->get_id_from_name(robot_name_).has_value())
        robot_name_.clear();
    if (robot_name_.empty())
        if (const auto robots = G_->get_nodes_by_type("robot"); not robots.empty())
            robot_name_ = robots.front().name();
    if (robot_name_.empty() or robot_name_ == src_frame_)
        return direct();

    // tgt←robot at the capture stamp (CLAMPED by DSR when the mask outruns the RT), then predicted forward.
    const auto base = inner_eigen_->get_transformation_matrix(tgt_frame_, robot_name_, stamp);
    if (not base.has_value())
        return direct();
    Mat::RTMat tgt_T_robot = base.value();
    rc::media::extrapolate_room_T_robot(G_, tgt_frame_, robot_name_, stamp, pose_extrap_max_dt_s_,
                                        base.value(), tgt_T_robot);

    // robot←src is the RIGID camera mount: it carries only its bootstrap timestamp, so a stamp-pinned
    // query FAILS on it — ask for "latest" (ts==0). Safe on the ts==0 cache because this instance is the
    // ingestor's own and every consumer refreshes on its main thread.
    const auto robot_T_src = inner_eigen_->get_transformation_matrix(robot_name_, src_frame_, 0);
    if (not robot_T_src.has_value())
        return direct();

    return to_m4(tgt_T_robot * robot_T_src.value());
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

    // points_opt / centroids / bboxes are NO LONGER REQUIRED. The producer publishes camera-frame points
    // only, and this class recomputes centroids/bboxes in the target frame after transforming; a producer
    // that still emits the legacy room-frame arrays simply has them available as a fallback below.
    if (not frame_opt or not count_opt or not labels_opt or not label_ids_opt or not confs_opt
        or not offsets_opt)
    {
        masks_packet_ = {};
        return false;
    }

    const int frame_id = frame_opt.value();
    if (frame_id == last_masks_frame_seen_)
        return false;                          // same frame republished (producer frozen) — not a new ingest
    if (frame_id < last_masks_frame_seen_)
        // mask_frame_id is a per-PROCESS publish counter; a backward jump means the producer (retina)
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
    const VecOpt source_opt      = G_->get_attrib_by_name<mask_source_att>(masks_node);
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
    // vref(), not .value() — these four are optional now (a current producer publishes none of them), and
    // .value() on an absent attribute is a bad_optional_access throw, not a graceful miss.
    const auto& support_flat   = vref(points_opt);
    const auto& centroids_flat = vref(centroids_opt);
    const auto& bbox_min_flat  = vref(bbox_min_opt);
    const auto& bbox_max_flat  = vref(bbox_max_opt);
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
    const auto& source_v          = vref(source_opt);
    const auto& color_rgb_v       = vref(color_rgb_opt);
    const auto& color_var_v       = vref(color_var_opt);
    const auto& color_neff_v      = vref(color_neff_opt);

    // Frame selection. The camera array is the contract (see header); the room array is only a fallback
    // for an OLD producer that still publishes it, or the explicit A/B escape hatch. Note the camera path
    // is chosen on the ARRAY being present, not on which agent this is — so a producer that drops the room
    // array needs no consumer change, and one that predates the camera array keeps working.
    const VecOpt points_cam_opt = G_->get_attrib_by_name<mask_support_points_cam_att>(masks_node);
    const bool have_cam = points_cam_opt and not points_cam_opt->get().empty();
    const bool use_cam  = transform_enabled_ and have_cam and not force_legacy_room_;
    if (not use_cam and support_flat.empty() and have_cam)
    {
        // Camera points are all we have and we cannot re-frame them: the room array is empty and the
        // transform is off/unavailable. Handing back camera coordinates that every consumer will treat as
        // room coordinates is the corruption this class exists to prevent — refuse the frame instead.
        ++dropped_no_transform_;
        masks_packet_ = {};
        return false;
    }
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

    // src→tgt transform: one matrix for the whole frame, pinned to the CAPTURE stamp (ts!=0 ⇒ correct
    // pinning AND no InnerEigenAPI cache). Built element-wise to dodge the Eigen-alignment ABI trap.
    //
    // ★NO IDENTITY FALLBACK. The old code fell back to identity when the chain was not resolvable, which
    // silently handed camera-frame points to a consumer that would fit them as room coordinates — the same
    // defect the producer had. An unresolvable chain means the robot is not localized at this stamp, and
    // the honest answer is "no observation this cycle", not "an observation at the origin".
    Eigen::Matrix4d T_tgt_src = Eigen::Matrix4d::Identity();
    if (use_cam)
    {
        const auto Topt = resolve_transform(packet.timestamp_ms);
        if (not Topt.has_value())
        {
            if (++dropped_no_transform_ % 100 == 1)
                std::println("[MaskIngestor] {}<-{} unresolvable at ts={} — frame {} dropped ({} total)",
                             tgt_frame_, src_frame_, packet.timestamp_ms, frame_id, dropped_no_transform_);
            masks_packet_ = {};
            last_masks_frame_seen_ = frame_id;   // consumed: do not re-attempt this same frame every cycle
            return false;
        }
        T_tgt_src = Topt.value();
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

    // RAW camera-frame support points (untransformed), 1-to-1 with support_points' indexing.
    // Pose-independent; this is what the object-anchor z_o path reads (table/hood/refrigerator/cabinet
    // build inst.obs_robot from it via the static body←zed mount).
    //
    // ★POPULATED ON BOTH PATHS. It used to be filled only when use_cam==false, i.e. only for consumers
    // that did NOT re-frame — so making the camera path the default would have silently emptied the array
    // and killed every object anchor with no compile error and no log. On the camera path it is simply the
    // untransformed source, so this is a copy, not a second computation.
    if (have_cam and points_cam_opt->get().size() == source_flat.size())
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
        // Sensor SOURCE — unambiguous, unlike has_depth (see MaskSlice::source). Absent ⇒ zed, which is
        // what a producer predating the field could only have been publishing.
        slice.source           = (static_cast<std::size_t>(i) < source_v.size()
                                  and source_v[static_cast<std::size_t>(i)] != 0.0f)
                                 ? MaskSource::ricoh : MaskSource::zed;
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
    if (not n.has_value()) { if (detail) *detail = "no 'masks' node (retina not up?)"; return false; }
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
