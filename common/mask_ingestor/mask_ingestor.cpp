/*
 * common/mask_ingestor/mask_ingestor.cpp  —  shared YOLO "masks" DSR node reader (implementation)
 *
 * SHARED across the concept agents: parses the retina-written PER-CAMERA mask nodes (discovered through the
 * "rgbd" camera nodes, see mask_nodes.h) into ONE merged MasksPacket (slices +
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
#include "../room_resolve/room_resolve.h"   // rc::room::resolve_frame — "room" means the CURRENT room (room_1, room_2, …)


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
}

// Override the default transform instance / target frame. `source_frame` is now ONLY a diagnostic default:
// each mask node is transformed from ITS OWN camera frame (the parent of the mask node).
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
std::optional<Eigen::Matrix4d> MaskIngestor::resolve_transform(const std::string& src_frame, std::uint64_t stamp) const
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
    // ★THE TARGET FRAME IS RESOLVED HERE, NOT LATCHED. `tgt_frame_` is what the agent CONFIGURED, and
    // the token "room" in a config is the fleet's word for "whichever room is current" — room_concept
    // names its rooms `room_1`, `room_2`, … so the literal is not a node name any more. Resolving it per
    // frame is what keeps this working across a room hand-over; a name taken once at enable_frame_transform
    // time would be a dead frame the moment the room changed, and cortex answers a dead frame with nullopt.
    const std::string tgt_frame = (G_ != nullptr) ? rc::room::resolve_frame(*G_, tgt_frame_) : tgt_frame_;

    if (G_ != nullptr)
        if (not G_->get_id_from_name(tgt_frame).has_value()
            or not G_->get_id_from_name(src_frame).has_value())
            return std::nullopt;

    // Direct lookup: correct whenever the chain can't be split (no robot node, src not under the robot),
    // and the only path when extrapolation is off. Still capture-stamp pinned.
    const auto direct = [&]() -> std::optional<Eigen::Matrix4d>
    {
        const auto T = inner_eigen_->get_transformation_matrix(tgt_frame, src_frame, stamp);
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
    if (robot_name_.empty() or robot_name_ == src_frame)
        return direct();

    // ── tgt←robot AT THE CAPTURE STAMP, EXTRAPOLATED BY THE QUERY ITSELF (2026-09-09) ─────────
    // TimeQuery::Extrapolated is Interpolated inside the RT ring and, past its ends, walks the pose
    // along the twist room_concept publishes in that same ring slot. This replaces
    // rc::media::extrapolate_room_T_robot, one of FOUR copies of that SE(2) step across the fleet.
    // ★TWO BEHAVIOUR CHANGES, BOTH DELIBERATE, SAY THEM RATHER THAN LET THEM BE FOUND:
    //   1. Inside the ring this was NEAREST (the old call passed no TimeQuery, and the default is
    //      Nearest); it is now interpolated. Strictly better — Nearest snaps to a block up to half a
    //      publish period away — but it is a change, and mask reprojection is what it moves.
    //   2. The horizon cap is enforced here rather than inside the helper. Cortex will not invent a
    //      cap; it reports how far it walked, and past the cap we take the un-extrapolated pose
    //      instead of predicting across a stalled producer.
    DSR::RT_API::TimeQueryInfo info;
    auto base = inner_eigen_->get_transformation_matrix(tgt_frame, robot_name_, stamp, "RT",
                                                        DSR::RT_API::TimeQuery::Extrapolated,
                                                        &info);
    if (not base.has_value())
        return direct();
    //   3. A CLAMPED query is now visible (info.clamped()). A mask baked against a pose from another
    //      instant lands off its object, which is the symptom this whole path exists to remove — so
    //      the condition is surfaced rather than absorbed. It is not treated as a failure: a clamp
    //      still returns the nearest measured pose, which beats dropping the frame.
    if (std::abs(info.applied_dt_ms) > static_cast<std::int64_t>(pose_extrap_max_dt_s_ * 1000.f))
    {
        base = inner_eigen_->get_transformation_matrix(tgt_frame, robot_name_, stamp, "RT",
                                                       DSR::RT_API::TimeQuery::Interpolated);
        if (not base.has_value())
            return direct();
    }
    const Mat::RTMat tgt_T_robot = base.value();

    // robot←src is the RIGID camera mount: it carries only its bootstrap timestamp, so a stamp-pinned
    // query FAILS on it — ask for "latest" (ts==0). Safe on the ts==0 cache because this instance is the
    // ingestor's own and every consumer refreshes on its main thread.
    const auto robot_T_src = inner_eigen_->get_transformation_matrix(robot_name_, src_frame, 0);
    if (not robot_T_src.has_value())
        return direct();

    return to_m4(tgt_T_robot * robot_T_src.value());
}

// ─── Ingest one camera's mask node ───────────────────────────────────────────────────────────────

bool MaskIngestor::camera_is_front(std::uint64_t camera_id) const
{
    return rc::masks::camera_is_front(*G_, camera_id);
}

// Parse ONE mask node (one camera) into `out`, with ITS camera as the source frame and ITS capture stamp.
// Support points are re-framed camera→target at that stamp; bearing-only azimuths (camera frame) are turned
// into ROOM-frame azimuths here, so everything downstream still reads slice.azimuth_room_rad in the room.
MaskIngestor::NodeResult MaskIngestor::ingest_node(const rc::masks::MaskNodeRef& ref, const DSR::Node& masks_node,
                                                   int frame_id, bool front, MasksPacket& packet)
{
    // TYPE-ATTRIBUTED reads (CLAUDE.md): compile-checked against dsr_attr_name.h. Held in named optionals so
    // the reference_wrapper payloads stay alive for the rest of the function (they reference masks_node, which
    // is in scope throughout). The vec() helper gives the previous "attr ? ->float_vec() : empty" semantics.
    static const std::vector<float> empty_flat;
    using VecOpt = std::optional<std::reference_wrapper<const std::vector<float>>>;
    const auto vref = [](const VecOpt& o) -> const std::vector<float>& { return o ? o->get() : empty_flat; };

    // REQUIRED core attributes — all must be present or this is not a valid mask packet.
    const auto count_opt     = G_->get_attrib_by_name<mask_count_att>(masks_node);
    const VecOpt label_ids_opt = G_->get_attrib_by_name<mask_label_ids_att>(masks_node);
    const VecOpt confs_opt     = G_->get_attrib_by_name<mask_confidences_att>(masks_node);
    const VecOpt offsets_opt   = G_->get_attrib_by_name<mask_support_offsets_att>(masks_node);
    const VecOpt centroids_opt = G_->get_attrib_by_name<mask_centroids_xyz_att>(masks_node);
    const VecOpt bbox_min_opt  = G_->get_attrib_by_name<mask_bbox_min_xyz_att>(masks_node);
    const VecOpt bbox_max_opt  = G_->get_attrib_by_name<mask_bbox_max_xyz_att>(masks_node);
    const auto labels_opt      = G_->get_attrib_by_name<mask_labels_att>(masks_node);   // ref_wrapper<const string>

    if (not count_opt or not labels_opt or not label_ids_opt or not confs_opt or not offsets_opt)
        return NodeResult::stale;

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
    const VecOpt points_cam_opt  = G_->get_attrib_by_name<mask_support_points_cam_att>(masks_node);

    const int mask_count = std::max(0, count_opt.value());
    const auto& labels = labels_opt.value().get();
    const auto& label_ids = label_ids_opt.value().get();
    const auto& confidences = confs_opt.value().get();
    const auto& offsets = offsets_opt.value().get();
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
    const auto& source_flat       = vref(points_cam_opt);   // CAMERA-frame points: the only 3-D array published

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

    packet.valid = true;
    packet.frame_id = frame_id;
    packet.timestamp_ms = ts_opt ? ts_opt.value() : 0;   // 0 → consumer falls back to latest pose
    if (cam_twist_opt and cam_twist_opt->get().size() >= 6)
        for (int k = 0; k < 6; ++k) packet.cam_twist[k] = cam_twist_opt->get()[k];
    packet.frame_dt_s = frame_dt_opt ? frame_dt_opt.value() : 0.0f;

    // src→tgt transform for THIS camera: one matrix for the node's frame, pinned to ITS capture stamp
    // (ts!=0 ⇒ correct pinning AND no InnerEigenAPI cache). Needed only when the frame carries 3-D points or
    // a bearing-only slice (camera-frame azimuth to convert); an empty frame is a valid, transform-free
    // liveness tick — it must not depend on the robot being localized.
    //
    // ★NO IDENTITY FALLBACK. An unresolvable chain means the robot is not localized at this stamp, and the
    // honest answer is "no observation this cycle", not "an observation at the origin".
    bool any_bearing_only = false;
    for (int i = 0; i < mask_count; ++i)
        if (static_cast<std::size_t>(i) < has_depth_v.size() and has_depth_v[static_cast<std::size_t>(i)] == 0.0f)
            any_bearing_only = true;
    const bool needs_transform = support_count > 0 or any_bearing_only;

    Eigen::Matrix4d T_tgt_src = Eigen::Matrix4d::Identity();
    if (needs_transform)
    {
        std::optional<Eigen::Matrix4d> Topt;
        if (transform_enabled_)
            Topt = resolve_transform(ref.camera, packet.timestamp_ms);
        if (not Topt.has_value())
        {
            if (++dropped_no_transform_ % 100 == 1)
                std::println("[MaskIngestor] {}<-{} unresolvable at ts={} — frame {} of '{}' dropped ({} total)",
                             tgt_frame_, ref.camera, packet.timestamp_ms, frame_id, ref.name, dropped_no_transform_);
            return NodeResult::dropped;
        }
        T_tgt_src = Topt.value();
    }
    const Eigen::Matrix3d R_tgt_src = T_tgt_src.block<3, 3>(0, 0);

    packet.support_points.reserve(support_count);
    packet.support_points_cam.reserve(support_count);
    for (std::size_t i = 0; i < support_count; ++i)
    {
        const Eigen::Vector4d q = T_tgt_src * Eigen::Vector4d(source_flat[i*3], source_flat[i*3+1], source_flat[i*3+2], 1.0);
        packet.support_points.emplace_back(static_cast<float>(q.x()), static_cast<float>(q.y()), static_cast<float>(q.z()));
        // RAW camera-frame support points (untransformed), 1-to-1 with support_points' indexing.
        // Pose-independent; this is what the object-anchor z_o path reads (table/hood/refrigerator/cabinet
        // build inst.obs_robot from it via the static body←zed mount). ★For a MERGED packet the array holds
        // each node's points in ITS OWN camera frame; those consumers only read zed slices' ranges.
        packet.support_points_cam.emplace_back(source_flat[i*3], source_flat[i*3+1], source_flat[i*3+2]);
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
        slice.timestamp_ms    = packet.timestamp_ms;
        slice.motion_var      = fetch1(motion_var_v);
        slice.motion_bias     = fetch1(motion_bias_v);
        slice.motion_dotd     = fetch1(motion_dotd_v);
        slice.trunc_frac      = fetch1(trunc_frac_v);
        slice.centroid_radius = fetch1(centroid_radius_v);
        slice.range           = fetch1(range_v);
        // Bearing-only (ricoh) channel: default has_depth=true when the producer predates the field.
        slice.has_depth        = (static_cast<std::size_t>(i) < has_depth_v.size())
                                 ? (has_depth_v[static_cast<std::size_t>(i)] != 0.0f) : true;
        // The producer publishes the bearing in the CAMERA frame; express it in the target (room) frame
        // here so every consumer keeps reading a room-frame azimuth. Direction only: rotate the unit ray.
        {
            const float az_cam = fetch1(azimuth_v);
            if (needs_transform)
            {
                const Eigen::Vector3d ray = R_tgt_src * Eigen::Vector3d(std::cos(az_cam), std::sin(az_cam), 0.0);
                slice.azimuth_room_rad = static_cast<float>(std::atan2(ray.y(), ray.x()));
            }
            else
                slice.azimuth_room_rad = az_cam;
        }
        slice.depth_var        = fetch1(depth_var_v);
        // Sensor SOURCE — unambiguous, unlike has_depth (see MaskSlice::source). Absent ⇒ a producer that
        // predates the field: the node's camera kind says it (front pinhole ⇒ zed, panoramic ⇒ ricoh).
        slice.source = (static_cast<std::size_t>(i) < source_v.size())
                       ? (source_v[static_cast<std::size_t>(i)] != 0.0f ? MaskSource::ricoh : MaskSource::zed)
                       : (front ? MaskSource::zed : MaskSource::ricoh);
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
        if (clamped_end > clamped_begin)
        {
            // Recompute centroid + bbox in the TARGET frame from the transformed support points (the
            // published camera-frame centroids/bboxes are stale once we re-frame the points).
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
    return NodeResult::ok;
}

// Append `src` (one node's packet) to `dst`, rebasing the slices' index ranges into the shared arrays.
void MaskIngestor::merge_into(MasksPacket& dst, MasksPacket&& src)
{
    const std::size_t sb = dst.support_points.size();
    const std::size_t pb = dst.mask_pixels.size();
    for (auto& sl : src.slices)
    {
        sl.support_begin += sb;  sl.support_end += sb;
        sl.pixel_begin   += pb;  sl.pixel_end   += pb;
        dst.slices.push_back(std::move(sl));
    }
    dst.support_points.insert(dst.support_points.end(), src.support_points.begin(), src.support_points.end());
    dst.support_points_cam.insert(dst.support_points_cam.end(), src.support_points_cam.begin(), src.support_points_cam.end());
    dst.mask_pixels.insert(dst.mask_pixels.end(), src.mask_pixels.begin(), src.mask_pixels.end());
}

// Re-read every camera's mask node; returns true only when a NEW frame (higher mask_frame_id) was ingested.
//
// MERGE POLICY. The packet is emitted once per FRONT-camera frame (the cadence every consumer was built
// around) and carries, besides that frame, the peripheral camera's slices IF that camera published a new
// frame since the last packet — each peripheral frame is delivered exactly once (it used to ride along,
// re-read, on every front frame). A peripheral frame that arrives first WAITS for the next front frame; with
// no front camera in the graph at all it is emitted on its own. The header (frame_id / timestamp / twist) is
// the front frame's; every slice also carries its own camera's capture stamp (MaskSlice::timestamp_ms).
bool MaskIngestor::refresh()
{
    if (not G_)
    {
        masks_packet_ = {};
        return false;
    }
    const auto refs = rc::masks::find_mask_nodes(*G_);
    if (refs.empty())
    {
        masks_packet_ = {};
        return false;
    }

    struct Fresh { rc::masks::MaskNodeRef ref; DSR::Node node; int frame_id; bool front; };
    std::vector<Fresh> fresh;
    bool any_front = false, front_fresh = false;
    for (const auto& ref : refs)
    {
        auto node = G_->get_node(ref.id);
        if (not node.has_value())
            continue;
        const auto fid = G_->get_attrib_by_name<mask_frame_id_att>(node.value());
        if (not fid.has_value())
            continue;
        auto& st = node_state_[ref.id];
        if (st.front < 0)
            st.front = camera_is_front(ref.camera_id) ? 1 : 0;
        const bool front = st.front == 1;
        any_front = any_front or front;
        if (fid.value() == st.last_frame)
            continue;                          // same frame republished (producer frozen) — not a new ingest
        if (fid.value() < st.last_frame)
            // mask_frame_id is a per-PROCESS publish counter; a backward jump means the producer (retina)
            // RESTARTED and reset it. Adopt the new stream (fall through to ingest) — otherwise refresh() would
            // reject every frame until the fresh counter climbs past the stale pre-restart value, silently
            // starving ALL mask consumers after a producer restart.
            std::println("[MaskIngestor] '{}' mask_frame_id {} < last-seen {} — producer restarted; adopting new stream",
                         ref.name, fid.value(), st.last_frame);
        fresh.push_back({ref, std::move(node.value()), fid.value(), front});
        front_fresh = front_fresh or front;
    }
    if (fresh.empty())
        return false;
    if (any_front and not front_fresh)
        return false;                          // a peripheral frame waits for the next front frame
    std::stable_partition(fresh.begin(), fresh.end(), [](const Fresh& f) { return f.front; });

    MasksPacket merged;
    for (auto& f : fresh)
    {
        MasksPacket np;
        const auto r = ingest_node(f.ref, f.node, f.frame_id, f.front, np);
        if (r == NodeResult::stale)
            continue;
        node_state_[f.ref.id].last_frame = f.frame_id;   // consumed (ingested, or dropped for want of a transform)
        if (r == NodeResult::dropped)
        {
            if (f.front)
            {
                // The front frame could not be framed: no observation this cycle (peripheral frames, not yet
                // consumed above this point, stay pending for the next one).
                masks_packet_ = {};
                return false;
            }
            continue;
        }
        if (not merged.valid)
        {
            merged.valid        = true;
            merged.frame_id     = np.frame_id;
            merged.timestamp_ms = np.timestamp_ms;
            merged.cam_twist    = np.cam_twist;
            merged.frame_dt_s   = np.frame_dt_s;
        }
        merge_into(merged, std::move(np));
    }
    if (not merged.valid)
        return false;

    masks_packet_ = std::move(merged);
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
    const auto refs = rc::masks::find_mask_nodes(*G_);
    if (refs.empty()) { if (detail) *detail = "no mask node under any rgbd camera (retina not up?)"; return false; }
    if (detail)
    {
        detail->clear();
        for (const auto& r : refs)
            *detail += (detail->empty() ? "" : ",") + r.name;
    }
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
