/*
 * graph_publisher.cpp — DSR semantic_grid exports (masks).
 */

// These pull in oneTBB (tbb/profiling.h). Include them BEFORE any header that activates Qt's `emit`
// keyword macro (dsr_api.h via graph_publisher.h, and <QtGlobal>), otherwise the empty `emit` macro
// mangles TBB's profiling.h and it fails to compile.
#include "yolo_processor.h"   // SegDetection
#include "yolo_human.h"       // rc::human_pose::PoseDetection, BODY18_FROM_COCO

#include "graph_publisher.h"

#include "../../common/motion_corruption/motion_corruption.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <print>
#include <sstream>
#include <unordered_map>
#include <utility>

#include <QtGlobal>

GraphPublisher::GraphPublisher(std::shared_ptr<DSR::DSRGraph> graph,
                               const VoxelizerParams& params,
                               std::function<void()> relayout)
    : G_(std::move(graph)), params_(params), relayout_(std::move(relayout))
{}

void GraphPublisher::publish(const RGBDData& rgbd, const Mat::RTMat& room_T_zed,
                             const std::vector<SegDetection>& detections, std::uint64_t frame_ts_ms,
                             const std::vector<BearingDetection>& bearing_detections)
{
    if (ensure_node("masks", "Plum", masks_ready_, /*relayout=*/true))
        upload_masks(rgbd, room_T_zed, detections, frame_ts_ms, bearing_detections);
}

bool GraphPublisher::ensure_node(const char* name, const char* color, bool& ready, bool relayout)
{
    if (ready)
        return true;
    if (G_->get_nodes_by_type("room").empty() or G_->get_nodes_by_type("robot").empty())
        return false;
    if (G_->get_node(name).has_value())
    {
        ready = true;
        return true;
    }

    auto zed_node = G_->get_node("zed");
    if (!zed_node.has_value())
    {
        qWarning() << "[GraphPublisher] WARNING: 'zed' node not found — cannot create" << name;
        return false;
    }

    auto node = DSR::Node::create<semantic_grid_node_type>(name);
    G_->add_or_modify_attrib_local<color_att>(node, std::string{color});
    G_->add_or_modify_attrib_local<level_att>(node, 4);
    G_->add_or_modify_attrib_local<parent_att>(node, zed_node.value().id());
    G_->add_or_modify_attrib_local<pos_x_att>(node, 105.849792f);
    G_->add_or_modify_attrib_local<pos_y_att>(node, 291.904266f);

    const auto id = G_->insert_node(node);
    if (!id.has_value())
    {
        qWarning() << "[GraphPublisher] ERROR: failed to insert" << name << "node";
        return false;
    }

    ready = true;
    qInfo() << "[GraphPublisher] Created" << name << "node id=" << *id << "(semantic_grid) under 'zed'";
    G_->get_rt_api()->insert_or_assign_edge_RT(zed_node.value(), *id, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f});
    if (relayout and relayout_)
        relayout_();
    return true;
}

// FRAME CONTRACT — the 'masks' node is a SHARED contract read by every concept agent through ONE
// component, common/mask_ingestor. Mask support points are DUAL-PUBLISHED, 1-to-1:
//   * mask_support_points      → ROOM frame  (consumed by table_concept, chair_concept — default path)
//   * mask_support_points_cam  → ZED/camera frame (consumed by bottle_concept via MaskIngestor's
//                                enable_frame_transform: zed→target, capture-stamp pinned + chain cov)
// Centroids/bboxes are published ROOM-frame only; the ingestor RECOMPUTES them in the target frame
// when a consumer opts into the camera path, so no *_cam centroid/bbox is needed.
// DO NOT drop either array: removing _cam breaks bottle_concept; removing the room arrays breaks
// table/chair. Frame choice is per-consumer at the ingestor, never a producer-side switch.
//
// EGO-MOTION CORRUPTION ANNOTATION (MaskMotion.enabled, default on; common/motion_corruption):
// while the camera moves, a mask is corrupted in a way the consumer can't reconstruct without the
// capture-time twist/exposure/timing. The producer attaches the DECOMPOSITION; the consumer decides
// gate-vs-downweight (per-concept b_max), per-DOF split, and folds the variance into its innovation cov.
//   per-mask (1-to-1 with mask_label_ids):
//     mask_motion_dotd     — metric position-corruption speed Z·‖ṡ‖ (m/s), diagnostic
//     mask_motion_bias     — systematic displacement from a KNOWN timing offset (m) — GATE on this
//     mask_motion_var      — variance to ADD to R (m²): exposure blur + timing jitter, ×peripheral
//     mask_trunc_frac      — fraction of silhouette pixels on the image border (truncation → bias)
//     mask_centroid_radius — normalized centroid radius from the principal point (periphery penalty)
//   per-frame (global): mask_cam_twist [vx,vy,vz,wx,wy,wz] (OPTICAL frame) + mask_frame_dt_s.
void GraphPublisher::upload_masks(const RGBDData& rgbd, const Mat::RTMat& room_T_zed,
                                  const std::vector<SegDetection>& detections, std::uint64_t frame_ts_ms,
                                  const std::vector<BearingDetection>& bearing_detections)
{
    // Keep masks stream progressing even when voxel grid processing is paused.
    const std::uint64_t sensing_frame = ++masks_publish_seq_;
    if (sensing_frame == last_masks_uploaded_frame_)
        return;
    last_masks_uploaded_frame_ = sensing_frame;

    auto masks_node_opt = G_->get_node("masks");
    if (!masks_node_opt.has_value())
        return;
    auto& masks_node = masks_node_opt.value();

    if (rgbd.depth.empty() || rgbd.width <= 0 || rgbd.height <= 0 || rgbd.focal_x <= 0.f || rgbd.focal_y <= 0.f)
    {
        G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_frame_id", static_cast<int>(sensing_frame));
        G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_timestamp_ms", frame_ts_ms);
        G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_count", 0);
        G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_labels", std::string{});
        G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_label_ids", std::vector<float>{});
        G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_confidences", std::vector<float>{});
        G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_support_offsets", std::vector<float>{0.0f});
        G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_support_points", std::vector<float>{});
        G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_support_points_cam", std::vector<float>{});
        G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_centroids_xyz", std::vector<float>{});
        G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_bbox_min_xyz", std::vector<float>{});
        G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_bbox_max_xyz", std::vector<float>{});
        G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_pixels_xy", std::vector<float>{});
        G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_pixel_offsets", std::vector<float>{0.0f});
        G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_depth_var", std::vector<float>{});
        if (params_.MASK_MOTION_ENABLED)
        {
            G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_motion_dotd", std::vector<float>{});
            G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_motion_bias", std::vector<float>{});
            G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_motion_var", std::vector<float>{});
            G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_trunc_frac", std::vector<float>{});
            G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_centroid_radius", std::vector<float>{});
            G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_range", std::vector<float>{});
            G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_cam_twist", std::vector<float>{0,0,0,0,0,0});
            G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_frame_dt_s", 0.0f);
            G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_rt_lag_s", -1.0f);
            G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_rt_gap_s", -1.0f);
        }
        G_->update_node(std::move(masks_node));
        return;
    }

    const Eigen::Matrix3f room_rotation = room_T_zed.linear().cast<float>();
    const Eigen::Vector3f room_translation = room_T_zed.translation().cast<float>();
    const float z_lift_m = params_.VOXEL_Z_LIFT_M;
    const float fx = rgbd.focal_x;
    const float fy = rgbd.focal_y;
    const float cx = static_cast<float>(rgbd.width) * 0.5f;
    const float cy = static_cast<float>(rgbd.height) * 0.5f;

    // Ego-motion camera twist (optical frame) for the per-mask corruption annotation. Finite-difference
    // the zed pose against the previous published frame, pinned to capture stamps; zero on the first
    // frame or when no stamp is available (corruption then evaluates to 0 → masks pass through unweighted).
    rc::motion::Twist cam_twist_optical;
    float frame_dt_s = 0.0f;
    if (params_.MASK_MOTION_ENABLED and have_last_zed_ and frame_ts_ms > 0 and last_zed_ts_ms_ > 0
        and frame_ts_ms > last_zed_ts_ms_)
    {
        frame_dt_s = static_cast<float>(frame_ts_ms - last_zed_ts_ms_) * 1e-3f;
        const auto tw = rc::motion::body_twist_from_poses(last_zed_R_, last_zed_t_,
                                                          room_rotation, room_translation, frame_dt_s);
        cam_twist_optical = rc::motion::vcam_to_optical(tw);
    }
    const rc::motion::CorruptionParams corruption_params{
        .exposure_s      = params_.MASK_MOTION_EXPOSURE_S,
        .timing_jitter_s = params_.MASK_MOTION_TIMING_JITTER_S,
        .timing_offset_s = params_.MASK_MOTION_TIMING_OFFSET_S};

    // Lazily open the diagnostic CSV (once). |v|,|w| are the frame twist magnitudes; per-mask columns
    // let you confirm dot_d≈0 when static and that it spikes under panning (rotation dominates).
    if (params_.MASK_MOTION_ENABLED and params_.MASK_MOTION_CSV_LOG and not motion_csv_open_attempted_)
    {
        motion_csv_open_attempted_ = true;
        motion_csv_.open("etc/mask_motion_log.csv", std::ios::out | std::ios::trunc);
        if (motion_csv_.is_open())
            motion_csv_ << "frame,ts_ms,dt_s,rt_lag_s,rt_gap_s,v_norm,w_norm,label,Z,xn,yn,radius,trunc,dotd,bias,var\n";
    }
    const float v_norm = cam_twist_optical.v.norm();
    const float w_norm = cam_twist_optical.w.norm();

    // Localization timing lag: capture stamp minus the NEWEST room→robot RT block stamp. This is the δt
    // the no-extrapolation clamp leaves when the robot-pose stream lags the camera — i.e. the residual
    // that drives the corruption BIAS. Measuring it here sizes MaskMotion.timing_offset_s and quantifies
    // the effect of raising room_concept's RT publish rate. -1 = couldn't read (edge/attr/stamp missing).
    float rt_lag_s = -1.0f;
    float rt_gap_s = -1.0f;   // spacing between the two newest RT blocks in this replica (block density)
    if (params_.MASK_MOTION_ENABLED and frame_ts_ms > 0)
    {
        const auto room_nodes  = G_->get_nodes_by_type("room");
        const auto robot_nodes = G_->get_nodes_by_type("robot");
        if (not room_nodes.empty() and not robot_nodes.empty())
        {
            // The dynamic localization RT edge is robot→room (robot-rooted bootstrap; room_concept
            // dsr_update_pose writes parent=robot, child=room). Try that first, then the reverse.
            auto edge = G_->get_rt_api()->get_edge_RT(robot_nodes.front(), room_nodes.front().id());
            if (not edge.has_value())
                edge = G_->get_rt_api()->get_edge_RT(room_nodes.front(), robot_nodes.front().id());
            if (edge.has_value())
            {
                if (auto ts = G_->get_attrib_by_name<rt_timestamps_att>(edge.value()); ts.has_value())
                {
                    // Two newest stamps in the consumer's ring: newest → rt_lag; (newest - second) →
                    // rt_gap, the consumer-side block spacing. rt_gap ~16ms = the 60Hz blocks actually
                    // reach this replica (dense ring → good interpolation); ~200ms = only ~5Hz arrive.
                    std::uint64_t newest = 0, second = 0;
                    for (const auto t : ts->get())   // ring buffer; 0 = empty slot
                    {
                        if (t > newest) { second = newest; newest = t; }
                        else if (t > second and t < newest) { second = t; }
                    }
                    if (newest > 0)
                        rt_lag_s = static_cast<float>(static_cast<std::int64_t>(frame_ts_ms)
                                                      - static_cast<std::int64_t>(newest)) * 1e-3f;
                    if (newest > 0 and second > 0)
                        rt_gap_s = static_cast<float>(newest - second) * 1e-3f;
                }
            }
        }
    }

    std::vector<float> label_ids;
    std::vector<float> confidences;
    std::vector<float> support_offsets;
    std::vector<float> support_points;
    // Dual-publish: the SAME support points in CAMERA (zed) frame (raw deprojected, no room transform),
    // 1-to-1 with support_points/support_offsets. Lets a consumer transform through the probabilistic
    // chain (zed→…→target) itself — pinned to the capture stamp + carrying localization covariance —
    // instead of inheriting the room transform baked here. Room-frame array kept for legacy consumers.
    std::vector<float> support_points_cam;
    std::vector<float> centroids_xyz;
    std::vector<float> bbox_min_xyz;
    std::vector<float> bbox_max_xyz;
    // Raw 2D mask silhouette (foreground pixels, BEFORE the depth gate) so downstream agents have
    // the RGB mask as an evidence channel independent of depth — flat (col,row) pairs per mask,
    // delimited by mask_pixel_offsets exactly like support_points/support_offsets.
    std::vector<float> mask_pixels_xy;
    std::vector<float> mask_pixel_offsets;
    // Per-mask ego-motion corruption decomposition (1-to-1 with label_ids). See common/motion_corruption.
    std::vector<float> motion_dotd;       // metric position-corruption speed (m/s)
    std::vector<float> motion_bias;       // systematic timing-offset displacement (m) — consumer gates on this
    std::vector<float> motion_var;        // variance to ADD to R (m²) — blur + jitter, ×peripheral
    std::vector<float> trunc_frac;        // fraction of silhouette pixels on the image border (truncation)
    std::vector<float> centroid_radius;   // normalized centroid radius from principal point (periphery)
    std::vector<float> mask_range;         // mean camera→mask depth Z (m): static range — consumer scales R + pose common-mode (a far view can't resolve orientation)
    std::vector<float> has_depth_flags;    // 1.0 = 3D slice (zed, or ricoh with lidar depth); 0.0 = ricoh bearing-only
    std::vector<float> mask_source;        // sensor SOURCE per mask: 0.0 = zed (front RGB-D), 1.0 = ricoh (360). Unlike
                                           //   mask_has_depth this is unambiguous (a lidar-depth ricoh mask is still 1.0).
    std::vector<float> azimuths;           // room-frame bearing (rad); meaningful only for has_depth==0 slices
    std::vector<float> depth_var;          // σ_range² (m²) to ADD to R along the mask ray (common/depth_projection).
                                           // 0 for dense-depth zed masks; the scored range_var for lidar-depth ricoh masks.
    std::ostringstream labels_joined;
    std::size_t total_support_points = 0;
    std::size_t fg_gate_dropped = 0, fg_gate_in = 0;   // foreground depth-gate telemetry (this frame)
    int         fg_gate_masks   = 0;
    support_offsets.push_back(0.0f);
    mask_pixel_offsets.push_back(0.0f);

    const std::size_t mask_stride = std::max<std::size_t>(1, params_.VOXEL_DECIMATION_FACTOR);

    for (std::size_t det_idx = 0; det_idx < detections.size(); ++det_idx)
    {
        const auto& det = detections[det_idx];
        if (det.mask.empty())
            continue;

        cv::Mat mask_bin;
        cv::threshold(det.mask, mask_bin, 127, 255, cv::THRESH_BINARY);
        const int min_x = std::max(0, det.bbox.x);
        const int min_y = std::max(0, det.bbox.y);
        const int max_x = std::min(rgbd.width, det.bbox.x + det.bbox.width);
        const int max_y = std::min(rgbd.height, det.bbox.y + det.bbox.height);

        // Pass 1: gather ALL masked room points with valid depth. No depth/range gate — every
        // masked pixel that deprojects is kept (radius outlier removal below still trims the
        // sparse silhouette-edge tail).
        std::vector<Eigen::Vector3f> gated;
        std::vector<Eigen::Vector3f> gated_cam;   // same points in camera (zed) frame, parallel to gated
        gated.reserve(static_cast<std::size_t>(det.bbox.area() / std::max<int>(1, static_cast<int>(mask_stride * mask_stride))));
        gated_cam.reserve(gated.capacity());
        std::vector<float> det_pixels;   // raw 2D foreground (col,row), depth-independent

        for (int row = min_y; row < max_y; row += static_cast<int>(mask_stride))
        {
            for (int col = min_x; col < max_x; col += static_cast<int>(mask_stride))
            {
                if (mask_bin.at<std::uint8_t>(row, col) == 0)
                    continue;

                det_pixels.push_back(static_cast<float>(col));
                det_pixels.push_back(static_cast<float>(row));

                const std::size_t depth_idx = static_cast<std::size_t>(row * rgbd.width + col);
                if (depth_idx >= rgbd.depth.size())
                    continue;

                const float depth = rgbd.depth[depth_idx];
                if (!std::isfinite(depth) || depth <= 0.0f)
                    continue;

                const float px = (static_cast<float>(col) - cx) * depth / fx;
                const float py = depth;
                const float pz = (cy - static_cast<float>(row)) * depth / fy;
                const Eigen::Vector3f point_cam(px, py, pz);
                Eigen::Vector3f point_room = room_rotation * point_cam + room_translation;
                point_room.z() += z_lift_m;

                gated.push_back(point_room);
                gated_cam.push_back(point_cam);
            }
        }

        if (gated.empty())
            continue;

        // Radius outlier removal: keep points that have enough neighbours nearby. The dense
        // object body survives; the sparse silhouette-edge tail is trimmed. Bucketed into a
        // uniform spatial hash grid (cell = MASK_OUTLIER_RADIUS_M) instead of all-pairs distance
        // checks: an isolated silhouette-edge point — exactly the case this filter is meant to
        // discard — used to scan the ENTIRE point set before concluding it has too few
        // neighbours (worst-case O(n²) on the very points it removes); the grid only visits the
        // 3x3x3 neighbouring cells, so cost tracks local density instead of total point count.
        std::vector<Eigen::Vector3f> mask_points_room;
        std::vector<Eigen::Vector3f> mask_points_cam;   // parallel to mask_points_room (same survivors)
        if (params_.MASK_OUTLIER_MIN_NEIGHBORS > 0 and params_.MASK_OUTLIER_RADIUS_M > 0.0f
            and gated.size() > static_cast<std::size_t>(params_.MASK_OUTLIER_MIN_NEIGHBORS))
        {
            const float radius = params_.MASK_OUTLIER_RADIUS_M;
            const float r2 = radius * radius;
            const int min_neighbours = params_.MASK_OUTLIER_MIN_NEIGHBORS;

            const auto cell_of = [radius](const Eigen::Vector3f& p) -> std::array<int, 3>
            {
                return {static_cast<int>(std::floor(p.x() / radius)),
                        static_cast<int>(std::floor(p.y() / radius)),
                        static_cast<int>(std::floor(p.z() / radius))};
            };
            struct CellHash
            {
                std::size_t operator()(const std::array<int, 3>& k) const noexcept
                {
                    std::size_t h = static_cast<std::size_t>(k[0]);
                    h = h * 1000003u ^ static_cast<std::size_t>(k[1]);
                    h = h * 1000003u ^ static_cast<std::size_t>(k[2]);
                    return h;
                }
            };

            std::unordered_map<std::array<int, 3>, std::vector<std::size_t>, CellHash> grid;
            grid.reserve(gated.size());
            for (std::size_t i = 0; i < gated.size(); ++i)
                grid[cell_of(gated[i])].push_back(i);

            mask_points_room.reserve(gated.size());
            mask_points_cam.reserve(gated.size());
            for (std::size_t i = 0; i < gated.size(); ++i)
            {
                const auto base = cell_of(gated[i]);
                int neighbours = 0;
                for (int dx = -1; dx <= 1 and neighbours < min_neighbours; ++dx)
                    for (int dy = -1; dy <= 1 and neighbours < min_neighbours; ++dy)
                        for (int dz = -1; dz <= 1 and neighbours < min_neighbours; ++dz)
                        {
                            const auto it = grid.find({base[0] + dx, base[1] + dy, base[2] + dz});
                            if (it == grid.end())
                                continue;
                            for (const std::size_t j : it->second)
                            {
                                if (j == i or (gated[i] - gated[j]).squaredNorm() > r2)
                                    continue;
                                if (++neighbours >= min_neighbours)
                                    break;
                            }
                        }
                if (neighbours >= min_neighbours)
                {
                    mask_points_room.push_back(gated[i]);
                    mask_points_cam.push_back(gated_cam[i]);
                }
            }
        }
        else
        {
            mask_points_room = std::move(gated);
            mask_points_cam = std::move(gated_cam);
        }

        if (mask_points_room.empty())
            continue;

        // FOREGROUND depth gate — kills the WALL-BEHIND points that bleed into the slice when the robot
        // moves: the RGB mask and depth map skew, so silhouette-edge pixels sample the background depth and
        // deproject onto the far wall. That's a COHERENT cluster the radius filter above can't remove. Anchor
        // on the near surface (low percentile of camera-frame depth py; zed frame y = depth) and drop points
        // more than MASK_DEPTH_GATE_BAND_M behind it — the object occludes the background, so those far
        // returns can't be the object. Physical object-depth prior, not a tuning cutoff (no-threshold-patches).
        if (params_.MASK_DEPTH_GATE_BAND_M > 0.0f and mask_points_cam.size() >= 8)
        {
            std::vector<float> depths;
            depths.reserve(mask_points_cam.size());
            for (const auto& pc : mask_points_cam) depths.push_back(pc.y());   // zed camera y = depth along axis
            const std::size_t knear = depths.size() / 10;                       // ~10th-pct = near surface
            std::nth_element(depths.begin(), depths.begin() + knear, depths.end());
            const float d_max = depths[knear] + params_.MASK_DEPTH_GATE_BAND_M;
            std::vector<Eigen::Vector3f> fg_room, fg_cam;
            fg_room.reserve(mask_points_room.size());
            fg_cam.reserve(mask_points_cam.size());
            for (std::size_t i = 0; i < mask_points_room.size(); ++i)
                if (mask_points_cam[i].y() <= d_max)
                {
                    fg_room.push_back(mask_points_room[i]);
                    fg_cam.push_back(mask_points_cam[i]);
                }
            if (const std::size_t dropped = mask_points_room.size() - fg_room.size(); dropped > 0)
            {
                fg_gate_dropped += dropped;
                fg_gate_in      += mask_points_room.size();
                ++fg_gate_masks;
            }
            mask_points_room = std::move(fg_room);
            mask_points_cam  = std::move(fg_cam);
            if (mask_points_room.empty())
                continue;
        }

        Eigen::Vector3f min_pt = Eigen::Vector3f::Constant(std::numeric_limits<float>::max());
        Eigen::Vector3f max_pt = Eigen::Vector3f::Constant(std::numeric_limits<float>::lowest());
        Eigen::Vector3f sum_pt = Eigen::Vector3f::Zero();
        for (const auto& point_room : mask_points_room)
        {
            sum_pt += point_room;
            min_pt = min_pt.cwiseMin(point_room);
            max_pt = max_pt.cwiseMax(point_room);
        }

        if (!labels_joined.str().empty())
            labels_joined << '|';
        labels_joined << det.label;

        label_ids.push_back(static_cast<float>(det.class_id));
        confidences.push_back(det.confidence);
        has_depth_flags.push_back(1.0f);   // zed slice: carries 3D support points
        mask_source.push_back(0.0f);       // source = zed
        azimuths.push_back(0.0f);
        depth_var.push_back(0.0f);          // zed dense depth ⇒ no extra range variance beyond R_base
        support_offsets.push_back(static_cast<float>(support_offsets.back() + static_cast<float>(mask_points_room.size())));
        mask_pixels_xy.insert(mask_pixels_xy.end(), det_pixels.begin(), det_pixels.end());
        mask_pixel_offsets.push_back(static_cast<float>(mask_pixel_offsets.back() + static_cast<float>(det_pixels.size() / 2)));

        const Eigen::Vector3f centroid = sum_pt / static_cast<float>(mask_points_room.size());
        centroids_xyz.push_back(centroid.x());
        centroids_xyz.push_back(centroid.y());
        centroids_xyz.push_back(centroid.z());

        bbox_min_xyz.push_back(min_pt.x());
        bbox_min_xyz.push_back(min_pt.y());
        bbox_min_xyz.push_back(min_pt.z());
        bbox_max_xyz.push_back(max_pt.x());
        bbox_max_xyz.push_back(max_pt.y());
        bbox_max_xyz.push_back(max_pt.z());

        for (const auto& point : mask_points_room)
        {
            support_points.push_back(point.x());
            support_points.push_back(point.y());
            support_points.push_back(point.z());
        }
        for (const auto& point : mask_points_cam)
        {
            support_points_cam.push_back(point.x());
            support_points_cam.push_back(point.y());
            support_points_cam.push_back(point.z());
        }

        if (params_.MASK_MOTION_ENABLED)
        {
            // Pixel centroid + truncation from the raw 2D foreground (det_pixels = col,row, depth-free).
            float sum_col = 0.0f, sum_row = 0.0f;
            int on_border = 0;
            const std::size_t n_px = det_pixels.size() / 2;
            for (std::size_t i = 0; i + 1 < det_pixels.size(); i += 2)
            {
                const float c = det_pixels[i], r = det_pixels[i + 1];
                sum_col += c;
                sum_row += r;
                if (c <= 0.0f or r <= 0.0f
                    or c >= static_cast<float>(rgbd.width - 1) or r >= static_cast<float>(rgbd.height - 1))
                    ++on_border;
            }
            const float tau = (n_px > 0) ? static_cast<float>(on_border) / static_cast<float>(n_px) : 0.0f;
            // Mean optical-axis depth from the camera-frame survivors (.y() is depth in the zed frame).
            float sum_depth = 0.0f;
            for (const auto& p : mask_points_cam)
                sum_depth += p.y();
            const float Z = mask_points_cam.empty() ? 0.0f : sum_depth / static_cast<float>(mask_points_cam.size());
            const float col_c = (n_px > 0) ? sum_col / static_cast<float>(n_px) : cx;
            const float row_c = (n_px > 0) ? sum_row / static_cast<float>(n_px) : cy;
            const float xn = (col_c - cx) / fx;
            const float yn = (row_c - cy) / fy;
            const auto corr = rc::motion::mask_corruption(cam_twist_optical, xn, yn, Z, corruption_params);
            motion_dotd.push_back(corr.dot_d);
            motion_bias.push_back(corr.bias_m);
            motion_var.push_back(corr.var_m);
            trunc_frac.push_back(tau);
            const float radius = std::sqrt(xn * xn + yn * yn);
            centroid_radius.push_back(radius);
            mask_range.push_back(Z);   // static range: distance carries info loss even at zero motion

            if (motion_csv_.is_open())
                motion_csv_ << sensing_frame << ',' << frame_ts_ms << ',' << frame_dt_s << ','
                            << rt_lag_s << ',' << rt_gap_s << ',' << v_norm << ',' << w_norm << ','
                            << det.label << ',' << Z << ',' << xn << ',' << yn << ',' << radius << ','
                            << tau << ',' << corr.dot_d << ',' << corr.bias_m << ',' << corr.var_m << '\n';
        }

        total_support_points += mask_points_room.size();
    }

    // Foreground-gate telemetry (throttled ≈1s): far wall-behind points removed this frame + masks affected.
    // Only prints when the gate actually dropped something, so a clean scene stays quiet. Tune the band from
    // these numbers: steady large drops on a static scene ⇒ band too tight (clipping the object).
    {
        static std::size_t fg_call_no = 0, fg_last_print = 0;
        ++fg_call_no;
        if (fg_gate_dropped > 0 and (fg_call_no - fg_last_print) >= 20)
        {
            fg_last_print = fg_call_no;
            std::println("[mask-fg-gate] dropped {}/{} pts across {} zed mask(s) (band {:.2f}m, frame={})",
                         fg_gate_dropped, fg_gate_in, fg_gate_masks, params_.MASK_DEPTH_GATE_BAND_M, sensing_frame);
        }
    }

    // Append the RGB-360 (ricoh) bearing-only detections as NO-DEPTH slices in the SAME node (Part B, see
    // RICOH_360_PERIPHERAL_DETECTION.md). They carry a room-frame bearing instead of 3D points:
    // support_begin==support_end (no points), no raw pixels, NaN centroid/bbox. Consumers skip !has_depth
    // slices until the bearing confirm/birth path (Part C) exists — inert-but-available today.
    // (Only emitted alongside a valid zed depth frame; the empty-depth early-return above drops them for
    //  that rare/transient frame, which is fine for peripheral evidence.)
    // Zed→room inverse, to also express any lidar-depth ricoh support points in the ZED frame so they
    // flow through the consumer's existing zed→target chain (no ingestor frame change / no camera tag).
    const Eigen::Matrix3f zed_R = room_T_zed.linear().cast<float>();
    const Eigen::Vector3f zed_t = room_T_zed.translation().cast<float>();
    const Eigen::Matrix3f zed_R_inv = zed_R.transpose();               // room→zed rotation
    const auto room_to_zed = [&](const Eigen::Vector3f& p_room) { return zed_R_inv * (p_room - zed_t); };

    for (const auto& b : bearing_detections)
    {
        if (!labels_joined.str().empty())
            labels_joined << '|';
        labels_joined << b.label;
        label_ids.push_back(b.class_id);
        confidences.push_back(b.confidence);
        mask_source.push_back(1.0f);       // source = ricoh (both bearing-only and lidar-depth ricoh slices)
        azimuths.push_back(b.azimuth_room_rad);
        mask_pixel_offsets.push_back(mask_pixel_offsets.back());    // no raw silhouette pixels either way

        if (b.has_lidar_depth and not b.support_room.empty())
        {
            // FULL 3D slice from the reprojected lidar: room + zed support points (so both consumer
            // paths work), real centroid/bbox, and range_var as the depth uncertainty to add to R.
            has_depth_flags.push_back(1.0f);
            depth_var.push_back(b.range_var);
            support_offsets.push_back(static_cast<float>(support_offsets.back() + static_cast<float>(b.support_room.size())));
            Eigen::Vector3f mn = Eigen::Vector3f::Constant(std::numeric_limits<float>::max());
            Eigen::Vector3f mx = Eigen::Vector3f::Constant(std::numeric_limits<float>::lowest());
            for (const auto& pr : b.support_room)
            {
                support_points.push_back(pr.x());     support_points.push_back(pr.y());     support_points.push_back(pr.z());
                const Eigen::Vector3f pz = room_to_zed(pr);
                support_points_cam.push_back(pz.x()); support_points_cam.push_back(pz.y()); support_points_cam.push_back(pz.z());
                mn = mn.cwiseMin(pr); mx = mx.cwiseMax(pr);
            }
            centroids_xyz.insert(centroids_xyz.end(), {b.centroid_room.x(), b.centroid_room.y(), b.centroid_room.z()});
            bbox_min_xyz.insert(bbox_min_xyz.end(),   {mn.x(), mn.y(), mn.z()});
            bbox_max_xyz.insert(bbox_max_xyz.end(),   {mx.x(), mx.y(), mx.z()});
            if (params_.MASK_MOTION_ENABLED)
            {
                // No zed-image interaction matrix for a ricoh mask ⇒ neutral motion terms; range_var
                // (depth_var) carries the ricoh-specific uncertainty. mask_range = the depth centroid range.
                motion_dotd.push_back(0.0f);  motion_bias.push_back(0.0f);  motion_var.push_back(0.0f);
                trunc_frac.push_back(0.0f);   centroid_radius.push_back(0.0f);
                mask_range.push_back(b.centroid_room.norm());
            }
        }
        else
        {
            // Bearing-only slice (no lidar depth): unchanged Part-B behaviour.
            has_depth_flags.push_back(0.0f);
            depth_var.push_back(0.0f);
            support_offsets.push_back(support_offsets.back());     // no 3D points → begin == end
            const float nan = std::numeric_limits<float>::quiet_NaN();
            centroids_xyz.insert(centroids_xyz.end(), {nan, nan, nan});
            bbox_min_xyz.insert(bbox_min_xyz.end(),   {nan, nan, nan});
            bbox_max_xyz.insert(bbox_max_xyz.end(),   {nan, nan, nan});
            if (params_.MASK_MOTION_ENABLED)
            {
                motion_dotd.push_back(0.0f);  motion_bias.push_back(0.0f);      motion_var.push_back(0.0f);
                trunc_frac.push_back(0.0f);   centroid_radius.push_back(0.0f);  mask_range.push_back(0.0f);
            }
        }
    }

    const int mask_count = static_cast<int>(label_ids.size());
    G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_frame_id", static_cast<int>(sensing_frame));
    G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_timestamp_ms", frame_ts_ms);
    G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_count", mask_count);
    G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_labels", labels_joined.str());
    G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_label_ids", label_ids);
    G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_confidences", confidences);
    G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_support_offsets", support_offsets);
    G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_support_points", support_points);
    G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_support_points_cam", support_points_cam);
    G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_centroids_xyz", centroids_xyz);
    G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_bbox_min_xyz", bbox_min_xyz);
    G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_bbox_max_xyz", bbox_max_xyz);
    G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_pixels_xy", mask_pixels_xy);
    G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_pixel_offsets", mask_pixel_offsets);
    G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_has_depth", has_depth_flags);
    G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_source", mask_source);
    G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_azimuth", azimuths);
    G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_depth_var", depth_var);
    if (params_.MASK_MOTION_ENABLED)
    {
        G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_motion_dotd", motion_dotd);
        G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_motion_bias", motion_bias);
        G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_motion_var", motion_var);
        G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_trunc_frac", trunc_frac);
        G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_centroid_radius", centroid_radius);
        G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_range", mask_range);
        const std::vector<float> twist_flat{
            cam_twist_optical.v.x(), cam_twist_optical.v.y(), cam_twist_optical.v.z(),
            cam_twist_optical.w.x(), cam_twist_optical.w.y(), cam_twist_optical.w.z()};
        G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_cam_twist", twist_flat);
        G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_frame_dt_s", frame_dt_s);
        G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_rt_lag_s", rt_lag_s);
        G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_rt_gap_s", rt_gap_s);
    }
    G_->update_node(std::move(masks_node));

    // Remember this zed pose for the next frame's twist finite-difference (only with a valid capture stamp).
    if (params_.MASK_MOTION_ENABLED and frame_ts_ms > 0)
    {
        last_zed_R_     = room_rotation;
        last_zed_t_     = room_translation;
        last_zed_ts_ms_ = frame_ts_ms;
        have_last_zed_  = true;
    }
    if (motion_csv_.is_open())
        motion_csv_.flush();

    if (params_.VERBOSE_DEBUG)
        qInfo() << "[Masks] Uploaded" << mask_count << "masks and" << total_support_points << "support points to DSR (frame=" << sensing_frame << ")";
}

void GraphPublisher::publish_skeletons(const RGBDData& rgbd,
                                       const std::vector<rc::human_pose::PoseDetection>& poses,
                                       std::uint64_t frame_ts_ms)
{
    if (ensure_node("skeleton", "MediumAquamarine", skeleton_ready_, /*relayout=*/true))
        upload_skeletons(rgbd, poses, frame_ts_ms);
}

// Deproject YOLO-pose keypoints to BODY_18 3D in the CAMERA (zed) frame and publish on the
// 'skeleton' node. Layout is fixed-stride (18 joints/body) so consumers slice by body index —
// no offsets array. Missing/low-confidence joints are written as NaN with conf 0. Coordinates are
// in metres, camera frame (x right, y forward/depth, z up) — the SAME convention upload_masks uses
// before its room transform, just stopped one step earlier. Re-parenting into room/robot frame is
// the consumer's job (human_concept owns the person node + its RT edge).
void GraphPublisher::upload_skeletons(const RGBDData& rgbd,
                                      const std::vector<rc::human_pose::PoseDetection>& poses,
                                      std::uint64_t frame_ts_ms)
{
    namespace hp = rc::human_pose;

    const std::uint64_t sensing_frame = ++skeleton_publish_seq_;
    if (sensing_frame == last_skeleton_uploaded_frame_)
        return;
    last_skeleton_uploaded_frame_ = sensing_frame;

    auto node_opt = G_->get_node("skeleton");
    if (!node_opt.has_value())
        return;
    auto& node = node_opt.value();

    const float fx = rgbd.focal_x;
    const float fy = rgbd.focal_y;
    const float cx = static_cast<float>(rgbd.width) * 0.5f;
    const float cy = static_cast<float>(rgbd.height) * 0.5f;
    const float NaN = std::numeric_limits<float>::quiet_NaN();
    const float kp_conf_min = params_.SKELETON_KP_CONF_MIN;
    const int   patch = std::max(0, params_.SKELETON_DEPTH_PATCH);

    // Median depth over a (2*patch+1) window around (col,row); 0 if no valid sample. A single-pixel
    // depth on a thin limb is often missing/noisy, so a small patch stabilises the joint range.
    auto sample_depth = [&](int col, int row) -> float
    {
        std::vector<float> vals;
        vals.reserve(static_cast<std::size_t>((2 * patch + 1) * (2 * patch + 1)));
        for (int r = row - patch; r <= row + patch; ++r)
            for (int c = col - patch; c <= col + patch; ++c)
            {
                if (r < 0 || c < 0 || r >= rgbd.height || c >= rgbd.width)
                    continue;
                const std::size_t idx = static_cast<std::size_t>(r) * rgbd.width + static_cast<std::size_t>(c);
                if (idx >= rgbd.depth.size())
                    continue;
                const float d = rgbd.depth[idx];
                if (std::isfinite(d) && d > 0.0f)
                    vals.push_back(d);
            }
        if (vals.empty())
            return 0.0f;
        std::nth_element(vals.begin(), vals.begin() + vals.size() / 2, vals.end());
        return vals[vals.size() / 2];
    };

    // Deproject one COCO keypoint to camera-frame metres; conf gates validity. Returns false if no
    // usable depth or below the confidence floor (caller writes NaN/0).
    auto deproject_coco = [&](const hp::PoseKp& kp, Eigen::Vector3f& out) -> bool
    {
        if (kp.conf < kp_conf_min)
            return false;
        const int col = static_cast<int>(std::lround(kp.x));
        const int row = static_cast<int>(std::lround(kp.y));
        if (col < 0 || row < 0 || col >= rgbd.width || row >= rgbd.height)
            return false;
        const float depth = sample_depth(col, row);
        if (depth <= 0.0f)
            return false;
        out = Eigen::Vector3f((static_cast<float>(col) - cx) * depth / fx,
                              depth,
                              (cy - static_cast<float>(row)) * depth / fy);
        return true;
    };

    std::vector<float> ids;
    std::vector<float> kp_xyz;    // count * 18 * 3, camera frame metres (NaN = missing)
    std::vector<float> kp_conf;   // count * 18, [0,1]
    ids.reserve(poses.size());
    kp_xyz.reserve(poses.size() * hp::NUM_BODY18 * 3);
    kp_conf.reserve(poses.size() * hp::NUM_BODY18);

    int written = 0;
    for (std::size_t p = 0; p < poses.size(); ++p)
    {
        const auto& pose = poses[p];

        // Deproject the 17 COCO joints first; BODY_18 then reorders + synthesizes the neck in 3D.
        std::array<Eigen::Vector3f, hp::NUM_COCO_KP> coco_xyz;
        std::array<bool,  hp::NUM_COCO_KP> coco_ok{};
        for (int k = 0; k < hp::NUM_COCO_KP; ++k)
            coco_ok[static_cast<std::size_t>(k)] = deproject_coco(pose.kp[static_cast<std::size_t>(k)],
                                                                  coco_xyz[static_cast<std::size_t>(k)]);

        ids.push_back(pose.track_id >= 0 ? static_cast<float>(pose.track_id) : static_cast<float>(p));

        for (int b = 0; b < hp::NUM_BODY18; ++b)
        {
            const int src = hp::BODY18_FROM_COCO[static_cast<std::size_t>(b)];
            if (src < 0)  // neck = 3D midpoint of shoulders (valid only if both shoulders are)
            {
                const int ls = hp::COCO_L_SHOULDER, rs = hp::COCO_R_SHOULDER;
                if (coco_ok[ls] && coco_ok[rs])
                {
                    const Eigen::Vector3f neck = 0.5f * (coco_xyz[ls] + coco_xyz[rs]);
                    kp_xyz.insert(kp_xyz.end(), {neck.x(), neck.y(), neck.z()});
                    kp_conf.push_back(std::min(pose.kp[ls].conf, pose.kp[rs].conf));
                }
                else
                {
                    kp_xyz.insert(kp_xyz.end(), {NaN, NaN, NaN});
                    kp_conf.push_back(0.0f);
                }
                continue;
            }
            if (coco_ok[static_cast<std::size_t>(src)])
            {
                const Eigen::Vector3f& v = coco_xyz[static_cast<std::size_t>(src)];
                kp_xyz.insert(kp_xyz.end(), {v.x(), v.y(), v.z()});
                kp_conf.push_back(pose.kp[static_cast<std::size_t>(src)].conf);
            }
            else
            {
                kp_xyz.insert(kp_xyz.end(), {NaN, NaN, NaN});
                kp_conf.push_back(0.0f);
            }
        }
        ++written;
    }

    G_->runtime_checked_add_or_modify_attrib_local(node, "skeleton_frame_id", static_cast<int>(sensing_frame));
    G_->runtime_checked_add_or_modify_attrib_local(node, "skeleton_timestamp_ms", frame_ts_ms);
    G_->runtime_checked_add_or_modify_attrib_local(node, "skeleton_count", written);
    G_->runtime_checked_add_or_modify_attrib_local(node, "skeleton_ids", ids);
    G_->runtime_checked_add_or_modify_attrib_local(node, "skeleton_kp_xyz", kp_xyz);
    G_->runtime_checked_add_or_modify_attrib_local(node, "skeleton_kp_conf", kp_conf);
    G_->update_node(std::move(node));

    // Detection signal: log when the people count changes (so you SEE 0→1 when someone steps in),
    // and always under verbose. valid_joints = keypoints that got a real depth (the rest are NaN).
    const long valid_joints = std::count_if(kp_conf.begin(), kp_conf.end(), [](float c){ return c > 0.0f; });
    // Log on a count change (0↔N transitions), plus a periodic heartbeat (~every 30 frames ≈ 3s @10Hz)
    // while anyone is present so the stream stays visible in steady state. std::cout (not qInfo) so it
    // shows on the same channel as the detector banners — RoboComp typically filters Qt info messages.
    const bool changed   = (written != last_logged_skeleton_count_);
    const bool heartbeat = (written > 0 and sensing_frame % 30 == 0);
    if (changed or heartbeat or params_.VERBOSE_DEBUG)
    {
        std::println("[Skeleton] {} people, {} of {} joints with depth (camera frame, frame={})",
                     written, valid_joints, written * 18, sensing_frame);
        last_logged_skeleton_count_ = written;
    }
}

void GraphPublisher::cleanup_semantic_grid_nodes()
{
    for (const auto& node : G_->get_nodes_by_type("semantic_grid"))
    {
        qInfo() << "[GraphPublisher] Removing stale '" << node.name().c_str() << "' node (id=" << node.id() << ") from DSR graph";
        G_->delete_node(node.id());
    }
    masks_ready_    = false;
    skeleton_ready_ = false;
}
