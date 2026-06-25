/*
 * graph_publisher.cpp — DSR semantic_grid exports (masks / tracks / voxels).
 */

// These pull in oneTBB (tbb/profiling.h). Include them BEFORE any header that activates Qt's `emit`
// keyword macro (dsr_api.h via graph_publisher.h, and <QtGlobal>), otherwise the empty `emit` macro
// mangles TBB's profiling.h and it fails to compile.
#include "unified_voxel_grid.h"
#include "voxel_processor.h"
#include "yolo_processor.h"   // SegDetection

#include "graph_publisher.h"

#include <algorithm>
#include <limits>
#include <print>
#include <sstream>
#include <utility>

#include <QtGlobal>

GraphPublisher::GraphPublisher(std::shared_ptr<DSR::DSRGraph> graph,
                               const VoxelizerParams& params,
                               VoxelProcessor* voxel_processor,
                               UnifiedVoxelGrid* voxel_grid,
                               std::function<void()> relayout)
    : G_(std::move(graph)), params_(params), voxel_processor_(voxel_processor),
      voxel_grid_(voxel_grid), relayout_(std::move(relayout))
{}

void GraphPublisher::publish(const RGBDData& rgbd, const Mat::RTMat& room_T_zed,
                             const std::vector<SegDetection>& detections, std::uint64_t frame_ts_ms)
{
    if (ensure_node("masks", "Plum", masks_ready_, /*relayout=*/true))
        upload_masks(rgbd, room_T_zed, detections, frame_ts_ms);

    // tracks deliberately does NOT relayout: a full twopi reshuffle racing another agent's node
    // insert/remove can paint a freed QGraphicsItem → SIGSEGV in the viewer. Cosmetic only.
    if (params_.PUBLISH_TRACKS and ensure_node("tracks", "SteelBlue", tracks_ready_, /*relayout=*/false))
        publish_tracks();

    if (params_.PUBLISH_VOXELS and ensure_node("voxels", "Khaki", voxels_ready_, /*relayout=*/true))
        upload_voxels();
}

void GraphPublisher::refresh_voxels_node()
{
    if (params_.PUBLISH_VOXELS and ensure_node("voxels", "Khaki", voxels_ready_, /*relayout=*/true))
        upload_voxels();
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

void GraphPublisher::upload_masks(const RGBDData& rgbd, const Mat::RTMat& room_T_zed,
                                  const std::vector<SegDetection>& detections, std::uint64_t frame_ts_ms)
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
        G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_centroids_xyz", std::vector<float>{});
        G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_bbox_min_xyz", std::vector<float>{});
        G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_bbox_max_xyz", std::vector<float>{});
        G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_pixels_xy", std::vector<float>{});
        G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_pixel_offsets", std::vector<float>{0.0f});
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

    std::vector<float> label_ids;
    std::vector<float> confidences;
    std::vector<float> support_offsets;
    std::vector<float> support_points;
    std::vector<float> centroids_xyz;
    std::vector<float> bbox_min_xyz;
    std::vector<float> bbox_max_xyz;
    // Raw 2D mask silhouette (foreground pixels, BEFORE the depth gate) so downstream agents have
    // the RGB mask as an evidence channel independent of depth — flat (col,row) pairs per mask,
    // delimited by mask_pixel_offsets exactly like support_points/support_offsets.
    std::vector<float> mask_pixels_xy;
    std::vector<float> mask_pixel_offsets;
    std::ostringstream labels_joined;
    std::size_t total_support_points = 0;
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
        gated.reserve(static_cast<std::size_t>(det.bbox.area() / std::max<int>(1, static_cast<int>(mask_stride * mask_stride))));
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
                Eigen::Vector3f point_room = room_rotation * Eigen::Vector3f(px, py, pz) + room_translation;
                point_room.z() += z_lift_m;

                gated.push_back(point_room);
            }
        }

        if (gated.empty())
            continue;

        // Radius outlier removal: keep points that have enough neighbours nearby. The dense
        // object body survives; the sparse silhouette-edge tail is trimmed.
        std::vector<Eigen::Vector3f> mask_points_room;
        if (params_.MASK_OUTLIER_MIN_NEIGHBORS > 0 and params_.MASK_OUTLIER_RADIUS_M > 0.0f
            and gated.size() > static_cast<std::size_t>(params_.MASK_OUTLIER_MIN_NEIGHBORS))
        {
            const float r2 = params_.MASK_OUTLIER_RADIUS_M * params_.MASK_OUTLIER_RADIUS_M;
            mask_points_room.reserve(gated.size());
            for (std::size_t i = 0; i < gated.size(); ++i)
            {
                int neighbours = 0;
                for (std::size_t j = 0; j < gated.size() and neighbours < params_.MASK_OUTLIER_MIN_NEIGHBORS; ++j)
                    if (i != j and (gated[i] - gated[j]).squaredNorm() <= r2)
                        ++neighbours;
                if (neighbours >= params_.MASK_OUTLIER_MIN_NEIGHBORS)
                    mask_points_room.push_back(gated[i]);
            }
        }
        else
            mask_points_room = std::move(gated);

        if (mask_points_room.empty())
            continue;

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

        total_support_points += mask_points_room.size();
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
    G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_centroids_xyz", centroids_xyz);
    G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_bbox_min_xyz", bbox_min_xyz);
    G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_bbox_max_xyz", bbox_max_xyz);
    G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_pixels_xy", mask_pixels_xy);
    G_->runtime_checked_add_or_modify_attrib_local(masks_node, "mask_pixel_offsets", mask_pixel_offsets);
    G_->update_node(std::move(masks_node));

    if (params_.VERBOSE_DEBUG)
        qInfo() << "[Masks] Uploaded" << mask_count << "masks and" << total_support_points << "support points to DSR (frame=" << sensing_frame << ")";
}

// Feed-forward, class-agnostic publisher. Exports ALL current voxel tracks (every category) into the
// 'tracks' node, room frame, as a struct-of-arrays keyed by persistent track id. Concept agents read
// these and do their own instance assignment. Uses runtime/dynamic attributes (like 'masks').
void GraphPublisher::publish_tracks()
{
    auto node_opt = G_->get_node("tracks");
    if (!node_opt.has_value())
        return;
    auto& node = node_opt.value();

    const auto& tracks        = voxel_processor_->last_track_candidates();
    const int   sensing_frame = voxel_processor_->last_frame_id();
    constexpr int kMaxPtsPerTrack = 400;

    std::vector<float> ids, label_ids, confidences, last_seen, voxel_counts;
    std::vector<float> centroids_xyz, bbox_min_xyz, bbox_max_xyz;
    std::vector<float> support_offsets{0.0f};   // prefix offsets, in POINTS
    std::vector<float> support_points;
    std::ostringstream labels_joined;
    int published = 0;

    for (const auto& t : tracks)
    {
        const auto pts = voxel_processor_->get_flat_pts_for_track(t.track_id, kMaxPtsPerTrack);
        if (pts.empty())
            continue;

        if (!labels_joined.str().empty())
            labels_joined << '|';
        labels_joined << t.category;

        ids.push_back(static_cast<float>(t.track_id));
        label_ids.push_back(0.0f);                 // class id slot (category string carries the label)
        confidences.push_back(1.0f);               // label-vote confidence (placeholder)
        last_seen.push_back(static_cast<float>(t.last_seen_frame));
        voxel_counts.push_back(static_cast<float>(t.voxel_count));

        centroids_xyz.insert(centroids_xyz.end(), {t.centroid.x(), t.centroid.y(), t.centroid.z()});
        bbox_min_xyz.insert(bbox_min_xyz.end(),   {t.min.x(), t.min.y(), t.min.z()});
        bbox_max_xyz.insert(bbox_max_xyz.end(),   {t.max.x(), t.max.y(), t.max.z()});

        support_points.insert(support_points.end(), pts.begin(), pts.end());
        support_offsets.push_back(static_cast<float>(support_points.size() / 3));
        ++published;
    }

    G_->runtime_checked_add_or_modify_attrib_local(node, "track_frame_id",      sensing_frame);
    G_->runtime_checked_add_or_modify_attrib_local(node, "track_count",         published);
    G_->runtime_checked_add_or_modify_attrib_local(node, "track_ids",           ids);
    G_->runtime_checked_add_or_modify_attrib_local(node, "track_labels",        labels_joined.str());
    G_->runtime_checked_add_or_modify_attrib_local(node, "track_label_ids",     label_ids);
    G_->runtime_checked_add_or_modify_attrib_local(node, "track_confidences",   confidences);
    G_->runtime_checked_add_or_modify_attrib_local(node, "track_last_seen",     last_seen);
    G_->runtime_checked_add_or_modify_attrib_local(node, "track_voxel_counts",  voxel_counts);
    G_->runtime_checked_add_or_modify_attrib_local(node, "track_centroids_xyz", centroids_xyz);
    G_->runtime_checked_add_or_modify_attrib_local(node, "track_bbox_min_xyz",  bbox_min_xyz);
    G_->runtime_checked_add_or_modify_attrib_local(node, "track_bbox_max_xyz",  bbox_max_xyz);
    G_->runtime_checked_add_or_modify_attrib_local(node, "track_support_offsets", support_offsets);
    G_->runtime_checked_add_or_modify_attrib_local(node, "track_support_points",  support_points);
    G_->update_node(std::move(node));

    if (sensing_frame % 30 == 0)
        std::println("[Tracks] frame={} published={} support_pts={}",
                     sensing_frame, published, support_points.size() / 3);
}

void GraphPublisher::upload_voxels()
{
    const int sensing_frame = voxel_processor_->last_frame_id();
    if (sensing_frame % 30 != 0)
        return;

    auto voxels_node_opt = G_->get_node("voxels");
    if (!voxels_node_opt.has_value())
        return;
    auto& voxels_node = voxels_node_opt.value();

    const auto exported = voxel_grid_->export_semantic_voxels();

    // Stride-5 encoding per voxel: [x, y, z, prob, track_id]. Receivers must interpret with stride 5.
    const std::size_t n = exported.points.size();
    std::vector<float> flat_pts;
    flat_pts.reserve(n * 5);
    for (std::size_t i = 0; i < n; ++i)
    {
        const auto& pt = exported.points[i];
        flat_pts.push_back(pt.x());
        flat_pts.push_back(pt.y());
        flat_pts.push_back(pt.z());
        flat_pts.push_back(i < exported.probs.size()    ? exported.probs[i]                    : 0.0f);
        flat_pts.push_back(i < exported.track_ids.size() ? static_cast<float>(exported.track_ids[i]) : -1.0f);
    }

    G_->add_or_modify_attrib_local<candidate_pts_att>(voxels_node, flat_pts);
    G_->add_or_modify_attrib_local<last_sensing_frame_att>(voxels_node, sensing_frame);
    G_->update_node(std::move(voxels_node));   // last use: move the voxel-grid blob into the engine (no deep copy under _mutex)

    if (params_.VERBOSE_DEBUG)
        qInfo() << "[Voxels] Uploaded" << n << "voxels (stride-5) to DSR (frame=" << sensing_frame << ")";
}

void GraphPublisher::cleanup_semantic_grid_nodes()
{
    for (const auto& node : G_->get_nodes_by_type("semantic_grid"))
    {
        qInfo() << "[GraphPublisher] Removing stale '" << node.name().c_str() << "' node (id=" << node.id() << ") from DSR graph";
        G_->delete_node(node.id());
    }
    masks_ready_  = false;
    tracks_ready_ = false;
    voxels_ready_ = false;
}
