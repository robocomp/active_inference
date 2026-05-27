#include "voxel_processor.h"

#ifdef emit
#undef emit
#endif

#include "unified_voxel_grid.h"
#include "voxel_opengl_viewer.h"

#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <execution>
#include <iterator>
#include <limits>
#include <numeric>
#include <print>
#include <unordered_set>

VoxelProcessor::VoxelProcessor(UnifiedVoxelGrid& voxel_grid)
    : voxel_grid_(voxel_grid)
{
}

void VoxelProcessor::configure(const Config& config)
{
    config_ = config;
}

void VoxelProcessor::clear_state(rc::VoxelOpenGLViewer* voxel_viewer)
{
    voxel_grid_.clear();
    active_tracks_.clear();
    last_track_points_.clear();
    next_track_id_ = 1;
    compute_frame_ = 0;
    last_voxel_viewer_update_ = {};

    if (voxel_viewer == nullptr)
        return;

    const std::vector<QVector3D> empty_points;
    const std::vector<std::string> empty_categories;
    voxel_viewer->update_voxels(empty_points);
    voxel_viewer->update_track_boxes(empty_points, empty_points, empty_categories);
}

void VoxelProcessor::fuse_lidar_support_points(
    const std::unordered_map<int, std::vector<Eigen::Vector3f>>& lidar_points_by_track)
{
    if (lidar_points_by_track.empty() || last_frame_id_ <= 0)
        return;

    for (const auto& [track_id, points] : lidar_points_by_track)
    {
        if (points.empty())
            continue;

        const auto track_it = active_tracks_.find(track_id);
        if (track_it == active_tracks_.end())
            continue;

        if (track_it->second.label != "table")
            continue;

        voxel_grid_.observe(track_id,
                            points,
                            track_it->second.label,
                            last_frame_id_,
                            {},
                            {},
                            0.3f);
    }
}

void VoxelProcessor::process_rgbd_frame(const RGBDData& rgbd,
                                        const std::vector<SegDetection>& detections,
                                        const Mat::RTMat& room_T_robot,
                                        const Mat::RTMat& room_T_zed,
                                        std::span<const GraphObjectBox> explained_boxes,
                                        rc::VoxelOpenGLViewer* voxel_viewer)
{
    const auto& raw_depth = rgbd.depth;
    const int img_w = rgbd.width;
    const int img_h = rgbd.height;
    const float fx = rgbd.focal_x;
    const float fy = rgbd.focal_y;

    if (raw_depth.empty() || img_w <= 0 || img_h <= 0
        || static_cast<int>(raw_depth.size()) != img_w * img_h
        || fx <= 0.f || fy <= 0.f)
        return;

    std::size_t valid_points = 0;
    std::size_t masked_points = 0;
    std::size_t selected_points = 0;
    std::size_t decimated_points = 0;
    std::size_t selected_tables = 0;
    std::size_t selected_chairs = 0;
    std::size_t selected_monitors = 0;
    std::size_t explained_points_skipped = 0;
    const std::size_t graph_box_count = explained_boxes.size();
    const std::size_t model_table_box_count = static_cast<std::size_t>(std::count_if(
        explained_boxes.begin(),
        explained_boxes.end(),
        [](const GraphObjectBox& box)
        {
            return box.category == "model_table";
        }));
    std::vector<float> det_median_range_m(detections.size(), std::numeric_limits<float>::quiet_NaN());
    std::vector<int32_t> pixel_owner(static_cast<std::size_t>(img_w * img_h), -1);
    build_owner_map_and_medians(rgbd, detections, pixel_owner, det_median_range_m);

    const float cx = static_cast<float>(img_w) * 0.5f;
    const float cy = static_cast<float>(img_h) * 0.5f;

    const std::size_t n_dets = detections.size();
    std::vector<std::vector<Eigen::Vector3f>> points_by_det(n_dets);
    std::vector<std::vector<Eigen::Vector3f>> sensed_points_by_det(n_dets);
    std::vector<std::size_t> selected_points_per_det(n_dets, 0);
    std::vector<Eigen::Vector3f> selected_points_camera;
    std::vector<std::size_t> selected_det_indices;
    selected_points_camera.reserve(static_cast<std::size_t>(img_w * img_h / 6));
    selected_det_indices.reserve(selected_points_camera.capacity());

    for (int row = 0; row < img_h; ++row)
    {
        for (int col = 0; col < img_w; ++col)
        {
            const std::size_t idx = static_cast<std::size_t>(row * img_w + col);
            const float depth_value = raw_depth[idx];
            if (!std::isfinite(depth_value) || depth_value <= 0.f)
                continue;

            ++valid_points;

            const int32_t owner = pixel_owner[idx];
            if (owner < 0)
                continue;

            // Camera frame convention in CameraAPI: +Y forward, +Z up.
            const float px = (static_cast<float>(col) - cx) * depth_value / fx;
            const float py = depth_value;
            const float pz = (cy - static_cast<float>(row)) * depth_value / fy;
            const float rng_sq = px * px + py * py + pz * pz;
            if (rng_sq < 0.01f || rng_sq > 100.0f)
                continue;
            const float rng = std::sqrt(rng_sq);

            const std::size_t det_idx = static_cast<std::size_t>(owner);
            if (det_idx >= n_dets)
                continue;

            const auto& det = detections[det_idx];
            const float ref_rng = det_median_range_m[det_idx];
            if (std::isfinite(ref_rng) && std::abs(rng - ref_rng) > 0.35f)
                continue;

            ++masked_points;
            ++selected_points;
            selected_points_camera.emplace_back(px, py, pz);
            selected_det_indices.push_back(det_idx);
            ++selected_points_per_det[det_idx];

            if (det.label == "table")
                ++selected_tables;
            else if (det.label == "chair")
                ++selected_chairs;
            else if (det.label == "monitor")
                ++selected_monitors;
        }
    }

    std::vector<Eigen::Vector3f> centroid_sums_by_det(n_dets, Eigen::Vector3f::Zero());
    std::vector<std::size_t> centroid_counts_by_det(n_dets, 0);
    std::vector<Eigen::Vector3f> sensed_centroid_sums_by_det(n_dets, Eigen::Vector3f::Zero());
    std::vector<std::size_t> sensed_centroid_counts_by_det(n_dets, 0);

    if (!selected_points_camera.empty())
    {
        for (std::size_t d = 0; d < n_dets; ++d)
        {
            points_by_det[d].reserve(selected_points_per_det[d]);
            sensed_points_by_det[d].reserve(selected_points_per_det[d]);
        }

        const std::size_t n_sel = selected_points_camera.size();
        // Depth unprojection yields camera-frame points. Convert with room <- zed.
        const Eigen::Matrix3f room_rotation = room_T_zed.linear().cast<float>();
        const Eigen::Vector3f room_translation = room_T_zed.translation().cast<float>();
        const float z_lift_m = config_.z_lift_m;
        std::vector<Eigen::Vector3f> selected_points_room(n_sel);

        std::transform(std::execution::par_unseq,
                       selected_points_camera.begin(),
                       selected_points_camera.end(),
                       selected_points_room.begin(),
                       [&](const Eigen::Vector3f& point)
                       {
                           Eigen::Vector3f room_point = room_rotation * point + room_translation;
                           room_point.z() += z_lift_m;
                           return room_point;
                       });

        for (std::size_t i = 0; i < n_sel; ++i)
        {
            const std::size_t det_idx = selected_det_indices[i];
            sensed_points_by_det[det_idx].push_back(selected_points_room[i]);
            sensed_centroid_sums_by_det[det_idx] += selected_points_room[i];
            ++sensed_centroid_counts_by_det[det_idx];
            if (point_explained_by_model(selected_points_room[i], detections[det_idx].label, explained_boxes))
            {
                ++explained_points_skipped;
                continue;
            }
            points_by_det[det_idx].push_back(selected_points_room[i]);
            centroid_sums_by_det[det_idx] += selected_points_room[i];
            ++centroid_counts_by_det[det_idx];
        }
    }

    const int frame_id = ++compute_frame_;
    std::vector<DetectionObservation> observations;
    observations.reserve(n_dets);

    for (std::size_t d = 0; d < n_dets; ++d)
    {
        if (sensed_points_by_det[d].empty())
            continue;
        if (!is_target_label(detections[d].label))
            continue;

        const Eigen::Vector3f centroid = sensed_centroid_sums_by_det[d] / static_cast<float>(sensed_centroid_counts_by_det[d]);

        observations.push_back(DetectionObservation{
            .det_index = d,
            .centroid = centroid,
            .label = detections[d].label,
            .confidence = detections[d].confidence
        });
    }

    std::vector<int> det_to_track(n_dets, -1);
    if (!observations.empty())
    {
        auto track_ids = associate_detections_hungarian(observations, frame_id);
        for (std::size_t i = 0; i < observations.size(); ++i)
            det_to_track[observations[i].det_index] = track_ids[i];
    }
    else
    {
        prune_stale_tracks(frame_id);
    }

    last_track_points_.clear();
    for (std::size_t d = 0; d < n_dets; ++d)
    {
        const int track_id = det_to_track[d];
        if (track_id < 0 || sensed_points_by_det[d].empty())
            continue;

        auto& track_points = last_track_points_[track_id];
        track_points.insert(track_points.end(), sensed_points_by_det[d].begin(), sensed_points_by_det[d].end());
    }

    std::vector<TrackBoxCandidate> box_candidates;
    const std::size_t voxel_decimation_step = std::max<std::size_t>(1, config_.voxel_decimation_factor);
    for (std::size_t d = 0; d < n_dets; ++d)
    {
        const int track_id = det_to_track[d];
        if (track_id < 0 || points_by_det[d].empty())
            continue;

        if (voxel_decimation_step == 1)
        {
            decimated_points += points_by_det[d].size();
            voxel_grid_.observe(track_id,
                                points_by_det[d],
                                detections[d].label,
                                frame_id,
                                {},
                                {},
                                detections[d].confidence);
            continue;
        }

        std::vector<Eigen::Vector3f> pts_decimated;
        pts_decimated.reserve((points_by_det[d].size() + voxel_decimation_step - 1) / voxel_decimation_step);
        for (std::size_t i = 0; i < points_by_det[d].size(); i += voxel_decimation_step)
            pts_decimated.push_back(points_by_det[d][i]);

        decimated_points += pts_decimated.size();
        voxel_grid_.observe(track_id,
                            pts_decimated,
                            detections[d].label,
                            frame_id,
                            {},
                            {},
                            detections[d].confidence);
    }

    box_candidates = build_track_box_candidates();
    merge_duplicate_tracks(box_candidates, frame_id);
    // Capture pre-suppression candidates and points for DSR sensing (TableCapture reads these).
    // The overlapping table track may be removed from the live voxel grid by model suppression,
    // so candidate_pts_att must come from this snapshot instead of the post-suppression state.
    last_box_candidates_ = box_candidates;
    std::unordered_set<int> snapshot_track_ids;
    snapshot_track_ids.reserve(last_box_candidates_.size() + last_track_points_.size());
    for (auto& candidate : last_box_candidates_)
    {
        snapshot_track_ids.insert(candidate.track_id);
        if (const auto it = last_track_points_.find(candidate.track_id); it != last_track_points_.end())
        {
            const auto stats = compute_point_cloud_stats(it->second);
            if (stats.count > 0)
            {
                candidate.min = stats.min;
                candidate.max = stats.max;
                candidate.centroid = stats.sum / static_cast<float>(stats.count);
                candidate.voxel_count = static_cast<int>(stats.count);
            }
            continue;
        }

        auto pts = voxel_grid_.get_points(candidate.track_id);
        if (pts.empty())
            continue;
        last_track_points_[candidate.track_id] = std::move(pts);
    }
    for (const auto& [track_id, pts] : last_track_points_)
    {
        if (pts.empty() || snapshot_track_ids.contains(track_id))
            continue;
        const auto track_it = active_tracks_.find(track_id);
        if (track_it == active_tracks_.end())
            continue;

        const auto stats = compute_point_cloud_stats(pts);
        if (stats.count == 0)
            continue;

        last_box_candidates_.push_back(TrackBoxCandidate{
            .track_id = track_id,
            .category = track_it->second.label,
            .min = stats.min,
            .max = stats.max,
            .centroid = stats.sum / static_cast<float>(stats.count),
            .voxel_count = static_cast<int>(stats.count),
            .last_seen_frame = track_it->second.last_seen_frame
        });
    }
    last_frame_id_ = frame_id;
    const std::size_t residual_tracks_suppressed = suppress_residual_tracks_near_models(box_candidates, explained_boxes);
    const std::size_t model_residual_table_track_count = static_cast<std::size_t>(std::count_if(
        box_candidates.begin(),
        box_candidates.end(),
        [&](const TrackBoxCandidate& box)
        {
            if (box.category != "table")
                return false;

            return std::ranges::any_of(explained_boxes,
                                        [&](const GraphObjectBox& model_box)
                                        {
                                            return model_matches_label(model_box, box.category)
                                                && boxes_overlap(box.min, box.max, model_box.min, model_box.max, model_track_padding_m_);
                                        });
        }));

    if (voxel_viewer != nullptr)
    {
        const auto now = std::chrono::steady_clock::now();
        const auto viewer_interval = config_.viewer_voxel_fps > 0
            ? std::chrono::milliseconds{1000 / config_.viewer_voxel_fps}
            : std::chrono::milliseconds{0};
        const bool should_update_voxels = viewer_interval.count() == 0
            || last_voxel_viewer_update_ == std::chrono::steady_clock::time_point{}
            || now - last_voxel_viewer_update_ >= viewer_interval;
        if (should_update_voxels)
        {
            last_voxel_viewer_update_ = now;
            const auto sem = voxel_grid_.export_display_semantic_voxels(config_.viewer_max_rendered_voxels);
            std::vector<QVector3D> qpts(sem.points.size());
            std::transform(std::execution::par_unseq,
                           sem.points.begin(),
                           sem.points.end(),
                           qpts.begin(),
                           [](const Eigen::Vector3f& point)
                           {
                               return QVector3D(point.x(), point.y(), point.z());
                           });
            voxel_viewer->update_voxels(qpts, sem.categories, sem.probs);
        }

        const auto filtered_boxes = filter_track_boxes_for_viewer(box_candidates);

        std::vector<QVector3D> box_mins;
        std::vector<QVector3D> box_maxs;
        std::vector<std::string> box_categories;
        box_mins.reserve(filtered_boxes.size());
        box_maxs.reserve(filtered_boxes.size());
        box_categories.reserve(filtered_boxes.size());
        for (const auto& box : filtered_boxes)
        {
            box_mins.emplace_back(box.min.x(), box.min.y(), box.min.z());
            box_maxs.emplace_back(box.max.x(), box.max.y(), box.max.z());
            box_categories.push_back(box.category);
        }

        voxel_viewer->update_track_boxes(box_mins, box_maxs, box_categories);
    }

    if (compute_frame_ % 30 == 0)
    {
        std::print("[ModelSuppression] graph_boxes={} model_table_boxes={} explained_pts={} selected_table_pts={} residual_table_tracks={} removed_residual_tracks={}\n",
                   graph_box_count,
                   model_table_box_count,
                   explained_points_skipped,
                   selected_tables,
                   model_residual_table_track_count,
                   residual_tracks_suppressed);

        // Print location of each model_table box (from DSR graph)
        for (const auto& box : explained_boxes)
        {
            if (box.category != "model_table") continue;
            const Eigen::Vector3f ctr = (box.min + box.max) * 0.5f;
            std::println("  [ModelBox '{}'] centre=({:.2f},{:.2f},{:.2f})  min=({:.2f},{:.2f},{:.2f})  max=({:.2f},{:.2f},{:.2f})",
                         box.category, ctr.x(), ctr.y(), ctr.z(),
                         box.min.x(), box.min.y(), box.min.z(),
                         box.max.x(), box.max.y(), box.max.z());
        }
        // Print location of each YOLO table track candidate
        for (const auto& box : box_candidates)
        {
            if (box.category != "table") continue;
            std::println("  [TrackBox '{}'] centre=({:.2f},{:.2f},{:.2f})  min=({:.2f},{:.2f},{:.2f})  max=({:.2f},{:.2f},{:.2f})",
                         box.category, box.centroid.x(), box.centroid.y(), box.centroid.z(),
                         box.min.x(), box.min.y(), box.min.z(),
                         box.max.x(), box.max.y(), box.max.z());
        }
    }

    if (config_.verbose_debug && compute_frame_ % 30 == 0)
    {
        const float ratio = valid_points > 0
            ? (100.0f * static_cast<float>(masked_points) / static_cast<float>(valid_points))
            : 0.0f;
        const int viewer_voxel_fps = config_.viewer_voxel_fps;
        std::println("[VoxelDebug] valid_pts={} masked_pts={} ({:.1f}%) selected_pts={} graph_boxes={} model_table_boxes={} explained_pts={} decimated_pts={} table={} chair={} monitor={} detections={} active_tracks={}",
                     valid_points,
                     masked_points,
                     ratio,
                     selected_points,
                     graph_box_count,
                     model_table_box_count,
                     explained_points_skipped,
                     decimated_points,
                     selected_tables,
                     selected_chairs,
                     selected_monitors,
                     detections.size(),
                     active_tracks_.size());
        if (viewer_voxel_fps > 0)
            std::println("[VoxelDebug] viewer_voxel_fps={} viewer_max_rendered_voxels={}",
                         viewer_voxel_fps,
                         config_.viewer_max_rendered_voxels);
    }

}

std::vector<float> VoxelProcessor::get_flat_pts_for_track(int track_id, int max_pts) const
{
    auto pts = voxel_grid_.get_points(track_id);
    if (pts.empty())
    {
        if (const auto it = last_track_points_.find(track_id); it != last_track_points_.end())
            pts = it->second;
    }
    if (pts.empty())
        return {};
    if (max_pts > 0 && static_cast<int>(pts.size()) > max_pts)
    {
        const std::size_t step = pts.size() / static_cast<std::size_t>(max_pts);
        std::vector<Eigen::Vector3f> dec;
        dec.reserve(static_cast<std::size_t>(max_pts));
        for (std::size_t i = 0; i < pts.size(); i += step)
            dec.push_back(pts[i]);
        pts = std::move(dec);
    }
    std::vector<float> flat;
    flat.reserve(pts.size() * 3);
    for (const auto& p : pts)
    {
        flat.push_back(p.x());
        flat.push_back(p.y());
        flat.push_back(p.z());
    }
    return flat;
}

bool VoxelProcessor::is_target_label(const std::string& label) const
{
    return label == "table" || label == "chair" || label == "monitor";
}

std::optional<std::string> VoxelProcessor::observable_category_for_model(const std::string& model_category) const
{
    if (model_category == "model_table")
        return std::string{"table"};

    return std::nullopt;
}

bool VoxelProcessor::model_matches_label(const GraphObjectBox& box, const std::string& label) const
{
    const auto observable_category = observable_category_for_model(box.category);
    return observable_category.has_value() && *observable_category == label;
}

bool VoxelProcessor::point_inside_box(const Eigen::Vector3f& point,
                                     const GraphObjectBox& box,
                                     float padding_m) const
{
    const Eigen::Vector3f min_with_padding = box.min.array() - padding_m;
    const Eigen::Vector3f max_with_padding = box.max.array() + padding_m;
    return (point.array() >= min_with_padding.array()).all()
        && (point.array() <= max_with_padding.array()).all();
}

bool VoxelProcessor::boxes_overlap(const Eigen::Vector3f& min_a,
                                   const Eigen::Vector3f& max_a,
                                   const Eigen::Vector3f& min_b,
                                   const Eigen::Vector3f& max_b,
                                   float padding_m) const
{
    const Eigen::Vector3f expanded_min_a = min_a.array() - padding_m;
    const Eigen::Vector3f expanded_max_a = max_a.array() + padding_m;
    const Eigen::Vector3f expanded_min_b = min_b.array() - padding_m;
    const Eigen::Vector3f expanded_max_b = max_b.array() + padding_m;

    return expanded_min_a.x() <= expanded_max_b.x() && expanded_max_a.x() >= expanded_min_b.x()
        && expanded_min_a.y() <= expanded_max_b.y() && expanded_max_a.y() >= expanded_min_b.y()
        && expanded_min_a.z() <= expanded_max_b.z() && expanded_max_a.z() >= expanded_min_b.z();
}

bool VoxelProcessor::point_explained_by_model(const Eigen::Vector3f& point,
                                             const std::string& label,
                                             std::span<const GraphObjectBox> explained_boxes) const
{
    for (const auto& box : explained_boxes)
    {
        if (!model_matches_label(box, label))
            continue;
        if (point_inside_box(point, box, model_point_padding_m_))
            return true;
    }
    return false;
}

void VoxelProcessor::build_owner_map_and_medians(const RGBDData& rgbd,
                                                 const std::vector<SegDetection>& detections,
                                                 std::vector<int32_t>& pixel_owner,
                                                 std::vector<float>& det_median_range_m) const
{
    const auto& raw_depth = rgbd.depth;
    const int img_w = rgbd.width;
    const int img_h = rgbd.height;
    const float fx = rgbd.focal_x;
    const float fy = rgbd.focal_y;
    if (raw_depth.empty() || fx <= 0.f || fy <= 0.f)
        return;

    const float cx = static_cast<float>(img_w) * 0.5f;
    const float cy = static_cast<float>(img_h) * 0.5f;

    for (std::size_t d = 0; d < detections.size(); ++d)
    {
        const auto& det = detections[d];
        if (!is_target_label(det.label) || det.mask.empty())
            continue;

        const int x0 = std::max(0, det.bbox.x);
        const int y0 = std::max(0, det.bbox.y);
        const int x1 = std::min({img_w, det.bbox.x + det.bbox.width, det.mask.cols});
        const int y1 = std::min({img_h, det.bbox.y + det.bbox.height, det.mask.rows});
        if (x0 >= x1 || y0 >= y1)
            continue;

        for (int row = y0; row < y1; ++row)
        {
            for (int col = x0; col < x1; ++col)
            {
                if (det.mask.at<uint8_t>(row, col) == 0)
                    continue;
                const std::size_t idx = static_cast<std::size_t>(row * img_w + col);
                if (pixel_owner[idx] == -1)
                    pixel_owner[idx] = static_cast<int32_t>(d);
            }
        }

        std::vector<float> ranges;
        ranges.reserve(static_cast<std::size_t>((x1 - x0) * (y1 - y0) / 4));
        for (int row = y0; row < y1; ++row)
        {
            for (int col = x0; col < x1; ++col)
            {
                if (det.mask.at<uint8_t>(row, col) == 0)
                    continue;

                const float depth_value = raw_depth[static_cast<std::size_t>(row * img_w + col)];
                if (!std::isfinite(depth_value) || depth_value <= 0.f)
                    continue;

                const float px = (static_cast<float>(col) - cx) * depth_value / fx;
                const float py = depth_value;
                const float pz = (static_cast<float>(row) - cy) * depth_value / fy;
                const float rng_sq = px * px + py * py + pz * pz;
                if (rng_sq >= 0.01f && rng_sq <= 100.0f)
                    ranges.push_back(std::sqrt(rng_sq));
            }
        }

        if (!ranges.empty())
        {
            auto mid = ranges.begin() + static_cast<std::ptrdiff_t>(ranges.size() / 2);
            std::nth_element(ranges.begin(), mid, ranges.end());
            det_median_range_m[d] = *mid;
        }
    }
}

std::vector<int> VoxelProcessor::hungarian_min_cost(const std::vector<std::vector<float>>& cost) const
{
    const std::size_t n = cost.size();
    if (n == 0)
        return {};

    std::size_t m = 0;
    for (const auto& row : cost)
        m = std::max(m, row.size());

    if (m == 0)
        return std::vector<int>(n, -1);

    const std::size_t dim = std::max(n, m);
    constexpr double big_cost = 1e9;
    std::vector<std::vector<double>> a(n, std::vector<double>(dim, big_cost));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < cost[i].size(); ++j)
            a[i][j] = static_cast<double>(cost[i][j]);

    std::vector<double> u(n + 1, 0.0);
    std::vector<double> v(dim + 1, 0.0);
    std::vector<std::size_t> p(dim + 1, 0);
    std::vector<std::size_t> way(dim + 1, 0);

    for (std::size_t i = 1; i <= n; ++i)
    {
        p[0] = i;
        std::size_t j0 = 0;
        std::vector<double> minv(dim + 1, std::numeric_limits<double>::infinity());
        std::vector<bool> used(dim + 1, false);

        do
        {
            used[j0] = true;
            const std::size_t i0 = p[j0];
            double delta = std::numeric_limits<double>::infinity();
            std::size_t j1 = 0;
            for (std::size_t j = 1; j <= dim; ++j)
            {
                if (used[j])
                    continue;
                const double cur = a[i0 - 1][j - 1] - u[i0] - v[j];
                if (cur < minv[j])
                {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if (minv[j] < delta)
                {
                    delta = minv[j];
                    j1 = j;
                }
            }

            for (std::size_t j = 0; j <= dim; ++j)
            {
                if (used[j])
                {
                    u[p[j]] += delta;
                    v[j] -= delta;
                }
                else
                {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        }
        while (p[j0] != 0);

        do
        {
            const std::size_t j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        }
        while (j0 != 0);
    }

    std::vector<int> assignment(n, -1);
    for (std::size_t j = 1; j <= dim; ++j)
    {
        if (p[j] == 0)
            continue;
        const std::size_t row = p[j] - 1;
        const std::size_t col = j - 1;
        if (row < n && col < m && col < cost[row].size())
            assignment[row] = static_cast<int>(col);
    }

    return assignment;
}

std::vector<int> VoxelProcessor::associate_detections_hungarian(const std::vector<DetectionObservation>& observations,
                                                                int frame_id)
{
    std::vector<int> out(observations.size(), -1);
    if (observations.empty())
        return out;

    if (active_tracks_.empty())
    {
        for (std::size_t i = 0; i < observations.size(); ++i)
        {
            const int new_id = next_track_id_++;
            active_tracks_[new_id] = InstanceTrack{
                .id = new_id,
                .centroid = observations[i].centroid,
                .label = observations[i].label,
                .last_seen_frame = frame_id
            };
            out[i] = new_id;
        }
        return out;
    }

    std::vector<int> track_ids;
    track_ids.reserve(active_tracks_.size());
    for (const auto& [track_id, _] : active_tracks_)
        track_ids.push_back(track_id);

    constexpr float impossible_cost = 1e6f;
    std::vector<std::vector<float>> cost(observations.size(), std::vector<float>(track_ids.size(), impossible_cost));
    for (std::size_t i = 0; i < observations.size(); ++i)
    {
        for (std::size_t j = 0; j < track_ids.size(); ++j)
        {
            const auto it = active_tracks_.find(track_ids[j]);
            if (it == active_tracks_.end())
                continue;
            const auto& tr = it->second;
            if (tr.label != observations[i].label)
                continue;
            cost[i][j] = (observations[i].centroid - tr.centroid).norm();
        }
    }

    const auto assignment = hungarian_min_cost(cost);
    for (std::size_t i = 0; i < observations.size(); ++i)
    {
        const int col = assignment[i];
        if (col < 0 || static_cast<std::size_t>(col) >= track_ids.size())
            continue;

        const float c = cost[i][static_cast<std::size_t>(col)];
        if (c > config_.track_association_max_distance_m || c >= impossible_cost * 0.5f)
            continue;

        const int track_id = track_ids[static_cast<std::size_t>(col)];
        out[i] = track_id;
        auto& tr = active_tracks_[track_id];
        tr.centroid = 0.65f * tr.centroid + 0.35f * observations[i].centroid;
        tr.label = observations[i].label;
        tr.last_seen_frame = frame_id;
    }

    for (std::size_t i = 0; i < observations.size(); ++i)
    {
        if (out[i] != -1)
            continue;

        const int new_id = next_track_id_++;
        active_tracks_[new_id] = InstanceTrack{
            .id = new_id,
            .centroid = observations[i].centroid,
            .label = observations[i].label,
            .last_seen_frame = frame_id
        };
        out[i] = new_id;
    }

    prune_stale_tracks(frame_id);
    return out;
}

void VoxelProcessor::prune_stale_tracks(int frame_id)
{
    std::erase_if(active_tracks_, [&](const auto& kv)
    {
        return (frame_id - kv.second.last_seen_frame) > config_.track_max_missed_frames;
    });
}

VoxelProcessor::PointCloudStats VoxelProcessor::compute_point_cloud_stats(std::span<const Eigen::Vector3f> points) const
{
    if (points.empty())
        return {};

    if (points.size() == 1)
    {
        return PointCloudStats{
            .min = points.front(),
            .max = points.front(),
            .sum = points.front(),
            .count = 1
        };
    }

    const auto combine = [](PointCloudStats lhs, const PointCloudStats& rhs)
    {
        if (lhs.count == 0)
            return rhs;
        if (rhs.count == 0)
            return lhs;

        lhs.min = lhs.min.cwiseMin(rhs.min);
        lhs.max = lhs.max.cwiseMax(rhs.max);
        lhs.sum += rhs.sum;
        lhs.count += rhs.count;
        return lhs;
    };

    const auto to_stats = [](const Eigen::Vector3f& point)
    {
        return PointCloudStats{
            .min = point,
            .max = point,
            .sum = point,
            .count = 1
        };
    };

    return std::transform_reduce(std::execution::par,
                                 points.begin(),
                                 points.end(),
                                 PointCloudStats{},
                                 combine,
                                 to_stats);
}

std::vector<VoxelProcessor::TrackBoxCandidate> VoxelProcessor::build_track_box_candidates() const
{
    std::vector<TrackBoxCandidate> candidates;

    const auto track_ids = voxel_grid_.get_all_track_ids();
    candidates.reserve(track_ids.size());

    for (const int tid : track_ids)
    {
        if (tid <= 0)
            continue;

        const int voxel_count = voxel_grid_.get_n_voxels(tid);
        if (voxel_count < 10)
            continue;

        const auto [dom_cat, _] = voxel_grid_.object_dominant_category(tid);
        const bool use_clustered_points = static_cast<std::size_t>(voxel_count) <= max_clustered_box_points_;
        auto pts = use_clustered_points
            ? voxel_grid_.get_points_clustered(tid, dom_cat)
            : voxel_grid_.get_points(tid);

        if (pts.size() < 10 && use_clustered_points)
            pts = voxel_grid_.get_points(tid);
        if (pts.size() < 10)
            continue;

        const auto stats = compute_point_cloud_stats(pts);
        if (stats.count == 0)
            continue;

        const Eigen::Vector3f centroid = stats.sum / static_cast<float>(stats.count);

        int last_seen_frame = -1;
        if (const auto it = active_tracks_.find(tid); it != active_tracks_.end())
            last_seen_frame = it->second.last_seen_frame;

        candidates.push_back(TrackBoxCandidate{
            .track_id = tid,
            .category = dom_cat,
            .min = stats.min,
            .max = stats.max,
            .centroid = centroid,
            .voxel_count = voxel_count,
            .last_seen_frame = last_seen_frame
        });
    }

    return candidates;
}

void VoxelProcessor::merge_duplicate_tracks(std::vector<TrackBoxCandidate>& candidates, int frame_id)
{
    if (candidates.empty())
        return;

    std::sort(candidates.begin(), candidates.end(),
              [this, frame_id](const TrackBoxCandidate& lhs, const TrackBoxCandidate& rhs)
              {
                  return prefer_candidate(lhs, rhs, frame_id);
              });

    std::unordered_set<int> merged_tracks;
    for (std::size_t i = 0; i < candidates.size(); ++i)
    {
        auto& keep = candidates[i];
        if (merged_tracks.contains(keep.track_id))
            continue;

        for (std::size_t j = i + 1; j < candidates.size(); ++j)
        {
            auto& drop = candidates[j];
            if (merged_tracks.contains(drop.track_id))
                continue;
            if (!boxes_look_duplicate(drop, keep))
                continue;

            const int moved = voxel_grid_.reassign_ownership(drop.track_id, keep.track_id);
            if (moved <= 0)
                continue;

            const float keep_weight = static_cast<float>(std::max(1, keep.voxel_count));
            const float drop_weight = static_cast<float>(std::max(1, drop.voxel_count));
            keep.centroid = (keep_weight * keep.centroid + drop_weight * drop.centroid) / (keep_weight + drop_weight);
            keep.min = keep.min.cwiseMin(drop.min);
            keep.max = keep.max.cwiseMax(drop.max);
            keep.voxel_count += drop.voxel_count;
            keep.last_seen_frame = std::max(keep.last_seen_frame, drop.last_seen_frame);

            auto keep_it = active_tracks_.find(keep.track_id);
            const auto drop_it = active_tracks_.find(drop.track_id);
            if (keep_it == active_tracks_.end())
            {
                active_tracks_[keep.track_id] = InstanceTrack{
                    .id = keep.track_id,
                    .centroid = keep.centroid,
                    .label = keep.category,
                    .last_seen_frame = keep.last_seen_frame
                };
                keep_it = active_tracks_.find(keep.track_id);
            }
            else
            {
                keep_it->second.centroid = keep.centroid;
                keep_it->second.label = keep.category;
                keep_it->second.last_seen_frame = keep.last_seen_frame;
            }

            if (drop_it != active_tracks_.end())
                active_tracks_.erase(drop_it);

            merged_tracks.insert(drop.track_id);
            if (config_.verbose_debug)
            {
                std::println("[TrackMerge] merged track={} into track={} category={} moved_voxels={}",
                             drop.track_id,
                             keep.track_id,
                             keep.category,
                             moved);
            }
        }
    }

    if (!merged_tracks.empty())
    {
        std::erase_if(candidates, [&](const TrackBoxCandidate& candidate)
        {
            return merged_tracks.contains(candidate.track_id);
        });
    }
}

std::size_t VoxelProcessor::suppress_residual_tracks_near_models(std::vector<TrackBoxCandidate>& candidates,
                                                                 std::span<const GraphObjectBox> explained_boxes)
{
    if (candidates.empty() || explained_boxes.empty())
        return 0;

    std::unordered_set<int> removed_tracks;
    for (const auto& candidate : candidates)
    {
        const bool overlaps_model = std::ranges::any_of(explained_boxes,
                                                        [&](const GraphObjectBox& model_box)
                                                        {
                                                            return model_matches_label(model_box, candidate.category)
                                                                && boxes_overlap(candidate.min,
                                                                                 candidate.max,
                                                                                 model_box.min,
                                                                                 model_box.max,
                                                                                 model_track_padding_m_);
                                                        });
        if (!overlaps_model)
            continue;

        // Keep the sensing-side track identity so Hungarian association can
        // reattach the next full table observation to the same track id even
        // though the residual voxels are suppressed from the live grid/viewer.
        voxel_grid_.remove(candidate.track_id);
        removed_tracks.insert(candidate.track_id);
    }

    if (!removed_tracks.empty())
    {
        std::erase_if(candidates, [&](const TrackBoxCandidate& candidate)
        {
            return removed_tracks.contains(candidate.track_id);
        });
    }

    return removed_tracks.size();
}

std::vector<VoxelProcessor::TrackBoxCandidate> VoxelProcessor::filter_track_boxes_for_viewer(const std::vector<TrackBoxCandidate>& candidates) const
{
    std::vector<TrackBoxCandidate> filtered_boxes;
    filtered_boxes.reserve(candidates.size());
    std::unordered_map<std::string, int> kept_per_category;

    for (const auto& candidate : candidates)
    {
        bool is_duplicate = false;
        for (const auto& kept : filtered_boxes)
        {
            if (!boxes_look_duplicate(candidate, kept))
                continue;

            is_duplicate = true;
            if (config_.verbose_debug)
            {
                std::println("[TrackBox] suppressing duplicate box track={} category={} kept_track={} overlap_vol={:.3f}",
                             candidate.track_id,
                             candidate.category,
                             kept.track_id,
                             intersection_volume(candidate, kept));
            }
            break;
        }
        if (is_duplicate)
            continue;

        const int max_instances = max_instances_for_category(candidate.category);
        const int kept_instances = kept_per_category[candidate.category];
        if (kept_instances >= max_instances)
        {
            if (config_.verbose_debug)
            {
                std::println("[TrackBox] suppressing by category cap track={} category={} cap={}",
                             candidate.track_id,
                             candidate.category,
                             max_instances);
            }
            continue;
        }

        filtered_boxes.push_back(candidate);
        ++kept_per_category[candidate.category];
    }

    return filtered_boxes;
}

float VoxelProcessor::axis_overlap(float amin, float amax, float bmin, float bmax) const
{
    return std::max(0.0f, std::min(amax, bmax) - std::max(amin, bmin));
}

float VoxelProcessor::box_volume(const TrackBoxCandidate& box) const
{
    const Eigen::Vector3f ext = (box.max - box.min).cwiseMax(Eigen::Vector3f::Zero());
    return ext.x() * ext.y() * ext.z();
}

float VoxelProcessor::intersection_volume(const TrackBoxCandidate& a, const TrackBoxCandidate& b) const
{
    const float ox = axis_overlap(a.min.x(), a.max.x(), b.min.x(), b.max.x());
    const float oy = axis_overlap(a.min.y(), a.max.y(), b.min.y(), b.max.y());
    const float oz = axis_overlap(a.min.z(), a.max.z(), b.min.z(), b.max.z());
    return ox * oy * oz;
}

bool VoxelProcessor::boxes_look_duplicate(const TrackBoxCandidate& a, const TrackBoxCandidate& b) const
{
    if (a.category != b.category)
        return false;

    const float inter = intersection_volume(a, b);
    if (inter <= 1e-5f)
        return false;

    const float a_vol = box_volume(a);
    const float b_vol = box_volume(b);
    const float smaller_vol = std::max(1e-5f, std::min(a_vol, b_vol));
    const float overlap_ratio = inter / smaller_vol;
    const float centroid_distance = (a.centroid - b.centroid).norm();
    const float smaller_diag = std::min((a.max - a.min).norm(), (b.max - b.min).norm());

    return overlap_ratio >= 0.30f
        && centroid_distance <= std::max(0.55f, 0.85f * std::max(smaller_diag, 1e-3f));
}

int VoxelProcessor::max_instances_for_category(const std::string& category) const
{
    if (category == "table")
        return 1;
    if (category == "monitor" || category == "blackboard")
        return 2;
    return std::numeric_limits<int>::max();
}

bool VoxelProcessor::prefer_candidate(const TrackBoxCandidate& lhs, const TrackBoxCandidate& rhs, int frame_id) const
{
    const bool lhs_seen_now = lhs.last_seen_frame == frame_id;
    const bool rhs_seen_now = rhs.last_seen_frame == frame_id;
    if (lhs_seen_now != rhs_seen_now)
        return lhs_seen_now;

    if (lhs.voxel_count != rhs.voxel_count)
        return lhs.voxel_count > rhs.voxel_count;

    return lhs.track_id < rhs.track_id;
}