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

void VoxelProcessor::process_rgbd_frame(const RoboCompCameraRGBDSimple::TRGBD& rgbd,
                                        const std::vector<SegDetection>& detections,
                                        const Mat::RTMat& room_T_robot,
                                        const Mat::RTMat& room_T_zed,
                                        rc::VoxelOpenGLViewer* voxel_viewer)
{
    const auto& raw_pts = rgbd.points.points;
    const int img_w = rgbd.image.width;
    const int img_h = rgbd.image.height;

    if (raw_pts.empty() || img_w <= 0 || img_h <= 0 || static_cast<int>(raw_pts.size()) != img_w * img_h)
        return;

    std::size_t valid_points = 0;
    std::size_t masked_points = 0;
    std::size_t selected_points = 0;
    std::size_t decimated_points = 0;
    std::unordered_map<std::string, std::size_t> selected_by_class;
    std::vector<float> det_median_range_m(detections.size(), std::numeric_limits<float>::quiet_NaN());
    std::vector<int32_t> pixel_owner(static_cast<std::size_t>(img_w * img_h), -1);
    const float point_scale = detect_point_scale_once(rgbd);
    build_owner_map_and_medians(rgbd, point_scale, detections, pixel_owner, det_median_range_m);

    const std::size_t n_dets = detections.size();
    std::vector<std::vector<Eigen::Vector3f>> points_by_det(n_dets);
    std::vector<std::vector<std::string>> labels_by_det(n_dets);
    std::vector<std::vector<float>> confs_by_det(n_dets);
    std::vector<Eigen::Vector3f> selected_points_robot;
    std::vector<std::size_t> selected_det_indices;
    selected_points_robot.reserve(static_cast<std::size_t>(img_w * img_h / 6));
    selected_det_indices.reserve(selected_points_robot.capacity());

    for (int row = 0; row < img_h; ++row)
    {
        for (int col = 0; col < img_w; ++col)
        {
            const std::size_t idx = static_cast<std::size_t>(row * img_w + col);
            const auto& p = raw_pts[idx];

            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
                continue;

            const float px = p.x * point_scale;
            const float py = p.y * point_scale;
            const float pz = p.z * point_scale;
            const float rng_sq = px * px + py * py + pz * pz;
            if (rng_sq < 0.01f || rng_sq > 100.0f)
                continue;
            const float rng = std::sqrt(rng_sq);

            ++valid_points;

            const int32_t owner = pixel_owner[idx];
            if (owner < 0)
                continue;

            const std::size_t det_idx = static_cast<std::size_t>(owner);
            if (det_idx >= n_dets)
                continue;

            const auto& det = detections[det_idx];
            const float ref_rng = det_median_range_m[det_idx];
            if (std::isfinite(ref_rng) && std::abs(rng - ref_rng) > 0.35f)
                continue;

            ++masked_points;
            ++selected_points;
            selected_points_robot.emplace_back(px, py, pz);
            selected_det_indices.push_back(det_idx);

            if (det.label == "table")
                ++selected_by_class["table"];
            else if (det.label == "chair")
                ++selected_by_class["chair"];
            else if (det.label == "monitor")
                ++selected_by_class["monitor"];
        }
    }

    if (!selected_points_robot.empty())
    {
        const std::size_t n_sel = selected_points_robot.size();
        Eigen::Matrix<double, 3, Eigen::Dynamic> pts_robot(3, static_cast<Eigen::Index>(n_sel));
        for (std::size_t i = 0; i < n_sel; ++i)
        {
            pts_robot(0, static_cast<Eigen::Index>(i)) = static_cast<double>(selected_points_robot[i].x());
            pts_robot(1, static_cast<Eigen::Index>(i)) = static_cast<double>(selected_points_robot[i].y());
            pts_robot(2, static_cast<Eigen::Index>(i)) = static_cast<double>(selected_points_robot[i].z());
        }

        Eigen::Matrix<double, 3, Eigen::Dynamic> pts_room =
            (room_T_robot.linear() * pts_robot).colwise() + room_T_zed.translation();

        for (std::size_t i = 0; i < n_sel; ++i)
        {
            const std::size_t det_idx = selected_det_indices[i];
            points_by_det[det_idx].emplace_back(
                static_cast<float>(pts_room(0, static_cast<Eigen::Index>(i))),
                static_cast<float>(pts_room(1, static_cast<Eigen::Index>(i))),
                static_cast<float>(pts_room(2, static_cast<Eigen::Index>(i))));
            labels_by_det[det_idx].push_back(detections[det_idx].label);
            confs_by_det[det_idx].push_back(detections[det_idx].confidence);
        }
    }

    const int frame_id = ++compute_frame_;
    std::vector<DetectionObservation> observations;
    observations.reserve(n_dets);

    for (std::size_t d = 0; d < n_dets; ++d)
    {
        if (points_by_det[d].empty())
            continue;
        if (!is_target_label(detections[d].label))
            continue;

        Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
        for (const auto& p : points_by_det[d])
            centroid += p;
        centroid /= static_cast<float>(points_by_det[d].size());

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

    std::vector<TrackBoxCandidate> box_candidates;
    const std::size_t voxel_decimation_step = std::max<std::size_t>(1, config_.voxel_decimation_factor);
    for (std::size_t d = 0; d < n_dets; ++d)
    {
        const int track_id = det_to_track[d];
        if (track_id < 0 || points_by_det[d].empty())
            continue;

        std::vector<Eigen::Vector3f> pts_decimated;
        std::vector<std::string> labels_decimated;
        std::vector<float> confs_decimated;

        pts_decimated.reserve((points_by_det[d].size() + voxel_decimation_step - 1) / voxel_decimation_step);
        labels_decimated.reserve(pts_decimated.capacity());
        confs_decimated.reserve(pts_decimated.capacity());

        for (std::size_t i = 0; i < points_by_det[d].size(); i += voxel_decimation_step)
        {
            pts_decimated.push_back(points_by_det[d][i]);
            labels_decimated.push_back(labels_by_det[d][i]);
            confs_decimated.push_back(confs_by_det[d][i]);
        }

        decimated_points += pts_decimated.size();
        voxel_grid_.observe(track_id,
                            pts_decimated,
                            detections[d].label,
                            frame_id,
                            labels_decimated,
                            confs_decimated,
                            detections[d].confidence);
    }

    box_candidates = build_track_box_candidates();
    merge_duplicate_tracks(box_candidates, frame_id);

    const auto sem = voxel_grid_.export_semantic_voxels();
    if (voxel_viewer != nullptr)
    {
        std::vector<QVector3D> qpts;
        qpts.reserve(sem.points.size());
        for (const auto& p : sem.points)
            qpts.emplace_back(p.x(), p.y(), p.z());
        voxel_viewer->update_voxels(qpts, sem.categories, sem.probs);

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

    if (config_.verbose_debug && compute_frame_ % 30 == 0)
    {
        const float ratio = valid_points > 0
            ? (100.0f * static_cast<float>(masked_points) / static_cast<float>(valid_points))
            : 0.0f;
        std::println("[VoxelDebug] valid_pts={} masked_pts={} ({:.1f}%) selected_pts={} decimated_pts={} table={} chair={} monitor={} detections={} active_tracks={}",
                     valid_points,
                     masked_points,
                     ratio,
                     selected_points,
                     decimated_points,
                     selected_by_class["table"],
                     selected_by_class["chair"],
                     selected_by_class["monitor"],
                     detections.size(),
                     active_tracks_.size());
    }
}

bool VoxelProcessor::is_target_label(const std::string& label) const
{
    return label == "table" || label == "chair" || label == "monitor";
}

float VoxelProcessor::detect_point_scale_once(const RoboCompCameraRGBDSimple::TRGBD& rgbd) const
{
    for (const auto& p : rgbd.points.points)
    {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
            continue;
        const float max_abs = std::max({std::abs(p.x), std::abs(p.y), std::abs(p.z)});
        return (max_abs > 50.0f) ? 0.001f : 1.0f;
    }
    return 1.0f;
}

void VoxelProcessor::build_owner_map_and_medians(const RoboCompCameraRGBDSimple::TRGBD& rgbd,
                                                 float point_scale,
                                                 const std::vector<SegDetection>& detections,
                                                 std::vector<int32_t>& pixel_owner,
                                                 std::vector<float>& det_median_range_m) const
{
    const auto& raw_pts = rgbd.points.points;
    const int img_w = rgbd.image.width;
    const int img_h = rgbd.image.height;

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

                const auto& p = raw_pts[static_cast<std::size_t>(row * img_w + col)];
                if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
                    continue;

                const float px = p.x * point_scale;
                const float py = p.y * point_scale;
                const float pz = p.z * point_scale;
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

std::vector<VoxelProcessor::TrackBoxCandidate> VoxelProcessor::build_track_box_candidates() const
{
    std::vector<TrackBoxCandidate> candidates;

    const auto track_ids = voxel_grid_.get_all_track_ids();
    candidates.reserve(track_ids.size());

    for (const int tid : track_ids)
    {
        if (tid <= 0)
            continue;

        const auto [dom_cat, _] = voxel_grid_.object_dominant_category(tid);
        auto pts = voxel_grid_.get_points_clustered(tid, dom_cat);
        if (pts.size() < 10)
            pts = voxel_grid_.get_points(tid);
        if (pts.size() < 10)
            continue;

        Eigen::Vector3f mn = pts.front();
        Eigen::Vector3f mx = pts.front();
        Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
        for (const auto& p : pts)
        {
            mn = mn.cwiseMin(p);
            mx = mx.cwiseMax(p);
            centroid += p;
        }
        centroid /= static_cast<float>(pts.size());

        int last_seen_frame = -1;
        if (const auto it = active_tracks_.find(tid); it != active_tracks_.end())
            last_seen_frame = it->second.last_seen_frame;

        candidates.push_back(TrackBoxCandidate{
            .track_id = tid,
            .category = dom_cat,
            .min = mn,
            .max = mx,
            .centroid = centroid,
            .voxel_count = voxel_grid_.get_n_voxels(tid),
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