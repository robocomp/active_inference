// RoomConcept, split out of room_concept.cpp (refactor step 2, docs/refactor_proposal_2026-10-13.md).
// Initial pose: orientation from points, yaw ambiguity, LiDAR bootstrap, seed validation, pose-fit score.

#include "room_concept.h"
#include "../../common/status_stream/status_sink.h"
#include <pthread.h>   // pthread_setname_np: name the worker so a per-thread CPU sample attributes itself
#include "pointcloud_center_estimator.h"
#include <numeric>
#include <set>
#include "room_gn_solver.h"
#include "room_obs_weights.h"
#include "room_gn_solver.h"
#include "image_edge_accumulate.h"
#include "image_edge_ops.h"
#include <algorithm>
#include <ranges>
#include <fstream>
#include <filesystem>
#include <limits>
#include <locale>
#include <ctime>
#include <print>
#include <sstream>
#include <sys/stat.h>
#include <QDebug>

namespace rc
{

    float RoomConcept::estimate_orientation_from_points(const std::vector<Eigen::Vector3f>& pts) const
    {
        if (pts.size() < 2)
            return 0.f;

        Eigen::Vector2f mean = Eigen::Vector2f::Zero();
        for (const auto& p : pts)
            mean += p.head<2>();
        mean /= static_cast<float>(pts.size());

        Eigen::Matrix2f cov = Eigen::Matrix2f::Zero();
        for (const auto& p : pts)
        {
            const Eigen::Vector2f d = p.head<2>() - mean;
            cov += d * d.transpose();
        }
        cov /= std::max<std::size_t>(1, pts.size() - 1);

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> es(cov);
        if (es.info() != Eigen::Success)
            return 0.f;

        const Eigen::Vector2f principal = es.eigenvectors().col(1);
        return std::atan2(principal.y(), principal.x());
    }

    void RoomConcept::resolve_initial_yaw_ambiguity(const std::vector<Eigen::Vector3f>& lidar_points, float prior_phi)
    {
        constexpr float kYawTieTolerance = 0.02f;
        if (model_ == nullptr || lidar_points.empty())
            return;

        auto wrap_angle = [](float a)
        {
            while (a > M_PI) a -= 2.f * M_PI;
            while (a < -M_PI) a += 2.f * M_PI;
            return a;
        };
        auto ang_dist = [&](float a, float b)
        {
            return std::abs(wrap_angle(a - b));
        };

        const auto state = get_current_state();
        const float x = state[2];
        const float y = state[3];
        const float theta = wrap_angle(state[4]);

        set_robot_pose(x, y, theta, false);
        const float sdf_theta = evaluate_pose_fit(lidar_points);

        const float theta_pi = wrap_angle(theta + static_cast<float>(M_PI));
        set_robot_pose(x, y, theta_pi, false);
        const float sdf_theta_pi = evaluate_pose_fit(lidar_points);

        const bool similar_fit = std::isfinite(sdf_theta) && std::isfinite(sdf_theta_pi)
                                 && std::abs(sdf_theta - sdf_theta_pi) <= kYawTieTolerance;
        const float d_theta = ang_dist(theta, prior_phi);
        const float d_theta_pi = ang_dist(theta_pi, prior_phi);

        if ((sdf_theta_pi < sdf_theta) || (similar_fit && d_theta_pi < d_theta))
            set_robot_pose(x, y, theta_pi, false);
        else
            set_robot_pose(x, y, theta, false);
    }

    bool RoomConcept::bootstrap_initialization_from_lidar()
    {
        if (is_initialized())
            return true;
        if (run_ctx_.high_lidar_buffer == nullptr)
            return false;

        if (estimating())
        {
            // No layout, no seed, no search: the first pose IS the origin of the map frame. The model
            // still needs SOME polygon to exist (the viewer, the SDF paths Estimate mode never
            // optimises against): the scan's oriented bounding box, until the walls replace it.
            std::vector<Eigen::Vector3f> pts;
            {
                const auto& [lidar_from_buffer] = run_ctx_.high_lidar_buffer->read_last();
                if (lidar_from_buffer.has_value()) pts = lidar_from_buffer->first;
            }
            if (pts.empty()) return false;
            // ⚠ THE FALLBACK BOX MUST NEVER BE LARGER THAN THE ROOM. initialize_rect() below BIRTHS
            // four wall landmarks on this rectangle, and a seeded wall that lies BEHIND the real one
            // is never hit by a beam again — absence cannot refute what is occluded, and this
            // codebase's rule is that occlusion HOLDS the belief. Those walls are then immortal: they
            // stay in the published cycle for ever, carrying their untouched prior sigma.
            // Measured in Webots room 2 (6.22 x 4.39 m), estimate_obb() failed and the old fixed
            // {+-10, +-10} fallback fired: after 2300 frames the polygon still ran out to x = +-10
            // with sigma 2.665 on one of them, while the room ITSELF was recovered to 6.18 x 3.98 m
            // with sigma 0.021 on its well-observed corners. The shape was not misestimated, it was
            // polluted. So the fallback is the scan's own extent, which by construction cannot sit
            // outside the walls that produced it.
            std::vector<Eigen::Vector2f> placeholder;
            {
                Eigen::Vector2f lo(1e9f, 1e9f), hi(-1e9f, -1e9f);
                for (const auto& q : pts) { const Eigen::Vector2f v(q.x(), q.y()); lo = lo.cwiseMin(v); hi = hi.cwiseMax(v); }
                if ((hi - lo).minCoeff() > 0.2f)
                    placeholder = {{lo.x(), lo.y()}, {hi.x(), lo.y()}, {hi.x(), hi.y()}, {lo.x(), hi.y()}};
                else
                    placeholder = {{-3.f, -3.f}, {3.f, -3.f}, {3.f, 3.f}, {-3.f, 3.f}};
            }
            PointcloudCenterEstimator estimator;
            if (const auto obb = estimator.estimate_obb(pts); obb.has_value())
            {
                const Eigen::Rotation2Df Rb(static_cast<float>(obb->rotation));
                const Eigen::Vector2f c = obb->center.cast<float>();
                const float hw = static_cast<float>(obb->width) * 0.5f, hh = static_cast<float>(obb->height) * 0.5f;
                if (hw > 0.1f and hh > 0.1f)
                    placeholder = {c + Rb * Eigen::Vector2f(-hw, -hh), c + Rb * Eigen::Vector2f(hw, -hh),
                                   c + Rb * Eigen::Vector2f(hw, hh),   c + Rb * Eigen::Vector2f(-hw, hh)};
            }
            init_use_polygon_ = false;          // no MODEL corners ⇒ the model-guided corner detector stays off
            init_polygon_vertices_.clear();
            set_polygon_room(placeholder);
            set_robot_pose(0.f, 0.f, 0.f, false);
            // MODEL-FIRST: the OBB rectangle IS the initial room model — the prior on shape and
            // size. Everything after is refinement of its sides plus splice jumps.
            if (wall_map_.walls.empty())
                wall_map_.initialize_rect(placeholder);
            {   // Never let the fallback fire silently again — it is invisible in the map and fatal to it.
                Eigen::Vector2f plo(1e9f, 1e9f), phi(-1e9f, -1e9f);
                for (const auto& v : placeholder) { plo = plo.cwiseMin(v); phi = phi.cwiseMax(v); }
                qInfo() << "[room][wall-slam] Estimate mode: origin = first pose; seed box"
                        << (phi.x() - plo.x()) << "x" << (phi.y() - plo.y()) << "m from"
                        << (estimator.estimate_obb(pts).has_value() ? "the scan's OBB" : "the scan EXTENT (OBB failed)")
                        << "-- walls seeded here must lie INSIDE the room or they can never die.";
            }
            return true;
        }

        Eigen::Vector2f init_xy = Eigen::Vector2f::Zero();
        float init_phi = 0.f;
        bool have_saved_pose = false;

        // Try saved pose first — if available we can initialize the model immediately
        // without needing a lidar scan (lidar is only required for pose estimation).
        if (!seed_pose_file_path_.empty())
        {
            std::ifstream in(seed_pose_file_path_);
            float sx = 0.f, sy = 0.f, st = 0.f;
            if (in.is_open() && (in >> sx >> sy >> st))
            {
                init_xy = Eigen::Vector2f(sx, sy);
                init_phi = st;
                have_saved_pose = true;
            }
            else
            {
                qWarning() << "RoomConcept bootstrap: seed pose not loaded from"
                           << QString::fromStdString(seed_pose_file_path_)
                           << "(file missing or parse error)";
            }
        }

        // Grab a sweep if one is already buffered. Without a saved pose we CANNOT proceed without it
        // (the initial position is estimated from the cloud); with one we can seed immediately, but we
        // still want the points — they are what tells us whether the seed is any good (below).
        std::vector<Eigen::Vector3f> pts;
        {
            const auto& [lidar_from_buffer] = run_ctx_.high_lidar_buffer->read_last();
            if (lidar_from_buffer.has_value())
                pts = lidar_from_buffer->first;
        }

        if (!have_saved_pose)
        {
            if (pts.empty())
                return false;

            PointcloudCenterEstimator estimator;
            Eigen::Vector2d room_center_in_robot = Eigen::Vector2d::Zero();
            if (const auto obb = estimator.estimate_obb(pts); obb.has_value())
            {
                room_center_in_robot = obb->center;
                init_phi = static_cast<float>(obb->rotation);
            }
            else if (const auto c = estimator.estimate(pts); c.has_value())
            {
                room_center_in_robot = c.value();
                init_phi = estimate_orientation_from_points(pts);
            }

            Eigen::Rotation2Df R(init_phi);
            init_xy = -(R * room_center_in_robot.cast<float>());
        }

        if (init_use_polygon_ && init_polygon_vertices_.size() >= 3)
        {
            set_polygon_room(init_polygon_vertices_);
            set_robot_pose(init_xy.x(), init_xy.y(), init_phi, false);
        }
        else
        {
            set_initial_state(init_room_width_, init_room_length_, init_xy.x(), init_xy.y(), init_phi);
        }

        bool used_grid_search = false;
        if (!have_saved_pose)
        {
            search_trigger_ = "seed";
            used_grid_search = grid_search_initial_pose(pts, 0.5f, static_cast<float>(M_PI_4));
            if (!used_grid_search)
                qWarning() << "RoomConcept bootstrap: grid search failed. Keeping estimator-based initialization.";
        }
        else
        {
            // The saved pose is a PRIOR, not certainty: the robot may have been moved while the agent
            // was down, the file may predate a layout/room-frame change, or it may simply be stale.
            // Check that it still explains the scan and fall through to the global search if it does
            // not. Deferred when no sweep has arrived yet (the seeded model is published meanwhile so
            // the viewer/planner have geometry) — validate_seed_pose runs on the first one instead.
            if (pts.empty())
                pending_seed_validation_ = true;
            else
                used_grid_search = !validate_seed_pose(pts);
        }

        // Needs points; on the deferred-validation path there are none yet, and the saved yaw is
        // carried as-is until the first sweep resolves it.
        if (!used_grid_search && !pts.empty())
            resolve_initial_yaw_ambiguity(pts, init_phi);

        return true;
    }

    /// Seeded-pose check. Returns true when no search could run (the seed stands by default), false when
    /// the mixture search ran — whatever it concluded, the yaw is then resolved and the caller must not
    /// run its own yaw-ambiguity pass on top.
    ///
    /// There is no pass/fail bar any more (it was RecoveryLossThreshold). The saved pose is a SEED of the
    /// search: it is polished and scored alongside every structural hypothesis, so it stands exactly when
    /// it is the best-supported mode — and a stale file loses to the mode that explains the scan.
    bool RoomConcept::validate_seed_pose(const std::vector<Eigen::Vector3f>& pts)
    {
        pending_seed_validation_ = false;
        if (pts.empty() or model_ == nullptr or not model_->has_state() or not params.reloc_enabled)
            return true;
        search_trigger_ = "seed_validation";
        if (not grid_search_initial_pose(pts, 0.5f, static_cast<float>(M_PI_4)))
        {
            qWarning() << "[RoomConcept] Seed validation: the search could not run; keeping the saved pose.";
            return true;
        }
        return false;
    }

    float RoomConcept::find_best_initial_orientation(const std::vector<Eigen::Vector3f>& lidar_points,
                                                        float x, float y, float base_phi)
    {
        if (model_ == nullptr || lidar_points.empty())
            return base_phi;

        // Test 4 orientations: base, +90°, +180°, +270° (covers symmetries)
        const std::vector<float> angle_offsets = {0.0f, M_PI_2, M_PI, 3.0f * M_PI_2};

        // Also test mirrored positions (x, y) and (-x, y) with all rotations
        // This handles axis-mirroring ambiguity
        const std::vector<std::pair<float, float>> position_variants = {
            {x, y},      // Original
            {-x, y},     // Mirror X
            {x, -y},     // Mirror Y
            {-x, -y}     // Mirror both
        };

        // Subsample points for faster evaluation
        std::vector<Eigen::Vector3f> sample_points;
        const int max_samples = params.orientation_search_max_samples;
        const int stride = std::max(1, static_cast<int>(lidar_points.size()) / max_samples);
        for (size_t i = 0; i < lidar_points.size(); i += stride)
            sample_points.push_back(lidar_points[i]);

        const ScanPoints& scan_pts = sample_points;

        float best_phi = base_phi;
        float best_x = x;
        float best_y = y;
        float best_loss = std::numeric_limits<float>::infinity();

        // Test all combinations of position variants and angle offsets
        for (const auto& [test_x, test_y] : position_variants)
        {
            for (float offset : angle_offsets)
            {
                float test_phi = base_phi + offset;
                // Normalize to [-pi, pi]
                while (test_phi > M_PI) test_phi -= 2.0f * M_PI;
                while (test_phi < -M_PI) test_phi += 2.0f * M_PI;

                // Temporarily set the pose
                model_->robot_pos = {test_x, test_y};
                model_->robot_theta = test_phi;

                // Evaluate SDF loss
                const Eigen::VectorXf sdf_vals = model_->sdf(scan_pts);
                const float loss = sdf_vals.size() > 0 ? sdf_vals.squaredNorm() / static_cast<float>(sdf_vals.size())
                                                       : std::numeric_limits<float>::infinity();

                if (loss < best_loss)
                {
                    best_loss = loss;
                    best_phi = test_phi;
                    best_x = test_x;
                    best_y = test_y;
                }
            }
        }

        // Set the best position found
        model_->robot_pos = {best_x, best_y};

        return best_phi;
    }

    float RoomConcept::evaluate_pose_fit(const std::vector<Eigen::Vector3f>& lidar_points,
                                                  int max_samples) const
    {
        if (model_ == nullptr || lidar_points.empty())
            return std::numeric_limits<float>::infinity();

        std::vector<Eigen::Vector3f> sampled;
        const int cap = std::max(1, max_samples);
        if (static_cast<int>(lidar_points.size()) > cap)
        {
            const int stride = static_cast<int>(lidar_points.size()) / cap;
            sampled.reserve(cap);
            for (size_t i = 0; i < lidar_points.size(); i += stride)
                sampled.push_back(lidar_points[i]);
        }
        else
        {
            sampled = lidar_points;
        }

        const ScanPoints& scan_pts = sampled;
        // Median, not mean: this result is compared against recovery_loss_threshold (validate_seed_pose)
        // and against itself across yaw hypotheses, and every other fit number in this class is a
        // median. Mean |SDF| runs high wherever furniture the polygon does not model is in view, which
        // made the seed check stricter than the bar it was measured against.
        return median_abs_sdf(model_->sdf(scan_pts));
    }

} // namespace rc
