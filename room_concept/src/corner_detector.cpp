#include "corner_detector.h"
#include <cmath>

namespace rc {

// ---------------------------------------------------------------------------
//  set_model_corners — keep only real corners and store wall edge directions
// ---------------------------------------------------------------------------
void CornerDetector::set_model_corners(const std::vector<Eigen::Vector2f>& polygon_vertices)
{
    model_corners_.clear();
    const int N = static_cast<int>(polygon_vertices.size());
    if (N < 3) return;

    for (int i = 0; i < N; ++i)
    {
        const Eigen::Vector2f& prev = polygon_vertices[(i + N - 1) % N];
        const Eigen::Vector2f& curr = polygon_vertices[i];
        const Eigen::Vector2f& next = polygon_vertices[(i + 1) % N];

        const Eigen::Vector2f d1 = (prev - curr).normalized();
        const Eigen::Vector2f d2 = (next - curr).normalized();
        const float dot = std::clamp(d1.dot(d2), -1.0f, 1.0f);
        const float angle_deg = std::acos(dot) * 180.0f / static_cast<float>(M_PI);

        if (angle_deg >= params_.min_corner_angle && angle_deg <= params_.max_corner_angle)
        {
            ModelCorner mc;
            mc.position       = curr;
            mc.edge_in_dir    = (curr - prev).normalized();  // wall arriving at corner
            mc.edge_out_dir   = (next - curr).normalized();  // wall leaving corner
            mc.convexity_sign = mc.edge_in_dir.x() * mc.edge_out_dir.y()
                              - mc.edge_in_dir.y() * mc.edge_out_dir.x();
            mc.wall_in_length  = (curr - prev).norm();
            mc.wall_out_length = (next - curr).norm();
            mc.original_index = i;
            model_corners_.push_back(mc);
        }
    }
}

// ---------------------------------------------------------------------------
//  detect — model-guided partitioning + PCA line fit + intersect
// ---------------------------------------------------------------------------
//  solve_hungarian — minimum-cost bipartite assignment (Kuhn-Munkres, O(N³))
//
//  Returns assignment[row] = col, or -1 if that row remains unassigned.
//  Infeasible pairs must be encoded as INFEASIBLE (defined below).
//  Rows with no feasible column end up unassigned (-1).
// ---------------------------------------------------------------------------
static constexpr float INFEASIBLE = 1e9f;

static std::vector<int> solve_hungarian(
        const std::vector<std::vector<float>>& cost, int R, int C)
{
    if (R == 0 || C == 0) return std::vector<int>(R, -1);

    const int N = std::max(R, C);   // pad to square

    auto cell = [&](int i, int j) -> float {
        return (i < R && j < C) ? cost[i][j] : INFEASIBLE;
    };

    // u[i]/v[j] — row/column potentials (1-indexed internally)
    // p[j]      — row currently matched to column j  (0 = free)
    // way[j]    — predecessor column on the shortest-path tree
    std::vector<float> u(N + 1, 0.f), v(N + 1, 0.f);
    std::vector<int>   p(N + 1, 0),   way(N + 1, 0);

    for (int i = 1; i <= N; ++i)
    {
        p[0] = i;
        int j0 = 0;
        std::vector<float> minv(N + 1, INFEASIBLE);
        std::vector<bool>  used(N + 1, false);

        do {
            used[j0] = true;
            const int i0 = p[j0];
            int   j1    = 1;           // safe fallback; always overwritten below
            float delta = INFEASIBLE;
            for (int j = 1; j <= N; ++j)
            {
                if (!used[j])
                {
                    const float c = cell(i0 - 1, j - 1) - u[i0] - v[j];
                    if (c < minv[j]) { minv[j] = c; way[j] = j0; }
                    if (minv[j] < delta) { delta = minv[j]; j1 = j; }
                }
            }
            for (int j = 0; j <= N; ++j)
            {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else          { minv[j] -= delta; }
            }
            j0 = j1;
        } while (p[j0] != 0);

        do {
            const int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }

    // Extract: p[j] = 1-indexed row assigned to column j
    // Reject padded or infeasible pairs.
    std::vector<int> assignment(R, -1);
    for (int j = 1; j <= C; ++j)
    {
        const int r = p[j] - 1;   // 0-based
        if (r >= 0 && r < R && cost[r][j - 1] < INFEASIBLE * 0.5f)
            assignment[r] = j - 1;
    }
    return assignment;
}

// ---------------------------------------------------------------------------
//  detect — two-phase pipeline:
//    Phase 1: per-model-corner PCA candidate generation
//    Phase 2: Hungarian assignment — prevents two model corners from claiming
//             the same physical detection when search discs overlap
// ---------------------------------------------------------------------------
CornerDetector::DetectionResult CornerDetector::detect(
        const std::vector<Eigen::Vector3f>& lidar_points,
        float robot_x, float robot_y, float robot_theta,
        float max_range) const
{
    DetectionResult result;
    if (model_corners_.empty() || lidar_points.empty())
        return result;

    // Build 2D lidar points in robot frame (drop z)
    std::vector<Eigen::Vector2f> pts2d;
    pts2d.reserve(lidar_points.size());
    for (const auto& p : lidar_points)
        pts2d.emplace_back(p.x(), p.y());

    // Rotation world→robot:  R(-θ)
    const float cos_t = std::cos(robot_theta);
    const float sin_t = std::sin(robot_theta);
    const Eigen::Vector2f t_world(robot_x, robot_y);

    auto to_robot = [&](const Eigen::Vector2f& v) -> Eigen::Vector2f {
        return {cos_t * v.x() + sin_t * v.y(),
               -sin_t * v.x() + cos_t * v.y()};
    };

    const float search_r2 = params_.search_radius * params_.search_radius;
    const float max_range2 = max_range * max_range;

    // ── Phase 1: per-model-corner detection attempt ───────────────────────
    // We record every in-FOV corner's predicted position for the cost matrix,
    // then collect all candidates that survive all quality filters.

    struct FOVCorner {
        const ModelCorner* mc;
        Eigen::Vector2f    predicted;   // robot-frame expected position
    };
    std::vector<FOVCorner>   fov_corners;
    std::vector<CornerMatch> candidates;

    for (const auto& mc : model_corners_)
    {
        const Eigen::Vector2f dw = mc.position - t_world;
        if (dw.squaredNorm() > max_range2)
            continue;

        const Eigen::Vector2f predicted = to_robot(dw);
        result.corners_in_fov++;
        fov_corners.push_back({&mc, predicted});

        const Eigen::Vector2f dir_in  = to_robot(mc.edge_in_dir);
        const Eigen::Vector2f dir_out = to_robot(mc.edge_out_dir);
        const Eigen::Vector2f normal_in (-dir_in.y(),  dir_in.x());
        const Eigen::Vector2f normal_out(-dir_out.y(), dir_out.x());

        // Gather neighbourhood, clipped to each wall's actual length
        // (+ 0.2 m slack for localisation error).
        std::vector<Eigen::Vector2f> group_in, group_out;
        group_in.reserve(128);
        group_out.reserve(128);
        const float in_limit  = mc.wall_in_length  + 0.2f;
        const float out_limit = mc.wall_out_length + 0.2f;
        const float wall_band = std::max(0.12f, 2.f * params_.ransac_threshold);

        for (const auto& p : pts2d)
        {
            const Eigen::Vector2f d = p - predicted;
            if (d.squaredNorm() > search_r2)
                continue;

            const float dist_to_in  = std::abs(normal_in.dot(d));
            const float dist_to_out = std::abs(normal_out.dot(d));

            const float along_in = -dir_in.dot(d);
            const float along_out = dir_out.dot(d);
            const bool in_candidate = (along_in >= -0.2f && along_in <= in_limit &&
                                       dist_to_in <= wall_band);
            const bool out_candidate = (along_out >= -0.2f && along_out <= out_limit &&
                                        dist_to_out <= wall_band);

            if (in_candidate && (!out_candidate || dist_to_in < dist_to_out))
                group_in.push_back(p);
            else if (out_candidate)
                group_out.push_back(p);
        }

        if (static_cast<int>(group_in.size())  < params_.min_points_per_line ||
            static_cast<int>(group_out.size()) < params_.min_points_per_line)
            continue;

        auto line_in  = fit_line_pca(group_in,  params_.min_points_per_line);
        auto line_out = fit_line_pca(group_out, params_.min_points_per_line);
        if (!line_in || !line_out)
            continue;

        result.corners_detected++;

        float angle_deg = 0.f;
        auto intersection = intersect(*line_in, *line_out, &angle_deg);
        if (!intersection)
            continue;

        if (angle_deg < params_.min_corner_angle || angle_deg > params_.max_corner_angle)
            { result.rej_angle++; continue; }

        const float isect_dist = (*intersection - predicted).norm();
        if (isect_dist > params_.max_match_distance)
            { result.rej_dist++; continue; }

        const float raw_dot_in  = line_in->direction().dot(dir_in);
        const float raw_dot_out = line_out->direction().dot(dir_out);
        {
            const float cos_thresh = std::cos(params_.max_orientation_dev
                                              * static_cast<float>(M_PI) / 180.f);
            if (std::abs(raw_dot_in) < cos_thresh || std::abs(raw_dot_out) < cos_thresh)
                { result.rej_orient++; continue; }
        }
        const Eigen::Vector2f ori_in  = (raw_dot_in  >= 0.f ? 1.f : -1.f) * line_in->direction();
        const Eigen::Vector2f ori_out = (raw_dot_out >= 0.f ? 1.f : -1.f) * line_out->direction();
        {
            const float detected_cross = ori_in.x() * ori_out.y() - ori_in.y() * ori_out.x();
            if (mc.convexity_sign * detected_cross < 0.50f * std::abs(mc.convexity_sign))
                { result.rej_convex++; continue; }
        }

        const float sigma = params_.ransac_threshold;
        CornerMatch cand;
        cand.detected    = *intersection;
        cand.predicted   = predicted;          // overwritten by Phase 2 winner
        cand.model_world = mc.position;        // overwritten by Phase 2 winner
        cand.model_index = mc.original_index;  // overwritten by Phase 2 winner
        cand.distance    = (*intersection - predicted).norm();
        cand.angle_deg   = angle_deg;
        cand.covariance  = sigma * sigma *
            (line_in->normal * line_in->normal.transpose() +
             line_out->normal * line_out->normal.transpose());
        candidates.push_back(cand);
    }

    if (candidates.empty())
        return result;

    // ── Phase 2: Hungarian assignment ────────────────────────────────────
    // cost[r][c] = distance from candidate c's detected position to fov
    // corner r's predicted position, or INFEASIBLE if beyond max_match_distance.
    // The solver guarantees each physical detection is claimed by at most one
    // model corner, choosing the globally cheapest 1-to-1 pairing.

    const int R = static_cast<int>(fov_corners.size());
    const int C = static_cast<int>(candidates.size());

    std::vector<std::vector<float>> cost(R, std::vector<float>(C, INFEASIBLE));
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < C; ++c)
        {
            const float d = (candidates[c].detected - fov_corners[r].predicted).norm();
            if (d <= params_.max_match_distance)
                cost[r][c] = d;
        }

    const std::vector<int> assignment = solve_hungarian(cost, R, C);

    for (int r = 0; r < R; ++r)
    {
        const int c = assignment[r];
        if (c < 0) continue;

        // Stamp model identity with the winning FOV corner
        CornerMatch m    = candidates[c];
        m.model_index    = fov_corners[r].mc->original_index;
        m.predicted      = fov_corners[r].predicted;
        m.model_world    = fov_corners[r].mc->position;
        m.distance       = (m.detected - m.predicted).norm();
        result.matches.push_back(m);
    }
    result.corners_accepted = static_cast<int>(result.matches.size());
    // Candidates that passed every quality gate but lost the 1-to-1 assignment.
    result.rej_unassigned = C - result.corners_accepted;

    return result;
}

// ---------------------------------------------------------------------------
//  fit_line_pca — least-squares line fit via PCA
// ---------------------------------------------------------------------------
std::optional<CornerDetector::Line2D> CornerDetector::fit_line_pca(
        const std::vector<Eigen::Vector2f>& pts, int min_points)
{
    if (static_cast<int>(pts.size()) < min_points)
        return std::nullopt;

    Eigen::Vector2f centroid = Eigen::Vector2f::Zero();
    for (const auto& p : pts)
        centroid += p;
    centroid /= static_cast<float>(pts.size());

    Eigen::Matrix2f scatter = Eigen::Matrix2f::Zero();
    for (const auto& p : pts)
    {
        Eigen::Vector2f dp = p - centroid;
        scatter += dp * dp.transpose();
    }

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> eig(scatter);
    Line2D line;
    line.normal = eig.eigenvectors().col(0);  // smallest eigenvalue = line normal
    line.d = line.normal.dot(centroid);
    return line;
}

// ---------------------------------------------------------------------------
//  intersect
// ---------------------------------------------------------------------------
std::optional<Eigen::Vector2f> CornerDetector::intersect(
        const Line2D& a, const Line2D& b, float* angle_deg)
{
    const float det = a.normal.x() * b.normal.y() - a.normal.y() * b.normal.x();
    if (std::abs(det) < 1e-6f)
        return std::nullopt;

    const float x = (b.normal.y() * a.d - a.normal.y() * b.d) / det;
    const float y = (a.normal.x() * b.d - b.normal.x() * a.d) / det;

    if (angle_deg)
    {
        // Angle between the two wall directions (= 180° - angle between normals)
        const float ndot = std::abs(a.normal.dot(b.normal));
        const float clamped = std::min(1.0f, ndot);
        *angle_deg = 180.0f - std::acos(clamped) * 180.0f / static_cast<float>(M_PI);
    }

    return Eigen::Vector2f(x, y);
}

} // namespace rc
