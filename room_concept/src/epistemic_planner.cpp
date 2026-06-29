#include "epistemic_planner.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>

namespace rc
{

EpistemicPlanner::EpistemicPlanner() : params{} {}
EpistemicPlanner::EpistemicPlanner(Params p) : params(std::move(p)) {}

// ---------------------------------------------------------------------------
// State setters
// ---------------------------------------------------------------------------
void EpistemicPlanner::set_room_bounds(const Eigen::Vector2f& min_corner,
                                      const Eigen::Vector2f& max_corner)
{
    room_min_ = min_corner;
    room_max_ = max_corner;
    room_bounds_set_ = true;
    grid_dirty_ = true;
    if (!visit_grid_.initialized)
        visit_grid_.init(room_min_, room_max_, params.ior_cell_size);
    // occupancy belief is configured from set_room_polygon() (called right after this at
    // room creation) so its in-room mask uses the actual polygon and isn't reset each cycle.
}

void EpistemicPlanner::set_room_polygon(const std::vector<Eigen::Vector2f>& vertices)
{
    room_corners_ = vertices;
    grid_dirty_ = true;
    maybe_configure_occupancy();
}

// (Re)build the occupancy belief grid once the room extent is known. The polygon
// (room_corners_) bounds the in-room cells; harmless to call repeatedly.
void EpistemicPlanner::maybe_configure_occupancy()
{
    if (!room_bounds_set_) return;
    OccupancyBelief::Params op;
    op.cell_size    = params.map_cell_size;
    op.sensor_range = params.map_sensor_range;
    op.n_rays       = params.map_rays;
    op.free_logodds = params.map_free_logodds;
    occ_.configure(room_min_, room_max_, room_corners_, op);
}

void EpistemicPlanner::set_robot_state(const Eigen::Affine2f& pose,
                                      const Eigen::Matrix3f& covariance)
{
    robot_pose_ = pose;
    robot_cov_ = covariance;
    robot_state_set_ = true;
    // Fold what the robot actually sees from here into the occupancy belief — this is
    // the per-cycle "observation" that drives the entropy down over the explored area.
    if (occ_.configured())
        occ_.observe_from(robot_pose_.translation());
}

void EpistemicPlanner::set_robot_footprint(float width_m, float length_m)
{
    const float footprint_radius = 0.5f * std::hypot(width_m, length_m);
    if (std::abs(robot_footprint_radius_ - footprint_radius) > 1e-6f)
    {
        robot_footprint_radius_ = footprint_radius;
        grid_dirty_ = true;
    }
}

void EpistemicPlanner::set_obstacle_footprints(std::vector<ObstacleFootprint> footprints)
{
    obstacle_footprints_ = std::move(footprints);

    // Mirror them as circle occluders so the occupancy belief casts object shadows
    // (unobserved cells behind objects the robot must move around to resolve).
    std::vector<OccupancyBelief::Obstacle> occ_obs;
    occ_obs.reserve(obstacle_footprints_.size());
    for (const auto& f : obstacle_footprints_)
        occ_obs.push_back({f.center, std::hypot(f.half_w, f.half_d)});
    occ_.set_obstacles(std::move(occ_obs));
}

// ===========================================================================
// Candidate generation
// ===========================================================================
std::vector<Eigen::Vector2f> EpistemicPlanner::generate_candidates() const
{
    if (!room_bounds_set_)
        return {};

    // Rebuild static grid cache when room geometry changes
    if (grid_dirty_)
    {
        cached_grid_.clear();
        const float res = params.grid_resolution;
        const float wall_margin = std::max(params.target_wall_margin, robot_footprint_radius_);
        const float wm2 = wall_margin * wall_margin;

        for (float x = room_min_.x() + res * 0.5f; x < room_max_.x(); x += res)
        {
            for (float y = room_min_.y() + res * 0.5f; y < room_max_.y(); y += res)
            {
                const Eigen::Vector2f p{x, y};
                if (!room_corners_.empty() &&
                    !corner_visibility::point_in_polygon(p, room_corners_))
                    continue;
                if (!room_corners_.empty())
                {
                    bool too_close = false;
                    for (std::size_t i = 0; i < room_corners_.size(); ++i)
                    {
                        const auto& a = room_corners_[i];
                        const auto& b = room_corners_[(i + 1) % room_corners_.size()];
                        const Eigen::Vector2f ab = b - a;
                        const float t = std::clamp((p - a).dot(ab) / ab.squaredNorm(), 0.f, 1.f);
                        if ((p - (a + t * ab)).squaredNorm() < wm2) { too_close = true; break; }
                    }
                    if (too_close) continue;
                }
                cached_grid_.emplace_back(p);
            }
        }
        grid_dirty_ = false;
    }

    // Filter cached grid by min_distance from robot (dynamic per cycle)
    const float min_d2 = params.min_distance * params.min_distance;
    std::vector<Eigen::Vector2f> candidates;
    candidates.reserve(std::min(static_cast<int>(cached_grid_.size()), params.max_candidates));

    for (const auto& p : cached_grid_)
    {
        if ((p - robot_pos()).squaredNorm() < min_d2)
            continue;

        // Reject candidates that fall inside (or too close to) any object/obstacle footprint.
        // Clearance = robot_footprint_radius_ so the robot body won't overlap the object.
        bool blocked = false;
        for (const auto& obs : obstacle_footprints_)
        {
            const Eigen::Vector2f d = p - obs.center;
            const float c = std::cos(-obs.yaw);
            const float s = std::sin(-obs.yaw);
            const float lx = c * d.x() - s * d.y();
            const float ly = s * d.x() + c * d.y();
            if (std::abs(lx) < obs.half_w + robot_footprint_radius_ &&
                std::abs(ly) < obs.half_d + robot_footprint_radius_)
            {
                blocked = true;
                break;
            }
        }
        if (blocked) continue;

        candidates.emplace_back(p);
        if (static_cast<int>(candidates.size()) >= params.max_candidates)
            break;
    }
    return candidates;
}

// ===========================================================================
// Angular dominance check
// ===========================================================================
bool EpistemicPlanner::is_angular_dominated() const
{
    const float sigma2_theta = robot_cov_(2, 2);
    const float max_pos = std::max(robot_cov_(0, 0), robot_cov_(1, 1));
    if (max_pos < 1e-9f) return false;
    return (sigma2_theta / max_pos) > params.angular_dominance_ratio;
}

// ===========================================================================
// FIM-based information-gain score
// ===========================================================================
float EpistemicPlanner::score_fim_gain(const Eigen::Vector2f& candidate,
                                      const Eigen::Matrix3f& prior_precision) const
{
    if (room_corners_.empty()) return 0.f;

    Eigen::Matrix3f fim = Eigen::Matrix3f::Zero();

    // (1) Corner landmarks — full 2D fixes (rank-2 each).
    if (const auto vis = corner_visibility::visible_corners(candidate, room_corners_,
                                                            params.fim_max_range);
        !vis.empty())
    {
        const Eigen::Vector2f dir = candidate - robot_pos();
        const float heading = (dir.squaredNorm() > 1e-4f)   // heading-independent for JᵀJ; kept for API
            ? std::atan2(-dir.x(), dir.y()) : robot_theta();
        fim += corner_visibility::corner_fim(candidate, heading, vis,
                                             room_corners_, params.fim_corner_sigma);
    }

    // (2) Wall surfaces — the localizer's actual point-to-wall SDF likelihood over a
    //     360° scan (rank-1, perpendicular-only). Captures the parallel-wall ambiguity.
    if (params.fim_use_walls)
        fim += corner_visibility::wall_fim(candidate, room_corners_, params.fim_wall_sigma,
                                           params.fim_max_range, params.fim_wall_rays,
                                           params.fim_wall_incidence_weight,
                                           params.fim_wall_incidence_min);

    if (fim.isZero(1e-12f)) return 0.f;
    // ΔH = ½·log det(I + Y_prior⁻¹·I_pred) in NATS — the expected reduction in DIFFERENTIAL
    // ENTROPY of the pose-belief Gaussian (the common EFE currency shared with aff_table).
    // d_optimality_gain returns the full log-det ( = 2·ΔH ), so apply ½. Mathematically ≥0 for a
    // PD prior + PSD FIM; clamp guards float roundoff so epistemic_gain is a non-negative benefit.
    return std::max(0.f, 0.5f * corner_visibility::d_optimality_gain(prior_precision, fim));
}

Eigen::Matrix3f EpistemicPlanner::current_prior_precision() const
{
    const Eigen::Matrix3f reg_cov = robot_cov_ + 1e-6f * Eigen::Matrix3f::Identity();
    // Floor Y_prior eigenvalues at a small prior precision. A degenerate/lost prior (huge
    // covariance → near-zero precision → det→0) then yields a LARGE-BUT-FINITE ΔH instead of
    // ∞/NaN: log det is bounded below by 3·log(floor), not by a determinant clamp artifact.
    // Equivalent to capping the assumed max covariance at 1/floor. Negligible once localized.
    return reg_cov.inverse() + params.fim_prior_precision_floor * Eigen::Matrix3f::Identity();
}

float EpistemicPlanner::live_epistemic_gain(const Eigen::Vector2f& viewpoint) const
{
    if (!robot_state_set_ || room_corners_.empty()) return 0.f;
    return score_fim_gain(viewpoint, current_prior_precision());
}

float EpistemicPlanner::live_total_epistemic_gain(const Eigen::Vector2f& viewpoint) const
{
    const float fim_gain = live_epistemic_gain(viewpoint);
    // Non-saturating exploration term: expected occupancy-entropy resolved by a scan from this
    // vantage. Same combination as evaluate_targets (t.score), so the advertised gain tracks the
    // ranking. 0 once the cell is covered → the total still decays to 0 when the room is DONE.
    const float map_gain = (params.w_map > 0.f && occ_.configured())
                           ? occ_.expected_info_gain(viewpoint) : 0.f;
    return fim_gain + params.w_map * map_gain;
}

// ===========================================================================
// evaluate_targets — FIM-based D-optimality scoring + cell score cache
// ===========================================================================
std::vector<EpistemicPlanner::Target> EpistemicPlanner::evaluate_targets() const
{
    // We update mutable cell_scores_ here
    auto& self = const_cast<EpistemicPlanner&>(*this);

    if (!room_bounds_set_ || !robot_state_set_)
    {
        self.cell_scores_.clear();
        self.ior_cells_.clear();
        return {};
    }

    if (is_angular_dominated())
    {
        Target rot;
        rot.position = robot_pos();
        rot.distance = 0.f;
        rot.rotate_in_place = true;
        rot.score = robot_cov_(2, 2);
        // Advertise the same grounded ΔH currency for the recovery action: rotating in place
        // sweeps a 360° scan from the current vantage, so its expected pose-entropy reduction is
        // the FIM gain here. Without this the heading-recovery action — needed exactly when the
        // robot is most lost — would publish gain=0 and lose every cross-affordance EFE compare.
        rot.eigenvector_score = live_epistemic_gain(robot_pos());
        self.cell_scores_.clear();
        self.ior_cells_.clear();
        return {rot};
    }

    const auto candidates = generate_candidates();
    if (candidates.empty())
    {
        self.cell_scores_.clear();
        self.ior_cells_.clear();
        return {};
    }

    const Eigen::Matrix3f prior_precision = current_prior_precision();
    const auto now = std::chrono::steady_clock::now();

    std::vector<Target> targets;
    targets.reserve(candidates.size());

    for (const auto& pos : candidates)
    {
        Target t;
        t.position = pos;
        t.distance = (pos - robot_pos()).norm();
        const float fim_gain = score_fim_gain(pos, prior_precision);

        // Update cell's running FIM average
        self.visit_grid_.update_fim(pos, fim_gain);

        // IoR: staleness of the target cell ∈ [0,1], 0=just visited, 1=fully stale
        const float staleness = visit_grid_.staleness(pos, params.ior_decay_time, now);

        // Path staleness: mean staleness of intermediate cells along the straight-line
        // path from robot to target. High = path goes through unexplored territory.
        const auto rpos = robot_pos();
        const int n_path = std::max(3, static_cast<int>(t.distance / params.ior_cell_size));
        float path_sum = 0.f;
        for (int s = 1; s < n_path; ++s)
        {
            const float alpha = static_cast<float>(s) / n_path;
            path_sum += visit_grid_.staleness(rpos + alpha * (pos - rpos), params.ior_decay_time, now);
        }
        const float path_staleness = path_sum / static_cast<float>(std::max(1, n_path - 1));  // ∈ [0,1]

        // Route staleness: blend of target cell and path cells.
        // w_path_interest=0 → target staleness only (old behaviour)
        // w_path_interest=1 → path staleness only
        // Use as multiplicative IoR suppressor so a visited path to an unvisited target
        // is penalised by the same power as a visited target, not just a small additive nudge.
        const float route_staleness = (1.f - params.w_path_interest) * staleness
                                    + params.w_path_interest * path_staleness;
        const float ior_suppressor = std::pow(route_staleness, params.w_ior);

        const float bonus = 1.f + params.w_exploration * t.distance;

        // Expected occupancy-entropy resolved by a scan from this vantage (bits). This is
        // the NON-saturating epistemic term: 0 once a cell is covered, regardless of how
        // well the pose is known. Additive with the (saturating) pose-FIM term — both are
        // information. w_map=0 ⇒ exactly the legacy IoR behaviour.
        const float map_gain = (params.w_map > 0.f && occ_.configured())
                               ? occ_.expected_info_gain(pos) : 0.f;

        // 1e-6 floor: when both terms vanish, tie-break by FIM gain.
        t.score = (fim_gain * ior_suppressor + params.w_map * map_gain) * bonus + 1e-6f;
        t.eigenvector_score = fim_gain;
        targets.push_back(t);
    }

    std::sort(targets.begin(), targets.end(),
              [](const Target& a, const Target& b) { return a.score > b.score; });

    // Build cell score + IoR freshness caches for visualisation (use grid cells)
    if (visit_grid_.initialized)
    {
        self.cell_scores_.clear();
        self.cell_scores_.reserve(visit_grid_.cells.size());
        self.ior_cells_.clear();
        for (int i = 0; i < static_cast<int>(visit_grid_.cells.size()); ++i)
        {
            const auto center = visit_grid_.cell_center(i);
            // Only include cells inside the room polygon
            if (!room_corners_.empty() &&
                !corner_visibility::point_in_polygon(center, room_corners_))
                continue;
            const auto& cell = visit_grid_.cells[i];
            const float stale = visit_grid_.staleness(center, params.ior_decay_time, now);
            // Heatmap = pose-info (FIM × staleness) + unresolved map entropy (the fog of
            // unexplored interior). Cheap per-cell entropy proxy of the candidate map term.
            const float combined = cell.fim_gain * std::pow(stale, params.w_ior)
                                 + params.w_map * occ_.entropy_at(center);
            self.cell_scores_.push_back({center, combined});
            // IoR freshness: 1=just visited, 0=stale/never; only store meaningful cells
            const float freshness = 1.f - stale;
            self.ior_cells_.push_back({center, freshness});  // include all: 0=dark fog, 1=bright visited
        }
    }

    return targets;
}

std::optional<EpistemicPlanner::Target> EpistemicPlanner::select_target()
{
    auto targets = evaluate_targets();
    if (targets.empty())
        return std::nullopt;

    // Rotate-in-place: no randomness needed
    if (targets.front().rotate_in_place)
        return targets.front();

    // Greedy argmax: targets are already sorted descending by score.
    // The highest-scored candidate maximises FIM gain under IoR suppression —
    // i.e. "the spot that reduces most uncertainty that we haven't visited recently".
    return targets.front();
}

// ===========================================================================
// mark_and_refresh — stamps current robot position + refreshes overlay.
// Used when the full update_target() path is skipped during navigation.
void EpistemicPlanner::mark_and_refresh()
{
    if (!robot_state_set_ || !visit_grid_.initialized) return;
    visit_grid_.mark_visited_with_falloff(robot_pos(), params.ior_path_radius, params.ior_decay_time);
    refresh_ior_overlay();
}

// refresh_ior_overlay — rebuilds ior_cells_ from the live visit grid.
// Cheap: no FIM computation.  Called every cycle so the path overlay
// reflects the robot's current position without waiting for target selection.
// ===========================================================================
void EpistemicPlanner::refresh_ior_overlay()
{
    if (!visit_grid_.initialized || !room_bounds_set_) return;
    const auto now = std::chrono::steady_clock::now();
    ior_cells_.clear();
    for (int i = 0; i < static_cast<int>(visit_grid_.cells.size()); ++i)
    {
        const auto center = visit_grid_.cell_center(i);
        if (!room_corners_.empty() &&
            !corner_visibility::point_in_polygon(center, room_corners_))
            continue;
        const float stale = visit_grid_.staleness(center, params.ior_decay_time, now);
        const float freshness = 1.f - stale;
        ior_cells_.push_back({center, freshness});  // include all: 0=dark fog, 1=bright visited
    }
}

// ===========================================================================
// update_target — handles dwell, arrival, and fresh selection
// ===========================================================================
std::optional<EpistemicPlanner::Target> EpistemicPlanner::update_target()
{
    // ---- Dwell check ----
    if (dwelling_)
    {
        if (std::chrono::steady_clock::now() < dwell_until_)
            return current_target_;   // still dwelling → keep current target
        dwelling_ = false;
        current_target_.reset();
    }

    // ---- Update visit grid: mark robot position with neighbourhood falloff ----
    visit_grid_.mark_visited_with_falloff(robot_pos(), params.ior_path_radius, params.ior_decay_time);

    // ---- Refresh IoR overlay every cycle so the viewer shows live path ----
    refresh_ior_overlay();

    // ---- Arrival check ----
    if (current_target_.has_value() && !current_target_->rotate_in_place)
    {
        const float dist = (current_target_->position - robot_pos()).norm();
        if (dist < params.arrival_distance)
        {
            visit_grid_.mark_visited_with_falloff(current_target_->position,
                                                  1.5f * params.ior_cell_size,
                                                  params.ior_decay_time);
            dwelling_ = true;
            dwell_until_ = std::chrono::steady_clock::now()
                         + std::chrono::milliseconds(static_cast<int>(params.dwell_time * 1000.f));
            return current_target_;
        }
    }

    // ---- Select new target if needed ----
    if (!current_target_.has_value())
    {
        auto target_opt = select_target();
        if (!target_opt)
            return std::nullopt;
        current_target_ = *target_opt;
    }

    return current_target_;
}

} // namespace rc
