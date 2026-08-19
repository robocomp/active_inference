#include "epistemic_planner.h"
#include <fstream>
#include <locale>
#include <format>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <print>

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
    // Same reason as set_room_polygon: this is called every cycle, and only a real change to the
    // bounds can invalidate the candidate grid or the observability mask.
    const bool changed = not room_bounds_set_
                      or (room_min_ - min_corner).squaredNorm() > 1e-8f
                      or (room_max_ - max_corner).squaredNorm() > 1e-8f;
    room_min_ = min_corner;
    room_max_ = max_corner;
    room_bounds_set_ = true;
    if (changed) { grid_dirty_ = true; mask_dirty_ = true; }
    if (!visit_grid_.initialized)
        visit_grid_.init(room_min_, room_max_, params.ior_cell_size);
}

void EpistemicPlanner::set_room_polygon(const std::vector<Eigen::Vector2f>& vertices)
{
    // Only a genuinely different contour invalidates the caches — this is called every cycle from
    // the graph, and rebuilding the candidate grid + observability mask on each one is pure waste.
    if (room_corners_.size() == vertices.size() and
        std::equal(room_corners_.begin(), room_corners_.end(), vertices.begin(),
                   [](const Eigen::Vector2f& a, const Eigen::Vector2f& b)
                   { return (a - b).squaredNorm() < 1e-8f; }))
        return;
    room_corners_ = vertices;
    grid_dirty_ = true;
    mask_dirty_ = true;
}

// ---------------------------------------------------------------------------
// Observability mask — see the header for why an unmasked grid poisons the route term.
// ---------------------------------------------------------------------------
void EpistemicPlanner::refresh_observable_mask()
{
    if (!visit_grid_.initialized) return;
    if (room_corners_.empty())
    {
        // No contour yet ⇒ we cannot say what is wall and what is room. Leave the mask empty so
        // is_observable() passes everything through, exactly as before this existed.
        visit_grid_.observable.clear();
        dbg_unobservable_ = 0;
        mask_dirty_ = false;
        return;
    }

    const float wall_margin = std::max(params.target_wall_margin, robot_footprint_radius_);
    // The robot marks every cell within ior_path_radius of wherever it stands, so a cell is
    // reachable-by-observation as long as SOME legal standing position (>= wall_margin from every
    // wall) is within that radius of it. Negative ⇒ every interior cell qualifies.
    const float min_wall_dist = wall_margin - params.ior_path_radius;

    visit_grid_.observable.assign(visit_grid_.cells.size(), 0);
    dbg_unobservable_ = 0;
    for (int i = 0; i < static_cast<int>(visit_grid_.cells.size()); ++i)
    {
        const Eigen::Vector2f c = visit_grid_.cell_center(i);
        bool ok = corner_visibility::point_in_polygon(c, room_corners_);
        if (ok and min_wall_dist > 0.f)
        {
            for (std::size_t k = 0; k < room_corners_.size(); ++k)
            {
                const auto& a = room_corners_[k];
                const auto& b = room_corners_[(k + 1) % room_corners_.size()];
                const Eigen::Vector2f ab = b - a;
                const float t = std::clamp((c - a).dot(ab) / ab.squaredNorm(), 0.f, 1.f);
                if ((c - (a + t * ab)).squaredNorm() < min_wall_dist * min_wall_dist)
                { ok = false; break; }
            }
        }
        visit_grid_.observable[i] = ok ? 1 : 0;
        if (!ok) ++dbg_unobservable_;
    }
    mask_dirty_ = false;
}

void EpistemicPlanner::set_robot_state(const Eigen::Affine2f& pose,
                                      const Eigen::Matrix3f& covariance)
{
    robot_pose_ = pose;
    robot_cov_ = covariance;
    robot_state_set_ = true;
}

void EpistemicPlanner::set_robot_footprint(float width_m, float length_m)
{
    const float footprint_radius = 0.5f * std::hypot(width_m, length_m);
    if (std::abs(robot_footprint_radius_ - footprint_radius) > 1e-6f)
    {
        robot_footprint_radius_ = footprint_radius;
        grid_dirty_ = true;
        mask_dirty_ = true;   // it feeds the effective wall margin the mask is built from
    }
}

void EpistemicPlanner::set_obstacle_footprints(std::vector<ObstacleFootprint> footprints)
{
    obstacle_footprints_ = std::move(footprints);
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
    const float obstacle_clearance = std::max(params.target_obstacle_clearance, robot_footprint_radius_);
    std::vector<Eigen::Vector2f> candidates;
    candidates.reserve(std::min(static_cast<int>(cached_grid_.size()), params.max_candidates));
    dbg_grid_ = static_cast<int>(cached_grid_.size());
    dbg_near_ = 0; dbg_blocked_ = 0;

    for (const auto& p : cached_grid_)
    {
        if ((p - robot_pos()).squaredNorm() < min_d2)
        { ++dbg_near_; continue; }

        // Reject candidates that fall inside (or too close to) any object/obstacle footprint.
        // Clearance must match the CONSUMER's near-goal requirement, not merely the robot body — a
        // target that only clears the footprint is one the executor will not stop at. See
        // Params::target_obstacle_clearance.
        bool blocked = false;
        for (const auto& obs : obstacle_footprints_)
        {
            const Eigen::Vector2f d = p - obs.center;
            const float c = std::cos(-obs.yaw);
            const float s = std::sin(-obs.yaw);
            const float lx = c * d.x() - s * d.y();
            const float ly = s * d.x() + c * d.y();
            if (std::abs(lx) < obs.half_w + obstacle_clearance and
                std::abs(ly) < obs.half_d + obstacle_clearance)
            {
                blocked = true;
                break;
            }
        }
        if (blocked) { ++dbg_blocked_; continue; }

        candidates.emplace_back(p);
        if (static_cast<int>(candidates.size()) >= params.max_candidates)
            break;
    }
    dbg_candidates_ = static_cast<int>(candidates.size());
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
Eigen::Matrix3f EpistemicPlanner::predicted_fim(const Eigen::Vector2f& viewpoint) const
{
    Eigen::Matrix3f fim = Eigen::Matrix3f::Zero();
    if (room_corners_.empty()) return fim;

    // (1) Corner landmarks — full 2D fixes (rank-2 each).
    if (const auto vis = corner_visibility::visible_corners(viewpoint, room_corners_,
                                                            params.fim_max_range);
        !vis.empty())
    {
        const Eigen::Vector2f dir = viewpoint - robot_pos();
        const float heading = (dir.squaredNorm() > 1e-4f)   // heading-independent for JᵀJ; kept for API
            ? std::atan2(-dir.x(), dir.y()) : robot_theta();
        fim += corner_visibility::corner_fim(viewpoint, heading, vis,
                                             room_corners_, params.fim_corner_sigma);
    }

    // (2) Wall surfaces — the localizer's actual point-to-wall SDF likelihood over a
    //     360° scan (rank-1, perpendicular-only). Captures the parallel-wall ambiguity.
    if (params.fim_use_walls)
        fim += corner_visibility::wall_fim(viewpoint, room_corners_, params.fim_wall_sigma,
                                           params.fim_max_range, params.fim_wall_rays,
                                           params.fim_wall_incidence_weight,
                                           params.fim_wall_incidence_min);
    return fim;
}

float EpistemicPlanner::info_gain_nats(const Eigen::Matrix3f& prior_precision,
                                       const Eigen::Matrix3f& fim)
{
    if (fim.isZero(1e-12f)) return 0.f;
    // ΔH = ½·log det(I + Y_prior⁻¹·I_pred) in NATS — the expected reduction in DIFFERENTIAL
    // ENTROPY of the pose-belief Gaussian (the common EFE currency shared with aff_table).
    // d_optimality_gain returns the full log-det ( = 2·ΔH ), so apply ½. Mathematically ≥0 for a
    // PD prior + PSD FIM; clamp guards float roundoff so epistemic_gain is a non-negative benefit.
    return std::max(0.f, 0.5f * corner_visibility::d_optimality_gain(prior_precision, fim));
}

float EpistemicPlanner::neglect_nats(const Eigen::Vector2f& pos,
                                     std::chrono::steady_clock::time_point now) const
{
    if (!visit_grid_.initialized) return 0.f;
    // log(1 + a/τ): the information recovered by re-observing a cell whose knowledge has been
    // decaying for `a` seconds. Strictly monotone and unbounded in a, so there is ALWAYS a unique
    // stalest cell — unlike the clamped staleness = min(1, a/τ) this replaces, which ties at 1.0
    // room-wide once everything is older than τ and leaves the ranking with nothing to steer on.
    // Never-visited cells carry the ~11-day sentinel age and so rank far above any visited cell,
    // which is what we want at start-up; the log keeps even that at a sane ~9 nats.
    const float age = visit_grid_.age_seconds(pos, now);
    return std::log1p(age / std::max(1e-3f, params.ior_decay_time));
}

Eigen::Matrix3f EpistemicPlanner::current_prior_precision() const
{
    const Eigen::Matrix3f reg_cov = robot_cov_ + 1e-6f * Eigen::Matrix3f::Identity();
    Eigen::Matrix3f Y_info = reg_cov.inverse();
    // Belief forgetting: while the robot is not actively exploring, decay the epistemic prior
    // precision toward its floor over belief_forget_time (exp decay). This is a random-walk covariance
    // inflation on the SCORING prior only — it never touches the localizer's covariance — so the
    // self-extinguishing pose-FIM ΔH recovers and afford_room re-activates for a periodic re-check.
    // refresh_belief() resets the clock during active exploration, so forgetting accrues only when idle.
    if (params.belief_forget_time > 0.f)
    {
        const float dt = std::chrono::duration<float>(
            std::chrono::steady_clock::now() - last_belief_refresh_).count();
        Y_info *= std::exp(-dt / params.belief_forget_time);
    }
    // Floor Y_prior eigenvalues at a small prior precision. A degenerate/lost prior (huge
    // covariance → near-zero precision → det→0) then yields a LARGE-BUT-FINITE ΔH instead of
    // ∞/NaN: log det is bounded below by 3·log(floor), not by a determinant clamp artifact.
    // Equivalent to capping the assumed max covariance at 1/floor. Negligible once localized.
    return Y_info + params.fim_prior_precision_floor * Eigen::Matrix3f::Identity();
}

float EpistemicPlanner::live_epistemic_gain(const Eigen::Vector2f& viewpoint) const
{
    if (!robot_state_set_ || room_corners_.empty()) return 0.f;
    return info_gain_nats(current_prior_precision(), predicted_fim(viewpoint));
}

float EpistemicPlanner::marginal_epistemic_gain(const Eigen::Vector2f& viewpoint) const
{
    if (!robot_state_set_ || room_corners_.empty()) return 0.f;
    const Eigen::Matrix3f Y = current_prior_precision();
    // Value of the vantage MINUS the value of the vantage we already have. Equivalently
    // ½·log[det(Y + I(v)) / det(Y + I(here))] — the prior log-det cancels, so this is the pure
    // "what does moving buy me" term. Clamped at 0: a candidate that sees LESS than the current
    // spot is simply worth no epistemic detour, it is not a cost in this channel (the distance
    // preference and the neglect drive decide where to go instead).
    return std::max(0.f, info_gain_nats(Y, predicted_fim(viewpoint))
                       - info_gain_nats(Y, predicted_fim(robot_pos())));
}

float EpistemicPlanner::live_total_epistemic_gain(const Eigen::Vector2f& viewpoint,
                                                  bool rotate_in_place) const
{
    // A heading recovery does not move, so the marginal-over-staying-put reading is meaningless
    // there and would advertise ~0 for the action needed exactly when the robot is most lost.
    const float info = rotate_in_place ? live_epistemic_gain(viewpoint)
                                       : marginal_epistemic_gain(viewpoint);
    // Neglect information of the destination cell. Unbounded in neglect age ⇒ the advertised gain
    // recovers on its own however long the room goes unattended, so afford_room can never fall
    // permanently out of the consumer's contention and exploration cannot stall. This is what the
    // old `patrol_gain_floor` was faking; here the gain is held up by a computed quantity instead.
    const float drive = (params.w_ior_drive > 0.f)
        ? params.w_ior_drive * neglect_nats(viewpoint, std::chrono::steady_clock::now())
        : 0.f;
    return info + drive;
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

    // Keep the observability mask in step with the contour before anything reads the visit grid.
    // Dirty-flagged, so this is a no-op on all but the cycles where the room geometry changed.
    if (mask_dirty_)
        self.refresh_observable_mask();

    if (is_angular_dominated())
    {
        Target rot;
        rot.position = robot_pos();
        rot.distance = 0.f;
        rot.rotate_in_place = true;
        rot.score = robot_cov_(2, 2);
        // Advertise the same grounded ΔH currency for the recovery action: rotating in place
        // sweeps a 360° scan from the current vantage, so its expected pose-entropy reduction is
        // the ABSOLUTE FIM gain here (not the marginal-over-staying-put reading, which is 0 by
        // definition for an action that does not move). Without this the heading-recovery action —
        // needed exactly when the robot is most lost — would publish gain=0 and lose every
        // cross-affordance EFE compare.
        rot.eigenvector_score = live_epistemic_gain(robot_pos());
        self.cell_scores_.clear();
        self.ior_cells_.clear();
        return {rot};
    }

    const auto candidates = generate_candidates();
    if (candidates.empty())
    {
        // Diagnose WHY nothing is selectable — the "stuck in open space, accepts no target" case.
        // Only runs on failure, so the recompute is cheap. Distinguishes the three starvation modes:
        //   grid==0            → polygon/wall-margin killed the whole room (geometry/config issue)
        //   beyond_mindist==0  → MinDistance too large for where the robot is
        //   blocked==beyond    → every reachable cell sits inside an obstacle footprint (obstacle flood,
        //                        e.g. residual_concept/table publishing boxes over the free space)
        int beyond_mindist = 0, blocked = 0;
        const float min_d2 = params.min_distance * params.min_distance;
        const float obstacle_clearance = std::max(params.target_obstacle_clearance, robot_footprint_radius_);
        for (const auto& p : cached_grid_)
        {
            if ((p - robot_pos()).squaredNorm() < min_d2) continue;
            ++beyond_mindist;
            for (const auto& obs : obstacle_footprints_)
            {
                const Eigen::Vector2f d = p - obs.center;
                const float c = std::cos(-obs.yaw), s = std::sin(-obs.yaw);
                const float lx = c * d.x() - s * d.y();
                const float ly = s * d.x() + c * d.y();
                if (std::abs(lx) < obs.half_w + obstacle_clearance and
                    std::abs(ly) < obs.half_d + obstacle_clearance) { ++blocked; break; }
            }
        }
        std::print("[planner] NO TARGET (STARVED): grid={} beyond_mindist={} blocked_by_obstacles={} "
                   "obstacles={} clearance={:.2f} robot=({:.2f},{:.2f}) min_dist={:.2f}\n",
                   cached_grid_.size(), beyond_mindist, blocked, obstacle_footprints_.size(),
                   obstacle_clearance, robot_pos().x(), robot_pos().y(), params.min_distance);
        std::fflush(stdout);
        self.cell_scores_.clear();
        self.ior_cells_.clear();
        return {};
    }

    const Eigen::Matrix3f prior_precision = current_prior_precision();
    const auto now = std::chrono::steady_clock::now();

    // Reference for the MARGINAL information term: what the robot can already see from where it
    // stands. Computed once — it is the null policy every candidate is scored against.
    const float here_gain = info_gain_nats(prior_precision, predicted_fim(robot_pos()));

    // Travel cost is normalised by the room diagonal and SUBTRACTED in nats. See Params::w_travel_cost
    // for why the sign is load-bearing: the neglect field is flat across large regions, so this term
    // decides the arg-max there, and a far-is-better sign turns that into corner-to-corner oscillation.
    const float room_diag = std::max(1e-3f, (room_max_ - room_min_).norm());

    std::vector<Target> targets;
    targets.reserve(candidates.size());
    float max_fim = 0.f;                      // best MARGINAL pose-info gain across candidates

    for (const auto& pos : candidates)
    {
        Target t;
        t.position = pos;
        t.distance = (pos - robot_pos()).norm();
        // MARGINAL pose information: what this vantage adds over the one we already occupy. Goes to
        // 0 wherever the candidate sees no geometry the robot cannot already see — which, in a room
        // whose layout is a fixed prior, is most of a convex region once the robot is inside it.
        // That is the "no increase in room certainty is possible" condition, computed rather than
        // thresholded, and it hands the ranking over to the neglect drive below all by itself.
        const float fim_gain = std::max(0.f, info_gain_nats(prior_precision, predicted_fim(pos))
                                             - here_gain);
        max_fim = std::max(max_fim, fim_gain);

        // Update cell's running FIM average
        self.visit_grid_.update_fim(pos, fim_gain);

        // IoR: staleness of the target cell ∈ [0,1], 0=just visited, 1=fully stale
        const float staleness = visit_grid_.staleness(pos, params.ior_decay_time, now);

        // Path staleness: mean staleness of intermediate cells along the straight-line
        // path from robot to target. High = path goes through unexplored territory.
        //
        // ONLY OBSERVABLE CELLS ARE SAMPLED. The straight line is a scoring device, not a route, and
        // in a non-convex apartment it clips wall interiors and pillar bays constantly. Such a cell
        // can never be marked visited, so it reports the never-visited sentinel age forever — a
        // permanent ~9-nat reward attached to whichever target pair happens to have one on its
        // segment, which is exactly how the planner ends up ping-ponging between two spots while
        // genuinely neglected territory ages. Skip them and average over what is left, rather than
        // counting them as zero: a skipped sample must not dilute the mean either, or the same
        // geometry would flip into a permanent PENALTY instead.
        const auto rpos = robot_pos();
        const int n_path = std::max(3, static_cast<int>(t.distance / params.ior_cell_size));
        float path_sum = 0.f;
        float path_neglect_sum = 0.f;
        int   n_sampled = 0;
        for (int s = 1; s < n_path; ++s)
        {
            const float alpha = static_cast<float>(s) / n_path;
            const Eigen::Vector2f cell = rpos + alpha * (pos - rpos);
            if (not visit_grid_.is_observable(cell))
                continue;
            path_sum += visit_grid_.staleness(cell, params.ior_decay_time, now);
            path_neglect_sum += neglect_nats(cell, now);
            ++n_sampled;
        }
        // A segment whose whole interior is wall says nothing about the route; fall back to the
        // destination's own reading so the blend below degrades to "target cell only" instead of
        // silently scoring it as freshly-swept.
        const float inv_path = (n_sampled > 0) ? 1.f / static_cast<float>(n_sampled) : 0.f;
        const float path_staleness = (n_sampled > 0) ? path_sum * inv_path : staleness;          // ∈[0,1]
        const float path_neglect   = (n_sampled > 0) ? path_neglect_sum * inv_path
                                                     : neglect_nats(pos, now);                    // nats

        // Route staleness: blend of target cell and path cells.
        // w_path_interest=0 → target staleness only (old behaviour)
        // w_path_interest=1 → path staleness only
        // Use as multiplicative IoR suppressor so a visited path to an unvisited target
        // is penalised by the same power as a visited target, not just a small additive nudge.
        // Staying CLAMPED is correct for this role: it only asks "did I just sweep this route",
        // and it modulates a term that already self-extinguishes. The long-horizon steering is
        // carried by route_neglect below, which does not saturate.
        const float route_staleness = (1.f - params.w_path_interest) * staleness
                                    + params.w_path_interest * path_staleness;
        const float ior_suppressor = std::pow(route_staleness, params.w_ior);

        // Neglect information of the route, in nats. Blended per-cell (not per-age) so the
        // never-visited sentinel age is log-compressed before it enters the mean and cannot swamp
        // an otherwise well-known path.
        const float route_neglect = (1.f - params.w_path_interest) * neglect_nats(pos, now)
                                  + params.w_path_interest * path_neglect;

        // Score = expected free energy, all in nats:
        //   (marginal pose info, suppressed if we just swept this route)
        // + (neglect information of the route — the non-saturating exploration drive)
        // − (travel cost)
        // The first term extinguishes itself as the room becomes fully visible; the second never
        // does, so the ranking degrades continuously into oldest-first inhibition-of-return with
        // no mode switch, no "info exhausted" flag and no gain floor. The third makes the choice
        // among equally-informative cells the CHEAPEST one, which is what turns the sweep into
        // contiguous coverage instead of an oscillation between extremes.
        // ★ATTEMPT-IoR MULTIPLIES THE REWARD, AND ONLY THE REWARD. This is the de-prioritisation
        // that mark_target_finished used to obtain by forging a visit; it now acts here, where it
        // belongs, leaving `neglect_nats` to report the TRUE observation age. A cell that was tried
        // and not reached therefore keeps its (correct, growing) neglect and simply stops being
        // proposed for a while — no blacklist, and it returns on its own as the attempt decays.
        const float attempt_supp = attempt_suppressor(pos, now);
        t.score = attempt_supp * (fim_gain * ior_suppressor
                                  + params.w_ior_drive * route_neglect)
                - params.w_travel_cost * (t.distance / room_diag);
        t.eigenvector_score = fim_gain;
        targets.push_back(t);
    }

    self.dbg_max_fim_  = max_fim;
    self.dbg_here_fim_ = here_gain;

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
            // Heatmap = pose-info (FIM × staleness).
            const float combined = cell.fim_gain * std::pow(stale, params.w_ior);
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

    // Rotate-in-place: no randomness needed. It MUST still report — this path used to return
    // silently, which made a planner stuck proposing heading-recovery indistinguishable from a
    // planner that had stopped selecting altogether. Note what it publishes: a target at the
    // robot's OWN position, which a position-following consumer completes instantly without ever
    // rotating.
    if (targets.front().rotate_in_place)
    {
        const float ratio = robot_cov_(2, 2) / std::max(1e-9f, std::max(robot_cov_(0, 0), robot_cov_(1, 1)));
        std::print("[planner] ROTATE-IN-PLACE at ({:.2f},{:.2f}) — angular dominance {:.1f} > {:.1f} "
                   "(σθ={:.3f}rad σxy={:.3f}m) gain={:.3f}\n",
                   robot_pos().x(), robot_pos().y(), ratio, params.angular_dominance_ratio,
                   std::sqrt(std::max(0.f, robot_cov_(2, 2))),
                   std::sqrt(std::max(0.f, std::max(robot_cov_(0, 0), robot_cov_(1, 1)))),
                   targets.front().eigenvector_score);
        std::fflush(stdout);
        return targets.front();
    }

    // ---- Selection diagnostic: shows WHY a target was picked (verifies IoR is steering, not the
    //      distance bonus). Printed once per new-target selection (update_target only calls this when
    //      there is no current target), so it is not spammy.
    {
        const auto now = std::chrono::steady_clock::now();
        const Eigen::Vector2f rp = robot_pos();
        // Advertised gain for the CHOSEN target — the quantity the consumer ranks on, so this line
        // also diagnoses "runs out of affordances" (gain falling below competitors / the patrol floor).
        const float pub_gain = live_total_epistemic_gain(targets.front().position);
        // here_fim = absolute pose info available WITHOUT moving; max_fim = best MARGINAL gain over
        // that. max_fim ≈ 0 with here_fim large is the healthy "room fully visible, nothing to learn
        // by driving" regime — the robot should then be steered purely by the neglect term, which
        // the per-target `neg=` column below shows.
        std::print("[planner] robot=({:.2f},{:.2f}) here_fim={:.3f} max_marg_fim={:.4f} pub_gain={:.3f}"
                   " | cand={}/{} (near={} blocked={} obst={}) masked_cells={} | top:",
                   rp.x(), rp.y(), dbg_here_fim_, dbg_max_fim_, pub_gain,
                   dbg_candidates_, dbg_grid_, dbg_near_, dbg_blocked_,
                   obstacle_footprints_.size(), dbg_unobservable_);
        const int n = std::min<int>(3, static_cast<int>(targets.size()));
        for (int i = 0; i < n; ++i)
        {
            const auto& t = targets[i];
            const float age = visit_grid_.age_seconds(t.position, now);
            std::print("  #{} ({:+.2f},{:+.2f}) score={:.3f} marg_fim={:.4f} neg={:.2f} age={:.0f}s d={:.2f}",
                       i, t.position.x(), t.position.y(), t.score, t.eigenvector_score,
                       neglect_nats(t.position, now), age, t.distance);
        }
        std::print("\n");
        std::fflush(stdout);
    }

    // ★★★THE SELECTION, AS DATA. Every scoring hypothesis tested on 2026-08-19 died because it was
    // seeded with GUESSES about this state — I asserted travel cost dominated the argmax and a
    // simulation over both a sparse field and a 348-cell dense grid refuted it twice. The rule is not
    // the unknown; the per-cell ages and suppressor values are. So dump the top candidates with every
    // term that entered their score, and let the bench replay a real selection instead of my model of
    // one. Self-describing (each value carries its key) after four positional-CSV misreadings today.
    // Throttled to one record per second: this is for offline replay, not per-cycle telemetry.
    {
        const auto now = std::chrono::steady_clock::now();   // the debug block above owns its own
        static std::ofstream sel_json;
        static bool sel_open = false;
        static std::chrono::steady_clock::time_point sel_last{};
        if (not sel_open)
        {
            sel_json.open("epistemic_select.jsonl", std::ios::out | std::ios::trunc);
            sel_json.imbue(std::locale::classic());   // decimal POINT under LANG=es_ES (see CLAUDE.md)
            sel_open = sel_json.is_open();
        }
        if (sel_open and std::chrono::duration<float>(now - sel_last).count() >= 1.0f)
        {
            sel_last = now;
            const auto rp2 = robot_pos();
            std::string cs;
            const int m = std::min<int>(8, static_cast<int>(targets.size()));
            for (int i = 0; i < m; ++i)
            {
                const auto& t = targets[i];
                const float age = visit_grid_.age_seconds(t.position, now);
                const float stale = visit_grid_.staleness(t.position, params.ior_decay_time, now);
                cs += std::format(
                    R"({}{{"x":{:.3f},"y":{:.3f},"score":{:.5f},"fim":{:.5f},"neg":{:.4f},)"
                    R"("age_s":{:.1f},"stale":{:.4f},"attempt_supp":{:.4f},"d":{:.3f}}})",
                    cs.empty() ? "" : ",", t.position.x(), t.position.y(), t.score,
                    t.eigenvector_score, neglect_nats(t.position, now), age, stale,
                    attempt_suppressor(t.position, now), t.distance);
            }
            sel_json << std::format(
                R"({{"rob_x":{:.3f},"rob_y":{:.3f},"n_cand":{},"n_grid":{},"near_rejected":{},)"
                R"("w_travel":{:.3f},"w_drive":{:.3f},"w_ior":{:.3f},"tau":{:.1f},"min_distance":{:.2f},)"
                R"("cands":[{}]}})" "\n",
                rp2.x(), rp2.y(), dbg_candidates_, dbg_grid_, dbg_near_,
                params.w_travel_cost, params.w_ior_drive, params.w_ior, params.ior_decay_time,
                params.min_distance, cs);
            sel_json.flush();
        }
    }

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

// mark_target_finished — a reached OR abandoned target is stamped as visited, so the neglect drive
// that selected it drops to ~0 there and the planner moves on instead of re-proposing it forever.
// See the header for why this is a de-prioritisation and not a blacklist.
void EpistemicPlanner::mark_target_finished(const Eigen::Vector2f& pos)
{
    if (!visit_grid_.initialized) return;
    // ★NO visit-grid stamp here any more — see the header. An attempt is not an observation, and
    // mark_and_refresh() already records, from the robot's real pose, everywhere it has actually been.
    note_attempt(pos);
    current_target_.reset();
    dwelling_ = false;
    refresh_ior_overlay();
}

void EpistemicPlanner::note_attempt(const Eigen::Vector2f& pos)
{
    const auto now = std::chrono::steady_clock::now();
    // Drop attempts that have fully decayed, and collapse a repeat at the same spot onto one entry
    // (the receptive field is what "the same spot" means everywhere else in this file).
    std::erase_if(attempts_, [&](const Attempt& a)
    {
        const bool expired = std::chrono::duration<float>(now - a.when).count()
                             >= std::max(0.1f, params.ior_decay_time);
        return expired or (a.pos - pos).norm() <= params.ior_path_radius;
    });
    attempts_.push_back({pos, now});
}

float EpistemicPlanner::attempt_suppressor(const Eigen::Vector2f& pos,
                                           std::chrono::steady_clock::time_point now) const
{
    // The most recent attempt whose receptive field covers `pos` decides. Linear recovery over
    // ior_decay_time, matching staleness()'s shape so the two IoR channels behave alike.
    float supp = 1.f;
    for (const auto& a : attempts_)
    {
        if ((a.pos - pos).norm() > params.ior_path_radius) continue;
        const float elapsed = std::chrono::duration<float>(now - a.when).count();
        supp = std::min(supp, std::min(1.f, elapsed / std::max(0.1f, params.ior_decay_time)));
    }
    return supp;
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
            refresh_belief();   // reached & scanned a target → knowledge fresh; restart the forget clock
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
