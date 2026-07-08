#pragma once

#include <vector>
#include <optional>
#include <chrono>
#include <random>
#include <Eigen/Dense>
#include "corner_visibility.h"

namespace rc
{

/**
 * EpistemicPlanner — Level 1: FIM-based information-gain target selection
 * =====================================================================
 * Generates candidate positions on a grid inside the room, scores them
 * using the Fisher-Information-Matrix D-optimality gain, exploration
 * distance bonus, and an Inhibition-of-Return (IoR) spatial visit grid,
 * then returns the best candidate as the planner's navigation target.
 *
 * Also handles angular-dominance detection (rotate-in-place), target
 * lifecycle (arrival, dwell timer), and visit-grid bookkeeping.
 */


class EpistemicPlanner
{
public:
    struct Params
    {
        float grid_resolution = 0.5f;        // spacing between candidate targets (m)
        float min_distance = 1.0f;           // ignore candidates closer than this to robot (m)
        int   max_candidates = 2000;         // cap on number of evaluated candidates
        float target_wall_margin = 1.0f;     // reject targets closer than this to walls (m)

        float angular_dominance_ratio = 50.0f; // σ²_θ / max(σ²_x, σ²_y) threshold
        float w_exploration  = 0.5f;         // weight: linear distance bonus (explore farther)

        // ---- Inhibition of Return (visit grid) ----
        float ior_cell_size   = 0.5f;        // spatial resolution of the visit grid (m)
        float ior_decay_time  = 120.0f;      // seconds until a visited cell is fully "stale" again
        float w_ior           = 2.0f;        // exponent for IoR suppressor: score *= staleness^w_ior
                                             //   (0=no suppression, 1=linear, 2=quadratic, etc.)
                                             //   just-visited → staleness=0 → score=0 (hard inhibition)
        float ior_path_radius = 1.0f;        // receptive-field radius for continuous path marking (m)
                                             //   = 2 grid cells at default ior_cell_size=0.5m
                                             //   cells within this radius are suppressed as the robot advances
        float w_path_interest = 0.3f;        // weight: bonus for paths that traverse unvisited cells
                                             //   path_interest = mean staleness of intermediate path cells ∈ [0,1]
                                             //   higher → prefer routes through unexplored territory
        // ADDITIVE IoR patrol drive (nats). w_ior (above) is a MULTIPLICATIVE suppressor on fim_gain,
        // so it vanishes once fim_gain→0 (localized) and map_gain→0 (room seen) — both info terms
        // saturate in a static convex room and the robot stops after a few affordances. This term is
        // ADDITIVE and NON-saturating: route_staleness recovers over ior_decay_time, so it keeps both
        // the selection score AND the published gain positive on long-unvisited cells → the robot
        // perpetually patrols the stalest reachable cell. It sits BELOW the info terms in magnitude so
        // genuine info-gathering still takes priority; it only dominates once info is exhausted.
        // 0 ⇒ off (robot rests once the room is fully known). Units: nats per unit staleness ∈ [0,1].
        float w_ior_drive     = 0.5f;

        // ---- Second-level Inhibition of Return: perpetual room patrol ----
        // The first-level IoR terms above (w_ior suppressor + w_ior_drive additive drive) both use
        // staleness() ∈ [0,1] CLAMPED at ior_decay_time. Once the whole room has gone unvisited for
        // longer than ior_decay_time, every reachable cell saturates at staleness=1: the recency signal
        // is lost, the selection argmax collapses onto the distance bonus, and the advertised gain flattens
        // — so the robot rests after only a few affordances even though there is always a *stalest* area to
        // revisit. This second level fixes both failure modes once pose-info is exhausted (max FIM gain <
        // info_exhausted_gain):
        //   (a) TARGET: re-rank candidates by RAW age (unclamped seconds since last visit), so the single
        //       least-recently-visited reachable cell always wins and the robot patrols oldest-first.
        //   (b) GAIN: floor afford_room's advertised gain at patrol_gain_floor nats so the consuming
        //       controller keeps selecting/executing it instead of withdrawing when the pose-FIM term → 0.
        // Keep patrol_gain_floor BELOW typical object-affordance ΔH so real objects still out-compete the
        // patrol when present. This is a superset of w_ior_drive: leave w_ior_drive for the smooth blend
        // while info is still trickling, and let this take over once info is truly exhausted.
        bool  patrol_enabled       = true;   // second-level IoR patrol master switch
        float patrol_gain_floor    = 0.0f;   // nats floored onto afford_room's gain in patrol mode (0 = off)
        float info_exhausted_gain  = 0.02f;  // max FIM gain (nats) below which info is "exhausted" → patrol

        // ---- FIM scoring ----
        float fim_corner_sigma  = 0.04f;     // isotropic corner detection noise σ (m)
        float fim_max_range     = 10.0f;     // max range for corner/wall visibility (m)
        float fim_prior_precision_floor = 1e-4f;  // floor on Y_prior eigenvalues (caps assumed max
                                                  // covariance at 1/floor) → degenerate prior gives a
                                                  // large-but-FINITE ΔH, no ∞/NaN

        // ---- Wall-surface SDF observations (match the localizer's likelihood) ----
        // Corners alone are full 2D fixes (rank-2) and miss the parallel-wall ambiguity.
        // The real localizer fits point-to-wall SDF over a 360° scan: each wall hit is a
        // rank-1 (perpendicular-only) constraint. Summing these makes the planner's
        // "evidence" match what the localizer actually gains, so it correctly prefers
        // vantage points that break the along-wall / heading ambiguity (corners, walls
        // seen at good incidence, far walls for heading leverage).
        bool  fim_use_walls             = true;
        float fim_wall_sigma            = 0.15f;  // SDF obs noise σ (m); ≈ RoomConcept.sigma_sdf
        int   fim_wall_rays             = 64;     // simulated 360° lidar rays for wall coverage
        bool  fim_wall_incidence_weight = true;   // down-weight grazing hits (|n·ray|)
        float fim_wall_incidence_min    = 0.2f;   // floor for the incidence weight

        // ---- Target lifecycle ----
        float arrival_distance = 0.15f;      // target reached threshold (m)
        float dwell_time       = 2.0f;       // seconds to stop after reaching a target

        // ---- Belief forgetting (re-activate exploration) ----
        // The pose-FIM ΔH self-extinguishes once the robot localizes against the static room prior
        // (robot_cov_ shrinks and stays tight), so afford_room goes silent "after a while". With
        // belief_forget_time > 0, the pose-prior precision used for epistemic SCORING decays toward its
        // floor over this timescale WHENEVER the robot is not actively exploring — a random-walk
        // covariance inflation on the epistemic prior only (never the localizer's covariance). The gain
        // then recovers and the robot periodically re-verifies the room. refresh_belief() resets the
        // clock while exploring, giving a breathe-in/out cycle of period ≈ belief_forget_time. 0 = off.
        float belief_forget_time = 0.0f;     // seconds; 0 disables (current self-extinguishing behaviour)
    };

    /// Scored candidate target in room frame
    struct Target
    {
        Eigen::Vector2f position{0.f, 0.f};
        float score = 0.0f;
        float distance = 0.0f;
        float eigenvector_score = 0.0f;
        bool  rotate_in_place = false;
    };

    EpistemicPlanner();
    explicit EpistemicPlanner(Params params);

    /// Axis-aligned oriented bounding box of an object/obstacle in room frame.
    struct ObstacleFootprint
    {
        Eigen::Vector2f center;   // room-frame position (m)
        float half_w = 0.f;       // half-width  along local X (m)
        float half_d = 0.f;       // half-depth  along local Y (m)
        float yaw    = 0.f;       // orientation relative to room frame (rad)
    };

    // ---- State setters ----
    void set_room_bounds(const Eigen::Vector2f& min_corner, const Eigen::Vector2f& max_corner);
    void set_room_polygon(const std::vector<Eigen::Vector2f>& vertices);
    void set_robot_state(const Eigen::Affine2f& pose, const Eigen::Matrix3f& covariance);
    void set_robot_footprint(float width_m, float length_m);

    /// Replace the list of occupied object/obstacle footprints used to exclude
    /// candidate targets.  Typically refreshed every compute cycle from the DSR graph.
    void set_obstacle_footprints(std::vector<ObstacleFootprint> footprints);

    // ---- Target selection (public API) ----
    std::vector<Target> evaluate_targets() const;
    std::optional<Target> select_target();

    /// Live grounded epistemic value (nats) of observing from `viewpoint`, evaluated
    /// against the CURRENT pose precision Y_prior: ΔH = ½·log det(I + Y_prior⁻¹·I_pred).
    /// This is the common EFE currency advertised on afford_room — recompute it every
    /// publish cycle so it decays as localization tightens (Y_prior grows ⇒ ΔH → 0) and
    /// falls through the consumer's withdrawal threshold. Returns 0 when nothing is
    /// visible. For a rotate-in-place recovery, pass robot_pos().
    float live_epistemic_gain(const Eigen::Vector2f& viewpoint) const;

    /// TOTAL epistemic gain advertised on afford_room = pose-FIM ΔH (nats, self-extinguishing
    /// once localized) + w_ior_drive · IoR patrol-staleness drive (NON-saturating: stays positive
    /// on long-unvisited cells even after the pose-FIM term saturates). Mirrors the selection
    /// currency in evaluate_targets so the published gain matches how targets are ranked. Use
    /// this as the published gain so exploration does not stall the moment the pose covariance
    /// tightens.
    float live_total_epistemic_gain(const Eigen::Vector2f& viewpoint) const;

    /// Called every plan cycle.  Returns the current navigation target,
    /// handling dwell and arrival logic internally.  Returns std::nullopt
    /// only when no valid target can be found.
    std::optional<Target> update_target();

    void clear_target() { current_target_.reset(); }
    const std::optional<Target>& current_target() const { return current_target_; }

    /// Mark the room-knowledge belief as FRESH — call while the robot is actively exploring or
    /// executing an epistemic affordance. Resets the belief-forgetting clock so the epistemic prior
    /// only decays (and exploration only re-activates) during genuine idle time. No-op unless
    /// Params::belief_forget_time > 0.
    void refresh_belief() { last_belief_refresh_ = std::chrono::steady_clock::now(); }

    /// Lightweight per-cycle update: stamps the current robot position in the
    /// visit grid and refreshes the IoR overlay. Call this when the full
    /// update_target() path is skipped (e.g. while a sibling agent is
    /// executing the affordance) so the path trail stays live in the viewer.
    void mark_and_refresh();

    // ---- Cell score data for visualisation ----
    struct CellScore
    {
        Eigen::Vector2f center;
        float score;          // combined probability weight
    };
    const std::vector<CellScore>& cell_scores() const { return cell_scores_; }
    /// Per-cell visit freshness for IoR overlay (1 = just visited, 0 = stale/never)
    struct IorCell
    {
        Eigen::Vector2f center;
        float freshness;      // 1=just visited, fades to 0 over ior_decay_time
    };
    const std::vector<IorCell>& ior_cells() const { return ior_cells_; }
    float cell_size() const { return params.ior_cell_size; }

    // ---- Accessors needed by Level 2 ----
    Eigen::Vector2f robot_pos() const { return robot_pose_.translation(); }
    float robot_theta() const { return std::atan2(robot_pose_.linear()(1,0), robot_pose_.linear()(0,0)); }
    const Eigen::Affine2f& robot_pose() const { return robot_pose_; }
    const Eigen::Matrix3f& robot_cov()  const { return robot_cov_; }
    const std::vector<Eigen::Vector2f>& room_corners() const { return room_corners_; }
    const Eigen::Vector2f& room_min() const { return room_min_; }
    const Eigen::Vector2f& room_max() const { return room_max_; }
    bool room_bounds_set() const { return room_bounds_set_; }

    Params params;

private:
    std::vector<Eigen::Vector2f> generate_candidates() const;
    bool is_angular_dominated() const;
    float score_fim_gain(const Eigen::Vector2f& candidate,
                         const Eigen::Matrix3f& prior_precision) const;
    /// Y_prior = inverse pose covariance with eigenvalues floored at
    /// fim_prior_precision_floor (a degenerate/lost prior then yields a
    /// large-but-finite ΔH, no ∞/NaN). Shared by evaluate_targets and live_epistemic_gain.
    Eigen::Matrix3f current_prior_precision() const;
    void refresh_ior_overlay();   // lightweight per-cycle rebuild of ior_cells_

    // ---- State ----
    Eigen::Vector2f room_min_{0, 0};
    Eigen::Vector2f room_max_{0, 0};
    bool room_bounds_set_ = false;

    std::vector<Eigen::Vector2f> room_corners_;
    mutable std::vector<Eigen::Vector2f> cached_grid_;
    mutable bool grid_dirty_ = true;

    Eigen::Affine2f robot_pose_ = Eigen::Affine2f::Identity();
    Eigen::Matrix3f robot_cov_ = Eigen::Matrix3f::Identity();
    bool robot_state_set_ = false;
    float robot_footprint_radius_ = 0.f;

    // Belief-forgetting clock: time of the last "belief is fresh" mark (active exploration). The
    // epistemic prior precision decays as (now - this) grows; refresh_belief() resets it. Initialised
    // to construction time so no spurious forgetting accrues before the first exploration.
    std::chrono::steady_clock::time_point last_belief_refresh_ = std::chrono::steady_clock::now();

    // Object/obstacle exclusion zones (updated each cycle from DSR graph)
    std::vector<ObstacleFootprint> obstacle_footprints_;

    // Persistent target
    std::optional<Target> current_target_;
    std::chrono::steady_clock::time_point dwell_until_{};
    bool dwelling_ = false;

    // ---- Inhibition of Return: spatial visit grid with accumulated scores ----
    struct VisitGrid
    {
        struct Cell
        {
            std::chrono::steady_clock::time_point last_visit{};
            float fim_gain = 0.f;   // running-average FIM info gain estimate
        };
        std::vector<Cell> cells;
        int cols = 0, rows = 0;
        float cell_size = 0.5f;
        Eigen::Vector2f origin{0.f, 0.f};
        bool initialized = false;

        void init(const Eigen::Vector2f& room_min, const Eigen::Vector2f& room_max, float cs)
        {
            cell_size = cs;
            origin = room_min;
            cols = static_cast<int>(std::ceil((room_max.x() - room_min.x()) / cell_size));
            rows = static_cast<int>(std::ceil((room_max.y() - room_min.y()) / cell_size));
            cols = std::max(1, cols);
            rows = std::max(1, rows);
            cells.assign(cols * rows, Cell{});
            initialized = true;
        }

        int to_index(const Eigen::Vector2f& pos) const
        {
            int c = static_cast<int>((pos.x() - origin.x()) / cell_size);
            int r = static_cast<int>((pos.y() - origin.y()) / cell_size);
            c = std::clamp(c, 0, cols - 1);
            r = std::clamp(r, 0, rows - 1);
            return r * cols + c;
        }

        Eigen::Vector2f cell_center(int idx) const
        {
            const int c = idx % cols;
            const int r = idx / cols;
            return {origin.x() + (c + 0.5f) * cell_size,
                    origin.y() + (r + 0.5f) * cell_size};
        }

        void mark_visited(const Eigen::Vector2f& pos)
        {
            if (!initialized) return;
            cells[to_index(pos)].last_visit = std::chrono::steady_clock::now();
        }

        void mark_visited_with_falloff(const Eigen::Vector2f& pos, float radius_m,
                                       float decay_s, float edge_staleness = 0.35f)
        {
            if (!initialized) return;

            const auto now = std::chrono::steady_clock::now();
            const float radius = std::max(cell_size, radius_m);
            const int radius_cells = std::max(1, static_cast<int>(std::ceil(radius / cell_size)));

            int center_c = static_cast<int>((pos.x() - origin.x()) / cell_size);
            int center_r = static_cast<int>((pos.y() - origin.y()) / cell_size);
            center_c = std::clamp(center_c, 0, cols - 1);
            center_r = std::clamp(center_r, 0, rows - 1);

            for (int dr = -radius_cells; dr <= radius_cells; ++dr)
            {
                const int row = center_r + dr;
                if (row < 0 || row >= rows) continue;

                for (int dc = -radius_cells; dc <= radius_cells; ++dc)
                {
                    const int col = center_c + dc;
                    if (col < 0 || col >= cols) continue;

                    const int idx = row * cols + col;
                    const float dist = (cell_center(idx) - pos).norm();
                    if (dist > radius)
                        continue;

                    const float normalized = std::clamp(dist / std::max(radius, 1e-6f), 0.f, 1.f);
                    const float staleness = normalized * std::clamp(edge_staleness, 0.f, 1.f);
                    const float elapsed_s = staleness * std::max(0.1f, decay_s);
                    const auto visit_tp = now - std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                                    std::chrono::duration<float>(elapsed_s));
                    if (cells[idx].last_visit < visit_tp)
                        cells[idx].last_visit = visit_tp;
                }
            }
        }

        /// Update running-average FIM gain for the cell containing pos.
        void update_fim(const Eigen::Vector2f& pos, float gain, float alpha = 0.3f)
        {
            if (!initialized) return;
            auto& cell = cells[to_index(pos)];
            cell.fim_gain = (1.f - alpha) * cell.fim_gain + alpha * gain;
        }

        float staleness(const Eigen::Vector2f& pos, float decay_s,
                        std::chrono::steady_clock::time_point now) const
        {
            if (!initialized) return 1.f;
            const auto& tp = cells[to_index(pos)].last_visit;
            if (tp == std::chrono::steady_clock::time_point{}) return 1.f;
            const float elapsed = std::chrono::duration<float>(now - tp).count();
            return std::min(1.f, elapsed / std::max(0.1f, decay_s));
        }

        // Raw seconds since this cell was last visited, UNCLAMPED (unlike staleness(), which saturates
        // at 1 after decay_s). Never-visited cells return kNeverVisitedAge so they always rank as the
        // "oldest". Used by the second-level IoR patrol to find a unique least-recently-visited reachable
        // cell even after every cell has saturated staleness()=1.
        static constexpr float kNeverVisitedAge = 1.0e6f;   // ~11.6 days; unvisited cells patrol first
        float age_seconds(const Eigen::Vector2f& pos,
                          std::chrono::steady_clock::time_point now) const
        {
            if (!initialized) return kNeverVisitedAge;
            const auto& tp = cells[to_index(pos)].last_visit;
            if (tp == std::chrono::steady_clock::time_point{}) return kNeverVisitedAge;
            return std::chrono::duration<float>(now - tp).count();
        }
    };
    VisitGrid visit_grid_;

    // Cell score cache (updated each evaluate_targets call)
    std::vector<CellScore> cell_scores_;
    // IoR freshness overlay (updated each evaluate_targets call)
    std::vector<IorCell> ior_cells_;

    // RNG for weighted random selection
    mutable std::mt19937 rng_{std::random_device{}()};
};

} // namespace rc
