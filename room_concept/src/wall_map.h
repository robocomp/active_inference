/*
 *  wall_map.h — the room's walls as LANDMARKS: association, birth, topology, re-anchoring.
 *
 *  WHAT A WALL IS
 *  --------------
 *  A wall is an independent 2-DOF line landmark in the map frame, Hesse form  n(φ)·p = d  —
 *  the 2-D analogue of the plane landmarks in S-Graphs+ (Bavle et al., arXiv 2212.11770). Its
 *  extent (s_min, s_max along the tangent) is NOT a variable: it is the running range of the points
 *  that were associated to it, kept for the viewer and for deciding which END of the wall a corner
 *  belongs to. Corners are DERIVED (the intersection of two adjacent walls), never estimated: a
 *  corner the robot never saw is still determined by the two walls it did see.
 *
 *  MANHATTAN, HIERARCHICALLY
 *  -------------------------
 *  The room carries one yaw θ₀. Each wall born with a decisive class k ∈ {0,1,2,3} is tied to it by
 *  a room↔wall factor  φᵢ − θ₀ − kπ/2 ~ N(0, σ_ε²)  (room_gn_solver's RoomWallFactor) — the
 *  S-Graphs+ room-to-wall factor in 2-D. A wall whose angle fits no class (a chamfer) is born with
 *  k = −1 and no such factor: it is still a wall, it just does not vote on the room's yaw.
 *
 *  BIRTH IS A MODEL COMPARISON, NOT A GATE
 *  ---------------------------------------
 *  Segments no wall explains are accumulated as CANDIDATES across frames (associated to each other
 *  with the same Mahalanobis test). A candidate becomes a wall when
 *      ΔF = F_without − F_with
 *         = Σ_i [ ln(A/(L√(2π)σ)) − r_i²/2σ² ] − Occam(φ,d) − Manhattan-class cost
 *  is decisive (> birth_nats, default ln 100). The per-point term is the evidence of "on this line"
 *  over "clutter, uniform over the scan area A" — the SAME model the segmenter's inlier band is
 *  derived from (wallseg::point_gain_nats). The Occam term is 0.5·ln det(Λ_post/Λ_prior) against a
 *  uniform prior on (φ, d), and a second observation frame is required because a line seen from ONE
 *  pose cannot separate a passer-by from a wall — the same "birth is tracker-only" rule as
 *  CONCEPT_AGENT_LIFECYCLE.md. ⚠ birth_nats is a decision constant; structure change cannot be gate-free.
 *
 *  θ₀ IS BORN FROM AGREEMENT, NOT FROM THE FIRST WALL
 *  ------------------------------------------------
 *  The room's yaw is defined the first time two walls agree modulo 90°. Until then every wall is
 *  class-less; at that moment all of them are classified. (Defining θ₀ from the first wall seen made
 *  a chamfer the reference and every real wall a "chamfer".)
 *
 *  INFORMATION
 *  -----------
 *  `information` is the carried (φ, d) precision the solver applies as a prior each solve. It grows
 *  only when a window slot is DROPPED (room_gn_solver's absorb), never from the live window — the live
 *  slots' factors already enter the solve, so carrying them too would count them twice.
 */
#pragma once

#include <Eigen/Dense>
#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "wall_segmenter.h"

namespace rc::wallmap
{
    struct Params
    {
        float manhattan_sigma_rad = 2.0f * static_cast<float>(M_PI) / 180.f;  // σ_ε of the room↔wall factor
        float manhattan_off_prior = 0.1f;   // prior mass of "this wall obeys no class" (a chamfer)
        // ── MODEL error of a wall estimate: how far the same physical wall can appear between VIEWS
        // for reasons the white-noise model does not carry — multi-ring projection spread, the band
        // mixing heights on a non-ideal surface, residual calibration bias. Without it the innovation
        // covariance holds only sensor noise, and once a wall's carried precision reaches millimetres
        // EVERY re-observation fails the gate and births a DUPLICATE wall (measured live: 490 walls in
        // one piso run, ~400 of them voted twins of the real six). This is exactly the corner
        // detector's map_sigma lesson (corner_detector.h: "the map is a hypothesis, not ground
        // truth") applied to the walls' own channel. Enters association, candidate matching and the
        // wall↔wall merge — never the solver factors (σ_obs already absorbs residuals there).
        float map_sigma_d       = 0.04f;    // m
        float map_sigma_phi_rad = 1.0f * static_cast<float>(M_PI) / 180.f;
        float assoc_chi2  = 5.991f;         // χ²₂ @95% — segment↔wall and segment↔candidate gate ⚠
        float merge_chi2  = 5.991f;         // χ²₂ @95% — two walls statistically one ⚠
        float obs_sigma   = 0.05f;          // m — σ_obs of the wall point factor (RoomConcept.ObsSigma)
        float huber_delta = 0.15f;          // m — its Huber knee (RoomConcept.HuberDelta)
        float sensor_range = 15.f;          // m — extent of the uniform prior on d (Occam term)
        float birth_nats  = 4.605f;         // ln 100 — decisive Bayes factor ⚠ decision constant
        int   birth_min_frames = 2;         // a landmark needs a second view (tracker-only birth)
        int   max_candidates = 32;
        float publish_corner_sigma = 0.06f; // m — every derived corner must be this sharp to publish
        // ── Existence (the step-back operator): absence removes, occlusion holds — PER BIN ───────
        // Every beam whose ray crosses a wall's extent testifies about the BIN it crossed: a return ON
        // the line supports that bin, a return BEYOND it passed through and refutes it (weighted by
        // P(detect)), a return SHORT of it is occlusion and says nothing. Log-odds per 25 cm bin of
        // extent, clamped so a long-lived wall stays refutable; a bin dies below −birth_nats, dead END
        // bins SHRINK the extent (that is how a wall whose room changed keeps its surviving portion,
        // and how an overshot corner heals), and the wall dies only when NO bin remains — the same
        // decisive bar as birth, symmetric. Never a timer: a wall out of view holds its odds. This is
        // the residual layer's per-bin lesson applied one level up: whole-wall odds killed a REAL wall
        // whose extent had merely lost a stretch.
        float exist_refute_pdet = 0.5f;     // P(detect): weight of a pass-through vs a support ⚠
        float exist_bin_m       = 0.25f;    // m — extent bin width (spatial resolution of refutation)
        // Adjacency votes are leaky: v ← (1−leak)·v per frame, +1 when the corner is observed. A corner
        // that stops being seen loses its claim on the wall's end, which is how a structure change
        // (a new wall now meeting this end) can ever win against months of old evidence. Same idea and
        // magnitude as the corner detector's yield_leak (~2.5 s at 20 Hz). ⚠ a time constant.
        float vote_leak = 0.02f;
    };

    struct WallLandmark
    {
        std::uint64_t id = 0;
        int   k = -1;                       // Manhattan class, −1 ⇒ no room↔wall factor
        float phi = 0.f, d = 0.f;           // map frame; n(φ) points INTO the room
        Eigen::Matrix2f information = Eigen::Matrix2f::Zero();   // carried (φ, d) precision
        float manhattan_var = 0.f;          // σ_ε² in force for this wall's room factor (0 ⇒ off)
        float room_factor_dF = 0.f;         // converged cost of that factor, nats (diagnostic)
        bool  has_extent = false;
        float s_min = 0.f, s_max = 0.f;     // tangent coordinates of the observed extent
        // Adjacency votes per END: [0] = low end (s_min), [1] = high end (s_max). With n pointing into
        // the room the CCW walk runs s DECREASING, so a wall's low end meets its successor's high end.
        std::array<std::map<std::uint64_t, float>, 2> votes;
        int   frames_seen = 0;
        int   points_seen = 0;
        // Existence log-odds (nats), PER extent bin of width Params::exist_bin_m; bins_s0 is the
        // tangent coordinate of bin 0's lower edge. exist_lodds is the summary (max over bins) for
        // display and merge. Seeded at birth with birth_nats; a bin dies below −birth_nats; dead end
        // bins shrink the extent; no bins left ⇒ the wall dies. See Params above.
        float exist_lodds = 0.f;
        std::vector<float> exist_bins;
        float bins_s0 = 0.f;

        Eigen::Vector2f normal() const { return linefit::normal_of(phi); }
        Eigen::Vector2f tangent() const { return linefit::tangent_of(phi); }
        linefit::Line2D line() const { linefit::Line2D l; l.normal = normal(); l.d = d; return l; }
        /// Best-voted neighbour at an end, if any.
        std::optional<std::uint64_t> neighbour(int end) const;
    };

    /// A line no wall explains, accumulated across frames until the model comparison decides.
    struct Candidate
    {
        float phi = 0.f, d = 0.f;
        Eigen::Matrix2f information = Eigen::Matrix2f::Zero();
        int   frames = 0;
        int   npts = 0;
        float gain = 0.f;                   // Σ point_gain_nats of its points about the fused line (nats)
        float s_min = 0.f, s_max = 0.f;
        int   this_frame_seg = -1;          // segment index that updated it THIS frame (−1 none)
        std::int64_t first_ms = 0, last_ms = 0;
    };

    /// One slot's observation of one wall, in the ROBOT frame of that slot: the points the segment
    /// claimed, their observation weights, and the association posterior. Consumed by
    /// gn::WallPointFactor. Points are copied (not indexed) because old slots are subsampled.
    struct WallAssoc
    {
        std::uint64_t wall_id = 0;
        Eigen::Matrix<float, Eigen::Dynamic, 2> pts;
        Eigen::VectorXf weights;            // range/incidence weights, mean 1 over the slot
        float pda = 1.f;                    // association posterior
    };

    struct Corner
    {
        std::uint64_t wall_a = 0, wall_b = 0;   // a = the wall whose HIGH end this is, b = its successor
        Eigen::Vector2f p = Eigen::Vector2f::Zero();
        bool  inferred = false;             // never observed meeting; intersected from the two lines
        float sigma = 0.f;                  // largest σ of the corner position (m), from the walls' information
    };

    struct Polygon
    {
        std::vector<Eigen::Vector2f> verts;      // CCW; edge i runs from verts[i] to verts[i+1]
        std::vector<std::uint64_t>   wall_of_edge;
        std::vector<Corner>          corners;    // corners[i] is verts[i]
        bool closed = false;
        bool publishable = false;
        float worst_corner_sigma = 0.f;
        std::string status;                      // human-readable: why not closed/publishable
    };

    /// Why a wall was just born — the line to read when the map misbehaves.
    struct BirthInfo
    {
        std::uint64_t id = 0;
        float phi = 0.f, d = 0.f;
        int   npts = 0, frames = 0;
        float dF = 0.f;
        std::uint64_t nearest_wall = 0;     // closest EXISTING wall at birth time…
        float nearest_chi2 = -1.f;          // …and how decisively the gate refused it (χ²)
    };

    struct FrameResult
    {
        std::vector<int>   seg_to_wall;     // per segment: index into walls, or −1
        std::vector<float> seg_pda;
        std::vector<WallAssoc> assoc;       // for the newest slot (robot frame)
        int births = 0;
        std::vector<BirthInfo> births_info;
        struct DeathInfo { std::uint64_t id = 0; float lodds = 0.f; int frames_seen = 0; int points_seen = 0; };
        int deaths = 0;
        std::vector<DeathInfo> deaths_info;
        int twins_fused = 0;                // candidates that turned out to BE an existing wall
        int segments_associated = 0;
        int merged = 0;
        int candidates = 0;
        int unexplained_points = 0;
        std::vector<int> seg_to_candidate; // per segment: candidate index or −1 (display)
    };

    class WallMap
    {
    public:
        Params params;

        bool  theta0_born = false;
        float theta0 = 0.f;
        float theta0_information = 0.f;
        std::vector<WallLandmark> walls;
        std::vector<Candidate>    candidates;

        /// The per-frame pipeline: associate this frame's segments to walls (Mahalanobis, Hungarian,
        /// PDA), record extents and adjacency votes, feed the rest to the candidates and let the model
        /// comparison bear walls. `pts_robot` are the points the segmenter ran on; `weights` their
        /// observation weights (empty ⇒ ones). `pose`/`pose_cov` is the slot's CURRENT estimate.
        FrameResult observe(const wallseg::Result& seg, const std::vector<Eigen::Vector2f>& pts_robot,
                            const Eigen::VectorXf& weights, const Eigen::Vector3f& pose,
                            const Eigen::Matrix3f& pose_cov, std::int64_t timestamp_ms);

        /// Fuse walls that have become statistically indistinguishable. Returns how many.
        int merge_indistinguishable();

        /// Derive the polygon from adjacency + intersection. Never throws; `status` says what is missing.
        Polygon build_polygon() const;

        /// One-shot gauge change: p' = R(−rot)·(p − c). Transforms every wall, candidate and θ₀.
        void reanchor(const Eigen::Vector2f& c, float rot);

        const WallLandmark* find(std::uint64_t id) const;
        WallLandmark*       find(std::uint64_t id);
        int index_of(std::uint64_t id) const;

        /// Manhattan class of an angle w.r.t. θ₀ and the residual to it. k = −1 with the off-class cost
        /// when no class is decisive against the "obeys no class" alternative.
        struct ClassChoice { int k = -1; float eps = 0.f; float cost = 0.f; };
        ClassChoice classify(float phi) const;

        /// Corner position and its σ from two walls' lines and information (public for the harness).
        static Corner intersect_walls(const WallLandmark& a, const WallLandmark& b, bool inferred);

        /// Segment (robot frame) → map-frame (φ, d) and the 2×3 Jacobian w.r.t. the pose.
        static void to_map(float phi_r, float d_r, const Eigen::Vector3f& pose,
                           float& phi_m, float& d_m, Eigen::Matrix<float, 2, 3>& H);

    private:
        std::uint64_t next_id_ = 1;
        int  birth_from_candidate(int ci, std::int64_t ts);
        void vote_adjacency(int wa, int wb, const Eigen::Vector2f& corner_map);
        /// Try to define θ₀ from two walls that agree modulo 90°; on success classify every wall.
        void try_birth_theta0();
        /// The step-back operator: update every wall's existence log-odds from this frame's beams
        /// (support / pass-through / occluded) and REMOVE the decisively refuted ones. Runs before
        /// association so nothing this frame holds an index into a dead wall.
        void update_existence(const std::vector<Eigen::Vector2f>& pts_robot, const Eigen::Vector3f& pose,
                              FrameResult& fr);
        void reclassify_all();
    };
} // namespace rc::wallmap
