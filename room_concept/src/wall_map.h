/*
 *  wall_map.h — the room as a CLOSED POLYGON MODEL: a cyclic order of wall landmarks, refined by
 *  evidence and changed only by splice JUMPS (model-first redesign, 2026-08-31).
 *
 *  WHAT THE MODEL IS
 *  -----------------
 *  Not a bag of walls. The state is an ordered CCW cycle of wall landmarks — Hesse lines
 *  n(φ)·p = d, the 2-D analogue of S-Graphs+ plane landmarks (Bavle et al., arXiv 2212.11770) —
 *  whose consecutive intersections ARE the polygon. Closure, contiguity and simplicity are
 *  properties of the state space, not tests applied afterwards. The first version kept a free wall
 *  set and derived the polygon from adjacency votes; live it grew hundreds of junk landmarks,
 *  because a free-floating wall costs nothing and the structural constraints never pushed back on
 *  the evidence. Here there is nowhere to put junk: every wall is an edge of the room.
 *
 *  INITIALISATION IS A PRIOR ON SHAPE AND SIZE
 *  -------------------------------------------
 *  The first scan's oriented bounding box seeds a RECTANGLE (initialize_rect): four walls, cyclic
 *  order, θ₀ from the box. rect_prior_sigma_* is that prior's strength — strong enough to anchor
 *  the yaw from frame 1 (with nothing absolute, odometry yaw drift re-bred rotated copies of every
 *  wall: the hairball), weak enough that the data moves every side.
 *
 *  EVIDENCE, LIKELIHOOD, PRIOR — as in every concept agent
 *  ------------------------------------------------------
 *  Segments (wall_segmenter) associate to the model's edges by a Mahalanobis test on (φ, d) that
 *  includes segment noise, edge uncertainty, pose uncertainty and a model-error term (map_sigma —
 *  the corner detector's lesson: without it a converged edge refuses its own re-observations and
 *  twins are born). Associated points become the solver's wall factors; the hierarchical Manhattan
 *  prior (θ₀ + per-edge class) constrains angles softly — a chamfer keeps k = −1 and no factor.
 *
 *  STRUCTURE CHANGES ARE JUMPS, NOT BIRTHS
 *  ---------------------------------------
 *  Unexplained segments accumulate as CANDIDATES; a candidate that earns a decisive ΔF (clutter
 *  comparison + Occam + class cost, > birth_nats, seen from ≥2 poses) proposes a SPLICE:
 *    - notch/extension on a host edge E: [E, jog, C, jog, E] (E appears twice; the room grows an
 *      alcove, gains the second SPACE seen through a wide opening, or loses a pillar-sized bite);
 *    - corner cut with a neighbour: [E, jog, C, N] or [P, C, jog, E] — how an OBB rectangle becomes
 *      the L-shaped truth — and obliquely without the jog: [E, C, N] (a chamfer, k = −1).
 *  A variant commits only if the resulting cycle builds a simple CCW polygon. No valid splice ⇒ no
 *  change, counted in splice_rejected — a refusal must never be mistaken for "never asked".
 *
 *  THE STEP-BACK OPERATOR (existence, per extent bin)
 *  --------------------------------------------------
 *  Every beam crossing an edge's extent testifies about the BIN it crossed: ON supports, BEYOND
 *  refutes (×P(detect)), SHORT is occlusion and holds. Dead end bins shrink the extent; an edge
 *  whose testable span is gone dies at −birth_nats — symmetric with birth, never a timer — and is
 *  spliced OUT, the cycle healing by collapsing parallel neighbours. (Whole-wall odds once killed a
 *  real wall that had merely lost a stretch: the residual layer's per-bin lesson, one level up.)
 *
 *  `information` is the carried (φ, d) precision the solver applies as a prior each solve. It grows
 *  only when a window slot is DROPPED (gn::absorb_wall_observations) — the live window's factors
 *  already enter the solve, so carrying them too would count them twice.
 */
#pragma once

#include <Eigen/Dense>
#include <cstdint>
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
        // ── MODEL error of an edge estimate between VIEWS (multi-ring spread, band mixing heights,
        // residual calibration). Without it the innovation holds only white noise, and once an edge's
        // carried precision reaches millimetres every re-observation fails the gate (measured live:
        // 490 walls from six, all twins). The corner detector's map_sigma lesson. Enters association,
        // candidate matching and the merge — never the solver factors.
        float map_sigma_d       = 0.04f;    // m
        float map_sigma_phi_rad = 1.0f * static_cast<float>(M_PI) / 180.f;
        float assoc_chi2  = 5.991f;         // χ²₂ @95% — segment↔edge and segment↔candidate gate ⚠
        float merge_chi2  = 5.991f;         // χ²₂ @95% — two edges statistically one ⚠
        float obs_sigma   = 0.05f;          // m — σ_obs of the wall point factor (RoomConcept.ObsSigma)
        float huber_delta = 0.15f;          // m — its Huber knee (RoomConcept.HuberDelta)
        float sensor_range = 15.f;          // m — extent of the uniform prior on d (Occam term)
        float birth_nats  = 4.605f;         // ln 100 — decisive Bayes factor ⚠ decision constant
        int   birth_min_frames = 2;         // a jump needs a second view (tracker-only birth)
        int   max_candidates = 32;
        float publish_corner_sigma = 0.06f; // m — every derived corner must be this sharp to publish
        // ── Model-first initialisation: the OBB rectangle prior on shape and size ────────────────
        // HONEST about what an OBB knows: on a non-convex cloud (an L) PCA tilts the box by 15-20°,
        // and a 5° prior then REFUSED the true walls' segments at the gate — the sides never rotated
        // onto the truth and the real walls were spliced in as extra edges instead (harness-caught).
        float rect_prior_sigma_d       = 0.50f;  // m — per-side offset prior strength
        float rect_prior_sigma_phi_rad = 15.0f * static_cast<float>(M_PI) / 180.f;
        // A candidate span end within this of a host edge's end "reaches the corner" — a corner-cut
        // splice instead of a notch. ⚠ a tolerance, tied to extent noise at range.
        float splice_end_tol = 0.5f;        // m
        // Stub jumps (a free-standing interior wall the boundary wraps around: [E, face, tip,
        // mirror-face, E]). OFF until jump selection is a GLOBAL free-energy comparison: with local
        // scoring the stub ties a notch's corner-cut on overlap and the tie-breaks are fragile
        // (measured: return-slivers on the notch test). The concavity MISSING from the estimate is
        // what the IoU metric is there to show meanwhile.
        bool  enable_stub_jumps = false;
        // ── Existence (the step-back operator), per extent bin — see the header comment ──────────
        float exist_refute_pdet = 0.5f;     // P(detect): weight of a pass-through vs a support ⚠
        float exist_bin_m       = 0.25f;    // m — extent bin width (spatial resolution of refutation)
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
        int   frames_seen = 0;
        int   points_seen = 0;
        std::int64_t last_seen_ms = 0;      // last frame a segment associated to this edge
        // Existence log-odds (nats), PER extent bin of width Params::exist_bin_m; bins_s0 is bin 0's
        // lower edge. exist_lodds is the summary (max over bins). Seeded at birth with birth_nats; a
        // bin dies below −birth_nats; dead END bins shrink the extent; no testable span ⇒ death.
        float exist_lodds = 0.f;
        std::vector<float> exist_bins;
        float bins_s0 = 0.f;

        Eigen::Vector2f normal() const { return linefit::normal_of(phi); }
        Eigen::Vector2f tangent() const { return linefit::tangent_of(phi); }
        linefit::Line2D line() const { linefit::Line2D l; l.normal = normal(); l.d = d; return l; }
    };

    /// A line no edge explains, accumulated across frames until the model comparison proposes a jump.
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

    /// One slot's observation of one edge, in the ROBOT frame of that slot. Consumed by
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
        std::uint64_t wall_a = 0, wall_b = 0;   // consecutive edges of the order
        Eigen::Vector2f p = Eigen::Vector2f::Zero();
        bool  inferred = false;             // kept for the viewer contract
        float sigma = 0.f;                  // largest σ of the corner position (m), from the edges' information
    };

    struct Polygon
    {
        std::vector<Eigen::Vector2f> verts;      // CCW; edge i runs from verts[i] to verts[i+1]
        std::vector<std::uint64_t>   wall_of_edge;
        std::vector<Corner>          corners;    // corners[i] is verts[i]
        bool closed = false;
        bool publishable = false;
        std::vector<int> crossing_edges;         // edge indices involved in self-crossings (repair input)
        float worst_corner_sigma = 0.f;
        std::string status;                      // human-readable: why not closed/publishable
    };

    /// Why a jump was just committed — the line to read when the map misbehaves.
    struct BirthInfo
    {
        std::uint64_t id = 0;               // the new C edge
        float phi = 0.f, d = 0.f;
        int   npts = 0, frames = 0;
        float dF = 0.f;
        std::uint64_t nearest_wall = 0;     // host edge of the splice
        float nearest_chi2 = -1.f;          // (CSV contract; −1 when not computed)
        int   seg = -1;                     // segment index that completed the jump (z attribution)
    };

    struct FrameResult
    {
        std::vector<int>   seg_to_wall;     // per segment: index into walls, or −1
        std::vector<float> seg_pda;
        std::vector<WallAssoc> assoc;       // for the newest slot (robot frame)
        int births = 0;                     // committed splices
        std::vector<BirthInfo> births_info;
        int splice_rejected = 0;            // qualified candidates no valid splice could place
        struct DeathInfo { std::uint64_t id = 0; float lodds = 0.f; int frames_seen = 0; int points_seen = 0; };
        int deaths = 0;
        std::vector<DeathInfo> deaths_info;
        int twins_fused = 0;                // candidates that turned out to BE an existing edge
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
        // ── FREE-SPACE EVIDENCE: a coarse log-odds grid the scans build directly ─────────────────
        // Every beam traverses free space and ends on matter; that is observed, not inferred. The
        // grid answers two questions the wall set cannot: WHERE the room's free region actually is
        // (the global re-derivation traces its contour), and where the FRONTIERS are (free cells
        // touching unknown — the epistemic explorer's targets, and the honest "am I done" test).
        struct FreeGrid
        {
            // 8 cm: a cell must be smaller than the thinnest wall the model must keep, or grazing
            // traversals through partially-occupied cells outvote endpoint hits and the beams CARVE
            // THROUGH thin interior walls (measured: 38 of 42 spur-face cells marked free at 15 cm).
            float cell = 0.08f;
            float x0 = 0.f, y0 = 0.f;
            int nx = 0, ny = 0;
            std::vector<float> lodds;             // + occupied, − free, 0 unknown; clamped ±4
            // Endpoint returns per cell, counted SEPARATELY from the odds: a return localises matter
            // in the cell near-certainly, while a traversal of a partially occupied cell (a thin wall
            // shares its cell with air) is weak and explicable. A saturating scalar is ORDER-DEPENDENT
            // — a cell driven deep-free by corridor traffic before its first frontal view could never
            // climb back — so returns are kept as their own evidence and three of them assert matter
            // regardless of how many beams grazed past.
            std::vector<unsigned short> hits;
            bool ready() const { return nx > 0; }
            int  idx(int i, int j) const { return j * nx + i; }
            bool in(int i, int j) const { return i >= 0 and i < nx and j >= 0 and j < ny; }
            Eigen::Vector2f at(int i, int j) const { return {x0 + (i + 0.5f) * cell, y0 + (j + 0.5f) * cell}; }
            void init(const Eigen::Vector2f& centre, float half_span);
            void mark(const Eigen::Vector2f& origin, const std::vector<Eigen::Vector2f>& pts_map);
            bool is_occupied(int i, int j) const
            { return in(i, j) and (hits[static_cast<size_t>(idx(i, j))] >= 3 or lodds[static_cast<size_t>(idx(i, j))] > 1.5f); }
            bool is_free(int i, int j) const
            { return in(i, j) and lodds[static_cast<size_t>(idx(i, j))] < -1.f and hits[static_cast<size_t>(idx(i, j))] < 3; }
            bool is_unknown(int i, int j) const
            { return in(i, j) and std::abs(lodds[static_cast<size_t>(idx(i, j))]) < 0.5f and hits[static_cast<size_t>(idx(i, j))] < 3; }
        };
        FreeGrid fgrid;
        /// Free cells adjacent to unknown cells — where the map still has questions (map frame).
        std::vector<Eigen::Vector2f> frontiers() const;
        /// GLOBAL re-derivation: trace the free region's contour, snap its runs to the evidence
        /// lines (walls ∪ candidates, Manhattan preferred), and ADOPT the resulting cycle iff it
        /// explains the observed free space better than the current one. The escape hatch from a
        /// wrong local topology that greedy jumps cannot leave (measured: a 107k-point wall homeless
        /// through 5140 rejections). Returns true when the cycle was swapped.
        bool re_derive(const Eigen::Vector2f& robot_map);

        // THE MODEL: wall ids in cyclic CCW order. A wall may appear twice (a notch splits its host
        // line into two runs). The polygon is these edges intersected in order; there is no other
        // topology state — no votes, no walk. Public so the harness can wound it.
        std::vector<std::uint64_t> order;

        /// Model-first initialisation: a CCW quad (the scan's OBB) — the prior on room shape and
        /// size. θ₀ is born from it; everything after is refinement plus splice jumps.
        void initialize_rect(const std::vector<Eigen::Vector2f>& rect_ccw);

        /// The per-frame pipeline: existence first (step-back), then segment↔edge association
        /// (many-to-one, Mahalanobis+PDA), candidate accumulation, qualified candidates proposing
        /// splices. `pose`/`pose_cov` is the slot's CURRENT estimate; weights empty ⇒ ones.
        FrameResult observe(const wallseg::Result& seg, const std::vector<Eigen::Vector2f>& pts_robot,
                            const Eigen::VectorXf& weights, const Eigen::Vector3f& pose,
                            const Eigen::Matrix3f& pose_cov, std::int64_t timestamp_ms);

        /// Fuse edges that have become statistically indistinguishable. Returns how many.
        int merge_indistinguishable();

        /// The model's polygon: the ordered edges intersected. Never throws.
        Polygon build_polygon() const;
        Polygon build_from(const std::vector<std::uint64_t>& ord) const;

        /// One-shot gauge change: p' = R(−rot)·(p − c). Transforms every edge, candidate and θ₀.
        void reanchor(const Eigen::Vector2f& c, float rot);

        const WallLandmark* find(std::uint64_t id) const;
        WallLandmark*       find(std::uint64_t id);
        int index_of(std::uint64_t id) const;

        struct ClassChoice { int k = -1; float eps = 0.f; float cost = 0.f; };
        ClassChoice classify(float phi) const;

        static Corner intersect_walls(const WallLandmark& a, const WallLandmark& b, bool inferred);

        /// Segment (robot frame) → map-frame (φ, d) and the 2×3 Jacobian w.r.t. the pose.
        static void to_map(float phi_r, float d_r, const Eigen::Vector3f& pose,
                           float& phi_m, float& d_m, Eigen::Matrix<float, 2, 3>& H);

    private:
        std::uint64_t next_id_ = 1;
        /// A qualified candidate becomes a JUMP: a notch on a host edge, or a corner cut with a
        /// neighbour. Committed only if the resulting cycle builds a simple CCW polygon. Returns the
        /// new C edge's index in walls, or −1 (rejected — counted, never silent).
        int  try_splice(const Candidate& c, FrameResult& fr, std::int64_t ts);
        /// Remove a dead edge from the order and HEAL the cycle (collapse parallel neighbours).
        void splice_out(std::uint64_t id);
        void heal_order();
        /// A self-crossing cycle is IMPOSSIBLE — persistent crossing is itself decisive refutation of
        /// the weakest edge involved (a wrong commit that beams alone cannot undo: they support it
        /// obliquely from the real walls it was mis-spliced onto). Counts consecutive crossing frames
        /// and removes the least-supported crossing edge after a short persistence.
        void repair_if_crossing();
        int  crossing_frames_ = 0;
        WallLandmark make_wall(float phi, float d, const Eigen::Matrix2f& info, float exist_seed,
                               std::int64_t ts);
        /// Give NEW (extent-less) edges their extent from the polygon they now bound.
        void seed_extents_from_polygon();
        void reclassify_all();
        void update_existence(const std::vector<Eigen::Vector2f>& pts_robot, const Eigen::Vector3f& pose,
                              FrameResult& fr);
    };
} // namespace rc::wallmap
