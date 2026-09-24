#pragma once
// ─────────────────────────────────────────────────────────────────────────────────────────────
//  THE BOX LAYOUT CHANNEL — the redesign of room_boxes.h wired into the live agent
//
//  Deliberately a SIDECAR. It consumes exactly what the bench consumes (a robot-frame scan, a
//  map-frame pose, that pose's covariance) and produces exactly what the bench scores (a polygon).
//  It reaches into nothing: no window, no solver, no wall map, no re-anchor. The existing wall-SLAM
//  layout keeps running untouched beside it, so turning `BoxLayout` on can change WHAT IS PUBLISHED
//  and nothing else, and turning it off restores the shipped behaviour exactly.
//
//  ★ WHY A SIDECAR AND NOT A REPLACEMENT. The live agent currently holds a 6.00 x 4.00 rectangle to
//    2-3 mm while driving ([[rectangle-survives-driving-four-fixes]], d307c84). That result was
//    expensive and is not being staked on an estimator whose ladder has open rows. Running both and
//    publishing one is how the comparison gets made on the robot instead of being argued about.
//
//  ⚠ THE BENCH AND THE AGENT MUST SHARE THE SAME SOURCE FILE, and they do — room_boxes.cpp is in
//    both targets. This is not a style preference: on 2026-09-19 the bench passed while the agent
//    failed for six minutes purely because bin/room_concept was older than the fix, and "the bench
//    proves something about the agent" is only true when there is one implementation.
// ─────────────────────────────────────────────────────────────────────────────────────────────

#include <Eigen/Dense>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "room_boxes.h"

namespace rc::boxch
{
    struct Params
    {
        bool  enabled      = false;   ///< publish the box polygon instead of the wall-map one
        int   every_frames = 40;      ///< how often structure is re-examined (COMPUTE, not a bar)
        float sensor_sigma = 0.02f;   ///< PHYSICAL: LiDAR range noise perpendicular to a wall
        float sigma_flat   = 0.010f;  ///< PHYSICAL: wall flatness / line scatter between views
        /// COMPUTE: voxel + clustering resolution. ⚠ IT MUST RESOLVE THE SMALLEST FEATURE THAT
        /// MATTERS. apartamento_layout.svg has 0.12 m fins standing on its centreline; at the old
        /// 0.10 m a fin was barely one cell wide and could not be separated from the noise around
        /// it. This is a RESOLUTION, not a tuning knob: it is set by the geometry to be recovered,
        /// and the cost is quadratic in cells, so it is not free.
        float cell         = 0.05f;
        int   min_cluster  = 12;      ///< COMPUTE: smallest cluster worth proposing
        float z_min        = 0.20f;   ///< PHYSICAL: floor rejection, in the robot's own frame
        /// PHYSICAL: the robot's passable half-width, from ROBOT_GEOMETRY.md's derived footprint
        /// (`WebotsProtoLoader::xy_hull()`: area 0.2182 m², inscribed 0.2300, circumscribed 0.3278).
        /// The planner may only route where a body of this radius fits, so a gap narrower than
        /// 2 x this is not a corridor. ⚠ EROSION IS AGAINST KNOWN OCCUPANCY ONLY, never against
        /// unknown space — a frontier cell is adjacent to unknown BY DEFINITION, so eroding against
        /// unknown forbids ever approaching the thing exploration exists to look at.
        /// ★ Using the INSCRIBED radius says "it fits going straight"; the circumscribed 0.3278
        /// would say "it fits at any heading, so it can also turn in place there". That is a
        /// behavioural choice ROBOT_GEOMETRY.md flags as wanting its own comparison; this is the
        /// permissive end, and it is the number to change if the robot clips corners in the real
        /// world.
        float body_radius  = 0.230f;
        /// ── THE OBSTACLE BAND: EVERYTHING BELOW THE WALL BAND ──────────────────────────────
        /// PHYSICAL, robot frame. Returns here are FURNITURE, not walls: the wall band is above
        /// them (z_min), which is how the robot separates the two — the high LiDAR sees over a
        /// table, the low one sees the table. They are recorded for NAVIGATION ONLY (obs_), never
        /// as evidence about the room. Measured in the bench: one band, with furniture on the
        /// floor, fits the furniture's faces as walls and loses the robot — IoU 0.543 against
        /// 0.983 with the bands separated.
        // ── MODEL CHOICES VALIDATED IN THE BENCH (were WS_* environment flags) ──────────────
        /// Fit the layout frame by the layout's own likelihood instead of the segment vote.
        bool  gauge_ml        = false;
        /// Build the free-space cover on the LAYOUT grid, not the map grid (else every proposed
        /// box is aligned with the map axes and the judge rewards a frame of exactly 0 degrees).
        bool  cover_layout    = false;
        /// COMPUTE budget of the greedy cover: how many rectangles it may propose. Not a model
        /// statement — MDL already refuses a box that does not pay for itself.
        int   cover_max_rects = 24;
        /// Refuse a candidate cover that would make the region more disconnected than it is. ⚠ A
        /// TRADE, not a win: the apartment gains (mean .887->.917, worst .473->.705) and the 50
        /// simple rooms lose at the tail (.969/.901 -> .956/.772). Off until it is priced in nats.
        bool  connected       = false;
        /// A one-scan seed's offsets carry the room's span, not sigma_flat: a box drawn round one
        /// scan is not known to a centimetre. Pairs with GrowParams::free_force and
        /// RegisterOptions::map_var.
        bool  seed_prior_span = false;
        /// Pass RegisterOptions::map_var when this channel registers (reproject).
        bool  reg_map_var     = false;
        /// Swept space pushes a face that excludes it — forwarded to GrowParams::free_force.
        bool  free_force      = false;

        float obs_z_min    = 0.15f;
        float obs_z_max    = 1.45f;
        float z_max        = 1.60f;   ///< PHYSICAL: below the ceiling — see
                                      ///  [[lidar-high-band-must-not-reach-ceiling]], where ceiling
                                      ///  returns in a 2-D wall SDF stopped a room stabilising while
                                      ///  the pose looked perfectly healthy.
    };

    /// The channel's own state. One instance lives in RoomConcept; nothing else may touch it.
    class Channel
    {
    public:
        void configure(const Params& p) { p_ = p; }
        const Params& params() const { return p_; }
        bool enabled() const { return p_.enabled; }

        /// ── WHERE SHOULD THE ROBOT GO NEXT? ─────────────────────────────────────────────────
        /// A path, in MAP-frame metres, to the viewpoint that most reduces what the layout does not
        /// know. The belief is the free-space map this channel already maintains: a cell is FREE
        /// (a beam went through it), OCCUPIED (a return landed there) or UNKNOWN. A frontier — a
        /// free cell touching unknown space — is where new information can actually be obtained,
        /// and the gain of standing there is the unknown area it would reveal.
        ///
        /// score = unknown cells within sensor horizon / (1 + path length)
        ///
        /// ⚠ THE PATH IS PLANNED THROUGH FREE CELLS, never as a straight line. The bench's fixed
        /// tours were validated against the TRUE polygon — information a robot does not have — and
        /// on the apartamento hall, whose two fins stand on the centreline, a straight run between
        /// two sensible points goes through a wall. Planning over the free map instead uses only
        /// what has been observed, which is the whole point of making the robot choose.
        ///
        /// Returns an empty path when no frontier is left: that is the honest termination
        /// condition for exploration, and it is what a frame cap has been standing in for.
        /// ★ An objective cannot be graded on an endpoint its own budget saturated —
        /// 594/594 runs of the previous explorer hit the cap and it measured NULL.
        /// ── TWO PHASES, BECAUSE COVERAGE IS A PRECONDITION AND PRECISION IS THE GOAL ────────
        /// EXPLORE  maximise unknown area revealed per metre — the frontier objective. It answers
        ///          "where has nobody been", which is occupancy entropy.
        /// REFINE   once there is no frontier, maximise the information gained about the WALL
        ///          OFFSET THAT IS WORST KNOWN. That is the estimand; occupancy never was.
        ///
        /// ★ The fixed tour this has to match drives TWO laps. The second lap reveals no new area
        ///   whatsoever — it contributes PRECISION, by seeing every wall again from new angles.
        ///   A pure frontier planner stops at the end of lap one and collects about half the data,
        ///   which is why it cannot reach the same layout however good its coverage is.
        ///
        /// Information about a face offset from a viewpoint goes as cos^2(incidence)/range: a wall
        /// seen edge-on says almost nothing about where it is. So REFINE prefers standing square
        /// to the worst-known wall at moderate range, which is also what keeps registration
        /// healthy — and pose error is what drives layout error
        /// ([[free-space-cost-must-be-symmetric]]: 0.356 m rms odometry-only vs 0.019 registered).
        enum class Phase { Explore, Refine, Done };
        Phase phase() const { return phase_; }
        const char* phase_name() const
        { return phase_ == Phase::Explore ? "explore" : phase_ == Phase::Refine ? "refine" : "done"; }

        std::vector<Eigen::Vector2f> plan_path(const Eigen::Vector2f& from_map, float horizon_m = 4.0f) const;

        /// ── CAN A BODY STAND HERE, IN THE ROBOT'S OWN BELIEF? ───────────────────────────────
        /// The same configuration-space predicate `plan_path` routes on, exposed so a path
        /// FOLLOWER can validate the curve it actually intends to drive. `RouteSpline` smooths
        /// across the planner's polyline and is APPROXIMATING, not interpolating, so the smoothed
        /// curve can leave the corridor the planner certified — near a 0.12 m fin that matters.
        /// ⚠ THE ANSWER MUST COME FROM THE MAP, NEVER FROM THE TRUE ROOM. A follower handed the
        /// truth polygon is an oracle, and it would look excellent for the same reason the
        /// true-pose corner channel did. This reads `free_` and `vmap_` — swept floor, and no
        /// KNOWN occupancy within `body_radius` — so it is exactly as wrong as the robot is.
        /// `m` is a MAP-frame point (the frame `plan_path` speaks).
        /// Signed distance (m) from a MAP-frame point to the believed layout boundary: negative
        /// inside the room. The robot's own belief, for a caller that must route by it (the bench's
        /// route optimiser, as controller_session routes by its GridPlanner EDT). +max if no layout.
        float model_sdf(const Eigen::Vector2f& m) const;
        /// WS_COVER_PROBE: connected components of the union (polygon() publishes only the first).
        int components() const;
        /// ── OBSTACLES ARE FOR NAVIGATION ONLY ───────────────────────────────────────────────
        /// The low LiDAR band sees floor furniture; the wall band sees over it. Low-band returns
        /// (robot-frame points at map-frame `pose`) are recorded HERE and nowhere else: plan_path()
        /// treats them as occupancy and obstacle_clearance() serves the route optimiser, but they
        /// never enter vmap_, free_, the layout, registration or the gauge. That is how the robot
        /// works — a table is something to drive around, not a wall to fit.
        void observe_obstacles(const std::vector<Eigen::Vector2f>& pts_robot, const Eigen::Vector3f& pose);
        /// Same, from the 3-D scan: keeps [obs_z_min, obs_z_max] — the band BELOW the wall band.
        void observe_obstacles(const std::vector<Eigen::Vector3f>& pts_robot, const Eigen::Vector3f& pose);
        /// Distance (m) from a MAP-frame point to the nearest recorded obstacle cell, searched
        /// within `horizon`; returns `horizon` when none is closer.
        float obstacle_clearance(const Eigen::Vector2f& m, float horizon = 1.5f) const;
        bool traversable(const Eigen::Vector2f& m) const;
        /// The swept cells as a set, for refit's free-space term (see rc::boxes::refit).
        void refresh_free_keys() const;
        /// Worst face variance, in metres — the quantity REFINE is driving down, and the honest
        /// stopping signal: when it stops improving there is nothing left to learn about shape.
        float worst_face_sigma() const { return worst_face_sigma_; }
        /// Expected information about the layout's offsets, in NATS, of the viewpoint the last
        /// plan chose — the epistemic half of G(v). Zero when the rate heuristic is driving.
        /// ★ THE FALSIFIER LIVES HERE: log this against the REALISED change in the estimator's
        /// total Fisher information on arrival and regress. A slope far from 1 means the currency
        /// is fiction and no lambda can rescue an objective whose value is made up.
        float last_info_nats() const { return last_info_nats_; }
        /// Raw predicted Fisher gain (sum of dH_k) of the chosen viewpoint — the quantity to
        /// regress against the REALISED change in info_total(). Same units on both sides.
        float last_dH_sum() const { return last_dH_sum_; }
        /// Unexplained residual, in nats, on the faces the chosen viewpoint would see — the
        /// structural half of the objective, and the only term voxel condensation cannot erase.
        float last_misfit_nats() const { return last_misfit_nats_; }
        /// Total Fisher information the estimator currently holds over all offsets, sum of 1/var.
        /// The realised side of that regression.
        double info_total() const
        {
            double t = 0.0;
            if (L_.cov.rows() != static_cast<long>(L_.n_offsets())) return t;
            for (long i = 0; i < L_.cov.rows(); ++i)
                if (L_.cov(i, i) > 0.f and std::isfinite(L_.cov(i, i))) t += 1.0 / L_.cov(i, i);
            return t;
        }
        /// Unknown cells the last plan expected to reveal — the gain it was chosen for.
        int last_gain() const { return last_gain_; }
    private:
        mutable int last_gain_ = 0;
        mutable Phase phase_ = Phase::Explore;
        mutable float worst_face_sigma_ = 1e9f;
        mutable float last_info_nats_ = 0.f;
        mutable float last_dH_sum_ = 0.f;
        mutable float last_misfit_nats_ = 0.f;
        /// The pose covariance the last scan was folded with — the prediction needs it to say what
        /// sigma a return at a given RANGE would be captured at.
        Eigen::Matrix3f last_cov_ = Eigen::Matrix3f::Identity() * 0.01f;
        mutable int refine_stall_ = 0;
    public:

        /// Fold one scan in. `pts_robot` is the robot-frame scan, `pose` the map-frame robot pose,
        /// `cov` that pose's 3x3 covariance. Cheap: a voxel update per point, no solve.
        /// `seg_phi` / `seg_len` are the agent's OWN wall segments for this scan: normal angle in
        /// the robot frame, and extent in metres. They carry the Manhattan vote.
        /// ⚠ THE VOTE MUST NOT READ CONSECUTIVE POINT DIFFERENCES. init_from_scan() does exactly
        /// that — it is the right estimator for a synthesised scan, which arrives in azimuthal
        /// order — but the agent's returns come through the media plane and a z-band filter, so
        /// consecutive entries are chords between unrelated returns. Measured live: the vote
        /// converged to +56.18 deg over 680 frames with 0.275 deg of drift, on a room whose true
        /// angle is ~0. ★ Averaging cannot rescue a BIASED estimator, only a noisy one, and a
        /// tight confidence interval around a wrong number is the most expensive kind of wrong.
        void observe(const std::vector<Eigen::Vector3f>& pts_robot,
                     const Eigen::Vector3f& pose, const Eigen::Matrix3f& cov,
                     const std::vector<float>& seg_phi, const std::vector<float>& seg_len);

        /// ── THE MAP FRAME MOVED ──────────────────────────────────────────────────────────────
        /// One-shot gauge change p' = R(-rot)(p - c), the same one RoomConcept::reanchor_map_frame
        /// applies to the window poses, the model tensors, the covariance and the wall map.
        /// ⚠ THE SIDECAR LIVES IN THE MAP FRAME AND MUST MOVE WITH IT. Its voxel map accumulates
        /// over the whole session, so a re-anchor that skips it leaves every past return stale by
        /// a rigid transform while new ones arrive in the new frame — TWO OFFSET COPIES OF THE
        /// SAME ROOM in one cloud. Measured live: 32.28% of cells more than 3 sigma outside the
        /// region and 31.89% more than 3 sigma inside, a near-symmetric split that no real room
        /// geometry produces, followed by extrusions chasing the ghost copy.
        /// ★★★ THE BENCH CANNOT REPRODUCE THIS. WS_BOXES runs odometry-only with no wall map, so
        /// it never re-anchors; the event does not exist there. That is the whole answer to "how
        /// can it work in the bench and not here" — the bench had no such event to get wrong.
        void reanchor(const Eigen::Vector2f& c, float rot);

        /// Re-examine the structure (and create the layout on the first call that can).
        /// Call once per frame; it self-throttles to `every_frames`.
        /// Returns true if the layout changed.
        bool step();

        /// The current layout, empty until the first scan has been folded in.
        const rc::boxes::Layout& layout() const { return L_; }
        /// The outer boundary, CCW, IN THE AGENT'S MAP FRAME, or empty if there is no layout yet.
        std::vector<Eigen::Vector2f> polygon() const;
        /// The layout's own rectilinear frame, as a rotation from the agent's map frame.
        float yaw() const { return yaw_; }

        // ── what the log needs to say whether this is working ──────────────────────────────────
        int   boxes()     const { return static_cast<int>(L_.boxes.size()); }
        int   proposed()  const { return n_proposed_; }
        int   admitted()  const { return n_admitted_; }
        int   removed()   const { return n_removed_; }
        float rms()       const { return rms_; }
        float last_dL()   const { return last_dL_; }
        std::size_t cells() const { return vmap_.size(); }
        /// Open the per-step log. Path is etc/box_layout.csv; grading SIZE needs the published
        /// polygon, and a canvas is not an instrument.
        void open_csv(const std::string& path);
        /// Largest 1-sigma pose uncertainty folded in on the last scan — the term whose absence
        /// let odometry drift buy structure at 449161 nats.
        float last_sigma() const { return last_sigma_; }
        /// Fraction of fused cells lying more than 3 sigma OUTSIDE / INSIDE the region, and the
        /// rms restricted to the cells that are neither. ★ A single rms cannot distinguish "the
        /// box is the wrong size" from "a tail of returns is somewhere the box will never reach":
        /// a run fitting to 0.108 m reported rms 0.337 m, and those two are only reconcilable if
        /// a minority of cells are very far off. Splitting the residual is what makes that
        /// testable instead of arguable.
        float frac_out() const { return frac_out_; }
        float frac_in()  const { return frac_in_; }
        float rms_core() const { return rms_core_; }

    private:
        struct Vox { Eigen::Vector2d acc{0.0, 0.0}; double w = 0.0; float smin = 1e9f; };

        /// ── KEYFRAMES: THE EVIDENCE, STILL IN THE FRAME IT WAS MEASURED IN ──────────────────
        /// ★★★★★ WITHOUT THESE THE ESTIMATOR IS A FILTER, NOT A SMOOTHER. `vmap_` and `free_` are
        /// per-cell counters: a return displaced by pose error lands in a DIFFERENT cell that is
        /// never merged back, and a cell swept in error can never be un-swept. Consolidation can
        /// then only OUTVOTE a mistake, never correct it — so a pose error above one cell (0.05 m)
        /// is permanent, which is exactly why the fixed tour at 0.037 m never suffers and an
        /// explorer at 0.19 m destroys the map.
        /// Keeping the scan in the ROBOT frame with the pose it was taken at makes the evidence
        /// re-projectable: when the layout changes materially, every keyframe is re-registered
        /// against the new layout and the whole occupancy is rebuilt from the corrected poses.
        /// That is a small bundle adjustment, and it is the difference between a map that can be
        /// repaired and one that can only be outvoted.
        struct KeyFrame
        {
            std::vector<Eigen::Vector2f> pts;   ///< robot frame, subsampled
            Eigen::Vector3f pose{0.f, 0.f, 0.f};
            float sigma = 0.f;                  ///< scan-WORST point sigma; logging only
            /// ⚠ THE POSE COVARIANCE, NOT ONE NUMBER. `sigma` alone was the scan-wide MAXIMUM, and
            /// reproject() stamped it on every point of the keyframe — so after the first rebuild
            /// every voxel's `smin` was the worst point of whichever scan happened to be a
            /// keyframe, and the per-point sigmas observe() computes so carefully were gone. A 5 m
            /// return with 0.5 deg of heading error alone is ~44 mm, which is why the worst-face
            /// posterior plateaued around 15 mm on faces whose sensor noise is 20 mm: the floor was
            /// manufactured by the rebuild, not by the geometry.
            /// Keeping the 3x3 lets point_sigma() re-derive each point's own sigma from the lever
            /// arm at ITS range, in the rebuild exactly as in the live fold — and lets a keyframe's
            /// evidence IMPROVE when re-registration sharpens its pose, which is what makes a
            /// second look at a wall worth anything at all.
            Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
        };
        /// 1-sigma uncertainty of WHERE a robot-frame return is, given the pose covariance:
        /// translation plus the lever arm of the heading error at that range. First-order
        /// propagation of cov through p = t + R(theta) q. Shared by observe() and reproject() so
        /// the live fold and the rebuild can never disagree about what a point is worth.
        static float point_sigma(const Eigen::Matrix3f& cov, float c, float s,
                                 const Eigen::Vector2f& q)
        {
            const Eigen::Vector2f jth(-s * q.x() - c * q.y(), c * q.x() - s * q.y());
            const Eigen::Vector2f pxth(cov(0, 2), cov(1, 2));
            const float tr = cov(0, 0) + cov(1, 1) + cov(2, 2) * jth.squaredNorm()
                           + 2.f * pxth.dot(jth);
            return std::sqrt(std::max(0.f, tr) * 0.5f);
        }
        std::vector<KeyFrame> keys_;
        std::uint64_t last_key_f_ = 0;
        std::uint64_t structure_steps_ = 0;
        /// Re-register every keyframe against the current layout and rebuild vmap_/free_ from the
        /// corrected poses. Returns the mean correction applied, in metres.
        float reproject();

        /// ── FREE SPACE: THE CELLS THE BEAMS PASSED THROUGH ──────────────────────────────────
        /// ★★★★★ THE LAYOUT IS BUILT FROM THE INSIDE OUT, NOT THE OUTSIDE IN.
        /// The first design seeded one box = the interval HULL of the returns and let growth carve
        /// it down. That is right when a room is nearly convex — a rectangle with columns — and
        /// catastrophically wrong for an apartment, where the hull is a big rectangle and MOST OF
        /// IT IS NOT THE APARTMENT. Every return then sits deep inside the hull, reads as
        /// unexplained matter, and the estimator carves: measured on an 8-box apartment,
        /// **74 boxes, 290 vertices against 32, IoU 0.195**.
        /// The region the robot has actually traversed is the honest starting point, and it is
        /// something the sensor MEASURES rather than something the hull ASSUMES: a cell a beam
        /// passed through is free, and free space is the room. This is why the box-decomposition
        /// literature builds from occupancy rather than from a bounding volume.
        std::map<std::pair<int, int>, int> free_;
        mutable std::set<std::pair<int, int>> free_keys_;
        /// The robot's own cell trail — always free, and the seed's guaranteed starting point.
        std::pair<int, int> last_cell_{0, 0};
        bool have_cell_ = false;

        /// Largest axis-aligned box of free cells containing `seed`, grown greedily one row or
        /// column at a time. No threshold: a side stops when the next line is not free.
        rc::boxes::Box grow_free_box(const std::pair<int, int>& seed) const;

        /// ── THE LAYOUT IS A BOX COVER OF THE OBSERVED FREE SPACE ────────────────────────────
        /// ★★★★★ Greedy local edits do not converge on apartment topology. Seeded from a hull and
        /// asked to carve, an 8-box apartment became 74 boxes at IoU 0.195; seeded from free space
        /// it became 17; with carves forbidden where the beams had swept, the churn simply moved
        /// into extrusions and it became 65. Each fix was right and none of them addressed the
        /// shape of the problem: an apartment is not a room with defects, it is a UNION, and it
        /// should be CONSTRUCTED as one rather than whittled toward one.
        ///
        /// So the layout is built directly: repeatedly take the largest axis-aligned rectangle of
        /// free cells not yet covered, and stop when the next one does not pay its own description
        /// length. That is the same MDL that governs grow(), applied to construction instead of to
        /// repair, and it is what the box-decomposition literature does — build from occupancy,
        /// not from a bounding volume. Returns then only REFINE the offsets, which is the job they
        /// are actually good at.
        bool rebuild_from_free();
        /// ── PERIODIC GLOBAL RE-EVALUATION: MEASURE THE COMPLEXITY, THEN FORCE IT DOWN ───────
        /// Every local rule answers "is this edit justified here?", and a staircase of forty boxes
        /// answers YES at every step while being globally indefensible. So periodically, take the
        /// whole layout, compute what it costs in nats (code + negative log-likelihood) and try
        /// every reduction available — delete a box, merge a pair into their bounding box, absorb
        /// a contained one — keeping any that LOWERS the total. A reduction is accepted when the
        /// parameters it saves outweigh the likelihood it gives up, which is the same trade that
        /// admits structure, run in the opposite direction.
        ///
        /// ⚠ It only ever accepts a STRICT decrease in box count, so it terminates and cannot
        /// oscillate. That matters here: re-derivation storms destroyed a correct map on
        /// 2026-09-12 because the global pass could both add and remove. This one cannot add.
        /// Returns the number of boxes it removed.
        int simplify();

        /// Collapse faces that differ by less than the measured wall residual — they are one wall.
        void snap_coplanar();
        /// Free-cell count at the last rebuild. The cover is O(cells x area) and at 0.05 m a 60 m2
        /// room is ~24000 cells, so rebuilding every structure step made one room exceed ten
        /// minutes. The cover only changes when the EVIDENCE changes, so it is recomputed when the
        /// free set has grown materially — not on a timer, and not every frame.
        std::size_t free_at_rebuild_ = 0;
        /// Once the free-space cover has built a layout, it owns it.
        bool have_cover_ = false;
        bool   adopted_once_ = false;

        Params p_;
        rc::boxes::Layout L_;
        /// ⚠ THE LAYOUT LIVES IN ITS OWN RECTILINEAR FRAME, ROTATED BY `yaw_` FROM THE AGENT'S MAP
        /// FRAME. init_from_scan() builds its hull along the axes the quadrupled-angle vote
        /// recovered, so the box it returns is NOT expressed in the frame of the points handed to
        /// it. The bench absorbs this by carrying the robot pose in the box frame; this channel
        /// must do the same explicitly, because the agent's map frame belongs to every other
        /// consumer and may not be rotated under them.
        ///   in : map -> box   is R(-yaw_)
        ///   out: box -> map   is R(+yaw_)
        /// Getting this wrong fits an axis-aligned box to a cloud that is rotated relative to it:
        /// measured live in Webots as rms 0.379 m (19x the sensor noise) and a 6.00 x 4.00 room
        /// published as 4.221 x 5.684.
        float yaw_ = 0.f;
        /// Running circular mean of the per-frame Manhattan vote, accumulated on the QUADRUPLED
        /// angle so that the four indistinguishable quarter-turns land on the same direction and
        /// average instead of cancelling. This is the whole reason the gauge can be estimated at
        /// all: at 4*theta a square's symmetry is the identity.
        double yaw4_cos_ = 0.0, yaw4_sin_ = 0.0;
        long   yaw_votes_ = 0;
        bool   qInfo_first_ = false;
        std::map<std::pair<int, int>, Vox> vmap_;
        std::map<std::pair<int, int>, int> obs_;    ///< low-band returns, map cells: NAVIGATION ONLY
        std::vector<rc::boxes::CloudPoint> cloud_;
        std::vector<Eigen::Vector2f> init_scan_;
        std::uint64_t frames_ = 0;
        int n_proposed_ = 0, n_admitted_ = 0, n_removed_ = 0;
        float rms_ = 0.f, last_dL_ = 0.f, last_sigma_ = 0.f;
        float frac_out_ = 0.f, frac_in_ = 0.f, rms_core_ = 0.f;
        std::ofstream csv_;

        void fuse();
        /// WS_GAUGE_ML: turn the layout frame by the rotation that maximises the layout's own
        /// likelihood of the fused returns (see the definition). Returns the rotation applied.
        float fit_gauge_ml(const rc::boxes::GrowParams& gp);
    };
}   // namespace rc::boxch
