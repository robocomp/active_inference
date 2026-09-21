#pragma once
// ─────────────────────────────────────────────────────────────────────────────────────────────
//  THE ROOM AS A UNION OF AXIS-ALIGNED BOXES
//
//  Written 2026-09-19, from zero, because the previous representation PERMITTED illegal states and
//  every defect found in two days of measurement was a rule policing that permission:
//
//    · the existence channel deleted walls below a rectangle with no floor and no price, while the
//      priced path refuses to — the published room became a TRIANGLE with the robot outside it;
//    · a span rule struck out a wall whose evidence bins were at the confidence clamp;
//    · heal_order deletes a near-parallel neighbour on POINT COUNT, with no evidence test;
//    · relaxing that kill let a near-parallel pair survive and their intersection ran away: a
//      3-vertex polygon with a corner at (-11752, -14336) m and a corner sigma of 441051.
//
//  A 3-vertex room cannot exist. In this representation it is not rejected — it is UNREPRESENTABLE.
//
//  STATE.  In the rectilinear map frame (the frame IS the gauge; there is no theta0 variable),
//
//      R  =  union of positive boxes  \  union of negative boxes
//
//  Each box is an axis-aligned interval product [x0,x1] x [y0,y1] with x0 < x1 and y0 < y1. The
//  region is therefore closed, Manhattan and valid BY CONSTRUCTION, for every value of every
//  parameter. There is no `order`, so nothing to heal; no cycle, so nothing to repair; no way to
//  express a triangle, an open polygon, or two near-parallel walls meeting 14 km away.
//
//  WHAT IS AN IDENTITY RATHER THAN A RULE
//    Manhattan     no parameter can express another angle
//    closure       a finite union of closed boxes is a closed region
//    simplicity    guaranteed for a connected union without holes (checked combinatorially, O(n^2),
//                  on the PROPOSAL only — never on the continuous state, which cannot leave it)
//    corner sigma  a vertex is (x_i, y_j): its covariance is a 2x2 sub-block of the offset
//                  covariance. No intersection of two lines, so no divergence when they are
//                  near-parallel — the failure above is arithmetically impossible here.
//
//  The one inequality is a box's positive width, and `width -> 0` IS the removal event, priced by
//  the same Bayes factor as insertion rather than policed by a point count.
//
//  See LAYOUT_REDESIGN.md for the threshold budget (~17 constants, 0 tuned decision constants,
//  against ~100 in the representation this replaces) and the migration plan.
// ─────────────────────────────────────────────────────────────────────────────────────────────

#include <Eigen/Dense>
#include <optional>
#include <set>
#include <vector>

namespace rc::boxes
{
    /// An axis-aligned box in the rectilinear map frame. Invariant: lo.x < hi.x and lo.y < hi.y.
    /// `positive` false makes it a subtracted box — a pier or column standing inside the room.
    struct Box
    {
        Eigen::Vector2f lo{0.f, 0.f}, hi{0.f, 0.f};
        bool positive = true;
        std::uint64_t id = 0;
        /// A column standing against a wall shares that wall's face. The shared offset is ONE
        /// parameter, not two: `attach` names which of this box's four offsets (0=lo.x, 1=lo.y,
        /// 2=hi.x, 3=hi.y) is the host's, and `host` which box it belongs to. -1 means free.
        /// ⚠ This is not bookkeeping. Leaving the two offsets independent let the fit pull a
        /// negative box 0.23 m clear of its wall, turning the column into a HOLE — which the
        /// representation forbids and polygon() silently hid by tracing only the outer boundary
        /// (w1 reported 4 vertices while holding 2 boxes). Tying them makes the detachment
        /// unrepresentable, and correctly costs the edit three parameters instead of four.
        /// 0..3  this box's lo.x/lo.y/hi.x/hi.y equals the host's SAME face   (a carve, inward)
        /// 4..7  this box's lo.x/lo.y/hi.x/hi.y equals the host's OPPOSITE face (an alcove, outward)
        /// An alcove abuts the room across the wall it opens through, so its inner face IS that
        /// wall. Same identity as the carve, same one-parameter saving, opposite direction.
        int attach = -1;
        std::uint32_t host = 0;

        float width()  const { return hi.x() - lo.x(); }
        float height() const { return hi.y() - lo.y(); }
        float area()   const { return width() * height(); }
        bool  valid()  const { return width() > 0.f and height() > 0.f; }
        bool  contains(const Eigen::Vector2f& p) const
        { return p.x() >= lo.x() and p.x() <= hi.x() and p.y() >= lo.y() and p.y() <= hi.y(); }
    };

    /// The layout. `yaw` is the robot-frame-to-rectilinear-frame rotation estimated at
    /// initialisation; the map frame is rectilinear BY DEFINITION, which is what removes theta0,
    /// the gauge factor and the re-anchor together.
    struct Layout
    {
        std::vector<Box> boxes;
        /// Joint covariance over the stacked offsets [b0.lo.x, b0.lo.y, b0.hi.x, b0.hi.y, b1...].
        Eigen::MatrixXf cov;

        std::size_t n_offsets() const { return boxes.size() * 4; }
        bool empty() const { return boxes.empty(); }

        /// Is p inside the region? (in a positive box, and in no negative one)
        bool inside(const Eigen::Vector2f& p) const;
        /// Signed distance to the region boundary, negative inside. The measurement model's chi.
        float sdf(const Eigen::Vector2f& p) const;
        /// The ordered outer boundary, CCW, for the fleet's delimiting_polygon contract.
        /// Cannot return fewer than 4 vertices: the smallest representable region is one box.
        std::vector<Eigen::Vector2f> polygon() const;
        /// Per-vertex 2x2 covariance, in the same order as polygon(). A vertex IS two offsets, so
        /// this is a sub-block lookup plus the flatness term — never an intersection.
        std::vector<Eigen::Matrix2f> polygon_cov(float sigma_flat) const;
    };

    /// Which offset of which box does the distance at `p` come from — the estimator's own
    /// attribution rule, as `4 * box + {0=lo.x, 1=lo.y, 2=hi.x, 3=hi.y}`, or -1 for an empty
    /// layout. refit() folds every return through this, so it is also the only honest answer to
    /// "which parameter could a look at this patch of wall improve?": where two boxes share a
    /// wall the returns land on ONE of the two coincident offsets and the other is redundant —
    /// unobserved by construction, and not a place worth sending a robot.
    int active_face(const Layout& L, const Eigen::Vector2f& p);

    /// ── INITIALISATION FROM ONE SCAN ─────────────────────────────────────────────────────────
    /// Estimate (yaw, one box) from a single scan in the robot frame. The yaw comes from the
    /// quadrupled-angle peak of the returns' local orientations — the standard Manhattan estimator,
    /// and the only search in the design; the box is then the interval hull along the recovered
    /// axes, at the quantile given by `edge_quantile` so a few stray returns cannot inflate it.
    /// ⚠ NO OBB. The previous code seeded from an oriented bounding box, which tilts 15-20 degrees
    /// on a concave first scan (documented at wall_map.h) — a quadrupled-angle vote does not.
    struct InitParams
    {
        float sensor_sigma  = 0.02f;   // m   — PHYSICAL: LiDAR range noise perpendicular to a wall
        float sigma_flat    = 0.010f;  // m   — PHYSICAL: wall flatness / line scatter between views
        int   yaw_bins      = 180;     // COMPUTE: resolution of the quarter-turn search
        float edge_quantile = 0.995f;  // COMPUTE: robustness of the interval hull to stray returns
    };
    /// Returns nullopt only if the scan is too small to estimate anything.
    std::optional<std::pair<float, Layout>>
    init_from_scan(const std::vector<Eigen::Vector2f>& pts_robot, const InitParams& p);

    /// ── REGISTRATION ─────────────────────────────────────────────────────────────────────────
    /// Solve the 3-DoF pose against a FIXED layout, Kinematic-ICP style: the map cost evaluated at
    /// the odometry prediction sets the regulariser's weight, so the prior dissolves exactly when
    /// odometry and the scan disagree, and holds when they agree. Translation only is regularised —
    /// heading is left to the LiDAR, which is the better channel and the best-observed DOF in a
    /// rectangular room. No gate, no threshold: beta is read off the data.
    struct RegisterResult
    {
        Eigen::Vector3f pose = Eigen::Vector3f::Zero();
        Eigen::Matrix3f cov  = Eigen::Matrix3f::Identity();
        float beta = 0.f;          ///< the map cost at the odometry prediction (the regulariser)
        float cost = 0.f;          ///< the map cost at the solution
        int   iterations = 0;
        bool  ok = false;
    };
    /// ── INCREMENTAL STRUCTURE: EXPLAIN THE POINTS WITH THE SIMPLEST REGION ──────────────────
    /// The problem, stated the way it should have been from the start: given {global poses, global
    /// point clouds}, find the SIMPLEST Manhattan, closed, simply-connected layout that explains
    /// them. Poses come from odometry alone at this stage — no relocalisation, no registration —
    /// so structure estimation is decoupled from pose estimation and cannot be corrupted by it.
    ///
    /// A return lands ON a surface, so under a correct layout every return has sdf ~ 0. A cluster
    /// of returns strictly INSIDE the region is therefore evidence of matter the region does not
    /// model — a column, a pier — and proposes a NEGATIVE box. A cluster OUTSIDE is evidence of
    /// room the region does not cover — an alcove, a door recess, a bay — and proposes a POSITIVE
    /// one attached across the wall it opens through.
    /// ⚠ BOTH DIRECTIONS, AND THE SECOND WAS MISSING UNTIL 2026-09-19. grow() had exactly one
    /// surprise test, `sdf < -3 sigma`, so it could carve and never extend: a door recess was
    /// UNREPRESENTABLE and its returns sat outside the box inflating the residual for ever (live
    /// Webots: a box fitted to 0.108 m still reporting rms 0.337 m). ★★★ The bench could not have
    /// caught it — every ladder room is a rectangle with COLUMNS, so the suite exercised the one
    /// direction that was implemented. A test suite that shares a blind spot with the code it
    /// tests cannot reveal it; `alcove` and `bay` are on the ladder now for exactly that reason.
    ///
    /// Admission is a Bayes factor in nats, with NO tuned margin: accept iff the likelihood gained
    /// exceeds the description length of the new parameters. BF > 1, i.e. 0 nats, is canonical;
    /// today's `birth_nats = 4.605` is tuned hysteresis and is not reproduced here.
    ///
    /// ⚠ A POINT IS NO SHARPER THAN THE POSE THAT PLACED IT. The first implementation of this
    /// scored every accumulated return at the sensor's own sigma (22 mm) while the poses came from
    /// raw odometry drifting 0.204 m. The drift smear then sat 9 sigma inside the region, was read
    /// as matter, and bought a negative box on the EMPTY rectangle: IoU 0.977 -> 0.026, with
    /// dL = 449161 nats because the likelihood summed 200000 points against a ~24-nat code length.
    /// Both halves of that are now fixed, and neither fix is a threshold:
    ///   · each point carries the pose sigma it was captured at, and enters at
    ///     sigma_eff^2 = sensor^2 + flat^2 + sigma_pose^2 — so a smeared point simply stops being
    ///     surprising, which is what it deserves, instead of being gated out;
    ///   · the cloud is voxelised before scoring, because a surface seen from 1000 poses is ONE
    ///     piece of geometric evidence, not 1000 independent ones. Same correlated-evidence
    ///     argument as the Woodbury common mode, applied across TIME instead of across a scan.
    struct CloudPoint
    {
        Eigen::Vector2f p{0.f, 0.f};
        float sigma_pose = 0.f;   ///< 1-sigma position uncertainty of this return in the map frame
    };

    /// A cell the beams passed through. `free` is the evidence that it is ROOM, which is a
    /// measurement, not an assumption — and the only thing that can stop growth from carving an
    /// apartment into scraps or extending it into space nobody has been.
    struct FreeCell { int x = 0, y = 0; };

    struct GrowParams
    {
        float sensor_sigma = 0.02f;   // PHYSICAL: range noise
        float sigma_flat   = 0.010f;  // PHYSICAL: wall flatness
        float cell         = 0.10f;   // COMPUTE: clustering / voxel resolution
        int   min_cluster  = 12;      // COMPUTE: smallest cluster worth proposing
        /// Log-odds that a cell of the room goes unswept by any beam over a whole tour. A MODEL
        /// parameter — the sensor's coverage statement — not a tuned bar: at 1 nat a claimed but
        /// never-observed cell is about e:1 against, so a merge must save more than one parameter
        /// per cell of empty space it swallows.
        float unobserved_nats = 1.0f;
    };
    struct GrowResult
    {
        int   proposed = 0;
        int   admitted = 0;
        float best_dL  = 0.f;         ///< nats gained by the admitted edit (>0 by construction)
    };
    /// ── FIT THE OFFSETS TO THE CLOUD ─────────────────────────────────────────────────────────
    /// The missing middle step: "size it with the initial lidar data, OPTIMISE IT, then look for
    /// what it fails to explain". Without this the box stays frozen at its single-scan hull — 4.19
    /// x 6.10 against a true 4.00 x 6.00 — so every wall's returns lie 5-10 cm INSIDE the region
    /// and grow() reads the whole perimeter as matter: c2 ended with five negative boxes, four of
    /// them 0.12-m slivers lying along the walls. Structure must never be asked to absorb a
    /// misfit the continuous parameters can remove themselves.
    ///
    /// Diagonal Gauss-Newton on the stacked offsets, weighted by each point's own 1/sigma^2. Each
    /// residual is sdf(q) and depends on exactly ONE offset (the active face), so the normal
    /// equations are diagonal and the step is a weighted mean — no linear algebra, no line search,
    /// no step-size constant. Returns the RMS residual after fitting.
    /// `freecells` — the cells the beams SWEPT, in cell coordinates. Optional, and the reason it
    /// exists: a face with no returns on it has no likelihood gradient, so nothing moves it. The
    /// tip of a fin faces into open room and collects no returns at all, so it stays wherever it
    /// was proposed — measured on the apartamento hall, a 1.69 m fin published at 3.23 m, closing a
    /// 2.45 m passage to 0.97 m, with its WIDTH correct to 8 mm and rms 0.033 m. The residual
    /// cannot see it either: the fin's sides fit their returns perfectly, and the phantom extends
    /// into space no beam ever returned from.
    /// ⚠ THE ROBOT DROVE THROUGH IT. One trajectory sample sat at (-0.038,-1.652) inside the
    /// published solid. A cell the robot has occupied cannot be structure, and that is not a
    /// likelihood statement — it is an invariant, and `mdl_cost` already charges it as `missed`.
    /// This makes the same evidence DIFFERENTIABLE so a face can be pushed by it instead of only
    /// being judged by it at a discrete edit.
    /// ★ ONE DIRECTION ONLY. A swept cell the layout excludes is proof of room. A claimed cell the
    /// beams never swept is NOT proof of matter — it may simply be room nobody has visited — so it
    /// stays a cost on structure edits and does not become a force on a face.
    float refit(Layout& L, const std::vector<CloudPoint>& cloud, const GrowParams& p, int iters = 10,
                const std::set<std::pair<int, int>>* freecells = nullptr);

    /// One structure step over the accumulated global cloud. Returns what it did; mutates `L`.
    /// ── WHAT THE WHOLE LAYOUT COSTS, IN NATS ─────────────────────────────────────────────────
    /// code length + negative log-likelihood: the quantity MDL says to minimise. Admission has
    /// always been priced this way LOCALLY — "does this one edit pay?" — and that is not the same
    /// question as "is this description the shortest one that fits". A staircase of forty boxes can
    /// be locally justified at every single step and globally absurd, which is exactly what the
    /// apartamento hall produced: IoU 0.926 with 196 published vertices for 32 real walls.
    /// Exposing the total makes a GLOBAL pass possible: try reductions, keep any that lowers it.
    /// `free` is the observed free space in LAYOUT-frame cells. ⚠ WITHOUT IT THE COST IS BLIND
    /// TO EMPTY CLAIMS: returns lie on walls, so a region that bulges into space the robot never
    /// swept loses almost no likelihood while saving real parameters, and a priced merge will take
    /// that trade every time. Measured on the apartamento hall — the layout fitted the walls to
    /// 0.118 m with a 0.046 m pose and still scored IoU 0.541, because simplify() had merged
    /// straight across the concavities. A claimed cell that no beam ever crossed is evidence
    /// AGAINST the claim, at `unobserved_nats` each.
    float mdl_cost(const Layout& L, const std::vector<CloudPoint>& cloud, const GrowParams& p,
                   const std::set<std::pair<int, int>>* free = nullptr);

    /// `free` is the observed free space, as cell coordinates on the same grid as `p.cell`, in
    /// the LAYOUT frame. Empty means "no free-space evidence", and grow() falls back to judging
    /// on returns alone — which is what it did before, and what fails on an apartment.
    GrowResult grow(Layout& L, const std::vector<CloudPoint>& cloud, const GrowParams& p,
                    const std::set<std::pair<int, int>>* free = nullptr);

    RegisterResult register_scan(const Layout& L,
                                 const std::vector<Eigen::Vector2f>& pts_robot,
                                 const Eigen::Vector3f& odom_pose,
                                 float sensor_sigma);
}   // namespace rc::boxes
