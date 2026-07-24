/*
 * cabinet_belief.h  —  AI2 cabinet-RUN belief (see CABINET.md)
 *
 * WHAT IS MODELLED, AND WHY IT IS NOT A CABINET
 * ---------------------------------------------
 * Kitchen cabinets stand flush side by side: no gaps, one continuous countertop, coplanar doors,
 * uniform colour. The boundary between two adjacent modules therefore carries essentially ZERO
 * sensor evidence, so a per-module box fit is unidentifiable — and every way of forcing it (seam
 * clustering, splitting at a standard width, gap thresholds) is exactly the threshold band-aid
 * CLAUDE.md forbids.
 *
 * So the latent is the RUN, not the module: a wall-anchored linear extrusion. One instance = one
 * TIER of one run (a base run and the wall units above it are two instances; an L-shaped kitchen is
 * two runs meeting at a corner — that falls out of the merge operator, no special case). The
 * individuation question does not get answered badly; it stops being asked.
 *
 * State θ = [cx, cy, yaw, L, d, z0, z1]   (7 DOF, room frame)
 *   cx,cy  run centre            yaw  run LONG-axis direction
 *   L      length along the axis  d   carcass depth (front face → back face)
 *   z0,z1  bottom / top height above the room floor
 *
 * ONE box primitive + clutter (a cabinet run has no legs — contrast the table's slab+4 legs), with
 * SOFT per-point responsibilities and the shared recursive variational-Laplace filter carrying a full
 * 7x7 covariance. Pure Eigen, no torch, no DSR — unit-testable in isolation (see self_test()).
 *
 * THE TWO STRUCTURAL TERMS THAT MAKE IT IDENTIFIABLE
 * --------------------------------------------------
 *  (1) The BACK FACE IS NEVER OBSERVED (it faces the wall), so `d` is depth-degenerate from any view
 *      — the same pathology as the bottle's radius. What supplies the missing information is the
 *      wall: a run is flush against it. accumulate_extra() folds in a wall-flush + wall-parallel
 *      factor whose precision is the marginal of a two-component mixture {flush against this wall,
 *      free-standing}, so it decays CONTINUOUSLY with the back face's distance to the wall. A
 *      free-standing island (Worktop3 in the apartamento kitchen sits 2.17 m / 1.18 m off both
 *      walls) finds no wall in range, the factor contributes ~nothing on its own, and `d` stays
 *      honestly wide. There is no `if (near_wall)`.
 *  (2) TIER is a DISCRETE latent, not an `if` on height. A Laplace filter cannot hold the bimodal
 *      prior on (z0,z1) that "base unit OR wall unit" really is, so the tier rides a sequential
 *      Bayesian mode comparison (mirroring the table's w<->h mode machinery): each frame accumulates
 *      log-evidence for the two prior sets and the MAP mode supplies the priors. No height threshold.
 *
 * Everything else is inherited: within-frame common-mode (Woodbury) saturation, range-dependent
 * covariance, pose-chain covariance — all from common/ai_belief/recursive_laplace.h.
 */

#pragma once

#include <array>
#include <vector>
#include <Eigen/Dense>

#include "../../common/ai_belief/recursive_laplace.h"   // shared predict/MAP/Woodbury engine
#include "../../common/ai_belief/lidar_ray_factor.h"    // shared YOLO-independent LiDAR first-hit factor

namespace rc
{

// Which tier of kitchen carcass this instance is. The two entries differ ONLY in their (d, z0, z1)
// priors; the geometry and every hook are identical. Selected by model evidence, never by a height cut.
enum class CabinetTier : int { Base = 0, Wall = 1 };

// The nearest room wall, as seen by THIS frame. Supplied by the fitter from room_concept's
// `delimiting_polygon_x/y` (a NOMINAL model loaded from apartamento_layout.svg — room_concept
// localizes against it and never fits its geometry to returns, so there is no circularity between
// "the wall" and "the cabinet front face"). ok=false ⇒ the wall factor is inert this frame.
struct WallRef
{
    bool            ok      = false;
    Eigen::Vector2f p       = Eigen::Vector2f::Zero();   // any point on the wall line (room frame, m)
    Eigen::Vector2f n       = Eigen::Vector2f::UnitX();  // UNIT normal, pointing INTO the room
    float           sigma_m = 0.02f;                     // wall position uncertainty (m)
    // The wall SEGMENT (its two corners, room frame), collinear-merged so an authored jog/door-notch does
    // not plant a phantom corner mid-run. A flush run's along-axis extent is bounded by these endpoints:
    // accumulate_extent censors the observed span at them (never grow past a corner) and adds a symmetric
    // inward hinge (retract an already-overgrown end back to the corner). has_segment=false ⇒ both inert.
    bool            has_segment = false;
    Eigen::Vector2f a       = Eigen::Vector2f::Zero();   // segment endpoint A (corner)
    Eigen::Vector2f b       = Eigen::Vector2f::Zero();   // segment endpoint B (corner)
    int             seg_id  = -1;                        // canonical wall id (stable across the merged run)
};

// Birth seed derived from a raw point cloud: the run's own 2-D principal axis + extents + z-band.
// This is the run analogue of the table's footprint second-moment — a GLOBAL sufficient statistic of
// the filled-box model, used only to START the fit near the data (a box seeded small and
// axis-aligned would see the real run as clutter and never grow into it).
struct RunSeed
{
    bool  ok = false;
    int   n  = 0;
    float cx = 0.0f, cy = 0.0f;
    float yaw = 0.0f;            // principal (major) axis angle, rad
    float L = 0.0f, d = 0.0f;    // full extents along major / minor axis (m)
    float z0 = 0.0f, z1 = 0.0f;  // robust vertical band (m)
    float aniso = 0.0f;          // (L−d)/(L+d): how well-defined the principal axis is
};

// Fitted cabinet-run state. Unlike the table there is NO symmetry fold on the extents: L >> d by
// construction and a run has a front (the doors), so w<->h is not a symmetry here. The one residual
// symmetry is the box's C2v yaw ambiguity (yaw vs yaw+pi), resolved geometrically in canonicalize()
// by requiring the front normal to point into the room.
struct CabinetBeliefState
{
    float cx = 0.0f, cy = 0.0f;
    float yaw = 0.0f;              // direction of the run's LONG axis (rad)
    float L = 1.0f;                // length along the axis (m)
    float d = 0.60f;               // carcass depth, front face -> back face (m)
    float z0 = 0.0f, z1 = 0.90f;   // bottom / top height above the room floor (m)

    Eigen::Matrix<float, 7, 1> vec() const
    { return (Eigen::Matrix<float, 7, 1>() << cx, cy, yaw, L, d, z0, z1).finished(); }
    static CabinetBeliefState from_vec(const Eigen::Matrix<float, 7, 1>& v)
    { return {v(0), v(1), v(2), v(3), v(4), v(5), v(6)}; }

    // Unit vector along the run's long axis, and the run's FRONT normal (canonicalize() fixes its sign
    // so it points into the room). Right-handed: n = R(+90 deg) * u.
    Eigen::Vector2f axis()   const { return {std::cos(yaw), std::sin(yaw)}; }
    Eigen::Vector2f normal() const { return {-std::sin(yaw), std::cos(yaw)}; }
    Eigen::Vector2f centre() const { return {cx, cy}; }
    float height() const { return std::max(0.0f, z1 - z0); }
    // Centre of the BACK face (the one against the wall) — the point the wall-flush factor drives.
    Eigen::Vector2f back_centre() const { return centre() - normal() * (0.5f * d); }
};

// Per-tier carcass priors. Values are the standard European kitchen carcass dimensions, entered as
// BROAD Gaussian priors (honest sigma, not clamps): they only break the degeneracy of an
// unobservable DOF, and the data overrides them wherever it actually speaks. They are deliberately
// laid out so a future room-type meta-concept (CONCEPT_AGENT_RECIPE.md §6, the `row/linear` schema)
// can SHARPEN them top-down as an empirical prior without restructuring anything here.
struct CabinetTierPrior
{
    float d_mean = 0.60f,  d_std  = 0.10f;    // carcass depth (m)
    float z0_mean = 0.00f, z0_std = 0.05f;    // bottom height (m)
    float z1_mean = 0.90f, z1_std = 0.06f;    // top height (m)  (base: worktop at 0.90)
};

struct CabinetBeliefParams
{
    // ── Tier priors (selected by model evidence — see resolve_tier) ────────────────────────────
    CabinetTierPrior base_tier{0.60f, 0.10f, 0.00f, 0.05f, 0.90f, 0.06f};
    CabinetTierPrior wall_tier{0.35f, 0.08f, 1.45f, 0.15f, 2.10f, 0.20f};

    // ── Observation model ─────────────────────────────────────────────────────────────────────
    // NOTE the value: `cabinet` masks are NOT SAM2-refined (voxelizer [Sam2] refine_labels omits
    // them), so they are raw semantic connected components with coarse boundaries. sigma_base_m is
    // correspondingly larger than the table's 0.03. This is an honest noise statement about a
    // coarser detector, not a fudge — tighten it if `cabinet` is added to refine_labels.
    float sigma_base_m    = 0.05f;
    float clutter_frac    = 0.10f;    // prior weight of the uniform clutter component
    float clutter_scale_m = 0.15f;    // a point further than ~this from the box is likely clutter

    // ── Priors on the data-driven DOFs ────────────────────────────────────────────────────────
    // L is deliberately very wide: run length is what the data is FOR, and (per the recipe's
    // lower-bound invariant) mask/LiDAR extent evidence is grow-only — shrinking is the free-space
    // channel's job, never a prior pulling L back in.
    float prior_L_mean = 1.0f,  prior_L_std = 3.0f;
    float prior_pos_std = 1.0f;       // cx, cy (m) — broad, position is well observed
    float prior_yaw_std = 1.0f;       // rad — broad, the wall factor supplies the real information

    // ── Wall-flush factor (accumulate_extra) ──────────────────────────────────────────────────
    // The precision is the marginal of a 2-component mixture {flush against the nearest wall,
    // free-standing}: lambda = wall_precision * exp(-(gap/wall_reach_m)^2) / (sigma_wall^2 + ...).
    // The exponential IS the (Gaussian) posterior weight of the flush component given the observed
    // gap — hence a continuous covariance, not a proximity gate. 0 = OFF.
    float wall_precision   = 400.0f;  // 1/m^2 at zero gap (~1/sigma^2 with sigma=5 cm)
    float wall_reach_m     = 0.35f;   // gap scale over which the flush hypothesis loses its weight
    float wall_parallel_precision = 200.0f;   // on sin(angle between run axis and wall) — 0 = OFF

    // ── Room-axis (Manhattan) yaw prior (accumulate_axis_alignment) ─────────────────────────────
    // A kitchen run is built parallel to a room wall — its yaw sits on a room axis (a multiple of
    // π/2 in the room frame). The wall_parallel term only supplies this while the run is flush along
    // its length against a DETECTED wall (weighted by the flush mixture), so a peninsula/island that
    // touches a wall on only one short side (U-/L-kitchen aisle), a merged/mis-positioned run, or a
    // run whose wall isn't currently seen feels nothing and drifts oblique. This factor adds the
    // room-independent Manhattan prior directly on yaw: a strong soft pull to the NEAREST π/2 axis,
    // silent once aligned (residual→0). Capture range bounds it so a genuinely oblique object outside
    // the range is left alone; ≤0 (or ≥π/4) ⇒ always active. Precision 0 = OFF.
    float room_axis_precision   = 300.0f;   // 1/rad^2 on the yaw→nearest-axis residual
    float room_axis_capture_rad = 0.0f;     // |Δyaw| beyond which the pull is released (0 ⇒ always on)

    // ── Along-axis CONTAINMENT factor (accumulate_extent) ─────────────────────────────────────
    // The per-point SDF mixture structurally cannot GROW a run: a point past the current end cap is
    // ~L/2 away, so the mixture cedes it to clutter and its gradient vanishes — the classic escape
    // valve (the table hits the same wall and answers it with a global footprint statistic). For a
    // run the right global statistic is the along-axis SPAN of the points that are compatible with
    // the run's cross-section.
    //
    // It enters as a CENSORED (one-sided) likelihood, which is the honest measurement model: an
    // observed span is a LOWER BOUND on the true extent — occlusion and foreshortening can only ever
    // shorten it, never lengthen it (the recipe's "a mask is a lower bound on extent" invariant). So
    // each end contributes a hinge residual that is exactly zero once the box contains the cloud, and
    // the factor is silent thereafter. Shrinking is NOT this channel's job; that belongs to the
    // occlusion-aware free-space/vacate evidence. 0 = OFF.
    float extent_precision = 800.0f;   // 1/m^2 on the end-containment residuals
    float extent_quantile  = 0.02f;    // robust order statistic for the span ends (0 ⇒ raw min/max)

    // ── Free-space / VACATE factor (accumulate_freespace) ─────────────────────────────────────────
    // The occlusion-aware UPPER bound that CLOSES the one-sided extent factor above (the "loan" its
    // comment promised). A LiDAR beam that traverses the carcass volume and lands BEYOND its far face
    // proves that crossing is empty; its midpoint is a witness point INSIDE the footprint, so a GLS pull
    // driving the FOOTPRINT SDF→0 there retreats the nearest face past it = shrink. A beam that RETURNS
    // inside the box (occupancy) has p_through≈0 ⇒ no vacate, so a real cabinet is untouched. p_through is
    // a {returned-from-surface, exited-far-face} mixture posterior (same construction as flush_weight) ⇒
    // continuous, no gate. Occlusion-safe by geometry: a beam stopped by an occluder never yields a witness
    // past it. Ported from table_belief; a cabinet carcass is SOLID [z0,z1] so the whole box is carved (no
    // top-slab z-gate, no hollow guard). 0 = OFF.
    float free_space_precision = 0.0f;   // 1/m^2 per through-beam (shrink-only)

    // ── Tier mode ─────────────────────────────────────────────────────────────────────────────
    // Per-frame evidence for the alternative tier is accumulated; the MAP mode supplies the priors.
    // tier_evidence_cap bounds the accumulator so a long run of agreeing frames cannot make the mode
    // unswitchable if the world changes (an honest forgetting factor, not a threshold on the data).
    float tier_evidence_cap = 25.0f;

    // ── Temporal transition: static furniture ⇒ tiny process noise ────────────────────────────
    float process_std_m   = 0.005f;
    float process_std_yaw = 0.01f;

    // ── Per-frame COMMON-MODE error (Woodbury saturation; see recursive_laplace.h) ────────────
    float common_mode_pos_std  = 0.04f;   // shared position error (m); chain_cov adds to it
    float common_mode_size_std = 0.03f;   // shared size error on L, d, z0, z1 (m)
    float common_mode_yaw_std  = 0.03f;   // shared yaw error (rad)

    // ── Optimiser ─────────────────────────────────────────────────────────────────────────────
    int   gn_iters = 4;
    float fd_eps   = 1e-3f;
};

// One fitted frame's evidence: room-frame points and per-point measurement variance R (m^2).
struct CabinetFrame
{
    std::vector<Eigen::Vector3f> points;
    std::vector<float>           R;          // per-point variance (m^2); empty ⇒ sigma_base^2 for all
    Eigen::Vector3f              cam_origin = Eigen::Vector3f::Zero();

    float chain_cov_xx  = 0.0f;   // extra shared position variance (m^2) from the pose chain
    float chain_cov_yy  = 0.0f;
    float chain_cov_yaw = 0.0f;   // grows with range ⇒ a distant vague mask cannot rotate a fitted run
    float chain_cov_size = 0.0f;  // grows with range ⇒ nor reshape it

    // The nearest room wall for the wall-flush factor (see WallRef).
    WallRef wall;

    // YOLO-INDEPENDENT LiDAR channel — the PRIMARY metric channel for a run (see CABINET.md):
    // masks answer "there is a cabinet here" (LiDAR cannot: a wall-flush run is geometrically a wall
    // offset by 0.6 m), while LiDAR answers "exactly where and how long", which coarse non-SAM2
    // masks on a white untextured door front do badly. `helios` rakes the front face densely ALONG
    // the run — precisely the along-axis extent evidence that defines L.
    rc::ai::LidarRays              lidar;
    std::vector<rc::ai::LidarRays> lidar_extra;   // per-device sets (e.g. the low `bpearl`), own origins
    // Beams whose SEGMENT (origin→endpoint) crosses the CURRENT fitted box — the through-beams that carry
    // the free-space/VACATE negative evidence. Selected against the box (not the mask centroid) precisely
    // because an overgrown end's carving beams land BEYOND it, far from the mask the range channel anchors on.
    rc::ai::LidarRays              lidar_freespace;
};

class CabinetBelief
{
public:
    static constexpr int N = 7;
    using State = CabinetBeliefState;
    using Frame = CabinetFrame;

    CabinetBelief() = default;
    CabinetBelief(const CabinetBeliefState& s, const CabinetBeliefParams& p) : state_(s), params_(p)
    { Sigma_.setZero(); Sigma_.diagonal() = prior_cov_diag(); }

    const CabinetBeliefState&         state()      const { return state_; }
    const Eigen::Matrix<float, 7, 7>& covariance() const { return Sigma_; }
    // ── Reported covariance: fold the discrete TIER-mode entropy into Σ ─────────────────────────
    // Σ_ carries only the WITHIN-tier carcass widths — it CANNOT hold the genuinely bimodal "base OR
    // wall unit" ambiguity on (d, z0, z1) that the tier rides outside the Gaussian. So the reported
    // covariance inflates those three diagonals by the discrete-mode entropy p(1−p)Δ², where
    // p = tier_posterior() (probability of the alternative tier) and Δ is the separation between the
    // two tiers' prior means for that DOF. A still-undecided tier (p≈½) advertises an honest large σ
    // spanning both carcasses; a resolved one (p→0) collapses back to Σ_. Consumed by the RT-edge cov
    // upload and the NBV planner. Mirrors Table/ChairBelief::covariance_reported (see TABLE.md §4.2).
    Eigen::Matrix<float, 7, 7> covariance_reported() const;
    const CabinetBeliefParams&        params()     const { return params_; }
    void set_state (const CabinetBeliefState&  s) { state_  = s; }
    void set_params(const CabinetBeliefParams& p) { params_ = p; }

    // ── Room interior reference, for the C2v yaw fold in canonicalize() ───────────────────────
    // Set once by the fitter from the room polygon centroid. The front normal must point toward it.
    void set_room_interior(const Eigen::Vector2f& c) { room_interior_ = c; has_room_interior_ = true; }

    // ── Discrete TIER mode ────────────────────────────────────────────────────────────────────
    CabinetTier tier()           const { return tier_; }
    float       tier_posterior() const;    // p(alternative tier) = sigmoid(-tier_evidence_)
    float       tier_evidence()  const { return tier_evidence_; }
    void        set_tier(CabinetTier t)  { tier_ = t; }
    // One sequential-comparison step over {current tier, other tier}: accumulate this frame's
    // log-evidence and adopt the better-supported mode (boundary at zero accumulated evidence = the
    // MAP over the mode; no tuned threshold). evidence_weight in [0,1] scales the frame's vote so an
    // ego-motion-corrupted / fragmentary mask does not decide the tier. Returns true iff it switched.
    bool        resolve_tier(const std::vector<Eigen::Vector3f>& pts, float R, float evidence_weight = 1.0f);
    // Mean per-point data energy (NLL proxy) at an ARBITRARY state — the mode-comparison test statistic.
    float       mean_energy(const std::vector<Eigen::Vector3f>& pts, const CabinetBeliefState& s, float R) const;
    const CabinetTierPrior& tier_prior() const
    { return tier_ == CabinetTier::Base ? params_.base_tier : params_.wall_tier; }

    // ── Inference (delegated to the shared engine) ────────────────────────────────────────────
    float update(const CabinetFrame& frame)
    { return ai::update<N>(*this, state_, Sigma_, prior_mean_, frame); }
    void  predict() { ai::predict<N>(*this, Sigma_, state_, prior_mean_); }
    void  inflate_for_age(float dt_s, float dt_nominal_s)
    { ai::inflate_for_age<N>(*this, Sigma_, state_, prior_mean_, dt_s, dt_nominal_s); }
    Eigen::Matrix<float, 7, 7> predicted_information(const std::vector<Eigen::Vector3f>& pts, float R) const
    { return ai::predicted_information<N>(*this, state_, pts, R); }

    // ── Generative-model hooks (called by the engine) ─────────────────────────────────────────
    float sdf_box (const Eigen::Vector3f& p, const CabinetBeliefState& s) const;
    float sdf_prim(const Eigen::Vector3f& p, const CabinetBeliefState& s, int prim) const;
    Eigen::Matrix<float, 7, 1> sdf_jacobian(const Eigen::Vector3f& p, const CabinetBeliefState& s, int prim) const;
    std::array<float, 2> responsibilities(const Eigen::Vector3f& p, const CabinetBeliefState& s, float R) const;
    float mixture_unnormalized(const Eigen::Vector3f& p, const CabinetBeliefState& s, float R,
                               std::array<float, 2>& u) const;
    void  apply_constraints(CabinetBeliefState& s) const;
    void  canonicalize(CabinetBeliefState& s) const;   // C2v yaw fold: front normal faces the room

    int   gn_iters() const { return params_.gn_iters; }
    int   n_prims()  const { return 1; }               // one box (clutter is the +1 mixture component)
    float sigma2()   const { return params_.sigma_base_m * params_.sigma_base_m; }
    Eigen::Matrix<float, 7, 7> transition() const { return Eigen::Matrix<float, 7, 7>::Identity(); }
    Eigen::Matrix<float, 7, 1> process_noise_diag() const;
    Eigen::Matrix<float, 7, 1> prior_cov_diag() const;
    Eigen::Matrix<float, 7, 1> prior_mean_vec() const;  // tier-dependent prior MEAN (d, z0, z1)
    Eigen::Matrix<float, 7, 1> common_mode_inv_diag(const CabinetFrame& frame) const;

    // Extra GN factors: the wall-flush + wall-parallel structural term, then the shared LiDAR
    // first-hit range factor (once per device ray-set, into the SAME normal equations).
    void accumulate_extra(const CabinetBeliefState& s, const CabinetFrame& f,
                          Eigen::Matrix<float, 7, 7>& Id, Eigen::Matrix<float, 7, 1>& bd) const;

    // ── Monitor instrumentation ───────────────────────────────────────────────────────────────
    float last_wall_gap()     const { return dbg_wall_gap_; }      // back face -> wall signed gap (m)
    float last_wall_lambda()  const { return dbg_wall_lambda_; }   // applied flush precision (1/m^2)
    int   last_lidar_rays()   const { return dbg_lidar_rays_; }
    float last_span_obs()     const { return dbg_span_obs_; }      // observed along-axis span (m)
    int   last_span_pts()     const { return dbg_span_pts_; }      // points inside the run cross-section
    int   last_vacate_beams() const { return dbg_vacate_beams_; }  // through-beams that carved this frame
    float last_axis_resid()   const { return dbg_axis_resid_; }    // signed yaw error to nearest room axis (rad)
    // Wall-segment domain instrumentation: is it binding? corner projections onto the run axis + retract residuals.
    int   last_seg_active()   const { return dbg_seg_active_; }    // 1 if the wall-segment terms are live this frame
    float last_seg_tlo()      const { return dbg_seg_tlo_; }       // −u corner projected onto the run axis (m)
    float last_seg_thi()      const { return dbg_seg_thi_; }       // +u corner projected onto the run axis (m)
    float last_seg_shi()      const { return dbg_seg_shi_; }       // observed +u span end (m) — grows past thi ⇒ censored
    float last_seg_qhi()      const { return dbg_seg_qhi_; }       // +u retract residual (box end past the corner, m)

    // Global birth statistic (see RunSeed). Pure geometry, no state — shared by the fitter's birth
    // path so the seed and the per-frame extent factor agree on what "the run's axis" means.
    // room_axis_snap: seed the run on the dominant ROOM axis (X or Y) instead of the raw PCA principal
    // axis. A kitchen run lies on a room axis; an L-shaped corner-spanning mask has a DIAGONAL PCA axis
    // and a long diagonal extent, so a PCA seed births one oblique box straddling both walls (which
    // grow-only extent can never retract). Snapping births the dominant arm axis-aligned and lets the
    // perpendicular arm fall off-surface (residual). See seed_from_points / the corner-merge note.
    static RunSeed seed_from_points(const std::vector<Eigen::Vector3f>& pts, bool room_axis_snap = false);

    static bool self_test();

private:
    void accumulate_wall(const CabinetBeliefState& s, const CabinetFrame& f,
                         Eigen::Matrix<float, 7, 7>& Id, Eigen::Matrix<float, 7, 1>& bd) const;
    void accumulate_extent(const CabinetBeliefState& s, const CabinetFrame& f,
                           Eigen::Matrix<float, 7, 7>& Id, Eigen::Matrix<float, 7, 1>& bd) const;
    // Posterior weight of the "flush against this wall" mixture component: exp(−(gap/reach)²), gap measured
    // from the run's back face. ONE definition shared by accumulate_wall (flush + parallel) and
    // accumulate_extent (wall-segment domain). 0 when no wall this frame ⇒ both terms inert (island).
    float flush_weight(const CabinetBeliefState& s, const CabinetFrame& f) const;
    void accumulate_axis_alignment(const CabinetBeliefState& s,
                                   Eigen::Matrix<float, 7, 7>& Id, Eigen::Matrix<float, 7, 1>& bd) const;
    // Free-space / VACATE: the occlusion-aware upper bound that closes accumulate_extent (see params).
    void accumulate_freespace(const CabinetBeliefState& s, const CabinetFrame& f,
                              Eigen::Matrix<float, 7, 7>& Id, Eigen::Matrix<float, 7, 1>& bd) const;

    CabinetBeliefState         state_;
    CabinetBeliefParams        params_;
    Eigen::Matrix<float, 7, 7> Sigma_      = Eigen::Matrix<float, 7, 7>::Identity();
    Eigen::Matrix<float, 7, 1> prior_mean_ = Eigen::Matrix<float, 7, 1>::Zero();

    CabinetTier     tier_          = CabinetTier::Base;
    float           tier_evidence_ = 0.0f;   // accumulated log-evidence FOR the current tier
    Eigen::Vector2f room_interior_ = Eigen::Vector2f::Zero();
    bool            has_room_interior_ = false;

    mutable float dbg_wall_gap_    = 0.0f;
    mutable float dbg_wall_lambda_ = 0.0f;
    mutable int   dbg_lidar_rays_  = 0;
    mutable float dbg_span_obs_    = 0.0f;
    mutable int   dbg_span_pts_    = 0;
    mutable int   dbg_vacate_beams_ = 0;   // through-beams that carved this frame (free-space diag)
    mutable float dbg_axis_resid_  = 0.0f;
    mutable int   dbg_seg_active_  = 0;
    mutable float dbg_seg_tlo_     = 0.0f;
    mutable float dbg_seg_thi_     = 0.0f;
    mutable float dbg_seg_shi_     = 0.0f;
    mutable float dbg_seg_qlo_     = 0.0f;
    mutable float dbg_seg_qhi_     = 0.0f;
};

}  // namespace rc
