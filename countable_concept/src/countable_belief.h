/*
 * countable_belief.h  —  AI2 bed belief (see CONCEPT_AGENT_RECIPE.md, COUNTABLE notes)
 *
 * The SIMPLEST box concept: a bed is ONE free-standing, floor-anchored cuboid — no legs, no top
 * slab, no w↔h symmetry, no round/square shape selection, no footprint-moment. ONE generative primitive
 * (a solid box occupying z∈[0,H]) + a uniform clutter component, soft per-point responsibilities (a
 * mixture, no hard min()/threshold), inferred by the shared recursive variational-Laplace Gaussian filter
 * carrying a FULL 6×6 covariance. Pure Eigen, no torch, no DSR — unit-testable in isolation (self_test()).
 *
 * State θ = [cx, cy, H, w, h, yaw]  (6 DOF ≡ the recipe's [cx,cy,yaw,w,d,h]):
 *   cx,cy = room-frame centre XY   ·   H = full height (box top; box spans floor→H, FLOOR-ANCHORED)
 *   w = width (local X)            ·   h = depth (local Y)   ·   yaw = rotation about +Z.
 * The box is asymmetric (w≠h resolves the 90° swap), so there is NO w↔h mode accumulator and NO canonical
 * fold — canonicalize() is a no-op. The honest 180° front/back yaw ambiguity is NOT folded; it is left in
 * the reported σ_yaw. n_prims()==1. Static furniture: transition() = I.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <stdexcept>
#include <variant>
#include <limits>
#include <optional>
#include <Eigen/Dense>

#include "countable_class_evidence.h"
#include "countable_vertical_span.h"                      // where the box stands: floor-anchored [0,H] or hanging [H−e,H]
#include "../../common/ai_belief/parts_geometry.h"       // the solid from the manifest's one parts table (K2a)
#include "../../common/concept_manifest/shape_parts.h"   // rc::manifest::ShapePart
#include "../../common/ai_belief/recursive_laplace.h"   // shared predict/MAP/Woodbury engine
#include "../../common/ai_belief/lidar_ray_factor.h"     // shared YOLO-independent LiDAR first-hit factor
#include "../../common/ai_belief/persistent_update.h"    // persistent shared-error latents (lidar-measures §7c.2)
#include "../../common/ai_belief/beam_ray.h"             // the smooth beam model of the persistent rays (lidar-measures §7c.8)
#include "../../common/ai_belief/persistent_engine.h"    // the shared persistent engine (shared-engine Task 3): this belief is its first model
#include "countable_viewpoint.h"                          // ChainLever, chain_B, chain_transport (lidar-measures §7c.7 b)

namespace rc
{

// 2D footprint second-moment of a point band (centroid + principal axis + full extents). Kept as a small,
// model-independent geometry helper for the fitter's optional birth seed; NOT part of the fit itself
// (n_prims()==1, no footprint-moment factor). ok=false when too few points fell in the band.
struct FootprintMoment
{
    bool  ok = false;
    int   n = 0;
    float cx = 0.0f, cy = 0.0f;                  // 2D centroid (m)
    float ext_major = 0.0f, ext_minor = 0.0f;    // full extents (m), major ≥ minor
    float phi = 0.0f;                            // major-axis angle (rad), atan2 in (−π/2, π/2]
};

// The nearest room wall, as seen by THIS frame (ported from cabinet_concept, simplified — a bed is a
// SINGLE box, so there is NO run / wall-id / collinear-merge / segment machinery). Supplied by the fitter
// from the room polygon (room_concept's `delimiting_polygon_x/y`, a NOMINAL model — no circularity between
// "the wall" and the fitted bed). ok=false ⇒ the wall factor is inert this frame (free-standing).
struct WallRef
{
    bool            ok      = false;
    Eigen::Vector2f p       = Eigen::Vector2f::Zero();   // any point on the wall line (room frame, m)
    Eigen::Vector2f n       = Eigen::Vector2f::UnitX();  // UNIT normal, pointing INTO the room
    float           sigma_m = 0.02f;                     // wall position uncertainty (m)
    // ★How long the wall segment is, because that is what says how well its DIRECTION is known: a wall
    // localised to sigma_m over a length L is known angularly to ~sigma_m/L. 0 ⇒ unknown, constant used.
    float           length_m = 0.0f;                     // polygon edge length (m)
    // The edge's identity as room_concept publishes the outline (F1.4: one wall latent per (room, edge)); edge −1 = unknown
    // (the engine then falls back to its unkeyed rule).
    std::uint64_t   room = 0;
    int             edge = -1;
};

// Appearance-based FRONT (door) cue: the room-frame direction the DOOR faces + a confidence in [0,1]. Emitted
// by CountableProjection::detect_front (project the FITTED box into the live ZED RGB, score vertical-edge /
// door-ness energy per visible face, pick the winner) and consumed by CountableBelief::resolve_front. Kept in
// the pure-Eigen belief header (no OpenCV/DSR) so both the belief resolver and the projection unit share it.
struct FrontCue
{
    float bearing_rad = 0.0f;   // room-frame yaw of the winning face's OUTWARD normal (direction the door faces)
    float confidence  = 0.0f;   // (max−second)/(max+eps) door-ness margin ∈ [0,1]; high when one face clearly wins
    // ★Room-frame bearing FROM THE COUNTABLE TO THE OBSERVER. Not the same thing as bearing_rad, and the
    // distinction is the whole point: bearing_rad is a property of the OBJECT (which way the door faces),
    // while this is where the look came FROM. front_acc_ needs the latter to tell a genuinely new view
    // from the same view repeated — keying novelty on the cue direction would suppress a fresh viewpoint
    // that happens to agree. NaN ⇒ unknown ⇒ no novelty weighting (the legacy sum).
    float view_bearing_rad = std::numeric_limits<float>::quiet_NaN();
};

// Fitted bed state θ = [cx, cy, H, w, h, yaw] (room frame). See the file header for the field map.
struct CountableBeliefState
{
    float cx = 0.0f, cy = 0.0f, H = 1.70f, w = 0.60f, h = 0.60f, yaw = 0.0f;

    // Inert chart flag (kept so the fitter's `use_quotient` assignment compiles). A single asymmetric box
    // needs no symmetry-quotient optimisation chart, so the plain (identity) chart is the only path.
    inline static bool use_quotient = false;

    Eigen::Matrix<float, 6, 1> vec() const
    { return (Eigen::Matrix<float, 6, 1>() << cx, cy, H, w, h, yaw).finished(); }
    static CountableBeliefState from_vec(const Eigen::Matrix<float, 6, 1>& v)
    { return {v(0), v(1), v(2), v(3), v(4), v(5)}; }
};

// Belief parameters: the observation/mixture model, priors, and the transition + common-mode covariances.
// All are measurement-model stds/precisions (never σ-floors or gates). Several fields are carried INERT so
// the shared fitter's parameter-fill compiles unchanged; they have no effect on this single-box model and
// are noted as such.
struct CountableBeliefParams
{
    // Observation model
    float sigma_base_m    = 0.03f;   // base on-surface noise std (m); R = σ² (+ motion_var + …) per point
    float range_noise_lat_per_m = 0.0f;   // AI2RangeNoiseLatPerM: the per-point R's lateral deprojection term, (k·range)²
    float clutter_frac    = 0.10f;   // ε: prior weight of the uniform clutter component
    float clutter_scale_m = 0.15f;   // a point further than ~this from the box surface is likely clutter
    float model_sigma_m   = 0.010f;  // residual model/surface std (m) added in quadrature (R floor)

    // Priors (broad; only break the empty-cloud degeneracy)
    float prior_pos_std  = 0.30f;    // position prior std (m) on cx,cy
    float prior_size_std = 0.30f;    // (legacy/inert now that the size prior is split into footprint vs height)
    float prior_yaw_std  = 0.60f;    // broad yaw prior std (rad)

    // ── FOOTPRINT + HEIGHT prior (a standard bed footprint ≈ 0.60×0.60 is TIGHT; HEIGHT varies a lot) ──
    // A Gaussian prior on the footprint DOFs (w,h≡depth) is strong (small std), so a partial front-only view
    // cannot let the unobserved depth float; the HEIGHT prior is deliberately BROAD (data-driven). These also
    // split the previously-uniform size entry of prior_cov_diag() into footprint (tight) vs height (broad).
    float prior_footprint_m   = 0.60f;   // mean of the w and h (depth) prior (m)
    float prior_footprint_std = 0.08f;   // TIGHT → strong footprint prior (m); λ = 1/std² folded into accumulate_extra
    // DEPTH std when the class declares its own ([prior.depth] std_m: a hood's depth spreads less than its width). ≤0 = the
    // footprint std (one std for both axes, every class before the hood).
    float prior_depth_std     = -1.0f;
    float depth_std() const { return prior_depth_std > 0.0f ? prior_depth_std : prior_footprint_std; }
    // DEPTH mean (local Y, front-to-back) when the class is NOT square in plan — a bed is 1.75 wide and 2.1 long,
    // a toilet 0.58 × 0.73. ≤0 = the same as prior_footprint_m (the refrigerator parent's square footprint).
    // The asymmetry is also evidence about the 90° yaw mode: only one of the two w↔h assignments fits it.
    float prior_depth_m       = -1.0f;
    float depth_mean() const { return prior_depth_m > 0.0f ? prior_depth_m : prior_footprint_m; }
    float prior_height_m      = 1.90f;   // mean of the H (vertical) static anchor (m) — a standard tall bed
    float prior_height_std    = 0.30f;   // anchor std (m); stops the box top floating above the cloud (accumulate_extra 2a)
    // DEPTH-OBSERVABILITY prior (front-only-view depth-collapse fix): depth (h) is identifiable only when the cloud
    // spans the depth extent (front AND back seen). A front-only view is a thin ly-slab whose many points still drag
    // depth to the clamp (size common-mode caps σ, not the mean). Extra depth precision applied ∝ (1 − observed
    // depth-extent/footprint) → single-face view holds depth at the footprint prior; relaxes once the back is seen.
    float depth_unobs_precision = 1500.0f;  // extra 1/m² when depth is unobserved (single face). 0 = OFF
    float depth_obs_band_m      = 0.10f;    // ly-spread (m) below which the cloud counts as a single depth face
    // TOP anchor: the box top is floor-anchored, so a "free" upper boundary — extending it ABOVE the cloud costs
    // nothing (empty surface unpenalised), so it ratchets up on the over-segmentation junk tail above the bed
    // (observed: H→2.37 for a 1.9 m mask). A firm TWO-SIDED anchor pins H to the observed robust cloud top
    // (frame.z_top_obs, the p97 top which sits BELOW the junk tail; the tail is separately faded via per-point R in
    // the fitter). Data still sets the LOWER bound (front points force H ≥ real top). Inert when z_top_obs < 0. 0 = OFF.
    // HEADBOARD (2026-11-14, owner: fit this Webots bed, headboard NOT the box top). A fixed-template panel on the
    // BACK face (local +Y; the front is −Y), full width, thickness headboard_t_m, from the floor to headboard_h_m.
    // It is part of the SAME primitive (SDF = min(box, panel)), so a headboard point is explained by the bed and
    // pulls the pose (and, being at one end, the 180° front/back mode) but never the mattress height H. Not a DOF:
    // the template is the manifest's. headboard_h_m <= 0 = a plain box (the refrigerator parent's model).
    float headboard_h_m = -1.0f;
    float headboard_t_m = 0.10f;
    // VERTICAL SPAN (K1): the box's base. ≤0 = floor-anchored, z ∈ [0, H] (every class before the hood); > 0 = the box HANGS
    // with this vertical extent below its top H, z ∈ [H − e, H] — the manifest's [model.geometry] support = "hangs" extent_m.
    float vertical_extent_m = -1.0f;
    rc::countable::VerticalSpan span() const { return {vertical_extent_m}; }
    // THE SOLID (K2a, common/ai_belief/parts_geometry.h): the class manifest's [[shape.part]] rows and its [shape] solid key. With
    // solid_parts the model's solid (camera SDF, beam parts) is the union of these boxes (∪ the headboard panel above); without,
    // it is the whole fitted box (∪ the panel) and the rows are the existence carve's only — every class before K2a. The carve
    // (countable_existence.cpp) reads the same rows through the same header.
    std::vector<rc::manifest::ShapePart> shape_parts;
    bool solid_parts = false;
    // K2a-L: the class PROPORTIONS the rows may name ([[shape.proportion]]): each a ρ = 1 latent of the engine (mean the
    // midpoint, variance Δ²/12). At most CountableEngineModel::kProportions; none ⇒ the latent stays exactly 0 (Δ = 0).
    std::vector<rc::manifest::ShapeProportion> shape_proportions;
    rc::ai::parts::Table parts_table() const
    { return {std::span<const rc::manifest::ShapePart>(shape_parts), solid_parts, headboard_h_m, headboard_t_m}; }

    // ── CLASS EXISTENCE (spec 2026-11-14 countable class existence) ──────────────────────────────────────
    // π_cls: P(a mask carrying this class's label is really this class), from the manifest's [prior.class]. The
    // "some other furniture" alternative is a Gaussian about the class mean with std class_alt_std_m per size
    // variable. The birth height density has noise σ_z = birth_height_noise_m and an outlier fraction ε =
    // birth_height_outlier_frac (a mask that bleeds onto the wall/person above the object).
    float class_p_label_true        = 0.8f;
    float class_fit_H_m = 0, class_fit_w_m = 0, class_fit_h_m = 0;   // per-class box-fit error (m)
    float class_alt_std_m           = 1.0f;
    float birth_height_noise_m      = 0.05f;
    float birth_height_outlier_frac = 0.05f;
    // The class's size prior in the evidence's (H, w, h) order. H_top = the tallest declared part: the headboard
    // when it is taller than the mattress layer, else the H mean.
    rc::countable::ClassShapePrior class_shape_prior() const
    {
        rc::countable::ClassShapePrior c;
        c.H_mean = prior_height_m;   c.H_std = prior_height_std;
        c.w_mean = prior_footprint_m; c.w_std = prior_footprint_std;
        c.h_mean = depth_mean();      c.h_std = depth_std();
        c.H_top  = headboard_h_m > prior_height_m ? headboard_h_m : prior_height_m;
        c.p_label_true = class_p_label_true;
        c.H_fit_std = class_fit_H_m; c.w_fit_std = class_fit_w_m; c.h_fit_std = class_fit_h_m;
        return c;
    }
    float top_no_float_precision = 10000.0f; // 1/m² firm two-sided anchor pinning H → observed robust cloud top
    float top_no_float_margin_m  = 0.02f;   // upward allowance (m) added to the observed top as the anchor target

    // ── "IS THIS REALLY A COUNTABLE?" plausibility + short-height prior (model-evidence mis-detection filter) ──
    // A bed is SQUARE-ish in footprint (w≈h≈0.60) and TALL (H≈1.4–2.0 m). A YOLO mis-detection (e.g. a
    // ~70 cm elongated cabinet mislabelled "bed") contradicts these shape priors → it is poorly
    // explained by the bed model. These parameters drive a CONTINUOUS plausibility score (no hard cut) and a
    // SOFT one-sided short-height prior. See countable_plausibility() / accumulate_extra() height term.
    float plaus_aspect_scale      = 0.15f;   // aspect = |w−h|/(w+h); aspect_ok = exp(−(aspect/scale)²) → 1 square
    float plaus_size_scale        = 0.15f;   // size_ok = exp(−((w−fp)²+(h−fp)²)/(2·scale²)), fp = prior_footprint_m
    float plaus_height_min        = 1.20f;   // logistic centre (m): height_ok ≈ 0.5 here, →1 tall, →0 below ~1 m
    float plaus_height_soft       = 0.15f;   // logistic softness (m) of the height_ok falloff
    float plaus_fe_ref            = 2.0f;    // "healthy bed fit" free-energy reference (NEEDS LIVE TUNING)
    float plaus_fe_scale          = 1.0f;    // fit_ok = exp(−max(0, FE−fe_ref)/fe_scale)
    // Short-height PRIOR (accumulate_extra): a one-sided GN factor nudging H up when H < plaus_height_min, with
    // precision GROWING as H falls further below (λ = gain·(deficit/soft)). Continuous covariance, NOT a clamp:
    // fitting a 70 cm cloud as a bed then fights this prior → worse fit → lower plausibility. 0 = OFF.
    float plaus_height_prior_gain = 2000.0f; // precision (1/m²) per unit (deficit/soft) below plaus_height_min

    // ── WALL-FLUSH + WALL-PARALLEL factor (accumulate_wall; ported from cabinet_concept) ─────────────────
    // A bed's BACK face rests against a room wall; the mask sees only front/sides, so depth (h) is
    // partial-view-degenerate. The precision is the marginal of a 2-component mixture {flush against this
    // wall, free-standing}: λ_flush = wall_precision·exp(−(gap/wall_reach_m)²)/(1/wall_precision+σ_wall²).
    // The exponential IS the flush component's Gaussian posterior weight given the back-to-wall gap — a
    // CONTINUOUS covariance, not a proximity gate: a genuine mid-room bed finds no wall in range and the
    // factor fades to nothing (depth stays honestly wide). 0 = OFF.
    float wall_precision          = 400.0f;   // 1/m² at zero gap (≈1/σ² with σ≈5 cm)
    float wall_reach_m            = 0.15f;    // gap scale over which the flush hypothesis loses its weight
    // π_flush: the CLASS prior that this object is wall-mounted, used when the depth axis is unobserved
    // and the gap cannot discriminate {flush, free-standing}. 0 = no class prior (previous behaviour).
    // ★A bed keeps 0: commonly against a wall, not definitionally. Only a hood declares 1.0.
    float wall_flush_prior        = 0.0f;
    // ROOM-AXIS ALIGNMENT (furniture prior, owner 2026-11-14): yaw prefers the room's axes, 4-fold, so it never
    // decides front/back. Residual ¼·sin(4Δ) ≈ Δ near alignment ⇒ this is the angle's std (rad). ≤0 = OFF.
    float room_axis_std           = 0.0f;
    float wall_parallel_precision = 200.0f;   // on (width-axis · wall-normal): back face parallel to the wall — 0 = OFF
    // WALL NO-CROSS (one-sided): resist the back face penetrating PAST the wall into the wall/exterior. Active only
    // when the gap goes negative (crossed) or within wall_no_cross_margin_m; precision GROWS with penetration
    // (λ = wall_no_cross_precision · how-far-past), leaving flush (gap≈0) untouched. Strong; continuous, no clamp.
    // PERSISTENT PATH (persistent = true, spec §7c.6): the term is a one-sided Gaussian with FIXED precision
    // λ = 1/σ_wall² (σ_wall = the staged wall's sigma_m), and wall_no_cross_precision > 0 is ONLY its on/off switch (its
    // value is unused there); a wall with σ_wall ≤ 0 makes the term inert. The flag-off path keeps λ as above.
    float wall_no_cross_precision = 2000.0f;  // 1/m² per m of penetration past the wall (>> wall_precision). 0 = OFF
    float wall_no_cross_margin_m  = 0.0f;     // interior margin (m) inside which the no-cross term activates
    // STAND-OFF (persistent path, F1.4): how far the back face of a real object of the class stands off its wall — the class
    // manifest's [prior.attachment] standoff_m / standoff_std_m. With std > 0 the flush prior is gap − δ ~ N(standoff_m,
    // standoff_std_m²) (δ = the wall latent), entered once; std ≤ 0 keeps the old flush form (mean 0, precision wall_precision).
    float wall_standoff_m         = 0.0f;
    float wall_standoff_std_m     = 0.0f;
    // ── WALL as a COMPETING EXPLANATION (per-point mixture component) ────────────────────────────────────
    // A room wall is a surface another concept (room_concept) has already inferred, so a mask point lying on it
    // is ALREADY EXPLAINED and must not be re-explained as bed surface. This is explaining-away, NOT an
    // exclusion zone: the wall enters the SAME per-point mixture as box/clutter, so (i) a wall point's box
    // responsibility collapses ⇒ it exerts no GN pull, and (ii) the marginal likelihood is already high there
    // WITHOUT a bed ⇒ box_log_evidence_gain ≈ 0 ⇒ the hypothesis earns nothing. A real bed is untouched:
    // its visible front face stands ~h off the wall plane, so those points stay with the box.
    // wall_explain_frac is the component's PRIOR weight — it should track how much the room is trusted; the
    // fitter scales it by the room's own confidence, so bad localization returns the bed hypothesis its
    // freedom rather than silently vetoing detections. 0 = OFF (no wall component).
    // Door CLEARANCE prior (door_clearance_logprior): nats of preference, at full flush, for the door mode
    // pointing straight INTO the room over one pointing straight into the wall. Scaled by the flush mixture
    // weight, so it vanishes for a genuine mid-room bed. 0 = OFF. A precision, not a gate.
    float door_clearance_gain = 3.0f;
    // ── INFERRED VOLATILITY (MODEL_HISTORY.md §3): "how fast does THIS object change?" ──────────────────
    // Q stops being a per-frame constant and becomes exp(ω), a per-DOF latent inferred from how much the
    // belief actually MOVES. This is the history stage: with a fixed Q, evidence is discarded exactly as
    // fast as it is acquired (σ_steady = (q/i)^¼ — a ratio of two RATES, with elapsed time absent), so hours
    // of touring leave the belief where seconds would, and an unobserved object's σ walks back to the prior
    // in ~6 minutes. Inferring ω fixes BOTH: retention τ ≈ Σ·exp(−ω) grows AND the floor (exp(ω)/i)^¼ falls.
    // Stiffness is then EARNED by evidence rather than imposed by a clamp, and the hyperprior pull-back keeps
    // it recoverable — a genuinely moved bed drives ω back up and adapts in seconds.
    bool  volatility_infer   = false;   // OFF ⇒ Q is the constant process_std_* exactly as before
    float volatility_lr      = 0.02f;   // gradient step on ω per MEASURED cycle
    float volatility_sigma   = 2.0f;    // hyperprior width σ_ω: how far evidence may pull ω from ω₀
    float volatility_omega_min = -18.0f; // numerical floor on ω (exp(−18) ≈ 1.5e-8 m²) — not a tuning knob
    float wall_explain_frac  = 0.25f;   // π_wall: prior weight of "this point belongs to a known room wall"
    float wall_explain_sigma_m = 0.05f; // extra std (m) on the wall plane: room-polygon + localization slack

    // Temporal transition (predict): rigid + static ⇒ small process noise per frame.
    // PERMANENCE (persistent path, owner ruling 11-25; the class manifest's [prior.permanence]): how often a real object of the
    // class is moved, and by how much. Per predict step (predict_period_s) the object is moved with probability
    // move_rate_per_day·dt/86400, by a step of std move_scale_m on (cx, cy) and move_scale_rad on yaw; its size jumps by the class
    // size prior std (a different instance of the class in the slot). That spike-and-slab is the volatility's hyperprior
    // (VolatilityConfig::jump_*), and its mean π·M is q₀ = CountableEngineModel::process_noise_diag(). rate 0 ⇒ q₀ = 0.
    float move_rate_per_day = 0.0f;
    float move_scale_m      = 0.0f;
    float move_scale_rad    = 0.0f;
    float predict_period_s  = 0.1f;   // the agent's predict period (Model.AI2AgeNominalDtS, the mask stream's nominal period)
    float process_std_m   = 0.005f;  // cx,cy,H,w,h per-frame process std (m)
    float process_std_yaw = 0.01f;   // yaw per-frame process std (rad)

    // Per-frame COMMON-MODE error (the error SHARED by all points of one mask; does NOT average out). The
    // frame's information SATURATES at this covariance (Woodbury), so N correlated points can't collapse σ.
    float common_mode_pos_std  = 0.03f;  // shared position error (m); pose-chain cov adds to it
    float common_mode_size_std = 0.02f;  // shared size error H,w,h (m)
    float common_mode_yaw_std  = 0.03f;  // shared yaw error (rad)

    // ── PERSISTENT SHARED ERRORS + THE CLASS PRIOR COUNTED ONCE (spec 2026-11-15 lidar-measures §7c.1–7c.2) ──────────
    // false ⇒ today's per-frame Woodbury path, bit-identical (lidar_golden_test). true ⇒ the belief carries latents
    // b = (e (δx, δy, δψ), b_cam (6), b_L (6), u_cam (6), u_L (6)) beside θ in a 33×33 double covariance (§7c.7; u_c, the
    // per-frame ego-motion share, rev 3.3.13); e is the robot's pose error in
    // the room, measured through B_chain = chain_B(ℓ) with a lever ℓ held as instance state; the camera points measure
    // θ + B_chain e + b_cam, the rays θ + B_chain e + b_L; the relational anchors (wall flush/parallel, room axis) enter once,
    // at the first admissible camera update; the footprint (2) and height (2a) anchors never (Σ₀ = the class P carries them);
    // predict clamps Σθθ to the class P (the engine's clamp_to_prior congruence). Set from Model.LidarMeasuresOnGated (Task 9).
    bool  persistent = false;
    // Δ per DOF (cx, cy, H, w, h, yaw): the class's layer spread (manifest [shape].layer_spread_m, spec shared-engine G4). The
    // persistent path's shape latents s_cam / s_L are born at Δ²/12 and carried with ρ = 1. Zeros ⇒ no shape latent.
    std::array<float, 6> layer_spread_m{};
    // The smooth beam model of the persistent path's rays (spec 7c.8 a, b): the Student-t at the sensor's precision and the
    // composite rule. Set by the fitter through rc::make_beam_config (countable_beam_config.h), never default-constructed: a
    // BeamConfig{} has no quantile levels and would integrate on the geometric breakpoints only, so the belief does not use
    // one (beam_ready() false ⇒ the rays are not a measurement, logged once per belief). Read only when persistent.
    rc::ai::beam::BeamConfig beam;

    // Optimiser
    int   gn_iters = 4;              // Gauss-Newton iterations per frame (the flag-off engine path)
    // Persistent path only (spec 7c.7 d + 7c.8 rev 3.3.7; solver constants, no config key): damped GN (Nielsen's λ rule, joint
    // damping) stops converged when the UNDAMPED (λ = 0) proposal from the current θ is below gn_tol_sigma·σ_k on every θ
    // component (rev 3.3.8; σ_k the posterior's at the current linearisation) — that step is applied if J does not rise (controller ruling, Task 19 fix round 2; an accepted damped step
    // below the tolerance no longer stops: under Nielsen's λ it is short however far the minimum is); else after
    // gn_iters_max = 25 (rev 3.3.7) steps (gn_conv = 0). A solver tolerance, not a modelling threshold. gn_reject_all_for_test is a TEST HOOK only (every step rejected, as if J never fell).
    int    gn_iters_max = 25;
    double gn_tol_m     = 1e-3;   // the 7c.7 test hook's tolerance (set_first_crossing_for_test)
    double gn_tol_rad   = 1e-3;
    // spec 7c.8 rev 3.3.8 "Convergence in posterior σ": stop when the undamped step is below gn_tol_sigma·√P_kk on every θ
    // component, P = (𝟙 + P̄W)⁻¹P̄ at the current linearisation. A solver tolerance, not a model threshold.
    double gn_tol_sigma = 0.1;
    bool   gn_reject_all_for_test = false;
    // DIAGNOSTIC only (Model.GnTrace; default 0 = off; no model effect): 1 = record every GN step of every persistent update
    // and, on updates that end unconverged (gn_conv = 0), a finite-difference check of each channel's gradient and curvature
    // at θ₀ and at the returned θ; 2 = the check on every update. Read by the fitter into gn_steps.csv / gn_fd.csv.
    int    gn_trace = 0;
    float fd_eps   = 1e-3f;          // finite-difference step for SDF Jacobians (m / rad)
    float jac_slope_clamp = 5.0f;    // clamp per-DOF SDF slope (guards a seam-point info spike; recipe §2)

    // ── INERT compatibility fields (set by the shared fitter; no effect on this single-box model) ────────
    float top_thickness = 0.03f;     // legacy (no top slab here)
    float leg_radius    = 0.025f;    // legacy (no legs here)
    float pixel_sigma_over_f     = 0.0015f;   // (footprint-residual path — OFF)
    float depth_sigma0_m         = 0.006f;
    float depth_sigma_range_coef = 0.004f;
    bool  footprint_residual     = false;
    float depth_bias_std         = 0.015f;
    float depth_scale_std        = 0.010f;
    float coverage_precision  = 0.0f;
    float coverage_robust_c_m = 0.15f;
    float free_space_precision = 0.0f;
    float footprint_moment_precision = 0.0f;
    float footprint_moment_completeness_gain = 0.0f;
    float footprint_moment_min_completeness  = 0.02f;
};

// One fitted frame's evidence: room-frame points and per-point measurement variance R (m²). Empty R ⇒ base
// σ² for every point. The chain/range/motion common-mode variances are folded into common_mode_inv_diag.
struct CountableFrame
{
    std::vector<Eigen::Vector3f> points;
    std::vector<float>           R;                 // per-point measurement variance (m²); empty ⇒ σ_base²
    Eigen::Vector3f              cam_origin = Eigen::Vector3f::Zero();  // (inert: footprint-residual path OFF)
    bool                         has_rays   = false;                    // (inert)
    std::vector<float>           point_azim;                            // (inert)
    float chain_cov_xx   = 0.0f;   // extra shared position variance (m²) from the pose chain + range + motion (cx)
    float chain_cov_yy   = 0.0f;   // ...                                                                      (cy)
    float chain_cov_yaw  = 0.0f;   // extra shared yaw variance (rad²): range + grazing-view cap + ego-motion
    float chain_cov_size = 0.0f;   // extra shared SIZE variance (m²) on H,w,h: range freezes afar + ego-motion
    float moment_extra_var = 0.0f; // (inert: no footprint-moment factor on a single asymmetric box)

    // Robust (high-quantile) observed cloud TOP z (m); the box top must not FLOAT above it (top-no-float factor).
    // <0 ⇒ not supplied this frame (factor inert). See accumulate_extra (2d).
    float z_top_obs = -1.0f;

    // The nearest room wall for the wall-flush / wall-parallel factor (see WallRef). ok=false ⇒ inert.
    WallRef wall;
    // The room's principal axis (rad, room frame; defined modulo 90°), from the room polygon's edges.
    bool  room_axis_ok = false;
    float room_axis    = 0.0f;

    // YOLO-INDEPENDENT LiDAR channel: range returns that fall on the box (room frame, sensor origin +
    // endpoints). Sphere-traced against THIS belief's own SDF by the shared factor. precision==0 ⇒ skipped.
    rc::ai::LidarRays lidar;
    // Additional per-device ray-sets (e.g. the low "bpearl" LiDAR); each keeps its OWN origin.
    std::vector<rc::ai::LidarRays> lidar_extra;

    // A frame that carries LiDAR rays ONLY (no camera points; spec 2026-11-15 lidar-measures §7b, C1 = A). It adds what was
    // measured on it (the rays) plus the ONE-SIDED constraints that are zero unless violated ((2b) short height, (2c) wall
    // no-cross). The Gaussian class anchors (flush/parallel, room axis, footprint, height) are NOT added: the predicted
    // belief already contains them, and re-adding them each frame would count the prior as a measurement.
    bool lidar_only = false;

    // ── Persistent shared errors (spec lidar-measures §7c.2). READ ONLY WHEN params.persistent; the defaults (ρ = 0, zero
    //    variances) are never read on the flag-off path. Order of the 6-vectors: θ's [cx, cy, H, w, h, yaw]. ──────────────
    // Lag-1 correlation of each latent since ITS channel's previous update of this instance (1 = parked, 0 = a new view).
    std::array<float, 6> rho_cam{};      // camera points' sensor-specific shared error b_cam
    std::array<float, 6> rho_L{};        // LiDAR rays' sensor-specific shared error b_L
    // The robot's pose error e (spec 7c.7 b): ρ_chain = exp(−½cᵀΣ_pose⁻¹c), c the localiser's SE(2) correction since e's
    // last update; Σ_pose = the robot's pose covariance in the ROOM on (x, y, ψ) at the capture stamp (double end to end,
    // ruling T12-4); chain_dr = the robot's room-frame displacement since this instance's last update (odometry; it moves
    // the lever and transports e by F = chain_transport(chain_dr)); robot_xy = the robot's room position at the capture
    // stamp (where the lever is taken when e is a new source).
    float           rho_chain  = 0.0f;
    Eigen::Matrix3d sigma_pose = Eigen::Matrix3d::Zero();
    Eigen::Vector2f chain_dr   = Eigen::Vector2f::Zero();
    Eigen::Vector2f robot_xy   = Eigen::Vector2f::Zero();
    // The camera's room position (floor plane) at this frame — where the offer gain's ρ path starts (shared-engine G5;
    // FrameLatents::camera_xy). Absent ⇒ robot_xy.
    std::optional<Eigen::Vector2f> camera_xy;
    // The latents' marginal variances at THIS frame (σ_now²): b_cam = floors + range + obliquity + moments (no chain, no
    // ego-motion); b_L = the config floors. 0 = a zero-variance latent.
    std::array<float, 6> sig2_cam{};
    std::array<float, 6> sig2_L{};
    // Per-frame FRESH (ego-motion) variances of each channel, marginalised inside the channel (ρ = 0 by construction).
    std::array<float, 6> fresh_cam{};
    std::array<float, 6> fresh_L{};
    // The persistent path's ray-set (spec 7c.8 a0): every return whose ray crosses the select box, with its persistent label m
    // and the set's Σq (rc::lidar_select::select_rays_beam; the caller sets sigma_s = model_sigma_m). READ ONLY WHEN
    // params.persistent, in place of `lidar` / `lidar_extra` (which the flag-off path and the instruments keep reading). The
    // LiDAR channel measured ⇔ the set is not empty and Σq > 0 (CountableBelief::lidar_measured; spec 7c.8 rev 3.3.11: the
    // energy carries no sweep weight, beam.kappa is an instrument).
    rc::ai::beam::BeamRays beam;
    // G3 (spec shared-engine; READ ONLY WHEN params.persistent): the other objects' previous-cycle published beliefs, a fixed
    // foreign density for this update — the camera points' neighbour mixture (FrameLatents::foreign_boxes) and, staged by
    // feed_beam, the rays' p_f. mask_error_rate = ε, their prior weight (Model.MaskPixelErrorRate). Empty ⇒ the old path.
    std::vector<rc::ai::beam::ForeignBox> foreign_boxes;
    double mask_error_rate = 0.0;
};

class CountableBelief;

// The countable belief as a model of the shared persistent engine (rc::ai::engine::PersistentModel; shared-engine Task 3). A
// view on one CountableBelief: the engine calls these hooks, which read (and, for the state and its covariance mirror, write)
// that belief. Its accumulate_extra is the PERSISTENT θ-only channel ((1) flush/parallel and the room axis while not yet in,
// (2b), (2c)), not CountableBelief::accumulate_extra (the flag-off path's, which the recursive-Laplace engine calls).
struct CountableEngineModel
{
    static constexpr int N = 6;
    static constexpr int kYaw = 5;                 // θ = [cx, cy, H, w, h, yaw]
    static constexpr int kHeight = 2;              // H: the DOF the LiDAR's top-grazing ring decorrelates (ρ_L height)
    using State = CountableBeliefState;
    using Frame = CountableFrame;
    using Mat6 = Eigen::Matrix<float, 6, 6>;
    using Vec6 = Eigen::Matrix<float, 6, 1>;

    CountableBelief* b = nullptr;

    // camera points (recursive_laplace.h hooks) and their energy
    float sdf_prim(const Eigen::Vector3f& p, const State& s, int prim) const;
    Vec6  sdf_jacobian(const Eigen::Vector3f& p, const State& s, int prim) const;
    std::array<float, 3> responsibilities(const Eigen::Vector3f& p, const State& s, float R) const;
    int   n_prims() const;
    float sigma2() const;
    double point_energy(const Eigen::Vector3f& p, const State& s, float R) const;
    // a STANDING look's per-point R at range d (the offer gain, shared-engine G5): the fitter's R without the ego-motion
    // term — σ_base² + (AI2RangeNoiseLatPerM·d)²; mask_depth_var is 0 for the ZED masks that fit (ricoh slices never fit)
    float standing_point_variance(float range_m) const;
    // The offer's context (shared-engine G5; PersistentEngine's HasOfferContext): what the camera-extra and θ-only terms read
    // from a frame besides its points — the wall (flush, no-cross, two_sided_fraction), the room axis, the robust top — and the
    // frame's common-mode fields, kept with them. offer_frame builds a frame of only these; offer_look puts a look's points
    // in it, and a look without points is LiDAR-only (lidar_only = points.empty()), as the fitter's frames are.
    struct OfferContext
    {
        WallRef wall;
        bool  room_axis_ok = false;
        float room_axis = 0.0f;
        float z_top_obs = -1.0f;
        float chain_cov_xx = 0.0f, chain_cov_yy = 0.0f, chain_cov_yaw = 0.0f, chain_cov_size = 0.0f;
    };
    OfferContext offer_context(const Frame& f) const;
    Frame offer_frame(const OfferContext& c) const;
    void  offer_look(Frame& f, std::span<const Eigen::Vector3f> pts, std::span<const float> R) const;
    // the camera channel's non-point terms: (2d) top and the depth_unobs_precision shares (flush boost, footprint boost)
    void accumulate_camera_extra(const State& s, const Frame& f, Mat6& Id, Vec6& bd, double* E) const;
    // the θ-only channel
    void accumulate_extra(const State& s, const Frame& f, Mat6& Id, Vec6& bd, double* E = nullptr) const;
    // the wall latent (F1.4): one per (room, edge), and the wall priors it carries, entered once at s = θ + B_w b
    static constexpr int kWallLatents = 1;
    rc::ai::engine::WallAnchor wall_anchor(const Frame& f) const;
    void accumulate_wall_prior(const State& s, const Frame& f, Mat6& Id, Vec6& bd, double* E) const;
    // the 7c.7 first-crossing LiDAR channel (set_first_crossing_for_test only); false = the engine's beam channel
    bool accumulate_lidar_override(const State& s, const Frame& f, Mat6* Id, Vec6* bd, double* E) const;
    // the beam channel
    std::vector<rc::ai::beam::BeamPart> beam_parts(const State& s) const;
    const rc::ai::beam::BeamRays&       beam_rays(const Frame& f) const { return f.beam; }
    const rc::ai::beam::BeamConfig&     beam_config() const;
    // K2a-L: the class shape proportions (persistent_engine.h kProp). ONE slot for every countable class (the engine's latent
    // layout is fixed at compile time); a class declaring none keeps it at Δ = 0 (exactly zero, zero variance).
    static constexpr int kProportions = 1;
    Eigen::Matrix<double, 1, 1> proportion_mean() const;     // the midpoint (lo + hi)/2; 0 when none is declared
    Eigen::Matrix<double, 1, 1> proportion_spread() const;   // Δ = hi − lo; 0 when none is declared
    Eigen::Matrix<double, 1, 1> proportions() const;         // the q the model evaluates at now
    void set_proportions(const Eigen::Matrix<double, 1, 1>& q) const;
    Eigen::Matrix<float, 1, 1> sdf_jacobian_q(const Eigen::Vector3f& p, const State& s, int prim) const;   // ∂sdf/∂q (central FD)
    double proportion_extent() const;   // H − z0 at the state: the band a proportion scales (the q common mode's σ maps through it)
    // e measured through B_chain = chain_B(ℓ) (spec 7c.7 b); θ does not enter it
    Eigen::Matrix<double, 6, 3> chain_jacobian(const State&, const Eigen::Vector2f& lever) const { return rc::ai::view::chain_B(lever); }
    void apply_constraints(State& s) const;
    void canonicalize(State& s) const;
    Vec6 shape_spread() const;   // Δ, the class's layer spread (params.layer_spread_m)
    Vec6 prior_cov_diag() const;
    Vec6 process_noise_diag() const;
    // the class prior at the class mean (spec 7c.1 amendment 11-17): seed cx, cy; class H, w, depth; the room axis' multiple
    // of 90° nearest the seed yaw when f.room_axis_ok, else the seed yaw
    Eigen::Matrix<double, 6, 1> class_prior_mean(const State& seed, const Frame& f) const;
    const State& state() const;
    void  set_state(const State& s);
    Mat6  covariance() const;
    void  set_covariance(const Mat6& S);
    rc::ai::engine::SolverConfig solver() const;
    rc::ai::engine::VolatilityConfig volatility() const;
    Vec6   jump_var() const;     // the class permanence prior's move variance per DOF (F1 round 6)
    double jump_prob() const;    // its move probability per predict step
};

// K2a-L: the same model with NO proportion slot. A class that declares no [[shape.proportion]] runs on this one, so its engine
// is the pre-K2a-L engine exactly (K = 3 + 6N + W, every matrix the size it was: bit-identical, which a zero-variance extra
// latent is not — the larger LU/GEMM sum in another order, 7e-12 relative on P, measured on the p1r8 toilet dump).
struct CountableEngineModelBox : CountableEngineModel
{
    static constexpr int kProportions = 0;
};

// The bed generative model wired onto the shared rc::ai recursive-Laplace engine. Supplies only the
// model hooks the engine calls (box SDF, responsibilities, Jacobian, constraints, and the Q/F/prior/common-
// mode diagonals) + the optional LiDAR range factor. N = 6, static (transition = I), one primitive.
class CountableBelief
{
public:
    static constexpr int N = 6;
    // K2a-L: the engine is ONE of two, chosen at construction and held in a std::variant (fix round 1, I2): Engine (no proportion
    // slot) for a class that declares none — bit-identical to before — and EngineQ (the proportion's latents appended after the
    // wall latent) for a class that does (the toilet). set_params refuses a change of the declared proportions (it would need
    // the other engine, un-seeded).
    using Engine  = rc::ai::engine::PersistentEngine<CountableEngineModelBox>;
    using EngineQ = rc::ai::engine::PersistentEngine<CountableEngineModel>;
    bool uses_q() const { return std::holds_alternative<EngineQ>(engine_); }
    CountableEngineModelBox model_of(const Engine&) const { return CountableEngineModelBox{{const_cast<CountableBelief*>(this)}}; }
    CountableEngineModel    model_of(const EngineQ&) const { return CountableEngineModel{const_cast<CountableBelief*>(this)}; }
    // the active engine handed to f (std::visit); defined before their uses, as their return types are deduced
    template <class F> decltype(auto) with_engine(F&& f) const { return std::visit(std::forward<F>(f), engine_); }
    template <class F> decltype(auto) with_engine_mut(F&& f) { return std::visit(std::forward<F>(f), engine_); }
    using State = CountableBeliefState;
    using Frame = CountableFrame;

    CountableBelief() = default;
    CountableBelief(const CountableBeliefState& s, const CountableBeliefParams& p) : state_(s), params_(p)
    {
        Sigma_.setZero(); Sigma_.diagonal() = prior_cov_diag();
        if (not p.shape_proportions.empty()) engine_.emplace<EngineQ>();
    }

    const CountableBeliefState&     state()      const { return state_; }
    const Eigen::Matrix<float, 6, 6>&  covariance() const { return Sigma_; }
    const CountableBeliefParams&    params()     const { return params_; }
    void set_state(const CountableBeliefState& s) { state_ = s; }
    // Refuses (throws std::logic_error) a change in whether the class declares proportions: the engine was chosen for it.
    void set_params(const CountableBeliefParams& p)
    {
        if (p.shape_proportions.empty() != params_.shape_proportions.empty())
            throw std::logic_error("CountableBelief::set_params: the declared shape proportions changed; the engine was chosen at "
                                   "construction for them — build a new belief");
        params_ = p;
    }

    // Room interior reference (polygon centroid). Used ONLY to pick which of the two depth faces is the
    // BACK face (the one whose outward normal points AWAY from the interior) — the face driven onto the wall.
    void set_room_interior(const Eigen::Vector2f& c) { room_interior_ = c; has_room_interior_ = true; }

    // Monitor instrumentation for the wall factor.
    float last_wall_gap()    const { return dbg_wall_gap_; }     // back face → wall signed gap (m)
    float last_wall_lambda() const { return dbg_wall_lambda_; }  // applied flush precision (1/m²)
    float last_wall_resp()   const { return dbg_wall_resp_; }    // mean per-point WALL responsibility (0..1):
                                                                 // →1 ⇒ the cloud is a wall, not a bed
    // Stage the wall the per-point mixture explains with, WITHOUT running an update. Only self_test needs this
    // (production stages it via update(frame)); it lets the test score clouds against a known wall directly.
    void set_frame_wall_for_test(const WallRef& w) { point_wall_ = w; }
    // Centre of the BACK face (the one against the wall): (cx,cy) + (h/2)·outward, outward = the depth axis
    // sign pointing AWAY from the room interior. Public so the fitter can query the nearest wall at it.
    Eigen::Vector2f back_centre(const CountableBeliefState& s) const;
    // The back face against a staged wall, ONE source for every wall term (F1 round 6.4; countable_belief.cpp)
    struct WallTerms { float gap = 0.0f, par = 0.0f; Eigen::Matrix<float, 6, 1> Jg = Eigen::Matrix<float, 6, 1>::Zero(),
                       Jp = Eigen::Matrix<float, 6, 1>::Zero(); };
    float     wall_gap(const CountableBeliefState& s, const WallRef& w) const;
    float     wall_par(const CountableBeliefState& s, const WallRef& w) const;
    WallTerms wall_terms(const CountableBeliefState& s, const WallRef& w, bool clamp) const;
    float     wall_parallel_lambda(const WallRef& w) const;

    // ── Inference (delegated to the shared engine) ────────────────────────────
    // Stash this frame's wall BEFORE the engine runs: the per-point mixture's wall component (explaining away)
    // needs it, and mixture_unnormalized is a const hook the engine calls with no access to the frame.
    // ★A degenerate Σ is SELF-SEALING and must never survive a cycle. Σ = 0 (or NaN/inf) makes the tracker's
    // Mahalanobis gate S = P + R²I collapse, so NO detection can ever associate again, so the belief can never
    // be updated, so Σ can never recover — the instance freezes at whatever garbage state it held and sits
    // there forever. Live (etc/ai2_log.csv 20:27): a volatility blow-up drove Q to inf, Σ through the inverse
    // to exactly 0 with σ_yaw = inf, and the instance took 3 measured frames out of 946 before freezing as a
    // 0.126 m slab at −102°. Re-seeding from the prior costs one cycle of confidence and breaks the trap.
    // (Belongs in common/ai_belief eventually — every agent on this engine has the same exposure.)
    void sanitize_covariance()
    {
        if (params_.persistent) { sanitize_joint(); return; }
        bool bad = not Sigma_.allFinite();
        for (int i = 0; i < N and not bad; ++i) bad = not (Sigma_(i, i) > 0.0f);
        if (not bad) return;
        Sigma_.setZero();
        Sigma_.diagonal() = prior_cov_diag();
        omega_ = omega0_; vol_n_ = 0.0f; vol_s_.setZero();   // the volatility estimate is suspect too
        size_lik_ = {};                                      // and so is the data summary built beside it
        std::printf("CountableBelief: DEGENERATE Σ (non-finite or non-positive diagonal) — re-seeded from prior\n");
    }

    float update(const CountableFrame& frame)
    {
        if (params_.persistent) return update_persistent_cycle(frame);
        point_wall_ = frame.wall;
        sanitize_covariance();
        const Eigen::Matrix<float, 6, 1> before = state_.vec();
        const float fe = ai::update<N>(*this, state_, Sigma_, prior_mean_, frame);
        sanitize_covariance();
        predict_size_likelihood(1.0f);                                 // ai::update predicted first, even with no points
        if (not frame.points.empty()) update_size_likelihood(frame);   // BEFORE update_volatility: same Q as the filter used
        // ω is inferred ONLY on a cycle that actually took a measurement. A predict-only cycle has δθ = 0,
        // which under the update below would read as "this object never moves" and drive ω down — the same
        // repetition-is-not-independence error that has bitten this codebase four times.
        // A LiDAR-only frame does not update ω either (spec lidar-measures §7b): its shift is the rays' correction of a
        // belief the camera cycles built, not a measurement of how much the object moves.
        if (not frame.lidar_only) update_volatility(state_.vec() - before);
        return fe;
    }
    // The engine runs a point-free frame as a measurement when this says it carries data: a LiDAR-only frame with rays of
    // usable precision. A point-free frame that is not lidar_only stays predict-only, exactly as before.
    bool has_extra_data(const CountableFrame& f) const
    {
        if (not f.lidar_only) return false;
        const auto usable = [](const rc::ai::LidarRays& r) { return not r.endpoints.empty() and r.precision > 0.0f; };
        if (usable(f.lidar)) return true;
        return std::ranges::any_of(f.lidar_extra, usable);
    }
    // "The LiDAR channel measured" (spec 7c.8 a0, R6-I4). Flag off: today's rule, a ray-set with endpoints and precision > 0
    // (f.lidar or an extra set). Flag on: the beam set staged non-empty with Σq > 0 (rev 3.3.11; f.lidar is not read), AND a BeamConfig
    // built by make (beam_ready; else logged once and not a measurement). frame_latents (the engine's LiDAR flag: the b_L predict, the channel) and the fitter's
    // rays_usable / record_channels read this; a sweep with Σq = 0 (the camera attributes nothing) is not a measurement (the belief stays at the prediction).
    bool lidar_measured(const CountableFrame& f) const;
    // The beam config was built by BeamConfig::make (it carries the quantile levels of the composite rule).
    bool beam_ready() const { return not params_.beam.e_levels.empty(); }
    // The model's convex parts for the beam path (spec 7c.8 a, R6-I1; K2a: common/ai_belief/parts_geometry.h): the box, or the
    // [[shape.part]] boxes of a solid = "parts" class, then the headboard panel when there is one (local y ∈ [h/2 − t, h/2],
    // t = min(headboard_t_m, h), z ∈ [0, headboard_h_m]). J = ∂(cx, cy, cz, yaw, hx, hy, hz)/∂θ, analytic.
    int n_beam_parts() const { return params_.parts_table().n_model_boxes(); }
    // The solid's table AT THE CURRENT PROPORTIONS (K2a-L): what the SDF, the beam parts and the existence carve read.
    rc::ai::parts::Table parts_table() const;
    // The proportions' current values (the engine's posterior mean after an update; the midpoints before) and their posterior σ
    // (the latent's marginal; 0 before it is born or when none is declared).
    double proportion_value(int k = 0) const
    { const auto& q = q_now(); return k < int(q.size()) ? q[std::size_t(k)] : std::numeric_limits<double>::quiet_NaN(); }
    double proportion_std(int k = 0) const
    { return uses_q() and k < CountableEngineModel::kProportions ? std::get<EngineQ>(engine_).latent_std()(EngineQ::kProp + k) : 0.0; }
    std::vector<rc::ai::beam::BeamPart> beam_parts_for_test(const CountableBeliefState& s) const { return beam_parts(s); }
    float sdf_dq_for_test(const Eigen::Vector3f& p, const CountableBeliefState& s) const { return model().sdf_jacobian_q(p, s, 0)(0); }
    // The last persistent update's beam log (spec 7c.8 stated limit 1): the staged ray count of the frame (0 when the LiDAR
    // channel did not measure), and the count of rays with π_i > 0.99 and e_i < −5c at the returned θ (e at the model's
    // nominal surface; a LOG counter, nothing reads it). last_beam_ms: wall time spent in the beam terms of that update.
    int    last_beam_staged() const { return with_engine([](const auto& e) -> decltype(auto) { return e.last_beam_staged(); }); }
    int    last_gap_rays()    const { return with_engine([](const auto& e) -> decltype(auto) { return e.last_gap_rays(); }); }
    double last_beam_ms()     const { return with_engine([](const auto& e) -> decltype(auto) { return e.last_beam_ms(); }); }
    // What the DATA alone says about the size (H, w, h): the posterior already contains the class size prior (blocks 2 and
    // 2a of accumulate_extra) every frame, so scoring it against the class would count that prior twice. Zero info until
    // the first frame with points. Spec 2026-11-14 countable class existence §9.
    const rc::countable::SizeGaussian& size_likelihood() const { return size_lik_; }

    // Per-DOF log-volatility ω (Q = exp ω) and its hyperprior mean ω₀ — the accumulated experience.
    // ω = log Q per DOF: the engine's on the persistent path (F1.2), else the legacy estimate's.
    Eigen::Matrix<float, 6, 1> log_volatility() const;
    // Retention time constant per DOF: τ = Σ_ii / exp(ω_i) frames — "how long this DOF remembers".
    Eigen::Matrix<float, 6, 1> retention_frames() const;
    void  predict()
    {
        sanitize_covariance();
        if (params_.persistent) predict_theta_joint(1.0f);
        else ai::predict<N>(*this, Sigma_, state_, prior_mean_);
        predict_size_likelihood(1.0f);
    }
    void  inflate_for_age(float dt_s, float dt_nominal_s)
    {
        if (params_.persistent) predict_theta_joint(dt_nominal_s > 1e-6f ? dt_s / dt_nominal_s : 1.0f);   // ai::inflate_for_age's q_scale
        else ai::inflate_for_age<N>(*this, Sigma_, state_, prior_mean_, dt_s, dt_nominal_s);
        predict_size_likelihood(dt_nominal_s > 1e-6f ? dt_s / dt_nominal_s : 1.0f);   // same q_scale as inflate_for_age
    }
    Eigen::Matrix<float, 6, 6> predicted_information(const std::vector<Eigen::Vector3f>& pts, float R) const
    { return ai::predicted_information<N>(*this, state_, pts, R); }
    // The persistent path's offer gain (shared-engine G5): the θ-marginal entropy drop of a camera look from standpoint `s`
    // under the engine's own joint update (PersistentEngine::predicted_gain), and what the last update delivered (the same
    // quantity; NaN after a predict-only update).
    double predicted_gain(const rc::ai::offer::Standpoint& s, const rc::ai::offer::OfferSensors& cam) const
    { return with_engine([&](const auto& e) { return e.predicted_gain(model_of(e), s, cam); }); }
    double last_delivered_nats() const { return with_engine([](const auto& e) -> decltype(auto) { return e.last_delivered_nats(); }); }

    // ── Persistent shared errors (params.persistent; spec lidar-measures §7c.1–7c.2) ─────────────────────────────────
    // The update runs on the shared engine (rc::ai::engine::PersistentEngine, common/ai_belief/persistent_engine.h; shared-engine
    // Task 3), this belief its model (CountableEngineModel). Joint state x = (θ, b), b = (e (δx, δy, δψ), b_cam, b_L, u_cam, u_L,
    // s_cam, s_L) (6 each but e); covariance() is P's θθ block in float (§7c.7). u_c (spec 7c.8 rev 3.3.13): the channel's
    // ego-motion share, a NEW source every persistent update. s_c: the shape misfit, zero variance until shared-engine Task 5.
    static constexpr int K = Engine::K;                   // latent DOFs of a class with no proportion (40); EngineQ::K = K + 1
    static constexpr int kChain = Engine::kChain, kCam = Engine::kCam, kL = Engine::kL;   // offsets inside b (e = kChain..+2)
    static constexpr int kFc = Engine::kFc, kFL = Engine::kFL, kSc = Engine::kSc, kSL = Engine::kSL;
    using MatJd = Engine::MatJd;
    using VecKd = Engine::VecKd;
    // K-sized views (dynamic: K + 1 for a class with a proportion, the proportion last).
    Eigen::VectorXd latent_mean() const { return with_engine([](const auto& e) { return Eigen::VectorXd(e.latent_mean()); }); }
    Eigen::VectorXd latent_std()  const { return with_engine([](const auto& e) { return Eigen::VectorXd(e.latent_std()); }); }
    Eigen::MatrixXd joint_covariance() const { return with_engine([](const auto& e) { return Eigen::MatrixXd(e.P()); }); }
    // e's lever ℓ (object − robot, room frame): set when e is a new source, then moved only by the odometry (spec 7c.7 b).
    const ChainLever&      chain_lever()      const { return with_engine([](const auto& e) -> decltype(auto) { return e.chain_lever(); }); }
    // The localisation share a publisher adds to covariance()'s (cx, cy) variances, given the fitter's per-frame chain cov
    // (compute_chain_cov). Flag off: that chain cov (the Woodbury path draws it afresh every frame and does not carry it).
    // Flag on: nothing — Σ_θθ is the joint's θθ marginal, which already integrates e through B_chain (spec 7c.7 b; adding
    // the chain cov again would count the robot's pose error twice, ruling T13-5).
    Eigen::Vector2f published_chain_var(float chain_xx, float chain_yy) const
    { return params_.persistent ? Eigen::Vector2f::Zero() : Eigen::Vector2f(chain_xx, chain_yy); }
    const Eigen::Matrix3d& last_sigma_pose()  const { return with_engine([](const auto& e) -> decltype(auto) { return e.last_sigma_pose(); }); }
    // The relational class anchors (ruling T6-1): the room axis enters once, on the first admissible camera update whose frame
    // stages it (axis_prior_added_). The WALL priors are the wall latent's (F1.4, F1 round 6.3): the engine enters them once per
    // physical wall at the instance's first association with it — no countable flag (the old wall_prior_added_ was dead on this
    // path and is gone). No recall path stores them or the latents: an LTOM recall cold-starts the belief in run_inference's lazy
    // init, so the anchors re-enter once and every latent is a new source. sanitize_covariance clears both (the priors re-enter
    // once). set_axis_prior_added is a test hook.
    bool wall_latent_set() const { return with_engine([](const auto& e) -> decltype(auto) { return e.wall_latent_set(); }); }   // the wall latent exists (its priors entered, F1.4)
    // F1.4: the wall latent's current mean for a staged wall (0 when the instance has none for that edge): the offset the
    // camera mixture and the beam's foreign density shift that wall by (mean-field, option A).
    float wall_offset_mean(const WallRef& w) const;
    // The birth seed's H (F1 round 6.2, the bed-depth diagnosis 11-25): the class-prior MAP N(prior_height_m, prior_height_std²)
    // combined with the robust top (weighted p95 of z) of the points the BOX explains, each point weighted by its box-versus-
    // headboard share under the box ∪ headboard shape at this belief's pose (exp(−d²/2R) per part, R = sigma_base²), the top's
    // own std = sigma_top (the detector's boundary error). EM: H₀ = the class mean, then shares → top → MAP, 10 iterations.
    // Without a headboard every share is 1. The p95 of the WHOLE cloud took the bed's headboard (1.02 against a box top 0.65).
    float seed_height(const std::vector<Eigen::Vector3f>& pts, float sigma_top) const;
    bool  last_wall_entered() const { return with_engine([](const auto& e) -> decltype(auto) { return e.last_wall_entered(); }); }
    double last_jump_p() const { return with_engine([](const auto& e) -> decltype(auto) { return e.last_jump_p(); }); }   // P(moved | data) of the last update (NaN: none)
    bool axis_prior_added() const { return axis_prior_added_; }
    void set_axis_prior_added(bool axis) { axis_prior_added_ = axis; }
    // The CLASS prior at the class mean, counted once (spec 7c.1 amendment 11-17): the instance's FIRST measured update (camera
    // or LiDAR-only, whichever comes first) has prior mean x̄_θ = (seed cx, seed cy, class H, class w, class depth, the room
    // axis' multiple of 90° nearest the seed yaw when f.room_axis_ok else the seed yaw) and P̄_θθ = diag(prior_cov_diag()),
    // cross terms 0; the data seed is only the GN start θ₀. From the second update on the recursion is unchanged (x̄ = the
    // previous posterior). The flag is set by that update, cleared by sanitize_covariance's re-seed, and travels with every
    // value copy (the fitter's step-bound snapshot). set_first_update_done is a test hook.
    // RECALL (ruling T6c-3): a belief re-created from a remembered POSTERIOR must be marked first_update_done = true, or the
    // class prior is pulled in a second time. No such path exists in this agent today (verified 11-17): an LTOM recall
    // (specificworker_lifecycle.cpp, rc::choose_recall → create_instance_from_detection) carries only the remembered number
    // and uid; the belief is built in exactly one place, run_inference's lazy init (countable_fitter.cpp,
    // `inst.ai2_belief = CountableBelief(s0, p)`), from THIS frame's cloud seed (or the tracker's birth defaults) — a data
    // seed, so the first-update class prior is correct there. A future path that restores a remembered state must call
    // set_first_update_done(true) behind params.persistent.
    bool first_update_done() const { return with_engine([](const auto& e) -> decltype(auto) { return e.first_update_done(); }); }
    void set_first_update_done(bool v) { with_engine_mut([&](auto& e) { e.set_first_update_done(v); }); }
    // spec 7c.8 rev 3.3.14: the birth cloud's EXCESS position covariance (room frame), added to σ_pos² on the (cx, cy) block of
    // the first update's class prior and of the birth Σ. Zero ⇒ today's diagonal σ_pos².
    void set_birth_position_cov(const Eigen::Matrix2f& excess)
    {
        Sigma_.topLeftCorner<2, 2>() += excess;
        with_engine_mut([&](auto& e) { e.set_birth_position_cov(excess); });
    }
    const Eigen::Matrix2f& birth_position_excess() const { return with_engine([](const auto& e) -> decltype(auto) { return e.birth_position_excess(); }); }
    // The persistent path's per-channel information (public so the tests can drive it): camera = points + (2d) top + the
    // depth_unobs_precision shares (footprint w/h boost, the flush boost); LiDAR = the beam set f.beam (spec 7c.8; f.lidar is
    // the flag-off path's and is not read here); θ-only = (2b), (2c) and,
    // on an admissible camera update whose anchor is not yet in (per-anchor flags), wall flush/parallel (1) / the room axis. Each channel
    // is linearised at its own state: s_th = θ_k, s_cam = θ_k + B_chain ē + b̄_cam, s_L = θ_k + B_chain ē + b̄_L.
    void accumulate_channels(const CountableBeliefState& s_th, const CountableBeliefState& s_cam,
                             const CountableBeliefState& s_L, const CountableFrame& f,
                             Eigen::Matrix<float, 6, 6>& Id_cam, Eigen::Matrix<float, 6, 1>& bd_cam,
                             Eigen::Matrix<float, 6, 6>& Id_L,   Eigen::Matrix<float, 6, 1>& bd_L,
                             Eigen::Matrix<float, 6, 6>& Id_th,  Eigen::Matrix<float, 6, 1>& bd_th) const
    { with_engine([&](const auto& e) { e.accumulate_channels(model_of(e), s_th, s_cam, s_L, f, not f.points.empty(), lidar_measured(f), Id_cam, bd_cam, Id_L, bd_L, Id_th, bd_th); }); }
    // Test/diagnostic record of the last persistent update: θ after each GN iterate, the predicted θθ block, and the
    // channel information at the converged state.
    const std::vector<Eigen::Matrix<float, 6, 1>>& dbg_iterates() const { return with_engine([](const auto& e) -> decltype(auto) { return e.dbg_iterates(); }); }   // ACCEPTED θ iterates
    // J = persistent_data_energy + prior_energy at θ₀ and after every accepted step of the last persistent update (spec 7c.7 d).
    const std::vector<double>& dbg_gn_energy() const { return with_engine([](const auto& e) -> decltype(auto) { return e.dbg_gn_energy(); }); }
    // Model.GnTrace (diagnostic): one row per GN step of the last persistent update, and the finite-difference check.
    using GnTraceStep = rc::ai::engine::GnTraceStep<N>;
    using GnTraceFd   = rc::ai::engine::GnTraceFd;
    const std::vector<GnTraceStep>& dbg_gn_steps() const { return with_engine([](const auto& e) -> decltype(auto) { return e.dbg_gn_steps(); }); }
    const std::vector<GnTraceFd>&   dbg_gn_fd()    const { return with_engine([](const auto& e) -> decltype(auto) { return e.dbg_gn_fd(); }); }
    long gn_update_serial() const { return with_engine([](const auto& e) -> decltype(auto) { return e.gn_update_serial(); }); }   // bumped by every persistent update (trace de-duplication)
    // The last persistent update's damped GN (spec 7c.7 d): steps taken (accepted + rejected), whether it converged (
    // the undamped (λ = 0) proposal from the current θ is below the tolerance, Task 19 fix round 2), and the final λ.
    // Logged (gn_it, gn_conv, gn_lambda); no gate reads them.
    int    last_gn_iters()     const { return with_engine([](const auto& e) -> decltype(auto) { return e.last_gn_iters(); }); }
    bool   last_gn_converged() const { return with_engine([](const auto& e) -> decltype(auto) { return e.last_gn_converged(); }); }
    double last_lambda()       const { return with_engine([](const auto& e) -> decltype(auto) { return e.last_lambda(); }); }
    // δ_last, the last ACCEPTED θ step (zero if none; ruling T14-1, kept by T14-4): P_θθ carries δ_last δ_lastᵀ on top of the joint posterior.
    const Eigen::Matrix<double, 6, 1>& last_gn_step() const { return with_engine([](const auto& e) -> decltype(auto) { return e.last_gn_step(); }); }
    // The data energy whose gradients accumulate_channels builds: camera (the points' mixture −log likelihood, (2d) top, the
    // depth_unobs_precision flush / footprint boosts) at s_cam, LiDAR (the smooth beam model of f.beam, spec 7c.8: Σ E_i, κ ≡ 1 since rev 3.3.11,
    // whose gradient accumulate_channels uses exactly) at s_L, θ-only ((1) flush/parallel and the room axis when they enter, (2b),
    // (2c)) at s_th — each term's own energy at its own linearisation point, with the per-frame gates of accumulate_channels.
    // ∇E = −(bd_cam + bd_L + bd_th) up to the state dependence of the blocks' mixture weights (flush posterior, two-sidedness,
    // the rays' n̂·u), which the GN information also holds fixed. The ego-motion share is the latent u_c (rev 3.3.13), so it
    // enters J through the prior term, as the step's G does (the old "zero while parked, step only" was wrong: live13trace).
    double persistent_data_energy(const CountableBeliefState& s_cam, const CountableBeliefState& s_L,
                                  const CountableBeliefState& s_th, const CountableFrame& f) const
    { return with_engine([&](const auto& e) { return e.data_energy(model_of(e), s_cam, s_L, s_th, f, not f.points.empty(), lidar_measured(f)); }); }
    const Eigen::Matrix<float, 6, 6>& dbg_Pbar_tt() const { return with_engine([](const auto& e) -> decltype(auto) { return e.dbg_Pbar_tt(); }); }
    const Eigen::Matrix<float, 6, 6>& dbg_Id_th()   const { return with_engine([](const auto& e) -> decltype(auto) { return e.dbg_Id_th(); }); }
    const Eigen::Matrix<float, 6, 6>& dbg_Id_cam()  const { return with_engine([](const auto& e) -> decltype(auto) { return e.dbg_Id_cam(); }); }
    const Eigen::Matrix<float, 6, 6>& dbg_Id_L()    const { return with_engine([](const auto& e) -> decltype(auto) { return e.dbg_Id_L(); }); }
    // Test hooks.
    // (on the ACTIVE engine; a latent vector of the wrong size throws)
    Eigen::Ref<Eigen::MatrixXd> joint_covariance_for_test()
    { return with_engine_mut([&](auto& e) { e.ensure_joint(Sigma_); return Eigen::Ref<Eigen::MatrixXd>(e.P_for_test()); }); }
    void set_latent_mean_for_test(const Eigen::VectorXd& b)
    {
        with_engine_mut([&](auto& e)
        {
            using E = std::decay_t<decltype(e)>;
            if (b.size() != E::K) throw std::invalid_argument("set_latent_mean_for_test: the latent vector has the wrong size");
            e.ensure_joint(Sigma_);
            e.set_latent_mean(typename E::VecKd(b));
        });
    }
    // The latent predict of one persistent update alone (e with its lever, and the latents of the channels f measures).
    void    predict_latents_for_test(const CountableFrame& f)
    { with_engine_mut([&](auto& e) { e.ensure_joint(Sigma_); e.predict_latents(model_of(e), frame_latents(f)); }); }
    void    apply_mode_rotation_for_test(int k) { apply_mode_rotation(k); }
    // TEST HOOK only (spec 7c.8 Pass, "Reduction"; not in the params a fitter fills): the persistent LiDAR channel reads f.lidar
    // through the 7c.6 first-crossing factor with the 7c.7 (d) solver (b̄ restart, θ-only damping, ×4/÷4 λ) — "7c.7's update" —
    // instead of the beam set, so a test can compare the two on one frame.
    void    set_first_crossing_for_test(bool v) { first_crossing_for_test_ = v; }

    // ── Generative-model hooks (called by the engine; also used as the SDF API) ─
    // sdf_box: the whole fitted box (the envelope, every class); sdf_headboard: the panel; sdf_compound: the MODEL's solid
    // (parts_geometry.h: box ∪ panel, or the [[shape.part]] boxes ∪ panel for a solid = "parts" class).
    float sdf_box(const Eigen::Vector3f& p, const CountableBeliefState& s) const;
    float sdf_prim(const Eigen::Vector3f& p, const CountableBeliefState& s, int prim) const;   // prim == 0
    float sdf_headboard(const Eigen::Vector3f& p, const CountableBeliefState& s) const;   // +inf when no headboard
    float sdf_compound(const Eigen::Vector3f& p, const CountableBeliefState& s) const;
    Eigen::Matrix<float, 6, 1> sdf_jacobian(const Eigen::Vector3f& p, const CountableBeliefState& s, int prim) const;
    // Soft responsibilities: [box, clutter, wall] (sum = 1) at measurement variance R. The engine reads only
    // index 0 (n_prims()==1); the rest exist so the marginal likelihood is honest about what else explains a point.
    std::array<float, 3> responsibilities(const Eigen::Vector3f& p, const CountableBeliefState& s, float R) const;
    // Un-normalised mixture components [box, clutter, wall] + their sum (= marginal likelihood p(point|model)).
    float mixture_unnormalized(const Eigen::Vector3f& p, const CountableBeliefState& s, float R,
                               std::array<float, 3>& u) const;
    // Mean per-point NLL of the SAME mixture with the BOX COMPONENT REMOVED — i.e. how well the scene is
    // explained WITHOUT positing a bed here. mean_energy() minus this is the log Bayes factor for "there is
    // a bed here", the quantity birth/existence evidence should use: a mask lying on a known wall is already
    // fully explained, so the factor is ≈0 no matter how well a box happens to fit it.
    float mean_energy_no_box(const std::vector<Eigen::Vector3f>& pts, float R) const;
    // log p(points | wall+clutter+box) − log p(points | wall+clutter) ≥ 0-ish. Large ⇒ the bed hypothesis
    // genuinely adds explanatory power; ≈0 ⇒ something already in the scene model accounts for these points.
    float box_log_evidence_gain(const std::vector<Eigen::Vector3f>& pts, const CountableBeliefState& s,
                                float R) const
    { return mean_energy_no_box(pts, R) - mean_energy(pts, s, R); }
    // Mean per-point mixture NEGATIVE LOG-LIKELIHOOD (clutter INCLUDED) = the honest free energy (rises with
    // misfit, unlike the engine's surface-only return). Used for the published/logged FE + convergence.
    float mean_energy(const std::vector<Eigen::Vector3f>& pts, const CountableBeliefState& s, float R) const;
    void  apply_constraints(CountableBeliefState& s) const;
    void  canonicalize(CountableBeliefState&) const {}   // asymmetric box: no symmetry fold

    int   gn_iters() const { return params_.gn_iters; }
    int   n_prims()  const { return 1; }             // one box (clutter is the +1 mixture component)
    float sigma2()   const { return params_.sigma_base_m * params_.sigma_base_m + params_.model_sigma_m * params_.model_sigma_m; }
    Eigen::Matrix<float, 6, 6> transition() const { return Eigen::Matrix<float, 6, 6>::Identity(); }  // static
    Eigen::Matrix<float, 6, 1> process_noise_diag() const;        // the Q in force (the engine's on the persistent path)
    Eigen::Matrix<float, 6, 1> configured_process_noise() const;  // Q₀ from the params (the volatility's prior mean)
    Eigen::Matrix<float, 6, 1> prior_cov_diag() const;
    Eigen::Matrix<float, 6, 1> common_mode_inv_diag(const CountableFrame& frame) const;
    // Extra GN factor (engine calls it via C++23 requires): the YOLO-independent LiDAR first-hit range factor.
    void accumulate_extra(const CountableBeliefState& s, const CountableFrame& f,
                          Eigen::Matrix<float, 6, 6>& Id, Eigen::Matrix<float, 6, 1>& bd) const;

    // ── Reported uncertainty / orientation-mode shims ────────────────────────────
    // The per-point fit has NO discrete w↔h mode (the box is asymmetric), so resolve_orientation / flip_evidence /
    // mode_posterior stay trivial (kept so the shared fitter/scene-graph compile). The GENUINE remaining ambiguity
    // is which way the DOOR faces — a square-ish, wall-flush bed leaves the yaw discrete-ambiguous. That is
    // resolved by APPEARANCE (resolve_front, below), and its residual entropy is folded into the reported σ_yaw.
    float mode_posterior()    const { return 0.0f; }
    float flip_evidence()     const { return 0.0f; }
    bool  resolve_orientation(const std::vector<Eigen::Vector3f>&, float, float = 1.0f) { return false; }

    // ── FRONT (door) yaw resolver — sequential Bayes over the discrete door-facing modes ─────────────
    // Fold the appearance FrontCue (CountableProjection::detect_front) into a per-instance accumulator over the
    // discrete yaw candidates the footprint/wall leave (0/90/180/270). Each cycle, add w·(front_dir_k · cue_dir)
    // to each candidate's evidence — w = confidence·evidence_weight — clamped to ±kFrontClamp so a settled belief
    // can still RECANT if the bed is physically turned. Adopt the argmax mode (rotate yaw; swap w↔h + Σ rows/
    // cols on a 90°/270° adoption). A clearly RECTANGULAR footprint is only 180°-ambiguous — the 90°/270° modes
    // would swap the fitted w↔h and fight the geometric fit + wall-flush, so they are excluded (geometry-derived
    // candidate restriction, NOT a belief threshold). Mirrors ChairBelief::resolve_orientation. Returns true iff
    // it adopted a new orientation. Door face = local −Y.
    bool  resolve_front(const FrontCue& cue, float evidence_weight = 1.0f);
    // Adopt the best door mode from the accumulated appearance evidence PLUS the geometric clearance prior,
    // with NO new appearance cue. Call every cycle: it is the only path that can fix a bed whose door faces
    // its own wall, which detect_front can never report (it needs 2 visible faces and never scores the back).
    bool  resolve_front_geometric();
    // Per-mode log-prior from door CLEARANCE: a door cannot open into the wall the bed is attached to.
    // Recomputed from current geometry wherever the mode is scored — a PRIOR, never accumulated.
    std::array<float, 4> door_clearance_logprior() const;
    // Posterior over the 4 discrete front modes = softmax(front_acc_) with disallowed modes zeroed (see
    // allowed_modes). Front diagnostics: the winning mode's posterior mass (→1 once resolved) and its index.
    std::array<float, 4> front_posterior() const;
    float front_confidence() const;   // max posterior mass (peakedness of the door decision)
    int   front_mode()       const;   // argmax mode index (0 = current believed door direction)

    // REPORTED yaw variance: within-mode Σ(5,5) PLUS the variance of the discrete door-mode offset Δ_k = k·π/2
    // under the front posterior. Undecided (accumulator ~0) → a wide, honest σ_yaw; resolved → collapses to Σ(5,5).
    float yaw_marginal_var()  const;
    Eigen::Matrix<float, 6, 6> covariance_reported() const;

    // ── Diagnostic getters (no footprint-moment / coverage / vacate on this model → all zero) ────────────
    int   last_vacate_beams()  const { return 0; }
    int   last_coverage_pts()  const { return 0; }
    int   last_moment_pts()    const { return 0; }
    float dbg_yaw_after_points() const { return state_.yaw; }
    float dbg_yaw_after_moment() const { return state_.yaw; }
    float dbg_moment_aniso()     const { return 0.0f; }
    float dbg_moment_r_yaw()     const { return 0.0f; }
    float dbg_moment_dyaw()      const { return 0.0f; }
    float dbg_moment_ext_major() const { return 0.0f; }
    float dbg_moment_ext_minor() const { return 0.0f; }
    float dbg_moment_phi()       const { return 0.0f; }
    int   dbg_moment_pts()       const { return 0; }

    // 2D footprint second-moment of the points whose z falls in [z_lo, z_hi]. Geometry helper for the
    // fitter's optional birth seed (gated OFF by default); pure, no state.
    static FootprintMoment footprint_moment(const std::vector<Eigen::Vector3f>& pts, float z_lo, float z_hi);

    // ── "IS THIS REALLY A COUNTABLE?" plausibility (model-evidence mis-detection filter) ─────────────────
    // Continuous plausibility ∈ (0,1] that this fitted box is a bed, as a product of soft factors (aspect,
    // footprint size, height, data-fit) — NOT a hard threshold. ~1 for a proper 0.6×0.6×1.75 low-FE bed,
    // →0 for an elongated/short one or a poor fit. `fe` = the belief's clutter-inclusive mean_energy. The
    // static overload is pure (self_test / birth-candidate use); the member evaluates the current state.
    static float countable_plausibility(const CountableBeliefState& s, float fe, const CountableBeliefParams& p);
    float countable_plausibility(float fe) const { return countable_plausibility(state_, fe, params_); }
    // Birth-candidate plausibility from a raw mask cloud (BEFORE any fit): footprint aspect + size from the
    // 2-D principal extents, and a rough height from the cloud z-range. Scales the tracker's birth evidence so
    // an elongated/short candidate never accumulates enough to birth. fit term = 1 (no fit yet). Pure/static.
    static float candidate_plausibility(const std::vector<Eigen::Vector3f>& pts, const CountableBeliefParams& p);
    // ── Verification ──────────────────────────────────────────────────────────
    static bool self_test();

private:
    // Wall-flush + wall-parallel structural factor (ported from cabinet_concept). Inert if f.wall.ok==false. The flush
    // precision is wgt·(lam_base + depth_unobs_precision·depth_unobs): `base` keeps the anchor share, `boost` the
    // depth-unobservability share, `parallel` the wall-parallel term (all true = the flag-off factor, bit for bit).
    void accumulate_wall(const CountableBeliefState& s, const CountableFrame& f,
                         Eigen::Matrix<float, 6, 6>& Id, Eigen::Matrix<float, 6, 1>& bd,
                         bool base = true, bool boost = true, bool parallel = true, double* E = nullptr) const;
    // The blocks of accumulate_extra, each adding its term into (Id, bd); accumulate_extra calls them in its historic order
    // (bit-identical), the persistent path routes them to channels.
    // E (persistent path, spec 7c.7 d): when non-null each block ALSO adds its own energy ½λr² (λ, r as it builds them) — the
    // GN acceptance objective; nothing in (Id, bd) changes.
    void accumulate_room_axis(const CountableBeliefState& s, const CountableFrame& f,
                              Eigen::Matrix<float, 6, 6>& Id, Eigen::Matrix<float, 6, 1>& bd, double* E = nullptr) const;
    // (2) footprint: anchor = the class Gaussian (λ = 1/prior_footprint_std²); the depth_unobs_precision boost always.
    void accumulate_footprint(const CountableBeliefState& s, const CountableFrame& f,
                              Eigen::Matrix<float, 6, 6>& Id, Eigen::Matrix<float, 6, 1>& bd, bool anchor,
                              double* E = nullptr) const;
    void accumulate_height(const CountableBeliefState& s, Eigen::Matrix<float, 6, 6>& Id, Eigen::Matrix<float, 6, 1>& bd) const;
    void accumulate_top(const CountableBeliefState& s, const CountableFrame& f,
                        Eigen::Matrix<float, 6, 6>& Id, Eigen::Matrix<float, 6, 1>& bd, double* E = nullptr) const;
    // fixed_precision (the persistent θ-only channel, spec §7c.6): ½λr² on violation with a CONSTANT λ (exact Hessian);
    // false: the historic λ ∝ violation (the flag-off path, bit-identical).
    void accumulate_short_height(const CountableBeliefState& s, Eigen::Matrix<float, 6, 6>& Id, Eigen::Matrix<float, 6, 1>& bd,
                                 bool fixed_precision = false, double* E = nullptr) const;
    void accumulate_no_cross(const CountableBeliefState& s, const CountableFrame& f,
                             Eigen::Matrix<float, 6, 6>& Id, Eigen::Matrix<float, 6, 1>& bd, bool fixed_precision = false,
                             double* E = nullptr) const;
    // first_crossing (the 7c.6 persistent LiDAR channel; since 7c.8 only the set_first_crossing_for_test hook): every ray-set with
    // LidarRays::first_crossing = true and surface_sigma_m = model_sigma_m; false: the ray-sets as staged (the historic sphere-trace).
    void accumulate_rays(const CountableBeliefState& s, const CountableFrame& f,
                         Eigen::Matrix<float, 6, 6>& Id, Eigen::Matrix<float, 6, 1>& bd, bool first_crossing = false) const;

    // Persistent path (params.persistent): the update runs on engine_ (shared-engine Task 3); these are the model-side
    // pieces around it.
    friend struct CountableEngineModel;
    CountableEngineModel model() const { return CountableEngineModel{const_cast<CountableBelief*>(this)}; }
   // the engine's view; the const
                                                                                                   // engine paths take it const
    void  sanitize_joint();                               // sanitize_covariance under persistent
    void  predict_theta_joint(float q_scale);             // the engine's θ predict; prior_mean_ follows θ as on the flag-off path
    // This frame's latent inputs for the engine (the channels' "measured" flags as the frame stages them).
    rc::ai::engine::FrameLatents<N> frame_latents(const CountableFrame& f) const;
    std::vector<rc::ai::beam::BeamPart> beam_parts(const CountableBeliefState& s) const;
    float update_persistent_cycle(const CountableFrame& frame);   // update() under persistent (size summary, ω as today)
    // Posterior weight of the "flush against this wall" mixture component: exp(−(gap/reach)²). 0 ⇒ inert.
    float two_sided_along(const CountableBeliefState& s, const CountableFrame& f, bool along_x) const;
    // The same identifiability statistic with no feature switch: fraction in [0, 1] of the cloud that spans the extent
    // along one footprint axis (0 = one face only or no points). two_sided_along = this behind the fit's switches.
    float two_sided_fraction(const CountableBeliefState& s, const CountableFrame& f, bool along_x) const;
    float flush_posterior(const CountableBeliefState& s, const CountableFrame& f) const;
    float flush_weight(const CountableBeliefState& s, const CountableFrame& f) const;

    // Per-point WALL mixture component (explaining away): its available prior weight this frame, and its
    // un-normalised density at p. Both 0 when no room polygon is known ⇒ the mixture is box+clutter as before.
    float wall_component_frac() const;
    float wall_component(const Eigen::Vector3f& p, float R, float pw) const;

    // Which of the 4 discrete door modes {0,90,180,270} the current footprint leaves open. A rectangular
    // footprint (|w−h|/(w+h) > kRectAspect) allows only {0,180} (a 90° swap would fight the fitted w↔h);
    // a near-square one allows all four. Shared by resolve_front and yaw_marginal_var so both agree.
    std::array<bool, 4> allowed_modes() const;
    // Adopt door mode k: rotate yaw by k·π/2 and, for k∈{1,3}, swap w↔h in the state AND swap Σ rows/cols 3↔4
    // (and prior_mean_ 3↔4) so the covariance stays consistent with the relabelled axes.
    void apply_mode_rotation(int k);

    // Door-facing accumulator: per-candidate accumulated appearance evidence (higher = more door-evidence),
    // relative to the current mode (front_acc_[0] re-baselined to the max on each adoption). All-zero = undecided.
    static constexpr float kRectAspect  = 0.10f;   // |w−h|/(w+h) above which the footprint is treated rectangular
    std::array<float, 4>   front_acc_ = {0.0f, 0.0f, 0.0f, 0.0f};
    // Mode-evidence budget already spent per observer-bearing bin, exactly as ChairBelief does for its
    // 4-way yaw modes. Without it front_acc_ SUMS the same static appearance evidence every frame, so the
    // door-mode confidence grows with dwell time rather than with information — the defect this fleet keeps
    // rediscovering (see CONCEPT_AGENT_INVARIANTS.md, mistake I).
    static constexpr int kFrontViewBins = 24;          // 15° bins around the bed
    std::array<float, kFrontViewBins> front_view_spent_{};
    float front_view_budget_ = 3.0f;                   // same scale as ChairBeliefParams::view_budget

    CountableBeliefState    state_;
    CountableBeliefParams   params_;
    Eigen::Matrix<float, 6, 6> Sigma_ = Eigen::Matrix<float, 6, 6>::Identity();  // posterior covariance
    Eigen::Matrix<float, 6, 1> prior_mean_ = Eigen::Matrix<float, 6, 1>::Zero(); // transition prior mean

    Eigen::Vector2f room_interior_     = Eigen::Vector2f::Zero();   // polygon centroid: picks the BACK face
    bool            has_room_interior_ = false;
    // This frame's nearest wall, stashed by update() so the const mixture hook can use it as a competing
    // explanation. ok==false (no room polygon) ⇒ the wall component is inert and the mixture is box+clutter.
    WallRef         point_wall_;
    // Inferred per-DOF log-volatility and its hyperprior mean. omega_ starts AT omega0_, so with
    // volatility_infer off the model is bit-identical to the constant-Q original.
    Eigen::Matrix<float, 6, 1> omega_  = Eigen::Matrix<float, 6, 1>::Zero();
    Eigen::Matrix<float, 6, 1> omega0_ = Eigen::Matrix<float, 6, 1>::Zero();
    bool                       omega_init_ = false;
    // Conjugate sufficient statistics for the volatility estimate: discounted observation count and per-DOF
    // discounted sum of squared steps. Q = (n₀·Q₀ + S)/(n₀ + n) — a weighted average, so it cannot diverge.
    float                      vol_n_ = 0.0f;
    Eigen::Matrix<float, 6, 1> vol_s_ = Eigen::Matrix<float, 6, 1>::Zero();
    void update_volatility(const Eigen::Matrix<float, 6, 1>& dtheta);
public:
    // self_test only: drive one volatility step directly, so the test exercises the production update.
    void update_volatility_for_test(const Eigen::Matrix<float, 6, 1>& d) { update_volatility(d); }
private:
    // Likelihood-only (H, w, h) summary, updated by update(); see size_likelihood(). size_prior_off_ switches the class SIZE
    // prior blocks (2, 2a) of accumulate_extra off for the one data-only accumulation that feeds it; false everywhere else.
    void update_size_likelihood(const CountableFrame& frame);
    void predict_size_likelihood(float q_scale);
    rc::countable::SizeGaussian size_lik_;
    mutable bool                size_prior_off_ = false;
    // K2a-L: the proportions the model evaluates at (one per declared [[shape.proportion]]); the midpoints until the engine sets
    // them (CountableEngineModel::set_proportions). q_now() sizes it lazily, so params set after construction are honoured.
    mutable std::vector<double> q_;
    const std::vector<double>& q_now() const
    {
        if (q_.size() != params_.shape_proportions.size())
        {
            q_.clear();
            for (const auto& pr : params_.shape_proportions) q_.push_back(0.5 * (double(pr.lo) + double(pr.hi)));
        }
        return q_;
    }
    mutable float   dbg_wall_gap_    = 0.0f;
    mutable float   dbg_wall_lambda_ = 0.0f;
    mutable float   dbg_wall_resp_   = 0.0f;   // mean wall responsibility over the last scored cloud

    // Persistent joint state (params.persistent): the engine holds P over x = (θ, b), b, the latents' last σ, e's lever, the
    // class-prior and birth-spread flags and the GN diagnostics; wall_/axis_prior_added_ = the relational anchors have entered
    // (model-side: the θ-only channel reads them). Sigma_ mirrors P's θθ block in float for every reader of covariance().
    // Neither e nor ℓ is stored by any recall (a pose error is not an object property).
    std::variant<Engine, EngineQ> engine_;   // the active engine, chosen at construction (I2)
    bool   axis_prior_added_ = false;   // the room axis has entered (T6-1)
    bool   first_crossing_for_test_ = false;   // see set_first_crossing_for_test
    mutable bool   beam_unready_logged_ = false;        // "BeamConfig not built by make" printed once per belief
    mutable bool   beam_inconsistent_logged_ = false;   // "ray-set refused (per-ray vectors misaligned)" printed once (N3)
};

}  // namespace rc
