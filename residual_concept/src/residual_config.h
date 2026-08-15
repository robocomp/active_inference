/*
 * residual_config.h
 *
 * Plain-data configuration for the residual_concept agent + a loader that fills it from a RoboComp
 * ConfigLoader. Much leaner than BottleConfig: no camera/mask/silhouette/support/epistemic/eval keys.
 * It carries the belief params, the clusterer (perception front-end) params, the tracker params, and a
 * handful of agent-loop knobs.
 */

#pragma once

#include <string>

#include "residual_belief.h"      // rc::ResidualBeliefParams
#include "residual_clusterer.h"   // rc::ResidualClusterParams
#include "residual_zed_boost.h"   // rc::ZedBoostParams
#include "residual_semantic.h"    // rc::SemanticFloorParams
#include "residual_floor_plane.h" // rc::FloorPlaneParams

class ConfigLoader;   // RoboComp config façade (defined in genericworker.h)

namespace rc
{

struct ResidualConfig
{
    // ── agent loop ──
    float fe_eps                = 5e-3f;   // |ΔFE| convergence band
    int   K_stable              = 20;      // consecutive converged frames before "settled"
    int   diverged_retire_frames = 0;      // legacy no-data-fit counter; 0 = OFF (removal is now evidence-based)
    int   log_period_frames     = 30;
    float write_threshold_m     = 0.02f;   // geometry dead-band: republish only past this centre/size change

    // ── belief (the AI2 box-footprint fit) ──
    ResidualBeliefParams belief;
    std::string ai2_csv_path = "";         // non-empty → per-cycle belief CSV (state + Σ diag)

    // Per-cluster diagnostic CSV (default ON): sweep/specialists/room_poly counts + each cluster's
    // geometry — so wall/subtraction problems are diagnosable by reading a file, not pasting the terminal.
    std::string diag_csv_path = "etc/residual_diag.csv";

    // ── clusterer (LiDAR residual filter + 3D DBSCAN — the perception front-end) ──
    ResidualClusterParams cluster;

    // ── ZED dense-depth boost: when a residual box is in the ZED FoV, backproject the depth in its image
    //    region and add those dense points to the belief likelihood (a much denser signal than LiDAR). ──
    bool          zed_boost_enabled = true;
    float         zed_sigma_m       = 0.02f;   // per-ZED-point obs noise std (m) → R (ZED close depth is crisp)
    ZedBoostParams zed_boost;                  // subsample / margins / depth range / cap
    // Feed the dense ZED FoV cloud into DETECTION (clustering). Safe once the dense floor/wall subtraction is
    // range-noise robust (zed_infra below) — otherwise dense depth bridges distinct objects into a mega-blob.
    bool          zed_detection_enabled = false;
    // Feed the dense ZED FoV cloud into the OCCUPANCY GRID (as a second sensor after LiDAR). LiDAR rings only
    // graze horizontal surfaces (tabletops) so they never fill; the ZED depth camera covers them densely, and
    // the extra evidence per cell is what makes the costmap costmap-stable. Own ray origin (camera) → carve OK.
    bool          grid_zed_enabled = true;
    // Temporal low-pass on the PUBLISHED belief field only (P/Var): P_pub = α·P_new + (1−α)·P_prev, per cell.
    // Damps frame-to-frame heatmap shimmer at obstacle edges WITHOUT touching the occupancy latch (instant
    // completeness preserved — the safety occupied set is unaffected). ASYMMETRIC: risk RISES at full α (fast
    // detection, safe) and only FALLS at α_down (slow, stable) so a flickering cell holds its risk. 1 = off.
    float grid_field_ema_up   = 1.0f;    // blend toward a HIGHER risk (instant — never lag a new obstacle)
    float grid_field_ema_down = 0.30f;   // blend toward a LOWER risk (slow — stabilises flicker; ↓ = stickier)
    // EGO-MOTION precision: when the robot moves, pose jitter + motion blur make the whole sweep less trustworthy,
    // so grid evidence is scaled by 1/(1 + |v|/vel0 + |ω|/omega0) → the stable accumulated field dominates during
    // motion and sharpens when stopped. (Combined with the per-hit range weight — see OccGridParams.)
    float motion_vel0_mps    = 0.5f;     // linear speed (m/s) at which sweep trust halves
    float motion_omega0_rps  = 0.6f;     // yaw rate (rad/s) at which sweep trust halves
    // ── FORGETTING + SELF-BODY (see OccGridParams for the full rationale) ──
    // The grid used to accumulate monotonically: an unobserved cell was never updated (so anything latched behind
    // an occluder was immortal) and the z-band could only widen (so clearing got progressively harder — measured
    // 85%→96% of see-throughs discarded over one run). These three give evidence a finite lifetime.
    float grid_forget_half_life_s = 10.0f;  // half-life of evidence in an UNOBSERVED cell (0 ⇒ never forget)
    // Robot body envelope used by the SENSOR model to discount returns off our own body at integration time.
    // Distinct from Clusterer.RobotRadiusM, which is only a READ-OUT mask around the current pose and therefore
    // cannot stop self-returns being latched into the map in the first place. Default matches the lidar3d_dds
    // driver's own [Footprint] radius so the two agree on one body model. 0 ⇒ term off.
    // Per-device floor band for HELIOS, mirroring Clusterer.BpearlFloorZ0. ★THIS IS A THRESHOLD and CLAUDE.md
    // says to justify one rather than reach for it, so: the model-level routes were tried and do not work here.
    //   - A per-device floor SIGMA (measured from helios's own floor fit) was implemented and had no effect:
    //     helios's floor error is a systematic BIAS of +13-17 cm, not scatter, and widening a Gaussian centred
    //     on the TRUE floor cannot explain returns that sit consistently above it.
    //   - A per-device floor PLANE (use helios's own fitted floor as its reference) would cancel a CONSTANT
    //     bias, but helios's bias is GRAZING, i.e. it grows with range. A plane fits the average, so near
    //     returns would be over-corrected — and an over-corrected near return means a real 10-15 cm obstacle
    //     right next to the robot reads as below its own floor reference and is discarded. Trading a false
    //     positive at range for a false negative up close is the wrong trade.
    // What remains is the sensor's documented role: ROBOT_GEOMETRY.md assigns helios the WALLS ("use for walls,
    // not the floor datum") and bpearl the floor and low obstacles. This band encodes that division explicitly
    // instead of letting helios's unusable near-floor returns latch as obstacles. The number is measured, not
    // guessed: helios reads the floor at 130-170 mm (room_concept's startup FloorCheck, both headings).
    // COST, stated plainly: helios contributes no obstacle evidence below this band, so low obstacles at range
    // are bpearl's responsibility alone until the robot closes on them.
    float cluster_helios_floor_z0 = 0.20f;
    // C-space inflation of the PUBLISHED hulls. 0 because the controller collides its real footprint against
    // the grid, so any inflation here is added twice. See the note at gp.inflate_radius_m.
    float grid_inflate_radius_m = 0.0f;
    float grid_self_body_radius_m = 0.55f;
    float grid_self_body_sigma_m  = 0.08f;  // positional uncertainty of the body surface (probit width)
    // ── TIGHTENING THE OCCUPIED CONDITION (see OccGridParams for the derivations) ──
    // The nav band was a hard step and a single return one millimetre above it latched a cell for good, while the
    // floor's own MEASURED scatter (the plane fit's residual RMS, ≈7 cm here) was thrown away. These put the floor
    // back in the per-return mixture and give a floor return its free-space meaning. Both default ON; they exist as
    // flags only so the pair can be A/B'd against the old behaviour on a live safety layer.
    bool  grid_floor_responsibility = true;   // hit weight ×= P(obstacle | z) in the {floor, obstacle} mixture
    bool  grid_floor_return_clears  = true;   // a below-band return marks its OWN cell free (breaks the ratchet)
    float grid_floor_sigma_min_m    = 0.03f;  // irreducible sensor range noise (m) — σ floor of the floor model
    // Robust infrastructure subtraction for the dense ZED cloud: floor/ceiling/wall removed with a band that
    // grows as the ZED's own depth noise (σ0 + q·range²), so a calibration offset / far-range blur can't leak.
    ResidualClusterer::DepthInfraParams zed_infra;

    // RGB-SEMANTIC floor down-weighting: a second, uncorrelated cue against ZED floor phantoms. The dense YOLO-sem
    // label map (retina `semantic` node under `zed`) is sampled per ZED grid-point; a floor-class label at a
    // NEAR-floor return down-weights its occupancy HIT (height-gated + freshness-decayed, never discards). Flag OFF.
    SemanticFloorParams semantic_floor;

    // DATA-DRIVEN FLOOR PLANE: estimate the floor z=a·x+b·y+c from the LiDAR each cycle and reference the grid's
    // obstacle band to it (instead of a fixed z=0). Fixes systematic floor phantoms when a new scenario's floor is
    // offset/tilted. Estimator always runs+logs; the grid uses it only when enabled (flag OFF ⇒ fixed band).
    FloorPlaneParams floor_plane;

    // ── instance tracker (shared rc::InstanceTracker) ──
    float tracker_gate_mahalanobis  = 9.0f;
    float tracker_gate_fallback_m   = 0.35f;
    int   tracker_birth_frames      = 4;
    int   tracker_death_frames      = 30;
    float tracker_birth_min_sep_m   = 0.25f;
    float tracker_detection_noise_m = 0.08f;   // ≥ cluster-centroid-vs-fit-centre offset
    float tracker_merge_overlap     = 0.30f;   // box-IoU merge threshold; 0 disables
    bool  tracker_nll_cost          = false;
    bool  tracker_death_enabled     = false;   // OFF: removal is now the evidence-based occupancy carve, not a
                                               // frame counter (a miss = failed association ≠ evidence of absence)
    // Negative-information death gate: an instance only accrues a death "miss" when it SHOULD be visible —
    // its centre is within this LiDAR range of the robot. Out of range → the miss is HELD → the obstacle
    // PERSISTS in the map when the robot moves away (no die+rebirth churn on revisit). It is retired only
    // when in-range-and-absent (genuinely removed). 0 ⇒ always expected-visible (legacy, churny).
    float tracker_expected_visible_range_m = 6.0f;

    // ── per-point observation reliability (fill the belief's per-point R with covariates instead of a flat
    //    σ — the AI2 pattern, so far/rotating returns can't distort the fit). R_i = σ² + (k_range·range)² +
    //    (lag·rot_rate·range)². Far points → high R → honest WIDE Σ (the box stops chasing sparse noise);
    //    during rotation the tangential smear (∝ range) downweights the whole frame. range = horiz distance
    //    from the sensor origin. ──
    float range_noise_coeff = 0.01f;   // m of extra std per m of range (LiDAR range noise + deprojection)
    float ego_motion_lag_s  = 0.05f;   // effective registration lag (s): rotation smear = lag·rot_rate·range

    // ── occupancy-evidence removal (Bayesian ray-carving — replaces the death-frame counter) ──
    // Each cycle the worker carves the LiDAR sweep against every in-range instance box (ResidualClusterer::
    // carve_box) and integrates the per-cycle occupancy log-odds. An obstacle is removed only when the
    // accumulated evidence says the space is empty — i.e. beams have repeatedly PASSED THROUGH where it
    // should be. Occlusion / out-of-range cycles are HELD (no evidence). No frame counter, no distance gate.
    ResidualClusterer::CarveParams carve;   // sensor detection/clutter rates + range σ (physical, not gates)
    // Decision boundary: remove when P(occupied) < removal_prob (i.e. log-odds < log(p/(1−p))). This is a
    // RISK tolerance (the cost of a wrong removal), not a tuning threshold — smaller = more compelling
    // evidence required before deleting. Default ≈ log-odds −2 (fairly sure it's gone).
    float removal_prob          = 0.12f;
    float occupancy_logodds_max = 4.0f;     // clamp |log-odds| (±) so evidence stays finite AND recoverable

    // ── residual-specific ──
    // Dissolve-on-explained-interior (the fission policy): if a specialist explains ≥ this fraction of an
    // instance's cluster support, invalidate it and let the next cycle's residual re-cluster rebuild fresh.
    float dissolve_explained_frac = 0.60f;

    // ── LiDAR media plane ──
    std::string lidar_frame_node = "lidar3D";   // DSR node whose frame the raw sweep arrives in
    bool use_bpearl = true;   // fuse the low "bpearl" plane with "helios" (legs/low obstacles). Toggle off
                              // to A/B if bpearl injects floor/low-structure artefacts (its low grazing
                              // geometry differs from helios, so the range-floor band may not catch it).

    // ── conservative publish (safe-clearance footprint) ──
    // LiDAR sees only an obstacle's NEAR faces, so the fit systematically UNDER-sizes it (the far side is
    // occluded). For navigation the published footprint must err LARGER — the robot must clear the unobserved
    // volume. Grow the published w/d by k·σ of the extent (occlusion inflates that σ, so the box grows exactly
    // on the poorly-observed side — no arbitrary margin) plus a flat robot-clearance margin (a physical safety
    // distance, not a tuning gate). The internal fit / carve keep using the raw estimate; only the PUBLISHED
    // geometry is inflated.
    float publish_extent_sigma_k  = 2.0f;    // grow published w,d by k·√Σ_ww, k·√Σ_dd (occlusion-aware)
    float publish_safety_margin_m = 0.05f;   // + this clearance on EACH footprint side (robot clearance)

    // ── published RT covariance ──
    bool  rt_cov_add_chain     = true;    // add J·Σ_chain·Jᵀ localization cov to the published RT cov
    float unobservable_var     = 9.87f;   // ≈π² — roll/pitch (rx,ry) are unobservable for a footprint box
};

// Fill a ResidualConfig from a RoboComp ConfigLoader (all keys + defaults).
ResidualConfig load_residual_config(const ConfigLoader& cfg);

}  // namespace rc
