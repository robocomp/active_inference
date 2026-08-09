/*
 * cabinet_instance.h  —  per-cabinet runtime state owned by the fitter (mirrors bottle_concept/bottle_instance.h).
 *
 * Bundles the geometry/state container, the AI2 full-covariance belief, convergence bookkeeping, the
 * per-frame sensor/diagnostics snapshots, the cabinet-owned voxel memory bank, the existence log-odds, and
 * the epistemic affordance request — everything CabinetFitter carries per tracked cabinet.
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

#include <Eigen/Dense>

#include "cabinet_model.h"        // CabinetModel / CabinetState
#include "cabinet_belief.h"       // AI2 full-covariance belief (CABINET.md)
#include "cabinet_affordance.h"   // CabinetAffordance
#include "../../common/existence_belief/existence_belief.h"   // per-instance existence log-odds (removal)

namespace rc {

struct CabinetInstance
{
    // ── Identity ──────────────────────────────────────────────────────────────────────────────────
    uint64_t    node_id;
    std::string node_name;

    // ── Geometry / state container ────────────────────────────────────────────────────────────────
    // Holds the accepted pose+dims and the compound SDF used to split the mask's support points into
    // on-surface (candidate) vs off-surface (residual) sets.
    CabinetModel  model;

    // ── AI2 belief (CABINET.md) ─────────────────────────────────────────────────────────────
    // Full-covariance recursive filter over θ=[cx,cy,H,w,h,yaw]. Lazily initialised from the model state on
    // the first cycle; its result is written back into `model` so downstream publish/viewer code is unchanged.
    CabinetBelief ai2_belief;
    bool        ai2_initialized = false;

    // Born from a residual cluster (SpecificWorker::birth_from_residual), not a mask detection. Such a run
    // shares its mask slice with the parent it split off from, so the tracker's single-assignment cannot
    // feed it — run_instance_tracker gives it a supplementary footprint-claim on overlapping slices.
    bool        residual_born = false;

    // The room wall this run is built against, fixed ONCE at birth (nearest wall to the birth centre) and
    // never recomputed — a per-cycle recompute flips walls as a transiently-tilted box drifts. The wall-split
    // (observe_slice) measures each point's perpendicular distance from THIS wall's line; a point beyond a
    // depth is the neighbouring L-arm. −1 ⇒ island / no wall within reach ⇒ no split. West-arm run keys on
    // the west wall, back-arm run on the back wall, so each keeps its own leg of an L-mask.
    int         wall_seg_id = -1;

    // ── Per-frame mask corruption / range (from the selected mask slice) ──────────────────────────
    // motion_var inflates the observation precision R (downweight); trunc_frac gates the update (truncated →
    // biased mask); dotd is the motion-corruption speed, kept for diagnostics (see MASK_MOTION_CORRUPTION.md).
    float last_motion_var      = 0.0f;
    float last_motion_dotd     = 0.0f;
    // FRAME ego-motion (camera twist magnitudes, from the masks node's mask_cam_twist) + capture dt. Unlike
    // last_motion_var (per-mask → R), this is a SHARED frame error → the belief COMMON-MODE, so the whole
    // frame's authority to move the geometry MEAN vanishes as motion rises (Woodbury), while existence still
    // confirms. "Still is the spot to update" — a fixation/VOR precision, not a motion gate.
    float last_ego_v           = 0.0f;   // |linear twist|  (m/s)
    float last_ego_w           = 0.0f;   // |angular twist| (rad/s)
    float last_ego_dt          = 0.0f;   // capture dt (s) — motion×dt = the shared ego-displacement this frame
    float last_trunc_frac      = 0.0f;
    float last_centroid_radius = 0.0f;
    float last_range           = 0.0f;   // mean camera→mask depth Z (m), this frame — static range weighting
    // Depth-uncertainty of the mask (common/depth_projection): σ_range² (m²) added to R, sibling to
    // last_motion_var. 0 for dense-depth zed masks; the scored range variance for lidar-depth ricoh masks.
    float last_depth_var       = 0.0f;

    // ── LiDAR range-channel diagnostics (this frame) ──────────────────────────────────────────────
    // #returns fed to the factor, #returns in a generous box, and their mean |dist| to the current model
    // surface. Few rays / large resid ⇒ wrong lidar→room frame.
    int   dbg_lidar_rays     = 0;
    int   dbg_lidar_raw      = 0;
    float dbg_lidar_resid_m  = -1.0f;
    float dbg_lidar_meanz_m  = -1.0f;   // mean z of selected returns (room frame) — vs H detects a z-offset
    float dbg_lidar_topz_m   = -1.0f;   // mean z of the HIGHEST 20% of selected returns ≈ observed tabletop z
    float dbg_lidar_floorz_m = -1.0f;   // mean z of the LOWEST 20% ≈ floor; should read ~0 if z-calib OK
    float dbg_lidar_cov_ang  = -1.0f;   // angular-coverage weight (1−R)^p ∈[0,1]; low ⇒ one-sided sweep
    int   dbg_lidar_bpearl_rays = 0;    // low-bpearl returns staged into the extra ray-set this frame (0 = OFF/none)

    // ── Divergence safety net (mirrors bottle) ────────────────────────────────────────────────────
    // Consecutive frames whose centre GN step was rejected as an outlier (exceeded cfg.max_step_m). Non-zero
    // ⇒ the fit tried to run away and was held.
    int   frames_diverged = 0;

    // ── Convergence / association bookkeeping ─────────────────────────────────────────────────────
    int  matched_frames        = 0;   // frames with fresh sensing data
    int  frames_converged      = 0;   // consecutive frames with |Δstate| < state_eps
    // Persistent wall commitment: the room-wall this run is anchored to, chosen ONCE and reused every frame
    // (a run does not migrate between walls). Replaces the per-frame nearest-wall argmin whose choice
    // flip-flops as the fit drifts near a corner (positive feedback → oblique drift + impossible depth).
    // -1 = not yet committed (free-standing runs commit to their nearest wall; the flush weight stays ~0).
    int  committed_wall_seg_id = -1;
    int  last_masks_frame_seen = -1;  // last masks packet frame consumed
    std::uint64_t last_mask_timestamp_ms = 0;  // capture stamp of the last consumed mask (chain-cov pinning)
    // Wall-clock (agent's own steady clock) of the last belief touch — measured on EVERY inference cycle, not
    // just fresh ones — so a stale cycle can inflate Σ by the real elapsed time (measurement-age → covariance).
    std::chrono::steady_clock::time_point last_belief_touch{};
    float chain_cov_xx = 0.0f, chain_cov_yy = 0.0f;  // Part B localization/chain cov (m²), added to the RT cov
    // Per-frame ROBOT-frame observation z_o (x,y,yaw) for the room localizer's object-anchor factor.
    // Computed by the fitter (gated), published by the scene-graph writer. See common/object_anchor.
    Eigen::Vector3f obs_robot = Eigen::Vector3f::Zero();
    // Anisotropic 2×2 measurement covariance R_o (ROBOT/body frame) for this single-view centroid: tight
    // ⊥-ray, loose along-ray so the near-face/partial-view bias falls in R_o's null-space and can't drag
    // the robot pose. Built in compute_object_observation via ray_anisotropic_cov; published as
    // obj_obs_robot_cov=[Rxx,Ryy,Rxy]. See CABINET_TRIANGULATION.md.
    Eigen::Matrix2f obs_robot_cov = Eigen::Matrix2f::Identity();
    bool obs_robot_valid = false;
    int  processed_cycles = 0;        // per-cabinet compute cycles for log throttling
    // Tracker's gated mask assignments for THIS frame (indices into the masks packet slices). With
    // multi-detection fusion a cabinet can be seen by BOTH the ZED and the ricoh-360 masks in one cycle, so
    // this is a LIST — process_cabinet_node runs one belief update per slice (sequential = joint likelihood,
    // each sensor keeping its own R + common-mode). Empty = no association this frame. Set by run_instance_tracker.
    std::vector<int>  assigned_mask_idxs;
    bool model_stable     = false;
    int  model_generation = 0;
    CabinetState prev_conv_state{};      // accepted state at the previous cycle (for state-delta convergence)
    bool       has_prev_conv_state = false;

    // ── Epistemic affordance state (Schmitt-trigger hysteresis, anti-oscillation) ─────────────────
    bool epistemic_pending   = false;
    int  epistemic_cooldown  = 0;   // cycles remaining before a satisfied cabinet may re-arm

    // ── write_rt_pose dead-band (suppress tiny pose oscillations) ─────────────────────────────────
    float last_written_cx = std::numeric_limits<float>::max();
    float last_written_cy = std::numeric_limits<float>::max();
    float last_written_z0 = std::numeric_limits<float>::max();   // tier switches move the run vertically
    // Last GEOMETRY published to the graph (dims + mesh). Gates the per-cycle mesh/dim rewrite so a settled
    // cabinet stops jittering the voxelizer mesh. Mirrors bottle_concept's last_pub_* publish gate.
    float last_pub_cx  = std::numeric_limits<float>::max();
    float last_pub_cy  = std::numeric_limits<float>::max();
    float last_pub_w   = std::numeric_limits<float>::max();
    float last_pub_h   = std::numeric_limits<float>::max();
    float last_pub_H   = std::numeric_limits<float>::max();
    float last_pub_yaw = std::numeric_limits<float>::max();
    // Trace of the last RT-edge covariance published, so a stationary-but-still-tightening cabinet refreshes its
    // edge covariance on a meaningful uncertainty change (not only on a pose move).
    float last_pub_cov_trace = std::numeric_limits<float>::quiet_NaN();

    // ── Cabinet-owned voxel memory bank (room frame), independent of per-frame uploads ──────────────
    std::vector<Eigen::Vector3f> voxel_bank_pts;
    std::unordered_set<std::uint64_t> voxel_bank_keys;
    // Most recent fresh-frame residual points (model-unexplained), held for the viewer.
    std::vector<Eigen::Vector3f> last_residual_pts;
    // Epistemic action request published to DSR (filled by the epistemic planner).
    CabinetAffordance affordance;

    // ── Active-perception aids for the controller's local lock-on search ──────────────────────────
    // Detection aliveness: how recently YOLO produced a "cabinet" mask for this instance, and the confidence
    // of the last one. The controller hill-climbs these during the micro-search.
    int   frames_since_detection = 100000;   // cycles since last fresh cabinet mask (0 = just detected)
    float last_mask_confidence   = 0.0f;     // YOLO confidence of the last cabinet detection
    bool  detection_alive        = false;    // frames_since_detection < threshold

    // ── Existence log-odds (removal) ──────────────────────────────────────────────────────────────
    // LiDAR free-space carve + mask-absence accrue negative evidence when the cabinet is demonstrably gone.
    // Removed only when the removal decision holds for existence_remove_frames consecutive cycles (debounce)
    // — deleting furniture warrants SUSTAINED evidence, not a transient hiccup.
    rc::exist::ExistenceBelief existence;
    int existence_remove_streak = 0;
    // Verification-gated removal (active-inference): a predicted-visible-but-absent observation from a view that
    // CANNOT resolve the cabinet (far / peripheral / edge-on) does NOT vote removal — it raises this decayed
    // go-VERIFY surprise. When it crosses existence_verify_surprise, wants_verification arms the epistemic planner
    // to drive the robot to a good ZED viewpoint; only a HIGH-P(detect) view can then confirm-or-remove.
    float verify_surprise    = 0.0f;
    bool  wants_verification = false;

    // ── EvidenceMonitor per-cycle snapshots (persisted copies of transient fit/existence values) ──
    int   dbg_n_zed_slices = 0;   // ZED mask slices fused this cycle (ricoh is bearing-only, never fused)
    int   dbg_cand_pts     = 0;   // this frame's on-surface (candidate) support point count
    int   dbg_resid_pts    = 0;   // this frame's off-surface (residual/unexplained) support point count
    float dbg_energy       = 0.0f;  // mean per-point data energy (NLL proxy) at the converged state
    float dbg_R            = 0.0f;  // per-point measurement variance used this frame (m²)
    // FE ATTENTION / SURPRISE (active-perception trigger). fe_baseline = the "well-fit" free-energy level this
    // cabinet settles to (slow EMA; tracks DOWN fast to consolidate a better fit, UP slow so a SUSTAINED rise — the
    // world changed, e.g. the cabinet was moved — shows as surprise before the baseline accepts it). fe_surprise =
    // the smoothed positive gap (FE − baseline): ~0 when the belief matches the data, spikes when the cabinet moves
    // and decays as the robot re-observes and the fit re-converges. This is the signal that will raise the cabinet's
    // epistemic priority in the controller's EFE (attend + dwell until FE settles). See active-perception thread.
    float fe_baseline = -1.0f;    // <0 = uninitialised
    float fe_surprise = 0.0f;
    // ★The PUBLISHED epistemic_gain — the number the controller's affordance selection ranks on
    // (efe_score = gain − lambda_cost*dist, lambda_cost = 0.2/m, switch_margin = 0.5). Logged because
    // cross-agent selection was UNAUDITABLE: door was the only agent recording its gain, so "why does the
    // robot never visit a door" could not be answered against what a table or a fridge actually offers.
    float dbg_nbv_gain = 0.0f;
    bool  dbg_gated   = false;    // truncation-gated (predict-only, no geometric update) this frame

    // Provenance of the mask packet this fit consumed (instrumentation; NO effect on the fit). The CSV showed
    // long runs of fits whose GEOMETRY was bit-identical (npts/range pinned) while the belief ratcheted — i.e.
    // the same capture re-integrated as independent evidence. These two columns say WHICH identity signal
    // moved: fid = mask_frame_id (a publish counter, advances on every republish), ts = mask_timestamp_ms
    // (the capture stamp; absent ⇒ 0 ⇒ the capture-stamp half of the freshness gate is disabled).
    int           dbg_pkt_frame_id = -1;
    std::uint64_t dbg_pkt_ts_ms    = 0;

    // ── Rogue-mask yaw diagnostics (instrumentation; NO effect on the fit) ─────────────────────────
    // Captured per fresh cycle to CATCH the abrupt-rotation frames (robot turning / entering-leaving FoV /
    // passing by, where an in-frame-but-partial mask snaps yaw). yaw at cycle entry + the per-channel yaw split,
    // plus the two view-geometry covariates the truncation gate does NOT measure: obliquity (how grazing the
    // tabletop view is → foreshortened footprint) and completeness (observed vs believed footprint area).
    float dbg_yaw_pre       = 0.0f;   // belief yaw at cycle start, before the update (rad)
    float dbg_obliquity_cos = 1.0f;   // |cos(incidence)| of camera→cabinet ray vs tabletop normal +z (1=top-down, →0 grazing)
    float dbg_completeness  = 1.0f;   // observed footprint area / believed w·h (<1 = partial/foreshortened mask)
    float dbg_dyaw_points   = 0.0f;   // this cycle's yaw move attributed to the per-point GN channel (rad)
    float dbg_dyaw_moment   = 0.0f;   // ...to the footprint-moment channel (rad)
    float dbg_dyaw_flip     = 0.0f;   // ...to the resolve_orientation 90° flip (rad)
    // Existence channel readouts from the last update_existence_and_remove (per-modality occ/free + counts).
    // *_free is the RAW sensor count; *_free_eff is what actually drove the log-odds after the range/occlusion
    // absence-confidence scaling AND the observed-guard (so raw→eff shows how much the absence was degraded).
    float dbg_ex_lidar_occ = 0.0f, dbg_ex_lidar_free = 0.0f, dbg_ex_lidar_free_eff = 0.0f;  int dbg_ex_lidar_n   = 0;
    float dbg_ex_sil_occ   = 0.0f, dbg_ex_sil_free   = 0.0f, dbg_ex_sil_free_eff   = 0.0f;  int dbg_ex_sil_ndet  = 0;
    // Verification-gated removal diagnostics: p_detect (how resolving this view is), central_frac, and the
    // decayed go-verify surprise. Show why an absence went to REMOVAL (high p_detect) vs VERIFICATION (low).
    float dbg_ex_pdetect = 1.0f, dbg_ex_central = 0.0f;

    // ── Predicted in-image cabinet ROI (current model projected through the camera extrinsic) ───────
    // Normalised so the controller is resolution-agnostic: drive offset→0 (centre the cabinet in the frame)
    // and fill→target (stand-off sweet spot) to maximise YOLO's firing probability.
    bool  roi_valid    = false;
    float roi_offset_x = 0.0f;   // [-1,1], 0 = horizontally centred in the image
    float roi_offset_y = 0.0f;   // [-1,1], 0 = vertically centred
    float roi_fill     = 0.0f;   // max(w/W, h/H): projected extent as a fraction of the image
};

}  // namespace rc
