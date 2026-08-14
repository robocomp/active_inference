/*
 * hood_fitter.h — the active-inference fit core of hood_concept (mirrors bottle_concept/bottle_fitter.h).
 *
 * Owns the per-hood instance map and runs the AI2 full-covariance belief update for each "hood_*" node:
 * instance lifecycle (ensure_instance + the HoodModel factory), observation (split the selected mask's
 * support points into on-surface vs off-surface sets), inference (voxel-bank ingest + one recursive HoodBelief
 * update with the mask-motion channel as the observation precision R / bias gate, written back into inst.model),
 * and the hood-owned voxel memory (ownership gate + FNV voxel keys). Collaborates with MaskIngestor (masks),
 * HoodSceneGraph (robot covariance), HoodProjection (camera projection), and HoodLidarRangeChannel;
 * SpecificWorker keeps the orchestration (process_hood_node), the DSR write-back, and the post-fit
 * epistemic / affordance / Qt-diagnostics steps. Plain class (no Q_OBJECT).
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/core.hpp>                    // ZED RGB frame for appearance-based FRONT (door) detection
#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>
#include <dsr/api/dsr_inner_gaussian_api.h>   // Part B: chain covariance propagation
#include <dsr/api/dsr_camera_api.h>

#include "hood_config.h"        // rc::HoodConfig
#include "hood_instance.h"      // rc::HoodInstance, HoodState
#include "hood_model.h"         // HoodModel / HoodModelParams
#include "hood_projection.h"    // rc::HoodProjection, rc::SilhouetteExistence
#include "hood_lidar_range_channel.h"   // rc::HoodLidarRangeChannel
#include "../../common/mask_ingestor/mask_ingestor.h"
#include "hood_scene_graph.h"

namespace rc {

class HoodFitter
{
public:
    struct HoodObservation
    {
        bool has_fresh_data = false;
        std::vector<Eigen::Vector3f> candidate_pts;
        std::vector<Eigen::Vector3f> residual_pts;
        // Ricoh slices are bearing-only and never fitted (see observe_slice / process_ricoh_bearings): a
        // peripheral 360 detection has a reliable DIRECTION but a biased centroid/extent, so it only drives
        // the attention path and never contributes candidate/residual points to a belief update.
    };

    HoodFitter(std::shared_ptr<DSR::DSRGraph> graph,
                DSR::InnerEigenAPI* inner_eigen,
                HoodConfig& cfg,
                MaskIngestor* mask_ingestor,
                HoodSceneGraph* scene_graph);

    // Create the instance for a "hood_*" node if absent (from prior/RT). Returns true the
    // first time it is created (so the worker can register Qt series / canvas pos). Latches room_id.
    bool ensure_instance(const DSR::Node& node, std::uint64_t room_id);

    // Build an observation from ONE assigned mask slice (index into the current packet): candidate/residual
    // split against the current SDF + this slice's R inputs (motion_var/depth_var/range) latched on the
    // instance. Called once per slice in the same cycle for multi-sensor fusion (ZED + ricoh). Sets
    // frames_since_detection=0 (a detection is alive). Does NOT gate on frame_id — the caller owns freshness.
    HoodObservation observe_slice(HoodInstance& inst, int slice_index);
    // Fallback path when the tracker assigned NO mask slice this cycle: candidate/residual point attributes
    // written directly on the node (legacy), else a stale observation (has_fresh_data=false → age the belief).
    HoodObservation observe(HoodInstance& inst, const DSR::Node& node);
    // One recursive full-covariance belief update (HoodBelief) on this frame's mask points, with the
    // mask-motion channel as the observation precision R / bias gate. Writes the result into inst.model
    // so all downstream publish/viewer code is unchanged. Returns the update free energy.
    float run_inference(HoodInstance& inst, const HoodObservation& observation);

    // Robot/camera ego-motion (room frame), from the transform chain — producer-INDEPENDENT. Call once per compute
    // cycle (before the instance loop); run_inference then gates the geometry update to STILLNESS ("be-still-to-
    // update"). Ported 1:1 from chair_concept. Uses the SAME room_T_zed extrinsic (HoodProjection) the fit uses.
    void  update_ego_motion();
    float ego_lin_mps() const   { return ego_lin_mps_; }
    float ego_ang_radps() const { return ego_ang_radps_; }
    // Robust combined ego-motion speed (m/s): max(|motion_dotd|, ego_lin + ai2_ang_lever_m·ego_ang). Works even if
    // the producer never populated motion_dotd. Feeds the continuous common-mode AND the discrete confirm_only gate.
    float motion_magnitude(const HoodInstance& inst) const;
    // Off-axis penalty ∈ [0,1]: 0 on the optical axis (a centred mask has no peripheral smear), → 1 at ai2_periph_ref.
    float periphery_penalty(const HoodInstance& inst) const;
    // CONFIRMATION-ONLY (no geometry-mean change) when the robot is MOVING (ego lin/ang OR motion_dotd above the
    // still-level) AND the mask is off-centre; a well-centred mask updates even while moving (the exception).
    // Gated behind cfg_.ai2_motion_confirm_only. True ⇒ run_inference takes the predict-only branch (like the
    // truncation gate). Ported 1:1 from chair_concept.
    bool  confirm_only(const HoodInstance& inst) const;
    // The SAME admissibility, evaluated on a raw mask slice that has no instance yet: may this frame move
    // geometry? The tracker gates BIRTH on it, so a frame the fit would refuse can never create an object.
    bool  frame_admissible(const rc::MaskIngestor::MaskSlice& sl) const;

    // One-shot unit self-check (motion_magnitude robustness, the confirm_only gate + its predict-only mean-hold).
    // Pure — builds a fitter with a local config and null graph (no DSR needed). Returns true on all-PASS.
    static bool self_test();

    std::unordered_map<std::uint64_t, HoodInstance>& instances() { return instances_; }
    void forget_node(std::uint64_t id) { instances_.erase(id); }

    // Room geometry for the wall-flush factor (ported from cabinet_concept, simplified — a fridge is a SINGLE
    // box, so no run / wall-id / segment machinery). Pushed each cycle by the worker from the room polygon +
    // its interior centroid. nearest_wall(p) returns the nearest polygon edge as a WallRef {foot, inward normal}.
    // How much of the room's INTERIOR is available to hold an object centred at q, ∈ [0,1]. 1 well inside;
    // decays over one fridge half-depth as q approaches a wall; 0 at or beyond it. This is the room model
    // stating where an object CAN be, so a detection outside the layout accrues no birth evidence — the
    // continuous counterpart of residual_concept's point_in_polygon reject, and the birth-side twin of the
    // belief's one-sided wall mixture component. 1.0 when no room polygon is known (nothing to say).
    float interior_support(const Eigen::Vector2f& q) const;

    void set_room_geometry(const Eigen::Vector2f& interior, std::vector<Eigen::Vector2f> polygon)
    { room_interior_ = interior; room_polygon_ = std::move(polygon); have_room_geometry_ = not room_polygon_.empty();
      // The projection unit needs the same walls: a silhouette sample behind one is NOT "predicted visible".
      if (projection_) projection_->set_room_polygon(room_polygon_); }
    // The REACHABLE region, for the NBV. A hood stands against a wall, so two of its four faces have
    // their viewpoints outside the room; without this the planner cannot tell, because the raw information
    // term is direction-blind and `is_reachable` imposes no constraint on an empty polygon. See the call in
    // step_epistemic() and rc::nbv::is_reachable.
    bool has_room_polygon() const { return room_polygon_.size() >= 3; }
    const std::vector<Eigen::Vector2f>& room_polygon() const { return room_polygon_; }
    WallRef nearest_wall(const Eigen::Vector2f& q) const;
    // Part B (chain covariance): enable adding the localization/chain term J·Σ_chain·Jᵀ (measurement
    // frame → room, capture-stamp pinned) to each instance, read by the scene-graph's RT-cov write.
    void set_chain_cov_source(DSR::InnerGaussianAPI* gaussian, std::string source_frame);
    // Object-anchor observation: publish this frame's ROBOT-frame fit (z_o) so the room localizer can
    // use the hood as an SE(2) pose landmark. robot_frame = the localizer's base node ("body"). OFF unless enabled.
    void set_object_observation(bool enabled, std::string robot_frame);
    // Room-frame XY a NEWLY born instance's model should start at (from the tracker's detection). The
    // room→hood RT edge written at birth is NOT reliably queryable in the same cycle, so ensure_instance
    // would read the 0,0 default and the model would freeze there. Consumed once by ensure_instance.
    //
    // XY only, on purpose. Cold-start GEOMETRY belongs to run_inference's lazy snap (centre + 95th-percentile
    // height + footprint moment, depth left at the prior until the cloud spans it, yaw committed only on a
    // clearly anisotropic footprint), which overwrites whatever is set here on the first frame with points.
    // A candidate's probation burst was tried as a richer seed and measurably hurt — see the ★ note in
    // run_inference. The burst is still collected, but it is used only to REFUSE implausible births
    // (specificworker_lifecycle.cpp), never to seed geometry.
    void note_birth(std::uint64_t id, const Eigen::Vector2f& xy) { birth_seeds_[id] = xy; }
    bool should_log(const HoodInstance& inst) const;

    // Emit a CSV row for an OUT-OF-VIEW instance whose EXISTENCE evidence integrated this cycle (LiDAR/silhouette
    // carve) but which the fitter did NOT fit (no fresh mask) — so the existence log-odds trajectory that drives a
    // removal is captured even across the silent out-of-FoV stretch where no fit row is written. Fit fields are the
    // last converged values (held); the existence columns are FRESH (integrated just before this call). Called from
    // HoodExistence::update_and_remove for non-observed instances only, so it never double-logs a fit row.
    void log_existence_cycle(const HoodInstance& inst)
    { log_ai2_csv(inst, 0, inst.dbg_R, /*gated=*/true, inst.dbg_energy); }

    // YOLO-independent LiDAR range channel: stage this cycle's sweep (room frame) + sensor origin for the
    // range factor. clear_lidar_sweep() each cycle first so a stale sweep never leaks into a frame with no
    // fresh LiDAR. Set/cleared from the compute() main thread by SpecificWorker (fed by ConceptLidarIngestor).
    void set_lidar_sweep(const std::vector<Eigen::Vector3f>& sweep_room, const Eigen::Vector3f& origin_room)
    { lidar_channel_.set_sweep(sweep_room, origin_room); }
    void set_lidar_sweep_bpearl(const std::vector<Eigen::Vector3f>& sweep_room, const Eigen::Vector3f& origin_room)
    { lidar_channel_.set_sweep_bpearl(sweep_room, origin_room); }
    void clear_lidar_sweep() { lidar_channel_.clear(); }

    // ZED RGB frame for appearance-based FRONT (door) detection. The worker pushes the newest decoded BGR frame
    // (HoodRgbIngestor) each cycle a NEW one arrived; run_inference then projects the fitted box into it
    // (HoodProjection::detect_front) and folds the door-ness cue into the belief (resolve_front). Cleared
    // at cycle start so a stalled stream never re-integrates a stale appearance. Deep-copied (thread-boundary-safe).
    void set_rgb_frame(const cv::Mat& bgr, std::uint64_t stamp_ms)
    { rgb_frame_ = bgr.clone(); rgb_stamp_ms_ = stamp_ms; rgb_fresh_ = not rgb_frame_.empty(); }
    void clear_rgb_frame() { rgb_fresh_ = false; }

    // PIXEL-LEVEL silhouette existence evidence (EXISTENCE_BELIEF_PLAN.md, mask channel). Delegates to
    // HoodProjection; see rc::SilhouetteExistence in hood_projection.h. Called from update_existence.
    SilhouetteExistence compute_silhouette_existence(const HoodInstance& inst)
    { return projection_->compute_silhouette_existence(inst); }

private:
    // Compute the localization/chain covariance term (J·Σ_chain·Jᵀ) at the hood centre by transforming
    // it from the measurement frame back to room with ZERO input cov; stored on the instance for the
    // RT-cov write. No-op unless set_chain_cov_source enabled it.
    void compute_chain_cov(HoodInstance& inst);
    // Periodic round-vs-square shape model-selection on the accumulated voxel bank → inst.subtype
    // (bounded log-Bayes-factor, no threshold). Runs every cfg_.shape_eval_period cycles.
    void evaluate_shape(HoodInstance& inst);

    HoodModelParams  make_model_params() const;

    // Append one AI2 belief row (state + Σ diag std + mask R/bias/trunc + gate flag) to cfg_.ai2_csv_path.
    void log_ai2_csv(const HoodInstance& inst, int point_count, float R, bool gated, float energy);

    std::shared_ptr<DSR::DSRGraph> G_;
    DSR::InnerEigenAPI*            inner_eigen_ = nullptr;
    DSR::InnerGaussianAPI*         gaussian_    = nullptr;   // Part B: chain covariance (set_chain_cov_source)
    std::string                    chain_src_frame_;          // measurement frame the chain cov is computed from
    bool                           chain_cov_enabled_ = false;
    bool                           obs_enabled_ = false;       // publish z_o (object-anchor observation)?
    std::string                    obs_robot_frame_ = "body";  // localizer base frame z_o is expressed in
    std::string                    obs_cam_frame_ = "zed";     // camera frame the raw mask cloud lives in
    void compute_object_observation(HoodInstance& inst);      // fill inst.obs_robot (gated)
    HoodConfig&                   cfg_;
    MaskIngestor*                  mask_ingestor_ = nullptr;
    HoodSceneGraph*               scene_graph_   = nullptr;
    std::unique_ptr<HoodProjection> projection_;   // camera-projection unit (owns the ZED CameraAPI)

    Eigen::Vector2f                room_interior_ = Eigen::Vector2f::Zero();   // polygon centroid: picks the BACK face
    std::vector<Eigen::Vector2f>   room_polygon_;                              // room walls: wall-flush factor
    bool                           have_room_geometry_ = false;

    // Ego-motion state (camera pose deltas → robot speed), updated once per cycle in update_ego_motion().
    // (Ported 1:1 from chair_concept — the "be-still-to-update" signal source.)
    float           ego_lin_mps_   = 0.0f;
    float           ego_ang_radps_ = 0.0f;
    Eigen::Vector3f prev_cam_pos_  = Eigen::Vector3f::Zero();
    Eigen::Vector3f prev_cam_fwd_  = Eigen::Vector3f::UnitY();
    bool            have_prev_cam_ = false;
    std::chrono::steady_clock::time_point prev_cam_tp_{};

    std::unordered_map<std::uint64_t, HoodInstance> instances_;
    std::unordered_map<std::uint64_t, Eigen::Vector2f> birth_seeds_;   // tracker-provided birth XY (see note_birth)
    std::uint64_t                  room_node_id_ = 0;   // latched per ensure_instance call
    std::ofstream                  ai2_csv_;            // per-cycle AI2 belief log (optional)

    // YOLO-independent LiDAR range channel: owns the staged per-cycle sweep + return selection (feed).
    HoodLidarRangeChannel         lidar_channel_{cfg_};

    // Newest ZED RGB frame for appearance-based FRONT (door) detection (see set_rgb_frame). Deep-copied BGR.
    cv::Mat        rgb_frame_;
    std::uint64_t  rgb_stamp_ms_ = 0;
    bool           rgb_fresh_    = false;   // a new frame arrived this cycle (cleared at cycle start)

    // Appearance FRONT (door) resolver: project the fitted box into rgb_frame_ and fold the door-ness cue into
    // the belief. No-op unless cfg_.front_detect_enabled and a fresh RGB frame is staged. Called from run_inference
    // after the accepted belief update. mode_evidence_weight down-weights the cue while the robot moves (a smeared
    // door face must barely vote on the discrete orientation), mirroring the w↔h mode weight.
    void resolve_front_from_rgb(HoodInstance& inst, float mode_evidence_weight);
};

}  // namespace rc
