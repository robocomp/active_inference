/*
 * bottle_fitter.h
 *
 * The active-inference core of bottle_concept. Owns the per-bottle instance map and
 * runs the free-energy fit for each "bottle_*" cylinder node every cycle:
 *   - instance lifecycle (ensure_instance),
 *   - observation: split the selected mask's support points into on-surface candidates vs
 *     residuals against the current SDF (observe),
 *   - inference: support-bank ingest + cold-start seed (with camera-ward de-projection)
 *     + recursive-Laplace belief update (common/ai_belief) + the occluding-contour silhouette
 *     factor (accumulate_extra) folded into the free-energy step,
 *   - the bottle-owned support-point memory (ownership gate + FNV cell keys).
 *
 * Pure belief engine (mirrors TableFitter): exposes ensure_instance → observe → run_inference and
 * does NOT write DSR. READS via MaskIngestor (masks) and BottleSceneGraph (table lookups / support
 * surface / robot covariance). The worker (SpecificWorker::process_bottle_node) owns the write-back
 * (scene_graph_->step_write_model) and the eval log. Plain class (no Q_OBJECT).
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>
#include <dsr/api/dsr_camera_api.h>

#include "bottle_config.h"
#include "bottle_instance.h"
#include "bottle_model.h"        // BottleModel / BottleState / BottleModelParams
#include "../../common/mask_ingestor/mask_ingestor.h"
#include "bottle_scene_graph.h"

namespace rc {

class BottleFitter
{
public:
    BottleFitter(std::shared_ptr<DSR::DSRGraph> graph,
                 DSR::InnerEigenAPI* inner_eigen,
                 BottleConfig& cfg,
                 MaskIngestor* perception,
                 BottleSceneGraph* scene_graph);

    // Per-cycle fresh observation: the selected mask's support split into on-surface candidates vs residuals.
    struct BottleObservation
    {
        bool has_fresh_data = false;
        float explanation_ratio = 1.0f;
        std::vector<Eigen::Vector3f> candidate_pts;
        std::vector<Eigen::Vector3f> residual_pts;
    };

    // Pure belief API (no DSR write-back: the worker orchestrates persist + eval). Mirrors TableFitter:
    //   ensure_instance → observe → run_inference; run_inference also does the support-surface decision
    //   and the table-top z anchor (object-specific belief steps), reading — never writing — via the
    //   scene_graph. Returns the free energy. The worker then calls scene_graph_->step_write_model and
    //   the evaluator.
    // Create the instance for a "bottle_*" node if absent. Returns true the first time it is created
    // (so the worker can do one-time setup); latches room_node_id every call.
    bool ensure_instance(const DSR::Node& node, std::uint64_t room_node_id);
    BottleObservation observe(BottleInstance& inst, const DSR::Node& node);

    // Stage the latest LiDAR sweep (ROOM frame) + the sensor origin (ROOM frame) for this cycle. Pumped once
    // per compute() by specificworker from the lidar3D media plane; feed_lidar() then selects per-instance.
    void set_lidar_sweep(std::vector<Eigen::Vector3f> pts_room, const Eigen::Vector3f& origin_room)
    { lidar_sweep_room_ = std::move(pts_room); lidar_origin_room_ = origin_room; lidar_have_sweep_ = true; }
    void clear_lidar_sweep() { lidar_sweep_room_.clear(); lidar_have_sweep_ = false; }
    // The fit: one recursive full-covariance AI2 belief update (bottle_belief.h) on this frame's mask
    // points + silhouette. Writes the result into inst.model so downstream publish/RT code is unchanged.
    float run_inference(BottleInstance& inst, const BottleObservation& observation);
    // P(something able to move this bottle is in contact with it) in [0,1] — the CAUSE that licenses the
    // position to be volatile. A bottle does not move by itself: absent a mover its centre is as static as
    // its size, and any apparent jump is a mis-association, not motion. Read from the `person` nodes
    // human_concept publishes; the robot's own gripper is the natural second source and ORs in here.
    //
    // Continuous by construction, with a PHYSICAL length scale (a human arm's reach) rather than a tuned
    // radius: p = exp(-0.5*(d/reach)^2), d = horizontal distance from the nearest mover. Nobody in the room
    // => 0 => Q_pos collapses to its static value and the association gate stays tight. Someone reaching for
    // it => ~1 => Q_pos opens and the gate widens with it, because the tracker gates on this same Sigma.
    float mover_belief(const BottleInstance& inst) const;

    bool should_log(const BottleInstance& inst) const;

    // Robot/camera ego-motion (room frame), from the transform chain — producer-INDEPENDENT. Call once per compute
    // cycle (before the instance loop); run_inference then gates the GEOMETRY update to STILLNESS ("be-still-to-
    // update"). Mirrors refrigerator/chair confirm_only. Uses the SAME room_T_zed extrinsic (room_T_zed_matrix) the
    // fit uses, pinned to the CURRENT pose (ts=0), diffed across cycles on the MAIN thread.
    void  update_ego_motion();
    float ego_lin_mps() const   { return ego_lin_mps_; }
    float ego_ang_radps() const { return ego_ang_radps_; }
    // Robust combined ego-motion speed (m/s): max(|motion_dotd|, ego_lin + ai2_ang_lever_m·ego_ang). Works even if
    // the producer never populated motion_dotd. The SAME motion measure the discrete confirm_only gate reads.
    float motion_magnitude(const BottleInstance& inst) const;
    // CONFIRMATION-ONLY (no geometry-mean change) when the robot is MOVING (ego lin/ang OR motion_dotd above the
    // still-level). Gated behind cfg_.ai2_motion_confirm_only. True ⇒ run_inference takes the predict-only branch.
    // A bottle is a yaw-symmetric CYLINDER, so this governs position + size (radius,height) only — there is no yaw.
    // The SAME admissibility, evaluated on a RAW mask slice that has no instance yet: may this frame move
    // geometry? Birth is gated on it, so a frame the fit would REFUSE can never create an object.
    // ★A frame that may not MOVE an existing belief must not CREATE one — see
    // common/instance_tracker/birth_evidence.h rule 2. Mirrors this agent's own `gated` computation; the
    // instance-only parts of that gate (anything read off a fitted projection) cannot apply pre-birth.
    bool  frame_admissible(const rc::MaskIngestor::MaskSlice& sl) const;
    bool  confirm_only(const BottleInstance& inst) const;

    // The live instance map (the validation sweep mutates it; del_node prunes it).
    std::unordered_map<std::uint64_t, BottleInstance>& instances() { return instances_; }
    void forget_node(std::uint64_t id) { instances_.erase(id); }
    // Room-frame XY a NEWLY born instance's model should start at (from the tracker's detection). The
    // room→bottle RT written at birth is not reliably composable by inner_eigen in the same cycle, so
    // the model could cold-start at 0,0; the fit normally drags it to the points, but seeding it removes
    // that dependency (and the matching tracker re-birth risk). Consumed once by ensure_instance.
    void note_birth(std::uint64_t id, const Eigen::Vector2f& xy) { birth_seeds_[id] = xy; }

private:
    // Object-specific belief pre-step: decide the resting surface (room vs a table) + set the table-top
    // z anchor (hysteretic re-parent). Reads via scene_graph_; called at the head of run_inference.
    void update_support_surface(BottleInstance& inst);
    // YOLO detection score → per-observation reliability weight w∈[0,1] (monotone map). Scales the
    // silhouette precision so a weak mask can't over-tighten radius. 1.0 when cfg_.mask_conf_weight is off.
    float mask_confidence_weight(float confidence) const;

    // Support bank (bottle-owned historical memory).
    void ingest_observation_support(BottleInstance& inst, const BottleObservation& observation);
    bool is_point_owned_by_bottle(const BottleInstance& inst, const Eigen::Vector3f& point) const;
    static std::uint64_t cell_key(const Eigen::Vector3f& point, float quantization_m);

    // Append one AI2 belief row (state + Σ diag std) to cfg_.ai2_csv_path. No-op if the path is empty.
    void log_ai2_csv(const BottleInstance& inst, int point_count, float R, float energy);

    // Feed the fitted model the RGB-mask edge rays as a silhouette likelihood.
    void feed_silhouette(BottleInstance& inst);
    // Select the LiDAR returns of the current sweep that fall near this instance and stage them onto the
    // frame's YOLO-independent range channel (no-op unless a sweep was set and cfg_.lidar_precision > 0).
    void feed_lidar(BottleInstance& inst, BottleFrame& frame, float range_scale) const;
    // Camera→object sensing distance (m) → precision-fade factor R_mult = max(1, range/near)^power (>=1). 1.0
    // within `near` (grasp range, full precision); grows beyond so far/receding objects are left ~untouched.
    float range_precision_scale(const BottleInstance& inst) const;
    // Set inst.expected_visible: true iff the bottle centre projects inside the camera frustum now. Drives
    // the tracker's negative-information DEATH gate (persist out-of-FoV; retire only if in-view & absent).
    void update_expected_visible(BottleInstance& inst);
    // room_T_zed (camera→room) as a plain 4×4, composed room→body→zed at ts=0 (alignment-safe).
    std::optional<Eigen::Matrix4d> room_T_zed_matrix(std::uint64_t timestamp_ms = 0) const;

    // Factory helper (config → model geometry/prior params; the model carries the SDF + state).
    BottleModelParams make_model_params() const;

    std::shared_ptr<DSR::DSRGraph>  G_;
    DSR::InnerEigenAPI*             inner_eigen_ = nullptr;
    BottleConfig&                    cfg_;
    MaskIngestor*               mask_ingestor_  = nullptr;
    BottleSceneGraph*               scene_graph_ = nullptr;   // READS only (support surface, table-top, robot cov)

    std::unique_ptr<DSR::CameraAPI> camera_api_;   // ZED intrinsics, lazily bound to the "zed" node
    std::unordered_map<std::uint64_t, BottleInstance> instances_;
    std::unordered_map<std::uint64_t, Eigen::Vector2f> birth_seeds_;   // tracker-provided birth XY (see note_birth)
    std::uint64_t                   room_node_id_ = 0;   // refreshed each process_bottle_node call
    std::ofstream                   ai2_csv_;            // per-cycle AI2 belief log (optional)

    // Latest LiDAR sweep for this cycle (ROOM frame) + sensor origin (ROOM frame); staged by set_lidar_sweep.
    std::vector<Eigen::Vector3f>    lidar_sweep_room_;
    Eigen::Vector3f                 lidar_origin_room_ = Eigen::Vector3f::Zero();
    bool                            lidar_have_sweep_  = false;

    // Ego-motion state (camera pose deltas → robot speed), updated once per cycle in update_ego_motion().
    // (Mirrors refrigerator/chair — the "be-still-to-update" signal source for the discrete confirm_only gate.)
    float           ego_lin_mps_   = 0.0f;
    float           ego_ang_radps_ = 0.0f;
    Eigen::Vector3f prev_cam_pos_  = Eigen::Vector3f::Zero();
    Eigen::Vector3f prev_cam_fwd_  = Eigen::Vector3f::UnitY();
    bool            have_prev_cam_ = false;
    std::chrono::steady_clock::time_point prev_cam_tp_{};
};

}  // namespace rc
