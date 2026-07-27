/*
 * door_fitter.h
 *
 * The active-inference core of door_concept (mirrors bottle_concept/bottle_fitter.h). Owns the
 * per-door instance map and runs the AI2 full-covariance belief update for each "door_*" node:
 *   - instance lifecycle (ensure_instance + the DoorModel factory),
 *   - observation: split the selected mask's support points into on-surface vs off-surface sets,
 *   - inference: voxel-bank ingest + one recursive belief update (DoorBelief) with the mask-motion
 *     channel as the observation precision R / bias gate, written back into inst.model,
 *   - the door-owned voxel memory (ownership gate + FNV voxel keys).
 *
 * Collaborates with MaskIngestor (masks) and DoorSceneGraph. SpecificWorker keeps the orchestration
 * (process_door_node), the DSR write-back call, and the post-fit epistemic / affordance / Qt steps.
 * Plain class (no Q_OBJECT).
 */

#pragma once

#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>
#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>
#include <dsr/api/dsr_inner_gaussian_api.h>   // Part B: chain covariance propagation
#include <dsr/api/dsr_camera_api.h>

#include "door_config.h"        // rc::DoorConfig
#include "door_instance.h"      // rc::DoorInstance, DoorState
#include "door_model.h"         // DoorModel / DoorModelParams
#include "../../common/mask_ingestor/mask_ingestor.h"
#include "door_scene_graph.h"

namespace rc {

class DoorFitter
{
public:
    struct DoorObservation
    {
        bool has_fresh_data = false;
        float explanation_ratio = 1.0f;
        std::vector<Eigen::Vector3f> candidate_pts;
        std::vector<Eigen::Vector3f> residual_pts;
    };

    DoorFitter(std::shared_ptr<DSR::DSRGraph> graph,
                DSR::InnerEigenAPI* inner_eigen,
                DoorConfig& cfg,
                MaskIngestor* mask_ingestor,
                DoorSceneGraph* scene_graph);

    // Create the instance for a "door_*" node if absent (from prior/RT). Returns true the
    // first time it is created (so the worker can register Qt series / canvas pos). Latches room_id.
    bool ensure_instance(const DSR::Node& node, std::uint64_t room_id);

    // Read the "masks" node → candidate/residual split against the current SDF.
    DoorObservation observe(DoorInstance& inst, const DSR::Node& node);
    // One recursive full-covariance belief update (DoorBelief) on this frame's mask points, with the
    // mask-motion channel as the observation precision R / bias gate. Writes the result into inst.model
    // so all downstream publish/viewer code is unchanged. Returns the update free energy.
    float run_inference(DoorInstance& inst, const DoorObservation& observation);

    std::unordered_map<std::uint64_t, DoorInstance>& instances() { return instances_; }
    void forget_node(std::uint64_t id) { instances_.erase(id); }
    // Line-of-sight occlusion test (room frame): is the camera→door ray blocked by a CLOSER object whose
    // bearing cone covers the door — another door instance, or any other detected mask this frame (table /
    // person / …)? Used to SUPPRESS the existence "vacate" negative: a door hidden behind something must not
    // be removed just because no mask reaches it. Conservative (over-suppress = keep) by design.
    bool los_occluded(const DoorInstance& inst) const;

    // ── Room-containment pose prior ───────────────────────────────────────────────────────────────
    // The room polygon (room frame) is a trusted NOMINAL model (authored from the SVG layout, never fitted),
    // so it is a legitimate strong prior: P(a door's centre outside the walls) ≈ 0. Loaded from the room
    // node's delimiting_polygon_{x,y}. Used to suppress out-of-room births AND remove an instance that a
    // localization glitch placed outside (it can't be reached by the sensor to vacate it — behind a wall).
    void set_room_geometry(const Eigen::Vector2f& interior, std::vector<Eigen::Vector2f> polygon)
    { room_interior_ = interior; room_polygon_ = std::move(polygon); }
    bool has_room_polygon() const { return room_polygon_.size() >= 3; }

    // Robot/camera ego-motion (room frame), from the transform chain — producer-independent. Call once per
    // compute cycle; run_inference then gates the pose/shape update to STILLNESS ("be-still-to-update").
    void  update_ego_motion();
    float ego_lin_mps() const   { return ego_lin_mps_; }
    float ego_ang_radps() const { return ego_ang_radps_; }
    // True when this frame must be CONFIRMATION-ONLY (no pose/shape change): robot linear/angular speed above
    // the still-level, OR the mask's own ego-motion corruption (motion_dotd) above its still-level.
    // (A/B FALLBACK path only — the hard gate; used when cfg.ai2_motion_confirm_only is true.)
    bool  confirm_only(const DoorInstance& inst) const;
    // Continuous frame reliability ∈ [0,1] (AIF): 1 = trustworthy (still OR centred), → 0 = moving AND peripheral.
    // Scales the existence NEGATIVE evidence so a smeared/off-axis frame can only CONFIRM, never argue a door away.
    float frame_reliability(const DoorInstance& inst) const;
    // ZED expected-detectability ∈ [0,1]: how reliably the ZED camera would detect a PRESENT door at this
    // instance's projected ROI — falls off toward the image edge (roi_offset) and with range. Used to weight
    // the existence VACATE so absence only removes to the degree ZED should have resolved it ("ZED removes").
    float zed_detectability(const DoorInstance& inst) const;
    float motion_magnitude(const DoorInstance& inst) const;   // combined ego-motion speed (m/s)
    float periphery_penalty(const DoorInstance& inst) const;  // off-axis penalty ∈ [0,1] (0 on-axis → 1 at periph_ref)
    // true if q is inside the polygon, or outside by no more than margin_m (tolerance for a wall-hugging door
    // whose centroid noise pokes through the wall). No polygon loaded ⇒ always true (unknown room → no prior).
    bool point_in_room(const Eigen::Vector2f& q, float margin_m = 0.0f) const;

    // Wall frame a door is anchored to: near corner O and along-wall unit u (room frame). A door lives IN a
    // wall, so its belief works in this frame (θ=[s,w,h]); the fitter associates the door to the nearest
    // room-polygon wall at birth and sets it on the belief. Mirrors cabinet_concept's build_wall_ref.
    struct WallFrame { bool ok = false; Eigen::Vector2f O{0.0f, 0.0f}, u{1.0f, 0.0f}; float len = 0.0f; };
    WallFrame nearest_wall(const Eigen::Vector2f& q) const;
    bool should_log(const DoorInstance& inst) const;
    // Part B (chain covariance): enable adding the localization/chain term J·Σ_chain·Jᵀ (measurement
    // frame → room, capture-stamp pinned) per instance, read by the scene-graph's RT-cov write.
    void set_chain_cov_source(DSR::InnerGaussianAPI* gaussian, std::string source_frame, bool enabled);
    // Room-frame XY a NEWLY born instance's model should cold-start at (from the tracker's detection).
    // The room→door RT written at birth is not reliably composable the same cycle, so without this the
    // model would start at 0,0; consumed once by ensure_instance.
    void note_birth(std::uint64_t id, const Eigen::Vector2f& xy) { birth_seeds_[id] = xy; }

    // Part C-birth: initialise `inst` as a bearing-only hypothesis — belief mean placed at `nominal_range`
    // along the ray from `robot_xy` at `azimuth`, with a broad along-ray / tight across-ray Σ (see
    // DoorBelief::seed_bearing). Sets ai2_initialized + is_bearing_hypothesis and writes the mean into the
    // model so the scene-graph/viewer show the hypothesis on the ray. No depth mask needed.
    void seed_bearing_hypothesis(DoorInstance& inst, const Eigen::Vector2f& robot_xy, float azimuth,
                                 float nominal_range, float along_std, float across_std, float yaw_std);

private:
    DoorBeliefParams make_belief_params() const;   // config → belief params (shared by init + hypothesis seed)
    // room_T_zed (camera→room). pose_ts_ms pins the room→body hop to the mask's capture time (Nearest RT
    // query); the rigid body→zed mount is always queried latest. 0 → current pose.
    std::optional<Eigen::Matrix4d> room_T_zed_matrix(std::uint64_t pose_ts_ms = 0) const;
    // Project the current model through the camera extrinsic → normalised in-image ROI (centre
    // offset + fill), stored on the instance for the controller's centring/dwell lock-on search.
    void compute_projected_roi(DoorInstance& inst);
    // Part B: localization/chain cov J·Σ_chain·Jᵀ at the door centre (measurement frame → room, zero
    // input cov), stored on the instance for the RT-cov write. No-op unless set_chain_cov_source enabled.
    void compute_chain_cov(DoorInstance& inst);

    void ingest_observation_voxels(DoorInstance& inst, const DoorObservation& observation);
    bool is_voxel_owned_by_door(const DoorInstance& inst, const Eigen::Vector3f& point) const;
    static std::uint64_t voxel_key(const Eigen::Vector3f& point, float quantization_m);

    DoorModelParams  make_model_params() const;

    // Append one AI2 belief row (state + Σ diag + range/motion) to cfg_.ai2_csv_path. No-op if empty.
    void log_ai2_csv(const DoorInstance& inst, int npts, float R, bool gated, float energy);

    std::shared_ptr<DSR::DSRGraph> G_;
    DSR::InnerEigenAPI*            inner_eigen_ = nullptr;
    DSR::InnerGaussianAPI*         gaussian_    = nullptr;   // Part B: chain covariance (set_chain_cov_source)
    std::string                    chain_src_frame_;
    bool                           chain_cov_enabled_ = false;
    DoorConfig&                   cfg_;
    MaskIngestor*                  mask_ingestor_ = nullptr;
    DoorSceneGraph*               scene_graph_   = nullptr;
    std::unique_ptr<DSR::CameraAPI> camera_api_;   // ZED intrinsics, lazily bound to the "zed" node

    std::unordered_map<std::uint64_t, DoorInstance> instances_;
    std::unordered_map<std::uint64_t, Eigen::Vector2f> birth_seeds_;   // tracker-provided birth XY (note_birth)
    std::uint64_t                  room_node_id_ = 0;   // latched per ensure_instance call
    std::vector<Eigen::Vector2f>   room_polygon_;       // room-frame delimiting polygon (containment prior)
    Eigen::Vector2f                room_interior_ = Eigen::Vector2f::Zero();   // a known-interior point (centroid)
    // Ego-motion state (camera pose deltas → robot speed), updated once per cycle in update_ego_motion().
    float           ego_lin_mps_   = 0.0f;
    float           ego_ang_radps_ = 0.0f;
    Eigen::Vector3f prev_cam_pos_  = Eigen::Vector3f::Zero();
    Eigen::Vector3f prev_cam_fwd_  = Eigen::Vector3f::UnitY();
    bool            have_prev_cam_ = false;
    std::chrono::steady_clock::time_point prev_cam_tp_{};
    std::ofstream                  ai2_csv_;            // per-cycle AI2 belief log (optional)
};

}  // namespace rc
