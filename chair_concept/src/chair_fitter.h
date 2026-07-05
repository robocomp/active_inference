/*
 * chair_fitter.h
 *
 * The active-inference core of chair_concept (mirrors bottle_concept/bottle_fitter.h). Owns the
 * per-chair instance map and runs the AI2 full-covariance belief update for each "chair_*" node:
 *   - instance lifecycle (ensure_instance + the ChairModel factory),
 *   - observation: split the selected mask's support points into on-surface vs off-surface sets,
 *   - inference: voxel-bank ingest + one recursive belief update (ChairBelief) with the mask-motion
 *     channel as the observation precision R / bias gate, written back into inst.model,
 *   - the chair-owned voxel memory (ownership gate + FNV voxel keys).
 *
 * Collaborates with MaskIngestor (masks) and ChairSceneGraph. SpecificWorker keeps the orchestration
 * (process_chair_node), the DSR write-back call, and the post-fit epistemic / affordance / Qt steps.
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

#include "chair_config.h"        // rc::ChairConfig
#include "chair_instance.h"      // rc::ChairInstance, ChairState
#include "chair_model.h"         // ChairModel / ChairModelParams
#include "../../common/mask_ingestor/mask_ingestor.h"
#include "chair_scene_graph.h"

namespace rc {

class ChairFitter
{
public:
    struct ChairObservation
    {
        bool has_fresh_data = false;
        float explanation_ratio = 1.0f;
        std::vector<Eigen::Vector3f> candidate_pts;
        std::vector<Eigen::Vector3f> residual_pts;
    };

    ChairFitter(std::shared_ptr<DSR::DSRGraph> graph,
                DSR::InnerEigenAPI* inner_eigen,
                ChairConfig& cfg,
                MaskIngestor* mask_ingestor,
                ChairSceneGraph* scene_graph);

    // Create the instance for a "chair_*" node if absent (from prior/RT). Returns true the
    // first time it is created (so the worker can register Qt series / canvas pos). Latches room_id.
    bool ensure_instance(const DSR::Node& node, std::uint64_t room_id);

    // Read the "masks" node → candidate/residual split against the current SDF.
    ChairObservation observe(ChairInstance& inst, const DSR::Node& node);
    // One recursive full-covariance belief update (ChairBelief) on this frame's mask points, with the
    // mask-motion channel as the observation precision R / bias gate. Writes the result into inst.model
    // so all downstream publish/viewer code is unchanged. Returns the update free energy.
    float run_inference(ChairInstance& inst, const ChairObservation& observation);

    std::unordered_map<std::uint64_t, ChairInstance>& instances() { return instances_; }
    void forget_node(std::uint64_t id) { instances_.erase(id); }
    bool should_log(const ChairInstance& inst) const;
    // Part B (chain covariance): enable adding the localization/chain term J·Σ_chain·Jᵀ (measurement
    // frame → room, capture-stamp pinned) per instance, read by the scene-graph's RT-cov write.
    void set_chain_cov_source(DSR::InnerGaussianAPI* gaussian, std::string source_frame, bool enabled);
    // Room-frame XY a NEWLY born instance's model should cold-start at (from the tracker's detection).
    // The room→chair RT written at birth is not reliably composable the same cycle, so without this the
    // model would start at 0,0; consumed once by ensure_instance.
    void note_birth(std::uint64_t id, const Eigen::Vector2f& xy) { birth_seeds_[id] = xy; }

    // Part C-birth: initialise `inst` as a bearing-only hypothesis — belief mean placed at `nominal_range`
    // along the ray from `robot_xy` at `azimuth`, with a broad along-ray / tight across-ray Σ (see
    // ChairBelief::seed_bearing). Sets ai2_initialized + is_bearing_hypothesis and writes the mean into the
    // model so the scene-graph/viewer show the hypothesis on the ray. No depth mask needed.
    void seed_bearing_hypothesis(ChairInstance& inst, const Eigen::Vector2f& robot_xy, float azimuth,
                                 float nominal_range, float along_std, float across_std, float yaw_std);

private:
    ChairBeliefParams make_belief_params() const;   // config → belief params (shared by init + hypothesis seed)
    // room_T_zed (camera→room). pose_ts_ms pins the room→body hop to the mask's capture time (Nearest RT
    // query); the rigid body→zed mount is always queried latest. 0 → current pose.
    std::optional<Eigen::Matrix4d> room_T_zed_matrix(std::uint64_t pose_ts_ms = 0) const;
    // Project the current model through the camera extrinsic → normalised in-image ROI (centre
    // offset + fill), stored on the instance for the controller's centring/dwell lock-on search.
    void compute_projected_roi(ChairInstance& inst);
    // Part B: localization/chain cov J·Σ_chain·Jᵀ at the chair centre (measurement frame → room, zero
    // input cov), stored on the instance for the RT-cov write. No-op unless set_chain_cov_source enabled.
    void compute_chain_cov(ChairInstance& inst);

    void ingest_observation_voxels(ChairInstance& inst, const ChairObservation& observation);
    bool is_voxel_owned_by_chair(const ChairInstance& inst, const Eigen::Vector3f& point) const;
    static std::uint64_t voxel_key(const Eigen::Vector3f& point, float quantization_m);

    ChairModelParams  make_model_params() const;

    // Append one AI2 belief row (state + Σ diag + range/motion) to cfg_.ai2_csv_path. No-op if empty.
    void log_ai2_csv(const ChairInstance& inst, int npts, float R, bool gated, float energy);

    std::shared_ptr<DSR::DSRGraph> G_;
    DSR::InnerEigenAPI*            inner_eigen_ = nullptr;
    DSR::InnerGaussianAPI*         gaussian_    = nullptr;   // Part B: chain covariance (set_chain_cov_source)
    std::string                    chain_src_frame_;
    bool                           chain_cov_enabled_ = false;
    ChairConfig&                   cfg_;
    MaskIngestor*                  mask_ingestor_ = nullptr;
    ChairSceneGraph*               scene_graph_   = nullptr;
    std::unique_ptr<DSR::CameraAPI> camera_api_;   // ZED intrinsics, lazily bound to the "zed" node

    std::unordered_map<std::uint64_t, ChairInstance> instances_;
    std::unordered_map<std::uint64_t, Eigen::Vector2f> birth_seeds_;   // tracker-provided birth XY (note_birth)
    std::uint64_t                  room_node_id_ = 0;   // latched per ensure_instance call
    std::ofstream                  ai2_csv_;            // per-cycle AI2 belief log (optional)
};

}  // namespace rc
