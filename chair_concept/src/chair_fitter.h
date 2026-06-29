/*
 * chair_fitter.h
 *
 * The active-inference core of chair_concept (mirrors bottle_concept/bottle_fitter.h). Owns the
 * per-chair instance map and runs the free-energy fit for each "chair_*" node:
 *   - instance lifecycle (ensure_instance + the ChairModel/SampleQueue factories),
 *   - observation: split the selected mask's support points into queue anchors vs residuals,
 *   - inference: voxel-bank ingest + cold-start snap + queue update + the belief evolve/accept
 *     step (warm-start observability policy) + RFE memory refresh,
 *   - the chair-owned voxel memory (ownership gate + FNV voxel keys).
 *
 * Collaborates with MaskIngestor (masks) and ChairSceneGraph (robot covariance). SpecificWorker
 * keeps the orchestration (process_chair_node), the DSR write-back call, and the post-fit epistemic
 * / affordance / Qt-diagnostics steps. Plain class (no Q_OBJECT).
 */

#pragma once

#include <array>
#include <cstdint>
#include <fstream>
#include <memory>
#include <unordered_map>
#include <vector>

#include <optional>

#include <Eigen/Dense>
#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>
#include <dsr/api/dsr_inner_gaussian_api.h>   // Part B: chain covariance propagation
#include <dsr/api/dsr_camera_api.h>

#include "chair_config.h"        // rc::ChairConfig
#include "chair_instance.h"      // rc::ChairInstance, ChairState
#include "chair_model.h"         // ChairModel / ChairModelParams / FreeEnergyDecomposition
#include "../../common/sample_queue/sample_queue.h"        // SampleQueue / SampleQueueParams
#include "prior_store.h"         // PriorStore / ChairPrior
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
                const std::vector<ChairPrior>& priors,
                MaskIngestor* mask_ingestor,
                ChairSceneGraph* scene_graph);

    // Create the instance for a "chair_*" node if absent (from prior/RT). Returns true the
    // first time it is created (so the worker can register Qt series / canvas pos). Latches room_id.
    bool ensure_instance(const DSR::Node& node, std::uint64_t room_id);

    // Read the "masks" node → candidate/residual split against the current SDF.
    ChairObservation observe(ChairInstance& inst, const DSR::Node& node);
    // One free-energy fit cycle (voxel-bank ingest + cold-start + queue + belief). Returns the FE.
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

private:
    struct ChairBeliefEvidence
    {
        std::vector<Eigen::Vector3f> fit_pts;
        std::vector<float> fit_weights;
        std::vector<Eigen::Vector3f> eval_pts;
        std::vector<float> eval_weights;
        int residual_count = 0;
        int trusted_point_count = 0;
        int historical_anchor_count = 0;
        float residual_precision = 0.0f;

        [[nodiscard]] bool has_evaluation() const { return not eval_pts.empty(); }
        [[nodiscard]] bool can_optimize() const { return not fit_pts.empty(); }
    };

    struct ChairBeliefPolicy
    {
        static float clamp01(float value);
        static float lerp(float start, float end, float gain);
        static float wrap_angle(float angle);
        static float angle_lerp(float start, float end, float gain);
        static ChairState apply_observability_warm_start(const ChairState& previous,
                                                         const ChairState& raw,
                                                         const ChairModelParams& params,
                                                         const ChairConfig& cfg,
                                                         float confidence,
                                                         const std::array<float, 6>& coverage,
                                                         int point_count,
                                                         float settle_gain = 1.0f,
                                                         float info_w = 0.0f,
                                                         float info_h = 0.0f,
                                                         const std::array<float, 8>* kalman_gain = nullptr);
        static float update_warm_confidence(float previous_confidence,
                                            const ChairConfig& cfg,
                                            const std::array<float, 6>& coverage,
                                            int point_count,
                                            int residual_count,
                                            float residual_precision);
    };

    void step_queue_update(ChairInstance& inst,
                           const std::vector<Eigen::Vector3f>& candidate_pts,
                           float observation_precision);
    float step_model_update(ChairInstance& inst,
                            const std::vector<Eigen::Vector3f>& residual_pts,
                            const std::vector<Eigen::Vector3f>& current_pts,
                            float residual_precision,
                            bool fresh_observation);
    ChairBeliefEvidence compose_belief_evidence(const ChairInstance& inst,
                                                const std::vector<Eigen::Vector3f>& residual_pts,
                                                const std::vector<Eigen::Vector3f>& current_pts,
                                                float residual_precision) const;
    void evolve_chair_belief(ChairInstance& inst, const ChairBeliefEvidence& evidence);
    float accept_chair_belief(ChairInstance& inst,
                              const ChairState& previous_state,
                              const ChairBeliefEvidence& evidence);
    void refresh_chair_memory(ChairInstance& inst);

    // RGB-mask silhouette: back-project the chair mask contour to room-frame rays and hand them to
    // the model for its differentiable silhouette term. No-op if mask_precision<=0 or no camera.
    void feed_silhouette(ChairInstance& inst);
    // room_T_zed (camera→room). pose_ts_ms pins the room→body hop to the mask's capture time (Nearest RT
    // query) so a moving base doesn't back-project a stale contour through the current pose; the rigid
    // body→zed mount is always queried latest. 0 → current pose (no timestamp available).
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
    SampleQueueParams make_queue_params() const;

    // Keep only points within ±cfg_.top_band_m of the model's estimated chair-top height — drops the
    // floor/under-chair/clutter population before the fit/extent terms. Identity if the gate is off.
    std::vector<Eigen::Vector3f> gate_to_top_band(const std::vector<Eigen::Vector3f>& pts,
                                                  const ChairModel& model) const;

    // Append one row of Fisher-filter evolution (state + per-DOF obs/accumulated info + posterior
    // std) to cfg_.fisher_csv_path. No-op if the path is empty. Lazily opens + writes the header.
    void log_fisher_csv(const ChairInstance& inst, bool fresh, float free_energy,
                        int point_count, float silres);

    std::shared_ptr<DSR::DSRGraph> G_;
    DSR::InnerEigenAPI*            inner_eigen_ = nullptr;
    DSR::InnerGaussianAPI*         gaussian_    = nullptr;   // Part B: chain covariance (set_chain_cov_source)
    std::string                    chain_src_frame_;
    bool                           chain_cov_enabled_ = false;
    ChairConfig&                   cfg_;
    const std::vector<ChairPrior>& priors_;
    MaskIngestor*                  mask_ingestor_ = nullptr;
    ChairSceneGraph*               scene_graph_   = nullptr;
    std::unique_ptr<DSR::CameraAPI> camera_api_;   // ZED intrinsics, lazily bound to the "zed" node

    std::unordered_map<std::uint64_t, ChairInstance> instances_;
    std::unordered_map<std::uint64_t, Eigen::Vector2f> birth_seeds_;   // tracker-provided birth XY (note_birth)
    std::uint64_t                  room_node_id_ = 0;   // latched per ensure_instance call
    std::ofstream                  fisher_csv_;         // per-cycle Fisher-filter evolution log (optional)

    // Shared per-DOF belief stabiliser (Fisher filter + Kalman acceptance + CUSUM/SPRT). Holds the
    // algorithm + params (refreshed from cfg_ each accept); per-chair state lives in inst.stab.
    BeliefStabilizer<8>            stabilizer_;
    void refresh_stabilizer_params();   // map cfg_ → stabilizer_ params + the chair DOF layout
};

}  // namespace rc
