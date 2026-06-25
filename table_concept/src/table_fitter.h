/*
 * table_fitter.h
 *
 * The active-inference core of table_concept (mirrors bottle_concept/bottle_fitter.h). Owns the
 * per-table instance map and runs the free-energy fit for each "table_*" node:
 *   - instance lifecycle (ensure_instance + the TableModel/SampleQueue factories),
 *   - observation: split the selected mask's support points into queue anchors vs residuals,
 *   - inference: voxel-bank ingest + cold-start snap + queue update + the belief evolve/accept
 *     step (warm-start observability policy) + RFE memory refresh,
 *   - the table-owned voxel memory (ownership gate + FNV voxel keys).
 *
 * Collaborates with MaskIngestor (masks) and TableSceneGraph (robot covariance). SpecificWorker
 * keeps the orchestration (process_table_node), the DSR write-back call, and the post-fit epistemic
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
#include <dsr/api/dsr_camera_api.h>

#include "table_config.h"        // rc::TableConfig
#include "table_instance.h"      // rc::TableInstance, TableState
#include "table_model.h"         // TableModel / TableModelParams / FreeEnergyDecomposition
#include "sample_queue.h"        // SampleQueue / SampleQueueParams
#include "prior_store.h"         // PriorStore / TablePrior
#include "mask_ingestor.h"
#include "table_scene_graph.h"

namespace rc {

class TableFitter
{
public:
    struct TableObservation
    {
        bool has_fresh_data = false;
        float explanation_ratio = 1.0f;
        std::vector<Eigen::Vector3f> candidate_pts;
        std::vector<Eigen::Vector3f> residual_pts;
    };

    TableFitter(std::shared_ptr<DSR::DSRGraph> graph,
                DSR::InnerEigenAPI* inner_eigen,
                TableConfig& cfg,
                const std::vector<TablePrior>& priors,
                MaskIngestor* mask_ingestor,
                TableSceneGraph* scene_graph);

    // Create the instance for a "table_*" node if absent (from prior/RT). Returns true the
    // first time it is created (so the worker can register Qt series / canvas pos). Latches room_id.
    bool ensure_instance(const DSR::Node& node, std::uint64_t room_id);

    // Read the "masks" node → candidate/residual split against the current SDF.
    TableObservation observe(TableInstance& inst, const DSR::Node& node);
    // One free-energy fit cycle (voxel-bank ingest + cold-start + queue + belief). Returns the FE.
    float run_inference(TableInstance& inst, const TableObservation& observation);

    std::unordered_map<std::uint64_t, TableInstance>& instances() { return instances_; }
    void forget_node(std::uint64_t id) { instances_.erase(id); }
    bool should_log_table(const TableInstance& inst) const;

private:
    struct TableBeliefEvidence
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

    struct TableBeliefPolicy
    {
        static float clamp01(float value);
        static float lerp(float start, float end, float gain);
        static float wrap_angle(float angle);
        static float angle_lerp(float start, float end, float gain);
        static TableState apply_observability_warm_start(const TableState& previous,
                                                         const TableState& raw,
                                                         const TableModelParams& params,
                                                         const TableConfig& cfg,
                                                         float confidence,
                                                         const std::array<float, 6>& coverage,
                                                         int point_count,
                                                         float settle_gain = 1.0f,
                                                         float info_w = 0.0f,
                                                         float info_h = 0.0f,
                                                         const std::array<float, 8>* kalman_gain = nullptr);
        static float update_warm_confidence(float previous_confidence,
                                            const TableConfig& cfg,
                                            const std::array<float, 6>& coverage,
                                            int point_count,
                                            int residual_count,
                                            float residual_precision);
    };

    void step_queue_update(TableInstance& inst,
                           const std::vector<Eigen::Vector3f>& candidate_pts,
                           float observation_precision);
    float step_model_update(TableInstance& inst,
                            const std::vector<Eigen::Vector3f>& residual_pts,
                            const std::vector<Eigen::Vector3f>& current_pts,
                            float residual_precision,
                            bool fresh_observation);
    TableBeliefEvidence compose_belief_evidence(const TableInstance& inst,
                                                const std::vector<Eigen::Vector3f>& residual_pts,
                                                const std::vector<Eigen::Vector3f>& current_pts,
                                                float residual_precision) const;
    void evolve_table_belief(TableInstance& inst, const TableBeliefEvidence& evidence);
    float accept_table_belief(TableInstance& inst,
                              const TableState& previous_state,
                              const TableBeliefEvidence& evidence);
    void refresh_table_memory(TableInstance& inst);

    // RGB-mask silhouette: back-project the table mask contour to room-frame rays and hand them to
    // the model for its differentiable silhouette term. No-op if mask_precision<=0 or no camera.
    void feed_silhouette(TableInstance& inst);
    // room_T_zed (camera→room). pose_ts_ms pins the room→body hop to the mask's capture time (Nearest RT
    // query) so a moving base doesn't back-project a stale contour through the current pose; the rigid
    // body→zed mount is always queried latest. 0 → current pose (no timestamp available).
    std::optional<Eigen::Matrix4d> room_T_zed_matrix(std::uint64_t pose_ts_ms = 0) const;
    // Project the current model through the camera extrinsic → normalised in-image ROI (centre
    // offset + fill), stored on the instance for the controller's centring/dwell lock-on search.
    void compute_projected_roi(TableInstance& inst);

    void ingest_observation_voxels(TableInstance& inst, const TableObservation& observation);
    bool is_voxel_owned_by_table(const TableInstance& inst, const Eigen::Vector3f& point) const;
    static std::uint64_t voxel_key(const Eigen::Vector3f& point, float quantization_m);

    TableModelParams  make_model_params() const;
    SampleQueueParams make_queue_params() const;

    // Keep only points within ±cfg_.top_band_m of the model's estimated table-top height — drops the
    // floor/under-table/clutter population before the fit/extent terms. Identity if the gate is off.
    std::vector<Eigen::Vector3f> gate_to_top_band(const std::vector<Eigen::Vector3f>& pts,
                                                  const TableModel& model) const;

    // Append one row of Fisher-filter evolution (state + per-DOF obs/accumulated info + posterior
    // std) to cfg_.fisher_csv_path. No-op if the path is empty. Lazily opens + writes the header.
    void log_fisher_csv(const TableInstance& inst, bool fresh, float free_energy,
                        int point_count, float silres);

    std::shared_ptr<DSR::DSRGraph> G_;
    DSR::InnerEigenAPI*            inner_eigen_ = nullptr;
    TableConfig&                   cfg_;
    const std::vector<TablePrior>& priors_;
    MaskIngestor*                  mask_ingestor_ = nullptr;
    TableSceneGraph*               scene_graph_   = nullptr;
    std::unique_ptr<DSR::CameraAPI> camera_api_;   // ZED intrinsics, lazily bound to the "zed" node

    std::unordered_map<std::uint64_t, TableInstance> instances_;
    std::uint64_t                  room_node_id_ = 0;   // latched per ensure_instance call
    std::ofstream                  fisher_csv_;         // per-cycle Fisher-filter evolution log (optional)
};

}  // namespace rc
