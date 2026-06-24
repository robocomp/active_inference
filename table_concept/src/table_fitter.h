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
                                                         float settle_gain = 1.0f);
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
                            float residual_precision);
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
    std::optional<Eigen::Matrix4d> room_T_zed_matrix() const;

    void ingest_observation_voxels(TableInstance& inst, const TableObservation& observation);
    bool is_voxel_owned_by_table(const TableInstance& inst, const Eigen::Vector3f& point) const;
    static std::uint64_t voxel_key(const Eigen::Vector3f& point, float quantization_m);

    TableModelParams  make_model_params() const;
    SampleQueueParams make_queue_params() const;

    std::shared_ptr<DSR::DSRGraph> G_;
    DSR::InnerEigenAPI*            inner_eigen_ = nullptr;
    TableConfig&                   cfg_;
    const std::vector<TablePrior>& priors_;
    MaskIngestor*                  mask_ingestor_ = nullptr;
    TableSceneGraph*               scene_graph_   = nullptr;
    std::unique_ptr<DSR::CameraAPI> camera_api_;   // ZED intrinsics, lazily bound to the "zed" node

    std::unordered_map<std::uint64_t, TableInstance> instances_;
    std::uint64_t                  room_node_id_ = 0;   // latched per ensure_instance call
};

}  // namespace rc
