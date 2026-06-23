/*
 * bottle_fitter.h
 *
 * The active-inference core of bottle_concept. Owns the per-bottle instance map and
 * runs the free-energy fit for each "bottle_*" cylinder node every cycle:
 *   - instance lifecycle (ensure_instance + the BottleModel/SampleQueue factories),
 *   - observation: split the selected mask's support points into queue anchors vs
 *     residuals against the current SDF (observe_bottle_node),
 *   - inference: voxel-bank ingest + cold-start seed (with camera-ward de-projection)
 *     + queue update + RGB silhouette likelihood + the gradient/free-energy step,
 *   - the bottle-owned voxel memory (ownership gate + FNV voxel keys).
 *
 * Collaborates with MaskIngestor (masks), BottleSceneGraph (table lookups, model
 * write-back, robot covariance) and BottleEvaluator (per-cycle eval logging). Plain
 * class (no Q_OBJECT) constructed by SpecificWorker after the other collaborators.
 */

#pragma once

#include <cstdint>
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
#include "sample_queue.h"        // SampleQueue / SampleQueueParams
#include "prior_store.h"         // BottlePrior
#include "mask_ingestor.h"
#include "bottle_scene_graph.h"
#include "bottle_evaluator.h"

namespace rc {

class BottleFitter
{
public:
    BottleFitter(std::shared_ptr<DSR::DSRGraph> graph,
                 DSR::InnerEigenAPI* inner_eigen,
                 BottleConfig& cfg,
                 const std::vector<BottlePrior>& priors,
                 MaskIngestor* perception,
                 BottleSceneGraph* scene_graph,
                 BottleEvaluator* evaluator);

    // Run one fit cycle for a bottle node: ensure its instance, observe, infer, anchor to the
    // table, write the model back (via scene_graph) and log the eval row (via evaluator).
    void process_bottle_node(const DSR::Node& node, std::uint64_t room_node_id);

    // The live instance map (the validation sweep mutates it; del_node prunes it).
    std::unordered_map<std::uint64_t, BottleInstance>& instances() { return instances_; }
    void forget_node(std::uint64_t id) { instances_.erase(id); }

private:
    struct BottleObservation
    {
        bool has_fresh_data = false;
        float explanation_ratio = 1.0f;
        std::vector<Eigen::Vector3f> candidate_pts;
        std::vector<Eigen::Vector3f> residual_pts;
    };

    void ensure_instance(const DSR::Node& node);
    bool should_log(const BottleInstance& inst) const;

    BottleObservation observe_bottle_node(BottleInstance& inst, const DSR::Node& node);
    float run_bottle_inference(BottleInstance& inst, const BottleObservation& observation);
    void step_queue_update(BottleInstance& inst,
                           const std::vector<Eigen::Vector3f>& candidate_pts,
                           float observation_precision);
    float step_model_update(BottleInstance& inst,
                            const std::vector<Eigen::Vector3f>& residual_pts,
                            float residual_precision);

    // Voxel bank (bottle-owned historical memory).
    void ingest_observation_voxels(BottleInstance& inst, const BottleObservation& observation);
    bool is_voxel_owned_by_bottle(const BottleInstance& inst, const Eigen::Vector3f& point) const;
    static std::uint64_t voxel_key(const Eigen::Vector3f& point, float quantization_m);

    // Feed the fitted model the RGB-mask edge rays as a silhouette likelihood.
    void feed_silhouette(BottleInstance& inst);
    // room_T_zed (camera→room) as a plain 4×4, composed room→body→zed at ts=0 (alignment-safe).
    std::optional<Eigen::Matrix4d> room_T_zed_matrix() const;

    // Factory helpers (config → model/queue params).
    BottleModelParams make_model_params() const;
    SampleQueueParams make_queue_params() const;

    std::shared_ptr<DSR::DSRGraph>  G_;
    DSR::InnerEigenAPI*             inner_eigen_ = nullptr;
    BottleConfig&                    cfg_;
    const std::vector<BottlePrior>& priors_;
    MaskIngestor*               mask_ingestor_  = nullptr;
    BottleSceneGraph*               scene_graph_ = nullptr;
    BottleEvaluator*                evaluator_   = nullptr;

    std::unique_ptr<DSR::CameraAPI> camera_api_;   // ZED intrinsics, lazily bound to the "zed" node
    std::unordered_map<std::uint64_t, BottleInstance> instances_;
    std::uint64_t                   room_node_id_ = 0;   // refreshed each process_bottle_node call
};

}  // namespace rc
