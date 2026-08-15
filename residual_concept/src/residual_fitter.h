/*
 * residual_fitter.h
 *
 * The active-inference core of residual_concept. Owns the per-obstacle instance map and runs the
 * box-footprint belief fit for each "residual_*" node every cycle:
 *   - instance lifecycle (ensure_instance),
 *   - observation: pick the LiDAR cluster the InstanceTracker assigned this instance (observe),
 *   - inference: lazy belief init (seed from the cluster PCA) → ResidualBelief.update → copy the posterior
 *     back into the model, anchoring cz/height from the cluster z-range + storing the convex hull.
 *
 * Pure belief engine (mirrors BottleFitter): ensure_instance → observe → run_inference, NO DSR writes (the
 * worker owns write-back via ResidualSceneGraph). Much simpler than bottle: no camera/mask/silhouette,
 * no lidar-ray factor, no support bank, no support surface. Plain class (no Q_OBJECT).
 */

#pragma once

#include <cstdint>
#include <fstream>
#include <memory>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>

#include "residual_config.h"
#include "residual_instance.h"
#include "residual_clusterer.h"   // rc::ResidualCluster (the detections)

namespace rc
{

class ResidualFitter
{
public:
    ResidualFitter(std::shared_ptr<DSR::DSRGraph> graph, DSR::InnerEigenAPI* inner_eigen, ResidualConfig& cfg);

    struct ResidualObservation
    {
        bool                   has_fresh_data = false;
        const ResidualCluster* cluster        = nullptr;
    };

    // Create the instance for a "residual_*" node if absent. Returns true the first time it is created.
    bool ensure_instance(const DSR::Node& node, std::uint64_t room_node_id);

    // Stage this cycle's clusters (owned by the worker). observe() reads inst.assigned_cluster_idx into it.
    void set_clusters(const std::vector<ResidualCluster>* clusters) { clusters_ = clusters; }
    // Per-cycle sensor context for the per-point reliability weighting: sensor origin (room frame) → point
    // range; |rot_rate| (rad/s) → ego-motion tangential smear. Set by the worker each compute().
    void set_sensor_context(const Eigen::Vector3f& origin_room, float rot_rate_abs)
    { sensor_origin_ = origin_room; rot_rate_ = rot_rate_abs; }

    ResidualObservation observe(ResidualInstance& inst);
    // The fit: one recursive full-covariance AI2 belief update on the cluster points; posterior copied into
    // inst.model (footprint), cz/height anchored from the cluster z-range, hull stored. Returns free energy.
    float run_inference(ResidualInstance& inst, const ResidualObservation& observation);

    bool should_log(const ResidualInstance& inst) const;

    std::unordered_map<std::uint64_t, ResidualInstance>& instances() { return instances_; }
    void forget_node(std::uint64_t id) { instances_.erase(id); }
    void note_birth(std::uint64_t id, const Eigen::Vector2f& xy) { birth_seeds_[id] = xy; }

private:
    void log_ai2_csv(const ResidualInstance& inst, int point_count, float R, float energy);

    std::shared_ptr<DSR::DSRGraph> G_;
    DSR::InnerEigenAPI*            inner_eigen_ = nullptr;
    ResidualConfig&               cfg_;

    const std::vector<ResidualCluster>*                 clusters_ = nullptr;   // this cycle's detections
    Eigen::Vector3f                                     sensor_origin_ = Eigen::Vector3f::Zero();
    float                                               rot_rate_ = 0.0f;      // |robot yaw rate| (rad/s)
    std::unordered_map<std::uint64_t, ResidualInstance> instances_;
    std::unordered_map<std::uint64_t, Eigen::Vector2f>  birth_seeds_;
    std::uint64_t                                       room_node_id_ = 0;
    std::ofstream                                       ai2_csv_;
};

}  // namespace rc
