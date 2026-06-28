/*
 * human_fitter.h
 *
 * The active-inference core of human_concept. Owns the per-person instance map and runs the
 * kinematic-model Laplace fit for each "person_*" node every cycle. Unlike bottle/table this is NOT
 * an SDF fit: it wraps the validated Eigen estimator in cpp/core (HumanKinematicModel +
 * AInfLaplacePoseEstimator). Pure belief engine — exposes ensure_instance → observe → run_inference
 * and does NOT write DSR (the worker owns the write-back via HumanSceneGraph).
 */

#pragma once

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

#include <dsr/api/dsr_api.h>

#include "human_config.h"
#include "human_instance.h"
#include "skeleton_source.h"
#include "human_kinematic_model.h"   // rc::human::HumanKinematicModel
#include "vfe_inference.h"           // rc::human::InferenceConfig

namespace rc {

class HumanFitter
{
public:
    HumanFitter(std::shared_ptr<DSR::DSRGraph> graph, HumanConfig& cfg);

    // Per-cycle fresh observation for this person: the keypoints for its track id (if present).
    struct HumanObservation
    {
        bool has_fresh_data = false;
        human::KpArray kp;
        std::optional<std::array<float, human::NUM_KP>> conf;
        float mean_conf  = 0.0f;   // mean per-joint confidence (0..100), or a valid-count proxy
        int   valid_count = 0;
    };

    // The worker hands the fitter this cycle's bodies (polled from the SkeletonSource) before the
    // per-node loop, so observe() can look each instance's track up by id.
    void set_frame(std::vector<SkeletonBody> bodies) { frame_ = std::move(bodies); }
    const std::vector<SkeletonBody>& frame() const { return frame_; }

    // Pure belief API (no DSR write-back). Mirrors the canonical concept-agent loop.
    bool ensure_instance(const DSR::Node& node, std::uint64_t room_node_id);
    HumanObservation observe(HumanInstance& inst, const DSR::Node& node);
    float run_inference(HumanInstance& inst, const HumanObservation& observation);
    bool should_log(const HumanInstance& inst) const;

    std::unordered_map<std::uint64_t, HumanInstance>& instances() { return instances_; }
    void forget_node(std::uint64_t id) { instances_.erase(id); }

    const human::HumanKinematicModel& model() const { return model_; }

private:
    human::InferenceConfig make_infer_config() const;

    std::shared_ptr<DSR::DSRGraph> G_;
    HumanConfig& cfg_;

    // Shared kinematic model (fixed segment lengths from the standard template). Declared BEFORE the
    // instance map so it outlives every per-instance estimator that references it.
    human::HumanKinematicModel model_;

    std::unordered_map<std::uint64_t, HumanInstance> instances_;
    std::vector<SkeletonBody> frame_;          // this cycle's bodies (set by the worker)
    std::uint64_t room_node_id_ = 0;
};

}  // namespace rc
