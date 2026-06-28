/*
 * human_scene_graph.h
 *
 * DSR node/RT I/O for human_concept. Owns everything that reads/writes the distributed graph for a
 * tracked person:
 *   - scaffolding missing "person_N" nodes from live skeleton track ids,
 *   - writing the fitted belief back (free energy, model uncertainty = tr(cov), the predicted skeleton
 *     as a mesh, detection-aliveness feedback attrs) and the room→person RT edge (pelvis pose + a
 *     covariance proxy).
 *
 * Plain class (no Q_OBJECT) constructed by SpecificWorker once G + the DSR APIs are ready.
 */

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_rt_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>

#include "human_config.h"
#include "human_instance.h"
#include "skeleton_source.h"
#include "human_kinematic_model.h"   // rc::human::KpArray

namespace rc {

class HumanSceneGraph
{
public:
    HumanSceneGraph(std::shared_ptr<DSR::DSRGraph> graph,
                    DSR::RT_API* rt_api,
                    DSR::InnerEigenAPI* inner_eigen,
                    HumanConfig& cfg,
                    std::function<void()> relayout = {});

    // Create a "person_<id>" node for any live track id not yet in the graph, anchored to the room at
    // the body's pelvis position.
    void scaffold_missing_person_nodes(const std::vector<SkeletonBody>& bodies,
                                       std::uint64_t room_node_id);

    // Publish the instance's fitted belief to its DSR node (attrs + skeleton mesh) and RT edge.
    void step_write_model(HumanInstance& inst, DSR::Node& node, float free_energy);

    // Write the room→person RT edge (pelvis pose) + a covariance proxy from tr(cov).
    void write_rt_pose(HumanInstance& inst);

    // Pelvis (root) position = midpoint of the hip keypoints; NaN if either hip is missing.
    static Eigen::Vector3f pelvis_of(const human::KpArray& kp);
    // Flat keypoint list (room frame) of a predicted skeleton: 18×3 floats (NaN-skipped downstream).
    static std::vector<float> skeleton_mesh(const human::KpArray& kp);

private:
    std::shared_ptr<DSR::DSRGraph> G_;
    DSR::RT_API*        rt_api_      = nullptr;
    DSR::InnerEigenAPI* inner_eigen_ = nullptr;
    HumanConfig&         cfg_;
    std::function<void()> relayout_;
};

}  // namespace rc
