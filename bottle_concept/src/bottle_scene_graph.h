/*
 * bottle_scene_graph.h
 *
 * DSR node/RT I/O layer for bottle_concept. Owns everything that reads or writes
 * the distributed graph for the bottle's geometry and pose:
 *   - table lookups (find_table_node / find_table_top) used to anchor the bottle,
 *   - scaffolding missing "bottle_N" cylinder nodes from priors matched to masks,
 *   - writing the fitted model back (geometry attrs + mesh + room→bottle RT edge
 *     with its Laplace covariance),
 *   - reading the robot's localisation covariance off the room→robot RT edge.
 *
 * Plain class (no Q_OBJECT) constructed by SpecificWorker once G + the DSR APIs
 * are ready. Runtime-varying inputs (priors, masks, the room node id) are passed
 * per-call rather than cached, so the collaborator never holds stale state.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <Eigen/Dense>

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_rt_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>

#include "agent_config.h"
#include "bottle_instance.h"
#include "bottle_model.h"        // BottleState
#include "bottle_perception.h"   // BottlePerception::MasksPacket
#include "prior_store.h"         // BottlePrior

class BottleSceneGraph
{
public:
    BottleSceneGraph(std::shared_ptr<DSR::DSRGraph> graph,
                     DSR::RT_API* rt_api,
                     DSR::InnerEigenAPI* inner_eigen,
                     AgentConfig& cfg);

    // The "table" node when the bottle's (bx,by) is over its footprint — the RT parent the bottle
    // hangs from (re-parents room→bottle to table→bottle). std::nullopt ⇒ hang from the room.
    std::optional<DSR::Node> find_table_node(float bx, float by) const;
    // Table-top z (room frame) for the same gate. std::nullopt if no table under (bx,by).
    std::optional<float> find_table_top(float bx, float by) const;

    // Create any "bottle_N" cylinder node named in priors that doesn't exist yet, matching each prior
    // to the nearest unused "bottle" mask slice and anchoring it to the table (or room) under it.
    void scaffold_missing_bottle_nodes(const std::vector<BottlePrior>& priors,
                                       const BottlePerception::MasksPacket& masks,
                                       std::uint64_t room_node_id);

    // Publish the instance's fitted model to its DSR node (geometry attrs + FE + mesh + RFE queue) and
    // RT edge — only when pose/size moved past the dead-band (FE jitter alone never triggers a rewrite).
    void step_write_model(BottleInstance& inst, DSR::Node& node, float free_energy);
    // Write the room-frame fit on the room/table→bottle RT edge (parent-frame transformed) + P_bottle.
    void write_rt_pose(BottleInstance& inst);

    // The robot's XY localisation covariance off the room→robot RT edge (0.01·I fallback).
    Eigen::Matrix2f read_robot_covariance(std::uint64_t room_node_id) const;

    // Flat triangle list (room frame) for the cylinder: side wall + top/bottom caps.
    static std::vector<float> make_cylinder_mesh(const BottleState& s, int segments = 16);

private:
    std::shared_ptr<DSR::DSRGraph> G_;
    DSR::RT_API*        rt_api_      = nullptr;
    DSR::InnerEigenAPI* inner_eigen_ = nullptr;
    AgentConfig&        cfg_;
};
