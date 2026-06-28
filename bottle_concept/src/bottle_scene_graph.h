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
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_rt_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>
#include <dsr/api/dsr_inner_gaussian_api.h>   // Part B: chain covariance propagation

#include "bottle_config.h"
#include "bottle_instance.h"
#include "bottle_model.h"        // BottleState
#include "../../common/mask_ingestor/mask_ingestor.h"   // MaskIngestor::MasksPacket
#include "prior_store.h"         // BottlePrior

namespace rc {

class BottleSceneGraph
{
public:
    BottleSceneGraph(std::shared_ptr<DSR::DSRGraph> graph,
                     DSR::RT_API* rt_api,
                     DSR::InnerEigenAPI* inner_eigen,
                     BottleConfig& cfg,
                     std::function<void()> relayout = {});

    // The "table" node when the bottle's (bx,by) is over its footprint — the RT parent the bottle
    // hangs from (re-parents room→bottle to table→bottle). std::nullopt ⇒ hang from the room.
    std::optional<DSR::Node> find_table_node(float bx, float by) const;
    // Table-top z (room frame) for the same gate. std::nullopt if no table under (bx,by).
    std::optional<float> find_table_top(float bx, float by) const;

    // Which surface does the bottle rest on? MAP over {room, every table_N} by vertical-support
    // log-evidence: the centre must lie inside a table's ORIENTED footprint AND the OBSERVED base
    // (base_z, not the anchored cz) must sit at its top. σ_z is inflated by the table's published
    // top-z variance, so a poorly-known table is a weak anchor. Returns the chosen RT parent + its
    // top z in room frame (NaN ⇒ room/floor, no z anchor) and the winner's log-evidence margin.
    struct SupportDecision
    {
        std::uint64_t parent_id   = 0;
        std::string   parent_name = "room";
        float         top_z       = std::numeric_limits<float>::quiet_NaN();
        float         margin      = 0.0f;
    };
    SupportDecision decide_support_surface(float cx, float cy, float base_z,
                                           std::uint64_t room_node_id) const;
    // Table top z (room frame) of a specific table node id, or NaN if it's not a valid table.
    float table_top_of(std::uint64_t table_id) const;

    // Create any "bottle_N" cylinder node named in priors that doesn't exist yet, matching each prior
    // to the nearest unused "bottle" mask slice and anchoring it to the table (or room) under it.
    void scaffold_missing_bottle_nodes(const std::vector<BottlePrior>& priors,
                                       const MaskIngestor::MasksPacket& masks,
                                       std::uint64_t room_node_id);

    // BIRTH (InstanceTracker): create a fresh auto-named "bottle_<N>" node at a detection centroid with
    // the default prior size, parented by the support decision. Returns the new node id (0 on failure).
    std::uint64_t create_instance_from_detection(const Eigen::Vector3f& centroid_room,
                                                 std::uint64_t room_node_id);

    // Publish the instance's fitted model to its DSR node (geometry attrs + FE + mesh + RFE queue) and
    // RT edge — only when pose/size moved past the dead-band (FE jitter alone never triggers a rewrite).
    void step_write_model(BottleInstance& inst, DSR::Node& node, float free_energy);
    // Write the room-frame fit on the room/table→bottle RT edge (parent-frame transformed) + P_bottle.
    void write_rt_pose(BottleInstance& inst);

    // Part B: enable adding the localization/chain covariance J·Σ_chain·Jᵀ (source frame → fit frame,
    // capture-stamp pinned) to the published RT-edge cov, via InnerGaussianAPI. Off until set.
    void set_chain_cov_source(DSR::InnerGaussianAPI* gaussian, std::string source_frame, bool enabled);

    // The robot's XY localisation covariance off the room→robot RT edge (0.01·I fallback).
    Eigen::Matrix2f read_robot_covariance(std::uint64_t room_node_id) const;

    // Flat triangle list (room frame) for the cylinder: side wall + top/bottom caps.
    static std::vector<float> make_cylinder_mesh(const BottleState& s, int segments = 16);

private:
    std::shared_ptr<DSR::DSRGraph> G_;
    DSR::RT_API*        rt_api_      = nullptr;
    DSR::InnerEigenAPI* inner_eigen_ = nullptr;
    DSR::InnerGaussianAPI* gaussian_ = nullptr;   // Part B: chain covariance (set via set_chain_cov_source)
    std::string         chain_src_frame_;          // measurement frame the chain cov is computed from
    bool                chain_cov_enabled_ = false;
    BottleConfig&        cfg_;
    std::function<void()> relayout_;   // re-run the graph twopi layout after a node is created/removed
};

}  // namespace rc
