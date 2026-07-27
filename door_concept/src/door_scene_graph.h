/*
 * door_scene_graph.h
 *
 * DSR node/RT I/O layer for door_concept (mirrors bottle_concept/bottle_scene_graph.h):
 * births "door_N" nodes from tracker detections, writes the fitted model back (geometry
 * attrs + mesh + RFE/voxel-bank export + room→door RT edge), reads the robot localisation
 * covariance, and writes the epistemic action proposal.
 *
 * Plain class (no Q_OBJECT) constructed by SpecificWorker once G + the DSR APIs are ready.
 * The graph relayout is injected as a callback to stay decoupled from the GUI (graph_viewers).
 */

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_rt_api.h>

#include "door_config.h"       // rc::DoorConfig
#include "door_instance.h"     // rc::DoorInstance, DoorState
#include "../../common/mask_ingestor/mask_ingestor.h"      // MaskIngestor::MasksPacket
#include "epistemic_planner.h"  // EpistemicProposal

namespace rc {

class DoorSceneGraph
{
public:
    DoorSceneGraph(std::shared_ptr<DSR::DSRGraph> graph,
                    DSR::RT_API* rt_api,
                    const DoorConfig& cfg,
                    std::function<void()> relayout);

    // Data-driven birth: create a fresh "door_N" node at the detection centroid (room frame), seeded
    // with the Tracker.Birth* default geometry. Returns the new node id (0 on failure). Used by the
    // worker's instance tracker — the only instance-lifecycle path.
    std::uint64_t create_instance_from_detection(const Eigen::Vector3f& centroid_room,
                                                 std::uint64_t room_node_id);

    // Publish the instance's fitted model to its DSR node (geometry + FE + mesh + RFE/voxel-bank) and
    // the room→door RT edge. persist_* resolves the node by id first; both no-op if the node is gone.
    bool persist_door_belief(DoorInstance& inst, std::uint64_t node_id, std::uint64_t room_id, float free_energy);
    void step_write_model(DoorInstance& inst, DSR::Node& node, std::uint64_t room_id, float free_energy);
    void write_rt_pose(std::uint64_t room_id, DoorInstance& inst);

    // Attach the door pose covariance (rt_covariance_att, 6×6 SE3) on the room→door RT edge, mapped
    // from the belief's full Σ. Writes when `force` (a geometry republish)
    // OR the covariance trace changed meaningfully since last write — so a stationary-but-tightening
    // door stays current without per-cycle edge churn. No-op if disabled / the edge is absent.
    void write_rt_covariance(std::uint64_t room_id, DoorInstance& inst, bool force);

    // Write the epistemic next-best-view proposal onto a door node.
    void write_epistemic_proposal(DSR::Node& node, const EpistemicProposal& prop);

    // Flat triangle list (room frame): top slab + 4 legs.
    static std::vector<float> make_door_mesh(const DoorState& s);

private:
    void write_door_mesh(DoorInstance& inst, DSR::Node& node);

    // A door belongs to a WALL. room_concept publishes one "wall_*" node per room-polygon edge, each with a
    // room→wall RT edge (origin = wall midpoint at half-room-height, yaw = wall tangent) and width_m = length.
    // resolve_wall picks the wall node whose segment is nearest the door centre (room frame) and returns its
    // frame, so the door can hang from it. ok=false when no wall node exists yet (→ fall back to the room).
    struct WallRef
    {
        std::uint64_t id = 0;
        Eigen::Vector2f mid{0.0f, 0.0f};   // wall midpoint (room frame)
        float yaw = 0.0f;                  // wall tangent yaw (room frame)
        float z0  = 0.0f;                  // wall-node origin height (room frame) = half room height
        float L   = 0.0f;                  // wall length (m)
        bool  ok  = false;
    };
    WallRef resolve_wall(std::uint64_t room_id, const Eigen::Vector2f& door_xy) const;

    std::shared_ptr<DSR::DSRGraph> G_;
    DSR::RT_API*          rt_api_ = nullptr;
    const DoorConfig&    cfg_;
    std::function<void()> relayout_;
};

}  // namespace rc
