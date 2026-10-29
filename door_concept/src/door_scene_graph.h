/*
 * door_scene_graph.h
 *
 * DSR node/RT I/O layer for door_concept (mirrors bottle_concept/bottle_scene_graph.h):
 * births "door_N" nodes from tracker detections, writes the fitted model back (geometry
 * attrs + mesh + RFE/support-bank export + room→door RT edge), reads the robot localisation
 * covariance, and writes the epistemic action proposal.
 *
 * Plain class (no Q_OBJECT) constructed by SpecificWorker once G + the DSR APIs are ready.
 * The graph relayout is injected as a callback to stay decoupled from the GUI (graph_viewers).
 */

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <Eigen/Dense>

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_rt_api.h>

#include "door_config.h"       // rc::DoorConfig
#include "door_instance.h"     // rc::DoorInstance, DoorState
#include "door_pragmatics.h"   // rc::door::InteractionState
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
    // `preferred_name` (identity RE-ACQUISITION): the remembered name of a door coming back at this spot; taken when
    // no live node wears it, else a fresh name. `next_door_index` is the agent's persisted name counter, advanced for
    // a fresh name (rc::door::allocate_door_name: names are opaque and never recycled).
    std::uint64_t create_instance_from_detection(const Eigen::Vector3f& centroid_room,
                                                 std::uint64_t room_node_id,
                                                 std::string_view preferred_name,
                                                 std::uint64_t& next_door_index);

    // ── ENTRY MIRROR under a PROTO-ROOM ──────────────────────────────────────────────────────────────
    // A proto-room (room_concept, a `room` node with a `proto` self-edge) is born when the robot crosses a
    // door into space the current room does not explain, and its frame IS that door's aperture. The door
    // it was entered through must hang from it — but the physical door already has ONE belief, the
    // instance in the current room, and a second fitted instance would count the same door twice (the
    // overlap merge would retire one of them). So the proto-room gets a MIRROR: a plain `object` node
    // `door_N` (object_subtype "door", parent = the proto-room) with no fitter instance, no affordances,
    // no ghost and no identity row. Its RT edge proto→mirror is `T_proto_source`, resolved once and never
    // rewritten — the proto-room's gauge is frozen, so the door's pose in it is a constant.
    // Returns the new node id (0 on failure). The mirror asks for its source's name (live ⇒ a fresh counter name);
    // `next_door_index` as for create_instance_from_detection.
    /// A door node under `room` with the given geometry (a door carried across from the room just left).
    std::uint64_t create_door_copy(const DSR::Node& room, float width, float depth, float height,
                                   std::optional<float> open_prob, std::optional<float> phi_rad,
                                   const Eigen::Vector3f& t, const Eigen::Vector3f& euler_xyz,
                                   std::uint64_t& next_door_index, const std::string& source_label,
                                   std::string_view preferred_name = {});
    std::uint64_t create_entry_mirror(const DSR::Node& proto_room,
                                      const DSR::Node& source_door,
                                      const Eigen::Vector3f& t_proto_source,
                                      const Eigen::Vector3f& euler_xyz_proto_source,
                                      std::uint64_t& next_door_index);

    // Publish the instance's fitted model to its DSR node (geometry + FE + mesh + RFE/support-bank) and
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

    // Publish the DOOR INTERACTION channel on the door node: the leaf angle, and the three probabilities
    // the pragmatic affordances' completion predicates are written against (door_phi_rad, door_open_prob,
    // door_reach_prob, door_crossing_progress).
    //
    // ★THESE EXIST BECAUSE A PREDICATE CAN ONLY BE WRITTEN AGAINST SOMETHING THAT IS ON THE GRAPH. The
    // executor evaluates a contract's goal clauses against attributes of the affordance's parent — this
    // node — so an affordance bound to a name nobody writes never satisfies and times out on every
    // attempt. That is the failure default_contract_for() records for cabinet and door, and the reason
    // these four names must stay in step with the contracts in door_pragmatics.h.
    // ★AND THEY ARE PERCEPTUAL, NOT PROTOCOL. None of them is ever set from a provider's reply: a door
    // provider's `Delivered` means it did what it was asked, never that the leaf moved. door_open_prob
    // comes from the image, marginalised over the leaf-angle posterior, like every other belief here.
    void write_interaction_state(DSR::Node& node, const rc::door::InteractionState& st);

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

    // `preferred_name` when no live node wears it, else `door_<next_door_index++>` (rc::door::allocate_door_name).
    // Shared by detection births, copies and entry mirrors so they can never hand out the same name.
    std::string next_door_name(std::string_view preferred_name, std::uint64_t& next_door_index) const;

    std::shared_ptr<DSR::DSRGraph> G_;
    DSR::RT_API*          rt_api_ = nullptr;
    const DoorConfig&    cfg_;
    std::function<void()> relayout_;
};

}  // namespace rc
