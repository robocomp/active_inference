// Recall nodes: ltom's offer of "this remembered object is probably still around here" to the concept agents.
// A recall node is a `metaconcept` named `ltom_recall_<uid decimal>`, an RT child of the live room at the remembered
// pose, with `object_uid` and `object_subtype` only. Main thread only (graph writes + the RT API).
#pragma once
#include "recall_select.h"

#include <dsr/api/dsr_api.h>

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>

#include "../../common/owned_nodes/owned_nodes.h"

namespace ltom
{
// The nodes this agent owns besides its agent node: the stale sweep (agent + test) and nothing else defines them.
inline const rc::owned::Spec kRecallSpec{.agent = "ltom_agent", .name_prefix = "ltom_recall_", .node_types = {"metaconcept"},
                                                  .legacy_parent_types = {}, .affordance_prefix = {}};

// The room recalls may be offered under: `room` unless it is a proto-room (provisional frame; graph_sightings records
// nothing there either, and a trailing-digit key could coincide with a remembered room's).
inline std::optional<std::uint64_t> recall_room(DSR::DSRGraph& G, std::optional<std::uint64_t> room)
{
    if (room and G.get_edge(*room, *room, "proto").has_value()) return std::nullopt;   // == rc::room::is_proto
    return room;
}

class RecallPublisher
{
public:
    explicit RecallPublisher(std::shared_ptr<DSR::DSRGraph> G);
    // Make the graph hold exactly `want` under room `room_id`: insert missing, move changed, delete the rest.
    // room_id == nullopt means "no room known": nothing is held. Only touches metaconcept nodes named ltom_recall_*.
    void sync(std::optional<std::uint64_t> room_id, std::span<const Recall> want);

private:
    std::shared_ptr<DSR::DSRGraph> G_;
    std::unique_ptr<DSR::RT_API> rt_api_;
    std::set<std::string> warned_;   // failures are logged once per (node, what)

    void warn_once(const std::string& name, const char* what);
    void insert(std::uint64_t room_id, const Recall& r);
    void write_pose(std::uint64_t room_id, std::uint64_t node_id, const std::string& name, const Recall& r);
    bool pose_current(const Recall& r) const;
    std::map<std::uint64_t, Keyframe> written_;   // uid -> keyframe last written to its node
};
}
