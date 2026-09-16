/*
 * passage_live.h — the LIVE passage between a proto-room and the room the robot came from.
 *
 * WHAT HAPPENS LIVE. The robot crosses a door; room_concept births a PROTO-ROOM `room_<k>` (a `room`
 * node with a `proto` SELF-edge, hung from the robot like the apartment, polygon empty) for the space
 * the current room does not explain. `current` STAYS on the room the robot came from while the new one
 * is proto. door_concept adds an ENTRY MIRROR `door_<N>` under the proto-room (`parent` = the proto-room,
 * RT proto→door): the same physical hole, seen from the new side.
 *
 * WHAT THIS DOES, once per cycle, on the LIVE graph only:
 *   - for every entry mirror, find its COUNTERPART: the door of the current room nearest to it, both
 *     poses read in the current room's frame through the RT tree (via the robot). An argmin over the
 *     doors that exist -- no distance cutoff, so no door is ever excluded by it;
 *   - keep exactly ONE `metaconcept` node `passage_<n>` (object_subtype "passage", NO RT edge, no pose)
 *     with a `match` edge to each of the two doors -- the same abstract passage memory keeps after an
 *     eviction (EVICTION.md), made visible while the robot is still in the doorway;
 *   - delete it the moment either door disappears.
 *
 * OWNERSHIP. ltsm_agent owns every live `metaconcept` named `passage_*` with subtype "passage". It does
 * not run the presence monitor, so an `[Owns]` entry would be read by nothing; ownership is enforced
 * here instead: remove_owned() on a graceful exit and sweep_stale() at startup for a crashed run. Both
 * act on the graph this object was constructed with -- the LIVE one -- and it is never handed the
 * memory graph, so memory's own `passage_N` on domain 2 is out of its reach by construction.
 */
#pragma once

#include <dsr/api/dsr_api.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <string>
#include <vector>

namespace ltsm
{

class PassageLive
{
public:
    explicit PassageLive(std::shared_ptr<DSR::DSRGraph> live);

    struct Step
    {
        int passages  = 0;   ///< live passages after the step
        int created   = 0;
        int rematched = 0;   ///< counterpart changed (a nearer door of the current room appeared)
        int deleted   = 0;   ///< an end disappeared
    };

    /// One cycle: retire passages whose ends are gone, then ensure one per entry mirror.
    Step step();

    /// Startup: delete live passages left behind by a crashed / SIGKILLed previous run.
    int sweep_stale();

    /// Graceful exit: delete every live passage this agent owns.
    int remove_owned() { return sweep_stale(); }

    /// The door pair the live passage matched for a proto-room: {the current room's door, the entry mirror}.
    [[nodiscard]] std::optional<std::pair<std::uint64_t, std::uint64_t>> pair_for(std::uint64_t proto_room_id) const;

    /// The live passages this agent owns (metaconcept, subtype "passage", name passage_*).
    [[nodiscard]] std::vector<DSR::Node> owned() const;

private:
    std::shared_ptr<DSR::DSRGraph> G_;
    bool warned_no_current_ = false;

    /// The room a node belongs to: climb `parent` until a `room`. 0 when there is none.
    [[nodiscard]] std::uint64_t owning_room(DSR::Node n) const;
};

}   // namespace ltsm
