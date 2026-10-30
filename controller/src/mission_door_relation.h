#pragma once
// DoorRelation — pick a door by what it LEADS TO, not by its name (door-identity-design.md, part 3).
//
// A `door_N` name is a graph slot, not an identity: door nodes are re-made on every crossing. What survives is the relation:
//   cross <room>   the live door of the current room that has a `recall_<room>` child (ltsm posts it under the door that
//                  leads to that remembered room);
//   cross new      a live door with NO recall child and NO passage `match` edge: a door nothing has been learnt about yet.
//                  Several: the one a door_concept look has targeted (aff_door_look_<name> exists or existed), else the
//                  nearest to the robot (ties by name, so the pick is deterministic).
// Resolution is LAZY: the recall (or the new door) may appear late, so the step holds (RelationHold) and re-resolves each
// poll. Pure: the controller gathers DoorFacts from the graph; no graph, no Qt here.
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "../../common/affordance_manager/affordance_room_scope.h"
#include "../../common/mission/mission_ir.h"

namespace rc
{
struct DoorFacts
{
    std::string name{};
    std::set<std::string> recall_rooms{};   ///< rooms R for which the door has a child `recall_R`
    bool has_match = false;                 ///< a passage `match` edge points at the door
    bool crossed = false;                   ///< a crossing fact names it (exit_fact::is_crossed_face): never `new`
    bool looked = false;                    ///< an aff_door_look_<name> node exists or existed
    double dist = std::numeric_limits<double>::infinity();   ///< metres from the robot (inf = unknown)
};

enum class DoorRelationKind { None, Room, New };

/// A Node-ref name that is a relation rather than a door: "new", or a room ("room_2").
[[nodiscard]] inline DoorRelationKind relation_of(const mission::Ref &r)
{
    if (r.kind != mission::Ref::Kind::Node) return DoorRelationKind::None;
    if (r.name == "new") return DoorRelationKind::New;
    if (r.name.starts_with("room")) return DoorRelationKind::Room;
    return DoorRelationKind::None;
}

/// The door the relation names among the live doors of the current room, if any resolves now.
[[nodiscard]] inline std::optional<std::string> pick_door(const mission::Ref &r, const std::vector<DoorFacts> &doors)
{
    const auto kind = relation_of(r);
    const DoorFacts *best = nullptr;
    for (const auto &d : doors)
    {
        if (kind == DoorRelationKind::Room)
        {
            if (not d.recall_rooms.contains(r.name)) continue;
            if (best == nullptr or d.name < best->name) best = &d;
        }
        else if (kind == DoorRelationKind::New)
        {
            if (not d.recall_rooms.empty() or d.has_match or d.crossed) continue;
            const bool better = best == nullptr or (d.looked != best->looked ? d.looked
                                                    : d.dist != best->dist   ? d.dist < best->dist
                                                                             : d.name < best->name);
            if (better) best = &d;
        }
    }
    return best != nullptr ? std::optional<std::string>(best->name) : std::nullopt;
}

enum class RoomEvent { None, Renamed, Changed };

/// The room the robot is in, tracked by node id. The id says WHICH room; the name is owned by room_concept and can change
/// under the same id (a recognised room adopts its remembered name in place: room_4 -> room_2). A rename is not a room
/// change.
inline RoomEvent track_room(std::uint64_t &id, std::string &name, std::uint64_t new_id, const std::string &new_name)
{
    if (new_id != id)
    {
        id = new_id; name = new_name;
        return RoomEvent::Changed;
    }
    if (name != new_name) { name = new_name; return RoomEvent::Renamed; }
    return RoomEvent::None;
}

/// Every name a room node has worn, by node id: a mission step names a room by the NAME it was written with ("room_4") and
/// that name can be renamed away under the same id (room_4 -> room_2). The id is the identity; this maps an old name to it.
using RoomAliases = std::map<std::string, std::uint64_t>;

/// The id a mission's room NAME stands for: the live graph node of that name if there is one (a name in use is its own
/// node's — an alias could be stale if the name was reused), else the node that used to wear it.
[[nodiscard]] inline std::optional<std::uint64_t> room_id_named(const RoomAliases &aliases, std::optional<std::uint64_t> live_id,
                                                               const std::string &name)
{
    if (live_id.has_value()) return live_id;
    if (const auto it = aliases.find(name); it != aliases.end()) return it->second;
    return std::nullopt;
}

/// Is the robot (room `cur_id` / `cur_name`) in the room the mission calls `name`? By id; the name only when the id is unknown.
[[nodiscard]] inline bool robot_in_room(std::uint64_t cur_id, const std::string &cur_name, const RoomAliases &aliases,
                                        std::optional<std::uint64_t> live_id, const std::string &name)
{
    if (cur_id == 0) return not cur_name.empty() and cur_name == name;
    if (const auto id = room_id_named(aliases, live_id, name); id.has_value()) return *id == cur_id;
    return cur_name == name;
}

/// `cross <room>`: ltsm names a door's recall child by the room's name AT THE TIME; the mission may use another name of the
/// same room (renamed since). `ids` = every name known -> node id (aliases + live rooms). A door whose recall names the
/// same id as `room` gets `room` added to its recall set, so pick_door matches it.
inline void expand_recalls(std::vector<DoorFacts> &doors, const RoomAliases &ids, const std::string &room)
{
    const auto it = ids.find(room);
    if (it == ids.end()) return;
    for (auto &d : doors)
        for (const auto &r : std::set<std::string>(d.recall_rooms))
            if (const auto j = ids.find(r); j != ids.end() and j->second == it->second) d.recall_rooms.insert(room);
}

/// The explore step's per-poll decision, by room ID (the name is no witness: a recognised room is renamed in place).
enum class ExploreRoom { Inside, Renamed, Outside };
[[nodiscard]] inline ExploreRoom explore_room_verdict(std::uint64_t step_id, std::uint64_t now_id, const std::string &step_name,
                                                      const std::string &now_name)
{
    if (now_id == 0) return ExploreRoom::Inside;                 // unresolved: not "left"
    if (now_id != step_id) return ExploreRoom::Outside;
    return (not now_name.empty() and now_name != step_name) ? ExploreRoom::Renamed : ExploreRoom::Inside;
}

/// A same-room rename of an explore's scope: swap ONLY the names. The scope CHANGE path of the session (a new room entering
/// the contest) releases the claim, lock-on and plan; a rename is the same room, so none of that may happen. Swaps only a
/// scope that is the old name (an empty scope — door passage standing — stays empty). True when swapped.
inline bool rename_scope(std::string &explore_room, std::string &manager_scope, const std::string &new_name)
{
    if (explore_room.empty() or manager_scope != explore_room) return false;
    explore_room = manager_scope = new_name;
    return true;
}

/// The session's per-cycle scope step, pure so a test can drive it with an executing claim. Applies a queued same-room rename
/// (names only), then compares the wanted scope with the manager's. `release` = the contest scope CHANGED to a non-empty room:
/// the caller must release the standing claim, lock-on and plan. A rename never sets it.
struct ScopeCycle { std::string explore_room, manager_scope, pending_rename; };
struct ScopeOutcome { bool changed = false; bool release = false; std::string scope; std::string manager_scope; /* what the manager MUST hold after this cycle */ };
[[nodiscard]] inline ScopeOutcome scope_cycle(ScopeCycle &s, bool command_explore, std::string_view door_required)
{
    if (not s.pending_rename.empty())
    {
        if (not rename_scope(s.explore_room, s.manager_scope, s.pending_rename) and not s.explore_room.empty())
            s.explore_room = s.pending_rename;
        s.pending_rename.clear();
    }
    ScopeOutcome o{.scope = affordance::contest_room_scope(command_explore, door_required, s.explore_room)};
    o.changed = o.scope != s.manager_scope;
    o.release = o.changed and not o.scope.empty();
    if (o.changed) s.manager_scope = o.scope;
    o.manager_scope = s.manager_scope;
    return o;
}

/// The step waits for its door to resolve; it has a budget so it cannot hang (like RoomHold).
class RelationHold
{
public:
    /// ⚑ FLAGGED CONSTANT: a MISSION-LEVEL timeout, not an ordering guess. The release is the FACT (pick_door resolves: ltsm
    /// marked recall_<room>, or a new door was seen); this only bounds how long the mission waits for a fact that may never
    /// come (`cross new` is gated on ltsm posting the way-back recall; ltsm down, door never seen) so the step fails as 'door_unresolved'. The step's own budget_s wins when smaller.
    static constexpr std::uint64_t kHoldBudgetMs = 120'000;

    enum class Verdict { Waiting, Ready, TimedOut };

    void hold(const mission::Step &s, std::uint64_t now_ms) { pending_ = s; since_ = now_ms; }
    bool holding() const { return pending_.has_value(); }
    void clear() { pending_.reset(); }

    /// `door` = what pick_door gives now. Ready releases the hold.
    Verdict check(const std::optional<std::string> &door, std::uint64_t now_ms)
    {
        if (door.has_value()) { pending_.reset(); return Verdict::Ready; }
        const double b = pending_.has_value() ? pending_->number("budget_s").value_or(0.0) : 0.0;
        const std::uint64_t budget = b > 0 ? std::min<std::uint64_t>(kHoldBudgetMs, std::uint64_t(b * 1000.0)) : kHoldBudgetMs;
        if (now_ms - since_ >= budget) { pending_.reset(); return Verdict::TimedOut; }
        return Verdict::Waiting;
    }

private:
    std::optional<mission::Step> pending_;
    std::uint64_t since_ = 0;
};
}   // namespace rc
