/*
 * common/affordance_manager/affordance_room_scope.h — AN `explore room_X` ORDER CONFINES THE CONTEST TO room_X.
 * SHARED, header-only, PURE (no DSR): the graph lookup is passed in, so the rule is pinned by a plain unit test
 * (controller/tests/test_explore_room_scope.cpp) and AffordanceManager::select_target applies it unchanged.
 *
 * ★WHY (user decision 2026-10-21, local-policy plan Task 4). A mission's `explore room_1` step opens the affordance
 *   contest on purpose (controller_session.cpp, command_explore_). Opened unscoped, it is the open contest: any
 *   producer's offer anywhere competes, so a table seen through a doorway in room_2 — or a mirror of a door in the
 *   proto-room beyond — can win on gain and drive the robot out of the room the order named, after which the step
 *   fails `outside_room`. The ORDER says which room; the contest must hear it. "Anything in room_X" is the user's
 *   wording: every kind of affordance stays eligible (room exploration, door looks, object looks), only its PLACE is
 *   restricted.
 *
 * ★A TARGET'S ROOM IS WHERE ITS OBJECT HANGS IN THE TREE, not where its standpoint lies. A standpoint for a door of
 *   room_1 may sit just inside room_2's polygon; the door still belongs to room_1. So the room is resolved by walking
 *   the structural parent chain (the `parent` attribute, which mirrors the RT tree) from the affordance up to the first
 *   node of type "room". Not by geometry, and never by a distance cutoff.
 *
 * ★UNRESOLVABLE ⇒ NOT ELIGIBLE. An affordance whose chain does not reach a room (no parent, a dangling parent, a
 *   cycle) cannot be shown to be in room_X, and the order is a restriction: it admits what it can confirm. With no
 *   explore order (scope "") nothing is filtered — the old behaviour, byte for byte.
 *
 * ★CROSS STEPS ARE UNTOUCHED. A door passage names its one entrant (AffordanceManager::set_required_name); while it
 *   does, the scope is "" (contest_room_scope) so the passage's own sequencing is all that restricts the contest.
 */
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace rc::affordance
{

/// The room the contest is confined to this cycle; "" = unscoped (no filter). Only a standing `explore` order with
/// a room scopes it, and a required entrant (door passage) always overrides — its sequencing is the restriction.
inline std::string contest_room_scope(bool explore_order, std::string_view required_name, std::string_view explore_room)
{
    if (not explore_order or not required_name.empty()) return {};
    return std::string(explore_room);
}

/// One link of the structural tree as the resolver sees it.
struct ChainNode
{
    std::string type;
    std::string name;
    std::optional<std::uint64_t> parent;   // the next link up; nullopt = a root (or no parent recorded)
};

/// Walk from `start` up the parent chain to the first node of type "room"; its NAME, or nullopt when the chain ends,
/// dangles (lookup → nullopt) or loops. `lookup(id) -> std::optional<ChainNode>`.
/// ★The depth bound is a CYCLE GUARD, not a model threshold: a well-formed tree is a handful of links deep (room →
///   wall → door → affordance), and 64 links can only mean a malformed graph, which is "unresolvable" — the same
///   answer as a dangling parent.
template <typename Lookup>
std::optional<std::string> room_of_chain(std::optional<std::uint64_t> start, Lookup &&lookup)
{
    constexpr int kCycleGuard = 64;
    std::optional<std::uint64_t> id = start;
    for (int i = 0; i < kCycleGuard and id.has_value(); ++i)
    {
        const std::optional<ChainNode> n = lookup(*id);
        if (not n.has_value()) return std::nullopt;
        if (n->type == "room") return n->name;
        id = n->parent;
    }
    return std::nullopt;
}

struct ExcludedCandidate
{
    std::string name;
    std::optional<std::string> room;   // where it resolved to; nullopt = unresolvable
};

/// Keep in `candidates` only those whose target room is `scope`; return the ones removed, in input order.
/// `scope` "" keeps everything (no order, no filter). `name_of(c) -> std::string`, `room_of(c) -> optional<string>`.
template <typename T, typename NameOf, typename RoomOf>
std::vector<ExcludedCandidate> scope_to_room(std::vector<T> &candidates, std::string_view scope, NameOf &&name_of,
                                             RoomOf &&room_of)
{
    std::vector<ExcludedCandidate> excluded;
    if (scope.empty()) return excluded;
    std::vector<T> kept;
    kept.reserve(candidates.size());
    for (auto &c : candidates)
    {
        std::optional<std::string> room = room_of(c);
        if (room.has_value() and *room == scope)
            kept.push_back(std::move(c));
        else
            excluded.push_back({name_of(c), std::move(room)});
    }
    candidates = std::move(kept);
    return excluded;
}

/// " 'aff_table_9'(room_2) 'aff_x'(unresolved)" — the [aff-select] log's account of what the order left out.
inline std::string describe_excluded(const std::vector<ExcludedCandidate> &excluded)
{
    std::string s;
    for (const auto &e : excluded)
        s += " '" + e.name + "'(" + (e.room.has_value() ? *e.room : std::string("unresolved")) + ")";
    return s;
}

}   // namespace rc::affordance
