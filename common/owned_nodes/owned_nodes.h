/*
 * common/owned_nodes/owned_nodes.h — the OWNERSHIP half of the agent lifecycle contract. SHARED.
 *
 * Every concept agent owns two families of DSR node: its instances ("<prefix>_*") and their affordance
 * children ("aff_<prefix>*"). Both must be reaped on a graceful shutdown AND on the startup stale-sweep that
 * recovers from a crashed or SIGKILLed previous run (see CLAUDE.md — a kill -9 leaks every owned node).
 * This module is that sweep, once, for all of them. Main-thread only (graph access).
 *
 * WHY SHARED — measured 2026-08-14 across the eight agents. The sweep was eight hand-written copies and they
 * had already diverged in ways that leak nodes:
 *
 *   - chair, door and human had NO "aff_<prefix>" NAME test — only the parent-type test. That branch is the
 *     ORPHAN-SAFE one: it is what catches an affordance whose parent object node was already deleted
 *     (retirement, or a crashed run reaped in the wrong order). Without it those three leak an affordance
 *     node with no parent, and nothing else in the fleet ever deletes it.
 *   - the LEGACY parent/node types kept for the pre-"object" schema migration were per-agent folklore:
 *     bottle swept {object, cylinder}, chair {object, chair}, door {object, door}, table {object, table},
 *     cabinet {object, box} — while hood and refrigerator swept "object" alone.
 *
 * Per the UNION rule, the shared sweep does BOTH tests for everyone and the agent declares only what is
 * genuinely its own: its name prefix and which legacy types its own history left behind.
 *
 * NOT NARROWED: a match on a DEDICATED parent type (chair/door/person/cylinder/…) still counts on its own,
 * exactly as those agents' hand-written versions did. The name-prefix filter is applied only to the GENERIC
 * "object" type, where the name is the only class discriminator. So no agent loses reach — three gain it.
 */

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <QDebug>
#include <dsr/api/dsr_api.h>

namespace rc::owned
{

// What this agent owns. `name_prefix` is the ONE thing that identifies the agent's nodes in the shared graph;
// everything else is either derived from it or a declaration about this agent's own history.
struct Spec
{
    std::string              agent;                // log label, e.g. "refrigerator_concept"
    std::string              name_prefix;          // e.g. "refrigerator" — every owned node name starts with it
    std::vector<std::string> node_types;           // instance node types to sweep, e.g. {"object", "cylinder"}
    std::vector<std::string> legacy_parent_types;  // pre-migration parent types that identify an affordance as
                                                   // ours on their OWN (no name filter), e.g. {"cylinder"}
    std::string              affordance_prefix;    // empty ⇒ "aff_" + name_prefix

    std::string aff_prefix() const
    { return affordance_prefix.empty() ? ("aff_" + name_prefix) : affordance_prefix; }
};

// Delete every instance node this agent owns: any node of a declared type whose name carries the prefix.
// Deleting the node drops its room→object RT edge too, so no separate edge cleanup is needed.
// Returns the number deleted.
inline int remove_instance_nodes(DSR::DSRGraph& G, const Spec& s)
{
    std::vector<std::uint64_t> to_delete;
    for (const auto& type : s.node_types)
        for (const auto& node : G.get_nodes_by_type(type))
            if (node.name().starts_with(s.name_prefix))
                to_delete.push_back(node.id());

    int n = 0;
    for (const auto id : to_delete)
        if (G.delete_node(id))
        {
            ++n;
            qInfo().noquote() << QString::fromStdString("[" + s.agent + "] removed instance node id="
                                                        + std::to_string(id));
        }
    return n;
}

// Delete every "affordance" node belonging to this agent — including stale ones left by a crashed previous run,
// whatever their (possibly DSR-collision-renamed) node name. Two independent tests, EITHER of which claims it:
//
//   (a) the deterministic "aff_<prefix>*" NAME — orphan-safe: it still fires after the parent object node is
//       gone, which is precisely the case a parent-type lookup cannot see.
//   (b) the parent: a legacy dedicated type on its own, or the generic "object" type carrying our name prefix.
//
// Returns the number deleted.
inline int remove_stale_affordances(DSR::DSRGraph& G, const Spec& s)
{
    const std::string aff_pfx = s.aff_prefix();
    int n = 0;
    for (const auto& aff : G.get_nodes_by_type("affordance"))
    {
        bool ours = aff.name().starts_with(aff_pfx);
        std::string_view why = "name";
        if (not ours)
            if (const auto pid = G.get_attrib_by_name<parent_att>(aff); pid.has_value())
                if (const auto parent = G.get_node(pid.value()); parent.has_value())
                {
                    // A dedicated legacy type identifies the parent on its own — the name is not the
                    // discriminator there, and those agents' hand-written versions did not filter on it.
                    for (const auto& lt : s.legacy_parent_types)
                        if (parent->type() == lt) { ours = true; why = "legacy parent type"; break; }
                    // Otherwise the parent must be one of OUR instance types AND carry our name prefix —
                    // under the generic "object" type the name is the only class discriminator.
                    if (not ours)
                        for (const auto& t : s.node_types)
                            if (parent->type() == t and parent->name().starts_with(s.name_prefix))
                            { ours = true; why = "parent instance node"; break; }
                }
        if (ours)
        {
            qInfo().noquote() << QString::fromStdString("[" + s.agent + "] removing affordance node "
                                                        + aff.name() + " id " + std::to_string(aff.id())
                                                        + " (" + std::string(why) + ")");
            G.delete_node(aff.id());
            ++n;
        }
    }
    return n;
}

}  // namespace rc::owned
