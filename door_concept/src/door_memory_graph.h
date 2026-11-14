#pragma once
/*
 * door_memory_graph.h — write a door's existence memory (memnote::NoteAttrs) onto its door node (2026-11-13, door memory
 * through ltsm). Header-only so a test can drive it on a standalone DSRGraph without the whole agent.
 *
 * ONLY the evidence fields are written: geometry already lives in width_m/height_m/depth_m + the RT edge, and the leaf side
 * and phi are written by write_interaction_state (door_leaf_*_prob, door_phi_rad). An absent optional is NOT written (never a
 * 0 placeholder: "never looked at" is not "looked at and empty"). door_adopted belongs to the restore path, not to this one.
 * No update_node here: the caller's own update_node (the interaction state's) carries it.
 */
#include "door_memory_note.h"

#include <dsr/api/dsr_api.h>

namespace rc::door::memnote
{
inline void write_memory_attrs(DSR::DSRGraph& G, DSR::Node& node, const NoteAttrs& a)
{
    if (a.exist_evidence) G.add_or_modify_attrib_local<door_exist_evidence_att>(node, *a.exist_evidence);
    if (a.exist_physical) G.add_or_modify_attrib_local<door_exist_physical_att>(node, *a.exist_physical);
    if (a.crossed)        G.add_or_modify_attrib_local<door_crossed_att>(node, *a.crossed);
    if (a.seen_ms)        G.add_or_modify_attrib_local<door_seen_ms_att>(node, *a.seen_ms);
    if (a.sighted_ms)     G.add_or_modify_attrib_local<door_sighted_ms_att>(node, *a.sighted_ms);
    if (a.place_sd)       G.add_or_modify_attrib_local<door_place_sd_att>(node, *a.place_sd);
    if (a.height_sd)      G.add_or_modify_attrib_local<door_height_sd_att>(node, *a.height_sd);
    if (a.phi_sd)         G.add_or_modify_attrib_local<door_phi_sd_att>(node, *a.phi_sd);
    if (a.ledger)         G.add_or_modify_attrib_local<door_view_ledger_att>(node, std::string(*a.ledger));
    if (a.visit)          G.add_or_modify_attrib_local<door_visit_att>(node, *a.visit);
}

/// The fields of `now` that differ from what this door last had written (`last` null = the first write: all of them), as a NoteAttrs
/// carrying only those (the rest absent, so write_memory_attrs skips them). A field absent in `now` is never written (gone_since / erase_gone remove what was written before).
[[nodiscard]] inline NoteAttrs changed_since(const NoteAttrs& now, const NoteAttrs* last)
{
    NoteAttrs d;
    d.xy = now.xy; d.yaw = now.yaw; d.width = now.width; d.height = now.height; d.depth = now.depth;
    const auto diff = [&](auto NoteAttrs::* m) { if ((now.*m).has_value() and not (last and (last->*m) == (now.*m))) d.*m = now.*m; };
    diff(&NoteAttrs::exist_evidence); diff(&NoteAttrs::exist_physical); diff(&NoteAttrs::place_sd); diff(&NoteAttrs::height_sd);
    diff(&NoteAttrs::phi_sd); diff(&NoteAttrs::crossed); diff(&NoteAttrs::seen_ms); diff(&NoteAttrs::sighted_ms);
    diff(&NoteAttrs::ledger); diff(&NoteAttrs::visit);
    return d;
}

/// Fold the fields `d` carries into `last` (what the node now holds).
inline void apply_changes(NoteAttrs& last, const NoteAttrs& d)
{
    last.xy = d.xy; last.yaw = d.yaw; last.width = d.width; last.height = d.height; last.depth = d.depth;
    const auto put = [&](auto NoteAttrs::* m) { if ((d.*m).has_value()) last.*m = d.*m; };
    put(&NoteAttrs::exist_evidence); put(&NoteAttrs::exist_physical); put(&NoteAttrs::place_sd); put(&NoteAttrs::height_sd);
    put(&NoteAttrs::phi_sd); put(&NoteAttrs::crossed); put(&NoteAttrs::seen_ms); put(&NoteAttrs::sighted_ms);
    put(&NoteAttrs::ledger); put(&NoteAttrs::visit);
}

/// Fields that WERE written (present in `last`) and are absent in `now`: their attribute on the node is stale and must go (a leaf
/// lost, a ledger emptied). `ledger_known` false = the ledger was not re-serialised this cycle (absent means "unchanged", not "gone").
struct Gone
{
    bool exist_evidence = false, exist_physical = false, place_sd = false, height_sd = false, phi_sd = false, crossed = false,
         seen_ms = false, sighted_ms = false, ledger = false, visit = false;
    [[nodiscard]] bool any() const
    {
        return exist_evidence or exist_physical or place_sd or height_sd or phi_sd or crossed or seen_ms or sighted_ms or ledger or visit;
    }
};

[[nodiscard]] inline Gone gone_since(const NoteAttrs& now, const NoteAttrs* last, bool ledger_known)
{
    Gone g;
    if (not last) return g;
    const auto gone = [&](auto NoteAttrs::* m) { return (last->*m).has_value() and not (now.*m).has_value(); };
    g.exist_evidence = gone(&NoteAttrs::exist_evidence); g.exist_physical = gone(&NoteAttrs::exist_physical);
    g.place_sd = gone(&NoteAttrs::place_sd); g.height_sd = gone(&NoteAttrs::height_sd); g.phi_sd = gone(&NoteAttrs::phi_sd);
    g.crossed = gone(&NoteAttrs::crossed); g.seen_ms = gone(&NoteAttrs::seen_ms); g.sighted_ms = gone(&NoteAttrs::sighted_ms);
    g.ledger = ledger_known and gone(&NoteAttrs::ledger); g.visit = gone(&NoteAttrs::visit);
    return g;
}

/// Remove the attributes `g` names from the node (type-attributed), and forget them in `last`.
inline void erase_gone(DSR::DSRGraph& G, DSR::Node& node, NoteAttrs& last, const Gone& g)
{
    if (g.exist_evidence) { G.remove_attrib_local<door_exist_evidence_att>(node);  last.exist_evidence.reset(); }
    if (g.exist_physical) { G.remove_attrib_local<door_exist_physical_att>(node);  last.exist_physical.reset(); }
    if (g.place_sd)       { G.remove_attrib_local<door_place_sd_att>(node);        last.place_sd.reset(); }
    if (g.height_sd)      { G.remove_attrib_local<door_height_sd_att>(node);       last.height_sd.reset(); }
    if (g.phi_sd)         { G.remove_attrib_local<door_phi_sd_att>(node);          last.phi_sd.reset(); }
    if (g.crossed)        { G.remove_attrib_local<door_crossed_att>(node);         last.crossed.reset(); }
    if (g.seen_ms)        { G.remove_attrib_local<door_seen_ms_att>(node);         last.seen_ms.reset(); }
    if (g.sighted_ms)     { G.remove_attrib_local<door_sighted_ms_att>(node);      last.sighted_ms.reset(); }
    if (g.ledger)         { G.remove_attrib_local<door_view_ledger_att>(node);     last.ledger.reset(); }
    if (g.visit)          { G.remove_attrib_local<door_visit_att>(node);           last.visit.reset(); }
}

/// True when `d` (from changed_since) carries any evidence field to write.
[[nodiscard]] inline bool any_change(const NoteAttrs& d)
{
    return d.exist_evidence or d.exist_physical or d.place_sd or d.height_sd or d.phi_sd or d.crossed or d.seen_ms or d.sighted_ms
           or d.ledger or d.visit;
}
}  // namespace rc::door::memnote
