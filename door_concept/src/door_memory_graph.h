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
    if (a.visit)                   G.add_or_modify_attrib_local<door_visit_att>(node, *a.visit);
}
}  // namespace rc::door::memnote
