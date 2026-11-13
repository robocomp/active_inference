#pragma once
/*
 * door_note_reader.h — read the `remembered_door` notes ltsm posts for the robot's current room (2026-11-13, door memory
 * through ltsm). Header-only on DSR so a test can drive it on a standalone DSRGraph.
 *
 * A note is a `metaconcept` node, object_subtype "remembered_door", parent = the live room node id, with the door's memory
 * attributes (the ones door_memory_graph.h writes) + width_m/height_m/depth_m, and an RT edge room -> note whose translation xy
 * and rotation_euler z are the door's pose in the room's own frame. The pose is read off the EDGE ATTRIBUTES (a ts==0 transform
 * would go through a cache an RT written this very cycle has not invalidated). A note not yet readable (no RT edge yet, no
 * geometry) is skipped WITHOUT being adopted: it may land completely on a later cycle. A note with door_adopted == true is
 * skipped (already restored; ltsm retires it). door_concept never deletes a note.
 */
#include "door_memory_note.h"

#include <dsr/api/dsr_api.h>

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace rc::door::memnote
{
struct NoteInput
{
    std::uint64_t id = 0;
    std::string   name;      ///< "remembered_door_5"
    NoteAttrs     attrs;
};

namespace detail
{
template <class Att>
[[nodiscard]] inline std::optional<std::string> str_attr(DSR::DSRGraph& G, const DSR::Node& n)
{
    if (const auto v = G.get_attrib_by_name<Att>(n); v.has_value()) return std::string(v.value().get());
    return std::nullopt;
}
}  // namespace detail

/// The unadopted, fully readable notes hanging from `room_id`.
[[nodiscard]] inline std::vector<NoteInput> read_notes(DSR::DSRGraph& G, std::uint64_t room_id)
{
    std::vector<NoteInput> out;
    for (const auto& n : G.get_nodes_by_type("metaconcept"))
    {
        if (G.get_attrib_by_name<object_subtype_att>(n).value_or(std::string{}) != "remembered_door") continue;
        if (G.get_attrib_by_name<parent_att>(n).value_or(0) != room_id) continue;
        if (G.get_attrib_by_name<door_adopted_att>(n).value_or(false)) continue;

        const auto w = G.get_attrib_by_name<width_m_att>(n);
        const auto h = G.get_attrib_by_name<height_m_att>(n);
        const auto d = G.get_attrib_by_name<depth_m_att>(n);
        const auto e = G.get_edge(room_id, n.id(), "RT");
        if (not (w.has_value() and h.has_value() and d.has_value() and e.has_value())) continue;
        const auto t = G.get_attrib_by_name<rt_translation_att>(e.value());
        const auto r = G.get_attrib_by_name<rt_rotation_euler_xyz_att>(e.value());
        if (not (t.has_value() and r.has_value() and t.value().get().size() >= 3 and r.value().get().size() >= 3)) continue;

        NoteInput in;
        in.id = n.id();
        in.name = n.name();
        NoteAttrs& a = in.attrs;
        a.xy = {t.value().get()[0], t.value().get()[1]};
        a.yaw = r.value().get()[2];
        a.width = w.value();  a.height = h.value();  a.depth = d.value();
        a.exist_evidence = G.get_attrib_by_name<door_exist_evidence_att>(n);
        a.exist_physical = G.get_attrib_by_name<door_exist_physical_att>(n);
        a.place_sd = G.get_attrib_by_name<door_place_sd_att>(n);
        a.height_sd = G.get_attrib_by_name<door_height_sd_att>(n);
        a.phi = G.get_attrib_by_name<door_phi_rad_att>(n);
        if (a.phi.has_value() and *a.phi < 0.0f) a.phi.reset();   // the published sentinel -1 = NOT MEASURED (write_interaction_state): no leaf memory
        a.phi_sd = G.get_attrib_by_name<door_phi_sd_att>(n);
        a.leaf_into_room_prob = G.get_attrib_by_name<door_leaf_into_room_prob_att>(n);
        a.leaf_hinge_plus_prob = G.get_attrib_by_name<door_leaf_hinge_plus_prob_att>(n);
        a.crossed = G.get_attrib_by_name<door_crossed_att>(n);
        a.seen_ms = G.get_attrib_by_name<door_seen_ms_att>(n);
        a.sighted_ms = G.get_attrib_by_name<door_sighted_ms_att>(n);
        a.ledger = detail::str_attr<door_view_ledger_att>(G, n);
        a.visit = G.get_attrib_by_name<door_visit_att>(n);
        out.push_back(std::move(in));
    }
    return out;
}

/// The identity rows the notes stand for (note "remembered_door_5" -> row "door_5" of `room`).
[[nodiscard]] inline std::vector<IdentityRow> rows_of_notes(std::span<const NoteInput> notes, const std::string& room)
{
    std::vector<IdentityRow> rows;
    rows.reserve(notes.size());
    for (const auto& n : notes) rows.push_back(row_of(n.attrs, room, face_live_name(n.name)));
    return rows;
}

/// door_concept's side of the handshake: the note was restored (or its place is live): ltsm deletes it.
inline bool mark_adopted(DSR::DSRGraph& G, std::uint64_t note_id)
{
    auto n = G.get_node(note_id);
    if (not n.has_value()) return false;
    G.add_or_modify_attrib_local<door_adopted_att>(n.value(), true);
    return G.update_node(n.value());
}
}  // namespace rc::door::memnote
