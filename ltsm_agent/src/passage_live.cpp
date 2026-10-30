/*
 * passage_live.cpp — see passage_live.h.
 */
#include "passage_live.h"

#include "room_eviction.h"                                   // ltsm::is_door
#include "../../common/room_resolve/room_resolve.h"          // rc::room::current_room / is_proto
#include "../../common/room_transition/exit_fact.h"           // rc::exit_fact::twin_of (link 3 = the passage)

#include <dsr/api/dsr_inner_eigen_api.h>

#include <QDebug>

#include <algorithm>
#include <charconv>
#include <format>
#include <limits>
#include <optional>

namespace ltsm
{

namespace
{
constexpr std::string_view kPrefix = "passage_";
}   // namespace

PassageLive::PassageLive(std::shared_ptr<DSR::DSRGraph> live) : G_(std::move(live)) {}

std::vector<DSR::Node> PassageLive::owned() const
{
    std::vector<DSR::Node> out;
    for (const auto &n : G_->get_nodes_by_type("metaconcept"))
        if (const auto st = G_->get_attrib_by_name<object_subtype_att>(n);
            st.has_value() and st.value() == "passage" and n.name().starts_with(kPrefix))
            out.push_back(n);
    return out;
}

std::uint64_t PassageLive::owning_room(DSR::Node n) const
{
    // Bounded by the depth of a real RT tree; a cycle in `parent` attributes must not hang the agent.
    for (int hops = 0; hops < 32; ++hops)
    {
        if (n.type() == "room") return n.id();
        const auto p = G_->get_attrib_by_name<parent_att>(n);
        if (not p.has_value()) return 0;
        const auto up = G_->get_node(p.value());
        if (not up.has_value()) return 0;
        n = up.value();
    }
    return 0;
}

std::optional<std::pair<std::uint64_t, std::uint64_t>> PassageLive::pair_for(std::uint64_t proto_room_id) const
{
    for (const auto &p : owned())
    {
        const auto m = DSR::DSRGraph::get_node_edges_by_type(p, "match");
        if (m.size() != 2) continue;
        for (int i = 0; i < 2; ++i)
            if (const auto entry = G_->get_node(m[i].to()); entry.has_value())
                if (const auto par = G_->get_attrib_by_name<parent_att>(entry.value());
                    par.has_value() and par.value() == proto_room_id)
                    return std::pair{m[1 - i].to(), m[i].to()};
    }
    return {};
}

std::optional<DSR::Node> PassageLive::passage_of(std::uint64_t door_id) const
{
    for (const auto &p : owned())
        if (G_->get_edge(p.id(), door_id, "match").has_value()) return p;
    return {};
}

int PassageLive::sweep_stale()
{
    int removed = 0;
    for (const auto &p : owned())
        if (G_->delete_node(p.id()))
        {
            ++removed;
            qInfo() << "[passage] removed" << QString::fromStdString(p.name());
        }
    return removed;
}

PassageLive::Step PassageLive::step()
{
    Step st;

    // ── 1. RETIRE: a passage lives exactly as long as both of its doors ────────────────────────
    const auto door_name = [this](std::uint64_t id)
    {
        const auto n = G_->get_node(id);
        return n.has_value() ? n->name() : std::string("<gone>");
    };
    for (const auto &p : owned())
    {
        const auto matches = DSR::DSRGraph::get_node_edges_by_type(p, "match");
        const bool whole = matches.size() == 2
                           and std::ranges::all_of(matches, [&](const auto &e) { return G_->get_node(e.to()).has_value(); });
        if (whole) continue;
        if (G_->delete_node(p.id()))
        {
            ++st.deleted;
            qInfo() << "[passage]" << QString::fromStdString(p.name())
                    << "deleted -- one of its doors is gone";
        }
    }

    // ── 2. ENSURE one passage per entry mirror ──────────────────────────────────────────────────
    std::vector<DSR::Node> doors;
    for (const auto &n : G_->get_nodes_by_type("object"))
        if (is_door(*G_, n)) doors.push_back(n);

    // ★ THE PAIR IS THE CROSSING FACT, NOT THE NEAREST DOOR (spec 2026-10-29 Rev 3, link 3): door_concept writes `exit`
    //   NEW_FACE -> OLD_FACE when it makes the face of the hole crossed in the room entered. That edge IS the passage: the
    //   two faces of one hole. (The argmin over the current room's doors it replaces joined two holes of one room —
    //   recog_proto8 — and needed a guard per case.)
    for (const auto &entry : doors)
    {
        const auto old_id = rc::exit_fact::twin_of(*G_, entry.id());
        if (not old_id.has_value()) continue;
        const auto old = G_->get_node(*old_id);
        if (not old.has_value()) continue;
        const std::optional<DSR::Node> best = old;
        const std::string where = owning_room(entry) != 0
            ? G_->get_node(owning_room(entry)).transform([](const auto &n) { return n.name(); }).value_or("?") : "?";
        {
            // The passage this entry already belongs to, if any.
            std::optional<DSR::Node> passage;
            for (const auto &p : owned())
                if (G_->get_edge(p.id(), entry.id(), "match").has_value()) { passage = p; break; }

            // ★ A PAIRING IS KEPT ONCE MADE (2026-10-11). It is made right at the crossing, when the robot is in the
            //   doorway and the two faces are measured through a fresh pose. Later the only route from the old room's
            //   doors to the new room runs through the robot's RETIRED RT edge (frozen at the crossing), so distances
            //   drift with every metre driven, and once `current` moves to the new room the "current room's doors" are
            //   the new room's own — a re-match would then pick a duplicate of the entry instead of its counterpart.
            //   The passage ends only when one of its doors is gone (step 1).
            if (passage.has_value()) continue;

            // THE NAME IS THE HOLE'S IDENTITY: memory's passage for this hole when it has one (by place, the current
            // room's door first, then the entry mirror), else the one shared numbering over live AND memory.
            std::string pname;
            std::string why = "new hole";
            if (ev_ != nullptr)
                for (const auto &d : {best.value(), entry})
                    if (const auto h = ev_->remembered_hole(d); h.has_value())
                    {
                        if (G_->get_node(h->passage.name()).has_value())
                        {
                            // two live entries resolved to one remembered hole: the name is taken live, so this one
                            // cannot carry it (DSR names are unique). Logged, numbered fresh — never two holes on one name.
                            qWarning().noquote() << QString::fromStdString(std::format(
                                "[passage] ⚠ {} resolves to remembered {} which is already live — numbered fresh",
                                d.name(), h->passage.name()));
                            break;
                        }
                        pname = h->passage.name();
                        why = std::format("remembered: {} at {:.2f} m from face {}", d.name(), h->distance, h->face.name());
                        break;
                    }
            if (pname.empty())
                pname = std::format("passage_{}", ev_ != nullptr ? ev_->next_passage_index() : next_passage_index(*G_, nullptr));

            // TYPE `metaconcept`, SUBTYPE "passage", NO RT EDGE: a relation between two doors, not a body.
            // Never type `door` -- door_concept's owned-node sweep deletes `door`-typed nodes by name.
            DSR::Node node;
            node.type("metaconcept");
            node.name(pname);
            G_->add_or_modify_attrib_local<object_subtype_att>(node, std::string("passage"));
            G_->add_or_modify_attrib_local<level_att>(node, 0);
            // Viewer placement only: between the two doors it links.
            const auto px = [&](const DSR::Node &d) { return G_->get_attrib_by_name<pos_x_att>(d).value_or(0.f); };
            const auto py = [&](const DSR::Node &d) { return G_->get_attrib_by_name<pos_y_att>(d).value_or(0.f); };
            G_->add_or_modify_attrib_local<pos_x_att>(node, 0.5f * (px(entry) + px(best.value())));
            G_->add_or_modify_attrib_local<pos_y_att>(node, 0.5f * (py(entry) + py(best.value())) + 60.f);

            // LVALUE insert_node: the rvalue overload is not instantiated in libdsr (README).
            const auto id = G_->insert_node(node);
            if (not id.has_value()) continue;
            auto to_cur   = DSR::Edge::create<match_edge_type>(id.value(), best->id());
            auto to_entry = DSR::Edge::create<match_edge_type>(id.value(), entry.id());
            G_->insert_or_assign_edge(std::move(to_cur));
            G_->insert_or_assign_edge(std::move(to_entry));
            ++st.created;
            qInfo().noquote() << QString::fromStdString(std::format(
                "[passage] {} match {} ↔ {}  (link 3: {} entered through {}; {})",
                node.name(), best->name(), entry.name(), where, best->name(), why));
        }
    }

    st.passages = static_cast<int>(owned().size());
    return st;
}

}   // namespace ltsm
