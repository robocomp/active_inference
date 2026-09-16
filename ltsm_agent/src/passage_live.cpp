/*
 * passage_live.cpp — see passage_live.h.
 */
#include "passage_live.h"

#include "room_eviction.h"                                   // ltsm::is_door
#include "../../common/room_resolve/room_resolve.h"          // rc::room::current_room / is_proto

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

std::optional<std::uint64_t> index_of(const std::string &name)
{
    if (not name.starts_with(kPrefix)) return {};
    std::uint64_t v = 0;
    const auto *b = name.data() + kPrefix.size();
    // from_chars, never stoull: es_ES locale (CLAUDE.md).
    if (const auto [ptr, ec] = std::from_chars(b, name.data() + name.size(), v);
        ec == std::errc{} and ptr == name.data() + name.size())
        return v;
    return {};
}
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

    std::vector<std::uint64_t> proto_rooms;
    for (const auto &r : G_->get_nodes_by_type("room"))
        if (rc::room::is_proto(*G_, r.id())) proto_rooms.push_back(r.id());

    const auto cur = rc::room::current_room(*G_);
    const auto cur_node = cur.has_value() ? G_->get_node(cur.value()) : std::optional<DSR::Node>{};

    auto ie = G_->get_inner_eigen_api();
    for (const auto proto : proto_rooms)
        for (const auto &entry : doors)
        {
            const auto parent = G_->get_attrib_by_name<parent_att>(entry);
            if (not parent.has_value() or parent.value() != proto) continue;

            // nullopt = nobody has said which room we are in -- never guess a counterpart.
            if (not cur_node.has_value())
            {
                if (not std::exchange(warned_no_current_, true))
                    qWarning() << "[passage] entry door" << QString::fromStdString(entry.name())
                               << "has no counterpart yet: the current room is unresolved";
                continue;
            }
            warned_no_current_ = false;

            const auto entry_in_cur = ie->get_transformation_matrix(cur_node->name(), entry.name());
            if (not entry_in_cur.has_value()) continue;

            // COUNTERPART = argmin distance over the current room's own doors (not mirrors).
            std::optional<DSR::Node> best;
            double best_d = std::numeric_limits<double>::max();
            for (const auto &d : doors)
            {
                if (d.id() == entry.id() or rc::room::is_proto_mirror(*G_, d)) continue;
                if (owning_room(d) != cur.value()) continue;
                const auto d_in_cur = ie->get_transformation_matrix(cur_node->name(), d.name());
                if (not d_in_cur.has_value()) continue;
                if (const double dist = (d_in_cur->translation() - entry_in_cur->translation()).norm(); dist < best_d)
                {
                    best_d = dist;
                    best = d;
                }
            }
            if (not best.has_value()) continue;

            // The passage this entry already belongs to, if any.
            std::optional<DSR::Node> passage;
            for (const auto &p : owned())
                if (G_->get_edge(p.id(), entry.id(), "match").has_value()) { passage = p; break; }

            if (passage.has_value())
            {
                std::uint64_t other = 0;
                for (const auto &e : DSR::DSRGraph::get_node_edges_by_type(passage.value(), "match"))
                    if (e.to() != entry.id()) other = e.to();
                if (other == best->id()) continue;
                if (other != 0) G_->delete_edge(passage->id(), other, "match");
                auto m = DSR::Edge::create<match_edge_type>(passage->id(), best->id());
                G_->insert_or_assign_edge(std::move(m));
                ++st.rematched;
                qInfo().noquote() << QString::fromStdString(std::format(
                    "[passage] {} rematch {} ↔ {} (was {})", passage->name(), best->name(), entry.name(),
                    other != 0 ? door_name(other) : std::string("<none>")));
                continue;
            }

            std::uint64_t n = 0;
            for (const auto &p : owned())
                if (const auto i = index_of(p.name()); i.has_value()) n = std::max(n, i.value() + 1);

            // TYPE `metaconcept`, SUBTYPE "passage", NO RT EDGE: a relation between two doors, not a body.
            // Never type `door` -- door_concept's owned-node sweep deletes `door`-typed nodes by name.
            DSR::Node node;
            node.type("metaconcept");
            node.name(std::format("passage_{}", n));
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
                "[passage] {} match {} ↔ {}  ({:.2f} m apart in {})",
                node.name(), best->name(), entry.name(), best_d, cur_node->name()));
        }

    st.passages = static_cast<int>(owned().size());
    return st;
}

}   // namespace ltsm
