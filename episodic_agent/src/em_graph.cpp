#include "em_graph.h"

#include <cmath>
#include <numbers>

#include <QJsonDocument>

namespace rc::em
{
namespace
{
std::string compact(const QJsonObject& o) { return QJsonDocument(o).toJson(QJsonDocument::Compact).toStdString(); }
}   // namespace

EmGraph::EmGraph(std::shared_ptr<DSR::DSRGraph> em) : G_(std::move(em)) {}

std::string EmGraph::stats_name(const Stats& st)
{
    return "stats_" + st.key_kind + "_" + (st.key_identity.empty() ? std::string("any") : st.key_identity);
}

std::optional<DSR::Node> EmGraph::ensure_node(const std::string& name, std::uint64_t parent_id)
{
    if (auto n = G_->get_node(name); n.has_value()) return n;
    const auto parent = G_->get_node(parent_id);
    if (not parent) return std::nullopt;
    auto n = DSR::Node::create<metaconcept_node_type>(name);
    G_->add_or_modify_attrib_local<parent_att>(n, parent_id);
    G_->add_or_modify_attrib_local<level_att>(n, G_->get_attrib_by_name<level_att>(*parent).value_or(0) + 1);
    // Viewer placement only: a growing spiral around the parent, so nodes do not pile on one spot.
    const double a = 0.6 * placed_, r = 60.0 + 4.0 * placed_;
    ++placed_;
    G_->add_or_modify_attrib_local<pos_x_att>(n, G_->get_attrib_by_name<pos_x_att>(*parent).value_or(0.f) + float(r * std::cos(a)));
    G_->add_or_modify_attrib_local<pos_y_att>(n, G_->get_attrib_by_name<pos_y_att>(*parent).value_or(0.f) + float(r * std::sin(a)));
    const auto id = G_->insert_node(n);
    if (not id) return std::nullopt;
    G_->insert_or_assign_edge(DSR::Edge::create<has_edge_type>(parent_id, *id));
    return G_->get_node(*id);
}

bool EmGraph::upsert(const Record& r)
{
    if (not G_ or r.id.empty()) return false;
    const auto root = G_->get_node("root");
    if (not root) return false;
    std::uint64_t parent_id = root->id();
    if (not r.parent.empty())
        if (const auto p = G_->get_node(r.parent); p.has_value()) parent_id = p->id();
    auto n = ensure_node(r.id, parent_id);
    if (not n) return false;
    G_->add_or_modify_attrib_local<em_record_att>(*n, compact(to_json(r)));
    return G_->update_node(*n);
}

bool EmGraph::upsert(const Stats& st)
{
    if (not G_) return false;
    const auto root = G_->get_node("root");
    if (not root) return false;
    auto n = ensure_node(stats_name(st), root->id());
    if (not n) return false;
    G_->add_or_modify_attrib_local<em_stats_att>(*n, compact(to_json(st)));
    return G_->update_node(*n);
}

bool EmGraph::remove(const std::string& record_id)
{
    if (not G_) return false;
    const auto n = G_->get_node(record_id);
    const auto root = G_->get_node("root");
    if (not n or not root) return false;
    // cortex delete_node neither cascades nor re-parents: a child still open when its parent is folded (a claim that
    // outlives its mission across midnight) would keep a dangling parent attr and no incoming `has`. Re-hang it.
    for (const auto& e : DSR::DSRGraph::get_node_edges_by_type(*n, "has"))
        if (auto child = G_->get_node(e.to()); child.has_value())
        {
            G_->add_or_modify_attrib_local<parent_att>(*child, root->id());
            G_->update_node(*child);
            G_->insert_or_assign_edge(DSR::Edge::create<has_edge_type>(root->id(), child->id()));
        }
    return G_->delete_node(record_id);
}

}   // namespace rc::em
