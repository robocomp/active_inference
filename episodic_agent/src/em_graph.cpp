#include "em_graph.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <functional>
#include <utility>
#include <vector>

#include <QJsonDocument>

namespace rc::em
{
namespace
{
std::string compact(const QJsonObject& o) { return QJsonDocument(o).toJson(QJsonDocument::Compact).toStdString(); }

// Viewer placement (em_graph.h), scene units.
constexpr float kColumn = 180.f;   // from one experience to the next, plus kIndent per level of its tree
constexpr float kIndent = 40.f;    // per tree level
constexpr float kRow = 60.f;       // one record per row as the tree hangs down
constexpr float kStatsX = -240.f;  // the stats column, left of root
constexpr int kMaxDepth = 8;       // a has-cycle must not hang the layout

/// Time order of record ids "<YYYYMMDD>-<seq>" (seq numerically); other names after them, by name.
std::pair<std::string, std::uint64_t> order_key(const std::string& name)
{
    const auto dash = name.find('-');
    std::uint64_t seq = 0;
    if (dash == std::string::npos) return {"~" + name, 0};
    const char* b = name.data() + dash + 1;
    const char* e = name.data() + name.size();
    if (const auto r = std::from_chars(b, e, seq); r.ec != std::errc{} or r.ptr != e) return {"~" + name, 0};
    return {name.substr(0, dash), seq};
}
}   // namespace

EmGraph::EmGraph(std::shared_ptr<DSR::DSRGraph> em) : G_(std::move(em)) {}

std::string EmGraph::stats_name(const Stats& st)
{
    // stats_<level>_<kind>[_<author>]_<identity|any>: the author only for missions (key_author is "" otherwise).
    return "stats_" + st.key_level + "_" + st.key_kind + (st.key_author.empty() ? std::string{} : "_" + st.key_author) + "_"
           + (st.key_identity.empty() ? std::string("any") : st.key_identity);
}

std::optional<DSR::Node> EmGraph::ensure_node(const std::string& name, std::uint64_t parent_id)
{
    if (auto n = G_->get_node(name); n.has_value()) return n;
    const auto parent = G_->get_node(parent_id);
    if (not parent) return std::nullopt;
    auto n = DSR::Node::create<metaconcept_node_type>(name);
    G_->add_or_modify_attrib_local<parent_att>(n, parent_id);
    G_->add_or_modify_attrib_local<level_att>(n, G_->get_attrib_by_name<level_att>(*parent).value_or(0) + 1);
    G_->add_or_modify_attrib_local<pos_x_att>(n, G_->get_attrib_by_name<pos_x_att>(*parent).value_or(0.f));
    G_->add_or_modify_attrib_local<pos_y_att>(n, G_->get_attrib_by_name<pos_y_att>(*parent).value_or(0.f));
    const auto id = G_->insert_node(n);
    if (not id) return std::nullopt;
    G_->insert_or_assign_edge(DSR::Edge::create<has_edge_type>(parent_id, *id));
    relayout();
    return G_->get_node(*id);
}

void EmGraph::relayout()
{
    const auto root = G_->get_node("root");
    if (not root) return;
    const auto children = [&](const DSR::Node& n)
    {
        std::vector<DSR::Node> out;
        for (const auto& e : DSR::DSRGraph::get_node_edges_by_type(n, "has"))
            if (auto c = G_->get_node(e.to()); c.has_value()) out.push_back(std::move(*c));
        std::ranges::sort(out, {}, [](const DSR::Node& c) { return order_key(c.name()); });
        return out;
    };
    const auto place = [&](DSR::Node n, float x, float y)
    {
        const float x0 = G_->get_attrib_by_name<pos_x_att>(n).value_or(NAN), y0 = G_->get_attrib_by_name<pos_y_att>(n).value_or(NAN);
        if (std::abs(x0 - x) < 0.5f and std::abs(y0 - y) < 0.5f) return;
        G_->add_or_modify_attrib_local<pos_x_att>(n, x);
        G_->add_or_modify_attrib_local<pos_y_att>(n, y);
        G_->update_node(n);
    };

    float x = kColumn, stats_y = 0.f;
    for (const DSR::Node& top : children(*root))
    {
        if (top.name().starts_with("stats_"))
        {
            place(top, kStatsX, stats_y);
            stats_y += kRow;
            continue;
        }
        int row = 0, deepest = 0;
        const std::function<void(const DSR::Node&, int)> hang = [&](const DSR::Node& n, int depth)
        {
            place(n, x + float(depth) * kIndent, float(row++) * kRow);
            deepest = std::max(deepest, depth);
            if (depth < kMaxDepth)
                for (const DSR::Node& c : children(n)) hang(c, depth + 1);
        };
        hang(top, 0);
        x += kColumn + float(deepest) * kIndent;
    }
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
    const bool deleted = G_->delete_node(record_id);
    if (deleted) relayout();
    return deleted;
}

}   // namespace rc::em
