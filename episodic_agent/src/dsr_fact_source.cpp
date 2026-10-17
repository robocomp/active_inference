#include "dsr_fact_source.h"

#include "../../common/room_resolve/room_resolve.h"   // rc::room::current_edge_room (SHARED)

#include <algorithm>
#include <cctype>
#include <charconv>
#include <ranges>
#include <set>

namespace rc::em
{
namespace
{
/// The `room` a node hangs from, walking parent_att (bounded: a cycle must not hang the poll).
std::optional<DSR::Node> room_of(DSR::DSRGraph& G, const DSR::Node& n)
{
    std::optional<DSR::Node> cur = n;
    for (int hops = 0; cur.has_value() and hops < 16; ++hops)
    {
        if (cur->type() == "room") return cur;
        const auto p = G.get_attrib_by_name<parent_att>(cur.value());
        if (not p.has_value() or p.value() == cur->id()) return std::nullopt;
        cur = G.get_node(p.value());
    }
    return std::nullopt;
}

/// ltsm_agent's room index: the room_id attribute, else the name's trailing digits (room_eviction.cpp room_index).
std::optional<std::uint64_t> room_index(DSR::DSRGraph& G, const DSR::Node& room)
{
    if (const auto id = G.get_attrib_by_name<room_id_att>(room); id.has_value()) return id.value();
    const std::string& name = room.name();
    auto it = name.end();
    while (it != name.begin() and std::isdigit(static_cast<unsigned char>(*(it - 1)))) --it;
    if (it == name.end()) return std::nullopt;
    std::uint64_t v = 0;
    const char* b = name.data() + (it - name.begin());
    const char* e = name.data() + name.size();
    if (const auto r = std::from_chars(b, e, v); r.ec != std::errc{} or r.ptr != e) return std::nullopt;
    return v;
}

/// A door is an `object` with object_subtype "door" (cortex's `door` type is unused); the name is the backstop.
bool is_door(DSR::DSRGraph& G, const DSR::Node& n)
{
    if (const auto st = G.get_attrib_by_name<object_subtype_att>(n); st.has_value() and st.value() == "door") return true;
    return n.name().starts_with("door_");
}

template <typename Att>
void put(DSR::DSRGraph& G, const DSR::Node& n, const char* key, std::map<std::string, double, std::less<>>& out)
{
    if (const auto v = G.get_attrib_by_name<Att>(n); v.has_value()) out[key] = static_cast<double>(v.value());
}
}   // namespace

DsrFactSource::DsrFactSource(std::shared_ptr<DSR::DSRGraph> live, std::shared_ptr<DSR::DSRGraph> ltsm)
    : live_(std::move(live)), ltsm_(std::move(ltsm)) {}

std::optional<Fact> DsrFactSource::fact(std::string_view node) const
{
    if (not live_ or node.empty()) return std::nullopt;
    const auto n = live_->get_node(std::string(node));
    if (not n.has_value()) return std::nullopt;
    auto& G = *live_;
    Fact f{.node = n->name(), .type = n->type(), .ltsm_id = {}, .values = {}};

    // The kFactAttrs whitelist, each through its typed getter (a bool reads 0/1).
    static_assert(kFactAttrs.size() == 5, "kFactAttrs changed: update the typed reads below");
    put<door_open_prob_att>(G, *n, "door_open_prob", f.values);
    put<door_reach_prob_att>(G, *n, "door_reach_prob", f.values);
    put<door_crossing_progress_att>(G, *n, "door_crossing_progress", f.values);
    put<obj_exist_prob_att>(G, *n, "obj_exist_prob", f.values);
    put<passage_passable_att>(G, *n, "passage_passable", f.values);

    // Pose + its xy variances: the RT edge from the node's parent (ruling R2: the parent room frame).
    if (const auto p = G.get_attrib_by_name<parent_att>(*n); p.has_value())
        if (const auto e = G.get_edge(p.value(), n->id(), "RT"); e.has_value())
        {
            if (const auto t = G.get_attrib_by_name<rt_translation_att>(e.value()); t.has_value() and t->get().size() >= 2)
            {
                f.values["x"] = t->get()[0];
                f.values["y"] = t->get()[1];
            }
            if (const auto r = G.get_attrib_by_name<rt_rotation_euler_xyz_att>(e.value()); r.has_value() and r->get().size() >= 3)
                f.values["yaw"] = r->get()[2];
            if (const auto c = G.get_attrib_by_name<rt_covariance_att>(e.value()); c.has_value() and c->get().size() >= 36)
            {
                f.values["cov_xx"] = c->get()[0];
                f.values["cov_yy"] = c->get()[7];
            }
        }

    if (is_door(G, *n)) f.ltsm_id = ltsm_id_of(*n);
    return f;
}

std::string DsrFactSource::current_room() const
{
    if (not live_) return {};
    const auto id = rc::room::current_edge_room(*live_);
    if (not id) return {};
    const auto n = live_->get_node(*id);
    return n ? n->name() : std::string{};
}

std::string DsrFactSource::ltsm_id_of(const DSR::Node& door) const
{
    if (not live_ or not ltsm_) return {};
    const auto room = room_of(*live_, door);
    if (not room) return {};
    const auto idx = room_index(*live_, *room);
    if (not idx) return {};
    const auto face = ltsm_->get_node("r" + std::to_string(*idx) + "_" + door.name());
    if (not face) return {};
    for (const auto& p : ltsm_->get_nodes_by_type("metaconcept"))
    {
        if (ltsm_->get_attrib_by_name<object_subtype_att>(p).value_or(std::string{}) != "passage") continue;
        const auto m = DSR::DSRGraph::get_node_edges_by_type(p, "match");
        if (std::ranges::any_of(m, [&](const DSR::Edge& e) { return e.to() == face->id(); })) return p.name();
    }
    return {};
}

std::vector<Event> LiveDiff::poll(DSR::DSRGraph& G)
{
    std::map<std::uint64_t, std::pair<std::string, std::string>> now;
    for (const char* type : kEventNodeTypes)
        for (const auto& n : G.get_nodes_by_type(type)) now.emplace(n.id(), std::pair{n.type(), n.name()});
    const auto cur = rc::room::current_edge_room(G);

    std::vector<Event> out;
    if (primed_)
    {
        const auto ev = [](const std::string& type, bool appeared) -> std::string
        {
            if (type == "affordance") return appeared ? "offered" : "withdrawn";
            return appeared ? "born" : "died";
        };
        for (const auto& [id, tn] : now)
            if (not known_.contains(id)) out.push_back(Event{.dt_ms = 0, .kind = ev(tn.first, true), .node = tn.second});
        for (const auto& [id, tn] : known_)
            if (not now.contains(id)) out.push_back(Event{.dt_ms = 0, .kind = ev(tn.first, false), .node = tn.second});
        if (cur and cur != current_)
            if (const auto n = G.get_node(*cur); n.has_value())
                out.push_back(Event{.dt_ms = 0, .kind = "current_moved", .node = n->name()});
    }
    primed_ = true;
    known_ = std::move(now);
    if (cur) current_ = cur;
    return out;
}

}   // namespace rc::em
