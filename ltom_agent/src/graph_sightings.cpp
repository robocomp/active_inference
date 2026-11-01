#include "graph_sightings.h"
#include "room_resolve/room_resolve.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <optional>
#include <print>
#include <set>
#include <utility>

namespace ltom
{
// ltsm_agent's room_index() rule: the room_id attribute, else the trailing digits of the name.
std::optional<std::int64_t> room_key_of(DSR::DSRGraph& G, const DSR::Node& room)
{
    if (const auto id = G.get_attrib_by_name<room_id_att>(room); id.has_value())
        return static_cast<std::int64_t>(id.value());
    const std::string& n = room.name();
    std::size_t i = n.size();
    while (i > 0 and n[i - 1] >= '0' and n[i - 1] <= '9') --i;
    std::int64_t v = 0;
    if (i < n.size() and std::from_chars(n.data() + i, n.data() + n.size(), v).ec == std::errc{})
        return v;
    return std::nullopt;
}

namespace
{
// The first ancestor of type room (bounded walk: a malformed parent loop must not hang the agent).
std::optional<DSR::Node> room_of(DSR::DSRGraph& G, const DSR::Node& n)
{
    std::optional<DSR::Node> cur = n;
    for (int depth = 0; depth < 16 and cur; ++depth)
    {
        const auto p = G.get_attrib_by_name<parent_att>(*cur);
        if (not p.has_value()) return std::nullopt;
        cur = G.get_node(p.value());
        if (cur and cur->type() == "room") return cur;
    }
    return std::nullopt;
}

std::optional<std::string> class_of(const std::string& name, std::span<const std::string> classes)
{
    for (const auto& c : classes)
        if (name.size() > c.size() and name.starts_with(c) and name[c.size()] == '_')
            return c;
    return std::nullopt;
}

bool finite(const Pose2& p)
{
    return std::isfinite(p.x) and std::isfinite(p.y) and std::isfinite(p.yaw)
       and std::isfinite(p.vx) and std::isfinite(p.vy) and std::isfinite(p.vyaw);
}

// A held instance is logged once per (uid, reason), not once per poll.
void log_held_once(std::uint64_t uid, const std::string& name, const char* why)
{
    static std::set<std::pair<std::uint64_t, std::string>> logged;
    if (logged.emplace(uid, why).second)
        std::println(stderr, "[ltom] {} uid={} is on the graph but unresolvable ({}): held, not counted as gone", name, uid, why);
}
}   // namespace

GraphView read_graph(DSR::DSRGraph& G, DSR::InnerEigenAPI& ie, std::span<const std::string> classes)
{
    GraphView out;

    for (const auto& a : G.get_nodes_by_type("agent"))
    {
        // spec 12: an owner that is gone OR Degraded is absent (graceful exit, flap).
        if (const auto st = G.get_attrib_by_name<agent_presence_state_att>(a); st.has_value() and st.value() == "Degraded")
            continue;
        const auto sp = a.name().rfind(' ');
        std::uint32_t id = 0;
        if (sp != std::string::npos and sp + 1 < a.name().size()
            and std::from_chars(a.name().data() + sp + 1, a.name().data() + a.name().size(), id).ec == std::errc{})
            out.owners_present.insert(id);
    }

    auto objects = G.get_nodes_by_type("object");
    std::ranges::sort(objects, {}, [](const DSR::Node& n) { return n.id(); });   // "first" is by node id

    std::set<std::uint64_t> seen;
    bool warned = false;
    for (const auto& n : objects)
    {
        const auto cls = class_of(n.name(), classes);
        if (not cls) continue;
        const auto uid = G.get_attrib_by_name<object_uid_att>(n);
        if (not uid.has_value()) continue;                       // pre-step-1 binary

        const auto room = room_of(G, n);
        if (not room) { out.present.insert(uid.value()); log_held_once(uid.value(), n.name(), "no room ancestor"); continue; }
        if (rc::room::is_proto(G, room->id())) { out.present.insert(uid.value()); continue; }   // out of scope by design, but still there
        const auto key = room_key_of(G, *room);

        Sighting s;
        s.uid = uid.value();
        s.cls = *cls;
        s.name = n.name();
        s.room_key = key.value_or(-1);
        s.w = G.get_attrib_by_name<width_m_att>(n).value_or(0.f);
        s.d = G.get_attrib_by_name<depth_m_att>(n).value_or(0.f);
        s.h = G.get_attrib_by_name<height_m_att>(n).value_or(0.f);
        s.born_ms = static_cast<std::int64_t>(G.get_attrib_by_name<timestamp_creation_att>(n).value_or(0));

        out.present.insert(s.uid);                               // on the graph from here on, resolvable or not
        const auto T = ie.get_transformation_matrix(room->name(), n.name());
        if (not T) { log_held_once(s.uid, s.name, "no transform to the room"); continue; }
        s.pose.x = (*T).translation().x();
        s.pose.y = (*T).translation().y();
        s.pose.yaw = std::atan2((*T).linear()(1, 0), (*T).linear()(0, 0));

        // Variances from the instance's own parent edge. Approximation for a bottle on a table: the table's
        // own uncertainty is not added.
        const auto parent = G.get_attrib_by_name<parent_att>(n);
        const auto edge = parent.has_value() ? G.get_edge(parent.value(), n.id(), "RT") : std::nullopt;
        const auto cov = edge ? G.get_attrib_by_name<rt_covariance_att>(*edge) : std::nullopt;
        if (not cov.has_value() or cov.value().get().size() < 36) { log_held_once(s.uid, s.name, "no covariance"); continue; }
        const auto& c = cov.value().get();
        s.pose.vx = c[0];
        s.pose.vy = c[7];
        s.pose.vyaw = c[35];

        if (not finite(s.pose)) { log_held_once(s.uid, s.name, "non-finite pose"); continue; }                        // a NaN would poison the keyframe chi2 for good
        if (not seen.insert(s.uid).second)
        {
            if (not std::exchange(warned, true))
                std::println(stderr, "[ltom] ⚠ duplicate object_uid {} on {}: keeping the first node", s.uid, s.name);
            continue;
        }
        out.alive.push_back(std::move(s));
    }
    return out;
}
}   // namespace ltom
