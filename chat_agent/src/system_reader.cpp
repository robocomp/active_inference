#include "system_reader.h"

#include <charconv>
#include <chrono>
#include <format>
#include <map>
#include <variant>

#include "../../common/media_transport/media_transport.h"

namespace rc::chat
{

namespace
{
std::uint64_t now_ns()
{
    return std::uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::system_clock::now().time_since_epoch()).count());
}

/// "controller 7" → "controller": AgentInfoAPI names the node "<agent_name> <agent_id>".
std::string agent_base_name(const std::string& node_name, std::uint32_t id)
{
    const std::string suffix = " " + std::to_string(id);
    return node_name.ends_with(suffix) ? node_name.substr(0, node_name.size() - suffix.size()) : node_name;
}

const char* sensor_label(const std::string& node)
{
    if (node == "zed") return "front camera";
    if (node == "ricoh") return "360-degree camera";
    if (node.starts_with("lidar")) return "3D LiDAR";
    if (node == "imu") return "IMU";
    return "sensor";
}

/// Any attribute value as text. std::format prints numbers with a point whatever the C locale (es_ES here).
std::string value_text(const DSR::Attribute& a)
{
    return std::visit([](const auto& v) -> std::string
    {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::string>) return v;
        else if constexpr (std::is_same_v<T, bool>) return v ? "true" : "false";
        else if constexpr (std::is_arithmetic_v<T>) return std::format("{}", v);
        else
        {
            std::string out = "[";
            for (std::size_t i = 0; i < v.size(); ++i) out += (i ? ", " : "") + std::format("{}", v[i]);
            return out + "]";
        }
    }, a.value());
}
}   // namespace

std::vector<AgentFact> SystemReader::agents() const
{
    std::vector<AgentFact> out;
    const std::uint64_t now = now_ns();
    for (const auto& n : G_->get_nodes_by_type("agent"))
    {
        AgentFact a{.id = G_->get_attrib_by_name<agent_id_att>(n).value_or(0)};
        a.name = agent_base_name(n.name(), a.id);
        a.presence = G_->get_attrib_by_name<agent_presence_state_att>(n).value_or(std::string{});
        a.fsm = G_->get_attrib_by_name<agent_fsm_state_att>(n).value_or(std::string{});
        // timestamp_agent: the last heartbeat, ns since the epoch (cortex get_unix_timestamp). 0 = none yet.
        if (const auto beat = G_->get_attrib_by_name<timestamp_agent_att>(n); beat.has_value() and *beat > 0)
            a.heartbeat_age_ms = now > *beat ? (now - *beat) / 1'000'000 : 0;
        if (const auto up = G_->get_attrib_by_name<timestamp_alivetime_att>(n); up.has_value()) a.uptime_s = *up;
        if (const auto cpu = G_->get_attrib_by_name<cpu_usage_att>(n); cpu.has_value()) a.cpu = *cpu;
        if (const auto mem = G_->get_attrib_by_name<memory_usage_att>(n); mem.has_value()) a.memory_kb = *mem;
        out.push_back(std::move(a));
    }
    std::ranges::sort(out, {}, &AgentFact::name);
    return out;
}

std::vector<SensorFact> SystemReader::sensors() const
{
    std::vector<SensorFact> out;
    for (const auto& n : G_->get_nodes())
    {
        const auto d = rc::media::descriptor_from_graph(*G_, n.name());
        if (not d.has_value()) continue;
        SensorFact s{.node = n.name(), .label = sensor_label(n.name())};
        for (const auto& [key, _] : d->streams) s.streams.push_back(key);
        if (d->model.rate_hz.has_value()) s.nominal_hz = *d->model.rate_hz;
        if (const auto bps = G_->get_attrib_by_name<media_bps_att>(n); bps.has_value()) s.media_bps = *bps;
        if (n.name() == "zed" or n.name() == "ricoh")
        {
            const auto st = cameras_.stats(n.name() == "zed" ? Camera::front : Camera::pano);
            s.fps = st.fps;
            s.frame_age_ms = st.frame_age_ms;
            s.width = st.width;
            s.height = st.height;
        }
        out.push_back(std::move(s));
    }
    std::ranges::sort(out, {}, &SensorFact::node);
    return out;
}

GraphSummary SystemReader::graph(const std::string& current_room) const
{
    GraphSummary g{.room = current_room, .agents = agents()};
    std::map<std::uint32_t, std::string> agent_names;
    for (const auto& a : g.agents) agent_names[a.id] = a.name;
    for (const auto& n : G_->get_nodes())
    {
        ++g.nodes_by_type[n.type()];
        const auto it = agent_names.find(n.agent_id());
        ++g.nodes_by_agent[it != agent_names.end() ? it->second : std::format("agent {}", n.agent_id())];
        for (const auto& [key, e] : n.fano()) ++g.edges_by_type[key.second];
    }
    return g;
}

std::optional<NodeView> SystemReader::inspect(const std::string& name_or_id) const
{
    std::optional<DSR::Node> node = G_->get_node(name_or_id);
    std::uint64_t id = 0;
    if (not node.has_value())
        if (const auto r = std::from_chars(name_or_id.data(), name_or_id.data() + name_or_id.size(), id);
            r.ec == std::errc{} and r.ptr == name_or_id.data() + name_or_id.size())
            node = G_->get_node(id);
    if (not node.has_value()) return std::nullopt;
    NodeView v{.name = node->name(), .type = node->type(), .id = node->id()};
    for (const auto& a : agents())
        if (a.id == node->agent_id()) v.created_by = a.name;
    for (const auto& [k, a] : node->attrs()) v.attrs[k] = value_text(a);
    const auto name_of = [this](std::uint64_t other)
    {
        const auto o = G_->get_node(other);
        return o.has_value() ? o->name() : std::format("#{}", other);
    };
    for (const auto& [key, e] : node->fano()) v.edges_out.emplace_back(key.second, name_of(key.first));
    for (const auto& e : G_->get_edges_to_id(node->id())) v.edges_in.emplace_back(e.type(), name_of(e.from()));
    return v;
}

}   // namespace rc::chat
