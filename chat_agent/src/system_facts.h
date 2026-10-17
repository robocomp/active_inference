#pragma once
/*
 * system_facts.h — what the chat may say about the SYSTEM itself (the working memory and the agents and sensors
 * feeding it), as plain data and pure text (no Qt, no DSR; tested in tests/test_system.cpp). system_reader.cpp fills
 * these from the graph and the media plane; everything here only formats, filters and measures. 2026-10-17.
 *
 * WHERE THE FACTS COME FROM
 *   agents   one node per agent, type "agent", named "<agent_name> <agent_id>", hanging from "mind" — written by
 *            cortex's AgentInfoAPI at 1 Hz (timestamp_agent = the last heartbeat in NANOSECONDS since the epoch,
 *            timestamp_alivetime = uptime s, cpu_usage = % of one core, memory_usage = RSS kB) and by
 *            common/agent_state_publisher (agent_presence_state, agent_fsm_state). The node can outlive the process
 *            (SIGKILL, persisted graph), so "running" is decided by the heartbeat's age, never by the node existing.
 *   sensors  the nodes carrying a media_descriptor (zed, ricoh, lidar3D, imu): streams from the descriptor,
 *            media_bps (bytes/s the producer publishes). Frames carry a monotonic frame_id, so the rate a consumer
 *            actually receives is measured here (RateMeter) from (frame_id, DDS source time) samples — exact even
 *            though the subscribers deliver only the newest frame.
 */

#include <algorithm>
#include <cstdint>
#include <deque>
#include <format>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace rc::chat
{

/// AgentInfoAPI beats at 1 Hz (its default period); three missed beats = not running. A protocol fact about the
/// heartbeat, not a model threshold: a live agent cannot be silent for longer than that.
inline constexpr std::uint64_t kAgentStaleMs = 3000;

struct AgentFact
{
    std::string name;
    std::uint32_t id = 0;
    std::string presence, fsm;                     ///< "" = not published
    std::optional<std::uint64_t> heartbeat_age_ms;   ///< nullopt = no heartbeat yet
    std::optional<std::uint64_t> uptime_s;
    std::optional<float> cpu;                       ///< % of one core (can exceed 100)
    std::optional<std::uint32_t> memory_kb;
};

[[nodiscard]] inline bool agent_running(const AgentFact& a)
{
    return a.heartbeat_age_ms.has_value() and *a.heartbeat_age_ms <= kAgentStaleMs;
}

namespace system_detail
{
inline std::string lower(std::string s)
{
    std::ranges::transform(s, s.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    return s;
}
inline std::string join(const std::vector<std::string>& v, std::string_view sep = ", ")
{
    std::string out;
    for (const auto& s : v) out += (out.empty() ? "" : std::string(sep)) + s;
    return out;
}
inline std::string hms(std::uint64_t s) { return std::format("{}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60); }
/// "a 18, b 4": most first, then by name.
inline std::string counts(const std::map<std::string, int>& m)
{
    std::vector<std::pair<std::string, int>> v(m.begin(), m.end());
    std::ranges::stable_sort(v, [](const auto& a, const auto& b) { return a.second > b.second; });
    std::vector<std::string> parts;
    for (const auto& [k, n] : v) parts.push_back(std::format("{} {}", k, n));
    return join(parts);
}
inline int total(const std::map<std::string, int>& m)
{
    int n = 0;
    for (const auto& [_, k] : m) n += k;
    return n;
}
}   // namespace system_detail

/// The agents() tool: one line each, running first.
[[nodiscard]] inline std::string answer_agents(std::vector<AgentFact> agents)
{
    using namespace system_detail;
    std::ranges::stable_sort(agents, [](const AgentFact& a, const AgentFact& b) { return agent_running(a) > agent_running(b); });
    const auto n = std::ranges::count_if(agents, agent_running);
    std::string out = std::format("{} agents running, {} agent nodes in the graph.\n", n, agents.size());
    for (const auto& a : agents)
    {
        out += std::format("- {} (id {}): ", a.name, a.id);
        if (not a.heartbeat_age_ms.has_value()) { out += "no heartbeat yet\n"; continue; }
        if (not agent_running(a))
        {
            out += std::format("not heard from for {} s (its node is left over)\n", *a.heartbeat_age_ms / 1000);
            continue;
        }
        std::vector<std::string> parts;
        if (not a.presence.empty()) parts.push_back(a.presence);
        if (not a.fsm.empty()) parts.push_back(a.fsm);
        if (a.uptime_s.has_value()) parts.push_back("up " + hms(*a.uptime_s));
        if (a.cpu.has_value()) parts.push_back(std::format("CPU {:.1f} %", *a.cpu));
        if (a.memory_kb.has_value()) parts.push_back(std::format("memory {:.0f} MB", double(*a.memory_kb) / 1024.0));
        out += (parts.empty() ? std::string("running") : join(parts)) + "\n";
    }
    return out;
}

/// The rate of a stream from (frame_id, source time) samples: frames counted, not samples, so a latest-only
/// subscriber polled slower than the stream still measures it exactly.
class RateMeter
{
public:
    /// Samples older than this are dropped: the rate follows changes within a few seconds.
    static constexpr std::uint64_t kRateWindowNs = 5'000'000'000;

    void push(std::uint64_t frame_id, std::uint64_t source_ns)
    {
        if (not samples_.empty())
        {
            if (frame_id == samples_.back().first) return;   // nothing new arrived since the last sample
            if (frame_id < samples_.back().first or source_ns <= samples_.back().second) samples_.clear();   // restart
        }
        samples_.emplace_back(frame_id, source_ns);
        while (samples_.size() > 2 and source_ns - samples_.front().second > kRateWindowNs) samples_.pop_front();
    }
    [[nodiscard]] std::optional<double> fps() const
    {
        if (samples_.size() < 2) return std::nullopt;
        const auto& [f0, t0] = samples_.front();
        const auto& [f1, t1] = samples_.back();
        return double(f1 - f0) * 1e9 / double(t1 - t0);
    }
    [[nodiscard]] std::uint64_t last_ns() const { return samples_.empty() ? 0 : samples_.back().second; }

private:
    std::deque<std::pair<std::uint64_t, std::uint64_t>> samples_;
};

struct SensorFact
{
    std::string node, label;
    std::vector<std::string> streams;
    std::optional<double> fps;                  ///< measured (RateMeter); nullopt = not measured or stalled
    std::optional<std::uint64_t> frame_age_ms;  ///< now − the newest frame's source time
    int width = 0, height = 0;
    std::optional<float> media_bps;             ///< bytes/s the producer reports
    std::optional<float> nominal_hz;            ///< the descriptor's rate_hz
};

/// A stream whose newest frame is older than this is reported as stalled rather than with a rate. Ten frame periods
/// of the slowest stream we carry (10 Hz LiDAR): a delivered stream is never that late.
inline constexpr std::uint64_t kStreamStallMs = 1000;

/// The sensors() tool. `filter` matches a node name or label, any case, as a substring; "" = all.
[[nodiscard]] inline std::string answer_sensors(const std::vector<SensorFact>& sensors, const std::string& filter)
{
    using namespace system_detail;
    std::string out;
    const std::string f = lower(filter);
    for (const auto& s : sensors)
    {
        if (not f.empty() and lower(s.node).find(f) == std::string::npos and lower(s.label).find(f) == std::string::npos)
            continue;
        out += std::format("- {} ({}): streams {}; ", s.node, s.label, join(s.streams));
        std::vector<std::string> parts;
        const bool stalled = s.frame_age_ms.has_value() and *s.frame_age_ms > kStreamStallMs;
        const std::string nominal = s.nominal_hz.has_value() ? std::format(" (nominal {:g} Hz)", *s.nominal_hz) : "";
        if (stalled) parts.push_back(std::format("no frames for {:.1f} s", double(*s.frame_age_ms) / 1000.0));
        else if (s.fps.has_value()) parts.push_back(std::format("{:.1f} frames/s measured{}", *s.fps, nominal));
        else parts.push_back("rate not measured" + nominal);
        if (s.width > 0) parts.push_back(std::format("{}x{}", s.width, s.height));
        if (s.frame_age_ms.has_value() and not stalled) parts.push_back(std::format("last frame {} ms ago", *s.frame_age_ms));
        if (s.media_bps.has_value()) parts.push_back(std::format("{:.1f} MB/s", *s.media_bps / 1e6));
        out += join(parts) + "\n";
    }
    if (out.empty())
    {
        std::vector<std::string> names;
        for (const auto& s : sensors) names.push_back(s.node);
        return std::format("no sensor matches '{}'; the sensors are: {}", filter, names.empty() ? "none" : join(names));
    }
    return out;
}

struct GraphSummary
{
    std::map<std::string, int> nodes_by_type, edges_by_type, nodes_by_agent;
    std::string room;
};

[[nodiscard]] inline std::string answer_graph(const GraphSummary& g)
{
    using namespace system_detail;
    return std::format("{} nodes, {} edges; current room {}\nnodes by type: {}\nedges by type: {}\nnodes created by: {}\n",
                       total(g.nodes_by_type), total(g.edges_by_type), g.room.empty() ? "unknown" : g.room,
                       counts(g.nodes_by_type), counts(g.edges_by_type), counts(g.nodes_by_agent));
}

struct NodeView
{
    std::string name, type;
    std::uint64_t id = 0;
    std::string created_by;
    std::map<std::string, std::string> attrs;   ///< name → value as text
    std::vector<std::pair<std::string, std::string>> edges_out, edges_in;   ///< (type, the other node's name)
};

/// Longest attribute value shown in full; longer ones are cut and their length given. Keeps one inspect() result
/// a few hundred tokens even for a node carrying a config file or a point list.
inline constexpr std::size_t kInspectValueChars = 80;

[[nodiscard]] inline std::string answer_inspect(const NodeView& v)
{
    std::string out = std::format("{} (type {}, id {}{})\n", v.name, v.type, v.id,
                                  v.created_by.empty() ? "" : ", created by " + v.created_by);
    for (const auto& [k, val] : v.attrs)   // a std::map: alphabetical
        out += val.size() <= kInspectValueChars
                   ? std::format("  {} = {}\n", k, val)
                   : std::format("  {} = {}… ({} chars)\n", k, val.substr(0, kInspectValueChars), val.size());
    const auto edges = [](const auto& list, std::string_view arrow)
    {
        std::vector<std::string> parts;
        for (const auto& [t, other] : list) parts.push_back(std::format("{} {} {}", t, arrow, other));
        return system_detail::join(parts);
    };
    if (not v.edges_out.empty()) out += "edges out: " + edges(v.edges_out, "→") + "\n";
    if (not v.edges_in.empty()) out += "edges in: " + edges(v.edges_in, "←") + "\n";
    return out;
}

}   // namespace rc::chat
