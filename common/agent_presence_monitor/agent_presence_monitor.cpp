#include "agent_presence_monitor.h"

#include <dsr/api/dsr_agent_info_api.h>
#include <dsr/api/dsr_api.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <queue>
#include <sstream>
#include <unordered_set>
#include <utility>

#include <QMetaType>

namespace
{
std::string join_ids(const std::vector<std::uint32_t> &ids)
{
    std::ostringstream stream;
    for (std::size_t index = 0; index < ids.size(); ++index)
    {
        if (index > 0)
            stream << ", ";
        stream << ids[index];
    }
    return stream.str();
}
}

AgentPresenceMonitor::AgentPresenceMonitor(const ConfigLoader &config_loader,
                                           std::shared_ptr<DSR::DSRGraph> graph,
                                           std::uint32_t local_agent_id)
    : config_loader(config_loader)
    , graph(std::move(graph))
    , local_agent_id(local_agent_id)
    , monitor_period_ms(config_loader.exists("Presence.monitor_period_ms") ? config_loader.get<int>("Presence.monitor_period_ms") : 500)
    , heartbeat_timeout_ms(config_loader.exists("Presence.heartbeat_timeout_ms") ? config_loader.get<int>("Presence.heartbeat_timeout_ms") : 3000)
    , rejoin_grace_ms(config_loader.exists("Presence.rejoin_grace_ms") ? config_loader.get<int>("Presence.rejoin_grace_ms") : 1500)
    , agent_info_period_ms(config_loader.exists("Presence.agent_info_period_ms") ? config_loader.get<int>("Presence.agent_info_period_ms") : 1000)
    , required_agent_ids(config_loader.exists("Presence.required_agent_ids")
                             ? sorted_unique(config_loader.get<std::vector<int>>("Presence.required_agent_ids"))
                             : std::vector<std::uint32_t>{})
    , optional_agent_ids(config_loader.exists("Presence.optional_agent_ids")
                             ? sorted_unique(config_loader.get<std::vector<int>>("Presence.optional_agent_ids"))
                             : std::vector<std::uint32_t>{})
    , required_agent_names(config_loader.exists("Presence.required_agent_names")
                               ? config_loader.get<std::vector<std::string>>("Presence.required_agent_names")
                               : std::vector<std::string>{})
    , optional_agent_names(config_loader.exists("Presence.optional_agent_names")
                               ? config_loader.get<std::vector<std::string>>("Presence.optional_agent_names")
                               : std::vector<std::string>{})
    , owned_node_names_(config_loader.exists("Owns.nodes")
                            ? config_loader.get<std::vector<std::string>>("Owns.nodes")
                            : std::vector<std::string>{})
    , owned_subtree_roots_(config_loader.exists("Owns.subtrees")
                               ? config_loader.get<std::vector<std::string>>("Owns.subtrees")
                               : std::vector<std::string>{})
    , node_requires_(config_loader.exists("Presence.node_requires")
                         ? config_loader.get<std::vector<std::string>>("Presence.node_requires")
                         : std::vector<std::string>{})
    , config_dir_(config_loader.exists("Presence.config_dir")
                      ? config_loader.get<std::string>("Presence.config_dir")
                      : "etc")
{
    timer.setInterval(monitor_period_ms);
}

AgentPresenceMonitor::~AgentPresenceMonitor()
{
    if (started)
    {
        agent_info_api.reset(); // stop heartbeat timer before deleting owned nodes to avoid re-creation race
        delete_owned_nodes();
    }
    stop();
}

void AgentPresenceMonitor::start()
{
    if (started)
        return;

    started = true;
    desired_local_ready = false;

    qRegisterMetaType<std::string>("std::string");
    qRegisterMetaType<std::uint32_t>("std::uint32_t");
    qRegisterMetaType<std::uint64_t>("std::uint64_t");
    qRegisterMetaType<std::vector<std::string>>("std::vector<std::string>");
    qRegisterMetaType<DSR::SignalInfo>("DSR::SignalInfo");

    agent_info_api = std::make_unique<DSR::AgentInfoAPI>(graph.get(), static_cast<std::uint32_t>(agent_info_period_ms));

    connections.emplace_back(QObject::connect(graph.get(),
                                              &DSR::DSRGraph::update_node_signal,
                                              &timer,
                                              [this](std::uint64_t id, const std::string &type, DSR::SignalInfo) {
                                                  if (type == "agent")
                                                      refresh_agent_node(id);
                                              },
                                              Qt::QueuedConnection));

    connections.emplace_back(QObject::connect(graph.get(),
                                              &DSR::DSRGraph::update_node_attr_signal,
                                              &timer,
                                              [this](std::uint64_t id, const std::vector<std::string> &, DSR::SignalInfo) {
                                                  refresh_agent_node(id);
                                              },
                                              Qt::QueuedConnection));

    connections.emplace_back(QObject::connect(graph.get(),
                                              &DSR::DSRGraph::del_node_signal,
                                              &timer,
                                              [this](std::uint64_t id, DSR::SignalInfo) {
                                                  handle_deleted_node(id);
                                              },
                                              Qt::QueuedConnection));

    QObject::connect(&timer, &QTimer::timeout, [&]() {
        initial_scan();
        refresh_timeouts();
        check_node_requires();
        update_local_ready_flag();
    });

    initial_scan();
    refresh_timeouts();
    timer.start();

    std::ostringstream req_desc;
    req_desc << "[";
    bool first = true;
    for (auto id : required_agent_ids)
    {
        if (!first)
            req_desc << ", ";
        req_desc << id;
        first = false;
    }
    for (const auto &name : required_agent_names)
    {
        if (!first)
            req_desc << ", ";
        req_desc << '"' << name << '"';
        first = false;
    }
    req_desc << "]";

    std::cout << "[Presence] monitor started for agent " << local_agent_id
              << " required=" << req_desc.str()
              << " optional=[" << join_ids(optional_agent_ids) << "]"
              << " timeout_ms=" << heartbeat_timeout_ms
              << " rejoin_grace_ms=" << rejoin_grace_ms << std::endl;
}

void AgentPresenceMonitor::stop()
{
    if (!started)
        return;

    desired_local_ready = false;
    update_local_ready_flag();

    timer.stop();
    for (const auto &connection : connections)
        QObject::disconnect(connection);
    connections.clear();

    agent_info_api.reset();
    started = false;
}

void AgentPresenceMonitor::set_local_ready(bool ready)
{
    desired_local_ready = ready;
    update_local_ready_flag();
}

bool AgentPresenceMonitor::all_required_ready() const
{
    return required_ready;
}

std::vector<std::uint32_t> AgentPresenceMonitor::missing_required() const
{
    return missing_required_ids;
}

std::vector<std::string> AgentPresenceMonitor::missing_required_names() const
{
    std::vector<std::string> result;
    for (const auto missing_id : missing_required_ids)
    {
        bool found_name = false;
        for (const auto &[name, resolved_id] : name_to_id)
        {
            if (resolved_id == missing_id)
            {
                result.push_back("\"" + name + "\"");
                found_name = true;
                break;
            }
        }
        if (!found_name)
            result.push_back("id:" + std::to_string(missing_id));
    }
    for (const auto &name : required_agent_names)
    {
        if (name_to_id.find(name) == name_to_id.end())
            result.push_back("\"" + name + "\" (unseen)");
    }
    return result;
}

std::vector<AgentPresenceMonitor::PeerSnapshot> AgentPresenceMonitor::snapshot() const
{
    std::vector<PeerSnapshot> values;
    values.reserve(peers.size());
    for (const auto &[agent_id, peer] : peers)
    {
        if (agent_id == local_agent_id)
            continue;
        values.push_back(peer);
    }
    return values;
}

void AgentPresenceMonitor::resolve_name_dependencies(const std::string &peer_name, std::uint32_t peer_id)
{
    if (required_agent_names.empty() && optional_agent_names.empty())
        return;

    std::string clean_name = peer_name;
    const auto open = peer_name.find(" ( ");
    const auto close = peer_name.rfind(" )");
    if (open != std::string::npos && close != std::string::npos && close > open + 3)
    {
        clean_name = peer_name.substr(open + 3, close - open - 3);
    }
    else
    {
        // DSR often stores node name as "agent_name id" (e.g. "robot_concept 6").
        // Strip the trailing numeric suffix so name-based matching works.
        const auto last_space = peer_name.rfind(' ');
        if (last_space != std::string::npos)
        {
            const auto suffix = peer_name.substr(last_space + 1);
            if (!suffix.empty() && std::all_of(suffix.begin(), suffix.end(), ::isdigit))
                clean_name = peer_name.substr(0, last_space);
        }
    }

    for (const auto &required_name : required_agent_names)
    {
        if (required_name != clean_name || name_to_id.count(required_name))
            continue;

        name_to_id[required_name] = peer_id;
        if (std::find(required_agent_ids.begin(), required_agent_ids.end(), peer_id) == required_agent_ids.end())
        {
            required_agent_ids.push_back(peer_id);
            std::sort(required_agent_ids.begin(), required_agent_ids.end());
            std::cout << "[Presence] resolved required name \"" << required_name
                      << "\" -> id " << peer_id << std::endl;
        }
    }

    for (const auto &opt_name : optional_agent_names)
    {
        if (opt_name != clean_name || optional_name_to_id.count(opt_name))
            continue;

        optional_name_to_id[opt_name] = peer_id;
        if (std::find(optional_agent_ids.begin(), optional_agent_ids.end(), peer_id) == optional_agent_ids.end())
        {
            optional_agent_ids.push_back(peer_id);
            std::sort(optional_agent_ids.begin(), optional_agent_ids.end());
            std::cout << "[Presence] resolved optional name \"" << opt_name
                      << "\" -> id " << peer_id << std::endl;
        }
    }
}

void AgentPresenceMonitor::initial_scan()
{
    for (const auto &node : graph->get_nodes_by_type("agent"))
        refresh_agent_node(node);
}

void AgentPresenceMonitor::refresh_agent_node(std::uint64_t graph_node_id)
{
    auto node = graph->get_node(graph_node_id);
    if (!node.has_value() || node->type() != "agent")
        return;

    refresh_agent_node(*node);
}

void AgentPresenceMonitor::refresh_agent_node(const DSR::Node &node)
{
    const auto agent_id = graph->get_attrib_by_name<agent_id_att>(node);
    const auto creation_timestamp = graph->get_attrib_by_name<timestamp_creation_att>(node);
    const auto heartbeat_timestamp = graph->get_attrib_by_name<timestamp_agent_att>(node);

    if (!agent_id.has_value() || !creation_timestamp.has_value() || !heartbeat_timestamp.has_value())
        return;

    const auto now_ns = current_time_ns();
    PeerSnapshot updated;
    if (auto it = peers.find(*agent_id); it != peers.end())
        updated = it->second;

    const auto before = updated;
    const bool restarted = before.ever_ready && before.creation_timestamp != 0 && before.creation_timestamp != *creation_timestamp;

    updated.graph_node_id = node.id();
    updated.agent_id = *agent_id;
    updated.agent_name = graph->get_attrib_by_name<agent_name_att>(node).value_or(node.name());
    updated.creation_timestamp = *creation_timestamp;
    updated.heartbeat_timestamp = *heartbeat_timestamp;
    updated.alive_time = graph->get_attrib_by_name<timestamp_alivetime_att>(node).value_or(0);
    updated.has_active_flag = graph->get_attrib_by_name<active_agent_att>(node).has_value();
    updated.active_flag = graph->get_attrib_by_name<active_agent_att>(node).value_or(true);

    resolve_name_dependencies(updated.agent_name, updated.agent_id);
    if (restarted)
    {
        updated.restart_grace_active = true;
        updated.restart_detected_at = now_ns;
        if (on_peer_restarted)
            on_peer_restarted(updated.agent_id);
    }
    else if (updated.restart_grace_active && now_ns >= updated.restart_detected_at + static_cast<std::uint64_t>(rejoin_grace_ms) * 1000000ULL)
    {
        updated.restart_grace_active = false;
    }

    updated.state = derive_state(updated, now_ns);
    if (updated.state == PeerState::Ready)
        updated.ever_ready = true;
    peers[updated.agent_id] = updated;
    graph_node_to_agent_id[node.id()] = updated.agent_id;

    if (updated.agent_id != local_agent_id)
        log_peer_transition(before, updated);

    recompute_required_status();
}

void AgentPresenceMonitor::handle_deleted_node(std::uint64_t graph_node_id)
{
    const auto node_it = graph_node_to_agent_id.find(graph_node_id);
    if (node_it == graph_node_to_agent_id.end())
        return;

    const auto peer_it = peers.find(node_it->second);
    if (peer_it == peers.end())
        return;

    auto before = peer_it->second;
    peer_it->second.graph_node_id = 0;
    peer_it->second.state = PeerState::Missing;
    peer_it->second.heartbeat_timestamp = 0;
    peer_it->second.restart_grace_active = false;

    if (peer_it->second.agent_id != local_agent_id)
        log_peer_transition(before, peer_it->second);

    graph_node_to_agent_id.erase(node_it);
    recompute_required_status();
}

void AgentPresenceMonitor::refresh_timeouts()
{
    const auto now_ns = current_time_ns();
    for (auto &[agent_id, peer] : peers)
    {
        auto before = peer;
        if (peer.restart_grace_active && now_ns >= peer.restart_detected_at + static_cast<std::uint64_t>(rejoin_grace_ms) * 1000000ULL)
            peer.restart_grace_active = false;

        peer.state = derive_state(peer, now_ns);
        if (peer.state == PeerState::Ready)
            peer.ever_ready = true;
        if (agent_id != local_agent_id)
            log_peer_transition(before, peer);
    }

    recompute_required_status();
}

void AgentPresenceMonitor::update_local_ready_flag()
{
    const auto it = peers.find(local_agent_id);
    if (it == peers.end() || it->second.graph_node_id == 0)
        return;

    auto node = graph->get_node(it->second.graph_node_id);
    if (!node.has_value())
        return;

    const auto current_ready = graph->get_attrib_by_name<active_agent_att>(*node).value_or(false);
    if (current_ready == desired_local_ready)
        return;

    graph->add_or_modify_attrib_local<active_agent_att>(*node, desired_local_ready);
    graph->update_node(*node);
}

void AgentPresenceMonitor::recompute_required_status()
{
    const auto previous_ready = required_ready;
    const auto previous_missing = missing_required_ids;

    missing_required_ids.clear();
    for (const auto required_id : required_agent_ids)
    {
        const auto it = peers.find(required_id);
        if (it == peers.end() || it->second.state != PeerState::Ready)
            missing_required_ids.push_back(required_id);
    }

    for (const auto required_id : missing_required_ids)
    {
        if (std::find(previous_missing.begin(), previous_missing.end(), required_id) == previous_missing.end())
        {
            for (const auto &[name, id] : name_to_id)
            {
                if (id == required_id)
                {
                    delete_peer_owned_nodes(name);
                    break;
                }
            }
        }
    }

    required_ready = missing_required_ids.empty() &&
                     node_requires_ok_ &&
                     std::all_of(required_agent_names.begin(), required_agent_names.end(),
                                 [this](const std::string &name) { return name_to_id.count(name) > 0; });

    if (required_ready != previous_ready || previous_missing != missing_required_ids)
    {
        log_required_status_change(previous_ready, previous_missing);
        if (required_ready && !previous_ready && on_required_ready)
            on_required_ready();
        else if (!required_ready && previous_ready && on_required_lost)
            on_required_lost();
    }

    for (const auto opt_id : optional_agent_ids)
    {
        const auto it = peers.find(opt_id);
        const bool now_ready = it != peers.end() && it->second.state == PeerState::Ready;
        const auto prev_it = optional_last_ready.find(opt_id);
        const bool was_ready = prev_it != optional_last_ready.end() && prev_it->second;

        if (now_ready == was_ready)
            continue;

        optional_last_ready[opt_id] = now_ready;

        std::string opt_name = "id:" + std::to_string(opt_id);
        for (const auto &[name, id] : optional_name_to_id)
        {
            if (id == opt_id)
            {
                opt_name = name;
                break;
            }
        }

        if (!now_ready)
        {
            delete_peer_owned_nodes(opt_name);
            std::cout << "[Presence] optional peer " << opt_id
                      << " (" << opt_name << ") lost" << std::endl;
            if (on_optional_agent_lost)
                on_optional_agent_lost(opt_name, opt_id);
        }
        else if (on_optional_agent_ready)
        {
            std::cout << "[Presence] optional peer " << opt_id
                      << " (" << opt_name << ") ready" << std::endl;
            on_optional_agent_ready(opt_name, opt_id);
        }
    }
}

void AgentPresenceMonitor::delete_owned_nodes()
{
    for (const auto &name : owned_node_names_)
        delete_single_owned_node(name);
    for (const auto &root : owned_subtree_roots_)
        delete_subtree(root);
}

void AgentPresenceMonitor::delete_single_owned_node(const std::string &name_or_pattern)
{
    if (!graph)
        return;

    if (!name_or_pattern.empty() && name_or_pattern.back() == '*')
    {
        const std::string prefix = name_or_pattern.substr(0, name_or_pattern.size() - 1);
        std::vector<std::string> to_delete;
        for (const auto &node : graph->get_nodes())
        {
            if (node.name().size() >= prefix.size() && node.name().substr(0, prefix.size()) == prefix)
                to_delete.push_back(node.name());
        }
        for (const auto &name : to_delete)
        {
            if (auto node = graph->get_node(name); node.has_value())
            {
                if (!graph->delete_node(node->id()))
                    std::cerr << "[Graph] failed to delete node '" << name << "'" << std::endl;
                else
                    std::cout << "[Graph] node '" << name << "' deleted" << std::endl;
            }
        }
    }
    else
    {
        if (auto node = graph->get_node(name_or_pattern); node.has_value())
        {
            if (!graph->delete_node(node->id()))
                std::cerr << "[Graph] failed to delete node '" << name_or_pattern << "'" << std::endl;
            else
                std::cout << "[Graph] node '" << name_or_pattern << "' deleted" << std::endl;
        }
    }
}

void AgentPresenceMonitor::delete_subtree(const std::string &root_name)
{
    if (!graph)
        return;
    auto root_opt = graph->get_node(root_name);
    if (!root_opt.has_value())
        return;

    std::vector<std::uint64_t> to_delete;
    std::unordered_set<std::uint64_t> visited;
    std::queue<std::uint64_t> queue;
    queue.push(root_opt->id());

    while (!queue.empty())
    {
        const std::uint64_t id = queue.front();
        queue.pop();
        if (!visited.insert(id).second)
            continue;
        to_delete.push_back(id);

        if (auto edges = graph->get_edges(id); edges.has_value())
        {
            for (const auto &[key, edge] : *edges)
            {
                if (key.second == "RT" && !visited.count(edge.to()))
                    queue.push(edge.to());
            }
        }
    }

    for (auto it = to_delete.rbegin(); it != to_delete.rend(); ++it)
    {
        if (auto node = graph->get_node(*it); node.has_value())
        {
            if (!graph->delete_node(*it))
                std::cerr << "[Graph] failed to delete node '" << node->name()
                          << "' (subtree of '" << root_name << ")" << std::endl;
            else
                std::cout << "[Graph] node '" << node->name() << "' deleted (subtree of '"
                          << root_name << "')" << std::endl;
        }
    }
}

void AgentPresenceMonitor::delete_peer_owned_nodes(const std::string &peer_clean_name)
{
    if (config_dir_.empty())
        return;
    const std::string peer_config_path = config_dir_ + "/" + peer_clean_name + ".toml";
    try
    {
        ConfigLoader peer_loader;
        peer_loader.load(peer_config_path);
        if (!peer_loader.exists("Owns.nodes"))
            return;
        const auto peer_nodes = peer_loader.get<std::vector<std::string>>("Owns.nodes");
        if (peer_nodes.empty())
            return;
        std::cout << "[Presence] cleaning up " << peer_nodes.size()
                  << " owned node(s) of crashed peer '" << peer_clean_name << "'" << std::endl;
        for (const auto &name : peer_nodes)
            delete_single_owned_node(name);
        if (peer_loader.exists("Owns.subtrees"))
        {
            const auto peer_subtrees = peer_loader.get<std::vector<std::string>>("Owns.subtrees");
            for (const auto &root : peer_subtrees)
                delete_subtree(root);
        }
    }
    catch (const std::exception &error)
    {
        std::cout << "[Presence] no config for peer '" << peer_clean_name
                  << "': " << error.what() << std::endl;
    }
}

void AgentPresenceMonitor::check_node_requires()
{
    if (node_requires_.empty())
        return;
    const bool all_ok = std::all_of(node_requires_.begin(), node_requires_.end(),
                                    [this](const std::string &name) { return graph->get_node(name).has_value(); });
    if (all_ok == node_requires_ok_)
        return;
    node_requires_ok_ = all_ok;
    if (!all_ok)
        std::cout << "[Presence] node_requires not satisfied (a required node disappeared)" << std::endl;
    else
        std::cout << "[Presence] node_requires satisfied (all required nodes present)" << std::endl;
    recompute_required_status();
}

AgentPresenceMonitor::PeerState AgentPresenceMonitor::derive_state(const PeerSnapshot &peer, std::uint64_t now_ns) const
{
    if (peer.graph_node_id == 0)
        return PeerState::Missing;

    const auto timeout_ns = static_cast<std::uint64_t>(heartbeat_timeout_ms) * 1000000ULL;
    if (peer.heartbeat_timestamp == 0 || now_ns > peer.heartbeat_timestamp + timeout_ns)
        return PeerState::Stale;

    const bool ready = peer.active_flag || !peer.has_active_flag;
    if (!ready || peer.restart_grace_active)
        return PeerState::Booting;

    return PeerState::Ready;
}

void AgentPresenceMonitor::log_peer_transition(const PeerSnapshot &before, const PeerSnapshot &after) const
{
    if (before.state == after.state && before.creation_timestamp == after.creation_timestamp)
        return;

    std::cout << "[Presence] peer " << after.agent_id
              << " (" << after.agent_name << ") "
              << to_string(before.state) << " -> " << to_string(after.state);
    if (before.creation_timestamp != 0 && before.creation_timestamp != after.creation_timestamp)
        std::cout << " [restart detected]";
    std::cout << std::endl;
}

void AgentPresenceMonitor::log_required_status_change(bool previous_ready, const std::vector<std::uint32_t> &previous_missing) const
{
    if (previous_ready == required_ready && previous_missing == missing_required_ids)
        return;

    if (required_ready)
        std::cout << "[Presence] all required agents are ready" << std::endl;
    else
        std::cout << "[Presence] required agents not ready: [" << join_ids(missing_required_ids) << "]" << std::endl;
}

std::uint64_t AgentPresenceMonitor::current_time_ns()
{
    const auto now = std::chrono::system_clock::now();
    return std::chrono::time_point_cast<std::chrono::nanoseconds>(now).time_since_epoch().count();
}

const char *AgentPresenceMonitor::to_string(PeerState state)
{
    switch (state)
    {
        case PeerState::Missing:
            return "Missing";
        case PeerState::Booting:
            return "Booting";
        case PeerState::Ready:
            return "Ready";
        case PeerState::Stale:
            return "Stale";
    }
    return "Unknown";
}

std::vector<std::uint32_t> AgentPresenceMonitor::sorted_unique(std::vector<int> ids)
{
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());

    std::vector<std::uint32_t> normalized;
    normalized.reserve(ids.size());
    for (const auto id : ids)
    {
        if (id >= 0)
            normalized.push_back(static_cast<std::uint32_t>(id));
    }
    return normalized;
}