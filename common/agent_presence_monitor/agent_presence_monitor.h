#ifndef AGENT_PRESENCE_MONITOR_H
#define AGENT_PRESENCE_MONITOR_H

#include <ConfigLoader/ConfigLoader.h>

#include <QMetaObject>
#include <QTimer>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace DSR {
class AgentInfoAPI;
class DSRGraph;
class Node;
}

class AgentPresenceMonitor
{
    public:
        enum class PeerState
        {
            Missing,
            Booting,
            Ready,
            Stale,
        };

        struct PeerSnapshot
        {
            std::uint64_t graph_node_id = 0;
            std::uint32_t agent_id = 0;
            std::string agent_name;
            std::uint64_t creation_timestamp = 0;
            std::uint64_t heartbeat_timestamp = 0;
            std::uint64_t alive_time = 0;
            bool active_flag = true;
            bool has_active_flag = false;
            bool ever_ready = false;
            bool restart_grace_active = false;
            std::uint64_t restart_detected_at = 0;
            PeerState state = PeerState::Missing;
        };

        AgentPresenceMonitor(const ConfigLoader &config_loader,
                            std::shared_ptr<DSR::DSRGraph> graph,
                            std::uint32_t local_agent_id);
        ~AgentPresenceMonitor();

        void start();
        void stop();

        void set_local_ready(bool ready);

        std::function<void()>              on_required_ready;
        std::function<void()>              on_required_lost;
        std::function<void(std::uint32_t)> on_peer_restarted;
        std::function<void(std::string name, std::uint32_t id)> on_optional_agent_lost;
        std::function<void(std::string name, std::uint32_t id)> on_optional_agent_ready;

        [[nodiscard]] bool all_required_ready() const;
        [[nodiscard]] std::vector<std::uint32_t> missing_required() const;
        [[nodiscard]] std::vector<std::string> missing_required_names() const;
        [[nodiscard]] std::vector<PeerSnapshot> snapshot() const;

        void delete_owned_nodes();

    private:
        void initial_scan();
        void refresh_agent_node(std::uint64_t graph_node_id);
        void refresh_agent_node(const DSR::Node &node);
        void handle_deleted_node(std::uint64_t graph_node_id);
        void refresh_timeouts();
        void update_local_ready_flag();
        void recompute_required_status();
        void resolve_name_dependencies(const std::string &peer_name, std::uint32_t peer_id);

        void delete_single_owned_node(const std::string &name_or_pattern);
        void delete_subtree(const std::string &root_name);
        void delete_peer_owned_nodes(const std::string &peer_clean_name);
        void check_node_requires();

        [[nodiscard]] PeerState derive_state(const PeerSnapshot &peer, std::uint64_t now_ns) const;
        void log_peer_transition(const PeerSnapshot &before, const PeerSnapshot &after) const;
        void log_required_status_change(bool previous_ready, const std::vector<std::uint32_t> &previous_missing) const;

        [[nodiscard]] static std::uint64_t current_time_ns();
        [[nodiscard]] static const char *to_string(PeerState state);
        [[nodiscard]] static std::vector<std::uint32_t> sorted_unique(std::vector<int> ids);
        [[nodiscard]] static std::string join_ids(const std::vector<std::uint32_t> &ids);

        const ConfigLoader &config_loader;
        std::shared_ptr<DSR::DSRGraph> graph;
        std::uint32_t local_agent_id;

        int monitor_period_ms;
        int heartbeat_timeout_ms;
        int rejoin_grace_ms;
        int agent_info_period_ms;
        int stale_grace_ms;

        bool desired_local_ready = false;
        bool started = false;
        bool required_ready = true;

        std::vector<std::uint32_t> required_agent_ids;
        std::vector<std::uint32_t> optional_agent_ids;
        std::vector<std::uint32_t> missing_required_ids;

        std::vector<std::string> required_agent_names;
        std::unordered_map<std::string, std::uint32_t> name_to_id;
        std::vector<std::string> optional_agent_names;
        std::unordered_map<std::string, std::uint32_t> optional_name_to_id;
        std::unordered_map<std::uint32_t, bool> optional_last_ready;

        std::vector<std::string> owned_node_names_;
        std::vector<std::string> owned_subtree_roots_;
        std::vector<std::string> node_requires_;
        std::string config_dir_;
        bool node_requires_ok_ = true;

        std::unordered_map<std::uint32_t, PeerSnapshot> peers;
        std::unordered_map<std::uint64_t, std::uint32_t> graph_node_to_agent_id;
        std::vector<QMetaObject::Connection> connections;
        QTimer timer;
        std::unique_ptr<DSR::AgentInfoAPI> agent_info_api;
        std::unordered_map<std::uint32_t, std::uint64_t> required_stale_since_;
};

#endif