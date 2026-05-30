#pragma once

#include <Eigen/Dense>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace DSR
{
class DSRGraph;
class Node;
}

namespace rc
{

class AffordanceManager
{
public:
    struct Target
    {
        std::uint64_t node_id = 0;
        std::string node_name;
        Eigen::Vector2f room_pos = Eigen::Vector2f::Zero();
        float yaw_rad = 0.f;
        float epistemic_gain = 0.f;
        bool epistemic_pending = false;
    };

    explicit AffordanceManager(std::string managed_node_name = {});

    void reset();

    void monitor_execution(const std::shared_ptr<DSR::DSRGraph> &graph);
    bool consume_completion_event();
    bool is_executing(const std::shared_ptr<DSR::DSRGraph> &graph);
    bool publish_target(const std::shared_ptr<DSR::DSRGraph> &graph,
                        std::uint64_t parent_id,
                        float tx,
                        float ty,
                        float yaw,
                        float gain,
                        const std::function<void()> &on_node_inserted = {},
                        const std::function<void()> &on_edge_inserted = {});

    std::optional<Target> select_target(const std::shared_ptr<DSR::DSRGraph> &graph);
    void mark_reached(const std::shared_ptr<DSR::DSRGraph> &graph);
    void clear_current();
    bool has_current() const;

private:
    enum class State
    {
        Idle,
        Searching,
        Following,
        Claiming,
        Completing
    };

    enum class ProtocolState
    {
        Missing,
        Offered,
        Executing,
        Completed,
        Invalid
    };

    static std::string_view state_name(State state);
    static std::string_view protocol_state_name(ProtocolState state);
    static ProtocolState decode_protocol_state(bool active, bool pending);

    void transition_to(State next, std::string_view reason, std::uint64_t node_id = 0, std::string_view node_name = {});
    void log_observation(std::uint64_t node_id,
                         std::string_view node_name,
                         bool active,
                         bool pending,
                         float x,
                         float y,
                         float yaw,
                         float gain);
    void reset_observation();

    std::optional<Target> read_target(const std::shared_ptr<DSR::DSRGraph> &graph, const DSR::Node &node) const;
    std::optional<DSR::Node> get_managed_node(const std::shared_ptr<DSR::DSRGraph> &graph);
    bool read_managed_flags(const std::shared_ptr<DSR::DSRGraph> &graph, bool &active, bool &pending);

    std::string managed_node_name_;
    std::uint64_t managed_node_id_ = 0;
    bool waiting_completion_ = false;
    bool completion_detected_ = false;
    bool last_managed_active_ = false;
    bool last_managed_pending_ = true;

    std::uint64_t current_affordance_id_ = 0;
    std::string current_affordance_name_;
    State state_ = State::Idle;
    std::optional<bool> last_observed_active_;
    std::optional<bool> last_observed_pending_;
};

} // namespace rc