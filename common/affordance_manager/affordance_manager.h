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
        std::uint64_t parent_node_id = 0;
        std::string parent_node_name;
        std::string parent_node_type;
        float shape_width_m = 0.f;
        float shape_depth_m = 0.f;
        bool has_shape = false;
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
    bool release_execution_claim(const std::shared_ptr<DSR::DSRGraph> &graph);

    // Grounded EFE selection weights: G = λ_cost·nav_dist − epistemic_gain (nats). switch_margin is
    // the commitment hysteresis (a held affordance must be beaten by this many nats to be dropped).
    void set_selection_params(float lambda_cost, float switch_margin);

    // robot_pos (room frame) feeds the nav-cost term. Pass std::nullopt (the default) to ignore
    // distance entirely — selection then uses epistemic_gain + hysteresis only. Do NOT pass a
    // Zero vector to mean "unknown": that would score every affordance by its distance from the
    // room origin (0,0), a real coordinate bias, not a disabled nav-cost.
    std::optional<Target> select_target(const std::shared_ptr<DSR::DSRGraph> &graph,
                                        std::optional<Eigen::Vector2f> robot_pos = std::nullopt);
    void mark_reached(const std::shared_ptr<DSR::DSRGraph> &graph);
    void clear_current();
    bool has_current() const;
    std::string current_name() const;

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
    // Grounded EFE selection (set_selection_params): nav-cost weight (nats/m) + hysteresis (nats).
    float select_lambda_cost_ = 0.2f;
    float select_switch_margin_ = 0.5f;
    std::uint64_t last_selected_id_ = 0;   // for commitment hysteresis across cycles
    State state_ = State::Idle;
    std::optional<bool> last_observed_active_;
    std::optional<bool> last_observed_pending_;
    std::string selected_target_debug_report_;
};

} // namespace rc