#pragma once

#include <Eigen/Dense>

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "../affordance_protocol/affordance_protocol.h"   // rc::affordance::Outcome

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

    // One evaluated affordance per select_target() call (for the controller's EFE plot/log).
    struct Candidate
    {
        std::string node_name;
        std::string parent_type;
        float gain = 0.f;        // epistemic value ΔH (nats)
        float efe_score = 0.f;   // selection objective gain − λ·dist (+hysteresis); higher = selected
        bool  eligible = false;  // Offered or Executing (i.e. actually competing)
        std::string state;       // protocol state: Offered / Executing / Completed / … (diagnostic)
    };
    const std::vector<Candidate> &last_candidates() const { return last_candidates_; }

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

    // ── TAKE AN AFFORDANCE OUT OF CONTENTION FOR A WHILE ─────────────────────────────────────────
    // The CONSUMER could not physically get there — repeated wedges, no footprint-feasible standpoint,
    // a goal the repair keeps landing on top of the object itself. That is NOT a completion (nothing was
    // observed) and NOT a defect in the producer's belief (the gain may be entirely real); it is a
    // statement about the APPROACH. Rejecting beats struggling: with other affordances on offer, grinding
    // at an unreachable one buys nothing and costs the robot.
    // Counted in selection ROUNDS rather than milliseconds so this stays clock-free — and it expires on
    // its own, because the world moves: an obstacle clears, the producer proposes a different viewpoint,
    // and the affordance deserves another chance without anyone having to remember to un-suppress it.
    // graph may be null (the suppression still applies locally); when present the refusal is RECORDED
    // on the node as epistemic_refused, which is a statement about the APPROACH — not a completion,
    // and not a claim that anything was observed.
    void suppress_target(const std::shared_ptr<DSR::DSRGraph> &graph, std::uint64_t node_id, int rounds);

    // Grounded EFE selection weights: G = λ_cost·nav_dist − epistemic_gain (nats). switch_margin is
    // the commitment hysteresis (a held affordance must be beaten by this many nats to be dropped).
    void set_selection_params(float lambda_cost, float switch_margin);

    // ── ROOM-vs-OBJECT ARBITRATION ────────────────────────────────────────────────────────────────
    // Scale applied to a ROOM affordance's epistemic gain WHILE at least one non-room affordance is
    // also in the running. Both gains are quoted in nats and the selector treats them as one currency,
    // but they are not equally spendable: re-localising is a standing background need that recovers on
    // its own and can be paid at any time, while an object's look is opportunistic — the robot is near
    // it NOW, and the chance is gone once it drives away. A room affordance that wins every contest
    // starves the objects permanently, which is what this expresses a preference against.
    // ★A PREFERENCE, NOT A GATE. The room can still win on a large enough gain, which is exactly what
    // should happen when the pose belief has genuinely degraded. 1.0 = no preference (old behaviour);
    // it is inert whenever no object affordance is competing, so a lost robot in an empty room is
    // unaffected. The principled fix is in whichever producer is overstating its ΔH — this is the
    // arbitration knob, not a correction to either belief.
    void set_room_gain_scale(float scale);

    // robot_pos (room frame) feeds the nav-cost term. Pass std::nullopt (the default) to ignore
    // distance entirely — selection then uses epistemic_gain + hysteresis only. Do NOT pass a
    // Zero vector to mean "unknown": that would score every affordance by its distance from the
    // room origin (0,0), a real coordinate bias, not a disabled nav-cost.
    std::optional<Target> select_target(const std::shared_ptr<DSR::DSRGraph> &graph,
                                        std::optional<Eigen::Vector2f> robot_pos = std::nullopt);
    // ★ THE OUTCOME IS REQUIRED, NOT DEFAULTED. A default would let every existing call site keep
    // compiling while silently claiming "satisfied", which is exactly the conflation this parameter
    // exists to end — and it would be invisible, because the wrong value looks like a successful look.
    // Making it explicit forces each terminal path to say what actually happened.
    void mark_reached(const std::shared_ptr<DSR::DSRGraph> &graph, rc::affordance::Outcome outcome);

    // Producer side: how the affordance this manager owns last ended. Valid after
    // consume_completion_event() returns true; Outcome::None before any completion.
    [[nodiscard]] rc::affordance::Outcome last_outcome() const { return last_outcome_; }

    /// The affordance node this manager owns (0 until published). Lets an agent's graph-signal slot
    /// recognise a change to ITS OWN affordance and latch the completion at the instant it happens,
    /// instead of waiting for the next poll — see monitor_execution's note on the missed-edge race.
    [[nodiscard]] std::uint64_t managed_node_id() const { return managed_node_id_; }

    // ── NO TWO IN A ROW ───────────────────────────────────────────────────────────────────────────
    // The affordance that was just completed is skipped by the next selection, however good its score.
    // Its producer re-offers it within a cycle or two (that is the protocol working as designed), and
    // its epistemic gain has not yet had time to fall — so the same affordance wins again immediately
    // and the robot loops on one object while everything else waits. Suppression lifts as soon as a
    // DIFFERENT affordance is selected, so this forbids repetition, not revisiting.
    // Name of the affordance suppressed on the last select_target call; empty when none was.
    [[nodiscard]] const std::string &suppressed_name() const { return suppressed_name_; }
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
    rc::affordance::Outcome last_outcome_ = rc::affordance::Outcome::None;

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
    float select_room_gain_scale_ = 1.0f;   // see set_room_gain_scale
    std::uint64_t last_completed_id_ = 0;   // skip this one on the next selection (see suppressed_name)
    // node id -> selection rounds still to skip. See suppress_target: the consumer could not reach it.
    std::map<std::uint64_t, int> unreachable_rounds_;
    std::string   suppressed_name_;         // what that skip cost, for the viewer
    std::uint64_t last_selected_id_ = 0;   // for commitment hysteresis across cycles
    std::vector<Candidate> last_candidates_;   // all affordances evaluated in the last select_target()
    State state_ = State::Idle;
    std::optional<bool> last_observed_active_;
    std::optional<bool> last_observed_pending_;
    std::string selected_target_debug_report_;
};

} // namespace rc