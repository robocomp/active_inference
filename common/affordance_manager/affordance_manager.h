#pragma once
#include <source_location>

#include <Eigen/Dense>

#include <cstdint>
#include <functional>
#include <map>
#include <set>
#include <utility>
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
    // ★WHY THE LAST SELECTION CHOSE NOTHING. Set on every select_target(); "" when one was chosen.
    // Exists because a frozen robot with valid offers on the wire is indistinguishable, from outside,
    // from a broken selector — and reading it out of stdout is not a diagnosis, it is a guess.
    [[nodiscard]] const std::string &last_reject_reason() const { return last_reject_reason_; }

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

    // ── A VERDICT READ OFF THE MAP, REMEMBERED AGAINST THE MAP ───────────────────────────────────
    // "The body does not fit in that cell" and "there is no route to it and nowhere closer" are
    // decided from the planner grid in a few milliseconds and cost the robot no motion at all. So the
    // pair can complete standpoints at loop rate while the base never moves: measured live
    // 2026-08-19, 21648 `unreachable` reports in 30 minutes — about twelve every second — and in
    // simulation with ten producers, 922 completions for 108 observations.
    // ★THE CACHE IS NOT A RATE LIMIT. Two rate limits were tried in protocol_multi_bench first: a
    // per-producer backoff (insufficient — with ten agents the consumer just turns to the next one's
    // offer, so a per-agent limit cannot bound a global resource) and a global quiet period on the
    // consumer (bounded the churn and starved the useful work with it, 120 observations down to 42).
    // What works has no duration in it: the verdict is a function of (cell, robot pose, map), so it
    // is remembered against exactly those three and cannot outlive any of them. Each cell yields at
    // most one such verdict per pose per map, and a new one requires the robot to drive — which costs
    // physical time by construction, so no zero-cost cycle remains for a livelock to live in.
    // ★FAILS OPEN: with no robot pose or no rasterised map, nothing is suppressed.
    // The owner of the planner grid stamps its identity here once per cycle; selection then consults
    // the cache against it without every caller having to carry the hash around.
    void set_map_identity(std::size_t h) { map_identity_ = h; }
    void note_map_verdict(const Eigen::Vector2f &cell, const Eigen::Vector2f &robot);
    [[nodiscard]] bool has_map_verdict(const Eigen::Vector2f &cell,
                                       const std::optional<Eigen::Vector2f> &robot) const;

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
    // ★WHO COMPLETED IT. mark_reached is the ONLY writer that clears epistemic_pending, so every
    // completion — including the phantom ones logged with the robot 3.11 m from the cell — passes
    // through here. source_location names the caller in one run instead of reading every call site.
    void mark_reached(const std::shared_ptr<DSR::DSRGraph> &graph, rc::affordance::Outcome outcome,
                      std::source_location loc = std::source_location::current());

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
    // ★★★HAVE WE ACTUALLY SEEN THIS ARMING PENDING? publish_target() sets waiting_completion_
    // OPTIMISTICALLY, so if the next poll reads pending==false — a stale read of a node we just armed —
    // the "pending -> not pending" edge fires and a completion is declared with NO consumer
    // involvement at all. Measured 2026-08-19: completions climbing 389→392 while the robot sat still
    // 2.3–3.1 m from the offered cell and the consumer held no target. The producer was completing its
    // own offers, never marking the cell visited (the robot was not there), re-offering it, and the
    // consumer answered "just-completed". Neither agent was driving the robot.
    // ★Same defect, same cure as RoomSceneGraph::armed_seen_live_: a LEVEL may only be believed as an
    // EDGE once the level has actually been observed on the other side first.
    bool pending_seen_ = false;
    bool completion_detected_ = false;
    bool last_managed_active_ = false;
    bool last_managed_pending_ = true;

    std::uint64_t current_affordance_id_ = 0;
    std::string current_affordance_name_;
    // Grounded EFE selection (set_selection_params): nav-cost weight (nats/m) + hysteresis (nats).
    float select_lambda_cost_ = 0.2f;
    float select_switch_margin_ = 0.5f;
    float select_room_gain_scale_ = 1.0f;   // see set_room_gain_scale
    // ★★★THE NODE IS NOT THE AFFORDANCE. "Do not immediately re-take the one that just finished" is
    // right for an object affordance, which owns its node. But room_concept publishes EVERY standpoint
    // through the single `afford_room` node, so keying this on the NODE ID blacklists ALL room
    // exploration from the first completion onwards. Measured live 2026-08-19 19:27: room held one
    // offer 1.96 m away for 60 s straight while the consumer logged
    //     tgt=0 plan=0 reject=just-completed elig=0
    // every 200 ms, robot moved 0.47 m. The only thing that clears this is the yield, and the yield
    // needs `suppressed_target`, which is assigned only inside the Offered branch the candidate never
    // reaches — so it never cleared.
    // ★So remember WHICH STANDPOINT finished, not merely which node. A different cell on the same node
    // is a different affordance and must be selectable at once. Same category error as suppress_target
    // (node-keyed, retires the whole channel) and as the first refusal-hold attempt.
    std::uint64_t last_completed_id_ = 0;   // skip this one on the next selection (see suppressed_name)
    // ★★★THE STANDPOINT WE ARE ACTUALLY EXECUTING, captured when we CLAIM it. mark_reached used to
    // recover it by RE-READING the node — but the producer overwrites that node with the NEXT cell as
    // soon as it sees the completion, so the read returned the incoming cell and we recorded it as
    // "just completed". We then suppressed exactly the cell we were supposed to take. Measured
    // 2026-08-19: robot arrived at (-1.00,+3.62) to 0.11 m, room offered (-3.50,+1.62) — 3.2 m away —
    // and the consumer rejected it as just-completed, 72% of cycles, robot issuing NO command for 48 s.
    // ★Identity must be carried as DATA at the moment of the decision, never inferred later from
    // mutable shared state. Reading a shared register to find out what you yourself just did is a race.
    float claimed_x_ = 0.f, claimed_y_ = 0.f;
    bool  claimed_pose_known_ = false;
    float last_completed_x_ = 0.f, last_completed_y_ = 0.f;
    bool  last_completed_pose_known_ = false;
    // node id -> selection rounds still to skip. See suppress_target: the consumer could not reach it.
    std::map<std::uint64_t, int> unreachable_rounds_;
    // ★★★WHEN EACH AFFORDANCE WAS LAST REFUSED. A refusal is a statement about the robot's CURRENT
    // situation, so re-offering it to the executor on the very next cycle is guaranteed to be refused
    // again — nothing can have changed in 50 ms. Without this the "yield rather than idle" rule below
    // re-takes the affordance it just refused, and producer and consumer spin at loop rate:
    //     REFUSED: already at this standpoint on the first cycle
    //     NONE ELIGIBLE — [afford_room (JustCompleted)]
    //     'afford_room' was the only affordance on offer — taking it again rather than idling
    //     REFUSED: already at this standpoint ...            (measured: ~104 completions/min)
    // ★The yield itself is RIGHT — a rule that can halt the agent for ever is a deadlock, not a
    // preference. What it lacked was any notion of time: it is only safe to retry once something
    // could have changed. So the yield still happens, just not before then.
    // ★KEYED ON THE STANDPOINT, NOT THE NODE. room_concept publishes EVERY standpoint through the one
    // `afford_room` node, so blocking the node would freeze all exploration for the hold; blocking the
    // POSE blocks only the spot that was actually refused and lets a different cell through at once.
    struct RefusedSpot { float x = 0.f, y = 0.f; std::uint64_t when_ms = 0; };
    std::map<std::uint64_t, RefusedSpot> refused_at_ms_;
    // ── MAP-ONLY VERDICTS, remembered against the map that produced them (see note_map_verdict) ──
    // (cell, pose) pairs already decided against on the CURRENT grid. Dropped whole when the grid's
    // identity moves: an answer cannot outlive the thing that made it true.
    std::set<std::pair<std::int64_t, std::int64_t>> map_verdicts_;
    std::size_t map_verdict_hash_ = 0;
    std::size_t map_identity_ = 0;
    std::string   suppressed_name_;         // what that skip cost, for the viewer
    std::uint64_t last_selected_id_ = 0;   // for commitment hysteresis across cycles
    std::vector<Candidate> last_candidates_;
    std::string last_reject_reason_;   // all affordances evaluated in the last select_target()
    State state_ = State::Idle;
    std::optional<bool> last_observed_active_;
    std::optional<bool> last_observed_pending_;
    std::string selected_target_debug_report_;
};

} // namespace rc