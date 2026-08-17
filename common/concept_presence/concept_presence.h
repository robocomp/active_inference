#pragma once
/*
 * common/concept_presence/concept_presence.h — the concept-agent presence protocol, wired once.
 *
 * WHAT WAS DUPLICATED. Every one of the seven object-concept agents opened initialize() with the same ~134
 * lines: configure the coordinator, attach the state machine, install three hook structs, start. hood,
 * refrigerator and table were BYTE-IDENTICAL to each other; cabinet differed by 6 lines, chair and door by
 * 21, bottle by 25 — and almost all of that difference was comment wording, `const std::string &name` versus
 * `const std::string& name`, and line wrapping. The parts that were NOT cosmetic were two agents MISSING
 * things the other five had (see the defect note below). ~940 fleet lines expressing one protocol.
 *
 * WHY THE PROTOCOL IS WORTH SHARING RATHER THAN JUST DEDUPLICATING. It encodes four hard-won rules, each of
 * which cost a debugging session, and each of which was previously restated in seven places:
 *
 *   1. ★DEGRADED MUST DEBOUNCE. A transient required-peer flap at startup (handshake, DSR churn, a peer
 *      restarting) fires presenceLost momentarily and recovers. Tearing down on entry deleted the agent's
 *      own node and exited "cleanly" — the symptom is `[Graph] node deleted` right after `monitor started`.
 *      So Degraded schedules a grace timer and shuts down only if a required peer is STILL missing.
 *   2. ★A STALLED PRIMARY INPUT IS RECOVERABLE, NOT A SHUTDOWN CAUSE. It routes through Degraded too, so it
 *      is flagged; the grace timer then finds all peers present and declines to exit, and the FSM bounces on
 *      to Waiting where the admission gate holds until the producer returns.
 *   3. ★PEERS-READY IS NECESSARY BUT NOT SUFFICIENT. Admission also needs the primary input actually LIVE —
 *      a fresh frame, not merely a persisting node, which would re-admit straight into another stall.
 *   4. ★THE ONE-SHOT SWEEP IS ONE-SHOT. `on_first_operating` clears leftovers from a CRASHED PREVIOUS run,
 *      which can only be true the first time. On a Degraded→Operating bounce the nodes in the graph are THIS
 *      run's live ones; sweeping every bounce deletes and re-creates them, and the controller sees an
 *      affordance vanish mid-approach.
 *
 * ★★THE STATE LIVES HERE, AND THAT IS THE POINT OF DOING IT THIS WAY. `operating_since_ms`,
 * `stall_reported`, `degraded_from_input` and `first_operating_done` are set by the state machine and read by
 * the stall predicate. Leaving them as agent members while the transitions moved into shared code would be
 * the worst of both — so the protocol object owns them and exposes `operating_since_ms()` for the predicate.
 * Rule 4's flag in particular stops being something a new agent can forget to declare.
 *
 * ★★★DEFECT THIS EXTRACTION FOUND (fixed separately in 44fcd5c, before the move): chair and door ran the
 * stale-affordance sweep on EVERY entry to Operating. Their own comment described the bug as the intent —
 * "Stale-node sweep on (re)entering Operating". Rule 4 is now structural.
 *
 * MAIN-THREAD ONLY. Every hook runs on the agent's main thread (the coordinator's monitor is poll-based on
 * purpose — see CLAUDE.md), so nothing here needs synchronisation.
 */

#include <cstdint>
#include <format>
#include <functional>
#include <memory>
#include <print>
#include <string>
#include <utility>

#include <QDateTime>
#include <QObject>
#include <QString>
#include <QTimer>
#include <QtGlobal>

#include "../agent_presence_coordinator/agent_presence_coordinator.h"

namespace rc::presence
{

// How long a required-peer loss must PERSIST before the agent exits. Rule 1: a flap must not be fatal.
inline constexpr int kRequiredLossGraceMs = 3000;

// How often the "why am I still Waiting" line may be printed, so a long wait does not flood the log.
inline constexpr std::int64_t kWaitLogThrottleMs = 2000;

/*
 * The agent-side seams. All are required except `on_first_operating`.
 *
 * The primary input is described by three questions rather than a type, so the same protocol serves an agent
 * gated on a `masks` graph node and one gated on a LiDAR media stream:
 *   pump          — advance the ingest while Waiting. An agent that polls (no free-running reader thread)
 *                   MUST do this or liveness never advances and it can neither notice the producer coming
 *                   back nor avoid the re-admit→instant-re-stall flap.
 *   age_ms        — ms since the last frame; NEGATIVE means none has ever arrived. Used only for logging.
 *   live / stalled— the two admission predicates (see rc::stream). `stalled` reports the age it judged.
 */
struct ConceptHooks
{
    std::function<bool()>              shutting_down;
    std::function<void()>              pump_primary_input;
    std::function<std::int64_t()>      primary_age_ms;
    std::function<bool()>              primary_live;
    std::function<bool(std::int64_t*)> primary_stalled;
    std::function<void()>              emit_ready;          // emit presenceReady()  — a signal, so a hook
    std::function<void()>              emit_lost;           // emit presenceLost()
    std::function<void()>              compute;
    std::function<void()>              terminal_shutdown;
    std::function<void()>              on_first_operating;  // one-shot, post-sync stale sweep (may be empty)
    std::function<void(const std::string&, std::uint32_t)> on_optional_peer_lost;
    std::function<void(const std::string&, std::uint32_t)> on_optional_peer_ready;
};

class ConceptProtocol
{
public:
    // Install the whole protocol. `owner` is the QObject the grace timer is parented to (the agent), so a
    // destroyed agent cannot be called back by a pending timer.
    void wire(AgentPresenceCoordinator& coord, QStateMachine* state_machine, QObject* owner,
              const ConfigLoader& config_loader, std::shared_ptr<DSR::DSRGraph> graph,
              std::uint32_t local_agent_id, ConceptHooks hooks)
    {
        coord_ = &coord;
        hooks_ = std::move(hooks);

        coord.configure(config_loader, std::move(graph), local_agent_id);
        // Colour this agent's graph node by its live health: the coordinator already publishes the presence
        // lifecycle, this adds the external FSM axis (Initialize/Compute/Emergency/Restore). Discovery is via
        // objectName(), so regenerating genericworker cannot break it.
        coord.attach_state_machine(state_machine);

        coord.set_transition_hooks({
            // Rule 3: peers-ready is necessary, not sufficient. Declines SILENTLY — this fires on every
            // presence event, and on_waiting_loop is what pumps and re-polls until the producer is real.
            .request_presence_ready = [this] { if (hooks_.primary_live()) hooks_.emit_ready(); },
            .request_presence_lost  = [this] { hooks_.emit_lost(); },
        });

        coord.set_peer_hooks({
            .on_peer_restarted = [](std::uint32_t id) { qInfo() << "[Presence] peer" << id << "restarted"; },
            .on_optional_peer_lost  = [this](const std::string& name, std::uint32_t id)
                                      { if (hooks_.on_optional_peer_lost) hooks_.on_optional_peer_lost(name, id); },
            .on_optional_peer_ready = [this](const std::string& name, std::uint32_t id)
                                      { if (hooks_.on_optional_peer_ready) hooks_.on_optional_peer_ready(name, id); },
        });

        coord.set_lifecycle_hooks({
            .on_waiting_enter   = [this] { on_waiting_enter(); },
            .on_waiting_loop    = [this] { on_waiting_loop(); },
            .on_operating_enter = [this] { on_operating_enter(); },
            .on_operating_loop  = [this] { on_operating_loop(); },
            .on_degraded_enter  = [this, owner] { on_degraded_enter(owner); },
        });

        coord.start();
    }

    // Baseline for the cold-start stall grace: the stall predicate measures from OPERATING ENTRY, so that a
    // producer still starting up is not read as a stall. Owned here because the transitions set it.
    [[nodiscard]] std::int64_t operating_since_ms() const { return operating_since_ms_; }

private:
    void on_waiting_enter() const
    {
        const auto missing = coord_->missing_required_names();
        if (missing.empty())
        {
            qInfo("[SM] -> Waiting");
            return;
        }
        QString m;
        for (const auto& label : missing)
            m += " " + QString::fromStdString(label);
        qInfo() << "[SM] -> Waiting (missing:" << m.trimmed() << ")";
    }

    void on_waiting_loop()
    {
        if (hooks_.shutting_down())
            return;
        if (hooks_.pump_primary_input)
            hooks_.pump_primary_input();                       // see ConceptHooks::pump — polling agents MUST

        const bool peers_ready = coord_->all_required_ready();
        const bool input_live  = hooks_.primary_live();
        if (peers_ready and input_live)                        // rule 3: both, or stay put
        {
            hooks_.emit_ready();
            return;
        }
        const auto now = QDateTime::currentMSecsSinceEpoch();
        if (now - last_wait_log_ms_ < kWaitLogThrottleMs)
            return;
        last_wait_log_ms_ = now;
        const auto age = hooks_.primary_age_ms ? hooks_.primary_age_ms() : -1;
        std::println("[SM] Waiting — peers {} | input {} (age {} ms)",
                     peers_ready ? "OK" : "MISSING", input_live ? "LIVE" : "stale",
                     age < 0 ? std::string("none") : std::to_string(age));
    }

    void on_operating_enter()
    {
        operating_since_ms_ = QDateTime::currentMSecsSinceEpoch();   // cold-start grace baseline
        stall_reported_     = false;                                 // re-arm the one-shot
        qInfo("[SM] -> Operating: all required peers present");

        // Rule 4. Post-sync on purpose: an initialize()-time sweep can run BEFORE leaked nodes arrive from
        // the persistent DSR server, and on_operating_enter fires before the first compute(), so this run has
        // created none of its own yet.
        if (not first_operating_done_)
        {
            first_operating_done_ = true;
            if (hooks_.on_first_operating)
                hooks_.on_first_operating();
        }
    }

    void on_operating_loop()
    {
        // Rule 2 / the primary-input gate: a dead producer means acting on stale evidence, so DEMOTE
        // (Operating→Degraded→Waiting) rather than re-integrating frozen frames. Belief Σ-aging is a
        // different axis and is untouched.
        std::int64_t age = 0;
        if (not stall_reported_ and hooks_.primary_stalled(&age))
        {
            stall_reported_      = true;
            degraded_from_input_ = true;
            std::println("[SM] Operating -> Waiting: primary input STALLED ({}) — not integrating stale evidence",
                         age < 0 ? std::string("no frame ever arrived")
                                 : std::format("last frame {} ms ago", age));
            hooks_.emit_lost();
            return;
        }
        hooks_.compute();
    }

    void on_degraded_enter(QObject* owner)
    {
        if (hooks_.shutting_down())
            return;
        if (degraded_from_input_)
        {
            // Rule 2: recoverable. Cleared here so the grace timer below sees all peers present and declines.
            degraded_from_input_ = false;
            qInfo("[SM] -> Degraded (input stall, peers intact) — passing through to Waiting, "
                  "re-admit on producer return");
        }
        else
            qInfo("[SM] -> Degraded: required peer lost — %d ms grace before shutdown", kRequiredLossGraceMs);

        // Rule 1: the debounce. Parented to `owner`, so a destroyed agent is never called back.
        QTimer::singleShot(kRequiredLossGraceMs, owner, [this]
        {
            if (hooks_.shutting_down())
                return;
            if (coord_->all_required_ready())
            {
                qInfo("[SM] required peers recovered during grace — staying alive");
                return;
            }
            qWarning("[SM] required peer still missing after grace — shutting down cleanly");
            hooks_.terminal_shutdown();
        });
    }

    AgentPresenceCoordinator* coord_ = nullptr;
    ConceptHooks              hooks_{};

    std::int64_t operating_since_ms_   = 0;
    std::int64_t last_wait_log_ms_     = 0;
    bool         stall_reported_       = false;   // one-shot per stall episode, re-armed on Operating entry
    bool         degraded_from_input_  = false;   // Degraded reason: recoverable input stall vs real peer loss
    bool         first_operating_done_ = false;   // ★rule 4 — a new agent can no longer forget to declare it
};

}  // namespace rc::presence
