#pragma once
/*
 * agent_status.h — the SINGLE mapping authority from an agent's live state to the colour its node
 * shows in the DSR graph view.
 *
 * Two independent axes are tracked, because neither alone answers "is this agent working?":
 *
 *   - Fsm       the generated GRAFCET/QStateMachine step (genericworker.cpp): Initialize, Compute,
 *               Emergency, Restore.
 *   - Presence  the agent-presence lifecycle (AgentPresenceCoordinator): Waiting, Operating,
 *               Degraded.
 *
 * They are NOT redundant: an agent parked in on_waiting_loop waiting for required peers is still in
 * FSM state "Compute" and would look perfectly healthy on the FSM axis alone. So the colour is the
 * WORST of the two axes — green requires Compute AND Operating.
 *
 * Colour names must have a valid Qt/SVG "dark<name>" counterpart, because GraphNode::set_color()
 * derives its pulse colour as "dark" + name. green/red/orange/gray all qualify; yellow does NOT
 * (there is no "darkyellow").
 */

#include <string>
#include <string_view>

namespace rc::agent_status
{

enum class Fsm
{
    Unknown,      // not attached to a state machine yet, or an unrecognised step name
    Initialize,
    Compute,
    Emergency,
    Restore,
};

enum class Presence
{
    Unknown,      // coordinator not started yet
    Waiting,
    Operating,
    Degraded,
};

// Colour of an agent node whose heartbeat has gone stale. This one is NOT self-published — a dead
// agent publishes nothing — so it is applied by the observer (see agent_status_overlay.h).
inline constexpr std::string_view STALE_COLOR = "gray";

// Severity ladder. Higher is worse; the published colour is the worse of the two axes.
enum class Severity
{
    Ok = 0,       // green
    Warn = 1,     // orange
    Bad = 2,      // red
};

[[nodiscard]] constexpr Severity severity_of(const Fsm f) noexcept
{
    switch (f)
    {
        case Fsm::Compute:    return Severity::Ok;
        case Fsm::Initialize: return Severity::Warn;   // transient, but not yet doing work
        case Fsm::Restore:    return Severity::Warn;   // recovering from Emergency
        case Fsm::Emergency:  return Severity::Bad;
        case Fsm::Unknown:    return Severity::Warn;   // "not known to be healthy" is not "healthy"
    }
    return Severity::Warn;
}

[[nodiscard]] constexpr Severity severity_of(const Presence p) noexcept
{
    switch (p)
    {
        case Presence::Operating: return Severity::Ok;
        case Presence::Waiting:   return Severity::Warn;
        case Presence::Degraded:  return Severity::Bad;
        case Presence::Unknown:   return Severity::Warn;
    }
    return Severity::Warn;
}

[[nodiscard]] constexpr std::string_view color_of(const Severity s) noexcept
{
    switch (s)
    {
        case Severity::Ok:   return "green";
        case Severity::Warn: return "orange";
        case Severity::Bad:  return "red";
    }
    return "orange";
}

// Worst-wins.
[[nodiscard]] constexpr std::string_view color_for(const Fsm f, const Presence p) noexcept
{
    const auto sf = severity_of(f);
    const auto sp = severity_of(p);
    return color_of(sf > sp ? sf : sp);
}

// ── Names written to / read from the graph ────────────────────────────────────────────────────
// The FSM names match the GRAFCETStep objectName()s built in genericworker.cpp, so parse/format
// round-trips exactly.

[[nodiscard]] constexpr std::string_view name_of(const Fsm f) noexcept
{
    switch (f)
    {
        case Fsm::Initialize: return "Initialize";
        case Fsm::Compute:    return "Compute";
        case Fsm::Emergency:  return "Emergency";
        case Fsm::Restore:    return "Restore";
        case Fsm::Unknown:    return "Unknown";
    }
    return "Unknown";
}

[[nodiscard]] constexpr std::string_view name_of(const Presence p) noexcept
{
    switch (p)
    {
        case Presence::Waiting:   return "Waiting";
        case Presence::Operating: return "Operating";
        case Presence::Degraded:  return "Degraded";
        case Presence::Unknown:   return "Unknown";
    }
    return "Unknown";
}

[[nodiscard]] inline Fsm fsm_from_name(const std::string_view s) noexcept
{
    if (s == "Initialize") return Fsm::Initialize;
    if (s == "Compute")    return Fsm::Compute;
    if (s == "Emergency")  return Fsm::Emergency;
    if (s == "Restore")    return Fsm::Restore;
    return Fsm::Unknown;
}

[[nodiscard]] inline Presence presence_from_name(const std::string_view s) noexcept
{
    if (s == "Waiting")   return Presence::Waiting;
    if (s == "Operating") return Presence::Operating;
    if (s == "Degraded")  return Presence::Degraded;
    return Presence::Unknown;
}

}   // namespace rc::agent_status
