/*
 * robot_capability.h — what the BASE CAN DO, read off the robot node, and the audit that holds a
 * consumer's own limits against it.
 *
 * ★★★CAPABILITY IS NOT POLICY, AND SUBSTITUTING ONE FOR THE OTHER IS THE DEFECT THIS EXISTS TO STOP.
 *
 *      CAPABILITY   what the hardware CAN do    owned by the base component (SVD48VBase's own
 *                                               etc/config_*.toml — the SAME file the real robot
 *                                               runs), read by robot_concept, published in SI on
 *                                               the robot node.
 *      POLICY       how fast we CHOOSE to go    a preference, and it stays in the consumer's config
 *                                               where a person can change it.
 *
 * A consumer asserts `policy <= capability`. It must never REPLACE one with the other. That is not a
 * style rule: room_concept's pose clamp took `POSE_CLAMP_W_MAX` from the controller's MaxRotSpeed
 * (`room_config.h:256` still says "// rad/s — controller MaxRotSpeed"), so a comfort setting became a
 * hard rate limiter on the PUBLISHED yaw — 0.8 rad/s while the robot was turning at 3.5. Nothing in
 * either agent was wrong on its own; the number had simply travelled from a place that owned it to a
 * place that did not.
 *
 * ★THIS HEADER REPORTS. IT DOES NOT RETUNE. `PolicyAudit::check` returns the policy unchanged and says
 * loudly when it exceeds the capability. Silently clamping four live control constants on the strength
 * of a first read is a behaviour change, and one that would land without anyone choosing it; the point
 * of the audit is to put the two numbers side by side so the choice can be made deliberately. The
 * exceedances are real and already known — see the measured table in controller_session.cpp — so an
 * auto-clamp here would not be a safety net, it would be an unreviewed retune of a running robot.
 *
 * ★ABSENT IS NOT ZERO. Every field is optional, and the producer publishes nothing for a key its base
 * config does not carry. Absent means "unknown — keep your own constant", never "0". A consumer that
 * defaulted these to 0 would clamp the robot to a standstill on a base that simply said less than
 * expected.
 *
 * ★ALL SI. That base config is in mm and mm/s; robot_concept converts on publish, so everything here
 * is m, m/s, m/s^2, rad/s. `maxRotSpeed` is the one field already in rad/s in the source file.
 *
 * ⚠ Requires the six `robot_max_*` / `robot_wheel_radius` / `robot_axes_length` REGISTER_TYPE lines in
 * cortex. Until cortex is reinstalled this header does not compile — the same gate the producer is
 * behind (robot_concept commit e4a5d95).
 */

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "dsr/api/dsr_api.h"

namespace rc
{

// ── WHAT THE HARDWARE CAN DO ──────────────────────────────────────────────────────────────────────
// Measured values, for orientation when reading a log (2026-08-29, both real files):
//                      Shadow / Differential      P3Bot / Omnidirectional
//   linear speed       0.9 m/s                    0.7 m/s
//   rot speed          2.0 rad/s                  1.5 rad/s
//   linear accel       0.5 m/s^2                  0.35 m/s^2
//   linear decel       1.0 m/s^2                  1.0 m/s^2
//   wheel radius       0.100 m                    0.080 m
//   axes length        0.518 m                    0.475 m
//   holonomic          false                      true
struct BaseCapability
{
    std::optional<float> max_linear_speed_mps;    // bounds forward AND lateral on an omni base
    std::optional<float> max_rot_speed_rps;
    std::optional<float> max_linear_accel_mps2;
    std::optional<float> max_linear_decel_mps2;
    std::optional<float> wheel_radius_m;
    std::optional<float> axes_length_m;
    // Derived by the producer from the base config's `baseType`, never declared by hand: the base
    // component already says "Differential"/"Omnidirectional" and always did.
    std::optional<bool>  holonomic;

    // True when the robot node carried at least one of these — i.e. the producer is publishing and the
    // consumer is entitled to say something about its own limits. False means the read happened but the
    // channel is not up (old robot_concept, no `Agent.base_config_file`, or a stale cortex).
    bool any() const
    {
        return max_linear_speed_mps.has_value() or max_rot_speed_rps.has_value()
            or max_linear_accel_mps2.has_value() or max_linear_decel_mps2.has_value()
            or wheel_radius_m.has_value() or axes_length_m.has_value() or holonomic.has_value();
    }
};

// Read the capability block off the robot node. Never throws; a missing node or a missing attribute
// yields an empty optional, which every caller must read as "keep your own constant".
BaseCapability read_base_capability(DSR::DSRGraph &graph, std::uint64_t robot_id);

// ── THE ASSERTION ─────────────────────────────────────────────────────────────────────────────────
// One row per consumer limit held against the capability that bounds it. Kept as data rather than
// printed on the spot so the whole comparison arrives as ONE block: a line per number in isolation is
// exactly how a 3x exceedance stayed invisible for as long as it did.
struct PolicyRow
{
    std::string what;                    // the consumer's config key, verbatim, so it can be edited
    std::string unit;
    float policy = 0.f;
    std::optional<float> capability;     // absent = the base did not say
    bool exceeds = false;                // policy > capability, to a tolerance
};

class PolicyAudit
{
public:
    // Records the pair and returns `policy` UNCHANGED — see the header note: this reports, it does not
    // retune. `tol_frac` absorbs a float round-trip through the graph, not a real difference.
    float check(const char *what, const char *unit, float policy, std::optional<float> capability,
                float tol_frac = 1e-3f);

    const std::vector<PolicyRow> &rows() const { return rows_; }
    bool any_exceeds() const;

private:
    std::vector<PolicyRow> rows_;
};

// One block at qInfo, or qCritical when something exceeds. `who` names the consumer ("controller"), so
// a reader of the shared log knows whose policy is on the table. Prints the capability block even when
// no policy was checked against it — the geometry (wheel radius, track) has no policy and is still
// worth stating once, because it is the first per-robot statement of it anywhere in the fleet.
void log_base_capability(const BaseCapability &cap, const PolicyAudit &audit, const char *who);

}   // namespace rc
