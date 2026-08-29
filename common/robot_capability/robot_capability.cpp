/*
 * robot_capability.cpp — see robot_capability.h
 */

#include "robot_capability.h"

#include <algorithm>
#include <cmath>

#include <QtGlobal>
#include <QDebug>

namespace rc
{

BaseCapability read_base_capability(DSR::DSRGraph &graph, std::uint64_t robot_id)
{
    BaseCapability cap;
    if (robot_id == 0) return cap;
    const auto node = graph.get_node(robot_id);
    if (not node.has_value()) return cap;

    // Type-attributed reads, never runtime_checked_* (CLAUDE.md): a typo here would be a runtime throw
    // rather than a compile error, on a path that runs once at startup and would therefore be found late.
    cap.max_linear_speed_mps  = graph.get_attrib_by_name<robot_max_linear_speed_att>(*node);
    cap.max_rot_speed_rps     = graph.get_attrib_by_name<robot_max_rot_speed_att>(*node);
    cap.max_linear_accel_mps2 = graph.get_attrib_by_name<robot_max_linear_accel_att>(*node);
    cap.max_linear_decel_mps2 = graph.get_attrib_by_name<robot_max_linear_decel_att>(*node);
    cap.wheel_radius_m        = graph.get_attrib_by_name<robot_wheel_radius_att>(*node);
    cap.axes_length_m         = graph.get_attrib_by_name<robot_axes_length_att>(*node);
    cap.holonomic             = graph.get_attrib_by_name<robot_holonomic_att>(*node);

    // A published ZERO is not a capability, it is a producer that read a key it could not parse. Treat it
    // as absent rather than as "this robot cannot move", which is the reading that would clamp a healthy
    // base to a standstill. (The producer does not publish absent keys at all, so this only ever fires on
    // a genuinely bad value.)
    const auto drop_nonpositive = [](std::optional<float> &v, const char *what)
    {
        if (v.has_value() and not (*v > 0.f))
        {
            qWarning() << "[capability]" << what << "published as" << *v
                       << "— not a usable limit; treating it as ABSENT.";
            v.reset();
        }
    };
    drop_nonpositive(cap.max_linear_speed_mps,  "robot_max_linear_speed");
    drop_nonpositive(cap.max_rot_speed_rps,     "robot_max_rot_speed");
    drop_nonpositive(cap.max_linear_accel_mps2, "robot_max_linear_accel");
    drop_nonpositive(cap.max_linear_decel_mps2, "robot_max_linear_decel");
    drop_nonpositive(cap.wheel_radius_m,        "robot_wheel_radius");
    drop_nonpositive(cap.axes_length_m,         "robot_axes_length");
    return cap;
}

float PolicyAudit::check(const char *what, const char *unit, float policy,
                         std::optional<float> capability, float tol_frac)
{
    PolicyRow row;
    row.what = what ? what : "";
    row.unit = unit ? unit : "";
    row.policy = policy;
    row.capability = capability;
    if (capability.has_value())
        row.exceeds = policy > *capability * (1.f + std::max(0.f, tol_frac));
    rows_.push_back(std::move(row));
    return policy;      // ★UNCHANGED, on purpose. See the header.
}

bool PolicyAudit::any_exceeds() const
{
    return std::ranges::any_of(rows_, [](const PolicyRow &r) { return r.exceeds; });
}

void log_base_capability(const BaseCapability &cap, const PolicyAudit &audit, const char *who)
{
    const QString whom = QString::fromUtf8(who ? who : "consumer");

    if (not cap.any())
    {
        // Said once, plainly, and NOT as an error: an older robot_concept, a config without
        // Agent.base_config_file, or a cortex that predates the attributes all land here, and in every
        // one of those cases the right behaviour is exactly today's behaviour.
        qInfo().noquote()
            << "[capability] robot node carries no base capability —" << whom
            << "keeps its own constants. (Producer: robot_concept + Agent.base_config_file.)";
        return;
    }

    const auto num = [](std::optional<float> v, int prec = 3)
    { return v.has_value() ? QString::number(*v, 'f', prec) : QStringLiteral("(absent)"); };

    QString block;
    block += QStringLiteral("\n[capability] what THIS base can do, from the base component's own config\n");
    block += QStringLiteral("  linear speed  %1 m/s\n").arg(num(cap.max_linear_speed_mps));
    block += QStringLiteral("  rot speed     %1 rad/s\n").arg(num(cap.max_rot_speed_rps));
    block += QStringLiteral("  linear accel  %1 m/s^2\n").arg(num(cap.max_linear_accel_mps2));
    block += QStringLiteral("  linear decel  %1 m/s^2\n").arg(num(cap.max_linear_decel_mps2));
    block += QStringLiteral("  wheel radius  %1 m       track %2 m\n")
                 .arg(num(cap.wheel_radius_m), num(cap.axes_length_m));
    block += QStringLiteral("  holonomic     %1\n")
                 .arg(cap.holonomic.has_value() ? (*cap.holonomic ? "true (lateral DOF)" : "false (no lateral DOF)")
                                                : "(absent)");

    if (not audit.rows().empty())
    {
        block += QStringLiteral("[policy] %1's own limits, held against the above. "
                                "A policy BELOW capability is normal and is the point; a policy ABOVE it "
                                "is not a limit at all, because the base saturates first.\n").arg(whom);
        for (const auto &r : audit.rows())
        {
            const QString mark = r.exceeds ? QStringLiteral("  ⚠ EXCEEDS")
                               : (r.capability.has_value() ? QStringLiteral("    ok    ")
                                                           : QStringLiteral("    ?     "));
            block += QStringLiteral("%1  %2 = %3 %4   (capability %5)\n")
                         .arg(mark,
                              QString::fromStdString(r.what),
                              QString::number(r.policy, 'f', 3),
                              QString::fromStdString(r.unit),
                              r.capability.has_value() ? QString::number(*r.capability, 'f', 3)
                                                       : QStringLiteral("absent"));
        }
        block += QStringLiteral("[policy] NOTHING WAS CHANGED by this report. These are live control "
                                "constants; retuning them is a decision, not a side effect of reading.\n");
    }

    if (audit.any_exceeds())
        qCritical().noquote() << block;
    else
        qInfo().noquote() << block;
}

}   // namespace rc
