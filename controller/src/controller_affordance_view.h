#pragma once

/*
 * controller_affordance_view.h — what the affordance is DOING, as data.
 *
 * An affordance is a small program: claim it, drive to a standpoint, turn to face the object, then run
 * the contract's servo loop until a set of AND-ed clauses holds for stable_n cycles or timeout_ms runs
 * out. All of that already exists (see common/affordance_protocol/affordance_protocol.h) — what did not
 * exist was any way to WATCH it. The state was spread across active_contract_, lockon_, control_output
 * and half a dozen session flags, so "why is it sitting there?" could only be answered by reading four
 * files and a CSV.
 *
 * This is the single snapshot the session publishes each cycle and the panel renders. One source, so
 * the panel and the diagnostics cannot disagree — the same reason the drive modes were separated.
 *
 * ★THE POINT IS THE `detail` AND `blocked_why` FIELDS, NOT THE PROGRESS BAR. A bar says "not finished",
 * which is what you already knew. The number next to its target says WHICH quantity is wrong and by how
 * much, and blocked_why says what is stopping it. A whole day went into a robot that planned, tracked
 * and commanded 0.33 m/s while the base output was disarmed: every subsystem was reporting correctly and
 * the one line that would have said so did not exist. This is that line.
 */

#include <cstdint>
#include <string>
#include <vector>

namespace rc
{

struct AffordanceStepView
{
    // Skipped is a first-class outcome: an Orient affordance never navigates, and a step list that
    // silently omitted it would be a different program from the one the contract describes.
    enum class State { Pending, Active, Done, Failed, Skipped };

    std::string label;              // "navigate", "align", "distance_m >= 0.35"
    State       state = State::Pending;
    float       progress = -1.f;    // 0..1; NEGATIVE means "this step has no meaningful fraction"
    std::string detail;             // the NUMBERS: "1.42 m to go of 3.10", "yaw 23.4 deg (tol 3.4)"
    std::string blocked_why;        // why it is not advancing; empty when it is
    float       elapsed_s = 0.f;    // time in this step, so a stall is visible as a growing number
};

struct AffordanceExecution
{
    bool          active = false;       // an affordance is being executed right now
    std::string   affordance;           // node name
    std::string   object;               // the parent object it services
    std::string   policy;               // Reach | Servo | Orient — decides which steps apply
    std::string   phase;                // the lock-on phase, or the pipeline stage before it
    float         elapsed_s = 0.f;
    float         timeout_s = 0.f;      // contract timeout; 0 = none. Shown as a clock that runs out.
    bool          contract_known = false;   // false ⇒ the rows below are the pipeline only, no clauses
    std::vector<AffordanceStepView> steps;

    // A short history of finished executions, newest first: "<name>  12.4 s  locked / gave up / reached".
    // Kept because these runs last seconds — by the time a window is opened the interesting one is over.
    std::vector<std::string> recent;
};

}   // namespace rc
