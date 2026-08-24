#pragma once
// ─────────────────────────────────────────────────────────────────────────────────────────────────
// THE CALIBRATION PIVOT, AS AN AFFORDANCE room PUBLISHES.
//
// The manoeuvre that measures the rotation scale is a CLOSURE PIVOT: turn through N complete turns
// and stop on the heading you started from, and the robot turned exactly 2*pi*N radians — a fact
// about turning, independent of the map, the survey and the localiser. Comparing the odometry's
// accumulated dtheta against that gives s_omega with nothing estimated in the denominator.
//
// ★IT NEEDS NO NEW POLICY AND NO CONTROLLER CHANGE. Policy::Orient already exists and the controller
// already executes it — rotate in place toward a target bearing, no navigation, completes when the
// heading is reached. So the manoeuvre is a SEQUENCE OF ORDINARY AFFORDANCES: offer a bearing 120
// degrees on, let the robot turn, offer the next. Twelve of those is four turns and the robot is back
// where it started. (No producer has ever authored an Orient contract, so that path is exercised here
// for the first time — which is a reason to watch it, not a reason to invent a new one.)
//
// ★WHY 120 AND NOT 180. At exactly half a turn the short-way rotation is ambiguous and the robot can
// take it back the way it came, oscillating instead of accumulating. At 120 every step turns the same
// way. Nothing else about the number matters — it is a third of a turn, so three steps close a turn.
//
// ★THE GATE IS THE CONSUMER'S OWN TEST, NOT A SECOND OPINION HERE. A pivot needs room to turn in, and
// this producer cannot know whether the body fits through every heading at a given spot — the
// controller can, and already asks exactly that (can_turn_here / nearest_rotatable) for any target
// whose policy ends in a turn. So the offer goes out at the robot's current pose and an `infeasible`
// reply is taken at face value: this spot will not do, stop asking, wait until the robot has moved
// somewhere else in the course of its ordinary work. That is the opportunism the whole design rests
// on — the calibration never drives anywhere, it waits for the robot to be somewhere suitable.
//
// ★AND IT NEVER ASSUMES THE TURN HAPPENED. The closure is only a fact if the robot really came back
// to the starting heading, so the sequence tracks the MEASURED heading, not the count of steps
// issued. A refused step, a preemption, or a step the robot took only halfway all leave the
// accumulated turn short, and the pivot simply is not closed yet.
// ─────────────────────────────────────────────────────────────────────────────────────────────────
#include "../../common/motion_calib/scale_estimator.h"

#include <cmath>
#include <optional>

namespace rc::calib
{

struct PivotParams
{
    // ★A CONSTANT-RATE PIVOT CANNOT TEACH A GYRO BIAS. Scale and bias land on the same component of
    // the correction and are separated ONLY by rotation-vs-time; at a fixed rate the two columns are
    // collinear and no estimator can tell them apart. The joint solve reports this honestly (its
    // normalised condition number went 14.5 -> 216.4 on exactly that case in the self-test), but the
    // manoeuvre still buys nothing for the bias. A pivot that VARIED its rate would. Not built:
    // the affordance offers one Orient contract with a fixed policy, and a varying-rate manoeuvre is
    // a different contract, not a parameter of this one.
    /// A third of a turn per step: same direction every time, three steps to the turn.
    double step_rad = 2.0 * M_PI / 3.0;
    /// Complete turns per pivot. Four is what the hand-run measurement used, and it sets the
    /// resolution: closing to 17 degrees over 1440 resolves the scale to about 1.2%.
    int turns = 4;
    /// How close the heading must come back for the pivot to count as closed. This is not a taste
    /// setting — it is the localiser's own heading accuracy, and it becomes the measurement's
    /// resolution, reported with the answer rather than hidden in it.
    double closure_tolerance_rad = 0.05;
};

/// What the pivot measured, once it closed.
struct PivotClosure
{
    double s_omega    = 0.0;   ///< (odometry turn) / (2*pi*N) - 1
    double resolution = 0.0;   ///< |heading miss| / |2*pi*N| — the finest claim this run supports
    double turned_rad = 0.0;   ///< what the odometry accumulated
    double truth_rad  = 0.0;   ///< 2*pi*N, the fact
    bool   usable     = false; ///< false when the scale is smaller than the run can resolve
};

class PivotAffordance
{
public:
    enum class State { Idle, Offering, Closed, SpotRefused };

    /// For logs. A state a human can read is the difference between a trace and a puzzle.
    [[nodiscard]] static constexpr const char* to_string(State s)
    {
        switch (s)
        {
            case State::Idle:        return "Idle";
            case State::Offering:    return "Offering";
            case State::Closed:      return "Closed";
            case State::SpotRefused: return "SpotRefused";
        }
        return "?";
    }

    explicit PivotAffordance(PivotParams p = PivotParams{}) : p_(p) {}

    [[nodiscard]] State state() const { return state_; }
    [[nodiscard]] int   steps_issued() const { return steps_; }
    [[nodiscard]] double accumulated_rad() const { return ref_turn_; }

    /// Should an offer go out this cycle, and at what bearing?
    ///
    /// `gain_nats` is what the manoeuvre is worth BEYOND the excitation the robot's ordinary tours
    /// are already expected to deliver — the marginal rule. The caller computes it from the
    /// estimator; this object only decides whether there is a manoeuvre to make and where it points.
    /// Returning nullopt means "nothing to offer", which is the normal state of a calibrated robot.
    /// ★NOT const, and the reason matters: the pivot BEGINS when the first offer goes out, so the
    /// starting heading is latched here. Latching it on the first OUTCOME instead loses the first
    /// step — twelve 120-degree steps then accumulate 1320 degrees, not 1440, and the closure never
    /// arrives. Caught by the unit case that counts the degrees rather than the steps.
    std::optional<double> next_bearing(double robot_heading_rad, double gain_nats)
    {
        if (state_ == State::SpotRefused or state_ == State::Closed) return std::nullopt;
        // ★NO THRESHOLD ON THE GAIN HERE. The offer competes on its gain in the same arbitration as
        // every exploration standpoint; if it is worth less than a cell, the cell wins and this never
        // runs. The only thing asked here is whether there is any gain at all to advertise.
        if (not (gain_nats > 0.0)) return std::nullopt;
        if (state_ == State::Idle)
        {
            state_ = State::Offering;
            start_heading_ = last_heading_ = robot_heading_rad;
            ref_turn_ = odom_turn_ = 0.0;
            steps_ = 0;
        }
        // ★★★ANCHORED TO THE STARTING HEADING, NOT TO WHERE THE LAST STEP HAPPENED TO STOP.
        // This returned `robot_heading + step`, and that quietly made closure impossible. The executor
        // completes an Orient when it is within its aligned band (0.05 rad = 2.9 deg), so every step
        // lands a couple of degrees short — and asking for "120 more from wherever you are" bakes each
        // shortfall in for ever. Measured live 2026-08-24, thirteen consecutive steps: 117.4 deg each,
        // never 120. Twelve of those is 1409 deg, so the robot ends 31 deg from where it started and
        // the heading test can never pass. The pivot ran and ran and could not finish.
        // Absolute bearings make the error self-correcting: step k asks for start + k*120 whatever the
        // last step achieved, so a step that fell short is made up by the next one, and step twelve
        // asks for start + 1440 deg — which IS the starting heading. The closure is built into the
        // sequence instead of being hoped for at the end of it.
        return wrap(start_heading_ + static_cast<double>(steps_ + 1) * p_.step_rad);
    }

    /// The consumer answered. `heading_rad` is the robot's measured heading NOW, and it is what the
    /// sequence advances on — never the step that was asked for.
    /// `ref_turn_rad` is the reference turn measured DURING this step, not differenced across the
    /// gap since the last one -- see the note in calib_channel.h::note_motion. `heading_rad` is still
    /// the measured heading now, because the closure test asks where the robot ENDED UP.
    void on_outcome(bool satisfied, bool spot_infeasible, double heading_rad, double odom_turn_rad,
                    double ref_turn_rad)
    {
        if (spot_infeasible)
        {
            // The controller says the body cannot turn here. Believe it, stop asking, and wait for
            // the robot to be somewhere else — it will be, in the course of its ordinary work.
            state_ = State::SpotRefused;
            return;
        }
        if (not satisfied) return;                 // timeout, preemption, anything else: nothing moved on

        ref_turn_ += ref_turn_rad;
        odom_turn_ += odom_turn_rad;
        last_heading_ = heading_rad;
        ++steps_;
        state_ = State::Offering;

        // Closed when the robot has been round at least the requested number of times AND the heading
        // has come back. Both conditions, because either alone is satisfiable by standing still.
        // ★COUNT THE TURNS, DO NOT COMPARE AGAINST AN ASSERTED TOTAL. The old test asked whether
        // |ref_turn_| had reached 2*pi*turns minus a tolerance, which pairs badly with the closure()
        // below asserting the same constant as the TRUTH — see there.
        if (std::abs(whole_turns()) >= p_.turns
            and std::abs(wrap(heading_rad - start_heading_)) <= p_.closure_tolerance_rad)
            state_ = State::Closed;
    }

    /// The robot has moved somewhere else; a spot that would not do no longer applies.
    void robot_moved() { if (state_ == State::SpotRefused) state_ = State::Idle; }

    [[nodiscard]] PivotClosure closure() const
    {
        PivotClosure c;
        if (state_ != State::Closed or steps_ == 0) return c;
        // ★★★THE TRUTH IS HOW MANY TURNS THE ROBOT ACTUALLY MADE, NOT HOW MANY WERE ASKED FOR.
        // This asserted 2*pi*turns — the CONFIGURED count — as the denominator of the scale. The whole
        // argument of a closure pivot is "the heading came back, therefore the robot turned a whole
        // number of turns"; WHICH whole number is a fact to be counted, not a parameter. With steps
        // falling short (see next_bearing) the sequence closes after thirteen turns, not four, and
        // dividing thirteen turns of odometry by four asserted ones reports s_omega = +226% — a
        // confident, catastrophically wrong calibration, which is far worse than a pivot that never
        // finishes. Rounding ref_turn_ to the nearest whole turn is exact here: the localiser's
        // heading error is degrees and the spacing is 360.
        c.truth_rad  = 2.0 * M_PI * whole_turns();
        c.turned_rad = odom_turn_;
        c.s_omega    = odom_turn_ / c.truth_rad - 1.0;
        c.resolution = std::abs(wrap(last_heading_ - start_heading_)) / std::abs(c.truth_rad);
        // ★A SCALE SMALLER THAN THE CLOSURE CAN RESOLVE IS NOT A MEASUREMENT. Reported, not silently
        // rounded to zero and not quietly quoted as if it were significant.
        c.usable     = std::abs(c.s_omega) > c.resolution;
        return c;
    }

    void reset() { state_ = State::Idle; steps_ = 0; ref_turn_ = odom_turn_ = 0.0; }

    /// Whole turns the reference heading says were made, signed. The closure's own count.
    [[nodiscard]] double whole_turns() const { return std::round(ref_turn_ / (2.0 * M_PI)); }

private:
    static double wrap(double a) { while (a > M_PI) a -= 2*M_PI; while (a < -M_PI) a += 2*M_PI; return a; }

    PivotParams p_;
    State  state_ = State::Idle;
    int    steps_ = 0;
    double start_heading_ = 0.0, last_heading_ = 0.0;
    double ref_turn_ = 0.0;     ///< measured turn, accumulated from the heading — never from the count
    double odom_turn_ = 0.0;    ///< the odometry's own accumulation over the same manoeuvre
};

}   // namespace rc::calib
