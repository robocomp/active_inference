#ifndef RC_PLAIN_TRACKER_H
#define RC_PLAIN_TRACKER_H

#include "tracker.h"

namespace rc
{

// THE PLAIN TRACKER — follows the route as smoothly as it can, and does nothing else.
//
// ★PREMISE, and it is the entire design: THE PATH IS SELF-ADJUSTING AND SAFE. The planner certifies the
// route with the footprint predicate, the band deforms it around what moves, and RouteFollower::repair
// splices a new stretch when a section is genuinely impassable. So this tracker performs no avoidance:
// no clearance bound, no gate, no scan. Note the constructor takes a PathWorld and NOT a FieldWorld —
// it is structurally incapable of querying an obstacle.
//
// ★WHY, and it cost four laps to learn. Its predecessor (ControlMode "route") carried its own clearance
// bound, and EVERY route-mode failure came from that bound — never from the steering law:
//   lap 1  omega was proportional to v, so a zero bound took the steering to zero with it. Deadlock:
//          568 of 1187 cycles frozen, with 0.50 m of clearance at the robot.
//   lap 2  a 0.10 m standoff forbade a route whose tightest CERTIFIED margin is 0.011 m. The planner
//          promises fit, not comfort.
//   lap 4  the bound scanned a FIXED 1 m at any speed, so one bad far-field sample stopped the robot
//          dead — and, stopped, it could not change the geometry that was stopping it. 178 stop
//          episodes of MEDIAN LENGTH 1 CYCLE, 24% of cycles at zero speed, cross-track +144% and
//          jerk +173% against the PD path it was built to beat.
//   lap 5  the bound was repaired to scan only the braking distance. The repair WORKED — stops fell to
//          19 episodes, 2.1% — and the lap was worse. Unbraked, mean commanded speed rose 0.35 -> 0.53,
//          pose sigma DOUBLED to 0.204 m, measured speed went non-physical on 27 cycles, and the robot
//          left the route by 4.3 m, forcing 9 replans. The broken bound had been holding the system
//          together by accident, as a speed limiter nobody had designed.
// The lesson was not "tune the bound". It is that avoidance and path-following are different jobs, and
// a tracker doing both duplicates the band — the same NULL that stage 1 measured when MPPI and the band
// shadowed each other. This one does path-following only, and is judged only on smoothness.
//
// ★WHAT PROTECTS THE ROBOT — all of it outside this class, by design:
//   • the planner's footprint predicate, at route build and at every repair
//   • the band, deforming the route around live obstacles
//   • detect_path_blockage -> replan, which runs immediately before this
//   • the pose-confidence ramp in controller_motion_commander (pose_xy_std_slow/stop), which scales adv
//     and rot down as the localiser's sigma grows. It is DOWNSTREAM of every mode, so this tracker must
//     NOT carry its own — that would be the identical duplication one layer lower. Lap 5's sigma blowup
//     belongs to it, and handing it a speed that is not already inflated is the most this can do.
class PlainTracker final : public Tracker
{
public:
    explicit PlainTracker(const PathWorld& world) : world_(world) {}

    const char* name() const override { return "plain"; }
    // ★RESET MEANS "RE-ACQUIRE", NOT "GO TO ZERO". The next cycle searches the WHOLE route for the
    // nearest point instead of the usual 2 m forward window, and adopts it. Both callers need that, and
    // they need different answers from it:
    //   • a mission STOP -> the robot is back at the start, so the nearest point IS the start. This is
    //     the requirement: stop, then start, and it goes to the beginning.
    //   • a route REPAIR (controller_session.cpp calls stop() mid-mission to install a repaired curve)
    //     -> the robot is somewhere in the middle, and the nearest point is right where it stands.
    // Forcing s_hint_ to 0 would satisfy the first and break the second — the tracker would steer at the
    // route's beginning from thirty metres away, which is the same "drives into a wall" failure.
    void reset() override { s_hint_.reset(); pivoting_ = false; start_align_ = true; holding_ = false; }
    // Resume with the arc length the CALLER knows, instead of searching for it.
    // ★start_align_ IS DELIBERATELY NOT ARMED. A re-acquisition arms it because the robot may be facing
    // anywhere; a LAP RESTART is not that. The robot arrives at the lap start along the closing hop,
    // already pointing down the first leg, and arming the start alignment would stop it dead to turn on
    // the spot for an alignment it already has. The lap boundary is a place the robot drives THROUGH:
    // nothing may stop it between arriving and departing.
    void resume_at(float s) override
    { s_hint_ = std::max(0.f, s); pivoting_ = false; start_align_ = false; holding_ = false; }

    // Is the tracker holding the wheels still and turning on the spot? For the overlay and the logs —
    // a robot that is deliberately stopped and one that is stuck look identical from outside.
    // ★Reports BOTH turn-in-place states — the start alignment and the mid-route pivot. The question
    // the overlay is asking is "are the wheels deliberately at zero", and both answer it yes.
    [[nodiscard]] bool pivoting() const { return holding_; }

    ControlOutput& compute(ControlOutput& out, const TrackerInput& in, const TrackerParams& p) override;

    // The arc length the control law actually used. Empty before the first cycle of a traversal.
    [[nodiscard]] std::optional<float> arc_length_hint() const { return s_hint_; }

    // Pin the projection. For OFFLINE BENCHES only (tools/tracker_sim), which sweep s themselves so the
    // steering law can be measured independently of the projection. The agent never calls this.
    void seed_arc_length(float s) { s_hint_ = s; }

    // Exercises the arc-length projection: that it is monotone within a traversal, that a reset is
    // free to move it backwards, and that a near-zero command does not cost it. Run from
    // `route_bench --self-test`. See the bottom of plain_tracker.cpp for what each case caught.
    static bool self_test();

private:
    const PathWorld& world_;
    // The monotone-forward arc-length projection. EMPTY means "re-acquire": one optional carries

    // both the value and its validity, so a stale hint cannot be read as a live one.

    std::optional<float> s_hint_;
    bool  reacquire_ = true;
    // STOP-TURN-GO. True while the wheels are held at zero and the robot is spinning to face the exit
    // of a turn it cannot drive through. Latched, because the two ends of the decision are asked of
    // different things: entry of the ROUTE ("does this turn reverse me?"), release of the POSE ("am I
    // facing the way out yet?"). A memoryless test cannot express that, and the two stateless versions
    // both failed — see the measured deadlocks in plain_tracker.cpp.
    bool  pivoting_ = false;
    // TURN AND ONLY THEN GO, AT THE START OF A PATH. Armed on every re-acquisition of the projection —
    // a mission start, a route repair, a new curve — and disarmed for the rest of that traversal the
    // moment the robot faces where the route leaves.
    // ★Distinct from pivoting_ because the two ask DIFFERENT QUESTIONS. The mid-route latch asks the
    // route ("does this turn reverse me?") before it will stop the wheels, because a robot that is
    // merely off-route on a straight must MOVE to get back on. At initiation there is nothing to move
    // back onto — the route is planned from the robot — so misalignment alone is the whole test, and
    // asking turn_ahead there is what let the robot arc onto a path it was facing 135 deg away from.
    bool  start_align_ = true;
    // Either turn-in-place state, as of the last cycle. For pivoting() — the overlay wants "are the
    // wheels deliberately at zero", not "which of the two reasons".
    bool  holding_ = false;
};

}   // namespace rc

#endif
