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
    void reset() override { reacquire_ = true; }

    ControlOutput& compute(ControlOutput& out, const TrackerInput& in, const TrackerParams& p) override;

    float arc_length_hint() const { return s_hint_; }

private:
    const PathWorld& world_;
    // The state this tracker keeps: a monotone-forward arc-length projection, and the previous position
    // used to bound how fast that projection may advance. The tour crosses itself, so a global
    // nearest-point search would snap onto a branch driven ten metres ago.
    float s_hint_ = 0.f;
    // Set by reset(); consumed by the next compute(), which then searches the whole route once.
    bool  reacquire_ = true;
};

}   // namespace rc

#endif
