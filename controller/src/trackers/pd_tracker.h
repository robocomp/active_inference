#ifndef RC_PD_TRACKER_H
#define RC_PD_TRACKER_H

#include "tracker.h"

namespace rc
{

// THE PD CARROT TRACKER — the controller that has measured best on the robot so far.
//
// Geometric pursuit: PD on the carrot bearing with a Stanley cross-track term, a cos-shaped speed law,
// a lateral bumper, an output EMA and a triggered safety gate. It DOES read the obstacle field (hence
// FieldWorld as well as PathWorld) — its gate rolls out the commanded arc for ~0.30 s, which is why it
// only ever consults the dense NEAR field and is not hurt by the far-field noise that killed the
// "route" tracker. See the post-mortem in etc/config.toml above ControlMode.
//
// ★Avoidance is still nominally the BAND's: this gate is a backstop, not a planner. Running "pd" with
// BandEnabled = false leaves the gate as the only thing between the tracker and an obstacle, and the
// agent warns about exactly that at startup.
class PdTracker final : public Tracker
{
public:
    PdTracker(const PathWorld& path, const FieldWorld& field) : path_(path), world_(field) {}

    const char* name() const override { return "pd"; }
    void reset() override { has_prev_vel_ = false; smoothed_vel_.setZero(); prev_angle_err_ = 0.f; }

    ControlOutput& compute(ControlOutput& out, const TrackerInput& in, const TrackerParams& p) override;

private:
    const PathWorld&  path_;
    const FieldWorld& world_;
    // ★These were members of TrajectoryController shared with the MPPI path. Only one mode runs in a
    // session, so nothing changes behaviourally by giving PD its own — but sharing smoothing state
    // between two different control laws was a latent bug waiting for someone to switch modes live.
    Eigen::Vector3f smoothed_vel_ = Eigen::Vector3f::Zero();
    bool  has_prev_vel_ = false;
    float prev_angle_err_ = 0.f;   // for the PD derivative term
};

}   // namespace rc

#endif
