// The pivot sequencer, exercised without a robot. Each case is a situation the live pair produced
// today, replayed against the decision object.
#include "calib_pivot.h"
#include <cstdio>
#include <cmath>
using rc::calib::PivotAffordance;
static int failures = 0;
static void check(bool ok, const char *what, const char *why = "")
{ if (!ok) { ++failures; std::printf("  ✗ %s  %s\n", what, why); } else std::printf("  ✓ %s\n", what); }

int main()
{
    std::printf("── the calibration pivot sequencer ──\n\n");

    // 1. A calibrated robot offers nothing.
    { PivotAffordance p; check(not p.next_bearing(0.0, 0.0).has_value(),
        "no gain, no offer — the normal state of a calibrated robot"); }

    // 2. A full pivot: twelve steps of 120 degrees, driven by MEASURED heading.
    {
        PivotAffordance p;
        double h = 0.3;                       // the robot's heading
        const double s_true = 0.07;           // the odometry reads 7% high
        for (int i = 0; i < 12; ++i)
        {
            const auto b = p.next_bearing(h, 0.5);
            if (not b.has_value()) break;
            const double turned = rc::calib::PivotAffordance::State::Idle == p.state() ? 0.0 : 0.0;
            (void)turned;
            const double step = 2.0*M_PI/3.0;
            h += step;                        // the robot really turns a third of a turn
            p.on_outcome(true, false, std::atan2(std::sin(h), std::cos(h)), step * (1.0 + s_true), step);
        }
        const auto c = p.closure();
        std::printf("  after %d steps: state=%d  accumulated %.1f deg  s_omega %.4f (truth %.3f) res %.4f\n",
                    p.steps_issued(), (int)p.state(), p.accumulated_rad()*180/M_PI, c.s_omega, s_true, c.resolution);
        check(p.state() == PivotAffordance::State::Closed, "twelve 120-degree steps close the pivot");
        check(std::abs(c.s_omega - s_true) < 1e-9, "and recover the injected scale exactly");
        check(c.usable, "and the run resolves it");
    }

    // 3. A step that did NOT happen must not advance the sequence.
    {
        PivotAffordance p;
        double h = 0.0;
        p.on_outcome(true, false, h, 0.0, 0.0);                  // step 1 lands
        const int after_one = p.steps_issued();
        p.on_outcome(false, false, h, 0.0, 0.0);                 // refused / timed out
        check(p.steps_issued() == after_one, "a step that did not complete does not advance the pivot",
              "the closure would then be a fiction");
    }

    // 4. The consumer says the body cannot turn here: stop asking, resume when the robot has moved.
    {
        PivotAffordance p;
        p.on_outcome(false, true, 0.0, 0.0, 0.0);
        check(not p.next_bearing(0.0, 5.0).has_value(), "an infeasible spot silences the offer",
              "it would hammer a place the body does not fit");
        p.robot_moved();
        check(p.next_bearing(0.0, 5.0).has_value(), "and it resumes once the robot is elsewhere");
    }

    // 5. Standing still does not close a pivot, however many outcomes arrive.
    {
        PivotAffordance p;
        for (int i = 0; i < 30; ++i) p.on_outcome(true, false, 0.0, 0.0, 0.0);
        check(p.state() != PivotAffordance::State::Closed,
              "no turn, no closure — the heading came back because it never left",
              "a stationary robot would report a scale");
    }

    // 6. A scale under the closure's resolution is reported as unusable rather than quoted.
    //    ★The first version of this case closed PERFECTLY (heading miss 0), which resolves everything
    //    — so the estimator was right to call a 0.01% scale usable and the test was wrong. Real
    //    closures miss: end 0.05 rad short of the mark and the run resolves ~0.8%, against which a
    //    0.01% claim is noise.
    {
        PivotAffordance p({.step_rad = 2.0*M_PI/3.0, .turns = 1, .closure_tolerance_rad = 0.10});
        double h = 0.0;
        (void)p.next_bearing(h, 1.0);                    // the pivot begins when the first offer goes out
        for (int i = 0; i < 3; ++i)
        {
            double step = 2.0*M_PI/3.0;
            if (i == 2) step -= 0.05;                    // stop 0.05 rad short of the mark
            h += step;
            p.on_outcome(true, false, std::atan2(std::sin(h), std::cos(h)), step * 1.0001, step);
        }
        const auto c = p.closure();
        std::printf("  imperfect closure: s_omega %.5f vs resolution %.5f -> usable=%d\n",
                    c.s_omega, c.resolution, c.usable);
        check(c.resolution > 0.005, "an imperfect closure reports a real resolution");
        check(not c.usable, "and a scale finer than it is not claimed as a measurement");
    }

    std::printf("\n%d check(s) FAILED\n", failures);
    return failures == 0 ? 0 : 1;
}
