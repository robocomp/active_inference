// The calibration channel, exercised without a robot: the passive observer, the marginal-gain rule
// that decides whether a deliberate pivot is worth anything, and the sequencing through the wire's
// own outcome vocabulary.
#include "calib_channel.h"
#include <cstdio>
#include <cmath>

using rc::calib::CalibChannel;
using rc::calib::CalibChannelParams;
using rc::calib::PivotAffordance;
using O = rc::affordance::Outcome;

static int failures = 0;
static void check(bool ok, const char *what, const char *why = "")
{ if (!ok) { ++failures; std::printf("  ✗ %s  %s\n", what, why); } else std::printf("  ✓ %s\n", what); }

// Drive `secs` seconds of motion at 20 Hz: the robot turns at `omega` rad/s and the odometry reads
// (1+s_true) times what really happened.
// ★COUNTED IN FRAMES, NOT ACCUMULATED IN SECONDS. `for (double e = 0; e < secs; e += dt)` runs 100 or
// 101 times depending on how 0.05 rounds, and one extra frame in twelve pivot steps reported the
// injected 5% scale as 5.96% — a test harness manufacturing a 1% calibration error out of floating
// point. The quantity under test is a ratio; the loop that produces it must be exact.
static void drive(CalibChannel &c, double &t, double &theta, double omega, double s_true, double secs,
                  double noise_rad = 0.0)
{
    const double dt = 0.05;
    const int frames = static_cast<int>(std::lround(secs / dt));
    for (int i = 0; i < frames; ++i)
    {
        const double d = omega * dt;
        // A deterministic wobble standing in for the localiser's own heading noise. Deterministic so
        // the test cannot pass or fail by luck.
        const double n = noise_rad * std::sin(1.7 * static_cast<double>(i));
        theta += d + n;
        t += dt;
        c.note_motion(t, std::atan2(std::sin(theta), std::cos(theta)), d * (1.0 + s_true), 0.0, true);
    }
}

int main()
{
    std::printf("── the calibration channel ──\n\n");

    // 1. THE FREE DATA. Ordinary tours turn, so the scale is identified without anyone asking for a
    //    manoeuvre. This is the claim the whole design rests on; if it fails the pivot is not an
    //    optimisation, it is the only source of data.
    {
        CalibChannelParams p; p.enabled = true;
        CalibChannel c(p);
        double t = 0.0, th = 0.0;
        drive(c, t, th, 0.5, 0.06, 400.0);          // a busy day: turning most of the time
        const auto post = c.posterior();
        std::printf("  passive only: s = %.4f +/- %.4f over %d windows, diet %.3f rad/s\n",
                    post.s, post.s_std, post.windows, c.passive_rate_rad_s());
        check(post.identifiable(), "ordinary turning alone identifies the rotation scale");
        check(std::abs(post.s - 0.06) < 0.005, "and recovers it to better than half a percent");
        check(std::abs(c.passive_rate_rad_s() - 0.5) < 0.05,
              "the passive diet is MEASURED, not assumed");
    }

    // 2. THE MANOEUVRE EXTINGUISHES ITSELF. Same robot, before and after that day's worth of turning:
    //    the deliberate pivot must be worth strictly less once the tours have done the work. This is
    //    what replaces a threshold on the posterior width.
    {
        CalibChannelParams p; p.enabled = true;
        CalibChannel c(p);
        const double before = c.marginal_gain_nats();
        double t = 0.0, th = 0.0;
        drive(c, t, th, 0.5, 0.06, 400.0);
        const double after = c.marginal_gain_nats();
        std::printf("  marginal gain: %.4f nats cold -> %.4f nats after 400 s of turning\n", before, after);
        check(before > 0.0, "a robot that has never turned has something to gain");
        check(after < before, "and a robot whose day is full of turning has less");
    }

    // 3. A ROBOT THAT ONLY EVER DRIVES STRAIGHT keeps the gain: no free rotation data arrives, so the
    //    deliberate manoeuvre stays worth making. The two cases together are the rule.
    {
        CalibChannelParams p; p.enabled = true;
        CalibChannel c(p);
        const double cold = c.marginal_gain_nats();
        double t = 0.0, th = 0.0;
        // Straight-line driving, with the localiser's heading noise present — because it always is,
        // and because a channel fed EXACTLY zero residuals is a situation no robot produces.
        drive(c, t, th, 0.0, 0.06, 400.0, 2e-3);
        std::printf("  straight-line day: diet %.4f rad/s, gain %.4f -> %.4f nats\n",
                    c.passive_rate_rad_s(), cold, c.marginal_gain_nats());
        check(c.passive_rate_rad_s() < 0.01, "no turning means no free rotation data");
        check(c.marginal_gain_nats() > 0.5 * cold, "so the pivot is still worth offering");
    }

    // 4. OFF IS OFF. A feature that has never driven a robot must not be able to.
    {
        CalibChannelParams p; p.enabled = false;
        CalibChannel c(p);
        check(not c.offer(0.0).has_value(), "disabled, it never offers, whatever the gain");
    }

    // 5. THE WHOLE SEQUENCE, through the wire's outcome words. Twelve satisfied 120-degree steps with
    //    an injected 5% odometry error must close and recover it.
    {
        CalibChannelParams p; p.enabled = true;
        CalibChannel c(p);
        double h = 0.4;
        const double s_true = 0.05, step = 2.0*M_PI/3.0;
        double t = 0.0, th = h;
        // ★WARM THE CHANNEL FIRST, because live it always is. The very first note_motion has no
        // previous heading to difference against, so it seats the reference and contributes nothing —
        // including its odometry. Opening a pivot on that exact cycle drops one frame in 1200 and
        // reported the injected 5% as 4.91%: a real 1.8% error in the answer, produced entirely by
        // the test starting the manoeuvre before the observer had a first sample.
        drive(c, t, th, 0.0, s_true, 0.5);
        for (int i = 0; i < 12; ++i)
        {
            const auto b = c.offer(h);
            if (not b.has_value()) { std::printf("  offer %d declined\n", i); break; }
            // The robot really turns a third of a turn; the odometry over-reports it. note_motion is
            // called throughout, exactly as it would be live — the pivot's windows are ordinary
            // windows, and excluding them would throw away the most informative motion of the day.
            drive(c, t, th, step / 5.0, s_true, 5.0);
            h += step;
            c.on_outcome(O::Satisfied, std::atan2(std::sin(h), std::cos(h)));
        }
        const auto cl = c.closure();
        std::printf("  pivot: state=%d  turned %.1f deg  s_omega %.4f (truth %.3f) res %.4f usable=%d\n",
                    (int)c.pivot().state(), c.pivot().accumulated_rad()*180/M_PI, cl.s_omega, s_true,
                    cl.resolution, (int)cl.usable);
        check(c.pivot().state() == PivotAffordance::State::Closed, "the sequence closes");
        check(std::abs(cl.s_omega - s_true) < 1e-6, "and recovers the injected scale");
    }

    // 6. INFEASIBLE IS BELIEVED, AND IT IS NOT FOREVER. The consumer alone can say the body cannot
    //    sweep its diagonal here; the producer stops asking and waits to be carried elsewhere.
    {
        CalibChannelParams p; p.enabled = true;
        CalibChannel c(p);
        check(c.offer(0.0).has_value(), "the first offer goes out");
        c.on_outcome(O::Infeasible, 0.0);
        c.note_robot_pos(0.0, 0.0);
        check(not c.offer(0.0).has_value(), "an infeasible spot silences the offer");
        c.note_robot_pos(0.20, 0.0);
        check(not c.offer(0.0).has_value(), "a shuffle of 20 cm is the SAME spot, not a new chance");
        c.note_robot_pos(1.50, 0.0);
        check(c.offer(0.0).has_value(), "a body-width and more away, it may ask again");
    }

    // 7. EVERY OTHER FAILURE LEAVES THE SEQUENCE WHERE IT WAS. A timeout is not a turn.
    {
        CalibChannelParams p; p.enabled = true;
        CalibChannel c(p);
        (void)c.offer(0.0);
        c.on_outcome(O::Timeout, 0.0);
        const int after_timeout = c.pivot().steps_issued();
        check(after_timeout == 0, "a step that timed out does not advance the pivot");
        check(c.offer(0.0).has_value(), "and the next offer still goes out");
    }

    // 8. ONE LIVE OFFER AT A TIME. The node is a single register; a second offer before the first is
    //    answered would be the same ABA hazard the epoch model exists to stop.
    {
        CalibChannelParams p; p.enabled = true;
        CalibChannel c(p);
        check(c.offer(0.0).has_value(), "the offer goes out");
        check(not c.offer(0.0).has_value(), "and nothing else is offered until it is answered");
    }

    if (failures == 0) std::printf("\nALL PASS\n\n");
    else               std::printf("\n%d check(s) FAILED\n\n", failures);
    return failures == 0 ? 0 : 1;
}
