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
        c.note_motion(t, 0.0, 0.0, std::atan2(std::sin(theta), std::cos(theta)), d * (1.0 + s_true), true);
    }
}

// Like drive(), but the odometry ALSO carries a constant bias: it reports b_true radians per second
// of rotation that never happened. That is what a gyro bias is, and it is invisible to a single-rate
// closure because over one manoeuvre it is indistinguishable from a scale error.
static void drive_biased(CalibChannel &c, double &t, double &theta, double omega, double s_true,
                         double b_true, double secs)
{
    const double dt = 0.05;
    const int frames = static_cast<int>(std::lround(secs / dt));
    for (int i = 0; i < frames; ++i)
    {
        const double d = omega * dt;
        theta += d;
        t += dt;
        c.note_motion(t, 0.0, 0.0, std::atan2(std::sin(theta), std::cos(theta)),
                      d * (1.0 + s_true) + b_true * dt, true);
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
        // ONE block: this case is about a single closure recovering a single scale. The manoeuvre's
        // default is two blocks at two rates (see 5d), which is a different question.
        p.pivot.blocks = 1;
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
        // ONE block: this case is about a single closure recovering a single scale. The manoeuvre's
        // default is two blocks at two rates (see 5d), which is a different question.
        p.pivot.blocks = 1;
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
        // ONE block: this case is about a single closure recovering a single scale. The manoeuvre's
        // default is two blocks at two rates (see 5d), which is a different question.
        p.pivot.blocks = 1;
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
        // ONE block: this case is about a single closure recovering a single scale. The manoeuvre's
        // default is two blocks at two rates (see 5d), which is a different question.
        p.pivot.blocks = 1;
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
            if (b.has_value()) c.mark_offered();   // the producer latches only once published
            if (not b.has_value()) { std::printf("  offer %d declined\n", i); break; }
            // The robot really turns a third of a turn; the odometry over-reports it. note_motion is
            // called throughout, exactly as it would be live — the pivot's windows are ordinary
            // windows, and excluding them would throw away the most informative motion of the day.
            // The claim is taken before the turn and released after it: only motion inside that
            // bracket is motion the pivot asked for.
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

    // 5c. A STEP THAT FALLS SHORT MUST NOT POISON THE SEQUENCE. The executor completes an Orient
    //     inside its aligned band, so every step lands a couple of degrees short of what was asked.
    //     Measured live 2026-08-24: thirteen consecutive steps of 117.4 deg, never 120. With bearings
    //     asked relative to where the last step stopped, twelve of those total 1409 deg and the robot
    //     ends 31 deg from its start — closure impossible, for ever. Anchored bearings make each step
    //     absorb the previous shortfall.
    {
        CalibChannelParams p; p.enabled = true;
        // ONE block: this case is about a single closure recovering a single scale. The manoeuvre's
        // default is two blocks at two rates (see 5d), which is a different question.
        p.pivot.blocks = 1;
        CalibChannel c(p);
        double t = 0.0, th = 0.0;
        const double s_true = 0.05, step = 2.0*M_PI/3.0;
        const double shortfall = 0.045;            // ~2.6 deg, the executor's band
        drive(c, t, th, 0.0, s_true, 0.5);
        double h = 0.0;
        for (int i = 0; i < 200 and c.pivot().state() != PivotAffordance::State::Closed; ++i)
        {
            const auto b = c.offer(h);
            if (not b.has_value()) break;
            c.mark_offered();
            // Turn to the ASKED bearing but stop short of it, exactly as the executor does.
            const double err = std::atan2(std::sin(*b - th), std::cos(*b - th));
            const double turn = err - (err > 0 ? shortfall : -shortfall);
            drive(c, t, th, turn / 5.0, s_true, 5.0);
            h = std::atan2(std::sin(th), std::cos(th));
            c.on_outcome(O::Satisfied, h);
        }
        const auto cl = c.closure();
        std::printf("  short steps: closed=%d after %d step(s), turned %.1f deg, truth %.1f deg, "
                    "s_omega %.4f (injected %.3f)\n",
                    (int)(c.pivot().state() == PivotAffordance::State::Closed),
                    c.pivot().steps_issued(), cl.turned_rad*180/M_PI, cl.truth_rad*180/M_PI,
                    cl.s_omega, s_true);
        check(c.pivot().state() == PivotAffordance::State::Closed,
              "a sequence of short steps still closes");
        check(c.pivot().steps_issued() <= 13, "and closes in about twelve, not forty");
        check(std::abs(cl.s_omega - s_true) < 5e-3,
              "and the scale it reports is the injected one, not a ratio against an asserted total");
    }

    // 5b. A DETOUR IN THE MIDDLE DOES NOT CORRUPT THE CLOSURE — because the truth is COUNTED.
    //     Between publishing a step and somebody claiming it, the selector may send the robot across
    //     the room, and it does. An earlier version GATED the accumulators on the consumer holding
    //     the claim so that rotation could not be credited to a step. That was the wrong cure and it
    //     broke the measurement: a closure is a TOTAL between two headings, so every radian in
    //     between belongs to it — including claim latency and the tail after the consumer lets go.
    //     Live 2026-08-25: fifteen steps, five real turns, only four counted, +4.34% reported against
    //     an estimator reading +0.34%. With truth = round(ref/2pi) a detour simply raises the turn
    //     count on BOTH sides and the ratio is untouched, which is what this checks.
    {
        CalibChannelParams p; p.enabled = true;
        // ONE block: this case is about a single closure recovering a single scale. The manoeuvre's
        // default is two blocks at two rates (see 5d), which is a different question.
        p.pivot.blocks = 1;
        CalibChannel c(p);
        double t = 0.0, th = 0.0;
        const double s_true = 0.05, step = 2.0*M_PI/3.0;
        drive(c, t, th, 0.0, s_true, 0.5);
        double h = 0.0;
        for (int i = 0; i < 60 and c.pivot().state() != PivotAffordance::State::Closed; ++i)
        {
            const auto b = c.offer(h);
            if (not b.has_value()) break;
            c.mark_offered();
            // Halfway through the sequence the robot is sent off on an errand and turns a whole extra
            // circle before anyone claims the step that is standing on the wire.
            if (i == 6) drive(c, t, th, 2.0*M_PI / 20.0, s_true, 20.0);
            const double err = std::atan2(std::sin(*b - th), std::cos(*b - th));
            drive(c, t, th, err / 5.0, s_true, 5.0);
            h = std::atan2(std::sin(th), std::cos(th));
            c.on_outcome(O::Satisfied, h);
        }
        const auto cl = c.closure();
        std::printf("  detour mid-pivot: closed=%d, truth %.0f deg (%.0f turns), s_omega %.4f\n",
                    (int)(c.pivot().state() == PivotAffordance::State::Closed),
                    cl.truth_rad*180/M_PI, cl.truth_rad/(2*M_PI), cl.s_omega);
        check(c.pivot().state() == PivotAffordance::State::Closed, "a pivot with a detour still closes");
        check(std::abs(cl.s_omega - s_true) < 5e-3,
              "and the detour's turning does not bias the scale, because the truth counts it too");
    }

    // 5d. TWO RATES SEPARATE SCALE FROM BIAS, ON HEADINGS ALONE. A closure at rate w measures
    //     k + b/w, so any single-rate pivot reports the two fused and cannot say which is which.
    //     Run the same closure at two rates and they come apart in closed form. This injects BOTH a
    //     scale error and a gyro bias and checks each is recovered — the bias in particular could not
    //     have been seen at all before, at any rate, by any amount of turning.
    {
        CalibChannelParams p; p.enabled = true;
        p.pivot_rates = {0.5, 0.25};
        p.pivot.blocks = 2;
        CalibChannel c(p);
        const double k_true = 0.03;          // 3% rotation scale
        const double b_true = 0.004;         // rad/s of phantom rotation while turning
        const double step = 2.0*M_PI/3.0;
        double t = 0.0, th = 0.0, h = 0.0;
        // Warm, with the robot still: a bias shows even here, which is the point of it.
        drive_biased(c, t, th, 0.0, k_true, b_true, 0.5);
        for (int i = 0; i < 200 and c.pivot().state() != PivotAffordance::State::Closed; ++i)
        {
            const auto b = c.offer(h);
            if (not b.has_value()) break;
            c.mark_offered();
            const double w = c.pivot_rate_rps();                  // the rate THIS block runs at
            const double err = std::atan2(std::sin(*b - th), std::cos(*b - th));
            drive_biased(c, t, th, (err < 0 ? -w : w), k_true, b_true, std::abs(err) / w);
            h = std::atan2(std::sin(th), std::cos(th));
            c.on_outcome(O::Satisfied, h);
        }
        const auto sep = c.separate_scale_and_bias();
        const auto &cs = c.pivot().closures();
        std::printf("  two-rate: %zu block(s) closed", cs.size());
        for (std::size_t i = 0; i < cs.size(); ++i)
            std::printf(" | s%zu %+.5f (res %.5f)", i + 1, cs[i].s_omega, cs[i].resolution);
        std::printf("\n  separated: k %+.5f +/- %.5f (true %+.3f), b %+.5f +/- %.5f rad/s (true %+.4f)\n",
                    sep.k_omega, sep.sigma_k, k_true, sep.b_omega, sep.sigma_b, b_true);
        check(cs.size() == 2, "both rate blocks close");
        check(sep.solved, "and the two closures separate");
        check(std::abs(sep.k_omega - k_true) < 5e-3, "the rotation scale comes back free of the bias");
        check(std::abs(sep.b_omega - b_true) < 1e-3, "and the BIAS comes back, which one rate cannot see");
    }

    // 6. INFEASIBLE IS BELIEVED, AND IT IS NOT FOREVER. The consumer alone can say the body cannot
    //    sweep its diagonal here; the producer stops asking and waits to be carried elsewhere.
    {
        CalibChannelParams p; p.enabled = true;
        // ONE block: this case is about a single closure recovering a single scale. The manoeuvre's
        // default is two blocks at two rates (see 5d), which is a different question.
        p.pivot.blocks = 1;
        CalibChannel c(p);
        check([&]{ const auto b = c.offer(0.0); if (b) c.mark_offered(); return b.has_value(); }(),
               "the first offer goes out");
        c.on_outcome(O::Infeasible, 0.0);
        c.note_robot_pos(0.0, 0.0);
        check(not c.offer(0.0).has_value(), "an infeasible spot silences the offer");
        c.note_robot_pos(0.20, 0.0);
        check(not c.offer(0.0).has_value(), "a shuffle of 20 cm is the SAME spot, not a new chance");
        c.note_robot_pos(1.50, 0.0);
        check([&]{ const auto b = c.offer(0.0); if (b) c.mark_offered(); return b.has_value(); }(),
               "a body-width and more away, it may ask again");
    }

    // 7. EVERY OTHER FAILURE LEAVES THE SEQUENCE WHERE IT WAS. A timeout is not a turn.
    {
        CalibChannelParams p; p.enabled = true;
        // ONE block: this case is about a single closure recovering a single scale. The manoeuvre's
        // default is two blocks at two rates (see 5d), which is a different question.
        p.pivot.blocks = 1;
        CalibChannel c(p);
        if (c.offer(0.0).has_value()) c.mark_offered();
        c.on_outcome(O::Timeout, 0.0);
        const int after_timeout = c.pivot().steps_issued();
        check(after_timeout == 0, "a step that timed out does not advance the pivot");
        check([&]{ const auto b = c.offer(0.0); if (b) c.mark_offered(); return b.has_value(); }(),
               "and the next offer still goes out");
    }

    // 8. ONE LIVE OFFER AT A TIME. The node is a single register; a second offer before the first is
    //    answered would be the same ABA hazard the epoch model exists to stop.
    {
        CalibChannelParams p; p.enabled = true;
        // ONE block: this case is about a single closure recovering a single scale. The manoeuvre's
        // default is two blocks at two rates (see 5d), which is a different question.
        p.pivot.blocks = 1;
        CalibChannel c(p);
        check([&]{ const auto b = c.offer(0.0); if (b) c.mark_offered(); return b.has_value(); }(),
               "the offer goes out");
        check(not c.offer(0.0).has_value(), "and nothing else is offered until it is answered");
    }

    // 9. A TELEPORT IS NOT ODOMETRY ERROR. The reference is the localiser's posterior and it jumps —
    //    on this robot 1.77% of cycles imply a speed above 1 m/s on a 0.6 m/s base, the worst 11.63 m
    //    in 50 ms. Charging that to the odometry puts a metre-scale outlier into a fit whose typical
    //    increment is a metre. Judged per FRAME: 3 m in a second is merely fast, 3 m in 50 ms is not
    //    possible, and testing against the window's length would let the second one through.
    {
        CalibChannelParams p; p.enabled = true;
        CalibChannel clean(p), jumped(p);
        double t1 = 0, th1 = 0, t2 = 0, th2 = 0;
        drive(clean, t1, th1, 0.4, 0.05, 60.0);
        drive(jumped, t2, th2, 0.4, 0.05, 30.0);
        // one 5 m relocalisation in a single 50 ms frame, then business as usual
        t2 += 0.05; th2 += 0.4 * 0.05;
        jumped.note_motion(t2, 5.0, 0.0, std::atan2(std::sin(th2), std::cos(th2)), 0.4 * 0.05 * 1.05, true);
        drive(jumped, t2, th2, 0.4, 0.05, 30.0);
        std::printf("  jump: %ld window(s) discarded; s clean %.4f vs jumped %.4f\n",
                    jumped.poisoned_windows(), clean.posterior().s, jumped.posterior().s);
        check(jumped.poisoned_windows() >= 1, "the frame carrying the teleport poisons its window");
        check(std::abs(jumped.posterior().s - clean.posterior().s) < 0.01,
              "and the scale is not dragged by it");
    }

    if (failures == 0) std::printf("\nALL PASS\n\n");
    else               std::printf("\n%d check(s) FAILED\n\n", failures);
    return failures == 0 ? 0 : 1;
}
