/*
 * stall_judge.h — "the robot is not getting anywhere". Why, and what may be done about it.
 *
 * WHY THIS EXISTS AS ITS OWN UNIT
 * The wedge detector used to live inline in ControllerSession, where it could not be exercised without a
 * DSR graph, a Qt loop and a robot. It shipped with a defect that made it structurally unable to fire in
 * the one situation it most needed to (below), and nothing could have caught that: there was nowhere to
 * write the test. So the decision is here, pure Eigen/STL, with a self_test() — and the session only
 * feeds it numbers.
 *
 * ★THE DEFECT THIS FIXES — A DETECTOR THAT CONCEALED ITSELF
 * A wedge is a PREDICTION ERROR: the robot is told to travel and does not get anywhere. It is judged by
 * integrating the commanded speed over a window and comparing that against NET DISPLACEMENT, with a
 * floor of slip_ratio (0.25) — so at cruise the bar is ~0.2 m, far above pose jitter.
 * But the command it integrated was taken AFTER the pose-covariance limiter had already scaled it down.
 * Measured 2026-08-16 (run 20260816-154346, route_s 32.30 m): the tracker asked for 0.071 m/s, the
 * limiter was floored at 0.15 and delivered 0.011 m/s, and the robot sat still for FIFTEEN SECONDS.
 * Over a 1500 ms window that command integrates to 0.0165 m, so the bar was 0.25 x that = 4.1 mm —
 * below pose jitter. The window passed every time. The run recorded one escape across 160 s containing
 * several multi-second freezes.
 * ★The throttle was lowering the very bar it would be judged against. The harder it held the robot down,
 * the more certainly the detector called it healthy. That is the property being removed.
 *
 * ★WHY THE OBVIOUS FIX IS WRONG, AND WHAT IS DONE INSTEAD
 * "Judge against the pre-throttle ask" alone is NOT sound: the base then gets blamed for travel it was
 * never told to make, and the response to a wedge is to REVERSE AND DROP AN OBSTACLE. Reversing does not
 * fix a speed limiter, so the reflex would fire, end, find the limiter still floored, and fire again —
 * the identical closed loop that "planner failure is not a wedge" was written to end, plus phantom
 * discs sprayed into the obstacle set at every stall.
 * So the window measures BOTH travels, and the verdict names the CAUSE:
 *
 *      achieved >= slip_ratio * ASKED                             -> None
 *      delivered >= slip_ratio AND achieved < slip_ratio*COMMANDED -> Wedge
 *      achieved  >  pose_sigma                                    -> None
 *      else                                                       -> ThrottleStall
 *
 * ★A WEDGE NEEDS BOTH HALVES: THE COMMAND WAS REAL, AND THE BASE FAILED IT. Each half alone is a
 * different bug, and this file shipped each of them in turn on 2026-08-16 before the pair was right.
 *   ONLY "the limiter did not take much" (`delivered >= slip`): blames the base for a command it may
 *   well have obeyed. Live consequence, run 20260816-202001: FOUR spurious escapes in one 138 s tour
 *   (escapes 1 -> 4). Every one had achieved 33–74% of the distance actually commanded — the base was
 *   obeying — but `delivered` sat at 0.29–0.46, just over the cut, so the robot reversed and dropped a
 *   phantom obstacle. The detector that had been too blind became too eager in the same afternoon.
 *   ONLY "the base failed its command" (the legacy predicate): correct in principle, but VACUOUS when
 *   the limiter has shrunk the command to millimetres — the bar becomes 4 mm and pose wander decides it.
 *   The self-test caught this: a 15 s deadlock came out as six throttle stalls and three wedges, i.e.
 *   three coin-flip escapes in the middle of exactly the freeze this file exists to report.
 * Conjoined, they say what a wedge actually is, and the escape set becomes a strict SUBSET of what the
 * legacy detector fired on — so escapes can never become more frequent than before this file existed.
 * That is the property to check first if this is ever edited again.
 *
 * ★THE THIRD LINE IS NOT DECORATION — the self-test rejected an earlier version that lacked it, on the
 * case of a robot creeping along at exactly the throttled speed. Being held to 15% is not the complaint;
 * being held to a STANDSTILL is. So "did it actually get anywhere" has to be asked in absolute terms,
 * and the honest yardstick is the pose uncertainty itself: NET DISPLACEMENT SMALLER THAN SIGMA IS NOT
 * EVIDENCE OF MOTION. No constant is invented — the scale comes from the same covariance that produced
 * the throttle — and it degrades the right way: a converged filter (sigma 0.06) lets a creep read as
 * motion, while one reporting 0.20 m cannot honestly claim 0.10 m of travel happened.
 * ⚠Two things stated plainly rather than assumed. (a) It is a real approximation: ABSOLUTE sigma
 * overstates SHORT-TERM blindness — a filter can carry 0.20 m of absolute error and still track 5 cm of
 * motion well — so this errs toward speaking up, which is the right direction for something that only
 * ever writes a line. (b) MEASURED, not argued: replaying three recorded tours, dropping this clause
 * changes 34 windows to 40 on the bad run and 1 to 2 on the good one. It is a refinement, not the thing
 * that makes the detector work — do not credit it with more than that.
 *
 * ★WHAT IT DOES ON REAL RUNS (replayed offline through this exact code, three complete tours):
 *      run       duration   NEW wedges   OLD wedges   NEW throttle stalls
 *      14:15      110 s          2            1        1 window  /  1.5 s   <- healthy: near-silent
 *      15:37      118 s          1            1       11 windows / 16.5 s
 *      15:43      160 s          2            0       34 windows / 51.1 s   <- the run that prompted this
 * The bottom row is the whole point: on the worst run, containing several multi-second freezes and one
 * of FIFTEEN seconds, the old rule found ZERO wedges. Not one. And the healthy run stays quiet, so the
 * new verdict is a signal and not a new source of noise.
 *
 * ★ORDER MATTERS: the Wedge branch is tested BEFORE sigma is consulted, so when the limiter is inert
 * (delivered == 1) the sigma test is unreachable and the behaviour is EXACTLY what it was before, to the
 * bit. Sigma can only ever affect the branch that exists because the limiter acted — and on that branch
 * a covariance is guaranteed to exist, because the limiter had to read one to throttle at all. The
 * normal case is the one nobody re-tests after a fix; it is left alone by construction.
 *
 * Wedge keeps its existing meaning and its existing response. ThrottleStall is REPORTED, never escaped:
 * the cure for "the limiter is holding us at 15%" is not a manoeuvre, and pretending otherwise is how a
 * recovery reflex becomes a second fault. What the caller gains is that the condition stops being
 * invisible.
 *
 * ★NO NEW TUNED NUMBER. slip_ratio and confirm_ms keep their meanings and their values; slip_ratio is
 * reused for the attribution because "a healthy fraction of what was asked" is the same notion in both
 * halves of the question.
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <Eigen/Dense>

namespace rc
{

enum class StallVerdict
{
    None,           // moving as intended, or nothing was asked of it
    Wedge,          // told to travel, did not — physical obstruction. Escape.
    ThrottleStall,  // our own speed limiter removed the travel. Report; do NOT escape.
    // ★TURNING FOREVER AND ARRIVING NOWHERE (added 2026-08-18 from a live failure). The base obeys
    // every command, so no "commanded vs achieved" test can see it: adv is 0, rot is at its cap, the
    // heading sweeps back and forth and the robot does not move. Measured on the robot at (1.57,-2.75):
    // cmd_adv 0.000 with cmd_rot alternating +0.800/-0.800 for 20+ s, track_s pinned at 0.00, while
    // stuck_ms sat at 0 the whole time because the window only ever opened on TRANSLATION.
    // It is a livelock, not an obstruction — but it needs the same thing a wedge does: something must
    // change the pose and force a replan, which is what the escape does.
    Spinning
};

class StallJudge
{
public:
    struct Params
    {
        float slip_ratio = 0.25f;    // achieved/asked below this over a full window is a failure
        float confirm_ms = 1500.f;   // how long the robot is watched before being judged
    };

    struct Report
    {
        StallVerdict verdict = StallVerdict::None;
        float window_s = 0.f;
        float asked_m = 0.f;       // travel the TRACKER asked for over the window
        float commanded_m = 0.f;   // travel actually commanded to the base over the window
        float achieved_m = 0.f;    // NET displacement from the window's anchor — immune to jitter
        float delivered = 1.f;     // commanded_m / asked_m: how much of the ask survived the limiter
        float asked_rot_rad = 0.f;     // heading the robot was TOLD to sweep over the window
        float achieved_rot_rad = 0.f;  // NET heading change over it — near zero for a chatter
    };

    void reset() { armed_ = false; since_ms_ = 0; asked_m_ = 0.f; commanded_m_ = 0.f; asked_rot_ = 0.f; }
    [[nodiscard]] bool armed() const { return armed_; }
    // The window's start, or 0 when none is open. ★`armed_` is a separate flag rather than `since_ms_ != 0`
    // because a timestamp of 0 is a legal instant — it is what a bench, a replay or a monotonic clock
    // starting at zero hands you, and folding "no window" into that value silently swallows the first
    // window in every one of those. The live agent passes epoch milliseconds and would never have shown it.
    [[nodiscard]] std::uint64_t since_ms() const { return armed_ ? since_ms_ : 0; }

    // One control cycle. `asked_mps` is the tracker's translation BEFORE any limiter; `commanded_mps` is
    // what was actually sent to the base. They are equal when nothing is limiting. `pose_sigma_m` is the
    // localiser's position sigma — the resolution at which "did it move" can be answered at all; it is
    // consulted ONLY on the throttled branch (see the header), so 0 is a safe value everywhere else.
    Report note(bool pursuing, float asked_mps, float commanded_mps, float pose_sigma_m,
                const Eigen::Vector2f &pos_room, float heading_rad, float commanded_rot_rps,
                std::uint64_t now_ms, const Params &p)
    {
        Report r;
        // ★GATED ON THE ASK, NOT ON THE COMMAND. Nothing is being predicted unless the tracker wants to
        // translate, so a terminal rotation or a deliberate pivot opens no window — correct, and the
        // reason the sharp-curve pivot at the head of a stall is not itself reported. Gating on the
        // COMMAND, as this did before, also closes the window whenever a limiter zeroes it, which is
        // precisely when the robot is most stuck.
        // ★ANY COMMANDED MOTION OPENS THE WINDOW, not translation alone. Gating on translation was
        // what made the spin invisible: a robot ordered to rotate at its cap forever is being asked for
        // something, and "nothing is predicted" was simply false. Rotation now carries its own
        // prediction, tested the same way translation is.
        if (not pursuing or not (asked_mps > 0.f or std::abs(commanded_rot_rps) > 0.f))
        {
            reset();
            return r;
        }
        if (not armed_)
        {
            armed_ = true;
            since_ms_ = now_ms;
            last_ms_ = now_ms;
            anchor_ = pos_room;
            anchor_heading_ = heading_rad;
            asked_m_ = 0.f;
            commanded_m_ = 0.f;
            asked_rot_ = 0.f;
            return r;
        }
        const float dt = static_cast<float>(now_ms - last_ms_) * 1e-3f;
        last_ms_ = now_ms;
        if (dt > 0.f)
        {
            asked_m_ += asked_mps * dt;
            commanded_m_ += std::max(0.f, commanded_mps) * dt;
            // How much heading the robot was TOLD to sweep — the rotational twin of asked_m_.
            asked_rot_ += std::abs(commanded_rot_rps) * dt;
        }

        // ★JUDGED AT THE END OF A FULL WINDOW, NEVER CONTINUOUSLY. Early on, the asked travel is a few
        // centimetres, so ANY pose jitter clears slip_ratio x that, restarts the clock, and the window
        // can never mature.
        if (static_cast<float>(now_ms - since_ms_) <= p.confirm_ms) return r;

        r.window_s = static_cast<float>(now_ms - since_ms_) * 1e-3f;
        r.asked_m = asked_m_;
        r.commanded_m = commanded_m_;
        r.achieved_m = (pos_room - anchor_).norm();
        r.delivered = asked_m_ > 1e-6f ? commanded_m_ / asked_m_ : 1.f;
        r.asked_rot_rad = asked_rot_;
        // NET heading change, the short way round. The discriminator for a spin is that this stays near
        // zero while asked_rot_ grows: a converging turn nets what it swept, a chatter nets nothing.
        r.achieved_rot_rad = std::abs(std::remainder(heading_rad - anchor_heading_,
                                                     2.f * static_cast<float>(M_PI)));

        // ★THE ROTATIONAL TEST FIRST, because a spin asks for no translation and every test below is
        // about translation. It is the SAME question the wedge test asks, on the other axis: you were
        // told to sweep this much heading — did you end up anywhere new? A converging turn nets what it
        // swept; a chatter sweeps radians and nets nothing, which is the whole signature.
        // Requires the robot to be going nowhere as well, so a robot turning WHILE driving is untouched.
        const bool going_nowhere = r.achieved_m < std::max(0.02f, p.slip_ratio * asked_m_);
        if (asked_rot_ > 0.f and going_nowhere
            and r.achieved_rot_rad < p.slip_ratio * asked_rot_)
            r.verdict = StallVerdict::Spinning;
        else if (r.achieved_m >= p.slip_ratio * asked_m_)
            r.verdict = StallVerdict::None;                    // going where it intended
        // A real command (the limiter left most of the ask alone) that the base then failed. BOTH halves
        // are required — see the header for the two live failures that each half alone produced.
        else if (r.delivered >= p.slip_ratio and r.achieved_m < p.slip_ratio * commanded_m_)
            r.verdict = StallVerdict::Wedge;
        else if (r.achieved_m > std::max(0.f, pose_sigma_m))
            r.verdict = StallVerdict::None;                    // base obeyed; throttled but still moving
        else
            r.verdict = StallVerdict::ThrottleStall;           // base obeyed; the limiter stopped us

        // Every verdict starts the next window HERE, so each one is about one stretch and a persisting
        // condition re-reports at the window cadence instead of latching or falling silent.
        since_ms_ = now_ms;
        anchor_ = pos_room;
        anchor_heading_ = heading_rad;
        asked_m_ = 0.f;
        commanded_m_ = 0.f;
        asked_rot_ = 0.f;
        return r;
    }

    static bool self_test();

private:
    bool armed_ = false;
    std::uint64_t since_ms_ = 0, last_ms_ = 0;
    Eigen::Vector2f anchor_ = Eigen::Vector2f::Zero();
    float anchor_heading_ = 0.f;
    float asked_m_ = 0.f, commanded_m_ = 0.f, asked_rot_ = 0.f;
};

// ── SELF TEST ────────────────────────────────────────────────────────────────────────────────────
// Inline so the unit stays header-only and every tool that includes it can run it.
inline bool StallJudge::self_test()
{
    bool ok = true;
    const auto check = [&ok](bool cond, const char *what)
    {
        if (not cond) { ok = false; std::printf("  FAIL: %s\n", what); }
    };
    const Params p;   // the shipped values: slip 0.25, confirm 1500 ms

    // Drive a scenario at 20 Hz for `secs`: a constant ask, a constant delivered fraction, a constant
    // real speed along +x, and a reported sigma.
    // ★JITTER IS PART OF THE FIXTURE, NOT NOISE IN IT. A robot standing still does not report the same
    // pose twice — it wanders a centimetre or two — and that wander is the entire reason the old
    // detector never fired: it cleared a bar of a few millimetres. A test on a PERFECTLY still robot
    // would "pass" against the old code too and prove nothing.
    // ★It must also not ALIAS with the window. A square wave of period 4 samples put the anchor and the
    // verdict on the same phase, so the wander cancelled to exactly zero and case (3) reported achieved
    // 0.0000 m — the reproduction quietly became the thing it was meant to rule out. Incommensurate
    // frequencies, deterministic so the test cannot flake.
    const auto wander = [](int i, float a) { return a * (std::sin(i * 2.399f) + 0.6f * std::sin(i * 0.717f)); };
    const auto run = [&](float asked_mps, float delivered, float real_mps, float sigma, float secs,
                         float jitter_m = 0.010f)
    {
        StallJudge j;
        Report last;
        Eigen::Vector2f truth{0.f, 0.f};
        for (int i = 0; i * 50 <= static_cast<int>(secs * 1000.f); ++i)
        {
            const std::uint64_t t = static_cast<std::uint64_t>(i) * 50u;
            const Eigen::Vector2f reported{truth.x() + wander(i, jitter_m),
                                           truth.y() + wander(i + 7, jitter_m)};
            // Pure-translation cases: heading fixed, no rotation commanded.
            const auto r = j.note(true, asked_mps, asked_mps * delivered, sigma, reported,
                                  0.f, 0.f, t, p);
            if (r.verdict != StallVerdict::None and last.verdict == StallVerdict::None) last = r;
            truth.x() += real_mps * 0.05f;
        }
        return last;
    };

    // (1) HEALTHY: asked 0.4, nothing limiting, and the robot really moves. Never a verdict.
    {
        const auto r = run(0.40f, 1.0f, 0.40f, 0.06f, 6.0f);
        std::printf("  healthy drive: verdict %d\n", static_cast<int>(r.verdict));
        check(r.verdict == StallVerdict::None, "a robot going where it was told must never be judged");
    }

    // (2) REAL WEDGE, LIMITER INERT: told to go at 0.4, nothing limiting, and it does not move.
    // ★This is the pre-existing behaviour and it must be unchanged — the normal case is the one nobody
    // re-tests after a fix. Note sigma is deliberately ABSURD (1 m) to prove the sigma test cannot
    // reach this branch: if it ever did, a large sigma would silently disable wedge detection.
    {
        const auto r = run(0.40f, 1.0f, 0.0f, 1.00f, 6.0f);
        std::printf("  wedge (limiter inert, sigma 1.00 m): verdict %d  asked %.3f m, commanded %.3f m, "
                    "achieved %.3f m\n", static_cast<int>(r.verdict), r.asked_m, r.commanded_m, r.achieved_m);
        check(r.verdict == StallVerdict::Wedge, "★an unlimited command that achieves nothing is a WEDGE, "
                                                "exactly as before — escape is the right response");
        check(r.delivered > 0.99f, "★and sigma must be UNREACHABLE here, or a bad covariance could switch "
                                   "off wedge detection altogether");
    }

    // (3) ★THE BUG, REPRODUCED FROM THE LIVE TRACE. The tracker asks 0.071 m/s, the covariance limiter is
    // floored at 0.15 and delivers 0.011, sigma is 0.20 m, and the robot does not move. These are the
    // measured numbers at route_s 32.30 m of run 20260816-154346, where it stood still for 15 s and
    // nothing fired.
    {
        const auto r = run(0.071f, 0.15f, 0.0f, 0.20f, 6.0f);
        std::printf("  throttled to 15%%, not moving: verdict %d  asked %.4f m, commanded %.4f m, "
                    "achieved %.4f m, delivered %.2f\n", static_cast<int>(r.verdict), r.asked_m,
                    r.commanded_m, r.achieved_m, r.delivered);
        check(r.verdict == StallVerdict::ThrottleStall,
              "★a robot held down by our OWN limiter must be reported as a throttle stall");
        check(r.verdict != StallVerdict::Wedge,
              "★...and must NOT be escaped: reversing does not fix a speed limiter, and the reflex would "
              "fire, end, find it still floored, and fire again");
        // The old detector's bar, reconstructed on the same window. It integrated the COMMANDED travel,
        // so the displacement it demanded was a few millimetres — which the pose wander above clears.
        const float old_bar = p.slip_ratio * r.commanded_m;
        std::printf("  old bar %.4f m vs jitter %.4f m achieved -> old verdict would be '%s'; "
                    "ask-based bar is %.4f m\n", old_bar, r.achieved_m,
                    r.achieved_m >= old_bar ? "healthy" : "wedge", p.slip_ratio * r.asked_m);
        check(r.achieved_m >= old_bar,
              "★THE DEFECT: pose wander alone must clear the old commanded-travel bar, so the old "
              "detector called this frozen robot healthy. If this ever fails the reproduction is wrong "
              "and the rest of this case proves nothing");
        check(p.slip_ratio * r.asked_m > r.achieved_m,
              "...while the ask-based bar is above the wander, which is what makes the verdict possible");
    }

    // (4) A THROTTLED ROBOT THAT IS STILL MAKING PROGRESS IS NOT STALLED. Same 15% limiter, but it is
    // genuinely creeping at the reduced speed with a converged filter. Slow is not stuck — and 46% of
    // the measured run was throttled, so a rule that fired here would bury the case that matters.
    // ★This case is why the sigma clause exists: an earlier version without it failed exactly here.
    {
        const auto r = run(0.40f, 0.15f, 0.40f * 0.15f, 0.06f, 6.0f);
        std::printf("  throttled but creeping (sigma 0.06): verdict %d\n", static_cast<int>(r.verdict));
        check(r.verdict == StallVerdict::None,
              "★achieving the throttled speed is still short of the ask — but this must not fire, or "
              "every legitimate slow section becomes a stall report");
    }

    // (5) PARTIAL THROTTLE THAT IS NOT THE CAUSE. The limiter took 40%, the base got a real command and
    // achieved none of it. That is a wedge, and the limiter is a bystander.
    {
        const auto r = run(0.40f, 0.60f, 0.0f, 0.06f, 6.0f);
        std::printf("  60%% delivered, achieving NOTHING: verdict %d (delivered %.2f)\n",
                    static_cast<int>(r.verdict), r.delivered);
        check(r.verdict == StallVerdict::Wedge,
              "a real command that achieves nothing is a WEDGE whatever the limiter was doing");
    }

    // (5b) ★THE SPURIOUS ESCAPE, REPRODUCED. The four false wedges of run 20260816-202001: the limiter
    // had taken ~71% of the ask, but the base then achieved roughly HALF of the distance it was actually
    // told to cover — it was obeying. The first version of this file asked "did the limiter take much?"
    // (delivered 0.29 > 0.25 ⇒ blame the base) and reversed the robot four times in one tour.
    // The base's obedience is the only thing a wedge may be judged on.
    // Numbers straight from the CSV row at t+45.5 s: asked 0.651 m, commanded 0.188 m, achieved 0.089 m.
    {
        const auto r = run(0.42f, 0.29f, 0.42f * 0.29f * 0.47f, 0.177f, 6.0f);
        std::printf("  limiter took 71%%, base achieved ~47%% of its command: verdict %d "
                    "(delivered %.2f, achieved/commanded %.0f%%)\n", static_cast<int>(r.verdict),
                    r.delivered, r.commanded_m > 1e-6f ? 100.f * r.achieved_m / r.commanded_m : 0.f);
        check(r.verdict != StallVerdict::Wedge,
              "★★a base that covered half of what it was TOLD to cover is not wedged — calling it one "
              "reverses a healthy robot and drops a phantom obstacle. This fired 4x in one live tour");
        check(r.verdict == StallVerdict::ThrottleStall,
              "...it is the limiter holding it down, which is a report, not a manoeuvre");
    }

    // (5c) AND THE ESCAPE PATH MUST NOT BE WEAKENED BY ANY OF THIS. A genuinely blocked robot, with the
    // limiter also active, still has to be escaped: it achieved almost none of its command.
    {
        const auto r = run(0.42f, 0.29f, 0.001f, 0.177f, 6.0f);
        std::printf("  limiter took 71%%, base achieved ~1%% of its command: verdict %d\n",
                    static_cast<int>(r.verdict));
        check(r.verdict == StallVerdict::Wedge,
              "★a throttled robot that ALSO cannot move is still wedged — the fix must not blunt the "
              "reflex it was not aimed at");
    }

    // (6) A PIVOT IS NOT A STALL — AND THE DISCRIMINATOR IS NET HEADING, NOT ABSENCE OF TRANSLATION.
    // ★THE CONTRADICTION THIS TEST CARRIED, RESOLVED. The case was written as "adv = 0 ⇒ no ask ⇒ no
    // window", and it passed only because it also passed commanded_rot_rps = 0 and a FROZEN heading —
    // i.e. it was not a pivot at all, it was a robot being asked for nothing. The moment rotation
    // became a first-class ask, the two claims ("a pivot must never fire" / "a spin must fire") were
    // pointed at the same scenario and one had to give.
    // ★WHAT SURVIVES: the pivot. What is withdrawn is the REASON — "it asks for no translation" was
    // never why a pivot is healthy. A pivot is healthy because IT GETS SOMEWHERE: it nets the heading
    // it sweeps. A spin sweeps the same radians and nets nothing. Those are distinguishable without
    // ever consulting adv, so nothing has to be exempted by category, which is the whole point.
    // (6a) A CONVERGING PIVOT: rotation at 0.8 rad/s, adv = 0, heading actually turning. Must NOT fire.
    {
        StallJudge j;
        bool fired = false;
        for (int i = 0; i < 200; ++i)
        {
            const float th = 0.8f * static_cast<float>(i) * 0.05f;   // the heading it was told to sweep
            if (j.note(true, 0.f, 0.f, 0.20f, {0.f, 0.f}, th, 0.8f,
                       static_cast<std::uint64_t>(i) * 50u, p).verdict
                != StallVerdict::None) fired = true;
        }
        std::printf("  10 s converging pivot (adv=0, heading turning): fired %s\n", fired ? "YES" : "no");
        check(not fired, "★a pivot that nets the heading it sweeps is doing its job — it is the sharp-curve "
                         "law working, and reversing it would drop a phantom obstacle mid-corner");
    }

    // (6b) THE SAME COMMAND, GOING NOWHERE: rotation at its cap alternating sign, heading oscillating
    // over a few degrees, net zero. This is the live signature — cmd_rot +0.800/-0.800 for 20 s at a
    // fixed pose. It MUST fire, and it must not be reachable by any test that looks only at adv.
    {
        StallJudge j;
        bool spun = false;
        for (int i = 0; i < 200; ++i)
        {
            const float th = 0.05f * std::sin(static_cast<float>(i) * 1.5f);   // ±3 deg, nets nothing
            const float rot = (i % 2 == 0) ? 0.8f : -0.8f;
            if (j.note(true, 0.f, 0.f, 0.20f, {0.f, 0.f}, th, rot,
                       static_cast<std::uint64_t>(i) * 50u, p).verdict
                == StallVerdict::Spinning) spun = true;
        }
        std::printf("  10 s chatter in place (rot at cap, net heading ~0): spinning %s\n",
                    spun ? "YES" : "no");
        check(spun, "★★sweeping radians and arriving nowhere is the livelock the robot sat in for 74 s of "
                    "a 144 s run; if this does not fire, nothing recovers it");
    }

    // (6c) TURNING WHILE DRIVING IS NEVER A SPIN, however little heading it nets — a robot covering
    // ground is going somewhere by definition, and the translation tests below are the ones that own it.
    {
        StallJudge j;
        bool spun = false;
        Eigen::Vector2f pos{0.f, 0.f};
        for (int i = 0; i < 200; ++i)
        {
            pos.x() += 0.4f * 0.05f;                                  // 0.4 m/s of real progress
            const float th = 0.05f * std::sin(static_cast<float>(i) * 1.5f);
            if (j.note(true, 0.4f, 0.4f, 0.20f, pos, th, (i % 2 == 0) ? 0.8f : -0.8f,
                       static_cast<std::uint64_t>(i) * 50u, p).verdict
                == StallVerdict::Spinning) spun = true;
        }
        check(not spun, "★a robot covering 0.4 m/s while its heading wanders is not spinning in place");
    }

    // (7) THE WINDOW MUST MATURE. A verdict before confirm_ms would be decided on a few centimetres of
    // asked travel, which the wander dominates.
    {
        const auto r = run(0.40f, 1.0f, 0.0f, 0.06f, 1.4f);   // just under confirm_ms
        check(r.verdict == StallVerdict::None, "no verdict may be reached before a full window");
    }

    // (8) A PERSISTING STALL MUST KEEP REPORTING, not latch once and fall silent. Fifteen seconds of
    // freeze should read as ten windows, because that is what tells a 1.5 s hiccup from a deadlock.
    {
        StallJudge j;
        int stalls = 0;
        Eigen::Vector2f pos{0.f, 0.f};
        for (int i = 0; i * 50 <= 15000; ++i)
        {
            const Eigen::Vector2f rep{pos.x() + 0.010f * std::sin(i * 2.399f), pos.y()};
            if (j.note(true, 0.071f, 0.011f, 0.20f, rep, 0.f, 0.f,
                       static_cast<std::uint64_t>(i) * 50u, p).verdict
                == StallVerdict::ThrottleStall) ++stalls;
        }
        std::printf("  15 s frozen: %d throttle-stall verdicts (one per ~%.1f s window)\n",
                    stalls, p.confirm_ms / 1000.f);
        check(stalls >= 8, "★a deadlock must report throughout, or the log cannot distinguish 15 s of "
                           "frozen from one bad window");
    }

    std::printf("StallJudge::self_test %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

}  // namespace rc
