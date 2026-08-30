#include "plain_tracker.h"

#include "../route_spline.h"

#include <algorithm>
#include <cmath>

namespace rc
{
namespace
{
float wrap_pi(float a) { return std::remainder(a, 2.f * static_cast<float>(M_PI)); }

// Reference speed at which the FEEDBACK's rotation authority is evaluated when the robot is crawling,
// so that authority never reaches zero at the end-of-route taper.
constexpr float kSteerFloorMps = 0.15f;
}   // namespace

// PLAIN TRACKER — curvature feedforward + critically damped Frenet feedback. Follows the route and does
// nothing else: no obstacle data of any kind reaches this file, which is why it takes a PathWorld and
// not a FieldWorld. Safety belongs to the planner's footprint predicate, to the band, and to
// detect_path_blockage -> replan, all of which live outside it.
//
//   omega = g_dc * [ v*kappa_bar(s + v*T_lag)  -  v_steer*( (2/L)e_psi + (1/L^2)e_y ) ]
//
// The feedforward is the kinematic relation omega = v*kappa, previewed by the identified actuator lag.
// It is the only term that reads the route, so the feedback sees a zero-mean residual and cannot
// double-count the nominal turn. The feedback is critically damped in the ARC-LENGTH domain, so an error
// decays over L metres at any speed and no gain is right at one speed while ringing at another.
ControlOutput& PlainTracker::compute(ControlOutput& out, const TrackerInput& in, const TrackerParams& p)
{
    const RouteSpline* spline = world_.route_spline();
    if (spline == nullptr or not spline->valid())
    {
        out.adv = 0.f; out.side = 0.f; out.rot = 0.f;
        return out;
    }
    const RouteSpline& sp = *spline;

    const Eigen::Vector2f pos{in.robot_pose.translation().x(), in.robot_pose.translation().y()};
    const float theta_pose = std::atan2(in.robot_pose.linear()(1, 0), in.robot_pose.linear()(0, 0));
    const float theta_fwd = wrap_pi(theta_pose + static_cast<float>(M_PI_2));   // FRAME conversion (1/2)

    // Monotone-forward projection: the tour crosses itself, so a global nearest-point search would snap
    // onto a branch driven ten metres ago.
    // ★An EMPTY s_hint_ means re-acquire; reset() empties it, and TrajectoryController::stop() and
    // set_route (on a curve change) both call reset(). A monotone projection is
    // only meaningful within ONE traversal — across a mission stop, or a repaired curve, it is stale,
    // and stale arc length pointed the tracker at the route's END POINT after a restart (route_s pinned
    // at 0.20 m, robot driving into a wall). Re-acquiring searches the whole route once and adopts the
    // nearest point: the route start after a stop, the robot's own position after a repair.
    // ★RE-ACQUISITION IS ALSO PATH INITIATION. The one moment "the robot is starting a path" is well
    // defined is the moment the projection is re-acquired: reset() fires on a mission start, on a route
    // repair and on a curve change, and on nothing else. That is where the start-alignment below is
    // armed, so it needs no notion of its own for "new path".
    const bool reacquiring = not s_hint_.has_value();
    if (reacquiring)
        s_hint_ = sp.project(pos, 0.f, sp.length());
    else
        // ★FORWARD-ONLY, THE SAME CLAMP RouteFollower::advance APPLIES (route_follower.cpp:481).
        // `RouteSpline::project` seeds its search at `s_hint - spacing/2` (route_spline.cpp:309) — a
        // deliberate half-cell of backward slack so a curve sampled at 5 cm can still find its own
        // nearest point — so EVERY call may return LESS than it was given. RouteFollower has always
        // clamped that away; this tracker never did, and at 20 Hz the leak is 0.025 m/cycle: ~20 s to
        // walk 10 m of arc length back to 0.00 and pin there. Measured live: `track_s` sat at ~0 on
        // 39% of rows and the longest unbroken advancing stretch in 282 s was 2.8 s, while the
        // cross-track error stayed at p50 0.017 m — the tracker was following its curve beautifully and
        // reporting that it had never left the start.
        // ★The clamp belongs HERE and not inside project(): the half-cell is what makes the search
        // correct, and the re-acquire branch above legitimately needs to move s BACKWARDS (a stop or a
        // repaired curve must be free to land the robot at s = 0, or wherever it now stands). Monotone
        // is a property of ONE traversal, which is exactly what the optional distinguishes.
    {
        // ★AND NO FURTHER FORWARD THAN THE ROBOT ACTUALLY MOVED. See last_pos_ in the header: the
        // forward-only clamp stops s walking BACK, and this stops it leaping AHEAD across a fold. The
        // allowance is the chord the robot covered since the last cycle plus one sample spacing — on
        // the curve those are the same quantity, off it arc must advance less, and the spacing is the
        // slack project() itself needs to find its nearest point. No new constant.
        // ★Absent last_pos_ (first cycle after a reset or a lap resume) means there is nothing to
        // measure against, so the bound is not applied — the projection is trusted exactly where it is
        // the only thing that knows where the robot is.
        const float advance_cap = last_pos_.has_value()
                                    ? (pos - *last_pos_).norm() + sp.spacing()
                                    : std::numeric_limits<float>::max();
        const float projected = sp.project(pos, *s_hint_, std::max(0.05f, p.plain_proj_window));
        s_hint_ = std::clamp(projected, *s_hint_, std::min(sp.length(), *s_hint_ + advance_cap));
    }
    const float s = *s_hint_;
    if (reacquiring) start_align_ = true;

    const Eigen::Vector2f r = sp.position_at(s);
    const float psi = sp.heading_at(s);
    // Cross-track, LEFT POSITIVE — the session's convention, deliberately NOT compute_pd's opposite one.
    const float e_y   = -std::sin(psi) * (pos.x() - r.x()) + std::cos(psi) * (pos.y() - r.y());
    const float e_psi = wrap_pi(theta_fwd - psi);

    // Speed: the route's own geometric profile (curvature and lateral-acceleration limits are already
    // folded into max_adv by route_speed_limit) plus a stop taper on the remaining arc. No field query.
    // ★sqrt(2*a*d) reaches zero exactly at the route end. It was sqrt(0.15^2 + 2*a*d), and that constant
    // under the root is a FLOOR: at zero remaining arc it still commanded 0.15 m/s, so the robot crept
    // past the end of its own route — measured, route_s pinned at 38.42 for the last ten cycles with
    // cmd_adv at 0.146 — and drifted out of RouteFollower::finished()'s 1 m end-reach check.
    // ── HOW THE END-OF-CURVE STOP WAS ACTUALLY BROKEN (2026-08-09/10, RESOLVED) ──────────────────
    // Symptom on the robot: driving a point target, it "jumps over the target and moves forward until it
    // hits an obstacle". Offline it never stopped at all -- 8 m past the end of a 2.5 m curve, speed
    // pinned at 0.068 m/s for ever.
    // ★NEITHER THE PROJECTION NOR THE TAPER WAS WRONG. Both were verified correct by tracing s against
    // length() cycle by cycle: s reaches the last sample and stays, and v_cmd matches sqrt(2*a*s_rem)
    // exactly. The defect was the SEAM between them -- see the s_remaining line below.
    // ★FOUR FIXES WERE TRIED AGAINST THE WRONG CAUSE AND ALL REVERTED. Three were killed by the same
    // fact, A TOUR ENDS WHERE IT BEGAN, so the endpoint sits beside the start:
    //   - taper bounded by the EUCLIDEAN distance to the endpoint -> brakes at the START of a loop
    //     (37 m tour: 88 s -> 400 s, TV(w)/m 2.07 -> 16.34)
    //   - zero the taper once PAST the endpoint's heading plane -> true on cycle one of a loop, same crawl
    //   - arm that guard only inside the braking distance v^2/(2a) -> tour restored, inert in the failing
    //     case, because s_remaining was the thing that was wrong
    //   - bound the PROJECTION's forward search by recent progress (5-cycle history, 3x the arc actually
    //     covered, so a closed turn cannot offer a nearer point across the chord): no effect on the
    //     terminal behaviour whatsoever, and the tour degraded (rms 41.2 -> 46.6 mm, TV(v)/m 0.581 ->
    //     1.048). Sound idea, wrong target -- the projection was already correct.
    // ★AND TWO OF THE MEASUREMENTS WERE HARNESS ARTEFACTS, which is why the wrong cause looked solid:
    //   - tools/tracker_sim sets p.cbf_max_decel = 1e6 to measure STEERING with no end taper. Every
    //     stop test run before 2026-08-10 was therefore of a tracker with no brake at all.
    //   - the stop test's own overshoot cutoff saturated, so every failing variant reported an identical
    //     "1.507 m" and they could not be told apart.
    // Reproduce the real thing: tools/tracker_sim <short route> --stop-test [--trace].
    const float a_dec = std::max(0.1f, p.cbf_max_decel);
    // ★s IS A SAMPLE, AND THE LAST SAMPLE IS NOT THE END. project() returns the nearest sample and they
    // are `spacing` apart, so length() - s has a FLOOR of (length mod spacing) -- 0.002 m on a 2.5 m
    // curve -- and sqrt(2*a*s_rem) therefore has a floor too: ~0.06 m/s that never decays. The robot
    // creeps through its own endpoint for ever (measured: 8 m past, v pinned at 0.068 m/s, s_rem pinned
    // at 0.002). This is the SAME defect the note above records for the old sqrt(0.15^2 + 2*a*d): that
    // explicit floor was removed, and this implicit one survived in the sampling grid.
    // The remaining arc is only known to within one spacing, so take the conservative end of that
    // interval: the taper can then reach zero, and it stops a fraction of a sample early rather than
    // never. No new constant -- the spacing is the curve's own.
    const float s_remaining = std::max(0.f, sp.length() - s - sp.spacing());
    const float v_stop = std::sqrt(2.f * a_dec * s_remaining);
    const float v_profile = std::clamp(std::min(p.max_adv, v_stop), 0.f, p.max_adv);

    // kappa_avg is CENTRED, so this preview is the one and only lookahead in the law.
    // ── THE PREVIEW MUST NOT LOOK PAST THE END OF THE CURVE ──────────────────────────────────────
    // ★ROOT CAUSE of "upon arriving it turns a bit, and it was not commanded to" (measured on the robot
    // 2026-08-13). kappa_avg(x, W) is a CENTRED difference — it reads heading_at(x ± W/2) — and
    // heading_at returns a FLAT 0 rad for any argument past the curve: position_at clamps both of its
    // probes to samples_.back(), the difference vector collapses, and the guarded fallback is 0.f.
    // ★That zero is not a sentinel. 0 rad is an ordinary heading (due +x), so there is no way for a
    // caller to tell "off the end" from "the route points east" — and this one could not.
    // The feedforward previews v*T_lag = 0.29 m at full speed, which runs off the end of a curve the
    // robot is about to finish, so kappa_avg differenced a real heading against that zero:
    //     k_ff = remainder(0 - 1.15, 2pi) / 0.40 = -2.87 1/m
    // on a stretch whose real curvature is about 1 1/m. omega saturated and the robot turned TOWARD
    // HEADING ZERO through its last 0.15 m — heading 1.1502 -> 0.8671 with e_y pinned at 0.011 m and
    // e_psi at 0.021, so the FEEDBACK was asking for nothing whatsoever.
    // ★FIXED HERE, NOT IN heading_at, AND THAT WAS MEASURED TOO. Clamping inside heading_at is the
    // tempting one-liner and it is a much larger change than it looks: the bogus curvature was also
    // feeding route_speed_limit, where it acted as an accidental END-OF-ROUTE BRAKE. Removing it made
    // the robot arrive too fast — the tour went 23 s -> 94 s, driven 9.7 -> 36.5 m, and the stop test
    // overshot the endpoint by 4.079 m instead of 0.303. Exactly the "broken bound holding the system
    // together by accident" this file's header records for lap 5. That brake deserves to exist on
    // purpose or not at all, and either way it is not this fix.
    // A centred window needs W/2 of curve on BOTH sides, so that — not length() — is the bound.
    const float k_ff = sp.kappa_avg(std::min(s + v_profile * p.plain_T_lag,
                                             sp.length() - 0.5f * p.plain_W), p.plain_W);
    const float L = std::max(0.05f, p.plain_L);
    const float k_fb = -(2.f / L) * e_psi - (1.f / (L * L)) * e_y;

    // ── ADVANCE IS BOUND BY THE TURN THE ROBOT CAN ACTUALLY DELIVER ─────────────────────────────
    // omega = g_dc*v*(k_ff + k_fb), so requiring |omega| <= max_rot inverts to a speed limit. Without
    // it omega simply CLIPS at max_rot while v carries on: the robot cannot come round, overshoots,
    // and the error that caused the demand persists.
    // ★MEASURED 20:54, the lap that hit the counter at the hairpin: |cmd_rot| sat at the 0.8 rad/s cap
    // on 1614 of 3039 cycles — 53.1% — with mean cmd_adv 0.677 m/s. It drove 63.9 m of a 37.6 m route
    // (ratio 1.70) with cross-track max 2.24 m. Mean |kappa| at those cycles was only 0.65, so v*kappa
    // = 0.44 and the FEEDFORWARD alone never saturates: it is the FEEDBACK, (1/L^2)*e_y = 2.78 per
    // metre, so 0.3 m off-route demands 0.83 rad/s by itself. Off-route -> saturate -> cannot converge
    // -> stay off-route.
    // ★The rotation is computed FIRST, at the speed the route would allow; then the advance is scaled by
    // exactly the factor the rotation had to be clipped by. That factor preserves the commanded ARC:
    // kappa = omega/v is unchanged, so a saturated turn becomes a slower turn of the SAME shape rather
    // than a wider one. Clipping omega alone silently changes the arc, which is the failure above.
    // ★No new parameter: max_rot and g_dc already exist, and this is omega = v*kappa read backwards.
    // ── THE STEERING FLOOR MUST DIE WITH THE ROUTE, OR THE ROBOT TURNS WHERE IT ARRIVES ──────────
    // Reported from the robot: "it advances OK but upon arriving it turns some 45 degrees, and it was
    // not commanded to". Confirmed in mppi_diag.csv — position frozen to the centimetre at (0.51,-1.25)
    // while cmd_rot sat on the -0.800 cap and the heading swept 1.612 -> 1.953 rad, still going.
    // ★THE SPEED TAPERS TO ZERO AND THE ROTATION AUTHORITY DOES NOT. v_profile = sqrt(2*a*s_remaining)
    // reaches zero at the endpoint, but the feedback is evaluated at max(v_profile, kSteerFloorMps), a
    // CONSTANT — so at the endpoint omega = g_dc*0.15*k_fb, which is not zero whenever any cross-track
    // is left. The robot cannot reduce e_y without translating, so it converts it into the only thing it
    // still has authority over: it spins, settling where k_fb = 0, i.e. e_psi = -e_y/(2L). At L = 0.50
    // that is a radian of turn per metre of leftover cross-track — 45 deg is 0.8 m of e_y.
    // ★The floor is right in what it was for and wrong about WHEN. It exists so a crawling robot can
    // still steer; a robot in the taper is crawling AND still going somewhere, and a robot at the
    // endpoint is crawling and has ARRIVED. The arc that remains is exactly what separates those two,
    // so the floor is weighted by it — over the length the feedback is designed to converge in, which is
    // the law's own L. Inert wherever more than L of route is left (everything but the last 0.5 m);
    // zero exactly where the route is. Continuous, and no new constant.
    // ★Deliberately NOT applied to k_fb itself: at speed the feedback should keep its full gain: it is
    // the artificial floor, not the feedback, that has no justification once there is nowhere left to go.
    const float steer_floor = kSteerFloorMps * std::clamp(s_remaining / L, 0.f, 1.f);
    const float omega_want = p.plain_g_dc * (v_profile * k_ff
                                             + std::max(v_profile, steer_floor) * k_fb);
    const float omega_ccw = std::clamp(omega_want, -p.max_rot, p.max_rot);
    const float scale = std::abs(omega_want) > 1e-6f
                      ? std::abs(omega_ccw) / std::abs(omega_want) : 1.f;
    // ★EXPONENTIAL BRAKE on the turn rate, times the saturation ratio's job — BOTH, not either:
    //   brake — "slow down as the turn demand grows". Acts BEFORE saturation, which is the approach to
    //           every hairpin: at omega just under max_rot the ratio is still 1 and the robot would
    //           otherwise drive flat out through the turn.
    //   scale — "do not demand a turn the robot cannot deliver". Acts only ONCE omega saturates,
    //           decaying as 1/demand.
    // ★THE EXPONENT USES THE DELIVERABLE RATE (omega_ccw, clamped), NOT THE DEMAND. With omega_want
    // unbounded the brake goes to ZERO and that DEADLOCKS: e_y can only be reduced by MOVING, so a
    // robot at v = 0 pivots at max rate forever and never approaches the path. Observed driving a
    // single-shot target, whose plan starts off-path: cmd_adv = 7.5e-19, cmd_rot pinned at -0.800,
    // demand ~10 rad/s (at L = 0.50 the feedback alone gives 4 rad/s per metre of cross-track).
    // ⚠CONSEQUENCE: r_om is pinned at 1 past the cap, so the brake is CONSTANT there at exp(-k) —
    // it no longer "keeps falling past saturation", and scale is what acts beyond the cap. That is why
    // both terms are kept. It also means the k = 0.25 optimum was measured on the UNBOUNDED exponent
    // (3 laps, 2026-08-05, before this bound) and has not been re-measured since.
    // At k = 0 this reduces to the saturation ratio alone.
    const float r_om = omega_ccw / std::max(0.05f, p.max_rot);
    const float brake = std::exp(-std::max(0.f, p.plain_brake_k) * r_om * r_om);

    // ── DO NOT TRANSLATE INTO SPACE YOU ARE NOT FACING ───────────────────────────────────────────
    // The lidar self-filter discards every return within max(0.40, body_max + 0.08) m of the robot
    // centre (trajectory_controller.cpp), so against a 0.325 m circumscribed body the robot is BLIND
    // within about 8 cm of its own hull. Nothing downstream can see furniture that is already inside
    // that annulus. The one thing that keeps the swept body out of unseen space is therefore a
    // property of the MOTION, not of any sensor: the robot may advance only into the space it is
    // pointing at, which is the space it has been looking at on the way in.
    // Reported from the robot: through a sharp turn it kept translating and clipped the furniture.
    // The speed profile alone cannot fix that — h_eff*w_max/kappa is a 1/kappa law that gets small
    // but never reaches zero, and the saturation ratio bottoms out at max_rot/|k_cmd|, which at the
    // tour's tightest is still ~0.09 m/s. Held for the ~4 s a point turn takes, that is a third of a
    // metre of unseen sideways sweep.
    //
    //   gate = max(0, cos(min(turn_ahead, misalign)))
    //
    // turn_ahead — how far the ROUTE turns over its own smoothing window: a property of the path.
    // misalign   — how far the ROBOT still has to turn to face where the route goes: a property of
    //              the pose.
    // ★THE MINIMUM OF THE TWO, and that is the whole design, not a detail:
    //   • on a STRAIGHT, turn_ahead is 0, so the gate is exactly 1 no matter how badly the robot is
    //     misaligned. This is what makes cross-track recovery impossible to deadlock — and a gate on
    //     misalignment ALONE does deadlock, at e_y > ~pi/2*L*2, because the feedback settles the robot
    //     at e_psi ~ e_y/(2L) off the tangent and cos of that is <= 0. e_y can only be reduced by
    //     MOVING; the same trap the unbounded brake exponent fell into (see the note above).
    //   • at a CUSP already aligned, misalign is 0, so the gate is 1 and the robot drives out of the
    //     turn instead of sitting in it. Only "the route turns hard here AND I am not yet pointing
    //     that way" stops the wheels, and rotating is what clears it, so it always terminates.
    // ★THE STEERING MUST BE RETARGETED WITH IT, AND GATING v ALONE DEADLOCKS. Measured, not argued:
    // with the gate on v only, the bench froze for 385 of 400 s above 90 deg of turn. The reason is
    // structural and is the whole difficulty here — this law's rotation is PROPORTIONAL TO v
    // (omega = g_dc*v*kappa), so it cannot turn in place, and the Frenet feedback is written in ARC
    // LENGTH, which presumes s is advancing. Stop the wheels and the curvature demand settles to zero
    // a few degrees from where it started, pointing at the tangent AT s while the gate is watching the
    // tangent AHEAD of s. Nothing then rotates the robot out of its own gate.
    // So the same gate blends the turn command from tracking toward a plain HEADING controller aimed
    // at psi_ahead:
    //     omega = gate*omega_track + (1 - gate)*omega_pivot,  omega_pivot = g_dc*v_floor*(2/L)*misalign
    // At gate = 1 this is exactly the previous law, untouched. At gate = 0 it is v = 0 with a heading
    // loop — a true point turn — and it always TERMINATES, because omega_pivot drives misalign down
    // and that is what re-opens the gate.
    // ★omega_pivot introduces NO new constant: it is the law's own heading term, g_dc*(2/L) evaluated
    // at the existing steering floor, retargeted from psi(s) to psi_ahead.
    const float psi_ahead   = sp.heading_at(std::min(s + p.plain_W, sp.length()));
    const float turn_ahead  = std::abs(wrap_pi(psi_ahead - psi));
    const float align_err   = wrap_pi(psi_ahead - theta_fwd);          // SIGNED: which way to spin

    // ── THE ONE BOUNDARY, USED TWICE, AND IT IS DERIVED RATHER THAN TUNED ────────────────────────
    // A quarter turn is where advancing STOPS MAKING PROGRESS toward where you are going: past 90 deg
    // the component of forward motion along the outgoing tangent changes sign. That is the same fact
    // whether it is asked of the route ("does this turn reverse me?") or of the pose ("am I facing the
    // way out yet?"), so both ends of the decision use it and there is no gain to pick.
    constexpr float kQuarterTurn = static_cast<float>(M_PI_2);
    // ENTRY asks the ROUTE and the POSE together. Release does NOT ask the same boundary back — see the
    // measured limit cycle below.
    //
    // ── "ALIGNED" IS A BOUND ON THE EXCURSION, AND IT IS DERIVED, NOT PICKED ─────────────────────
    // A heading loop is asymptotic, so a release needs a bound, and the honest one is what the RESIDUAL
    // BUYS. Released onto the path with heading error e_psi, the critically damped Frenet feedback gives
    // e_y(s) = e_psi*s*exp(-s/L), peaking at s = L:  e_y_max = e_psi*L/e. Require that peak to stay
    // inside the resolution at which the route is even KNOWN — one sample spacing — and invert:
    //     |align_err| <= e*spacing/L
    // 0.27 rad at the shipped L = 0.50 and RouteSpacing = 0.05. No new constant: one is the law's, one
    // is the curve's. ★The excursion is written out rather than folded away so it can be RE-DERIVED: if
    // this tracker is ever handed the corridor width the planner actually certified, that width — not
    // the sample spacing — is what the peak should be measured against.
    const float align_tol = 2.71828183f * sp.spacing() / L;

    // ── FACE THE PATH BEFORE DRIVING IT ──────────────────────────────────────────────────────────
    // ★AT PATH INITIATION MISALIGNMENT ALONE IS THE CRITERION, and that is NOT the deadlock the
    // stateless gate fell into. min(turn_ahead, misalign) exists because e_y can only be reduced by
    // MOVING, so a gate on misalignment alone freezes a robot that is off-route on a straight. That
    // premise is FALSE at the start of a path: the route is planned FROM the robot, so there is no
    // cross-track to work off — only heading — and rotating reduces the whole of the error. Which is
    // why this asks the POSE only, and the mid-route latch below still asks the ROUTE as well.
    // ★WHY IT MUST NOT ARC ON. Every avoidance guarantee this tracker leans on is a property of the
    // PATH: the planner's footprint predicate certified those samples, the band deforms those samples,
    // repair splices those samples. An arc flown onto the path from a misaligned start is off the
    // certified curve by exactly however much it bulges, and NOTHING has ever looked at that space.
    // ★Armed once per acquisition and it cannot re-arm within a traversal, so it can never invent a
    // mid-route stop.
    // ★IT AIMS AT THE TANGENT AT THE PROJECTION, psi(s) — NOT at psi_ahead, which is the mid-route
    // pivot's target. They are different questions and the bench caught the difference: tracker_sim
    // starts the robot exactly ON the route and exactly TANGENT to it (P.th = heading_at(0)), which is
    // a correctly-started robot with nothing to align, yet aiming this at psi(s+W) made it pivot off
    // the tangent on any curving start — chasing a chord 0.40 m ahead instead of the direction the
    // route actually leaves in. rms 40.4 -> 47.4 mm on route_world.txt. Against psi(s) the same start
    // gives align error 0, the state releases on cycle one, and the bench is bit-for-bit unchanged.
    // ★And nothing is skipped: with the robot facing psi(s), the arc length whose heading it matches IS
    // s, so the adopt below is a no-op here. A start pivot must not spend route the robot never drove.
    const float start_err = wrap_pi(psi - theta_fwd);      // SIGNED: which way to spin, to the TANGENT
    // ── AND NO TURN-IN-PLACE ONCE THERE IS NO ROUTE LEFT TO TURN ONTO ────────────────────────────
    // A pivot is a decision to stop and face where the robot is about to DRIVE. Past the end of the
    // curve there is no "about to drive": psi_ahead is clamped to heading_at(length), so the robot would
    // hold the wheels at zero and spin to face the final tangent — a second, independent way of turning
    // where it arrives, and the same complaint from the robot's point of view. Requiring the window the
    // pivot looks through to still fit on the curve says exactly that, in the pivot's own terms and with
    // no new constant.
    const bool route_ahead = s + p.plain_W < sp.length();
    bool released_now = false;
    const bool start_align_active = start_align_;
    if (start_align_)
    {
        if (std::abs(start_err) <= align_tol or not route_ahead)
        { start_align_ = false; released_now = true; }
    }
    else if (not pivoting_ and route_ahead
             and turn_ahead >= kQuarterTurn and std::abs(align_err) >= kQuarterTurn)
        pivoting_ = true;
    else if (pivoting_ and not route_ahead)
    { pivoting_ = false; released_now = true; }
    // ── THE MID-ROUTE RELEASE IS THE ALIGNMENT BOUND, NOT THE QUARTER TURN ──────────────────────
    // ★THE PREVIOUS NOTE HERE WAS WRONG, AND THE ROBOT FALSIFIED IT (2026-08-23). It read: "the
    // release only LOOKED guilty of the chatter; what made the pair oscillate is that release handed
    // the robot to a law aiming somewhere else. Fix that (the adopt below) and the quarter turn is
    // sound again" — and it priced the alternative at "rms 40.4 -> 47.4 mm for nothing". The adopt is
    // in and the pair still chatters, so that "nothing" was the whole of the benefit.
    // MEASURED, live, on the shipped build: 14.6 s in one 78 min run with cmd_rot alternating between
    // the two caps — HOLD cycles at +0.800 (pivot, wheels at zero) and released cycles at -0.800
    // (Frenet feedback) — in blocks of 8-11 cycles, e_psi oscillating in 0.84..1.04 rad and never
    // converging, s frozen at 3.25 m, and the CROSS-TRACK ERROR GROWING MONOTONICALLY 0.23 -> 0.93 m.
    // The robot walked 0.8 m off a path nothing had certified it to be on, at 0.106 m/s, for 15 s.
    // ★WHY THE ADOPT CANNOT CLOSE THIS. It searches [s, s+W] for the arc length whose heading matches
    // the one the robot achieved. Released at the quarter turn the robot has achieved NOTHING — it is
    // still 90 deg off the outgoing tangent — so no such arc length exists, `best` stays at `s`, and
    // e_psi survives the release at ~1 rad. The feedback then spends the released cycles turning the
    // other way (-(2/L)*e_psi alone is -4 rad/m here) until align_err climbs back over the quarter
    // turn and the entry test, which is the EXACT COMPLEMENT of this one while turn_ahead >= 90 deg,
    // fires again. Zero hysteresis, two laws saturated in opposite directions, 0.62 s of plant dead
    // time so neither command is ever finished: a limit cycle, not a handover.
    // ★AND THE CROSS-TRACK TERM ALONE WOULD STILL REVERSE IT. Even with a perfect adopt (e_psi = 0),
    // (1/L^2)*e_y at e_y = 0.79 m is 3.16 1/m — four times what the base can deliver. So the release
    // cannot be made safe by making the two laws agree; it has to happen where the turn is DONE.
    // ★SO RELEASE ON THE ALIGNMENT BOUND, WHICH IS THIS FILE'S OWN DERIVED CONSTANT (align_tol =
    // e*spacing/L, the heading residual whose cross-track excursion peaks inside one sample spacing).
    // Entry asks the quarter turn, release asks the alignment bound: the hysteresis is now wide and
    // both ends are derived rather than picked. It also REPAIRS the adopt instead of competing with
    // it — a robot released facing the outgoing tangent has an arc length in [s, s+W] that matches,
    // so `best` moves, e_psi starts near zero, and entry cannot re-fire because align_err is small.
    // ★THE COST IS THE ONE ALREADY MEASURED and it is worth paying: rms 40.4 -> 47.4 mm of tracking
    // error, against a 0.93 m departure from the certified curve. See the bench numbers in the commit.
    else if (pivoting_ and std::abs(align_err) <= align_tol)
    { pivoting_ = false; released_now = true; }
    // One heading loop, two targets: the tangent the robot is starting from, or the tangent it must
    // leave a turn on. Captured before the flag is cleared so the release cycle still commands the
    // error it was regulating.
    const float pivot_err = start_align_active ? start_err : align_err;

    // ── ARRIVE ALREADY STOPPED ───────────────────────────────────────────────────────────────────
    // MEASURED, and it is why a reactive gate is not enough: with the wheels commanded to zero from the
    // moment the turn is detected, the robot still swept 0.23 m through the kitchen's hairpin — and
    // 0.14 m of that was pure plant. The command was already at most 0.018 m/s; the base carries a
    // 0.20 s transport delay, a 0.22 s lag and a slew limit, so ~0.4 s of motion is in flight at all
    // times. Deciding to stop AT the corner is deciding half a second too late.
    // So brake for the pivot the way the end of the route is braked: find the first arc length ahead
    // where the route reverses, and bound the speed by what can still be shed before reaching it.
    // ★Scanned over the ROUTE only. A heading-dependent scan would need to guess the robot's future
    // heading, and the whole point is that the approach is decided before the robot gets there.
    // ★This CANNOT deadlock, and that is why the zero lives here and not in route_speed_limit: it is a
    // bound on the approach to a point AHEAD (ds > 0 gives sqrt(2*a*ds) > 0), never a bound at the
    // robot's own s. A hard zero in the speed profile would pin v = 0 at the apex for ever, because the
    // profile is a pure route property and pivoting does not change it.
    float v_pivot_approach = p.max_adv;
    {
        const float horizon = p.max_adv * p.max_adv / (2.f * a_dec) + p.plain_W;
        for (float ds = 0.05f; ds <= horizon; ds += 0.05f)
        {
            const float x = s + ds;
            if (x > sp.length()) break;
            const float turn_x = std::abs(wrap_pi(sp.heading_at(std::min(x + p.plain_W, sp.length()))
                                                  - sp.heading_at(x)));
            if (turn_x < kQuarterTurn) continue;      // drivable: no stop required there
            // ★THE DEAD TIME IS PART OF THE STOPPING DISTANCE, and leaving it out is what made the
            // first attempt useless. sqrt(2*a*ds) is the textbook bound and it assumes braking starts
            // NOW; this base does not begin to slow for T_lag seconds (0.42 s identified), during which
            // it covers v*T_lag. Solving v*T_lag + v^2/(2a) = ds for v:
            //     v = -a*T_lag + sqrt((a*T_lag)^2 + 2*a*ds)
            // At 5 cm out that is 0.106 m/s where the textbook form allowed 0.316 — a factor of three,
            // and the difference between arriving stopped and arriving at a third of full speed.
            // No new constant: T_lag is this law's own identified lag, already used by the feedforward.
            const float at = a_dec * std::max(0.f, p.plain_T_lag);
            v_pivot_approach = std::min(v_pivot_approach,
                                        std::max(0.f, -at + std::sqrt(at * at + 2.f * a_dec * ds)));
            break;                                     // the FIRST one is the binding one
        }
    }

    // While pivoting the wheels are held at zero and the turn command becomes a plain HEADING loop on
    // the outgoing tangent. It introduces no new constant: it is this law's own heading term,
    // g_dc*(2/L) evaluated at the existing steering floor, retargeted from psi(s) to psi_ahead.
    // ★omega must NOT be scaled down with v. The rotation is the only thing that clears the condition
    // holding the robot, and a law whose omega is proportional to v cannot turn on the spot — that is
    // exactly how the stateless version froze for 385 of 400 s in the bench.
    const float omega_pivot = std::clamp(p.plain_g_dc * kSteerFloorMps * (2.f / L) * pivot_err,
                                         -p.max_rot, p.max_rot);
    // The release cycle still commands the pivot. The command below is built from the arc length this
    // cycle projected, and the release is about to move it, so driving on the old one for one cycle
    // would issue exactly the wrong turn — small, 50 ms, but it is the only cycle that could.
    const bool holding = start_align_ or pivoting_ or released_now;
    const float omega_blend = holding ? omega_pivot : omega_ccw;

    const float v_cmd = holding
                      ? 0.f
                      : std::clamp(std::min(v_profile * std::min(scale, brake), v_pivot_approach),
                                   0.f, p.max_adv);
    holding_ = holding;

    // ── A PIVOT MUST CARRY THE ARC LENGTH WITH IT, OR THE FEEDBACK UNDOES IT NEXT CYCLE ──────────
    // MEASURED 2026-08-13 (mppi_diag.csv, 4900 cycles / 246 s): the robot sat at s = 0.000 — the very
    // start of its path — with cmd_rot alternating -0.800 / +0.800 every 8 cycles (0.40 s) and its
    // heading oscillating inside a 0.15 rad band. It never turned at all. The base's identified lag is
    // 0.42 s on top of 0.20 s of transport, so every command was reversed before the wheels had
    // finished obeying it; meanwhile v sat at the saturation floor, 0.047 m/s, and the robot crept
    // 0.7 m FORWARD while facing 135 deg away from where its path went.
    // ★THE TWO AUTHORITIES AIMED AT DIFFERENT HEADINGS. The pivot drives theta -> psi(s+W); the Frenet
    // feedback drives theta -> psi(s), through -(2/L)*e_psi. Where the route turns hard those differ by
    // the whole turn — 135 deg over 0.40 m at that route's start — so the two command OPPOSITE
    // rotations, and the latch handed control between them across the one boundary where they disagree
    // most.
    // ★AND THE HYSTERESIS WAS NOT THERE. "Entry asks the route AND the pose, release asks the pose
    // alone" is an asymmetry only while turn_ahead is free to fall below 90 deg. It is NOT, in exactly
    // the case the pivot exists for: with turn_ahead >= 90 held, entry reduces to |align_err| >= 90,
    // which is the negation of the release test, and the pair chatters on the boundary.
    // ★So a pivot ENDS BY ADOPTING THE ARC LENGTH IT TURNED TO FACE. The robot turned through that
    // corner in place instead of driving it, so that arc is spent: psi(s) then agrees with the heading
    // the pivot achieved, e_psi is ~0 at release, and the feedback CONTINUES the pivot instead of
    // spending the next cycle undoing it. Bounded by the same window W the pivot aimed at, so it can
    // never skip more route than the turn it was looking at, and the projection is monotone-forward so
    // the adopted value cannot be walked back.
    if (released_now)
    {
        float best = s;
        float best_err = std::abs(wrap_pi(psi - theta_fwd));
        for (float ds = sp.spacing(); ds <= p.plain_W; ds += sp.spacing())
        {
            const float x = std::min(s + ds, sp.length());
            const float err = std::abs(wrap_pi(sp.heading_at(x) - theta_fwd));
            if (err < best_err) { best_err = err; best = x; }
            if (x >= sp.length()) break;
        }
        s_hint_ = best;
    }

    last_pos_ = pos;          // for the advance bound above, next cycle

    out.adv  = v_cmd;
    out.side = 0.f;
    out.rot  = -omega_blend;                      // FRAME conversion (2/2)
    out.cross_track_m = e_y;
    out.carrot_bearing_rad = e_psi;               // the heading error, not a bearing
    out.carrot_dist_m = v_cmd * p.plain_T_lag;    // the preview distance, for the overlay
    out.dist_to_goal = s_remaining;
    out.gate_speed_scale = p.max_adv > 1e-3f ? v_cmd / p.max_adv : 1.f;
    out.safety_guard_triggered = false;           // nothing in this mode can trigger; keep it honest
    return out;
}

// ═══ SELF-TEST ═══════════════════════════════════════════════════════════════════════════════════
// The arc-length projection is the one piece of state this tracker carries across cycles, and it was
// silently leaking backwards for the whole life of the class while every OTHER measurement said the
// tracker was healthy (cross-track p50 0.017 m). Nothing could have caught that from outside: a tracker
// that follows its curve perfectly and reports it has not moved looks like a good tracker until you ask
// it how far along it is. These cases ask.
namespace
{
struct StubWorld : PathWorld
{
    const RouteSpline *sp = nullptr;
    [[nodiscard]] const RouteSpline *route_spline() const override { return sp; }
    // PlainTracker reads neither — same assertion tools/tracker_sim.cpp makes, for the same reason:
    // if a future change makes one fire, fail loudly rather than measure a made-up robot.
    float body_extent(const Eigen::Vector2f &, float) const override { std::abort(); }
    float body_extent_max() const override { std::abort(); }
};
}   // namespace

bool PlainTracker::self_test()
{
    bool ok = true;
    auto check = [&](bool c, const char *m) { if (not c) { ok = false; std::printf("  FAIL: %s\n", m); } };

    // A gentle S, sampled at 5 cm — the spacing matters, because the backward slack the projection
    // seeds is half of it (route_spline.cpp:309) and that half-cell is the entire leak.
    std::vector<Eigen::Vector2f> poly;
    for (int i = 0; i <= 120; ++i)
    {
        const float x = 0.1f * static_cast<float>(i);
        poly.emplace_back(x, 0.6f * std::sin(0.35f * x));
    }
    RouteSpline sp;
    const auto free_everywhere = [](const Eigen::Vector2f &, float) { return true; };
    if (not sp.build(poly, 0.05f, free_everywhere, 0.40f))
    { std::printf("PlainTracker::self_test SKIPPED (spline build failed)\n"); return true; }

    StubWorld world;
    world.sp = &sp;
    TrackerParams p;
    const auto pose_at = [&](float s)
    {
        const Eigen::Vector2f r = sp.position_at(s);
        Eigen::Affine2f a = Eigen::Affine2f::Identity();
        // The tracker reads theta as the BODY angle and converts; heading_at is the forward tangent,
        // so the body angle is that minus 90 degrees. Getting this wrong only changes the commands,
        // not the projection, but a test that lies about the pose is worth nothing later.
        a.linear() = Eigen::Rotation2Df(sp.heading_at(s) - static_cast<float>(M_PI_2)).matrix();
        a.translation() = r;
        return a;
    };
    const auto drive = [&](PlainTracker &t, float s)
    {
        ControlOutput out; TrackerInput in;
        in.robot_pose = pose_at(s);
        t.compute(out, in, p);
        return out;
    };

    // (1) ★ORDINARY LOCALISER JITTER, WHICH IS THE LIVE SIGNATURE. The robot creeps forward while its
    // reported pose wanders a few centimetres ALONG the curve — every localiser does this, and PLAIN
    // drives at 20 Hz, so the wander per cycle is comparable to the advance per cycle.
    // Unclamped, the projection follows the wander DOWN as readily as up, but only the down direction
    // sticks: `project` searches [s_hint - spacing/2, s_hint + window], so a step back becomes the new
    // origin and the ground is never recovered except by real motion. That asymmetry — not any single
    // large event — is what pinned `track_s` at ~0 on 39% of a 282 s run while the cross-track error
    // stayed at p50 0.017 m.
    // ★What this case does NOT claim: an earlier version asserted the hint leaks while the robot stands
    // perfectly still. It does not — a stationary pose re-projects to the same sample, and that version
    // passed with and without the fix. The leak needs pose MOTION along the curve, so this drives.
    {
        PlainTracker t(world);
        drive(t, 3.0f);
        float s_true = 3.0f, prev = t.arc_length_hint().value_or(0.f);
        float worst_back = 0.f;
        const float s_start = prev;
        for (int i = 0; i < 300; ++i)
        {
            s_true += 0.01f;                                          // 0.2 m/s at 20 Hz
            // Deterministic wander, +-4 cm along the curve. No RNG: a test that fails one run in ten is
            // worse than no test.
            const float jitter = 0.04f * std::sin(static_cast<float>(i) * 2.399f);
            drive(t, std::max(0.f, s_true + jitter));
            const float now = t.arc_length_hint().value_or(0.f);
            worst_back = std::min(worst_back, now - prev);
            prev = now;
        }
        std::printf("  300 cycles driving with +-4 cm of along-track pose jitter: s %.3f -> %.3f m "
                    "(true advance 3.00 m), worst single-cycle change %.4f m\n",
                    s_start, prev, worst_back);
        check(worst_back >= -1e-4f,
              "★★the projection stepped BACKWARDS on ordinary pose jitter — every such step is kept, "
              "and that ratchet is what walks the arc length to 0 and pins it there");
        check(prev > s_start + 2.0f,
              "and it must still ADVANCE: a clamp that froze the hint would pass the test above and "
              "break the tracker completely");
    }

    // (2) MONOTONE UNDER A POSE THAT JUMPS BACKWARDS. A localiser correction, a replan, a bad frame —
    // the pose can legitimately move backwards along the curve. The robot's PROGRESS through this
    // traversal has not. RouteFollower::advance has always taken this position (route_follower.cpp:481);
    // this tracker now takes the same one, so the two projections cannot drift apart.
    {
        PlainTracker t(world);
        drive(t, 4.0f);
        float worst_back = 0.f, prev = t.arc_length_hint().value_or(0.f);
        for (int i = 0; i < 60; ++i)
        {
            // forward a little, then yanked back a lot, repeatedly
            const float s = 4.0f + 0.05f * static_cast<float>(i) - ((i % 10 == 0) ? 1.5f : 0.f);
            drive(t, std::max(0.f, s));
            const float now = t.arc_length_hint().value_or(0.f);
            worst_back = std::min(worst_back, now - prev);
            prev = now;
        }
        std::printf("  pose yanked 1.5 m backwards every 10th cycle: worst single-cycle change %.4f m\n",
                    worst_back);
        check(worst_back >= -1e-4f, "★arc length is progress through ONE traversal and may not decrease; "
                                    "a pose that jumps back is a pose event, not an un-driving");
    }

    // (3) ...BUT A RE-ACQUIRE MUST STILL BE FREE TO GO BACKWARDS, and this is why the clamp lives in the
    // tracker and not inside project(). reset() means "new traversal": a mission stop puts the robot at
    // the START, and a monotone rule that outlived the traversal would steer at the route's end point
    // from thirty metres away — the "it drives into a wall after a restart" failure the header records.
    {
        PlainTracker t(world);
        drive(t, 9.0f);
        check(t.arc_length_hint().value_or(0.f) > 8.f, "precondition: the tracker is well along the curve");
        t.reset();
        drive(t, 0.2f);
        const float after = t.arc_length_hint().value_or(-1.f);
        std::printf("  reset at s ~ 9 m, robot back at the start: re-acquired at %.3f m\n", after);
        check(after >= 0.f and after < 1.0f,
              "★a reset is a NEW traversal — monotonicity must not survive it, or a restart steers at "
              "the end of the route");
    }

    // (4) A NEAR-ZERO COMMAND MUST NOT COST THE PROJECTION. The tracker emits adv ~ 0 in two ordinary
    // states — holding the start alignment, and the stop taper at the end — and the session used to
    // read that as "nothing is happening" and call TrajectoryController::stop(), which reset() s
    // this hint away. The loop closed on itself. This asserts the tracker's own half: whatever it
    // commands, it keeps its place. Any loss of arc length is therefore an EXTERNAL reset(), which is
    // where to look.
    {
        PlainTracker t(world);
        // Start badly misaligned so the start-alignment hold fires and the command really is ~zero.
        ControlOutput out; TrackerInput in;
        Eigen::Affine2f a = pose_at(2.0f);
        a.linear() = Eigen::Rotation2Df(sp.heading_at(2.0f) + static_cast<float>(M_PI_2)).matrix();
        in.robot_pose = a;
        t.compute(out, in, p);
        const float s_before = t.arc_length_hint().value_or(-1.f);
        for (int i = 0; i < 40; ++i) { ControlOutput o2; t.compute(o2, in, p); out = o2; }
        const float s_after = t.arc_length_hint().value_or(-1.f);
        std::printf("  40 cycles of a near-zero command (adv %.4f m/s): s %.3f -> %.3f m\n",
                    out.adv, s_before, s_after);
        check(s_before >= 0.f and s_after >= 0.f,
              "★★the tracker must never discard its own projection: a zero command is a STATE it is "
              "holding, not the absence of a traversal");
        check(std::abs(s_after - s_before) < 0.2f, "and it must not drift while nothing moves");
    }

    std::printf("PlainTracker::self_test %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

}   // namespace rc
