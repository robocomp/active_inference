/*
 * tracker_sim — closed-loop offline bench for the ROUTE TRACKER, against the IDENTIFIED plant.
 *
 * WHY THIS EXISTS. On the robot, cross-track rms came out 0.086 and 0.153 m on two runs of the SAME
 * configuration. Every tracker effect we are chasing is smaller than that spread, so a lap cannot
 * adjudicate a gain choice and four laps can barely adjudicate a design. Here the plant is the one
 * IDENTIFIED from the robot (tau 0.213-0.236 s, pure delay 0.20 s, DC gain 0.89, r^2 0.94-0.95 on two
 * independent laps), so a conclusion about tracking dynamics transfers; and it is deterministic, so a
 * gain sweep costs seconds instead of a morning.
 *
 * WHAT IT DOES NOT MODEL, deliberately — and what that means for reading the output:
 *   - No obstacles, so no ESDF: the clearance speed bound and the safety gate are both absent. This
 *     bench answers TRACKING questions only. The jerk that the gate injects on the robot (79% of all
 *     command roughness) cannot appear here, so TV(v) from this bench is a floor, not a prediction.
 *   - Consequently no carrot CLIP. On the robot the clip binds on 75% of cycles and pulls the achieved
 *     lookahead to a p50 of 1.06 m against a 2.0 m parameter, which quadruples pure pursuit's loop gain
 *     (K = 2v/L_d^2) exactly in tight spots. The PD arm here therefore runs the UNCLIPPED ideal and
 *     flatters itself; treat it as an optimistic baseline, not as the robot.
 *   - Room-frame math convention (x = cos(theta), y = sin(theta)), NOT the production body frame
 *     (+Y forward, +X right, heading clockwise). This bench validates the control LAW; the frame
 *     plumbing is validated separately by the sign audit on CW/CCW circles in the tracker itself.
 *
 * Usage:
 *   tracker_sim [route_world.txt]              both arms, default gains
 *   tracker_sim [route_world.txt] --sweep      sweep L for the feedforward arm (the margin test)
 */

#include "../src/route_follower.h"
#include "../src/route_spline.h"
#include "../src/route_optimizer.h"
#include "../src/trackers/plain_tracker.h"
#include "../src/grid_planner.h"
#include "../../common/robot_footprint/robot_footprint.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deque>
#include <fstream>
#include <numbers>
#include <sstream>
#include <string>
#include <vector>

using Eigen::Vector2f;

namespace
{
constexpr float kPi = std::numbers::pi_v<float>;
// The REAL cbf_max_decel (controller_types.h). The arms below disable the end taper for tracking
// measurements; --stop-test restores it, because a stop test with no taper cannot ever stop.
constexpr float kADecReal = 1.0f;
bool  g_stop_test = false;   // run to an actual stop instead of breaking 0.15 m early
float wrap(float a) { return std::remainder(a, 2.f * kPi); }

// ── THE IDENTIFIED PLANT ─────────────────────────────────────────────────────────────────────────
// command -> [50 ms slew limiter, as ControllerMotionCommander::output_loop] -> [pure delay]
//         -> [first-order lag, DC gain] -> realised (v, omega) -> unicycle
// The slew limiter is part of the plant from the tracker's point of view: it exists downstream, the
// tracker does not know about it, and leaving it out would make every command look smoother than the
// wheels ever see.
struct Plant
{
    float tau = 0.22f;          // identified: 0.213-0.236 s
    float delay_s = 0.20f;      // identified
    float dc_gain = 0.89f;      // identified: the base delivers 89% of commanded rate
    float max_lin_accel = 1.5f; // ControllerMotionCommander
    float max_rot_accel = 4.0f;

    float x = 0.f, y = 0.f, th = 0.f;      // room frame, math convention
    float v_real = 0.f, w_real = 0.f;      // what the wheels are actually doing
    float v_slew = 0.f, w_slew = 0.f;      // after the output-tick limiter
    std::deque<std::pair<float, float>> pipe;   // delay line of slew-limited commands

    void step(float v_cmd, float w_cmd, float dt)
    {
        // Output-tick slew limit
        const float dv = std::clamp(v_cmd - v_slew, -max_lin_accel * dt, max_lin_accel * dt);
        const float dw = std::clamp(w_cmd - w_slew, -max_rot_accel * dt, max_rot_accel * dt);
        v_slew += dv; w_slew += dw;

        // Pure transport delay
        pipe.emplace_back(v_slew, w_slew);
        const std::size_t depth = static_cast<std::size_t>(std::lround(delay_s / dt));
        float v_in = 0.f, w_in = 0.f;
        if (pipe.size() > depth) { v_in = pipe.front().first; w_in = pipe.front().second; pipe.pop_front(); }

        // First-order lag with the identified DC gain
        const float a = std::exp(-dt / tau);
        v_real = a * v_real + (1.f - a) * dc_gain * v_in;
        w_real = a * w_real + (1.f - a) * dc_gain * w_in;

        // Unicycle
        x  += v_real * std::cos(th) * dt;
        y  += v_real * std::sin(th) * dt;
        th  = wrap(th + w_real * dt);
    }
};

// ── SPEED PROFILE (shared by both arms, so speed never confounds a steering comparison) ──────────
// The curvature limit the session already computes: comfort sqrt(a_lat/kappa) on the point value and
// kinematic omega_max/kappa on the AVERAGED value, back-propagated so the robot can shed the
// difference in time. Reuses RouteSpline::kappa_avg, i.e. the production estimator.
// ── CANDIDATE: A ROTATION BUDGET THAT SHRINKS AS THE TURN TIGHTENS ───────────────────────────────
// Measured on this route (--brake): from |kappa| = 0.3 upward the ROTATION limit h*w_max/kappa is the
// only thing setting the speed — comfort never binds again, and the turn-entry (rotational
// acceleration) limit never binds at all. So "slow down more in a sharp turn" has exactly one place to
// live, and a FIXED h cannot express it: h*w_max/kappa scales every turn by the same 1/kappa, so a
// body-radius hairpin is treated as just a scaled-up gentle bend.
//   h_eff(kappa) = h / (1 + q * (kappa * R_body)^2)
// The covariate is dimensionless because it is measured against the ROBOT: kappa*R_body = 1 is a turn
// whose radius equals the body's own circumscribed circle, which is where driving becomes pivoting.
// ★SHIPPED 2026-08-12 as ControllerRuntimeParams::sharp_turn_slowdown (config SharpTurnSlowdown,
// 0.25). This mirror must track ControllerSession::route_speed_limit and RouteSpline route_ideal —
// three copies of one speed law, which is the pre-existing duplication in this bench, not a new one.
float g_turn_q = 0.25f;
const float kBodyR = rc::RobotFootprint::shadow().circumscribed_radius();

float speed_limit(const rc::RouteSpline &sp, float s_now, float v_cap,
                  float a_lat, float a_dec, float w_max, float W, float headroom = 1.0f)
{
    float v = v_cap;
    // ★Mirrors ControllerSession::route_speed_limit, INCLUDING the 2026-08-05 fix: the noise floor may
    // not overrule the rotation budget. Without that, a cusp demands a turn rate the robot cannot
    // deliver, the command saturates, and the FF arm diverges — which is exactly what this bench caught.
    float v_rot_min = v_cap;
    const float horizon = v_cap * v_cap / (2.f * a_dec) + 1.0f;
    for (float ds = 0.f; ds <= horizon; ds += 0.10f)
    {
        const float k_pt = std::abs(sp.curvature_at(s_now + ds));
        const float k_av = std::abs(sp.kappa_avg(s_now + ds + 0.5f * W, W));
        if (k_pt < 1e-3f and k_av < 1e-3f) continue;
        const float v_lat = k_pt > 1e-3f ? std::sqrt(a_lat / k_pt) : v_cap;
        const float kr = k_av * kBodyR;
        const float h_eff = headroom / (1.f + std::max(0.f, g_turn_q) * kr * kr);
        const float v_rot = k_av > 1e-3f ? h_eff * w_max / k_av : v_cap;
        v_rot_min = std::min(v_rot_min, v_rot);
        v = std::min(v, std::sqrt(std::min(v_lat, v_rot) * std::min(v_lat, v_rot) + 2.f * a_dec * ds));
    }
    return std::clamp(v, std::min(0.15f, v_rot_min), v_cap);
}

// The clearance bound and its phantom test went with the "route" tracker on 2026-08-05: the PLAIN
// tracker queries no obstacle field, so there is nothing left here to exercise. The two arms below are
// now exactly the two live modes — ControlMode "pd" and ControlMode "plain".
//
// ── ARM A: "pd" — the carrot + Stanley + PD + EMA + Gaussian brake law, unclipped ────────────────
struct PdArm
{
    float lookahead = 2.0f, k_ct = 1.4f, ct_soft = 0.30f;
    float Kp = 1.2f, Kd = 0.3f, smoothing = 0.60f, gauss_k = 0.5f, cos_p = 1.0f;
    float sm_v = 0.f, sm_w = 0.f, prev_err = 0.f;
    bool  has_prev = false;

    void control(const rc::RouteSpline &sp, float s, float px, float py, float th,
                 float v_limit, float w_max, float &v_cmd, float &w_cmd)
    {
        // Carrot: walk forward `lookahead` metres along the curve from the projection.
        const Vector2f c = sp.position_at(std::min(s + lookahead, sp.length()));
        const float bearing = wrap(std::atan2(c.y() - py, c.x() - px) - th);
        const Vector2f r = sp.position_at(s);
        const float psi = sp.heading_at(s);
        const float e_y = -std::sin(psi) * (px - r.x()) + std::cos(psi) * (py - r.y());
        const float v_ref = std::max(ct_soft, std::abs(sm_v));
        const float cross_term = std::atan2(-k_ct * e_y, v_ref);   // steer back toward the route
        const float err = bearing + cross_term;
        const float d_err = has_prev ? err - prev_err : 0.f;
        prev_err = err; has_prev = true;
        const float raw_w = std::clamp(Kp * err + Kd * d_err, -w_max, w_max);
        const float raw_v = v_limit * std::pow(std::max(0.f, std::cos(bearing)), cos_p);
        sm_v = smoothing * sm_v + (1.f - smoothing) * raw_v;       // EMA inside the loop
        sm_w = smoothing * sm_w + (1.f - smoothing) * raw_w;
        const float ratio = sm_w / w_max;
        v_cmd = sm_v * std::exp(-gauss_k * ratio * ratio);         // Gaussian brake AFTER the smoother
        w_cmd = sm_w;
    }
};

// ── ARM B: "plain" — THE REAL TRACKER, LINKED, NOT COPIED ──────────────────────────────────────
// This used to be a hand-written replica of plain_tracker.cpp, kept in sync by a comment. It drifted
// TWICE: first to a reverted feedforward design, then (silently, after the comment was written) the
// robot moved its brake exponent to the CLAMPED rate while the copy kept the unbounded demand — so the
// k = 0.25 sweep optimised a law the robot was no longer running. PlainTracker::compute depends only on
// PathWorld::route_spline() — never on body_extent, never on a field — so a four-line stub world links
// the real thing and the class of bug is gone.
struct SimWorld : rc::PathWorld
{
    const rc::RouteSpline *sp = nullptr;
    [[nodiscard]] const rc::RouteSpline *route_spline() const override { return sp; }
    // PlainTracker never calls these — the compiler cannot know that, but the whole point of the
    // PathWorld/FieldWorld split is that this file can assert it. If a future tracker change makes one
    // of them fire, the bench aborts loudly rather than measuring a made-up robot.
    float body_extent(const Eigen::Vector2f &, float) const override { std::abort(); }
    float body_extent_max() const override { std::abort(); }
};

struct FfArm
{
    float L = 0.50f;            // THE free parameter (robot is at 0.50): error-decay length, metres of arc
    float T_lag = 0.42f;        // identified total lag
    float g_dc = 1.f / 0.89f;   // identified DC gain, inverted
    float W = 0.40f;            // route's own smoothing scale
    float brake_k = 0.25f;      // shipped

    // The sim drives s itself (it sweeps the route open-loop), so the tracker's own projection is
    // overridden each step to keep the two benches comparable.
    void control(const rc::RouteSpline &sp, float s, float px, float py, float th,
                 float v_limit, float w_max, float &v_cmd, float &w_cmd) const
    {
        // ★LOCAL, not a member. A member tracker holds a REFERENCE to a member world, so copying the
        // arm (the sweeps do: `FfArm a; a.L = x;` then pass it on) leaves the copy's tracker pointing at
        // the ORIGINAL's world — sp stays null, PlainTracker returns adv = 0, and every sweep row reads
        // a flawless 0.0 mm because the robot never moved. PlainTracker's only state is s_hint_, which
        // this bench seeds every call, so a fresh one per step is exactly equivalent and cannot alias.
        SimWorld world;
        world.sp = &sp;
        rc::PlainTracker tracker{world};
        rc::TrackerParams p;
        p.plain_L = L; p.plain_T_lag = T_lag; p.plain_g_dc = g_dc; p.plain_W = W;
        p.plain_brake_k = brake_k;
        p.plain_proj_window = 0.f;      // clamped to 0.05 inside; s is re-seeded below anyway
        p.max_adv = v_limit; p.max_rot = w_max;
        // ★THE TAPER IS DISABLED HERE ON PURPOSE — but only for the steering measurements. 1e6 makes
        // sqrt(2*a*s_rem) unbounded, so v_profile is just max_adv and the robot cruises through the end
        // of its own curve. That is correct for a TRACKING benchmark and catastrophic for --stop-test,
        // which asks whether the robot comes to REST: with the taper off it never can, and the result
        // reads as "PLAIN has no terminal behaviour" when the harness is what removed it.
        p.cbf_max_decel = g_stop_test ? kADecReal : 1e6f;

        rc::TrackerInput in;
        in.robot_pose = Eigen::Affine2f::Identity();
        in.robot_pose.translation() = Eigen::Vector2f{px, py};
        // PlainTracker takes the ROBOT frame convention (+Y forward); the sim carries maths heading.
        in.robot_pose.linear() = Eigen::Rotation2Df(th - static_cast<float>(M_PI_2)).toRotationMatrix();

        tracker.seed_arc_length(s);
        rc::ControlOutput out;
        tracker.compute(out, in, p);
        v_cmd = out.adv;
        w_cmd = -out.rot;               // back to CCW-positive, the sim's convention
    }
};

struct Result
{
    // First loss of the route, for locating a divergence instead of only sizing it.
    float s_lost = -1.f, kappa_lost = 0.f, v_lost = 0.f, t_lost = 0.f, s_end = 0.f;
    // --stop-test only: what the robot did at the END of the curve.
    float end_dist = -1.f, end_speed = -1.f, max_past = 0.f;
    bool  came_to_rest = false;
    double rot_effort = 0.0;   // integral |omega| dt — the quantity the L policy constrains

    int n = 0;
    double e_sq = 0, tv_v = 0, tv_w = 0, dist = 0, t_end = 0;
    float e_max = 0;
    std::vector<float> e, kappa;      // for the correlation and its lag
    // ── BRAKE DIAGNOSTIC (--brake) ───────────────────────────────────────────────────────────────
    // The plain tracker cuts the profile speed by min(scale, brake), and neither factor is observable
    // from outside. They are RECOVERABLE here: v_cmd/v_lim IS that product-of-one, and the brake alone
    // is exp(-k*(w_cmd/w_max)^2) because the exponent uses the DELIVERED (clamped) rate, which is
    // w_cmd. So "which factor is binding, and where" can be measured without touching the tracker.
    std::vector<float> d_kappa, d_vlim, d_vcmd, d_wcmd;
    // ── UNSEEN SWEEP (--pivot) ───────────────────────────────────────────────────────────────────
    // The quantity the collision complaint is actually about: how far the BODY translates while the
    // route is turning hard. The lidar is blind within ~8 cm of the hull, so every metre travelled
    // sideways-ish through a corner is a metre of body sweeping through space nothing can see.
    // Bucketed by how much the route turns over its smoothing window, which is what "sharp" means.
    double trans_45 = 0, trans_90 = 0, trans_135 = 0;   // metres translated where turn_ahead exceeds
    double time_45 = 0, time_90 = 0, time_135 = 0;      // seconds spent there
    float  vmax_90 = 0;                                 // fastest the robot ever moved past 90 deg
};

double corr(const std::vector<float> &a, const std::vector<float> &b, int lag)
{
    // corr(a[i], b[i - lag]); lag < 0 compares a with the FUTURE of b.
    double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0; int n = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const long j = static_cast<long>(i) - lag;
        if (j < 0 or j >= static_cast<long>(b.size())) continue;
        sa += a[i]; sb += b[j]; saa += double(a[i]) * a[i]; sbb += double(b[j]) * b[j];
        sab += double(a[i]) * b[j]; ++n;
    }
    if (n < 30) return 0.0;
    const double na = saa - sa * sa / n, nb = sbb - sb * sb / n;
    return (na > 1e-12 and nb > 1e-12) ? (sab - sa * sb / n) / std::sqrt(na * nb) : 0.0;
}

float g_plant_tau = 0.22f, g_plant_delay = 0.20f, g_plant_gain = 0.89f;   // the ROBOT, for sim2real tests
float g_headroom = 1.0f;
bool  g_heading_gate = false;   // FAILED experiment, kept as a negative result; see the A/B below
bool  g_proj_robust = false; // --proj-robust: progress-following projection (see project_robust)
bool  g_optimise = false;   // --optimise: build the route through the OPTIMISER, not fit-only
float g_w_jerk = 0.f;       // w_jerk=<v>: the dkappa/ds prior, and it implies --optimise
bool  g_trace = false;      // --trace: dump the terminal approach cycle by cycle
float g_proj_window = 2.0f;    // forward arc-length search window for the projection   // fraction of the omega budget the FEEDFORWARD may claim; the rest is
                           // reserved for feedback authority. 1.0 = the naive limit.

// ── PROGRESS-FOLLOWING PROJECTION (--proj-robust) ────────────────────────────────────────────────
// RouteSpline::project searches [s_hint, s_hint + window] for the nearest sample, with window fixed at
// 2 m. Two consequences, and the second is the one that matters at the end of a curve:
//   • In a CLOSED TURN the chord is short, so a point 1-2 m further along can be nearer in straight-line
//     distance than the true corresponding point. The projection then JUMPS THE CHORD, skipping arc it
//     never drove.
//   • A window that wide is also free to sit still: nothing ties the estimate to how far the robot has
//     actually moved, so `s` can stall while the robot keeps going — which is what starves the stop
//     taper (s_remaining stops shrinking, sqrt(2*a*s_remaining) never falls, the robot drives past).
// So: bound the forward search by RECENT PROGRESS instead of a constant. The robot cannot have advanced
// much more along the curve than it advanced through space, so a few cycles of measured advance is the
// honest bound, and a candidate beyond it is the chord rather than travel. Keeping the last 5 keeps one
// noisy cycle from either freezing or unlocking the window.
struct Projector
{
    std::deque<float> hist;          // last N accepted arc lengths
    float s = 0.f;
    bool  init = false;
    static constexpr std::size_t kN = 5;

    float update(const rc::RouteSpline &sp, const Vector2f &p, float fallback_window)
    {
        if (not init)
        {
            s = sp.project(p, 0.f, sp.length());   // re-acquire: whole curve, once
            init = true;
            hist.assign(kN, s);
            return s;
        }
        // Advance over the retained history — the arc the robot demonstrably covered in those cycles.
        const float advanced = std::max(0.f, s - hist.front());
        // Allow appreciably more than that (acceleration is real), but not the metres a chord jump needs.
        // Floored so a stopped robot can still re-acquire a little, and capped by the caller's window so
        // this can never search WIDER than the code it replaces.
        const float window = std::min(fallback_window, std::max(0.15f, 3.f * advanced));
        s = sp.project(p, s, window);
        hist.push_back(s);
        while (hist.size() > kN) hist.pop_front();
        return s;
    }
};


template <typename Arm>

Result run(const rc::RouteFollower &route, Arm arm, float v_cap, float w_max,
           float a_lat, float a_dec, float W, bool verbose)
{
    const rc::RouteSpline &sp = route.spline();
    Result R;
    Plant P;
    P.tau = g_plant_tau; P.delay_s = g_plant_delay; P.dc_gain = g_plant_gain;
    const Vector2f p0 = sp.position_at(0.f);
    P.x = p0.x(); P.y = p0.y(); P.th = sp.heading_at(0.f);

    constexpr float kCtrlDt = 0.05f;    // 20 Hz, the data-driven control rate
    constexpr float kSubDt  = 0.005f;   // plant integration
    float s_hint = 0.f, last_v = 0.f, last_w = 0.f;
    float v_cmd_prev = 0.f;
    Projector projector;   // --stop-test: "at rest" means the PLANT has stopped AND nothing new was asked
    float lx = P.x, ly = P.y; bool have_last = false;
    bool  first = true;

    for (float t = 0.f; t < 400.f; t += kCtrlDt)
    {
        // ★Rate limit REVERTED with the tracker (see plain_tracker.cpp): it forbade ACQUISITION, not
        // just jumping. This bench never exercised that, because it always starts exactly on the route.
        // ── PROJECTION: reject branches the robot is pointing AWAY from ──────────────────────────
        // At a hairpin the outbound and inbound legs are centimetres apart in SPACE, so a nearest-point
        // search cannot tell them apart and snaps to the returning leg — the tip is never projected onto
        // and the robot turns around where the two legs touch. Distance cannot separate them; HEADING
        // can: the two branches are anti-parallel, and a robot cannot be following a segment that points
        // backwards relative to it. Accept a sample only where cos(psi(s) - theta) > 0.
        (void)lx; (void)ly; (void)have_last;
        if (g_heading_gate)
        {
            const Vector2f rp{P.x, P.y};
            const float step = std::max(0.01f, sp.spacing());
            float bs = s_hint, bd = 1e9f; bool any = false;
            for (float ds = -0.5f * step; ds <= 2.0f; ds += step)
            {
                const float t = s_hint + ds;
                if (t < 0.f or t > sp.length()) continue;
                if (std::cos(wrap(sp.heading_at(t) - P.th)) <= 0.f) continue;   // wrong branch
                const float d = (sp.position_at(t) - rp).norm();
                if (d < bd) { bd = d; bs = t; any = true; }
            }
            s_hint = any ? bs : sp.project(rp, s_hint, 2.0f);   // all rejected -> fall back
        }
        else if (g_proj_robust) s_hint = projector.update(sp, {P.x, P.y}, g_proj_window);
        else s_hint = sp.project({P.x, P.y}, s_hint, g_proj_window);
        // ── WHERE THE RUN ENDS ───────────────────────────────────────────────────────────────────
        // ★THE DEFAULT STOPS 0.15 m EARLY, AND THAT IS A BLIND SPOT. This bench was built for MISSION
        // routes, where RouteFollower::finished() accepts arrival within a metre, so "did it come to
        // rest ON the endpoint" was never a question worth asking and the loop breaks as soon as the
        // route is essentially covered. The consequence is that the terminal approach — the only regime
        // that matters for a point target, e.g. an affordance standpoint — has never been simulated at
        // all. Both a 39 m tour and a 2.5 m hop reported "ended at s = length - 0.16 m", which is this
        // line, not the tracker.
        // --stop-test runs the same closed loop to an actual STOP and records what it did there.
        {
            const Vector2f endp = sp.position_at(sp.length());
            const float d_end = (Vector2f(P.x, P.y) - endp).norm();
            const float psi_e = sp.heading_at(sp.length());
            // Signed progress along the curve's final heading: positive means BEYOND the endpoint.
            const float past = std::cos(psi_e) * (P.x - endp.x()) + std::sin(psi_e) * (P.y - endp.y());
            if (g_stop_test)
            {
                R.max_past = std::max(R.max_past, past);
                // ★TRACE: is the projection actually stuck, or is the command ignoring it? Printed for
                // the approach and the first metres beyond, so the two hypotheses are distinguishable.
                if (g_trace and past > -1.0f and R.n % 4 == 0)
                    std::printf("      t=%5.2f  s=%7.3f / %7.3f  s_rem=%6.3f  past=%6.3f  v_cmd=%5.3f  v_real=%5.3f\n",
                                t, s_hint, sp.length(), sp.length() - s_hint, past, v_cmd_prev, P.v_real);
                const bool at_rest = std::abs(P.v_real) < 0.02f and std::abs(v_cmd_prev) < 0.02f;
                if (at_rest and t > 1.0)
                { R.came_to_rest = true; R.end_dist = d_end; R.end_speed = std::abs(P.v_real); R.t_end = t; break; }
                if (past > 8.0f or t > 120.0)   // far enough past that no stop is coming
                { R.end_dist = d_end; R.end_speed = std::abs(P.v_real); R.t_end = t; break; }
            }
            else if (s_hint >= sp.length() - 0.15f) { R.t_end = t; break; }
        }

        const float v_lim = speed_limit(sp, s_hint, v_cap, a_lat, a_dec, w_max, W, g_headroom);
        float v_cmd = 0.f, w_cmd = 0.f;
        arm.control(sp, s_hint, P.x, P.y, P.th, v_lim, w_max, v_cmd, w_cmd);

        if (not first) { R.tv_v += std::abs(v_cmd - last_v); R.tv_w += std::abs(w_cmd - last_w); }
        R.rot_effort += std::abs(P.w_real) * kCtrlDt;   // REALISED turn, not commanded
        last_v = v_cmd; last_w = w_cmd; first = false;
        v_cmd_prev = v_cmd;

        const Vector2f r = sp.position_at(s_hint);
        const float psi = sp.heading_at(s_hint);
        const float e = -std::sin(psi) * (P.x - r.x()) + std::cos(psi) * (P.y - r.y());
        R.e_sq += double(e) * e; R.e_max = std::max(R.e_max, std::abs(e)); ++R.n;
        R.e.push_back(e);
        R.kappa.push_back(sp.kappa_avg(s_hint, W));
        R.d_kappa.push_back(std::abs(sp.kappa_avg(s_hint, W)));
        R.d_vlim.push_back(v_lim); R.d_vcmd.push_back(v_cmd); R.d_wcmd.push_back(w_cmd);
        // WHERE does it come apart? An rms figure cannot distinguish "slightly loose everywhere" from
        // "fine until s=X, then gone", and those want completely different fixes.
        if (R.s_lost < 0.f and std::abs(e) > 0.5f)
        {
            R.s_lost = s_hint;
            R.kappa_lost = sp.kappa_avg(s_hint, W);
            R.v_lost = v_lim;
            R.t_lost = t;
        }
        R.s_end = s_hint;

        // How sharply the route turns over its own smoothing window, at the robot's projection —
        // exactly the `turn_ahead` the tracker's translation gate uses.
        const float turn_ahead = std::abs(wrap(sp.heading_at(std::min(s_hint + W, sp.length()))
                                               - sp.heading_at(s_hint)));

        const float x0 = P.x, y0 = P.y;
        for (float u = 0.f; u < kCtrlDt - 1e-6f; u += kSubDt) P.step(v_cmd, w_cmd, kSubDt);
        const float step_m = std::hypot(P.x - x0, P.y - y0);
        constexpr float kD45 = 0.785f, kD90 = 1.571f, kD135 = 2.356f;
        if (turn_ahead > kD45)  { R.trans_45  += step_m; R.time_45  += kCtrlDt; }
        if (turn_ahead > kD90)  { R.trans_90  += step_m; R.time_90  += kCtrlDt;
                                  R.vmax_90 = std::max(R.vmax_90, step_m / kCtrlDt); }
        if (turn_ahead > kD135) { R.trans_135 += step_m; R.time_135 += kCtrlDt; }
        R.dist += step_m;
        R.t_end = t;
    }
    if (verbose)
        std::printf("    (ran %d cycles, %.1f m, %.1f s)\n", R.n, R.dist, R.t_end);
    return R;
}

// J, exactly as MissionRunner computes it (controller_mission.cpp:888-891):
//   smooth_lin = (lin_accel_effort/v_max)/distance,  smooth_rot = (rot_accel_effort/w_max)/distance,
//   dev_norm   = cross_track_rms/planner_cell_size,  J = the sum. Per metre, dimensionless, lower better.
// ⚠TV(v)/TV(w) are the TOTAL VARIATION OF THE COMMAND — the closest this bench has to the agent's accel
// efforts, which integrate |a| over the profile. Same normalisation, so J here RANKS configurations; it
// is NOT numerically comparable to the robot's J.
constexpr float kCellSizeM = 0.06f;   // Planner.CellSize — the length scale dev_norm is measured in
double mission_J(const Result &R, float v_max, float w_max)
{
    if (R.dist < 0.1 or R.n == 0) return 1e9;
    const double rms = std::sqrt(R.e_sq / R.n);
    return (R.tv_v / v_max) / R.dist + (R.tv_w / w_max) / R.dist + rms / kCellSizeM;
}

void report(const char *tag, const Result &R)
{
    const double rms = R.n ? std::sqrt(R.e_sq / R.n) : 0.0;
    // Peak of corr(e(t), kappa(t - lag)) over +-1 s. Negative lag = the error ANTICIPATES curvature.
    int best_lag = 0; double best = 0;
    for (int lag = -20; lag <= 20; ++lag)
        if (const double c = corr(R.e, R.kappa, lag); std::abs(c) > std::abs(best)) { best = c; best_lag = lag; }
    std::printf("  %-14s rms %6.1f mm | max %6.1f mm | corr(e,k) %+.3f | peak %+.3f @ %+4d ms"
                " | TV(v)/m %.3f | TV(w)/m %.3f | %.0f s\n",
                tag, rms * 1000, R.e_max * 1000, corr(R.e, R.kappa, 0), best, best_lag * 50,
                R.dist > 0.1 ? R.tv_v / R.dist : 0.0, R.dist > 0.1 ? R.tv_w / R.dist : 0.0, R.t_end);
    if (R.s_lost >= 0.f)
        std::printf("      ↳ LOST THE ROUTE at s=%.1f m (t=%.0f s): |e_y| passed 0.5 m where "
                    "kappa_avg=%.2f 1/m and the speed profile allowed %.3f m/s; ended at s=%.1f\n",
                    R.s_lost, R.t_lost, R.kappa_lost, R.v_lost, R.s_end);
}

// ── WHICH FACTOR ACTUALLY REFRAINS THE ADVANCE, AND WHERE? ───────────────────────────────────────
// v_cmd = v_profile * min(scale, brake), and the QUESTION this answers is whether the exponential
// brake is doing any work in the turns it was written for. Bucketed by |kappa_avg| because "sharp
// turn" is a statement about curvature, not about a time average over a lap.
//   ratio  = v_cmd / v_profile               — the whole reduction the tracker applies
//   brake  = exp(-k * (w_cmd/w_max)^2)       — recoverable exactly (the exponent uses the CLAMPED rate)
//   binds  = share of cycles where ratio < brake, i.e. the SATURATION ratio is the binding one and the
//            brake contributes nothing at all
void brake_diag(const char *tag, const Result &R, float brake_k, float w_max)
{
    struct Bucket { float lo, hi; int n = 0; double vlim = 0, vcmd = 0, brake = 0, ratio = 0; int binds = 0; };
    Bucket b[] = {{0.f, 0.3f}, {0.3f, 1.0f}, {1.0f, 2.0f}, {2.0f, 4.0f}, {4.0f, 1e9f}};
    for (std::size_t i = 0; i < R.d_kappa.size(); ++i)
    {
        const float k = R.d_kappa[i], vl = R.d_vlim[i], vc = R.d_vcmd[i];
        if (vl < 1e-4f) continue;
        const float r_om = std::abs(R.d_wcmd[i]) / std::max(0.05f, w_max);
        const float br = std::exp(-brake_k * r_om * r_om);
        const float ratio = vc / vl;
        for (auto &q : b)
            if (k >= q.lo and k < q.hi)
            { ++q.n; q.vlim += vl; q.vcmd += vc; q.brake += br; q.ratio += ratio; q.binds += (ratio < br - 1e-3f); break; }
    }
    std::printf("\n  %s — where does the advance actually get refrained? (brake_k = %.2f)\n", tag, brake_k);
    std::printf("    |kappa| 1/m       n   v_profile    v_cmd    brake    ratio   scale binds\n");
    for (const auto &q : b)
    {
        if (q.n == 0) continue;
        char range[24];
        if (q.hi > 1e8f) std::snprintf(range, sizeof range, ">= %.1f", q.lo);
        else             std::snprintf(range, sizeof range, "%.1f - %.1f", q.lo, q.hi);
        std::printf("    %-12s %6d    %8.3f %8.3f %8.3f %8.3f   %5.1f%%\n", range, q.n,
                    q.vlim / q.n, q.vcmd / q.n, q.brake / q.n, q.ratio / q.n, 100.0 * q.binds / q.n);
    }
}

// How much body sweep happens inside a hard turn — the collision question, not the tracking one.
void pivot_report(const char *tag, const Result &R)
{
    std::printf("  %-22s >45deg: %5.2f m / %4.1f s | >90deg: %5.2f m / %4.1f s (v_max %.3f) | "
                ">135deg: %5.2f m / %4.1f s | rms %5.1f mm | %.0f s\n",
                tag, R.trans_45, R.time_45, R.trans_90, R.time_90, R.vmax_90,
                R.trans_135, R.time_135, R.n ? std::sqrt(R.e_sq / R.n) * 1000 : 0.0, R.t_end);
}

struct World
{
    rc::GridPlanner planner;
    std::vector<Vector2f> wp_safe;
    Vector2f start{0.f, 0.f};
    int laps = 1;
    float spacing = 0.05f, smoothing = 0.40f, v_max = 0.7f, a_lat = 1.0f, standoff = 0.6f;
    rc::RouteOptimizerConfig opt;   // read from the world file; only USED when --optimise is given
};

bool load_world(const std::string &path, World &w)
{
    std::ifstream f(path);
    if (not f.is_open()) { std::printf("cannot open %s\n", path.c_str()); return false; }
    std::string line;
    while (std::getline(f, line))
    {
        if (line.empty() or line[0] == '#') continue;
        std::istringstream ls(line);
        std::string key; ls >> key;
        if (key == "grid") break;
        else if (key == "laps")    ls >> w.laps;
        else if (key == "start")   ls >> w.start.x() >> w.start.y();
        else if (key == "wp_safe") { Vector2f p; ls >> p.x() >> p.y(); w.wp_safe.push_back(p); }
        else if (key == "fit")     ls >> w.spacing >> w.smoothing >> w.v_max >> w.a_lat >> w.standoff;
        else if (key == "opt")
            // Same field order as route_bench, so one snapshot replays identically in both benches.
            ls >> w.opt.d_target >> w.opt.rho >> w.opt.sigma_a >> w.opt.clearance_floor
               >> w.opt.w_kappa >> w.opt.w_clear >> w.opt.w_gauge >> w.opt.clear_peak
               >> w.opt.anchor_huber >> w.opt.iterations >> w.opt.kappa_peak >> w.opt.safety_bias;
    }
    f.clear(); f.seekg(0);
    if (not w.planner.read_grid(f)) { std::printf("no readable grid in %s\n", path.c_str()); return false; }
    return w.wp_safe.size() >= 2;
}
}   // namespace

int main(int argc, char **argv)
{
    std::string path = "route_world.txt";
    bool sweep = false, brake_mode = false, pivot_mode = false;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "--sweep") sweep = true; else if (a == "--stop-test") g_stop_test = true;
        else if (a == "--proj-robust") g_proj_robust = true;
        else if (a == "--trace") g_trace = true;
        else if (a == "--optimise" or a == "--optimize") g_optimise = true;
        else if (a == "--brake") brake_mode = true;
        else if (a == "--pivot") pivot_mode = true;
        else if (a.rfind("w_jerk=", 0) == 0) { g_optimise = true; g_w_jerk = std::stof(a.substr(7)); } else path = a;
    }

    World w;
    if (not load_world(path, w)) return 1;

    rc::RouteFollower route;
    // Capture the RAW A* polyline as it comes back from the planner, before the spline fit, the
    // feasibility pass or the optimiser touch it. That is stage 1 of the pipeline, and the only way to
    // tell WHICH stage loses the clearance the waypoints plainly have.
    std::vector<Vector2f> raw_poly;
    auto plan = [&w, &raw_poly](const Vector2f &a, const Vector2f &b)
    {
        auto seg = w.planner.plan(a, b);           // std::optional<std::vector<Vector2f>>
        if (seg) for (const auto &q : *seg) raw_poly.push_back(q);
        return seg;
    };
    auto free_at = [&w](const Vector2f &p, float h) { return w.planner.pose_free(p, h); };
    // ── OPTIONALLY DRIVE THE OPTIMISED ROUTE ─────────────────────────────────────────────────────
    // Default OFF: this bench measures the TRACKER, and every published baseline here (tour 41.2 mm rms
    // / TV(w)/m 2.068 / 80 s) was taken on the fit-only route. Turning the optimiser on changes the
    // STIMULUS, so a run under it measures the route, not the control law — which is exactly what you
    // want when asking whether a geometry prior buys smoother COMMANDS, and useless for anything else.
    if (g_optimise)
    {
        rc::RouteOptimizerConfig o = w.opt;
        o.enabled = true;
        o.w_jerk = g_w_jerk;
        o.distance = [&w](const Vector2f &p) { return w.planner.distance_at(p); };
        o.distance_gradient = [&w](const Vector2f &p) { return w.planner.distance_gradient_at(p); };
        o.anchors = w.wp_safe;
        route.set_optimizer(o);
        std::printf("  [optimiser ON] w_kappa %.2f w_clear %.2f w_jerk %.2f\n",
                    o.w_kappa, o.w_clear, o.w_jerk);
    }
    if (not route.build(w.start, w.wp_safe, w.laps, plan, free_at, w.spacing, w.smoothing))
    { std::printf("route build failed\n"); return 1; }

    constexpr float kWmax = 0.8f, kADec = 1.0f, kW = 0.40f;
    std::printf("tracker_sim: %s — route %.1f m, %zu waypoints, v_max %.2f, plant tau 0.22 delay 0.20 gain 0.89\n\n",
                path.c_str(), route.length(), w.wp_safe.size(), w.v_max);

    if (pivot_mode)
    {
        // ── HOW MUCH BODY SWEEP HAPPENS INSIDE A HARD TURN ───────────────────────────────────────
        // The robot clips furniture in sharp turns, and the lidar cannot see anything within ~8 cm of
        // its hull, so the number that matters is METRES TRANSLATED WHILE TURNING — not rms, not lap
        // time. Reported at three sharpness bands so the effect is visible where it is meant to bite
        // and absent where it is not. Compare against the same binary with the gate removed.
        g_headroom = 0.70f;
        std::printf("\n  body sweep inside a turn (route %.1f m, L=0.50, q=%.2f)\n",
                    route.length(), g_turn_q);
        for (const float L : {0.50f})
        {
            FfArm arm; arm.L = L; arm.brake_k = 0.25f;
            pivot_report("shipped", run(route, arm, w.v_max, kWmax, w.a_lat, kADec, kW, false));
        }
        for (const float q : {0.00f, 0.25f})
        {
            g_turn_q = q;
            FfArm arm; arm.L = 0.50f; arm.brake_k = 0.25f;
            char tag[40]; std::snprintf(tag, sizeof tag, "q=%.2f", q);
            pivot_report(tag, run(route, arm, w.v_max, kWmax, w.a_lat, kADec, kW, false));
        }
        g_turn_q = 0.25f;
        return 0;
    }

    if (brake_mode)
    {
        // ── HOW HARD IS THE ADVANCE REFRAINED IN A SHARP TURN, AND BY WHAT? ──────────────────────
        // The k=0.25 optimum on record was measured against the UNBOUNDED exponent; the shipped law
        // clamps it, which caps the brake at exp(-k) and hands the sharp-turn regime to the saturation
        // ratio instead. So the sweep has to be re-run before any claim about "more braking" is safe.
        g_headroom = 0.70f;
        std::printf("\n  brake_k sweep on the SHIPPED law (L=0.50, headroom 0.70)\n");
        for (const float k : {0.00f, 0.25f, 0.50f, 1.00f, 2.00f, 4.00f})
        {
            FfArm arm; arm.L = 0.50f; arm.brake_k = k;
            char tag[32]; std::snprintf(tag, sizeof tag, "k=%.2f", k);
            const Result R = run(route, arm, w.v_max, kWmax, w.a_lat, kADec, kW, false);
            report(tag, R);
        }
        {
            FfArm arm; arm.L = 0.50f; arm.brake_k = 0.25f;
            brake_diag("SHIPPED k=0.25", run(route, arm, w.v_max, kWmax, w.a_lat, kADec, kW, false), 0.25f, kWmax);
        }
        // ── THE CANDIDATE: CURVATURE-DEPENDENT ROTATION BUDGET ───────────────────────────────────
        // h_eff = h / (1 + q*(kappa*R_body)^2). q = 0 must reproduce the shipped row exactly, which is
        // what makes the rest of the column readable as an effect rather than as a rebuild difference.
        std::printf("\n  turn-tightness dial q — h_eff = h/(1 + q*(kappa*R_body)^2), R_body = %.3f m\n", kBodyR);
        std::printf("    (h_eff/h at a 1 m radius, at the body radius 0.325 m, and at the route's "
                    "tightest 0.13 m)\n");
        for (const float q : {0.00f, 0.10f, 0.25f, 0.50f, 1.00f, 2.00f})   // 0.25 is shipped
        {
            g_turn_q = q;
            FfArm arm; arm.L = 0.50f; arm.brake_k = 0.25f;
            const Result R = run(route, arm, w.v_max, kWmax, w.a_lat, kADec, kW, false);
            char tag[40];
            const auto f = [q](float kap) { const float kr = kap * kBodyR; return 1.f / (1.f + q * kr * kr); };
            std::snprintf(tag, sizeof tag, "q=%.2f", q);
            report(tag, R);
            std::printf("      ↳ h_eff/h: %.2f (r=1 m)  %.2f (r=0.325 m)  %.2f (r=0.13 m) | "
                        "rot/m %.3f | driven %.1f m\n",
                        f(1.f), f(1.f / kBodyR), f(7.65f),
                        R.dist > 0.1 ? R.rot_effort / R.dist : 0.0, R.dist);
        }
        g_turn_q = 0.25f;   // restore the shipped value for the probes below
        // ── WHAT THE THREE CANDIDATE LIMITS WOULD ALLOW, ALONG THIS ROUTE ────────────────────────
        // Two are already in route_speed_limit (comfort sqrt(a_lat/kappa), kinematic h*w_max/kappa_avg).
        // The third is the one that is MISSING: omega = v*kappa means the robot must SPIN UP into a
        // bend at d(omega)/dt = v^2*dkappa/ds, and the base's slew limiter caps that at max_rot_accel.
        // Printing all three side by side says whether a turn-entry limit could bind at all before any
        // of it is written into the controller.
        {
            const rc::RouteSpline &sp = route.spline();
            std::printf("\n  what each limit would allow (a_lat %.2f, h*w_max %.2f, alpha 4.0 rad/s^2)\n",
                        w.a_lat, 0.70f * kWmax);
            std::printf("    |kappa| 1/m       n    |dk/ds|   v_lat    v_rot   v_entry   binding\n");
            struct B { float lo, hi; int n = 0; double kp = 0, vl = 0, vr = 0, ve = 0; };
            B b[] = {{0.f, 0.3f}, {0.3f, 1.0f}, {1.0f, 2.0f}, {2.0f, 4.0f}, {4.0f, 1e9f}};
            for (float s = 0.f; s + kW <= sp.length(); s += 0.05f)
            {
                const float k0 = std::abs(sp.kappa_avg(s, kW));
                // ONE difference of an ALREADY-AVERAGED curvature, not a third difference of samples —
                // the same reasoning route_speed_limit gives for preferring kappa_avg to curvature_at.
                const float kp = std::abs(sp.kappa_avg(s + kW, kW) - sp.kappa_avg(s, kW)) / kW;
                const float vl = k0 > 1e-3f ? std::sqrt(w.a_lat / k0) : w.v_max;
                const float vr = k0 > 1e-3f ? 0.70f * kWmax / k0 : w.v_max;
                const float ve = kp > 1e-3f ? std::sqrt(4.0f / kp) : w.v_max;
                for (auto &q : b)
                    if (k0 >= q.lo and k0 < q.hi) { ++q.n; q.kp += kp; q.vl += vl; q.vr += vr; q.ve += ve; break; }
            }
            for (const auto &q : b)
            {
                if (q.n == 0) continue;
                char range[24];
                if (q.hi > 1e8f) std::snprintf(range, sizeof range, ">= %.1f", q.lo);
                else             std::snprintf(range, sizeof range, "%.1f - %.1f", q.lo, q.hi);
                const double vl = q.vl / q.n, vr = q.vr / q.n, ve = q.ve / q.n;
                const char *bind = (vr <= vl and vr <= ve) ? "rot" : (vl <= ve ? "lat" : "ENTRY");
                std::printf("    %-12s %6d %9.2f %8.3f %8.3f %9.3f   %s\n",
                            range, q.n, q.kp / q.n, vl, vr, ve, bind);
            }
        }
        return 0;
    }

    if (sweep)
    {
        // THE MARGIN TEST. Theory says the closed-loop time constant must stay >= 3x the total lag,
        // i.e. L >= 3*T_lag*v_max ~ 0.88 m, and that below roughly 0.3-0.45 m the loop should ring.
        // If oscillation appears where predicted, the margin calculation is validated; if it appears
        // much earlier or not at all, the model is wrong and so is the default.
        std::printf("  headroom sweep at L=0.60 (fraction of omega_max the feedforward may claim)\n");
        for (const float hr : {1.00f, 0.85f, 0.70f, 0.55f, 0.40f})
        {
            g_headroom = hr;
            FfArm arm; arm.L = 0.60f;
            char tag[32]; std::snprintf(tag, sizeof tag, "headroom %.2f", hr);
            report(tag, run(route, arm, w.v_max, kWmax, w.a_lat, kADec, kW, false));
        }
        g_headroom = 0.70f;
        // ── SIM2REAL / ADAPTATION TEST ───────────────────────────────────────────────────────────
        // The design puts ALL plant knowledge into two identified scalars (T_lag, g_dc). If that is
        // true, then on a DIFFERENT robot the tracker degrades in a way that re-identifying those two
        // numbers — which the existing offline fit already produces from logged (cmd_rot, meas_rot) —
        // fully recovers. If it does not recover, the plant knowledge is smeared elsewhere and
        // per-mission adaptation cannot work.
        std::printf("\n  sim2real: a DIFFERENT robot (tau 0.35, delay 0.30, gain 0.75) at L=0.60, headroom 0.70\n");
        {
            struct Case { const char *tag; float tau, delay, gain, T_lag, g_dc; };
            const Case cases[] = {
                {"matched (baseline)", 0.22f, 0.20f, 0.89f, 0.42f, 1.f / 0.89f},
                {"new robot, STALE  ", 0.35f, 0.30f, 0.75f, 0.42f, 1.f / 0.89f},
                {"new robot, ADAPTED", 0.35f, 0.30f, 0.75f, 0.65f, 1.f / 0.75f},
            };
            for (const auto &c : cases)
            {
                g_plant_tau = c.tau; g_plant_delay = c.delay; g_plant_gain = c.gain;
                FfArm arm; arm.L = 0.60f; arm.T_lag = c.T_lag; arm.g_dc = c.g_dc;
                report(c.tag, run(route, arm, w.v_max, kWmax, w.a_lat, kADec, kW, false));
            }
            // ★ IS L PART OF THE ADAPTATION? The margin rule that set L in the first place is
            // L >= 3*T_lag*v_max, so a robot with MORE lag needs a LONGER length scale — L is not an
            // independent gain, it is a function of the identified lag. If that is right, sweeping L on
            // the slow robot should peak near 3*0.65*0.7 = 1.37 m and recover the rest of the gap; if
            // the peak stays near the fast robot's 0.60, then L is independent and adaptation is two
            // numbers, not three.
            std::printf("\n  does L follow T_lag? slow robot (tau 0.35, delay 0.30, gain 0.75), T_lag/g_dc adapted\n");
            g_plant_tau = 0.35f; g_plant_delay = 0.30f; g_plant_gain = 0.75f;
            for (const float L : {0.60f, 0.90f, 1.20f, 1.37f, 1.60f, 2.00f})
            {
                FfArm arm; arm.L = L; arm.T_lag = 0.65f; arm.g_dc = 1.f / 0.75f;
                char tag[32]; std::snprintf(tag, sizeof tag, "L=%.2f (rule 1.37)", L);
                report(tag, run(route, arm, w.v_max, kWmax, w.a_lat, kADec, kW, false));
            }
            g_plant_tau = 0.22f; g_plant_delay = 0.20f; g_plant_gain = 0.89f;
        }
        g_headroom = 1.0f;
        std::printf("\n  L sweep — expect TV(w)/m to climb sharply below L ~ 0.3-0.45 m\n");
        for (const float L : {2.00f, 1.50f, 1.00f, 0.80f, 0.60f, 0.45f, 0.35f, 0.25f, 0.15f})
        {
            FfArm arm; arm.L = L;
            char tag[32]; std::snprintf(tag, sizeof tag, "L=%.2f m", L);
            report(tag, run(route, arm, w.v_max, kWmax, w.a_lat, kADec, kW, false));
        }
        return 0;
    }


    {
        // ── PROJECTION SELF-TEST: is the Frenet anchor even correct? ─────────────────────────────
        // Put a PERFECT robot exactly ON the curve and walk it forward. project() should then return
        // the arc length it was handed, to within a sample. Any deviation is the anchor binding to the
        // wrong part of the route — and since the plain tracker's entire error pair (e_y, e_psi) is
        // measured against that anchor, an anchor error IS a phantom tracking error that the feedback
        // will faithfully steer at. Needs no controller, no plant and no noise.
        const rc::RouteSpline &sp = route.spline();
        float hint = 0.f, worst = 0.f, worst_s = 0.f, first_bad = -1.f;
        int n_bad = 0, n = 0;
        for (float s_true = 0.f; s_true <= sp.length(); s_true += 0.05f)
        {
            ++n;
            hint = sp.project(sp.position_at(s_true), hint, 2.0f);
            const float err = std::fabs(hint - s_true);
            if (err > worst) { worst = err; worst_s = s_true; }
            if (err > 0.15f) { ++n_bad; if (first_bad < 0.f) first_bad = s_true; }
        }
        std::printf("\n  projection self-test (perfect robot ON the curve, window 2.0 m):\n");
        std::printf("    worst |s_est - s_true| = %.3f m at s=%.1f;  %d of %d samples off by >0.15 m",
                    worst, worst_s, n_bad, n);
        if (first_bad >= 0.f) std::printf(";  FIRST at s=%.2f m", first_bad);
        std::printf("\n");
        // Curvature profile, and specifically the LAP SEAM: a 2-lap route rejoins its own start, and
        // whatever the route builder does there is traversed at speed by a law whose feedforward is a
        // curvature. A cusp or a curvature step at the seam would be invisible on a 1-lap route.
        {
            float kmax = 0.f, kmax_s = 0.f;
            for (float t = 0.f; t <= sp.length(); t += 0.05f)
            {
                const float k = std::fabs(sp.kappa_avg(t, 0.40f));
                if (k > kmax) { kmax = k; kmax_s = t; }
            }
            std::printf("    |kappa_avg| max %.2f 1/m (r=%.2f m) at s=%.1f\n", kmax, 1.f / std::max(kmax, 1e-3f), kmax_s);
            const float seam = 0.5f * sp.length();
            float kseam = 0.f;
            for (float t = seam - 3.f; t <= seam + 3.f; t += 0.05f)
                kseam = std::max(kseam, std::fabs(sp.kappa_avg(t, 0.40f)));
            std::printf("    |kappa_avg| max within +-3 m of the half-way point (s=%.1f): %.2f 1/m\n", seam, kseam);
            // Heading continuity: a cusp shows as a large heading jump between adjacent samples.
            float dpsi_max = 0.f, dpsi_s = 0.f;
            for (float t = 0.05f; t <= sp.length(); t += 0.05f)
            {
                const float d = std::fabs(wrap(sp.heading_at(t) - sp.heading_at(t - 0.05f)));
                if (d > dpsi_max) { dpsi_max = d; dpsi_s = t; }
            }
            std::printf("    max heading step between 5 cm samples: %.3f rad (%.1f deg) at s=%.1f\n",
                        dpsi_max, dpsi_max * 180.f / kPi, dpsi_s);
        }
    }

    const float sp_len_for_sweep = route.spline().length();
    {
        // ── THE RECORDED FAILURE GEOMETRY, CHECKED DIRECTLY ──────────────────────────────────────
        // ★The closed-loop A/B above is INCONCLUSIVE and must not be read as validation: this bench
        // never reaches kappa*e_y anywhere near 1 (its errors stay ~50 mm), so the failure cannot occur
        // here at all. What CAN be checked is the arithmetic at the state the robot actually logged at
        // t=461.6 s: kappa = +3.5 1/m, e_y = +0.51 m, v ~ 0.15 m/s.
        {
            const float kap = 3.5f, ey = 0.51f, v = 0.15f, gdc = 1.f / 0.89f, Lg = 0.60f, epsi = 0.05f;
            const float ke = kap * ey;
            const float fb = gdc * std::max(v, 0.15f) * (-(2.f / Lg) * epsi - (1.f / (Lg * Lg)) * ey);
            const float ff_old = gdc * v * kap;
            const float draw = 1.f - ke, dmn = 0.4f;
            const float keff = kap / (draw >= 0.f ? std::max(draw, dmn) : std::min(draw, -dmn));
            const float gate = 1.f / (1.f + (ke / 0.5f) * (ke / 0.5f));
            const float ff_new = gdc * (v * gate) * keff;
            std::printf("\n  recorded failure geometry (kappa=%.1f, e_y=%.2f, v=%.2f):  kappa*e_y = %.2f\n", kap, ey, v, ke);
            std::printf("    feedback                     %+.3f rad/s  (toward the route)\n", fb);
            std::printf("    OLD feedforward  (raw kappa) %+.3f rad/s  -> sum %+.3f  CANCELS, robot goes straight on\n",
                        ff_old, ff_old + fb);
            std::printf("    NEW feedforward  kappa_eff=%+.2f, speed gate x%.3f  %+.3f rad/s  -> sum %+.3f  REINFORCES\n",
                        keff, gate, ff_new, ff_new + fb);
        }
    }

    {
        // ── CAN THE BAND EVEN MOVE? CLEARANCE AT THE AUTHORED WAYPOINTS ──────────────────────────
        // The route carries an ANCHOR likelihood pinning it to the waypoints the user clicked. If those
        // are themselves close to obstacles, no optimiser can win: the clearance preference and the
        // anchor are in direct opposition, and the anchor is the mission. Measuring this decides whether
        // "the band gains 1 mm" is a band defect or a MISSION defect.
        const rc::RobotFootprint &fp = rc::RobotFootprint::shadow();
        int tight = 0; float worst = 1e9f; std::size_t worst_i = 0;
        std::printf("\n  clearance AT the authored waypoints (body inscribed %.3f circumscribed %.3f):\n",
                    fp.inscribed_radius(), fp.circumscribed_radius());
        for (std::size_t i = 0; i < w.wp_safe.size(); ++i)
        {
            const float d = w.planner.distance_at(w.wp_safe[i]);
            if (d < worst) { worst = d; worst_i = i; }
            if (d < fp.circumscribed_radius()) ++tight;
        }
        std::printf("    worst waypoint: #%zu at %.3f m from an obstacle;  %d of %zu are closer than the "
                    "circumscribed radius\n", worst_i, worst, tight, w.wp_safe.size());
        std::printf("    all: ");
        for (std::size_t i = 0; i < w.wp_safe.size(); ++i)
            std::printf("%.2f ", w.planner.distance_at(w.wp_safe[i]));
        std::printf("\n");
    }

    {
        // ── WHICH STAGE LOSES THE CLEARANCE? ─────────────────────────────────────────────────────
        // The authored waypoints have 0.48-1.08 m of room, yet the driven route sits at ~0.19 m. Three
        // stages sit between them: the A* polyline, the spline fit + feasibility pass, and the optimiser.
        // Measure clearance on the same footing at each.
        auto stats = [&w](const std::vector<Vector2f> &pts, float &mn, float &p05)
        {
            std::vector<float> d;
            for (std::size_t i = 1; i < pts.size(); ++i)
            {
                const Vector2f a = pts[i-1], b = pts[i];
                const float len = (b - a).norm();
                const int n = std::max(1, static_cast<int>(len / 0.05f));
                for (int k = 0; k <= n; ++k) d.push_back(w.planner.distance_at(a + (b - a) * (float(k) / n)));
            }
            if (d.empty()) { mn = p05 = 0.f; return; }
            std::sort(d.begin(), d.end());
            mn = d.front(); p05 = d[static_cast<std::size_t>(d.size() * 0.05)];
        };
        std::vector<Vector2f> spline_pts;
        for (float t = 0.f; t <= route.spline().length(); t += 0.05f)
            spline_pts.push_back(route.spline().position_at(t));
        float rmn, rp5, smn, sp5;
        stats(raw_poly, rmn, rp5);
        stats(spline_pts, smn, sp5);
        float wmn = 1e9f;
        for (const auto &q : w.wp_safe) wmn = std::min(wmn, w.planner.distance_at(q));
        std::printf("\n  WHERE THE CLEARANCE GOES (raw ESDF distance, body needs 0.230-0.325 m):\n");
        std::printf("    authored waypoints        min %.3f\n", wmn);
        std::printf("    stage 1  A* polyline      min %.3f   p05 %.3f   (%zu pts)\n", rmn, rp5, raw_poly.size());
        std::printf("    stage 2+3 fitted route    min %.3f   p05 %.3f   (%zu samples)\n", smn, sp5, spline_pts.size());
    }


    {
        // ── rms(L): IS THE ADAPTATION PROBLEM EVEN WELL-POSED? ───────────────────────────────────
        // The policy minimises cross_track_rms, the one quantity that repeats on the robot (cv 2.3%
        // over nine laps, through a mission edit and a 119 mm waypoint repair). But minimising it is
        // only safe if rms(L) has an INTERIOR minimum. If rms fell monotonically as L shrank, the policy
        // would drive the gains up without bound until the loop rings — and 2/L, 1/L^2 grow fast.
        std::printf("\n  rms(L) — does minimising cross-track have an interior optimum?\n");
        std::printf("      %-8s %10s %10s %10s %9s\n", "L", "rms mm", "max mm", "TV(w)/m", "time s");
        float bestR = 1e9f, bestL = 0.f;
        for (const float Lv : {0.25f, 0.30f, 0.40f, 0.50f, 0.60f, 0.75f, 0.90f, 1.10f, 1.40f})
        {
            FfArm arm; arm.L = Lv;
            const Result R = run(route, arm, w.v_max, kWmax, w.a_lat, kADec, kW, false);
            const double rms = R.n ? std::sqrt(R.e_sq / R.n) : 0.0;
            const bool fin = R.s_end >= route.spline().length() - 0.5f;
            std::printf("      %-8.2f %10s %10.1f %10.3f %9.0f%s\n", Lv,
                        fin ? (std::to_string(static_cast<int>(rms*1000))).c_str() : "DNF",
                        R.e_max*1000, R.dist>0.1 ? R.tv_w/R.dist : 0.0, R.t_end,
                        fin && rms*1000 < bestR ? "   <- best" : "");
            if (fin && rms*1000 < bestR) { bestR = static_cast<float>(rms*1000); bestL = Lv; }
        }
        std::printf("      MINIMUM rms = %.1f mm at L = %.2f   (current L = 0.60)\n", bestR, bestL);
    }

    {
        std::printf("\n  projection heading gate (hairpin fix) — same world, same gains:\n");
        std::printf("      %-22s %9s %9s %9s %9s\n", "projection", "rms mm", "max mm", "driven m", "time s");
        for (const bool gate : {false, true})
        {
            g_heading_gate = gate;
            const Result R = run(route, FfArm{}, w.v_max, kWmax, w.a_lat, kADec, kW, false);
            const double rms = R.n ? std::sqrt(R.e_sq / R.n) : 0.0;
            const bool fin = R.s_end >= route.spline().length() - 0.5f;
            std::printf("      %-22s %9.1f %9.1f %9.2f %9.0f%s\n",
                        gate ? "heading-gated" : "plain 2 m window",
                        rms*1000, R.e_max*1000, R.dist, R.t_end, fin ? "" : "   DID NOT FINISH");
        }
        g_heading_gate = false;
    }

    {
        // ── DOES THE PIPELINE PUSH THE ROUTE ONTO ITSELF? ────────────────────────────────────────
        // In a narrow passage driven twice, the clearance term pushes BOTH legs toward the medial axis
        // — and nothing in the objective keeps the route away from itself, so they converge. Where they
        // meet, a nearest-point projection cannot tell the outbound leg from the inbound one, the search
        // snaps across, and the robot turns around where the legs touch instead of driving to the tip.
        // Measure it: the closest approach between two points of the route that are FAR APART in arc
        // length (so genuinely different passes, not neighbours on the same curve).
        auto self_min = [](const std::vector<Vector2f> &pts, float arc_sep, float step)
        {
            float best = 1e9f; std::size_t bi = 0, bj = 0;
            for (std::size_t i = 0; i < pts.size(); ++i)
                for (std::size_t j = i + static_cast<std::size_t>(arc_sep / step); j < pts.size(); ++j)
                {
                    const float d = (pts[i] - pts[j]).norm();
                    if (d < best) { best = d; bi = i; bj = j; }
                }
            return std::tuple<float, std::size_t, std::size_t>{best, bi, bj};
        };
        const rc::RouteSpline &sp2 = route.spline();
        std::vector<Vector2f> curve;
        for (float t = 0.f; t <= sp2.length(); t += 0.05f) curve.push_back(sp2.position_at(t));
        const auto [d_raw, ri, rj] = self_min(raw_poly, 2.0f, 0.10f);
        const auto [d_fit, fi, fj] = self_min(curve, 2.0f, 0.05f);
        const rc::RobotFootprint &fp2 = rc::RobotFootprint::shadow();
        std::printf("\n  route SELF-PROXIMITY (closest approach between passes >2 m apart in arc):\n");
        std::printf("      A* polyline      %.3f m\n", d_raw);
        std::printf("      fitted+optimised %.3f m   at (%.2f,%.2f)\n", d_fit,
                    curve[fi].x(), curve[fi].y());
        std::printf("      the body is %.3f m wide (circumscribed %.3f).\n",
                    2.f * fp2.circumscribed_radius(), fp2.circumscribed_radius());
        // How wide is the passage there? Two lanes need 2*body + a gap; one lane needs body + margin.
        const float free_here = w.planner.distance_at(curve[fi]);
        std::printf("      passage at that point: %.3f m to the nearest obstacle, so ~%.3f m wide.\n",
                    free_here, 2.f * free_here);
        std::printf("      two lanes would need %.3f m; %s\n", 4.f * fp2.circumscribed_radius(),
                    4.f * fp2.circumscribed_radius() <= 2.f * free_here
                        ? "IT FITS — the passes could be split left/right."
                        : "IT DOES NOT FIT — one lane only, the legs must share a line.");
    }

    {
        // ── THE PROJECTION WINDOW ────────────────────────────────────────────────────────────────
        // The route is a DIRECTED curve: arc length only moves forward, so a projection cannot
        // legitimately jump 2.5 m in one cycle however close the two points are in space. It does only
        // because the forward search window is 2 m and the fold at a hairpin sits inside it. The robot
        // covers ~0.035 m per cycle, so 2 m is ~57x what TRACKING needs — the width exists for catch-up,
        // which reacquire_ now handles separately.
        std::printf("\n  projection window sweep (robot moves ~0.035 m/cycle):\n");
        std::printf("      %-10s %9s %9s %9s %9s\n", "window m", "rms mm", "max mm", "driven m", "time s");
        for (const float wdw : {2.00f})
        {
            g_proj_window = wdw;
            const Result R = run(route, FfArm{}, w.v_max, kWmax, w.a_lat, kADec, kW, false);
            const double rms = R.n ? std::sqrt(R.e_sq / R.n) : 0.0;
            const bool fin = R.s_end >= route.spline().length() - 0.5f;
            std::printf("      %-10.2f %9.1f %9.1f %9.2f %9.0f%s   ended at s=%.2f of %.2f\n",
                        wdw, rms*1000, R.e_max*1000, R.dist, R.t_end,
                        fin ? "" : "   STALLED", R.s_end, route.spline().length());
            if (not fin)
            {
                // Where it stalled, and what the route is doing there.
                const float st = R.s_end;
                std::printf("                 stall point: kappa_avg=%.2f 1/m  heading turns %.0f deg\n",
                            route.spline().kappa_avg(st, 0.40f),
                            180.f / kPi * std::fabs(wrap(route.spline().heading_at(st + 1.0f)
                                                       - route.spline().heading_at(st - 0.2f))));
                const Vector2f sp_pos = route.spline().position_at(st);
                std::printf("                 at (%.2f, %.2f)\n", sp_pos.x(), sp_pos.y());
            }
        }
        g_proj_window = 2.0f;
    }

    {
        // ── WILL THE POLICY'S NEXT STEP BUST THE BUDGET? ─────────────────────────────────────────
        // The constraint is rot_effort/distance (integral |omega| dt per metre), NOT TV(w). They differ:
        // rot_effort is how much heading the robot actually turns through, which is mostly set by the
        // ROUTE; TV(w) is how roughly it does it. Tightening L should raise the second a lot and the
        // first only a little — if so the constraint barely binds and the policy will walk L to its
        // floor. Worth knowing before spending three laps per step.
        std::printf("\n  does the constraint bind? (robot at L=0.60: rot/m 0.831, budget 0.870)\n");
        std::printf("      %-8s %9s %10s %10s\n", "L", "rms mm", "rot/m", "TV(w)/m");
        for (const float Lv : {0.45f, 0.50f, 0.60f, 0.75f, 1.00f})
        {
            FfArm arm; arm.L = Lv;
            const Result R = run(route, arm, w.v_max, kWmax, w.a_lat, kADec, kW, false);
            const double rms = R.n ? std::sqrt(R.e_sq / R.n) : 0.0;
            std::printf("      %-8.2f %9.1f %10.3f %10.3f\n", Lv, rms*1000,
                        R.dist > 0.1 ? R.rot_effort / R.dist : 0.0,
                        R.dist > 0.1 ? R.tv_w / R.dist : 0.0);
        }
    }


    // (The steer-floor sweep lived here. It cannot run any more: the floor is a constant inside
    // PlainTracker, not a parameter, and this bench now LINKS the tracker instead of copying it —
    // sweeping it would mean adding a knob to the robot to serve the bench. Its answer is already
    // recorded in etc/config.toml; re-run it by editing kSteerFloorMps if it is ever reopened.)

    if (g_stop_test)
    {
        // ── DOES IT COME TO REST ON THE ENDPOINT? ────────────────────────────────────────────────
        // The question a POINT target asks and a mission route never does. Reported as the two numbers
        // that decide it: how far from the endpoint the robot finally stopped, and how far PAST the
        // endpoint it got on the way — a tracker can satisfy the first by creeping back after
        // overshooting, and on the robot that is a collision, not an arrival.
        std::printf("\n  ── TERMINAL APPROACH (run to rest, not to length-0.15) ─────────────────────\n");
        std::printf("      %-18s %10s %10s %10s %8s\n", "arm", "stopped", "end dist", "max past", "end v");
        for (auto &[nm, R] : std::initializer_list<std::pair<const char *, Result>>{
                 {"pd",    run(route, PdArm{}, w.v_max, kWmax, w.a_lat, kADec, kW, true)},
                 {"plain", run(route, FfArm{}, w.v_max, kWmax, w.a_lat, kADec, kW, true)}})
            std::printf("      %-18s %10s %8.3f m %8.3f m %6.3f m/s%s\n", nm,
                        R.came_to_rest ? "yes" : "NO", R.end_dist, R.max_past, R.end_speed,
                        R.max_past > 0.25f ? "   <- OVERSHOT the endpoint" : "");
        std::printf("\n  (max past > 0 means it drove BEYOND the end of the curve.)\n");
        return 0;
    }

    report("pd", run(route, PdArm{}, w.v_max, kWmax, w.a_lat, kADec, kW, true));
    report("plain (FF+Frenet)", run(route, FfArm{}, w.v_max, kWmax, w.a_lat, kADec, kW, true));
    std::printf("\n  reminder: no obstacles here, so no gate and no carrot clip — the PD arm is an\n"
                "  OPTIMISTIC baseline (on the robot its carrot is clipped on 75%% of cycles).\n");
    return 0;
}
