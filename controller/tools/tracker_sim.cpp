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
float speed_limit(const rc::RouteSpline &sp, float s_now, float v_cap,
                  float a_lat, float a_dec, float w_max, float W, float headroom = 1.0f)
{
    float v = v_cap;
    const float horizon = v_cap * v_cap / (2.f * a_dec) + 1.0f;
    for (float ds = 0.f; ds <= horizon; ds += 0.10f)
    {
        const float k_pt = std::abs(sp.curvature_at(s_now + ds));
        const float k_av = std::abs(sp.kappa_avg(s_now + ds + 0.5f * W, W));
        if (k_pt < 1e-3f and k_av < 1e-3f) continue;
        const float v_lat = k_pt > 1e-3f ? std::sqrt(a_lat / k_pt) : v_cap;
        const float v_rot = k_av > 1e-3f ? headroom * w_max / k_av : v_cap;
        v = std::min(v, std::sqrt(std::min(v_lat, v_rot) * std::min(v_lat, v_rot) + 2.f * a_dec * ds));
    }
    return std::clamp(v, 0.15f, v_cap);
}

// ── THE CLEARANCE SPEED BOUND, as compute_route_tracker computes it ──────────────────────────────
// The production code queries the tracker's robot-frame ESDF (rebuilt each cycle from lidar); here the
// stand-in is the recorded grid's exact EDT, which IS the static geometry the route was planned
// against. That difference matters for the absolute numbers but not for the question this reproduces:
// does d_min collapse where the robot plainly has room?
struct Clearance
{
    const rc::GridPlanner *grid = nullptr;
    float scan_m = 1.0f, standoff = 0.0f;   // production default; see kRouteStandoffM
    bool  directional = true;      // false = the circumscribed disc (the first, frozen version)

    // Returns v_clear and writes the limiting d_min.
    float v_clear(const rc::RouteSpline &sp, float s, float th_fwd, float a_dec, float &d_min_out) const
    {
        const rc::RobotFootprint &fp = rc::RobotFootprint::shadow();
        float d_min = std::numeric_limits<float>::max();
        for (float ds = 0.f; ds <= scan_m; ds += 0.10f)
        {
            const Vector2f q = sp.position_at(s + ds);
            const float d = grid->distance_at(q);
            float extent = fp.circumscribed_radius();
            if (directional)
            {
                const Vector2f g = grid->distance_gradient_at(q);
                const float dpsi = wrap(sp.heading_at(s + ds) - th_fwd);
                extent = fp.support_radius(-dpsi, -g);
            }
            d_min = std::min(d_min, d - extent);
        }
        d_min_out = d_min;
        return std::sqrt(2.f * a_dec * std::max(0.f, d_min - standoff));
    }
};

// ── ARM A: the CURRENT law (carrot + Stanley + PD + EMA + Gaussian brake), unclipped ─────────────
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

// ── ARM B: the design under test — curvature feedforward + critically damped Frenet feedback ─────
struct FfArm
{
    float L = 1.0f;             // the ONE chosen gain; both feedback gains follow by critical damping
    float T_lag = 0.42f;        // identified total lag
    float g_dc = 1.f / 0.89f;   // identified DC gain, inverted
    float W = 0.40f;            // route's own smoothing scale

    const Clearance *clear = nullptr;   // null = no obstacles (the original, tracking-only bench)
    mutable float last_d_min = 1e9f, last_v_clear = 1e9f;

    void control(const rc::RouteSpline &sp, float s, float px, float py, float th,
                 float v_limit, float w_max, float &v_cmd, float &w_cmd) const
    {
        const Vector2f r = sp.position_at(s);
        const float psi = sp.heading_at(s);
        const float e_y = -std::sin(psi) * (px - r.x()) + std::cos(psi) * (py - r.y());  // left positive
        const float e_psi = wrap(th - psi);
        v_cmd = v_limit;
        if (clear != nullptr)
        {
            float d_min = 1e9f;
            const float vc = clear->v_clear(sp, s, th, 1.0f, d_min);
            last_d_min = d_min; last_v_clear = vc;
            v_cmd = std::min(v_cmd, vc);
        }
        // Preview the FEEDFORWARD only, by exactly the measured lag. kappa_avg is centred, so this is
        // the only lookahead in the law — a forward-windowed estimator would add a second one.
        const float k_ff = sp.kappa_avg(s + v_cmd * T_lag, W);
        // Feedforward on the REAL speed; feedback on a floored one, so steering authority never
        // vanishes with the clearance bound (the deadlock that froze the first route lap).
        const float v_steer = std::max(v_cmd, 0.15f);
        w_cmd = std::clamp(g_dc * (v_cmd * k_ff
                                   + v_steer * (-(2.f / L) * e_psi - (1.f / (L * L)) * e_y)),
                           -w_max, w_max);
    }
};

struct Result
{
    int n = 0;
    double e_sq = 0, tv_v = 0, tv_w = 0, dist = 0, t_end = 0;
    float e_max = 0;
    std::vector<float> e, kappa;      // for the correlation and its lag
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
float g_headroom = 1.0f;   // fraction of the omega budget the FEEDFORWARD may claim; the rest is
                           // reserved for feedback authority. 1.0 = the naive limit.

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
    bool  first = true;

    for (float t = 0.f; t < 400.f; t += kCtrlDt)
    {
        s_hint = sp.project({P.x, P.y}, s_hint, 2.0f);
        if (s_hint >= sp.length() - 0.15f) { R.t_end = t; break; }

        const float v_lim = speed_limit(sp, s_hint, v_cap, a_lat, a_dec, w_max, W, g_headroom);
        float v_cmd = 0.f, w_cmd = 0.f;
        arm.control(sp, s_hint, P.x, P.y, P.th, v_lim, w_max, v_cmd, w_cmd);

        if (not first) { R.tv_v += std::abs(v_cmd - last_v); R.tv_w += std::abs(w_cmd - last_w); }
        last_v = v_cmd; last_w = w_cmd; first = false;

        const Vector2f r = sp.position_at(s_hint);
        const float psi = sp.heading_at(s_hint);
        const float e = -std::sin(psi) * (P.x - r.x()) + std::cos(psi) * (P.y - r.y());
        R.e_sq += double(e) * e; R.e_max = std::max(R.e_max, std::abs(e)); ++R.n;
        R.e.push_back(e);
        R.kappa.push_back(sp.kappa_avg(s_hint, W));

        const float x0 = P.x, y0 = P.y;
        for (float u = 0.f; u < kCtrlDt - 1e-6f; u += kSubDt) P.step(v_cmd, w_cmd, kSubDt);
        R.dist += std::hypot(P.x - x0, P.y - y0);
        R.t_end = t;
    }
    if (verbose)
        std::printf("    (ran %d cycles, %.1f m, %.1f s)\n", R.n, R.dist, R.t_end);
    return R;
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
}

struct World
{
    rc::GridPlanner planner;
    std::vector<Vector2f> wp_safe;
    Vector2f start{0.f, 0.f};
    int laps = 1;
    float spacing = 0.05f, smoothing = 0.40f, v_max = 0.7f, a_lat = 1.0f, standoff = 0.6f;
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
    }
    f.clear(); f.seekg(0);
    if (not w.planner.read_grid(f)) { std::printf("no readable grid in %s\n", path.c_str()); return false; }
    return w.wp_safe.size() >= 2;
}
}   // namespace

int main(int argc, char **argv)
{
    std::string path = "route_world.txt";
    bool sweep = false;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "--sweep") sweep = true; else path = a;
    }

    World w;
    if (not load_world(path, w)) return 1;

    rc::RouteFollower route;
    auto plan = [&w](const Vector2f &a, const Vector2f &b) { return w.planner.plan(a, b); };
    auto free_at = [&w](const Vector2f &p, float h) { return w.planner.pose_free(p, h); };
    if (not route.build(w.start, w.wp_safe, w.laps, plan, free_at, w.spacing, w.smoothing))
    { std::printf("route build failed\n"); return 1; }

    constexpr float kWmax = 0.8f, kADec = 1.0f, kW = 0.40f;
    std::printf("tracker_sim: %s — route %.1f m, %zu waypoints, v_max %.2f, plant tau 0.22 delay 0.20 gain 0.89\n\n",
                path.c_str(), route.length(), w.wp_safe.size(), w.v_max);

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
        // ── REPRODUCE THE ROBOT'S MID-ROUTE STOP ────────────────────────────────────────────────
        // Walk the whole route and evaluate the clearance bound at every metre, exactly as the tracker
        // would. If d_min collapses below the standoff anywhere the robot plainly has room, this prints
        // where — which is the diagnostic the CSV could not give because d_min is not logged.
        Clearance cl; cl.grid = &w.planner;
        const rc::RouteSpline &sp = route.spline();
        const rc::RobotFootprint &fp = rc::RobotFootprint::shadow();
        std::printf("  clearance scan along the route (standoff %.2f m, body inscribed %.3f circumscribed %.3f)\n",
                    cl.standoff, fp.inscribed_radius(), fp.circumscribed_radius());
        int n_zero_dir = 0, n_zero_disc = 0; float worst_dir = 1e9f, worst_disc = 1e9f, worst_s = 0.f;
        for (float s = 0.f; s < sp.length(); s += 0.25f)
        {
            const float th = sp.heading_at(s);
            float d_dir = 0.f, d_disc = 0.f;
            cl.directional = true;  const float v_dir  = cl.v_clear(sp, s, th, 1.0f, d_dir);
            cl.directional = false; const float v_disc = cl.v_clear(sp, s, th, 1.0f, d_disc);
            if (v_dir  <= 1e-3f) ++n_zero_dir;
            if (v_disc <= 1e-3f) ++n_zero_disc;
            if (d_dir < worst_dir) { worst_dir = d_dir; worst_s = s; }
            worst_disc = std::min(worst_disc, d_disc);
        }
        const int n = static_cast<int>(sp.length() / 0.25f) + 1;
        std::printf("    directional extent : d_min worst %+.3f m at s=%.1f, v_clear==0 at %d/%d points\n",
                    worst_dir, worst_s, n_zero_dir, n);
        std::printf("    circumscribed disc : d_min worst %+.3f m,                v_clear==0 at %d/%d points\n",
                    worst_disc, n_zero_disc, n);
        // What standoff does this route actually admit? The bound reaches zero at d_min = standoff, so
        // a standoff larger than the route's tightest body margin forbids a route the planner already
        // certified as FEASIBLE — the planner's job is fit, not comfort.
        std::printf("    standoff sweep (v_clear==0 count, and the slowest crawl it permits):\n");
        for (const float c : {0.10f, 0.05f, 0.02f, 0.0f})
        {
            cl.standoff = c; cl.directional = true;
            int zeros = 0; float v_slowest = 1e9f;
            for (float s = 0.f; s < sp.length(); s += 0.25f)
            {
                float d = 0.f;
                const float v = cl.v_clear(sp, s, sp.heading_at(s), 1.0f, d);
                if (v <= 1e-3f) ++zeros;
                v_slowest = std::min(v_slowest, v);
            }
            std::printf("      standoff %.2f m -> %2d/%d zero, slowest %.3f m/s\n", c, zeros, n, v_slowest);
        }
        cl.standoff = 0.0f;
        // Does the clearance bound disagree with route_speed_limit's own 0.15 m/s floor at the tight
        // point? The floor lives INSIDE route_speed_limit (curvature only), and the tracker takes a
        // min() of that with v_clear — so v_clear may legitimately go below 0.15 and does win. Checked
        // rather than assumed, because two floors that disagree is how a robot ends up crawling or
        // refusing to move for reasons nobody can find.
        {
            float d = 0.f;
            const float v = cl.v_clear(sp, worst_s, sp.heading_at(worst_s), 1.0f, d);
            std::printf("    at the tightest point (s=%.1f): d_min %+.3f m -> v_clear %.3f m/s;"
                        " route_speed_limit floors at 0.150 and the tracker takes the min => %.3f\n\n",
                        worst_s, d, v, std::min(v, 0.150f));
        }
    }

    report("PD (current)", run(route, PdArm{}, w.v_max, kWmax, w.a_lat, kADec, kW, true));
    report("FF+Frenet", run(route, FfArm{}, w.v_max, kWmax, w.a_lat, kADec, kW, true));
    std::printf("\n  reminder: no obstacles here, so no gate and no carrot clip — the PD arm is an\n"
                "  OPTIMISTIC baseline (on the robot its carrot is clipped on 75%% of cycles).\n");
    return 0;
}
