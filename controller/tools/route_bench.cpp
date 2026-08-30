/*
 * route_bench.cpp — build the real tour OFFLINE, against the real world, and measure the ROUTE.
 *
 * WHY THIS EXISTS. The route optimiser's weights act on GEOMETRY: how tight the worst corner is, how
 * far the curve sits from the walls, how far it drifts from what was clicked. Measuring a geometric
 * effect by driving a lap costs a run per weight, and the run-to-run noise floor on this robot is
 * 14.5% on reversals and 44% on min clearance — so a single lap cannot even resolve the difference
 * between two weight settings. Here the same route is rebuilt from the same snapshot in milliseconds,
 * and the geometry is exact: no noise, no localisation, no MPPI.
 *
 * WHAT IT IS NOT. It measures the ROUTE, never the driving. Lap time here is an idealised traversal of
 * the curve under v = sqrt(a_lat/kappa); it says what the route ALLOWS, not what the robot achieves.
 * A route that is better here can still be driven worse — that is exactly the question a live run
 * answers and this one cannot. The point is to arrive at the live run with one candidate instead of
 * ten.
 *
 * FIDELITY. The world comes from ControllerSession::dump_route_world as the planner's RASTER, and the
 * route is built with the same RouteFollower/RouteSpline/optimize_route the controller calls, at the
 * same spacing and smoothing, through the same A*. The only thing that is reconstructed rather than
 * replayed is nothing at all.
 *
 *   bin/route_bench [route_world.txt] [key=value,value,... ...]
 *
 * With no keys it prints the world, the unoptimised route, and the route under the snapshot's own
 * optimiser settings. Keys take comma-separated lists and are swept as a cartesian product:
 *
 *   bin/route_bench route_world.txt w_kappa=1,2,4,8 clear_peak=0,4
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <clocale>

#include "../../common/robot_footprint/mesh_hull.h"
#include "../src/grid_planner.h"
#include "../src/route_follower.h"
#include "../src/route_optimizer.h"
// Header-only, pure, and it guards a SAFETY reflex — so it belongs in the one command that runs every
// self-test rather than in a scratch main that ages. See stall_judge.h for what it caught.
#include "../src/stall_judge.h"
#include "../src/trackers/plain_tracker.h"

using Eigen::Vector2f;

namespace
{

struct World
{
    rc::GridPlanner planner;
    std::string mission = "?";
    std::vector<Vector2f> wp_raw, wp_safe;
    Vector2f start{0.f, 0.f};
    int   laps = 1;
    float spacing = 0.05f, smoothing = 0.40f;
    float v_max = 0.7f, a_lat = 1.0f, standoff = 0.6f;
    rc::RouteOptimizerConfig opt;   // as the controller configured it at build time
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
        if (key == "grid") break;   // everything from here is the raster; GridPlanner parses its own format
        else if (key == "mission") { std::getline(ls, w.mission); if (not w.mission.empty() and w.mission[0] == ' ') w.mission.erase(0, 1); }
        else if (key == "laps")    ls >> w.laps;
        else if (key == "start")   ls >> w.start.x() >> w.start.y();
        else if (key == "wp_raw")  { Vector2f p; ls >> p.x() >> p.y(); w.wp_raw.push_back(p); }
        else if (key == "wp_safe") { Vector2f p; ls >> p.x() >> p.y(); w.wp_safe.push_back(p); }
        else if (key == "fit")     ls >> w.spacing >> w.smoothing >> w.v_max >> w.a_lat >> w.standoff;
        else if (key == "opt")
            // Trailing fields are optional: a stream that runs out leaves the rest at their defaults, so a
            // snapshot written before a field existed still replays as the optimiser that wrote it.
            ls >> w.opt.d_target >> w.opt.rho >> w.opt.sigma_a >> w.opt.clearance_floor
               >> w.opt.w_kappa >> w.opt.w_clear >> w.opt.w_gauge >> w.opt.clear_peak
               >> w.opt.anchor_huber >> w.opt.iterations >> w.opt.kappa_peak >> w.opt.safety_bias;
    }
    // Rewind and let GridPlanner find its own header: it skips whatever it does not recognise, so the
    // two parsers never need to agree on where the boundary between the sections is.
    f.clear();
    f.seekg(0);
    if (not w.planner.read_grid(f)) { std::printf("no readable grid in %s\n", path.c_str()); return false; }
    if (w.wp_safe.size() < 2) { std::printf("fewer than 2 waypoints in %s\n", path.c_str()); return false; }
    return true;
}

// ── Route metrics ────────────────────────────────────────────────────────────────────────────────
struct Metrics
{
    bool  built = false;
    float length = 0.f;
    int   samples = 0, corrections = 0;
    float kappa_max = 0.f, kappa_rms = 0.f, r_min = 0.f;
    float arc_below_inscribed = 0.f, arc_below_half = 0.f;
    int   pivots = 0;              // separate stretches asking for a radius the robot cannot turn
    float clear_min = 0.f, clear_p05 = 0.f, clear_mean = 0.f;
    int   infeasible = 0;          // samples whose footprint (at the tangent heading) is in collision
    float dev_max = 0.f, dev_mean = 0.f;   // route vs the RECORDED waypoints
    // Per waypoint, so "the route passes far from one of them" can name WHICH. The aggregate above
    // cannot: a single badly-missed point and a route uniformly loose look the same in dev_max/dev_mean.
    std::vector<float> dev_per_wp;
    float t_ideal = 0.f;           // traversal time under the curvature speed law
    rc::RouteOptimizerReport opt;
    // Each pivot as (arc length, radius, position), worst first. The POSITION is the point of it: a pivot
    // is only actionable once you know whether it sits on an authored waypoint (only the tour can fix it)
    // or in open route between two (the optimiser can).
    struct Pivot { float s, r; Vector2f p; };
    std::vector<Pivot> tight;
};

// Menger curvature over three consecutive samples. The samples are uniform in arc length, so this is a
// second difference in disguise — but written this way it is exact for a circle at any spacing, which
// matters because the whole point is to read off a MINIMUM TURN RADIUS and compare it to the robot's.
std::vector<float> curvature_profile(const std::vector<Vector2f> &p)
{
    std::vector<float> k(p.size(), 0.f);
    // The first and last few samples are skipped: the fitted curve's end spans are straight by
    // construction (RouteSpline triples the end control points) but their PARAMETERISATION is degenerate,
    // so the resample can leave near-coincident samples there and a three-point curvature estimate on
    // those is 0/0. Chasing that artefact once already cost an afternoon — it read as a 2 mm turn radius
    // on a curve whose tightest real corner was 0.32 m.
    constexpr std::size_t kEndSkip = 3;
    if (p.size() <= 2 * kEndSkip + 2) return k;
    for (std::size_t i = kEndSkip; i + kEndSkip < p.size(); ++i)
    {
        const float a = (p[i] - p[i - 1]).norm(), b = (p[i + 1] - p[i]).norm(), c = (p[i + 1] - p[i - 1]).norm();
        const float area2 = std::abs((p[i].x() - p[i - 1].x()) * (p[i + 1].y() - p[i - 1].y())
                                   - (p[i + 1].x() - p[i - 1].x()) * (p[i].y() - p[i - 1].y()));
        k[i] = (a * b * c > 1e-12f) ? area2 / (a * b * c) : 0.f;
    }
    return k;
}

// Idealised traversal time: v = min(v_max, sqrt(a_lat/kappa)), made reachable by a backward then forward
// pass at a longitudinal acceleration limit. This is the same physics as the controller's speed ceiling
// (route_speed_limit), evaluated over the whole curve instead of a lookahead window. It is a property of
// the CURVE — it contains no robot, no tracking error and no obstacle reaction.
float ideal_time(const std::vector<Vector2f> &p, const std::vector<float> &k, float v_max, float a_lat, float a_lon)
{
    const std::size_t n = p.size();
    if (n < 2) return 0.f;
    std::vector<float> v(n, v_max), ds(n, 0.f);
    for (std::size_t i = 1; i < n; ++i) ds[i] = (p[i] - p[i - 1]).norm();
    for (std::size_t i = 0; i < n; ++i)
        if (k[i] > 1e-4f) v[i] = std::min(v[i], std::sqrt(a_lat / k[i]));
    for (std::size_t i = n - 1; i-- > 0;)                       // braking (backward)
        v[i] = std::min(v[i], std::sqrt(v[i + 1] * v[i + 1] + 2.f * a_lon * ds[i + 1]));
    for (std::size_t i = 1; i < n; ++i)                          // accelerating (forward)
        v[i] = std::min(v[i], std::sqrt(v[i - 1] * v[i - 1] + 2.f * a_lon * ds[i]));
    float t = 0.f;
    for (std::size_t i = 1; i < n; ++i)
        t += ds[i] / std::max(0.05f, 0.5f * (v[i] + v[i - 1]));
    return t;
}

Metrics evaluate(World &w, const rc::RouteOptimizerConfig *opt, float smoothing, float spacing)
{
    Metrics m;
    rc::RouteFollower route;
    if (opt != nullptr)
    {
        // The snapshot carries weights but cannot carry the field itself — it is a pair of callables over
        // the planner. Bound here, to the SAME GridPlanner EDT the controller hands the optimiser.
        rc::RouteOptimizerConfig o = *opt;
        o.distance = [&w](const Vector2f &p) { return w.planner.distance_at(p); };
        o.distance_gradient = [&w](const Vector2f &p) { return w.planner.distance_gradient_at(p); };
        route.set_optimizer(o);
    }

    auto plan = [&w](const Vector2f &a, const Vector2f &b) { return w.planner.plan(a, b); };
    auto free_at = [&w](const Vector2f &p, float h) { return w.planner.pose_free(p, h); };
    if (not route.build(w.start, w.wp_safe, w.laps, plan, free_at, spacing, smoothing)) return m;

    m.built = true;
    m.length = route.length();
    m.corrections = route.corrections();
    m.opt = route.spline().last_optimizer_report();

    const auto &s = route.path();
    m.samples = static_cast<int>(s.size());
    const auto k = curvature_profile(s);
    const float r_inscribed = rc::RobotFootprint::shadow().inscribed_radius();

    double k2 = 0.0;
    for (std::size_t i = 0; i < k.size(); ++i)
    {
        m.kappa_max = std::max(m.kappa_max, k[i]);
        k2 += static_cast<double>(k[i]) * k[i];
    }
    m.kappa_rms = static_cast<float>(std::sqrt(k2 / std::max<std::size_t>(1, k.size())));
    m.r_min = m.kappa_max > 1e-6f ? 1.f / m.kappa_max : 1e9f;

    // Arc length spent below a given turn radius, and the PIVOTS — contiguous stretches asking for a
    // radius the robot cannot turn at all. The count matters more than the total length: each stretch is
    // one place where a differential drive must stop translating and rotate on the spot, which is where
    // the reversals in a lap come from. Ten centimetres of it in one place is one pivot; the same ten
    // centimetres spread over five places is five.
    bool in_tight = false;
    float tight_worst = 0.f, tight_s = 0.f, s_acc = 0.f;
    Vector2f tight_p = Vector2f::Zero();
    for (std::size_t i = 0; i < s.size(); ++i)
    {
        const float step = i ? (s[i] - s[i - 1]).norm() : 0.f;
        s_acc += step;
        const float r = k[i] > 1e-6f ? 1.f / k[i] : 1e9f;
        if (r < r_inscribed) m.arc_below_inscribed += step;
        if (r < 0.5f) m.arc_below_half += step;
        if (r < r_inscribed)
        {
            if (not in_tight) { in_tight = true; tight_worst = r; tight_s = s_acc; tight_p = s[i]; }
            else if (r < tight_worst) { tight_worst = r; tight_s = s_acc; tight_p = s[i]; }
        }
        else if (in_tight) { in_tight = false; ++m.pivots; m.tight.push_back({tight_s, tight_worst, tight_p}); }
    }
    if (in_tight) { ++m.pivots; m.tight.push_back({tight_s, tight_worst, tight_p}); }
    std::sort(m.tight.begin(), m.tight.end(),
              [](const auto &a, const auto &b) { return a.r < b.r; });

    // Clearance is centre-to-nearest-occupied, the same field the optimiser descends. Reported as p05 as
    // well as min because min is the noisiest statistic in the whole stack (44% run to run live) — though
    // here, with no perception in the loop, min is at least repeatable.
    std::vector<float> clr;
    clr.reserve(s.size());
    for (std::size_t i = 0; i < s.size(); ++i)
    {
        clr.push_back(w.planner.distance_at(s[i]));
        const Vector2f t = (i + 1 < s.size()) ? (s[i + 1] - s[i]) : (s[i] - s[i - 1]);
        if (not w.planner.pose_free(s[i], std::atan2(t.y(), t.x()))) ++m.infeasible;
    }
    auto sorted = clr;
    std::sort(sorted.begin(), sorted.end());
    m.clear_min = sorted.front();
    m.clear_p05 = sorted[static_cast<std::size_t>(0.05 * (sorted.size() - 1))];
    m.clear_mean = static_cast<float>(std::accumulate(clr.begin(), clr.end(), 0.0) / clr.size());

    // Fidelity to the tour AS RECORDED: for each authored waypoint, the closest the route comes to it.
    // Against wp_raw, not the repaired set — the repair is already a departure from what was asked, and
    // folding it in would hide it.
    double dev_sum = 0.0;
    for (const auto &q : w.wp_raw)
    {
        float best = 1e9f;
        for (const auto &p : s) best = std::min(best, (p - q).norm());
        m.dev_max = std::max(m.dev_max, best);
        m.dev_per_wp.push_back(best);
        dev_sum += best;
    }
    m.dev_mean = static_cast<float>(dev_sum / std::max<std::size_t>(1, w.wp_raw.size()));

    m.t_ideal = ideal_time(s, k, w.v_max, w.a_lat, 1.0f);
    return m;
}

void print_header()
{
    std::printf("\n%-26s %7s %6s %6s %6s %5s %7s %7s %6s %6s %6s %5s %8s\n",
                "config", "len_m", "R_min", "a<Rin", "a<0.5", "piv", "clr_min", "clr_p05",
                "dev_mx", "dev_mn", "t_idl", "corr", "S_after");
    std::printf("%s\n", std::string(120, '-').c_str());
}

void print_row(const std::string &label, const Metrics &m)
{
    if (not m.built) { std::printf("%-26s  BUILD FAILED\n", label.c_str()); return; }
    std::printf("%-26s %7.2f %6.3f %6.2f %6.2f %5d %7.3f %7.3f %6.3f %6.3f %6.1f %5d %8.4f%s\n",
                label.c_str(), m.length, m.r_min, m.arc_below_inscribed, m.arc_below_half, m.pivots,
                m.clear_min, m.clear_p05, m.dev_max, m.dev_mean, m.t_ideal, m.corrections,
                m.opt.ran ? m.opt.cost_after : 0.f,
                m.opt.rejected ? "  REJECTED" : (m.infeasible ? "  INFEASIBLE" : ""));
}

void print_detail(const std::string &label, const Metrics &m)
{
    if (not m.built) return;
    std::printf("  %s: kappa max %.3f rms %.3f | infeasible samples %d | samples %d\n",
                label.c_str(), m.kappa_max, m.kappa_rms, m.infeasible, m.samples);
    // Which waypoint the route gives up on. dev_max alone cannot tell a single abandoned point from a
    // uniformly loose route, and "it passes far from one of them" is a per-waypoint claim.
    if (not m.dev_per_wp.empty())
    {
        std::printf("    waypoint fit (m):");
        for (std::size_t i = 0; i < m.dev_per_wp.size(); ++i)
            std::printf(" %zu:%.2f", i, m.dev_per_wp[i]);
        const auto it = std::max_element(m.dev_per_wp.begin(), m.dev_per_wp.end());
        std::printf("   <- worst wp%zu at %.3f m\n",
                    static_cast<std::size_t>(it - m.dev_per_wp.begin()), *it);
    }
    if (m.opt.ran)
        std::printf("    optimiser: %d iters, S %.4f -> %.4f  (kappa %.4f clear %.4f anchor %.4f gauge %.4f)"
                    "  max_move %.3f m  clearance %.3f -> %.3f%s\n",
                    m.opt.iterations, m.opt.cost_before, m.opt.cost_after,
                    m.opt.e_kappa, m.opt.e_clear, m.opt.e_anchor, m.opt.e_gauge,
                    m.opt.max_move_m, m.opt.min_clearance_before, m.opt.min_clearance_after,
                    m.opt.rejected ? "  [REVERTED]" : "");
    if (not m.tight.empty())
    {
        std::printf("    pivots (R < inscribed):");
        for (std::size_t i = 0; i < m.tight.size() and i < 6; ++i)
            std::printf("  s=%.1f R=%.3f at (%+.2f,%+.2f)",
                        m.tight[i].s, m.tight[i].r, m.tight[i].p.x(), m.tight[i].p.y());
        std::printf("\n");
    }
}

// Write s, position, curvature, the curvature-allowed speed and clearance for every sample. This is what
// makes "is the curvature ceiling what is slowing the robot down?" answerable: joined against a run's
// profile CSV on route_s_m, it puts the speed the ROUTE allows beside the speed the robot chose.
void dump_profile(const std::string &path, const World &w, const std::vector<Vector2f> &p,
                  const std::vector<float> &k)
{
    std::ofstream f(path);
    if (not f.is_open()) { std::printf("cannot write %s\n", path.c_str()); return; }
    f << "s_m,x,y,kappa,r_m,v_curv_mps,clearance_m\n";
    float s_acc = 0.f;
    for (std::size_t i = 0; i < p.size(); ++i)
    {
        if (i) s_acc += (p[i] - p[i - 1]).norm();
        const float kk = k[i];
        const float v = kk > 1e-4f ? std::min(w.v_max, std::sqrt(w.a_lat / kk)) : w.v_max;
        f << s_acc << ',' << p[i].x() << ',' << p[i].y() << ',' << kk << ','
          << (kk > 1e-6f ? 1.f / kk : 1e9f) << ',' << v << ','
          << w.planner.distance_at(p[i]) << '\n';
    }
    std::printf("wrote %s (%zu samples)\n", path.c_str(), p.size());
}

// key=v1,v2,... — a sweep axis. Order is preserved so the product is printed in a predictable order.
struct Axis { std::string key; std::vector<float> values; };

bool apply_key(rc::RouteOptimizerConfig &o, float &smoothing, float &spacing, const std::string &k, float v)
{
    // handled by the caller (they belong to the World, not the optimiser) — accepted here so the key check passes
    if (k == "a_lat" or k == "v_max") return true;
    if      (k == "w_kappa")      o.w_kappa = v;
    else if (k == "w_clear")      o.w_clear = v;
    else if (k == "w_gauge")      o.w_gauge = v;
    else if (k == "w_jerk")       o.w_jerk = v;    // dkappa/ds prior — see route_optimizer.h
    else if (k == "clear_peak")   o.clear_peak = v;
    else if (k == "kappa_peak")   o.kappa_peak = v;
    else if (k == "sigma_a")      o.sigma_a = v;
    else if (k == "d_target")     o.d_target = v;
    else if (k == "rho")          o.rho = v;
    else if (k == "anchor_huber") o.anchor_huber = v;
    else if (k == "iterations")   o.iterations = static_cast<int>(v);
    else if (k == "clearance_floor") o.clearance_floor = v;
    else if (k == "safety_bias")  o.safety_bias = v;
    else if (k == "curvature_bound") o.curvature_bound = v != 0.f;
    else if (k == "min_seg_m")    o.min_seg_m = v;
    else if (k == "smoothing")    smoothing = v;
    else if (k == "spacing")      spacing = v;
    else return false;
    return true;
}

}  // namespace

int main(int argc, char **argv)
{
    std::string path = "route_world.txt";
    std::string dump_path;
    std::vector<Axis> axes;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        // The route modules all carry a self_test() and NOTHING CALLED THEM — they ran once from a
        // throwaway main and then only ever aged. A tool that already links exactly these translation
        // units is the natural place for them to live, so they are one command away from being run.
        if (a == "--self-test")
        {
            // ★SET THE AGENT'S LOCALE FIRST. rc::mesh parses an OBJ, and these machines run es_ES where the
            // decimal separator is a COMMA — but a standalone harness has no Qt, so nothing calls
            // setlocale() and the truncation bug SIMPLY DOES NOT OCCUR here. Without this line the mesh
            // test would pass while the agent broke, which is the exact contradiction CLAUDE.md says to
            // chase before any other explanation.
            std::setlocale(LC_ALL, "");
            const bool ok = rc::mesh::self_test()
                          & rc::RobotFootprint::self_test()
                          & rc::GridPlanner::self_test()
                          & rc::RouteSpline::self_test()
                          & rc::RouteFollower::self_test()
                          & rc::route_optimizer_self_test()
                          & rc::PlainTracker::self_test()
                          & rc::StallJudge::self_test();
            std::printf("\n%s\n", ok ? "ALL SELF-TESTS PASS" : "SELF-TESTS FAILED");
            return ok ? 0 : 1;
        }
        const auto eq = a.find('=');
        if (eq == std::string::npos) { path = a; continue; }
        if (a.compare(0, 5, "dump=") == 0) { dump_path = a.substr(5); continue; }
        Axis ax{a.substr(0, eq), {}};
        std::stringstream ss(a.substr(eq + 1));
        std::string tok;
        while (std::getline(ss, tok, ',')) if (not tok.empty()) ax.values.push_back(std::stof(tok));
        rc::RouteOptimizerConfig probe; float sm = 0.f, sp = 0.f;
        if (not apply_key(probe, sm, sp, ax.key, 0.f)) { std::printf("unknown key '%s'\n", ax.key.c_str()); return 2; }
        if (not ax.values.empty()) axes.push_back(std::move(ax));
    }

    World w;
    if (not load_world(path, w)) return 1;

    std::printf("world: %s | mission '%s' | %d laps | %zu waypoints (%zu repaired-set) | start (%.2f,%.2f)\n",
                path.c_str(), w.mission.c_str(), w.laps, w.wp_raw.size(), w.wp_safe.size(),
                w.start.x(), w.start.y());
    std::printf("grid : %d x %d cells @ %.3f m, %ld occupied (%.0f%%) | footprint inscribed %.3f circumscribed %.3f\n",
                w.planner.width(), w.planner.height(), w.planner.params.cell_size_m,
                w.planner.occupied_cells(),
                100.0 * w.planner.occupied_cells() / std::max(1, w.planner.width() * w.planner.height()),
                rc::RobotFootprint::shadow().inscribed_radius(),
                rc::RobotFootprint::shadow().circumscribed_radius());
    std::printf("fit  : spacing %.3f smoothing %.3f | v_max %.2f a_lat %.2f standoff %.2f\n",
                w.spacing, w.smoothing, w.v_max, w.a_lat, w.standoff);
    std::printf("opt  : d_target %.3f rho %.3f sigma_a %.3f floor %.3f | w_kappa %.2f w_clear %.2f "
                "w_gauge %.3f clear_peak %.2f kappa_peak %.2f huber %.2f iters %d safety_bias %.2f\n",
                w.opt.d_target, w.opt.rho, w.opt.sigma_a, w.opt.clearance_floor,
                w.opt.w_kappa, w.opt.w_clear, w.opt.w_gauge, w.opt.clear_peak, w.opt.kappa_peak,
                w.opt.anchor_huber, w.opt.iterations, w.opt.safety_bias);

    // The distance field is what the optimiser descends; realise it once so the timing below is the
    // solve, not the field.
    (void)w.planner.distance_at(w.start);

    print_header();
    std::vector<std::pair<std::string, Metrics>> results;

    // BASELINE — the route as it is driven today, optimiser off. Every other row is only meaningful
    // against this one.
    {
        auto m = evaluate(w, nullptr, w.smoothing, w.spacing);
        print_row("baseline (opt off)", m);
        results.emplace_back("baseline (opt off)", m);
    }

    if (not dump_path.empty())
    {
        rc::RouteOptimizerConfig o = w.opt;
        o.enabled = true; o.verbose = false;
        rc::RouteFollower route;
        rc::RouteOptimizerConfig ob = o;
        ob.distance = [&w](const Vector2f &p) { return w.planner.distance_at(p); };
        ob.distance_gradient = [&w](const Vector2f &p) { return w.planner.distance_gradient_at(p); };
        route.set_optimizer(ob);
        auto plan = [&w](const Vector2f &a, const Vector2f &b) { return w.planner.plan(a, b); };
        auto free_at = [&w](const Vector2f &p, float h) { return w.planner.pose_free(p, h); };
        if (route.build(w.start, w.wp_safe, w.laps, plan, free_at, w.spacing, w.smoothing))
            dump_profile(dump_path, w, route.path(), curvature_profile(route.path()));
    }

    // The cartesian product of whatever axes were given; with none, the snapshot's own settings.
    std::vector<std::size_t> idx(axes.size(), 0);
    bool more = true;
    while (more)
    {
        rc::RouteOptimizerConfig o = w.opt;
        o.enabled = true;
        o.verbose = false;
        float sm = w.smoothing, sp = w.spacing;
        // a_lat and v_max belong to the WORLD, not the optimiser: they set both the cornering speed the
        // route allows and (through rho = v_max^2/a_lat) how hard the optimiser works to avoid corners.
        // Changing one without the other would let the route be shaped for one speed budget and driven
        // under another — the two must not disagree.
        for (std::size_t ax = 0; ax < axes.size(); ++ax)
        {
            if (axes[ax].key == "a_lat") w.a_lat = axes[ax].values[idx[ax]];
            if (axes[ax].key == "v_max") w.v_max = axes[ax].values[idx[ax]];
        }
        o.rho = w.v_max * w.v_max / std::max(0.05f, w.a_lat);
        std::string label = axes.empty() ? "opt (snapshot weights)" : "";
        for (std::size_t a = 0; a < axes.size(); ++a)
        {
            apply_key(o, sm, sp, axes[a].key, axes[a].values[idx[a]]);
            char buf[64];
            std::snprintf(buf, sizeof buf, "%s%s=%g", label.empty() ? "" : " ",
                          axes[a].key.c_str(), axes[a].values[idx[a]]);
            label += buf;
        }
        auto m = evaluate(w, &o, sm, sp);
        print_row(label, m);
        results.emplace_back(label, m);

        more = false;
        for (std::size_t a = axes.size(); a-- > 0;)
        {
            if (++idx[a] < axes[a].values.size()) { more = true; break; }
            idx[a] = 0;
        }
        if (axes.empty()) break;
    }

    std::printf("\ndetail:\n");
    for (const auto &[label, m] : results) print_detail(label, m);
    std::printf("\nR_min is the tightest turn the route demands; the robot's inscribed radius is %.3f m.\n"
                "t_idl is the curve's own traversal time under v=sqrt(a_lat/kappa) — a property of the "
                "ROUTE, not a prediction of the run.\n",
                rc::RobotFootprint::shadow().inscribed_radius());
    return 0;
}
