/*
 * mppi_bench.cpp — replay ONE recorded control cycle offline, through the real TrajectoryController.
 *
 * WHY THIS EXISTS. Every module in this agent that carries a self_test() — route_spline, route_follower,
 * route_optimizer, grid_planner — was iterated on offline in milliseconds. The 6000 lines that do not
 * (trajectory_controller, controller_session, obstacle_tracker) are exactly where the regressions
 * happened, and each one cost a lap, a revert, and in one case driving into furniture. The asymmetry is
 * not a coincidence: a cost term that can only be evaluated by driving will be evaluated by driving.
 *
 * The specific question this answers, which nothing else could: "if I change a cost term, does the
 * command change, and do rollouts that touch obstacles become admissible?" Tonight that was answered by
 * deleting a term, driving, and watching the robot hit things.
 *
 * FIDELITY. The snapshot records what compute() CONSUMES — pose, the cloud it used, the obstacle and wall
 * points, the path, the parameters — and the replay hands them back to the SAME compute(). The ESDF is
 * deliberately NOT recorded: producer and replayer are the same build_esdf on the same inputs, so the
 * inputs are the smaller and more honest artefact. (route_bench records the RASTER instead, because
 * there the two rasterisers could disagree. Same principle, opposite conclusion.)
 *
 * DETERMINISM. The sampler is seeded (TrajectoryController::set_seed), so a replay is repeatable and a
 * comparison measures the change rather than the sampling noise. The live agent seeds from
 * std::random_device and is NOT repeatable — which is correct for a robot and useless for an experiment.
 *
 *   bin/mppi_bench <snapshot> [key=value,value,... ...] [--seeds N]
 *
 * Keys sweep as a cartesian product, e.g.
 *   bin/mppi_bench mppi_cycle.txt lambda_lateral_bumper=0,18 lambda_goal=5,20
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "../src/trajectory_controller.h"

using Eigen::Vector2f;
using Eigen::Vector3f;

namespace
{

struct Row
{
    std::string label;
    float adv = 0.f, rot = 0.f;
    float ess = 0.f; int K = 0;
    float lambda_used = 0.f, cost_range = 0.f;
    float g_goal = 0.f, g_obs = 0.f, g_lat = 0.f, g_cbf = 0.f, g_vel = 0.f, g_smooth = 0.f;
    int   n_collisions = 0;
    float min_esdf = 0.f, p_free = 0.f;
    // What the COMMAND would actually do: the closest the body comes to an obstacle if the returned
    // command is held open-loop for the horizon. This is the number a cost change must not quietly ruin,
    // and it is the one that was missing when a term was deleted and the robot started hitting things.
    float cmd_clearance = 1e9f;
    float esdf_on_plan = 0.f, body_extent = 0.f;
};

// The closest the BODY comes to an obstacle along the plan the controller actually returned.
//
// ★FIRST VERSION OF THIS WAS WRONG and is worth recording: it held the FIRST command constant for the
// whole horizon. At rot = 0.55 rad/s that is 159 degrees of rotation over 5 s — a trajectory the
// controller never intended, since it re-solves every 100 ms. It reported the body 0.317 m INSIDE an
// obstacle on a cycle where not one rollout was infeasible, which is not a finding about the controller
// but about the measurement. Instruments that mix a measurement with an assumption about the thing
// measured cannot be read as evidence; that lesson is written down twice already in this project.
//
// The honest quantity is the weighted-average trajectory the solver exports — the plan it actually
// chose — transformed back into the robot frame the ESDF lives in, with the body's real reach at each
// point rather than a disc.
// Returns {near-horizon body clearance, near-horizon raw ESDF} along the plan.
//
// ★NEAR, not the whole horizon — the third correction this one metric needed, and the most instructive.
// The controller GRADES its horizon on purpose: a collision within hard_collision_horizon_s (1.2 s)
// makes a rollout infeasible, beyond that it is only a soft penalty, because the plan is re-solved every
// 100 ms and the tail is a prediction rather than a commitment. Scoring the whole 5 s reported the plan
// 0.317 m inside an obstacle on every configuration — invariant under a 70x change in temperature, which
// is the signature of measuring something the mechanism does not control. The tail always dips; that is
// what a tail is for. The commitment is the first 1.2 s, so that is what this measures.
std::pair<float, float> command_clearance(rc::TrajectoryController &tc,
                                          const std::vector<Vector2f> &avg_traj_room,
                                          const Eigen::Affine2f &pose)
{
    if (avg_traj_room.empty()) return {1e9f, 1e9f};
    const Eigen::Affine2f to_robot = pose.inverse();
    const float body = tc.body_extent_here();
    const int n_near = std::max(1, static_cast<int>(tc.params.hard_collision_horizon_s
                                                  / std::max(1e-3f, tc.params.trajectory_dt)));
    float worst_esdf = 1e9f;
    int i = 0;
    for (const auto &p_room : avg_traj_room)
    {
        if (i++ > n_near) break;
        const Vector2f p = to_robot * p_room;
        worst_esdf = std::min(worst_esdf, tc.clearance_at(p.x(), p.y()));
    }
    return {worst_esdf - body, worst_esdf};
}

struct Axis { std::string key; std::vector<float> values; };

bool apply_key(rc::TrajectoryController::Params &p, const std::string &k, float v)
{
    if      (k == "lambda_obstacle")           p.lambda_obstacle = v;
    else if (k == "lambda_goal")               p.lambda_goal = v;
    else if (k == "lambda_lateral_clearance")  p.lambda_lateral_clearance = v;
    else if (k == "lambda_lateral_bumper")     p.lambda_lateral_bumper = v;
    else if (k == "lateral_corner_bias_gain")  p.lateral_corner_bias_gain = v;
    else if (k == "lateral_balance_gain")      p.lateral_balance_gain = v;
    else if (k == "lambda_cbf")                p.lambda_cbf = v;
    else if (k == "lambda_velocity")           p.lambda_velocity = v;
    else if (k == "lambda_smooth")             p.lambda_smooth = v;
    else if (k == "mppi_lambda")               p.mppi_lambda = v;
    else if (k == "sigma_rot")                 p.sigma_rot = v;
    else if (k == "lambda_fixed")              p.lambda_fixed = v != 0.f;
    else if (k == "sigma_adv")                 p.sigma_adv = v;
    else if (k == "lambda_max")                p.lambda_max = v;
    else if (k == "warm_start_rot_weight")     p.warm_start_rot_weight = v;
    else if (k == "d_safe")                    p.d_safe = v;
    else if (k == "max_adv")                   p.max_adv = v;
    else if (k == "num_samples")               p.num_samples = static_cast<int>(v);
    else if (k == "trajectory_steps")          p.trajectory_steps = static_cast<int>(v);
    else if (k == "trajectory_dt")             p.trajectory_dt = v;
    else if (k == "rot_cost_factor")           p.rot_cost_factor = v;
    else if (k == "obstacle_cost_cap")         p.obstacle_cost_cap = v;
    else if (k == "close_obstacle_margin")     p.close_obstacle_margin = v;
    else if (k == "hard_collision_horizon_s")  p.hard_collision_horizon_s = v;
    else if (k == "cbf_max_decel")             p.cbf_max_decel = v;
    else if (k == "stop_pivot_seeds")          p.enable_stop_pivot_seeds = v != 0.f;
    else return false;
    return true;
}

// ── CLOSED-LOOP REPLAY ───────────────────────────────────────────────────────────────────────────
// Run N successive cycles from one snapshot: solve, execute the returned command for dt, move the robot,
// re-express the world in the new robot frame, solve again — with the SAME controller instance, so the
// warm start, the adaptive temperature and the last-command continuity all carry across exactly as they
// do on the robot.
//
// WHY THIS AND NOT A SINGLE CYCLE. A weave is a property of a SEQUENCE of decisions; a single-cycle
// replay cannot show it, however many seeds it averages. Here the world is FROZEN — the same cloud, no
// new sensing, no localisation noise, no moving obstacles — so anything oscillatory that appears is
// generated by the controller alone. That is exactly the question: can successive MPPI solves on an
// unchanging straight-ahead problem produce a 0.7 Hz rotation oscillation?
// ★What it cannot tell us: whether the REAL weave has the same cause. If the oscillation does not
// reproduce here, the cause needs the changing world (new returns each cycle, pose jitter) and this
// harness has ruled the controller out on its own — which is worth just as much.
struct Step { float t, adv, rot, x, y, theta, p_free, ess, lambda, min_esdf, best_rot; int ncol, best_idx; };

std::vector<Step> closed_loop(const std::string &snapshot, int n_cycles, std::uint32_t seed,
                              const std::vector<Axis> &axes, const std::vector<std::size_t> &idx)
{
    std::vector<Step> out;
    std::ifstream f(snapshot);
    rc::TrajectoryController tc;
    Eigen::Affine2f pose;
    std::vector<Vector3f> lidar_robot;
    if (not f.is_open() or not tc.load_snapshot(f, pose, lidar_robot)) return out;
    for (std::size_t a = 0; a < idx.size(); ++a) apply_key(tc.params, axes[a].key, axes[a].values[idx[a]]);
    const auto path_copy = tc.get_path();
    tc.set_path_presmoothed(path_copy);
    tc.set_seed(seed);

    // The cloud was recorded in the ROBOT frame of the snapshot's pose. The world does not move, so lift
    // it to the room once and push it back down into whatever frame the robot occupies each cycle.
    std::vector<Vector2f> lidar_room;
    lidar_room.reserve(lidar_robot.size());
    for (const auto &p : lidar_robot) lidar_room.push_back(pose * Vector2f(p.x(), p.y()));
    const std::vector<float> z(lidar_robot.size(), 0.3f);

    const float dt = tc.params.trajectory_dt;
    for (int c = 0; c < n_cycles; ++c)
    {
        const Eigen::Affine2f to_robot = pose.inverse();
        std::vector<Vector3f> cloud;
        cloud.reserve(lidar_room.size());
        for (std::size_t i = 0; i < lidar_room.size(); ++i)
        {
            const Vector2f q = to_robot * lidar_room[i];
            cloud.emplace_back(q.x(), q.y(), z[i]);
        }
        const auto o = tc.compute(pose, cloud);
        const float th = Eigen::Rotation2Df(pose.linear()).angle();
        out.push_back({c * dt, o.adv, o.rot, pose.translation().x(), pose.translation().y(), th,
                       o.p_free, o.ess_K ? o.ess / o.ess_K : 0.f, o.lambda_used, o.min_esdf,
                       o.best_seed_rot, o.n_collisions, o.best_seed_idx});
        // Execute for one control period. Robot forward is +y in its own frame.
        const float nth = th + o.rot * dt;
        pose = Eigen::Translation2f(pose.translation().x() - o.adv * dt * std::sin(th),
                                    pose.translation().y() + o.adv * dt * std::cos(th))
             * Eigen::Rotation2Df(nth);
    }
    return out;
}

void print_header()
{
    std::printf("\n%-34s %6s %6s %6s %7s %7s %8s %8s %8s %6s %7s %7s %7s\n",
                "config", "adv", "rot", "ESS/K", "lambda", "range", "g_goal", "g_obs", "g_lat",
                "ncol", "p_free", "cmd_clr", "esdf_pl");
    std::printf("%s\n", std::string(126, '-').c_str());
}

void print_row(const Row &r)
{
    std::printf("%-34s %6.3f %6.3f %6.3f %7.1f %7.1f %8.2f %8.2f %8.2f %6d %7.3f %7.3f %7.3f\n",
                r.label.c_str(), r.adv, r.rot, r.K ? r.ess / r.K : 0.f, r.lambda_used, r.cost_range,
                r.g_goal, r.g_obs, r.g_lat, r.n_collisions, r.p_free, r.cmd_clearance, r.esdf_on_plan);
}

}  // namespace

int main(int argc, char **argv)
{
    std::string path;
    std::vector<Axis> axes;
    int n_seeds = 1;
    bool show_plan = false;
    int n_loop = 0;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "--seeds" and i + 1 < argc) { n_seeds = std::max(1, std::atoi(argv[++i])); continue; }
        if (a == "--plan") { show_plan = true; continue; }
        if (a == "--loop" and i + 1 < argc) { n_loop = std::max(2, std::atoi(argv[++i])); continue; }
        const auto eq = a.find('=');
        if (eq == std::string::npos) { path = a; continue; }
        Axis ax{a.substr(0, eq), {}};
        std::stringstream ss(a.substr(eq + 1));
        std::string tok;
        while (std::getline(ss, tok, ',')) if (not tok.empty()) ax.values.push_back(std::stof(tok));
        rc::TrajectoryController::Params probe;
        if (not apply_key(probe, ax.key, 0.f)) { std::printf("unknown key '%s'\n", ax.key.c_str()); return 2; }
        if (not ax.values.empty()) axes.push_back(std::move(ax));
    }
    if (path.empty()) { std::printf("usage: mppi_bench <snapshot> [key=v,v ...] [--seeds N] [--plan]\n"); return 2; }

    // One controller per evaluation: the class carries cycle-to-cycle state (warm start, adaptive K/T/λ,
    // last command) and reusing it would make each row depend on the rows before it.
    const auto evaluate = [&](const std::string &label, const Axis *ax_set, const std::vector<std::size_t> &idx,
                              std::uint32_t seed) -> Row
    {
        std::ifstream f(path);
        rc::TrajectoryController tc;
        Eigen::Affine2f pose;
        std::vector<Vector3f> lidar;
        Row r; r.label = label;
        if (not f.is_open() or not tc.load_snapshot(f, pose, lidar)) { r.label += " [LOAD FAILED]"; return r; }
        for (std::size_t a = 0; a < idx.size(); ++a) apply_key(tc.params, ax_set[a].key, ax_set[a].values[idx[a]]);
        // ★Re-install the path AFTER the keys. adaptive_K_/adaptive_T_/adaptive_lambda_ are initialised in
        // reset_mppi_state(), which runs when the path is set — inside load_snapshot, i.e. before these
        // keys existed. Without this, a K sweep silently does nothing and reports a perfect null result:
        // K=100/300/1000 gave output identical to three decimals, which is what tipped me off.
        // Copy first: passing the controller's own path back into a setter that clears it is aliasing.
        const auto path_copy = tc.get_path();
        tc.set_path_presmoothed(path_copy);
        tc.set_seed(seed);
        const auto out = tc.compute(pose, lidar);
        r.adv = out.adv; r.rot = out.rot;
        r.ess = out.ess; r.K = out.ess_K;
        r.lambda_used = out.lambda_used; r.cost_range = out.cost_range;
        r.g_goal = out.g_goal; r.g_obs = out.g_obs; r.g_lat = out.g_lat;
        r.g_cbf = out.g_cbf; r.g_vel = out.g_vel; r.g_smooth = out.g_smooth;
        r.n_collisions = out.n_collisions; r.min_esdf = out.min_esdf; r.p_free = out.p_free;
        const auto [clr, raw] = command_clearance(tc, out.average_trajectory_room, pose);
        r.cmd_clearance = clr; r.esdf_on_plan = raw; r.body_extent = tc.body_extent_here();
        return r;
    };

    std::printf("snapshot: %s   (%d seed%s per config)\n", path.c_str(), n_seeds, n_seeds > 1 ? "s" : "");

    if (n_loop > 0)
    {
        std::vector<std::size_t> lidx(axes.size(), 0);
        bool lmore = true;
        while (lmore)
        {
            std::string label = axes.empty() ? "as recorded" : "";
            for (std::size_t a = 0; a < axes.size(); ++a)
            {
                char buf[64];
                std::snprintf(buf, sizeof buf, "%s%s=%g", label.empty() ? "" : " ",
                              axes[a].key.c_str(), axes[a].values[lidx[a]]);
                label += buf;
            }
            const auto tr = closed_loop(path, n_loop, 1234u, axes, lidx);
            if (tr.empty()) { std::printf("  %s: closed loop failed to start\n", label.c_str()); break; }
            // Reversals with the SAME deadband the run metric uses, so the two numbers are comparable.
            constexpr float kDB = 0.05f;
            int rev = 0, sign = 0;
            float sum_abs = 0.f, dist = 0.f;
            std::vector<float> flips;
            for (std::size_t i = 0; i < tr.size(); ++i)
            {
                const int c = tr[i].rot > kDB ? 1 : (tr[i].rot < -kDB ? -1 : 0);
                if (c) { if (sign and c != sign) { ++rev; flips.push_back(tr[i].t); } sign = c; }
                sum_abs += std::abs(tr[i].rot);
                if (i) dist += std::hypot(tr[i].x - tr[i-1].x, tr[i].y - tr[i-1].y);
            }
            float mean_gap = 0.f;
            for (std::size_t i = 1; i < flips.size(); ++i) mean_gap += flips[i] - flips[i-1];
            if (flips.size() > 1) mean_gap /= (flips.size() - 1);
            std::printf("\n  CLOSED LOOP  %s : %d cycles (%.1f s simulated, %.2f m driven)\n",
                        label.c_str(), n_loop, tr.back().t, dist);
            std::printf("    reversals %d   mean |rot| %.3f   mean gap between flips %.2f s%s\n",
                        rev, sum_abs / tr.size(), mean_gap,
                        mean_gap > 1e-3f ? "" : "  (fewer than two flips)");
            if (mean_gap > 1e-3f)
                std::printf("    -> implied oscillation %.2f Hz   (the lap's weave is 0.69 Hz)\n",
                            1.f / (2.f * mean_gap));
            if (axes.empty())
            {
                std::printf("    %5s %7s %7s %8s %8s %7s %7s\n",
                            "t", "adv", "cmd_rot", "best_idx", "best_rot", "ESS/K", "lambda");
                int jumps = 0;
                for (std::size_t i = 1; i < tr.size(); ++i)
                    if (tr[i].best_idx != tr[i-1].best_idx) ++jumps;
                for (std::size_t i = 0; i < tr.size(); ++i)
                    if (i < 4 or (i % 4) == 0)
                        std::printf("    %5.1f %7.3f %+7.3f %8d %+8.3f %7.3f %7.1f\n",
                                    tr[i].t, tr[i].adv, tr[i].rot, tr[i].best_idx, tr[i].best_rot,
                                    tr[i].ess, tr[i].lambda);
                std::printf("    winning seed changed on %d of %zu cycles\n", jumps, tr.size() - 1);
            }
            else
            {
                float mean_adv = 0.f, mean_ess = 0.f, mean_lam = 0.f;
                for (const auto &st : tr) { mean_adv += st.adv; mean_ess += st.ess; mean_lam += st.lambda; }
                std::printf("    mean adv %.3f   mean ESS/K %.3f   mean lambda %.0f   driven %.2f m\n",
                            mean_adv / tr.size(), mean_ess / tr.size(), mean_lam / tr.size(), dist);
            }
            lmore = false;
            for (std::size_t a = axes.size(); a-- > 0;)
            {
                if (++lidx[a] < axes[a].values.size()) { lmore = true; break; }
                lidx[a] = 0;
            }
            if (axes.empty()) break;
        }
        return 0;
    }
    if (show_plan)
    {
        // The plan, point by point, in the ESDF's own frame. An aggregate that does not move when the
        // temperature changes 70x is not measuring what its name says — the way to find out is to look
        // at the samples rather than the minimum over them.
        std::ifstream f(path);
        rc::TrajectoryController tc;
        Eigen::Affine2f pose; std::vector<Vector3f> lidar;
        if (f.is_open() and tc.load_snapshot(f, pose, lidar))
        {
            tc.set_seed(1234u);
            const auto out = tc.compute(pose, lidar);
            const Eigen::Affine2f to_robot = pose.inverse();
            std::printf("\nplan (robot frame): body_extent_here %.3f\n", tc.body_extent_here());
            std::printf("%4s %8s %8s %9s\n", "i", "x", "y", "esdf");
            int i = 0;
            for (const auto &p_room : out.average_trajectory_room)
            {
                const Vector2f q = to_robot * p_room;
                if (i < 8 or (i % 8) == 0)
                    std::printf("%4d %8.3f %8.3f %9.4f\n", i, q.x(), q.y(), tc.clearance_at(q.x(), q.y()));
                ++i;
            }
        }
    }
    print_header();

    std::vector<std::size_t> idx(axes.size(), 0);
    bool more = true;
    while (more)
    {
        std::string label = axes.empty() ? "as recorded" : "";
        for (std::size_t a = 0; a < axes.size(); ++a)
        {
            char buf[64];
            std::snprintf(buf, sizeof buf, "%s%s=%g", label.empty() ? "" : " ",
                          axes[a].key.c_str(), axes[a].values[idx[a]]);
            label += buf;
        }
        // Several seeds per config, because one sample of a stochastic optimiser is not a measurement of
        // it. The spread across seeds is the resolution limit of every comparison below.
        std::vector<Row> rows;
        for (int s = 0; s < n_seeds; ++s) rows.push_back(evaluate(label, axes.data(), idx, 1234u + s));
        Row mean = rows.front();
        if (n_seeds > 1)
        {
            const auto avg = [&](auto get) {
                float t = 0.f; for (const auto &r : rows) t += get(r); return t / rows.size(); };
            mean.adv = avg([](const Row &r){ return r.adv; });
            mean.rot = avg([](const Row &r){ return r.rot; });
            mean.ess = avg([](const Row &r){ return r.ess; });
            mean.lambda_used = avg([](const Row &r){ return r.lambda_used; });
            mean.cost_range = avg([](const Row &r){ return r.cost_range; });
            mean.g_goal = avg([](const Row &r){ return r.g_goal; });
            mean.g_obs = avg([](const Row &r){ return r.g_obs; });
            mean.g_lat = avg([](const Row &r){ return r.g_lat; });
            mean.n_collisions = static_cast<int>(avg([](const Row &r){ return float(r.n_collisions); }));
            mean.cmd_clearance = avg([](const Row &r){ return r.cmd_clearance; });
        }
        print_row(mean);
        if (n_seeds > 1)
        {
            // ★THE DITHER MEASUREMENT. Every seed sees the SAME world, the same path and the same costs;
            // the only difference is the sampling noise. So the spread of the commanded rotation across
            // seeds is precisely the part of the command that is Monte-Carlo noise rather than decision.
            // If sd(rot) is comparable to the rotation magnitude itself, the controller is dithering and
            // the sign flips need no obstacle, no mode, and no geometry to explain them.
            const auto sd = [&](auto get) {
                double m = 0; for (const auto &r : rows) m += get(r); m /= rows.size();
                double v = 0; for (const auto &r : rows) v += (get(r) - m) * (get(r) - m);
                return std::sqrt(v / std::max<std::size_t>(1, rows.size() - 1)); };
            const double sd_rot = sd([](const Row &r){ return double(r.rot); });
            const double sd_adv = sd([](const Row &r){ return double(r.adv); });
            int sign_flips = 0;
            for (std::size_t i = 1; i < rows.size(); ++i)
                if ((rows[i].rot > 0) != (rows[i-1].rot > 0)) ++sign_flips;
            float lo = 1e9f, hi = -1e9f;
            for (const auto &r : rows) { lo = std::min(lo, r.cmd_clearance); hi = std::max(hi, r.cmd_clearance); }
            std::printf("%-34s   sd(rot) %.4f  sd(adv) %.4f  |rot|/sd %.1f  sign-disagreements %d/%d"
                        "  cmd_clr [%.3f, %.3f]\n", "", sd_rot, sd_adv,
                        sd_rot > 1e-9 ? std::abs(mean.rot) / sd_rot : 999.0, sign_flips,
                        int(rows.size()) - 1, lo, hi);
        }

        more = false;
        for (std::size_t a = axes.size(); a-- > 0;)
        {
            if (++idx[a] < axes[a].values.size()) { more = true; break; }
            idx[a] = 0;
        }
        if (axes.empty()) break;
    }

    std::printf("\ncmd_clr is the closest the BODY comes to an obstacle along the plan the solver actually\n"
                "returned (its weighted-average trajectory) — negative means contact. It is the number a\n"
                "cost change must not quietly ruin, and the one that was missing when a term was deleted\n"
                "on a hunch.\n");
    return 0;
}
