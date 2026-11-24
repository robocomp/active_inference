// replay — re-run a countable agent's replay dump through the CURRENT engine (shared-engine plan revision 2, Task R1).
//
// The agent writes the dump with Model.ReplayDump = true (etc/replay/<node>_<run>_<serial>.bin, one per instance birth;
// format in src/countable_replay.h). This tool rebuilds the belief from the recorded seed (state + params), re-applies every
// recorded call (birth position covariance, room interior, updates, predicts, inflate_for_age, step-bound restores, front
// cues) and prints one CSV row per update. Columns use the ai2 log's names where the quantity is the same:
//   upd                 the update's index in the dump (0-based)
//   cycle,gated,npts    the ai2 log's cycle (processed_cycles), gated flag and camera point count of that update
//   lidar_only          the frame was LiDAR-only (a gated cycle's mask-attributed rays)
//   beam_staged         rays in the frame's beam set (the ai2 log's beam_staged)
//   cx..yaw             θ after the replayed update
//   std_cx..std_h       √ of the θθ diagonal; std_yaw = the marginal (door-mode) yaw std, std_yaw_within = √Σ(5,5)
//   gn_it,gn_conv,gn_lambda   the damped GN of the replayed update
//   omega_cx..omega_yaw the log-volatility ω (the ai2 log has omega_w and omega_yaw)
//   fe_upd              update()'s return value (NOT the ai2 log's `energy`, which is the mean mixture energy)
//   pstd_cx..pstd_yaw   √ of the PREDICTED θθ diagonal this update started from (Σ̄: the class prior on the first update)
//   dth_live            max |θ_replay − θ_live| (the live result the agent recorded; 0 ⇒ this engine reproduces the live run)
//   dP_live             max relative |diag P_replay − diag P_live| over the joint P (θ and every latent)
// Run: ./replay DUMP.bin [--from N] [--to N] [--allow-defaults] [--sig2-cam-measured] [--neighbour-size-std MANIFEST_DIR]
//   (N = update index; rows outside are computed, not printed). --sig2-cam-measured re-stages every frame's camera latent
//   marginal with today's rule (rc::persistent_latent_inputs: the measured ZED boundary std, F1.6) from the frame's yaw entry,
//   so a dump recorded before that change replays with it; dth_live / dP_live then measure the change, not a stream defect.
//   --neighbour-size-std DIR does the same for the neighbours' size std (F1.7: Δ_w/√12, Δ_h/√12 of the neighbour class's
//   manifest in DIR). The dump does not record a neighbour's class, so the TOOL infers it from the footprint: a box longer than
//   1.2 m is a bed, else a nightstand — true of every neighbour in objects_replay1 (room_3: one bed, its nightstands), not
//   general; the agent reads the class from the node name.
// Exit status: 0 = the whole dump replayed; 1 = a stream error (truncated / malformed), or a field the code has was absent
// from the dump (missing) or unreadable (bad) — the replay then ran with that field's DEFAULT, which is not the live run, so
// it fails unless --allow-defaults says that is intended (e.g. a dump older than a new params member).
// The last lines (stderr) are always the summary: the final θ and σ, the max σ per DOF over every event after the first update (predicts
// included),
// the total GN steps, and the max dth_live and dP_live over the whole dump.
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <set>
#include <string>
#include <string_view>
#include "countable_belief.h"
#include "countable_latent_inputs.h"
#include "countable_neighbour_size.h"
#include "countable_replay.h"

namespace
{
bool parse_int(std::string_view s, long& out)
{
    const auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), out);
    return ec == std::errc{} and p == s.data() + s.size();
}

void print_set(const char* what, const std::set<std::string>& s)
{
    if (s.empty()) return;
    std::fprintf(stderr, "replay: %s fields (%zu):", what, s.size());
    for (const auto& n : s) std::fprintf(stderr, " %s", n.c_str());
    std::fprintf(stderr, "\n");
}
}   // namespace

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "usage: %s DUMP.bin [--from N] [--to N] [--allow-defaults] [--sig2-cam-measured] [--neighbour-size-std DIR]\n", argv[0]);
        return 2;
    }
    long from = 0, to = std::numeric_limits<long>::max();
    bool allow_defaults = false, sig2_cam_measured = false;
    std::string neighbour_manifests;
    for (int i = 2; i < argc; ++i)
    {
        const std::string_view k = argv[i];
        if (k == "--allow-defaults") { allow_defaults = true; continue; }
        if (k == "--sig2-cam-measured") { sig2_cam_measured = true; continue; }
        if (k == "--neighbour-size-std" and i + 1 < argc) { neighbour_manifests = argv[++i]; continue; }
        long* dst = k == "--from" ? &from : k == "--to" ? &to : nullptr;
        if (dst == nullptr or i + 1 >= argc or not parse_int(argv[i + 1], *dst))
        {
            std::fprintf(stderr, "replay: bad argument %s\n", argv[i]);
            return 2;
        }
        ++i;
    }

    rc::replay::Reader reader;
    if (not reader.open(argv[1]))
    {
        std::fprintf(stderr, "replay: %s\n", reader.error().c_str());
        return 1;
    }
    std::printf("upd,cycle,gated,npts,lidar_only,beam_staged,cx,cy,H,w,h,yaw,std_cx,std_cy,std_H,std_w,std_h,std_yaw,"
                "std_yaw_within,gn_it,gn_conv,gn_lambda,omega_cx,omega_cy,omega_H,omega_w,omega_h,omega_yaw,fe_upd,"
                "pstd_cx,pstd_cy,pstd_H,pstd_w,pstd_h,pstd_yaw,dth_live,dP_live\n");

    std::string pending;   // the row of the last update, printed once its live result (if any) is known
    const auto flush = [&](double dth, double dP)
    {
        if (pending.empty()) return;
        std::printf("%s,%.9g,%.9g\n", pending.c_str(), dth, dP);
        pending.clear();
    };
    rc::CountableBelief belief;
    double worst_th = 0.0, worst_P = 0.0;
    long compared = 0, gn_total = 0;
    std::array<double, 6> sd_max{};
    long upd = -1;
    const auto track = [&](const rc::CountableBelief& b)
    {   // after the first update only: before it Σ is the birth prior, not anything the estimator concluded
        if (upd < 0) return;
        for (int i = 0; i < 6; ++i) sd_max[std::size_t(i)] = std::max(sd_max[std::size_t(i)], std::sqrt(std::max(0.0, double(b.covariance()(i, i)))));
    };
    const bool ok = rc::replay::play(reader, belief,
        [&](const rc::replay::UpdateEv& ev, const rc::CountableBelief& b, float fe)
        {
            const double nan = std::numeric_limits<double>::quiet_NaN();
            flush(nan, nan);   // the previous update had no live result recorded
            ++upd;
            if (upd < from or upd > to) return;
            const auto& s = b.state();
            const auto& S = b.covariance();
            const auto sd = [&](int i) { return std::sqrt(std::max(0.0f, S(i, i))); };
            const auto& om = b.log_volatility();
            const auto& Pb = b.dbg_Pbar_tt();
            const auto pd = [&](int i) { return std::sqrt(std::max(0.0f, Pb(i, i))); };
            gn_total += b.last_gn_iters();
            char row[1024];
            std::snprintf(row, sizeof row,
                          "%ld,%lld,%d,%zu,%d,%zu,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%d,%d,%.6g,"
                          "%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.9g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g",
                          upd, static_cast<long long>(ev.cycle), ev.gated ? 1 : 0, ev.frame.points.size(),
                          ev.frame.lidar_only ? 1 : 0, ev.frame.beam.endpoints.size(),
                          s.cx, s.cy, s.H, s.w, s.h, s.yaw, sd(0), sd(1), sd(2), sd(3), sd(4),
                          std::sqrt(std::max(0.0f, b.yaw_marginal_var())), sd(5),
                          b.last_gn_iters(), b.last_gn_converged() ? 1 : 0, b.last_lambda(),
                          om(0), om(1), om(2), om(3), om(4), om(5), fe, pd(0), pd(1), pd(2), pd(3), pd(4), pd(5));
            pending = row;
        },
        [&](const rc::replay::ResultEv& live, const rc::CountableBelief& b)
        {
            const auto& s = b.state();
            const double dth = std::max({std::abs(double(s.cx) - live.theta.cx), std::abs(double(s.cy) - live.theta.cy),
                                         std::abs(double(s.H) - live.theta.H), std::abs(double(s.w) - live.theta.w),
                                         std::abs(double(s.h) - live.theta.h), std::abs(double(s.yaw) - live.theta.yaw)});
            const auto& P = b.joint_covariance();
            double dP = live.p_diag.size() == std::size_t(P.rows()) ? 0.0 : std::numeric_limits<double>::infinity();
            for (int i = 0; std::isfinite(dP) and i < P.rows(); ++i)
            {
                const double a = P(i, i), l = live.p_diag[std::size_t(i)], m = std::max(std::abs(a), std::abs(l));
                if (m > 0.0) dP = std::max(dP, std::abs(a - l) / m);
            }
            worst_th = std::max(worst_th, dth);
            worst_P  = std::max(worst_P, dP);
            ++compared;
            flush(dth, dP);
        },
        track,
        [&](rc::CountableFrame& f)
        {
            if (sig2_cam_measured)
            {
                const float yaw_std = std::sqrt(std::max(0.0f, f.sig2_cam[5]));
                f.sig2_cam = rc::persistent_latent_inputs({0.0f, 0.0f, yaw_std}, rc::LatentView{}).sig2_cam;
            }
            if (not neighbour_manifests.empty())
                for (auto& nb : f.foreign_boxes)
                {
                    const std::string cls = std::max(nb.w, nb.h) > 1.2f ? "bed" : "nightstand";
                    if (const auto sd = rc::neighbour_size_std(neighbour_manifests + "/" + cls + ".concept.toml"))
                    { nb.std(3) = (*sd)[0]; nb.std(4) = (*sd)[1]; }
                }
        });
    flush(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());

    print_set("missing (default kept)", reader.diag().missing);
    print_set("unknown (skipped)", reader.diag().unknown);
    print_set("BAD (size mismatch, default kept)", reader.diag().bad);
    if (reader.skipped_events() > 0) std::fprintf(stderr, "replay: %d events of unknown type skipped\n", reader.skipped_events());
    if (not ok)
        std::fprintf(stderr, "replay: stream error: %s\n", reader.error().empty() ? "an event before the seed" : reader.error().c_str());
    {
        const auto& s = belief.state();
        const auto& S = belief.covariance();
        const auto sd = [&](int i) { return std::sqrt(std::max(0.0f, S(i, i))); };
        std::fprintf(stderr, "replay: final theta cx %.6f cy %.6f H %.6f w %.6f h %.6f yaw %.6f\n", s.cx, s.cy, s.H, s.w, s.h, s.yaw);
        std::fprintf(stderr, "replay: final std   cx %.6f cy %.6f H %.6f w %.6f h %.6f yaw %.6f\n", sd(0), sd(1), sd(2), sd(3), sd(4), sd(5));
        std::fprintf(stderr, "replay: max std     cx %.6f cy %.6f H %.6f w %.6f h %.6f yaw %.6f (every event after the first update)\n", sd_max[0], sd_max[1],
                     sd_max[2], sd_max[3], sd_max[4], sd_max[5]);
        std::fprintf(stderr, "replay: gn steps %ld\n", gn_total);
    }
    const bool defaults = not reader.diag().missing.empty() or not reader.diag().bad.empty();
    if (defaults and not allow_defaults)
        std::fprintf(stderr, "replay: FAIL — fields ran at their defaults (missing/bad above); pass --allow-defaults if intended\n");
    std::fprintf(stderr, "replay: %ld updates; against the %ld live results: max dth_live %.3g, max dP_live %.3g%s\n", upd + 1,
                 compared, worst_th, worst_P, compared > 0 and worst_th == 0.0 and worst_P == 0.0 ? " (bit-identical)" : "");
    return (not ok or (defaults and not allow_defaults)) ? 1 : 0;
}
