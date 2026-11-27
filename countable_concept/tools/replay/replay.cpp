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
//   wall_in             1 when this update entered the wall latent's priors (a new wall edge for the instance, F1.4)
//   back_gap            the back face's gap to the frame's STAGED wall, (back_centre − wall.p)·n (< 0: through it; NaN: no wall)
//   gn_cap              1 when the update's GN ran into its iteration cap without converging
//   range_m             the camera's (else the robot's) distance to θ's centre at this update (for NEES by range bin)
//   jump_p              P(moved | data) of the update's volatility statistic (NaN: none computed)
//   q,std_q             the class proportion's posterior mean and σ after the update (K2a-L; NaN σ: the class declares none)
//   pstd_cx..pstd_yaw   √ of the PREDICTED θθ diagonal this update started from (Σ̄: the class prior on the first update)
//   dth_live            max |θ_replay − θ_live| (the live result the agent recorded; 0 ⇒ this engine reproduces the live run)
//   dP_live             max relative |diag P_replay − diag P_live| over the joint P (θ and every latent)
// Run: ./replay DUMP.bin [--from N] [--to N] [--allow-defaults] [--sig2-cam-measured] [--neighbour-size-std MANIFEST_DIR]
//              [--standoff MANIFEST_DIR] [--permanence MANIFEST_DIR] [--parts MANIFEST_DIR]
//   (N = update index; rows outside are computed, not printed). --sig2-cam-measured re-stages every frame's camera latent
//   marginal with today's rule (rc::persistent_latent_inputs: the measured ZED boundary std, F1.6) from the frame's yaw entry,
//   so a dump recorded before that change replays with it; dth_live / dP_live then measure the change, not a stream defect.
//   --neighbour-size-std DIR does the same for the neighbours' size std (F1.7: Δ_w/√12, Δ_h/√12 of the neighbour class's
//   manifest in DIR), the class read from the recorded ForeignBox::cls (the neighbour's node name without its serial, F1 round
//   6.7). A dump recorded before cls existed carries none: those neighbours are left as recorded and counted (stderr).
//   --standoff DIR sets the seed's wall stand-off (F1.4) from the manifest of the dumped node's class in DIR
//   (rc::manifest::wall_standoff), as a dump recorded before the stand-off existed lacks it. --permanence DIR does the same for
//   the class permanence prior (F1 round 6; predict_period_s then keeps its default 0.1 s, the configs' AI2AgeNominalDtS).
//   --placement DIR sets the seed's placement prior P(against a wall) (CB1) from the manifest of the dumped node's class in DIR
//   ([prior.placement] p_against_wall; −1 when the class declares none), as a dump recorded before CB1 lacks it.
//   --rebirth FILE (CB1, COUNTERFACTUAL) replaces the recorded seed state of this dump with the one FILE gives for its stem (written
//   by tools/replay/birth_placement: the birth re-derived from the dump's first camera frame with the view-conditional placement),
//   and sets that birth's position covariance (the p_wall column, the birth's placement posterior, is informational: the mode prior
//   uses the class p_against_wall, --placement). One line per dump: stem changed p_wall cx cy H w h yaw c00 c01 c11 (decimal points, from_chars); the state and covariance are used
//   only when changed = 1 (a FREE -> WALL change: the recorded stream then holds no birth covariance to supersede). A dump not in
//   FILE is as recorded.
//   --parts DIR sets the seed's solid (K2a: params.shape_parts and params.solid_parts, common/ai_belief/parts_geometry.h) from the
//   manifest of the dumped node's class in DIR ([[shape.part]] and [shape] solid), as a dump recorded before K2a lacks it (and ran
//   the one-box solid). The dumped FRAMES are unchanged: what the live agent staged with its own θ trajectory (see the K2a report).
//   --seed-h-rule re-derives the seed's H with the birth rule (CountableBelief::seed_height, F1 round 6.2) from the dump's FIRST
//   camera frame (the birth cloud itself is not recorded), at the recorded born pose.
//   --f1-restage GAP re-stages every frame's rho_cam / rho_L with task-F1's rule (rc::ai::view: the viewpoints' SEPARATION from each
//   latent's anchor beyond its paid frontier, the kernels unchanged) for a dump recorded BEFORE F1, whose frames carry the old arc-length
//   rho and no viewpoints. It mirrors CountableFitter::stage_persistent / consume_paths from what the dump does record: the odometry
//   displacement since the last accepted update = the frame's chain_dr (an approximation: chain_dr also carries the pose increment over
//   a stretch the odometry could not integrate, which the agent's own F1 viewpoint skips); the camera position = camera_xy; the LiDAR
//   origin = the staged beam's origin (on a frame without beam rays, the last one shifted with the camera); GAP = the Helios max ring
//   gap the run logged ("[lidar-measures] Helios SensorModel: max ring gap"). A rejected update (RestoreEv) commits no anchor. The
//   result is a COUNTERFACTUAL replay — not the live run — so dth_live / dP_live then measure the change. Off: frames as recorded.
//   --theta-cov appends, after dP_live, the 21 upper-triangle entries S_i_j (i <= j, i, j = 0..5 in the order cx cy H w h yaw) of the
//   FULL 6x6 theta covariance the update left (b.covariance(); the yaw entry is the within-mode Sigma(5,5)), %.9g. Off by default: the
//   default output is byte-identical to before. --wall-geom (CW1) appends, after dP_live and before the covariance, corner_gap = the
//   smallest gap of the footprint's four corners to the frame's STAGED wall line, min_k (p_k − wall.p)·n (< 0: a corner past it;
//   NaN: no wall staged), corner_gap_true = the same against the TRUE wall (the staged line moved by the wall latent's mean δ̂ along n,
//   where the engine evaluates the containment), and room_gap = the smallest signed distance of the corners to the frame's room polygon (< 0: a corner
//   outside the room; NaN: the frame carries no polygon, e.g. a dump recorded before CW1). corner_gap and corner_gap_true are against the
//   staged wall's LINE even when a polygon is present (room_gap is the polygon's). Off by default. The full block is NOT in the dump (the live result stores only diag P), so replaying
//   is the only way to recover it (first_visit_nees.py, AuRo test (a)).
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
#include <fstream>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include "countable_belief.h"
#include "countable_latent_inputs.h"
#include "countable_neighbour_size.h"
#include "../../common/concept_manifest/wall_standoff.h"
#include "../../common/concept_manifest/permanence.h"
#include "../../common/concept_manifest/shape_parts.h"
#include "countable_replay.h"
#include "countable_containment.h"

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
        std::fprintf(stderr, "usage: %s DUMP.bin [--from N] [--to N] [--allow-defaults] [--sig2-cam-measured] [--neighbour-size-std DIR] [--standoff DIR] [--placement DIR] [--rebirth FILE] [--permanence DIR] [--parts DIR] [--seed-h-rule] [--theta-cov] [--wall-geom] [--f1-restage GAP]\n", argv[0]);
        return 2;
    }
    long from = 0, to = std::numeric_limits<long>::max();
    bool allow_defaults = false, sig2_cam_measured = false;
    std::string neighbour_manifests, standoff_manifests, permanence_manifests, parts_manifests, placement_manifests, rebirth_file;
    bool seed_h_rule = false, theta_cov = false, wall_geom = false;
    bool f1_restage = false;
    float f1_gap = 0.0f;
    long neighbours_without_class = 0;
    for (int i = 2; i < argc; ++i)
    {
        const std::string_view k = argv[i];
        if (k == "--allow-defaults") { allow_defaults = true; continue; }
        if (k == "--sig2-cam-measured") { sig2_cam_measured = true; continue; }
        if (k == "--neighbour-size-std" and i + 1 < argc) { neighbour_manifests = argv[++i]; continue; }
        if (k == "--standoff" and i + 1 < argc) { standoff_manifests = argv[++i]; continue; }
        if (k == "--placement" and i + 1 < argc) { placement_manifests = argv[++i]; continue; }
        if (k == "--rebirth" and i + 1 < argc) { rebirth_file = argv[++i]; continue; }
        if (k == "--permanence" and i + 1 < argc) { permanence_manifests = argv[++i]; continue; }
        if (k == "--parts" and i + 1 < argc) { parts_manifests = argv[++i]; continue; }
        if (k == "--seed-h-rule") { seed_h_rule = true; continue; }
        if (k == "--theta-cov") { theta_cov = true; continue; }
        if (k == "--wall-geom") { wall_geom = true; continue; }
        if (k == "--f1-restage" and i + 1 < argc)
        {
            const std::string_view a = argv[++i];
            const auto [p, ec] = std::from_chars(a.data(), a.data() + a.size(), f1_gap);
            if (ec != std::errc{} or p != a.data() + a.size() or not (f1_gap > 0.0f))
            {
                std::fprintf(stderr, "replay: bad --f1-restage %s (the Helios max ring gap, rad, > 0)\n", argv[i]);
                return 2;
            }
            f1_restage = true;
            continue;
        }
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
                "pstd_cx,pstd_cy,pstd_H,pstd_w,pstd_h,pstd_yaw,wall_in,back_gap,gn_cap,range_m,jump_p,q,std_q,dth_live,dP_live");
    if (wall_geom) std::printf(",corner_gap,corner_gap_true,room_gap");
    if (theta_cov)
        for (int i = 0; i < 6; ++i)
            for (int j = i; j < 6; ++j) std::printf(",S_%d_%d", i, j);
    std::printf("\n");

    std::string geom_tail; // --wall-geom: corner_gap, corner_gap_true, room_gap of the update (CW1)
    std::string cov_tail;  // --theta-cov: the update's 21 covariance entries, printed after the live comparison columns
    std::string pending;   // the row of the last update, printed once its live result (if any) is known
    const auto flush = [&](double dth, double dP)
    {
        if (pending.empty()) return;
        std::printf("%s,%.9g,%.9g%s%s\n", pending.c_str(), dth, dP, geom_tail.c_str(), cov_tail.c_str());
        pending.clear();
    };
    rc::CountableBelief belief;
    // --f1-restage: the fitter's F1 bookkeeping, carried across the dump (see the header note).
    struct F1Restage
    {
        rc::ai::view::ViewState v;                         // the viewpoints + anchors after the last ACCEPTED update
        rc::ai::view::ViewState staged;                    // the last replayed update's staged view
        bool pending = false, restored = false, camera = false, lidar = false;
        float rho_cam = 1.0f, rho_lat = 1.0f, rho_H = 1.0f;
        std::optional<Eigen::Vector3f> L;                  // the last LiDAR origin seen (room frame)
        std::optional<Eigen::Vector2f> L_minus_cam;        // its offset from the camera then
        long frames = 0, no_beam_origin = 0;
    } f1;
    const auto f1_frame = [&](rc::CountableFrame& f)
    {
        namespace vw = rc::ai::view;
        if (f1.pending and not f1.restored) f1.v = vw::view_commit(f1.staged, f1.camera, f1.lidar, f1.rho_cam, f1.rho_lat, f1.rho_H);
        f1.pending = false; f1.restored = false;
        ++f1.frames;
        if (not f.beam.endpoints.empty())
        {
            f1.L = f.beam.origin.template cast<float>();
            if (f.camera_xy.has_value()) f1.L_minus_cam = f1.L->head<2>() - *f.camera_xy;
        }
        else if (f1.L.has_value() and f1.L_minus_cam.has_value() and f.camera_xy.has_value())
        {
            f1.L->head<2>() = *f.camera_xy + *f1.L_minus_cam;
            ++f1.no_beam_origin;
        }
        const auto& s = belief.state();
        const Eigen::Vector2f c(s.cx, s.cy);
        const std::optional<Eigen::Vector2f> Lxy = f1.L.has_value() ? std::optional<Eigen::Vector2f>(f1.L->head<2>()) : std::nullopt;
        f.view = vw::view_stage(f1.v, f.camera_xy, Lxy, f.chain_dr, c);
        float rho_cam = 1.0f;   // as CountableFitter::stage_persistent
        if (f.camera_xy.has_value())
        {
            const Eigen::Vector2f r = *f.camera_xy - c;
            const float z = r.norm();
            if (z > 0.0f)
            {
                const Eigen::Vector2f t(-r.y() / z, r.x() / z);
                const Eigen::Vector2f ux(std::cos(s.yaw), std::sin(s.yaw)), uy(-std::sin(s.yaw), std::cos(s.yaw));
                const float w_perp = s.w * std::abs(ux.dot(t)) + s.h * std::abs(uy.dot(t));
                if (w_perp > 0.0f) rho_cam = vw::rho_camera(vw::anchor_separation(f.view.a_cam, f.view.cam), z, w_perp);
            }
        }
        float rho_lat = 1.0f, rho_H = 1.0f;
        if (f1.L.has_value())
        {
            const float zl = (f1.L->head<2>() - c).norm();
            rho_lat = vw::rho_lidar_lateral(vw::anchor_separation(f.view.a_lat, f.view.L), f1_gap);
            const auto span = belief.params().span();
            const float z_edge = span.hangs() ? span.z0(s.H) : s.H;
            if (zl > 0.0f)
                rho_H = vw::rho_lidar_height(vw::anchor_separation(f.view.a_H, f.view.L), zl, std::atan2(z_edge - f1.L->z(), zl), f1_gap);
        }
        f.rho_cam.fill(rho_cam);
        f.rho_L = {rho_lat, rho_lat, rho_H, rho_lat, rho_lat, rho_lat};
        f1.staged = f.view;
        f1.camera = not f.points.empty();
        f1.lidar = belief.lidar_measured(f);
        f1.rho_cam = rho_cam; f1.rho_lat = rho_lat; f1.rho_H = rho_H;
        f1.pending = true;
    };
    std::optional<std::array<float, 11>> rebirth;   // CB1: changed p_wall cx cy H w h yaw c00 c01 c11 for this dump's stem
    if (not rebirth_file.empty())
    {
        std::ifstream rf(rebirth_file);
        if (not rf) { std::fprintf(stderr, "replay: cannot open --rebirth %s\n", rebirth_file.c_str()); return 2; }
        std::string stem = argv[1];
        if (const auto sl = stem.find_last_of('/'); sl != std::string::npos) stem = stem.substr(sl + 1);
        if (const auto dot = stem.rfind(".bin"); dot != std::string::npos) stem = stem.substr(0, dot);
        std::string line;
        while (std::getline(rf, line))
        {
            std::vector<std::string_view> tok;
            for (std::size_t a = 0; a < line.size();)
            {
                const std::size_t b = line.find(' ', a);
                const std::size_t e = b == std::string::npos ? line.size() : b;
                if (e > a) tok.emplace_back(line.data() + a, e - a);
                a = e + 1;
            }
            if (tok.size() != 12 or tok[0] != stem) continue;
            std::array<float, 11> v{};
            bool good = true;
            for (int j = 0; j < 11; ++j)
            {
                const auto t = tok[std::size_t(j + 1)];
                const auto [p, ec] = std::from_chars(t.data(), t.data() + t.size(), v[std::size_t(j)]);
                good = good and ec == std::errc{} and p == t.data() + t.size();
            }
            if (not good) { std::fprintf(stderr, "replay: malformed --rebirth line for %s\n", stem.c_str()); return 2; }
            rebirth = v;
        }
        std::fprintf(stderr, "replay: rebirth %s\n", rebirth ? "from the re-derived birth (COUNTERFACTUAL)" : "none for this dump (as recorded)");
    }
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
                          "%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.9g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%d,%.6g,%d,%.4f,%.4g,%.9g,%.6g",
                          upd, static_cast<long long>(ev.cycle), ev.gated ? 1 : 0, ev.frame.points.size(),
                          ev.frame.lidar_only ? 1 : 0, ev.frame.beam.endpoints.size(),
                          s.cx, s.cy, s.H, s.w, s.h, s.yaw, sd(0), sd(1), sd(2), sd(3), sd(4),
                          std::sqrt(std::max(0.0f, b.yaw_marginal_var())), sd(5),
                          b.last_gn_iters(), b.last_gn_converged() ? 1 : 0, b.last_lambda(),
                          om(0), om(1), om(2), om(3), om(4), om(5), fe, pd(0), pd(1), pd(2), pd(3), pd(4), pd(5), b.last_wall_entered() ? 1 : 0,
                          ev.frame.wall.ok ? double((b.back_centre(s) - ev.frame.wall.p).dot(ev.frame.wall.n))
                                           : std::numeric_limits<double>::quiet_NaN(),
                          (b.last_gn_iters() >= b.params().gn_iters_max and not b.last_gn_converged()) ? 1 : 0,
                          double((ev.frame.camera_xy.value_or(ev.frame.robot_xy) - Eigen::Vector2f(s.cx, s.cy)).norm()),
                          b.last_jump_p(), b.proportion_value(0), b.params().shape_proportions.empty()
                                                              ? std::numeric_limits<double>::quiet_NaN() : b.proportion_std(0));
            pending = row;
            geom_tail.clear();
            if (wall_geom)
            {
                namespace ct = rc::containment;
                const auto& fw = ev.frame.wall;
                const double nan = std::numeric_limits<double>::quiet_NaN();
                double cg = nan, ct_true = nan, rg = nan;
                if (fw.ok)
                {
                    cg = ct_true = std::numeric_limits<double>::infinity();
                    const double dl = b.wall_offset_mean(fw);   // the TRUE wall is the staged line + δ̂·n (the engine's θ + B_w b)
                    for (const auto& k : ct::corners(s.cx, s.cy, s.w, s.h, s.yaw))
                    {
                        cg = std::min(cg, double((k.p - fw.p).dot(fw.n)));
                        ct_true = std::min(ct_true, double((k.p - fw.p).dot(fw.n)) - dl);
                    }
                }
                if (ev.frame.room_polygon.size() >= 3)
                    rg = ct::min_corner_distance(s.cx, s.cy, s.w, s.h, s.yaw, ev.frame.room_polygon, false, fw.p, fw.n);
                char g[96];
                std::snprintf(g, sizeof g, ",%.6g,%.6g,%.6g", cg, ct_true, rg);
                geom_tail = g;
            }
            cov_tail.clear();
            if (theta_cov)
                for (int i = 0; i < 6; ++i)
                    for (int j = i; j < 6; ++j)
                    {
                        char c[40];
                        std::snprintf(c, sizeof c, ",%.9g", double(S(i, j)));
                        cov_tail += c;
                    }
        },
        [&](const rc::replay::ResultEv& live, const rc::CountableBelief& b)
        {
            const auto& s = b.state();
            const double dth = std::max({std::abs(double(s.cx) - live.theta.cx), std::abs(double(s.cy) - live.theta.cy),
                                         std::abs(double(s.H) - live.theta.H), std::abs(double(s.w) - live.theta.w),
                                         std::abs(double(s.h) - live.theta.h), std::abs(double(s.yaw) - live.theta.yaw)});
            const auto& P = b.joint_covariance();
            // A dump recorded before K2a-L lacks the appended proportion latent(s) (the last rows of P): the common prefix is compared.
            const std::size_t n_cmp = live.p_diag.size() == std::size_t(P.rows()) ? live.p_diag.size()
                                    : live.p_diag.size() + std::size_t(rc::CountableEngineModel::kProportions) == std::size_t(P.rows())
                                        ? live.p_diag.size() : 0;
            double dP = n_cmp > 0 ? 0.0 : std::numeric_limits<double>::infinity();
            for (int i = 0; std::isfinite(dP) and i < int(n_cmp); ++i)
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
            if (f1_restage) f1_frame(f);
            if (sig2_cam_measured)
            {
                const float yaw_std = std::sqrt(std::max(0.0f, f.sig2_cam[5]));
                f.sig2_cam = rc::persistent_latent_inputs({0.0f, 0.0f, yaw_std}, rc::LatentView{}).sig2_cam;
            }
            if (not neighbour_manifests.empty())
                for (auto& nb : f.foreign_boxes)
                {
                    if (nb.cls.empty()) { ++neighbours_without_class; continue; }
                    if (const auto sd = rc::neighbour_size_std(neighbour_manifests + "/" + nb.cls + ".concept.toml"))
                    { nb.std(3) = (*sd)[0]; nb.std(4) = (*sd)[1]; }
                }
        },
        [&](rc::replay::SeedEv& seed)
        {
            if (rebirth)
            {
                const auto& r = *rebirth;
                if (r[0] != 0.0f) seed.state = rc::CountableBeliefState{r[2], r[3], r[4], r[5], r[6], r[7]};
            }
            if (seed_h_rule)
            {   // F1 round 6.2's rule on the first camera frame (the birth cloud is not in the dump), at the recorded born pose
                const auto pts = rc::replay::first_camera_points(argv[1]);
                const float H0 = seed.state.H;
                if (not pts.empty()) seed.state.H = rc::CountableBelief(seed.state, seed.params).seed_height(pts, rc::kZedYoloBoundaryStdM);
                std::fprintf(stderr, "replay: seed H %.4f -> %.4f (the rule on the first camera frame, %zu points)\n", H0, seed.state.H, pts.size());
            }
            if (not permanence_manifests.empty())
                if (const auto pm = rc::manifest::permanence(permanence_manifests + "/" + rc::node_class(seed.node) + ".concept.toml"))
                {
                    seed.params.move_rate_per_day = pm->move_rate_per_day;
                    seed.params.move_scale_m = pm->move_scale_m;
                    seed.params.move_scale_rad = pm->move_scale_rad;
                }
            if (not parts_manifests.empty())
            {
                const std::string path = parts_manifests + "/" + rc::node_class(seed.node) + ".concept.toml";
                seed.params.shape_parts = rc::manifest::load_shape_parts(path);   // throws on a malformed table: a stream error
                seed.params.solid_parts = rc::manifest::load_solid_is_parts(path);
                seed.params.shape_proportions = rc::manifest::load_shape_proportions(path);
                std::fprintf(stderr, "replay: solid %s (%zu part(s)) from %s\n", seed.params.solid_parts ? "parts" : "box",
                             seed.params.shape_parts.size(), path.c_str());
            }
            if (not placement_manifests.empty())
            {   // CB1: the class's placement mixture (a dump recorded before CB1 lacks placement_p_against_wall)
                const std::string path = placement_manifests + "/" + rc::node_class(seed.node) + ".concept.toml";
                seed.params.placement_p_against_wall = rc::manifest::placement_p_against_wall(path).value_or(-1.0f);
                std::fprintf(stderr, "replay: placement p_against_wall %.2f from %s\n", seed.params.placement_p_against_wall, path.c_str());
            }
            if (standoff_manifests.empty()) return;
            const std::string path = standoff_manifests + "/" + rc::node_class(seed.node) + ".concept.toml";
            if (const auto so = rc::manifest::wall_standoff(path))
            {
                seed.params.wall_standoff_m = so->mean_m;
                seed.params.wall_standoff_std_m = so->std_m;
                std::fprintf(stderr, "replay: stand-off %.3f +- %.3f m from %s\n", so->mean_m, so->std_m, path.c_str());
            }
        },
        [&](const rc::CountableBelief&) { f1.restored = true; },
        [&](rc::CountableBelief& b)
        {
            if (not rebirth or (*rebirth)[0] == 0.0f) return;
            const auto& r = *rebirth;
            Eigen::Matrix2f C;
            C << r[8], r[9], r[9], r[10];
            b.set_birth_position_cov(C);
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
        if (not belief.params().shape_proportions.empty())
            std::fprintf(stderr, "replay: final proportion %s %.6f std %.6f (prior mean %.6f std %.6f)\n",
                         belief.params().shape_proportions[0].name.c_str(), belief.proportion_value(0), belief.proportion_std(0),
                         0.5 * double(belief.params().shape_proportions[0].lo + belief.params().shape_proportions[0].hi),
                         double(belief.params().shape_proportions[0].hi - belief.params().shape_proportions[0].lo) / std::sqrt(12.0));
    }
    if (f1_restage)
        std::fprintf(stderr, "replay: COUNTERFACTUAL --f1-restage %.4f: %ld frames re-staged with F1's rho (viewpoint separation), %ld of "
                     "them with the LiDAR origin carried from the last beam — not the live run\n", f1_gap, f1.frames, f1.no_beam_origin);
    if (neighbours_without_class > 0)
        std::fprintf(stderr, "replay: %ld recorded neighbours carry no class (a dump older than ForeignBox::cls): their size std "
                     "left as recorded\n", neighbours_without_class);
    const bool defaults = not reader.diag().missing.empty() or not reader.diag().bad.empty();
    if (defaults and not allow_defaults)
        std::fprintf(stderr, "replay: FAIL — fields ran at their defaults (missing/bad above); pass --allow-defaults if intended\n");
    std::fprintf(stderr, "replay: %ld updates; against the %ld live results: max dth_live %.3g, max dP_live %.3g%s\n", upd + 1,
                 compared, worst_th, worst_P, compared > 0 and worst_th == 0.0 and worst_P == 0.0 ? " (bit-identical)" : "");
    return (not ok or (defaults and not allow_defaults)) ? 1 : 0;
}
