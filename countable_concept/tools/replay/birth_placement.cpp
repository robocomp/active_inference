// birth_placement — re-derive a countable birth from a replay dump and replay the belief from it (CB1, COUNTERFACTUAL).
//
// The dump does not record the birth cloud as such, but its FIRST camera update is the birth frame: the fitter seeds the belief
// from that frame's candidate + residual points and updates on the same frame in the same cycle (G1x: the point counts of the
// "cold-start footprint" log line and of update 0 agree on every chair dump). From it this tool rebuilds what the fitter's birth
// saw — the cold-start footprint (CountableFitter::run_inference), the camera (frame.camera_xy at the ZED's height above the floor,
// ROBOT_GEOMETRY.md: body 0.0325 + zed 0.945), the robot's pose covariance (frame.sigma_pose, frame.robot_xy) and the wall staged on
// update 0 (frame.wall) — and runs rc::seed_birth twice: WITHOUT the view (BASE's comparison) and WITH it (CB1). It then replays
// the dump's events through the CURRENT engine from the CB1 seed (the recorded seed when the branch did not change) and prints the
// held yaw against the staged wall.
//
// Reconstructed, not recorded (the report states it): the room's span along the wall normal is 2·(room interior − wall)·n, the
// recorded room interior point standing in for the polygon the fitter had (a rectangle's centre ⇒ exact for these rooms); the
// nearest wall is the one update 0 staged (the wall nearest the SEEDED back centre — the cloud's nearest wall for a wall chair).
//
// Run: ./birth_placement CONFIG_DIR CONFIG_FILE MANIFEST_DIR [--rebirth-out FILE] DUMP.bin [DUMP.bin ...]
//      (--rebirth-out appends, per dump, the CB1 placement posterior and — when the branch changed FREE -> WALL — the re-derived
//      seed and birth covariance, for replay --rebirth)
//      e.g. ./birth_placement ../../../chair_concept etc/countable.toml ../../../common/concept_manifest g1x_tour_S03/replay/chair_18_*.bin
// One line per dump on stdout: tour-free fields, BASE / CB1 branch, the live (BASE engine) yaw after update 0 and at the end, the
// CB1 replay's, each with back·(−n) (+1 = back on the staged wall, −1 = back to the room).
#include <clocale>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <locale>
#include <string>
#include <vector>

#include <genericworker.h>   // the test stub: the real ConfigLoader (manifest_stub/)
#include "countable_birth_seed.h"
#include "countable_config.h"
#include "countable_latent_inputs.h"
#include "countable_replay.h"
#include "../../common/concept_manifest/wall_standoff.h"

namespace
{
constexpr float kZedHeightM = 0.0325f + 0.945f;   // ROBOT_GEOMETRY.md: body above the floor + zed above body

float back_on_wall(const rc::CountableBeliefState& s, const rc::WallRef& w)
{   // the back (local +Y) against the wall's outward direction
    return Eigen::Vector2f(-std::sin(s.yaw), std::cos(s.yaw)).dot(-w.n);
}
}  // namespace

int main(int argc, char** argv)
{
    std::setlocale(LC_ALL, "");
    if (argc < 5)
    {
        std::fprintf(stderr, "usage: %s CONFIG_DIR CONFIG_FILE MANIFEST_DIR DUMP.bin [DUMP.bin ...]\n", argv[0]);
        return 2;
    }
    const std::filesystem::path here = std::filesystem::current_path();
    std::vector<std::string> dumps;
    std::string rebirth_out;   // --rebirth-out FILE: the CB1 seeds that changed branch, for replay --rebirth
    for (int i = 4; i < argc; ++i)
    {
        if (std::string(argv[i]) == "--rebirth-out" and i + 1 < argc) { rebirth_out = std::filesystem::absolute(argv[++i]).string(); continue; }
        dumps.push_back(std::filesystem::absolute(argv[i]).string());
    }
    std::ofstream rb;
    if (not rebirth_out.empty()) { rb.open(rebirth_out, std::ios::app); rb.imbue(std::locale::classic()); }
    const std::string manifests = std::filesystem::absolute(argv[3]).string();
    std::filesystem::current_path(argv[1]);
    ConfigLoader cl;
    cl.load(argv[2]);
    const rc::CountableConfig cfg = rc::load_countable_config(cl);
    std::filesystem::current_path(here);
    std::printf("dump,npts,cam_across,cam_range,mid_n,p_wall,base_branch,cb1_branch,live_yaw0,live_back0,live_yawN,live_backN,cb1_yaw0,"
                "cb1_back0,cb1_yawN,cb1_backN,n_upd\n");
    for (const auto& path : dumps)
    {
        std::vector<rc::replay::Event> evs;
        {
            rc::replay::Reader r;
            if (not r.open(path)) { std::fprintf(stderr, "%s: cannot open\n", path.c_str()); continue; }
            rc::replay::Event e;
            while (r.next(e)) evs.push_back(e);
        }
        const rc::replay::SeedEv* seed = nullptr;
        const rc::replay::UpdateEv* u0 = nullptr;
        const rc::replay::RoomInteriorEv* ri = nullptr;
        for (const auto& e : evs)
        {
            if (const auto* s = std::get_if<rc::replay::SeedEv>(&e); s and not seed) seed = s;
            if (const auto* u = std::get_if<rc::replay::UpdateEv>(&e); u and not u0 and not u->frame.points.empty()) u0 = u;
            if (const auto* c = std::get_if<rc::replay::RoomInteriorEv>(&e); c and not ri) ri = c;
        }
        if (not seed or not u0 or not u0->frame.camera_xy) { std::fprintf(stderr, "%s: no seed / camera frame\n", path.c_str()); continue; }
        const auto& pts = u0->frame.points;
        const rc::WallRef wall = u0->frame.wall;
        // the cold-start footprint, as CountableFitter::run_inference builds it
        rc::CountableBeliefState s0 = seed->state;
        {
            Eigen::Vector3f sum = Eigen::Vector3f::Zero();
            std::vector<float> zs;
            for (const auto& p : pts) { sum += p; zs.push_back(p.z()); }
            s0.cx = sum.x() / float(pts.size()); s0.cy = sum.y() / float(pts.size());
            const std::size_t k = static_cast<std::size_t>(0.95f * float(zs.size() - 1));
            std::nth_element(zs.begin(), zs.begin() + std::ptrdiff_t(k), zs.end());
            s0.H = zs[k];
        }
        const float z_hi = s0.H + 3.0f * cfg.ai2_sigma_base_m;
        const float z_lo = rc::countable::VerticalSpan{cfg.vertical_extent_m}.z0(s0.H) + 3.0f * cfg.ai2_sigma_base_m;
        if (const auto mom = rc::CountableBelief::footprint_moment(pts, z_lo, z_hi); mom.ok)
        {
            s0.cx = mom.cx; s0.cy = mom.cy;
            s0.w = std::max(0.10f, mom.ext_major);
            s0.h = mom.ext_minor > cfg.ai2_depth_obs_band_m ? mom.ext_minor : (cfg.ai2_prior_depth_m > 0.f ? cfg.ai2_prior_depth_m : cfg.ai2_prior_footprint_m);
            const float sm = mom.ext_major + mom.ext_minor;
            s0.yaw = (sm > 1e-4f and (mom.ext_major - mom.ext_minor) / sm > 0.10f) ? mom.phi : seed->state.yaw;
        }
        const rc::NearestWall nw = [&](const Eigen::Vector2f&) { return wall; };
        rc::BirthView view;
        view.have_cam = true;
        view.cam = Eigen::Vector3f(u0->frame.camera_xy->x(), u0->frame.camera_xy->y(), kZedHeightM);
        view.have_pose = true;
        view.sigma_pose = u0->frame.sigma_pose;
        view.robot_xy = u0->frame.robot_xy;
        const float span = ri ? 2.0f * (ri->c - wall.p).dot(wall.n) : 0.0f;
        view.room_span = [span](const rc::WallRef&) { return span; };
        const std::string name = std::filesystem::path(path).stem().string();
        rc::CountableBeliefState sb = s0, sa = s0;
        const auto pc_base = rc::seed_birth(sb, pts, z_lo, z_hi, cfg, true, nw, name + " BASE");
        float p_wall = -1.0f;
        const auto pc_cb1  = rc::seed_birth(sa, pts, z_lo, z_hi, cfg, true, nw, name + " CB1", &view, &p_wall);
        // the live (BASE engine) trajectory: the recorded results
        rc::CountableBeliefState l0{}, lN{};
        int nres = 0;
        for (const auto& e : evs)
            if (const auto* r = std::get_if<rc::replay::ResultEv>(&e)) { if (nres == 0) l0 = r->theta; lN = r->theta; ++nres; }
        // the CB1 replay: from the CB1 seed when the branch changed, else the recorded one; the class placement prior set
        rc::CountableBeliefParams params = seed->params;
        params.placement_p_against_wall = p_wall >= 0.0f ? p_wall : cfg.placement_p_against_wall.value_or(-1.0f);   // as the fitter
        const bool changed = pc_base.has_value() != pc_cb1.has_value();
        rc::CountableBeliefState start = seed->state;
        if (changed)
        {
            start = sa;
            rc::CountableBelief tmp(start, params);
            start.H = tmp.seed_height(pts, rc::kZedYoloBoundaryStdM);   // F1 round 6.2, as the fitter re-derives it
        }
        if (rb.is_open())
        {   // every dump: its placement posterior; the re-derived seed and birth covariance when the branch changed FREE -> WALL
            const std::string stem = std::filesystem::path(path).stem().string();
            if (changed and not pc_cb1)
                std::fprintf(stderr, "%s: WALL -> FREE is not written for replay --rebirth (the recorded birth covariance would stand)\n", path.c_str());
            const bool w = changed and pc_cb1.has_value();
            const Eigen::Matrix2f C = w ? *pc_cb1 : Eigen::Matrix2f::Zero();
            rb << std::setprecision(9) << stem << ' ' << int(w) << ' ' << params.placement_p_against_wall << ' ' << start.cx << ' '
               << start.cy << ' ' << start.H << ' ' << start.w << ' ' << start.h << ' ' << start.yaw << ' ' << C(0, 0) << ' ' << C(0, 1)
               << ' ' << C(1, 1) << '\n';
        }
        rc::CountableBelief b(start, params);
        if (changed and pc_cb1) b.set_birth_position_cov(*pc_cb1);
        rc::CountableBelief pre;
        rc::CountableBeliefState a0{}, aN{};
        int nupd = 0;
        for (auto& e : evs)
        {
            std::visit([&]<class E>(E& ev)
            {
                if constexpr (std::is_same_v<E, rc::replay::BirthPosCovEv>) { if (not changed) b.set_birth_position_cov(ev.excess); }
                else if constexpr (std::is_same_v<E, rc::replay::RoomInteriorEv>) b.set_room_interior(ev.c);
                else if constexpr (std::is_same_v<E, rc::replay::UpdateEv>)
                {
                    pre = b;
                    b.update(ev.frame);
                    if (nupd == 0)
                    {
                        a0 = b.state();
                        // diagnostic: the frame's points' mixture NLL at the four yaw modes of the updated state (k·π/2, w ↔ h on
                        // the quarter turns), relative to the held mode — does the cloud itself tell the modes apart?
                        const float R = params.sigma_base_m * params.sigma_base_m;
                        const double n = double(ev.frame.points.size());
                        const double e0 = n * b.mean_energy(ev.frame.points, a0, R);
                        std::fprintf(stderr, "[%s] mode NLL (update 0, %zu pts, rel. to held):", name.c_str(), ev.frame.points.size());
                        for (int k = 0; k < 4; ++k)
                        {
                            rc::CountableBeliefState st = a0;
                            st.yaw = std::remainder(a0.yaw + float(k) * float(M_PI_2), float(2.0 * M_PI));
                            if (k % 2) std::swap(st.w, st.h);
                            std::fprintf(stderr, " k%d %+.1f", k, n * b.mean_energy(ev.frame.points, st, R) - e0);
                        }
                        std::fprintf(stderr, "\n");
                    }
                    aN = b.state();
                    ++nupd;
                }
                else if constexpr (std::is_same_v<E, rc::replay::PredictEv>) b.predict();
                else if constexpr (std::is_same_v<E, rc::replay::InflateEv>) b.inflate_for_age(ev.dt_s, ev.dt_nominal_s);
                else if constexpr (std::is_same_v<E, rc::replay::RestoreEv>) b = pre;
                else if constexpr (std::is_same_v<E, rc::replay::ResolveFrontEv>) b.resolve_front(ev.cue, ev.weight);
            }, e);
        }
        const float mid = [&]
        {
            std::vector<float> v;
            for (const auto& p : pts) if (p.z() >= z_lo and p.z() <= z_hi) v.push_back((p.head<2>() - wall.p).dot(wall.n));
            std::sort(v.begin(), v.end());
            return v.empty() ? NAN : 0.5f * (v[std::size_t(0.05f * float(v.size() - 1))] + v[std::size_t(0.95f * float(v.size() - 1))]);
        }();
        const Eigen::Vector2f cam_xy = *u0->frame.camera_xy;
        std::printf("%s,%zu,%.3f,%.3f,%.3f,%.3g,%s,%s,%.1f,%+.2f,%.1f,%+.2f,%.1f,%+.2f,%.1f,%+.2f,%d\n", name.c_str(), pts.size(),
                    (cam_xy - wall.p).dot(wall.n), (cam_xy - Eigen::Vector2f(s0.cx, s0.cy)).norm(), mid, p_wall,
                    pc_base ? "WALL" : "FREE", pc_cb1 ? "WALL" : "FREE", l0.yaw * 57.29578f, back_on_wall(l0, wall),
                    lN.yaw * 57.29578f, back_on_wall(lN, wall), a0.yaw * 57.29578f, back_on_wall(a0, wall), aN.yaw * 57.29578f,
                    back_on_wall(aN, wall), nupd);
        std::fflush(stdout);
    }
    return 0;
}
