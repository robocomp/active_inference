// mount_factors_test -- the floor + vertical LiDAR mount factors (common/sensor_calib/mount_factors.h) and this
// agent's stage that feeds them (src/lidar_mount_stage.*). Synthetic, no DDS/DSR.
//   make -C build mount_factors_test && ../bin/mount_factors_test [2|3|P|S]
// Parts 2, 3 and P moved VERBATIM from room_concept/tools/mount_factors_selftest.cpp on 2026-10-29 (plan Phase 3);
// their expected output is room_concept/tools/oracle/mount_factors_selftest.txt as recorded before the move.
#include "../../common/sensor_calib/mount_factors.h"
#include "../src/lidar_mount_stage.h"
#include "../../robot_concept/src/mount_overlay.h"   // the overlay a planted error is written in

#include <charconv>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <array>
#include <random>
#include <string>
#include <string_view>
#include <vector>

static int failures = 0;
static void check(bool ok, const char *w) { std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", w); failures += not ok; }
using namespace rc::mountf;
static constexpr double kDegD = M_PI / 180.0;
// ─────────────────────────────── shared synthetic room + LiDAR ray caster ───────────────────────────────
namespace syn
{
    using V3 = Eigen::Vector3d;
    struct Room { double hx = 2.0, hy = 3.0, h = 2.6; };          // 4 x 6 m, 2.6 m ceiling, world frame
    /// Small 6-DoF mount error about the sensor origin s: p_true = R(w)(p_nom - s) + s + t.
    struct M6 { V3 t = V3::Zero(), w = V3::Zero(); };
    inline Eigen::Matrix3d rot(const V3 &w)
    {
        const double a = w.norm();
        return a > 0 ? Eigen::AngleAxisd(a, w / a).toRotationMatrix() : Eigen::Matrix3d::Identity();
    }
    /// Ray from o along unit d (WORLD frame) to the first of: floor z=0, ceiling z=h, the 4 walls. Returns t.
    inline double cast(const Room &r, const V3 &o, const V3 &d, bool ceiling = true)
    {
        double best = 1e9;
        auto take = [&](double t) { if (t > 1e-6 and t < best) best = t; };
        if (d.z() < -1e-9) take(-o.z() / d.z());
        if (ceiling and d.z() > 1e-9) take((r.h - o.z()) / d.z());
        if (d.x() > 1e-9) take((r.hx - o.x()) / d.x());
        if (d.x() < -1e-9) take((-r.hx - o.x()) / d.x());
        if (d.y() > 1e-9) take((r.hy - o.y()) / d.y());
        if (d.y() < -1e-9) take((-r.hy - o.y()) / d.y());
        return best;
    }
    struct Pose { double x = 0, y = 0, th = 0; };   // body -> world: p_w = R(th) p_b + (x, y), z unchanged
    /// One sweep in the BODY frame as the NOMINAL mount transform produces it (p_nom = M^-1(p_true)).
    /// s = nominal sensor origin (body), elevations (rad, about the horizontal), azimuth step.
    inline std::vector<Eigen::Vector3f> sweep(const Room &room, const Pose &P, const V3 &s, const M6 &M,
                                              const std::vector<double> &elev, double az_step, double noise,
                                              std::mt19937 &rng, bool ceiling = true)
    {
        std::normal_distribution<double> n01(0.0, 1.0);
        const Eigen::Matrix3d R = rot(M.w);
        const double c = std::cos(P.th), sn = std::sin(P.th);
        auto to_w = [&](const V3 &b) { return V3(c * b.x() - sn * b.y() + P.x, sn * b.x() + c * b.y() + P.y, b.z()); };
        auto to_b = [&](const V3 &w) { const V3 q(w.x() - P.x, w.y() - P.y, w.z()); return V3(c * q.x() + sn * q.y(), -sn * q.x() + c * q.y(), q.z()); };
        const V3 s_true = s + M.t;                                   // the sensor origin, truly
        std::vector<Eigen::Vector3f> out;
        for (double az = 0.0; az < 2 * M_PI; az += az_step)
            for (const double el : elev)
            {
                const V3 d_nom(std::cos(el) * std::cos(az), std::cos(el) * std::sin(az), std::sin(el));
                const V3 d_b = R * d_nom;                              // the true ray, body frame
                const V3 o_w = to_w(s_true);
                const V3 d_w(c * d_b.x() - sn * d_b.y(), sn * d_b.x() + c * d_b.y(), d_b.z());
                const double t = cast(room, o_w, d_w, ceiling);
                if (t > 50.0) continue;
                V3 p_true = to_b(o_w + t * d_w) + noise * V3(n01(rng), n01(rng), n01(rng));
                const V3 p_nom = R.transpose() * (p_true - s - M.t) + s;   // M^-1
                out.emplace_back(p_nom.cast<float>());
            }
        return out;
    }
    inline std::vector<double> range(double a, double b, double step)
    { std::vector<double> v; for (double x = a; x <= b + 1e-9; x += step) v.push_back(x * kDegD); return v; }
}

// ─────────────────────────────── Part 2: the floor plane ───────────────────────────────
static void part2()
{
    using namespace syn;
    std::printf("Part 2: FloorPlaneMount (bpearl roll, pitch, z from the floor seen head-on)\n");
    const Room room;
    const V3 s_bp(0.0, 0.14, 0.7025);                 // Shadow frame (r2.2 item 5)
    const auto elev = range(-88.0, -2.0, 3.0);        // inverted dome, pointing down
    M6 M; M.w = V3(0.5 * kDegD, -0.8 * kDegD, 0.0); M.t = V3(0.0, 0.0, 0.01);
    auto drive = [&](FloorPlaneMount &fm, int sweeps, double furniture, double flat_sd, unsigned seed)
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> ux(-1.0, 1.0), uth(-M_PI, M_PI), u01(0.0, 1.0), uz(0.3, 0.6);
        std::normal_distribution<double> n01(0.0, 1.0);
        for (int k = 0; k < sweeps; ++k)
        {
            const Pose P{ux(rng), 1.5 * ux(rng), uth(rng)};
            auto pts = sweep(room, P, s_bp, M, elev, 2.0 * kDegD, 0.005, rng, false);
            const double off = flat_sd * n01(rng);      // the whole sweep's floor offset (common mode)
            for (auto &p : pts)
            {
                const V3 pd = p.cast<double>();
                const V3 pt = rot(M.w) * (pd - s_bp) + s_bp + M.t;   // back to true to edit the scene
                V3 q = pt;
                if (std::abs(pt.z()) < 0.03)
                {
                    q.z() += off;
                    if (pt.head<2>().norm() < 1.5 and u01(rng) < furniture) q.z() = uz(rng);   // a seat, a table top
                }
                p = (rot(M.w).transpose() * (q - s_bp - M.t) + s_bp).cast<float>();
            }
            fm.observe_sweep(pts, s_bp.cast<float>());
        }
    };
    auto show = [](const char *tag, const FloorPlaneMount &fm)
    {
        const auto i = fm.info(); const auto m = i.mean(), s = i.sigma();
        std::printf("    %-30s roll %+.3f +- %.4f deg | pitch %+.3f +- %.4f deg | dz %+6.2f +- %.2f mm | share %.2f sigma_f %.1f mm\n",
                    tag, m[0] / kDegD, s[0] / kDegD, m[1] / kDegD, s[1] / kDegD, m[2] * 1e3, s[2] * 1e3,
                    fm.floor_share(), fm.sigma_floor() * 1e3);
    };
    // 2 sigma PLUS 1 % of the plant: the rows are first order in omega, the synthetic plant is an exact
    // rotation, so an O(omega^2) linearisation residue (~0.002 deg on 0.8 deg) is EXPECTED, and with
    // sigma ~0.0006 deg "within 2 sigma" alone would test it rather than the estimator (review item 9).
    auto agrees = [&](const FloorPlaneMount &fm, bool with_dz)
    {
        const auto i = fm.info(); const auto m = i.mean(), s = i.sigma();
        const double lin = 0.01 * M.w.norm();
        return std::abs(m[0] - M.w.x()) < 2 * s[0] + lin and std::abs(m[1] - M.w.y()) < 2 * s[1] + lin
               and (not with_dz or std::abs(m[2] - M.t.z()) < 2 * s[2] + 0.01 * std::abs(M.t.z()));
    };
    {
        FloorPlaneMount fm; drive(fm, 100, 0.0, 0.0, 11); show("A clean floor", fm);
        check(agrees(fm, true), "A: planted roll 0.5, pitch -0.8 deg, dz 10 mm recovered (2 sigma + 1 % linearisation)");
    }
    {
        FloorPlaneMount fm; drive(fm, 100, 0.2, 0.0, 12); show("B 20% furniture near", fm);
        check(agrees(fm, true), "B: 20 % of near floor points on furniture -> still recovered (mixture, no band)");
    }
    {
        const int S = 100; const double flat = 0.005;
        FloorPlaneMount fm; drive(fm, S, 0.0, flat, 13); show("C per-sweep floor offset 5 mm", fm);
        const auto s = fm.info().sigma();
        std::printf("    sigma_flat/sqrt(sweeps) = %.2f mm\n", flat / std::sqrt(double(S)) * 1e3);
        check(s[2] > 0.8 * flat / std::sqrt(double(S)), "C: dz sigma does not collapse below ~sigma_flat/sqrt(sweeps)");
        check(agrees(fm, true), "C: and the planted values are still recovered");
        check(s[0] < 0.02 * kDegD and s[1] < 0.02 * kDegD, "C: the per-sweep dz nuisance does NOT absorb roll/pitch (tilt stays sharp)");
    }
}

// ─────────────────────────────── Part 3: verticals (walls vertical, ceiling horizontal) ───────────────────────────────
static void part3()
{
    using namespace syn;
    std::printf("Part 3: VerticalMount (helios tilt from walls + ceiling; bpearl planar relative to the map)\n");
    const Room room;
    const V3 s_h(0.0, -0.155, 1.1075), s_bp(0.0, 0.14, 0.7025);     // Shadow frame (r2.2 item 5)
    const auto elev_h  = range(-15.5, 54.5, 2.0);                    // inverted helios: the fan points UP
    const auto elev_bp = range(-88.0, -2.0, 3.0);
    const Eigen::Matrix3f wide = Eigen::Vector3f(0.05f, 0.05f, float(2.0 * kDegD)).cwiseAbs2().asDiagonal();
    // the wall map in the BODY frame, as the localiser's pose estimate puts it
    auto walls_body = [&](const Pose &P)
    {
        const double c = std::cos(P.th), sn = std::sin(P.th);
        auto tb = [&](double x, double y) { const double qx = x - P.x, qy = y - P.y; return Eigen::Vector2f(float(c * qx + sn * qy), float(-sn * qx + c * qy)); };
        const std::array<Eigen::Vector2f, 4> v{tb(-room.hx, -room.hy), tb(room.hx, -room.hy), tb(room.hx, room.hy), tb(-room.hx, room.hy)};
        std::vector<VerticalMount::Wall> w;
        for (int k = 0; k < 4; ++k) w.push_back({v[k], v[(k + 1) % 4]});
        return w;
    };
    struct Arm { VerticalMount::Role role; M6 M; double pose_sd_xy, pose_sd_th; int sweeps; };
    auto drive = [&](const Arm &a, unsigned seed)
    {
        VerticalMount vm; VerticalMount::Params p; p.role = a.role; vm.set_params(p);
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> ux(-1.0, 1.0), uth(-M_PI, M_PI);
        std::normal_distribution<double> n01(0.0, 1.0);
        const bool helios = a.role == VerticalMount::Role::HeliosTilt;
        const V3 s = helios ? s_h : s_bp;
        for (int k = 0; k < a.sweeps; ++k)
        {
            const Pose P{ux(rng), 1.5 * ux(rng), uth(rng)};
            const auto pts = sweep(room, P, s, a.M, helios ? elev_h : elev_bp, 1.5 * kDegD, 0.01, rng, helios);
            const Pose Pe{P.x + a.pose_sd_xy * n01(rng), P.y + a.pose_sd_xy * n01(rng), P.th + a.pose_sd_th * n01(rng)};
            vm.observe_sweep(pts, s.cast<float>(), walls_body(Pe), wide, helios ? float(room.h) : 0.f);
        }
        return vm;
    };
    auto show = [](const char *tag, const VerticalMount &vm)
    {
        const auto i = vm.info(); const auto m = i.mean(), s = i.sigma();
        std::printf("    %-32s dx %+5.1f+-%4.1f dy %+5.1f+-%4.1f dz %+5.1f+-%4.1f mm | roll %+.3f+-%.3f pitch %+.3f+-%.3f yaw %+.3f+-%.3f deg | share %.2f sw %.1f mm\n",
                    tag, m[0] * 1e3, s[0] * 1e3, m[1] * 1e3, s[1] * 1e3, m[2] * 1e3, s[2] * 1e3, m[3] / kDegD, s[3] / kDegD,
                    m[4] / kDegD, s[4] / kDegD, m[5] / kDegD, s[5] / kDegD, vm.wall_share(), vm.sigma_wall() * 1e3);
    };
    const VerticalMount::Params defaults;
    const auto prior_sd = defaults.prior_sigma;
    auto at_prior = [&](const VerticalMount &vm, std::initializer_list<int> idx)
    {
        const auto i = vm.info(); const auto s = i.sigma(), m = i.mean();
        bool ok = true;
        for (int k : idx) ok = ok and std::abs(s[k] - prior_sd[k]) < 1e-9 * prior_sd[k] + 1e-15 and std::abs(m[k]) < 1e-12;
        return ok;
    };
    const double lin = 0.01;   // 1 % of the plant: first-order rows vs an exact synthetic rotation (see part 2)
    {   // A: planted helios tilt, exact pose
        Arm a{VerticalMount::Role::HeliosTilt, {}, 0.0, 0.0, 60};
        a.M.w = V3(0.7 * kDegD, -0.4 * kDegD, 0.0);
        const auto vm = drive(a, 21); show("A helios tilt 0.7/-0.4 deg", vm);
        const auto i = vm.info(); const auto m = i.mean(), s = i.sigma();
        check(std::abs(m[3] - a.M.w.x()) < 2 * s[3] + lin * a.M.w.norm() and std::abs(m[4] - a.M.w.y()) < 2 * s[4] + lin * a.M.w.norm(),
              "A: helios roll/pitch recovered within 2 sigma (+1 % linearisation)");
        check(s[3] < 0.05 * kDegD and s[4] < 0.05 * kDegD, "A: and informed (sigma < 0.05 deg)");
    }
    {   // C: random per-sweep POSE error (3 cm, 1 deg), ZERO tilt -> pose error is never read as tilt (Review Focus #1)
        Arm a{VerticalMount::Role::HeliosTilt, {}, 0.03, 1.0 * kDegD, 60};
        const auto vm = drive(a, 23); show("C pose error 3 cm/1 deg, no tilt", vm);
        const auto i = vm.info(); const auto m = i.mean(), s = i.sigma();
        check(std::abs(m[3]) < 2 * s[3] and std::abs(m[4]) < 2 * s[4], "C: per-sweep pose error does NOT leak into tilt (within 2 sigma of 0)");
    }
    {   // D: helios dz planted 2 cm -> unobservable (the ceiling height is a nuisance): stays EXACTLY at its prior
        Arm a{VerticalMount::Role::HeliosTilt, {}, 0.0, 0.0, 30};
        a.M.t = V3(0.0, 0.0, 0.02);
        const auto vm = drive(a, 24); show("D helios dz 2 cm", vm);
        check(at_prior(vm, {2}), "D: helios dz stays at its prior (sigma == prior, mean 0)");
    }
    {   // E (r2.2 item 6): helios with a planted 30 mm lever and 1 deg yaw, CORRECT map -> NO planar information
        Arm a{VerticalMount::Role::HeliosTilt, {}, 0.0, 0.0, 30};
        a.M.t = V3(0.03, -0.02, 0.0); a.M.w = V3(0.0, 0.0, 1.0 * kDegD);
        const auto vm = drive(a, 25); show("E helios lever 30 mm + 1 deg yaw", vm);
        check(at_prior(vm, {0, 1, 5}), "E: the vertical factor gives the helios ZERO planar information (sigma == prior)");
        const auto i = vm.info(); const auto m = i.mean(), s = i.sigma();
        check(std::abs(m[3]) < 2 * s[3] and std::abs(m[4]) < 2 * s[4], "E: and the lever is not read as tilt");
    }
    {   // B: bpearl planar offset RELATIVE to the map (dx 2 cm, dy -3 cm, dyaw 0.8 deg), small true pose noise
        Arm a{VerticalMount::Role::BpearlPlanar, {}, 0.01, 0.3 * kDegD, 400};
        a.M.t = V3(0.02, -0.03, 0.0); a.M.w = V3(0.0, 0.0, 0.8 * kDegD);
        const auto vm = drive(a, 22); show("B bpearl planar 20/-30 mm, 0.8 deg", vm);
        const auto i = vm.info(); const auto m = i.mean(), s = i.sigma();
        check(std::abs(m[0] - 0.02) < 2 * s[0] and std::abs(m[1] + 0.03) < 2 * s[1]
              and std::abs(m[5] - a.M.w.z()) < 2 * s[5] + lin * a.M.w.norm(), "B: bpearl planar (relative) recovered within 2 sigma");
        check(std::abs(m[0]) > 2 * s[0] and std::abs(m[1]) > 2 * s[1] and std::abs(m[5]) > 2 * s[5], "B: and detected");
        check(at_prior(vm, {2, 3, 4}), "B: bpearl verticals carry NO tilt/z (the floor owns those)");
    }
}

// ─────────────────────────────── Part S: the agent's stage ───────────────────────────────
static void part_stage()
{
    std::printf("Part S: LidarMountStage (frames, throttle, pose-less sweeps)\n");
    // A 4 x 6 room centred on the origin; robot at (1, 0.5) facing +90 deg.
    const std::vector<Eigen::Vector2f> poly{{-2.f, -3.f}, {2.f, -3.f}, {2.f, 3.f}, {-2.f, 3.f}};
    Eigen::Affine3d T = Eigen::Affine3d::Identity();
    T.translate(Eigen::Vector3d(1.0, 0.5, 0.0));
    T.rotate(Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitZ()));
    const auto walls = sc::walls_in_robot(poly, T);
    check(walls.size() == 4, "S1 one wall per polygon edge");
    // room (2, -3) -> robot: R^T (p - t) with R = rot(90): (x', y') = (dy, -dx) = (-3.5, -1)
    check((walls[1].a - Eigen::Vector2f(-3.5f, -1.f)).norm() < 1e-5f, "S2 polygon vertex lands in the robot frame (inverse of room_T_robot)");
    check(sc::place_of(T) == rc::mountf::place_key(1.f, 0.5f, float(M_PI / 2)), "S3 place key from the pose (x, y, heading)");
    check(sc::walls_in_robot({{0.f, 0.f}, {1.f, 0.f}}, T).empty(), "S4 a degenerate polygon gives no walls");

    sc::LidarMountStageConfig cfg; cfg.rate_hz = 2.0; cfg.log_dir = "";
    sc::LidarMountStage st(cfg);
    sc::MountSweep sw; sw.sensor = "bpearl"; sw.origin_robot = Eigen::Vector3f(0.f, 0.14f, 0.7025f);
    for (int i = 0; i < 400; ++i) sw.points_robot.emplace_back(0.5f + 0.01f * i, 0.2f, 0.f);   // a flat floor patch
    sw.stamp_ms = 1'000'000;
    check(st.observe(sw, poly, 0.f), "S5 first sweep is fed");
    sw.stamp_ms += 100;
    check(not st.observe(sw, poly, 0.f), "S6 a sweep inside 1/rate_hz is skipped (compute budget)");
    sw.stamp_ms += 500;
    check(st.observe(sw, poly, 0.f), "S7 the next one past 1/rate_hz is fed");
    const auto s = st.state();
    check(s.sweeps_bpearl_floor == 2 and s.sweeps_bpearl_walls == 0, "S8 without a pose: floor rows only, no wall rows");
    sw.sensor = "helios"; sw.stamp_ms += 1000;
    check(st.observe(sw, poly, 0.f) and st.state().sweeps_helios == 0, "S9 helios without a pose feeds nothing");
    sw.sensor = "zed";
    check(not st.observe(sw, poly, 0.f), "S10 another sensor is ignored");
}

// ─────────────────────────────── Part O: an error planted through robot_concept's overlay ───────────────────────────────
// The Webots protocol (plan Phase 3 rulings): write a CORRECTION C into robot_concept/etc/mount_calib_<robot>_webots.txt.
// The graph then believes mount = compose(nominal, C) while Webots still mounts the sensor at the nominal pose, so
// every consumer sees the same wrong extrinsic. This part runs that chain end to end -- true points -> device frame
// through the TRUE (nominal) mount -> robot frame through the OVERLAID mount (rc::mount::compose, robot_concept's own
// code) -> the stage -- and pins the SIGN: the factors estimate the correction that RESTORES the truth, so they read
// -C (to first order), and the stage's sensor origin is the overlaid one.
static void part_overlay()
{
    using namespace syn;
    std::printf("Part O: error planted through the mount overlay -> the stage reads -C\n");
    const Room room;
    const std::vector<Eigen::Vector2f> poly{{float(-room.hx), float(-room.hy)}, {float(room.hx), float(-room.hy)},
                                            {float(room.hx), float(room.hy)}, {float(-room.hx), float(room.hy)}};
    struct Sensor { const char *name; V3 t_nom; rc::mount::Vec3 rpy_nom; rc::mount::Correction C; std::vector<double> elev; };
    // Nominal mounts in the Shadow frame (ROBOT_GEOMETRY.md), both hung inverted (Ry(pi)).
    rc::mount::Correction Ch; Ch.rpy = {0.4 * kDegD, -0.3 * kDegD, 0.0};                       // helios tilt
    rc::mount::Correction Cb; Cb.t = {0.010, 0.0, 0.010}; Cb.rpy = {0.5 * kDegD, -0.8 * kDegD, 1.0 * kDegD};   // bpearl
    const std::array<Sensor, 2> sensors{{{"helios", V3(0.0, -0.155, 1.1075), {0.0, M_PI, 0.0}, Ch, range(-15.5, 54.5, 2.0)},
                                         {"bpearl", V3(0.0, 0.14, 0.7025), {0.0, M_PI, 0.0}, Cb, range(-88.0, -2.0, 3.0)}}};
    sc::LidarMountStageConfig cfg; cfg.log_dir = "";
    sc::LidarMountStage st(cfg);
    std::mt19937 rng(11);
    std::uniform_real_distribution<double> ux(-1.0, 1.0), uth(-M_PI, M_PI);
    std::int64_t stamp = 1'000'000;
    for (int k = 0; k < 300; ++k)
    {
        const Pose P{ux(rng), 1.5 * ux(rng), uth(rng)};
        Eigen::Affine3d T = Eigen::Affine3d::Identity();
        T.translate(V3(P.x, P.y, 0.0)); T.rotate(Eigen::AngleAxisd(P.th, V3::UnitZ()));
        stamp += 600;   // past 1/rate_hz: every sweep is fed
        for (const auto &S : sensors)
        {
            const auto truth = sweep(room, P, S.t_nom, M6{}, S.elev, 1.5 * kDegD, 0.01, rng, std::string(S.name) == "helios");
            const Eigen::Matrix3d Rn = rc::mount::euler_to_matrix(S.rpy_nom);
            const rc::mount::Mount G = rc::mount::compose({S.t_nom.x(), S.t_nom.y(), S.t_nom.z()}, S.rpy_nom, S.C);
            const Eigen::Matrix3d Rg = rc::mount::euler_to_matrix(G.rpy);
            const V3 tg(G.t[0], G.t[1], G.t[2]);
            sc::MountSweep sw; sw.sensor = S.name; sw.stamp_ms = stamp; sw.origin_robot = tg.cast<float>(); sw.room_T_robot = T;
            for (const auto &p : truth)
            {
                const V3 p_dev = Rn.transpose() * (p.cast<double>() - S.t_nom);   // what the device reports
                sw.points_robot.emplace_back((Rg * p_dev + tg).cast<float>());      // what the graph makes of it
            }
            st.observe(sw, poly, float(room.h));
        }
    }
    const auto s = st.state();
    const Eigen::Vector3d fm = s.floor.mean(), fs = s.floor.sigma();
    const auto hm = s.vert_helios.mean(), hs = s.vert_helios.sigma();
    const auto bm = s.vert_bpearl.mean(), bs = s.vert_bpearl.sigma();
    std::printf("    %s\n", st.summary().c_str());
    auto near = [](double est, double sd, double want) { return std::abs(est - want) < std::max(3.0 * sd, 0.1 * std::abs(want)); };
    check(near(fm(0), fs(0), -Cb.rpy[0]) and near(fm(1), fs(1), -Cb.rpy[1]), "O1 bpearl floor roll/pitch read -C (planted +0.5 / -0.8 deg)");
    check(near(fm(2), fs(2), -Cb.t[2]), "O2 bpearl floor z reads -C (planted +10 mm)");
    check(near(hm(3), hs(3), -Ch.rpy[0]) and near(hm(4), hs(4), -Ch.rpy[1]), "O3 helios tilt reads -C (planted +0.4 / -0.3 deg)");
    check(near(bm(0), bs(0), -Cb.t[0]) and near(bm(5), bs(5), -Cb.rpy[2]), "O4 bpearl planar x / yaw read -C (planted +10 mm / +1 deg)");
}

int main(int argc, char **argv)
{
    std::setlocale(LC_ALL, "");   // CLAUDE.md: a harness that parses agent data must run under the agent's locale
    const std::string only = argc > 1 ? argv[1] : "";
    if (only.empty() or only == "2") part2();
    if (only.empty() or only == "3") part3();
    if (only.empty() or only == "P")
    // ── PLACE CELLS (fix A, 2026-10-08): repeated views of one place saturate; new places keep adding ──
    {
        std::printf("\nP. place-level common mode (PlaceCells)\n");
        using PC = rc::mountf::PlaceCells<3>;
        const Eigen::Vector3d psig(0.03, 0.03, 0.0087);           // place error: 3 cm, 3 cm, 0.5 deg
        const Eigen::Matrix3d Hs = Eigen::Vector3d(1.0 / (0.01 * 0.01), 1.0 / (0.01 * 0.01), 1.0 / (0.0035 * 0.0035)).asDiagonal();
        const Eigen::Vector3d place_err(0.048, -0.032, 0.006);    // the parked spot's own error (the live case)
        PC pc; pc.place_prec = psig.cwiseAbs2().cwiseInverse();
        for (int i = 0; i < 2461; ++i) pc.add(42, Hs, Hs * place_err, 1.0);
        for (int i = 0; i < 1000; ++i) pc.add(1000 + i % 200, Hs, Eigen::Vector3d::Zero(), 1.0);
        Eigen::Matrix3d H; Eigen::Vector3d b; pc.effective(H, b);
        const Eigen::Vector3d est = H.ldlt().solve(b), sd = H.inverse().diagonal().cwiseSqrt();
        std::printf("    one parked place + 200 others: est (%+.1f, %+.1f) mm (%+.2f deg), sd (%.1f, %.1f) mm\n",
                    est(0) * 1e3, est(1) * 1e3, est(2) * 180 / M_PI, sd(0) * 1e3, sd(1) * 1e3);
        check(std::abs(est(0)) < 0.002 and std::abs(est(1)) < 0.002, "P1 the parked place counts as ONE place (est ~ 0)");
        PC flat; flat.place_prec = Eigen::Vector3d::Constant(1e12);
        for (int i = 0; i < 2461; ++i) flat.add(42, Hs, Hs * place_err, 1.0);
        for (int i = 0; i < 1000; ++i) flat.add(1000 + i % 200, Hs, Eigen::Vector3d::Zero(), 1.0);
        flat.effective(H, b);
        const Eigen::Vector3d est0 = H.ldlt().solve(b);
        std::printf("    without the place nuisance: est (%+.1f, %+.1f) mm\n", est0(0) * 1e3, est0(1) * 1e3);
        check(est0(0) > 0.025, "P2 control: without it the parked view dominates (the live defect)");
        PC one; one.place_prec = psig.cwiseAbs2().cwiseInverse();
        for (int i = 0; i < 100000; ++i) one.add(7, Hs, Eigen::Vector3d::Zero(), 1.0);
        one.effective(H, b);
        check(H(0, 0) <= 1.0 / (0.03 * 0.03) * 1.0001, "P3 one place saturates at 1/sigma_place^2");
        PC f; f.place_prec = psig.cwiseAbs2().cwiseInverse();
        for (int i = 0; i < 5000; ++i) f.add(i, Hs, Eigen::Vector3d::Zero(), 0.99);
        check(f.size() < 5000, "P4 forgotten cells are pruned");
        check(rc::mountf::place_key(1.1f, 2.2f, 0.1f) == rc::mountf::place_key(1.2f, 2.3f, 0.2f), "P5 same cell, same key");
        check(rc::mountf::place_key(1.1f, 2.2f, 0.1f) != rc::mountf::place_key(1.1f, 2.2f, 1.7f), "P5 other heading octant, other key");
    }

    if (only.empty() or only == "S") part_stage();
    if (only.empty() or only == "O") part_overlay();

    std::printf("%s (%d failures)\n", failures ? "FAILED" : "ALL PASS", failures);
    return failures ? 1 : 0;
}
