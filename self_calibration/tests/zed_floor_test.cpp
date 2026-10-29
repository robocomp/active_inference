// zed_floor_test -- Task 4.1: the ZED camera's roll, pitch and z from the floor in its depth image. Synthetic: the
// notched 4x6 m room rendered by a small 3-D ray caster (floor z = 0, vertical walls from the polygon, a 0.7 m
// table top), no DDS/DSR. The camera sits 1.08 m up, pitched 25 deg down; the TRUE camera pose differs from the
// NOMINAL one the stage is told by the planted error, so the nominal-mount points carry the error E(p) = R(w)(p - t)
// + t + (0,0,dz) (w = (roll, pitch) about the ZED origin t, robot axes) and the stage must read the correction
// -(roll, pitch, z).
//   make -C build zed_floor_test && ../bin/zed_floor_test
#include "../src/zed_floor_stage.h"
#include "synthetic_room.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <numbers>
#include <random>

using namespace sc;
using namespace sc::calib::testing;

static int g_fail = 0;
#define CHECK(c, ...) do { if (not (c)) { ++g_fail; std::printf("FAIL %s:%d  %s  -- ", __FILE__, __LINE__, #c); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static constexpr double kPi = std::numbers::pi, kDeg = kPi / 180.0;
static constexpr std::uint32_t kFmtZ16 = 3, kFmtF32 = 4;

// ───────────────────────────────────── part 1: back-projection ─────────────────────────────────────
static ImageIn plane_image(int w, int h, double depth_m, bool f32)
{
    ImageIn im; im.sensor = "zed"; im.stream = "depth"; im.width = w; im.height = h;
    im.format = f32 ? kFmtF32 : kFmtZ16; im.step = w * (f32 ? 4 : 2);
    im.data.resize(std::size_t(im.step) * h);
    for (int v = 0; v < h; ++v)
        for (int u = 0; u < w; ++u)
        {
            if (f32) { const float d = float(depth_m); std::memcpy(&im.data[std::size_t(v) * im.step + u * 4], &d, 4); }
            else     { const std::uint16_t d = std::uint16_t(std::llround(depth_m * 1000.0)); std::memcpy(&im.data[std::size_t(v) * im.step + u * 2], &d, 2); }
        }
    return im;
}

static void part1()
{
    std::printf("Part 1: depth_to_points\n");
    const PinholeIntrinsics K{300, 300, 640, 360};
    for (const bool f32 : {true, false})
    {
        const auto pts = depth_to_points(plane_image(640, 360, 2.0, f32), K, 1);
        bool ok = pts.size() == 640u * 360u;
        double worst = 0;
        std::size_t i = 0;
        for (int v = 0; v < 360 and ok; ++v)
            for (int u = 0; u < 640; ++u, ++i)
            {
                const auto& p = pts[i];
                worst = std::max({worst, std::abs(p.y() - 2.0), std::abs(p.x() - (u - 320) * 2.0 / 300.0),
                                  std::abs(p.z() - (180 - v) * 2.0 / 300.0)});
            }
        CHECK(ok and worst < 1e-5, "%s plane Y=2 m: %zu points, worst error %.2e", f32 ? "F32" : "Z16", pts.size(), worst);
    }
    {   // 1280x720 depth against 640x360 intrinsics: fx, fy, cx, cy scale by 2, so the same ray direction per pixel pair
        const auto pts = depth_to_points(plane_image(1280, 720, 2.0, true), K, 1);
        CHECK(pts.size() == 1280u * 720u, "size %zu", pts.size());
        const auto& p = pts[std::size_t(100) * 1280 + 1000];   // u = 1000, v = 100
        CHECK(std::abs(p.x() - (1000 - 640) * 2.0 / 600.0) < 1e-5 and std::abs(p.z() - (360 - 100) * 2.0 / 600.0) < 1e-5,
              "scaled intrinsics: x %.5f z %.5f", p.x(), p.z());
    }
    {   // stride and invalid depths
        auto im = plane_image(64, 36, 1.0, true);
        const float nan = std::nanf(""), zero = 0.f;
        std::memcpy(&im.data[0], &nan, 4); std::memcpy(&im.data[4], &zero, 4);
        const auto pts = depth_to_points(im, PinholeIntrinsics{30, 30, 64, 36}, 2);
        CHECK(pts.size() == 32u * 18u - 1u, "stride 2 skips the NaN pixel (u=0) and keeps the zero's neighbour: %zu", pts.size());
    }
}

// ───────────────────────────────────── part 2: the planted floor ─────────────────────────────────────
// A tiny 3-D scene in the ROOM frame: the floor, the room walls to 2.5 m, an axis-aligned table top (and its sides).
struct Scene
{
    Room2D room = notched_room();
    bool walls = false;
    Eigen::Vector3d tmin{0.6, -0.5, 0.0}, tmax{1.4, 0.3, 0.7};   // the table box
    bool table = false;
};

// Distance t along (o + t d) to the nearest hit; d's zed-Y component is 1, so t IS the depth value.
static double cast(const Scene& s, const Eigen::Vector3d& o, const Eigen::Vector3d& d)
{
    double best = std::numeric_limits<double>::infinity();
    if (d.z() < -1e-12) best = -o.z() / d.z();                                    // the floor
    if (s.walls)
        for (std::size_t i = 0; i < s.room.poly.size(); ++i)
        {
            const Eigen::Vector2d p = s.room.poly[i], e = s.room.poly[(i + 1) % s.room.poly.size()] - p;
            const Eigen::Vector2d dd = d.head<2>(), w = p - o.head<2>();
            const double den = dd.x() * e.y() - dd.y() * e.x();
            if (std::abs(den) < 1e-12) continue;
            const double t = (w.x() * e.y() - w.y() * e.x()) / den, sp = (w.x() * dd.y() - w.y() * dd.x()) / den;
            const double z = o.z() + t * d.z();
            if (t > 1e-9 and sp >= 0.0 and sp <= 1.0 and z >= 0.0 and z <= 2.5) best = std::min(best, t);
        }
    if (s.table)   // slab test
    {
        double t0 = 1e-9, t1 = std::numeric_limits<double>::infinity();
        for (int a = 0; a < 3; ++a)
        {
            if (std::abs(d[a]) < 1e-12) { if (o[a] < s.tmin[a] or o[a] > s.tmax[a]) { t1 = -1; break; } continue; }
            double ta = (s.tmin[a] - o[a]) / d[a], tb = (s.tmax[a] - o[a]) / d[a];
            if (ta > tb) std::swap(ta, tb);
            t0 = std::max(t0, ta); t1 = std::min(t1, tb);
        }
        if (t0 <= t1) best = std::min(best, t0);
    }
    return best;
}

static Eigen::Affine3d nominal_mount()   // robot_from_zed: 1.08 m up, 0.1 m forward, pitched 25 deg down about X
{
    Eigen::Affine3d T = Eigen::Affine3d::Identity();
    T.translation() = Eigen::Vector3d(0.0, 0.1, 1.08);
    T.linear() = Eigen::AngleAxisd(-25.0 * kDeg, Eigen::Vector3d::UnitX()).toRotationMatrix();
    return T;
}

// The TRUE mount: the nominal one with E^-1 applied (E as in the file header), so Mnom * Mtrue^-1 = E.
static Eigen::Affine3d true_mount(const Eigen::Vector3d& w, double dz)
{
    const Eigen::Affine3d Mn = nominal_mount();
    const Eigen::Vector3d t = Mn.translation();
    Eigen::Affine3d Einv = Eigen::Affine3d::Identity();
    const double ang = w.norm();
    const Eigen::Matrix3d R = ang > 0 ? Eigen::AngleAxisd(ang, w / ang).toRotationMatrix() : Eigen::Matrix3d::Identity();
    Einv.linear() = R.transpose();
    Einv.translation() = t - R.transpose() * (t + Eigen::Vector3d(0, 0, dz));
    return Einv * Mn;
}

static ImageIn render(const Scene& sc, const Eigen::Vector3d& pose, const Eigen::Affine3d& robot_from_zed,
                      const PinholeIntrinsics& K, double rel_noise, std::mt19937& rng, std::int64_t stamp)
{
    ImageIn im; im.sensor = "zed"; im.stream = "depth"; im.width = unsigned(K.width); im.height = unsigned(K.height);
    im.format = kFmtF32; im.step = im.width * 4; im.stamp_ms = stamp;
    im.data.resize(std::size_t(im.step) * im.height);
    Eigen::Affine3d room_T_robot = Eigen::Affine3d::Identity();
    room_T_robot.linear() = Eigen::AngleAxisd(pose.z(), Eigen::Vector3d::UnitZ()).toRotationMatrix();
    room_T_robot.translation() = Eigen::Vector3d(pose.x(), pose.y(), 0.0);
    const Eigen::Affine3d W = room_T_robot * robot_from_zed;
    std::normal_distribution<double> nd(0.0, 1.0);
    for (unsigned v = 0; v < im.height; ++v)
        for (unsigned u = 0; u < im.width; ++u)
        {
            const Eigen::Vector3d dz((u - 0.5 * K.width) / K.fx, 1.0, (0.5 * K.height - v) / K.fy);
            double t = cast(sc, W.translation(), W.linear() * dz);
            float d = std::isfinite(t) ? float(t * (1.0 + rel_noise * nd(rng))) : std::nanf("");
            std::memcpy(&im.data[std::size_t(v) * im.step + u * 4], &d, 4);
        }
    return im;
}

struct Result { Eigen::Vector3d m, s; long n = 0; };

static Result run(const Scene& sc, const Eigen::Vector3d& w, double dz, unsigned seed, int nposes = 30)
{
    const PinholeIntrinsics K{300, 300, 640, 360};
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> ux(-1.3, 1.3), uy(-2.0, 2.0), uth(-kPi, kPi);
    ZedFloorStage::Config cfg; cfg.rate_hz = 0.0; cfg.stride = 4; cfg.log_dir = "";
    ZedFloorStage st(cfg);
    const Eigen::Affine3d Mn = nominal_mount(), Mt = true_mount(w, dz);
    std::int64_t stamp = 1'000'000;
    for (int k = 0; k < nposes;)
    {
        const Eigen::Vector3d P(ux(rng), uy(rng), uth(rng));
        if (sc.table and P.x() > sc.tmin.x() - 0.4 and P.x() < sc.tmax.x() + 0.4 and P.y() > sc.tmin.y() - 0.4 and P.y() < sc.tmax.y() + 0.4)
            continue;   // not inside the table
        ImageIn im = render(sc, P, Mt, K, 0.002, rng, stamp += 500);
        im.robot_from_sensor = Eigen::Affine3f(Mn.cast<float>());
        Eigen::Affine3d room_T_robot = Eigen::Affine3d::Identity();
        room_T_robot.linear() = Eigen::AngleAxisd(P.z(), Eigen::Vector3d::UnitZ()).toRotationMatrix();
        room_T_robot.translation() = Eigen::Vector3d(P.x(), P.y(), 0.0);
        st.observe(im, K, room_T_robot);
        ++k;
    }
    std::printf("    %s\n", st.summary().c_str());
    return {st.info().mean(), st.info().sigma(), st.frames()};
}

static void check_recovered(const char* tag, const Result& r, const Eigen::Vector3d& w, double dz)
{
    // 2 sigma PLUS 1 % of the plant: the rows are first order in the rotation and the plant is an exact rotation.
    const Eigen::Vector3d truth(-w.x(), -w.y(), -dz);
    const double lin = 0.01 * w.norm();
    CHECK(std::abs(r.m[0] - truth[0]) < 2 * r.s[0] + lin, "%s roll %+.4f +- %.4f deg, want %+.3f", tag, r.m[0] / kDeg, r.s[0] / kDeg, truth[0] / kDeg);
    CHECK(std::abs(r.m[1] - truth[1]) < 2 * r.s[1] + lin, "%s pitch %+.4f +- %.4f deg, want %+.3f", tag, r.m[1] / kDeg, r.s[1] / kDeg, truth[1] / kDeg);
    CHECK(std::abs(r.m[2] - truth[2]) < 2 * r.s[2] + 0.01 * std::abs(dz), "%s dz %+.2f +- %.2f mm, want %+.1f", tag, r.m[2] * 1e3, r.s[2] * 1e3, truth[2] * 1e3);
    // informative: not a vacuous 2 sigma
    CHECK(r.s[0] < 0.25 * kDeg and r.s[1] < 0.25 * kDeg and r.s[2] < 0.01, "%s sigma not tight: %.3f %.3f deg %.2f mm", tag, r.s[0] / kDeg, r.s[1] / kDeg, r.s[2] * 1e3);
}

static void part2()
{
    std::printf("Part 2: planted_floor (floor only)\n");
    const Eigen::Vector3d w(0.6 * kDeg, 1.0 * kDeg, 0.0);
    Scene sc;
    for (const unsigned seed : {1u, 2u, 3u})
        check_recovered("floor", run(sc, w, 0.010, seed), w, 0.010);
    // the control: no planted error reads back ~0
    const auto r0 = run(sc, Eigen::Vector3d::Zero(), 0.0, 4);
    CHECK(std::abs(r0.m[0]) < 2 * r0.s[0] + 1e-4 and std::abs(r0.m[1]) < 2 * r0.s[1] + 1e-4 and std::abs(r0.m[2]) < 2 * r0.s[2] + 1e-3,
          "no-error control reads %+.4f %+.4f deg %+.2f mm", r0.m[0] / kDeg, r0.m[1] / kDeg, r0.m[2] * 1e3);
}

static void part3()
{
    std::printf("Part 3: walls_do_not_bias (walls + a 0.7 m table in view)\n");
    const Eigen::Vector3d w(0.6 * kDeg, 1.0 * kDeg, 0.0);
    Scene sc; sc.walls = true; sc.table = true;
    for (const unsigned seed : {1u, 2u, 3u})
        check_recovered("walls+table", run(sc, w, 0.010, seed), w, 0.010);
}

int main()
{
    part1(); part2(); part3();
    std::printf("%s (%d failures)\n", g_fail ? "FAILED" : "PASSED", g_fail);
    return g_fail ? 1 : 0;
}
