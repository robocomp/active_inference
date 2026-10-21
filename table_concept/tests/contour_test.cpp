// contour_test — the table's contour existence channel against a ray-cast synthetic scene.
//
// Scores the SAME silhouette the agent does (table_contour.h) with the SAME scorers (common/contour_edge),
// on images rendered from a pinhole camera 1.2 m up, pitched down, looking at a floor + back wall with or
// without a table slab. The question each case answers: does the channel CONFIRM a table that is there,
// REFUTE one that is not, and stay confirming when the belief is a little off (pose error) or the table has
// the floor's own colour (depth alone must carry it)?
//
// Build: see CMakeLists.txt (target contour_test). Exit code 0 = every expectation held.
#include <cmath>
#include <cstdio>
#include <optional>
#include <random>
#include <string>

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include "table_contour.h"
#include "../../common/contour_edge/contour_edge_check.h"
#include "../../common/contour_edge/contour_depth_check.h"

namespace
{
constexpr int    kW = 640, kH = 360;
constexpr double kF = 360.0;                  // focal (px), ~83° HFoV
constexpr double kCamZ = 1.2, kPitch = 0.45;  // camera height (m), downward pitch (rad)

struct Camera
{
    Eigen::Vector3d C{0.0, 0.0, kCamZ};
    Eigen::Vector3d right{1, 0, 0}, fwd{0, std::cos(kPitch), -std::sin(kPitch)}, up{0, std::sin(kPitch), std::cos(kPitch)};
    std::optional<rc::edges::ProjectedVertex> project(const Eigen::Vector3d& P) const
    {
        const Eigen::Vector3d d = P - C;
        const double x = right.dot(d), y = fwd.dot(d), z = up.dot(d);
        if (y <= 0.20) return std::nullopt;
        rc::edges::ProjectedVertex v;
        v.px = cv::Point2f(static_cast<float>(kW / 2.0 + kF * x / y), static_cast<float>(kH / 2.0 - kF * z / y));
        v.depth_m = static_cast<float>(y);
        return v;
    }
};

struct World
{
    bool        has_table = true;
    rc::TableSlab table;
    int         table_grey = 200;            // tabletop intensity (floor tiles are ~70-130)
};

// Ray/slab intersection in the slab's local frame (the round table is approximated by its square for
// rendering only when round == false; for round we test the ellipse on the top/bottom planes + side).
double hit_slab(const rc::TableSlab& t, const Eigen::Vector3d& o, const Eigen::Vector3d& d)
{
    const double c = std::cos(-t.yaw), s = std::sin(-t.yaw);
    const auto loc = [&](const Eigen::Vector3d& p) { return Eigen::Vector3d(c * p.x() - s * p.y(), s * p.x() + c * p.y(), p.z()); };
    const Eigen::Vector3d ol = loc(o - Eigen::Vector3d(t.cx, t.cy, 0.0)), dl = loc(d);
    const double z0 = t.z_top - t.thickness, z1 = t.z_top, hw = 0.5 * t.w, hd = 0.5 * t.d;
    if (t.round)
    {
        double best = 1e9;
        for (const double z : {z0, z1})                           // caps
            if (std::abs(dl.z()) > 1e-9)
            {
                const double k = (z - ol.z()) / dl.z();
                const Eigen::Vector3d p = ol + k * dl;
                if (k > 0 and std::pow(p.x() / hw, 2) + std::pow(p.y() / hd, 2) <= 1.0) best = std::min(best, k);
            }
        for (int i = 0; i < 2000; ++i)                            // side: march (test-only, cheap enough)
        {
            const double k = 0.2 + i * 0.004;
            const Eigen::Vector3d p = ol + k * dl;
            if (p.z() >= z0 and p.z() <= z1 and std::pow(p.x() / hw, 2) + std::pow(p.y() / hd, 2) <= 1.0) { best = std::min(best, k); break; }
        }
        return best;
    }
    double tmin = -1e9, tmax = 1e9;
    const std::array<double, 3> lo{-hw, -hd, z0}, hi{hw, hd, z1};
    for (int a = 0; a < 3; ++a)
    {
        if (std::abs(dl[a]) < 1e-12) { if (ol[a] < lo[a] or ol[a] > hi[a]) return 1e9; continue; }
        double k1 = (lo[a] - ol[a]) / dl[a], k2 = (hi[a] - ol[a]) / dl[a];
        if (k1 > k2) std::swap(k1, k2);
        tmin = std::max(tmin, k1); tmax = std::min(tmax, k2);
    }
    return (tmax >= tmin and tmin > 0) ? tmin : 1e9;
}

void render(const Camera& cam, const World& w, cv::Mat& bgr, cv::Mat& depth, std::uint32_t seed)
{
    bgr.create(kH, kW, CV_8UC3);
    depth.create(kH, kW, CV_32F);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> tile(70, 130);
    std::vector<int> tiles(64 * 64);
    for (auto& v : tiles) v = tile(rng);
    std::normal_distribution<float> noise(0.0f, 3.0f), dnoise(0.0f, 1.0f);
    for (int r = 0; r < kH; ++r)
        for (int col = 0; col < kW; ++col)
        {
            const Eigen::Vector3d d = (cam.fwd + cam.right * ((col + 0.5 - kW / 2.0) / kF) - cam.up * ((r + 0.5 - kH / 2.0) / kF)).normalized();
            double k = 1e9; int grey = 0;
            if (d.z() < 0)                                           // floor
            {
                k = -cam.C.z() / d.z();
                const Eigen::Vector3d p = cam.C + k * d;
                const int ix = static_cast<int>(std::floor(p.x() / 0.25)) & 63, iy = static_cast<int>(std::floor(p.y() / 0.25)) & 63;
                grey = tiles[static_cast<std::size_t>(ix * 64 + iy)];
            }
            if (d.y() > 1e-6)                                        // back wall at y = 6
            {
                const double kw = (6.0 - cam.C.y()) / d.y();
                if (kw < k) { k = kw; grey = 150; }
            }
            if (w.has_table)
            {
                const double kt = hit_slab(w.table, cam.C, d);
                if (kt < k)
                {
                    const Eigen::Vector3d p = cam.C + kt * d;
                    k = kt; grey = (p.z() > w.table.z_top - 1e-3) ? w.table_grey : 50;   // top vs edge band
                }
            }
            const float g = std::clamp(grey + noise(rng), 0.0f, 255.0f);
            bgr.at<cv::Vec3b>(r, col) = cv::Vec3b(static_cast<uchar>(g), static_cast<uchar>(g), static_cast<uchar>(g));
            const double fwd_depth = (k * d).dot(cam.fwd);
            depth.at<float>(r, col) = k < 1e8 ? static_cast<float>(fwd_depth * (1.0 + 0.01 * dnoise(rng))) : 0.0f;
        }
}

struct Verdict { float edge = 0, depth = 0, total = 0; int ne = 0, nd = 0; bool measured = false; };

Verdict score(const Camera& cam, const rc::TableSlab& belief, const cv::Mat& bgr, const cv::Mat& depth)
{
    Verdict v;
    const auto cs = rc::table_contour_set(belief, Eigen::Vector2d(cam.C.x(), cam.C.y()),
                                          [&](const Eigen::Vector3d& P) { return cam.project(P); });
    if (not cs.face.valid()) return v;
    std::vector<std::vector<cv::Point>> ctl;
    for (const auto& c : cs.controls) ctl.push_back(c.px);
    const auto es = rc::edges::contour_edge_support(rc::edges::prepare_frame(bgr), cs.face.px, ctl);
    if (es.n_samples > 0 and es.n_controls > 0) { v.edge = es.excess; v.ne = es.n_samples; }
    rc::edges::ContourDepthParams dp;
    dp.sigma_depth_m = std::hypot(0.05f, rc::inward_depth_slack_m(cs.face, dp.offset_px_max));
    const auto ds = rc::edges::contour_depth_support(depth, cs.face, cs.controls, dp);
    if (ds.n_samples > 0) { v.depth = ds.verdict; v.nd = ds.n_samples; }
    v.total = v.edge + v.depth;
    v.measured = v.ne > 0 or v.nd > 0;
    return v;
}
}   // namespace

int main()
{
    const Camera cam;
    int failures = 0;
    const auto check = [&](const std::string& name, const World& w, const rc::TableSlab& belief, int sign)
    {
        cv::Mat bgr, depth;
        render(cam, w, bgr, depth, 7);
        const Verdict v = score(cam, belief, bgr, depth);
        const bool ok = v.measured and (sign > 0 ? v.total > 0.0f : v.total < 0.0f);
        failures += ok ? 0 : 1;
        std::printf("%-44s edge %+.3f (n %3d)  depth %+.3f (n %3d)  total %+.3f  tanh %+.3f  %s\n",
                    name.c_str(), v.edge, v.ne, v.depth, v.nd, v.total, std::tanh(v.total), ok ? "OK" : "FAIL");
    };

    rc::TableSlab sq;  sq.cx = 0.3;  sq.cy = 2.2; sq.yaw = 0.3; sq.w = 1.2; sq.d = 0.8; sq.z_top = 0.75; sq.thickness = 0.03;
    rc::TableSlab rd = sq; rd.round = true; rd.w = rd.d = 1.0; rd.yaw = 0.0;
    World present{true, sq}, absent{false, sq}, round_present{true, rd}, round_absent{false, rd};
    World camo{true, sq, 100};                                    // tabletop the floor's own grey

    check("square, present",                 present,       sq, +1);
    check("square, absent (blank floor)",    absent,        sq, -1);
    check("round, present",                  round_present, rd, +1);
    check("round, absent",                   round_absent,  rd, -1);
    check("square, floor-coloured top",      camo,          sq, +1);
    rc::TableSlab off = sq; off.cx += 0.08; off.cy -= 0.06; off.yaw += 0.08;
    check("square, belief 10 cm / 4.6 deg off",  present,   off, +1);
    rc::TableSlab far = sq; far.cy = 4.0;
    World far_present{true, far};
    check("square at 4 m, present",          far_present,   far, +1);
    check("square at 4 m, absent",           World{false, far}, far, -1);
    std::printf("%s (%d failure%s)\n", failures == 0 ? "PASS" : "FAIL", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
