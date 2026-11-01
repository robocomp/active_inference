// wall_gap_replay.cpp — the "wall, then the hole" door estimator (src/door_wall_gap.h) run OFFLINE on a door_concept frame
// record (etc/anchor_frames.bin) and scored against the Webots truth (gt_error.csv), frame by frame (spec 2026-11-01).
//
// Per anchor (a record gap > 2 s starts a new one): the pose starts at the anchor's CREATION prior (the room frame's
// opening, as live); between records the robot's motion is recovered from the record (prior_k = move(post_{k-1})), as
// replay_anchor_frames --chain does, and moves both the pose and THE MEMORY: every point seen during the approach, kept in
// the robot frame, moved by that motion, thinned on a 3 cm grid (user 2026-11-01: "the lack of memory ... makes the result
// brittle"). Each frame measures the wall and the hole on memory + the new points and fuses them with the moved pose.
//
//   Build + run (from door_concept/):
//     Q=/usr/include/x86_64-linux-gnu/qt5; g++ -std=c++23 -O2 -fPIC -I/usr/include/eigen3 -Isrc -I../common -I$Q -I$Q/QtCore
//       -I$Q/QtGui -I$Q/QtWidgets tests/wall_gap_replay.cpp -o build/wall_gap_replay -lfastdds -lfastcdr -lQt5Core
//     ./build/wall_gap_replay RUN/anchor_frames.bin RUN/gt_error.csv [--no-memory] [--lintel H]
#include "../src/anchor_frame_record.h"
#include "../src/door_wall_gap.h"

#include <charconv>
#include <cstdlib>
#include <random>
#include <cstring>
#include <fstream>
#include <map>
#include <print>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace
{
bool num(const char* s, double& v) { const auto r = std::from_chars(s, s + std::strlen(s), v); return r.ec == std::errc(); }
struct GtRow { std::int64_t t; double x, y, th; };
std::vector<GtRow> read_gt(const std::string& path)   // as replay_anchor_frames: robot world yaw = -gt_theta_raw - pi/2
{
    std::vector<GtRow> out; std::ifstream f(path); std::string line;
    if (not std::getline(f, line)) return out;
    std::vector<std::string> head; { std::stringstream ss(line); std::string c; while (std::getline(ss, c, ',')) head.push_back(c); }
    auto col = [&](const char* n) { return int(std::find(head.begin(), head.end(), n) - head.begin()); };
    const int it = col("ts_ms"), ix = col("gt_x_world"), iy = col("gt_y_world"), ith = col("gt_theta_raw");
    while (std::getline(f, line))
    {
        std::vector<std::string> c; std::stringstream ss(line); std::string v; while (std::getline(ss, v, ',')) c.push_back(v);
        if (int(c.size()) <= std::max({it, ix, iy, ith})) continue;
        GtRow r{}; double t = 0, raw = 0;
        if (num(c[it].c_str(), t) and num(c[ix].c_str(), r.x) and num(c[iy].c_str(), r.y) and num(c[ith].c_str(), raw))
        { r.t = std::int64_t(t); r.th = -raw - M_PI / 2; out.push_back(r); }
    }
    return out;
}
std::optional<GtRow> gt_at(const std::vector<GtRow>& gt, std::int64_t t)
{
    auto it = std::lower_bound(gt.begin(), gt.end(), t, [](const GtRow& r, std::int64_t v) { return r.t < v; });
    if (it == gt.end() or it == gt.begin()) return std::nullopt;
    const GtRow& g1 = *it; const GtRow& g0 = *std::prev(it);
    if (g1.t - g0.t > 200 or g1.t == g0.t) return std::nullopt;
    const double a = double(t - g0.t) / double(g1.t - g0.t);
    return GtRow{t, g0.x + a * (g1.x - g0.x), g0.y + a * (g1.y - g0.y), g0.th + a * std::remainder(g1.th - g0.th, 2 * M_PI)};
}
constexpr double kDoorX = -4.77, kDoorY = 9.425;   // DOOR_1's opening on the wall's mid-plane (replay_anchor_frames)
// error of an anchor estimate against the truth: depth / lateral (m) in the TRUE anchor's axes, yaw (rad), and the robot's
// true distance to the plane (the anchor's axes may be flipped: compare modulo pi)
struct Err { double depth, lat, yaw, dist; };
std::optional<Err> err_at(const std::vector<GtRow>& gt, std::int64_t t, const rc::interaction::RelPose& est)
{
    const auto g = gt_at(gt, t);
    if (not g.has_value()) return std::nullopt;
    const double c = std::cos(g->th), s = std::sin(g->th), dx = kDoorX - g->x, dy = kDoorY - g->y;
    const Eigen::Vector2d p_true(c * dx + s * dy, -s * dx + c * dy);
    double yaw_true = std::remainder(-g->th, 2 * M_PI);
    if (std::abs(std::remainder(double(est.x.z()) - yaw_true, 2 * M_PI)) > M_PI / 2) yaw_true += M_PI;
    const Eigen::Vector2d a(std::cos(yaw_true), std::sin(yaw_true)), n(-a.y(), a.x());
    const Eigen::Vector2d e = est.x.head<2>().cast<double>() - p_true;
    return Err{e.dot(n), e.dot(a), std::remainder(double(est.x.z()) - yaw_true, 2 * M_PI), std::abs(p_true.dot(n))};
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc < 3) { std::print(stderr, "usage: wall_gap_replay anchor_frames.bin gt_error.csv [--no-memory] [--lintel H]\n"); return 2; }
    bool memory = true, room_yaw = false;
    double room_yaw_sigma_deg = 2.0;
    std::mt19937 rng(17);
    std::normal_distribution<float> yaw_noise(0.f, 1.f * float(M_PI) / 180.f);   // the localiser's heading error (~1 deg)
    rc::door::wallgap::Params pp;
    for (int i = 3; i < argc; ++i)
    {
        double v;
        if (std::strcmp(argv[i], "--no-memory") == 0) memory = false;
        else if (std::strcmp(argv[i], "--room-yaw") == 0) room_yaw = true;
        else if (std::strcmp(argv[i], "--room-yaw-sigma") == 0 and i + 1 < argc and num(argv[++i], v)) room_yaw_sigma_deg = v;
        else if (std::strcmp(argv[i], "--lintel") == 0 and i + 1 < argc and num(argv[++i], v)) pp.lintel = float(v);
    }
    std::ifstream in(argv[1], std::ios::binary);
    std::vector<rc::door::rec::Frame> frames; std::uint32_t version = 0;
    if (not rc::door::rec::read_all(in, frames, &version)) { std::print(stderr, "{}: not a frame record\n", argv[1]); return 2; }
    const auto gt = read_gt(argv[2]);
    std::print("{}: {} frames (v{}), memory {}, lintel {:.2f} m\n", argv[1], frames.size(), version, memory ? "ON" : "OFF", pp.lintel);
    rc::interaction::RelPose x, live;
    std::vector<Eigen::Vector3f> mem;
    int anchor = 0;
    struct Bin { std::vector<double> d, l, y, dl, ll, yl; };
    std::map<int, std::map<int, Bin>> bins;   // anchor -> distance bin (0.4 m) -> errors (new, live)
    const std::int64_t t0 = frames.empty() ? 0 : frames.front().t_ms;
    for (std::size_t k = 0; k < frames.size(); ++k)
    {
        const auto& f = frames[k];
        const bool restart = k == 0 or f.t_ms - frames[k - 1].t_ms > 2000;
        if (restart)
        {
            ++anchor; x = f.prior; mem.clear();
            std::print("anchor {} at t {:.1f} s: creation prior ({:+.2f},{:+.2f}) yaw {:+.1f} deg\n", anchor, (f.t_ms - t0) * 1e-3,
                       f.prior.x.x(), f.prior.x.y(), f.prior.x.z() * 57.2958f);
        }
        else
        {
            // the robot's motion between the records (side, adv, dtheta), recovered as replay_anchor_frames does
            const auto& a = frames[k - 1].post; const auto& b = f.prior;
            const float dth = rc::interaction::wrap_pi(a.x.z() - b.x.z());
            const float c = std::cos(dth), sn = std::sin(dth);
            const float side = a.x.x() - (c * b.x.x() - sn * b.x.y()), adv = a.x.y() - (sn * b.x.x() + c * b.x.y());
            Eigen::Matrix3f J, G;
            rc::interaction::predict_jacobians(a.x, side, adv, dth, J, G);
            const Eigen::Matrix3f Q = b.P - J * a.P * J.transpose();
            rc::interaction::predict_jacobians(x.x, side, adv, dth, J, G);
            x.x = rc::interaction::move_mean(x.x, side, adv, dth);
            x.P = J * x.P * J.transpose() + Q;
            for (auto& p : mem) { const Eigen::Vector3f q = rc::interaction::move_mean(Eigen::Vector3f(p.x(), p.y(), 0.f), side, adv, dth); p.x() = q.x(); p.y() = q.y(); }
        }
        // the points the agent's wall-then-hole saw: the staged bpearl rays, plus (record v5) the Helios returns above the lintel
        std::vector<Eigen::Vector3f> rays = f.staging.prior.rays;
        rays.insert(rays.end(), f.helios.begin(), f.helios.end());
        if (memory)
        {
            mem.insert(mem.end(), rays.begin(), rays.end());
            std::unordered_set<long long> seen; std::vector<Eigen::Vector3f> thin; thin.reserve(mem.size());
            for (auto it = mem.rbegin(); it != mem.rend(); ++it)   // newest first: a cell keeps its latest return
            {
                const long long key = ((long long)std::floor(it->x() / 0.03f) * 73856093LL) ^ ((long long)std::floor(it->y() / 0.03f) * 19349663LL)
                                      ^ ((long long)std::floor(it->z() / 0.03f) * 83492791LL);
                if (seen.insert(key).second) thin.push_back(*it);
            }
            mem.swap(thin);
        }
        pp.width = f.width > 0.f ? f.width : pp.width;
        pp.thickness = f.thickness > 0.f ? f.thickness : pp.thickness;
        // --room-yaw: the agent's room-given wall direction (anchor_from_room), emulated: the TRUE wall direction in the
        // robot frame + the localiser's heading error, sigma 2 deg — reset every frame, as door_concept now does
        if (room_yaw)
            if (const auto g = gt_at(gt, f.t_ms); g.has_value())
            {
                float yaw = float(std::remainder(-g->th, 2 * M_PI)) + yaw_noise(rng);
                if (std::abs(rc::interaction::wrap_pi(yaw - x.x.z())) > float(M_PI_2)) yaw = rc::interaction::wrap_pi(yaw + float(M_PI));
                x.x.z() = yaw; x.P(2, 2) = float(std::pow(room_yaw_sigma_deg * M_PI / 180.0, 2.0));
            }
        const auto m = rc::door::wallgap::measure(x, memory ? mem : rays, pp);
        if (std::getenv("WG_DEBUG") != nullptr and anchor == std::atoi(std::getenv("WG_DEBUG")) and k % 15 == 0)
        {
            int n_above = 0; for (const auto& p : (memory ? mem : rays)) n_above += p.z() > pp.lintel ? 1 : 0;
            const auto e0 = err_at(gt, f.t_ms, x);
            std::print("  dbg t {:.1f}s dist {:.2f}: wall {} d_face {:+.3f} n_in {} slope {:+.3f} | gap {} [{:+.2f},{:+.2f}] n_plane {} | pts above lintel {} of {} | prior err depth {:+.2f} lat {:+.2f}\n",
                       (f.t_ms - t0) * 1e-3, e0 ? e0->dist : -1.0, m.wall, m.d_face, m.n_wall, m.slope, m.gap, m.u_left, m.u_right, m.n_plane,
                       n_above, (memory ? mem : rays).size(), e0 ? e0->depth : 0.0, e0 ? e0->lat : 0.0);
        }
        x = rc::door::wallgap::fuse(x, m, pp);
        const auto e = err_at(gt, f.t_ms, x), el = err_at(gt, f.t_ms, f.post);
        if (e.has_value() and el.has_value())
        {
            auto& bn = bins[anchor][int(e->dist / 0.4)];
            bn.d.push_back(e->depth); bn.l.push_back(e->lat); bn.y.push_back(e->yaw);
            bn.dl.push_back(el->depth); bn.ll.push_back(el->lat); bn.yl.push_back(el->yaw);
        }
    }
    auto med = [](std::vector<double> v) { if (v.empty()) return std::nan(""); std::ranges::sort(v); return v[v.size() / 2]; };
    auto amed = [&](const std::vector<double>& v) { std::vector<double> a; for (double x : v) a.push_back(std::abs(x)); return med(a); };
    for (const auto& [an, bs] : bins)
    {
        std::print("anchor {}: |error| medians, NEW (wall, then hole) vs LIVE (the record's posterior)\n", an);
        for (const auto& [bi, bn] : bs)
            std::print("   {:.1f}-{:.1f} m from the plane, n {:4d}: depth {:.3f} vs {:.3f}  lateral {:.3f} vs {:.3f}  yaw {:5.1f} vs {:5.1f} deg\n",
                       bi * 0.4, bi * 0.4 + 0.4, bn.d.size(), amed(bn.d), amed(bn.dl), amed(bn.l), amed(bn.ll),
                       amed(bn.y) * 57.2958, amed(bn.yl) * 57.2958);
    }
    return 0;
}
