// reloc_handover_replay.cpp — the relocaliser's search OFFLINE on the frames room_concept dumps after a hand-over
// (tmp/reloc_doorway_points.csv: the room outline once, then every high-band return in the robot frame per frame), ROOM-WIDE
// (lattice maxima over the whole polygon) vs THE ENTRY AREA (Params::local_seeds, from the door anchor), each scored
// against a truth file (ts_ms,x,y,theta in the same room frame). 2026-11-01, pingpong39 leg 6: the room-wide search took a
// 4.35 m / 179° twin one frame after a recall placed 8° off.
//
//   g++ -std=c++23 -O2 -I/usr/include/eigen3 -Isrc tools/reloc_handover_replay.cpp src/reloc_search.cpp src/wall_segmenter.cpp \
//       -o build/reloc_handover_replay
//   build/reloc_handover_replay DUMP.csv HANDOVER_MS TRUTH.csv  ax ay bx by  s_in width
//     a, b: the crossed door's mouth (room frame); s_in: how far past the mouth the body was at the hand-over (m); width (m).
// The door anchor is built as passage_prior.h's anchor_at_handover does (forward + reversed, yaw spread from the body fit)
// and carried by the TRUTH's motion (= perfect odometry). Mode choice: evidence + log A(x), A = (1 − p_kid)·Σ w N + p_kid·U.
#include "reloc_search.h"

#include <charconv>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <map>
#include <print>
#include <sstream>
#include <string>
#include <vector>

namespace
{
bool num(const std::string& s, double& v) { const auto r = std::from_chars(s.data(), s.data() + s.size(), v); return r.ec == std::errc(); }
std::vector<std::string> split(const std::string& l) { std::vector<std::string> c; std::stringstream ss(l); std::string v; while (std::getline(ss, v, ',')) c.push_back(v); return c; }
float wrap(float a) { return std::remainder(a, 2.f * float(M_PI)); }
struct Anc { Eigen::Vector3f pose; Eigen::Matrix3f cov; float w; };
Eigen::Vector3f compose(const Eigen::Vector3f& a, const Eigen::Vector3f& u)
{
    const float c = std::cos(a.z()), s = std::sin(a.z());
    return {a.x() + c * u.x() - s * u.y(), a.y() + s * u.x() + c * u.y(), wrap(a.z() + u.z())};
}
Eigen::Vector3f between(const Eigen::Vector3f& a, const Eigen::Vector3f& b)   // a⁻¹ ∘ b
{
    const float c = std::cos(a.z()), s = std::sin(a.z());
    const Eigen::Vector2f d = b.head<2>() - a.head<2>();
    return {c * d.x() + s * d.y(), -s * d.x() + c * d.y(), wrap(b.z() - a.z())};
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc < 10) { std::print(stderr, "usage: reloc_handover_replay DUMP HANDOVER_MS TRUTH ax ay bx by s_in width\n"); return 2; }
    double v[6];
    for (int i = 0; i < 6; ++i) if (not num(argv[4 + i], v[i])) { std::print(stderr, "bad number {}\n", argv[4 + i]); return 2; }
    const std::string ho = argv[2];
    const Eigen::Vector2f A(v[0], v[1]), B(v[2], v[3]);
    const float s_in = float(v[4]), width = float(v[5]);

    std::vector<Eigen::Vector2f> poly;
    std::map<long long, std::vector<Eigen::Vector2f>> frames;
    std::map<long long, Eigen::Vector3f> live_pose;
    {
        std::ifstream f(argv[1]); std::string l; std::getline(f, l);
        while (std::getline(f, l))
        {
            const auto c = split(l);
            if (c.size() < 12 or c[1] != ho) continue;
            double t, x, y, px, py, pt;
            if (not (num(c[0], t) and num(c[4], x) and num(c[5], y) and num(c[9], px) and num(c[10], py) and num(c[11], pt))) continue;
            if (c[2] == "poly") poly.emplace_back(x, y);
            else { frames[(long long)t].emplace_back(x, y); live_pose[(long long)t] = {float(px), float(py), float(pt)}; }
        }
    }
    std::map<long long, Eigen::Vector3f> truth;
    {
        std::ifstream f(argv[3]); std::string l; std::getline(f, l);
        while (std::getline(f, l)) { const auto c = split(l); double t, x, y, th; if (c.size() >= 4 and num(c[0], t) and num(c[1], x) and num(c[2], y) and num(c[3], th)) truth[(long long)t] = {float(x), float(y), float(th)}; }
    }
    auto truth_at = [&](long long t) -> std::optional<Eigen::Vector3f> { auto it = truth.lower_bound(t); if (it == truth.end()) return std::nullopt; return it->second; };
    std::print("{} frames, polygon {} vertices, truth {} rows\n", frames.size(), poly.size(), truth.size());
    if (frames.empty() or poly.size() < 3) return 1;

    // the anchor at the hand-over (passage_prior.h anchor_at_handover, body 0.46 x 0.48 m, sig_t = sig_s = 0.05 m)
    const Eigen::Vector2f tg = (B - A).normalized(), c0 = 0.5f * (A + B);
    Eigen::Vector2f n_in(-tg.y(), tg.x());
    double area2 = 0; for (std::size_t i = 0, j = poly.size() - 1; i < poly.size(); j = i++) area2 += double(poly[j].x()) * poly[i].y() - double(poly[i].x()) * poly[j].y();
    Eigen::Vector2f cen = Eigen::Vector2f::Zero(); for (const auto& p : poly) cen += p; cen /= float(poly.size());
    if (n_in.dot(cen - c0) < 0.f) n_in = -n_in;   // into the room
    const float th_in = std::atan2(-n_in.x(), n_in.y());
    const float play = std::max(0.f, width - 0.46f), psi = std::atan2(play, 0.48f);
    Eigen::Matrix2f R; R.col(0) = tg; R.col(1) = n_in;
    Anc fwd{{0, 0, 0}, Eigen::Matrix3f::Zero(), 1.f};
    const Eigen::Vector2f xy = c0 + n_in * s_in;
    fwd.pose = {xy.x(), xy.y(), th_in};
    fwd.cov.topLeftCorner<2, 2>() = R * Eigen::Vector2f(play * play / 12.f + 0.0025f, 0.0025f).asDiagonal() * R.transpose();
    fwd.cov(2, 2) = 0.25f * psi * psi;
    Anc rev = fwd; rev.pose.z() = wrap(th_in + float(M_PI));
    const auto t_first = truth_at(frames.begin()->first);
    // heading at the crossing ⇒ the components' weights (the truth's here; live: the localiser's)
    fwd.w = std::abs(wrap(t_first->z() - th_in)) < float(M_PI_2) ? 1.f : 0.f; rev.w = 1.f - fwd.w;
    std::print("anchor: ({:+.2f},{:+.2f}) yaw {:+.1f} deg, sigma yaw {:.1f} deg, w fwd {:.0f}\n", fwd.pose.x(), fwd.pose.y(),
               fwd.pose.z() * 57.2958f, std::sqrt(fwd.cov(2, 2)) * 57.2958f, fwd.w);

    rc::reloc::Params rp;
    rp.sigma_sdf = 0.15f; rp.outlier_frac = 0.30f; rp.wall_offset_sigma = 0.03f; rp.body_clearance = 0.30f; rp.max_points = 150;
    {
        rc::reloc::Aperture ap; ap.a = A; ap.b = B; ap.n_out = -n_in; ap.open = std::getenv("AP_OPEN") ? std::strtof(std::getenv("AP_OPEN"), nullptr) : 0.9f; ap.p_shut = std::getenv("AP_SHUT") ? std::strtof(std::getenv("AP_SHUT"), nullptr) : 0.05f; ap.sigma_along = 0.05f;
        if (std::getenv("NO_APERTURE") == nullptr) rp.apertures = {ap};
    }
    rp.beyond_extent = 20.f;
    rc::reloc::LikelihoodRaster raster;
    raster.build(poly, rp);
    const double log_u = -std::log(0.5 * std::abs(area2) * 2.0 * M_PI);
    int frame_no = 0;
    for (const auto& [t, pts] : frames)
    {
        ++frame_no;
        const auto tr = truth_at(t);
        if (not tr.has_value()) continue;
        // the anchor carried by the truth's motion since the first frame
        const Eigen::Vector3f u = between(*t_first, *tr);
        std::vector<Anc> anc = {fwd, rev};
        for (auto& a : anc) a.pose = compose(a.pose, u);
        const double p_kid = 1.0 - std::pow(1.0 - 1e-3, double(frame_no * 2 + 1));   // frames at 20 Hz, dump every ~60 ms
        auto log_a_rel = [&](const rc::reloc::Mode& m)
        {
            double ld = -INFINITY;
            for (const auto& a : anc)
            {
                if (not (a.w > 0.f)) continue;
                Eigen::Vector3f d = m.pose - a.pose; d.z() = wrap(d.z());
                const Eigen::LDLT<Eigen::Matrix3f> L(Eigen::Matrix3f(a.cov + m.cov));
                const double lc = std::log(a.w) - 0.5 * d.dot(L.solve(d)) - 0.5 * (3 * std::log(2 * M_PI) + L.vectorD().array().abs().log().sum());
                const double h = std::max(ld, lc); ld = h + std::log(std::exp(ld - h) + std::exp(lc - h));
            }
            const double x1 = std::log1p(-p_kid) + ld, x2 = std::log(p_kid) + log_u, h = std::max(x1, x2);
            return h + std::log(std::exp(x1 - h) + std::exp(x2 - h)) - log_u;
        };
        for (const bool local : {false, true})
        {
            rc::reloc::Params p = rp;
            if (local)
                for (const auto& a : anc)
                {
                    if (not (a.w > 0.f)) continue;
                    const float sth = std::sqrt(a.cov(2, 2));
                    for (const int k : {-2, -1, 0, 1, 2}) p.local_seeds.emplace_back(a.pose.x(), a.pose.y(), wrap(a.pose.z() + float(k) * sth));
                }
            auto res = rc::reloc::search(raster, poly, pts, {live_pose.at(t)}, p, std::uint32_t(t & 0xffffffff));
            int best = -1; double best_lp = -INFINITY;
            for (int i = 0; i < int(res.modes.size()); ++i)
            {
                const double lp = double(rc::reloc::log_evidence(res.modes[i])) + log_a_rel(res.modes[i]);
                if (lp > best_lp) { best_lp = lp; best = i; }
            }
            if (best < 0) { std::print("  {:5d} ms {}: no mode\n", t - std::stoll(ho), local ? "ENTRY" : "ROOM "); continue; }
            const auto& m = res.modes[best];
            const Eigen::Vector3f e = m.pose - *tr;
            // the best mode by evidence alone, for the record
            const auto& me = res.modes.front();
            std::print("  {:5d} ms {}: {} modes | chosen ({:+.2f},{:+.2f},{:+6.1f}) err {:.2f} m {:+5.1f} deg | top-by-likelihood err {:.2f} m {:+6.1f} deg | live pose err {:.2f} m {:+6.1f} deg\n",
                       t - std::stoll(ho), local ? "ENTRY" : "ROOM ", res.modes.size(), m.pose.x(), m.pose.y(), m.pose.z() * 57.2958f,
                       e.head<2>().norm(), wrap(e.z()) * 57.2958f, (me.pose - *tr).head<2>().norm(), wrap(me.pose.z() - tr->z()) * 57.2958f,
                       (live_pose.at(t) - *tr).head<2>().norm(), wrap(live_pose.at(t).z() - tr->z()) * 57.2958f);
        }
    }
    return 0;
}
