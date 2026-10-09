// Replay a live innovation log (tmp/noise_innov/innov_<ts>.csv) through rc::preint::MotionNoiseInnov — the
// AGENT'S estimator code, not a re-implementation — with the outlier mixture off and on, and print the
// coefficients each fit ends with plus the calibration (mean / median of y/m, inliers weighted) per motion class.
//
// Build:  g++ -std=c++23 -O2 -I/usr/include/eigen3 tools/noise_innov_replay.cpp -o /tmp/innov_replay
// Run:    /tmp/innov_replay tmp/noise_innov/innov_<ts>.csv [tmp/heading/heading_<ts>.csv]
// Logs written before the while-moving term have no p_move column: pass the heading log and P(moving) is joined
// from its rest_gain_tr / rest_gain_ro on ts_ms (dt = 0.05 s). Without either, the term's regressor is 0.
#include "../src/motion_noise_innov.h"
#include "../src/pose_field_bias.h"
#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <clocale>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using rc::preint::MotionNoiseInnov;

namespace
{
std::vector<std::string> split(const std::string &line)
{
    std::vector<std::string> out; std::string cell; std::istringstream ss(line);
    while (std::getline(ss, cell, ',')) out.push_back(cell);
    return out;
}
double num(const std::string &s)
{
    double v = 0.0;
    const auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), v);   // locale-independent (CLAUDE.md)
    return ec == std::errc{} ? v : std::numeric_limits<double>::quiet_NaN();
}
struct Row { std::array<MotionNoiseInnov::Vec, 3> x; std::array<double, 3> y; double of, ol, ot;
             long long ts = 0; double zx = NAN, zy = NAN, zt = NAN; };

void fit(const std::vector<Row> &rows, const std::array<double, 6> &k0, bool mixture, bool move_term)
{
    MotionNoiseInnov L;
    MotionNoiseInnov::Params p;
    p.k0 = k0; p.mixture = mixture;
    if (not move_term) p.mv_sd = {1e-12, 1e-12, 1e-12};
    L.set_params(p);
    std::map<std::string, std::array<std::vector<double>, 3>> ratio;
    // With the full model, also feed the pose-field bias estimator exactly as observe_innovation does, and
    // write what it would add per cycle to /tmp/pf_replay.csv (ts_ms, pf_sigma_b, pf_xx, pf_tt).
    const bool feed_pf = mixture and move_term and std::isfinite(rows.front().zx);
    rc::PoseFieldBias pf;
    std::ofstream pfo;
    if (feed_pf) { pf.set_params({}); pfo.open("/tmp/pf_replay.csv"); pfo.imbue(std::locale::classic()); pfo << "ts_ms,pf_sigma_b,pf_xx,pf_tt\n"; }
    long long prev_ts = -1;
    for (const auto &r : rows)
    {
        L.observe_rows(r.x, r.y);
        if (feed_pf)
        {
            const bool chain = prev_ts > 0 and r.ts - prev_ts <= 80;
            prev_ts = r.ts;
            const double th = r.zt - 0.5 * r.ot;   // body -> world, the agent's convention
            const Eigen::Vector3f odom(float(-std::sin(th) * r.of + std::cos(th) * r.ol),
                                       float( std::cos(th) * r.of + std::sin(th) * r.ol), float(r.ot));
            // the scan's claimed covariance, recovered from the regressors: x(NK+a) = u_a' (R_n + R_{n-1}) u_a
            const Eigen::Matrix3f R = Eigen::Vector3f(float(0.5 * r.x[0](6)), float(0.5 * r.x[1](7)), float(0.5 * r.x[2](8))).asDiagonal();
            pf.observe(Eigen::Vector3f(float(r.zx), float(r.zy), float(r.zt)), R, odom,
                       float(std::getenv("PF_ALL_ODOM") ? L.odom_var(0) + L.odom_var(1)
                                                        : L.odom_var_accum(0) + L.odom_var_accum(1)), 0.f, chain);
            const Eigen::Matrix3f Sb = pf.covariance(R);
            pfo << r.ts << ',' << pf.sigma_b() << ',' << 0.5 * (Sb(0, 0) + Sb(1, 1)) << ',' << Sb(2, 2) << '\n';
        }
        const bool mov = std::abs(r.of) > 0.002, turn = std::abs(r.ot) > 0.005;
        const std::string c = (not mov and not turn) ? "parked" : (turn and not mov) ? "pivot" : "driving";
        for (int a = 0; a < 3; ++a)
            if (L.last_m(a) > 0.0) ratio[c][a].push_back(L.last_y(a) / L.last_m(a));
    }
    std::printf("\n== %s%s\n", mixture ? "MIXTURE (outlier component on)" : "PLAIN Gaussian",
                move_term ? " + WHILE-MOVING term" : "");
    const char *kn[6] = {"k_long", "k_lat", "k_lat_turn", "k_th_turn", "k_t_trans", "k_t_rot"};
    for (int j = 0; j < 6; ++j) std::printf("  %-10s %.3e  (configured %.3e, x%.2f)\n", kn[j], L.k(j), k0[j], L.k(j) / k0[j]);
    std::printf("  s fwd/lat/th %.2e %.2e %.2e | q fwd/lat/th %.2e %.2e %.2e\n", L.s(0), L.s(1), L.s(2), L.q(0), L.q(1), L.q(2));
    if (feed_pf) std::printf("  POSE-FIELD bias: sigma_b %.1f mm, l %.2f m, %ld pairs (per-cycle term -> /tmp/pf_replay.csv)\n",
                             pf.sigma_b() * 1e3, pf.length(), pf.pairs());
    if (move_term) std::printf("  while moving mv fwd/lat/th %.2e %.2e %.2e (per s)\n", L.mv(0), L.mv(1), L.mv(2));
    if (mixture)
        std::printf("  outlier share pi %.3f %.3f %.3f | variance ratio kappa %.1f %.1f %.1f\n",
                    L.pi(0), L.pi(1), L.pi(2), L.kappa(0), L.kappa(1), L.kappa(2));
    std::printf("  y/m (mean / median; for the mixture the MEAN includes the outliers it names):\n");
    for (const auto &[c, v] : ratio)
    {
        std::printf("    %-8s", c.c_str());
        for (int a = 0; a < 3; ++a)
        {
            auto w = v[a];
            if (w.empty()) continue;
            double mean = 0; for (double t : w) mean += t; mean /= w.size();
            std::nth_element(w.begin(), w.begin() + w.size() / 2, w.end());
            std::printf("  %s %.2f / %.3f", a == 0 ? "fwd" : a == 1 ? "lat" : "th ", mean, w[w.size() / 2]);
        }
        std::printf("  (n=%zu)\n", v.at(0).size());
    }
}
}   // namespace

int main(int argc, char **argv)
{
    std::setlocale(LC_ALL, "");   // run under the agent's locale: from_chars must not care (CLAUDE.md)
    if (argc < 2) { std::fprintf(stderr, "usage: %s innov.csv\n", argv[0]); return 1; }
    std::ifstream f(argv[1]);
    std::string line;
    if (not std::getline(f, line)) return 1;
    const auto hdr = split(line);
    std::map<std::string, int> col;
    for (int i = 0; i < int(hdr.size()); ++i) col[hdr[i]] = i;
    const char *ax[3] = {"fwd", "lat", "th"};
    std::map<long long, double> pmove_by_ts;                  // from the heading log, if given
    if (argc > 2)
    {
        std::ifstream hf(argv[2]);
        std::string hl; std::map<std::string, int> hc;
        while (std::getline(hf, hl))
        {
            if (hl.empty() or hl[0] == '#') continue;
            const auto c = split(hl);
            if (hc.empty()) { for (int k = 0; k < int(c.size()); ++k) hc[c[k]] = k; continue; }
            if (c.size() <= std::size_t(std::max(hc["rest_gain_tr"], hc["rest_gain_ro"]))) continue;
            const double tr = num(c[hc["rest_gain_tr"]]), ro = num(c[hc["rest_gain_ro"]]);
            pmove_by_ts[static_cast<long long>(num(c[hc["ts_ms"]]))] =
                1.0 - (1.0 - std::clamp(tr, 0.0, 1.0)) * (1.0 - std::clamp(ro, 0.0, 1.0));
        }
    }
    constexpr int NOLD = MotionNoiseInnov::NK + MotionNoiseInnov::NS + MotionNoiseInnov::NQ;
    long joined = 0;
    std::vector<Row> rows;
    std::array<double, 6> k0{};
    bool first = true;
    while (std::getline(f, line))
    {
        const auto c = split(line);
        if (c.size() != hdr.size()) continue;                  // a partial last line
        Row r;
        bool ok = true;
        for (int a = 0; a < 3; ++a)
        {
            for (int j = 0; j < MotionNoiseInnov::NP; ++j)
            {
                const auto it = col.find("x" + std::to_string(j) + "_" + ax[a]);
                r.x[a](j) = it != col.end() ? num(c[it->second]) : 0.0;
            }
            if (col.find("x" + std::to_string(NOLD) + "_" + ax[a]) == col.end())
            {   // an older log: the while-moving regressor from the joined P(moving)
                const auto it = pmove_by_ts.find(static_cast<long long>(num(c[col["ts_ms"]])));
                const double pm = it != pmove_by_ts.end() ? it->second : 0.0;
                if (a == 0 and it != pmove_by_ts.end()) ++joined;
                r.x[a](NOLD + a) = pm * 0.05;
            }
            r.y[a] = num(c[col[std::string("y_") + ax[a]]]);
            ok = ok and r.x[a].allFinite() and std::isfinite(r.y[a]);
        }
        r.of = num(c[col["odom_fwd"]]); r.ol = num(c[col["odom_lat"]]); r.ot = num(c[col["odom_th"]]);
        r.ts = static_cast<long long>(num(c[col["ts_ms"]]));
        if (col.count("z_x")) { r.zx = num(c[col["z_x"]]); r.zy = num(c[col["z_y"]]); r.zt = num(c[col["z_th"]]); }
        if (first)
        {   // the first row's coefficients are the configured priors (the learner starts there)
            const char *kn[6] = {"k_long", "k_lat", "k_lat_turn", "k_th_turn", "k_t_trans", "k_t_rot"};
            for (int j = 0; j < 6; ++j) k0[j] = num(c[col[kn[j]]]);
            first = false;
        }
        if (ok) rows.push_back(r);
    }
    std::printf("%s: %zu cycles (%ld joined to a heading-log P(moving))\n", argv[1], rows.size(), joined);
    fit(rows, k0, false, false);
    fit(rows, k0, true, false);
    fit(rows, k0, true, true);
    return 0;
}
