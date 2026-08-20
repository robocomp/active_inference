// ─────────────────────────────────────────────────────────────────────────────────────────────────
// SAME LOG, BOTH ESTIMATORS — the test that matters for phase 0.
//
// build: g++ -std=c++23 -O2 -o scale_replay scale_replay.cpp
// run:   ./scale_replay <sdf_localizer log.csv> [stride_frames]
//
// scale_estimator.h claims to be the online form of the fit in room_concept/tools/motion_calib.cpp.
// The unit tests show it agrees with that fit on SYNTHETIC data, which proves only that it is the
// same arithmetic. This drives both from ONE parse of a REAL run: same frames, same windows, same
// rejections — so any disagreement is about the estimators and nothing else.
//
// ★WHY ONE PARSE AND NOT TWO PROGRAMS. Running motion_calib separately and eyeballing the two outputs
// compares numbers computed from differently-built windows, and this project has already paid for
// comparing things that were not measuring the same thing (three false findings in one session,
// all from unaligned windows). One parse, two consumers, one table.
//
// WHAT IT CANNOT TEST, and where phase 0's remaining risk lives:
//   · a scale that DRIFTS — the batch fit assumes one constant, so on a log where the scale really
//     moved the two SHOULD differ, and the log cannot say which is right;
//   · widening while idle, and the marginal-gain rule — properties the offline tool has no notion of;
//   · the truth. Neither estimator knows it. That needs a closure pivot (turn N times, return to the
//     same heading, truth is exactly 2*pi*N — map-free) or a log carrying inj_* columns.
// ─────────────────────────────────────────────────────────────────────────────────────────────────
#include "scale_estimator.h"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace
{
// std::from_chars only: these machines run es_ES, where strtof stops at the decimal POINT and returns
// the integer part, silently. The project has a whole memory note about the day that cost.
bool parse_double(std::string_view s, double &out)
{
    while (not s.empty() and (s.front() == ' ' or s.front() == '"')) s.remove_prefix(1);
    while (not s.empty() and (s.back()  == ' ' or s.back()  == '"' or s.back() == '\r')) s.remove_suffix(1);
    return std::from_chars(s.data(), s.data() + s.size(), out).ec == std::errc{};
}

void split(std::string_view line, std::vector<std::string_view> &out)
{
    out.clear();
    std::size_t start = 0;
    for (std::size_t i = 0; i <= line.size(); ++i)
        if (i == line.size() or line[i] == ',') { out.push_back(line.substr(start, i - start)); start = i + 1; }
}

struct Frame
{
    double ts_s = 0;
    double px = 0, py = 0, pth = 0;      // POSTERIOR pose = pred + innovation
    double ox = 0, oy = 0, oth = 0;      // measured odometry increment for this frame
    bool   ok = false;
};

double wrap(double a) { while (a >  M_PI) a -= 2*M_PI; while (a < -M_PI) a += 2*M_PI; return a; }
}   // namespace

int main(int argc, char **argv)
{
    if (argc < 2) { std::fprintf(stderr, "usage: scale_replay <log.csv> [stride_frames]\n"); return 1; }
    const int stride = (argc > 2) ? std::atoi(argv[2]) : 40;
    std::ifstream in(argv[1]);
    if (not in) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }

    std::string line;
    std::map<std::string, int> col;
    std::vector<std::string_view> buf;
    if (not std::getline(in, line)) return 2;
    { std::string_view hv(line); split(hv, buf);
      for (std::size_t i = 0; i < buf.size(); ++i) col[std::string(buf[i])] = static_cast<int>(i); }

    const char *need[] = {"ts_ms","pred_x","pred_y","pred_theta","innov_x","innov_y","innov_theta",
                          "meas_dx","meas_dy","meas_dth","meas_valid","meas_fresh"};
    for (const char *n : need)
        if (not col.count(n)) { std::fprintf(stderr, "column '%s' missing — wrong log format\n", n); return 2; }

    std::vector<Frame> fr;
    const std::size_t ncols = col.size();
    int ragged = 0;
    while (std::getline(in, line))
    {
        std::string_view lv(line); split(lv, buf);
        if (buf.size() < ncols) { ++ragged; continue; }     // a short row shifts every later field
        Frame f; double ix, iy, ith, mv, mf, ts;
        bool ok = parse_double(buf[col["ts_ms"]], ts);
        ok = ok and parse_double(buf[col["pred_x"]], f.px) and parse_double(buf[col["pred_y"]], f.py)
                and parse_double(buf[col["pred_theta"]], f.pth)
                and parse_double(buf[col["innov_x"]], ix) and parse_double(buf[col["innov_y"]], iy)
                and parse_double(buf[col["innov_theta"]], ith)
                and parse_double(buf[col["meas_dx"]], f.ox) and parse_double(buf[col["meas_dy"]], f.oy)
                and parse_double(buf[col["meas_dth"]], f.oth)
                and parse_double(buf[col["meas_valid"]], mv) and parse_double(buf[col["meas_fresh"]], mf);
        if (not ok) continue;
        f.ts_s = ts / 1000.0;
        f.px += ix; f.py += iy; f.pth = wrap(f.pth + ith);   // posterior = prediction + innovation
        f.ok = (mv != 0.0) and (mf != 0.0);
        fr.push_back(f);
    }
    std::printf("replay: %s — %zu frames (%d ragged), stride %d\n", argv[1], fr.size(), ragged, stride);
    if (fr.size() < static_cast<std::size_t>(stride) * 4) { std::fprintf(stderr, "too few frames\n"); return 3; }

    // ── ONE PASS, TWO CONSUMERS ─────────────────────────────────────────────────────────────────
    rc::calib::ScaleEstimator rot({.scale_walk_density = 0.0, .prior_std = 1e6});   // prior off: the
    rc::calib::ScaleEstimator tra({.scale_walk_density = 0.0, .prior_std = 1e6});   // batch has none
    std::vector<double> d_rot, e_rot, T_rot, d_tra, e_tra, T_tra;
    int rejected_gap = 0, rejected_jump = 0;

    for (std::size_t a = 0; a + stride < fr.size(); a += stride)
    {
        const std::size_t b = a + stride;
        bool good = true;
        for (std::size_t i = a; i <= b; ++i) if (not fr[i].ok) { good = false; break; }
        if (not good) { ++rejected_gap; continue; }
        const double T = fr[b].ts_s - fr[a].ts_s;
        if (not (T > 0.0)) { ++rejected_gap; continue; }

        // Reference motion over the window, from the localiser's posterior.
        const double ref_dth = wrap(fr[b].pth - fr[a].pth);
        // ★PROJECT THE ERROR ON THE DIRECTION OF TRAVEL, which is what motion_calib does and what I
        // got wrong twice before reading it. A scale error acts ALONG the path; the cross-path part is
        // a heading error and belongs to the rotation channel, so mixing them lets a yaw error
        // masquerade as a speed error. Comparing summed odometry steps against the straight-line
        // displacement read every curve as a positive scale (+0.019 on this log); comparing them
        // against the summed posterior displacements read the localiser's own jitter as a negative one
        // (-0.031). Both were geometry, not calibration. The displacement vectors, projected on the
        // reference's direction, are the quantity the model is actually about.
        const double ref_dx = fr[b].px - fr[a].px, ref_dy = fr[b].py - fr[a].py;
        const double ref_straight = std::hypot(ref_dx, ref_dy);

        // ★A RELOCALISATION IS NOT ODOMETRY ERROR. The reference is the localiser's posterior, and it
        // jumps: measured on 2026-08-20, 2172 of 122841 cycles imply a speed above 1 m/s on a 0.6 m/s
        // base, the worst 11.63 m in 50 ms. Charging that to the odometry puts a metre-scale outlier
        // into a fit whose typical increment is a metre. The bound is kinematic and generous — it can
        // only reject a physical impossibility.
        if (not rc::calib::ScaleEstimator::window_is_physical(ref_straight, T, 0.6))
        { ++rejected_jump; continue; }

        // Odometry increment over the same window, summed in the body's own terms.
        // The odometry's displacement over the same window, accumulated in the room frame so it can be
        // projected against the reference direction.
        double odo_dth = 0.0, odo_x = 0.0, odo_y = 0.0;
        for (std::size_t i = a + 1; i <= b; ++i)
        { odo_dth += fr[i].oth; odo_x += fr[i].ox; odo_y += fr[i].oy; }

        // ROTATION: the regressor is the REFERENCE turn, the response the odometry's excess.
        rot.add(ref_dth, odo_dth - ref_dth, T);
        d_rot.push_back(ref_dth); e_rot.push_back(odo_dth - ref_dth); T_rot.push_back(T);

        // TRANSLATION: magnitude of the reference displacement as the regressor, the odometry error
        // projected on its direction as the response. Below a tenth of a millimetre the direction is
        // meaningless, so the window contributes nothing rather than a random direction.
        if (ref_straight > 1e-4)
        {
            const double ux = ref_dx / ref_straight, uy = ref_dy / ref_straight;
            const double e = (odo_x - ref_dx) * ux + (odo_y - ref_dy) * uy;
            tra.add(ref_straight, e, T);
            d_tra.push_back(ref_straight); e_tra.push_back(e); T_tra.push_back(T);
        }
    }

    // The batch fit, exactly as motion_calib computes it.
    auto batch = [](const std::vector<double> &d, const std::vector<double> &e, const std::vector<double> &T)
    {
        double sxx = 0, sxy = 0;
        for (std::size_t i = 0; i < d.size(); ++i) { const double w = 1.0/T[i]; sxx += w*d[i]*d[i]; sxy += w*d[i]*e[i]; }
        const double s = (sxx > 1e-18) ? sxy/sxx : 0.0;
        double ss = 0;
        for (std::size_t i = 0; i < d.size(); ++i) { const double r = e[i] - s*d[i]; ss += r*r/T[i]; }
        return std::pair{s, (d.size() > 1) ? std::sqrt(ss/static_cast<double>(d.size()-1)) : 0.0};
    };

    std::printf("windows: %zu used, %d dropped (gap/invalid), %d dropped (pose jump)\n\n",
                d_rot.size(), rejected_gap, rejected_jump);
    std::printf("%-12s %12s %12s %12s %12s %10s\n", "channel", "online s", "batch s", "online sigma",
                "batch sigma", "s_std");
    std::printf("%-12s %12s %12s %12s %12s %10s\n", "-----------", "----------", "----------",
                "----------", "----------", "--------");
    struct Row { const char *name; rc::calib::ScaleEstimator &on;
                 const std::vector<double> &d, &e, &T; };
    for (auto &r : {Row{"rotation", rot, d_rot, e_rot, T_rot},
                    Row{"translation", tra, d_tra, e_tra, T_tra}})
    {
        const auto c = r.on.posterior();
        const auto [bs, bsig] = batch(r.d, r.e, r.T);
        std::printf("%-12s %12.6f %12.6f %12.6f %12.6f %10.6f%s\n", r.name, c.s, bs, c.sigma, bsig,
                    c.s_std, c.identifiable() ? "" : "   (not identifiable — too little excitation)");
        if (std::abs(c.s - bs) > 1e-9 or std::abs(c.sigma - bsig) > 1e-9)
            std::printf("             ★ DISAGREEMENT: online and batch differ on the same windows — "
                        "one of the two is wrong.\n");
    }
    std::printf("\nreminder: neither number is truth. For that, a closure pivot (turn N times, return\n"
                "to the same heading; truth is exactly 2*pi*N and survives any survey error) or a log\n"
                "carrying inj_* ground truth.\n");
    return 0;
}
