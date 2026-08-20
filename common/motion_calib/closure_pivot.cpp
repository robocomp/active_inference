// ─────────────────────────────────────────────────────────────────────────────────────────────────
// THE CLOSURE PIVOT — the only measurement here that sees the truth.
//
// build: g++ -std=c++23 -O2 -o closure_pivot closure_pivot.cpp
// run:   ./closure_pivot <log.csv> --turns N [--from <ts_ms>] [--to <ts_ms>]
//
// Every other number in this directory is measured against the LOCALISER, which is an estimate. This
// one is not. Spin the robot through N complete turns and stop on the heading it started from: it
// turned 2*pi*N radians, exactly, as a fact about turning. No map, no survey, no localiser — a room
// polygon 2% too large does not touch it. That is why the rotation scale is claimable and the
// translation scale is not: translation has no equivalent closure, so it inherits whatever error the
// survey has, confidently.
//
//     s_omega = (sum of odometry dtheta) / (2*pi*N) - 1
//
// WHAT THE OPERATOR MUST GET RIGHT, and what the tool checks:
//   · N is COUNTED, not inferred from the odometry. Inferring it from the thing under test would
//     assume the answer — round(sum/2pi) returns the truth the estimator wants whenever the error is
//     under half a turn, which is always.
//   · The final heading must MATCH the initial one. The residual is the measurement's resolution:
//     closing to 17 deg over 1440 deg resolves ~1.2%, so a 1.4% scale is barely separable and a 0.5%
//     claim from that run would be noise. The tool prints the resolution and refuses to quote a
//     precision better than it.
//   · The robot must not have travelled. A pivot that walks is a pivot whose heading closure no longer
//     means what it should, because the reference is then a different pose.
//
// The same windows also feed the online estimator, so its answer can be scored against the closure
// rather than against another estimate — this is phase 0's level-4 test, the one the unit checks and
// the replay cannot perform.
// ─────────────────────────────────────────────────────────────────────────────────────────────────
#include "scale_estimator.h"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace
{
bool parse_double(std::string_view s, double &out)
{
    while (not s.empty() and (s.front() == ' ' or s.front() == '"')) s.remove_prefix(1);
    while (not s.empty() and (s.back() == ' ' or s.back() == '"' or s.back() == '\r')) s.remove_suffix(1);
    return std::from_chars(s.data(), s.data() + s.size(), out).ec == std::errc{};
}
void split(std::string_view line, std::vector<std::string_view> &out)
{
    out.clear();
    std::size_t start = 0;
    for (std::size_t i = 0; i <= line.size(); ++i)
        if (i == line.size() or line[i] == ',') { out.push_back(line.substr(start, i - start)); start = i + 1; }
}
double wrap(double a) { while (a > M_PI) a -= 2*M_PI; while (a < -M_PI) a += 2*M_PI; return a; }
constexpr double kDeg = 180.0 / M_PI;
}   // namespace

int main(int argc, char **argv)
{
    const char *path = nullptr;
    double turns = 0, from_ms = -1, to_ms = -1;
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--turns") == 0 and i + 1 < argc) turns = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--from") == 0 and i + 1 < argc) from_ms = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--to") == 0 and i + 1 < argc) to_ms = std::atof(argv[++i]);
        else if (argv[i][0] != '-') path = argv[i];
    }
    if (path == nullptr or turns == 0)
    {
        std::fprintf(stderr,
            "usage: closure_pivot <log.csv> --turns N [--from ts_ms] [--to ts_ms]\n"
            "  N is the number of COMPLETE turns you counted — signed, so -4 for four clockwise.\n"
            "  It is not inferred from the odometry: inferring it from the quantity under test would\n"
            "  assume the answer.\n");
        return 1;
    }

    std::ifstream in(path);
    if (not in) { std::fprintf(stderr, "cannot open %s\n", path); return 2; }
    std::string line; std::map<std::string,int> col; std::vector<std::string_view> buf;
    if (not std::getline(in, line)) return 2;
    { std::string_view hv(line); split(hv, buf);
      for (std::size_t i = 0; i < buf.size(); ++i) col[std::string(buf[i])] = static_cast<int>(i); }
    for (const char *n : {"ts_ms","pred_theta","innov_theta","meas_dth","meas_dx","meas_dy",
                          "pred_x","pred_y","innov_x","innov_y","meas_valid","meas_fresh"})
        if (not col.count(n)) { std::fprintf(stderr, "column '%s' missing — wrong log format\n", n); return 2; }

    double odo_dth_sum = 0, odo_abs_sum = 0, odo_dx = 0, odo_dy = 0;
    double ref_unwrapped = 0, prev_ref = 0, first_ref = 0, first_x = 0, first_y = 0;
    double last_x = 0, last_y = 0, t0 = -1, t1 = 0;
    bool   first = true;
    long   n_rows = 0, n_bad = 0;
    const std::size_t ncols = col.size();

    // Windows for the online estimator: one per second of the manoeuvre, so its answer can be scored
    // against the closure truth rather than against another estimate.
    rc::calib::ScaleEstimator rot({.scale_walk_density = 0.0, .prior_std = 1e6, .prior_density_windows = 0.0});
    double w_odo = 0, w_ref = 0, w_t0 = -1;

    while (std::getline(in, line))
    {
        std::string_view lv(line); split(lv, buf);
        if (buf.size() < ncols) { ++n_bad; continue; }
        double ts, pth, ith, dth, dx, dy, px, py, ix, iy, mv, mf;
        if (not (parse_double(buf[col["ts_ms"]], ts) and parse_double(buf[col["pred_theta"]], pth)
             and parse_double(buf[col["innov_theta"]], ith) and parse_double(buf[col["meas_dth"]], dth)
             and parse_double(buf[col["meas_dx"]], dx) and parse_double(buf[col["meas_dy"]], dy)
             and parse_double(buf[col["pred_x"]], px) and parse_double(buf[col["pred_y"]], py)
             and parse_double(buf[col["innov_x"]], ix) and parse_double(buf[col["innov_y"]], iy)
             and parse_double(buf[col["meas_valid"]], mv) and parse_double(buf[col["meas_fresh"]], mf)))
        { ++n_bad; continue; }
        if (from_ms >= 0 and ts < from_ms) continue;
        if (to_ms   >= 0 and ts > to_ms)   continue;
        if (mv == 0.0 or mf == 0.0) { ++n_bad; continue; }

        const double ref_th = wrap(pth + ith), rx = px + ix, ry = py + iy;
        if (first) { first_ref = prev_ref = ref_th; first_x = rx; first_y = ry; t0 = ts; first = false; }
        // The reference heading, UNWRAPPED: a closure of four turns is 1440 degrees, not 0.
        const double d_ref = wrap(ref_th - prev_ref);
        ref_unwrapped += d_ref;
        prev_ref = ref_th;
        odo_dth_sum += dth;  odo_abs_sum += std::abs(dth);
        odo_dx += dx; odo_dy += dy;
        last_x = rx; last_y = ry; t1 = ts; ++n_rows;

        // ★The window's REFERENCE turn is the same increment added to ref_unwrapped above; taking it
        // again from (ref_th - prev_ref) after prev_ref was already advanced returns zero, which is
        // what a first version of this loop accumulated. Reuse the increment instead of re-deriving
        // it from state that has moved on.
        if (w_t0 < 0) w_t0 = ts;
        w_odo += dth;
        w_ref += d_ref;
        if (ts - w_t0 >= 1000.0)                       // one window per second of manoeuvre
        { rot.add(w_ref, w_odo - w_ref, (ts - w_t0) / 1000.0); w_odo = w_ref = 0; w_t0 = ts; }
    }
    if (n_rows < 10) { std::fprintf(stderr, "only %ld usable rows in that range\n", n_rows); return 3; }

    const double truth        = 2.0 * M_PI * turns;             // THE fact: N complete turns
    const double closure_err  = wrap(prev_ref - first_ref);      // how well the heading came back
    const double net_travel   = std::hypot(last_x - first_x, last_y - first_y);
    const double s_omega      = odo_dth_sum / truth - 1.0;
    // The measurement can be no finer than the closure: if the heading missed by e over a total of
    // |truth|, the scale is only known to e/|truth|.
    const double resolution   = std::abs(closure_err) / std::abs(truth);

    std::printf("closure pivot — %s\n", path);
    std::printf("  rows %ld (%ld unusable)   span %.1f s\n", n_rows, n_bad, (t1 - t0) / 1000.0);
    std::printf("  counted turns          %+.2f  =  %+.1f deg of TRUTH\n", turns, truth * kDeg);
    std::printf("  odometry accumulated   %+.1f deg   (|turn| total %.1f deg)\n",
                odo_dth_sum * kDeg, odo_abs_sum * kDeg);
    std::printf("  localiser accumulated  %+.1f deg   — a cross-check, not the truth\n", ref_unwrapped * kDeg);
    std::printf("  heading closure error  %+.2f deg   ⇒ this run resolves the scale to %.2f%%\n",
                closure_err * kDeg, resolution * 100.0);
    std::printf("  net travel             %.3f m      — a pivot that walks measures something else\n",
                net_travel);
    std::printf("\n  s_omega = %.4f  (%.2f%%)  ±%.2f%% from closure alone\n",
                s_omega, s_omega * 100.0, resolution * 100.0);

    // ── PHASE 0'S LEVEL-4 CHECK: the online estimator, scored against truth ──────────────────────
    const auto c = rot.posterior();
    std::printf("\n  online estimator over the same manoeuvre: s = %.4f ± %.4f  (%d windows)\n",
                c.s, c.s_std, c.windows);
    if (c.windows >= 3)
    {
        const double gap = std::abs(c.s - s_omega);
        std::printf("  vs the closure truth: %s (gap %.4f, its own 3 sigma is %.4f)\n",
                    gap <= std::max(3.0 * c.s_std, resolution) ? "AGREES" : "★ DISAGREES",
                    gap, 3.0 * c.s_std);
    }
    else std::printf("  (too few windows to score it — a longer manoeuvre would fix that)\n");

    // ── THE HONESTY CHECKS ──────────────────────────────────────────────────────────────────────
    int problems = 0;
    if (std::abs(s_omega) < resolution)
    {
        ++problems;
        std::printf("\n  ⚠ THE SCALE IS SMALLER THAN THIS RUN CAN RESOLVE. %.2f%% measured against a\n"
                    "    %.2f%% resolution is not a measurement — turn more times, or close better.\n",
                    std::abs(s_omega) * 100.0, resolution * 100.0);
    }
    if (std::abs(ref_unwrapped - truth) > std::abs(truth) * 0.10)
    {
        ++problems;
        std::printf("\n  ⚠ THE LOCALISER DISAGREES ABOUT HOW FAR THE ROBOT TURNED (%.1f deg vs %.1f).\n"
                    "    Either the turn count is wrong, or the run contains a relocalisation — check\n"
                    "    before believing the number above.\n", ref_unwrapped * kDeg, truth * kDeg);
    }
    if (net_travel > 0.30)
    {
        ++problems;
        std::printf("\n  ⚠ THE ROBOT MOVED %.2f m DURING THE PIVOT. Heading closure still holds, but a\n"
                    "    pivot that walks has drifted through the map, so treat the cross-check above\n"
                    "    with suspicion.\n", net_travel);
    }
    if (problems == 0)
        std::printf("\n  ✓ closure, turn count and net travel are all consistent — this number is usable.\n");

    std::printf("\n  what this does NOT measure: the translation scale. There is no closure for it —\n"
                "  a pivot does not translate, and an out-and-back needs a surveyed distance, so s_v\n"
                "  inherits the map's error. Quote s_omega; flag s_v as survey-limited.\n");
    return problems == 0 ? 0 : 4;
}
