// card_vs_probe.h -- layer (b) of the perception-card truth test: the shared evaluator's geometric P(detect)
// against what an agent LOGGED in its detect_probe CSV (common/detect_probe/detect_probe.h).
//
// Header-only so the synthetic test calls analyse() directly. Columns are addressed BY NAME, so old logs
// (no `oyaw`) and new ones both load. Numbers are parsed with std::from_chars (CLAUDE.md locale trap).
//
// Reports
//   * bias_logged   = mean(p_detect_logged - fired)            -- the agent's own prediction vs the outcome
//   * bias_geo      = mean(p_geo - fired)                      -- the evaluator's prediction vs the outcome
//   * bias_geo_vs_logged = mean(p_geo - p_detect_logged)       -- evaluator vs agent (mean predicted - logged p)
//   * coverage      = fraction of roi_valid rows with |roi_fill - fill_pred| <= 2 sigma
//   * 10-bin reliability tables (predicted p vs observed detection rate), for both predictors.
// The geometric part needs a card AND the object yaw (`oyaw`); without them only the logged-p part is filled.
// Geometry NOT in the log and therefore a PARAMETER (Options): the ZED intrinsics and the belief sigmas
// (extent, position) behind the fill sigma. Walls are not logged: the world has no polygon (walls unknown).
#pragma once

#include "perception_card/evaluator.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace rc::card::probe
{
struct Row
{
    double rx = 0, ry = 0, cam_z = 0, ocx = 0, ocy = 0, oyaw = 0;
    double roi_fill = 0, p_detect = 0;
    bool roi_valid = false, fired = false;
};

struct Log { std::vector<Row> rows; bool has_oyaw = false; std::string error; };

inline bool num(std::string_view s, double& v)
{
    double x = 0;
    const auto r = std::from_chars(s.data(), s.data() + s.size(), x);
    if (r.ec != std::errc() or r.ptr != s.data() + s.size()) return false;
    v = x;
    return true;
}

inline std::vector<std::string_view> split(std::string_view l)
{
    std::vector<std::string_view> out;
    std::size_t a = 0;
    for (;;)
    {
        const auto b = l.find(',', a);
        out.push_back(l.substr(a, b == std::string_view::npos ? b : b - a));
        if (b == std::string_view::npos) break;
        a = b + 1;
    }
    return out;
}

// Rows with an unparsable required field are skipped (a log cut mid-write ends in one).
inline Log read_log(const std::string& path)
{
    Log log;
    std::ifstream f(path);
    if (not f) { log.error = "cannot open " + path; return log; }
    f.imbue(std::locale::classic());
    std::string line;
    if (not std::getline(f, line)) { log.error = "empty " + path; return log; }
    std::map<std::string, std::size_t, std::less<>> col;
    {
        const auto h = split(line);
        for (std::size_t i = 0; i < h.size(); ++i) col[std::string(h[i])] = i;
    }
    for (const char* need : {"rx", "ry", "cam_z", "ocx", "ocy", "roi_fill", "roi_valid", "p_detect", "fired"})
        if (not col.contains(need)) { log.error = std::string("missing column ") + need; return log; }
    log.has_oyaw = col.contains("oyaw");
    while (std::getline(f, line))
    {
        const auto c = split(line);
        const auto get = [&](const char* n, double& v)
        {
            const auto it = col.find(n);
            return it != col.end() and it->second < c.size() and num(c[it->second], v);
        };
        Row r; double ro = 0, fi = 0;
        if (not (get("rx", r.rx) and get("ry", r.ry) and get("cam_z", r.cam_z) and get("ocx", r.ocx) and get("ocy", r.ocy)
                 and get("roi_fill", r.roi_fill) and get("roi_valid", ro) and get("p_detect", r.p_detect) and get("fired", fi)))
            continue;
        if (log.has_oyaw) get("oyaw", r.oyaw);
        r.roi_valid = ro > 0.5; r.fired = fi > 0.5;
        log.rows.push_back(r);
    }
    return log;
}

struct Options
{
    double zed_f = 448, zed_w = 1280, zed_h = 720;   // the sim ZED (as the evaluator tests); NOT in the log
    float sigma_extent = 0.05f, sigma_pos = 0.05f;   // belief sigmas behind the fill sigma; NOT in the log
};

struct Bin { int n = 0; double mean_pred = 0, rate = 0; };
using Reliability = std::array<Bin, 10>;

struct Report
{
    std::size_t n = 0, n_fired = 0;
    double bias_logged = 0;                          // mean(p_detect - fired)
    Reliability rel_logged{};
    bool geometric = false;                          // card present and oyaw logged
    std::size_t n_geo = 0, n_cov = 0;
    double bias_geo = 0, bias_geo_vs_logged = 0, coverage = 0;
    Reliability rel_geo{};
    std::string note;
};

inline void fill_rel(Reliability& rel, const std::vector<double>& pred, const std::vector<char>& hit)
{
    std::array<double, 10> sp{}, sh{};
    for (std::size_t i = 0; i < pred.size(); ++i)
    {
        const int b = std::clamp(int(pred[i] * 10.0), 0, 9);
        ++rel[b].n; sp[b] += pred[i]; sh[b] += hit[i];
    }
    for (int b = 0; b < 10; ++b)
        if (rel[b].n > 0) { rel[b].mean_pred = sp[b] / rel[b].n; rel[b].rate = sh[b] / rel[b].n; }
}

inline Report analyse(const Log& log, const Card* card, const Options& o = {})
{
    Report rep;
    rep.n = log.rows.size();
    std::vector<double> pl; std::vector<char> hit;
    for (const Row& r : log.rows) { pl.push_back(std::clamp(r.p_detect, 0.0, 1.0)); hit.push_back(r.fired); rep.n_fired += r.fired; }
    if (rep.n == 0) { rep.note = "no rows"; return rep; }
    for (std::size_t i = 0; i < pl.size(); ++i) rep.bias_logged += pl[i] - hit[i];
    rep.bias_logged /= double(rep.n);
    fill_rel(rep.rel_logged, pl, hit);

    if (not card) { rep.note = "no card given: geometric comparison skipped (use --agent-dir)"; return rep; }
    if (not log.has_oyaw)
    {
        rep.note = "log has no `oyaw` column (written by logs from this commit on): the hypothesis cannot be rebuilt; "
                   "the geometric comparison needs new logs. Only the agent's own logged p is scored.";
        return rep;
    }
    rep.geometric = true;
    rep.note = "viewpoint = robot xy facing the target; walls unknown; zed f=" + std::to_string(o.zed_f);
    World w;
    rc::nbv::Sensor s;
    s.hfov_rad = float(2.0 * std::atan(0.5 * o.zed_w / o.zed_f));
    s.vfov_rad = float(2.0 * std::atan(0.5 * o.zed_h / o.zed_f));
    w.zed = s;
    const SubtypeParams& sp = params_for(*card, "");
    std::vector<double> pg; std::vector<char> hg;
    double sb = 0, sl = 0; std::size_t cov_ok = 0;
    for (const Row& r : log.rows)
    {
        w.zed_mount_z = float(r.cam_z); s.height_m = float(r.cam_z); s.env = sp.env;
        Hypothesis h; h.concept_ = card->concept_; h.x = float(r.ocx); h.y = float(r.ocy); h.yaw = float(r.oyaw);
        const Eigen::Vector2f from{float(r.rx), float(r.ry)};
        const ChannelResult c = evaluate_camera(*card, w, h, {from, std::nullopt});
        const double p = std::clamp<double>(c.p, 0.0, 1.0);
        pg.push_back(p); hg.push_back(r.fired);
        sb += p - r.fired; sl += p - std::clamp(r.p_detect, 0.0, 1.0);
        if (r.roi_valid)
        {
            rc::nbv::Target t;
            const GeometryPrior g = card->geometry;
            t.cx = h.x; t.cy = h.y; t.yaw = h.yaw; t.w = g.w; t.h = g.d; t.z0 = g.z0; t.z1 = g.z1;
            t.sigma_extent_m = o.sigma_extent; t.sigma_pos_m = o.sigma_pos;
            const double fp = rc::nbv::predicted_fill(t, from, s);
            const double sg = rc::nbv::predicted_fill_sigma(t, from, s);
            if (std::isfinite(fp) and sg > 0.0)
            {
                ++rep.n_cov;
                cov_ok += std::abs(r.roi_fill - fp) <= 2.0 * sg;
            }
        }
    }
    rep.n_geo = pg.size();
    rep.bias_geo = sb / double(rep.n_geo);
    rep.bias_geo_vs_logged = sl / double(rep.n_geo);
    rep.coverage = rep.n_cov ? double(cov_ok) / double(rep.n_cov) : std::numeric_limits<double>::quiet_NaN();
    fill_rel(rep.rel_geo, pg, hg);
    return rep;
}

inline void print_rel(std::FILE* o, const char* title, const Reliability& rel)
{
    std::fprintf(o, "  reliability (%s):\n    bin        n   mean_pred  observed\n", title);
    for (int b = 0; b < 10; ++b)
        if (rel[b].n > 0)
            std::fprintf(o, "    %.1f-%.1f %6d   %.3f      %.3f\n", 0.1 * b, 0.1 * (b + 1), rel[b].n, rel[b].mean_pred, rel[b].rate);
}

inline void print(std::FILE* o, const std::string& name, const Report& r)
{
    std::fprintf(o, "== %s: %zu rows, %zu fired (%.1f%%)\n", name.c_str(), r.n, r.n_fired, r.n ? 100.0 * r.n_fired / r.n : 0.0);
    if (r.n == 0) { std::fprintf(o, "  %s\n", r.note.c_str()); return; }
    std::fprintf(o, "  bias, agent's logged p - outcome:   %+.4f\n", r.bias_logged);
    print_rel(o, "agent's logged p", r.rel_logged);
    if (r.geometric)
    {
        std::fprintf(o, "  bias, evaluator p - outcome:        %+.4f   (n=%zu)\n", r.bias_geo, r.n_geo);
        std::fprintf(o, "  bias, evaluator p - logged p:       %+.4f\n", r.bias_geo_vs_logged);
        std::fprintf(o, "  2-sigma fill coverage:              %.3f   (n=%zu, nominal 0.954)\n", r.coverage, r.n_cov);
        print_rel(o, "evaluator p", r.rel_geo);
    }
    std::fprintf(o, "  note: %s\n", r.note.c_str());
}

inline bool write_csv(const std::string& path, const std::string& name, const Report& r)
{
    std::ofstream f(path, std::ios::app);
    if (not f) return false;
    f.imbue(std::locale::classic());
    f << "# " << name << " bias_logged=" << r.bias_logged << " bias_geo=" << r.bias_geo << " coverage=" << r.coverage << '\n'
      << "source,bin_lo,n,mean_pred,observed\n";
    for (const auto& [tag, rel] : {std::pair{"logged", &r.rel_logged}, std::pair{"geo", &r.rel_geo}})
        for (int b = 0; b < 10; ++b)
            if ((*rel)[b].n > 0) f << tag << ',' << 0.1 * b << ',' << (*rel)[b].n << ',' << (*rel)[b].mean_pred << ',' << (*rel)[b].rate << '\n';
    return true;
}
}  // namespace rc::card::probe
