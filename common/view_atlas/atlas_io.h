/*
 * atlas_io.h — archive of dead objects' atlases + the fitted-constants loader (spec: VIEW_ATLAS.md).
 *
 * Text index (greppable) + binary blob (descriptors, no decimal round-trip), the place_map.h precedent.
 * Locale per CLAUDE.md: imbue(classic) + max_digits10 to write, std::from_chars to read. Binary written
 * field by field. Every refusal says which field disagreed. Record format is agent-agnostic on purpose:
 * a future long-term object memory absorbs these files unchanged.
 */
#pragma once

#include "view_atlas.h"

#include <charconv>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace rc::atlas
{

inline constexpr std::uint32_t kArchiveVersion = 1;

struct ArchiveRecord
{
    std::uint64_t   object_uid = 0;
    std::string     agent, cls, name;
    Eigen::Vector3f pose = Eigen::Vector3f::Zero();       // x, y, yaw at death (world frame)
    Eigen::Matrix3f cov  = Eigen::Matrix3f::Identity();
    std::uint64_t   born_ms = 0, died_ms = 0;
    AtlasState      atlas;
};

namespace detail
{
template <class T> void put(std::ofstream& b, T v) { b.write(reinterpret_cast<const char*>(&v), sizeof(T)); }
template <class T> bool get(std::ifstream& b, T& v) { return bool(b.read(reinterpret_cast<char*>(&v), sizeof(T))); }
inline void put_vec(std::ofstream& b, const Eigen::VectorXf& v, int dim)
{
    for (int i = 0; i < dim; ++i) put<float>(b, i < v.size() ? v[i] : 0.f);
}
inline bool get_vec(std::ifstream& b, Eigen::VectorXf& v, int dim)
{
    v.resize(dim);
    for (int i = 0; i < dim; ++i) if (not get<float>(b, v[i])) return false;
    return true;
}
template <class T> bool parse(std::string_view s, T& out)
{
    const auto r = std::from_chars(s.data(), s.data() + s.size(), out);
    return r.ec == std::errc() and r.ptr == s.data() + s.size();
}
inline std::vector<std::string_view> split(std::string_view s, char sep)
{
    std::vector<std::string_view> out;
    std::size_t a = 0;
    for (std::size_t i = 0; i <= s.size(); ++i)
        if (i == s.size() or s[i] == sep) { out.push_back(s.substr(a, i - a)); a = i + 1; }
    return out;
}
inline std::string_view trim(std::string_view s)
{
    while (not s.empty() and (s.front() == ' ' or s.front() == '\t')) s.remove_prefix(1);
    while (not s.empty() and (s.back() == ' ' or s.back() == '\t' or s.back() == '\r')) s.remove_suffix(1);
    return s;
}
} // namespace detail

inline bool save_archive(const std::string& base, const std::vector<ArchiveRecord>& recs, std::string* why_not)
{
    auto fail = [&](std::string m) { if (why_not) *why_not = std::move(m); return false; };
    for (const auto& r : recs)
        for (const auto* s : {&r.agent, &r.cls, &r.name})
            if (s->find(',') != std::string::npos or s->find('\n') != std::string::npos)
                return fail("text field contains a comma or newline: '" + *s + "'");
    const int sd = recs.empty() ? 0 : recs.front().atlas.shape_dim;
    const int cd = recs.empty() ? 0 : recs.front().atlas.colour_dim;
    const int nb = recs.empty() ? 0 : int(recs.front().atlas.bins.size());
    for (const auto& r : recs)
        if (r.atlas.shape_dim != sd or r.atlas.colour_dim != cd or int(r.atlas.bins.size()) != nb)
            return fail("records disagree on shape_dim/colour_dim/n_bins");

    std::ofstream b(base + ".blob", std::ios::binary | std::ios::trunc);
    if (not b) return fail("cannot open " + base + ".blob");
    b.write("RCVA", 4);
    for (std::uint32_t f : {kArchiveVersion, std::uint32_t(sd), std::uint32_t(cd), std::uint32_t(nb),
                            std::uint32_t(recs.size())})
        detail::put(b, f);
    for (const auto& r : recs)
    {
        detail::put_vec(b, r.atlas.colour, cd);
        detail::put<float>(b, r.atlas.n_c);
        for (const auto& bin : r.atlas.bins)
        {
            detail::put<float>(b, bin.n);
            detail::put<float>(b, bin.bearing);
            detail::put<std::uint64_t>(b, bin.stamp_ms);
            detail::put<std::uint32_t>(b, std::uint32_t(bin.poses.size()));
            for (const auto& p : bin.poses) { detail::put<float>(b, p.x()); detail::put<float>(b, p.y()); }
            detail::put_vec(b, bin.mean, sd);
        }
    }
    if (not b) return fail("write failed: " + base + ".blob");
    b.close();

    std::ofstream f(base + ".idx", std::ios::trunc);
    if (not f) return fail("cannot open " + base + ".idx");
    f.imbue(std::locale::classic());
    f << std::setprecision(std::numeric_limits<float>::max_digits10);
    f << "# view_atlas archive v" << kArchiveVersion << " shape_dim=" << sd << " colour_dim=" << cd
      << " n_bins=" << nb << "\n"
      << "uid,agent,cls,name,x,y,yaw,c00,c01,c02,c11,c12,c22,born_ms,died_ms\n";
    for (const auto& r : recs)
        f << r.object_uid << ',' << r.agent << ',' << r.cls << ',' << r.name << ','
          << r.pose.x() << ',' << r.pose.y() << ',' << r.pose.z() << ','
          << r.cov(0, 0) << ',' << r.cov(0, 1) << ',' << r.cov(0, 2) << ','
          << r.cov(1, 1) << ',' << r.cov(1, 2) << ',' << r.cov(2, 2) << ','
          << r.born_ms << ',' << r.died_ms << '\n';
    return bool(f) or fail("write failed: " + base + ".idx");
}

inline bool load_archive(const std::string& base, const AtlasParams& p, int shape_dim, int colour_dim,
                         std::vector<ArchiveRecord>& out, std::string* why_not)
{
    out.clear();
    auto fail = [&](std::string m) { out.clear(); if (why_not) *why_not = std::move(m); return false; };

    std::ifstream f(base + ".idx");
    if (not f) return fail("cannot open " + base + ".idx");
    std::vector<ArchiveRecord> recs;
    std::string line;
    while (std::getline(f, line))
    {
        if (line.empty() or line[0] == '#' or line.starts_with("uid,")) continue;
        const auto c = detail::split(line, ',');
        if (c.size() != 15) return fail("index line has " + std::to_string(c.size()) + " fields, expected 15");
        ArchiveRecord r;
        float v[9];
        bool ok = detail::parse(c[0], r.object_uid) and detail::parse(c[13], r.born_ms) and detail::parse(c[14], r.died_ms);
        for (int i = 0; i < 9; ++i) ok = ok and detail::parse(c[std::size_t(4 + i)], v[i]);
        if (not ok) return fail("unparseable number in index line: " + line);
        r.agent = std::string(c[1]); r.cls = std::string(c[2]); r.name = std::string(c[3]);
        r.pose = Eigen::Vector3f(v[0], v[1], v[2]);
        r.cov << v[3], v[4], v[5], v[4], v[6], v[7], v[5], v[7], v[8];
        recs.push_back(std::move(r));
    }

    std::ifstream b(base + ".blob", std::ios::binary);
    if (not b) return fail("cannot open " + base + ".blob");
    char magic[4];
    if (not b.read(magic, 4) or std::memcmp(magic, "RCVA", 4) != 0) return fail("bad magic in " + base + ".blob");
    std::uint32_t ver, sd, cd, nb, n;
    if (not (detail::get(b, ver) and detail::get(b, sd) and detail::get(b, cd) and detail::get(b, nb) and detail::get(b, n)))
        return fail("blob short: header");
    if (ver != kArchiveVersion) return fail("version " + std::to_string(ver) + " != " + std::to_string(kArchiveVersion));
    if (int(sd) != shape_dim) return fail("shape_dim " + std::to_string(sd) + " != " + std::to_string(shape_dim));
    if (int(cd) != colour_dim) return fail("colour_dim " + std::to_string(cd) + " != " + std::to_string(colour_dim));
    if (int(nb) != p.n_bins) return fail("n_bins " + std::to_string(nb) + " != " + std::to_string(p.n_bins));
    if (n != recs.size()) return fail("record count: blob " + std::to_string(n) + ", index " + std::to_string(recs.size()));

    for (auto& r : recs)
    {
        r.atlas.shape_dim = shape_dim; r.atlas.colour_dim = colour_dim;
        if (not detail::get_vec(b, r.atlas.colour, colour_dim) or not detail::get(b, r.atlas.n_c))
            return fail("blob short: colour");
        r.atlas.bins.resize(nb);
        for (auto& bin : r.atlas.bins)
        {
            std::uint32_t np;
            if (not (detail::get(b, bin.n) and detail::get(b, bin.bearing) and detail::get(b, bin.stamp_ms)
                     and detail::get(b, np)))
                return fail("blob short: bin header");
            if (int(np) > p.poses_per_bin) return fail("poses per bin " + std::to_string(np) + " > poses_per_bin");
            bin.poses.resize(np);
            for (auto& ps : bin.poses)
            {
                float x, y;
                if (not (detail::get(b, x) and detail::get(b, y))) return fail("blob short: poses");
                ps = Eigen::Vector2f(x, y);
            }
            if (not detail::get_vec(b, bin.mean, shape_dim)) return fail("blob short: bin mean");
        }
    }
    out = std::move(recs);
    return true;
}

/// `key = value` lines; '#' starts a comment. Unknown keys and unparseable values are refused (loudly):
/// a misspelt constant silently keeping its default is the failure this loader exists to prevent.
/// The result must satisfy the model's invariants (positive scales, positive-definite covariances); a file
/// that does not is refused and `p` is left untouched.
inline bool load_params(const std::string& path, AtlasParams& p, std::string* why_not)
{
    auto fail = [&](std::string m) { if (why_not) *why_not = std::move(m); return false; };
    std::ifstream f(path);
    if (not f) return fail("cannot open " + path);
    AtlasParams q = p;
    auto sym = [](Eigen::Matrix2f& m, float v) { m(0, 1) = m(1, 0) = v; };
    std::string line;
    while (std::getline(f, line))
    {
        std::string_view s = detail::trim(line);
        if (s.empty() or s.front() == '#') continue;
        const auto eq = s.find('=');
        if (eq == std::string_view::npos) return fail("no '=' in line: " + line);
        const std::string key(detail::trim(s.substr(0, eq)));
        float v;
        if (not detail::parse(detail::trim(s.substr(eq + 1)), v)) return fail("unparseable value for " + key + ": " + line);
        if      (key == "ell_rad")          q.ell_rad = v;
        else if (key == "repeat_ell_rad")   q.repeat_ell_rad = v;
        else if (key == "repeat_ell_logr")  q.repeat_ell_logr = v;
        else if (key == "n0")               q.n0 = v;
        else if (key == "mu_own_d")         q.mu_own.x() = v;
        else if (key == "mu_own_c")         q.mu_own.y() = v;
        else if (key == "mu_bg_d")          q.mu_bg.x() = v;
        else if (key == "mu_bg_c")          q.mu_bg.y() = v;
        else if (key == "cov_own_dd")       q.cov_own(0, 0) = v;
        else if (key == "cov_own_dc")       sym(q.cov_own, v);
        else if (key == "cov_own_cc")       q.cov_own(1, 1) = v;
        else if (key == "cov_bg_dd")        q.cov_bg(0, 0) = v;
        else if (key == "cov_bg_dc")        sym(q.cov_bg, v);
        else if (key == "cov_bg_cc")        q.cov_bg(1, 1) = v;
        else if (key == "cov_n_dd")         q.cov_n(0, 0) = v;
        else if (key == "cov_n_dc")         sym(q.cov_n, v);
        else if (key == "cov_n_cc")         q.cov_n(1, 1) = v;
        else if (key == "lambda_nats")      q.lambda_nats = v;
        else if (key == "max_rate_hz")      q.max_rate_hz = v;
        else if (key == "pi_move")          q.pi_move = v;
        else if (key == "quality_px_scale") q.quality_px_scale = v;
        else return fail("unknown key: " + key);
    }
    // Invariants checked after all lines are parsed, so a refused file never half-applies.
    if (not (q.n0 > 0.f))              return fail("n0 must be > 0");
    if (not (q.ell_rad > 0.f))         return fail("ell_rad must be > 0");
    if (not (q.repeat_ell_rad > 0.f))  return fail("repeat_ell_rad must be > 0");
    if (not (q.repeat_ell_logr > 0.f)) return fail("repeat_ell_logr must be > 0");
    if (not (q.quality_px_scale > 0.f)) return fail("quality_px_scale must be > 0");
    auto pd = [](const Eigen::Matrix2f& m) { return Eigen::LLT<Eigen::Matrix2f>(m).info() == Eigen::Success; };
    if (not pd(q.cov_own)) return fail("cov_own is not positive definite");
    if (not pd(q.cov_bg))  return fail("cov_bg is not positive definite");
    if (not pd(q.cov_n))   return fail("cov_n is not positive definite");
    p = q;
    return true;
}

} // namespace rc::atlas
