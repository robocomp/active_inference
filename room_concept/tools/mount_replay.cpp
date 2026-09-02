/*
 *  Copyright (C) 2026 by Pablo Bustos
 *
 *  This program is free software: you can redistribute it and/or modify it under the terms of the
 *  GNU General Public License as published by the Free Software Foundation, either version 3 or
 *  any later version. See <http://www.gnu.org/licenses/>.
 */

/**
 *  ARM 7 — can a WRONG extrinsic be recovered, and can the estimator say WHICH device was wrong?
 *  (VALIDATION_THREE_DEVICE_CORNERS §2b. Pre-registered 2026-09-02, before this tool existed.)
 *
 *  ── What it does ────────────────────────────────────────────────────────────────────────────────
 *  Replays a recorded drive under a DELIBERATELY WRONG extrinsic. `r = uv_image − uv_lidar`, and an
 *  extrinsic perturbation changes `uv_lidar` deterministically from the LiDAR corner in the robot
 *  frame — which the pair CSV now carries — so the four injection legs are four analyses of ONE
 *  drive instead of four tours. That also removes the route variation that
 *  [[rgb-corner-calibration-experiment]] measured swamping between-run comparisons.
 *
 *  ── Why it rebuilds the rows through the agent's own header ─────────────────────────────────────
 *  ★ It calls `rc::mount::make_pair_from` and `rc::mount::Accum`, the LIVE code, rather than a
 *    python replica of them. A replica is a second implementation that can drift from the estimator
 *    it is supposed to grade, and this codebase has already paid for that once
 *    ([[measurement-tools-that-lied]]). The only thing this file adds is the perturbation and the
 *    bookkeeping around it.
 *
 *  ── The three channels, and which one is the falsifier ──────────────────────────────────────────
 *  Injecting into a camera's own mount and recovering it is nearly tautological. The experiment that
 *  can FAIL uses the third device: each injection site has a DIFFERENT signature across the three
 *  channels, and the LiDAR row is the one that tests attribution.
 *
 *      injected +δ yaw │ ricoh mount │ zed mount │ closure (ricoh − zed)
 *      ricoh mount     │     −δ      │     0     │        −δ
 *      zed mount       │      0      │    −δ     │        +δ
 *      LiDAR           │     −δ      │    −δ     │   0 — up to PARALLAX, which this tool computes
 *
 *  ★★★ The published table said the LiDAR row leaves the closure at exactly 0. That is only exact if
 *      both cameras' `uv_lidar` shift by the same ANGLE. A LiDAR yaw error rotates every corner about
 *      the LiDAR's own origin, and the two cameras sit at different places, so a parallax term
 *      survives the difference. A falsifier with no tolerance band would read that leftover as
 *      refuting the attribution logic when it is only geometry — so the leg is reported against the
 *      leakage MEASURED on the same rows, and refuses outright if the sidecar could not resolve
 *      where the LiDAR is.
 *
 *  ── The self-check that has to pass before any of it means anything ─────────────────────────────
 *  With δ = 0 the replay must reproduce the LIVE solve. Same rows, same covariance (the CSV carries
 *  the off-diagonal for exactly this), same code ⇒ the numbers must agree to rounding. `--verify`
 *  does that against `etc/camera_calib_<robot>_<camera>.txt`.
 *  ⚠ It only holds when the live evidence came from THIS run alone: the pool resumes from disk
 *    across sessions while the CSV is truncated each run. Delete the evidence file before the
 *    recording drive — which the format-1 note in §2b already requires.
 *
 *  ── The approximation that was stated, and then removed ─────────────────────────────────────────
 *  §2b warned that a replay must hold `cov` FIXED while the extrinsic moves, because the LiDAR half
 *  of it depends on the mount through `Pxy = (P·cam_R_robot).leftCols<2>()`. That is no longer
 *  necessary: the pair CSV carries that half separately (`cl_*`), so the replay recovers the corner's
 *  own covariance `cov_xy = Pxy⁻¹·cov_lidar·Pxy⁻ᵀ` at the NOMINAL mount and rebuilds the weight under
 *  the injected one. ★ `--fixed-cov` keeps the old behaviour, so the size of the approximation that
 *  was going to be assumed small is a number this tool prints rather than a claim it makes.
 *
 *  Build:  make -C build mount_replay && ../bin/mount_replay --help
 */

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "mount_lidar_pair.h"

namespace
{
constexpr double kRad2Deg = 180.0 / M_PI;

/// One replayable row. Everything the agent wrote that the rebuild needs, and nothing else.
struct Row
{
    std::int64_t    ts = 0;
    int             cam = 0;          ///< index into the run's cameras
    int             vertex = -1;
    bool            ceiling = false;
    Eigen::Vector2f uv_image = Eigen::Vector2f::Zero();
    Eigen::Matrix2f cov = Eigen::Matrix2f::Identity();
    Eigen::Matrix2f cov_lidar = Eigen::Matrix2f::Zero();   ///< the part of `cov` that moves with the mount
    float           assoc_prob = 1.f;
    Eigen::Vector3f p_robot = Eigen::Vector3f::Zero();
};

struct Camera
{
    std::string             name;
    rc::mount::ReplayContext ctx;
    std::vector<Row>        rows;     ///< kept per camera; the closure walks them merged by time
    bool                    has_cov_lidar = false;   ///< cl_* present ⇒ the weight can be rebuilt
};

/// CLAUDE.md: these machines run es_ES, where `strtof` stops at a decimal POINT and returns the
/// integer part SILENTLY. `from_chars` is locale-independent by definition and reports failure.
bool to_num(std::string_view s, double& out)
{
    const char* p = s.data();
    const char* e = p + s.size();
    while (p < e and (*p == ' ' or *p == '\t')) ++p;
    return std::from_chars(p, e, out).ec == std::errc{};
}

std::vector<std::string_view> split(std::string_view line, std::vector<std::string_view>& into)
{
    into.clear();
    for (std::size_t a = 0, b; a <= line.size(); a = b + 1)
    {
        b = line.find(',', a);
        if (b == std::string_view::npos) b = line.size();
        into.push_back(line.substr(a, b - a));
    }
    return into;
}

/// Reads a pair CSV BY COLUMN NAME. Not by position: this file has gained columns three times, and
/// a positional reader would have kept working while reading the wrong ones.
bool load_pairs(const std::string& path, Camera& cam, std::string& err)
{
    std::ifstream f(path);
    if (not f.is_open()) { err = "cannot open " + path; return false; }
    std::string line;
    if (not std::getline(f, line)) { err = path + " is empty"; return false; }

    std::vector<std::string_view> tok;
    split(line, tok);
    std::map<std::string, int> col;
    for (int i = 0; i < static_cast<int>(tok.size()); ++i) col[std::string(tok[static_cast<std::size_t>(i)])] = i;
    const auto need = [&](const char* k, int& idx) {
        const auto it = col.find(k);
        if (it == col.end()) { err = path + ": no '" + k + "' column"; return false; }
        idx = it->second;
        return true;
    };
    int c_ts = 0, c_vx = 0, c_ceil = 0, c_ui = 0, c_vi = 0, c_su = 0, c_sv = 0, c_cuv = 0;
    int c_ap = 0, c_px = 0, c_py = 0, c_pz = 0, c_luu = 0, c_luv = 0, c_lvv = 0;
    if (not (need("ts_ms", c_ts) and need("vertex", c_vx) and need("u_img", c_ui)
             and need("v_img", c_vi) and need("sigu", c_su) and need("sigv", c_sv)
             and need("assoc_prob", c_ap) and need("px_robot", c_px) and need("py_robot", c_py)
             and need("pz_robot", c_pz)))
        return false;
    // The two columns added for the replay. Their ABSENCE is not a missing field to default: a file
    // without them was written by a binary that could not have produced a replayable row, and
    // guessing (cov diagonal, ceiling=false) would silently answer a different question.
    if (not need("cuv", c_cuv))
    {
        err = path + ": no 'cuv' column — recorded before 2026-09-02, so the replay cannot"
                     " reproduce the live weighting. Re-record; do not assume a diagonal covariance.";
        return false;
    }
    if (not need("ceiling", c_ceil))
    {
        err = path + ": no 'ceiling' column — the loop closure keys on vertex*2+ceiling and would"
                     " difference floor corners against ceiling ones. Re-record.";
        return false;
    }
    // Optional, and the ONLY optional columns here: without them the replay still runs but must hold
    // the covariance fixed, which is the approximation these columns exist to remove. Their absence
    // is reported, not defaulted away.
    const bool have_lid = need("cl_uu", c_luu) and need("cl_uv", c_luv) and need("cl_vv", c_lvv);
    if (not have_lid)
        std::printf("  ⚠ %s has no cl_* columns: the LiDAR share of the covariance was not recorded,"
                    " so this file can only be replayed with --fixed-cov\n", path.c_str());
    err.clear();

    long bad = 0;
    while (std::getline(f, line))
    {
        if (line.empty()) continue;
        split(line, tok);
        const int wanted = 1 + std::max({c_ts, c_vx, c_ceil, c_ui, c_vi, c_su, c_sv, c_cuv, c_ap,
                                         c_px, c_py, c_pz, have_lid ? c_lvv : 0});
        if (static_cast<int>(tok.size()) < wanted) { ++bad; continue; }
        double ts = 0, vx = 0, ce = 0, ui = 0, vi = 0, su = 0, sv = 0, cuv = 0, ap = 0, px = 0, py = 0, pz = 0;
        const bool ok = to_num(tok[static_cast<std::size_t>(c_ts)], ts)
                    and to_num(tok[static_cast<std::size_t>(c_vx)], vx)
                    and to_num(tok[static_cast<std::size_t>(c_ceil)], ce)
                    and to_num(tok[static_cast<std::size_t>(c_ui)], ui)
                    and to_num(tok[static_cast<std::size_t>(c_vi)], vi)
                    and to_num(tok[static_cast<std::size_t>(c_su)], su)
                    and to_num(tok[static_cast<std::size_t>(c_sv)], sv)
                    and to_num(tok[static_cast<std::size_t>(c_cuv)], cuv)
                    and to_num(tok[static_cast<std::size_t>(c_ap)], ap)
                    and to_num(tok[static_cast<std::size_t>(c_px)], px)
                    and to_num(tok[static_cast<std::size_t>(c_py)], py)
                    and to_num(tok[static_cast<std::size_t>(c_pz)], pz);
        if (not ok) { ++bad; continue; }
        Row r;
        r.ts = static_cast<std::int64_t>(ts);
        r.vertex = static_cast<int>(vx);
        r.ceiling = ce > 0.5;
        r.uv_image = Eigen::Vector2f(static_cast<float>(ui), static_cast<float>(vi));
        r.cov(0, 0) = static_cast<float>(su * su);
        r.cov(1, 1) = static_cast<float>(sv * sv);
        r.cov(0, 1) = r.cov(1, 0) = static_cast<float>(cuv);
        r.assoc_prob = static_cast<float>(ap);
        r.p_robot = Eigen::Vector3f(static_cast<float>(px), static_cast<float>(py), static_cast<float>(pz));
        if (have_lid)
        {
            double luu = 0, luv = 0, lvv = 0;
            if (to_num(tok[static_cast<std::size_t>(c_luu)], luu)
                and to_num(tok[static_cast<std::size_t>(c_luv)], luv)
                and to_num(tok[static_cast<std::size_t>(c_lvv)], lvv))
            {
                r.cov_lidar(0, 0) = static_cast<float>(luu);
                r.cov_lidar(0, 1) = r.cov_lidar(1, 0) = static_cast<float>(luv);
                r.cov_lidar(1, 1) = static_cast<float>(lvv);
            }
        }
        cam.rows.push_back(r);
    }
    cam.has_cov_lidar = have_lid;
    if (bad > 0) std::printf("  ⚠ %s: %ld unparsable rows skipped\n", path.c_str(), bad);
    return not cam.rows.empty();
}

/// ── The injection ───────────────────────────────────────────────────────────────────────────────
/// The axes are the SAME ones J is built from (mount_lidar_pair.h): pitch = rotation about the
/// camera x axis, height = translation along camera z, yaw = rotation about camera z. Injecting on
/// the same axis the estimator differentiates is what makes "recovery fraction" mean anything.
enum class Axis { Yaw, Pitch, Height };

Eigen::Matrix3f rot_about(const Eigen::Vector3f& axis, float ang)
{
    return Eigen::AngleAxisf(ang, axis.normalized()).toRotationMatrix();
}

/// A camera-mount injection moves the point in CAMERA coordinates: pc' = R·pc (+ h·z for height),
/// which is the same as perturbing the mount that produced pc.
void inject_camera(Axis ax, float mag, Eigen::Matrix3f& R, Eigen::Vector3f& t)
{
    if (ax == Axis::Height) { t += mag * Eigen::Vector3f(0.f, 0.f, 1.f); return; }
    const Eigen::Vector3f axis = (ax == Axis::Yaw) ? Eigen::Vector3f(0.f, 0.f, 1.f)
                                                   : Eigen::Vector3f(1.f, 0.f, 0.f);
    const Eigen::Matrix3f Rr = rot_about(axis, mag);
    R = Rr * R;
    t = Rr * t;
}

/// A LiDAR injection rotates every corner about the LIDAR'S OWN ORIGIN in the robot frame — not
/// about the robot origin. That distinction IS the parallax the closure row turns on, so the origin
/// comes from the sidecar and the leg refuses if it was not resolvable.
Eigen::Vector3f inject_lidar(Axis ax, float mag, const Eigen::Vector3f& p_robot,
                             const Eigen::Vector3f& lidar_t_robot)
{
    if (ax == Axis::Height) return p_robot + mag * Eigen::Vector3f(0.f, 0.f, 1.f);
    // Yaw about the robot's vertical through the LiDAR; pitch about the robot x through it. Stated
    // in ROBOT axes on purpose: the helios hangs inverted, so "the LiDAR's own z" would need a sign
    // convention the sidecar does not carry, while "yaw" as a physical rotation is unambiguous.
    const Eigen::Vector3f axis = (ax == Axis::Yaw) ? Eigen::Vector3f(0.f, 0.f, 1.f)
                                                   : Eigen::Vector3f(1.f, 0.f, 0.f);
    return rot_about(axis, mag) * (p_robot - lidar_t_robot) + lidar_t_robot;
}

struct Leg
{
    std::string site;      ///< a camera name, or "lidar", or "" for the baseline
    Axis        axis = Axis::Yaw;
    float       mag  = 0.f;   ///< rad, or metres for Height
};

/// One camera's solve under one leg, plus the rebuilt rows the closure needs.
struct CamResult
{
    rc::mount::Accum::Solution sol;
    /// The raw normal equations, kept so `--verify` can compare the ACCUMULATION against the live
    /// evidence and not merely the row count. Two runs can agree on how many pairs they saw and
    /// still be weighting them differently, which is exactly the failure the covariance columns
    /// were added to rule out.
    Eigen::Matrix4d H = Eigen::Matrix4d::Zero();
    Eigen::Vector4d b = Eigen::Vector4d::Zero();
    double rTr = 0.0;
    long   n = 0;
    long   cov_held = 0;   ///< rows whose nominal Pxy would not invert, so the weight was NOT rebuilt
    /// (key, ts) → residual in RADIANS, the same quantity loop_closure_observe compares.
    std::vector<std::tuple<std::int64_t, int, double, double>> obs;   // ts, key, du_rad, dv_rad
};

/// The weight this row carried live, rebuilt under the INJECTED mount.
///
/// `cov = cov_image + cov_lidar + floor`, and only `cov_lidar = Pxy·cov_xy·Pxyᵀ` depends on the
/// extrinsic. So recover the corner's own covariance at the NOMINAL mount and push it through the
/// injected projection instead: `cov' = (cov − cov_lidar) + Pxy'·cov_xy·Pxy'ᵀ`.
/// ⚠ `Pxy` can be singular — a bearing-only direction at a shallow corner has no metric inverse — so
///   a row whose nominal Pxy will not invert keeps its recorded covariance and is COUNTED, not
///   silently patched.
bool rebuilt_cov(const Eigen::Matrix2f& cov, const Eigen::Matrix2f& cov_lidar,
                 const Eigen::Matrix<float, 2, 3>& P0, const Eigen::Matrix3f& R0,
                 const Eigen::Matrix<float, 2, 3>& P1, const Eigen::Matrix3f& R1,
                 Eigen::Matrix2f& out)
{
    const Eigen::Matrix2f Pxy0 = (P0 * R0).leftCols<2>();
    const Eigen::Matrix2f Pxy1 = (P1 * R1).leftCols<2>();
    const float det = Pxy0.determinant();
    if (not std::isfinite(det) or std::abs(det) < 1e-9f) return false;
    const Eigen::Matrix2f inv = Pxy0.inverse();
    const Eigen::Matrix2f cov_xy = inv * cov_lidar * inv.transpose();
    out = (cov - cov_lidar) + Pxy1 * cov_xy * Pxy1.transpose();
    return out.allFinite();
}

CamResult solve_leg(const Camera& c, const Leg& leg, double offset_sigma_px, bool fixed_cov,
                    bool& refused, std::string& why)
{
    CamResult out;
    refused = false;
    Eigen::Matrix3f R = c.ctx.cam_R_robot;
    Eigen::Vector3f t = c.ctx.cam_t_robot;
    const bool lidar_leg = (leg.site == "lidar");
    if (lidar_leg and not c.ctx.lidar_known)
    {
        refused = true;
        why = "the sidecar carries no lidar_t_robot, and a LiDAR rotation about the wrong centre is"
              " a different experiment — not an approximation of this one";
        return out;
    }
    if (not leg.site.empty() and not lidar_leg and leg.site == c.name)
        inject_camera(leg.axis, leg.mag, R, t);

    rc::mount::Accum acc;
    acc.offset_sigma_px = offset_sigma_px;
    out.obs.reserve(c.rows.size());
    // A leg that moves nothing must reproduce the live solve BIT FOR BIT — that is the only
    // self-check this tool has — so the covariance rebuild is skipped when there is no injection,
    // rather than round-tripped through an inverse that would perturb the last digits.
    const bool moved = not leg.site.empty() and leg.mag != 0.f;
    const bool do_rebuild = moved and not fixed_cov and c.has_cov_lidar;
    for (const Row& r : c.rows)
    {
        const Eigen::Vector3f p = lidar_leg
            ? inject_lidar(leg.axis, leg.mag, r.p_robot, c.ctx.lidar_t_robot)
            : r.p_robot;
        Eigen::Matrix2f cov = r.cov;
        if (do_rebuild)
        {
            // The nominal geometry of the SAME row, only to get Pxy at the mount the covariance was
            // recorded under.
            const rc::mount::PairObs o0 = rc::mount::make_pair_from(
                r.vertex, r.assoc_prob, r.p_robot, r.uv_image, r.cov, c.ctx.cam,
                c.ctx.cam_R_robot, c.ctx.cam_t_robot,
                c.ctx.sigma_pitch, c.ctx.sigma_height, c.ctx.sigma_yaw);
            const rc::mount::PairObs o1 = rc::mount::make_pair_from(
                r.vertex, r.assoc_prob, p, r.uv_image, r.cov, c.ctx.cam, R, t,
                c.ctx.sigma_pitch, c.ctx.sigma_height, c.ctx.sigma_yaw);
            Eigen::Matrix2f cv;
            if (o0.ok and o1.ok and rebuilt_cov(r.cov, r.cov_lidar, o0.P, c.ctx.cam_R_robot,
                                                o1.P, R, cv))
                cov = cv;
            else
                ++out.cov_held;
        }
        const rc::mount::PairObs o =
            rc::mount::make_pair_from(r.vertex, r.assoc_prob, p, r.uv_image, cov, c.ctx.cam,
                                      R, t, c.ctx.sigma_pitch, c.ctx.sigma_height, c.ctx.sigma_yaw);
        if (not o.ok) continue;
        acc.add(o);
        ++out.n;
        // The SAME local scale the agent now uses, per row (image_edge_ops.h px_per_rad_at).
        const Eigen::Vector2f ppr = rc::img::px_per_rad_at(c.ctx.cam, o.uv_image);
        if (ppr.x() > 0.f and ppr.y() > 0.f)
            out.obs.emplace_back(r.ts, r.vertex * 2 + (r.ceiling ? 1 : 0),
                                 static_cast<double>(o.r.x() / ppr.x()),
                                 static_cast<double>(o.r.y() / ppr.y()));
    }
    out.sol = acc.solve();
    out.H = acc.H; out.b = acc.b; out.rTr = acc.rTr;
    return out;
}

/// The closure, recomputed from the two cameras' REBUILT rows by loop_closure_observe's own rule:
/// same corner, different camera, within 60 ms, differenced in radians. Recomputing it (rather than
/// reading etc/camera_loop.csv) is what makes it respond to an injection at all.
struct Closure { double du_deg = 0, dv_deg = 0, sd_du_deg = 0, sd_dv_deg = 0; long n = 0; };

Closure closure_of(const CamResult& a, const CamResult& b, std::int64_t max_dt_ms)
{
    // Merge by key, then walk each key's two time series once.
    std::map<int, std::vector<std::pair<std::int64_t, Eigen::Vector2d>>> A, B;
    for (const auto& [ts, key, du, dv] : a.obs) A[key].emplace_back(ts, Eigen::Vector2d(du, dv));
    for (const auto& [ts, key, du, dv] : b.obs) B[key].emplace_back(ts, Eigen::Vector2d(du, dv));
    Closure c;
    double su = 0, sv = 0, su2 = 0, sv2 = 0;
    for (auto& [key, va] : A)
    {
        const auto itb = B.find(key);
        if (itb == B.end()) continue;
        auto& vb = itb->second;
        std::ranges::sort(va, {}, &std::pair<std::int64_t, Eigen::Vector2d>::first);
        std::ranges::sort(vb, {}, &std::pair<std::int64_t, Eigen::Vector2d>::first);
        std::size_t j = 0;
        for (const auto& [ts, ra] : va)
        {
            while (j + 1 < vb.size() and std::abs(vb[j + 1].first - ts) <= std::abs(vb[j].first - ts)) ++j;
            if (std::abs(vb[j].first - ts) > max_dt_ms) continue;
            const double ddu = ra.x() - vb[j].second.x(), ddv = ra.y() - vb[j].second.y();
            su += ddu; sv += ddv; su2 += ddu * ddu; sv2 += ddv * ddv; ++c.n;
        }
    }
    if (c.n > 0)
    {
        const double mu = su / static_cast<double>(c.n), mv = sv / static_cast<double>(c.n);
        c.du_deg = mu * kRad2Deg;
        c.dv_deg = mv * kRad2Deg;
        c.sd_du_deg = std::sqrt(std::max(0.0, su2 / static_cast<double>(c.n) - mu * mu)) * kRad2Deg;
        c.sd_dv_deg = std::sqrt(std::max(0.0, sv2 / static_cast<double>(c.n) - mv * mv)) * kRad2Deg;
    }
    return c;
}

/// Parameters come back in units of the prior sigma, with the sign convention of the live report
/// (`-sol.p(i) * sigma`), so these are directly comparable with what the agent logs.
double param_deg(const rc::mount::Accum::Solution& s, int i, const rc::mount::ReplayContext& c)
{
    const double sig = (i == 0) ? c.sigma_pitch : (i == 1) ? c.sigma_height : c.sigma_yaw;
    return -s.p(i) * sig * ((i == 1) ? 1.0 : kRad2Deg);   // height in metres, angles in degrees
}
double sigma_deg(const rc::mount::Accum::Solution& s, int i, const rc::mount::ReplayContext& c)
{
    const double sig = (i == 0) ? c.sigma_pitch : (i == 1) ? c.sigma_height : c.sigma_yaw;
    return s.sigma(i) * sig * ((i == 1) ? 1.0 : kRad2Deg);
}

const char* axis_name(Axis a) { return a == Axis::Yaw ? "yaw" : a == Axis::Pitch ? "pitch" : "height"; }

// ════ SELFTEST ══════════════════════════════════════════════════════════════════════════════════
// A drive whose truth is set here, so the tool is shown to recover a KNOWN injection before it is
// ever pointed at a real one. The pre-registration turns on three signatures being distinguishable
// (§2b); a tool that cannot tell them apart on data built to have them would say nothing about data
// that might not.
//
// ★ The rows are generated at the NOMINAL mount, so the baseline residual is noise only and every
//   leg's answer is the injection itself. The nuisance is left OFF here on purpose: this test asks
//   whether the REPLAY works, and the nuisance's own behaviour was verified against synthetic truth
//   when it was written (f634c87).
namespace selftest
{
std::uint32_t rng_state = 12345u;
float urand()   // deterministic: a failing selftest must be reproducible
{
    rng_state = rng_state * 1664525u + 1013904223u;
    return static_cast<float>((rng_state >> 8) & 0xFFFFFu) / static_cast<float>(0x100000);
}
float nrand() { return (urand() + urand() + urand() + urand() - 2.f) * 1.2f; }   // ~N(0,1)

/// ★ The two cameras are modelled as the REAL pair — the ricoh a 360 panorama, the zed a pinhole —
///   because the two do not convert pixels to angles the same way, and the closure differences them
///   in ANGLE. `px_per_rad` is exact for the panorama and, in its own words, "exact at the principal
///   point" for the pinhole: off axis the true scale is fx·sec²θ. Modelling both as pinholes hid
///   that behind a symmetric error; modelling them as they are is what lets the selftest measure it.
rc::mount::ReplayContext make_ctx(const std::string& name, const Eigen::Vector3f& cam_pos_robot,
                                  const Eigen::Vector3f& lidar_pos_robot, bool panorama)
{
    rc::mount::ReplayContext c;
    c.robot = "Synth";
    c.camera = name;
    if (panorama)
    {
        c.cam.kind = rc::CameraModel::Kind::Equirect;
        c.cam.width = 1920.f; c.cam.height = 960.f;
        c.cam.cx = 960.f; c.cam.cy = 480.f;
        c.cam.fx = c.cam.fy = 0.f;
        c.cam.azimuth_sign = 1.f; c.cam.azimuth_offset = 0.f;
    }
    else
    {
        c.cam.kind = rc::CameraModel::Kind::Pinhole;
        c.cam.fx = c.cam.fy = 700.f;
        c.cam.cx = 640.f; c.cam.cy = 360.f;
        c.cam.width = 1280.f; c.cam.height = 720.f;
    }
    c.cam.valid = true;
    // ★ THIS CODEBASE'S PINHOLE CONVENTION IS y = DEPTH, z = UP (image_edge_ops.h, and the ZED's
    //   own frame). A z-forward camera — the textbook one — projects nothing here, which is how this
    //   selftest first came out with zero rows. Camera x = right = −robot y, y = depth = robot x,
    //   z = up = robot z; that alignment is also why "yaw about camera z" IS yaw about the robot.
    Eigen::Matrix3f R;
    R << 0.f, -1.f, 0.f,
         1.f,  0.f, 0.f,
         0.f,  0.f, 1.f;
    c.cam_R_robot = R;
    c.cam_t_robot = -R * cam_pos_robot;            // p_cam = R (p_robot - cam_pos)
    c.sigma_pitch = 0.0035f; c.sigma_height = 0.010f; c.sigma_yaw = 0.0035f;
    c.lidar_t_robot = lidar_pos_robot;
    c.lidar_known = true;
    return c;
}

/// 23 vertices, each seen from many places — the real drive's shape (a few corners, many sightings),
/// which is exactly the clustering that makes the honest yaw sigma large.
Camera make_camera(const std::string& name, const Eigen::Vector3f& cam_pos,
                   const Eigen::Vector3f& lidar_pos, int n_vertex, int n_frames, bool panorama)
{
    Camera cam;
    cam.name = name;
    cam.ctx = make_ctx(name, cam_pos, lidar_pos, panorama);
    cam.has_cov_lidar = true;
    for (int f = 0; f < n_frames; ++f)
        for (int v = 0; v < n_vertex; ++v)
        {
            // The corner in the ROBOT frame: ahead of the robot, spread in bearing and range, and
            // moving frame to frame as the robot drives past it.
            const float range = 2.f + 6.f * urand();
            const float bear  = (static_cast<float>(v) / n_vertex - 0.5f) * 1.2f + 0.15f * nrand();
            const float z     = (v % 2 == 0) ? 0.05f : 2.45f;      // floor and ceiling corners
            const Eigen::Vector3f p(range * std::cos(bear), range * std::sin(bear), z);
            const Eigen::Vector3f pc = cam.ctx.cam_R_robot * p + cam.ctx.cam_t_robot;
            if (pc.y() < 0.5f) continue;   // y is DEPTH here
            Eigen::Vector2d uv;
            if (not rc::img::project_with_model(cam.ctx.cam, pc.cast<double>(), uv)) continue;
            if (uv.x() < 0 or uv.x() > cam.ctx.cam.width or uv.y() < 0 or uv.y() > cam.ctx.cam.height) continue;
            Row r;
            r.ts = 100 * f;                       // both cameras land on the same stamp: shared corners
            r.vertex = v;
            r.ceiling = (z > 1.f);
            r.uv_image = Eigen::Vector2f(static_cast<float>(uv.x()) + 0.5f * nrand(),
                                         static_cast<float>(uv.y()) + 0.5f * nrand());
            r.p_robot = p;
            // The LiDAR half of the weight, built the way make_pair builds it, so the replay's
            // reconstruction has something real to reconstruct.
            Eigen::Matrix<double, 2, 3> P;
            if (not rc::img::project_jacobian_model(cam.ctx.cam, pc.cast<double>(), P)) continue;
            const Eigen::Matrix2f Pxy = (P.cast<float>() * cam.ctx.cam_R_robot).leftCols<2>();
            const Eigen::Matrix2f cov_xy = (0.02f * 0.02f) * Eigen::Matrix2f::Identity();
            r.cov_lidar = Pxy * cov_xy * Pxy.transpose();
            r.cov = 0.25f * Eigen::Matrix2f::Identity() + r.cov_lidar
                    + 0.01f * Eigen::Matrix2f::Identity();
            cam.rows.push_back(r);
        }
    return cam;
}

int run()
{
    std::printf("mount_replay selftest — a drive whose truth is known\n\n");
    const Eigen::Vector3f lidar(0.00f, 0.00f, 1.075f);
    std::vector<Camera> cams{ make_camera("ricoh", Eigen::Vector3f(0.02f, 0.00f, 1.02f), lidar, 23, 60, true),
                              make_camera("zed",   Eigen::Vector3f(0.18f, 0.09f, 0.945f), lidar, 23, 60, false) };
    std::printf("  ricoh %zu rows, zed %zu rows\n\n", cams[0].rows.size(), cams[1].rows.size());

    int failures = 0;
    const auto check = [&](const char* what, bool ok, const std::string& detail) {
        std::printf("  %-58s %s   %s\n", what, ok ? "PASS" : "FAIL", detail.c_str());
        if (not ok) ++failures;
    };
    const auto fmt = [](const char* f, double a, double b = 0.0) {
        char buf[160]; std::snprintf(buf, sizeof buf, f, a, b); return std::string(buf);
    };

    bool refused = false; std::string why;
    std::vector<CamResult> base;
    for (const Camera& c : cams) base.push_back(solve_leg(c, Leg{}, 0.0, false, refused, why));
    const Closure base_cl = closure_of(base[0], base[1], 60);
    check("baseline yaw is zero on data generated at the nominal mount",
          std::abs(param_deg(base[0].sol, 2, cams[0].ctx)) < 0.02,
          fmt("ricoh yaw %+.4f deg", param_deg(base[0].sol, 2, cams[0].ctx)));

    // ── Leg 1: inject the ricoh's own mount ─────────────────────────────────────────────────────
    const double dlt = 1.0;                       // deg
    Leg L1{"ricoh", Axis::Yaw, static_cast<float>(dlt / kRad2Deg)};
    std::vector<CamResult> l1;
    for (const Camera& c : cams) l1.push_back(solve_leg(c, L1, 0.0, false, refused, why));
    const double d_ricoh = param_deg(l1[0].sol, 2, cams[0].ctx) - param_deg(base[0].sol, 2, cams[0].ctx);
    const double d_zed   = param_deg(l1[1].sol, 2, cams[1].ctx) - param_deg(base[1].sol, 2, cams[1].ctx);
    check("ricoh injection: the ricoh mount recovers it", std::abs(std::abs(d_ricoh / dlt) - 1.0) < 0.05,
          fmt("moved %+.4f deg, recovery %.3f", d_ricoh, d_ricoh / dlt));
    check("ricoh injection: the zed mount does NOT move", std::abs(d_zed) < 0.02,
          fmt("zed moved %+.4f deg", d_zed));
    const Closure c1 = closure_of(l1[0], l1[1], 60);
    // The panorama's px→rad scale is exact, so a ricoh injection must arrive in the closure at
    // full size. (A ZED injection would NOT: it converts through fx, which is exact only at the
    // principal point — see the leakage the LiDAR leg reports below.)
    check("ricoh injection: the closure moves by the injection",
          std::abs(std::abs(c1.du_deg - base_cl.du_deg) / dlt - 1.0) < 0.05,
          fmt("closure du moved %+.4f deg (%.3f of the injection)", c1.du_deg - base_cl.du_deg,
              (c1.du_deg - base_cl.du_deg) / dlt));

    // ── Leg 2: inject the LIDAR — the falsifier row ─────────────────────────────────────────────
    Leg L2{"lidar", Axis::Yaw, static_cast<float>(dlt / kRad2Deg)};
    std::vector<CamResult> l2;
    for (const Camera& c : cams) l2.push_back(solve_leg(c, L2, 0.0, false, refused, why));
    const double l2_ricoh = param_deg(l2[0].sol, 2, cams[0].ctx) - param_deg(base[0].sol, 2, cams[0].ctx);
    const double l2_zed   = param_deg(l2[1].sol, 2, cams[1].ctx) - param_deg(base[1].sol, 2, cams[1].ctx);
    check("LiDAR injection: BOTH mounts move by it",
          std::abs(std::abs(l2_ricoh / dlt) - 1.0) < 0.10 and std::abs(std::abs(l2_zed / dlt) - 1.0) < 0.10,
          fmt("ricoh %+.4f, zed %+.4f deg", l2_ricoh, l2_zed));
    const Closure c2 = closure_of(l2[0], l2[1], 60);
    const double leak = c2.du_deg - base_cl.du_deg;
    check("LiDAR injection: the closure keeps it, up to parallax",
          std::abs(leak) < 0.25 * dlt,
          fmt("closure du moved %+.4f deg = %.1f%% of the injection — the PARALLAX the published"
              " table wrote as exactly 0", leak, 100.0 * leak / dlt));

    // ── Splitting the leakage: geometry, or a defect in the live channel? ───────────────────────
    // The leftover above has two sources and they have different remedies. PARALLAX is unavoidable —
    // the two cameras sit at different distances from the LiDAR, so one sees a rotation about it
    // slightly differently. The SCALE error is not: `px_per_rad` returns fx for a pinhole, exact only
    // at the principal point, while the true local scale is fx·sec²θ — so the ZED's contribution to
    // the closure is inflated off-axis. Re-running the same injection with BOTH cameras modelled as
    // panoramas (exact scale, unchanged geometry) leaves parallax alone, and the difference is the
    // part that is worth fixing in loop_closure_observe rather than tolerated in the falsifier.
    {
        std::vector<Camera> exact{ cams[0],
                                   make_camera("zed", Eigen::Vector3f(0.18f, 0.09f, 0.945f), lidar,
                                               23, 60, /*panorama=*/true) };
        std::vector<CamResult> eb, el;
        for (const Camera& c : exact) eb.push_back(solve_leg(c, Leg{}, 0.0, false, refused, why));
        for (const Camera& c : exact) el.push_back(solve_leg(c, L2, 0.0, false, refused, why));
        const double leak_geom = closure_of(el[0], el[1], 60).du_deg - closure_of(eb[0], eb[1], 60).du_deg;
        std::printf("\n  LiDAR-injection leakage into the closure: %.4f deg total\n"
                    "     parallax alone (both cameras on an exact px->rad scale): %.4f deg\n"
                    "     the pinhole fx approximation in px_per_rad:              %.4f deg\n"
                    "  ⇒ the falsifier's tolerance is NOT zero, and %.0f%% of it is a scale error the\n"
                    "    live loop_closure_observe could remove.\n",
                    leak, leak_geom, leak - leak_geom,
                    (std::abs(leak) > 1e-9) ? 100.0 * std::abs(leak - leak_geom) / std::abs(leak) : 0.0);
    }

    // ── The covariance rebuild: how much did the old approximation matter? ──────────────────────
    std::vector<CamResult> l1_fixed;
    for (const Camera& c : cams) l1_fixed.push_back(solve_leg(c, L1, 0.0, true, refused, why));
    const double d_fixed = param_deg(l1_fixed[0].sol, 2, cams[0].ctx) - param_deg(base[0].sol, 2, cams[0].ctx);
    std::printf("\n  cov rebuilt %+.5f deg vs held fixed %+.5f deg — the approximation §2b was going"
                " to assume small is %.3f%% of the injection\n",
                d_ricoh, d_fixed, 100.0 * std::abs(d_ricoh - d_fixed) / dlt);

    // ── The refusal, which is a behaviour and therefore testable ────────────────────────────────
    Camera blind = cams[0];
    blind.ctx.lidar_known = false;
    solve_leg(blind, L2, 0.0, false, refused, why);
    check("a LiDAR leg REFUSES when the sidecar has no lidar origin", refused, why.substr(0, 60));

    std::printf("\n%s\n", failures == 0 ? "ALL PASS" : "FAILURES");
    return failures == 0 ? 0 : 1;
}
} // namespace selftest

void usage()
{
    std::printf(
        "mount_replay — arm 7: replay a recorded drive under a deliberately wrong extrinsic.\n\n"
        "  --pair FILE        a pair CSV (repeatable; the sidecar etc/image_edge_replay_<cam>.txt\n"
        "                     is found beside it). Two cameras are needed for the closure channel.\n"
        "  --inject SITE:AXIS=VALUE   SITE = a camera name or 'lidar'; AXIS = yaw|pitch|height;\n"
        "                     VALUE in degrees (yaw/pitch) or metres (height). Repeatable: each is\n"
        "                     run as its own leg against the same baseline.\n"
        "  --sigma-px X       per-vertex offset nuisance prior (px). 0 = the pre-2026-09-02 solve.\n"
        "                     Pass it TWICE-over by running with 0 and with 5.3: the contrast is\n"
        "                     outcome 3 of the pre-registration, not a tuning knob.\n"
        "  --closure-ms N     max stamp difference for a shared corner (default 60, the live rule).\n"
        "  --fixed-cov        do NOT rebuild the covariance under the injected mount (the old\n"
        "                     approximation). Run it both ways: the difference is the size of what\n"
        "                     §2b was going to assume away.\n"
        "  --verify FILE      check the delta=0 solve against a live evidence file.\n"
        "  --selftest         replay a drive whose truth is set in the tool, and check that the\n"
        "                     three injection signatures come out distinguishable. Run this before\n"
        "                     trusting a leg on real data.\n"
        "  --help\n");
}
} // namespace

int main(int argc, char** argv)
{
    std::vector<std::string> pair_files;
    std::vector<Leg> legs;
    double sigma_px = 0.0;
    bool fixed_cov = false;
    std::int64_t closure_ms = 60;
    std::string verify_file;

    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        const auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
        if (a == "--help") { usage(); return 0; }
        else if (a == "--selftest") return selftest::run();
        else if (a == "--pair") pair_files.push_back(next());
        else if (a == "--sigma-px") { double v = 0; if (to_num(next(), v)) sigma_px = v; }
        else if (a == "--closure-ms") { double v = 0; if (to_num(next(), v)) closure_ms = static_cast<std::int64_t>(v); }
        else if (a == "--verify") verify_file = next();
        else if (a == "--fixed-cov") fixed_cov = true;
        else if (a == "--inject")
        {
            const std::string spec = next();
            const auto colon = spec.find(':');
            const auto eq = spec.find('=');
            if (colon == std::string::npos or eq == std::string::npos or eq < colon)
            { std::printf("bad --inject '%s' (want SITE:AXIS=VALUE)\n", spec.c_str()); return 2; }
            Leg L;
            L.site = spec.substr(0, colon);
            const std::string ax = spec.substr(colon + 1, eq - colon - 1);
            double v = 0;
            if (not to_num(std::string_view(spec).substr(eq + 1), v))
            { std::printf("bad --inject value in '%s'\n", spec.c_str()); return 2; }
            if (ax == "yaw") { L.axis = Axis::Yaw; L.mag = static_cast<float>(v / kRad2Deg); }
            else if (ax == "pitch") { L.axis = Axis::Pitch; L.mag = static_cast<float>(v / kRad2Deg); }
            else if (ax == "height") { L.axis = Axis::Height; L.mag = static_cast<float>(v); }
            else { std::printf("bad axis '%s' (yaw|pitch|height)\n", ax.c_str()); return 2; }
            legs.push_back(L);
        }
        else { std::printf("unknown argument '%s'\n", a.c_str()); usage(); return 2; }
    }
    if (pair_files.empty()) { usage(); return 2; }

    // ── Load ────────────────────────────────────────────────────────────────────────────────────
    std::vector<Camera> cams;
    for (const std::string& p : pair_files)
    {
        Camera c;
        // The sidecar sits beside the CSV: image_edge_pair_<cam>.csv → image_edge_replay_<cam>.txt.
        const auto slash = p.find_last_of('/');
        const std::string dir = (slash == std::string::npos) ? std::string() : p.substr(0, slash + 1);
        const std::string base = (slash == std::string::npos) ? p : p.substr(slash + 1);
        const std::string tag = "image_edge_pair_";
        if (not base.starts_with(tag) or not base.ends_with(".csv"))
        { std::printf("✗ %s: expected etc/image_edge_pair_<cam>.csv\n", p.c_str()); return 1; }
        c.name = base.substr(tag.size(), base.size() - tag.size() - 4);
        const std::string side = dir + "image_edge_replay_" + c.name + ".txt";
        if (not rc::mount::read_replay_context(side, c.ctx))
        {
            std::printf("✗ %s: no usable run-constants sidecar. A pair row cannot be rebuilt under a\n"
                        "  different mount without the camera model and the nominal extrinsic, and\n"
                        "  hard-coding them is how an offline tool and the agent come to disagree.\n",
                        side.c_str());
            return 1;
        }
        std::string err;
        if (not load_pairs(p, c, err)) { std::printf("✗ %s\n", err.c_str()); return 1; }
        std::printf("  %-8s %8zu rows   mount yaw axis from %s   lidar origin %s\n",
                    c.name.c_str(), c.rows.size(), side.c_str(),
                    c.ctx.lidar_known ? "known" : "UNRESOLVED (LiDAR leg will refuse)");
        cams.push_back(std::move(c));
    }

    // ── Baseline: delta = 0 ─────────────────────────────────────────────────────────────────────
    std::printf("\n── baseline (no injection), nuisance sigma %.2f px ──\n", sigma_px);
    const Leg base_leg{};
    std::vector<CamResult> base;
    for (const Camera& c : cams)
    {
        bool refused = false; std::string why;
        CamResult r = solve_leg(c, base_leg, sigma_px, fixed_cov, refused, why);
        std::printf("  %-8s n=%ld clusters=%d  pitch %+8.4f deg ± %.4f | height %+8.4f m ± %.4f |"
                    " yaw %+8.4f deg ± %.4f  chi2/dof %.2f\n",
                    c.name.c_str(), r.n, r.sol.clusters,
                    param_deg(r.sol, 0, c.ctx), sigma_deg(r.sol, 0, c.ctx),
                    param_deg(r.sol, 1, c.ctx), sigma_deg(r.sol, 1, c.ctx),
                    param_deg(r.sol, 2, c.ctx), sigma_deg(r.sol, 2, c.ctx), r.sol.chi2_dof);
        if (not r.sol.ok) std::printf("    ⚠ solve not ok (too few pairs?)\n");
        base.push_back(std::move(r));
    }
    Closure base_cl;
    if (cams.size() >= 2)
    {
        base_cl = closure_of(base[0], base[1], closure_ms);
        std::printf("  closure %s−%s: du %+8.4f ± %.4f deg, dv %+8.4f ± %.4f deg over %ld shared corners\n",
                    cams[0].name.c_str(), cams[1].name.c_str(), base_cl.du_deg, base_cl.sd_du_deg,
                    base_cl.dv_deg, base_cl.sd_dv_deg, base_cl.n);
    }
    else
        std::printf("  (one camera only — the closure channel, and with it the attribution test,"
                    " needs two)\n");

    // ── The delta = 0 self-check against the live evidence ──────────────────────────────────────
    if (not verify_file.empty())
    {
        std::printf("\n── verify: delta=0 replay vs the live evidence in %s ──\n", verify_file.c_str());
        std::printf("  ⚠ this can only agree if the live pool held THIS run's rows alone — the pool\n"
                    "    resumes across sessions while the CSV is truncated each run.\n");
        std::ifstream f(verify_file);
        if (not f.is_open()) std::printf("  ✗ cannot open it\n");
        else
        {
            // The evidence file is (H, b, rTr, n) — the aggregate the old model solved. Comparing n
            // is the cheap half; comparing the SOLVE is the half that matters, and the caller can do
            // that against the agent's own [camcal] log line for the same run.
            std::string line; long n_live = -1;
            double rTr_live = 0.0;
            Eigen::Matrix4d H_live = Eigen::Matrix4d::Zero();
            Eigen::Vector4d b_live = Eigen::Vector4d::Zero();
            std::vector<std::string_view> tok;
            while (std::getline(f, line))
            {
                if (line.empty() or line[0] == '#') continue;
                split(line, tok);
                double v = 0, i = 0, j = 0;
                if (tok[0] == "n" and tok.size() > 1 and to_num(tok[1], v)) n_live = static_cast<long>(v);
                else if (tok[0] == "rTr" and tok.size() > 1) to_num(tok[1], rTr_live);
                else if (tok[0] == "H" and tok.size() > 3 and to_num(tok[1], i) and to_num(tok[2], j)
                         and to_num(tok[3], v))
                    H_live(static_cast<int>(i), static_cast<int>(j)) =
                        H_live(static_cast<int>(j), static_cast<int>(i)) = v;
                else if (tok[0] == "b" and tok.size() > 2 and to_num(tok[1], i) and to_num(tok[2], v))
                    b_live(static_cast<int>(i)) = v;
            }
            const CamResult& r0 = base[0];
            std::printf("  pairs: live %ld, replay %ld  %s\n", n_live, r0.n,
                        (n_live == r0.n) ? "✓ same rows"
                                         : "✗ DIFFERENT — the two are not describing one run");
            // The accumulation itself. A relative comparison, because H's entries span orders of
            // magnitude and an absolute tolerance would be a different test on each one.
            const auto rel = [](double a, double c) {
                const double d = std::max(std::abs(a), std::abs(c));
                return (d > 1e-12) ? std::abs(a - c) / d : 0.0;
            };
            double worst = 0.0; int wi = 0, wj = 0;
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    if (const double e = rel(H_live(i, j), r0.H(i, j)); e > worst) { worst = e; wi = i; wj = j; }
            double worst_b = 0.0; int wb = 0;
            for (int i = 0; i < 4; ++i)
                if (const double e = rel(b_live(i), r0.b(i)); e > worst_b) { worst_b = e; wb = i; }
            const double worst_r = rel(rTr_live, r0.rTr);
            std::printf("  worst relative difference:  H(%d,%d) %.3e   b(%d) %.3e   rTr %.3e\n",
                        wi, wj, worst, wb, worst_b, worst_r);
            const double tol = 1e-5;   // float rows, double accumulation: the CSV's own printed precision
            const bool ok = n_live == r0.n and worst < tol and worst_b < tol and worst_r < tol;
            std::printf("  %s\n", ok
                ? "✓ the replay reproduces the live accumulation — a leg's numbers can be believed"
                : "✗ THE REPLAY IS NOT THE LIVE SOLVE. Every injection result below is void until this"
                  " agrees: a difference here is either a stale evidence file (pool resumed across"
                  " sessions) or a real divergence between the two paths.");
        }
    }

    // ── The legs ────────────────────────────────────────────────────────────────────────────────
    for (const Leg& L : legs)
    {
        const double shown = (L.axis == Axis::Height) ? L.mag : L.mag * kRad2Deg;
        std::printf("\n── inject %s %s %+.4f %s ──\n", L.site.c_str(), axis_name(L.axis), shown,
                    (L.axis == Axis::Height) ? "m" : "deg");
        const int pi = (L.axis == Axis::Pitch) ? 0 : (L.axis == Axis::Height) ? 1 : 2;
        std::vector<CamResult> leg;
        bool any_refused = false;
        for (std::size_t k = 0; k < cams.size(); ++k)
        {
            bool refused = false; std::string why;
            CamResult r = solve_leg(cams[k], L, sigma_px, fixed_cov, refused, why);
            if (refused)
            {
                std::printf("  %-8s REFUSED: %s\n", cams[k].name.c_str(), why.c_str());
                any_refused = true;
                leg.push_back(std::move(r));
                continue;
            }
            const double d = param_deg(r.sol, pi, cams[k].ctx) - param_deg(base[k].sol, pi, cams[k].ctx);
            const double sg = sigma_deg(r.sol, pi, cams[k].ctx);
            std::printf("  %-8s %s moved %+8.4f (± %.4f) against an injection of %+.4f"
                        "  ⇒ recovery %.3f  [%.2f sigma]\n",
                        cams[k].name.c_str(), axis_name(L.axis), d, sg, shown,
                        (std::abs(shown) > 0) ? d / shown : 0.0,
                        (sg > 0) ? std::abs(d) / sg : 0.0);
            if (r.cov_held > 0)
                std::printf("           %ld of %zu rows kept their recorded weight (singular Pxy)\n",
                            r.cov_held, cams[k].rows.size());
            leg.push_back(std::move(r));
        }
        if (cams.size() >= 2 and not any_refused)
        {
            const Closure cl = closure_of(leg[0], leg[1], closure_ms);
            const double dd = cl.du_deg - base_cl.du_deg;
            // The closure's own precision on these rows, not a nominal one: sd/sqrt(n) of the very
            // differences being compared.
            const double se = (cl.n > 0) ? cl.sd_du_deg / std::sqrt(static_cast<double>(cl.n)) : 0.0;
            std::printf("  closure  du moved %+8.4f deg (se %.4f, n %ld)  ⇒ %.3f of the injection\n",
                        dd, se, cl.n, (std::abs(shown) > 0) ? dd / shown : 0.0);
            if (L.site == "lidar")
                std::printf("    ★ the falsifier: the attribution rule says a LiDAR error moves BOTH\n"
                            "      mounts and leaves this UNCHANGED. Unchanged means within the\n"
                            "      parallax above — the two cameras sit at different distances from\n"
                            "      the LiDAR, so exact cancellation was never the prediction.\n");
        }
    }
    std::printf("\n");
    return 0;
}
