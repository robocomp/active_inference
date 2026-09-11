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

// ── THE PARAMETER PROBE, AND WHY IT BORROWS THE DEAD COLUMN ──────────────────────────────────────
// The pool estimates 3 of the mount's 6 degrees of freedom: pitch about x̂_cam, height along ẑ_cam,
// yaw about ẑ_cam. `J`'s fourth column is the `dt` slot, which make_pair_from leaves at zero because
// the image/LiDAR offset is not a property of the mount. This borrows that slot to carry ONE
// candidate parameter at a time, so "is a parameter missing?" is answered by the AGENT'S OWN
// marginalised solve rather than by a replica of it — the same reason every other leg here does.
//   0 roll      rotation about the camera frame's ŷ (the depth axis of a pinhole; a fixed direction
//               in the panorama frame, where there is no single optical axis)
//   1 t_lateral translation along x̂_cam
//   2 t_depth   translation along ŷ_cam
//   3 NULL      a deterministic pseudo-random column, the negative control. Whatever the real
//               candidates score that this scores too is the fit's slack and not a parameter.
int          g_probe = -1;          ///< -1 = the live 3-parameter model, unchanged
float        g_probe_sigma = 0.f;   ///< the candidate's prior sigma, so it stays in prior-sigma units
int          g_skip_vertex = -9999; ///< leave-one-CORNER-out: the corner is the sample unit
std::int64_t g_win_lo = 0, g_win_hi = 0;   ///< 0,0 = no window filter

const char* probe_name(int i)
{ return i == 0 ? "roll" : i == 1 ? "t_lateral" : i == 2 ? "t_depth" : i == 3 ? "NULL-control" : "?"; }
const char* probe_unit(int i) { return i == 0 ? "deg" : i == 3 ? "-" : "m"; }

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
    /// The self-calibration correction in force when this row was written (pitch rad, height m,
    /// yaw rad). With the live loop on, the mount MOVES during a run; a row must be rebuilt against
    /// the mount that actually measured it, not the one the file happened to open with.
    Eigen::Vector3f corr = Eigen::Vector3f::Zero();
    float           assoc_prob = 1.f;
    Eigen::Vector3f p_robot = Eigen::Vector3f::Zero();
};

struct Camera
{
    /// A mount correction applied before anything else, so the BASELINE is evaluated under it.
    /// This is factor B of the chapter's 2x2: B0 leaves it zero (the nominal extrinsic from the
    /// graph), B1 sets it to the self-calibrated estimate. The rows are the same measured image
    /// points in both, which is what makes the pairing exact — changing the mount changes the
    /// PREDICTION and leaves the measurement untouched.
    Eigen::Vector3f mount_apply = Eigen::Vector3f::Zero();   // pitch rad, height m, yaw rad
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
    int c_cp = 0, c_ch = 0, c_cy = 0;
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
    // Written only by a binary that can move the mount mid-run. Absent means the mount was static,
    // which is exactly what a zero correction encodes — so this one CAN default, and safely.
    const bool have_corr = need("corr_pitch", c_cp) and need("corr_height", c_ch) and need("corr_yaw", c_cy);
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
        if (have_corr)
        {
            double cp = 0, ch = 0, cy = 0;
            if (to_num(tok[static_cast<std::size_t>(c_cp)], cp)
                and to_num(tok[static_cast<std::size_t>(c_ch)], ch)
                and to_num(tok[static_cast<std::size_t>(c_cy)], cy))
                r.corr = Eigen::Vector3f(static_cast<float>(cp), static_cast<float>(ch), static_cast<float>(cy));
        }
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

/// ── REPLAYING A DRIVE RECORDED BEFORE THE REPLAY EXISTED ────────────────────────────────────────
/// A pre-2026-09-02 pair CSV has no `p_robot`. It does not need one for a CAMERA-mount leg: that
/// injection acts entirely in camera coordinates, `pc' = R_inj·pc`, and `pc` is recoverable from the
/// row — `(u_lidar, v_lidar)` is its bearing and `range_m` its magnitude, both logged. So the
/// question "can this estimator recover a camera misalignment" is answerable on data already on disk,
/// without driving. What such a file CANNOT support is stated and enforced below, not glossed:
///
///   ✗ the LiDAR leg          — needs `p_robot` and the LiDAR origin; the rotation centre is the
///                              whole content of that leg, so it refuses rather than approximating.
///   ✗ the closure channel    — only the driving camera ever wrote rows; there is no second camera.
///   ⚠ the covariance         — only its diagonal was logged. Baseline and injected legs are weighted
///                              the SAME way, so a recovery fraction is affected only at second order,
///                              but the absolute sigmas are not the live ones.
///   ⚠ the pitch axis         — the panorama's azimuth zero and handedness are not in the file. Yaw
///                              and height are rotations/translations about the VERTICAL and are
///                              unaffected by that; pitch turns about x̂_cam, whose direction depends
///                              on the azimuth convention. Read the pitch leg as indicative.
///
/// ★ The reconstruction is CHECKED, not asserted: `r` is recomputed from the reconstructed `pc` and
///   compared against the `ru, rv` the agent wrote. If the inversion were wrong the two would differ.
bool load_legacy(const std::string& path, float width, float height, Camera& cam, std::string& err)
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
        if (it == col.end()) { err = path + ": no '" + std::string(k) + "' column"; return false; }
        idx = it->second; return true;
    };
    int c_ts=0,c_vx=0,c_ui=0,c_vi=0,c_ul=0,c_vl=0,c_ru=0,c_rv=0,c_su=0,c_sv=0,c_ap=0,c_rg=0;
    if (not (need("ts_ms",c_ts) and need("vertex",c_vx) and need("u_img",c_ui) and need("v_img",c_vi)
             and need("u_lidar",c_ul) and need("v_lidar",c_vl) and need("ru",c_ru) and need("rv",c_rv)
             and need("sigu",c_su) and need("sigv",c_sv) and need("assoc_prob",c_ap)
             and need("range_m",c_rg)))
        return false;

    cam.ctx.robot = "legacy";
    cam.ctx.camera = cam.name;
    cam.ctx.cam.kind = rc::CameraModel::Kind::Equirect;
    cam.ctx.cam.width = width; cam.ctx.cam.height = height;
    cam.ctx.cam.cx = 0.5f * width; cam.ctx.cam.cy = 0.5f * height;
    cam.ctx.cam.azimuth_sign = 1.f; cam.ctx.cam.azimuth_offset = 0.f;
    cam.ctx.cam.valid = true;
    cam.ctx.cam_R_robot = Eigen::Matrix3f::Identity();   // the replay works in CAMERA coordinates
    cam.ctx.cam_t_robot = Eigen::Vector3f::Zero();
    cam.ctx.sigma_pitch = 0.0035f; cam.ctx.sigma_height = 0.010f; cam.ctx.sigma_yaw = 0.0035f;
    cam.ctx.lidar_known = false;                         // ⇒ any LiDAR leg refuses
    cam.has_cov_lidar = false;

    double worst = 0.0, sum = 0.0; long checked = 0, bad = 0;
    while (std::getline(f, line))
    {
        if (line.empty()) continue;
        split(line, tok);
        if (static_cast<int>(tok.size()) <= std::max({c_ts,c_vx,c_ui,c_vi,c_ul,c_vl,c_ru,c_rv,c_su,c_sv,c_ap,c_rg}))
        { ++bad; continue; }
        double ts=0,vx=0,ui=0,vi=0,ul=0,vl=0,ru=0,rv=0,su=0,sv=0,ap=0,rg=0;
        if (not (to_num(tok[(size_t)c_ts],ts) and to_num(tok[(size_t)c_vx],vx) and to_num(tok[(size_t)c_ui],ui)
                 and to_num(tok[(size_t)c_vi],vi) and to_num(tok[(size_t)c_ul],ul) and to_num(tok[(size_t)c_vl],vl)
                 and to_num(tok[(size_t)c_ru],ru) and to_num(tok[(size_t)c_rv],rv) and to_num(tok[(size_t)c_su],su)
                 and to_num(tok[(size_t)c_sv],sv) and to_num(tok[(size_t)c_ap],ap) and to_num(tok[(size_t)c_rg],rg)))
        { ++bad; continue; }
        if (not (rg > 1e-3)) { ++bad; continue; }
        // Invert the equirect projection: v carries the elevation, u the azimuth, range_m the norm.
        const double el = (vl / height - 0.5) * M_PI;      // asin(-z/r)
        const double z  = -rg * std::sin(el);
        const double rho = std::sqrt(std::max(0.0, rg * rg - z * z));
        const double az = (ul / width - 0.5) * 2.0 * M_PI; // atan2(x, y)
        Row r;
        r.ts = static_cast<std::int64_t>(ts);
        r.vertex = static_cast<int>(vx);
        r.ceiling = false;                                  // not logged; only the closure uses it
        r.uv_image = Eigen::Vector2f(static_cast<float>(ui), static_cast<float>(vi));
        r.cov = Eigen::Matrix2f::Zero();
        r.cov(0,0) = static_cast<float>(su * su);
        r.cov(1,1) = static_cast<float>(sv * sv);
        r.assoc_prob = static_cast<float>(ap);
        r.p_robot = Eigen::Vector3f(static_cast<float>(rho * std::sin(az)),
                                    static_cast<float>(rho * std::cos(az)),
                                    static_cast<float>(z));
        // ★ The check: does the reconstructed point reproduce the residual the agent recorded?
        const rc::mount::PairObs o = rc::mount::make_pair_from(
            r.vertex, r.assoc_prob, r.p_robot, r.uv_image, r.cov, cam.ctx.cam,
            cam.ctx.cam_R_robot, cam.ctx.cam_t_robot,
            cam.ctx.sigma_pitch, cam.ctx.sigma_height, cam.ctx.sigma_yaw);
        if (o.ok)
        {
            const double d = std::max(std::abs(o.r.x() - ru), std::abs(o.r.y() - rv));
            worst = std::max(worst, d); sum += d; ++checked;
        }
        cam.rows.push_back(r);
    }
    if (cam.rows.empty()) { err = path + ": no usable rows"; return false; }
    std::printf("  %-8s %8zu rows reconstructed from (u_lidar, v_lidar, range_m)\n",
                cam.name.c_str(), cam.rows.size());
    std::printf("           residual round-trip vs the agent's own ru/rv: mean %.4f px, worst %.4f px"
                " over %ld rows%s\n", (checked > 0) ? sum / static_cast<double>(checked) : 0.0, worst,
                checked, (worst < 0.05) ? "   ✓ the inversion is faithful"
                                        : "   ✗ CHECK THIS before reading anything below");
    if (bad > 0) std::printf("           %ld unparsable rows skipped\n", bad);
    return true;
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
    /// The per-vertex partials, kept so a CLUSTER BOOTSTRAP can rebuild a solve from a resampled set
    /// of corners without re-reading a single row: H, b and rTr are sums over these blocks, so a
    /// replicate is an addition rather than a re-accumulation.
    std::map<int, rc::mount::VertexBlock> per_vertex;
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
    if (not c.mount_apply.isZero())   // factor B: yaw then pitch, the ingestor's own order
    {
        inject_camera(Axis::Yaw,    c.mount_apply.z(), R, t);
        inject_camera(Axis::Pitch,  c.mount_apply.x(), R, t);
        inject_camera(Axis::Height, c.mount_apply.y(), R, t);
    }
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
        // The two subsetting filters, before any work: a corner left out for the jackknife, and a
        // time window for the between-window scatter. Both default to off.
        if (r.vertex == g_skip_vertex) continue;
        if (g_win_hi > 0 and (r.ts < g_win_lo or r.ts >= g_win_hi)) continue;
        // ★ THE MOUNT THIS ROW WAS ACTUALLY MEASURED AGAINST. The sidecar's extrinsic already holds
        //   the correction in force when the file opened; a row written later carries its own. The
        //   difference is what has to be re-applied here, or every row after the first correction is
        //   rebuilt against a mount that never took it.
        Eigen::Matrix3f Rr = R; Eigen::Vector3f tr = t;
        if (const Eigen::Vector3f d = r.corr - c.ctx.applied; not d.isZero())
        {
            // ⚠ YAW FIRST, THEN PITCH. Each inject LEFT-multiplies, and the ingestor composes
            //   R = R_x(pitch)·R_z(yaw)·R_base, so yaw has to go on first to end up on the inside.
            //   Rotations do not commute and the discrepancy is second order in pitch x yaw — which
            //   is why it hid on the ricoh (pitch 0.006 deg) and showed on the zed (pitch 0.20 deg,
            //   yaw 0.28 deg): H agreed to 3.6e-4 on one camera and 3.1e-2 on the other.
            inject_camera(Axis::Yaw,    d.z(), Rr, tr);
            inject_camera(Axis::Pitch,  d.x(), Rr, tr);
            inject_camera(Axis::Height, d.y(), Rr, tr);
        }
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
                r.vertex, r.assoc_prob, p, r.uv_image, r.cov, c.ctx.cam, Rr, tr,
                c.ctx.sigma_pitch, c.ctx.sigma_height, c.ctx.sigma_yaw);
            Eigen::Matrix2f cv;
            if (o0.ok and o1.ok and rebuilt_cov(r.cov, r.cov_lidar, o0.P, c.ctx.cam_R_robot,
                                                o1.P, R, cv))
                cov = cv;
            else
                ++out.cov_held;
        }
        rc::mount::PairObs o =
            rc::mount::make_pair_from(r.vertex, r.assoc_prob, p, r.uv_image, cov, c.ctx.cam,
                                      Rr, tr, c.ctx.sigma_pitch, c.ctx.sigma_height, c.ctx.sigma_yaw);
        if (not o.ok) continue;
        if (g_probe >= 0)
        {
            // The SAME construction the three live columns use (mount_lidar_pair.h): a rotation is
            // P·(axis × pc) and a translation is P·axis, each scaled by the candidate's prior sigma
            // so the solve stays in prior-sigma units and Accum's unit prior still applies to it.
            const Eigen::Vector3f pc = Rr * p + tr;
            const Eigen::Vector3f x_cam(1.f, 0.f, 0.f), y_cam(0.f, 1.f, 0.f);
            Eigen::Vector2f col = Eigen::Vector2f::Zero();
            if      (g_probe == 0) col = o.P * y_cam.cross(pc);
            else if (g_probe == 1) col = o.P * x_cam;
            else if (g_probe == 2) col = o.P * y_cam;
            else if (g_probe == 3)
            {
                // Deterministic in the row — a failing probe has to be reproducible — and built
                // from no geometry, so it can carry nothing a real parameter would carry.
                const std::uint64_t h = static_cast<std::uint64_t>(r.ts) * 2654435761ull
                                      + static_cast<std::uint64_t>(r.vertex) * 40503ull;
                col = Eigen::Vector2f(static_cast<float>((h & 1023ull) / 511.5 - 1.0),
                                      static_cast<float>(((h >> 10) & 1023ull) / 511.5 - 1.0));
            }
            o.J.col(3) = g_probe_sigma * col;
        }
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
    out.H = acc.H; out.b = acc.b; out.rTr = acc.rTr; out.per_vertex = acc.per_vertex;
    return out;
}

/// The closure, recomputed from the two cameras' REBUILT rows by loop_closure_observe's own rule:
/// same corner, different camera, within 60 ms, differenced in radians. Recomputing it (rather than
/// reading etc/camera_loop.csv) is what makes it respond to an injection at all.
struct Closure { double du_deg = 0, dv_deg = 0, sd_du_deg = 0, sd_dv_deg = 0; long n = 0; };
/// Per-CORNER sums of the differenced sightings. The corner is the resampling unit for the bootstrap
/// below, for the same reason it is the unit everywhere else here: sightings of one corner are not
/// independent draws.
struct VertexClosure { double du_sum = 0, dv_sum = 0; long n = 0; };

Closure closure_of(const CamResult& a, const CamResult& b, std::int64_t max_dt_ms,
                   std::map<int, VertexClosure>* by_vertex = nullptr)
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
            if (by_vertex != nullptr)
            {
                VertexClosure& vc = (*by_vertex)[key / 2];   // fold floor and ceiling into the corner
                vc.du_sum += ddu; vc.dv_sum += ddv; ++vc.n;
            }
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

/// ── THE CLOSED LOOP, SIMULATED ──────────────────────────────────────────────────────────────────
/// A feedback path into a perception input has to be shown not to run away before it is switched on
/// in a robot. This drives the real loop: rows are MEASURED against the mount as it currently is,
/// the pooled solve is applied through Accum::apply_correction, and the next batch is measured
/// against the mount that results. Two properties are checked, and they are the two ways such a
/// loop fails.
///   1. an INFORMED axis converges to the truth and then stays there (no ratchet on repetition)
///   2. an UNINFORMED axis decays back to the graph value instead of wandering
int run_loop()
{
    std::printf("\nclosed-loop selftest — the feedback path, driven for 12 cycles\n\n");
    int failures = 0;
    const Eigen::Vector3f lidar(0.f, 0.f, 1.075f);
    const double truth_deg = 1.0;                       // the mount really is 1 degree out in yaw

    Camera base = make_camera("zed", Eigen::Vector3f(0.18f, 0.09f, 0.945f), lidar, 23, 20, false);
    const Eigen::Matrix3f R_nominal = base.ctx.cam_R_robot;
    const Eigen::Vector3f t_nominal = base.ctx.cam_t_robot;

    rc::mount::Accum acc;
    acc.offset_sigma_px = 5.3;
    Eigen::Vector3f corr = Eigen::Vector3f::Zero();     // pitch, height, yaw — what the agent applied
    double last_total = 0.0, max_step_after_settle = 0.0;
    for (int cycle = 0; cycle < 12; ++cycle)
    {
        // The mount as it now is: nominal, plus what the loop has applied so far.
        Eigen::Matrix3f R = R_nominal; Eigen::Vector3f t = t_nominal;
        inject_camera(Axis::Yaw, corr.z(), R, t);
        // The rows the robot would see: the TRUE mount produced the image, the current mount predicts it.
        Eigen::Matrix3f R_true = R_nominal; Eigen::Vector3f t_true = t_nominal;
        inject_camera(Axis::Yaw, static_cast<float>(truth_deg / kRad2Deg), R_true, t_true);
        for (const Row& r : base.rows)
        {
            const rc::mount::PairObs truth = rc::mount::make_pair_from(
                r.vertex, r.assoc_prob, r.p_robot, r.uv_image, r.cov, base.ctx.cam, R_true, t_true,
                base.ctx.sigma_pitch, base.ctx.sigma_height, base.ctx.sigma_yaw);
            if (not truth.ok) continue;
            // uv_image as the true mount would have placed it, predicted by the mount we have.
            const rc::mount::PairObs o = rc::mount::make_pair_from(
                r.vertex, r.assoc_prob, r.p_robot, truth.uv_lidar, r.cov, base.ctx.cam, R, t,
                base.ctx.sigma_pitch, base.ctx.sigma_height, base.ctx.sigma_yaw);
            if (o.ok) acc.add(o);
        }
        const auto sol = acc.solve();
        if (not sol.ok) { std::printf("  cycle %2d: solve failed\n", cycle); ++failures; break; }
        Eigen::Vector4d dp = -sol.p; dp(3) = 0.0;      // see Accum::applied on the sign
        acc.apply_correction(dp);
        corr.z() += static_cast<float>(dp(2)) * base.ctx.sigma_yaw;
        const double total = corr.z() * kRad2Deg;
        if (cycle >= 4) max_step_after_settle = std::max(max_step_after_settle, std::abs(total - last_total));
        last_total = total;
        if (cycle < 3 or cycle == 11)
            std::printf("  cycle %2d  applied total %+7.4f deg   remaining estimate %+7.4f deg\n",
                        cycle, total, -dp(2) * base.ctx.sigma_yaw * kRad2Deg);
    }
    const double err = std::abs(last_total - truth_deg);
    std::printf("  %-58s %s   total %+.4f deg vs a truth of %+.4f\n",
                "informed axis converges to the truth", (err < 0.06) ? "PASS" : "FAIL", last_total, truth_deg);
    if (err >= 0.06) ++failures;
    std::printf("  %-58s %s   largest late step %.5f deg\n",
                "and then STAYS — no ratchet on repetition",
                (max_step_after_settle < 0.02) ? "PASS" : "FAIL", max_step_after_settle);
    if (max_step_after_settle >= 0.02) ++failures;

    // An axis with no information at all: height, on rows whose geometry cannot see it, must decay.
    rc::mount::Accum idle;
    idle.offset_sigma_px = 5.3;
    idle.applied = Eigen::Vector4d(0.0, 3.0, 0.0, 0.0);   // as if 3 prior-sigmas had been applied
    double h = 3.0;
    for (int cycle = 0; cycle < 12; ++cycle)
    {
        const auto sol = idle.solve(0);
        if (not sol.ok) break;
        Eigen::Vector4d dp = -sol.p; dp(3) = 0.0;
        idle.apply_correction(dp);
        h = idle.applied(1);
    }
    std::printf("  %-58s %s   3.000 -> %.4f prior sigmas\n",
                "uninformed axis DECAYS to the graph value", (std::abs(h) < 0.05) ? "PASS" : "FAIL", h);
    if (std::abs(h) >= 0.05) ++failures;
    return failures;
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

    failures += run_loop();
    std::printf("\n%s\n", failures == 0 ? "ALL PASS" : "FAILURES");
    return failures == 0 ? 0 : 1;
}
} // namespace selftest

/// ── THE DISAGREEMENT BETWEEN THE TWO ROUTES, WITH THEIR CORRELATION IN IT ───────────────────────
/// The mount solves and the closure are NOT independent estimates of the camera-vs-camera yaw: both
/// are computed from the same corner sightings, and the closure is literally a difference of the same
/// per-camera residuals that drive the solves. Combining their sigmas in quadrature therefore states
/// an interval for D = (yaw_a − yaw_b) − closure that assumes a covariance of zero, and there is no
/// reason for it to be zero.
///
/// ★ A CLUSTER BOOTSTRAP settles it without needing the covariance in closed form: resample CORNERS
///   with replacement, recompute BOTH routes on the same resampled corners, and take the spread of
///   their difference. The correlation is carried along by construction. The corner is the unit for
///   the reason it is the unit everywhere else here — sightings of one corner are not independent.
/// ★ It is exact and it is cheap, because H, b and rTr are sums over the per-vertex blocks: a
///   replicate adds up the selected blocks instead of re-reading 100 000 rows.
void run_bootstrap(std::vector<Camera>& cams, const std::vector<CamResult>& base,
                   double sigma_px, std::int64_t closure_ms, int reps)
{
    if (cams.size() < 2) { std::printf("  (the bootstrap compares two routes and needs two cameras)\n"); return; }
    std::map<int, VertexClosure> by_vertex;
    const Closure full = closure_of(base[0], base[1], closure_ms, &by_vertex);

    std::vector<int> verts;                     // the resampling universe: every corner either saw
    for (const auto& [v, blk] : base[0].per_vertex) verts.push_back(v);
    for (const auto& [v, blk] : base[1].per_vertex)
        if (std::ranges::find(verts, v) == verts.end()) verts.push_back(v);
    if (verts.size() < 3) { std::printf("  (too few corners to resample)\n"); return; }

    const auto solve_subset = [&](const CamResult& r, const std::vector<int>& pick) {
        rc::mount::Accum a;
        a.offset_sigma_px = sigma_px;
        for (const int v : pick)
        {
            const auto it = r.per_vertex.find(v);
            if (it == r.per_vertex.end()) continue;
            rc::mount::VertexBlock& d = a.per_vertex[v];
            // A corner drawn twice contributes twice — that IS the resampling.
            d.A += it->second.A; d.c += it->second.c; d.D += it->second.D;
            d.b += it->second.b; d.e += it->second.e; d.rTr += it->second.rTr; d.n += it->second.n;
            a.n += it->second.n;
        }
        return a.solve();
    };

    // ── The closure's cluster statistics, computed directly ─────────────────────────────────────
    // ⚠ Reported because a standard error is exactly the kind of number that can be produced two
    //   ways that disagree. sd(per sighting)/sqrt(k) is NOT a cluster standard error: it puts the
    //   WITHIN-corner spread where the BETWEEN-corner spread belongs, and here that is a factor of
    //   four. The estimator below is the mean of per-corner means, weighted by sightings, with its
    //   spread taken over the corners themselves.
    {
        double wsum = 0, num = 0;
        std::vector<std::pair<double, double>> cm;   // (corner mean, weight)
        for (const auto& [v, vc] : by_vertex)
            if (vc.n > 0)
            {
                const double m = vc.du_sum / static_cast<double>(vc.n) * kRad2Deg;
                cm.emplace_back(m, static_cast<double>(vc.n));
                num += vc.du_sum * kRad2Deg; wsum += static_cast<double>(vc.n);
            }
        const double gm = num / wsum;
        double var_un = 0, var_w = 0, wtot = 0;
        for (const auto& [m, w] : cm) { var_un += (m - gm) * (m - gm); var_w += w * (m - gm) * (m - gm); wtot += w; }
        const double k = static_cast<double>(cm.size());
        const double sd_corner = std::sqrt(var_un / (k - 1.0));
        std::printf("  closure by CORNER: %.0f corners, mean %+.4f deg, sd of corner means %.4f,"
                    " se = sd/sqrt(k) %.4f\n", k, gm, sd_corner, sd_corner / std::sqrt(k));
        std::printf("                     (sightings-weighted sd %.4f; per-SIGHTING sd %.4f, whose"
                    " /sqrt(k) is %.4f and is NOT a cluster se)\n",
                    std::sqrt(var_w / wtot), full.sd_du_deg, full.sd_du_deg / std::sqrt(k));
    }

    std::uint32_t rng = 20260904u;
    const auto rnd = [&] { rng = rng * 1664525u + 1013904223u; return rng >> 8; };
    std::vector<double> dm, dc, dd;             // mount difference, closure, and their difference
    // ── AND THE COMMON MODE, which is the quantity a LiDAR mount parameter would estimate ────────
    // ★★★ The closure is the DIFFERENCE of the two cameras and is therefore blind to anything they
    //     share; the MEAN is the other half of the same basis, and a yaw common to both cameras is
    //     not a camera fault at all — it is where a LiDAR yaw error appears. That matters the moment
    //     the measured mount is published into the shared body->camera edge (ImageEdge.mountPublish):
    //     publishing charges the WHOLE camera-vs-LiDAR disagreement to the camera, so a non-zero
    //     common mode is a LiDAR error being written into two camera mounts, where it will look
    //     self-consistent and be externally wrong.
    // ⚠ It is only an INDICATION of a LiDAR error, never a measurement of one: a rotation common to
    //   all three devices is unobservable by construction, and the parallax that separates a LiDAR
    //   rotation from a matched pair of camera rotations is 3.2% of the signal (arm 7's own
    //   falsifier band). So read it as "is there something the cameras share?", and note that a
    //   common mode consistent with zero is exactly the licence the publish needs.
    std::vector<double> cmn;
    dm.reserve(reps); dc.reserve(reps); dd.reserve(reps); cmn.reserve(reps);
    for (int r = 0; r < reps; ++r)
    {
        std::vector<int> pick;
        pick.reserve(verts.size());
        for (std::size_t k = 0; k < verts.size(); ++k) pick.push_back(verts[rnd() % verts.size()]);
        const auto sa = solve_subset(base[0], pick), sb = solve_subset(base[1], pick);
        if (not sa.ok or not sb.ok) continue;
        double du = 0; long n = 0;
        for (const int v : pick)
            if (const auto it = by_vertex.find(v); it != by_vertex.end())
            { du += it->second.du_sum; n += it->second.n; }
        if (n == 0) continue;
        const double ya = param_deg(sa, 2, cams[0].ctx), yb = param_deg(sb, 2, cams[1].ctx);
        const double mount = ya - yb;
        const double clo   = du / static_cast<double>(n) * kRad2Deg;
        dm.push_back(mount); dc.push_back(clo); dd.push_back(mount - clo);
        cmn.push_back(0.5 * (ya + yb));
    }
    const auto stat = [](const std::vector<double>& v) {
        double m = 0; for (double x : v) m += x; m /= static_cast<double>(v.size());
        double s = 0; for (double x : v) s += (x - m) * (x - m);
        return std::pair{m, std::sqrt(s / static_cast<double>(v.size() - 1))};
    };
    if (dd.size() < 10) { std::printf("  (bootstrap produced too few usable replicates)\n"); return; }
    const auto [mm, sm] = stat(dm);
    const auto [mc, sc] = stat(dc);
    const auto [md, sd] = stat(dd);
    double cov = 0;
    for (std::size_t i = 0; i < dd.size(); ++i) cov += (dm[i] - mm) * (dc[i] - mc);
    cov /= static_cast<double>(dd.size() - 1);
    const double rho = cov / (sm * sc);

    std::printf("\n── the two routes to the camera-vs-camera yaw, %zu corners, %zu bootstrap replicates ──\n",
                verts.size(), dd.size());
    std::printf("  mount solves differ   %+8.4f deg   bootstrap sd %.4f\n", mm, sm);
    std::printf("  closure says          %+8.4f deg   bootstrap sd %.4f   (%ld differenced sightings)\n",
                mc, sc, full.n);
    std::printf("  correlation between the two routes  rho = %+.3f\n", rho);
    std::printf("  they differ by        %+8.4f deg   bootstrap sd %.4f   = %.2f sigma\n",
                md, sd, std::abs(md) / sd);
    std::printf("  ★ quadrature, which assumes rho = 0, would have said sd %.4f (%.2f sigma) — %s\n",
                std::hypot(sm, sc), std::abs(md) / std::hypot(sm, sc),
                (sd < std::hypot(sm, sc)) ? "so independence UNDERSTATES the disagreement"
                                          : "so independence OVERSTATES the disagreement");
    // ── The common mode: what the two cameras SHARE, i.e. where a LiDAR yaw would sit ────────────
    {
        const auto [mk, sk] = stat(cmn);
        std::printf("\n  common mode (mean of the two camera yaws) %+.4f +/- %.4f deg = %.2f sigma"
                    " from zero\n", mk, sk, sk > 0 ? std::abs(mk / sk) : 0.0);
        std::printf("    the closure differences the cameras and cannot see this; a yaw they SHARE is"
                    " not a camera\n    fault but where a LiDAR yaw error appears. Publishing a mount"
                    " charges the whole\n    camera-vs-LiDAR disagreement to the CAMERA, so this is"
                    " the number that licenses it.\n");
        std::printf("    ⚠ an INDICATION, not a measurement: a rotation common to all three devices is"
                    " unobservable,\n      and parallax separates a LiDAR rotation from a matched"
                    " camera pair by only 3.2%% of the signal.\n");
    }
}

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
        "  --apply CAM:AXIS=VALUE     evaluate from a mount already corrected by this much —\n"
        "                     factor B of the 2x2. Same rows, different prediction.\n"
        "  --bootstrap N      resample CORNERS with replacement N times and report the mount-solve\n"
        "                     difference, the closure, their correlation, and the spread of their\n"
        "                     difference — which combining sigmas in quadrature cannot give.\n"
        "  --fixed-cov        do NOT rebuild the covariance under the injected mount (the old\n"
        "                     approximation). Run it both ways: the difference is the size of what\n"
        "                     §2b was going to assume away.\n"
        "  --probe            ask whether a FOURTH mount parameter is needed: roll, a lateral\n"
        "                     translation, a depth translation, and a null control, each in\n"
        "                     turn in the dead dt column, plus a leave-one-CORNER-out on the\n"
        "                     strongest. The pooled sigma and the jackknife disagree by ~67x;\n"
        "                     the jackknife is the one that answers the question.\n"
        "  --scatter-sweep    run the window-length ladder and fit sigma_window ~ T^-alpha per\n"
        "                     axis. alpha = 1/2 is the independent-rows null, and ONLY there\n"
        "                     is either ratio a property of the estimator rather than of the\n"
        "                     cadence. Run this before quoting a ratio.\n"
        "  --scatter[-ms N]   the M4 honesty check: solve N ms windows (default 5000, the live\n"
        "                     cadence) and compare their\n"
        "                     scatter against the formal sigma, BOTH ways round — against a\n"
        "                     window's own sigma and against the pooled one. They differ by\n"
        "                     sqrt(k) and answer different questions.\n"
        "  --verify FILE      check the delta=0 solve against a live evidence file.\n"
        "  --selftest         replay a drive whose truth is set in the tool, and check that the\n"
        "                     three injection signatures come out distinguishable. Run this before\n"
        "                     trusting a leg on real data.\n"
        "  --help\n");
}
} // namespace

int main(int argc, char** argv)
{
    std::vector<std::string> pair_files, legacy_files;
    float legacy_w = 1920.f, legacy_h = 960.f;
    std::vector<Leg> legs;
    std::vector<std::tuple<std::string, std::string, double>> applies;
    double sigma_px = 0.0;
    bool fixed_cov = false;
    bool do_probe = false;
    bool do_scatter = false;
    std::int64_t scatter_ms = 5000;
    bool do_sweep = false;
    int reps = 0;
    std::int64_t closure_ms = 60;
    std::string verify_file;

    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        const auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
        if (a == "--help") { usage(); return 0; }
        else if (a == "--selftest") return selftest::run();
        else if (a == "--pair") pair_files.push_back(next());
        else if (a == "--legacy-pair") legacy_files.push_back(next());
        else if (a == "--legacy-equirect")
        { double w = 0, h = 0; if (to_num(next(), w)) legacy_w = (float)w; if (to_num(next(), h)) legacy_h = (float)h; }
        else if (a == "--sigma-px") { double v = 0; if (to_num(next(), v)) sigma_px = v; }
        else if (a == "--closure-ms") { double v = 0; if (to_num(next(), v)) closure_ms = static_cast<std::int64_t>(v); }
        else if (a == "--verify") verify_file = next();
        else if (a == "--fixed-cov") fixed_cov = true;
        else if (a == "--probe") do_probe = true;
        else if (a == "--scatter") do_scatter = true;
        else if (a == "--scatter-sweep") do_sweep = true;
        else if (a == "--scatter-ms")
        { double v = 0; if (to_num(next(), v) and v > 0) { do_scatter = true; scatter_ms = static_cast<std::int64_t>(v); } }
        else if (a == "--bootstrap") { double v = 0; if (to_num(next(), v)) reps = static_cast<int>(v); }
        else if (a == "--apply")
        {
            // --apply CAM:AXIS=VALUE, repeatable. Same spelling as --inject, different meaning:
            // --inject asks what a WRONG mount would do, --apply says which mount to start from.
            const std::string spec = next();
            const auto colon = spec.find(':'), eq = spec.find('=');
            if (colon == std::string::npos or eq == std::string::npos or eq < colon)
            { std::printf("bad --apply '%s' (want CAM:AXIS=VALUE)\n", spec.c_str()); return 2; }
            double v = 0;
            if (not to_num(std::string_view(spec).substr(eq + 1), v))
            { std::printf("bad --apply value in '%s'\n", spec.c_str()); return 2; }
            applies.emplace_back(spec.substr(0, colon), spec.substr(colon + 1, eq - colon - 1), v);
        }
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
    if (pair_files.empty() and legacy_files.empty()) { usage(); return 2; }

    // ── Load ────────────────────────────────────────────────────────────────────────────────────
    std::vector<Camera> cams;
    for (const std::string& p : legacy_files)
    {
        Camera c;
        const auto slash = p.find_last_of('/');
        const std::string base = (slash == std::string::npos) ? p : p.substr(slash + 1);
        const std::string tag = "image_edge_pair_";
        c.name = base.starts_with(tag) ? base.substr(tag.size(), base.size() - tag.size() - 4) : "legacy";
        std::printf("  ⚠ LEGACY MODE on %s — camera-mount legs only: no p_robot (no LiDAR leg), one\n"
                    "    camera (no closure), covariance diagonal only, pitch axis convention-dependent.\n",
                    p.c_str());
        std::string err;
        if (not load_legacy(p, legacy_w, legacy_h, c, err)) { std::printf("✗ %s\n", err.c_str()); return 1; }
        cams.push_back(std::move(c));
    }
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

    for (const auto& [cam, ax, v] : applies)
    {
        bool hit = false;
        for (Camera& c : cams)
        {
            if (c.name != cam) continue;
            hit = true;
            if      (ax == "pitch")  c.mount_apply.x() = static_cast<float>(v / kRad2Deg);
            else if (ax == "yaw")    c.mount_apply.z() = static_cast<float>(v / kRad2Deg);
            else if (ax == "height") c.mount_apply.y() = static_cast<float>(v);
            else { std::printf("bad --apply axis '%s'\n", ax.c_str()); return 2; }
        }
        if (not hit) { std::printf("--apply names camera '%s', which was not loaded\n", cam.c_str()); return 2; }
    }
    for (const Camera& c : cams)
        if (not c.mount_apply.isZero())
            std::printf("  %-8s evaluated under a mount corrected by pitch %+.4f deg, height %+.4f m,"
                        " yaw %+.4f deg\n", c.name.c_str(), c.mount_apply.x() * kRad2Deg,
                        c.mount_apply.y(), c.mount_apply.z() * kRad2Deg);

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

    // ── IS A FOURTH MOUNT PARAMETER NEEDED? ─────────────────────────────────────────────────────
    // ★★★ THE POOLED SIGMA AND THE JACKKNIFE GIVE OPPOSITE ANSWERS, AND THE JACKKNIFE IS RIGHT.
    //     Measured on arm7_0903_1104: a depth translation comes out 62.6 sigma from zero on the
    //     ricoh by its own posterior, and 1.01 sigma once corners rather than sightings are the
    //     sample. The per-vertex nuisance does NOT protect against this: it integrates out a
    //     per-corner CONSTANT offset, and a candidate whose covariate varies WITHIN a corner (range)
    //     is untouched by it. So the marginalisation makes the three FITTED parameters honest and
    //     says nothing about a NEW column — a model-selection question needs the leave-one-out.
    if (do_probe)
    {
        std::printf("\n── parameter probe: the live 3 DOF + ONE candidate in the dead dt column ──\n"
                    "   a candidate the data needs must come out many sigmas from zero, drop"
                    " chi2/dof,\n   and SURVIVE the jackknife — while the NULL control does none of"
                    " it.\n");
        for (size_t ci = 0; ci < cams.size(); ++ci)
        {
            const Camera& c = cams[ci];
            std::printf("  %s: baseline chi2/dof %.4f over %d corners\n",
                        c.name.c_str(), base[ci].sol.chi2_dof, base[ci].sol.clusters);
            for (int pi = 0; pi < 4; ++pi)
            {
                g_probe = pi;
                // A rotation is asked on pitch/yaw's own prior and a translation on height's, so the
                // candidate is asked on the same terms as the parameter it would sit beside.
                g_probe_sigma = (pi == 0) ? c.ctx.sigma_pitch : (pi == 3) ? 1.f : c.ctx.sigma_height;
                bool refused = false; std::string why;
                const CamResult r = solve_leg(c, base_leg, sigma_px, fixed_cov, refused, why);
                const double sc  = (pi == 0) ? g_probe_sigma * kRad2Deg : (pi == 3) ? 1.0 : g_probe_sigma;
                const double val = -r.sol.p(3) * sc, sig = r.sol.sigma(3) * sc;
                std::printf("    %-12s %+9.4f %-3s ± %.4f (%6.2f σ) | chi2/dof %.4f (%+.4f)"
                            " | cond %5.1f | pitch/height/yaw now %+.4f %+.4f %+.4f\n",
                            probe_name(pi), val, probe_unit(pi), sig,
                            sig > 0 ? std::abs(val / sig) : 0.0,
                            r.sol.chi2_dof, r.sol.chi2_dof - base[ci].sol.chi2_dof, r.sol.cond,
                            param_deg(r.sol, 0, c.ctx), param_deg(r.sol, 1, c.ctx),
                            param_deg(r.sol, 2, c.ctx));
            }
            // Leave-one-CORNER-out on the strongest candidate. A parameter carried by one corner is
            // a corner, and no formal sigma computed over sightings can say so.
            g_probe = 2; g_probe_sigma = c.ctx.sigma_height;
            double su = 0, su2 = 0, lo = 1e9, hi = -1e9;
            int k = 0, lo_v = -1, hi_v = -1;
            for (const auto& [v, blk] : base[ci].per_vertex)
            {
                g_skip_vertex = v;
                bool rf = false; std::string wy;
                const CamResult r = solve_leg(c, base_leg, sigma_px, fixed_cov, rf, wy);
                g_skip_vertex = -9999;
                if (not r.sol.ok) continue;
                const double val = -r.sol.p(3) * c.ctx.sigma_height;
                su += val; su2 += val * val; ++k;
                if (val < lo) { lo = val; lo_v = v; }
                if (val > hi) { hi = val; hi_v = v; }
            }
            if (k > 1)
            {
                const double m = su / k;
                const double sd = std::sqrt(std::max(0.0, su2 / k - m * m));
                const double se = sd * std::sqrt(static_cast<double>(k) - 1.0);   // jackknife SE
                std::printf("    t_depth leave-one-corner-out, %d corners: mean %+.4f m,"
                            " jackknife SE %.4f m ⇒ %.2f σ  (v%d → %+.4f, v%d → %+.4f)\n",
                            k, m, se, se > 0 ? std::abs(m / se) : 0.0, lo_v, lo, hi_v, hi);
            }
            g_probe = -1;
        }
    }

    // ── M4: THE BETWEEN-WINDOW SCATTER AGAINST THE FORMAL SIGMA ─────────────────────────────────
    // ★★★ TWO RATIOS EXIST, AND sqrt(k) IS NOT A CONVERSION BETWEEN THEM. The scatter of window
    //     estimates against a WINDOW's own sigma asks whether one window's error bar is honest; the
    //     same scatter over sqrt(k) against the POOLED sigma asks whether the pooled one is. They
    //     would differ by exactly sqrt(k) only if pooling k windows shrank sigma by sqrt(k) — i.e.
    //     only if the windows were independent samples of one static quantity carrying no
    //     information the others lack. Whether that holds is measurable, and it is what the last
    //     column reports: pool_gain = sigma_window / (sqrt(k) * sigma_pooled).
    //
    //     ★★★ THE POOLED RATIO IS ONLY INTERPRETABLE WHERE pool_gain ~ 1.
    //       pool_gain >> 1  pooling uses information NO window has — on this pair that is the range
    //                       diversity across windows breaking the pitch/height ridge, which the live
    //                       log already announces as "POOLING BROKE THE RIDGE". sd/sqrt(k) is then
    //                       not the pooled estimator's spread at all, so a large pooled ratio there
    //                       is the ridge, NOT a dishonest interval.
    //       pool_gain ~ 1   the windows behave like independent draws and the pooled ratio is a
    //                       valid honesty test.
    //       pool_gain << 1  pooling buys nothing, because the axis is limited by the number of
    //                       DISTINCT CORNERS and every window already sees them (the panorama's
    //                       yaw). sigma_pooled ~ sigma_window, so dividing the scatter by sqrt(k)
    //                       drives the ratio far below one for the same invalid reason.
    //     ⇒ THE PER-WINDOW FORM IS THE ROBUST INSTRUMENT: it assumes nothing about how information
    //       combines across windows. Quote the pooled form only beside its pool_gain.
    //
    //     The live [mount/pool] line prints window scatter beside the POOLED sigma, which is neither
    //     of these forms — so a ratio recorded from that line is not comparable with either.
    if (do_scatter or do_sweep)
    {
        // One camera, one window length: the scatter of the window estimates, the mean formal sigma
        // OF a window, and the pooled sigma. Everything the two ratios and pool_gain are built from.
        const auto scatter_once = [&](size_t ci, std::int64_t ms,
                                      double sd[3], double sw[3], double sp[3]) -> int
        {
            const Camera& c = cams[ci];
            if (c.rows.empty()) return 0;
            std::int64_t t0 = c.rows.front().ts, t1 = t0;
            for (const Row& r : c.rows) { t0 = std::min(t0, r.ts); t1 = std::max(t1, r.ts); }
            double su[3] = {0, 0, 0}, su2[3] = {0, 0, 0}, sf[3] = {0, 0, 0};
            int k = 0;
            for (std::int64_t w = t0; w < t1; w += ms)
            {
                g_win_lo = w; g_win_hi = w + ms;
                bool rf = false; std::string wy;
                const CamResult r = solve_leg(c, base_leg, sigma_px, fixed_cov, rf, wy);
                g_win_lo = g_win_hi = 0;
                // A window holding one or two corners cannot separate a mount from those corners'
                // own offsets. It is not a poor estimate of the mount; it is not an estimate of it.
                if (not r.sol.ok or r.sol.clusters < 3) continue;
                for (int i = 0; i < 3; ++i)
                {
                    const double v = param_deg(r.sol, i, c.ctx);
                    su[i] += v; su2[i] += v * v; sf[i] += sigma_deg(r.sol, i, c.ctx);
                }
                ++k;
            }
            if (k < 2) return k;
            for (int i = 0; i < 3; ++i)
            {
                const double m = su[i] / k;
                sd[i] = std::sqrt(std::max(0.0, su2[i] / k - m * m));
                sw[i] = sf[i] / k;
                sp[i] = sigma_deg(base[ci].sol, i, c.ctx);
            }
            return k;
        };
        const char* nm[3] = {"pitch", "height", "yaw"};

        if (do_scatter)
        {
            std::printf("\n── M4 honesty: %.1f s windows (the live cadence is 5 s), ≥3 corners each ──\n",
                        scatter_ms / 1000.0);
            for (size_t ci = 0; ci < cams.size(); ++ci)
            {
                double sd[3], sw[3], sp[3];
                const int k = scatter_once(ci, scatter_ms, sd, sw, sp);
                if (k < 2)
                { std::printf("  %-8s only %d usable windows\n", cams[ci].name.c_str(), k); continue; }
                std::printf("  %-8s %d windows\n", cams[ci].name.c_str(), k);
                const double rk = std::sqrt(static_cast<double>(k));
                for (int i = 0; i < 3; ++i)
                    std::printf("    %-7s window sd %9.5f | per-window σ %9.5f → %6.2f"
                                " | pooled σ %9.5f vs sd/√k %9.5f → %6.2f | pool_gain %6.2f\n",
                                nm[i], sd[i], sw[i], sw[i] > 0 ? sd[i] / sw[i] : 0.0,
                                sp[i], sd[i] / rk, sp[i] > 0 ? (sd[i] / rk) / sp[i] : 0.0,
                                sp[i] > 0 ? sw[i] / (rk * sp[i]) : 0.0);
            }
        }

        // ── THE ASSUMPTION TEST BEHIND BOTH RATIOS: how a window's own sigma scales with duration ──
        // ★★★ NEITHER RATIO HAS A WINDOW-LENGTH-FREE VALUE, and the exponent says why.
        //     Write sigma_window ~ T^-alpha. If a window's rows were independent information,
        //     doubling T would halve the variance: alpha = 1/2. Then, and ONLY then,
        //     sigma_window/sqrt(k) is invariant (k ~ 1/T), so pool_gain is invariant and the pooled
        //     ratio is a property of the estimator rather than of the cadence.
        //     Measured: pool_gain ∝ T^(1/2 - alpha), which reproduces every axis of this tour to a
        //     few per cent — so a pool_gain that "drifts with window length" is ARITHMETIC, and a
        //     drift test cannot classify an axis. The exponent can.
        //     alpha ~ 1/2   a window's rows carry independent information (ricoh pitch, 0.484)
        //     alpha < 1/2   information SATURATES: a longer window adds sightings, not distinct
        //                   corners. Both ratios then grow with T and neither has a fixed value.
        //     alpha ~ 0     the axis is pinned at its prior and duration buys nothing (ricoh yaw).
        // ★★★ AND THE per-window RATIO GROWING WITH T IS ITSELF A RESULT: sigma_window falls while
        //     the SCATTER of the window estimates does not, so the disagreement between windows is
        //     REAL and not sampling noise. It is a per-window bias field, which no single interval
        //     can represent, and it is what makes the ratio unbounded in T rather than convergent.
        if (do_sweep)
        {
            static const std::int64_t ladder[] = {1500, 2000, 3000, 4000, 5000,
                                                  7000, 10000, 15000, 20000, 30000};
            std::printf("\n── window-length sweep: is either ratio a property of the estimator? ──\n");
            for (size_t ci = 0; ci < cams.size(); ++ci)
            {
                std::printf("  %s\n      T     k |%s\n", cams[ci].name.c_str(),
                            "   pitch  sd/σ  gain |  height  sd/σ  gain |     yaw  sd/σ  gain");
                std::vector<double> lt, ls[3];
                for (std::int64_t ms : ladder)
                {
                    double sd[3], sw[3], sp[3];
                    const int k = scatter_once(ci, ms, sd, sw, sp);
                    if (k < 2) continue;
                    const double rk = std::sqrt(static_cast<double>(k));
                    std::printf("  %6.1f %5d |", ms / 1000.0, k);
                    for (int i = 0; i < 3; ++i)
                        std::printf(" %7.5f %5.2f %5.2f |", sw[i], sw[i] > 0 ? sd[i] / sw[i] : 0.0,
                                    sp[i] > 0 ? sw[i] / (rk * sp[i]) : 0.0);
                    std::printf("\n");
                    lt.push_back(std::log(ms / 1000.0));
                    for (int i = 0; i < 3; ++i) ls[i].push_back(std::log(std::max(1e-12, sw[i])));
                }
                if (lt.size() < 3) { std::printf("      (too few usable window lengths)\n"); continue; }
                const double n = static_cast<double>(lt.size());
                double mx = 0; for (double v : lt) mx += v; mx /= n;
                double sxx = 0; for (double v : lt) sxx += (v - mx) * (v - mx);
                for (int i = 0; i < 3; ++i)
                {
                    double my = 0; for (double v : ls[i]) my += v; my /= n;
                    double sxy = 0;
                    for (size_t q = 0; q < lt.size(); ++q) sxy += (lt[q] - mx) * (ls[i][q] - my);
                    const double a = -sxy / sxx;                       // sigma_window ~ T^-a
                    double sse = 0;
                    for (size_t q = 0; q < lt.size(); ++q)
                    { const double e = ls[i][q] - (my - a * (lt[q] - mx)); sse += e * e; }
                    const double se = std::sqrt((sse / (n - 2)) / sxx);
                    std::printf("      %-7s alpha = %+.3f ± %.3f   %s\n", nm[i], a, se,
                                std::abs(a - 0.5) < 3 * se
                                    ? "consistent with independent rows ⇒ the pooled ratio is"
                                      " well defined here"
                                    : (a < 0.05 ? "pinned at the prior ⇒ duration buys nothing"
                                                : "information SATURATES ⇒ NEITHER ratio has a"
                                                  " window-length-free value"));
                }
            }
        }
    }

    if (reps > 0) run_bootstrap(cams, base, sigma_px, closure_ms, reps);

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
            Eigen::Vector4d applied_live = Eigen::Vector4d::Zero();
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
                else if (tok[0] == "applied" and tok.size() > 4)
                    for (int q = 0; q < 4; ++q) to_num(tok[static_cast<std::size_t>(q + 1)], applied_live(q));
                else if (tok[0] == "H" and tok.size() > 3 and to_num(tok[1], i) and to_num(tok[2], j)
                         and to_num(tok[3], v))
                    H_live(static_cast<int>(i), static_cast<int>(j)) =
                        H_live(static_cast<int>(j), static_cast<int>(i)) = v;
                else if (tok[0] == "b" and tok.size() > 2 and to_num(tok[1], i) and to_num(tok[2], v))
                    b_live(static_cast<int>(i)) = v;
            }
            // ★ A LIVE SNAPSHOT LAGS, AND THAT IS NOT A MISMATCH. The pool is saved on its own
            //   cadence while the CSV keeps appending, so a file copied from a running agent holds
            //   FEWER pairs than the CSV. They are appended together and in order — every added pair
            //   writes its row immediately after — so the saved pool is exactly the CSV's first n
            //   rows, and re-accumulating that prefix is the comparison the check actually wants.
            //   Without this the verify can only ever pass on a stopped agent, which is when it is
            //   least convenient to run it.
            CamResult r0 = base[0];
            if (n_live > 0 and n_live < r0.n)
            {
                Camera trunc = cams[0];
                trunc.rows.resize(static_cast<std::size_t>(n_live));
                bool rf = false; std::string rw;
                r0 = solve_leg(trunc, Leg{}, sigma_px, fixed_cov, rf, rw);
                std::printf("  the snapshot was taken from a RUNNING agent: re-accumulated the CSV's"
                            " first %ld rows to match the saved pool\n", n_live);
            }
            std::printf("  pairs: live %ld, replay %ld  %s\n", n_live, r0.n,
                        (n_live == r0.n) ? "✓ same rows"
                                         : "✗ DIFFERENT — the two are not describing one run");
            // The accumulation itself. A relative comparison, because H's entries span orders of
            // magnitude and an absolute tolerance would be a different test on each one.
            const auto rel = [](double a, double c) {
                const double d = std::max(std::abs(a), std::abs(c));
                return (d > 1e-12) ? std::abs(a - c) / d : 0.0;
            };
            // ★ GRADED AGAINST THE MATRIX SCALE, NOT ENTRY BY ENTRY. H's off-diagonals run four
            //   orders of magnitude below its diagonal, so a per-entry RELATIVE error on a
            //   near-zero cross term is large while being numerically nothing — measured here at
            //   9.1e-3 on an entry worth 0.04% of the diagonal, i.e. 3.6e-6 of the matrix. A test
            //   that cannot pass when the estimator is right is not a test.
            double scale = 1e-300;
            for (int i = 0; i < 4; ++i) scale = std::max(scale, std::abs(H_live(i, i)));
            double worst = 0.0, worst_rel = 0.0; int wi = 0, wj = 0;
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                {
                    const double e = std::abs(H_live(i, j) - r0.H(i, j)) / scale;
                    worst_rel = std::max(worst_rel, rel(H_live(i, j), r0.H(i, j)));
                    if (e > worst) { worst = e; wi = i; wj = j; }
                }
            double bscale = 1e-300;
            for (int i = 0; i < 4; ++i) bscale = std::max(bscale, std::abs(b_live(i)));
            double worst_b = 0.0; int wb = 0;
            for (int i = 0; i < 4; ++i)
                if (const double e = std::abs(b_live(i) - r0.b(i)) / bscale; e > worst_b) { worst_b = e; wb = i; }
            const double worst_r = rel(rTr_live, r0.rTr);
            std::printf("  H: worst %.3e of the matrix scale at (%d,%d)   [worst per-entry relative %.3e]\n",
                        worst, wi, wj, worst_rel);
            std::printf("  b: worst %.3e of its scale at (%d)            rTr %.3e relative\n",
                        worst_b, wb, worst_r);
            // ⚠ b and rTr are RE-REFERENCED by an applied correction while H is not, so a pool that
            //   has fed anything back cannot be graded on them — and saying so beats failing a check
            //   that was never applicable.
            const bool rebased = applied_live.norm() > 0.0;
            const double tol = 1e-4;
            const bool ok = n_live == r0.n and worst < tol and (rebased or (worst_b < tol and worst_r < tol));
            if (rebased)
                std::printf("  (this pool has applied a correction: b and rTr are rebased by it, so only"
                            " H is comparable)\n");
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
