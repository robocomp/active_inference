// Offline replay of door_concept's interaction-anchor frame record (Task 8, fix 3): reads etc/anchor_frames.bin (written
// live by step_passage_anchor, format in src/anchor_frame_record.h), re-runs update_door_anchor on every record with the
// recorded inputs, and prints per frame the recorded vs replayed P(door in view) — p_lidar with its three factors
// (opening coverage · P(door) · walls), p_cam, p_view — and the replayed posterior's distance to the recorded one.
// A replay that disagrees with its own record (pose Δ ≫ 0, or p_lidar off) means the update is no longer the one that
// ran live: rebuild, or check the format version.
//
//   Build + run (from door_concept/):
//     Q=/usr/include/x86_64-linux-gnu/qt5; g++ -std=c++23 -O2 -fPIC -I/usr/include/eigen3 -Isrc -I../common -I$Q -I$Q/QtCore
//       -I$Q/QtGui -I$Q/QtWidgets tests/replay_anchor_frames.cpp -o build/replay_anchor_frames -lfastdds -lfastcdr -lQt5Core
//     ./build/replay_anchor_frames [etc/anchor_frames.bin] [--every N]
//     ./build/replay_anchor_frames FILE --chain [--time-scale S] [--freeze] [--from T0] [--to T1] [--window A B] [--every N]
//
// The recorded p_cam / p_view are AS PUBLISHED (fix 1: the last mask frame held for kAnchorStaleMs, its camera term
// recomputed at the posterior); the replayed ones are this frame's own (p_cam > 0 only on a frame that carried a mask),
// so they differ on mask-less frames by design — compare p_lidar for the update itself.
// Format version 4 (Task 10: the persistent-bias factor; versions 3 and 2 read too). A version-1 file (cc39a46c, e.g. try66) is read too (Task 9): its records hold the
// PRE-budget rays, so the replay stages them itself with the CURRENT model — the replayed numbers are then today's
// model on that run's inputs, not a re-run of the update that ran live (the pose Δ is the model change, not an error).
//
// --chain (Task 9): the CALLER's half replayed too. Each record's prior was predicted live from the previous posterior;
// the step's motion is recovered exactly from that pair (the inverse of relative_pose.h move_mean) together with the
// covariance the prediction added, and the anchor is then CHAINED: predicted from the replay's own posterior by that motion
// and updated on the record's inputs. A gap > 2 s (a new anchor) restarts the chain from the record's prior.
//   --time-scale S   the step motion × S: the twist integrated over the SIMULATION clock (S = Δsim/Δwall, try66 ≈ 0.94),
//                    which is what the agent now does (rc::rt::SimClockRate); 1 = the live wall-clock integration.
//   --freeze         the door's width and wall run taken from the first record of each chain and kept (the agent now reads
//                    them with the pose, at creation / re-init); without it each record's own (the live per-cycle values).
//   --window A B     also summarise p_view over A ≤ t ≤ B s (the chain itself still runs from --from).
//   --motion-csv F   the step motion from the robot's LOCALISED poses in F (gt_error.csv: ts_ms, est_x, est_y, est_theta;
//                    room frame, est_theta = the robot frame's yaw) instead of the record — for a record whose live
//                    prediction was broken (try68: the f.r.1 clock bug, dt = 0 for the whole run). Its covariance growth:
//   --motion-sd A,S,R  per-second velocity σ of adv, side, rot (default 0.011, 0.011, 0.010: try67's live per-step growth).
//   --width W        a what-if: the anchor's model width forced to W (e.g. a door's CLEAR width between its frames).
//   --gt F           Task 10: score every posterior against the Webots truth in F (gt_error.csv: gt_x_world, gt_y_world,
//                    gt_theta_raw; DOOR_1 of tworooms-piso.wbt) and print, per leg and window, the CALIBRATION of the
//                    claimed σ on the published frames (p_view ≥ 0.5): |err|, σ, |err|/σ medians and 2σ coverage.
//   --diag           with --gt: per frame, the prior's, the data-only optimum's (x̄ + I_L⁺ I_L ν) and the posterior's error,
//                    the frame's robust data σ, the gain, and the persistent bias's σ (Task 10's diagnosis).
//   --dump F         with --chain: every posterior to F as CSV (t_ms, x, y, yaw, P row-major, p_view).
//   --rot-scale K    Task 12: add the per-step rotation-scale variance (K·|Δθ|)² through the step's Jacobian (relative_pose.h
//                    kRotScaleSigma = 0.09) — for records written before the live predict added it.
//   --no-bias        Task 10 ablation: the persistent bias factor is NOT carried between frames (the agent carries it).
//   --casing C       the caller's leaf for records written before it (leaf.len = 0, v1/v2): hinged at and spanning the clear
//                    width rc::door::anchor_clear_width(width, C) — what the agent now passes (DoorConcept.FrameCasingM, 0.06).
//   --lost-check     Task D-B, with --chain and --gt: the per-cycle lost check (AnchorLostMode: IMM mixing before each update,
//                    evidence after it, the two modes' combination as the scored pose) — the room init each cycle is the
//                    chain's first record (a fresh anchor's prior is the room init) carried through the TRUE robot pose.
//                    Prints a LOST row per frame: P(lost), the room channel, the mixing weight, tracker vs published pose.
//   --lost-no-feedback  diagnosis: the mode's output only, the tracker never mixed (see lost_no_feedback).
// The held camera term is replayed as live (the last mask frame for kAnchorStaleMs, carried by the step motion), so the
// chained p_view is the one that would be PUBLISHED. ⚠ The rays are the ones select_anchor_rays kept AT THE LIVE PRIOR: a
// chained prior far from the live one is scored on that selection (floor returns through the opening it predicts are
// missing), which can only understate it.
#include <charconv>
#include <cmath>
#include <cstring>
#include <format>
#include <fstream>
#include <print>
#include <string>
#include <vector>

#include "anchor_frame_record.h"
#include "door_anchor_open.h"   // Task D-B: --lost-check (AnchorLostMode)
#include "interaction_anchor/interaction_anchor.h"   // kAnchorStaleMs
#include <algorithm>
#include <locale>
#include <optional>
#include <sstream>

namespace
{
bool no_bias = false;   // --no-bias: the ablation — no bias state carried (each frame's own bias only, L = 0 before it)
bool diag = false;
double rot_scale = 0.0;   // --rot-scale K (Task 12): the per-step rotation-scale variance the live caller now adds      // --diag: per frame, the prior's / the data-only optimum's / the posterior's error against the truth
std::ofstream dump;     // --dump F: every chained posterior as CSV (t_ms, x, y, yaw, P row-major, p_view) — for the servo check
bool num(const char* a, double& out)
{
    const auto [p, ec] = std::from_chars(a, a + std::strlen(a), out);
    return ec == std::errc{} and p == a + std::strlen(a);
}

// The robot's localised poses (ts_ms, x, y, yaw), from gt_error.csv's ts_ms / est_x / est_y / est_theta columns.
struct PoseRow { std::int64_t t; double x, y, th; };
std::vector<PoseRow> read_motion_csv(const std::string& path)
{
    std::vector<PoseRow> out;
    std::ifstream f(path);
    std::string line;
    if (not std::getline(f, line)) return out;
    std::vector<std::string> head;
    { std::stringstream ss(line); std::string c; while (std::getline(ss, c, ',')) head.push_back(c); }
    auto col = [&](const char* n) { return int(std::find(head.begin(), head.end(), n) - head.begin()); };
    const int it = col("ts_ms"), ix = col("est_x"), iy = col("est_y"), ith = col("est_theta");
    while (std::getline(f, line))
    {
        std::vector<std::string> c; std::stringstream ss(line); std::string v; while (std::getline(ss, v, ',')) c.push_back(v);
        const int need = std::max({it, ix, iy, ith});
        if (int(c.size()) <= need) continue;
        PoseRow r{}; double t = 0;
        if (num(c[it].c_str(), t) and num(c[ix].c_str(), r.x) and num(c[iy].c_str(), r.y) and num(c[ith].c_str(), r.th))
        { r.t = std::int64_t(t); out.push_back(r); }
    }
    return out;
}
std::optional<PoseRow> pose_at(const std::vector<PoseRow>& rows, std::int64_t t)
{
    if (rows.empty()) return std::nullopt;
    const auto it = std::lower_bound(rows.begin(), rows.end(), t, [](const PoseRow& r, std::int64_t v) { return r.t < v; });
    if (it == rows.end()) return rows.back();
    return *it;
}

// ─── Ground truth (Task 10): the robot's Webots pose (gt_error.csv: ts_ms, gt_x_world, gt_y_world, gt_theta_raw) and the
// door's opening, so each frame's posterior can be scored against the truth and its claimed σ CALIBRATED.
//   The robot's world yaw is −gt_theta_raw − π/2 (Task 8/9's convention, checked against the robot's motion).
//   DOOR_1 (tworooms-piso.wbt): the gap between ROOM2_WALL_SOUTH_WEST and _EAST is x ∈ [−5.30, −4.24] at the walls'
//   mid-plane y = 9.425 (the frame's own translation, (−4.769, 9.380), is the door node, 4.5 cm off the mid-plane:
//   that offset is in DEPTH only), the wall runs along world x, so the anchor's true yaw is −(robot world yaw).
struct GtRow { std::int64_t t; double x, y, th; };
std::vector<GtRow> read_gt_csv(const std::string& path)
{
    std::vector<GtRow> out;
    std::ifstream f(path);
    std::string line;
    if (not std::getline(f, line)) return out;
    std::vector<std::string> head;
    { std::stringstream ss(line); std::string c; while (std::getline(ss, c, ',')) head.push_back(c); }
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
constexpr double kDoorX = -4.77, kDoorY = 9.425;
// The posterior's error against the truth, in the TRUE anchor's axes: lateral (along the wall), yaw, and the robot's
// true normal coordinate s; plus the claimed σ along the same axis. The anchor's axes may be flipped (the robot on the
// other side): then the yaw differs by π and the along-wall axis is reversed.
struct GtErr { bool ok = false; double lat = 0, yaw = 0, s = 0, sig_lat = 0, sig_yaw = 0; };
// The truth at t, interpolated between the two bracketing rows (≈ 20 Hz; the next row alone is up to 110 ms late,
// 3° of a 0.5 rad/s turn).
std::optional<GtRow> gt_at(const std::vector<GtRow>& gt, std::int64_t t)
{
    auto it = std::lower_bound(gt.begin(), gt.end(), t, [](const GtRow& r, std::int64_t v) { return r.t < v; });
    if (it == gt.end() or it == gt.begin()) return std::nullopt;
    const GtRow& g1 = *it; const GtRow& g0 = *std::prev(it);
    if (g1.t - g0.t > 200 or g1.t == g0.t) return std::nullopt;
    const double a_ = double(t - g0.t) / double(g1.t - g0.t);
    return GtRow{t, g0.x + a_ * (g1.x - g0.x), g0.y + a_ * (g1.y - g0.y), g0.th + a_ * std::remainder(g1.th - g0.th, 2 * M_PI)};
}
GtErr gt_error(const std::vector<GtRow>& gt, std::int64_t t, const rc::interaction::RelPose& post)
{
    GtErr e;
    if (gt.empty()) return e;
    const auto gi_ = gt_at(gt, t);
    if (not gi_.has_value()) return e;
    const GtRow gi = gi_.value();
    const GtRow* it2 = &gi;
    const double c = std::cos(it2->th), s = std::sin(it2->th), dx = kDoorX - it2->x, dy = kDoorY - it2->y;
    const double tx = c * dx + s * dy, ty = -s * dx + c * dy, tz = std::remainder(-it2->th, 2 * M_PI);
    const Eigen::Vector2d a(std::cos(tz), std::sin(tz));
    double lat = (post.x.x() - tx) * a.x() + (post.x.y() - ty) * a.y();
    double ye = std::remainder(post.x.z() - tz, 2 * M_PI);
    if (std::abs(ye) > M_PI / 2) { ye = std::remainder(ye + M_PI, 2 * M_PI); lat = -lat; }
    e.ok = true; e.lat = lat; e.yaw = ye; e.s = -(-std::sin(tz) * tx + std::cos(tz) * ty);
    const Eigen::Matrix2d P2 = post.P.topLeftCorner<2, 2>().cast<double>();
    e.sig_lat = std::sqrt(std::max(0.0, a.dot(P2 * a)));
    e.sig_yaw = std::sqrt(std::max(0.0, double(post.P(2, 2))));
    return e;
}
struct Calib
{
    std::vector<double> zl, zy, el, ey, sl, sy;
    void add(const GtErr& e)
    {
        zl.push_back(std::abs(e.lat) / std::max(e.sig_lat, 1e-9)); zy.push_back(std::abs(e.yaw) / std::max(e.sig_yaw, 1e-9));
        el.push_back(std::abs(e.lat)); ey.push_back(std::abs(e.yaw)); sl.push_back(e.sig_lat); sy.push_back(e.sig_yaw);
    }
    static double med(std::vector<double> v) { if (v.empty()) return 0; std::ranges::nth_element(v, v.begin() + std::ptrdiff_t(v.size() / 2)); return v[v.size() / 2]; }
    static double mx(const std::vector<double>& v) { return v.empty() ? 0 : std::ranges::max(v); }
    static double cov2(const std::vector<double>& z) { if (z.empty()) return 0; return 100.0 * double(std::ranges::count_if(z, [](double q) { return q <= 2.0; })) / double(z.size()); }
    void print(const char* what) const
    {
        if (zl.empty()) return;
        std::print("  calib {} ({} published frames): lat |err| med {:.3f} max {:.3f} m, σ_lat med {:.3f} m, |err|/σ med {:.2f}, 2σ cover {:.0f} % | "
                   "yaw |err| med {:.1f} max {:.1f}°, σ_yaw med {:.2f}°, |err|/σ med {:.2f}, 2σ cover {:.0f} %\n",
                   what, zl.size(), med(el), mx(el), med(sl), med(zl), cov2(zl), med(ey) * 180 / M_PI, mx(ey) * 180 / M_PI,
                   med(sy) * 180 / M_PI, med(zy), cov2(zy));
    }
};

struct LegStats
{
    int n = 0, published = 0, ge06 = 0, pub_pred = 0; double sum = 0, min = 1; std::int64_t t0 = 0, t1 = 0, last_seen = -1; Calib cal;
    // Task 12 fix B: the doorway band (TRUE |s| ≤ 0.5 m, needs --gt): frames seen, seen-or-predicted under the 3-s wall-clock
    // window, and under the band-CHARGED window (rc::door::anchor_unseen_charge at the last seen posterior, CrossClearanceM 0.80).
    int band_n = 0, band_seen = 0, band_old = 0, band_new = 0;
    std::int64_t charge = 0, charge_t = -1;
    rc::interaction::RelPose seen_pose;
};
constexpr float kReplayCrossClearanceM = 0.80f;   // door_concept's DoorAffordance.CrossClearanceM default (the doorway band)
void print_leg(const LegStats& s, int k)
{
    if (s.n == 0) return;
    std::print("leg {}: t {:.1f}–{:.1f} s, {} frames: p_view mean {:.3f} min {:.3f}, ≥ 0.6 on {} ({:.0f} %), published (≥ 0.5) {}, "
               "published if the cross were claimed (seen, or predicted ≤ kAnchorPredictMs) {}\n", k,
               s.t0 * 1e-3, s.t1 * 1e-3, s.n, s.sum / s.n, s.min, s.ge06, 100.0 * s.ge06 / s.n, s.published, s.pub_pred);
    if (s.band_n > 0)
        std::print("  doorway band (true |s| ≤ 0.5 m): {} frames — seen {}, seen-or-predicted {} with the 3-s wall window, {} with the "
                   "band-charged window (Task 12 fix B)\n", s.band_n, s.band_seen, s.band_old, s.band_new);
    s.cal.print("seen");
}

// --lost-check (Task D-B): the per-cycle lost check (door_anchor_open.h AnchorLostMode) replayed with the anchor. The
// room initialisation each cycle is rebuilt from the chain's FIRST record — a fresh anchor's prior IS the room
// initialisation (hung from the room) — taken into the world with the robot's true pose at that instant (--gt), and then
// brought back into each later frame's robot frame by the true pose then; its P0 is the first record's, carried in anchor
// axes. So the replayed room init holds the live room-frame error at the hang and no later localisation noise (stated).
bool lost_check = false;
bool lost_no_feedback = false;   // --lost-no-feedback: the mode's OUTPUT only — the tracker is never mixed (diagnosis: the
                                 // records' rays were selected at the LIVE prior, so a tracker the mixing moves is scored on
                                 // rays chosen for somewhere else; this isolates what the output alone does)

float wk_lost(const rc::door::rec::Frame& f) { return rc::door::anchor_clear_width(f.width, 0.06f); }   // pa.clear_width (FrameCasingM 0.06)

int chain(const std::vector<rc::door::rec::Frame>& frames, double scale, bool freeze, double from_s, double to_s, int every,
          double win_a, double win_b, double width_override, double casing, const std::vector<PoseRow>& motion,
          const Eigen::Vector3f& motion_var, const std::vector<GtRow>& gt)
{
    rc::door::AnchorLostMode lost_mode;
    bool lost_ok = false;                       // an opening in the world for this chain (its first record + the truth)
    Eigen::Vector3d open_world = Eigen::Vector3d::Zero();   // x, y, yaw of the room init's opening, world
    Eigen::Matrix3f P0_axes = Eigen::Matrix3f::Zero();       // its P0 in anchor axes
    const auto room_init_at = [&](std::int64_t t) -> std::optional<rc::interaction::RelPose>
    {
        if (not lost_ok) return std::nullopt;
        const auto g = gt_at(gt, t);
        if (not g.has_value()) return std::nullopt;
        const double c = std::cos(g->th), s = std::sin(g->th), dx = open_world.x() - g->x, dy = open_world.y() - g->y;
        rc::interaction::RelPose r;
        r.x = Eigen::Vector3f(float(c * dx + s * dy), float(-s * dx + c * dy), float(std::remainder(open_world.z() - g->th, 2 * M_PI)));
        r.P = rc::door::anchor_axes_to_robot(P0_axes, r.x.z());
        r.stamp_ms = t;
        return r;
    };
    using rc::interaction::RelPose;
    const std::int64_t t0 = frames.empty() ? 0 : frames.front().t_ms;
    std::print("{:>8} {:>5} | {:>7} {:>7} {:>7} | {:>6} {:>6} | {:>6} = {:>5} x {:>5} x {:>5} | {:>5} {:>6} | {:>6} {}\n", "t_ms", "rays", "x", "y",
               "yaw", "e", "s", "pl", "open", "door", "walls", "pc", "pv", "pv_rec", "| sig_lat sig_yaw");
    RelPose x;
    rc::door::AnchorBias bias;   // Task 10: the persistent bias factor, carried as the agent carries it
    rc::door::CameraViewHold hold;
    float w = 0, u0 = 0, u1 = 0;
    bool have = false;
    LegStats leg, win;
    int legs = 0;
    for (std::size_t k = 0; k < frames.size(); ++k)
    {
        const auto& f = frames[k];
        const double t = (f.t_ms - t0) * 1e-3;
        if (t < from_s or t > to_s) continue;
        const bool restart = not have or k == 0 or f.t_ms - frames[k - 1].t_ms > 2000;
        if (restart)
        {
            if (have) print_leg(leg, ++legs);
            leg = {};
            leg.t0 = f.t_ms - t0;
            x = f.prior;
            bias = f.bias;
            hold = {};
            lost_mode = {};
            lost_ok = false;
            if (lost_check)
                if (const auto g = gt_at(gt, f.t_ms); g.has_value())
                {
                    const double c = std::cos(g->th), sn = std::sin(g->th);
                    open_world = {g->x + c * f.prior.x.x() - sn * f.prior.x.y(), g->y + sn * f.prior.x.x() + c * f.prior.x.y(),
                                  std::remainder(g->th + f.prior.x.z(), 2 * M_PI)};
                    P0_axes = rc::door::robot_to_anchor_axes(f.prior.P, f.prior.x.z());
                    lost_ok = true;
                    std::print("LOST-CHECK chain at {} ms: room init opening (world) ({:+.3f},{:+.3f}) yaw {:+.3f}; P0 σ (anchor axes) "
                               "({:.3f},{:.3f},{:.3f})\n", f.t_ms - t0, open_world.x(), open_world.y(), open_world.z(),
                               std::sqrt(P0_axes(0, 0)), std::sqrt(P0_axes(1, 1)), std::sqrt(P0_axes(2, 2)));
                }
            w = f.width; u0 = f.wall_u_min; u1 = f.wall_u_max;
            have = true;
        }
        else
        {
            // The live step: prior_k = move_mean(post_{k-1}, side, adv, dθ) + Q — recovered, then scaled and re-applied.
            const auto& a = frames[k - 1].post;
            const auto& b = f.prior;
            const float dth = rc::interaction::wrap_pi(a.x.z() - b.x.z());
            const float c = std::cos(dth), sn = std::sin(dth);
            const float side = a.x.x() - (c * b.x.x() - sn * b.x.y()), adv = a.x.y() - (sn * b.x.x() + c * b.x.y());
            Eigen::Matrix3f J, G;
            rc::interaction::predict_jacobians(a.x, side, adv, dth, J, G);
            const Eigen::Matrix3f Q = b.P - J * a.P * J.transpose();
            float s_ = float(scale);
            Eigen::Matrix3f Qm = Q;
            const float wk_model = width_override > 0 ? float(width_override) : freeze ? w : f.width;
            if (not motion.empty())
                if (const auto p0 = pose_at(motion, frames[k - 1].t_ms), p1 = pose_at(motion, f.t_ms); p0.has_value() and p1.has_value())
                {
                    // The robot's motion in its own frame at k−1: world = pos + R(θ)·p_robot (x right, y forward).
                    const double c0 = std::cos(p0->th), s0 = std::sin(p0->th), dxw = p1->x - p0->x, dyw = p1->y - p0->y;
                    const float m_side = float(c0 * dxw + s0 * dyw), m_adv = float(-s0 * dxw + c0 * dyw);
                    const float m_dth = rc::interaction::wrap_pi(float(p1->th - p0->th));
                    const float dts = float(f.t_ms - frames[k - 1].t_ms) * 1e-3f;
                    Eigen::Matrix3f J2, G2;
                    rc::interaction::predict_jacobians(x.x, m_side, m_adv, m_dth, J2, G2);
                    RelPose p; p.x = rc::interaction::move_mean(x.x, m_side, m_adv, m_dth);
                    p.P = J2 * x.P * J2.transpose() + G2 * motion_var.asDiagonal() * G2.transpose() * (dts * dts);
                    p.stamp_ms = b.stamp_ms;
                    rc::door::transport_bias(bias, x.x, m_side, m_adv, m_dth, wk_model);
                    x = p;
                    rc::door::carry_held_mask(hold, m_side, m_adv, m_dth);
                    goto staged;
                }
            rc::interaction::predict_jacobians(x.x, s_ * side, s_ * adv, s_ * dth, J, G);
            RelPose p;
            p.x = rc::interaction::move_mean(x.x, s_ * side, s_ * adv, s_ * dth);
            p.P = J * x.P * J.transpose() + s_ * s_ * Q;
            // --rot-scale K: the rotation-scale variance (relative_pose.h kRotScaleSigma) on records written without it.
            if (rot_scale > 0.0)
            {
                const float sr = float(rot_scale) * std::abs(s_ * dth);
                p.P += (sr * sr) * (G.col(2) * G.col(2).transpose());
            }
            p.stamp_ms = b.stamp_ms;
            rc::door::transport_bias(bias, x.x, s_ * side, s_ * adv, s_ * dth, wk_model);
            x = p;
            rc::door::carry_held_mask(hold, s_ * side, s_ * adv, s_ * dth);
            (void)Qm;
        }
        staged:
        const auto room_now = lost_check ? room_init_at(f.t_ms) : std::nullopt;
        float lost_w = 0.f;
        if (lost_check)   // every record is an update: the mode's transition, and the IMM mixing when a room init exists
        {
            rc::interaction::RelPose x_mixed = x;
            rc::door::AnchorBias b_mixed = bias;
            lost_w = rc::door::lost_mode_predict(lost_mode, x_mixed, no_bias ? nullptr : &b_mixed,
                                                 room_now.has_value() ? &room_now.value() : nullptr, wk_lost(f));
            if (not lost_no_feedback) { x = x_mixed; bias = b_mixed; }
        }
        rc::door::AnchorStaging st = f.staging;   // v2: the record's budgeted sets; v1: empty, its pre-budget rays are staged
        st.replay = f.version != 1;
        const float wk = width_override > 0 ? float(width_override) : freeze ? w : f.width, lo = freeze ? u0 : f.wall_u_min, hi = freeze ? u1 : f.wall_u_max;
        rc::door::LeafPrior leaf = f.leaf;
        if (casing >= 0.0 and leaf.known and leaf.len <= 0.f)
        {
            const float clear = rc::door::anchor_clear_width(wk, float(casing));
            leaf.hinge_u = (leaf.hinge_u >= 0.f ? 0.5f : -0.5f) * clear;
            leaf.len = clear;
        }
        const RelPose xprior_ = x;
        auto [post, ev] = rc::door::update_door_anchor(x, wk, f.thickness, lo, hi, f.lidar_origin,
                                                       f.version == 1 ? f.rays_v1 : std::vector<Eigen::Vector3f>{}, f.mask, f.cam,
                                                       f.pi_open, leaf, f.floor_below, &st, no_bias ? nullptr : &bias);
        if (not f.mask.empty()) rc::door::hold_mask_frame(hold, f.mask, f.t_ms);
        ev.p_cam = rc::door::held_camera_view(hold, f.t_ms, rc::interaction::kAnchorStaleMs, f.cam, wk, post);
        ev.p_view = rc::door::combine_view(ev.p_lidar, ev.p_cam);
        x = post;
        if (lost_check) rc::door::lost_mode_view(lost_mode, x, ev.p_view, wk_lost(f));
        if (room_now.has_value())
        {
            const auto le = rc::door::lost_mode_posterior(lost_mode, x, room_now.value(), wk_lost(f));
            post = rc::door::lost_mode_output(x, room_now.value(), lost_mode.p);   // what would be PUBLISHED
            const auto gt_t = gt_error(gt, f.t_ms, x), gt_o = gt_error(gt, f.t_ms, post);
            if (k % std::size_t(every) == 0)
                std::print("{:8d} LOST P {:.3f} (d2 {:7.1f} room_llr {:+7.1f}) w_mix {:.3f} | tracker ({:+.3f},{:+.3f},{:+.3f}) out ({:+.3f},{:+.3f},{:+.3f}) "
                           "σ_out ({:.3f},{:.3f}) | room ({:+.3f},{:+.3f},{:+.3f}){}\n",
                           f.t_ms - t0, lost_mode.p, le.d2, le.room_llr, lost_w, x.x.x(), x.x.y(), x.x.z(), post.x.x(), post.x.y(), post.x.z(),
                           std::sqrt(post.P(0, 0)), std::sqrt(post.P(1, 1)), room_now->x.x(), room_now->x.y(), room_now->x.z(),
                           gt_t.ok ? std::format(" | GT err tracker lat {:+.2f} yaw {:+.0f}° out lat {:+.2f} yaw {:+.0f}°", gt_t.lat,
                                                 gt_t.yaw * 180 / M_PI, gt_o.lat, gt_o.yaw * 180 / M_PI) : std::string{});
        }
        ++leg.n; leg.sum += ev.p_view; leg.min = std::min(leg.min, double(ev.p_view));
        leg.published += ev.p_view >= 0.5f ? 1 : 0; leg.ge06 += ev.p_view >= 0.6f ? 1 : 0; leg.t1 = f.t_ms - t0;
        const GtErr ge = gt_error(gt, f.t_ms, post);
        auto note_pred = [&](LegStats& L)
        {
            if (ev.p_view >= 0.5f) { L.last_seen = f.t_ms; L.charge = 0; L.seen_pose = post; }
            else if (L.charge_t >= 0 and L.last_seen >= 0)
                L.charge = rc::door::anchor_unseen_charge(L.charge, f.t_ms - L.charge_t,
                                                          rc::door::anchor_doorway_band(L.seen_pose, kReplayCrossClearanceM));
            L.charge_t = f.t_ms;
            const bool old_rule = L.last_seen >= 0 and f.t_ms - L.last_seen <= rc::interaction::kAnchorPredictMs;
            L.pub_pred += old_rule ? 1 : 0;
            if (ge.ok and std::abs(ge.s) <= 0.5)
            {
                ++L.band_n;
                L.band_seen += ev.p_view >= 0.5f ? 1 : 0;
                L.band_old  += old_rule ? 1 : 0;
                L.band_new  += (L.last_seen >= 0 and L.charge <= rc::interaction::kAnchorPredictMs) ? 1 : 0;
            }
        };
        note_pred(leg);
        if (dump.is_open())
        {
            dump << (f.t_ms - t0) << ',' << post.x.x() << ',' << post.x.y() << ',' << post.x.z();
            for (int a = 0; a < 3; ++a) for (int c = 0; c < 3; ++c) dump << ',' << post.P(a, c);
            dump << ',' << ev.p_view << '\n';
        }
        if (ge.ok and ev.p_view >= 0.5f) leg.cal.add(ge);
        if (t >= win_a and t <= win_b)
        {
            if (win.n == 0) win.t0 = f.t_ms - t0;
            ++win.n; win.sum += ev.p_view; win.min = std::min(win.min, double(ev.p_view));
            win.published += ev.p_view >= 0.5f ? 1 : 0; win.ge06 += ev.p_view >= 0.6f ? 1 : 0; win.t1 = f.t_ms - t0;
            note_pred(win);
            if (ge.ok and ev.p_view >= 0.5f) win.cal.add(ge);
        }
        if (k % std::size_t(every) != 0) continue;
        const float cz = std::cos(post.x.z()), sz = std::sin(post.x.z());
        std::print("{:8d} {:5d} | {:+7.3f} {:+7.3f} {:+7.3f} | {:+6.2f} {:+6.2f} | {:6.3f} = {:5.3f} x {:5.3f} x {:5.3f} | {:5.3f} {:6.3f} | {:6.3f} | {:.4f} {:.4f}\n",
                   f.t_ms - t0, f.version == 1 ? std::uint32_t(f.rays_v1.size()) : f.staging.prior.n_pre, post.x.x(), post.x.y(), post.x.z(),
                   -(cz * post.x.x() + sz * post.x.y()), -(-sz * post.x.x() + cz * post.x.y()), ev.p_lidar, ev.lidar_open, ev.lidar_door,
                   ev.lidar_wall, ev.p_cam, ev.p_view, f.p_view,
                   std::sqrt(std::max(0.f, Eigen::Vector2f(cz, sz).dot(post.P.topLeftCorner<2, 2>() * Eigen::Vector2f(cz, sz)))),
                   std::sqrt(std::max(0.f, post.P(2, 2))));
        if (ge.ok and diag)
        {
            const float tz = post.x.z();
            const Eigen::Vector3f al(std::cos(tz), std::sin(tz), 0), ay(0, 0, 1);
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(ev.info);
            Eigen::Matrix3f Ip = Eigen::Matrix3f::Zero();
            for (int q = 0; q < 3; ++q) if (es.eigenvalues()(q) > 1e-9f * es.eigenvalues().maxCoeff()) Ip += es.eigenvectors().col(q) * es.eigenvectors().col(q).transpose() / es.eigenvalues()(q);
            const Eigen::Matrix3f LL = bias.L * bias.L.transpose();
            auto sd = [&](const Eigen::Matrix3f& V, const Eigen::Vector3f& a) { return std::sqrt(std::max(0.f, a.dot(V * a))); };
            RelPose xl = post; xl.x = xprior_.x + Ip * ev.info_innov;
            const GtErr gl = gt_error(gt, f.t_ms, xl), gp = gt_error(gt, f.t_ms, xprior_);
            const Eigen::Matrix3f Vr = Ip * (ev.misfit + ev.info) * Ip;
            std::print("{:8d} DIAG prior err lat {:+.3f} yaw {:+.2f}° (σ {:.3f} {:.2f}°) | data-only err lat {:+.3f} yaw {:+.2f}° (robust σ {:.3f} {:.2f}°) | "
                       "gain lat {:.2f} yaw {:.2f} | L σ lat {:.3f} yaw {:.2f}° | P_z σ lat {:.3f}\n", f.t_ms - t0,
                       gp.lat, gp.yaw * 180 / M_PI, gp.sig_lat, gp.sig_yaw * 180 / M_PI, gl.lat, gl.yaw * 180 / M_PI, sd(Vr, al), sd(Vr, ay) * 180 / M_PI,
                       al.dot(ev.gain * al), ev.gain(2, 2), sd(LL, al), sd(LL, ay) * 180 / M_PI, sd(Eigen::Matrix3f(post.P - LL), al));
        }
        if (ge.ok) std::print("{:8d} GT lat {:+.3f} yaw {:+.2f}° s {:+.2f} | z_lat {:.2f} z_yaw {:.2f}\n", f.t_ms - t0, ge.lat, ge.yaw * 180 / M_PI, ge.s,
                              std::abs(ge.lat) / std::max(ge.sig_lat, 1e-9), std::abs(ge.yaw) / std::max(ge.sig_yaw, 1e-9));
    }
    if (have) print_leg(leg, ++legs);
    if (win.n > 0) { std::print("window "); print_leg(win, 0); }
    return 0;
}
}  // namespace

int main(int argc, char** argv)
{
    std::string path = "etc/anchor_frames.bin";
    int every = 1;
    bool chained = false, freeze = false;
    double scale = 1.0, from_s = -1e9, to_s = 1e9, win_a = 1e9, win_b = -1e9, width_override = 0, casing = -1;
    std::string motion_csv, gt_csv;
    Eigen::Vector3f motion_var(0.011f * 0.011f, 0.011f * 0.011f, 0.010f * 0.010f);
    for (int i = 1; i < argc; ++i)
    {
        double v = 0;
        if (std::strcmp(argv[i], "--every") == 0 and i + 1 < argc) { if (num(argv[++i], v) and v >= 1) every = int(v); }
        else if (std::strcmp(argv[i], "--chain") == 0) chained = true;
        else if (std::strcmp(argv[i], "--no-bias") == 0) no_bias = true;
        else if (std::strcmp(argv[i], "--diag") == 0) diag = true;
        else if (std::strcmp(argv[i], "--rot-scale") == 0 and i + 1 < argc) { if (num(argv[++i], v) and v >= 0) rot_scale = v; }
        else if (std::strcmp(argv[i], "--dump") == 0 and i + 1 < argc)
        {
            dump.open(argv[++i]);
            dump.imbue(std::locale::classic());
            dump << "t_ms,x,y,yaw,P00,P01,P02,P10,P11,P12,P20,P21,P22,p_view\n";
        }
        else if (std::strcmp(argv[i], "--freeze") == 0) freeze = true;
        else if (std::strcmp(argv[i], "--lost-check") == 0) lost_check = true;
        else if (std::strcmp(argv[i], "--lost-no-feedback") == 0) lost_no_feedback = true;
        else if (std::strcmp(argv[i], "--time-scale") == 0 and i + 1 < argc) { if (num(argv[++i], v) and v > 0) scale = v; }
        else if (std::strcmp(argv[i], "--from") == 0 and i + 1 < argc) { if (num(argv[++i], v)) from_s = v; }
        else if (std::strcmp(argv[i], "--to") == 0 and i + 1 < argc) { if (num(argv[++i], v)) to_s = v; }
        else if (std::strcmp(argv[i], "--motion-csv") == 0 and i + 1 < argc) motion_csv = argv[++i];
        else if (std::strcmp(argv[i], "--gt") == 0 and i + 1 < argc) gt_csv = argv[++i];
        else if (std::strcmp(argv[i], "--motion-sd") == 0 and i + 1 < argc)
        {
            std::stringstream ss(argv[++i]); std::string tok; int j = 0;
            while (std::getline(ss, tok, ',') and j < 3) { double d = 0; if (num(tok.c_str(), d)) motion_var(j) = float(d * d); ++j; }
        }
        else if (std::strcmp(argv[i], "--casing") == 0 and i + 1 < argc) { if (num(argv[++i], v) and v >= 0) casing = v; }
        else if (std::strcmp(argv[i], "--width") == 0 and i + 1 < argc) { if (num(argv[++i], v) and v > 0) width_override = v; }
        else if (std::strcmp(argv[i], "--window") == 0 and i + 2 < argc) { if (num(argv[++i], v)) win_a = v; if (num(argv[++i], v)) win_b = v; }
        else path = argv[i];
    }
    std::ifstream in(path, std::ios::binary);
    if (not in.is_open()) { std::print(stderr, "cannot open {}\n", path); return 2; }
    std::vector<rc::door::rec::Frame> frames;
    std::uint32_t version = 0;
    if (not rc::door::rec::read_all(in, frames, &version)) { std::print(stderr, "{}: not an anchor frame record (or another version)\n", path); return 2; }
    std::print("{}: version {}, {} frames\n", path, version, frames.size());
    if (chained)
    {
        std::print("CHAINED: step motion × {:.3f}, width/run {}{}\n", scale, freeze ? "frozen at each chain's first record" : "per record (live)",
                   width_override > 0 ? std::format(", width forced to {:.2f}", width_override) : std::string{});
        const auto motion = motion_csv.empty() ? std::vector<PoseRow>{} : read_motion_csv(motion_csv);
        if (not motion_csv.empty()) std::print("motion from {} ({} poses)\n", motion_csv, motion.size());
        const auto gt = gt_csv.empty() ? std::vector<GtRow>{} : read_gt_csv(gt_csv);
        if (not gt_csv.empty()) std::print("ground truth from {} ({} poses)\n", gt_csv, gt.size());
        return chain(frames, scale, freeze, from_s, to_s, every, win_a, win_b, width_override, casing, motion, motion_var, gt);
    }
    std::print("{:>8} {:>5} {:>4} {:>4} | {:>6} {:>6} {:>6} | {:>6} = {:>5} x {:>5} x {:>5} | {:>5} {:>5} | {:>6} {:>6}\n", "t_ms", "rays", "F", "mask",
               "pv_rec", "pl_rec", "pc_rec", "pl_rep", "open", "door", "walls", "pc", "pv", "dxy_m", "dyaw");
    const std::int64_t t0 = frames.empty() ? 0 : frames.front().t_ms;
    double worst_xy = 0, worst_yaw = 0, worst_pl = 0;
    for (std::size_t k = 0; k < frames.size(); ++k)
    {
        const auto& f = frames[k];
        if ((f.t_ms - t0) * 1e-3 < from_s or (f.t_ms - t0) * 1e-3 > to_s) continue;
        rc::door::AnchorBias fb = f.bias;   // v4: the factor the live update was handed (v1–v3: none)
        rc::door::AnchorStaging st = f.staging;
        st.replay = f.version != 1;   // v2: the record's budgeted sets stand for the frame; v1: stage its pre-budget rays
        const auto [post, ev] = rc::door::update_door_anchor(f.prior, f.width, f.thickness, f.wall_u_min, f.wall_u_max, f.lidar_origin,
                                                             f.version == 1 ? f.rays_v1 : std::vector<Eigen::Vector3f>{}, f.mask,
                                                             f.cam, f.pi_open, f.leaf, f.floor_below, &st, &fb);
        const double dxy = (post.x.head<2>() - f.post.x.head<2>()).norm();
        const double dyaw = std::abs(rc::interaction::wrap_pi(post.x.z() - f.post.x.z()));
        worst_xy = std::max(worst_xy, dxy); worst_yaw = std::max(worst_yaw, dyaw);
        worst_pl = std::max(worst_pl, double(std::abs(ev.p_lidar - f.p_lidar)));
        if (k % std::size_t(every) != 0) continue;
        std::print("{:8d} {:5d} {:4d} {:4d} | {:6.3f} {:6.3f} {:6.3f} | {:6.3f} = {:5.3f} x {:5.3f} x {:5.3f} | {:5.3f} {:5.3f} | {:6.4f} {:6.4f}{}\n",
                   f.t_ms - t0, f.version == 1 ? std::uint32_t(f.rays_v1.size()) : f.staging.prior.n_pre, ev.n_budget, f.mask.size(), f.p_view,
                   f.p_lidar, f.p_cam, ev.p_lidar, ev.lidar_open, ev.lidar_door, ev.lidar_wall, ev.p_cam, ev.p_view, dxy, dyaw,
                   f.p_view < 0.5f ? "  (not published)" : "");
    }
    std::print("replay vs record: worst pose Δ {:.4f} m / {:.4f} rad, worst |Δ p_lidar| {:.4f}\n", worst_xy, worst_yaw, worst_pl);
    return 0;
}
