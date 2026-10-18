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
// Format version 3 (Task 9 fix round 1; version 2 read too). A version-1 file (cc39a46c, e.g. try66) is read too (Task 9): its records hold the
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
//   --casing C       the caller's leaf for records written before it (leaf.len = 0, v1/v2): hinged at and spanning the clear
//                    width rc::door::anchor_clear_width(width, C) — what the agent now passes (DoorConcept.FrameCasingM, 0.06).
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
#include "interaction_anchor/interaction_anchor.h"   // kAnchorStaleMs
#include <algorithm>
#include <optional>
#include <sstream>

namespace
{
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

struct LegStats { int n = 0, published = 0, ge06 = 0, pub_pred = 0; double sum = 0, min = 1; std::int64_t t0 = 0, t1 = 0, last_seen = -1; };
void print_leg(const LegStats& s, int k)
{
    if (s.n == 0) return;
    std::print("leg {}: t {:.1f}–{:.1f} s, {} frames: p_view mean {:.3f} min {:.3f}, ≥ 0.6 on {} ({:.0f} %), published (≥ 0.5) {}, "
               "published if the cross were claimed (seen, or predicted ≤ kAnchorPredictMs) {}\n", k,
               s.t0 * 1e-3, s.t1 * 1e-3, s.n, s.sum / s.n, s.min, s.ge06, 100.0 * s.ge06 / s.n, s.published, s.pub_pred);
}

int chain(const std::vector<rc::door::rec::Frame>& frames, double scale, bool freeze, double from_s, double to_s, int every,
          double win_a, double win_b, double width_override, double casing, const std::vector<PoseRow>& motion,
          const Eigen::Vector3f& motion_var)
{
    using rc::interaction::RelPose;
    const std::int64_t t0 = frames.empty() ? 0 : frames.front().t_ms;
    std::print("{:>8} {:>5} | {:>7} {:>7} {:>7} | {:>6} {:>6} | {:>6} = {:>5} x {:>5} x {:>5} | {:>5} {:>6} | {:>6} {}\n", "t_ms", "rays", "x", "y",
               "yaw", "e", "s", "pl", "open", "door", "walls", "pc", "pv", "pv_rec", "| sig_lat sig_yaw");
    RelPose x;
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
            hold = {};
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
                    x = p;
                    rc::door::carry_held_mask(hold, m_side, m_adv, m_dth);
                    goto staged;
                }
            rc::interaction::predict_jacobians(x.x, s_ * side, s_ * adv, s_ * dth, J, G);
            RelPose p;
            p.x = rc::interaction::move_mean(x.x, s_ * side, s_ * adv, s_ * dth);
            p.P = J * x.P * J.transpose() + s_ * s_ * Q;
            p.stamp_ms = b.stamp_ms;
            x = p;
            rc::door::carry_held_mask(hold, s_ * side, s_ * adv, s_ * dth);
            (void)Qm;
        }
        staged:
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
        auto [post, ev] = rc::door::update_door_anchor(x, wk, f.thickness, lo, hi, f.lidar_origin,
                                                       f.version == 1 ? f.rays_v1 : std::vector<Eigen::Vector3f>{}, f.mask, f.cam,
                                                       f.pi_open, leaf, f.floor_below, &st);
        if (not f.mask.empty()) rc::door::hold_mask_frame(hold, f.mask, f.t_ms);
        ev.p_cam = rc::door::held_camera_view(hold, f.t_ms, rc::interaction::kAnchorStaleMs, f.cam, wk, post);
        ev.p_view = rc::door::combine_view(ev.p_lidar, ev.p_cam);
        x = post;
        ++leg.n; leg.sum += ev.p_view; leg.min = std::min(leg.min, double(ev.p_view));
        leg.published += ev.p_view >= 0.5f ? 1 : 0; leg.ge06 += ev.p_view >= 0.6f ? 1 : 0; leg.t1 = f.t_ms - t0;
        auto note_pred = [&](LegStats& L)
        {
            if (ev.p_view >= 0.5f) L.last_seen = f.t_ms;
            L.pub_pred += (L.last_seen >= 0 and f.t_ms - L.last_seen <= rc::interaction::kAnchorPredictMs) ? 1 : 0;
        };
        note_pred(leg);
        if (t >= win_a and t <= win_b)
        {
            if (win.n == 0) win.t0 = f.t_ms - t0;
            ++win.n; win.sum += ev.p_view; win.min = std::min(win.min, double(ev.p_view));
            win.published += ev.p_view >= 0.5f ? 1 : 0; win.ge06 += ev.p_view >= 0.6f ? 1 : 0; win.t1 = f.t_ms - t0;
            note_pred(win);
        }
        if (k % std::size_t(every) != 0) continue;
        const float cz = std::cos(post.x.z()), sz = std::sin(post.x.z());
        std::print("{:8d} {:5d} | {:+7.3f} {:+7.3f} {:+7.3f} | {:+6.2f} {:+6.2f} | {:6.3f} = {:5.3f} x {:5.3f} x {:5.3f} | {:5.3f} {:6.3f} | {:6.3f} | {:.4f} {:.4f}\n",
                   f.t_ms - t0, f.version == 1 ? std::uint32_t(f.rays_v1.size()) : f.staging.prior.n_pre, post.x.x(), post.x.y(), post.x.z(),
                   -(cz * post.x.x() + sz * post.x.y()), -(-sz * post.x.x() + cz * post.x.y()), ev.p_lidar, ev.lidar_open, ev.lidar_door,
                   ev.lidar_wall, ev.p_cam, ev.p_view, f.p_view,
                   std::sqrt(std::max(0.f, Eigen::Vector2f(cz, sz).dot(post.P.topLeftCorner<2, 2>() * Eigen::Vector2f(cz, sz)))),
                   std::sqrt(std::max(0.f, post.P(2, 2))));
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
    std::string motion_csv;
    Eigen::Vector3f motion_var(0.011f * 0.011f, 0.011f * 0.011f, 0.010f * 0.010f);
    for (int i = 1; i < argc; ++i)
    {
        double v = 0;
        if (std::strcmp(argv[i], "--every") == 0 and i + 1 < argc) { if (num(argv[++i], v) and v >= 1) every = int(v); }
        else if (std::strcmp(argv[i], "--chain") == 0) chained = true;
        else if (std::strcmp(argv[i], "--freeze") == 0) freeze = true;
        else if (std::strcmp(argv[i], "--time-scale") == 0 and i + 1 < argc) { if (num(argv[++i], v) and v > 0) scale = v; }
        else if (std::strcmp(argv[i], "--from") == 0 and i + 1 < argc) { if (num(argv[++i], v)) from_s = v; }
        else if (std::strcmp(argv[i], "--to") == 0 and i + 1 < argc) { if (num(argv[++i], v)) to_s = v; }
        else if (std::strcmp(argv[i], "--motion-csv") == 0 and i + 1 < argc) motion_csv = argv[++i];
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
        return chain(frames, scale, freeze, from_s, to_s, every, win_a, win_b, width_override, casing, motion, motion_var);
    }
    std::print("{:>8} {:>5} {:>4} {:>4} | {:>6} {:>6} {:>6} | {:>6} = {:>5} x {:>5} x {:>5} | {:>5} {:>5} | {:>6} {:>6}\n", "t_ms", "rays", "F", "mask",
               "pv_rec", "pl_rec", "pc_rec", "pl_rep", "open", "door", "walls", "pc", "pv", "dxy_m", "dyaw");
    const std::int64_t t0 = frames.empty() ? 0 : frames.front().t_ms;
    double worst_xy = 0, worst_yaw = 0, worst_pl = 0;
    for (std::size_t k = 0; k < frames.size(); ++k)
    {
        const auto& f = frames[k];
        if ((f.t_ms - t0) * 1e-3 < from_s or (f.t_ms - t0) * 1e-3 > to_s) continue;
        rc::door::AnchorStaging st = f.staging;
        st.replay = f.version != 1;   // v2: the record's budgeted sets stand for the frame; v1: stage its pre-budget rays
        const auto [post, ev] = rc::door::update_door_anchor(f.prior, f.width, f.thickness, f.wall_u_min, f.wall_u_max, f.lidar_origin,
                                                             f.version == 1 ? f.rays_v1 : std::vector<Eigen::Vector3f>{}, f.mask,
                                                             f.cam, f.pi_open, f.leaf, f.floor_below, &st);
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
