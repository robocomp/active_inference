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
// Format version 2 (fix round 1). A version-1 file (cc39a46c, e.g. try66) is read too (Task 9): its records hold the
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
//   --width W        a what-if: the anchor's model width forced to W (e.g. a door's CLEAR width between its frames).
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

namespace
{
bool num(const char* a, double& out)
{
    const auto [p, ec] = std::from_chars(a, a + std::strlen(a), out);
    return ec == std::errc{} and p == a + std::strlen(a);
}

struct LegStats { int n = 0, published = 0, ge06 = 0; double sum = 0, min = 1; std::int64_t t0 = 0, t1 = 0; };
void print_leg(const LegStats& s, int k)
{
    if (s.n == 0) return;
    std::print("leg {}: t {:.1f}–{:.1f} s, {} frames: p_view mean {:.3f} min {:.3f}, ≥ 0.6 on {} ({:.0f} %), published (≥ 0.5) {}\n", k,
               s.t0 * 1e-3, s.t1 * 1e-3, s.n, s.sum / s.n, s.min, s.ge06, 100.0 * s.ge06 / s.n, s.published);
}

int chain(const std::vector<rc::door::rec::Frame>& frames, double scale, bool freeze, double from_s, double to_s, int every,
          double win_a, double win_b, double width_override)
{
    using rc::interaction::RelPose;
    const std::int64_t t0 = frames.empty() ? 0 : frames.front().t_ms;
    std::print("{:>8} {:>5} | {:>7} {:>7} {:>7} | {:>6} {:>6} | {:>6} = {:>5} x {:>5} x {:>5} | {:>5} {:>6} | {:>6}\n", "t_ms", "rays", "x", "y",
               "yaw", "e", "s", "pl", "open", "door", "walls", "pc", "pv", "pv_rec");
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
            const float s_ = float(scale);
            rc::interaction::predict_jacobians(x.x, s_ * side, s_ * adv, s_ * dth, J, G);
            RelPose p;
            p.x = rc::interaction::move_mean(x.x, s_ * side, s_ * adv, s_ * dth);
            p.P = J * x.P * J.transpose() + s_ * s_ * Q;
            p.stamp_ms = b.stamp_ms;
            x = p;
            rc::door::carry_held_mask(hold, s_ * side, s_ * adv, s_ * dth);
        }
        rc::door::AnchorStaging st;
        st.replay = f.version != 1;
        const float wk = width_override > 0 ? float(width_override) : freeze ? w : f.width, lo = freeze ? u0 : f.wall_u_min, hi = freeze ? u1 : f.wall_u_max;
        auto [post, ev] = rc::door::update_door_anchor(x, wk, f.thickness, lo, hi, f.lidar_origin,
                                                       f.version == 1 ? f.rays_v1 : std::vector<Eigen::Vector3f>{}, f.mask, f.cam,
                                                       f.pi_open, f.leaf, f.floor_below, &st);
        if (not f.mask.empty()) rc::door::hold_mask_frame(hold, f.mask, f.t_ms);
        ev.p_cam = rc::door::held_camera_view(hold, f.t_ms, rc::interaction::kAnchorStaleMs, f.cam, wk, post);
        ev.p_view = rc::door::combine_view(ev.p_lidar, ev.p_cam);
        x = post;
        ++leg.n; leg.sum += ev.p_view; leg.min = std::min(leg.min, double(ev.p_view));
        leg.published += ev.p_view >= 0.5f ? 1 : 0; leg.ge06 += ev.p_view >= 0.6f ? 1 : 0; leg.t1 = f.t_ms - t0;
        if (t >= win_a and t <= win_b)
        {
            if (win.n == 0) win.t0 = f.t_ms - t0;
            ++win.n; win.sum += ev.p_view; win.min = std::min(win.min, double(ev.p_view));
            win.published += ev.p_view >= 0.5f ? 1 : 0; win.ge06 += ev.p_view >= 0.6f ? 1 : 0; win.t1 = f.t_ms - t0;
        }
        if (k % std::size_t(every) != 0) continue;
        const float cz = std::cos(post.x.z()), sz = std::sin(post.x.z());
        std::print("{:8d} {:5d} | {:+7.3f} {:+7.3f} {:+7.3f} | {:+6.2f} {:+6.2f} | {:6.3f} = {:5.3f} x {:5.3f} x {:5.3f} | {:5.3f} {:6.3f} | {:6.3f}\n",
                   f.t_ms - t0, f.version == 1 ? std::uint32_t(f.rays_v1.size()) : f.staging.prior.n_pre, post.x.x(), post.x.y(), post.x.z(),
                   -(cz * post.x.x() + sz * post.x.y()), -(-sz * post.x.x() + cz * post.x.y()), ev.p_lidar, ev.lidar_open, ev.lidar_door,
                   ev.lidar_wall, ev.p_cam, ev.p_view, f.p_view);
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
    double scale = 1.0, from_s = -1e9, to_s = 1e9, win_a = 1e9, win_b = -1e9, width_override = 0;
    for (int i = 1; i < argc; ++i)
    {
        double v = 0;
        if (std::strcmp(argv[i], "--every") == 0 and i + 1 < argc) { if (num(argv[++i], v) and v >= 1) every = int(v); }
        else if (std::strcmp(argv[i], "--chain") == 0) chained = true;
        else if (std::strcmp(argv[i], "--freeze") == 0) freeze = true;
        else if (std::strcmp(argv[i], "--time-scale") == 0 and i + 1 < argc) { if (num(argv[++i], v) and v > 0) scale = v; }
        else if (std::strcmp(argv[i], "--from") == 0 and i + 1 < argc) { if (num(argv[++i], v)) from_s = v; }
        else if (std::strcmp(argv[i], "--to") == 0 and i + 1 < argc) { if (num(argv[++i], v)) to_s = v; }
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
        return chain(frames, scale, freeze, from_s, to_s, every, win_a, win_b, width_override);
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
