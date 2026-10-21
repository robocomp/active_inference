// anchor_replay — the passage anchor re-run from the DoorCycleInput tape (door_inputs.bin) and the decision log
// (door_decisions.bin) (Task D-B fix round). The anchor slice is not in the DoorCore yet, so this is the SMALLEST anchor
// replay the tapes allow: the caller's half of step_passage_anchor (specificworker_anchor.cpp) rebuilt from the recorded
// inputs, the update itself (door_anchor_update.h) and the lost check (door_anchor_open.h) run as they run live.
//
// ★Why it exists: the anchor frame record (etc/anchor_frames.bin) is TRUNCATED at every agent start-up; rtD1h's (the alias
//  of leg 5) was overwritten by the next run within the hour. door_inputs.bin is collected with the run.
//
// Reconstructed (stated, not recorded — the door's belief is the agent's own state, not on the input tape):
//   · the anchor's birth pose: the AnchorHung decision row (x, y, yaw), P0 in anchor axes = the log's σ (--p0 lat,dep,yaw);
//   · the door's geometry and leaf: --width (the wall gap; clear width = width − 2·FrameCasingM 0.06), --thickness, the
//     wall run ±1.5·width (the agent's default for an unknown run), the leaf from the log's "carries its leaf" line
//     (--leaf hinge_u,dir_u,swing_y,phi_deg,phi_sigma_deg, p_hinge = p_swing = the MAP's probability, 1 here);
//   · the motion between cycles: the localised robot pose (room←robot, ts 0) while the room frame is unchanged, else the
//     body twist × the wall interval; its variance: robot_current_speed_variance × dt² (the live predict's).
// Recorded, used as live: the bpearl robot-frame sweep and its origin (brought to now by the twist, to_now), the mask's
// door slices in the zed frame through robot←zed, the zed's field of view.
// Scored against the Webots truth (--gt gt_error.csv, DOOR_1 at (−4.77, 9.425), the wall along world x), as
// replay_anchor_frames does: lateral and yaw error of the tracker and of the published pose.
//
//   Build + run (from door_concept/):
//     Q=/usr/include/x86_64-linux-gnu/qt5; g++ -std=c++23 -O2 -fPIC -I/usr/include/eigen3 -Isrc -I../common -I$Q -I$Q/QtCore \
//       tools/anchor_replay.cpp src/core/door_cycle_io.cpp -o build/anchor_replay -lzstd -lfastdds -lfastcdr -lQt5Core
//     ./build/anchor_replay INPUTS DECISIONS --hang N [--frames K] [--gt gt_error.csv] [--no-lost] [--width W]
//         [--thickness T] [--p0 a,b,c] [--leaf hu,du,sy,phi_deg,sig_deg]
//   Prints one row per measured frame and a summary line "SUMMARY hang N frames F lat_med … yaw_max …".
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <format>
#include <fstream>
#include <optional>
#include <print>
#include <sstream>
#include <string>
#include <vector>

#include "core/door_cycle_io.h"
#include "decision_log.h"
#include "door_anchor_open.h"
#include "door_anchor_update.h"

namespace
{
using rc::interaction::RelPose;
bool num(const std::string& s, double& v)
{
    const auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
    return ec == std::errc{} and p == s.data() + s.size();
}
std::vector<double> nums(const std::string& s)
{
    std::vector<double> out; std::stringstream ss(s); std::string tok;
    while (std::getline(ss, tok, ',')) { double v = 0; if (num(tok, v)) out.push_back(v); }
    return out;
}
// p_now = the point a capture `age_s` old saw, in the robot frame of NOW (specificworker_anchor.cpp to_now).
Eigen::Vector3f to_now(const Eigen::Vector3f& p, const rc::door::core::Twist& tw, float age_s)
{
    const Eigen::Vector3f q = rc::interaction::move_mean(Eigen::Vector3f(p.x(), p.y(), 0.f), tw.linear.x() * age_s,
                                                         tw.linear.y() * age_s, tw.angular.z() * age_s);
    return {q.x(), q.y(), p.z()};
}
struct GtRow { std::int64_t t; double x, y, th; };
std::vector<GtRow> read_gt(const std::string& path)
{
    std::vector<GtRow> out; std::ifstream f(path); std::string line;
    if (not std::getline(f, line)) return out;
    std::vector<std::string> head; { std::stringstream ss(line); std::string c; while (std::getline(ss, c, ',')) head.push_back(c); }
    auto col = [&](const char* n) { return int(std::find(head.begin(), head.end(), n) - head.begin()); };
    const int it = col("ts_ms"), ix = col("gt_x_world"), iy = col("gt_y_world"), ith = col("gt_theta_raw");
    while (std::getline(f, line))
    {
        std::vector<std::string> c; std::stringstream ss(line); std::string v; while (std::getline(ss, v, ',')) c.push_back(v);
        if (int(c.size()) <= std::max({it, ix, iy, ith})) continue;
        double t = 0, x = 0, y = 0, raw = 0;
        if (num(c[it], t) and num(c[ix], x) and num(c[iy], y) and num(c[ith], raw)) out.push_back({std::int64_t(t), x, y, -raw - M_PI / 2});
    }
    return out;
}
// The truth's error of an anchor pose (replay_anchor_frames' gt_error): lateral along the true wall, and yaw.
std::optional<std::pair<double, double>> gt_err(const std::vector<GtRow>& gt, std::int64_t t, const RelPose& p)
{
    auto it = std::lower_bound(gt.begin(), gt.end(), t, [](const GtRow& r, std::int64_t v) { return r.t < v; });
    if (it == gt.end() or it == gt.begin()) return std::nullopt;
    const GtRow& g1 = *it; const GtRow& g0 = *std::prev(it);
    if (g1.t - g0.t > 200 or g1.t == g0.t) return std::nullopt;
    const double a_ = double(t - g0.t) / double(g1.t - g0.t);
    const double gx = g0.x + a_ * (g1.x - g0.x), gy = g0.y + a_ * (g1.y - g0.y), th = g0.th + a_ * std::remainder(g1.th - g0.th, 2 * M_PI);
    constexpr double kDoorX = -4.77, kDoorY = 9.425;
    const double c = std::cos(th), s = std::sin(th), dx = kDoorX - gx, dy = kDoorY - gy;
    const double tx = c * dx + s * dy, ty = -s * dx + c * dy, tz = std::remainder(-th, 2 * M_PI);
    double lat = (p.x.x() - tx) * std::cos(tz) + (p.x.y() - ty) * std::sin(tz);
    double ye = std::remainder(double(p.x.z()) - tz, 2 * M_PI);
    if (std::abs(ye) > M_PI / 2) { ye = std::remainder(ye + M_PI, 2 * M_PI); lat = -lat; }
    return std::pair{lat, ye};
}
double med(std::vector<double> v) { if (v.empty()) return 0; std::ranges::nth_element(v, v.begin() + std::ptrdiff_t(v.size() / 2)); return v[v.size() / 2]; }
}  // namespace

int main(int argc, char** argv)
{
    if (argc < 3) { std::print(stderr, "usage: anchor_replay INPUTS DECISIONS --hang N [options]\n"); return 2; }
    const std::string in_path = argv[1], dec_path = argv[2];
    int hang = 0, max_frames = 1 << 30;
    bool lost = true, no_mask = false;
    double width = 1.0, thickness = 0.10, casing = 0.06;
    std::vector<double> p0{0.309, 0.203, 0.101}, leafv;
    std::string gt_path;
    for (int i = 3; i < argc; ++i)
    {
        const std::string a = argv[i];
        const auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        double v = 0;
        if (a == "--hang" and num(next(), v)) hang = int(v);
        else if (a == "--frames") { if (num(next(), v)) max_frames = int(v); }
        else if (a == "--no-lost") lost = false;
        else if (a == "--no-mask") no_mask = true;   // diagnosis: the camera's mask left out of the update
        else if (a == "--width") { if (num(next(), v)) width = v; }
        else if (a == "--thickness") { if (num(next(), v)) thickness = v; }
        else if (a == "--p0") p0 = nums(next());
        else if (a == "--leaf") leafv = nums(next());
        else if (a == "--gt") gt_path = next();
    }
    using K = rc::door::DoorDecisionKind;
    const auto decs = rc::door::DoorDecisionLog::read_all(dec_path);
    if (not decs.has_value()) { std::print(stderr, "cannot read {}\n", dec_path); return 2; }
    std::vector<rc::door::DoorDecision> hangs;
    for (const auto& d : *decs) if (d.kind == static_cast<std::uint16_t>(K::AnchorHung)) hangs.push_back(d);
    if (hang < 0 or hang >= int(hangs.size())) { std::print(stderr, "{} hangs on the tape\n", hangs.size()); return 2; }
    const auto& H = hangs[std::size_t(hang)];
    // The passage ends at the anchor's removal or its first retry/re-init row — after that the live anchor was re-seeded
    // from the room, which this replay does not reproduce.
    std::uint64_t end_cycle = ~0ull;
    // (Any door id: a room switch relinks the anchor to the next room's door instance.)
    for (const auto& d : *decs)
        if (d.cycle > H.cycle
            and (d.kind == static_cast<std::uint16_t>(K::AnchorRemoved) or d.kind == static_cast<std::uint16_t>(K::AnchorHung)
                 or (d.kind == static_cast<std::uint16_t>(K::AnchorState)
                     and (d.note.starts_with("retry") or d.note.starts_with("re-initialised")))))
        { end_cycle = d.cycle; break; }
    const auto gt = gt_path.empty() ? std::vector<GtRow>{} : read_gt(gt_path);
    std::print("hang {} at cycle {} (end cycle {}): born ({:+.3f},{:+.3f},{:+.3f}), P0 σ ({:.3f},{:.3f},{:.3f}), width {:.2f} t {:.2f}, lost check {}\n",
               hang, H.cycle, end_cycle == ~0ull ? -1 : std::int64_t(end_cycle), H.a, H.b, H.c, p0[0], p0[1], p0[2], width, thickness, lost ? "ON" : "off");

    std::ifstream f(in_path, std::ios::binary);
    std::uint32_t version = 0;
    if (not rc::door::core::read_header(f, rc::door::core::RecordingKind::Input, &version).has_value()) { std::print(stderr, "{}: not a DCIN tape\n", in_path); return 2; }

    RelPose pose;
    pose.x = Eigen::Vector3f(float(H.a), float(H.b), float(H.c));
    pose.P = rc::door::anchor_axes_to_robot(Eigen::Vector3f(float(p0[0] * p0[0]), float(p0[1] * p0[1]), float(p0[2] * p0[2])).asDiagonal(), pose.x.z());
    const RelPose room_seed = pose;   // the room init at the hang, carried in the WORLD by the truth (below) for the lost check
    rc::door::AnchorBias bias;
    rc::door::AnchorLostMode mode;
    std::optional<Eigen::Vector3d> open_world;   // the room init's opening in the world, at the hang (needs --gt)
    const float clear = rc::door::anchor_clear_width(float(width), float(casing));
    rc::door::LeafPrior leaf;
    if (leafv.size() >= 5)
    {
        leaf.known = true; leaf.hinge_u = float(leafv[0]); leaf.dir_u = float(leafv[1]); leaf.swing_y = float(leafv[2]);
        leaf.len = clear; leaf.phi_max = 2.0944f; leaf.phi_known = true;
        leaf.phi = float(leafv[3] * M_PI / 180.0); leaf.phi_sigma = float(leafv[4] * M_PI / 180.0);
        leaf.p_hinge = 1.f; leaf.p_swing = 1.f;
    }
    std::int64_t last_lidar = 0; int last_mask = -1;
    std::optional<Eigen::Matrix4d> prev_room_T_robot; std::optional<std::uint64_t> prev_room;
    std::int64_t prev_t = 0;
    std::vector<double> el_t, ey_t, el_o, ey_o, zl, zy;   // zl/zy: the tracker's |error|/σ on frames whose fit was taken (calibration)
    int frames = 0;
    bool first_printed = false;
    std::int64_t t_hang = 0;
    while (frames < max_frames)
    {
        auto in = rc::door::core::read_input(f, version);
        if (not in.has_value() or in->cycle >= end_cycle) break;
        if (in->cycle < H.cycle) continue;
        const auto& rb = in->robot;
        const std::int64_t now = in->t_wall_ms;
        // ── PREDICT: the robot's motion since the last cycle, in the robot frame of then ──────────────────────────────
        if (prev_t > 0 and now > prev_t)
        {
            const float dt = 1e-3f * float(now - prev_t);
            float side = 0, adv = 0, dth = 0;
            const auto cur_room = in->rooms.graph_current_room;
            if (rb.room_T_robot.has_value() and prev_room_T_robot.has_value() and cur_room == prev_room)
            {
                const Eigen::Matrix4d D = prev_room_T_robot->inverse() * rb.room_T_robot.value();
                side = float(D(0, 3)); adv = float(D(1, 3)); dth = float(std::atan2(D(1, 0), D(0, 0)));
            }
            else if (rb.twist.has_value()) { side = rb.twist->linear.x() * dt; adv = rb.twist->linear.y() * dt; dth = rb.twist->angular.z() * dt; }
            Eigen::Matrix3f J, G;
            rc::interaction::predict_jacobians(pose.x, side, adv, dth, J, G);
            Eigen::Vector3f var = Eigen::Vector3f::Zero();
            if (rb.speed_variance.size() >= 3) var = {rb.speed_variance[0], rb.speed_variance[1], rb.speed_variance[2]};
            const Eigen::Vector3f x_before = pose.x;
            pose.x = rc::interaction::move_mean(pose.x, side, adv, dth);
            pose.P = J * pose.P * J.transpose() + G * var.asDiagonal() * G.transpose() * (dt * dt);
            rc::door::transport_bias(bias, x_before, side, adv, dth, float(width));
        }
        prev_t = now;
        if (rb.room_T_robot.has_value()) prev_room_T_robot = rb.room_T_robot;
        prev_room = in->rooms.graph_current_room;
        // ── the room initialisation this cycle (lost check): the hang's room init fixed in the WORLD, brought into the robot
        //    frame of now by the truth (replay_anchor_frames --lost-check's convention: the live room-frame error at the hang,
        //    no later localisation noise) ──────────────────────────────────────────────────────────────────────────────
        std::optional<RelPose> room_now;
        if (lost and not gt.empty())
        {
            auto it = std::lower_bound(gt.begin(), gt.end(), now, [](const GtRow& r, std::int64_t v) { return r.t < v; });
            if (it != gt.end())
            {
                const double c = std::cos(it->th), s = std::sin(it->th);
                if (not open_world.has_value())
                    open_world = Eigen::Vector3d(it->x + c * room_seed.x.x() - s * room_seed.x.y(), it->y + s * room_seed.x.x() + c * room_seed.x.y(),
                                                 std::remainder(it->th + room_seed.x.z(), 2 * M_PI));
                const double dx = open_world->x() - it->x, dy = open_world->y() - it->y;
                RelPose r;
                r.x = Eigen::Vector3f(float(c * dx + s * dy), float(-s * dx + c * dy), float(std::remainder(open_world->z() - it->th, 2 * M_PI)));
                r.P = rc::door::anchor_axes_to_robot(Eigen::Vector3f(float(p0[0] * p0[0]), float(p0[1] * p0[1]), float(p0[2] * p0[2])).asDiagonal(), r.x.z());
                room_now = r;
            }
        }
        // ── the lost check's transition and IMM mixing at a capture (an update), BEFORE the rays are selected (as live) ─
        float w_mix = 0.f;
        const bool capture = (in->bpearl.has_value() and in->bpearl->robot_fresh and in->bpearl->robot_stamp_ms > last_lidar)
                             or (in->masks.has_value() and in->masks->valid and in->masks->frame_id != last_mask);
        if (lost and capture)
            w_mix = rc::door::lost_mode_predict(mode, pose, &bias, room_now.has_value() ? &room_now.value() : nullptr, clear);
        // ── UPDATE inputs: the bpearl sweep (robot frame, brought to now) and the mask's door slice ─────────────────────
        const rc::door::core::Twist tw = rb.twist.value_or(rc::door::core::Twist{});
        std::vector<Eigen::Vector3f> rays, mask_pts;
        Eigen::Vector3f origin = Eigen::Vector3f::Zero();
        if (in->bpearl.has_value() and in->bpearl->robot_fresh and in->bpearl->robot_stamp_ms > last_lidar)
        {
            last_lidar = in->bpearl->robot_stamp_ms;
            const float age = 1e-3f * float(now - last_lidar);
            origin = to_now(in->bpearl->origin_robot, tw, age);
            std::vector<Eigen::Vector3f> sweep;
            sweep.reserve(in->bpearl->pts_robot.size());
            for (const auto& p : in->bpearl->pts_robot) sweep.push_back(to_now(p, tw, age));
            rays = rc::door::select_anchor_rays(pose, float(width), float(thickness), 0.05f, 2.0f, origin, sweep, 2.0f);
        }
        rc::door::CameraFrustum cam;
        if (rb.robot_T_zed.has_value())
        {
            const Eigen::Matrix4d& M = rb.robot_T_zed.value();
            float half_fov = 0.f;
            if (in->camera.cam_rgb_width.has_value() and in->camera.cam_rgb_focalx.has_value() and *in->camera.cam_rgb_focalx > 0)
                half_fov = std::atan(0.5f * float(*in->camera.cam_rgb_width) / float(*in->camera.cam_rgb_focalx));
            else if (in->camera.cam_fov.has_value()) half_fov = 0.5f * *in->camera.cam_fov;
            if (half_fov > 0.f)
            {
                cam.known = true; cam.origin = M.block<2, 1>(0, 3).cast<float>(); cam.forward = M.block<2, 1>(0, 1).cast<float>().normalized();
                cam.half_fov = half_fov; cam.max_range = 8.f;
            }
            if (in->masks.has_value() and in->masks->valid and in->masks->frame_id != last_mask)
            {
                last_mask = in->masks->frame_id;
                const float age = in->masks->timestamp_ms > 0 ? 1e-3f * float(now - std::int64_t(in->masks->timestamp_ms)) : 0.f;
                float best = std::numeric_limits<float>::infinity();
                for (const auto& sl : in->masks->slices)
                {
                    if (sl.label != "door" or sl.source != 0 or not sl.has_depth or sl.support_end <= sl.support_begin
                        or sl.support_end > in->masks->support_points_cam.size()) continue;
                    std::vector<Eigen::Vector3f> pts; Eigen::Vector2f mean = Eigen::Vector2f::Zero();
                    for (std::size_t i = sl.support_begin; i < sl.support_end; ++i)
                    {
                        const auto& q = in->masks->support_points_cam[i];
                        const Eigen::Vector4d r = M * Eigen::Vector4d(q.x(), q.y(), q.z(), 1.0);
                        pts.push_back(to_now(Eigen::Vector3f(float(r.x()), float(r.y()), float(r.z())), tw, age));
                        mean += pts.back().head<2>();
                    }
                    mean /= float(pts.size());
                    if (const float d = (mean - pose.x.head<2>()).norm(); d < best) { best = d; mask_pts = std::move(pts); }
                }
            }
        }
        if (no_mask) mask_pts.clear();
        if (rays.empty() and mask_pts.size() < 2) continue;   // not a measured cycle
        const RelPose prior = pose;
        auto [post, ev] = rc::door::update_door_anchor(pose, float(width), float(thickness), -1.5f * float(width), 1.5f * float(width), origin,
                                                       rays, mask_pts, cam, 1.0f, leaf, 0.05f, nullptr, &bias);
        pose = post;
        RelPose out = pose;
        float pv_out = ev.p_view;
        if (lost) rc::door::lost_mode_view(mode, pose, ev.p_view, clear);
        if (room_now.has_value())
        {
            rc::door::lost_mode_posterior(mode, pose, room_now.value(), clear);
            out = rc::door::lost_mode_output(pose, room_now.value(), mode.p);
            const auto vr = rc::door::door_view(room_now.value(), float(width), float(thickness), -1.5f * float(width), 1.5f * float(width),
                                                origin, rays, mask_pts, cam, 1.0f, leaf, 0.05f);
            pv_out = rc::door::anchor_view_mixture(ev.p_view, vr.p_view, mode.p);
        }
        ++frames;
        const auto et = gt_err(gt, now, pose), eo = gt_err(gt, now, out);
        if (et.has_value()) { el_t.push_back(std::abs(et->first)); ey_t.push_back(std::abs(et->second) * 180 / M_PI); }
        if (et.has_value() and ev.p_fit > 0.99f)
        {
            const Eigen::Matrix3f Pa = rc::door::robot_to_anchor_axes(pose.P, pose.x.z());
            zl.push_back(std::abs(et->first) / std::sqrt(std::max(1e-12f, Pa(0, 0))));
            zy.push_back(std::abs(et->second) / std::sqrt(std::max(1e-12f, Pa(2, 2))));
        }
        if (eo.has_value()) { el_o.push_back(std::abs(eo->first)); ey_o.push_back(std::abs(eo->second) * 180 / M_PI); }
        if (not first_printed)
        {
            first_printed = true;
            std::print("FIRST cycle {} prior ({:+.3f},{:+.3f},{:+.3f}) → post ({:+.3f},{:+.3f},{:+.3f}) σ ({:.3f},{:.3f},{:.3f}) P(fit) {:.3f} (alias log-ratio {:+.1f}) p_view {:.2f} rays {} mask {}{}\n",
                       in->cycle, prior.x.x(), prior.x.y(), prior.x.z(), pose.x.x(), pose.x.y(), pose.x.z(), std::sqrt(pose.P(0, 0)),
                       std::sqrt(pose.P(1, 1)), std::sqrt(pose.P(2, 2)), ev.p_fit, ev.fit_llr, ev.p_view, rays.size(), mask_pts.size(),
                       et.has_value() ? std::format(" | GT err lat {:+.2f} yaw {:+.1f}°", et->first, et->second * 180 / M_PI) : std::string{});
        }
        if (t_hang == 0) t_hang = now;
        std::print("{:6d} t {:6.2f} | track ({:+.3f},{:+.3f},{:+.3f}) fit {:.3f} llr {:+.1f} pv {:.2f} | w {:.3f} q {:.3f} P(lost) {:.3f} out ({:+.3f},{:+.3f},{:+.3f}) pv_out {:.2f}{}\n",
                   in->cycle, 1e-3 * double(now - t_hang), pose.x.x(), pose.x.y(), pose.x.z(),
                   ev.p_fit, ev.fit_llr, ev.p_view, w_mix, mode.q(), mode.p, out.x.x(), out.x.y(), out.x.z(), pv_out,
                   et.has_value() ? std::format(" | GT track lat {:+.2f} yaw {:+.1f}° out lat {:+.2f} yaw {:+.1f}°", et->first,
                                                et->second * 180 / M_PI, eo->first, eo->second * 180 / M_PI) : std::string{});
    }
    std::print("SUMMARY hang {} frames {} tracker lat_med {:.3f} lat_max {:.3f} yaw_med {:.1f} yaw_max {:.1f} | published lat_med {:.3f} lat_max {:.3f} yaw_med {:.1f} yaw_max {:.1f}\n",
               hang, frames, med(el_t), el_t.empty() ? 0 : std::ranges::max(el_t), med(ey_t), ey_t.empty() ? 0 : std::ranges::max(ey_t),
               med(el_o), el_o.empty() ? 0 : std::ranges::max(el_o), med(ey_o), ey_o.empty() ? 0 : std::ranges::max(ey_o));
    if (not zl.empty())
    {
        const auto frac2 = [](const std::vector<double>& v) { return double(std::ranges::count_if(v, [](double z) { return z <= 2.0; })) / double(v.size()); };
        std::print("CALIB hang {} frames {} (fit taken): |lat err|/σ median {:.2f}, ≤2σ {:.0f} % | |yaw err|/σ median {:.2f}, ≤2σ {:.0f} %\n",
                   hang, zl.size(), med(zl), 100 * frac2(zl), med(zy), 100 * frac2(zy));
    }
    return 0;
}
