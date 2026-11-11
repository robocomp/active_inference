/*
 *    Copyright (C) 2026 by RoboLab at the University of Extremadura
 *    This file is part of RoboComp — see room_scene_graph.h.
 */
// ── THE POSE BETWEEN TWO CORRECTIONS (owner, 2026-11-11; Shadow spec "pose_tiempo_real" §4.1/§4.5) ─────────────────
// PosePropagator integrates from the last correction (the BASE): heading with the gyro (trapezoid between samples), the
// wheels where there is no gyro and where the wheels say the robot is stopped (ZUPT); translation with the wheels along
// the segment's MID heading. Each channel carries the calibration the motion prior APPLIED (UpdateResult::applied_motion,
// filled from MotionCalibration's acting accessors), with the prior's own split:
//   gyro   dpsi = k_w * omega_gyro * dt - b_w * dt              (scale, bias — gyro only)
//   wheels dpsi = k_ww * rot * dt + dk_wheel * forward           (radius/track + scrubbing, asymmetry — wheels only)
//   body   (dx, dy) = (k_lat * side * dt, k_v * adv * dt), rotated by th + 0.5*dpsi + yaw_offset
// (The Shadow spec applied k_w/b_w to the wheel channel and dk_wheel to the gyro one; this copy's prior keeps them
// apart — room_concept_odometry_window.cpp fuse_segment_heading — and the propagator follows the prior.)
// Body frame: +Y forward (adv), +X lateral (side).
//
// ONE CLOCK: every stamp handed in is a WALL stamp (IMU effective_ts_ms, odometry effective_ts_ms, the LiDAR-stamped
// correction t_c). In Webots the producer's rates are per SIM second, so integrating them over wall intervals
// under-counts by the sim/wall ratio (~6%) over the propagated span (tens of ms); the owner chose one clock over a
// sim-clock map inside the publisher. On the real robot both clocks are the same and nothing changes.
//
// plan_correction() is what PosePublisher::maybe_publish_corrected_pose does with it when a correction arrives: the
// SPREAD of the correction's jump over the blocks already written in (t_k, t_c), the reset to the new base, and the TAIL
// (every written block after t_c re-integrated from the correction, plus the newest IMU sample). Pure — no DSR, no Qt —
// so tools/pose_clamp_test.cpp runs the code the agent runs. Not thread-safe: its owner locks it.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <numbers>
#include <optional>
#include <vector>

namespace rc
{
    /// The motion-model calibration the optimiser APPLIED to the correction's prior (nominal where not acting).
    struct PropagatorCalib
    {
        float k_w = 1.f, b_w = 0.f;              ///< gyro scale, gyro bias (rad/s, physical sign: gyro reads w + b)
        float k_v = 1.f, k_lat = 1.f;            ///< wheel forward / lateral scale
        float yaw_offset = 0.f;                  ///< body->room yaw of the displacement (0 while it acts LiDAR-side)
        float dk_wheel = 0.f;                    ///< rad per metre forward (wheel asymmetry)
        float k_ww = 1.f;                        ///< wheel heading scale
    };

    struct PropagatorParams
    {
        std::int64_t odom_lag_ms  = 32;          ///< RoomConcept.OdomLagMs (toml 0)
        bool         zupt         = true;        ///< RoomConcept.ZuptEnabled
        float        zupt_rot_eps = 0.002f;      ///< rad/s, RoomConcept.ZuptWheelRotEps (the prior's own ZUPT condition)
        float        zupt_lin_eps = 0.001f;      ///< m/s,   RoomConcept.ZuptWheelLinEps
        std::int64_t keep_ms      = 2000;        ///< history kept behind the newest sample
    };

    class PosePropagator
    {
    public:
        struct Pose { std::int64_t t_ms = 0; float x = 0.f, y = 0.f, th = 0.f; };

        explicit PosePropagator(PropagatorParams p = {}) : p_(p) {}
        void set_params(const PropagatorParams& p) { p_ = p; }

        void add_imu(std::int64_t t_ms, float gyro_z)
        {
            if (not imu_.empty() and t_ms <= imu_.back().t) return;
            imu_.push_back({t_ms, gyro_z});
            while (imu_.size() > 2 and imu_.front().t < t_ms - p_.keep_ms) imu_.pop_front();
        }
        /// The producer's stamp; the lag is applied here.
        void add_odometry(std::int64_t t_ms, float adv, float side, float rot)
        {
            const std::int64_t t = t_ms - p_.odom_lag_ms;
            if (not odom_.empty() and t <= odom_.back().t) return;
            odom_.push_back({t, adv, side, rot});
            while (odom_.size() > 2 and odom_.front().t < t - p_.keep_ms) odom_.pop_front();
        }
        /// A new correction at t_c: integrated straight away up to the newest IMU sample.
        void reset(std::int64_t t_c, float x, float y, float th, const PropagatorCalib& c)
        {
            base_ = Pose{t_c, x, y, th}; calib_ = c; run_ = base_;
            if (not imu_.empty() and imu_.back().t > t_c) advance(*run_, imu_.back().t);
        }
        [[nodiscard]] bool has_base() const { return base_.has_value(); }
        [[nodiscard]] std::int64_t base_ms() const { return base_ ? base_->t_ms : 0; }
        [[nodiscard]] std::optional<Pose> latest() const { return run_; }   ///< the last integrated pose
        /// Pose at t >= base. Advances from the last one asked for; an earlier t starts again from the base.
        std::optional<Pose> at(std::int64_t t)
        {
            if (not base_ or t < base_->t_ms) return std::nullopt;
            if (t < run_->t_ms) run_ = base_;
            advance(*run_, t);
            return run_;
        }
        /// Body twist now (calibrated), for the blocks' twist slots.
        [[nodiscard]] float omega() const
        {
            if (imu_.empty()) return odom_.empty() ? 0.f : odom_.back().rot * calib_.k_ww;
            return imu_.back().gz * calib_.k_w - calib_.b_w;
        }
        [[nodiscard]] float v_forward() const { return odom_.empty() ? 0.f : odom_.back().adv * calib_.k_v; }
        [[nodiscard]] float v_lateral() const { return odom_.empty() ? 0.f : odom_.back().side * calib_.k_lat; }

    private:
        struct Imu  { std::int64_t t; float gz; };
        struct Odom { std::int64_t t; float adv, side, rot; };

        [[nodiscard]] std::optional<double> gyro_at(std::int64_t t) const
        {
            if (imu_.empty() or t < imu_.front().t) return std::nullopt;
            if (t >= imu_.back().t) return imu_.back().gz;
            const auto it = std::ranges::upper_bound(imu_, t, {}, &Imu::t);
            const auto& a = *(it - 1); const auto& b = *it;
            return a.gz + (b.gz - a.gz) * (static_cast<double>(t - a.t) / static_cast<double>(b.t - a.t));
        }
        [[nodiscard]] const Odom* odom_at(std::int64_t t) const
        {
            if (odom_.empty()) return nullptr;
            const auto it = std::ranges::upper_bound(odom_, t, {}, &Odom::t);
            return it == odom_.begin() ? &odom_.front() : &*(it - 1);
        }
        [[nodiscard]] std::int64_t next_break(std::int64_t t, std::int64_t end) const
        {
            std::int64_t n = end;
            if (const auto it = std::ranges::upper_bound(imu_, t, {}, &Imu::t); it != imu_.end()) n = std::min(n, it->t);
            if (const auto it = std::ranges::upper_bound(odom_, t, {}, &Odom::t); it != odom_.end()) n = std::min(n, it->t);
            return n;
        }
        void advance(Pose& s, std::int64_t t) const
        {
            while (s.t_ms < t)
            {
                const std::int64_t a = s.t_ms, b = next_break(a, t);
                const float dt = static_cast<float>(b - a) * 1e-3f;
                const Odom* o = odom_at(a);
                const float adv = o ? o->adv : 0.f, side = o ? o->side : 0.f, rot = o ? o->rot : 0.f;
                const float dx = side * dt * calib_.k_lat, dy = adv * dt * calib_.k_v;
                const bool stopped = p_.zupt and std::abs(rot) < p_.zupt_rot_eps
                                     and std::abs(adv) < p_.zupt_lin_eps and std::abs(side) < p_.zupt_lin_eps;
                const auto w0 = gyro_at(a), w1 = gyro_at(b);
                const float dth = (not stopped and w0 and w1)
                                  ? static_cast<float>(0.5 * (*w0 + *w1)) * dt * calib_.k_w - calib_.b_w * dt
                                  : rot * dt * calib_.k_ww + calib_.dk_wheel * dy;
                const float thm = s.th + 0.5f * dth + calib_.yaw_offset;
                s.x += dx * std::cos(thm) - dy * std::sin(thm);
                s.y += dx * std::sin(thm) + dy * std::cos(thm);
                s.th += dth; s.t_ms = b;
            }
        }

        PropagatorParams p_; PropagatorCalib calib_;
        std::deque<Imu> imu_; std::deque<Odom> odom_;
        std::optional<Pose> base_, run_;
    };

    /// What one correction writes besides itself, in the same frame as the correction.
    struct CorrectionPlan
    {
        std::vector<PosePropagator::Pose> spread;   ///< written blocks in (t_k, t_c), each moved by alpha * Delta
        std::vector<PosePropagator::Pose> tail;     ///< written blocks after t_c + the newest IMU sample, from the correction
    };

    /// The correction at t_c, pose `c` (what is written), with calibration `calib`. `written` = stamps of the blocks
    /// already on the edge after earlier corrections (ascending). `may_spread` = spreading is on AND the frame is
    /// continuous across (t_k, t_c] (no relocalisation, no re-anchor of the estimator's frame); `max_gap_ms` the
    /// longest t_c - t_k over which the old base still describes the motion (RoomConcept.PoseClampMaxDt).
    ///   SPREAD: with the OLD base (t_k), before the reset. Delta = correction - prediction at t_c; each written block
    ///           in (t_k, t_c) receives alpha * Delta, alpha = (t - t_k) / (t_c - t_k), so the edge has no step at t_c.
    ///   TAIL:   after the reset, every written block after t_c re-integrated from the correction, then the newest IMU
    ///           sample if it is newer still. The propagator is left at the newest sample for the next tick.
    inline CorrectionPlan plan_correction(PosePropagator& prop, const std::deque<std::int64_t>& written,
                                          const PosePropagator::Pose& c, const PropagatorCalib& calib,
                                          bool may_spread, std::int64_t max_gap_ms)
    {
        CorrectionPlan out;
        const std::int64_t t_c = c.t_ms, t_k = prop.base_ms();
        if (may_spread and prop.has_base() and t_c > t_k and t_c - t_k <= max_gap_ms)
        {
            for (const auto t : written)                      // ascending: at() advances without restarting
                if (t > t_k and t < t_c)
                    if (const auto s = prop.at(t)) out.spread.push_back(*s);
            if (const auto pc = prop.at(t_c))
            {
                const float dx = c.x - pc->x, dy = c.y - pc->y;
                const float dth = std::remainder(c.th - pc->th, 2.f * std::numbers::pi_v<float>);
                for (auto& s : out.spread)
                {
                    const float a = static_cast<float>(s.t_ms - t_k) / static_cast<float>(t_c - t_k);
                    s.x += a * dx; s.y += a * dy; s.th += a * dth;
                }
            }
            else out.spread.clear();
        }
        prop.reset(t_c, c.x, c.y, c.th, calib);
        const auto newest = prop.latest();
        for (const auto t : written)
            if (t > t_c)
                if (const auto s = prop.at(t)) out.tail.push_back(*s);
        if (newest and newest->t_ms > t_c and (out.tail.empty() or out.tail.back().t_ms < newest->t_ms))
            out.tail.push_back(*newest);
        if (newest) prop.at(newest->t_ms);
        return out;
    }

    /// The edge after a rewrite from its earliest block: what was written before it stays, everything from it on is
    /// exactly the blocks just written. `written` keeps only the non-correction stamps (the corrections themselves
    /// are bases, never spread or re-integrated), ascending, and forgets what is older than `keep_ms` behind the newest.
    inline void record_rewrite(std::deque<std::int64_t>& written, std::int64_t t_c, const CorrectionPlan& plan,
                               std::int64_t keep_ms = 1000)
    {
        const std::int64_t from = plan.spread.empty() ? t_c : std::min(t_c, plan.spread.front().t_ms);
        std::erase_if(written, [from](std::int64_t t) { return t >= from; });
        for (const auto& s : plan.spread) written.push_back(s.t_ms);
        for (const auto& s : plan.tail)   written.push_back(s.t_ms);
        while (written.size() > 1 and written.front() < written.back() - keep_ms) written.pop_front();
    }
}   // namespace rc
