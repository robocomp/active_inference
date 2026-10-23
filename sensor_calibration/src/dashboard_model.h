/*
 * dashboard_model.h -- the PURE (no Qt, no DSR) aggregation behind the sensor_calibration dashboard window.
 *
 * DISPLAY ONLY. Nothing here feeds back into the harvester or any estimate: the agent hands over copies of what it
 * already has (the selected Pairs, keyframe verdicts, stall samples) and the widgets read the result.
 *
 *  - Label of a fragment (yaw | lever): which DoF group gains the larger PRIOR-SCALED information from the pair,
 *        lever:  (I_dx,dx + I_dy,dy) * var_xy        yaw:  I_psi,psi * var_yaw
 *    (var = prior sigma^2, so each term is "data precision / prior precision" of that group). A label for the
 *    reader, never a gate: the pair has already been selected and its information already counted.
 *  - Per-DoF ratio: accumulated data precision (sum of the pairs' info diagonal) / prior precision = info_diag * var.
 *    1 means the data are worth exactly the prior.
 *  - Time axis. Fragment stamps are the HELIOS source clock; stall samples are taken on the wall clock at each tick.
 *    note_lidar_stamp() keeps offset = helios_stamp - wall at the latest sweep, and stall samples are mapped
 *    through it, so both live on the helios clock without assuming it is the wall clock.
 *  - Timeline pruning keeps `window_ms` (5 min) behind the newest time seen (a compute budget for the display).
 */
#pragma once

#include "../../common/sensor_calib/trajectory_harvester.h"

#include <array>
#include <cstdint>
#include <deque>
#include <vector>

namespace sc::dash
{

enum class Label { yaw, lever };

struct Fragment { std::int64_t t0_ms = 0, t1_ms = 0; Label label = Label::yaw; };
struct StallInterval { std::int64_t t0_ms = 0, t1_ms = 0; };

struct Priors { double var_xy = 0.05 * 0.05, var_yaw = 0.0175 * 0.0175; };

/// Label of one pair from its information matrix (see header).
[[nodiscard]] Label classify(const Eigen::Matrix3d& info, const Priors& pr);

class DashboardModel
{
public:
    explicit DashboardModel(Priors pr = {}, std::int64_t window_ms = 5 * 60 * 1000) : pr_(pr), window_ms_(window_ms) {}

    // --- feeds (main thread) ---
    void add_pair(const calib::Pair& p);                         ///< a selected pair: counters, ratios, timeline
    void note_keyframe(bool captured, bool stalled);             ///< result of harvester.add_keyframe and the verdict passed to it
    void note_lidar_stamp(std::int64_t stamp_ms, std::int64_t wall_ms);
    void sample_stall(std::int64_t wall_ms, bool stalled);       ///< each tick: any stream (helios included) stalled?
    void close_episode();                                        ///< the harvester closed its episode: per-episode counters restart

    // --- reads ---
    [[nodiscard]] long keyframes() const { return kf_; }
    [[nodiscard]] long dropped_stalled() const { return kf_dropped_; }
    [[nodiscard]] long pairs_total(Label l) const { return total_[idx(l)]; }
    [[nodiscard]] long pairs_episode(Label l) const { return episode_[idx(l)]; }
    [[nodiscard]] long pairs_total() const { return total_[0] + total_[1]; }
    [[nodiscard]] long pairs_episode() const { return episode_[0] + episode_[1]; }
    /// data precision / prior precision for (dx, dy, psi), cumulative over the run.
    [[nodiscard]] std::array<double, 3> ratios() const;
    [[nodiscard]] const std::deque<Fragment>& fragments() const { return frags_; }
    [[nodiscard]] const std::deque<StallInterval>& stalls() const { return stalls_; }   ///< includes the open one (ends at newest sample)
    /// Newest time on the helios clock the model knows (end of the timeline axis).
    [[nodiscard]] std::int64_t now_ms() const { return newest_ms_; }
    [[nodiscard]] std::int64_t window_ms() const { return window_ms_; }

private:
    static constexpr std::size_t idx(Label l) { return l == Label::yaw ? 0 : 1; }
    void advance(std::int64_t t_ms);
    void prune();

    Priors pr_;
    std::int64_t window_ms_;
    long kf_ = 0, kf_dropped_ = 0;
    long total_[2] = {0, 0}, episode_[2] = {0, 0};
    Eigen::Vector3d info_diag_ = Eigen::Vector3d::Zero();
    std::deque<Fragment> frags_;
    std::deque<StallInterval> stalls_;
    bool stall_open_ = false;
    bool have_offset_ = false;
    std::int64_t offset_ms_ = 0;      // helios stamp - wall
    std::int64_t newest_ms_ = 0;
};

}   // namespace sc::dash
