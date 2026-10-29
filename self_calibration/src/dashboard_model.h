/*
 * dashboard_model.h -- the PURE (no Qt, no DSR) aggregation behind the self_calibration dashboard window.
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
 *    The `>` in classify() is a DISPLAY-ONLY LABEL on an already-selected pair, NOT a threshold: it gates nothing, feeds
 *    nothing back, and a tie simply reads "yaw".
 *  - ONE CLOCK: the wall clock. Pair stamps are the HELIOS source clock (sim time may not run at wall rate), so
 *    each fragment is converted ONCE, when it is taken, with the helios->wall offset current at that moment
 *    (offset = helios_stamp - wall at the latest sweep) and never touched again: a later offset change cannot tear
 *    it. Stall samples are wall-clock already. A backward wall-clock step resets the timeline (timeline_resets()).
 *  - Counters: total (cumulative) and a rolling last-60 s count taken from the fragment list.
 *  - Timeline pruning keeps `window_ms` (5 min) behind the newest wall time seen (a compute budget for the display).
 *  - Stream rows: StreamRates turns cumulative arrival totals into rates over >= 1 s windows and a status
 *    (waiting = never delivered, stalled, live) per stream, media/IMU and odometry/pose/polygon alike.
 */
#pragma once

#include "../../common/sensor_calib/trajectory_harvester.h"

#include <array>
#include <cstdint>
#include <deque>
#include <map>
#include <string>
#include <vector>

namespace sc::dash
{

enum class Label { yaw, lever };

struct Fragment { std::int64_t t0_ms = 0, t1_ms = 0; Label label = Label::yaw; };
struct StallInterval { std::int64_t t0_ms = 0, t1_ms = 0; };

enum class StreamStatus { waiting, live, stalled };
struct StreamSample { std::string key; std::uint64_t total = 0; bool ever_seen = false, stalled = false, show_rate = true; };
struct StreamRow { std::string key; double rate_hz = 0.0; StreamStatus status = StreamStatus::waiting; bool show_rate = true; };

/// Rates from cumulative totals over windows of >= 1 s (between windows the last rate is held).
class StreamRates
{
public:
    [[nodiscard]] std::vector<StreamRow> update(std::int64_t wall_ms, const std::vector<StreamSample>& samples);
private:
    std::int64_t window_start_ms_ = -1;
    std::map<std::string, std::uint64_t> totals_;
    std::map<std::string, double> rates_;
};

struct Priors { double var_xy = 0.05 * 0.05, var_yaw = 0.0175 * 0.0175; };

/// Label of one pair from its information matrix (see header).
[[nodiscard]] Label classify(const Eigen::Matrix3d& info, const Priors& pr);

class DashboardModel
{
public:
    explicit DashboardModel(Priors pr = {}, std::int64_t window_ms = 5 * 60 * 1000) : pr_(pr), window_ms_(window_ms) {}

    // --- feeds (main thread) ---
    void add_pair(const calib::Pair& p, std::int64_t wall_ms);   ///< a selected pair taken at wall_ms: converted once to the wall clock
    void note_keyframe(bool captured, bool stalled);             ///< result of harvester.add_keyframe and the verdict passed to it
    void note_lidar_stamp(std::int64_t stamp_ms, std::int64_t wall_ms);
    void sample_stall(std::int64_t wall_ms, bool stalled);       ///< each tick: any stream (helios included) stalled?

    // --- reads ---
    [[nodiscard]] long keyframes() const { return kf_; }
    [[nodiscard]] long dropped_stalled() const { return kf_dropped_; }
    [[nodiscard]] long pairs_total(Label l) const { return total_[idx(l)]; }
    /// Fragments (any label / one label) whose end lies in the last 60 s; independent of the harvester's log period.
    [[nodiscard]] long pairs_recent(Label l) const;
    [[nodiscard]] long pairs_recent() const { return pairs_recent(Label::yaw) + pairs_recent(Label::lever); }
    [[nodiscard]] static constexpr std::int64_t recent_ms() { return 60 * 1000; }
    [[nodiscard]] long timeline_resets() const { return resets_; }
    [[nodiscard]] long pairs_total() const { return total_[0] + total_[1]; }
    /// data precision / prior precision for (dx, dy, psi), cumulative over the run.
    [[nodiscard]] std::array<double, 3> ratios() const;
    [[nodiscard]] const std::deque<Fragment>& fragments() const { return frags_; }
    [[nodiscard]] const std::deque<StallInterval>& stalls() const { return stalls_; }   ///< includes the open one (ends at newest sample)
    /// Newest wall time the model knows (end of the timeline axis).
    [[nodiscard]] std::int64_t now_ms() const { return newest_ms_; }
    [[nodiscard]] std::int64_t window_ms() const { return window_ms_; }

private:
    static constexpr std::size_t idx(Label l) { return l == Label::yaw ? 0 : 1; }
    void advance(std::int64_t wall_ms);
    void prune();

    Priors pr_;
    std::int64_t window_ms_;
    long kf_ = 0, kf_dropped_ = 0;
    long total_[2] = {0, 0}, resets_ = 0;
    Eigen::Vector3d info_diag_ = Eigen::Vector3d::Zero();
    std::deque<Fragment> frags_;
    std::deque<StallInterval> stalls_;
    bool stall_open_ = false;
    bool have_offset_ = false;
    std::int64_t offset_ms_ = 0;      // helios stamp - wall
    std::int64_t newest_ms_ = 0;
};

}   // namespace sc::dash
