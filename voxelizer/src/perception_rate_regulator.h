#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>

namespace rc
{

// Homeostatic perception-rate regulator (decoupled from any voxel/grid code).
//
// Goal: hold compute() near a target rate by decimating the OPTIONAL perception
// model (human-pose) via AIMD with hysteresis. It is fed only the per-cycle compute
// cost and the processed-frame stamp, and derives everything else:
//   - processed_hz : EMA of the wall rate of processed (fresh-frame) cycles
//   - feed_hz      : the INPUT stream rate, estimated as 1000/(min frame_ts gap)
//                    over each control window (the fastest observed cadence)
//   - compute_ms   : EMA of the cycle cost
//
// Control philosophy (per the design discussion): react to COST + frame-drop, never
// to output Hz alone. If the INPUT feed itself is below target, decimation cannot
// help (we'd just process the same frames), so we HOLD. Only the pose model is an
// actuator — YOLO-seg is the core product (its masks feed the concept agents, and
// decimating it would republish stale masks for moving objects), so it is never
// decimated here; when pose is already at its cap and we're still behind, the
// warning is the signal to drop to a lighter seg model / lower input resolution.
struct RateRegulatorConfig
{
    float  target_hz        = 20.0f;
    int    pose_decim_min   = 1;     // floor = the user-configured HumanPose.decimation
    int    pose_decim_max   = 4;     // never skip more than this
    double control_period_s = 2.0;   // adapt at most this often (slow adaptation)
    double deadband_hz      = 2.0;   // hysteresis band around target (avoid thrash)
};

class PerceptionRateRegulator
{
public:
    void configure(const RateRegulatorConfig& c)
    {
        cfg_ = c;
        cfg_.pose_decim_max = std::max(c.pose_decim_min, c.pose_decim_max);
        pose_decim_ = cfg_.pose_decim_min;
    }

    // Call once per PROCESSED (fresh-frame) cycle.
    void update(double compute_ms, std::uint64_t frame_ts_ms)
    {
        using namespace std::chrono;
        const auto now = steady_clock::now();

        if (have_last_)
        {
            const double dt = duration<double, std::milli>(now - last_tp_).count();
            if (dt > 0.0) { const double hz = 1000.0 / dt; proc_hz_ = proc_hz_ > 0.0 ? 0.9 * proc_hz_ + 0.1 * hz : hz; }
        }
        last_tp_ = now;
        have_last_ = true;

        comp_ms_ = comp_ms_ > 0.0 ? 0.9 * comp_ms_ + 0.1 * compute_ms : compute_ms;

        // Feed-period estimate: the smallest source-stamp gap in this window ≈ true cadence.
        if (last_ts_ != 0 && frame_ts_ms > last_ts_)
        {
            const double gap = static_cast<double>(frame_ts_ms - last_ts_);
            if (gap > 0.0 && gap < win_min_gap_ms_) win_min_gap_ms_ = gap;
        }
        last_ts_ = frame_ts_ms;

        if (ctrl_tp_.time_since_epoch().count() == 0) ctrl_tp_ = now;
        if (duration<double>(now - ctrl_tp_).count() >= cfg_.control_period_s)
        {
            ctrl_tp_ = now;
            feed_hz_ = (win_min_gap_ms_ < 1e8 && win_min_gap_ms_ > 0.0) ? 1000.0 / win_min_gap_ms_ : 0.0;
            win_min_gap_ms_ = 1e9;   // reset the window
            control();
        }
    }

    [[nodiscard]] int    pose_decimation() const { return pose_decim_; }
    [[nodiscard]] double processed_hz()    const { return proc_hz_; }
    [[nodiscard]] double feed_hz()         const { return feed_hz_; }
    [[nodiscard]] double compute_ms()      const { return comp_ms_; }
    [[nodiscard]] bool   changed()         const { return changed_; }          // decim moved on the last control tick
    [[nodiscard]] bool   below_target()    const { return proc_hz_ > 0.0 && proc_hz_ < cfg_.target_hz - cfg_.deadband_hz; }
    // True ⇒ the INPUT feed (not us) is the limiter, so decimation would not help.
    [[nodiscard]] bool   feed_limited()    const { return feed_hz_ > 0.0 && feed_hz_ < cfg_.target_hz - cfg_.deadband_hz; }
    [[nodiscard]] bool   at_pose_cap()     const { return pose_decim_ >= cfg_.pose_decim_max; }

private:
    void control()
    {
        changed_ = false;
        const double budget_ms = 1000.0 / std::max(1.0f, cfg_.target_hz);
        const bool behind = proc_hz_ > 0.0 && proc_hz_ < cfg_.target_hz - cfg_.deadband_hz;
        const bool bound  = comp_ms_ >= 0.8 * budget_ms;                       // compute is eating the budget
        const bool feedok = feed_hz_ <= 0.0 || feed_hz_ >= cfg_.target_hz - cfg_.deadband_hz;
        const bool ample  = proc_hz_ > cfg_.target_hz + cfg_.deadband_hz && comp_ms_ < 0.6 * budget_ms;

        // Additive-increase decimation only when we're behind BECAUSE of compute (bound) and the
        // feed can actually supply the target; multiplicative-safe additive-decrease when we have room.
        if (behind && bound && feedok)
        {
            if (pose_decim_ < cfg_.pose_decim_max) { ++pose_decim_; changed_ = true; }
        }
        else if (ample)
        {
            if (pose_decim_ > cfg_.pose_decim_min) { --pose_decim_; changed_ = true; }
        }
    }

    RateRegulatorConfig cfg_{};
    int    pose_decim_ = 1;
    double proc_hz_ = 0.0, comp_ms_ = 0.0, feed_hz_ = 0.0;
    double win_min_gap_ms_ = 1e9;
    std::uint64_t last_ts_ = 0;
    std::chrono::steady_clock::time_point last_tp_{}, ctrl_tp_{};
    bool have_last_ = false, changed_ = false;
};

} // namespace rc
