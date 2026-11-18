/*
 * tick_timing.h — per-stage wall-clock timing of one compute() tick (diagnostic only; changes no result).
 *
 * A TickTiming is zeroed at the top of each tick; StageTimer (RAII, steady_clock) ADDS its elapsed ms to one
 * field on destruction, so a stage entered once per instance is summed over the tick. One row per tick goes to
 * timing_log.csv (ms, 3 decimals). The fitter holds a borrowed pointer to the worker's record (single thread).
 */
#pragma once

#include <chrono>
#include <fstream>
#include <iomanip>
#include <locale>

namespace rc
{
struct TickTiming
{
    // Stages (ms, summed over the tick). Nested ones are noted: infer INCLUDES membership+feed; inst_total
    // INCLUDES observe, infer and publish.
    double mask_refresh = 0, tracker = 0, foreign_claims = 0, lidar_pump = 0, rgb_stage = 0;
    double ego_motion = 0, inst_total = 0, observe = 0, membership = 0, lidar_feed = 0, infer = 0;
    double publish = 0, ricoh = 0, existence = 0, class_factor = 0, dashboard = 0, tick = 0;
    int n_inst = 0, n_cam = 0, n_ray = 0, n_none = 0;   // instances: camera data / LiDAR rays only / nothing

    void reset() { *this = TickTiming{}; }

    static const char* header()
    {
        return "ts_ms,n_inst,n_cam,n_ray,n_none,mask_refresh,tracker,foreign_claims,lidar_pump,rgb_stage,ego_motion,"
               "inst_total,observe,membership,lidar_feed,infer,publish,ricoh,existence,class_factor,dashboard,tick\n";
    }

    void write_row(std::ofstream& f, long long ts_ms) const
    {
        f << ts_ms << ',' << n_inst << ',' << n_cam << ',' << n_ray << ',' << n_none << std::fixed << std::setprecision(3);
        for (const double v : {mask_refresh, tracker, foreign_claims, lidar_pump, rgb_stage, ego_motion, inst_total,
                               observe, membership, lidar_feed, infer, publish, ricoh, existence, class_factor,
                               dashboard, tick})
            f << ',' << v;
        f << '\n';
        f.flush();
    }
};

// Adds elapsed ms to `slot` on scope exit. A null `slot` pointer-owner (no timing record) costs nothing.
class StageTimer
{
public:
    explicit StageTimer(double* slot) : slot_(slot), t0_(std::chrono::steady_clock::now()) {}
    ~StageTimer()
    {
        if (slot_)
            *slot_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0_).count();
    }
    StageTimer(const StageTimer&) = delete;
    StageTimer& operator=(const StageTimer&) = delete;
private:
    double* slot_;
    std::chrono::steady_clock::time_point t0_;
};
}  // namespace rc
