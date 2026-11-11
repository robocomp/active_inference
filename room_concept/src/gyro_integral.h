/*
 *    Copyright (C) 2026 by RoboLab at the University of Extremadura
 *    This file is part of RoboComp — see room_scene_graph.h.
 */
// THE GYRO HEADING INTEGRAL, ONE IMPLEMENTATION. Extracted 2026-11-11 from RoomConcept::OdomSources::imu_dtheta
// (room_concept_odometry_window.cpp), which now calls it, so the motion prior and the published prediction
// (rc::publish::predict_to_newest_imu, the second block of every correction's RT rewrite) integrate the gyro the
// same way: trapezoid across each sample pair, clipped to the segment, refused unless the history brackets the
// WHOLE segment (a partial integral silently under-rotates).
#pragma once

#include "common_types.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace rc::imu
{
/// Visits every piece of [seg_start, seg_end] lying between two consecutive samples, oldest first, as
/// visit(s, e, dtheta_raw): the piece's bounds on `stamp`'s clock and the RAW gyro increment over it (rad, no
/// calibration). `stamp(sample)` is the clock the bounds are on (the integration key in simulation, see
/// ImuReading::integration_ts_ms). Returns false — and visits nothing — unless the history brackets the whole
/// segment; also false if no piece had positive length.
template <class StampFn, class Visit>
bool for_each_gyro_step(const std::vector<ImuReading>& history, StampFn&& stamp,
                        std::int64_t seg_start, std::int64_t seg_end, Visit&& visit)
{
    if (history.size() < 2) return false;
    if (stamp(history.front()) > seg_start or stamp(history.back()) < seg_end)
        return false;
    bool any = false;
    for (std::size_t k = 0; k + 1 < history.size(); ++k)
    {
        const std::int64_t a = stamp(history[k]);
        const std::int64_t b = stamp(history[k + 1]);
        if (b <= a) continue;
        const std::int64_t s = std::max(a, seg_start);
        const std::int64_t e = std::min(b, seg_end);
        if (e <= s) continue;
        // Trapezoid across the sample pair, so a rate ramping through the segment is not
        // biased by holding the leading sample flat.
        const double w0 = history[k].gyro_z, w1 = history[k + 1].gyro_z;
        const double span = static_cast<double>(b - a);
        const double f0 = static_cast<double>(s - a) / span;
        const double f1 = static_cast<double>(e - a) / span;
        const double wm = 0.5 * ((w0 + (w1 - w0) * f0) + (w0 + (w1 - w0) * f1));
        visit(s, e, wm * static_cast<double>(e - s) * 0.001);
        any = true;
    }
    return any;
}
}   // namespace rc::imu
