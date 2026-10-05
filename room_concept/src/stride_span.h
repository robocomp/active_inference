/*  stride_span.h — what the NEWEST window slot's motion factor must carry under the strided window.
 *
 *  Every frame builds a slot: it either REPLACES the newest slot (not enough motion since the last
 *  admission) or is APPENDED after it. So the newest slot is always the PREVIOUS frame, and a slot's
 *  motion factor links it to the slot before it in the window (room_gn_solver.cpp: MotionFactor(i-1, i)):
 *    replace -> this frame links to the newest's predecessor: it spans  newest's factor  ⊕ this frame;
 *    append  -> this frame links to the newest, i.e. the previous frame: it spans this frame ONLY.
 *
 *  ★ THE DEFECT THIS REPLACES (2026-10-05): one accumulator, `+=` every frame, used for BOTH cases and
 *    zeroed after an append. An appended slot then carried the whole stride (~0.10-0.15 m, or ~0.15 rad)
 *    as if it happened in one frame, and the next replaced slot carried one frame while spanning two.
 *    The solve was pushed forward by a few cm at every admission and snapped back one cycle later: sign-
 *    alternating ±3-9 cm / 2-4° correction pairs every 0.10 m of travel or 0.15 rad of turn (26/29 pairs
 *    pointed FORWARD while driving; heading_2026-10-04_21-33-03.csv). tools/stride_span_selftest.cpp.
 */
#pragma once
#include <Eigen/Dense>
#include "se2_preintegration.h"

namespace rc
{
    class StrideSpan
    {
    public:
        struct Factor
        {
            Eigen::Vector3f delta = Eigen::Vector3f::Zero();   ///< global-frame increment, additive
            Eigen::Matrix3f cov   = Eigen::Matrix3f::Zero();   ///< summed per-frame covariances (legacy)
            rc::preint::Interval preint{};                     ///< chained interval (samples == 0: none)
        };

        /// This frame REPLACES the newest slot: newest's span ⊕ this frame.
        const Factor& replace(const Eigen::Vector3f &d, const Eigen::Matrix3f &c, const rc::preint::Interval *p)
        {
            f_.delta += d;
            f_.cov   += c;
            if (p) f_.preint = rc::preint::chain(f_.preint, *p);
            return f_;
        }
        /// This frame is APPENDED after the newest (the previous frame): this frame only.
        const Factor& append(const Eigen::Vector3f &d, const Eigen::Matrix3f &c, const rc::preint::Interval *p)
        {
            f_.delta  = d;
            f_.cov    = c;
            f_.preint = p ? *p : rc::preint::Interval{};
            return f_;
        }
        [[nodiscard]] const Factor& current() const noexcept { return f_; }
        void reset() { f_ = Factor{}; }

    private:
        Factor f_{};
    };
}
