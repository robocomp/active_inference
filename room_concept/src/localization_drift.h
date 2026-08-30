/*
 * localization_drift.h — the LOCALIZATION metric: how far the estimate's own motion departs from
 * the true motion, per unit of motion. Display-only and header-only; nothing here feeds the
 * estimator.
 *
 * ★★★ WHY NOT "DISTANCE FROM THE TRUE POSITION". Ground truth is the Webots WORLD frame; the
 * estimate is the ROOM frame, whose orientation room_concept picks from its own fit and which is
 * free to differ between runs (observed sitting in stable modes tens of degrees apart). |est - gt|
 * is therefore dominated by an arbitrary frame choice that changes when nothing about the
 * localiser changed, and comparing two runs on it compares two frame choices. This measures
 * RELATIVE pose error instead — the disagreement in motion over a fixed span — which is invariant
 * to the frame entirely, and which is exactly what the motion-model parameters (k_v, k_omega,
 * eps_yaw) act on.
 *
 * ★★★ NORMALISED BY MOTION, NOT BY TIME. A parked robot predicts nothing and so always looks
 * accurate; a per-second error would reward standing still. mm per metre travelled and deg per
 * radian turned are comparable between arms that drove different amounts, which is the whole
 * point when the arms are an A/B.
 *
 * ★★★ THE TWO HEADING COLUMNS DO NOT SHARE A CONVENTION, AND NOTHING DECLARES IT.
 *   - `robot_gt_angle` is published with an INVERTED sign (room_viewer.cpp already negates it;
 *     see SpecificWorker::gt_convention_report).
 *   - the room-frame heading is the BODY frame's, forward = +y, where the supervisor's is
 *     forward = +x. Measured on gt_error_2026-08-29_21-54-45.csv as mean(course - theta):
 *     GT +0.01 deg (R = 1.000) against EST +89.77 deg (R = 0.904).
 * Missing that second one produced a relative translation error of 1361 mm/m — 27x wrong — that
 * read as a catastrophically broken localiser, while the ROTATION channel looked fine throughout
 * (a difference of angle changes cannot see a constant offset). Both corrections are applied here
 * from the frame documentation and then VERIFIED against the incoming data; if the check fails the
 * meter reports `suspect` and publishes nothing, rather than drawing a confident wrong line.
 *
 * ★ THE CORRECTIONS ARE DECLARED, NEVER FITTED. A free constant angular offset is exactly the
 * shape of eps_yaw (the mount-yaw parameter the calibration estimates), so fitting the convention
 * would silently absorb the quantity under test.
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <deque>
#include <numbers>

namespace rc
{

class LocalizationDriftMeter
{
public:
    struct Config
    {
        /// One sample per this much GT travel / rotation. Spans are NON-OVERLAPPING so successive
        /// samples are independent: overlapping ones reuse the same motion and make any spread
        /// look tighter than the data supports.
        double span_trans_m   = 1.0;
        double span_rot_rad   = 0.5;
        /// Below this the robot is not moving and the pair carries no information about the model.
        double min_step_m     = 0.005;
        /// The declared conventions, from the frame documentation. NOT knobs.
        bool   negate_gt_theta = true;                          ///< robot_gt_angle's inverted sign
        double est_yaw_offset  = std::numbers::pi / 2.0;        ///< room/body frame forward = +y
        /// How far the measured convention may sit from the declared one before the meter refuses.
        double convention_tol_rad = 15.0 * std::numbers::pi / 180.0;
        /// Samples needed before the convention check can speak at all.
        std::size_t check_min_n = 40;
    };

    struct Out
    {
        bool   has_trans = false;
        double mm_per_m  = 0.0;   ///< |true motion - estimated motion| per metre travelled
        bool   has_rot   = false;
        double deg_per_rad = 0.0; ///< heading disagreement per radian turned
    };

    void configure(const Config &c) noexcept { cfg_ = c; }

    /// True when the incoming data contradicts the declared heading conventions. While set, the
    /// meter emits nothing: a metric that is silently measuring the wrong thing is worse than a
    /// gap in the plot, because only one of the two looks like a problem.
    [[nodiscard]] bool suspect() const noexcept { return suspect_; }
    [[nodiscard]] const char *suspect_reason() const noexcept { return reason_; }
    /// mean(course - theta) actually measured, radians, for the report line. NaN until enough motion.
    [[nodiscard]] double measured_gt_offset() const noexcept { return circ_mean(gs_, gc_, gn_); }
    [[nodiscard]] double measured_est_offset() const noexcept { return circ_mean(es_, ec_, en_); }

    /// Feed one cycle. `gt_theta` and `est_theta` are the RAW values as published; the declared
    /// corrections are applied inside, so callers cannot apply them twice.
    Out push(double gt_x, double gt_y, double gt_theta,
             double est_x, double est_y, double est_theta)
    {
        Out out;
        const double gth = wrap((cfg_.negate_gt_theta ? -gt_theta : gt_theta));
        const double eth = wrap(est_theta + cfg_.est_yaw_offset);

        if (not buf_.empty())
        {
            const S &p = buf_.back();
            const double dxg = gt_x - p.gx, dyg = gt_y - p.gy;
            const double step = std::hypot(dxg, dyg);
            if (step > cfg_.min_step_m)
            {
                // The convention check, on real motion only: for each trajectory, how far its OWN
                // reported heading sits from its OWN direction of travel. Each is measured against
                // its own x/y, so neither frame is assumed and the comparison never crosses them.
                accum(std::atan2(dyg, dxg) - gth, gs_, gc_, gn_);
                const double dxe = est_x - p.ex, dye = est_y - p.ey;
                if (std::hypot(dxe, dye) > cfg_.min_step_m)
                    accum(std::atan2(dye, dxe) - eth, es_, ec_, en_);
                check_convention();
            }
            acc_s_ += step;
            acc_r_ += std::abs(wrap(gth - p.gth));
        }

        buf_.push_back(S{gt_x, gt_y, gth, est_x, est_y, eth, acc_s_, acc_r_});
        while (buf_.size() > 4096) buf_.pop_front();
        if (suspect_) return out;

        if (acc_s_ - anchor_t().s >= cfg_.span_trans_m)
        {
            out.has_trans = true;
            out.mm_per_m = rpe_trans(anchor_t(), buf_.back()) / cfg_.span_trans_m * 1000.0;
            at_ = buf_.size() - 1;
        }
        if (acc_r_ - anchor_r().r >= cfg_.span_rot_rad)
        {
            out.has_rot = true;
            out.deg_per_rad = rpe_rot(anchor_r(), buf_.back()) / cfg_.span_rot_rad
                            * 180.0 / std::numbers::pi;
            ar_ = buf_.size() - 1;
        }
        // Indices are into a deque that pops from the front; rebase them rather than letting them
        // silently point at the wrong sample.
        if (at_ >= buf_.size()) at_ = 0;
        if (ar_ >= buf_.size()) ar_ = 0;
        return out;
    }

    void reset() noexcept
    {
        buf_.clear();
        acc_s_ = acc_r_ = 0.0;
        at_ = ar_ = 0;
        gs_ = gc_ = es_ = ec_ = 0.0;
        gn_ = en_ = 0;
        suspect_ = false;
        reason_ = "";
    }

private:
    struct S { double gx, gy, gth, ex, ey, eth, s, r; };

    static double wrap(double a) noexcept
    {
        constexpr double tau = 2.0 * std::numbers::pi;
        return std::fmod(a + std::numbers::pi + tau * 4.0, tau) - std::numbers::pi;
    }
    static void accum(double a, double &s, double &c, std::size_t &n) noexcept
    { s += std::sin(a); c += std::cos(a); ++n; }
    static double circ_mean(double s, double c, std::size_t n) noexcept
    { return n == 0 ? NAN : std::atan2(s / double(n), c / double(n)); }

    /// Relative pose error: the true displacement and the estimated one, each expressed in ITS OWN
    /// trajectory's body frame at the start of the span, then differenced. No shared frame is ever
    /// assumed, which is what makes this immune to the room frame's arbitrary orientation.
    static double rpe_trans(const S &a, const S &b) noexcept
    {
        const double cg = std::cos(a.gth), sg = std::sin(a.gth);
        const double ce = std::cos(a.eth), se = std::sin(a.eth);
        const double dxg = b.gx - a.gx, dyg = b.gy - a.gy;
        const double dxe = b.ex - a.ex, dye = b.ey - a.ey;
        const double gx =  cg * dxg + sg * dyg, gy = -sg * dxg + cg * dyg;
        const double ex =  ce * dxe + se * dye, ey = -se * dxe + ce * dye;
        return std::hypot(gx - ex, gy - ey);
    }
    static double rpe_rot(const S &a, const S &b) noexcept
    { return std::abs(wrap((b.gth - a.gth) - (b.eth - a.eth))); }

    const S &anchor_t() const noexcept { return buf_[at_ < buf_.size() ? at_ : 0]; }
    const S &anchor_r() const noexcept { return buf_[ar_ < buf_.size() ? ar_ : 0]; }

    void check_convention() noexcept
    {
        if (suspect_ or gn_ < cfg_.check_min_n or en_ < cfg_.check_min_n) return;
        // Both are corrected before they get here, so BOTH should now sit on their own course.
        if (std::abs(wrap(circ_mean(gs_, gc_, gn_))) > cfg_.convention_tol_rad)
        { suspect_ = true; reason_ = "ground-truth heading is not on its own course after the "
                                     "declared sign flip — robot_gt_angle's convention changed"; }
        else if (std::abs(wrap(circ_mean(es_, ec_, en_))) > cfg_.convention_tol_rad)
        { suspect_ = true; reason_ = "estimated heading is not on its own course after the declared "
                                     "+90 deg — the room/body frame convention changed"; }
    }

    Config cfg_{};
    std::deque<S> buf_;
    double acc_s_ = 0.0, acc_r_ = 0.0;
    std::size_t at_ = 0, ar_ = 0;
    double gs_ = 0.0, gc_ = 0.0, es_ = 0.0, ec_ = 0.0;
    std::size_t gn_ = 0, en_ = 0;
    bool suspect_ = false;
    const char *reason_ = "";
};

}   // namespace rc
