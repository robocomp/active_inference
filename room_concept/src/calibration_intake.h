/*  calibration_intake.h — where calibration data comes from, and what is allowed in.
 *
 *  THE IDEA: THE ESTIMATOR DOES NOT CARE WHY THE ROBOT MOVED.
 *  ---------------------------------------------------------
 *  Motion is motion. The robot spends its day serving standpoints and every one of those tours turns
 *  and translates, so ordinary work is free calibration data. A deliberate manoeuvre is a SECOND
 *  source, not a better one — it is offered as an affordance, executed by the controller at a time
 *  nobody here chooses, and may be pre-empted by anything more urgent. It may never run at all.
 *
 *  So this class sits at the back and takes what arrives. It has no opinion about the schedule, no
 *  state machine tracking a manoeuvre, and no way to make anything happen. Data shows up; it decides
 *  whether the data is USABLE; the estimator sees only what survives.
 *
 *  ★★★PROVENANCE MUST NOT CHANGE THE WEIGHT. An episode produced during a calibration pivot is not
 *  more trustworthy than one produced while fetching a cup — same robot, same sensors, same physics.
 *  It is only more INFORMATIVE, and that is already carried by its covariates through the Jacobian.
 *  Weighting by provenance on top would count the excitation twice. Source is recorded for reporting
 *  and for answering "did the manoeuvre we offered actually deliver anything", never for arithmetic.
 *
 *  ★★★AN INTERRUPTED MANOEUVRE IS NOT LOST DATA. If the controller abandons a pivot halfway because
 *  something more urgent arrived, the robot still turned, and that turning was still measured. The
 *  episodes stay. Discarding them because the TASK failed would confuse a task outcome with a
 *  measurement — and the affordance protocol's own history is full of that confusion. Only a chunk
 *  that is physically corrupt gets dropped.
 *
 *  ★★★"NOTHING ARRIVED" AND "EVERYTHING WAS REJECTED" MUST NOT LOOK ALIKE. They are opposite
 *  problems — an idle robot versus a broken localiser — and from outside both are simply an estimate
 *  that stopped moving. Every rejection is therefore counted BY REASON. This is the same defect that
 *  hid two wiring bugs earlier in this work: a learner reading zeros and a correctly-idle learner are
 *  indistinguishable unless something counts what was refused.
 *
 *  WHAT IS *NOT* A REJECTION CRITERION: "is this episode informative enough". There is no such test
 *  and there must not be. An uninformative episode carries a covariate near zero, the solve's own
 *  weighting gives it near-zero influence, and it costs nothing to admit. A threshold there would
 *  throw away the small, numerous episodes that actually make a window well-conditioned.
 */
#pragma once

#include "calibration_estimator.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <string_view>

namespace rc::calib
{
    /// Where an episode came from. Recorded, never weighted — see the header.
    enum class Source : int { Passive = 0, Manoeuvre = 1, COUNT = 2 };

    /// Why an episode was refused. Counted separately so an idle robot and a broken one differ.
    enum class Verdict : int
    {
        Accepted = 0,
        NonFinite,        ///< NaN/inf anywhere — a corrupt frame, not a quiet one
        NotTracking,      ///< the localiser's own fit was poor, so its correction is recovery,
                          ///< not model error. An estimator driven by another estimator's residuals
                          ///< inherits its failure modes.
        Implausible,      ///< the motion implies a speed the base cannot reach: a relocalisation
                          ///< jump charged to odometry would put a metre-scale outlier into a fit
                          ///< whose typical increment is a metre.
        COUNT
    };

    [[nodiscard]] constexpr std::string_view verdict_name(Verdict v)
    {
        switch (v)
        {
            case Verdict::Accepted:    return "accepted";
            case Verdict::NonFinite:   return "non-finite";
            case Verdict::NotTracking: return "localiser not tracking";
            case Verdict::Implausible: return "kinematically impossible";
            default:                   return "?";
        }
    }

    struct IntakeParams
    {
        /// Above this localiser residual the fit is not trustworthy enough for its correction to be
        /// read as model error. Measured 2026-08-23: normal operation runs ~0.027 m; the pathological
        /// windows where the optimizer fires every cycle ran 0.05-0.26 and manufactured 497 fragment
        /// episodes in 180 s, dragging a healthy parameter far out of range.
        float max_fit_residual_m = 0.10f;
        /// The base's own envelope. A localiser jump is not travel.
        float max_speed_mps = 0.8f;
        /// Sliding window handed to the joint solve.
        std::size_t window = 512;
    };

    /// Owns the admission policy and the estimator. Knows nothing about Qt, DSR, the affordance
    /// protocol, or who is driving — it is fed, and it reports.
    class CalibrationIntake
    {
    public:
        void configure(const IntakeParams &ip, const Prior &prior)
        { p_ = ip; est_.configure(prior, ip.window); }

        /// Offer one CLOSED PIVOT. It bypasses the episode admission policy on purpose: that policy
        /// exists to reject episodes whose reference — the optimizer's correction — is untrustworthy,
        /// and a closure has no such reference. Its truth is that the robot came back to the heading
        /// it left, which no localiser, map or fit residual can spoil. The one thing that can spoil
        /// it is not closing, and PivotAffordance refuses to report that as a closure at all.
        void offer_closure(double truth_rad, double turned_rad, double rate_rad_s, double sigma_s)
        { est_.add_closure(truth_rad, turned_rad, rate_rad_s, sigma_s); }
        [[nodiscard]] std::size_t closures() const noexcept { return est_.closures(); }

        /// Persist / restore the WINDOW (evidence), and forget it. See BatchEstimator::save.
        bool        save(const std::string& path) const { return est_.save(path); }
        std::size_t load(const std::string& path)       { return est_.load(path); }
        void        reset() noexcept                    { est_.reset(); }

        /// Offer one episode. Returns why it was accepted or refused. `fit_residual_m` is the
        /// localiser's own worst residual over the episode; `elapsed_s` its duration.
        Verdict offer(const Episode &e, Source src, float fit_residual_m)
        {
            const Verdict v = judge(e, fit_residual_m);
            ++counts_[static_cast<int>(v)];
            if (v == Verdict::Accepted)
            {
                est_.add(e);
                ++accepted_by_source_[static_cast<int>(src)];
                if (const auto r = est_.solve(); r.ok) last_ = r;
            }
            return v;
        }

        [[nodiscard]] const Result &estimate() const noexcept { return last_; }
        [[nodiscard]] std::size_t   pool() const noexcept { return est_.size(); }
        [[nodiscard]] int count(Verdict v) const noexcept { return counts_[static_cast<int>(v)]; }
        [[nodiscard]] int accepted_from(Source s) const noexcept
        { return accepted_by_source_[static_cast<int>(s)]; }
        /// Total offered, so "nothing arrived" is visible as a number rather than inferred.
        [[nodiscard]] int offered() const noexcept
        {
            int n = 0; for (int c : counts_) n += c; return n;
        }

    private:
        [[nodiscard]] Verdict judge(const Episode &e, float fit) const
        {
            const bool finite = std::isfinite(e.d_forward) and std::isfinite(e.d_lateral)
                            and std::isfinite(e.d_theta)   and std::isfinite(e.duration)
                            and std::isfinite(e.r_forward) and std::isfinite(e.r_lateral)
                            and std::isfinite(e.r_theta);
            if (not finite) return Verdict::NonFinite;
            if (std::isfinite(fit) and fit > p_.max_fit_residual_m) return Verdict::NotTracking;
            // Judged on the EPISODE's own displacement against its own duration. A long episode can
            // legitimately cover a long distance; what cannot happen is covering it faster than the
            // base can move.
            if (e.duration > 1e-3f)
            {
                const float speed = std::hypot(e.d_forward, e.d_lateral) / e.duration;
                if (speed > p_.max_speed_mps) return Verdict::Implausible;
            }
            return Verdict::Accepted;
        }

        IntakeParams p_{};
        BatchEstimator est_;
        Result last_{};
        std::array<int, static_cast<int>(Verdict::COUNT)> counts_{};
        std::array<int, static_cast<int>(Source::COUNT)>  accepted_by_source_{};
    };
}
