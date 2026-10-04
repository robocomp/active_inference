/*  surprise.h — how much the LiDAR had to tell the motion model: KL(posterior || prediction), in nats.
 *
 *  WHY
 *  ---
 *  Self-calibration has no ground truth on the real robot, so "is the calibrated motion model better?"
 *  needs a measure the robot can compute about itself. The localiser fuses a PREDICTION, the motion
 *  model's N(pred, P_pred), with the room's evidence and returns a POSTERIOR N(est, P_post). The
 *  information the evidence had to inject is the KL divergence of the posterior from the prediction,
 *      KL = 1/2 [ tr(P_pred^-1 P_post) - 3 + ln det P_pred - ln det P_post ]   (information term)
 *         + 1/2  c^T P_pred^-1 c,                c = est - pred              (mismatch term)
 *  which is the COMPLEXITY term of the free energy, the cost of updating beliefs. A better motion model
 *  needs less of it per metre and per second.
 *  The two halves answer different questions. The INFORMATION term is how much the measurement
 *  sharpened the pose: it falls when the prediction is already tight (a well-calibrated model is
 *  allowed to be tight). The MISMATCH term is how far the prediction was wrong, in its own units. Under
 *  a model that is CALIBRATED (prediction and posterior both honest), the correction c is Gaussian with
 *  covariance P_pred - P_post, so
 *      E[mismatch] = 1/2 tr(P_pred^-1 (P_pred - P_post)) = 1/2 (3 - tr(P_pred^-1 P_post)),
 *  which `expected` reports. Over a run, sum(mismatch) / sum(expected) is ~1 for an honest model, > 1
 *  when the prediction is biased or over-confident, < 1 when it is under-confident. That ratio is
 *  scale-free and needs no ground truth; it is the per-run calibration verdict.
 *
 *  WHEN IT IS SCORED
 *  -----------------
 *  Only on cycles where the estimate was CORRECTED (an optimiser solve, or a polish). On an early exit
 *  the posterior IS the prediction, so the KL is zero; the open-loop stretch is not lost, it is scored
 *  whole at the next correction, against a P_pred that accumulated every cycle's motion noise since the
 *  last one (the caller keeps that sum). The early-exit gate chooses WHEN to score, so the cycles scored
 *  are the ones where the prediction looked worst: the measure is conservative, and comparable between
 *  runs that share the gate.
 */
#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <limits>

namespace rc::surprise
{
    struct Surprise
    {
        float kl       = 0.f;   ///< nats, KL(posterior || prediction) = mismatch + info
        float mismatch = 0.f;   ///< nats, 1/2 c^T P_pred^-1 c
        float expected = 0.f;   ///< nats, E[mismatch] under a calibrated model
        float info     = 0.f;   ///< nats, the covariance part of the KL
        bool  scored   = false; ///< a correction was scored this cycle
        // Per axis, in the BODY frame (forward, lateral, heading), filled by the caller: the correction and
        // the predictive / posterior variances on that axis. Logged to attribute an over- or under-confident
        // motion covariance to an axis before anything is rescaled (2026-10-04).
        float c_fwd = 0.f, c_lat = 0.f, c_th = 0.f;
        float pp_fwd = 0.f, pp_lat = 0.f, pp_th = 0.f;
        float pq_fwd = 0.f, pq_lat = 0.f, pq_th = 0.f;
        int   open_cycles = 0;  ///< prediction cycles accumulated into P_pred since the last scored correction
    };

    /// c = est - pred (heading already wrapped). P_pred, P_post: 3x3 pose covariances (x, y, theta).
    /// Returns scored = false when P_pred is not positive definite.
    [[nodiscard]] inline Surprise kl_gauss(const Eigen::Vector3f &c, const Eigen::Matrix3f &P_pred,
                                           const Eigen::Matrix3f &P_post)
    {
        Surprise s;
        if (not c.allFinite() or not P_pred.allFinite() or not P_post.allFinite()) return s;
        const Eigen::Matrix3d Pp = P_pred.cast<double>(), Pq = P_post.cast<double>();
        const Eigen::LLT<Eigen::Matrix3d> lp(Pp), lq(Pq);
        if (lp.info() != Eigen::Success or lq.info() != Eigen::Success) return s;
        const Eigen::Vector3d cd = c.cast<double>();
        const double maha = cd.dot(lp.solve(cd));
        const double tr   = lp.solve(Pq).trace();
        const auto logdet = [](const Eigen::LLT<Eigen::Matrix3d> &l)
        { return 2.0 * l.matrixL().toDenseMatrix().diagonal().array().log().sum(); };
        const double info = 0.5 * (tr - 3.0 + logdet(lp) - logdet(lq));
        s.mismatch = float(0.5 * maha);
        s.expected = float(0.5 * (3.0 - tr));
        s.info     = float(info);
        s.kl       = s.mismatch + s.info;
        s.scored   = true;
        return s;
    }
}
