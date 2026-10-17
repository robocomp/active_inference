// RoomConcept, split out of room_concept.cpp (refactor step 2, docs/refactor_proposal_2026-10-13.md).
// Pose solve: scan linearisation, the GN loop, posterior covariance, boundary hyper-precision.

#include "room_concept.h"
#include "../../common/status_stream/status_sink.h"
#include <pthread.h>   // pthread_setname_np: name the worker so a per-thread CPU sample attributes itself
#include "pointcloud_center_estimator.h"
#include <numeric>
#include <set>
#include "room_gn_solver.h"
#include "room_obs_weights.h"
#include "room_gn_solver.h"
#include "image_edge_accumulate.h"
#include "image_edge_ops.h"
#include <algorithm>
#include <ranges>
#include <fstream>
#include <filesystem>
#include <limits>
#include <locale>
#include <ctime>
#include <print>
#include <sstream>
#include <sys/stat.h>
#include <QDebug>

namespace rc
{
    namespace
    {

        /// Conversion between the two absolute-residual estimators this class used to mix.
        /// For a half-normal residual of scale sigma, E|x| = sqrt(2/pi)*sigma = 0.798*sigma while
        /// median|x| = 0.674*sigma. Until 2026-08-13 UpdateResult::sdf_mse was a MEAN on the
        /// early-exit path and a MEDIAN on the optimized path; it is now a median everywhere, so a
        /// threshold that was calibrated against the old mean must be multiplied by this factor to
        /// keep its meaning. Applied once to each affected config value (see config.toml)
        /// and once in boundary_weight_now().
        /// ⚠ PROVISIONAL. 0.845 is the ratio for a HALF-NORMAL residual, and this residual is not
        /// half-normal: roughly two thirds of the floor is systematic map mismatch (furniture,
        /// doorway gaps, wall thickness the polygon omits) rather than zero-mean noise. A systematic
        /// component drives the mean and the median TOWARDS each other — for a pure offset the ratio
        /// is 1 — so 0.845 is expected to OVER-correct, leaving the rescaled thresholds slightly too
        /// tight. Do not reason about the shape of the distribution; measure the ratio:
        ///
        ///     python3 tools/measure_sdf_ratio.py tmp/sdf_localizer/log_<newest>.csv
        ///
        /// On an early-exit frame the log now carries both statistics of the SAME residuals at the
        /// SAME pose (sdf_mse = median, early_exit_metric = mean), so their ratio IS this constant.
        /// The tool prints the measured value and re-derives all six thresholds from their original
        /// mean-calibrated numbers. Replace the value here and in config.toml together.
        constexpr float kMedianOverMeanAbs = 0.674f / 0.798f;   // = 0.845, pending measurement
    }

    bool RoomConcept::scan_linearize(const ScanPoints &pts, float x, float y, float th,
                                     Eigen::Matrix3f &H, Eigen::Vector3f &bb)
    {
        H.setZero(); bb.setZero();
        if (pts.empty() or not model_) return false;
        // SdfFactor's own linearization (room_obs_weights.cpp), with this scan's door weights.
        const auto lin = linearize_observation(*model_, params, pts, Eigen::Vector3f(x, y, th), &doors_robot_);
        H = lin.H; bb = lin.g;
        return H.allFinite() and bb.allFinite() and H.trace() > 0.f;
    }

    void RoomConcept::scan_only_pose(const ScanPoints &pts, float x, float y, float th)
    {
        innov_ok_ = false;
        last_scan_iters_ = 0;
        // ★ ITERATED (2026-10-08). z is DEFINED as the scan-only optimum. One Gauss-Newton step from the
        //   dead-reckoned prediction recovered only g = 0.81-0.88 of a planted mount offset against ground
        //   truth (z - pred = g·(gt - pred), tools/kinematic_shortfall_diag.py); the missing (1 - g) then
        //   arrived as a burst at the next solve, which the noise learner's outlier mixture down-weighted, so
        //   every consumer of z (noise learner, pose-field bias, kinematic mount factor) saw a shrunk signal.
        //   Re-linearise at each new pose up to ScanOnlyIters times; stop when a step is numerically
        //   negligible (a convergence test, not a model threshold). R = H^-1 at the FINAL pose.
        Eigen::Vector3f z(x, y, th);
        Eigen::Matrix3f R = Eigen::Matrix3f::Zero();
        bool have = false;
        for (int it = 0; it < std::max(1, params.scan_only_iters); ++it)
        {
            Eigen::Matrix3f H; Eigen::Vector3f bb;
            if (not scan_linearize(pts, z.x(), z.y(), z.z(), H, bb)) break;
            const Eigen::LDLT<Eigen::Matrix3f> ldlt(H);
            if (ldlt.info() != Eigen::Success) break;
            const Eigen::Vector3f step = -ldlt.solve(bb);
            const Eigen::Matrix3f Ri = ldlt.solve(Eigen::Matrix3f::Identity());
            if (not step.allFinite() or not Ri.allFinite() or Ri(0, 0) <= 0.f or Ri(2, 2) <= 0.f) break;
            z += step;
            z[2] = std::remainder(z[2], 2.f * float(M_PI));
            R = Ri;          // the curvature at the pose this step started from; refreshed next pass
            have = true;
            ++last_scan_iters_;
            if (step.head<2>().norm() < 1e-4f and std::abs(step[2]) < 1e-4f) break;
        }
        if (not have) return;
        innov_z_ = z;
        innov_R_ = 0.5f * (R + R.transpose());
        innov_ok_ = true;
    }

    // Hierarchical boundary precision (HIERARCHICAL_PRECISION.md, Stage 1 + Option A).
    // Treat the boundary-prior precision scale as a random variable π=exp(u_b_) with a log-precision
    // hyperprior u ~ N(g(v), σ_u²), g(v)=u0+g_gain·v, and a slow in-process hyper-state v (map_trust)
    // with its own prior v ~ N(0, σ_v²). One fast VB step on u_b_ and one slow step on v per converged
    // frame. Well-posed only because μ,Λ_b are FEJ-frozen (fixed linearization point) — see the FEJ+Schur
    // marginalization. No-op unless enabled ⇒ zero effect on the legacy quality-gate path.
    void RoomConcept::update_boundary_hyperprecision(const Eigen::Matrix3f &sigma_x)
    {
        if (not params.hier_prec_boundary_enabled) return;

        const auto &bp = window_mgr_.boundary_prior;
        // Same guard as the boundary loss term: it only contributes when the prior is valid AND the
        // oldest slot is not the current one (window.size() > 1). Otherwise there is no residual to learn from.
        if (not bp.valid or window_mgr_.size() <= 1) return;

        // Oldest surviving pose x̂ (post-optimization) — the state the boundary prior anchors.
        Eigen::Vector3f d = window_mgr_.window.front().pose - bp.mu;
        d[2] = std::atan2(std::sin(d[2]), std::cos(d[2]));   // wrap the angle residual

        // Expected boundary residual ⟨r_b⟩ = Δᵀ Λ_b Δ + tr(Λ_b Σ_x): the point residual plus the
        // pose-uncertainty discount (a shaky posterior should not over-drive the precision down).
        const float quad  = d.dot(bp.precision * d);
        const float trace = (bp.precision * sigma_x).trace();
        const float r_b   = quad + std::max(0.0f, trace);

        constexpr float d_dim = 3.0f;                        // boundary factor dimensionality
        if (not u_b_init_)                                   // seed to the top-down prediction g(v)
        {
            u_b_ = params.hier_prec_u0 + params.hier_prec_g_gain * map_trust_v_;
            u_b_init_ = true;
        }
        const float g_v = params.hier_prec_u0 + params.hier_prec_g_gain * map_trust_v_;

        // Fast VB step on u=log π:  ∂F/∂u = ½ eᵘ r_b − d/2 + (u − g(v))/σ_u²  (minimize ⇒ u -= lr·∂F/∂u).
        const float grad_u = 0.5f * std::exp(u_b_) * r_b - 0.5f * d_dim
                             + (u_b_ - g_v) / params.hier_prec_sigma_u2;
        u_b_ -= params.hier_prec_lr_u * grad_u;
        // Pure numeric guard on exp(u_b_) (≈[2e-9, 5e8]); NOT a behavioural gate — the operating band is
        // a few units around g(v). Prevents inf/denorm if a pathological residual ever appears.
        u_b_ = std::clamp(u_b_, -20.0f, 20.0f);

        // Slow VB step on the map_trust hyper-state v (Option A):
        //   F_v = (u − g(v))²/(2σ_u²) + v²/(2σ_v²),  ∂F_v/∂v = −g_gain·(u − g(v))/σ_u² + v/σ_v².
        const float grad_v = -params.hier_prec_g_gain * (u_b_ - g_v) / params.hier_prec_sigma_u2
                             + map_trust_v_ / params.hier_prec_sigma_v2;
        map_trust_v_ -= params.hier_prec_lr_v * grad_v;

        // A/B trace: boundary_weight is exp(u_b_) AFTER this update (i.e. what next frame will apply).
        log_hier_prec_row("opt", r_b, quad, std::max(0.0f, trace), /*reloc_fired=*/false);
    }


    // =========================================================================
    //  Backend-shared helpers
    // =========================================================================
    float RoomConcept::boundary_weight_now() const
    {
        if (params.hier_prec_boundary_enabled)
            return std::exp(u_b_);
        if (params.rfe_boundary_quality_gate and prev_sdf_mse_ > 1e-6f)
        {
            // prev_sdf_mse_ became a MEDIAN absolute residual on 2026-08-13 (it was a mean on the
            // early-exit path, which is >98% of frames). A median of a half-normal is 0.674*sigma
            // against the mean's 0.798*sigma, so the denominator shrank by kMedianOverMeanAbs and
            // this weight would otherwise have risen by 1/0.845 = 1.18 across the board. The factor
            // restores the calibration; it is applied here rather than to sigma_sdf because
            // sigma_sdf also sets the early-exit gate, which still tests the mean and must not move.
            const float sigma2 = params.sigma_sdf * params.sigma_sdf;
            return std::min(1.0f, kMedianOverMeanAbs * sigma2 / prev_sdf_mse_);
        }
        return 1.0f;
    }

    std::vector<Eigen::Vector3f> RoomConcept::read_window_poses() const
    {
        std::vector<Eigen::Vector3f> out;
        out.reserve(window_mgr_.window.size());
        for (const auto& slot : window_mgr_.window)
            out.push_back(slot.pose);
        return out;
    }

    void RoomConcept::write_window_poses(const std::vector<Eigen::Vector3f>& poses)
    {
        if (poses.size() != window_mgr_.window.size()) return;
        for (size_t i = 0; i < poses.size(); ++i)
            window_mgr_.window[i].pose = poses[i];
    }

    std::pair<float, int> RoomConcept::run_gn_loop(const OdometryPrior& odometry_prior)
    {
        // velocity_adaptive_weights is deliberately not applied here: it preconditions Adam's
        // gradient, and a curvature-correct step does not want a preconditioner. Same minimum.
        gn::Input in;
        in.model           = model_.get();
        in.params          = &params;
        in.window          = &window_mgr_.window;
        in.boundary_prior  = &window_mgr_.boundary_prior;
        in.boundary_weight = boundary_weight_now();
        // ── ONCE THE LAYOUT IS FROZEN, LOCALISE THE WAY GIVEN MODE DOES ──────────────────────────
        // While the room is being learnt the walls ARE the estimate, so the solver takes them as
        // landmarks, runs with no_sdf (the polygon is DERIVED from those same walls — scoring a pose
        // against it would be scoring the walls against themselves) and needs a gauge, because the
        // map frame is otherwise unobservable.
        // None of those three reasons survives the freeze. The walls no longer move, so they are no
        // longer being estimated; the polygon is no longer derived from anything live, so its SDF is
        // an independent measurement of the pose exactly as a loaded layout's is; and the frame is
        // pinned by the re-anchor, so there is nothing left to gauge-fix. From here the room is GIVEN
        // in the full sense rather than in name, and the pose is solved against its SDF — the same
        // channel, the same terms and the same constants a layout loaded from file uses.
        // Door apertures go to BOTH regimes: a through-the-door return is just as wrong for the
        // wall landmarks as for the room's SDF.
        in.doors_robot = &doors_robot_;
        if (searching())
        {
            in.walls             = &wall_map_;
            in.no_sdf            = true;     // the polygon is derived from these walls: no SDF on it
            in.gauge_fix         = true;     // first-pose gauge until the boundary prior exists
            in.gauge_sigma_xy    = params.wall_gauge_sigma_xy;
            in.gauge_sigma_theta = params.wall_gauge_sigma_theta;
            in.wall_max_slots    = params.wall_max_slots;
        }

        gn::Options opts;
        opts.max_iters    = params.gn_max_iters;
        opts.lambda_init  = params.gn_lambda_init;
        opts.step_tol     = params.gn_step_tol;
        opts.loss_rel_tol = params.gn_loss_rel_tol;

        // ── Private landmark variables ──────────────────────────────────────────────────────────────
        // One per distinct object anchor in the window. Born from the producing agent's belief (mean +
        // Σ_o as the prior), then refined by room's observations alone and NEVER written back to the
        // graph. The prior is spent on the birth frame: Σ_o came through the robot pose, so re-applying
        // it every frame would keep feeding the same evidence back and shrink both estimates without
        // new information (the incest that makes two agents confidently agree on the wrong thing).
        std::vector<gn::Landmark> landmarks;
        if (params.object_anchor_optimize_landmark and params.object_anchor.enable)
        {
            std::scoped_lock lk(object_anchors_mutex_);
            for (const auto& slot : window_mgr_.window)
                for (const auto& a : slot.object_anchors)
                {
                    if (std::ranges::any_of(landmarks, [&](const auto& l) { return l.id == a.node_id; }))
                        continue;
                    gn::Landmark lm;
                    lm.id = a.node_id;
                    auto it = landmark_estimates_.find(a.node_id);
                    if (it == landmark_estimates_.end())
                    {
                        const Eigen::Matrix2f sig = a.map_cov.topLeftCorner<2, 2>();
                        if (not sig.allFinite() or sig(0, 0) <= 0.f or sig(1, 1) <= 0.f)
                            continue;                       // no usable birth prior yet
                        lm.p                 = a.pose_world.head<2>();
                        lm.prior_mean        = lm.p;
                        lm.prior_information = sig.inverse();
                        lm.information       = lm.prior_information;
                        lm.has_prior         = true;        // birth frame only
                        qInfo() << "[room][landmark] born" << QString::fromStdString(a.type)
                                << "#" << static_cast<qulonglong>(a.node_id)
                                << "at (" << lm.p.x() << "," << lm.p.y() << ") prior σ="
                                << std::sqrt(std::max(sig(0, 0), sig(1, 1))) * 1000.f << "mm";
                    }
                    else
                    {
                        lm.p           = it->second.p;
                        lm.information = it->second.information;
                        lm.has_prior   = false;
                    }
                    landmarks.push_back(lm);
                }
            in.landmarks = landmarks.empty() ? nullptr : &landmarks;
        }

        auto poses = read_window_poses();
        const auto r = gn::solve(in, poses, opts);
        if (not r.ok)
        {
            // Singular system / non-finite loss: keep the window as predicted rather than hand the caller a
            // NaN pose. (This used to fall back to Adam; it fired 0 times in try42-45, and Adam went with
            // torch on 2026-10-17.) One iteration, not zero: zero would read as a prediction early-exit.
            qWarning() << "[gn] solve failed (loss" << r.loss_init << ") — keeping the predicted window";
            return {r.loss_init, 1};
        }

        write_window_poses(poses);
        if (in.landmarks != nullptr)
        {
            std::scoped_lock lk(object_anchors_mutex_);
            for (const auto& lm : landmarks)
                landmark_estimates_[lm.id] = {lm.p, lm.information};
        }
        // ── What this window says about the newest pose, both ways ──────────────────────────────
        // Taken HERE because this is the only place the assembled Input and the converged poses
        // exist together; recomputing either elsewhere would be comparing two different problems.
        // One extra linearize (~0.3 ms) on an OPTIMISED frame only, and those are ~0.7% of frames.
        if (params.hessian_check or params.covariance_from_solver)
        {
            const auto nm = gn::newest_pose_marginal(in, poses);
            last_marg_ok_         = nm.ok;
            last_marg_prec_       = nm.marginal;
            last_block_prec_      = nm.block;
            last_marg_dof_        = nm.n_marginalised;
            last_gn_window_slots_ = static_cast<int>(poses.size());
        }
        else last_marg_ok_ = false;
        last_lbfgs_grad_norm_ = r.grad_norm;
        last_adam_losses_.clear();
        last_adam_losses_.push_back(r.loss_init);
        last_adam_losses_.push_back(r.loss);
        last_loss_init_ = r.loss_init;
        return {r.loss, r.iterations};
    }

    std::pair<Eigen::Matrix3f, float> RoomConcept::compute_posterior_covariance(
        const ScanPoints& scan_pts)
    {
        // Estimate mode: the only covariance is the window's own (the GN marginal). The autograd
        // fallback below takes a Hessian of the SDF against the polygon, which here is DERIVED from
        // the very walls the solve just moved — or a placeholder before closure. Keep the predicted one.
        if (estimating() and not (params.covariance_from_solver and last_marg_ok_))
            return {current_covariance, 1.f};

        // ── THE WINDOW'S OWN ANSWER, WHEN THE SOLVER HAS ONE ────────────────────────────────────
        // The GN backend assembles H = sum J'WJ over the whole window and solves it. Λ_marg is the
        // newest pose's marginal out of exactly that system — every other pose and landmark
        // Schur-complemented away — so it already contains the boundary prior, the motion chain and
        // every slot's observations. It IS the posterior the estimator just computed.
        //
        // What it replaces is a SECOND estimator layered on the first: prior = last posterior, plus
        // a double-backward autograd Hessian of the newest slot alone. Three things were wrong with
        // that, all measured 2026-08-29:
        //   • 108 ms of a 131 ms optimised frame, against 18 ms for the solve itself — a Hessian
        //     built twice, the second time the expensive way (optimizer_timing.csv);
        //   • it RATCHETED. The recursion multiplied the same window's observations in every frame,
        //     so the published sigma_x reached 1.6 mm against the single slot's 1.9 and the window's
        //     own 5.5 — tighter than either input. A marginal cannot do that;
        //   • it needed the guard that started this: `det > 1e-10` refused 89 of 102 updates because
        //     a determinant cannot tell a PRECISE covariance from a SINGULAR one.
        // Between solves the covariance still inflates through predict_step(), which is where a
        // prediction step belongs. The recursion is gone; the propagation is not.
        // ★ The GN Hessian is the Gauss-Newton approximation, so it drops second-order terms the
        //   autograd one keeps. hessian_check.csv logs both, which is how this was measured and how
        //   a regression would be seen; CovarianceFromSolver = false restores the old path.
        if (params.covariance_from_solver and last_marg_ok_)
        {
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig(last_marg_prec_);
            const Eigen::Vector3f ev = eig.eigenvalues().cwiseMax(params.eigenvalue_clamp_posterior);
            const Eigen::Matrix3f prec = eig.eigenvectors() * ev.asDiagonal() * eig.eigenvectors().transpose()
                                       + params.covariance_regularization * Eigen::Matrix3f::Identity();
            const Eigen::Matrix3f P = prec.inverse();
            const float cond = ev.maxCoeff() / std::max(1e-8f, ev.minCoeff());
            hess_pred_cov_   = current_covariance;      // P_pred, for the NIS column
            hess_obs_prec_   = last_marg_prec_;         // "obs" is now the window marginal
            hess_cov_det_    = static_cast<double>(P.determinant());
            hess_cov_cond_   = static_cast<double>(cond);
            const bool ok = P.allFinite() and ev.minCoeff() > 0.f and cond < params.condition_number_max;
            hess_cov_accept_ = ok ? 1 : (not P.allFinite() ? -1 : ev.minCoeff() > 0.f ? -3 : -2);
            if (ok) { current_covariance = P; return {P, cond}; }
            return {current_covariance, cond};
        }

        try {
            // P_pred, kept before the update overwrites it. This is what predict_step() left in
            // current_covariance, i.e. F·P_prev·Fᵀ + Q — the quantity the innovation has to be
            // consistent with, and the one the process noise actually sets.
            hess_pred_cov_ = current_covariance;
            auto& newest = window_mgr_.newest();
            // Gauss-Newton information of this scan — the solver's own linearization. (An autograd
            // Hessian of the same loss until 2026-10-17, which also kept the SDF's curvature.)
            Eigen::Matrix3f H_likelihood = linearize_observation(*model_, params, scan_pts, newest.pose).H;

            // The RGB edge term is a LIKELIHOOD, so once it drives the pose it must also inform the
            // reported covariance — otherwise the agent would act on evidence it does not admit to
            // having, and sigma_theta (the pre-registered success criterion, and what feeds the
            // controller's speed governor and every concept agent's precision) could never move.
            // ★ Gated on DRIVE, not merely enable: shadow mode must not change any published number.
            // Its Hessian is taken by central differences of the same loss (it has no standalone
            // analytic form outside the GN factor list; 3 DOF, 19 evaluations).
            if (params.image_edge.enable and params.image_edge.drive and not newest.image_edges.empty())
            {
                const auto L = [&](const Eigen::Vector3f& x)
                { return static_cast<double>(ImageEdgeFactor::loss(newest.image_edges, x, params.image_edge)); };
                const Eigen::Vector3f h(1e-3f, 1e-3f, 1e-3f);   // m, m, rad
                const Eigen::Vector3f x0 = newest.pose;
                const double f0 = L(x0);
                for (int i = 0; i < 3; ++i)
                    for (int j = i; j < 3; ++j)
                    {
                        Eigen::Vector3f ei = Eigen::Vector3f::Zero(), ej = Eigen::Vector3f::Zero();
                        ei[i] = h[i]; ej[j] = h[j];
                        const double hij = (i == j)
                            ? (L(x0 + ei) - 2.0 * f0 + L(x0 - ei)) / (double(h[i]) * h[i])
                            : (L(x0 + ei + ej) - L(x0 + ei - ej) - L(x0 - ei + ej) + L(x0 - ei - ej))
                              / (4.0 * h[i] * h[j]);
                        H_likelihood(i, j) += static_cast<float>(hij);
                        if (i != j) H_likelihood(j, i) += static_cast<float>(hij);
                    }
            }
            hess_obs_prec_ = H_likelihood;   // this slot's observation alone, before the clamp

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig(H_likelihood);
            Eigen::Vector3f evals = eig.eigenvalues().cwiseMax(params.eigenvalue_clamp_posterior);
            H_likelihood = eig.eigenvectors() * evals.asDiagonal() * eig.eigenvectors().transpose();

            Eigen::Matrix3f prior_precision = current_covariance.inverse();
            const float lambda = params.covariance_regularization;
            Eigen::Matrix3f posterior_precision = prior_precision + H_likelihood
                                                 + lambda * Eigen::Matrix3f::Identity();

            Eigen::Matrix3f new_cov = posterior_precision.inverse();

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(posterior_precision);
            const auto eigenvalues = solver.eigenvalues();
            const float max_ev = eigenvalues.maxCoeff();
            const float min_ev = eigenvalues.minCoeff();
            const float cond = (min_ev > 1e-8f) ? (max_ev / min_ev) : 1e8f;

            // ── ACCEPT THE UPDATE, OR SAY WHICH CONDITION REFUSED IT ────────────────────────────
            // ★ 2026-08-29: the determinant test that used to stand here REFUSED 89 of 102 updates,
            //   so the published sigma never absorbed an observation — it was F·P·Fᵀ + Q compounding
            //   and nothing else. For a 3x3 pose covariance det = (sx·sy·st)², so
            //   `det > covariance_det_min` (1e-10) forbade anything tighter than sigma = 21.5 mm on
            //   equal axes, while the observation alone supports ~2.9 mm — refused by 3e6.
            //   A SINGULAR covariance and a PRECISE one both have a small determinant, and a
            //   determinant cannot tell them apart. It was rejecting precision as degeneracy, and it
            //   is the root cause of a sigma that stayed at 0.086 through a 3.97 m pose jump.
            // The two tests that DO discriminate were already here:
            //   min_ev > 0                    positive-definiteness — a real covariance, not a
            //                                 degenerate one collapsed onto a subspace;
            //   cond < condition_number_max   degeneracy by ratio, which is scale-free and therefore
            //                                 says nothing about how precise the estimate is.
            // Together they reject exactly the matrices that are unusable and none that are merely
            // small, which is the distinction the old test could not make.
            hess_cov_det_  = static_cast<double>(new_cov.determinant());
            hess_cov_cond_ = static_cast<double>(cond);
            const bool finite_ok = new_cov.allFinite();
            const bool pd_ok     = min_ev > 0.f;
            const bool cond_ok   = cond < params.condition_number_max;
            hess_cov_accept_ = not finite_ok ? -1 : not pd_ok ? -2 : not cond_ok ? -3 : 1;
            if (finite_ok and pd_ok and cond_ok)
            {
                current_covariance = new_cov;
                return {new_cov, cond};
            }
            return {current_covariance, cond};
        } catch (const std::exception &e) {
            std::cerr << "RFE covariance update failed: " << e.what() << std::endl;
            return {current_covariance, -1.0f};
        }
    }

} // namespace rc
