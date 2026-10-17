// RoomConcept, split out of room_concept.cpp (refactor step 2, docs/refactor_proposal_2026-10-13.md).
// The sliding window: RoomConcept::WindowManager.

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

    // =====================================================================
    //  WindowManager methods
    // =====================================================================

    bool RoomConcept::WindowManager::append(WindowSlot slot, int max_window_size,
                                             float mu_quality_threshold, bool fej_schur)
    {
        bool slid = false;
        if (static_cast<int>(window.size()) >= max_window_size)
        {
            // FEJ+Schur path: marginalize_oldest() (called before this) already folded the dropping
            // slot into the boundary prior at a FROZEN linearization point. Touching mu here would be
            // exactly the non-FEJ re-anchoring we are removing — so just pop.
            if (not fej_schur)
            {
                // Solution C: only update boundary mu if the dropped slot had acceptable
                // localization quality. If the slot was confused (displacement, obstacle),
                // keep the previous mu so the prior continues anchoring to the last good pose.
                const bool slot_is_good = (window.front().sdf_mse_final < mu_quality_threshold)
                                          || !boundary_prior.valid;
                if (slot_is_good)
                    boundary_prior.mu = window.front().pose;
            }

            window.pop_front();
            slid = true;
        }
        window.push_back(std::move(slot));
        return slid;
    }

    void RoomConcept::WindowManager::subsample_old_slots(int max_pts_per_slot)
    {
        if (window.size() <= 1 || max_pts_per_slot <= 0) return;
        for (size_t i = 0; i < window.size() - 1; i++)
        {
            auto& slot = window[i];
            if (slot.subsampled) continue;   // already done — skip
            const auto n_pts = slot.lidar_points.size();
            if (n_pts > static_cast<std::size_t>(max_pts_per_slot))
            {
                const auto stride = n_pts / static_cast<std::size_t>(max_pts_per_slot);
                ScanPoints kept;
                kept.reserve(n_pts / stride + 1);
                for (std::size_t k = 0; k < n_pts; k += stride) kept.push_back(slot.lidar_points[k]);
                slot.lidar_points = std::move(kept);
            }
            slot.subsampled = true;
        }
    }

    RoomConcept::WindowManager::LossBreakdown
    RoomConcept::WindowManager::compute_rfe_loss_breakdown(const Model& model, const Params& params) const
    {
        LossBreakdown bd;
        if (window.empty()) return bd;
        const auto wrap = [](float a) { return std::atan2(std::sin(a), std::cos(a)); };

        // 1. Boundary prior
        if (boundary_prior.valid) {
            Eigen::Vector3f diff = window.front().pose - boundary_prior.mu;
            diff[2] = wrap(diff[2]);
            bd.boundary = 0.5f * diff.dot(boundary_prior.precision * diff);
        }

        // 2. Observation factors
        float obs_acc = 0.f;
        for (const auto& slot : window) {
            if (params.sdf_current_slot_only && &slot != &window.back())
                continue;
            obs_acc += compute_observation_loss(model, params, slot.lidar_points, slot.pose,
                                                &slot == &window.back() ? doors_newest_robot : nullptr);
        }
        bd.obs = obs_acc;

        // 3. Motion factors
        float mot_acc = 0.f;
        for (size_t i = 1; i < window.size(); i++) {
            const auto& curr = window[i];
            Eigen::Vector3f r = (curr.pose - window[i - 1].pose) - curr.odometry_delta;
            r[2] = wrap(r[2]);
            mot_acc += 0.5f * r.dot(curr.motion_prec * r);
        }
        bd.motion = mot_acc;

        // 4. Corner factors
        if (params.enable_corner_tracking) {
            const float corner_gain  = params.corner_precision_gain;
            const float corner_huber = params.corner_huber_sigma;
            const int corner_start = std::max(0, static_cast<int>(window.size()) - params.corner_max_slots);
            float cor_acc = 0.f;
            for (size_t si = corner_start; si < window.size(); si++) {
                const auto& slot = window[si];
                const float cos_th = std::cos(slot.pose.z()), sin_th = std::sin(slot.pose.z());
                for (const auto& obs : slot.corner_obs) {
                    const Eigen::Vector2f dw = obs.model_corner_world - slot.pose.head<2>();
                    const Eigen::Vector2f predicted(cos_th * dw.x() + sin_th * dw.y(),
                                                    -sin_th * dw.x() + cos_th * dw.y());
                    const Eigen::Vector2f residual = obs.detected_robot - predicted;
                    const float maha_sq = corner_gain * residual.dot(obs.information * residual);
                    const float m = std::sqrt(maha_sq + 1e-8f);
                    const float hw = m <= corner_huber ? 1.f : corner_huber / m;
                    cor_acc += 0.5f * hw * maha_sq;
                }
            }
            bd.corner = cor_acc;
        }

        // 5. Object-anchor factors
        if (params.object_anchor.enable) {
            const int obj_start = std::max(0, static_cast<int>(window.size()) - params.object_anchor_max_slots);
            float obj_acc = 0.f;
            for (size_t si = obj_start; si < window.size(); si++)
                obj_acc += ObjectAnchorFactor::loss(window[si].object_anchors, window[si].pose, params.object_anchor);
            bd.object = obj_acc;
        }

        // 6. RGB structural contours — same terms as the GN factor list.
        if (params.image_edge.enable and params.image_edge.drive) {
            const int img_start = std::max(0, static_cast<int>(window.size()) - std::max(1, params.image_edge_max_slots));
            float img_acc = 0.f;
            for (size_t si = img_start; si < window.size(); si++)
                img_acc += ImageEdgeFactor::loss(window[si].image_edges, window[si].pose, params.image_edge);
            bd.image = img_acc;
        }

        return bd;
    }

    void RoomConcept::WindowManager::recompute_boundary_prior(
        const Model& model, const Params& params)
    {
        if (window.empty())
            return;

        const auto& oldest = window.front();

        // ── mu update (Solution C already applied in append()) ────────────────
        // By this point boundary_prior.mu was already conditionally updated when the
        // slot was dropped from the window. We only need to update it here for the
        // "recompute after recovery / reset" path where append() was not called.
        // Unconditional write is safe: recompute is only called when window_slid==true
        // and the slot that slid out already had its quality checked in append().
        // (No additional guard needed here — the pose in mu was set correctly.)

        // ── Hessian computation (Solution B) ─────────────────────────────────
        // If the oldest slot's scan was of poor quality (sdf_mse_final exceeds the
        // threshold), the H_obs term would encode a high-confidence direction toward
        // a contaminated pose. In that case we use only the kinematic (motion) factor
        // for the precision matrix, which is always trustworthy.
        const bool use_obs_hessian =
            (oldest.sdf_mse_final < params.boundary_hessian_quality_threshold);

        Eigen::Matrix3f H = Eigen::Matrix3f::Zero();

        if (use_obs_hessian)
        {
            // ── Full path: H_obs + H_motion + H_prior ───
            // H_obs is the Gauss-Newton information of the oldest scan (the solver's own linearization);
            // the motion and prior terms are quadratic, so their Hessians are exact: Ω (J = −I wrt the
            // oldest pose, wrap' = 1) and the previous prior's precision. Until 2026-10-17 this was an
            // autograd Hessian of the same loss, which also kept the SDF's second-order curvature.
            H = linearize_observation(model, params, oldest.lidar_points, oldest.pose).H;
            if (window.size() > 1)
                H += window[1].motion_prec;
            if (boundary_prior.valid)
                H += boundary_prior.precision;
        }
        else
        {
            // ── Degraded path: kinematic precision only ───────────────────────
            // H_obs is not included because the scan was contaminated (obstacle,
            // displacement confusion). Use only the motion-factor precision from the
            // next slot, which reflects purely the odometry model uncertainty.
            // This gives a conservative, direction-agnostic anchor.
            if (window.size() > 1)
                H = window[1].motion_prec;
            else
            {
                // No motion link available: fall back to a weak isotropic prior.
                H = Eigen::Matrix3f::Identity() * params.eigenvalue_clamp_boundary;
            }
        }

        // ── Eigenvalue clamping (floor + ceiling) ─────────────────────────────
        // Floor prevents degenerate (zero-precision) directions.
        // Ceiling (eigenvalue_clamp_boundary_max) prevents over-confident priors
        // regardless of cause — acts as a safety net for both paths.
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig(H);
        Eigen::Vector3f evals = eig.eigenvalues()
            .cwiseMax(params.eigenvalue_clamp_boundary)
            .cwiseMin(params.eigenvalue_clamp_boundary_max);
        H = eig.eigenvectors() * evals.asDiagonal() * eig.eigenvectors().transpose();

        boundary_prior.precision = H;
        boundary_prior.valid = true;
    }

    // =====================================================================
    //  FEJ + Schur marginalization of the dropping slot (boundary_fej_schur)
    //
    //  Called BEFORE append() pops, with x₀ = window.front() (dropping) and
    //  x₁ = window[1] (its Markov blanket via the motion factor). Forms the exact
    //  Schur complement of x₀ over x₁ from the factors touching x₀ — the previous
    //  marginal prior on x₀, x₀'s (soft quality-weighted) obs factor, and the
    //  x₀↔x₁ motion factor — and stores a First-Estimates-Jacobian-frozen prior on x₁.
    //
    //     Λ₀₀ = Λ_prev + w₀·H_obs(x₀*) + Ω     Λ₀₁ = Λ₁₀ = −Ω     Λ₁₁ = Ω
    //     g₀  = Λ_prev·Δ_prev + g_prev + w₀·g_obs − Ω·r          g₁ = +Ω·r
    //     Λ_marg = Ω − Ω Λ₀₀⁻¹ Ω               g_marg = Ω·r + Ω Λ₀₀⁻¹ g₀
    //
    //  All Jacobians are evaluated at the CURRENT (converged-from-previous-frame)
    //  estimates x₀*, x₁* and then frozen — that fixed linearization point is what
    //  prevents the "erroneous information gain" ratchet (loss_boundary 0→559).
    // =====================================================================
    void RoomConcept::WindowManager::marginalize_oldest(
        const Model& model, const Params& params)
    {
        // Only meaningful when a slide is imminent (window full) and there IS a blanket slot.
        if (static_cast<int>(window.size()) < params.rfe_window_size || window.size() < 2)
            return;

        const auto& x0slot = window.front();   // dropping
        const auto& x1slot = window[1];        // Markov blanket (motion-linked survivor)

        auto wrap = [](float a) { return std::atan2(std::sin(a), std::cos(a)); };

        const Eigen::Vector3f x0 = x0slot.pose;   // FEJ linearization points
        const Eigen::Vector3f x1 = x1slot.pose;

        // ── Motion factor x₀↔x₁ (precision Ω, residual r), linearized at (x₀*, x₁*) ──
        // odometry_delta of x₁ is the odometry delta from x₀ to x₁; J₀ = −I, J₁ = +I.
        const Eigen::Matrix3f Omega = x1slot.motion_prec;
        const Eigen::Vector3f odom = x1slot.odometry_delta;
        Eigen::Vector3f r = (x1 - x0) - odom;
        r[2] = wrap(r[2]);

        // ── Dropped slot's obs factor: H_obs, g_obs at x₀*, softly quality-weighted ──
        // w₀ = 1/(1+(sdf/σ_q)²): a poorly-fit dropped slot injects weak, high-covariance
        // information that cannot anchor the survivor (continuous replacement of the hard gate).
        Eigen::Matrix3f H_obs = Eigen::Matrix3f::Zero();
        Eigen::Vector3f g_obs = Eigen::Vector3f::Zero();
        {
            const float sq = std::max(1e-6f, params.boundary_quality_sigma);
            const float ratio = x0slot.sdf_mse_final / sq;
            const float w0 = 1.0f / (1.0f + ratio * ratio);

            // Gauss-Newton information + gradient of the dropped scan, the solver's own linearization
            // (an autograd Hessian until 2026-10-17, which also kept the SDF's curvature).
            const auto lin = linearize_observation(model, params, x0slot.lidar_points, x0);
            g_obs = w0 * lin.g;
            H_obs = w0 * lin.H;
            H_obs = 0.5f * (H_obs + H_obs.transpose().eval());
        }

        // ── Assemble the joint linear system over (x₀, x₁) from factors touching x₀ ──
        Eigen::Matrix3f L00 = H_obs + Omega;     // + Λ_prev below
        Eigen::Matrix3f L01 = -Omega;            // L10 = L01ᵀ = −Ω (symmetric)
        Eigen::Matrix3f L11 = Omega;
        Eigen::Vector3f g0  = g_obs - Omega * r; // motion: J₀ᵀΩr = −Ωr
        Eigen::Vector3f g1  =         Omega * r; // motion: J₁ᵀΩr = +Ωr

        // Previous marginal prior on x₀ (FEJ form 0.5·ΔᵀΛΔ + gᵀΔ, Δ = wrap(x₀* − mu_prev)).
        if (boundary_prior.valid)
        {
            Eigen::Vector3f dprev = x0 - boundary_prior.mu;
            dprev[2] = wrap(dprev[2]);
            L00 += boundary_prior.precision;
            g0  += boundary_prior.precision * dprev + boundary_prior.grad;
        }

        // ── Schur complement: eliminate x₀ ──
        const Eigen::Matrix3f L00inv =
            L00.ldlt().solve(Eigen::Matrix3f::Identity());
        Eigen::Matrix3f Lmarg = L11 - L01.transpose() * L00inv * L01;  // Ω − ΩL₀₀⁻¹Ω
        Eigen::Vector3f gmarg = g1 - L01.transpose() * L00inv * g0;    // Ωr + ΩL₀₀⁻¹g₀

        // ── Convert to (mean, precision) form BEFORE clamping ──
        // The marginal's mode is  mean = x₁* − Λ_marg⁻¹·g_marg  (≈ x₀*+odom, the odometry-propagated
        // anchor). We store the prior as a pure quadratic 0.5·(x−mean)ᵀΛ(x−mean) rather than a frozen
        // point + linear term, because the eigenvalue clamp below scales Λ: clamping Λ while KEEPING
        // g_marg would move the effective mode by Λ_clamped⁻¹·g_marg (with Ω~1e4 that is a ~0.2 m
        // phantom anchor that drags the window off the walls — the bug seen 2026-07-16). Computing the
        // mean from the UNCLAMPED Λ_marg fixes the mode; clamping then only softens the confidence.
        const Eigen::Vector3f mean = x1 - Lmarg.ldlt().solve(gmarg);

        // Eigenvalue clamp (floor + ceiling) for numerical safety, as in the legacy path.
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig(Lmarg);
        Eigen::Vector3f evals = eig.eigenvalues()
            .cwiseMax(params.eigenvalue_clamp_boundary)
            .cwiseMin(params.eigenvalue_clamp_boundary_max);
        Lmarg = eig.eigenvectors() * evals.asDiagonal() * eig.eigenvectors().transpose();

        Eigen::Vector3f mu = mean;
        mu[2] = wrap(mu[2]);
        if (not Lmarg.allFinite() or not mu.allFinite())
            return;   // keep the previous prior rather than poison with NaNs

        boundary_prior.mu        = mu;                       // FROZEN marginal mode (FEJ anchor)
        boundary_prior.precision = Lmarg;
        boundary_prior.grad      = Eigen::Vector3f::Zero();  // mean form ⇒ no separate linear term
        boundary_prior.valid     = true;
    }

} // namespace rc
