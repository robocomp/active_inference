#pragma once
// Split out of room_concept.h (refactor step 3, docs/refactor_proposal_2026-10-13.md). The types live in
// rc::loc and RoomConcept re-exports each under its old name (RoomConcept::<same name>), so no caller changed.
// Contents: WindowSlot, BoundaryPrior, WindowManager.

#include <cstdint>
#include <deque>
#include <optional>
#include <vector>
#include <Eigen/Dense>
#include "room_model.h"
#include "door_apertures.h"
#include "image_edge_types.h"
#include "object_anchor_types.h"
#include "wall_map.h"
#include "localizer_params.h"

namespace rc::loc
{
    // ===== Sliding Window (RFE) types =====
    struct WindowSlot
    {
        Eigen::Vector3f pose = Eigen::Vector3f::Zero();   // {x, y, theta}
        ScanPoints      lidar_points;                     // stored observation, robot frame
        Eigen::Vector3f odometry_delta = Eigen::Vector3f::Zero(); // delta from prev slot
        Eigen::Matrix3f motion_cov = Eigen::Matrix3f::Identity(); // Σ_dyn
        Eigen::Matrix3f pose_cov = Eigen::Matrix3f::Identity() * 0.1f;   // pose covariance at slot creation (publish-test marginal)
        int64_t timestamp_ms = 0;

        Eigen::Matrix3f motion_prec = Eigen::Matrix3f::Identity();   // Σ_dyn^{-1}, set at append time
        // Precision of the "this interval had no motion" hypothesis, for the ZERO-delta motion factor.
        // Empty unless Params::zupt_as_factor — see NoiseModel::zupt_as_factor.
        std::optional<Eigen::Matrix3f> zupt_prec;
        bool subsampled = false;           // true once old-slot subsampling has been applied

        // Quality of this slot's localization (set after Adam/early-exit, read when slot
        // becomes the oldest and its Hessian is used for the boundary prior).
        float sdf_mse_final = 0.0f;

        // Corner observations for this slot (robot frame)
        struct CornerObs {
            Eigen::Vector2f model_corner_world;  // world-frame model corner
            Eigen::Vector2f detected_robot;      // detected position in robot frame
            // Graded per-detection precision Λ_det (robot frame) from CornerDetector: the loss applies
            // this anisotropically (rᵀΛ_det r), so shallow/marginal corners contribute weakly along
            // ill-determined directions and clean asymmetric corners contribute strongly. Rank-1 for a
            // near-parallel (aperture-ambiguous) corner. Replaces the old scalar corner_obs_sigma·I.
            Eigen::Matrix2f information = Eigen::Matrix2f::Identity();
        };
        std::vector<CornerObs> corner_obs;

        // Wall-SLAM (MapMode = Estimate): this slot's observations of the wall landmarks, in ITS robot
        // frame, as the segmenter/associator left them. Consumed by gn::WallPointFactor; folded into
        // the walls' carried information when the slot is dropped (gn::absorb_wall_observations).
        std::vector<wallmap::WallAssoc> wall_assoc;

        // Object-anchor observations for this slot (validated modelled objects as SE(2) landmarks).
        std::vector<ObjectAnchorObs> object_anchors;
        // RGB edge evidence for this slot. The normal search already happened ONCE, on the compute
        // thread at the pre-solve pose, so uv_meas is a CONSTANT of the solve — required, or LM's
        // accept/reject test tracks a moving target (see image_edge_source.h).
        ImageEdgeObs                 image_edges;
    };

    struct BoundaryPrior
    {
        // Legacy (boundary_fej_schur=false): mu = dropped state's MAP pose, precision = quality-gated
        // Hessian, grad unused. Boundary loss = 0.5·(x_front−mu)ᵀ·precision·(x_front−mu).
        //
        // FEJ+Schur (boundary_fej_schur=true): mu = FROZEN marginal MODE (mean form) of the surviving
        // slot, computed from the unclamped Schur complement as mu = x₁* − Λ_marg⁻¹·g_marg (≈ x₀*+odom);
        // precision = Λ_marg = Ω − ΩΛ₀₀⁻¹Ω, eigenvalue-clamped AFTER the mode is fixed (so clamping only
        // softens confidence, never moves the anchor). Same pure-quadratic loss 0.5·Δᵀ·precision·Δ,
        // Δ = wrap(x_front − mu). mu is NOT updated between marginalizations (the FEJ discipline).
        Eigen::Vector3f mu = Eigen::Vector3f::Zero();            // dropped-state pose (legacy) / frozen marginal mode (Schur)
        Eigen::Matrix3f precision = Eigen::Matrix3f::Identity(); // Σ^{-1} (legacy Hessian) / Λ_marg (Schur)
        Eigen::Vector3f grad = Eigen::Vector3f::Zero();          // vestigial (mean form uses no linear term)
        bool valid = false;
    };

   // ===== Window Manager (RFE sliding window) =====
   struct WindowManager
   {
       std::deque<WindowSlot> window;
       BoundaryPrior boundary_prior;
       /// The doorways in the NEWEST slot's robot frame (RoomConcept::doors_robot_), for its SDF term. Set by the
       /// owner each cycle; null ⇒ no door term.
       const std::vector<DoorAperture>* doors_newest_robot = nullptr;

       bool empty() const { return window.empty(); }
       size_t size() const { return window.size(); }
       void clear() { window.clear(); boundary_prior.valid = false; }
       WindowSlot& newest() { return window.back(); }
       const WindowSlot& newest() const { return window.back(); }

       /// Slide if full, append new slot.  Returns true if window was slid.
       /// mu_quality_threshold: only update boundary_prior.mu from the dropped slot if its
       /// sdf_mse_final is below this value (Solution C — legacy path only).
       /// fej_schur: when true, do NOT touch boundary_prior here — marginalize_oldest() (called
       /// BEFORE append) has already folded the dropped slot into the FEJ+Schur prior.
       bool append(WindowSlot slot, int max_window_size,
                   float mu_quality_threshold = std::numeric_limits<float>::max(),
                   bool fej_schur = false);

       /// Subsample lidar in all slots except the newest.
       void subsample_old_slots(int max_pts_per_slot);

       /// Per-term loss breakdown — called once after Adam for diagnostic logging.
       struct LossBreakdown {
           float boundary = 0.f;
           float obs      = 0.f;
           float motion   = 0.f;
           float corner   = 0.f;
           float object   = 0.f;
           float image    = 0.f;   ///< RGB structural-contour term (0 unless ImageEdge.enable)
       };
       LossBreakdown compute_rfe_loss_breakdown(const Model& model, const Params& params) const;

       /// Recompute boundary prior Hessian from oldest surviving slot (LEGACY path).
       void recompute_boundary_prior(const Model& model, const Params& params);

       /// FEJ + Schur marginalization (boundary_fej_schur path). Call BEFORE append() pops, while
       /// both the dropping slot (window.front() = x₀) and its blanket (window[1] = x₁) are live.
       /// Forms the exact Schur complement of x₀ over x₁ from {previous marginal on x₀, x₀'s obs
       /// factor (soft quality-weighted), x₀↔x₁ motion factor}, and stores a FEJ-frozen prior on x₁.
       /// No-op unless the window is full (>= max_window_size) with size > 1.
       void marginalize_oldest(const Model& model, const Params& params);
   };
} // namespace rc::loc
