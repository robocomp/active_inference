/*
 *  object_anchor_factor.h — the torch/autograd side of the object-anchor subsystem.
 *
 *  Adds one SE(2) landmark factor per validated object to the room localizer's RFE loss.
 *  It is the direct generalisation of the existing corner factor in compute_rfe_loss():
 *  same predicted-observation geometry z_hat = R(-θ)·(p_o − t), but with an orientation
 *  channel and a full 3×3 innovation precision Λ (from the object's belief covariance)
 *  instead of a scalar corner σ.  Autograd supplies the Jacobian wrt the robot pose.
 *
 *  Pure torch — no DSR here (see object_anchor_types.h for the rationale).
 */
#pragma once

#include <torch/torch.h>
#include <vector>

#include "object_anchor_types.h"

namespace rc
{
    class ObjectAnchorFactor
    {
    public:
        struct Params
        {
            bool  enable      = false;  // master switch for the object-anchor factor family
            float weight      = 1.0f;   // global scale (keep < walls: objects refine, never outvote geometry)
            float huber_delta = 3.0f;   // robust cutoff in WHITENED (σ) units — bounds a confident-but-wrong fit
        };

        /// Sum of SE(2) landmark factors for ONE slot's robot pose.
        ///   pose_xy    : [2]  (x, y),  requires_grad
        ///   pose_theta : [1] or scalar (yaw), requires_grad
        /// Returns a scalar loss on `device`.  No-op (zero) when disabled or no anchors.
        static torch::Tensor loss(const std::vector<ObjectAnchorObs>& anchors,
                                  const torch::Tensor& pose_xy,
                                  const torch::Tensor& pose_theta,
                                  const Params& params,
                                  torch::Device device);
    };
} // namespace rc
