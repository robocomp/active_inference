/*
 *    Copyright (C) 2026 by RoboLab at the University of Extremadura
 *    This file is part of RoboComp
 *
 *    RoboComp is free software: you can redistribute it and/or modify it under
 *    the terms of the GNU General Public License as published by the Free
 *    Software Foundation, either version 3 of the License, or (at your option)
 *    any later version. See <http://www.gnu.org/licenses/>.
 */

#pragma once

/*
 *  image_edge_factor.h — the torch/autograd side of the RGB edge-alignment subsystem.
 *
 *  The MIRROR of the GN factor in room_gn_solver.cpp. It must evaluate the same objective, term for
 *  term, or two things break at once: LM's accept/reject test measures a different function than the
 *  one it linearizes, and compute_posterior_covariance() — which differentiates compute_rfe_loss
 *  through autograd — reports a sigma that omits this term entirely. The second matters more than
 *  the shadow log, because the pre-registered success criterion IS that sigma.
 *
 *  Pure torch — no DSR, no image, no OpenCV (see image_edge_types.h for the rationale). The pixel
 *  search has already happened upstream on the compute thread; what arrives here is a fixed set of
 *  (p_room, n_hat, uv_meas, sigma, pi_vis, h) tuples, i.e. plain measurements.
 */

#include <torch/torch.h>
#include <vector>

#include "image_edge_types.h"

namespace rc
{
    class ImageEdgeFactor
    {
    public:
        struct Params
        {
            bool  enable = false;   ///< master switch for the family
            bool  drive  = false;   ///< does it enter the GN factor list (vs shadow-only)?
            float search_sigmas = 3.0f;         ///< Gaussian truncation of the derived search window
            float mount_pitch_sigma  = 0.0035f; ///< rad — shared nuisance priors. PHYSICAL, not knobs:
            float mount_height_sigma = 0.010f;  ///< m     they are how well the mount is known, and they
            float mount_yaw_sigma    = 0.0035f; ///< rad   are what caps a wall at ~one observation.
            /// NOTE: deliberately NO `weight`. If this term needs a hand-set scalar to behave, its
            /// covariance model is wrong. Do not add one — see room_config.h.
        };

        /// Sum of RGB edge factors for ONE slot's robot pose.
        ///   pose_xy    : [2] (x, y), requires_grad
        ///   pose_theta : [1] or scalar (yaw), requires_grad
        /// Returns a scalar loss on `device`. Zero when disabled or when there is no evidence.
        ///
        /// The loss is the EM SURROGATE with the mixture responsibilities gamma FROZEN at the values
        /// computed by the caller for this linearisation — NOT the true mixture NLL. That choice is
        /// what lets evaluate() and linearize() use one identical expression and avoids the
        /// IRLS-weight-vs-loss-weight split that Se2LandmarkFactor:274-286 records as a real past bug.
        static torch::Tensor loss(const ImageEdgeObs& obs,
                                  const torch::Tensor& pose_xy,
                                  const torch::Tensor& pose_theta,
                                  const Params& params,
                                  torch::Device device);
    };
}  // namespace rc
