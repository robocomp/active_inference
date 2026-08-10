/*
 *  room_obs_weights.h — the SDF observation term, shared by every solver backend.
 *
 *  These two functions used to live in an anonymous namespace inside room_concept.cpp, which was
 *  fine while Adam/L-BFGS were the only consumers. room_gn_solver needs the SAME per-point weights
 *  to build its analytic Jacobian: if the two backends disagreed about the weighting, the shadow
 *  comparison would be measuring that disagreement instead of the solvers. One definition, two
 *  callers — do not re-derive the weights anywhere else.
 *
 *  Definitions are still in room_concept.cpp (they are unchanged); only the linkage moved.
 */
#pragma once

#include <torch/torch.h>

#include "room_concept.h"
#include "room_model.h"

namespace rc
{
    /// Per-point observation weights (range and/or incidence), normalised to mean 1 and DETACHED —
    /// they are constants at the linearization point, which is exactly what IRLS/Gauss-Newton needs.
    torch::Tensor build_observation_weights(const Model& model,
                                            const RoomConcept::Params& params,
                                            const torch::Tensor& points_robot,
                                            const torch::Tensor& pose_theta,
                                            const Model::SdfQueryResult& query);

    /// 0.5·σ_obs⁻²·mean_i(w_i · huber_δ(d_i)) for one slot, given an already-evaluated SDF query.
    torch::Tensor compute_observation_loss_from_query(const Model& model,
                                                      const RoomConcept::Params& params,
                                                      const torch::Tensor& points_robot,
                                                      const torch::Tensor& pose_theta,
                                                      const Model::SdfQueryResult& query);
} // namespace rc
