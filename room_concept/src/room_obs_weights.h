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

#include <Eigen/Dense>
#include <vector>

#include "door_apertures.h"
#include "localizer_params.h"
#include "room_model.h"

namespace rc
{
    /// Per-point observation weights (range and/or incidence), normalised to mean 1 — constants at the
    /// linearization point, which is exactly what IRLS/Gauss-Newton needs.
    /// `doors_robot` is every known doorway of this room as a segment IN THE ROBOT FRAME, so the beam
    /// origin is the origin and no pose translation is needed here. A point whose beam crossed an open
    /// aperture is weighted by (1 - p_open): its return came from the next room and says nothing about
    /// this one. Null or empty ⇒ no door term, which is what a room with no known doors gets.
    Eigen::VectorXf build_observation_weights(const Model& model,
                                              const loc::Params& params,
                                              const ScanPoints& points_robot,
                                              float pose_theta,
                                              const Model::SdfQueryResult& query,
                                              const std::vector<DoorAperture>* doors_robot = nullptr);

    /// The same weights from EXPLICIT room-frame normals (one row per point). The SDF path passes
    /// query.closest_normals; the wall-landmark factors pass each point's wall normal, so both terms
    /// weight a point identically. Empty normals ⇒ range weighting only.
    Eigen::VectorXf weights_from_normals(const loc::Params& params,
                                         const ScanPoints& points_robot,
                                         float pose_theta,
                                         const Eigen::Matrix<float, -1, 2>& normals,
                                         const std::vector<DoorAperture>* doors_robot = nullptr);

    /// 0.5·σ_obs⁻²·mean_i(w_i · huber_δ(d_i)) for one slot, given an already-evaluated SDF query.
    float compute_observation_loss_from_query(const Model& model,
                                              const loc::Params& params,
                                              const ScanPoints& points_robot,
                                              float pose_theta,
                                              const Model::SdfQueryResult& query,
                                              const std::vector<DoorAperture>* doors_robot = nullptr);

    /// The same loss at a pose: queries the model, then compute_observation_loss_from_query.
    float compute_observation_loss(const Model& model,
                                   const loc::Params& params,
                                   const ScanPoints& points_robot,
                                   const Eigen::Vector3f& pose,
                                   const std::vector<DoorAperture>* doors_robot = nullptr);

    /// Loss, gradient and Gauss-Newton Hessian of the observation term at `pose` — the SAME linearization
    /// room_gn_solver's SdfFactor adds to its normal equations (IRLS Huber, weights frozen at the point).
    /// Used wherever a 3x3 information of one scan is needed (window marginalisation, posterior
    /// covariance). It replaced the autograd Hessians of the torch backend (2026-10-17); it drops the
    /// second-order SDF curvature autograd kept — the Gauss-Newton approximation, as in the solver.
    struct ObsLinearization
    {
        float           loss = 0.f;
        Eigen::Vector3f g = Eigen::Vector3f::Zero();
        Eigen::Matrix3f H = Eigen::Matrix3f::Zero();
    };
    ObsLinearization linearize_observation(const Model& model,
                                           const loc::Params& params,
                                           const ScanPoints& points_robot,
                                           const Eigen::Vector3f& pose,
                                           const std::vector<DoorAperture>* doors_robot = nullptr);
} // namespace rc
