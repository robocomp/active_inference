// The SDF observation term (see room_obs_weights.h). Moved out of room_concept.cpp and off torch on
// 2026-10-17; the formulas are the torch ones term for term.
#include "room_obs_weights.h"

#include <algorithm>
#include <cmath>

namespace rc
{
    namespace
    {
        constexpr float kObsWeightEps = 1e-6f;

        float huber(float d, float delta)
        {
            const float ad = std::abs(d);
            return ad < delta ? 0.5f * d * d : delta * (ad - 0.5f * delta);
        }
    }

    Eigen::VectorXf build_observation_weights(const Model& /*model*/,
                                              const RoomConcept::Params& params,
                                              const ScanPoints& points_robot,
                                              float pose_theta,
                                              const Model::SdfQueryResult& query,
                                              const std::vector<DoorAperture>* doors_robot)
    {
        return weights_from_normals(params, points_robot, pose_theta, query.closest_normals, doors_robot);
    }

    Eigen::VectorXf weights_from_normals(const RoomConcept::Params& params,
                                         const ScanPoints& points_robot,
                                         float pose_theta,
                                         const Eigen::Matrix<float, -1, 2>& normals,
                                         const std::vector<DoorAperture>* doors_robot)
    {
        const auto n = static_cast<Eigen::Index>(points_robot.size());
        Eigen::VectorXf weights = Eigen::VectorXf::Ones(n);
        bool any_weighting = false;

        if (n <= 1)
            return weights;

        const auto range = [&](Eigen::Index i)
        {
            const auto& p = points_robot[static_cast<std::size_t>(i)];
            return std::sqrt(p.x() * p.x() + p.y() * p.y());
        };

        if (params.far_points_weight)
        {
            Eigen::VectorXf dists_alpha(n);
            for (Eigen::Index i = 0; i < n; ++i)
                dists_alpha(i) = std::pow(range(i), params.far_points_exponent);
            const float denom = dists_alpha.mean() + kObsWeightEps;
            for (Eigen::Index i = 0; i < n; ++i)
                weights(i) *= std::max(dists_alpha(i) / denom, params.far_points_min_weight);
            any_weighting = true;
        }

        if (params.incidence_angle_weight and normals.rows() > 0)
        {
            const float c = std::cos(pose_theta), s = std::sin(pose_theta);
            for (Eigen::Index i = 0; i < n; ++i)
            {
                const auto& p = points_robot[static_cast<std::size_t>(i)];
                const float rn = std::max(range(i), kObsWeightEps);
                const float rx = p.x() / rn, ry = p.y() / rn;                 // ray direction, robot frame
                const float wx = c * rx - s * ry, wy = s * rx + c * ry;       // ... in the room frame
                const float nn = std::max(std::sqrt(normals(i, 0) * normals(i, 0) + normals(i, 1) * normals(i, 1)),
                                          kObsWeightEps);
                const float incidence = std::abs(wx * normals(i, 0) / nn + wy * normals(i, 1) / nn);
                const float w = std::pow(std::max(incidence, kObsWeightEps), params.incidence_angle_exponent);
                weights(i) *= std::max(w, params.incidence_angle_min_weight);
            }
            any_weighting = true;
        }

        // ── A BEAM THAT WENT THROUGH AN OPEN DOOR MEASURED THE NEXT ROOM ────────────────────
        // The apertures arrive already in the ROBOT frame, so the beam origin is the origin here.
        // Weight = the per-beam mixture (DoorApertures::weight), prior 1 - p_open (½ unmeasured): a shut door keeps its returns
        // at full weight, because a closed door IS a wall surface and its returns are wall evidence.
        // ⚠ WHAT THIS DOES AND DOES NOT DO. These weights are normalised to mean 1 on the last
        // line, so the door term is RELATIVE — it moves the fit away from through-the-door returns
        // and toward real wall returns, which is exactly what a weighted least squares needs, and
        // it is scale-free because sigma_obs sets the scale. It does NOT reduce the REPORTED
        // sdf_mse, which is a separate unweighted reduction over the same points: an open door
        // will still show up there until that reduction is given the same weights.
        if (doors_robot != nullptr and not doors_robot->empty())
        {
            const Eigen::Vector2f origin(0.f, 0.f);
            Eigen::VectorXf door_w(n);
            int discounted = 0;
            for (Eigen::Index i = 0; i < n; ++i)
            {
                const auto& p = points_robot[static_cast<std::size_t>(i)];
                door_w(i) = DoorApertures::weight(*doors_robot, origin, Eigen::Vector2f(p.x(), p.y()),
                                                  params.rfe_obs_sigma);
                if (door_w(i) < 1.f) ++discounted;
            }
            if (discounted > 0)
            {
                weights = weights.cwiseProduct(door_w);
                any_weighting = true;
            }
        }

        if (not any_weighting)
            return weights;
        return weights / (weights.mean() + kObsWeightEps);
    }

    float compute_observation_loss_from_query(const Model& model,
                                              const RoomConcept::Params& params,
                                              const ScanPoints& points_robot,
                                              float pose_theta,
                                              const Model::SdfQueryResult& query,
                                              const std::vector<DoorAperture>* doors_robot)
    {
        const auto n = query.size();
        if (n == 0)
            return 0.f;
        const float inv_var = 1.0f / (params.rfe_obs_sigma * params.rfe_obs_sigma);
        const auto w = build_observation_weights(model, params, points_robot, pose_theta, query, doors_robot);
        double sum = 0.0;
        for (Eigen::Index i = 0; i < n; ++i)
            sum += static_cast<double>(huber(query.sdf(i), params.rfe_huber_delta) * w(i));
        return 0.5f * inv_var * static_cast<float>(sum / static_cast<double>(n));
    }

    float compute_observation_loss(const Model& model,
                                   const RoomConcept::Params& params,
                                   const ScanPoints& points_robot,
                                   const Eigen::Vector3f& pose,
                                   const std::vector<DoorAperture>* doors_robot)
    {
        const auto query = model.sdf_query_at_pose(points_robot, pose);
        return compute_observation_loss_from_query(model, params, points_robot, pose.z(), query, doors_robot);
    }

    ObsLinearization linearize_observation(const Model& model,
                                           const RoomConcept::Params& params,
                                           const ScanPoints& points_robot,
                                           const Eigen::Vector3f& pose,
                                           const std::vector<DoorAperture>* doors_robot)
    {
        ObsLinearization out;
        const auto q = model.sdf_query_at_pose(points_robot, pose, /*want_grad=*/true);
        const auto n = q.size();
        if (n == 0)
            return out;

        const auto w = build_observation_weights(model, params, points_robot, pose.z(), q, doors_robot);
        const float inv_var = 1.0f / (params.rfe_obs_sigma * params.rfe_obs_sigma);
        const float delta = params.rfe_huber_delta;
        const float c = std::cos(pose.z()), s = std::sin(pose.z());

        // IRLS Huber: huber_δ(d) = 0.5·u·d² with u = min(1, δ/|d|), so the loss is 0.5·Σ a_i d_i² with
        // a_i = 0.5·σ_obs⁻²·w_i·u_i/N. Row of J for point p (robot frame), g = ∇SDF, q = R(θ)p:
        //     ∂d/∂(x,y) = gᵀ ,   ∂d/∂θ = gᵀ·(J·q) = -g_x·q_y + g_y·q_x
        for (Eigen::Index i = 0; i < n; ++i)
        {
            const float d = q.sdf(i);
            if (not std::isfinite(d)) continue;
            const float ad = std::abs(d);
            const float u = (ad <= delta or ad < 1e-9f) ? 1.0f : delta / ad;
            const float a = 0.5f * inv_var * w(i) * u / static_cast<float>(n);
            const auto& p = points_robot[static_cast<std::size_t>(i)];
            const float qx = c * p.x() - s * p.y();
            const float qy = s * p.x() + c * p.y();
            const float gx = q.grad(i, 0), gy = q.grad(i, 1);
            const Eigen::Vector3f J(gx, gy, -gx * qy + gy * qx);
            out.H.noalias() += a * J * J.transpose();
            out.g.noalias() += (a * d) * J;
        }
        out.loss = compute_observation_loss_from_query(model, params, points_robot, pose.z(), q, doors_robot);
        return out;
    }
} // namespace rc
