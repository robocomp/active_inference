/*
 * lidar_ray_factor.h  —  shared LiDAR (range-sensor) first-hit factor for the AI2 concept beliefs
 *
 * SHARED, header-only: one YOLO-independent range-evidence channel every recursive-Laplace belief folds in
 * unchanged via its accumulate_extra hook (it touches only sdf_prim / sdf_jacobian / n_prims).
 *
 * A second, YOLO-INDEPENDENT evidence channel for the recursive-Laplace beliefs (bottle/table/chair). The
 * existing per-point SDF term pushes the surface toward each point along the SURFACE NORMAL, so a one-sided
 * (front-arc) cloud is depth-degenerate: the model can slide camera-ward and keep explaining the points
 * (see bottle-sdf-depth-bias). A range ray pins depth directly, because its residual is measured ALONG THE
 * VIEWING RAY, not normal to the surface:
 *
 *     e = ρ_hit(θ) − ρ_obs          (predicted first-hit range − observed range)
 *
 * ρ_hit(θ) is found by sphere-tracing the model's own SDF, so this factor is SHAPE-AGNOSTIC: it calls only
 * Model::sdf_prim / Model::sdf_jacobian / Model::n_prims — the hooks every concept belief already exposes —
 * and therefore works UNCHANGED for the cylinder (N=5), the table (N=6) and the chair (N=3). It folds into
 * the SAME Gauss-Newton normal equations as everything else (Id += w·JJᵀ, bd += −w·J·e), via the model's
 * accumulate_extra hook, so it informs both the MEAN and the posterior Σ.
 *
 * AI2 note (no thresholds): membership is decided by GEOMETRY, not a distance gate — a ray only contributes
 * if the model's SDF is actually crossed along it (sphere-trace hit); a ray that misses the model carries no
 * term at all. Returns that don't sit on the surface fade out through a continuous Cauchy responsibility, and
 * the grazing-incidence ill-conditioning is absorbed as a growing 1/(n̂·u) in the Jacobian (→ low information
 * after weighting), never a grazing cutoff.
 */

#pragma once

#include <vector>
#include <cmath>
#include <Eigen/Dense>

namespace rc::ai
{

// ─── One sweep of range returns (the factor's input) ─────────────────────────────────────────────

// One frame of range returns, in the SAME (room) frame as the model state. Each return is the ray from
// `origin` (sensor centre) to `endpoints[i]`; the observed range is |endpoint − origin|. `precision` (1/m²,
// ≈ 1/σ_range²) is the base per-ray weight AND the on/off switch: 0 ⇒ the factor is skipped entirely, exactly
// like BottleFrame::sil_precision. Because all returns of one sweep share the sensor-pose error, they are
// CORRELATED — the engine's common-mode Woodbury saturation de-correlates them, same as the depth points.
struct LidarRays
{
    Eigen::Vector3f              origin    = Eigen::Vector3f::Zero();  // sensor centre (room frame)
    std::vector<Eigen::Vector3f> endpoints;                           // returns (room frame)
    float precision   = 0.0f;    // base per-ray precision 1/m² (0 ⇒ OFF). Match the depth term (≈1/σ²)
    float robust_c_m  = 0.05f;   // Cauchy scale (m): returns this far off the model surface fade out

    // Sphere-trace controls (cm-scale objects; a few cm tolerance is plenty).
    float surf_eps_m  = 2e-3f;   // surface-hit tolerance + finite-difference step (m)
    float max_range_m = 6.0f;    // stop marching (scene bound)
    int   max_steps   = 64;      // sphere-trace iteration cap
};

// ─── First-hit range factor (sphere-trace the model's own SDF → GN normal equations) ─────────────

// Generic first-hit range factor. Templated on the belief model so it reuses the model's own SDF; N is the
// state dimension (deduced call sites pass it explicitly). Folds into (Id, bd) with the engine convention.
template <int N, class Model, class State>
void accumulate_lidar_rays(const Model& m, const State& s, const LidarRays& rays,
                           Eigen::Matrix<float, N, N>& Id, Eigen::Matrix<float, N, 1>& bd)
{
    if (rays.precision <= 0.0f or rays.endpoints.empty())
        return;

    const int   P    = m.n_prims();
    const float c2   = rays.robust_c_m * rays.robust_c_m;
    const float tcap = std::min(rays.max_range_m, 1e9f);

    // Model SDF = min over primitives; also reports the winning primitive (needed for the Jacobian).
    const auto sdf_min = [&](const Eigen::Vector3f& p, int& best) -> float
    {
        best = 0;
        float d = m.sdf_prim(p, s, 0);
        for (int k = 1; k < P; ++k)
        {
            const float dk = m.sdf_prim(p, s, k);
            if (dk < d) { d = dk; best = k; }
        }
        return d;
    };

    for (const auto& ep : rays.endpoints)
    {
        Eigen::Vector3f dir = ep - rays.origin;
        const float rho_obs = dir.norm();
        if (rho_obs < 1e-4f) continue;
        dir /= rho_obs;                                   // unit viewing direction

        // Sphere-trace from the sensor to the first surface crossing of the model along this ray.
        float t = 0.0f; int prim = 0; bool hit = false;
        for (int k = 0; k < rays.max_steps; ++k)
        {
            const float d = sdf_min(rays.origin + t * dir, prim);
            if (d < rays.surf_eps_m) { hit = true; break; }
            t += std::max(d, rays.surf_eps_m);            // conservative forward march
            if (t > std::min(tcap, rho_obs + 5.0f * rays.robust_c_m)) break;
        }
        if (not hit or t < rays.surf_eps_m) continue;     // model not on this ray (soft geometric membership)

        const Eigen::Vector3f p = rays.origin + t * dir;
        int best = prim; sdf_min(p, best);                // arg-primitive AT the hit

        const float e = t - rho_obs;                      // range residual

        // Range Jacobian by implicit differentiation of SDF(origin + t·dir; θ) = 0:
        //     dt/dθ = −(∂SDF/∂θ) / (∂SDF/∂t),   ∂SDF/∂t = ∇SDF·dir = n̂·dir  (incidence cosine).
        int tmp;
        const float h    = rays.surf_eps_m;
        const float dSdt = (sdf_min(rays.origin + (t + h) * dir, tmp)
                          - sdf_min(rays.origin + (t - h) * dir, tmp)) / (2.0f * h);
        if (std::abs(dSdt) < 1e-3f) continue;             // grazing: range uninformative (incidence, not a gate)
        const Eigen::Matrix<float, N, 1> J = -m.sdf_jacobian(p, s, best) / dSdt;

        // Continuous Cauchy responsibility: a return far off the surface (background / a different object)
        // fades out smoothly. No association gate; the grazing 1/(n̂·dir) blow-up is already carried in J.
        const float w = rays.precision / (1.0f + (e * e) / c2);
        Id.noalias() += w * (J * J.transpose());
        bd.noalias() += -w * J * e;
    }
}

}  // namespace rc::ai
