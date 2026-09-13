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

// ─── First-hit range COST (the same march, as a scalar objective) ───────────────────────────────
// ★THE FACTOR BELOW RETURNS NORMAL EQUATIONS; THIS RETURNS THE OBJECTIVE ITSELF. A belief that optimises
// one coordinate over a GRID — a door's hinge angle, say — needs to COMPARE hypotheses, and (Id, bd) does
// not compare: it is the local linearisation about one state. Same march, same robust weight, same
// residual, evaluated rather than differentiated, so the two can never disagree about what is being
// minimised.
// ★AND IT IS THE MEASUREMENT A DOORWAY ACTUALLY MAKES. An aperture is a HOLE: a closed leaf stops the ray
// at the wall plane, an open one lets it fly through into the next room. That difference lives in WHERE
// THE RAY STOPS — including the rays that do not stop — and it is invisible to any cost built from return
// points alone, because the through-rays' endpoints are somewhere else entirely. e = t - rho_obs is
// signed and says which way the model is wrong: positive, the model let a ray through that was stopped;
// negative, the model stopped a ray that flew on.
template <int N, class Model, class State>
[[nodiscard]] float lidar_ray_cost(const Model& m, const State& s, const LidarRays& rays)
{
    if (rays.precision <= 0.0f or rays.endpoints.empty())
        return 0.0f;
    const float tcap = std::min(rays.max_range_m, 1e9f);
    const auto sdf_min = [&](const Eigen::Vector3f& p, int& prim)
    {
        float d = m.sdf_prim(p, s, 0); prim = 0;
        for (int k = 1; k < m.n_prims(); ++k)
        { const float dk = m.sdf_prim(p, s, k); if (dk < d) { d = dk; prim = k; } }
        return d;
    };
    double acc = 0.0;
    int    used = 0;
    for (const auto& ep : rays.endpoints)
    {
        const Eigen::Vector3f dir = ep - rays.origin;
        const float rho_obs = dir.norm();
        if (rho_obs < 1e-3f) continue;
        const Eigen::Vector3f u = dir / rho_obs;
        float t = 0.0f; int prim = 0; bool hit = false;
        for (int it = 0; it < rays.max_steps and t < tcap; ++it)
        {
            const float d = sdf_min(rays.origin + t * u, prim);
            if (d < rays.surf_eps_m) { hit = true; break; }
            t += std::max(d, rays.surf_eps_m);
        }
        // ★A MODEL THAT PREDICTS NO HIT IS A PREDICTION, NOT A MISSING MEASUREMENT. The ray is expected to
        // fly past everything the model knows about, so the honest predicted range is the march cap: if
        // the sensor DID stop it, that is a real, large residual — which is precisely how an open-leaf
        // hypothesis is refuted by a door that is in fact shut. Skipping these rays would make every
        // "open" hypothesis unfalsifiable, since its rays mostly predict no hit.
        // ★★★A BEAM MODEL, NOT A ROBUST SQUARED RESIDUAL — AND THE DIFFERENCE IS WHY THIS FUNCTION
        // EXISTS. A Cauchy (or any robust) kernel COMPRESSES large residuals by design: that is what
        // makes it robust. But for the question "did this ray get through, or was it stopped?" the entire
        // signal LIVES in large residuals, so the kernel destroys exactly the information being sought.
        // Measured 2026-09-13 on a door: with c = 0.05 m every ray saturated identically and the
        // likelihood was perfectly flat (the weight sat at 1/25 across 25 hypotheses); raising c to the
        // aperture width, 1.0 m, only moved the saturation point — "stopped 2.8 m early" scored 0.44 and
        // "flew 4 m past" scored 0.47, still indistinguishable. No value of c fixes it, because both
        // outcomes are far beyond any c that also tolerates sensor noise.
        // The right statistic is the standard range-sensor beam model, which is also what this fleet
        // already uses for silhouettes (rc::exist's occupancy/free log-odds):
        //   model predicts a hit at t  ⇒  p(rho) = (1-w_u)·N(rho; t, sigma) + w_u·U
        //   model predicts no hit      ⇒  p(rho) =                            w_u·U
        // A hypothesis that puts a surface where the ray actually stopped scores the Gaussian peak; one
        // that predicts free space there, or a block where the ray flew on, falls to the uniform floor.
        // The gap between those is a FIXED amount per ray that ACCUMULATES over hundreds of rays instead
        // of saturating — which is what makes hundreds of rays decisive rather than 0.7% of flat.
        // ★w_u is the share of returns the model cannot be expected to explain (the far wall seen through
        // an open doorway, the floor, a person). It is what stops an unexplained ray from being read as a
        // refutation: unexplained is the uniform floor, not minus infinity.
        const float sig  = rays.robust_c_m;            // range noise scale (m) — reused, now as sigma
        const float w_u  = 0.2f;                       // unexplained-return share
        const float uni  = 1.0f / std::max(1e-3f, tcap);
        double p = w_u * uni;
        if (hit)
        {
            const float e = t - rho_obs;
            p += (1.0 - w_u) * std::exp(-0.5 * (e * e) / (sig * sig))
                 / (sig * 2.5066282746f);              // sqrt(2*pi)
        }
        acc += -std::log(std::max(p, 1e-12));
        ++used;
    }
    return used > 0 ? static_cast<float>(acc / used) : 0.0f;   // PER RAY: see the door's phi_free_energy
}

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
