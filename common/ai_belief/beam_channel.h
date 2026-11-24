/*
 * common/ai_belief/beam_channel.h  —  the persistent engine's LiDAR channel on the smooth beam model (header-only, pure)
 *
 * Spec 2026-11-15 lidar-measures §7c.8 (a, b, a0) and 2026-11-23 shared-engine ("Engine shape"). Moved out of
 * countable_belief.cpp so every model on rc::ai::engine::PersistentEngine reads its rays the same way. It is generic over
 * the model's convex parts (std::span<const BeamPart>, each carrying its own ∂(centre, yaw, half extents)/∂θ) and knows
 * nothing about the model that built them:
 *
 *   - beam_readiness: whether a staged ray-set is a measurement (rev 3.3.11: non-empty and Σq > 0; refused when the per-ray
 *     vectors are misaligned or the BeamConfig was not built by BeamConfig::make — the caller logs those once);
 *   - BeamChannel<N>: Σ E_i of the set at θ, its exact gradient (bd −= ∇E) and the PSD curvature (Id += W), with the
 *     per-update cache (one build per state: the GN evaluates J at a candidate and, if accepted, linearises there next);
 *   - count_gap_rays: the stated-limit-1 log counter (rays the parts block almost surely whose return lies > 5c beyond the
 *     nominal surface).
 *
 * The ray STAGING (select_rays_beam by ray geometry, the persistent label, p_f against the room's known surfaces) is the
 * caller's: it reads the sweep, the mask membership and the room polygon, none of which belong to the engine
 * (countable_concept/src/countable_beam_staging.h stages the countable's).
 */
#pragma once

#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <span>
#include <vector>

#include <Eigen/Core>

#include "beam_ray.h"

namespace rc::ai::beam
{

// Whether a staged ray-set measures (spec 7c.8 rev 3.3.11): kEmpty (no rays or Σq = 0: the camera attributes nothing),
// kInconsistent (m / p_f not one per ray: a staging error), kUnready (a BeamConfig{} would integrate on the geometric
// breakpoints only), else kMeasured.
enum class Readiness { kMeasured, kEmpty, kInconsistent, kUnready };
inline Readiness beam_readiness(const BeamRays& rays, const BeamConfig& cfg)
{
    if (rays.endpoints.empty() or not (rays.q_sum > 0.0)) return Readiness::kEmpty;
    if (not detail::p_f_consistent(rays)) return Readiness::kInconsistent;
    if (cfg.e_levels.empty()) return Readiness::kUnready;
    return Readiness::kMeasured;
}

// Stated limit 1 (spec 7c.8, gaps under furniture): the rays the parts block almost surely (π_i = Φ(−d_i/σ_s) > 0.99) whose
// return lies more than 5c BEYOND the nominal surface (e_i = t_i(0) − ρ_i < −5c, t_i(0) the entry into the solid, 0 when the
// sensor is inside it). A LOG counter only: nothing reads it.
inline int count_gap_rays(std::span<const BeamPart> parts, const BeamRays& rays, const BeamConfig& cfg)
{
    const double sig = rays.sigma_s, c = cfg.st.c, rho_max = rays.rho_max;
    const Eigen::Vector3d& o = rays.origin;
    int n = 0;
    for (const auto& e : rays.endpoints)
    {
        const Eigen::Vector3d du = e - o;
        const double rho = du.norm();
        if (not (rho > 0.0)) continue;
        const Eigen::Vector3d u = du / rho;
        double d = std::numeric_limits<double>::infinity(), t_in = std::numeric_limits<double>::infinity();
        for (const auto& P : parts)
        {
            d = std::min(d, closest_approach(P, o, u, rho_max, nullptr));
            // slab entry of o + t u into the part (local frame), t ∈ [0, ρ_max]
            const double cy = std::cos(P.yaw), sy = std::sin(P.yaw), dx = o.x() - P.c.x(), dy = o.y() - P.c.y();
            const Eigen::Vector3d ol(cy * dx + sy * dy, -sy * dx + cy * dy, o.z() - P.c.z());
            const Eigen::Vector3d ul(cy * u.x() + sy * u.y(), -sy * u.x() + cy * u.y(), u.z());
            double a = 0.0, b = rho_max;
            bool hit = true;
            for (int i = 0; i < 3 and hit; ++i)
            {
                if (ul(i) == 0.0) { hit = std::abs(ol(i)) <= P.half(i); continue; }
                double t1 = (-P.half(i) - ol(i)) / ul(i), t2 = (P.half(i) - ol(i)) / ul(i);
                if (t1 > t2) std::swap(t1, t2);
                a = std::max(a, t1); b = std::min(b, t2);
                hit = a <= b;
            }
            if (hit) t_in = std::min(t_in, a);
        }
        const double pi = Phi_upper(d / sig);
        if (pi > 0.99 and std::isfinite(t_in) and t_in - rho < -5.0 * c) ++n;
    }
    return n;
}

// The LiDAR channel of one belief: E = Σ E_i (κ ≡ 1 since rev 3.3.11), g = ∇E, W the PSD curvature; the GN convention is
// bd −= g, Id += W, in double inside beam_ray.h, the blocks float (as every channel of the engine). Holds the per-update
// cache and the wall time spent in the beam terms; mutable state only (the energy of a state does not depend on it).
template <int N>
class BeamChannel
{
public:
    using Key = std::array<float, N>;

    // Σ E_i at the state whose float vector is `key`, its gradient and curvature into (*Id, *bd) when both are given, the
    // energy alone otherwise. `parts()` builds the model's parts at that state (called only when the cache misses).
    template <class PartsFn>
    double accumulate(const Key& key, PartsFn&& parts, const BeamRays& rays, const BeamConfig& cfg,
                      Eigen::Matrix<float, N, N>* Id, Eigen::Matrix<float, N, 1>* bd) const
    {
        const auto t0 = std::chrono::steady_clock::now();
        double E = 0.0;
        if (cache_.on)
        {   // inside an update: the full build, once per state
            int k = -1;
            for (int j = 0; j < 2; ++j) if (cache_.slot[j].valid and cache_.slot[j].s == key) k = j;
            if (k < 0)
            {
                k = 1 - cache_.last;
                auto& c = cache_.slot[k];
                c.W.setZero(); c.g.setZero(); c.E = 0.0;
                accumulate_beam_rays<N>(parts(), rays, cfg, c.W, c.g, &c.E);
                c.s = key; c.valid = true;
            }
            cache_.last = k;
            const auto& c = cache_.slot[k];
            if (Id and bd) { *Id += c.W.template cast<float>(); *bd -= c.g.template cast<float>(); }
            ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            return c.E;
        }
        const auto pv = parts();
        if (Id and bd)
        {
            Eigen::Matrix<double, N, N> W = Eigen::Matrix<double, N, N>::Zero();
            Eigen::Matrix<double, N, 1> g = Eigen::Matrix<double, N, 1>::Zero();
            accumulate_beam_rays<N>(pv, rays, cfg, W, g, &E);
            *Id += W.template cast<float>();
            *bd -= g.template cast<float>();
        }
        else
            E = beam_rays_energy(pv, rays, cfg);
        ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return E;
    }

    // The cache lives for one update (the frame is fixed inside it): on while a Scope is alive, cleared on both ends.
    struct Scope
    {
        const BeamChannel& ch;
        explicit Scope(const BeamChannel& c) : ch(c) { ch.cache_.on = true; ch.cache_.clear(); }
        ~Scope() { ch.cache_.on = false; ch.cache_.clear(); }
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;
    };

    double ms() const { return ms_; }
    void   reset_ms() { ms_ = 0.0; }

private:
    struct Cache
    {
        struct Slot
        {
            bool valid = false;
            Key s{};
            double E = 0.0;
            Eigen::Matrix<double, N, N> W = Eigen::Matrix<double, N, N>::Zero();
            Eigen::Matrix<double, N, 1> g = Eigen::Matrix<double, N, 1>::Zero();
        };
        bool on = false;
        std::array<Slot, 2> slot{};   // two states: the accepted iterate survives the rejected trials after it
        int last = 0;                 // the slot used last (the other one is overwritten next)
        void clear() { slot[0].valid = slot[1].valid = false; }
    };
    mutable Cache  cache_;
    mutable double ms_ = 0.0;
};

}  // namespace rc::ai::beam
