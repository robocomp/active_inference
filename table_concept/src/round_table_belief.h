/*
 * round_table_belief.h  —  ROUND (disc-top) table belief for the square-vs-round model-selection test.
 *
 * A minimal `Model`-concept struct wired onto the SAME shared rc::ai recursive-Laplace engine as
 * TableBelief (common/ai_belief/recursive_laplace.h). Used two ways: the offline A/B harness
 * (tests/compare_models) and the LIVE per-instance shape model-selection (TableFitter::evaluate_shape,
 * which fits it to the accumulated support bank and compares free energy vs the square). Header-only.
 *
 * State θ = [cx, cy, H, radius]  (N=4). A round table is rotationally symmetric → NO yaw, NO w/h, NO
 * canonical fold, NO orientation-mode machinery (that whole class of TableBelief code is structurally
 * absent here — which is exactly why a disc's posterior cannot carry an unidentifiable yaw).
 *
 * Two base variants (RoundBase):
 *   - Ring     : disc top + 4 legs on a ring  → 5 surface prims (matched to the square's top+4legs).
 *                This is the CLEAN discriminator: identical primitive cardinality ⇒ the mixture prior
 *                π_surf=(1−ε)/5 matches the square, so the free-energy baseline cancels and the ONLY
 *                difference measured is disc-vs-box TOP shape.
 *   - Pedestal : disc top + one central pedestal → 2 surface prims. Faithful to the real table, BUT its
 *                π_surf=(1−ε)/2 differs, shifting the FE baseline in its favour — so use it only as a
 *                physical-realism cross-check, never as the verdict (compare_models asserts on Ring).
 *
 * Everything else (top_thickness, sigma_base, clutter ε/scale, priors, common-mode) is copied from
 * TableBeliefParams so R and the mixture floor are IDENTICAL for a fair comparison.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include <Eigen/Dense>

#include "table_belief.h"   // rc::TableFrame (reused as Frame) + the rc::ai engine (transitively)

namespace rc
{

namespace round_detail
{
// Vertical-cylinder SDF — copied VERBATIM from table_belief.cpp (anon namespace, not visible here) so the
// disc-top / leg / pedestal SDF math is byte-identical to the square model's legs.
inline float cylinder_sdf(float dlx, float dly, float dlz, float r, float hh)
{
    const float d_radial = std::sqrt(dlx * dlx + dly * dly) - r;
    const float d_vert   = std::abs(dlz) - hh;
    const float outside  = std::sqrt(std::max(d_radial, 0.0f) * std::max(d_radial, 0.0f) +
                                     std::max(d_vert, 0.0f)   * std::max(d_vert, 0.0f));
    const float inside   = std::min(std::max(d_radial, d_vert), 0.0f);
    return outside + inside;
}
}  // namespace round_detail

enum class RoundBase { Ring, Pedestal };

struct RoundTableState
{
    float cx = 0.0f, cy = 0.0f, H = 0.75f, radius = 0.5f;
    Eigen::Matrix<float, 4, 1> vec() const
    { return (Eigen::Matrix<float, 4, 1>() << cx, cy, H, radius).finished(); }
    static RoundTableState from_vec(const Eigen::Matrix<float, 4, 1>& v)
    { return {v(0), v(1), v(2), v(3)}; }
};

struct RoundTableParams
{
    // Geometry (fixed) — mirror TableBeliefParams where shared.
    float top_thickness = 0.03f;
    float leg_radius    = 0.025f;   // ring-leg radius (Ring variant)
    float ped_radius    = 0.06f;    // central pedestal radius (Pedestal variant)
    float ring_frac     = 0.80f;    // ring legs sit at ring_frac·radius
    // Observation / mixture (identical to the square so R and the clutter floor match).
    float sigma_base_m    = 0.03f;
    float clutter_frac    = 0.10f;
    float clutter_scale_m = 0.12f;
    // Priors / transition / common-mode.
    float prior_size_std       = 0.30f;
    float process_std_m        = 0.005f;
    float common_mode_pos_std  = 0.03f;
    float common_mode_size_std = 0.02f;
    // Optimiser.
    int   gn_iters = 4;
    float fd_eps   = 1e-3f;
    float jac_clamp = 1.5f;   // the disc has an on-axis radial kink the box lacks → clamp the FD slope
};

// Round-table generative model on the shared rc::ai engine. N = 4, static (transition = I).
class RoundTableBelief
{
public:
    static constexpr int N = 4;
    using State = RoundTableState;
    using Frame = TableFrame;   // reuse the square model's frame; only .points/.R are touched

    RoundTableBelief() = default;
    RoundTableBelief(const RoundTableState& s, const RoundTableParams& p, RoundBase base)
        : state_(s), params_(p), base_(base)
    { Sigma_.setZero(); Sigma_.diagonal() = prior_cov_diag(); }

    const RoundTableState&            state()      const { return state_; }
    const Eigen::Matrix<float, 4, 4>& covariance() const { return Sigma_; }
    RoundBase                         base()       const { return base_; }
    int                               n_legs()     const { return base_ == RoundBase::Ring ? 4 : 1; }

    // ── SDF primitives ────────────────────────────────────────────────────────
    float sdf_top(const Eigen::Vector3f& p, const RoundTableState& s) const
    {
        const float half_t = 0.5f * params_.top_thickness;
        const float top_cz = s.H - half_t;
        return round_detail::cylinder_sdf(p.x() - s.cx, p.y() - s.cy, p.z() - top_cz, s.radius, half_t);
    }
    Eigen::Vector2f leg_center(const RoundTableState& s, int k) const
    {
        if (base_ == RoundBase::Pedestal) return {s.cx, s.cy};
        const float rr  = params_.ring_frac * s.radius;
        const float ang = 0.25f * static_cast<float>(M_PI) + k * 0.5f * static_cast<float>(M_PI);
        return {s.cx + rr * std::cos(ang), s.cy + rr * std::sin(ang)};
    }
    float leg_half_height(const RoundTableState& s) const
    { return 0.5f * std::max(0.0f, s.H - params_.top_thickness); }
    float sdf_leg(const Eigen::Vector3f& p, const RoundTableState& s, int k) const
    {
        const auto  ctr = leg_center(s, k);
        const float hh  = leg_half_height(s);
        const float r   = (base_ == RoundBase::Pedestal) ? params_.ped_radius : params_.leg_radius;
        return round_detail::cylinder_sdf(p.x() - ctr.x(), p.y() - ctr.y(), p.z() - hh, r, hh);
    }
    float sdf_prim(const Eigen::Vector3f& p, const RoundTableState& s, int prim) const
    { return prim == 0 ? sdf_top(p, s) : sdf_leg(p, s, prim - 1); }

    // ── Mixture responsibilities [top, legs…, clutter] — same structure/floor as the square ──
    float mixture_unnormalized(const Eigen::Vector3f& p, const RoundTableState& s, float R,
                               std::array<float, 6>& u) const
    {
        const int   P       = n_prims();
        const float eps     = std::clamp(params_.clutter_frac, 0.0f, 0.99f);
        const float pi_surf = (1.0f - eps) / static_cast<float>(P);
        const float inv2R   = 0.5f / std::max(1e-9f, R);
        const float z_join  = s.H - params_.top_thickness;
        const float band    = std::max(0.5f * params_.top_thickness, 1e-3f);
        const float leg_z   = 1.0f / (1.0f + std::exp((p.z() - z_join) / band));   // →1 below join
        const float top_z   = 1.0f - leg_z;                                        // →1 above join (tabletop)

        u = {};
        const float dt = sdf_top(p, s);
        u[0] = pi_surf * top_z * std::exp(-dt * dt * inv2R);
        for (int k = 0; k < n_legs(); ++k)
        {
            const float d = sdf_leg(p, s, k);
            u[1 + k] = pi_surf * leg_z * std::exp(-d * d * inv2R);
        }
        const float cs = params_.clutter_scale_m;
        u[P] = eps * std::exp(-cs * cs * inv2R);   // clutter floor at index n_prims()

        float sum = 0.0f;
        for (int i = 0; i <= P; ++i) sum += u[i];
        return sum;
    }
    std::array<float, 6> responsibilities(const Eigen::Vector3f& p, const RoundTableState& s, float R) const
    {
        std::array<float, 6> u{};
        const float sum = mixture_unnormalized(p, s, R, u);
        const int   P   = n_prims();
        if (sum <= 0.0f) { u = {}; u[P] = 1.0f; return u; }
        for (int i = 0; i <= P; ++i) u[i] /= sum;
        return u;
    }
    // TRUE free energy: mean per-point −log mixture marginal likelihood, clutter-inclusive. The (2πR)^-3/2
    // normaliser is dropped — it cancels vs the square (same R, same points). IDENTICAL formula to
    // TableBelief::mean_energy so the two are directly comparable.
    float mean_energy(const std::vector<Eigen::Vector3f>& pts, const RoundTableState& s, float R) const
    {
        if (pts.empty()) return 0.0f;
        std::array<float, 6> u{};
        double e = 0.0;
        for (const auto& p : pts)
        {
            const float sum = mixture_unnormalized(p, s, R, u);
            e += -std::log(std::max(sum, 1e-12f));
        }
        return static_cast<float>(e / static_cast<double>(pts.size()));
    }

    // ── Jacobian (central FD, slope-clamped) ────────────────────────────────────
    Eigen::Matrix<float, 4, 1> sdf_jacobian(const Eigen::Vector3f& p, const RoundTableState& s, int prim) const
    {
        Eigen::Matrix<float, 4, 1> J;
        const Eigen::Matrix<float, 4, 1> base = s.vec();
        const float e = params_.fd_eps;
        for (int j = 0; j < 4; ++j)
        {
            Eigen::Matrix<float, 4, 1> vp = base, vm = base;
            vp(j) += e; vm(j) -= e;
            const float g = (sdf_prim(p, State::from_vec(vp), prim) -
                             sdf_prim(p, State::from_vec(vm), prim)) / (2.0f * e);
            J(j) = std::clamp(g, -params_.jac_clamp, params_.jac_clamp);
        }
        return J;
    }

    // ── Constraints / engine diagonals ──────────────────────────────────────────
    void apply_constraints(RoundTableState& s) const
    {
        s.radius = std::max(s.radius, 0.10f);
        s.H      = std::max(s.H, params_.top_thickness + 0.05f);
    }
    void canonicalize(RoundTableState&) const {}   // rotational symmetry: nothing to fold

    int   gn_iters() const { return params_.gn_iters; }
    int   n_prims()  const { return 1 + n_legs(); }   // disc top + legs/pedestal (clutter is +1)
    float sigma2()   const { return params_.sigma_base_m * params_.sigma_base_m; }
    Eigen::Matrix<float, 4, 4> transition() const { return Eigen::Matrix<float, 4, 4>::Identity(); }
    Eigen::Matrix<float, 4, 1> process_noise_diag() const
    {
        const float qm = params_.process_std_m * params_.process_std_m;
        return (Eigen::Matrix<float, 4, 1>() << qm, qm, qm, qm).finished();
    }
    Eigen::Matrix<float, 4, 1> prior_cov_diag() const
    {
        const float ss = params_.prior_size_std * params_.prior_size_std;   // H, radius
        return (Eigen::Matrix<float, 4, 1>() << 0.30f * 0.30f, 0.30f * 0.30f, ss, ss).finished();
    }
    Eigen::Matrix<float, 4, 1> common_mode_inv_diag(const TableFrame& frame) const
    {
        const float p2 = params_.common_mode_pos_std  * params_.common_mode_pos_std;
        const float s2 = params_.common_mode_size_std * params_.common_mode_size_std;
        const float cs = frame.chain_cov_size;
        return (Eigen::Matrix<float, 4, 1>() <<
                1.0f / std::max(1e-9f, p2 + frame.chain_cov_xx),
                1.0f / std::max(1e-9f, p2 + frame.chain_cov_yy),
                1.0f / std::max(1e-9f, s2 + cs),
                1.0f / std::max(1e-9f, s2 + cs)).finished();
    }

    // One recursive-Laplace update on a fresh frame (returns the engine's surface-only energy — DISCARD it,
    // recompute mean_energy at the converged state, exactly as TableBelief::update / table_fitter do).
    float update(const TableFrame& frame)
    { return ai::update<N>(*this, state_, Sigma_, prior_mean_, frame); }

private:
    RoundTableState            state_;
    RoundTableParams           params_;
    RoundBase                  base_       = RoundBase::Ring;
    Eigen::Matrix<float, 4, 4> Sigma_      = Eigen::Matrix<float, 4, 4>::Identity();
    Eigen::Matrix<float, 4, 1> prior_mean_ = Eigen::Matrix<float, 4, 1>::Zero();
};

}  // namespace rc
