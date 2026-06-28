/*
 * chair_model.cpp
 *
 * Compound SDF (box-top + 4 cylindrical legs) ported from the Python prototype
 * box_concept/src/objects/chair/sdf.py, replacing PyTorch ops with Eigen3.
 * Gradient computed with PyTorch autograd (Adam/SGD optimizers).
 */

#include "chair_model.h"
#include "../../common/robust_metrics/robust_metrics_torch.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <print>
#include <torch/torch.h>


namespace rc {
using namespace torch::indexing;

// ─── Helpers ─────────────────────────────────────────────────────────────────

namespace
{

bool finite_state(const ChairState& state)
{
    return std::isfinite(state.cx) && std::isfinite(state.cy) &&
           std::isfinite(state.cz) && std::isfinite(state.yaw) &&
           std::isfinite(state.seat_w) && std::isfinite(state.seat_d) &&
           std::isfinite(state.seat_h) && std::isfinite(state.back_h);
}

/** Standard box SDF with smooth interior (log-sum-exp inside). */
float box_sdf(float dx, float dy, float dz)
{
    // Outside component: Euclidean distance from the surface
    float ox = std::max(dx, 0.0f);
    float oy = std::max(dy, 0.0f);
    float oz = std::max(dz, 0.0f);
    float outside = std::sqrt(ox*ox + oy*oy + oz*oz);

    // Inside component: smooth min via log-sum-exp
    float inside = 0.0f;
    if (dx < 0.0f and dy < 0.0f and dz < 0.0f)
    {
        constexpr float k = ChairModel::SDF_SMOOTH_K;
        constexpr float s = ChairModel::SDF_INSIDE_SCALE;
        // logsumexp(-[dx,dy,dz]/k)
        float a = dx / k, b = dy / k, c = dz / k;         // all negative → -a,-b,-c positive
        float mx = std::max({-a, -b, -c});
        float lse = mx + std::log(std::exp(-a - mx) + std::exp(-b - mx) + std::exp(-c - mx));
        inside = s * (-k * lse);                            // negative (inside box)
    }

    return outside + inside;
}

/**
 * Robust IRLS weight ω(r) = ρ'(r)/(2r), normalised so the quadratic loss gives ω≡1. This is the
 * per-residual multiplier that turns a squared error into the M-estimator's effective precision;
 * used to down-weight outliers in the Fisher information (an outlier should contribute little
 * curvature, exactly as it contributes little gradient). Mirrors robust_loss_value's scale family.
 */
float robust_irls_weight(float residual, RobustLossType type, float scale)
{
    const float c  = std::max(scale, 1e-6f);
    const float c2 = c * c;
    const float r2 = residual * residual;
    switch (type)
    {
        case RobustLossType::Quadratic:
            return 1.0f;
        case RobustLossType::Huber:
            return std::abs(residual) <= c ? 1.0f : c / std::max(std::abs(residual), 1e-9f);
        case RobustLossType::GemanMcClure:
        {
            const float d = r2 + c2;
            return (c2 * c2) / (d * d);
        }
        case RobustLossType::Welsch:
            return std::exp(-r2 / c2);
        case RobustLossType::TukeyBiweight:
        {
            if (r2 >= c2) return 0.0f;
            const float u = 1.0f - r2 / c2;
            return u * u;
        }
    }
    return 1.0f;
}

/** Robust percentile of a copy of `vals` (q in [0,1]); `vals` is consumed. */
float percentile_inplace(std::vector<float>& vals, float q)
{
    if (vals.empty()) return 0.0f;
    const std::size_t k = std::min(vals.size() - 1,
                                   static_cast<std::size_t>(q * static_cast<float>(vals.size())));
    std::nth_element(vals.begin(), vals.begin() + static_cast<std::ptrdiff_t>(k), vals.end());
    return vals[k];
}

/**
 * Footprint-extent energy: the top-face rectangle (centre offset + half-extents) should match the
 * observed point extent in the chair-local frame. Penalises the box being off-centre or larger/
 * smaller than the 2–98 percentile point span. This is what makes position/size non-degenerate
 * when only the (flat) top face is observed. Scalar mirror of the autograd term in fe_torch_impl.
 */
float footprint_extent_energy(const ChairState& s, const std::vector<Eigen::Vector3f>& pts, float lambda)
{
    if (lambda <= 0.0f || pts.size() < 8) return 0.0f;
    const float c = std::cos(s.yaw), sn = std::sin(s.yaw);
    std::vector<float> lx, ly;
    lx.reserve(pts.size());
    ly.reserve(pts.size());
    for (const auto& p : pts)
    {
        const float px = p.x() - s.cx, py = p.y() - s.cy;
        lx.push_back(px * c + py * sn);
        ly.push_back(-px * sn + py * c);
    }
    const float lx_lo = percentile_inplace(lx, 0.02f);
    const float lx_hi = percentile_inplace(lx, 0.98f);
    const float ly_lo = percentile_inplace(ly, 0.02f);
    const float ly_hi = percentile_inplace(ly, 0.98f);
    const float ex = (lx_hi - lx_lo) * 0.5f, ey = (ly_hi - ly_lo) * 0.5f;
    const float mx = (lx_hi + lx_lo) * 0.5f, my = (ly_hi + ly_lo) * 0.5f;
    const float dw = s.seat_w * 0.5f - ex, dh = s.seat_d * 0.5f - ey;
    return lambda * (dw * dw + dh * dh + mx * mx + my * my);
}

} // anonymous namespace

namespace
{

torch::Tensor box_sdf_tensor(const torch::Tensor& dx,
                              const torch::Tensor& dy,
                              const torch::Tensor& dz)
{
    const auto ox = torch::clamp_min(dx, 0.0f);
    const auto oy = torch::clamp_min(dy, 0.0f);
    const auto oz = torch::clamp_min(dz, 0.0f);
    // +eps under sqrt: gradient of sqrt(x) is infinite at x=0 (point inside the
    // box → ox=oy=oz=0), which produces NaN gradients in autograd and locks the
    // optimizer. The eps offset (~3e-5 m) is negligible for a chair SDF.
    const auto outside = torch::sqrt(ox * ox + oy * oy + oz * oz + 1e-9f);

    constexpr float k = ChairModel::SDF_SMOOTH_K;
    constexpr float s = ChairModel::SDF_INSIDE_SCALE;
    const auto a = dx / k;
    const auto b = dy / k;
    const auto c = dz / k;
    const auto na = -a;
    const auto nb = -b;
    const auto nc = -c;
    const auto mx = torch::max(torch::max(na, nb), nc);
    const auto lse = mx + torch::log(torch::exp(na - mx) + torch::exp(nb - mx) + torch::exp(nc - mx));
    const auto inside_raw = s * (-k * lse);
    const auto inside_mask = (dx < 0.0f) & (dy < 0.0f) & (dz < 0.0f);
    const auto inside = torch::where(inside_mask, inside_raw, torch::zeros_like(dx));

    return outside + inside;
}

void apply_constraints_to_tensor(torch::Tensor& arr)
{
    torch::NoGradGuard no_grad;
    // Wooden-chair physical bounds (match ChairModel::apply_constraints) — keep the seat slab from
    // ballooning during the optimiser steps, not just at the end.
    arr.index_put_({4}, torch::clamp(arr.index({4}), 0.30f, 0.65f));   // seat_w
    arr.index_put_({5}, torch::clamp(arr.index({5}), 0.30f, 0.65f));   // seat_d
    arr.index_put_({6}, torch::clamp(arr.index({6}), 0.38f, 0.55f));   // seat_h
    arr.index_put_({7}, torch::clamp(arr.index({7}), 0.30f, 0.65f));   // back_h
}

torch::Tensor fe_torch_impl(const ChairModelParams& params,
                             const ChairState& prior,
                             const torch::Tensor& state_tensor,
                             const torch::Tensor& points_tensor,
                             const torch::Tensor& weights_tensor,
                             float robust_scale)
{
    const auto cx     = state_tensor.index({0});
    const auto cy     = state_tensor.index({1});
    const auto cz     = state_tensor.index({2});
    const auto yaw    = state_tensor.index({3});
    const auto seat_w = state_tensor.index({4});
    const auto seat_d = state_tensor.index({5});
    const auto seat_h = state_tensor.index({6});
    const auto back_h = state_tensor.index({7});

    const auto px = points_tensor.index({Slice(), 0}) - cx;
    const auto py = points_tensor.index({Slice(), 1}) - cy;
    const auto pz = points_tensor.index({Slice(), 2}) - cz;

    const auto c = torch::cos(yaw);
    const auto s = torch::sin(yaw);
    const auto local_x = px * c + py * s;
    const auto local_y = -px * s + py * c;
    const auto local_z = pz;

    const auto half_w = seat_w * 0.5f;
    const auto half_d = seat_d * 0.5f;
    const float half_t = ChairModel::SEAT_THICKNESS * 0.5f;

    const auto seat_cz = seat_h - half_t;
    const auto sdf_seat = box_sdf_tensor(torch::abs(local_x) - half_w, torch::abs(local_y) - half_d, torch::abs(local_z - seat_cz) - half_t);

    const auto back_cy = -half_d + half_t;
    const auto back_cz = seat_h + back_h * 0.5f;
    const auto sdf_back = box_sdf_tensor(torch::abs(local_x) - half_w, torch::abs(local_y - back_cy) - half_t, torch::abs(local_z - back_cz) - back_h * 0.5f);

    const auto leg_top    = seat_h - ChairModel::SEAT_THICKNESS;
    const auto leg_cz     = leg_top * 0.5f;
    const auto leg_half_z = leg_top * 0.5f;
    const float R = ChairModel::LEG_HALF;
    const std::array<std::array<float, 2>, 4> sgn = {{{1.f,1.f},{-1.f,1.f},{-1.f,-1.f},{1.f,-1.f}}};
    torch::Tensor sdf_legs;
    for (std::size_t i = 0; i < 4; ++i)
    {
        const auto ox = sgn[i][0] * (half_w - R);
        const auto oy = sgn[i][1] * (half_d - R);
        const auto leg = box_sdf_tensor(torch::abs(local_x - ox) - R, torch::abs(local_y - oy) - R, torch::abs(local_z - leg_cz) - leg_half_z);
        sdf_legs = (i == 0) ? leg : torch::minimum(sdf_legs, leg);
    }
    const auto sdf = torch::minimum(sdf_seat, torch::minimum(sdf_back, sdf_legs));

    const float inv_sigma2 = 1.0f / (params.sigma_obs * params.sigma_obs);
    const auto point_loss = robust_loss_value(sdf, params.robust_loss, robust_scale);
    // Normalise by point count to match Python prototype (belief_manager.py line 385):
    //   weighted_likelihood = torch.sum(weights * sdf**2) / len(points)
    const auto likelihood = (weights_tensor * point_loss * inv_sigma2).sum() /
                            static_cast<double>(points_tensor.size(0));

    const float sigma = params.prior_size_std > 0.f ? params.prior_size_std : 0.10f;
    const float inv_prior_sigma2 = 1.0f / (sigma * sigma);
    const auto dsw = seat_w - params.prior_seat_w;
    const auto dsd = seat_d - params.prior_seat_d;
    const auto dsh = seat_h - params.prior_seat_h;
    const auto dbh = back_h - params.prior_back_h;
    const auto size_energy = params.lambda_size * (dsw*dsw + dsd*dsd + dsh*dsh + dbh*dbh) * inv_prior_sigma2;
    const auto dpx = cx - prior.cx;
    const auto dpy = cy - prior.cy;
    const auto pos_energy = params.lambda_pos * (dpx*dpx + dpy*dpy);
    const auto s_sw = seat_w - prior.seat_w; const auto s_sd = seat_d - prior.seat_d;
    const auto s_sh = seat_h - prior.seat_h; const auto s_bh = back_h - prior.back_h;
    const auto state_energy = params.lambda_state * (s_sw*s_sw + s_sd*s_sd + s_sh*s_sh + s_bh*s_bh);
    auto dyaw = yaw - prior.yaw;
    dyaw = dyaw - (2.0f * M_PIf) * torch::floor((dyaw + M_PIf) / (2.0f * M_PIf));
    const auto angle_energy = params.lambda_angle * dyaw * dyaw;
    auto total = likelihood + size_energy + pos_energy + state_energy + angle_energy;

    // Footprint-extent term — see footprint_extent_energy(): pulls the top rectangle (centre +
    // half-extents) onto the observed 2–98 percentile point span in chair-local frame, removing
    // the flat-interior degeneracy that otherwise lets the box drift/oversize at ~zero energy.
    if (params.lambda_extent > 0.0f && points_tensor.size(0) >= 8)
    {
        const auto q = torch::tensor({params.extent_pct_lo, params.extent_pct_hi}, points_tensor.options());
        const auto qx = torch::quantile(local_x, q);
        const auto qy = torch::quantile(local_y, q);
        const auto ex = (qx.index({1}) - qx.index({0})) * 0.5f;
        const auto ey = (qy.index({1}) - qy.index({0})) * 0.5f;
        const auto mx = (qx.index({1}) + qx.index({0})) * 0.5f;
        const auto my = (qy.index({1}) + qy.index({0})) * 0.5f;
        const auto dw = half_w - ex;
        const auto dh = half_d - ey;
        total = total + params.lambda_extent * (dw * dw + dh * dh + mx * mx + my * my);
    }

    return total;
}

} // anonymous namespace

// ─── ChairModel ──────────────────────────────────────────────────────────────

ChairModel::ChairModel(const ChairState& prior, const ChairModelParams& params)
    : state_(prior), prior_(prior), params_(params)
{
    apply_constraints();
}

// ─── SDF ─────────────────────────────────────────────────────────────────────

float ChairModel::sdf_point_at(const Eigen::Vector3f& p, const ChairState& s) const
{
    const float cos_t = std::cos(-s.yaw);
    const float sin_t = std::sin(-s.yaw);
    const float px = p.x() - s.cx;
    const float py = p.y() - s.cy;
    const float lx = px * cos_t - py * sin_t;
    const float ly = px * sin_t + py * cos_t;
    const float lz = p.z() - s.cz;

    const float half_w = s.seat_w * 0.5f;
    const float half_d = s.seat_d * 0.5f;
    const float half_t = SEAT_THICKNESS * 0.5f;

    const float seat_cz = s.seat_h - half_t;
    const float sdf_seat = box_sdf(std::abs(lx) - half_w, std::abs(ly) - half_d, std::abs(lz - seat_cz) - half_t);

    const float back_cy = -half_d + half_t;
    const float back_cz =  s.seat_h + s.back_h * 0.5f;
    const float sdf_back = box_sdf(std::abs(lx) - half_w, std::abs(ly - back_cy) - half_t, std::abs(lz - back_cz) - s.back_h * 0.5f);

    const float leg_top    = s.seat_h - SEAT_THICKNESS;
    const float leg_cz     = leg_top * 0.5f;
    const float leg_half_z = leg_top * 0.5f;
    const float ox = half_w - LEG_HALF;
    const float oy = half_d - LEG_HALF;
    const float leg_off[4][2] = {{ ox, oy}, {-ox, oy}, {-ox, -oy}, { ox, -oy}};
    float sdf_legs = std::numeric_limits<float>::max();
    for (const auto& o : leg_off)
        sdf_legs = std::min(sdf_legs, box_sdf(std::abs(lx - o[0]) - LEG_HALF, std::abs(ly - o[1]) - LEG_HALF, std::abs(lz - leg_cz) - leg_half_z));

    return std::min(sdf_seat, std::min(sdf_back, sdf_legs));
}

float ChairModel::sdf_point(const Eigen::Vector3f& p) const
{
    return sdf_point_at(p, state_);
}

std::vector<float> ChairModel::compute_sdf(const std::vector<Eigen::Vector3f>& points) const
{
    std::vector<float> out;
    out.reserve(points.size());
    for (const auto& p : points)
        out.push_back(sdf_point(p));
    return out;
}

std::array<float, 8> ChairModel::observation_information(
    const std::vector<Eigen::Vector3f>& points,
    const std::vector<float>& weights) const
{
    std::array<float, 8> info{};
    const std::size_t n = points.size();
    if (n == 0)
        return info;

    const float inv_sigma2 = 1.0f / (params_.sigma_obs * params_.sigma_obs);

    // Central-difference step per DOF (metres for translations/sizes, radians for yaw). Kept small
    // so the linearisation is local but large enough to stay clear of float-cancellation noise.
    const std::array<float, 8> eps = {1e-3f, 1e-3f, 1e-3f, 1e-3f, 1e-3f, 1e-3f, 1e-3f, 1e-3f};

    // (B) Clamp the finite-difference slope. The box SDF is min() over faces+legs, hence non-smooth:
    // a point straddling a face-switch/corner makes (gp-gm) a discontinuity JUMP, so g=(gp-gm)/2eps
    // blows up ~1000× and a single point would dominate the Fisher (→ K→1, stabiliser bypassed). For a
    // well-behaved SDF point |∂SDF/∂θ| is O(1), so clamping the slope rejects the kink without
    // distorting genuine curvature. 0 disables.
    const float gmax = params_.fisher_grad_clamp;

    const auto base = state_.to_array();
    for (std::size_t i = 0; i < n; ++i)
    {
        const Eigen::Vector3f& p = points[i];
        const float r  = sdf_point_at(p, state_);
        const float wi = (weights.empty() ? 1.0f : weights[i]) *
                         robust_irls_weight(r, params_.robust_loss, params_.robust_loss_scale);
        if (wi <= 0.0f)
            continue;                       // hard-rejected outlier — no curvature contribution
        for (int j = 0; j < 8; ++j)
        {
            auto plus = base, minus = base;
            plus[j]  += eps[j];
            minus[j] -= eps[j];
            const float gp = sdf_point_at(p, ChairState::from_array(plus));
            const float gm = sdf_point_at(p, ChairState::from_array(minus));
            float g  = (gp - gm) / (2.0f * eps[j]);   // ∂SDF/∂θ_j
            if (gmax > 0.0f)
                g = std::clamp(g, -gmax, gmax);
            info[j] += wi * inv_sigma2 * g * g;
        }
    }
    return info;
}

ExtentDiagnostics ChairModel::extent_diagnostics(const std::vector<Eigen::Vector3f>& points) const
{
    ExtentDiagnostics diag;
    diag.n_total = static_cast<int>(points.size());
    if (points.empty())
        return diag;

    const float cos_t = std::cos(-state_.yaw);
    const float sin_t = std::sin(-state_.yaw);
    const float half_t = SEAT_THICKNESS * 0.5f;
    const float seat_cz = state_.seat_h - half_t;   // seat slab centre (local z)

    std::vector<float> lx, ly;
    lx.reserve(points.size());
    ly.reserve(points.size());
    for (const auto& p : points)
    {
        const float px = p.x() - state_.cx, py = p.y() - state_.cy;
        const float local_x = px * cos_t - py * sin_t;
        const float local_y = px * sin_t + py * cos_t;
        lx.push_back(local_x);
        ly.push_back(local_y);

        // Attribution: points inside the seat z-band count as seat, the rest as legs/back.
        const float lz = p.z() - state_.cz;
        (std::abs(lz - seat_cz) <= half_t * 2.0f ? diag.n_top : diag.n_leg) += 1;
    }

    const std::array<std::pair<float, float>, 3> qs = {{{0.02f, 0.98f}, {0.05f, 0.95f}, {0.10f, 0.90f}}};
    for (int k = 0; k < 3; ++k)
    {
        std::vector<float> cx = lx, cy = ly;   // percentile_inplace consumes its argument
        const float xlo = percentile_inplace(cx, qs[k].first);
        std::vector<float> cx2 = lx;
        const float xhi = percentile_inplace(cx2, qs[k].second);
        const float ylo = percentile_inplace(cy, qs[k].first);
        std::vector<float> cy2 = ly;
        const float yhi = percentile_inplace(cy2, qs[k].second);
        diag.half_ex[k] = 0.5f * (xhi - xlo);
        diag.half_ey[k] = 0.5f * (yhi - ylo);
        if (k == 0) { diag.off_x = 0.5f * (xhi + xlo); diag.off_y = 0.5f * (yhi + ylo); }
    }
    return diag;
}

// ─── Free Energy ─────────────────────────────────────────────────────────────

float ChairModel::prior_energy(const ChairState& s) const
{
    const float sigma = params_.prior_size_std > 0.0f ? params_.prior_size_std : 0.10f;
    const float inv_sigma2 = 1.0f / (sigma * sigma);

    // Size prior: penalise deviation from prior dimensions
    float dsw_p = s.seat_w - params_.prior_seat_w;
    float dsd_p = s.seat_d - params_.prior_seat_d;
    float dsh_p = s.seat_h - params_.prior_seat_h;
    float dbh_p = s.back_h - params_.prior_back_h;
    float size_energy = params_.lambda_size * (dsw_p*dsw_p + dsd_p*dsd_p + dsh_p*dsh_p + dbh_p*dbh_p) * inv_sigma2;

    // Position transition prior: penalise drift from prior centre
    float dpx = s.cx - prior_.cx;
    float dpy = s.cy - prior_.cy;
    float pos_energy = params_.lambda_pos * (dpx*dpx + dpy*dpy);

    // Size state transition prior: penalise change from prior state
    float dsw = s.seat_w - prior_.seat_w;
    float dsd = s.seat_d - prior_.seat_d;
    float dsh = s.seat_h - prior_.seat_h;
    float dbh = s.back_h - prior_.back_h;
    float state_energy = params_.lambda_state * (dsw*dsw + dsd*dsd + dsh*dsh + dbh*dbh);

    // Angle transition prior
    float dyaw = s.yaw - prior_.yaw;
    // Wrap to [-π, π]
    dyaw -= 2.0f * M_PIf * std::floor((dyaw + M_PIf) / (2.0f * M_PIf));
    float angle_energy = params_.lambda_angle * dyaw * dyaw;

    return size_energy + pos_energy + state_energy + angle_energy;
}

// Per-ray silhouette residual: closest approach of the ray to the box+legs SDF (tangent mode), or
// the chair-top-plane 2-D box-SDF (legacy). 0 on the boundary; >0 outside; <0 piercing.
float ChairModel::silhouette_ray_metric(const Eigen::Vector3f&, const ChairState&) const
{
    // Silhouette term inert for the chair (first build). See header note.
    return 0.0f;
}

float ChairModel::silhouette_energy_at(const ChairState&) const
{
    return 0.0f;
}

float ChairModel::silhouette_residual() const
{
    return 0.0f;
}

float ChairModel::fe_at(const ChairState& s,
                         const std::vector<Eigen::Vector3f>& pts,
                         const std::vector<float>& weights) const
{
    return fe_terms_at(s, pts, weights, 0).total_fe;
}

FreeEnergyDecomposition ChairModel::fe_terms_at(const ChairState& s,
                                                const std::vector<Eigen::Vector3f>& pts,
                                                const std::vector<float>& weights,
                                                std::size_t historical_count) const
{
    FreeEnergyDecomposition terms;
    const float inv_sigma2 = 1.0f / (params_.sigma_obs * params_.sigma_obs);
    const bool uniform = weights.empty();
    const std::size_t split = std::min(historical_count, pts.size());

    for (std::size_t i = 0; i < pts.size(); ++i)
    {
        const float sdf = sdf_point_at(pts[i], s);
        const float w = uniform ? 1.0f : weights[i];
        const float contribution = w * robust_loss_value(sdf, params_.robust_loss, params_.robust_loss_scale) * inv_sigma2;

        terms.effective_weight_mass += w;
        if (i < split)
        {
            terms.likelihood_historical += contribution;
            terms.historical_weight_mass += w;
        }
        else
        {
            terms.likelihood_current += contribution;
            terms.current_weight_mass += w;
        }
    }

    // Normalise by point count to match Python prototype (belief_manager.py line 385):
    //   weighted_likelihood = torch.sum(weights * sdf**2) / len(points)
    const float inv_N = 1.0f / static_cast<float>(std::max(pts.size(), std::size_t{1}));
    terms.likelihood_current *= inv_N;
    terms.likelihood_historical *= inv_N;

    terms.prior = prior_energy(s) + footprint_extent_energy(s, pts, params_.lambda_extent)
                + silhouette_energy_at(s);
    terms.total_fe = terms.likelihood_current + terms.likelihood_historical + terms.prior;
    return terms;
}

float ChairModel::compute_free_energy(const std::vector<Eigen::Vector3f>& points,
                                      const std::vector<float>& weights) const
{
    if (points.empty())
        return prior_energy(state_);
    return fe_at(state_, points, weights);
}

FreeEnergyDecomposition ChairModel::compute_free_energy_decomposition(
    const std::vector<Eigen::Vector3f>& points,
    const std::vector<float>& weights,
    std::size_t historical_count) const
{
    return fe_terms_at(state_, points, weights, historical_count);
}

// ─── Gradient step ───────────────────────────────────────────────────────────

float ChairModel::gradient_step(const std::vector<Eigen::Vector3f>& points,
                                 const std::vector<float>& weights,
                                 std::size_t historical_count,
                                 const IterationObserver& observer,
                                 float gnc_start_override)
{
    if (points.empty())
        return prior_energy(state_);

    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

    auto points_tensor = torch::empty({static_cast<long>(points.size()), 3}, options);
    {
        auto pts_acc = points_tensor.accessor<float, 2>();
        for (std::size_t i = 0; i < points.size(); ++i)
        {
            pts_acc[i][0] = points[i].x();
            pts_acc[i][1] = points[i].y();
            pts_acc[i][2] = points[i].z();
        }
    }

    torch::Tensor weights_tensor;
    if (weights.empty())
    {
        weights_tensor = torch::ones({static_cast<long>(points.size())}, options);
    }
    else
    {
        weights_tensor = torch::empty({static_cast<long>(weights.size())}, options);
        auto w_acc = weights_tensor.accessor<float, 1>();
        for (std::size_t i = 0; i < weights.size(); ++i)
            w_acc[i] = weights[i];
    }

    auto arr = state_.to_array();
    auto theta = torch::tensor(
        {arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7]},
        options).set_requires_grad(true);

    auto theta_to_state = [&](const torch::Tensor& tensor) -> ChairState
    {
        auto cpu = tensor.detach().to(torch::kCPU);
        auto acc = cpu.accessor<float, 1>();
        return ChairState{acc[0], acc[1], acc[2], acc[3], acc[4], acc[5], acc[6], acc[7]};
    };

    const ChairState fallback_state = state_;

    // Graduated non-convexity: geometric anneal of the robust-loss scale from the wide start
    // down to the target across the iterations. A flat top-face SDF gives covered points zero
    // gradient, and a sharp Tukey scale hard-rejects the uncovered points as outliers (constant
    // loss, zero gradient) — together a half-offset pose is a flat local minimum. Starting wide
    // keeps those uncovered points inside the robust band so they pull the box over before the
    // scale sharpens. Disabled (start == target) when robust_gnc_start_scale <= robust_loss_scale.
    const float gnc_target = params_.robust_loss_scale;
    // The fitter decays the start scale over the instance lifetime (wide → target) so coarse
    // alignment happens early and the converged pose stops being re-widened; -1 = use the param.
    const float gnc_start  = (gnc_start_override >= 0.0f) ? gnc_start_override : params_.robust_gnc_start_scale;
    const bool  gnc_active = gnc_start > gnc_target && params_.optimization_iters > 1;
    auto scale_for_iter = [&](int iter) -> float
    {
        if (!gnc_active)
            return gnc_target;
        const float t = static_cast<float>(iter) / static_cast<float>(params_.optimization_iters - 1);
        return gnc_start * std::pow(gnc_target / gnc_start, t);   // start → target, geometric
    };

    // RGB-mask silhouette term: each back-projected contour ray (dir d from camera C) is intersected
    // with the chair-top plane z=H → room-XY point B; B should lie on the top rectangle boundary, so
    // its 2-D box-SDF is driven to 0. Differentiable in cx,cy,w,h,yaw (and H via the intersection).
    const bool sil_on = params_.mask_precision > 0.0f && !sil_dirs_.empty();
    torch::Tensor sil_dx, sil_dy, sil_dz;
    if (sil_on)
    {
        const long n = static_cast<long>(sil_dirs_.size());
        sil_dx = torch::empty({n}, options);
        sil_dy = torch::empty({n}, options);
        sil_dz = torch::empty({n}, options);
        auto ax = sil_dx.accessor<float, 1>();
        auto ay = sil_dy.accessor<float, 1>();
        auto az = sil_dz.accessor<float, 1>();
        for (long i = 0; i < n; ++i) { ax[i] = sil_dirs_[i].x(); ay[i] = sil_dirs_[i].y(); az[i] = sil_dirs_[i].z(); }
    }
    const float sil_Cx = sil_cam_.x(), sil_Cy = sil_cam_.y(), sil_Cz = sil_cam_.z();
    using torch::indexing::Slice;
    auto silhouette_loss = [&](const torch::Tensor&, float) -> torch::Tensor
    {
        // Silhouette term inert for the chair (first build). mask_precision default 0.
        (void)sil_on; (void)sil_Cx; (void)sil_Cy; (void)sil_Cz;
        return torch::zeros({}, options);
    };

    auto run_loop = [&](auto& optimizer) -> bool
    {
        for (int iter = 0; iter < params_.optimization_iters; ++iter)
        {
            optimizer.zero_grad();
            auto loss = fe_torch_impl(params_, prior_, theta, points_tensor, weights_tensor, scale_for_iter(iter))
                      + silhouette_loss(theta, scale_for_iter(iter));
            const float loss_value = loss.item<float>();
            if (!std::isfinite(loss_value))
                return false;
            loss.backward();

            if (theta.grad().defined() && params_.grad_clip > 0.0f)
            {
                torch::NoGradGuard no_grad;
                theta.mutable_grad().clamp_(-params_.grad_clip, params_.grad_clip);
            }

            optimizer.step();
            apply_constraints_to_tensor(theta);

            const ChairState iter_state = theta_to_state(theta);
            if (!finite_state(iter_state))
                return false;

            if (observer)
            {
                observer(iter, iter_state, fe_terms_at(iter_state, points, weights, historical_count));
            }
        }
        return true;
    };

    bool ok = false;
    const std::string opt = params_.optimizer_type;
    if (opt == "sgd")
    {
        torch::optim::SGD optimizer(
            {theta},
            torch::optim::SGDOptions(params_.optimization_lr).momentum(params_.sgd_momentum));
        ok = run_loop(optimizer);
    }
    else
    {
        torch::optim::Adam optimizer(
            {theta},
            torch::optim::AdamOptions(params_.optimization_lr));
        ok = run_loop(optimizer);
    }

    if (!ok)
    {
        state_ = fallback_state;
        apply_constraints();
        return fe_at(state_, points, weights);
    }

    state_ = theta_to_state(theta);
    if (!finite_state(state_))
    {
        state_ = fallback_state;
        apply_constraints();
        return fe_at(state_, points, weights);
    }
    apply_constraints();

    // Report the final FE at the SHARP target scale (matches fe_terms_at / acceptance), not the
    // annealed scale, so convergence/acceptance see the true objective.
    const float final_fe = (fe_torch_impl(params_, prior_, theta.detach(), points_tensor, weights_tensor,
                                          params_.robust_loss_scale)
                            + silhouette_loss(theta.detach(), params_.robust_loss_scale)).item<float>();
    if (!std::isfinite(final_fe))
    {
        state_ = fallback_state;
        apply_constraints();
        return fe_at(state_, points, weights);
    }

    return final_fe;
}

// ─── Constraints ─────────────────────────────────────────────────────────────

void ChairModel::apply_constraints()
{
    // Physical bounds for a wooden chair (Webots WoodenChair ≈ 0.45×0.45, seat ~0.46 m, total ~0.92).
    // A hard ceiling stops the flat seat slab from ballooning to swallow contaminated mask points,
    // independent of the (soft) size prior.
    state_.seat_w = std::clamp(state_.seat_w, 0.30f, 0.65f);
    state_.seat_d = std::clamp(state_.seat_d, 0.30f, 0.65f);
    state_.seat_h = std::clamp(state_.seat_h, 0.38f, 0.55f);
    state_.back_h = std::clamp(state_.back_h, 0.30f, 0.65f);
}

// ─── Bounding box ────────────────────────────────────────────────────────────

std::pair<Eigen::Vector3f, Eigen::Vector3f> ChairModel::bounding_box() const
{
    const float s = std::sin(state_.yaw);
    const float c = std::cos(state_.yaw);

    // Half-extents of the seat footprint in local frame
    const float hw = state_.seat_w * 0.5f;
    const float hd = state_.seat_d * 0.5f;

    // Project corners to room XY
    float min_x =  std::numeric_limits<float>::max();
    float max_x = -std::numeric_limits<float>::max();
    float min_y =  std::numeric_limits<float>::max();
    float max_y = -std::numeric_limits<float>::max();

    for (float lx : {-hw, hw})
    {
        for (float ly : {-hd, hd})
        {
            float rx = state_.cx + c * lx - s * ly;
            float ry = state_.cy + s * lx + c * ly;
            min_x = std::min(min_x, rx);
            max_x = std::max(max_x, rx);
            min_y = std::min(min_y, ry);
            max_y = std::max(max_y, ry);
        }
    }

    return {
        Eigen::Vector3f{min_x, min_y, state_.cz},
        Eigen::Vector3f{max_x, max_y, state_.cz + state_.seat_h + state_.back_h}
    };
}

}  // namespace rc
