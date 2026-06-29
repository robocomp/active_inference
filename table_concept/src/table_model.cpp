/*
 * table_model.cpp
 *
 * Compound SDF (box-top + 4 cylindrical legs) ported from the Python prototype
 * box_concept/src/objects/table/sdf.py, replacing PyTorch ops with Eigen3.
 * Gradient computed with PyTorch autograd (Adam/SGD optimizers).
 */

#include "table_model.h"
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

bool finite_state(const TableState& state)
{
    return std::isfinite(state.cx) && std::isfinite(state.cy) &&
           std::isfinite(state.w) && std::isfinite(state.h) &&
           std::isfinite(state.table_height) && std::isfinite(state.leg_length) &&
           std::isfinite(state.yaw) && std::isfinite(state.leg_inset);
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
        constexpr float k = TableModel::SDF_SMOOTH_K;
        constexpr float s = TableModel::SDF_INSIDE_SCALE;
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
 * observed point extent in the table-local frame. Penalises the box being off-centre or larger/
 * smaller than the 2–98 percentile point span. This is what makes position/size non-degenerate
 * when only the (flat) top face is observed. Scalar mirror of the autograd term in fe_torch_impl.
 */
float footprint_extent_energy(const TableState& s, const std::vector<Eigen::Vector3f>& pts, float lambda)
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
    const float dw = s.w * 0.5f - ex, dh = s.h * 0.5f - ey;
    return lambda * (dw * dw + dh * dh + mx * mx + my * my);
}

/** Finite cylinder SDF (radius r, half-height hh, centred at origin). */
float cylinder_sdf(float dx_local, float dy_local, float dz_local, float r, float hh)
{
    float d_radial   = std::sqrt(dx_local*dx_local + dy_local*dy_local) - r;
    float d_vertical = std::abs(dz_local) - hh;

    float or_ = std::max(d_radial, 0.0f);
    float ov  = std::max(d_vertical, 0.0f);
    float outside = std::sqrt(or_*or_ + ov*ov);
    float inside  = std::min(std::max(d_radial, d_vertical), 0.0f);
    return outside + inside;
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
    // optimizer. The eps offset (~3e-5 m) is negligible for a table SDF.
    const auto outside = torch::sqrt(ox * ox + oy * oy + oz * oz + 1e-9f);

    constexpr float k = TableModel::SDF_SMOOTH_K;
    constexpr float s = TableModel::SDF_INSIDE_SCALE;
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

torch::Tensor cylinder_sdf_tensor(const torch::Tensor& dx_local,
                                   const torch::Tensor& dy_local,
                                   const torch::Tensor& dz_local,
                                   float r,
                                   float hh)
{
    const auto d_radial = torch::sqrt(dx_local * dx_local + dy_local * dy_local + 1e-9f) - r;
    const auto d_vertical = torch::abs(dz_local) - hh;

    const auto or_ = torch::clamp_min(d_radial, 0.0f);
    const auto ov = torch::clamp_min(d_vertical, 0.0f);
    const auto outside = torch::sqrt(or_ * or_ + ov * ov + 1e-9f);
    const auto inside = torch::clamp_max(torch::max(d_radial, d_vertical), 0.0f);
    return outside + inside;
}

void apply_constraints_to_tensor(torch::Tensor& arr)
{
    torch::NoGradGuard no_grad;
    arr.index_put_({2}, torch::clamp_min(arr.index({2}), 0.1f));
    arr.index_put_({3}, torch::clamp_min(arr.index({3}), 0.1f));
    arr.index_put_({4}, torch::clamp_min(arr.index({4}), 0.05f + TableModel::TOP_THICKNESS));
    const auto max_leg = arr.index({4}) - TableModel::TOP_THICKNESS;
    arr.index_put_({5}, torch::clamp(arr.index({5}), 0.05f, max_leg.item<float>()));
    // leg_inset FROZEN at the outer edge — pin it each iteration so the optimiser's gradient on
    // index 7 is discarded (legs stay flush with the top edge; see apply_constraints()).
    arr.index_put_({7}, TableModel::LEG_RADIUS);
}

torch::Tensor fe_torch_impl(const TableModelParams& params,
                             const TableState& prior,
                             const torch::Tensor& state_tensor,
                             const torch::Tensor& points_tensor,
                             const torch::Tensor& weights_tensor,
                             float robust_scale,
                             float top_split_z)
{
    const auto cx = state_tensor.index({0});
    const auto cy = state_tensor.index({1});
    const auto w = state_tensor.index({2});
    const auto h = state_tensor.index({3});
    const auto table_height = state_tensor.index({4});
    const auto leg_length = state_tensor.index({5});
    const auto yaw = state_tensor.index({6});
    const auto leg_inset = state_tensor.index({7});

    const auto px = points_tensor.index({Slice(), 0}) - cx;
    const auto py = points_tensor.index({Slice(), 1}) - cy;
    const auto pz = points_tensor.index({Slice(), 2});

    const auto c = torch::cos(yaw);
    const auto s = torch::sin(yaw);
    const auto local_x = px * c + py * s;
    const auto local_y = -px * s + py * c;
    const auto local_z = pz;

    const auto half_w = w * 0.5f;
    const auto half_h = h * 0.5f;
    const float half_t = TableModel::TOP_THICKNESS * 0.5f;
    const auto top_cz = table_height - half_t;

    const auto dx_top = torch::abs(local_x) - half_w;
    const auto dy_top = torch::abs(local_y) - half_h;
    const auto dz_top = torch::abs(local_z - top_cz) - half_t;
    const auto sdf_top = box_sdf_tensor(dx_top, dy_top, dz_top);

    const auto leg_center_z = leg_length * 0.5f;
    const auto leg_half_h = leg_length * 0.5f;

    const std::array<std::array<float, 2>, 4> leg_offsets = {{
        { 1.0f,  1.0f},
        {-1.0f,  1.0f},
        {-1.0f, -1.0f},
        { 1.0f, -1.0f},
    }};

    torch::Tensor sdf_leg_min;
    for (std::size_t i = 0; i < leg_offsets.size(); ++i)
    {
        const auto off_x = leg_offsets[i][0] * (half_w - leg_inset);
        const auto off_y = leg_offsets[i][1] * (half_h - leg_inset);
        const auto dlx = local_x - off_x;
        const auto dly = local_y - off_y;
        const auto dlz = local_z - leg_center_z;
        const auto leg = cylinder_sdf_tensor(dlx, dly, dlz, TableModel::LEG_RADIUS, leg_half_h.item<float>());
        if (i == 0)
            sdf_leg_min = leg;
        else
            sdf_leg_min = torch::minimum(sdf_leg_min, leg);
    }

    // Height-based top/leg attribution (replaces nearest-primitive min(top,legs)). A point above the
    // split height is a TOP observation even when it lies horizontally near a corner leg — under the
    // old min() an undersized box let the corner leg cylinders "absorb" the top-edge points (≈2/3 of
    // all points) and PIN the size, so the box sat ~15% under the mask and never grew. The split is a
    // CONSTANT (anchored to the observed top face, see top_split_z) so the gate is a fixed per-point
    // classification — no gradient through the discrete choice — which also removes the min()-SDF kink
    // the Fisher clamp was fighting.
    const auto top_mask = (local_z >= top_split_z).to(points_tensor.dtype());
    const auto sdf = top_mask * sdf_top + (1.0f - top_mask) * sdf_leg_min;

    const float inv_sigma2 = 1.0f / (params.sigma_obs * params.sigma_obs);
    const auto point_loss = robust_loss_value(sdf, params.robust_loss, robust_scale);
    // Normalise by point count to match Python prototype (belief_manager.py line 385):
    //   weighted_likelihood = torch.sum(weights * sdf**2) / len(points)
    const auto likelihood = (weights_tensor * point_loss * inv_sigma2).sum() /
                            static_cast<double>(points_tensor.size(0));

    const float sigma = prior.leg_length > 0.0f ? params.prior_size_std : 0.15f;
    const float inv_prior_sigma2 = 1.0f / (sigma * sigma);

    const auto dw = w - params.prior_w;
    const auto dh = h - params.prior_h;
    const auto dt = table_height - params.prior_table_height;
    const auto size_energy = params.lambda_size * (dw * dw + dh * dh + dt * dt) * inv_prior_sigma2;

    const auto dpx = cx - prior.cx;
    const auto dpy = cy - prior.cy;
    const auto pos_energy = params.lambda_pos * (dpx * dpx + dpy * dpy);

    const auto dsw = w - prior.w;
    const auto dsh = h - prior.h;
    const auto dst = table_height - prior.table_height;
    const auto dsl = leg_length - prior.leg_length;
    const auto dsi = leg_inset - prior.leg_inset;
    const auto state_energy = params.lambda_state * (dsw * dsw + dsh * dsh + dst * dst + dsl * dsl + dsi * dsi);

    auto dyaw = yaw - prior.yaw;
    dyaw = dyaw - (2.0f * M_PIf) * torch::floor((dyaw + M_PIf) / (2.0f * M_PIf));
    const auto angle_energy = params.lambda_angle * dyaw * dyaw;

    auto total = likelihood + size_energy + pos_energy + state_energy + angle_energy;

    // Footprint-extent term — see footprint_extent_energy(): pulls the top rectangle (centre +
    // half-extents) onto the observed 2–98 percentile point span in table-local frame, removing
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
        const auto dh = half_h - ey;
        total = total + params.lambda_extent * (dw * dw + dh * dh + mx * mx + my * my);
    }

    return total;
}

} // anonymous namespace

void set_torch_threads(int n)
{
    torch::set_num_threads(std::max(1, n));
}

// ─── Top/leg attribution split ─────────────────────────────────────────────────

void TableModel::update_top_reference(const std::vector<Eigen::Vector3f>& pts) const
{
    if (pts.size() < 4)
        return;   // too few points to estimate a top face — keep the previous reference
    std::vector<float> z;
    z.reserve(pts.size());
    for (const auto& p : pts)
        z.push_back(p.z());
    // 95th percentile of z = robust observed top face (rejects a few high outliers/noise).
    const std::size_t k = static_cast<std::size_t>(0.95f * (z.size() - 1));
    std::nth_element(z.begin(), z.begin() + k, z.end());
    top_ref_z_ = z[k];
}

float TableModel::top_split_z(const TableState& s) const
{
    // Top face: the observed cloud top when available, else the belief surface height. A point is a
    // top-slab observation if it sits no more than (slab thickness + one obs-sigma) below that face.
    const float top_face = std::isfinite(top_ref_z_) ? top_ref_z_ : s.table_height;
    return top_face - (TOP_THICKNESS + params_.sigma_obs);
}

// ─── TableModel ──────────────────────────────────────────────────────────────

TableModel::TableModel(const TableState& prior, const TableModelParams& params)
    : state_(prior), prior_(prior), params_(params)
{
    apply_constraints();
}

// ─── SDF ─────────────────────────────────────────────────────────────────────

float TableModel::sdf_point_at(const Eigen::Vector3f& p, const TableState& s) const
{
    const float cx = s.cx, cy = s.cy;
    const float w = s.w, h = s.h;
    const float table_height = s.table_height;
    const float leg_length   = s.leg_length;
    const float yaw          = s.yaw;

    // Transform to table-local frame
    const float cos_t = std::cos(-yaw);
    const float sin_t = std::sin(-yaw);
    const float px = p.x() - cx;
    const float py = p.y() - cy;
    const float local_x = px * cos_t - py * sin_t;
    const float local_y = px * sin_t + py * cos_t;
    const float local_z = p.z();

    // ── TABLE TOP ────────────────────────────────────────────────────────────
    const float half_w = w * 0.5f;
    const float half_h = h * 0.5f;
    const float half_t = TOP_THICKNESS * 0.5f;
    const float top_cz = table_height - half_t;

    const float dx_top = std::abs(local_x) - half_w;
    const float dy_top = std::abs(local_y) - half_h;
    const float dz_top = std::abs(local_z - top_cz) - half_t;

    const float sdf_top = box_sdf(dx_top, dy_top, dz_top);

    // ── LEGS (4 cylinders at corners) ────────────────────────────────────────
    const float leg_inset    = s.leg_inset;
    const float leg_center_z = leg_length * 0.5f;
    const float leg_half_h   = leg_length * 0.5f;

    const float leg_offsets[4][2] = {
        { half_w - leg_inset,  half_h - leg_inset},
        {-half_w + leg_inset,  half_h - leg_inset},
        {-half_w + leg_inset, -half_h + leg_inset},
        { half_w - leg_inset, -half_h + leg_inset},
    };

    float sdf_leg_min = std::numeric_limits<float>::max();
    for (const auto& off : leg_offsets)
    {
        const float dlx = local_x - off[0];
        const float dly = local_y - off[1];
        const float dlz = local_z - leg_center_z;
        sdf_leg_min = std::min(sdf_leg_min, cylinder_sdf(dlx, dly, dlz, LEG_RADIUS, leg_half_h));
    }

    // Height-based attribution — scalar mirror of fe_torch_impl (split anchored to the observed top).
    return (local_z >= top_split_z(s)) ? sdf_top : sdf_leg_min;
}

float TableModel::sdf_point(const Eigen::Vector3f& p) const
{
    return sdf_point_at(p, state_);
}

std::vector<float> TableModel::compute_sdf(const std::vector<Eigen::Vector3f>& points) const
{
    std::vector<float> out;
    out.reserve(points.size());
    for (const auto& p : points)
        out.push_back(sdf_point(p));
    return out;
}

std::array<float, 8> TableModel::observation_information(
    const std::vector<Eigen::Vector3f>& points,
    const std::vector<float>& weights) const
{
    std::array<float, 8> info{};
    const std::size_t n = points.size();
    if (n == 0)
        return info;

    update_top_reference(points);   // same top/leg split as the data term sees

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
            const float gp = sdf_point_at(p, TableState::from_array(plus));
            const float gm = sdf_point_at(p, TableState::from_array(minus));
            float g  = (gp - gm) / (2.0f * eps[j]);   // ∂SDF/∂θ_j
            if (gmax > 0.0f)
                g = std::clamp(g, -gmax, gmax);
            info[j] += wi * inv_sigma2 * g * g;
        }
    }
    return info;
}

ExtentDiagnostics TableModel::extent_diagnostics(const std::vector<Eigen::Vector3f>& points) const
{
    ExtentDiagnostics diag;
    diag.n_total = static_cast<int>(points.size());
    if (points.empty())
        return diag;

    update_top_reference(points);
    const float z_split = top_split_z(state_);

    const float cos_t = std::cos(-state_.yaw);
    const float sin_t = std::sin(-state_.yaw);

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

        // Attribution by HEIGHT (matches fe_torch_impl / sdf_point_at): above the split ⇒ a top
        // observation, below ⇒ a leg observation. (Was nearest-primitive min(top,legs).)
        (p.z() >= z_split ? diag.n_top : diag.n_leg) += 1;
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

float TableModel::prior_energy(const TableState& s) const
{
    const float sigma = prior_.leg_length > 0.0f ? params_.prior_size_std : 0.15f;
    const float inv_sigma2 = 1.0f / (sigma * sigma);

    // Size prior: penalise deviation from prior dimensions
    float dw = s.w            - params_.prior_w;
    float dh = s.h            - params_.prior_h;
    float dt = s.table_height - params_.prior_table_height;
    float size_energy = params_.lambda_size * (dw*dw + dh*dh + dt*dt) * inv_sigma2;

    // Position transition prior: penalise drift from prior centre
    float dpx = s.cx - prior_.cx;
    float dpy = s.cy - prior_.cy;
    float pos_energy = params_.lambda_pos * (dpx*dpx + dpy*dpy);

    // Size state transition prior: penalise change from prior state
    float dsw = s.w            - prior_.w;
    float dsh = s.h            - prior_.h;
    float dst = s.table_height - prior_.table_height;
    float dsl = s.leg_length   - prior_.leg_length;
    float dsi = s.leg_inset    - prior_.leg_inset;
    float state_energy = params_.lambda_state * (dsw*dsw + dsh*dsh + dst*dst + dsl*dsl + dsi*dsi);

    // Angle transition prior
    float dyaw = s.yaw - prior_.yaw;
    // Wrap to [-π, π]
    dyaw -= 2.0f * M_PIf * std::floor((dyaw + M_PIf) / (2.0f * M_PIf));
    float angle_energy = params_.lambda_angle * dyaw * dyaw;

    return size_energy + pos_energy + state_energy + angle_energy;
}

// Per-ray silhouette residual: closest approach of the ray to the box+legs SDF (tangent mode), or
// the table-top-plane 2-D box-SDF (legacy). 0 on the boundary; >0 outside; <0 piercing.
float TableModel::silhouette_ray_metric(const Eigen::Vector3f& d, const TableState& s) const
{
    const int N = params_.sil_tangent_samples;
    if (N > 0)
    {
        const float t_top = std::max(0.0f, (s.table_height - sil_cam_.z()) / d.z());
        const float t_flr = (0.0f - sil_cam_.z()) / d.z();
        float best = std::numeric_limits<float>::max();
        for (int k = 0; k < N; ++k)
        {
            const float a = (N > 1) ? static_cast<float>(k) / static_cast<float>(N - 1) : 0.0f;
            const float t = t_top + a * (t_flr - t_top);
            best = std::min(best, sdf_point_at(sil_cam_ + t * d, s));
        }
        return best;
    }
    const float c = std::cos(s.yaw), sn = std::sin(s.yaw);
    const float t = (s.table_height - sil_cam_.z()) / d.z();
    const float bx = sil_cam_.x() + t * d.x() - s.cx;
    const float by = sil_cam_.y() + t * d.y() - s.cy;
    const float lx = bx * c + by * sn, ly = -bx * sn + by * c;
    const float dx = std::abs(lx) - s.w * 0.5f, dy = std::abs(ly) - s.h * 0.5f;
    const float ox = std::max(dx, 0.0f), oy = std::max(dy, 0.0f);
    return std::sqrt(ox * ox + oy * oy) + std::min(std::max(dx, dy), 0.0f);
}

float TableModel::silhouette_energy_at(const TableState& s) const
{
    if (params_.mask_precision <= 0.0f || sil_dirs_.empty())
        return 0.0f;
    float acc = 0.0f;
    for (const auto& d : sil_dirs_)
    {
        if (std::abs(d.z()) < 1e-6f) continue;
        acc += robust_loss_value(silhouette_ray_metric(d, s), params_.robust_loss, params_.robust_loss_scale);
    }
    return params_.mask_precision * sil_conf_ * acc / static_cast<float>(sil_dirs_.size());
}

float TableModel::silhouette_residual() const
{
    if (sil_dirs_.empty())
        return 0.0f;
    float acc = 0.0f;
    int n = 0;
    for (const auto& d : sil_dirs_)
    {
        if (std::abs(d.z()) < 1e-6f) continue;
        acc += std::abs(silhouette_ray_metric(d, state_));
        ++n;
    }
    return n > 0 ? acc / static_cast<float>(n) : 0.0f;
}

float TableModel::fe_at(const TableState& s,
                         const std::vector<Eigen::Vector3f>& pts,
                         const std::vector<float>& weights) const
{
    return fe_terms_at(s, pts, weights, 0).total_fe;
}

FreeEnergyDecomposition TableModel::fe_terms_at(const TableState& s,
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

float TableModel::compute_free_energy(const std::vector<Eigen::Vector3f>& points,
                                      const std::vector<float>& weights) const
{
    if (points.empty())
        return prior_energy(state_);
    return fe_at(state_, points, weights);
}

FreeEnergyDecomposition TableModel::compute_free_energy_decomposition(
    const std::vector<Eigen::Vector3f>& points,
    const std::vector<float>& weights,
    std::size_t historical_count) const
{
    return fe_terms_at(state_, points, weights, historical_count);
}

// ─── Gradient step ───────────────────────────────────────────────────────────

float TableModel::gradient_step(const std::vector<Eigen::Vector3f>& points,
                                 const std::vector<float>& weights,
                                 std::size_t historical_count,
                                 const IterationObserver& observer,
                                 float gnc_start_override)
{
    if (points.empty())
        return prior_energy(state_);

    update_top_reference(points);                 // anchor the top/leg split to this frame's cloud
    const float top_split = top_split_z(state_);  // constant over the inner iterations

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

    auto theta_to_state = [&](const torch::Tensor& tensor) -> TableState
    {
        auto cpu = tensor.detach().to(torch::kCPU);
        auto acc = cpu.accessor<float, 1>();
        return TableState{acc[0], acc[1], acc[2], acc[3], acc[4], acc[5], acc[6], acc[7]};
    };

    const TableState fallback_state = state_;

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
    // with the table-top plane z=H → room-XY point B; B should lie on the top rectangle boundary, so
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
    auto silhouette_loss = [&](const torch::Tensor& th, float robust_scale) -> torch::Tensor
    {
        if (!sil_on) return torch::zeros({}, options);
        const auto cx = th.index({0}), cy = th.index({1});
        const auto w = th.index({2}), h = th.index({3});
        const auto half_w = w * 0.5f, half_h = h * 0.5f;
        const auto H = th.index({4}), leg_len = th.index({5}), yaw = th.index({6}), inset = th.index({7});
        const auto c = torch::cos(yaw), s = torch::sin(yaw);

        torch::Tensor metric;   // [Nrays] per-ray contour residual, driven to 0
        const int Nsamp = params_.sil_tangent_samples;
        if (Nsamp > 0)
        {
            // Height-agnostic occluding contour: sample each ray over the table's vertical band
            // [z=H .. z=0] and take the closest approach to the box+legs SDF (tangent ⇒ 0). This
            // matches top, sides AND legs in one term, with no plane assumption.
            const int N = std::max(2, Nsamp);
            const auto t_top = torch::clamp_min((H - sil_Cz) / sil_dz, 0.0f);   // [Nrays]
            const auto t_flr = (0.0f - sil_Cz) / sil_dz;                        // [Nrays]
            const auto a = torch::linspace(0.0f, 1.0f, N, options);             // [N]
            const auto t  = t_top.index({Slice(), torch::indexing::None})
                          + a.index({torch::indexing::None, Slice()}) * (t_flr - t_top).index({Slice(), torch::indexing::None});  // [Nrays,N]
            const auto px = sil_Cx + t * sil_dx.index({Slice(), torch::indexing::None});   // [Nrays,N]
            const auto py = sil_Cy + t * sil_dy.index({Slice(), torch::indexing::None});
            const auto pz = sil_Cz + t * sil_dz.index({Slice(), torch::indexing::None});
            const auto rx = px - cx, ry = py - cy;
            const auto lx = rx * c + ry * s;
            const auto ly = -rx * s + ry * c;
            // top slab
            const float half_t = TableModel::TOP_THICKNESS * 0.5f;
            const auto top_cz = H - half_t;
            auto sdf = box_sdf_tensor(torch::abs(lx) - half_w, torch::abs(ly) - half_h, torch::abs(pz - top_cz) - half_t);
            // 4 legs (vertical extent detached as a scalar, consistent with the data term)
            const float lr = TableModel::LEG_RADIUS;
            const float leg_hh = (leg_len * 0.5f).item<float>();
            const auto  leg_cz = leg_len * 0.5f;
            const std::array<std::array<float, 2>, 4> sg = {{ {1, 1}, {-1, 1}, {-1, -1}, {1, -1} }};
            for (const auto& g : sg)
            {
                const auto off_x = g[0] * (half_w - inset);
                const auto off_y = g[1] * (half_h - inset);
                sdf = torch::minimum(sdf, cylinder_sdf_tensor(lx - off_x, ly - off_y, pz - leg_cz, lr, leg_hh));
            }
            metric = std::get<0>(sdf.min(1));   // [Nrays] closest approach along each ray
        }
        else
        {
            // Legacy: intersect the table-top plane; 2-D box-SDF to the footprint rectangle.
            const auto t = (H - sil_Cz) / sil_dz;
            const auto rx = (sil_Cx + t * sil_dx) - cx, ry = (sil_Cy + t * sil_dy) - cy;
            const auto lx = rx * c + ry * s, ly = -rx * s + ry * c;
            const auto dx = torch::abs(lx) - half_w, dy = torch::abs(ly) - half_h;
            const auto ox = torch::clamp_min(dx, 0.0f), oy = torch::clamp_min(dy, 0.0f);
            metric = torch::sqrt(ox * ox + oy * oy + 1e-9f) + torch::clamp_max(torch::max(dx, dy), 0.0f);
        }
        const auto loss = robust_loss_value(metric, params_.robust_loss, robust_scale);
        return params_.mask_precision * sil_conf_ * loss.mean();
    };

    auto run_loop = [&](auto& optimizer) -> bool
    {
        for (int iter = 0; iter < params_.optimization_iters; ++iter)
        {
            optimizer.zero_grad();
            auto loss = fe_torch_impl(params_, prior_, theta, points_tensor, weights_tensor, scale_for_iter(iter), top_split)
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

            const TableState iter_state = theta_to_state(theta);
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
                                          params_.robust_loss_scale, top_split)
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

void TableModel::apply_constraints()
{
    state_.w            = std::max(state_.w, 0.1f);
    state_.h            = std::max(state_.h, 0.1f);
    state_.table_height = std::max(state_.table_height, 0.05f + TOP_THICKNESS);
    const float max_leg = state_.table_height - TOP_THICKNESS;
    state_.leg_length   = std::clamp(state_.leg_length, 0.05f, max_leg);
    // leg_inset is FROZEN (not estimated): legs sit at the OUTER edge of the top, their outer rim
    // flush with the table edge (centre at half_w − LEG_RADIUS). Estimating it let the legs migrate
    // inward and fight the top edges; pinning it removes a degenerate DOF and keeps the corner-leg
    // points tied to the true footprint.
    state_.leg_inset    = LEG_RADIUS;
}

// ─── Bounding box ────────────────────────────────────────────────────────────

std::pair<Eigen::Vector3f, Eigen::Vector3f> TableModel::bounding_box() const
{
    const float s = std::sin(state_.yaw);
    const float c = std::cos(state_.yaw);

    // Half-extents of the table top in local frame
    const float hw = state_.w * 0.5f;
    const float hh = state_.h * 0.5f;

    // Project corners to room XY
    float min_x =  std::numeric_limits<float>::max();
    float max_x = -std::numeric_limits<float>::max();
    float min_y =  std::numeric_limits<float>::max();
    float max_y = -std::numeric_limits<float>::max();

    for (float lx : {-hw, hw})
    {
        for (float ly : {-hh, hh})
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
        Eigen::Vector3f{min_x, min_y, 0.0f},
        Eigen::Vector3f{max_x, max_y, state_.table_height}
    };
}

}  // namespace rc
