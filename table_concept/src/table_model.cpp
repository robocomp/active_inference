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

using namespace torch::indexing;

// ─── Helpers ─────────────────────────────────────────────────────────────────

namespace
{

bool finite_state(const TableState& state)
{
    return std::isfinite(state.cx) && std::isfinite(state.cy) &&
           std::isfinite(state.w) && std::isfinite(state.h) &&
           std::isfinite(state.table_height) && std::isfinite(state.leg_length) &&
           std::isfinite(state.yaw);
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
}

torch::Tensor fe_torch_impl(const TableModelParams& params,
                             const TableState& prior,
                             const torch::Tensor& state_tensor,
                             const torch::Tensor& points_tensor,
                             const torch::Tensor& weights_tensor)
{
    const auto cx = state_tensor.index({0});
    const auto cy = state_tensor.index({1});
    const auto w = state_tensor.index({2});
    const auto h = state_tensor.index({3});
    const auto table_height = state_tensor.index({4});
    const auto leg_length = state_tensor.index({5});
    const auto yaw = state_tensor.index({6});

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

    const float leg_inset = TableModel::LEG_RADIUS + 0.02f;
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

    const auto sdf = torch::minimum(sdf_top, sdf_leg_min);

    const float inv_sigma2 = 1.0f / (params.sigma_obs * params.sigma_obs);
    const auto point_loss = robust_loss_value(sdf, params.robust_loss, params.robust_loss_scale);
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
    const auto state_energy = params.lambda_state * (dsw * dsw + dsh * dsh + dst * dst + dsl * dsl);

    auto dyaw = yaw - prior.yaw;
    dyaw = dyaw - (2.0f * M_PIf) * torch::floor((dyaw + M_PIf) / (2.0f * M_PIf));
    const auto angle_energy = params.lambda_angle * dyaw * dyaw;

    return likelihood + size_energy + pos_energy + state_energy + angle_energy;
}

} // anonymous namespace

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
    const float leg_inset    = LEG_RADIUS + 0.02f;
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

    return std::min(sdf_top, sdf_leg_min);
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
    float state_energy = params_.lambda_state * (dsw*dsw + dsh*dsh + dst*dst + dsl*dsl);

    // Angle transition prior
    float dyaw = s.yaw - prior_.yaw;
    // Wrap to [-π, π]
    dyaw -= 2.0f * M_PIf * std::floor((dyaw + M_PIf) / (2.0f * M_PIf));
    float angle_energy = params_.lambda_angle * dyaw * dyaw;

    return size_energy + pos_energy + state_energy + angle_energy;
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

    terms.prior = prior_energy(s);
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
                                 const IterationObserver& observer)
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
        {arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6]},
        options).set_requires_grad(true);

    auto theta_to_state = [&](const torch::Tensor& tensor) -> TableState
    {
        auto cpu = tensor.detach().to(torch::kCPU);
        auto acc = cpu.accessor<float, 1>();
        return TableState{acc[0], acc[1], acc[2], acc[3], acc[4], acc[5], acc[6]};
    };

    const TableState fallback_state = state_;

    auto run_loop = [&](auto& optimizer) -> bool
    {
        for (int iter = 0; iter < params_.optimization_iters; ++iter)
        {
            optimizer.zero_grad();
            auto loss = fe_torch_impl(params_, prior_, theta, points_tensor, weights_tensor);
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

    const float final_fe = fe_torch_impl(params_, prior_, theta.detach(), points_tensor, weights_tensor).item<float>();
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
