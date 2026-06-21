/*
 * bottle_model.cpp
 *
 * Single vertical-cylinder SDF + free-energy fit for a bottle instance.
 * Geometry differs from TableModel (one cylinder, no compound min, no yaw); the
 * FE / autograd / robust-loss machinery is the same so the perception loop is
 * shared bar this file.
 */

#include "bottle_model.h"
#include "../../common/robust_metrics/robust_metrics_torch.h"
#include <algorithm>
#include <cmath>
#include <print>
#include <torch/torch.h>

using namespace torch::indexing;

// ─── Helpers ─────────────────────────────────────────────────────────────────

namespace
{

    bool finite_state(const BottleState& s)
    {
        return std::isfinite(s.cx) && std::isfinite(s.cy) && std::isfinite(s.cz) &&
            std::isfinite(s.radius) && std::isfinite(s.height);
    }

    /** Finite cylinder SDF (radius r, half-height hh), point already centred. */
    float cylinder_sdf(float dx, float dy, float dz, float r, float hh)
    {
        const float d_radial   = std::sqrt(dx*dx + dy*dy) - r;
        const float d_vertical = std::abs(dz) - hh;

        const float or_ = std::max(d_radial, 0.0f);
        const float ov  = std::max(d_vertical, 0.0f);
        const float outside = std::sqrt(or_*or_ + ov*ov);
        const float inside  = std::min(std::max(d_radial, d_vertical), 0.0f);
        return outside + inside;
    }

    // radius / half-height carried as tensors so the optimiser can size the bottle.
    torch::Tensor cylinder_sdf_tensor(const torch::Tensor& dx,
                                    const torch::Tensor& dy,
                                    const torch::Tensor& dz,
                                    const torch::Tensor& r,
                                    const torch::Tensor& hh)
    {
        // +eps under sqrt: grad of sqrt(x) is infinite at x=0 (a point on the axis),
        // which NaNs autograd and locks the optimiser. ~3e-5 m offset is negligible.
        const auto d_radial   = torch::sqrt(dx * dx + dy * dy + 1e-9f) - r;
        const auto d_vertical = torch::abs(dz) - hh;

        const auto or_ = torch::clamp_min(d_radial, 0.0f);
        const auto ov  = torch::clamp_min(d_vertical, 0.0f);
        const auto outside = torch::sqrt(or_ * or_ + ov * ov + 1e-9f);
        const auto inside  = torch::clamp_max(torch::max(d_radial, d_vertical), 0.0f);
        return outside + inside;
    }

    void apply_constraints_to_tensor(torch::Tensor& arr)
    {
        torch::NoGradGuard no_grad;
        arr.index_put_({3}, torch::clamp_min(arr.index({3}), BottleModel::MIN_RADIUS));
        arr.index_put_({4}, torch::clamp_min(arr.index({4}), BottleModel::MIN_HEIGHT));
    }

    torch::Tensor fe_torch_impl(const BottleModelParams& params,
                                const BottleState& prior,
                                const torch::Tensor& state_tensor,
                                const torch::Tensor& points_tensor,
                                const torch::Tensor& weights_tensor)
    {
        const auto cx     = state_tensor.index({0});
        const auto cy     = state_tensor.index({1});
        const auto cz     = state_tensor.index({2});
        const auto radius = state_tensor.index({3});
        const auto height = state_tensor.index({4});

        const auto px = points_tensor.index({Slice(), 0}) - cx;
        const auto py = points_tensor.index({Slice(), 1}) - cy;
        const auto pz = points_tensor.index({Slice(), 2}) - cz;

        const auto half_h = height * 0.5f;
        const auto sdf = cylinder_sdf_tensor(px, py, pz, radius, half_h);

        const float inv_sigma2 = 1.0f / (params.sigma_obs * params.sigma_obs);
        const auto point_loss = robust_loss_value(sdf, params.robust_loss, params.robust_loss_scale);
        // Normalise by point count (matches TableModel / Python prototype).
        const auto likelihood = (weights_tensor * point_loss * inv_sigma2).sum() /
                                static_cast<double>(points_tensor.size(0));

        const float sigma = params.prior_size_std > 0.0f ? params.prior_size_std : 0.03f;
        const float inv_prior_sigma2 = 1.0f / (sigma * sigma);

        const auto dr = radius - params.prior_radius;
        const auto dh = height - params.prior_height;
        const auto size_energy = params.lambda_size * (dr * dr + dh * dh) * inv_prior_sigma2;

        const auto dpx = cx - prior.cx;
        const auto dpy = cy - prior.cy;
        const auto dpz = cz - prior.cz;
        const auto pos_energy = params.lambda_pos * (dpx * dpx + dpy * dpy + dpz * dpz);

        const auto dsr = radius - prior.radius;
        const auto dsh = height - prior.height;
        const auto state_energy = params.lambda_state * (dsr * dsr + dsh * dsh);

        return likelihood + size_energy + pos_energy + state_energy;
    }

} // anonymous namespace

// ─── BottleModel ───────────────────────────────────────────────────────────────

BottleModel::BottleModel(const BottleState& prior, const BottleModelParams& params)
    : state_(prior), prior_(prior), params_(params)
{
    apply_constraints();
}

// ─── SDF ─────────────────────────────────────────────────────────────────────

float BottleModel::sdf_point_at(const Eigen::Vector3f& p, const BottleState& s) const
{
    const float dx = p.x() - s.cx;
    const float dy = p.y() - s.cy;
    const float dz = p.z() - s.cz;
    return cylinder_sdf(dx, dy, dz, s.radius, s.height * 0.5f);
}

float BottleModel::sdf_point(const Eigen::Vector3f& p) const
{
    return sdf_point_at(p, state_);
}

std::vector<float> BottleModel::compute_sdf(const std::vector<Eigen::Vector3f>& points) const
{
    std::vector<float> out;
    out.reserve(points.size());
    for (const auto& p : points)
        out.push_back(sdf_point(p));
    return out;
}

// ─── Free Energy ─────────────────────────────────────────────────────────────

float BottleModel::prior_energy(const BottleState& s) const
{
    const float sigma = params_.prior_size_std > 0.0f ? params_.prior_size_std : 0.03f;
    const float inv_sigma2 = 1.0f / (sigma * sigma);

    // Size prior: penalise deviation from prior dimensions
    const float dr = s.radius - params_.prior_radius;
    const float dh = s.height - params_.prior_height;
    const float size_energy = params_.lambda_size * (dr*dr + dh*dh) * inv_sigma2;

    // Position transition prior: penalise drift from prior centre
    const float dpx = s.cx - prior_.cx;
    const float dpy = s.cy - prior_.cy;
    const float dpz = s.cz - prior_.cz;
    const float pos_energy = params_.lambda_pos * (dpx*dpx + dpy*dpy + dpz*dpz);

    // Size state transition prior: penalise change from prior state
    const float dsr = s.radius - prior_.radius;
    const float dsh = s.height - prior_.height;
    const float state_energy = params_.lambda_state * (dsr*dsr + dsh*dsh);

    return size_energy + pos_energy + state_energy;
}

float BottleModel::silhouette_energy_at(const BottleState& s) const
{
    if (params_.mask_precision <= 0.0f or sil_dirs_.empty())
        return 0.0f;

    // Occluding-contour residual for a vertical cylinder: δ(ray) − radius, where δ is the
    // horizontal distance from the axis (cx,cy) to the back-projected edge ray. The vertical
    // axis cancels the ray's z-component, so δ depends only on (cx,cy) — independent of cz/height.
    const float Cx = sil_cam_xy_.x(), Cy = sil_cam_xy_.y();
    float acc = 0.0f;
    for (const auto& d : sil_dirs_)
    {
        const float den = std::hypot(d.x(), d.y());
        if (den < 1e-6f) continue;
        const float delta = std::abs(d.x() * (Cy - s.cy) - d.y() * (Cx - s.cx)) / den;
        acc += robust_loss_value(delta - s.radius, params_.robust_loss, params_.robust_loss_scale);
    }
    return params_.mask_precision * sil_conf_ * acc / static_cast<float>(sil_dirs_.size());
}

float BottleModel::fe_at(const BottleState& s,
                         const std::vector<Eigen::Vector3f>& pts,
                         const std::vector<float>& weights) const
{
    return fe_terms_at(s, pts, weights, 0).total_fe + silhouette_energy_at(s);
}

FreeEnergyDecomposition BottleModel::fe_terms_at(const BottleState& s,
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

    const float inv_N = 1.0f / static_cast<float>(std::max(pts.size(), std::size_t{1}));
    terms.likelihood_current *= inv_N;
    terms.likelihood_historical *= inv_N;

    terms.prior = prior_energy(s);
    terms.total_fe = terms.likelihood_current + terms.likelihood_historical + terms.prior;
    return terms;
}

float BottleModel::compute_free_energy(const std::vector<Eigen::Vector3f>& points,
                                       const std::vector<float>& weights) const
{
    if (points.empty())
        return prior_energy(state_);
    return fe_at(state_, points, weights);
}

FreeEnergyDecomposition BottleModel::compute_free_energy_decomposition(
    const std::vector<Eigen::Vector3f>& points,
    const std::vector<float>& weights,
    std::size_t historical_count) const
{
    return fe_terms_at(state_, points, weights, historical_count);
}

// ─── Gradient step ───────────────────────────────────────────────────────────

float BottleModel::gradient_step(const std::vector<Eigen::Vector3f>& points,
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
        {arr[0], arr[1], arr[2], arr[3], arr[4]},
        options).set_requires_grad(true);

    auto theta_to_state = [&](const torch::Tensor& tensor) -> BottleState
    {
        auto cpu = tensor.detach().to(torch::kCPU);
        auto acc = cpu.accessor<float, 1>();
        return BottleState{acc[0], acc[1], acc[2], acc[3], acc[4]};
    };

    const BottleState fallback_state = state_;

    // Silhouette (RGB-mask) edge rays → a differentiable tangent-distance term δ(ray;cx,cy) − radius,
    // added to the loss so the mask drives (cx,cy,radius) jointly with the depth points.
    const bool sil_on = params_.mask_precision > 0.0f and not sil_dirs_.empty();
    torch::Tensor sil_dx, sil_dy;
    if (sil_on)
    {
        sil_dx = torch::empty({static_cast<long>(sil_dirs_.size())}, options);
        sil_dy = torch::empty({static_cast<long>(sil_dirs_.size())}, options);
        auto ax = sil_dx.accessor<float, 1>();
        auto ay = sil_dy.accessor<float, 1>();
        for (std::size_t i = 0; i < sil_dirs_.size(); ++i) { ax[i] = sil_dirs_[i].x(); ay[i] = sil_dirs_[i].y(); }
    }
    const float sil_Cx = sil_cam_xy_.x(), sil_Cy = sil_cam_xy_.y();
    auto silhouette_loss = [&](const torch::Tensor& th) -> torch::Tensor
    {
        if (not sil_on) return torch::zeros({}, options);
        const auto cx = th.index({0}), cy = th.index({1}), radius = th.index({3});
        const auto den   = (sil_dx * sil_dx + sil_dy * sil_dy).sqrt();
        const auto delta = (sil_dx * (sil_Cy - cy) - sil_dy * (sil_Cx - cx)).abs() / den;
        const auto loss  = robust_loss_value(delta - radius, params_.robust_loss, params_.robust_loss_scale);
        return params_.mask_precision * sil_conf_ * loss.mean();
    };

    // Free-space / visibility term: carve a sample `freespace_margin` IN FRONT of each point (toward
    // the camera, horizontal ray — the axis is vertical) and penalise it being INSIDE the cylinder.
    // One-sided ⇒ the surface may sit at/behind the points but not bulge forward into observed free
    // space, which is exactly the depth freedom the symmetric SDF leaves open. Reuses the silhouette
    // camera (sil_cam_xy_), so it is active on the same frames the mask is.
    using torch::indexing::Slice;
    const bool fs_on = params_.lambda_freespace > 0.0f and sil_on;
    auto freespace_loss = [&](const torch::Tensor& th) -> torch::Tensor
    {
        if (not fs_on) return torch::zeros({}, options);
        const auto cx = th.index({0}), cy = th.index({1}), cz = th.index({2});
        const auto radius = th.index({3}), half_h = th.index({4}) * 0.5f;
        const auto px = points_tensor.index({Slice(), 0});
        const auto py = points_tensor.index({Slice(), 1});
        const auto pz = points_tensor.index({Slice(), 2});
        const auto rx = px - sil_Cx, ry = py - sil_Cy;                 // camera → point (horizontal)
        const auto rn = (rx * rx + ry * ry + 1e-9f).sqrt();
        const auto sx = px - params_.freespace_margin * rx / rn;       // carve sample, in front of p
        const auto sy = py - params_.freespace_margin * ry / rn;
        const auto sdf = cylinder_sdf_tensor(sx - cx, sy - cy, pz - cz, radius, half_h);
        const auto viol = torch::clamp_min(-sdf, 0.0f);               // penalise carve sample INSIDE
        return params_.lambda_freespace * (weights_tensor * viol).sum()
             / static_cast<double>(points_tensor.size(0));
    };

    auto run_loop = [&](auto& optimizer) -> bool
    {
        for (int iter = 0; iter < params_.optimization_iters; ++iter)
        {
            optimizer.zero_grad();
            auto loss = fe_torch_impl(params_, prior_, theta, points_tensor, weights_tensor)
                      + silhouette_loss(theta) + freespace_loss(theta);
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

            const BottleState iter_state = theta_to_state(theta);
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

    const float final_fe = (fe_torch_impl(params_, prior_, theta.detach(), points_tensor, weights_tensor)
                            + silhouette_loss(theta.detach())).item<float>();
    if (!std::isfinite(final_fe))
    {
        state_ = fallback_state;
        apply_constraints();
        return fe_at(state_, points, weights);
    }

    return final_fe;
}

// ─── Pose covariance (Laplace approximation) ───────────────────────────────────

Eigen::Matrix3f BottleModel::pose_covariance(const std::vector<Eigen::Vector3f>& points,
                                             const std::vector<float>& weights,
                                             float fd_step,
                                             float max_var) const
{
    const Eigen::Matrix3f fallback = Eigen::Matrix3f::Identity() * max_var;

    // F as a function of the centre (cx,cy,cz) only; radius/height held at θ*.
    const float h = fd_step;
    auto fe_centre = [&](float dcx, float dcy, float dcz) -> float
    {
        BottleState s = state_;
        s.cx += dcx; s.cy += dcy; s.cz += dcz;
        return points.empty() ? prior_energy(s) : fe_at(s, points, weights);
    };

    const float f0 = fe_centre(0.0f, 0.0f, 0.0f);
    if (!std::isfinite(f0))
        return fallback;

    // compute_free_energy normalises the likelihood by point count (mean, not
    // sum) for optimiser stability — but a covariance must accumulate precision
    // with independent evidence, so undo the 1/N here by scaling the curvature
    // by the effective sample size (Σwᵢ, or N for uniform weights). Without this
    // the covariance reflects only coverage geometry, never how MANY points we
    // have, and P_bottle would not tighten as observations pile up.
    float eff_n = static_cast<float>(points.size());
    if (not weights.empty())
    {
        eff_n = 0.0f;
        for (const float w : weights) eff_n += w;
    }
    eff_n = std::max(1.0f, eff_n);
    eff_n *= std::max(1e-4f, params_.cov_eff_scale);   // correlated-evidence discount → NEES-calibrated P

    const float inv_h2 = eff_n / (h * h);
    Eigen::Matrix3f H = Eigen::Matrix3f::Zero();

    // Diagonal: central second difference.
    const std::array<std::array<float, 3>, 3> e = {{{h,0,0},{0,h,0},{0,0,h}}};
    for (int i = 0; i < 3; ++i)
    {
        const float fp = fe_centre(e[i][0], e[i][1], e[i][2]);
        const float fm = fe_centre(-e[i][0], -e[i][1], -e[i][2]);
        H(i, i) = (fp - 2.0f * f0 + fm) * inv_h2;
    }
    // Off-diagonal: mixed central difference.
    for (int i = 0; i < 3; ++i)
        for (int j = i + 1; j < 3; ++j)
        {
            const float fpp = fe_centre(e[i][0]+e[j][0], e[i][1]+e[j][1], e[i][2]+e[j][2]);
            const float fpm = fe_centre(e[i][0]-e[j][0], e[i][1]-e[j][1], e[i][2]-e[j][2]);
            const float fmp = fe_centre(-e[i][0]+e[j][0], -e[i][1]+e[j][1], -e[i][2]+e[j][2]);
            const float fmm = fe_centre(-e[i][0]-e[j][0], -e[i][1]-e[j][1], -e[i][2]-e[j][2]);
            const float v = (fpp - fpm - fmp + fmm) * 0.25f * inv_h2;
            H(i, j) = v;
            H(j, i) = v;
        }

    if (!H.allFinite())
        return fallback;

    // Σ = H⁻¹, but only trust it if H is sufficiently positive-definite. A tiny or
    // negative eigenvalue (flat/saddle FE — i.e. under-observed) ⇒ huge variance,
    // so clamp each direction's variance to max_var.
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(H);
    if (es.info() != Eigen::Success)
        return fallback;

    const float min_prec = 1.0f / max_var;            // variance ≤ max_var
    Eigen::Vector3f eig = es.eigenvalues();
    Eigen::Vector3f inv_eig;
    for (int i = 0; i < 3; ++i)
        inv_eig(i) = 1.0f / std::max(eig(i), min_prec);

    const Eigen::Matrix3f V = es.eigenvectors();
    return V * inv_eig.asDiagonal() * V.transpose();
}

// ─── Constraints ─────────────────────────────────────────────────────────────

void BottleModel::apply_constraints()
{
    state_.radius = std::max(state_.radius, MIN_RADIUS);
    state_.height = std::max(state_.height, MIN_HEIGHT);
}

// ─── Bounding box ────────────────────────────────────────────────────────────

std::pair<Eigen::Vector3f, Eigen::Vector3f> BottleModel::bounding_box() const
{
    const float r  = state_.radius;
    const float hh = state_.height * 0.5f;
    return {
        Eigen::Vector3f{state_.cx - r, state_.cy - r, state_.cz - hh},
        Eigen::Vector3f{state_.cx + r, state_.cy + r, state_.cz + hh}
    };
}
