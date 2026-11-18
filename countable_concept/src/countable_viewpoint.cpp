#include "countable_viewpoint.h"

#include <cmath>
#include <algorithm>
#include <limits>

#include "place_memory/place_map.h"

namespace rc {

void accumulate_path(ViewpointPath& p, const Eigen::Vector2f& sensor_xy_room,
                     const Eigen::Vector2f& step_xy_room, const Eigen::Vector2f& object_xy)
{
    const Eigen::Vector2f r = sensor_xy_room - object_xy;
    const float z = r.norm();
    // Degenerate geometry (the sensor sits at the object centre: r̂ undefined), not a tuning threshold.
    if (z < 1e-3f)
        return;
    const Eigen::Vector2f rhat = r / z;
    const Eigen::Vector2f that(-rhat.y(), rhat.x());
    p.dphi_rad += std::abs(step_xy_room.dot(that)) / z;
    p.dz_m     += std::abs(step_xy_room.dot(rhat));
}

Eigen::Vector2f odometry_step_room(float v_adv, float v_side, float yaw_room, float dt_s)
{
    const float c = std::cos(yaw_room), s = std::sin(yaw_room);
    const Eigen::Vector2f robot(v_side * dt_s, v_adv * dt_s);   // x right, y forward
    return {c * robot.x() - s * robot.y(), s * robot.x() + c * robot.y()};
}

float rho_camera(const ViewpointPath& p, float z_m, float w_perp_m)
{
    const float half_angle = std::atan(w_perp_m / (2.f * z_m));
    return std::exp(-std::abs(p.dphi_rad) / half_angle - std::abs(p.dz_m) / z_m);
}

float rho_lidar_lateral(const ViewpointPath& p, float g)
{
    if (g <= 0.f) return 1.f;
    return std::exp(-std::abs(p.dphi_rad) / g);
}

float rho_lidar_height(const ViewpointPath& p, float z_m, float eps_rad, float g)
{
    if (g <= 0.f) return 1.f;
    return std::exp(-std::abs(p.dz_m) * std::abs(std::tan(eps_rad)) / (g * z_m));
}

namespace {
float wrap_pi(float a)
{
    a = std::fmod(a + static_cast<float>(M_PI), 2.f * static_cast<float>(M_PI));
    if (a <= 0.f) a += 2.f * static_cast<float>(M_PI);
    return a - static_cast<float>(M_PI);
}
}  // namespace

Eigen::Vector3f odometry_step_room_se2(float v_adv, float v_side, float omega, float yaw_room, float dt_s)
{
    const Eigen::Vector2f d = odometry_step_room(v_adv, v_side, yaw_room, dt_s);
    return {d.x(), d.y(), omega * dt_s};
}

void chain_observe(ChainTracker& t, const Eigen::Vector3f& pose, std::optional<Eigen::Vector3f> odo_step)
{
    if (not t.have)
    {
        t.ref = t.prev = pose;
        t.odo.setZero();
        t.have = true;
        return;
    }
    if (odo_step.has_value()) t.odo += *odo_step;
    else                      t.ref += pose - t.prev;   // unknown odometry: not a correction (fail closed)
    t.prev = pose;
}

Eigen::Vector3f chain_correction(const ChainTracker& t)
{
    if (not t.have) return Eigen::Vector3f::Zero();
    Eigen::Vector3f c = (t.prev - t.ref) - t.odo;
    c.z() = wrap_pi(c.z());
    return c;
}

void chain_rebase(ChainTracker& t)
{
    t.ref = t.prev;
    t.odo.setZero();
}

float rho_chain(const Eigen::Vector3f& c, const Eigen::Matrix3f& S)
{
    // Double arithmetic, symmetrised Σ, eigen-decomposition. Eigenvalues ≤ n·ε_double·λ_max (negative ones included: the
    // PSD projection) are the numerical null space — the standard numerical-rank rule, not a model threshold. Σ⁻¹ acts
    // on the range only; a component of c in the null space cannot be explained by Σ at all ⇒ ρ = 0.
    const Eigen::Vector3d cd = c.cast<double>();
    Eigen::Matrix3d Sd = S.cast<double>();
    Sd = 0.5 * (Sd + Sd.transpose());
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(Sd);
    const Eigen::Vector3d lam = es.eigenvalues();
    const double tol = 3.0 * std::numeric_limits<double>::epsilon() * std::max(lam.maxCoeff(), 0.0);
    double maha = 0.0, null_comp2 = 0.0;
    for (int i = 0; i < 3; ++i)
    {
        const double p = es.eigenvectors().col(i).dot(cd);
        if (lam(i) > tol) maha += p * p / lam(i);
        else              null_comp2 += p * p;
    }
    if (std::sqrt(null_comp2) > 3.0 * std::numeric_limits<double>::epsilon() * cd.norm()) return 0.f;
    return static_cast<float>(std::exp(-0.5 * maha));
}

Eigen::Matrix3f pose_cov_in_room(std::span<const float> rt_cov36, const Eigen::Vector3f& edge_pose)
{
    if (rt_cov36.size() < 36) return Eigen::Matrix3f::Zero();
    constexpr int idx[3] = {0, 1, 5};
    Eigen::Matrix3d S;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            S(i, j) = rt_cov36[static_cast<std::size_t>(idx[i] * 6 + idx[j])];
    const Eigen::Matrix3d J = place::se2_inverse_jacobian(edge_pose).cast<double>();
    Eigen::Matrix3d R = J * S * J.transpose();   // double: float J Σ Jᵀ would put ~1e-7 noise on a null direction
    R = 0.5 * (R + R.transpose());
    return R.cast<float>();
}

Eigen::Matrix3d chain_transport(const Eigen::Vector2f& dr)
{
    Eigen::Matrix3d F = Eigen::Matrix3d::Identity();
    F(0, 2) = -dr.y();
    F(1, 2) =  dr.x();
    return F;
}

void lever_start(ChainLever& L, const Eigen::Vector2f& seed_c, const Eigen::Vector2f& robot_r)
{
    L.l = seed_c - robot_r;
    L.set = true;
}

void lever_step(ChainLever& L, const Eigen::Vector2f& dr, float rho, const Eigen::Vector2f& c, const Eigen::Vector2f& r)
{
    L.l = rho * (L.l - dr) + (1.f - rho) * (c - r);
}

Eigen::Matrix<double, 6, 3> chain_B(const Eigen::Vector2f& l)
{
    Eigen::Matrix<double, 6, 3> B = Eigen::Matrix<double, 6, 3>::Zero();
    B(0, 0) = 1.0; B(0, 2) = -l.y();
    B(1, 1) = 1.0; B(1, 2) =  l.x();
    B(5, 2) = 1.0;
    return B;
}

}  // namespace rc
