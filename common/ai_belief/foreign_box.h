/*
 * common/ai_belief/foreign_box.h — a neighbour object's published belief as a known surface of the foreign density
 * (spec 2026-11-23 shared-engine G3, "Answers to the builder check" 5; the 11-22 candidate 2 in
 * docs/superpowers/results/2026-11-22-fixA-live14.md). SHARED, header-only, pure.
 *
 * A YOLO-seg mask can take in a neighbour (a nightstand mask that swallowed the bed: live14fixA, a 2.46 m cloud for a
 * 0.51 m nightstand). The foreign density of the camera points and of the LiDAR rays knew the room's walls and floor
 * (beam_room.h) but not the other objects, so those points and returns had nowhere to go but this object. A neighbour
 * enters as a FIXED foreign density for one update, built from its previous-cycle published belief (pose, size, std):
 *
 *   - camera point p, measurement variance R:  g(p) = √(R / (R + σ_d²)) · exp(−½ d² / (R + σ_d²)),
 *       d = the neighbour box's signed distance at p, σ_d² = Σ_j (∂d/∂θ_j · std_j)² its published std pushed through the
 *       SDF (diagonal: the publishers carry no cross terms). It is the normalised N(d; 0, R + σ_d²) on the same scale the
 *       model's own components use (their common 1/√(2πR) dropped), so the membership compares like with like;
 *   - LiDAR ray o + t·u:  the neighbour is one more surface the ray can cross first (beam_room.h), its σ along the ray the
 *       face's σ_d over |n̂·u|, as a wall's.
 *
 * The box is upright, standing on z0 (the fleet's RT convention: origin at the base, top = z0 + H), its local frame
 * R(−yaw)(p − c). θ order of `std` = (cx, cy, H, w, h, yaw), w along the local x, h along the local y.
 *
 * A neighbour whose std is missing, zero or not finite is NOT a belief (unseeded, or a publisher that never wrote one); it
 * is skipped (`usable`), never used as a zero-width surface (plan Review Focus 5).
 */
#pragma once

#include <cmath>
#include <limits>

#include <Eigen/Core>

namespace rc::ai::beam
{

struct ForeignBox
{
    Eigen::Vector2f c = Eigen::Vector2f::Zero();     // footprint centre, room frame
    float H = 0.0f, w = 0.0f, h = 0.0f, yaw = 0.0f;  // height, extent along local x, along local y; yaw about +z
    Eigen::Matrix<float, 6, 1> std = Eigen::Matrix<float, 6, 1>::Zero();   // published std of (cx, cy, H, w, h, yaw)
    float z0 = 0.0f;                                 // base height, room frame (0 for a floor-standing object)
};

// Whether a neighbour is a belief the foreign density may use: every std component finite and ≥ 0, not all zero, and a
// box with positive extents (Review Focus 5).
inline bool usable(const ForeignBox& b)
{
    if (not (b.H > 0.0f and b.w > 0.0f and b.h > 0.0f)) return false;
    if (not (std::isfinite(b.c.x()) and std::isfinite(b.c.y()) and std::isfinite(b.yaw) and std::isfinite(b.z0))) return false;
    bool any = false;
    for (int j = 0; j < 6; ++j)
    {
        if (not (std::isfinite(b.std(j)) and b.std(j) >= 0.0f)) return false;
        any = any or b.std(j) > 0.0f;
    }
    return any;
}

// The neighbour box's signed distance at p and its variance under the published std (σ_d² above).
struct ForeignSdf { double d = 0.0, var = 0.0; };
inline ForeignSdf foreign_sdf(const ForeignBox& b, const Eigen::Vector3d& p)
{
    const double cy = std::cos(double(b.yaw)), sy = std::sin(double(b.yaw));
    const double dx = p.x() - double(b.c.x()), dy = p.y() - double(b.c.y());
    const Eigen::Vector3d pl(cy * dx + sy * dy, -sy * dx + cy * dy, p.z() - double(b.z0) - 0.5 * double(b.H));
    const Eigen::Vector3d half(0.5 * double(b.w), 0.5 * double(b.h), 0.5 * double(b.H));
    const Eigen::Vector3d q = pl.cwiseAbs() - half;
    const Eigen::Vector3d qp = q.cwiseMax(0.0);
    const double n = qp.norm();
    ForeignSdf out;
    out.d = n + std::min(0.0, q.maxCoeff());
    // ga = ∂d/∂|p_local| (the closest-feature direction outside, the face normal inside or on the surface)
    Eigen::Vector3d ga = Eigen::Vector3d::Zero();
    if (n > 0.0) ga = qp / n;
    else { int k = 0; q.maxCoeff(&k); ga(k) = 1.0; }
    const Eigen::Vector3d g(pl.x() < 0.0 ? -ga.x() : ga.x(), pl.y() < 0.0 ? -ga.y() : ga.y(),
                            pl.z() < 0.0 ? -ga.z() : ga.z());        // ∂d/∂p_local
    Eigen::Matrix<double, 6, 1> J;
    J(0) = -(cy * g.x() - sy * g.y());                               // ∂d/∂cx
    J(1) = -(sy * g.x() + cy * g.y());                               // ∂d/∂cy
    J(2) = 0.5 * (-g.z() - ga.z());                                  // the centre z = z0 + H/2 and half_z = H/2
    J(3) = -0.5 * ga.x();                                            // ∂d/∂w
    J(4) = -0.5 * ga.y();                                            // ∂d/∂h
    J(5) = g.x() * pl.y() - g.y() * pl.x();                          // ∂p_local/∂yaw = (p_ly, −p_lx)
    out.var = J.cwiseProduct(b.std.cast<double>()).squaredNorm();
    return out;
}

// g(p) for one camera point of measurement variance R (see the header): the neighbour's surface density on the model's scale.
inline double foreign_point_density(const ForeignBox& b, const Eigen::Vector3d& p, double R)
{
    const ForeignSdf s = foreign_sdf(b, p);
    const double v = R + s.var;
    return std::sqrt(R / v) * std::exp(-0.5 * s.d * s.d / v);
}

// The first entry of o + t·u, t ∈ (0, ρ_max], into the neighbour box (t = +inf: none, or the sensor inside it), and the hit
// face's σ along the ray, σ_d / |n̂·u|.
struct BoxCrossing { double t = std::numeric_limits<double>::infinity(), sigma_r = 0.0; };
inline BoxCrossing first_box_crossing(const ForeignBox& b, const Eigen::Vector3d& o, const Eigen::Vector3d& u, double rho_max)
{
    const double cy = std::cos(double(b.yaw)), sy = std::sin(double(b.yaw));
    const double dx = o.x() - double(b.c.x()), dy = o.y() - double(b.c.y());
    const Eigen::Vector3d ol(cy * dx + sy * dy, -sy * dx + cy * dy, o.z() - double(b.z0) - 0.5 * double(b.H));
    const Eigen::Vector3d ul(cy * u.x() + sy * u.y(), -sy * u.x() + cy * u.y(), u.z());
    const Eigen::Vector3d half(0.5 * double(b.w), 0.5 * double(b.h), 0.5 * double(b.H));
    double a = 0.0, e = rho_max;
    int face = -1;
    for (int i = 0; i < 3; ++i)
    {
        if (ul(i) == 0.0)
        {
            if (std::abs(ol(i)) > half(i)) return {};                // parallel to this slab and outside it
            continue;
        }
        double t1 = (-half(i) - ol(i)) / ul(i), t2 = (half(i) - ol(i)) / ul(i);
        if (t1 > t2) std::swap(t1, t2);
        if (t1 > a) { a = t1; face = i; }
        e = std::min(e, t2);
        if (a > e) return {};
    }
    if (face < 0 or not (a > 0.0)) return {};                         // the sensor inside the box: no entry
    BoxCrossing c;
    c.t = a;
    const double nu = std::abs(ul(face));                             // |n̂·u|, n̂ the hit face's normal
    c.sigma_r = std::sqrt(foreign_sdf(b, o + a * u).var) / nu;
    return c;
}

}  // namespace rc::ai::beam
