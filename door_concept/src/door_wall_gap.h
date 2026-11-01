#pragma once
/*
 * door_wall_gap.h — the crossing's door as A HOLE IN A WALL: find the wall, then the hole (spec 2026-11-01
 * door-wall-then-hole). A pure function of one frame's points in the ROBOT frame and the anchor's prior.
 *
 *   1. WALL (depth, yaw). The wall's direction is the prior's (the room's Manhattan wall, ~1 deg): a tight prior on the
 *      slope, so a side wall at 90 deg is not a candidate. Its offset is the MODE of the points' signed distance d across
 *      the prior's wall line, over the points near the opening's span — above the lintel (the wall spans the opening there
 *      and the leaf never hides it) and beside the opening below it. Perpendicular structure (the stub's side walls, the
 *      open leaf) spreads along d and makes no peak; the empty corridor contributes nothing. A line d = d0 + a·u through
 *      the mode's inliers, with the slope's prior, refines the yaw.
 *   2. HOLE (lateral centre, width). The points ON the wall plane (within its thickness) below the lintel, along the wall
 *      (u): the opening is the empty interval best matching the door's width near the prediction; its ends are the jambs.
 *      The open leaf meets the plane only at its hinge, i.e. at a jamb.
 * Each part reports whether it measured anything; the caller fuses what was measured with the motion-predicted prior
 * (fuse_wall_gap) and carries the rest.
 */
#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>

#include <Eigen/Dense>

#include "../../common/interaction_anchor/relative_pose.h"

namespace rc::door::wallgap
{
struct Params
{
    float width = 0.9f;          // the opening (m)
    float thickness = 0.1f;      // the wall (m)
    float lintel = 2.1f;         // the opening's top (m, robot-frame z)
    float floor_z = 0.10f;       // below: a floor return
    float range_sigma = 0.03f;   // a return's range noise (m)
    float span_margin = 0.6f;    // |u| <= w/2 + this: the points that can be this wall near the opening (m)
    float sigma_slope = 0.03f;   // the wall direction's prior (rad): the room's Manhattan wall
};
struct Result
{
    bool wall = false, gap = false;
    float d_face = 0.f;          // the near face's offset across the PRIOR's line (m, + = toward the far side)
    float slope = 0.f;           // d = d_face + slope·u: the yaw correction (rad, small)
    float sigma_d = 0.f, sigma_slope = 0.f;
    int n_wall = 0;
    float u_left = 0.f, u_right = 0.f;   // the hole's ends along the prior's wall line (m)
    int n_plane = 0;
};

/// p_robot -> (u, d) in the prior's anchor frame (u along the wall, d across it, the robot on d < 0).
inline Eigen::Vector2f to_anchor(const rc::interaction::RelPose& prior, const Eigen::Vector3f& p)
{
    const float c = std::cos(prior.x.z()), s = std::sin(prior.x.z());
    const Eigen::Vector2f q = p.head<2>() - prior.x.head<2>();
    return {c * q.x() + s * q.y(), -s * q.x() + c * q.y()};
}

inline Result measure(const rc::interaction::RelPose& prior, const std::vector<Eigen::Vector3f>& pts, const Params& pp)
{
    Result r;
    // the prior's depth spread: where the wall can be
    const float c = std::cos(prior.x.z()), s = std::sin(prior.x.z());
    const Eigen::Vector2f n(-s, c);
    const float sd_prior = std::sqrt(std::max(0.f, n.dot(prior.P.topLeftCorner<2, 2>() * n)));
    const float win = 3.f * std::hypot(sd_prior, pp.range_sigma) + 0.5f * pp.thickness;
    const float half = 0.5f * pp.width;
    // ── 1. the wall ──
    std::vector<Eigen::Vector2f> cand;   // (u, d)
    for (const auto& p : pts)
    {
        if (p.z() < pp.floor_z) continue;
        const Eigen::Vector2f a = to_anchor(prior, p);
        if (std::abs(a.x()) > half + pp.span_margin or std::abs(a.y() + 0.5f * pp.thickness) > win) continue;
        const bool above = p.z() > pp.lintel;
        const bool beside = std::abs(a.x()) > half;
        if (above or beside) cand.push_back(a);
    }
    if (cand.size() >= 8)
    {
        // the mode of d: a kernel density at the range noise, evaluated on a grid across the window
        const float h = std::max(pp.range_sigma, 0.02f);
        float best_d = 0.f, best_v = -1.f;
        for (float d = -0.5f * pp.thickness - win; d <= -0.5f * pp.thickness + win; d += 0.5f * h)
        {
            float v = 0.f;
            for (const auto& a : cand) { const float z = (a.y() - d) / h; v += std::exp(-0.5f * z * z); }
            if (v > best_v) { best_v = v; best_d = d; }
        }
        // the line through the mode's inliers, the slope under its prior (weighted least squares, 2 params)
        Eigen::Matrix2f A = Eigen::Matrix2f::Zero();
        Eigen::Vector2f b = Eigen::Vector2f::Zero();
        int n_in = 0;
        for (const auto& a : cand)
            if (std::abs(a.y() - best_d) < 2.f * h)
            {
                const Eigen::Vector2f j(1.f, a.x());
                A += j * j.transpose(); b += j * a.y(); ++n_in;
            }
        if (n_in >= 6)
        {
            const float r2 = pp.range_sigma * pp.range_sigma;
            Eigen::Matrix2f H = A / r2;
            H(1, 1) += 1.f / (pp.sigma_slope * pp.sigma_slope);
            const Eigen::Vector2f sol = H.ldlt().solve(b / r2);
            r.wall = true;
            r.d_face = sol.x();
            r.slope = sol.y();
            r.n_wall = n_in;
            // the points share a view (correlated): their count is capped like the mask's, 30
            const float neff = float(std::min(n_in, 30));
            const Eigen::Matrix2f Hc = A / r2 * (neff / float(n_in));
            Eigen::Matrix2f Hcp = Hc; Hcp(1, 1) += 1.f / (pp.sigma_slope * pp.sigma_slope);
            const Eigen::Matrix2f C = Hcp.inverse();
            r.sigma_d = std::sqrt(std::max(0.f, C(0, 0)));
            r.sigma_slope = std::sqrt(std::max(0.f, C(1, 1)));
        }
    }
    // ── 2. the hole: points ON the wall plane, below the lintel, along the wall ──
    const float d_near = r.wall ? r.d_face : -0.5f * pp.thickness;
    std::vector<float> us;
    for (const auto& p : pts)
    {
        if (p.z() < pp.floor_z or p.z() > pp.lintel - 2.f * pp.range_sigma) continue;
        const Eigen::Vector2f a = to_anchor(prior, p);
        const float d_line = d_near + (r.wall ? r.slope * a.x() : 0.f);
        if (a.y() < d_line - 2.f * pp.range_sigma or a.y() > d_line + pp.thickness + 2.f * pp.range_sigma) continue;
        if (std::abs(a.x()) > half + pp.span_margin + 0.5f) continue;
        us.push_back(a.x());
    }
    r.n_plane = int(us.size());
    if (us.size() >= 2)
    {
        std::ranges::sort(us);
        // the empty intervals; the hole = the one whose width and centre best match the door's (both in units of sigma)
        float best = std::numeric_limits<float>::infinity();
        for (std::size_t i = 0; i + 1 < us.size(); ++i)
        {
            const float l = us[i], rr = us[i + 1], w = rr - l;
            if (w < 0.3f * pp.width) continue;   // not a door-sized hole (a third of it: the gap between two returns)
            const float su = std::sqrt(std::max(1e-6f, prior.P(0, 0) + prior.P(1, 1)));
            const float cost = std::pow((w - pp.width) / 0.1f, 2.f) + std::pow((0.5f * (l + rr)) / std::max(su, 0.05f), 2.f);
            if (cost < best) { best = cost; r.u_left = l; r.u_right = rr; r.gap = true; }
        }
    }
    return r;
}

/// The prior moved by what was measured, each part fused with its own prior variance (1-D, anchor axes), and returned as
/// the new anchor in the robot frame. The opening's centre is on the wall's MID-plane (d = 0), as the anchor is.
inline rc::interaction::RelPose fuse(const rc::interaction::RelPose& prior, const Result& m, const Params& pp)
{
    rc::interaction::RelPose out = prior;
    const float c = std::cos(prior.x.z()), s = std::sin(prior.x.z());
    const Eigen::Vector2f a(c, s), n(-s, c);
    const Eigen::Matrix2f Pxy = prior.P.topLeftCorner<2, 2>();
    float var_u = std::max(1e-8f, a.dot(Pxy * a)), var_d = std::max(1e-8f, n.dot(Pxy * n)), var_y = std::max(1e-8f, prior.P(2, 2));
    float du = 0.f, dd = 0.f, dyaw = 0.f;
    if (m.wall)
    {
        const float z_d = m.d_face + 0.5f * pp.thickness, r_d = m.sigma_d * m.sigma_d + 1e-6f;   // mid-plane offset
        const float k = var_d / (var_d + r_d); dd = k * z_d; var_d *= (1.f - k);
        const float z_y = std::atan(m.slope), r_y = m.sigma_slope * m.sigma_slope + 1e-8f;
        const float ky = var_y / (var_y + r_y); dyaw = ky * z_y; var_y *= (1.f - ky);
    }
    if (m.gap)
    {
        const float z_u = 0.5f * (m.u_left + m.u_right), r_u = pp.range_sigma * pp.range_sigma;
        const float k = var_u / (var_u + r_u); du = k * z_u; var_u *= (1.f - k);
    }
    const Eigen::Vector2f shift = du * a + dd * n;
    out.x.head<2>() += shift;
    out.x.z() = rc::interaction::wrap_pi(prior.x.z() + dyaw);
    const float c2 = std::cos(out.x.z()), s2 = std::sin(out.x.z());
    Eigen::Matrix2f R; R << c2, -s2, s2, c2;
    Eigen::Matrix2f D = Eigen::Vector2f(var_u, var_d).asDiagonal();
    out.P.setZero();
    out.P.topLeftCorner<2, 2>() = R * D * R.transpose();
    out.P(2, 2) = var_y;
    return out;
}
}  // namespace rc::door::wallgap
