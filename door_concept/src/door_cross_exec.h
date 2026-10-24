/*
 * door_cross_exec.h — P(the cross is EXECUTABLE from where the robot stands now): the cross contract's `pre:` clause.
 *
 * WHY (gold13 leg 2, 2026-10-24): the robot stood BESIDE a leaf opened ~90 deg into its own room; `cross` was on offer
 * (P(passable) 1.00 x P(at the door) 0.67) and the passage skipped the approach to it, then pressed against the leaf.
 * P(passable) is a property of the DOOR (the clear span) and P(at the door) a property of the robot's RANGE and facing;
 * neither says whether the straight run the cross will drive — robot -> aperture centre -> through-pose — is free of the
 * LEAF. That is a third factor, and it is a likelihood, not a gate:
 *
 *     P(cross executable) = P(at the door) x P(corridor clear of the leaf | phi, hinge, swing beliefs) x P(passable)
 *
 * The middle factor marginalises the leaf footprint (the segment hinge -> free edge, leaf_pose) over EVERY hypothesis the
 * agent holds: the phi posterior (phi_curve_log), the hinge side and the swing side (their posterior log-odds, mean-field
 * as phi_posterior's (hinge x swing) marginal). For each, the gap between the body's swept run (a segment, widened by the
 * body half-width) and the leaf segment is turned into a probability by the SAME Gaussian CDF the band uses
 * (prob_within), with the same sigma floor, so there is no distance threshold: a leaf grazing the corridor is 50 %.
 *
 * Pure (Eigen + std + door_geometry/door_passability): tests/test_door_cross_exec.cpp.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "door_geometry.h"
#include "door_passability.h"

namespace rc::door
{

// Shortest distance between segments [p,p+r] and [q,q+s] (2-D). Touching/crossing => 0.
[[nodiscard]] inline float segment_gap(const Eigen::Vector2f& p, const Eigen::Vector2f& p2,
                                       const Eigen::Vector2f& q, const Eigen::Vector2f& q2)
{
    const auto cross2 = [](const Eigen::Vector2f& a, const Eigen::Vector2f& b) { return a.x() * b.y() - a.y() * b.x(); };
    const auto point_seg = [](const Eigen::Vector2f& x, const Eigen::Vector2f& a, const Eigen::Vector2f& b)
    {
        const Eigen::Vector2f ab = b - a;
        const float L2 = ab.squaredNorm();
        const float t = L2 > 1e-12f ? std::clamp((x - a).dot(ab) / L2, 0.0f, 1.0f) : 0.0f;
        return (x - (a + t * ab)).norm();
    };
    const Eigen::Vector2f r = p2 - p, s = q2 - q;
    const float den = cross2(r, s);
    if (std::abs(den) > 1e-9f)
    {
        const float t = cross2(q - p, s) / den, u = cross2(q - p, r) / den;
        if (t >= 0.0f and t <= 1.0f and u >= 0.0f and u <= 1.0f) return 0.0f;
    }
    return std::min({point_seg(p, q, q2), point_seg(p2, q, q2), point_seg(q, p, p2), point_seg(q2, p, p2)});
}

struct CrossCorridorBelief
{
    Aperture ap;
    std::vector<std::pair<float, double>> phi_curve_log;   // (phi, log weight): the leaf-angle posterior
    HingeSide hinge = HingeSide::Near;   // MAP hinge, with P(MAP is right) = sigma(hinge_llr)
    float     swing = +1.0f;             // MAP swing, likewise
    float     hinge_llr = 0.0f, swing_llr = 0.0f;
};

// P(the body's run robot -> through-point is clear of the leaf), body half-width `r_body`, gap uncertainty `sigma`.
// An empty phi curve is "not measured": 1 is NOT assumed — the caller multiplies a P(passable) that is itself unknown then.
[[nodiscard]] inline float prob_corridor_clear(const CrossCorridorBelief& b, const Eigen::Vector2f& robot,
                                               const Eigen::Vector2f& through, float r_body, float sigma)
{
    double m = -std::numeric_limits<double>::infinity();
    for (const auto& [phi, lw] : b.phi_curve_log) if (std::isfinite(phi) and not std::isnan(lw)) m = std::max(m, lw);
    if (not std::isfinite(m)) return 0.0f;
    const auto sigm = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
    const float ph = sigm(b.hinge_llr), ps = sigm(b.swing_llr);
    const HingeSide other_h = b.hinge == HingeSide::Near ? HingeSide::Far : HingeSide::Near;
    const float s = std::max(sigma, 1e-3f);
    double num = 0.0, den = 0.0;
    for (const auto& [phi, lw] : b.phi_curve_log)
    {
        if (not std::isfinite(phi) or std::isnan(lw)) continue;
        const double w = std::exp(lw - m);
        float clear = 0.0f;
        for (const auto& [h, wh] : {std::pair{b.hinge, ph}, std::pair{other_h, 1.0f - ph}})
            for (const auto& [sw, wsw] : {std::pair{b.swing, ps}, std::pair{-b.swing, 1.0f - ps}})
            {
                const LeafPose L = leaf_pose(b.ap, LeafState{.phi = phi, .hinge = h, .swing = sw});
                const Eigen::Vector2f tip = L.hinge_xy + L.ex * (2.0f * L.half_w);
                const float gap = segment_gap(robot, through, L.hinge_xy, tip);
                clear += wh * wsw * prob_within(r_body, gap, s);   // P(r_body <= gap)
            }
        num += w * clear;
        den += w;
    }
    return den > 0.0 ? static_cast<float>(num / den) : 0.0f;
}

// The pre: clause value. p_at_door = P(in band/within reach) x P(facing) (InteractionState::p_within), p_passable the clear-span P.
[[nodiscard]] inline float cross_exec_prob(float p_at_door, float p_corridor_clear, float p_passable)
{
    return std::clamp(p_at_door, 0.0f, 1.0f) * std::clamp(p_corridor_clear, 0.0f, 1.0f) * std::clamp(p_passable, 0.0f, 1.0f);
}

}  // namespace rc::door
