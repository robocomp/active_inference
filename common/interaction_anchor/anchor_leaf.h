/*
 * anchor_leaf.h — Task 12 fix A: the door's SWUNG LEAF, carried on the interaction anchor in ANCHOR coordinates, so the
 * controller's servo can treat it as an obstacle the model already knows (the LiDAR self-filter hides a leaf within
 * ~0.4 m of the body; try70: every failed crossing was on the side the 90° leaf swings toward).
 *
 * ─── THE ENCODING (existing attributes only, no cortex change) ─────────────────────────────────────────────────────────
 * Anchor frame (interaction_anchor.h): origin = the opening's centre on the wall's mid-plane, x along the wall, +y the
 * wall normal, the robot on −y when the anchor was created (so −y = "the robot's side", +y = "away from it").
 * The leaf is a slab turning about a vertical hinge. Its CENTRE LINE is the segment from the hinge H to the free edge T;
 * its half-thickness is kLeafHalfThickM (the leaf model's own, the one door_concept's anchor update fits with).
 *   delimiting_polygon_x / delimiting_polygon_y  (anchor frame, metres) — K LEAF HYPOTHESES of 4 vertices each:
 *       v0 = H                        the hinge, on the jamb the leaf hangs from (x = ±w_clear/2), on the face it swings past
 *       v1 = T(φ)                     the free edge at the leaf's MAP angle φ
 *       v2 = T(φ + σ_φ)               the free edge one σ further open, about the same hinge (σ_φ = 0 ⇒ v2 = v1)
 *       v3 = (p, 0)                   NOT a point: the hypothesis's PROBABILITY p (fix round 1, I-4), in x
 *     The hypotheses are the hinge × swing configurations door_concept's belief holds (p = P(hinge)·P(swing), from its
 *     hinge and swing log-odds), most probable first, Σp = 1. A 3-vertex list (the first Task 12 encoding) is one leaf, p 1.
 *   door_phi_rad                       φ itself (≥ 0; 0 = the leaf closes the opening), as the door node publishes it.
 * Each hypothesis carries everything the controller needs and no convention it must know: WHICH jamb is the hinge (the
 * sign of v0.x), WHICH side the leaf swings to (the sign of v1.y; it reads both the robot's side and the far side), how far
 * open it is, its length |v1 − v0|, the angular σ (the angle between v1 − v0 and v2 − v0) and how likely it is. Absence = no
 * leaf known: the attributes are REMOVED, never set to a sentinel (a polygon at the origin would be a legal leaf).
 * Pure Eigen + std; the DSR read/write lives in interaction_anchor.{h,cpp}.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <utility>
#include <optional>
#include <vector>

#include <Eigen/Dense>

namespace rc::interaction
{

// The leaf's half-thickness (m): the leaf MODEL's dimension (door_concept door_anchor_update.h kLeafHalfThick IS this value),
// a physical size shared by the writer and the reader — not a decision level. A double, so the anchor update's own constant
// is bit-identical to what it was (the chained replay of try70 leg 5 is sensitive to a 5e-10 m change: see the Task 12 report).
inline constexpr double kLeafHalfThickM = 0.02;

struct AnchorLeaf
{
    Eigen::Vector2f hinge{0.f, 0.f};   // H, anchor frame
    Eigen::Vector2f tip{0.f, 0.f};     // T(φ), anchor frame
    float phi = 0.f;                   // the MAP opening angle (rad), as published
    float sigma_phi = 0.f;             // its σ (rad)
    float half_thick = float(kLeafHalfThickM);
    float open_sense = 1.f;            // the rotation sense (CCW +1) that opens it further; the writer's (not encoded)
    float p = 1.f;                     // the hypothesis's probability (fix round 1, I-4)
    [[nodiscard]] float length() const { return (tip - hinge).norm(); }
};

// The leaf from the door's belief, in anchor coordinates — door_anchor_update.h leaf_hit's slab: a hinge pin on the jamb's
// SWING-SIDE face (hinge_u, face_y), the shut direction across the opening along x by dir_u (= −sign(hinge_u)), the leaf
// swinging toward swing_y (±1) by φ ≥ 0, unit direction D = (dir_u cos φ, swing_y sin φ). The slab lies on the side
// n_off = dir_u·swing_y·(D.y, −D.x) of its hinge line (toward the opening when swung 90°, into the wall's thickness when shut),
// so its CENTRE LINE starts half a thickness off the pin.
[[nodiscard]] inline AnchorLeaf make_anchor_leaf(float hinge_u, float face_y, float dir_u, float swing_y, float len, float phi,
                                                 float sigma_phi)
{
    AnchorLeaf l;
    const Eigen::Vector2f D(dir_u * std::cos(phi), swing_y * std::sin(phi));
    const Eigen::Vector2f n_off = (dir_u * swing_y) * Eigen::Vector2f(D.y(), -D.x());
    l.hinge = Eigen::Vector2f(hinge_u, face_y) + l.half_thick * n_off;
    l.tip = l.hinge + len * D;
    l.phi = phi;
    l.sigma_phi = std::max(0.f, sigma_phi);
    l.open_sense = dir_u * swing_y >= 0.f ? 1.f : -1.f;   // cross((dir_u, 0), (0, swing_y))
    return l;
}

// The vertex lists the anchor node carries: 4 per hypothesis (see the header).
[[nodiscard]] inline std::pair<std::vector<float>, std::vector<float>> encode_anchor_leaves(const std::vector<AnchorLeaf>& leaves)
{
    std::vector<float> xs, ys;
    xs.reserve(4 * leaves.size()); ys.reserve(4 * leaves.size());
    for (const auto& l : leaves)
    {
        const Eigen::Vector2f d = l.tip - l.hinge;
        // One σ further open: the leaf's own opening sense, σ_φ more.
        const float a = l.open_sense * l.sigma_phi, c = std::cos(a), s = std::sin(a);
        const Eigen::Vector2f t2 = l.hinge + Eigen::Vector2f(c * d.x() - s * d.y(), s * d.x() + c * d.y());
        xs.insert(xs.end(), {l.hinge.x(), l.tip.x(), t2.x(), std::clamp(l.p, 0.f, 1.f)});
        ys.insert(ys.end(), {l.hinge.y(), l.tip.y(), t2.y(), 0.f});
    }
    return {std::move(xs), std::move(ys)};
}

// The reader's half: the hypotheses, renormalised to Σp = 1 (empty = no leaf / malformed: wrong size, non-finite, a
// zero-length leaf, no probability mass).
[[nodiscard]] inline std::vector<AnchorLeaf> decode_anchor_leaves(const std::vector<float>& xs, const std::vector<float>& ys,
                                                                  float phi)
{
    std::vector<AnchorLeaf> out;
    if (xs.size() != ys.size()) return out;
    const bool legacy = xs.size() == 3;
    if (not legacy and (xs.empty() or xs.size() % 4 != 0)) return out;
    for (std::size_t i = 0; i < xs.size(); ++i)
        if (not std::isfinite(xs[i]) or not std::isfinite(ys[i])) return {};
    const std::size_t stride = legacy ? 3 : 4;
    float mass = 0.f;
    for (std::size_t b = 0; b + 2 < xs.size(); b += stride)
    {
        AnchorLeaf l;
        l.hinge = {xs[b], ys[b]};
        l.tip = {xs[b + 1], ys[b + 1]};
        const Eigen::Vector2f d1 = l.tip - l.hinge, d2 = Eigen::Vector2f(xs[b + 2], ys[b + 2]) - l.hinge;
        if (not (d1.norm() > 1e-4f)) return {};
        l.sigma_phi = std::abs(std::atan2(d1.x() * d2.y() - d1.y() * d2.x(), d1.dot(d2)));
        l.phi = std::isfinite(phi) ? phi : 0.f;
        l.p = legacy ? 1.f : std::clamp(xs[b + 3], 0.f, 1.f);
        mass += l.p;
        out.push_back(l);
    }
    if (not (mass > 0.f)) return {};
    for (auto& l : out) l.p /= mass;
    return out;
}

// The hinge × swing hypotheses from a belief: the MAP leaf (hinge_u, dir_u, swing_y) with P(MAP hinge) = σ(hinge_llr) and
// P(MAP swing) = σ(swing_llr) (log-odds of the MAP over the other, ≥ 0 after the MAP flip), and the three others — the
// other jamb (hinge_u, dir_u negated) and/or the other side (swing_y and face_y negated). Most probable first.
[[nodiscard]] inline std::vector<AnchorLeaf> anchor_leaf_hypotheses(float hinge_u, float face_y, float dir_u, float swing_y,
                                                                    float len, float phi, float sigma_phi,
                                                                    float hinge_llr, float swing_llr)
{
    const auto sig = [](float l) { return 1.f / (1.f + std::exp(-l)); };
    const float ph = sig(hinge_llr), ps = sig(swing_llr);
    std::vector<AnchorLeaf> out;
    for (const float hs : {1.f, -1.f})
        for (const float ss : {1.f, -1.f})
        {
            AnchorLeaf l = make_anchor_leaf(hs * hinge_u, ss * face_y, hs * dir_u, ss * swing_y, len, phi, sigma_phi);
            l.p = (hs > 0.f ? ph : 1.f - ph) * (ss > 0.f ? ps : 1.f - ps);
            out.push_back(l);
        }
    std::ranges::stable_sort(out, [](const AnchorLeaf& a, const AnchorLeaf& b) { return a.p > b.p; });
    return out;
}

}  // namespace rc::interaction
