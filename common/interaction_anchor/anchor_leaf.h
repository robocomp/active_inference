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
 *   delimiting_polygon_x / delimiting_polygon_y  (3 vertices, anchor frame, metres):
 *       v0 = H                        the hinge, on the jamb the leaf hangs from (x = ±w_clear/2), on the face it swings past
 *       v1 = T(φ)                     the free edge at the leaf's MAP angle φ
 *       v2 = T(φ + σ_φ)               the free edge one σ further open, about the same hinge (σ_φ = 0 ⇒ v2 = v1)
 *   door_phi_rad                       φ itself (≥ 0; 0 = the leaf closes the opening), as the door node publishes it.
 * The polygon carries everything the controller needs and no convention it must know: WHICH jamb is the hinge (the sign
 * of v0.x), WHICH side the leaf swings to (the sign of v1.y; it reads both the robot's side and the far side), how far
 * open it is, its length |v1 − v0| and the angular σ (the angle between v1 − v0 and v2 − v0). Absence = no leaf known:
 * the attributes are REMOVED, never set to a sentinel (a 3-vertex polygon at the origin would be a legal leaf).
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

// The three vertices (x and y lists) the anchor node carries.
[[nodiscard]] inline std::pair<std::vector<float>, std::vector<float>> encode_anchor_leaf(const AnchorLeaf& l)
{
    const Eigen::Vector2f d = l.tip - l.hinge;
    // One σ further open: the leaf's own opening sense, σ_φ more.
    const float a = l.open_sense * l.sigma_phi, c = std::cos(a), s = std::sin(a);
    const Eigen::Vector2f t2 = l.hinge + Eigen::Vector2f(c * d.x() - s * d.y(), s * d.x() + c * d.y());
    return {{l.hinge.x(), l.tip.x(), t2.x()}, {l.hinge.y(), l.tip.y(), t2.y()}};
}

// The reader's half. nullopt = not a leaf (wrong size, non-finite, a zero-length leaf).
[[nodiscard]] inline std::optional<AnchorLeaf> decode_anchor_leaf(const std::vector<float>& xs, const std::vector<float>& ys,
                                                                  float phi)
{
    if (xs.size() != 3 or ys.size() != 3) return std::nullopt;
    for (int i = 0; i < 3; ++i)
        if (not std::isfinite(xs[i]) or not std::isfinite(ys[i])) return std::nullopt;
    AnchorLeaf l;
    l.hinge = {xs[0], ys[0]};
    l.tip = {xs[1], ys[1]};
    const Eigen::Vector2f d1 = l.tip - l.hinge, d2 = Eigen::Vector2f(xs[2], ys[2]) - l.hinge;
    if (not (d1.norm() > 1e-4f)) return std::nullopt;
    l.sigma_phi = std::abs(std::atan2(d1.x() * d2.y() - d1.y() * d2.x(), d1.dot(d2)));
    l.phi = std::isfinite(phi) ? phi : 0.f;
    return l;
}

}  // namespace rc::interaction
