/*
 * door_room_band.h — WHERE A DOOR SITS RELATIVE TO THE ROOM POLYGON: the wall band.
 *
 * room_concept publishes the room as a polygon drawn at the ROOM-SIDE FACE of its walls. A door is a hole
 * IN a wall, so its aperture does not lie on that polygon and does not lie inside it: it lies in the BAND
 * between the room-side face (signed offset δ = 0) and the far face (δ = T, the wall thickness), wherever
 * the door node happens to be registered (leaf line, wall centre line, far face). room_concept measured a
 * 0.19-0.20 m recess; the Webots walls are 0.11 m. Pure geometry + Eigen: no DSR, unit-testable
 * (door_concept/tests/test_door_frame_evidence.cpp, section (f)).
 *
 * Two consumers, ONE model, so they cannot disagree about where the wall is:
 *
 *  (1) THE CONTAINMENT PRIOR (specificworker.cpp, update_existence_beliefs). It used to be
 *      `point_in_room(aperture centre, 0.40 m) ? 0 : −1.5 per frame`, which is wrong twice: the normal
 *      position of a real door (δ ∈ [0, T]) is OUTSIDE the polygon, so the test was only ever saved by its
 *      margin; and a prior integrated as −1.5 PER FRAME is not a prior but a fake sensor, firing at 10 Hz on
 *      a fact that did not change. Now it is a static log-prior term
 *            log π(δ) = −½ (dist(δ, [0, T]) / σ)²
 *      — exactly 0 anywhere in the wall band, a Gaussian falloff in the pose + polygon uncertainty σ outside
 *      it, on BOTH sides (a door well inside the room is as wrong as one far outside) — that REPLACES its
 *      previous value each cycle instead of accumulating. A door far off the band is pushed to the floor
 *      once and stays there (removal then needs the usual debounce, advanced by 1 − π); a door on the band
 *      pays nothing, ever.
 *
 *  (2) THE LEAF BEYOND THE APERTURE (door_fitter.cpp, compute_silhouette_existence). A leaf swung OUT of
 *      the robot's room is seen, if at all, THROUGH the opening: past the room-side face, behind the jambs
 *      and the casing, edge-on. The silhouette model's P(detect) is a model of a leaf in the room the camera
 *      is in or flush in its own aperture plane; it has no term for the opening's edges as occluders, so a
 *      leaf sample past the room-side face is a view it does not represent and weighs 1 − P(through).
 *      Measured from the APERTURE LINE: the room-side face lies an unknown r ≥ 0 past it (r ~ N(r0, σ²)
 *      truncated to r ≥ 0, r0 = max(−δ_ap, 0) — the aperture's own depth inside the polygon, 0 for one in
 *      the wall), and a sample at across-offset a is through the opening with probability P(r < a). A
 *      sample ON the aperture line (a shut leaf, the hinge edge) is never "through" — exactly 0, not ½ —
 *      so a shut door, and a leaf swung INTO the room, are untouched.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <Eigen/Dense>

namespace rc::door
{

struct RoomBand
{
    float wall_thickness_m = 0.20f;   // T: room-side face → far face (Existence.WallThicknessM)
    float sigma_m          = 0.10f;   // σ: door registration ⊕ room-polygon face uncertainty (Existence.BoundarySigmaM)
};

// Signed distance from q to the polygon boundary: > 0 outside, < 0 inside (even-odd rule). NaN if the polygon
// has fewer than 3 vertices (no room ⇒ no band ⇒ callers impose nothing). `nearest` receives the closest
// boundary point when non-null.
[[nodiscard]] inline float signed_boundary_distance(const std::vector<Eigen::Vector2f>& poly, const Eigen::Vector2f& q,
                                                    Eigen::Vector2f* nearest = nullptr)
{
    const std::size_t n = poly.size();
    if (n < 3)
        return std::numeric_limits<float>::quiet_NaN();
    bool inside = false;
    float best2 = std::numeric_limits<float>::max();
    Eigen::Vector2f best_p = poly[0];
    for (std::size_t i = 0, j = n - 1; i < n; j = i++)
    {
        const Eigen::Vector2f& a = poly[i];
        const Eigen::Vector2f& b = poly[j];
        if (((a.y() > q.y()) != (b.y() > q.y()))
            and (q.x() < (b.x() - a.x()) * (q.y() - a.y()) / (b.y() - a.y() + 1e-12f) + a.x()))
            inside = not inside;
        const Eigen::Vector2f ab = b - a;
        const float len2 = ab.squaredNorm();
        const float t = (len2 > 1e-12f) ? std::clamp((q - a).dot(ab) / len2, 0.0f, 1.0f) : 0.0f;
        const Eigen::Vector2f p = a + t * ab;
        if (const float d2 = (q - p).squaredNorm(); d2 < best2) { best2 = d2; best_p = p; }
    }
    if (nearest != nullptr)
        *nearest = best_p;
    const float d = std::sqrt(best2);
    return inside ? -d : d;
}

// Distance from δ to the wall band [0, T] (0 inside the band).
[[nodiscard]] inline float distance_to_band(float delta, const RoomBand& b)
{
    return std::max({0.0f, -delta, delta - std::max(0.0f, b.wall_thickness_m)});
}

// log π(δ): the containment log-prior of an aperture at signed offset δ. 0 on the band, −½(d/σ)² off it.
// NaN δ (no polygon) ⇒ 0: no room, no prior.
[[nodiscard]] inline float band_log_prior(float delta, const RoomBand& b)
{
    if (not std::isfinite(delta))
        return 0.0f;
    const float u = distance_to_band(delta, b) / std::max(b.sigma_m, 1e-3f);
    return -0.5f * u * u;
}
[[nodiscard]] inline float p_on_band(float delta, const RoomBand& b) { return std::exp(band_log_prior(delta, b)); }

// Unit normal of the aperture's wall pointing OUT of the room: whichever of ±across the signed distance
// increases along. The probe step is the wall thickness — the scale over which "out" is defined.
[[nodiscard]] inline Eigen::Vector2f outward_normal(const std::vector<Eigen::Vector2f>& poly, const Eigen::Vector2f& ap_xy,
                                                    const Eigen::Vector2f& across, const RoomBand& b)
{
    const float step = std::max(0.05f, b.wall_thickness_m);
    const float dp = signed_boundary_distance(poly, ap_xy + step * across);
    const float dm = signed_boundary_distance(poly, ap_xy - step * across);
    return (std::isfinite(dp) and std::isfinite(dm) and dm > dp) ? Eigen::Vector2f(-across) : across;
}

// Φ(x).
[[nodiscard]] inline float std_normal_cdf(float x) { return 0.5f * std::erfc(-x * 0.70710678f); }

// P(a point at across-wall offset `a` past the APERTURE LINE (toward the outside) lies past the room-side
// face, i.e. is seen THROUGH the opening), with that face r ~ N(r0, σ²) past the line, truncated to r ≥ 0.
// a ≤ 0 (on the line or on the room side) ⇒ exactly 0.
[[nodiscard]] inline float p_beyond_far_face(float a, float r0, float sigma)
{
    if (not (a > 0.0f) or not std::isfinite(a))
        return 0.0f;
    const float s  = std::max(sigma, 1e-3f);
    const float lo = std_normal_cdf(-r0 / s);                    // mass the truncation removes (r < 0)
    const float num = std_normal_cdf((a - r0) / s) - lo;
    return std::clamp(num / std::max(1.0f - lo, 1e-6f), 0.0f, 1.0f);
}

// r0 for an aperture at signed offset δ_ap: how far past the aperture line the ROOM-SIDE FACE lies — the
// aperture's depth inside the polygon, 0 for an aperture in (or beyond) the wall.
[[nodiscard]] inline float face_offset_from_aperture(float delta_ap)
{
    return std::isfinite(delta_ap) ? std::max(0.0f, -delta_ap) : 0.0f;
}

}   // namespace rc::door
