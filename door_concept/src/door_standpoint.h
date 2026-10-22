/*
 * door_standpoint.h — WHERE a door is (which wall of the room polygon holds its aperture) and WHERE the robot
 * stands to use it (approach / open / cross poses). Pure plan-view geometry: Eigen + door_geometry.h, no DSR,
 * no Qt, so it is exercised offline by tests/test_door_standpoint.cpp on a real remembered room outline.
 *
 * ─── WHY THIS EXISTS (2026-10-12, live try25/28/30: `aff_door_24_approach` OUTSIDE the room) ──────────────
 * room_1's outline is non-convex: a 10 cm partition wall runs along x≈0.05 from y≈0.39 up to the top wall, and
 * the region x>0.09, y>2.4 is NOT room_1. Webots DOOR_0 (door_24) sits in the y≈2.4 wall, aperture x≈0.1..1.1,
 * hinged at the corner (0.09,2.4). Two defects composed:
 *
 *  1. WALL ASSOCIATION BY THE MASK CENTROID. With the leaf swung ~90° away from the room, it lies along y at
 *     x≈0.12, y≈2.5..3.5 — in the LINE of the perpendicular partition wall, 10 cm behind it. The centroid's
 *     nearest polygon edge is therefore the partition (0.11 m) and not the doorway wall (0.28 m), and the
 *     belief converged as a CLOSED door in the partition: centre (−0.01,2.93), yaw 1.56, w = 1.00 — exactly
 *     the swung leaf (hinge (0.12,2.49) + 1 m along +y ⇒ centre (0.12,2.99)). No centroid rule can separate
 *     "closed leaf in wall A" from "open leaf of a door in wall B" at a corner: the leaf IS where both say.
 *     What does separate them is HOW THE LEAF WAS SEEN. A camera inside the room sees the leaf through the
 *     aperture, so every sightline camera→leaf point that leaves the room polygon leaves it THROUGH the wall
 *     that holds the door. For a closed door that is its own wall (the leaf sits in, or just behind, the
 *     face), for a swung leaf it is the doorway wall — never the partition, which the robot was on the
 *     wrong side of to see it as a closed door. associate_wall() votes per point on that exit edge.
 *
 *  2. APPROACH SIDE = SIGN OF THE ROBOT'S DISTANCE TO THE INFINITE APERTURE PLANE. In a non-convex room the
 *     robot can be inside the room and still geometrically "beyond" the plane of a door it can only reach
 *     from the other side (here: robot in the lower-right region, at x>0, beyond the partition's plane ⇒
 *     standpoint (1.84,2.78), in the part of the plan that is not room_1). Which side to stand on is not a
 *     half-plane question, it is a REGION question: stand on the side of the aperture that is in the same
 *     region as the robot — inside the room polygon iff the robot is. approach_side() asks exactly that, and
 *     falls back to the half-plane sign only when the polygon cannot tell the two sides apart.
 *
 * The collinear merge used to compare |cos|: a polygon that doubles back on itself (a spike or a slit, as
 * room_1 has at (3.69,2.36)→(4.14,2.36)→(0.09,2.4)) had its two anti-parallel edges merged into ONE "wall",
 * whose direction, near corner and inward normal were then those of whichever edge the walk started from.
 * Same-direction only now: an anti-parallel neighbour is the other face of a sliver, not more of this wall.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <span>
#include <vector>

#include <Eigen/Dense>

#include "door_geometry.h"

namespace rc::door
{

// ─── polygon primitives ──────────────────────────────────────────────────────────────────────────

// Even-odd interior test. A sliver (a doubled-back slit) has zero area and so contains nothing — which is
// the right answer: nobody can stand in a 0-width notch.
inline bool polygon_contains(const Eigen::Vector2f& q, std::span<const Eigen::Vector2f> poly)
{
    const std::size_t n = poly.size();
    if (n < 3)
        return false;
    bool inside = false;
    for (std::size_t i = 0, j = n - 1; i < n; j = i++)
    {
        const Eigen::Vector2f& a = poly[i];
        const Eigen::Vector2f& b = poly[j];
        if (((a.y() > q.y()) != (b.y() > q.y()))
            and (q.x() < (b.x() - a.x()) * (q.y() - a.y()) / (b.y() - a.y()) + a.x()))
            inside = not inside;
    }
    return inside;
}

inline double polygon_area2(std::span<const Eigen::Vector2f> poly)   // twice the signed area; >0 ⇔ CCW
{
    double a2 = 0.0;
    const std::size_t n = poly.size();
    for (std::size_t i = 0; i < n; ++i)
    {
        const Eigen::Vector2f& p0 = poly[i];
        const Eigen::Vector2f& p1 = poly[(i + 1) % n];
        a2 += static_cast<double>(p0.x()) * p1.y() - static_cast<double>(p1.x()) * p0.y();
    }
    return a2;
}

// Index of the edge (i → i+1) nearest to q, or poly.size() when there is none.
inline std::size_t nearest_edge(const Eigen::Vector2f& q, std::span<const Eigen::Vector2f> poly)
{
    const std::size_t n = poly.size();
    float best = std::numeric_limits<float>::max();
    std::size_t best_i = n;
    for (std::size_t i = 0; i < n; ++i)
    {
        const Eigen::Vector2f& a = poly[i];
        const Eigen::Vector2f ab = poly[(i + 1) % n] - a;
        const float len2 = ab.squaredNorm();
        if (len2 < 1e-9f) continue;
        const float t = std::clamp((q - a).dot(ab) / len2, 0.0f, 1.0f);
        if (const float d2 = (q - (a + t * ab)).squaredNorm(); d2 < best) { best = d2; best_i = i; }
    }
    return best_i;
}

// The FIRST polygon edge the segment cam→target crosses (smallest fraction along it), if any.
inline std::optional<std::size_t> first_crossed_edge(const Eigen::Vector2f& cam, const Eigen::Vector2f& target,
                                                     std::span<const Eigen::Vector2f> poly)
{
    const auto cross = [](const Eigen::Vector2f& p, const Eigen::Vector2f& q) { return p.x() * q.y() - p.y() * q.x(); };
    const Eigen::Vector2f d = target - cam;
    const std::size_t n = poly.size();
    float best_t = std::numeric_limits<float>::max();
    std::optional<std::size_t> best;
    for (std::size_t i = 0; i < n; ++i)
    {
        const Eigen::Vector2f a = poly[i];
        const Eigen::Vector2f ab = poly[(i + 1) % n] - a;
        const float rxs = cross(d, ab);
        if (std::abs(rxs) < 1e-12f) continue;              // parallel: never a crossing
        const float t = cross(a - cam, ab) / rxs;          // fraction along cam→target
        const float u = cross(a - cam, d)  / rxs;          // fraction along the edge
        if (t > 0.0f and t <= 1.0f and u >= 0.0f and u <= 1.0f and t < best_t) { best_t = t; best = i; }
    }
    return best;
}

// ─── the logical wall an edge belongs to ─────────────────────────────────────────────────────────

// A polygon has chamfered corners / notches that split one physical wall into several short edges, so an edge
// is COLLINEAR-MERGED into its logical wall: walk both ways while neighbours keep the SAME direction (within
// ~3°). O = the run's start corner, u = along-wall unit, len = run length. ★The normal n = (−u.y, u.x) points
// INTO the room whatever the polygon's winding (a CW polygon has its run traversed backwards) — LeafState::swing
// and Openable.SwingDir are signs against it, and approach_side() relies on it.
struct WallRun
{
    bool ok = false;
    Eigen::Vector2f O{0.0f, 0.0f}, u{1.0f, 0.0f};
    float len = 0.0f;
    std::size_t lo = 0, hi = 0;   // first and last polygon edge of the run
};

// ★ A DOORWAY NOTCH DOES NOT END THE WALL (2026-10-18). A learnt room outline follows the scan into the doorway, so
//   one physical wall reads  main → step in → the opening's floor line → step out → main. The collinear walk stopped
//   at the first step, the door was anchored to ONE piece, and apply_constraints (its span must fit [0, wall_len])
//   pinned it there: live, room_2's door_5 sat at s = 0 for 700 cycles, half a door width off the real gap, while
//   the ZED segmented it squarely. With recess_m > 0 the walk also crosses a detour of up to two edges when the
//   next same-direction edge lies within recess_m of the run's line — a recess no deeper than the wall itself is
//   the doorway in that wall, not another wall. recess_m = 0 is the old behaviour exactly.
inline WallRun wall_run_of_edge(std::span<const Eigen::Vector2f> poly, std::size_t edge, float recess_m = 0.0f)
{
    WallRun w;
    const std::size_t n = poly.size();
    if (n < 3 or edge >= n)
        return w;
    constexpr float kColinCos = 0.99863f;   // cos(3°) — what "the same wall" means for a chamfer split
    const auto edge_dir = [&](std::size_t i) -> Eigen::Vector2f
    {
        const Eigen::Vector2f d = poly[(i + 1) % n] - poly[i];
        const float l = d.norm();
        return l > 1e-6f ? Eigen::Vector2f(d / l) : Eigen::Vector2f::Zero();
    };
    const Eigen::Vector2f d0 = edge_dir(edge);
    const Eigen::Vector2f P0 = poly[edge];
    const Eigen::Vector2f n0(-d0.y(), d0.x());
    const auto on_line = [&](std::size_t i) { return std::abs(n0.dot(poly[i % n] - P0)) <= recess_m; };
    const auto same_wall_edge = [&](std::size_t j)
    { return edge_dir(j).dot(d0) >= kColinCos and on_line(j) and on_line(j + 1); };
    std::size_t lo = edge, hi = edge, used = 1;
    bool detoured = false;
    while (used < n)   // backward while the SAME direction (no |·|: see header)
    {
        const std::size_t prev = (lo + n - 1) % n;
        if (prev == hi) break;
        if (edge_dir(prev).dot(d0) >= kColinCos) { lo = prev; ++used; continue; }
        bool crossed = false;
        if (recess_m > 0.0f)
            for (std::size_t k = 1; k <= 2 and used + k < n; ++k)
            {
                const std::size_t j = (lo + n - 1 - k) % n;
                if (j == hi) break;
                if (same_wall_edge(j)) { lo = j; used += k + 1; crossed = detoured = true; break; }
            }
        if (not crossed) break;
    }
    while (used < n)   // forward
    {
        const std::size_t nxt = (hi + 1) % n;
        if (nxt == lo) break;
        if (edge_dir(nxt).dot(d0) >= kColinCos) { hi = nxt; ++used; continue; }
        bool crossed = false;
        if (recess_m > 0.0f)
            for (std::size_t k = 1; k <= 2 and used + k < n; ++k)
            {
                const std::size_t j = (hi + 1 + k) % n;
                if (j == lo) break;
                if (same_wall_edge(j)) { hi = j; used += k + 1; crossed = detoured = true; break; }
            }
        if (not crossed) break;
    }
    const Eigen::Vector2f a   = poly[lo];
    const Eigen::Vector2f far = poly[(hi + 1) % n];
    if (detoured)
    {
        // The run's ends may sit on the recessed line; the wall is the SEED edge's line, so project onto it.
        const Eigen::Vector2f O = P0 + (a - P0).dot(d0) * d0;
        const float len = (far - O).dot(d0);
        if (len < 1e-6f)
            return w;
        w.O = O; w.u = d0; w.len = len; w.lo = lo; w.hi = hi; w.ok = true;
    }
    else
    {
        const Eigen::Vector2f ab  = far - a;
        const float len = ab.norm();
        if (len < 1e-6f)
            return w;
        w.O = a; w.u = ab / len; w.len = len; w.lo = lo; w.hi = hi; w.ok = true;
    }
    if (polygon_area2(poly) < 0.0) { w.O = w.O + w.len * w.u; w.u = -w.u; }   // CW: traverse backwards ⇒ n inward
    return w;
}

// ─── which wall holds the door ───────────────────────────────────────────────────────────────────

// Each observed door point votes for one logical wall: the wall the sightline camera→point LEAVES the room
// through (the light from that point entered the room through the door's aperture), or — when the point is
// in front of every wall (no crossing) — the wall nearest to it. The wall with the most votes holds the door.
// No camera, or a camera that is not inside this room (wrong room frame / standing in the doorway), cannot
// say which edge a sightline leaves through, so then every point votes for its nearest wall. With a single
// point and no camera this is exactly the old nearest-edge rule.
inline WallRun associate_wall(std::span<const Eigen::Vector2f> poly, std::span<const Eigen::Vector2f> pts,
                              const std::optional<Eigen::Vector2f>& cam, float recess_m = 0.0f)
{
    const std::size_t n = poly.size();
    if (n < 3 or pts.empty())
        return {};
    const bool use_sight = cam.has_value() and polygon_contains(*cam, poly);
    std::vector<std::size_t> run_of(n, n);   // edge → its run's first edge, computed lazily
    std::vector<double> votes(n, 0.0);
    for (const auto& p : pts)
    {
        std::optional<std::size_t> e;
        if (use_sight)
            e = first_crossed_edge(*cam, p, poly);
        if (not e.has_value())
            if (const std::size_t ne = nearest_edge(p, poly); ne < n) e = ne;
        if (not e.has_value()) continue;
        if (run_of[*e] == n)
            if (const WallRun r = wall_run_of_edge(poly, *e, recess_m); r.ok) run_of[*e] = r.lo;
        if (run_of[*e] < n) votes[run_of[*e]] += 1.0;
    }
    const auto it = std::ranges::max_element(votes);
    if (*it <= 0.0)
        return {};
    return wall_run_of_edge(poly, static_cast<std::size_t>(std::distance(votes.begin(), it)), recess_m);
}

// ★Task D-C review M-1: A REMEMBERED DOOR (no mask points: restored from memory, or carried across) is associated with the
// wall its remembered POSE lies in — the place AND the yaw vote. The cost of edge e is how far the aperture would have to
// move to lie in it, in metres of the door's own geometry: the place's distance to the edge, and the jamb displacement a
// yaw mismatch δ implies, (w/2)·|sin δ| (the same measure the RT dead-band uses). The nearest-edge rule alone put DOOR_0 on
// the PARTITION meeting its wall 7 cm from the jamb whenever the remembered place sat a few cm off its own wall line.
inline WallRun associate_wall_remembered(std::span<const Eigen::Vector2f> poly, const Eigen::Vector2f& place, float yaw,
                                         float half_w, float recess_m = 0.0f)
{
    const std::size_t n = poly.size();
    if (n < 3 or not place.allFinite() or not std::isfinite(yaw))
        return {};
    std::size_t best = n;
    double best_cost = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; ++i)
    {
        const Eigen::Vector2f a = poly[i], b = poly[(i + 1) % n], ab = b - a;
        const float len2 = ab.squaredNorm();
        if (len2 < 1e-10f) continue;
        const float t = std::clamp((place - a).dot(ab) / len2, 0.0f, 1.0f);
        const double dist = (place - (a + t * ab)).norm();
        const double dyaw = std::sin(static_cast<double>(yaw) - std::atan2(ab.y(), ab.x()));
        const double cost = dist * dist + std::pow(static_cast<double>(half_w) * dyaw, 2.0);
        if (cost < best_cost) { best_cost = cost; best = i; }
    }
    return best < n ? wall_run_of_edge(poly, best, recess_m) : WallRun{};
}

// ─── the poses ───────────────────────────────────────────────────────────────────────────────────

struct Pose2
{
    float x = 0.0f, y = 0.0f, yaw = 0.0f;
};

// The stand-off pose in front of the aperture on side `side` (its sign; 0 counts as +1), facing through it.
inline Pose2 front_pose(const Aperture& a, float side, float standoff_m)
{
    const float sgn = (side < 0.0f) ? -1.0f : +1.0f;
    const Eigen::Vector2f n = a.across_u();
    const Eigen::Vector2f c = a.centre_xy();
    const Eigen::Vector2f p = c + sgn * standoff_m * n;
    // Face the aperture: the heading is from the stand-off toward the centre, i.e. -sgn·n.
    return {p.x(), p.y(), std::atan2(-sgn * n.y(), -sgn * n.x())};
}

// The target on the FAR side of `side`, and the heading that carries the robot through.
inline Pose2 through_pose(const Aperture& a, float side, float standoff_m)
{
    const float sgn = (side < 0.0f) ? -1.0f : +1.0f;
    const Eigen::Vector2f n = a.across_u();
    const Eigen::Vector2f c = a.centre_xy();
    const Eigen::Vector2f p = c - sgn * standoff_m * n;
    return {p.x(), p.y(), std::atan2(-sgn * n.y(), -sgn * n.x())};
}

// Which side of the aperture (+1 along across_u, −1 against it) the robot should stand on to use the door.
// The side whose stand-off point is in the SAME REGION as the robot: inside the room polygon iff the robot
// is. That is a statement about connectivity, which is what "the side I am on" means in a non-convex room;
// the signed distance to the aperture's infinite plane (`robot_side`) answers it only for a convex one and is
// kept as the answer when the polygon cannot tell the two sides apart (both or neither in the robot's region,
// e.g. no polygon, or a door in a partition with the same room on both sides).
inline float approach_side(const Aperture& a, const Eigen::Vector2f& robot, float robot_side, float standoff_m,
                           std::span<const Eigen::Vector2f> poly)
{
    const float half_plane = robot_side < 0.0f ? -1.0f : +1.0f;
    if (poly.size() < 3)
        return half_plane;
    const bool robot_in = polygon_contains(robot, poly);
    const Eigen::Vector2f n = a.across_u();
    const Eigen::Vector2f c = a.centre_xy();
    const bool plus_ok  = polygon_contains(c + standoff_m * n, poly) == robot_in;
    const bool minus_ok = polygon_contains(c - standoff_m * n, poly) == robot_in;
    if (plus_ok != minus_ok)
        return plus_ok ? +1.0f : -1.0f;
    return half_plane;
}

// Which side of the aperture is THIS ROOM's interior (+1 along across_u, −1 against it): the side whose stand-off
// point lies inside the room polygon. A crossing LEAVES the current room, so this — not where the robot happens
// to stand — is the side a crossing starts from. ★Not approach_side: that one follows the robot, and a robot
// already 0.2 m past the plane (a crossing that stalled in the doorway, try43) re-latched to the far side and was
// sent BACK into the room it was leaving. `fallback` (the robot's side) when the polygon cannot tell the two apart.
inline float interior_side(const Aperture& a, float standoff_m, std::span<const Eigen::Vector2f> poly, float fallback)
{
    if (poly.size() < 3)
        return fallback;
    const Eigen::Vector2f n = a.across_u();
    const Eigen::Vector2f c = a.centre_xy();
    const bool plus_in  = polygon_contains(c + standoff_m * n, poly);
    const bool minus_in = polygon_contains(c - standoff_m * n, poly);
    if (plus_in != minus_in)
        return plus_in ? +1.0f : -1.0f;
    return fallback;
}

}  // namespace rc::door
