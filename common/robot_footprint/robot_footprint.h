/*
 * robot_footprint.h — the robot's real 2-D shape, in ONE place.
 *
 * WHY THIS EXISTS
 * Navigation in this stack was carrying SIX independent robot-size margins, owned by three different agents,
 * none of which knew the others existed:
 *
 *     residual  C-space dilation      inflate_radius_m          0.25 m
 *     residual  robot read-out mask   Clusterer.RobotRadiusM    0.45 m
 *     residual  wall margin           Clusterer.WallMarginM     0.35 m
 *     planner   obstacle expansion    robot_width_m · 0.5       0.20 m
 *     MPPI      clearance at the goal near_safe                 0.425 m
 *     planner   target repair         (its own copy of the above)
 *
 * Because they compose additively wherever two stages meet, the sum is what the robot actually obeys — and no
 * one number can be tuned without moving a boundary somebody else depends on. Observed consequences, all from
 * this arithmetic rather than from perception: the robot's own exclusion disc cancelled exactly
 * (0.45 − 0.25 − 0.20 = 0.00) so it was reported INSIDE an obstacle and no path could leave it; targets were
 * "repaired" to 0.20 m clearance that the controller then refused at 0.425 m, so it hunted at the goal
 * forever; and ~0.90 m was eaten out of every gap, closing passages the robot fits through easily.
 *
 * The cure is not better numbers. C-space inflation is an APPROXIMATION that buys a cheap point-in-polygon
 * test by pretending the robot is a disc; the price is that every stage has to guess a radius. Testing the
 * actual footprint against the world removes the guess: there is one feasibility predicate,
 *
 *     footprint(pose) ∩ occupied == ∅
 *
 * which is exact, parameter-free, and CANNOT stack because there is only one of it. Standoff beyond that stops
 * being a hard constraint and becomes a soft preference (the MPPI's distance cost), so tuning comfort can never
 * again make a goal unreachable.
 *
 * WHERE THE NUMBERS COME FROM — measured, not assumed
 * `shadow()` is the XY convex hull of the BODY ∪ the FOUR WHEELS, simplified 74 → 18 vertices (+0.3% area).
 *
 *   BODY: webots-shadow/protos/meshes/shadow.stl — 18809 triangles, binary STL, metres, z-up, x lateral,
 *   y forward (already this codebase's robot-frame convention, so coordinates transfer with no re-mapping).
 *   This file is byte-identical (md5 06153a46…) to lidar3d_dds/robots/Shadow/shadow.stl, the mesh the LiDAR
 *   driver's Embree self-filter actually loads, AND it is what Shadow.proto instantiates — so in simulation it
 *   is the very geometry that GENERATES the returns. Nothing can be more authoritative than that.
 *   Sampled over triangle SURFACES, not vertices: the base is a few very large triangles, so a vertex-only
 *   profile falsely reports empty height bands.  Body alone: max radius 0.2863, base |x|max 0.1770.
 *
 *   WHEELS: four Solids in Shadow.proto at (±0.21, −0.10, 0.015) and (±0.21, +0.16, 0.015), each a
 *   boundingObject Cylinder of radius 0.05 and height 0.072. The body STL has the wheel wells CUT OUT — its
 *   base half-width is 0.177 against the wheels' 0.246+ — so the wheels are the one piece of real geometry the
 *   STL genuinely omits, and they are the widest thing on the robot.
 *
 *     hull area              0.2343 m²
 *     inscribed radius       0.2299 m   (largest disc that fits INSIDE — the passable half-width)
 *     circumscribed radius   0.3252 m   (smallest disc that CONTAINS it)
 *     |x|max 0.2716   |y|max 0.2300
 *
 * ★ ONE UNRESOLVED ITEM, STATED RATHER THAN BURIED.
 * The wheel cylinders sit under two nested rotations (Solid `rotation 0 0 1 1.57`, then boundingObject Pose
 * `rotation 1 0 0 -1.5708`), and the resulting axis direction could not be settled by reading the proto alone.
 * The three candidates give outer |x| = 0.2460 (axis lateral), 0.2600 (forward) or 0.2600 (vertical). Rather
 * than pick one, the wheels are bounded ORIENTATION-INDEPENDENTLY: whatever the axis, a cylinder fits inside a
 * sphere of radius √(r² + (h/2)²) = 0.0616 m about its centre, giving outer |x| = 0.2716. That is at most
 * 2.6 cm pessimistic and it is a stated geometric bound, not a hidden margin — measure the axis and tighten it.
 *
 * ★ WHY THE DRIVER'S 0.55 m DISC IS NOT IN THIS POLYGON.
 * lidar3d_dds self-filters with a 0.55 m footprint disc plus a 0.75 m near-floor skirt, and its config claims
 * "the STL models only the ±0.24 m central column". That claim is measurably FALSE — ±0.24 m IS the whole
 * robot's half-width and the STL contains the full base sides. The discs were introduced by commit 4151dbc2 in
 * the SAME change that disabled the [Floor] z cut, and their z_min = 110 mm is literally the old floor-cut
 * value: they are a radius-limited reimplementation of it, tuned by chasing stray returns, never derived from
 * geometry. Live evidence (controller/proximity_obstacles.csv, 2044 frames): the nearest UNFILTERED return sits
 * at p50 0.2493 m — matching the wheels' 0.246 m outer extent to 3 mm — with z max 0.154 m robot frame, i.e.
 * 0.109 m body frame, agreeing with the discs' z_min = 110 mm to 0.6 mm. So the real self-hits are the WHEELS,
 * BELOW the discs' band, and the discs miss them entirely while deleting 0.648 m² of real world data
 * full-height (1.465 m² below 0.60 m) — 6.5× the robot's own footprint. Folding 0.55 m into this polygon would
 * therefore encode a tuning artefact as geometry. It stays out. The driver-side fix is a separate, ordered
 * migration (fix the helios 45 mm mount error first, then mesh ∪ wheels, then an occlusion raycast, then step
 * the discs to zero under a per-reason census).
 *
 * Pure Eigen/STL, DSR-free → unit-testable in isolation (self_test()).
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>
#include <Eigen/Dense>

namespace rc
{

// The robot's ground-plane silhouette in the ROBOT frame: x to the right, y forward, origin at the rotation
// centre. Convex and counter-clockwise. Convexity is not required by the collision test but it is what the
// mesh yields, and it keeps `contains()` exact and branch-free.
class RobotFootprint
{
public:
    // The Shadow robot, measured from its own mesh (see the file header for provenance and the 0.55 m caveat).
    // ★STILL COMPILED IN, AND STILL SHADOW'S. It has three jobs now: the fallback when no mesh can be read,
    // the reference every startup report is diffed against, and Shadow's ACTUAL body — because shadow.obj
    // omits the four wheels (|x|max 0.2420 against the true 0.2464; ROBOT_GEOMETRY.md), so for that one robot
    // the mesh is NOT the whole silhouette and deriving from it would be a 4.4 mm regression.
    static RobotFootprint shadow();

    // ── A BODY THAT IS NOT COMPILED IN ────────────────────────────────────────────────────────────
    // Validating constructor. Rejects rather than half-accepts: fewer than 3 vertices, zero/negative area,
    // clockwise winding, or an origin outside the polygon. `why` (optional) receives the reason.
    // ★THE POLYGON MUST BE IN THE ROBOT FRAME — x right, y FORWARD, origin at the rotation centre. A mesh in
    // some other frame is the caller's problem to rotate BEFORE calling this, and getting that wrong is
    // silent: the class cannot tell a body from the same body turned 90 degrees.
    static std::optional<RobotFootprint> from_polygon(std::vector<Eigen::Vector2f> poly,
                                                      std::string *why = nullptr);

    // Where this body came from — "compiled:shadow" or "mesh:<path> yaw+90". Carried so a log line, a mission
    // manifest or a route snapshot can never attribute a run to the wrong robot.
    const std::string &source() const { return source_; }
    void set_source(std::string s) { source_ = std::move(s); }

    // Everything a caller needs to PRINT about a mesh-derived body, in one object. Deliberately not a bool:
    // "it loaded" is not the question — the question is what shape it loaded and how that differs from the
    // body the robot was driving a moment ago.
    struct MeshReport
    {
        bool ok = false;
        std::string reason;                  // why not, when !ok
        std::string path;
        float yaw_offset_rad = 0.f;
        long  vertices = 0, vertices_rejected = 0;
        Eigen::Vector3f bb_min = Eigen::Vector3f::Zero();   // MESH frame
        Eigen::Vector3f bb_max = Eigen::Vector3f::Zero();
        std::size_t hull_raw = 0, hull_simplified = 0;
        float area_growth_frac = 0.f;
        // ROBOT frame, after the yaw offset. Margin NOT applied.
        float area_m2 = 0.f, inscribed = 0.f, circumscribed = 0.f, x_max = 0.f, y_max = 0.f;
        Eigen::Vector2f centroid = Eigen::Vector2f::Zero();
    };

    // Read the mesh, rotate it into the robot frame, hull it, and check it looks like a mobile base.
    // `yaw_offset_rad` turns MESH coordinates into ROBOT coordinates (P3Bot's mesh is native forward=+x while
    // the robot frame is forward=+y, so it needs +pi/2; Shadow's mesh is already robot-framed, so 0).
    static std::optional<RobotFootprint> from_obj(const std::string &path, float yaw_offset_rad,
                                                  MeshReport &report);

    // An explicit, uniform outward margin (m). This is the ONE place a standoff belongs: it is a stated
    // preference, not a geometric fact, and keeping it separate from the polygon means the true shape stays
    // honest and the margin stays auditable. 0 ⇒ collide against the bare metal.
    void  set_safety_margin(float m) { safety_margin_m_ = std::max(0.0f, m); }
    float safety_margin() const { return safety_margin_m_; }

    // The silhouette itself, WITHOUT the safety margin.
    const std::vector<Eigen::Vector2f>& polygon() const { return poly_; }

    // Largest disc centred on the origin that fits entirely INSIDE the footprint. This is the number a gap has
    // to beat for the robot to fit through it — 0.231 m, against the ~0.95 m the stacked margins were demanding.
    float inscribed_radius() const;
    // Smallest disc centred on the origin that CONTAINS the footprint. Useful as a cheap broad-phase reject.
    float circumscribed_radius() const;

    // Is a point (robot frame) inside the footprint, grown by the safety margin?
    bool contains(const Eigen::Vector2f& p_robot) const;

    // Same question about a WORLD point, for a body standing at `centre` facing room-yaw `yaw`. The
    // quarter turn lives here for the reason stated below: the footprint's forward axis is +y while a
    // room yaw has forward at +x, and every hand-rolled copy of that conversion is a 4.2 cm error
    // waiting to happen silently. Three such copies were found and fixed on 2026-08-16/17; this is so
    // there is no reason to write a fourth.
    bool contains_yaw(const Eigen::Vector2f& p_world, const Eigen::Vector2f& centre, float yaw) const
    {
        const float th = yaw - 1.57079633f;
        const Eigen::Vector2f d = p_world - centre;
        const float c = std::cos(th), s = std::sin(th);
        return contains({c * d.x() + s * d.y(), -s * d.x() + c * d.y()});
    }

    // The footprint placed at a world pose: translation + heading (rad), safety margin applied.
    std::vector<Eigen::Vector2f> at_pose(const Eigen::Vector2f& pos, float theta) const;

    // Body extent in a given WORLD direction, for a body at heading `theta`. For a CONVEX footprint this is
    // the support function, and it is exactly the right number to compare an ESDF reading against: the ESDF
    // gives the distance from the body ORIGIN to the nearest obstacle, and the gradient gives the direction,
    // so the body collides iff esdf < support_radius(theta, -grad). That turns a distance field query into an
    // exact footprint test for the price of a few dot products — no disc approximation, and it is direction
    // dependent, which is the whole point: a rectangle reaches further along its diagonal than across its
    // width, and a disc model has to assume the worst case in every direction.
    float support_radius(float theta, const Eigen::Vector2f& dir_world) const;

    // Same, for a body whose heading is given as a YAW in the usual robotics sense — forward =
    // (cos yaw, sin yaw), measured counter-clockwise from +x — which is what a room-frame robot pose
    // carries. This footprint's own forward axis is +y, hence the quarter turn. Frame conventions are the
    // easiest thing in this stack to get silently wrong (the hull is nearly x-symmetric, so a sign error
    // costs millimetres and never shows up as a visible fault), so every one of them lives HERE.
    float support_radius_yaw(float yaw, const Eigen::Vector2f& dir_world) const
    { return support_radius(yaw - 1.57079633f, dir_world); }

    // Grid-collision support. Returns the integer cell offsets a footprint at heading `theta` covers, relative
    // to the cell holding the robot's origin, for a grid of `cell_size` metres. Precompute ONCE per heading and
    // a collision test becomes a handful of array lookups — which is what makes exact footprint checking cheap
    // enough to run inside a planner's inner loop instead of approximating it with an inflated obstacle set.
    // Conservative: a cell is included if ANY part of it is covered, so the test never under-reports overlap.
    std::vector<Eigen::Vector2i> cell_offsets(float cell_size, float theta) const;

    // Same, for a body facing room-yaw `yaw` — the quarter turn applied once, here, rather than at each
    // caller. GridPlanner's heading buckets ARE yaws (its move table is (cos, sin) of them), and feeding
    // them to cell_offsets() raw is precisely the defect fixed on 2026-08-16: the body was rasterised
    // 90 degrees across its own direction of travel for the class's whole life, costing 4.2 cm of width
    // where a corridor closes, silently, because the hull is nearly symmetric.
    std::vector<Eigen::Vector2i> cell_offsets_yaw(float cell_size, float yaw) const
    { return cell_offsets(cell_size, yaw - 1.57079633f); }

    static bool self_test();

private:
    std::vector<Eigen::Vector2f> poly_;
    std::string source_ = "compiled:shadow";
    float safety_margin_m_ = 0.0f;
};

}  // namespace rc
