/*
 * door_bearing_range.h — turn a peripheral BEARING into a RANGE, so a 360 detection can become
 * "GO THERE and CHECK" instead of only "glance that way".
 *
 * THE GAP THIS FILLS. voxelizer already publishes ricoh detections into the `masks` node. A slice with
 * lidar depth (mask_has_depth = 1) carries a real range + range_var and flows down the normal fit path —
 * nothing to do. A slice WITHOUT depth carries only `azimuth_room_rad`, and door_concept's Part C-birth
 * places its hypothesis at a CONSTANT `Bearing.NominalRangeM` along that ray. A constant range is not a
 * belief about where the thing is, so the hypothesis can only ever author an Orient (rotate-in-place)
 * affordance: there is no trustworthy (x,y) to navigate to.
 *
 * THE OBSERVATION. residual_concept already publishes, on the `residual` node, exactly the field needed:
 * `grid_occupancy_prob` is P(occupied ∧ ¬explained) — every cell an already-modelled object accounts for
 * is soft-collapsed toward zero (residual_occupancy_grid.cpp: "SOFT collapse: attenuate by the explained
 * prob"). So marching the bearing ray through that field and taking the first surviving cell answers
 * precisely the right question: "how far along this bearing is the first thing no known object explains?"
 * table/cabinet/refrigerator already consume this same attribute trio, so this adds a consumer, not a
 * coupling.
 *
 * ★σ IS DERIVED, NOT PICKED. The returned sigma is built from the two error sources that actually exist:
 *   · cell quantisation — the march resolves range to the grid pitch, so σ ≥ cell/2;
 *   · bearing error at range — an azimuth error ε displaces the ray by d·ε across the scene, and at long
 *     range that is what makes the march hit a DIFFERENT object than the one detected. So σ grows as
 *     d·sigma_az. This is why a far candidate is correctly less trusted than a near one without any
 *     distance gate: the NBV marginalises over this σ and simply stands further back.
 *   · plus half the depth of the occupied run, since a wall or a leaf has extent along the ray.
 * Nothing here is a tuned threshold; `p_occupied` is the same "confidently unexplained" level the birth
 * probe already uses, and max_range_m is a search bound, not a model choice.
 *
 * FAILS CLOSED. No field / nothing found along the ray ⇒ nullopt, and the caller keeps the previous
 * constant-range behaviour. A hypothesis without a range stays a glance, which is the honest answer.
 *
 * ★THE DURABLE PART IS `BearingRange`, NOT THIS PROVIDER. Range for a peripheral detection is a
 * PRECISION-ORDERED CASCADE, and the grid march is only today's middle rung:
 *
 *     1. dense ricoh depth        (YOLO-Depth + the RGB360 fit — landing soon)   tightest σ
 *     2. lidar-fused ricoh depth  (mask_has_depth = 1, carries range_var)        tight σ
 *     3. residual-grid ray march  (this file)                                    wide σ
 *     4. Bearing.NominalRangeM    (a constant, i.e. no belief at all)            widest σ
 *
 * Every rung answers with the same pair (range, σ) and the consumer never asks which rung it came from —
 * it just propagates σ into the along-ray Σ, and the NBV's marginalisation does the rest. So when dense
 * ricoh depth arrives it is inserted as rung 1 and NOTHING below changes: bearing-only slices become rare,
 * this provider quietly stops being reached, and no call site needs editing. Do not let a caller branch on
 * the source; branch on σ, which is what actually differs.
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <numbers>
#include <optional>
#include <vector>

#include <Eigen/Dense>

namespace rc::door
{

// One snapshot of residual_concept's published residual field. Row-major, i = y*width + x.
// Mirrors table_concept's rc::GridField; kept local while this is a prototype rather than promoted to
// common/, so nothing else has to change to try it.
struct ResidualField
{
    std::vector<float> prob;                     // P(occupied ∧ ¬explained)
    float xmin = 0.f, ymin = 0.f, cell = 0.f;    // room-frame origin of cell (0,0), cell size (m)
    int   width = 0, height = 0;

    bool valid() const
    { return width > 0 and height > 0 and cell > 0.f and static_cast<int>(prob.size()) >= width * height; }

    // Nearest-cell lookup; returns 0 outside the grid so a ray that leaves the room simply finds nothing.
    float at_world(float wx, float wy) const
    {
        if (not valid()) return 0.f;
        const int x = static_cast<int>((wx - xmin) / cell);
        const int y = static_cast<int>((wy - ymin) / cell);
        if (x < 0 or y < 0 or x >= width or y >= height) return 0.f;
        return prob[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)];
    }
};

struct BearingRangeParams
{
    float p_occupied   = 0.5f;    // "confidently unexplained-occupied" — same level as the birth probe
    float max_range_m  = 8.0f;    // how far along the ray to look (search bound, not a model choice)
    float sigma_az_rad = 0.05f;   // 360 bearing accuracy; drives the range-dependent σ term
    float min_range_m  = 0.4f;    // ignore returns inside the robot's own footprint
};

struct BearingRange
{
    float range_m = 0.f;          // distance along the ray to the first unexplained occupied run
    float sigma_m = 0.f;          // derived 1σ (see the header note) — feeds the along-ray Σ
    int   run_cells = 0;          // depth of the occupied run in cells (diagnostic)
};

// March `az` from `from` through the residual field; return the first confidently-unexplained run.
// Pure geometry + array indexing: no DSR, no Qt, so it is directly unit-testable.
inline std::optional<BearingRange> range_along_bearing(const ResidualField& f,
                                                       const Eigen::Vector2f& from, float az,
                                                       const BearingRangeParams& p = {})
{
    if (not f.valid()) return std::nullopt;

    const Eigen::Vector2f dir(std::cos(az), std::sin(az));
    const float step = 0.5f * f.cell;                 // half-cell: never step over a one-cell obstacle
    const int   n    = static_cast<int>(p.max_range_m / std::max(1e-3f, step));

    for (int i = 0; i < n; ++i)
    {
        const float d = static_cast<float>(i) * step;
        if (d < p.min_range_m) continue;
        const Eigen::Vector2f q = from + dir * d;
        if (f.at_world(q.x(), q.y()) < p.p_occupied) continue;

        // First hit. Walk the run out so its DEPTH can widen σ — a wall met obliquely is a long run and
        // genuinely tells us less about where along it the detection was.
        int run = 0;
        for (int j = i; j < n; ++j)
        {
            const Eigen::Vector2f r = from + dir * (static_cast<float>(j) * step);
            if (f.at_world(r.x(), r.y()) < p.p_occupied) break;
            ++run;
        }
        const float run_m = static_cast<float>(run) * step;

        BearingRange out;
        out.range_m   = d;
        out.run_cells = run;
        // Quadrature of the three independent contributions (see the header note).
        const float s_quant  = 0.5f * f.cell;
        const float s_bear   = d * p.sigma_az_rad;
        const float s_extent = 0.5f * run_m;
        out.sigma_m = std::sqrt(s_quant * s_quant + s_bear * s_bear + s_extent * s_extent);
        return out;
    }
    return std::nullopt;   // nothing unexplained along this ray ⇒ caller keeps the glance
}

// ─── rung 2.5: PARALLAX. Two bearings from two places are a range ────────────────────────────────────────
//
// A bearing-only detection is missing exactly one DOF: range. A MOVING observer recovers it for free —
// two bearings taken from different points triangulate, with a baseline that grows the further the robot
// travels. This is why "keep going and check later" is not merely the cheaper option but the BETTER
// estimate: stopping to look is the one action that freezes the baseline at zero, buying a nicer image of
// the DOF that was never the problem.
//
// σ HAS THE RIGHT LIMITS BY CONSTRUCTION. For two rays meeting at parallax angle γ, a bearing error ε
// displaces the intersection along the ray by ≈ d·ε/sin γ. So σ → ∞ as γ → 0 (no baseline ⇒ no range,
// which is the honest answer, not a failure) and shrinks as the robot travels further off the original
// ray. No gate on "enough baseline" is needed: a poor geometry simply reports a large σ and the NBV's
// marginalisation stands the robot further back until it has earned a better one.
struct BearingFix
{
    Eigen::Vector2f xy{0.f, 0.f};   // triangulated position, room frame
    float sigma_m      = 0.f;       // 1σ along the ray (the dominant error; across-ray is ~d·σ_az)
    float parallax_rad = 0.f;       // angle between the two rays — the geometry that earned the fix
};

// Intersect two bearing rays. `sigma_az` is the per-observation bearing 1σ; the two combine in quadrature.
// Returns nullopt when the rays are parallel (no baseline) or intersect BEHIND either observer, which is
// the degenerate case that would otherwise place a confident object in the wrong half-plane.
inline std::optional<BearingFix> triangulate_bearings(const Eigen::Vector2f& from_a, float az_a,
                                                      const Eigen::Vector2f& from_b, float az_b,
                                                      float sigma_az = 0.05f)
{
    const Eigen::Vector2f da(std::cos(az_a), std::sin(az_a));
    const Eigen::Vector2f db(std::cos(az_b), std::sin(az_b));
    const auto cross = [](const Eigen::Vector2f& u, const Eigen::Vector2f& v)
    { return u.x() * v.y() - u.y() * v.x(); };

    const float den = cross(da, db);
    if (std::abs(den) < 1e-4f) return std::nullopt;      // parallel ⇒ no baseline ⇒ no range

    const Eigen::Vector2f D = from_b - from_a;
    const float ta = cross(D, db) / den;                 // distance along ray A to the intersection
    const float tb = cross(D, da) / den;                 // …and along ray B
    if (ta <= 0.f or tb <= 0.f) return std::nullopt;     // intersection behind an observer

    BearingFix fix;
    fix.xy = from_a + da * ta;
    // Parallax = the angle between the rays, folded to [0, π/2]: what matters is how far from PARALLEL the
    // geometry is, and rays meeting head-on (γ→π) are just as informative as perpendicular ones.
    const float g = std::asin(std::clamp(std::abs(den), 0.f, 1.f));
    fix.parallax_rad = g;
    const float s_az = sigma_az * std::sqrt(2.0f);       // two independent bearings
    fix.sigma_m = ta * s_az / std::max(1e-3f, std::sin(std::max(g, 1e-3f)));
    return fix;
}

// ─── association: is this new bearing the SAME object as one we already hold? ─────────────────────────────
//
// THE DEDUP AND THE FUSION ARE THE SAME TEST. A proto that was never checked stays in the graph, so the
// next sighting of it must associate rather than birth a twin. Testing the new ray against the proto's
// PLACED POINT is wrong for exactly the reason the range cascade exists: a bearing-only proto's placed
// range is a guess, so the same door seen from a new angle lands metres away and a point test births a
// duplicate. The invariant that actually holds across observations is the BEARING, so the gate lives in
// bearing space: does the new ray pass within the proto's positional uncertainty as seen from HERE?
//
// A point at distance d with positional σ_p subtends σ_p/d of bearing, so the two error sources add in
// quadrature and the gate widens automatically for a distant or poorly-known proto — no separate
// near/far rule. `k` is a Mahalanobis width (2 ≈ 95%), not a metric threshold.
inline bool bearing_consistent(const Eigen::Vector2f& from, float az,
                               const Eigen::Vector2f& proto_xy, float proto_sigma_m,
                               float sigma_az = 0.05f, float k = 2.0f)
{
    const Eigen::Vector2f r = proto_xy - from;
    const float d = r.norm();
    if (d < 1e-3f) return true;                          // standing on it: any bearing is consistent
    float resid = std::atan2(r.y(), r.x()) - az;
    while (resid >  std::numbers::pi_v<float>) resid -= 2.0f * std::numbers::pi_v<float>;
    while (resid < -std::numbers::pi_v<float>) resid += 2.0f * std::numbers::pi_v<float>;
    const float s_pos = proto_sigma_m / d;               // positional σ expressed as bearing
    const float gate  = k * std::sqrt(sigma_az * sigma_az + s_pos * s_pos);
    return std::abs(resid) <= gate;
}

}   // namespace rc::door
