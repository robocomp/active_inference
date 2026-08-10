/*
 * common/nbv/viewpoint_score.h  —  the shared DETECTION-WEIGHTED next-best-view core (header-only, Eigen only).
 *
 * THE PROBLEM THIS FIXES, IN ONE LINE. Every concept agent scored a candidate viewpoint by the information a
 * detection there would yield, and none of them multiplied by the probability that a detection HAPPENS. So the
 * winner was the most informative view, not the most informative view we can actually get — and the robot paid
 * real travel for an expected gain that never materialised.
 *
 * The Expected-Free-Energy quantity is the expectation over the detection OUTCOME, not over the outcome we hope
 * for. Marginalising the binary outcome d ∈ {detect, miss}:
 *
 *     E[ΔH | pose] = P(detect|pose)·ΔH_detect(pose) + (1 − P(detect|pose))·ΔH_miss(pose)
 *                  ≈ P(detect|pose)·ΔH_detect(pose)
 *
 * The approximation is not laziness: a miss carries ~0 information about the fit DOFs (extent, yaw, centre),
 * which is what ΔH_detect measures. A miss is NOT worthless in general — it is exactly the evidence the removal
 * channel integrates — but that information lands on the EXISTENCE belief, on a different currency, and is
 * already weighted by this same P(detect) there. `Score` exposes p_detect so a caller that wants the full
 * mixture can add its own ΔH_miss term rather than re-deriving the geometry.
 *
 * The consequence is the point: a viewpoint the detector cannot fire from has ~0 expected gain, so it can never
 * out-bid the travel cost in the controller's G = λ_cost·nav_dist − epistemic_gain. Undetectable poses stop
 * being chosen because they are worthless, not because a rule forbids them.
 *
 * ─── WHAT LIVES HERE (and therefore stops being copy-pasted per agent) ────────────────────────────────────
 *
 *   · fill prediction on BOTH image axes, pinhole-exact, from the camera's REAL intrinsics + mount height.
 *     `roi_fill` — the quantity the detector envelope is a function of, and the one the agents measure — is
 *     max(Δcol/W, Δrow/H). A horizontal-only model is blind to the axis that actually binds for a low, wide
 *     tabletop viewed from a camera at 1 m: the VERTICAL one.
 *   · MARGINALISATION over the belief. P(detect) is unimodal in fill, so E[p(fill)] ≠ p(E[fill]) — the gap is
 *     largest exactly where it matters, near the shoulders. Propagating the belief's own σ through the fill
 *     and integrating (5-node Gauss-Hermite) makes an uncertain object want a SAFER framing automatically.
 *     This is what replaces the old hand-picked `kStandOffSafetyMarginM = 0.45`: the safety margin becomes a
 *     consequence of precision, which is the repo's modelling rule rather than an exception to it.
 *   · CONTINUOUS visibility. The frustum + occluder test yields visible_frac ∈ [0,1], not a boolean. A
 *     candidate whose target is 5 % clipped should be scored 5 % worse, not deleted (the per-agent
 *     `if (sight_blocked(...)) continue;` was the one hard gate this work had left).
 *   · the stand-off argmax and the usable BAND, found by scanning the model above — so they inherit the
 *     vertical axis, the mount height and the belief σ instead of inverting a horizontal-only formula.
 *   · the four-vertical-face enumeration + ranking that table / cabinet / refrigerator / chair / door each
 *     had their own copy of.
 *
 * WHAT DOES NOT LIVE HERE: the belief. Each agent keeps its own Σ and its own predicted_information(); it hands
 * this module a `raw_gain(face, standoff)` callback. That split is deliberate — this module knows the SENSOR,
 * the agent knows the OBJECT, and neither needs to learn the other's competence. It is the same split the
 * affordance protocol already draws between producer (what view it needs) and controller (global feasibility).
 *
 * Pure geometry + Eigen: no DSR, no Qt, no agent types. Sibling of common/occlusion/occlusion.h, which answers
 * the same "could it have been seen?" question for the existence channel.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <span>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "../detectability/detectability.h"   // rc::detect — P(detect | fill, visible_frac)

namespace rc::nbv
{

// ─── Sensor: the forward model, to the depth the detector envelope actually needs ─────────────────────────
//
// Both FoVs come from the REAL intrinsics (hfov = 2·atan(W/2fx), vfov = 2·atan(H/2fy)); the defaults are a
// conservative stand-in for a caller with no CameraAPI bound yet, not a specification of any camera here.
struct Sensor
{
    float hfov_rad     = 70.0f * std::numbers::pi_v<float> / 180.0f;
    float vfov_rad     = 0.0f;    // ≤0 ⇒ vertical channel off (horizontal-only fill, the old behaviour)
    float height_m     = 0.0f;    // optical centre above the floor, room frame — sets which axis binds
    rc::detect::DetectorEnvelope env{};

    bool has_vertical() const { return vfov_rad > 0.05f and vfov_rad < 3.0f; }
    float tan_half_h()  const { return std::tan(0.5f * std::clamp(hfov_rad, 0.1f, 3.0f)); }
    float tan_half_v()  const { return std::tan(0.5f * std::clamp(vfov_rad, 0.1f, 3.0f)); }

    // ★Is this a COMPLETE camera model, or a partially-populated one?
    //
    // This exists because the failure it guards is silent and expensive. Both fields are filled from the graph
    // and both can be missing for a while after startup: the intrinsics only appear once robot_concept starts
    // publishing frames, and height_m comes from the room→zed transform, so it stays 0 for as long as
    // room_concept takes to publish the ROOM node — much longer. A Sensor missing either one still ANSWERS:
    // vfov 0 ⇒ has_vertical() false ⇒ the fill model silently collapses to horizontal-only, which is exactly
    // the bug this module was built to remove, and it returns a confident number with no signal anything is
    // wrong. Measured on a live 0.64 x 0.64 x 1.92 m refrigerator: complete model → 3.43 m stand-off, missing
    // vertical → 0.64 m. The robot drives nose-to-nose with the box and no mask is possible.
    //
    // So an incomplete model must REFUSE, not guess — plan_faces() returns an invalid Plan. These are
    // completeness conditions on the model's inputs, not tuning thresholds: a real camera has a vertical field
    // of view and sits above the floor, or we do not yet have its model. height_m > 0 is "we read the mount",
    // since a floor-level optical centre is not a thing on this robot.
    bool complete() const { return has_vertical() and height_m > 1e-3f and hfov_rad > 0.05f; }
};

// ─── Target: the object as the BELIEF currently holds it — estimate AND uncertainty ───────────────────────
//
// σ_pos/σ_extent are what make the framing precision-aware. Left at 0 the module degrades gracefully to the
// point-estimate behaviour, so an agent can adopt it in two steps.
struct Target
{
    float cx = 0.0f, cy = 0.0f, yaw = 0.0f;   // footprint pose, room frame
    float w  = 0.0f, h  = 0.0f;               // footprint extent along the object's own x / y axes
    float z0 = 0.0f, z1 = 0.0f;               // vertical span, room frame; z1 ≤ z0 ⇒ vertical channel off
    float sigma_pos_m    = 0.0f;              // belief σ on the centre  (marginalisation)
    float sigma_extent_m = 0.0f;              // belief σ on the extent  (marginalisation)

    Eigen::Vector2f centre() const { return {cx, cy}; }
    Eigen::Vector2f axis_x() const { return {std::cos(yaw), std::sin(yaw)}; }
    Eigen::Vector2f axis_y() const { return {-std::sin(yaw), std::cos(yaw)}; }
    // The radius the object can present from an ARBITRARY bearing. Used for the published band and the
    // collision floor, where one number must hold from every approach direction.
    float circumscribed_radius() const { return 0.5f * std::sqrt(w * w + h * h); }
    bool  has_vertical() const { return z1 > z0 + 1e-3f; }

    // Exact projected half-width of the oriented footprint along image-horizontal `u` — the support function
    // of a rectangle, a|u·ex| + b|u·ey|. Exact where the circumscribed radius is merely an upper bound: seen
    // square-on, a 2.4×0.9 m table presents 1.2 m, not its 1.28 m half-diagonal.
    float support_half_width(const Eigen::Vector2f& u) const
    {
        return 0.5f * w * std::abs(u.dot(axis_x())) + 0.5f * h * std::abs(u.dot(axis_y()));
    }
};

// An oriented footprint that can be stood inside or looked through. Room frame.
struct Obstacle { float cx = 0.0f, cy = 0.0f, w = 0.0f, h = 0.0f, yaw = 0.0f; };

// ─── walls as obstacles ───────────────────────────────────────────────────────────────────────────────────
//
// THE ROOM ITSELF IS AN OCCLUDER, and until now nothing told the planner so: `collect_graph_obstacles` reads
// only `object`/`box` nodes — furniture — so the walls were invisible to the sight test. For an object set
// INTO a wall (a door) that is not a detail, it is the whole answer: the leaf's two EDGE faces have normals
// lying IN the wall plane, so their viewpoints sit on (indeed inside) the wall, several metres along it, and
// nothing marked those sightlines blocked. Live proof, door_concept 2026-08-07: the winner was `face=-x`, the
// HINGE edge, at d=3.67 m — a point on the wall, which is exactly what the user saw the affordance drive to.
//
// Modelling note: this is why an angular "view the door only from a cone about its normal" rule is NOT needed.
// The cone is the CONSEQUENCE of the wall being opaque — a grazing view has to traverse metres of wall to
// reach the leaf — so it falls out of `visible_fraction`'s continuous ray test with no threshold anywhere.
//
// A gap in the wall run: an aperture the sensor genuinely sees THROUGH (a doorway). Without these, the wall
// would also block the honest head-on views, because the leaf sits INSIDE the wall's own footprint.
struct WallGap { Eigen::Vector2f centre{0.0f, 0.0f}; float half_w = 0.0f; };

// Each polygon edge → a thin oriented box, minus the spans covered by `gaps`. Pure geometry (no DSR), so it
// stays testable in the standalone harness. `thickness_m` is the wall's depth: occlusion by a plane barely
// depends on it (a ray either crosses the plane or runs along it), so callers should DERIVE it from something
// physical they already have rather than invent a knob — door_concept passes its own fitted leaf thickness,
// the depth of material the aperture is cut through.
inline std::vector<Obstacle> wall_obstacles(std::span<const Eigen::Vector2f> polygon, float thickness_m,
                                            std::span<const WallGap> gaps = {})
{
    std::vector<Obstacle> out;
    const std::size_t n = polygon.size();
    if (n < 3 or not (thickness_m > 0.0f))
        return out;

    for (std::size_t i = 0, j = n - 1; i < n; j = i++)
    {
        const Eigen::Vector2f A = polygon[j], B = polygon[i];
        const Eigen::Vector2f d = B - A;
        const float len = d.norm();
        if (len < 1e-3f)
            continue;
        const Eigen::Vector2f u = d / len;                       // along the wall
        const Eigen::Vector2f nrm(-u.y(), u.x());                // across it
        const float yaw = std::atan2(u.y(), u.x());

        // Spans of THIS edge that are open. A gap belongs to this edge only if it sits on it: its centre must
        // project inside [0,len] and lie within the wall's own half-thickness of the edge line — otherwise a
        // door on the opposite wall would punch a hole here.
        std::vector<std::pair<float, float>> open;
        for (const auto& g : gaps)
        {
            const Eigen::Vector2f r = g.centre - A;
            const float s = r.dot(u);
            if (std::abs(r.dot(nrm)) > 0.5f * thickness_m + g.half_w)
                continue;
            const float s0 = std::max(0.0f, s - g.half_w), s1 = std::min(len, s + g.half_w);
            if (s1 > s0)
                open.emplace_back(s0, s1);
        }
        std::sort(open.begin(), open.end());

        // Emit the SOLID complement: the runs of wall between the openings.
        // (NOT named `emit` — Qt #defines that to nothing, and this header is included from Qt agents.)
        const auto add_run = [&](float a, float b)
        {
            if (b - a < 1e-3f)
                return;
            const Eigen::Vector2f c = A + u * (0.5f * (a + b));
            out.push_back(Obstacle{c.x(), c.y(), b - a, thickness_m, yaw});
        };
        float cursor = 0.0f;
        for (const auto& [s0, s1] : open)
        {
            add_run(cursor, s0);
            cursor = std::max(cursor, s1);
        }
        add_run(cursor, len);
    }
    return out;
}

// One robot pose to score, with the information a DETECTION from it would yield (the agent's own Σ maths).
struct Candidate
{
    Eigen::Vector2f pos{0.0f, 0.0f};
    float yaw      = 0.0f;    // heading (rad); the camera's optical axis in the horizontal plane
    float raw_gain = 0.0f;    // ΔH_detect (nats) — what the agent's belief expects GIVEN a detection
    int   face     = -1;      // provenance, for the caller's ranked-face array
};

// What the model says about a candidate. `expected_gain` is the EFE quantity; the rest is why.
struct Score
{
    float fill          = 0.0f;   // predicted roi_fill = max over both image axes
    float fill_min      = 0.0f;   // the SHORT axis — what decides "enough pixels to segment"
    float fill_sigma    = 0.0f;   // its σ, propagated from the belief
    float visible_frac  = 0.0f;   // frustum ∧ unoccluded fraction of the silhouette
    float p_detect      = 0.0f;   // E_belief[ P(detect | fill, visible_frac) ]
    float expected_gain = 0.0f;   // p_detect · raw_gain  ← the only number selection should compare
    bool  stands_inside = false;  // the pose is inside an obstacle (feasibility, not detectability)
};

// ─── numerics ─────────────────────────────────────────────────────────────────────────────────────────────
namespace detail
{
// 5-node Gauss-Hermite in PROBABILISTS' form: ∫ f(x) N(x;0,1) dx ≈ Σ wₖ f(xₖ). Nodes/weights are exact
// constants of the quadrature, not tuning: 5 nodes integrate a degree-9 polynomial exactly, which is far more
// than the two logistics need. This is a numerical integration rule; it makes no decision.
inline constexpr std::array<float, 5> kGHx{-2.856970f, -1.355626f, 0.0f, 1.355626f, 2.856970f};
inline constexpr std::array<float, 5> kGHw{ 0.011257f,  0.222076f, 0.533333f, 0.222076f, 0.011257f};

// Slab test: does the segment a→b cross the oriented rectangle `o`? (Shared by the sight test.)
inline bool segment_hits_box(const Eigen::Vector2f& a, const Eigen::Vector2f& b, const Obstacle& o)
{
    const float c = std::cos(-o.yaw), s = std::sin(-o.yaw);
    const Eigen::Vector2f d = b - a;
    const Eigen::Vector2f half(0.5f * o.w, 0.5f * o.h);
    const Eigen::Vector2f p0(c * (a.x() - o.cx) - s * (a.y() - o.cy),
                             s * (a.x() - o.cx) + c * (a.y() - o.cy));
    const Eigen::Vector2f dd(c * d.x() - s * d.y(), s * d.x() + c * d.y());
    float t0 = 0.0f, t1 = 1.0f;
    for (int k = 0; k < 2; ++k)
    {
        if (std::abs(dd(k)) < 1e-6f)
        {
            if (std::abs(p0(k)) > half(k)) return false;   // parallel and outside this slab
        }
        else
        {
            float ta = (-half(k) - p0(k)) / dd(k), tb = (half(k) - p0(k)) / dd(k);
            if (ta > tb) std::swap(ta, tb);
            t0 = std::max(t0, ta);
            t1 = std::min(t1, tb);
        }
    }
    return t0 < t1 and t1 > 1e-3f and t0 < 1.0f - 1e-3f;
}
// Even-odd ray-cast: is q strictly inside the polygon? (Shared by the reachability test.)
inline bool point_in_polygon(const Eigen::Vector2f& q, std::span<const Eigen::Vector2f> poly)
{
    const std::size_t n = poly.size();
    if (n < 3)
        return true;                       // no polygon ⇒ no constraint
    bool inside = false;
    for (std::size_t i = 0, j = n - 1; i < n; j = i++)
        if (((poly[i].y() > q.y()) != (poly[j].y() > q.y())) and
            (q.x() < (poly[j].x() - poly[i].x()) * (q.y() - poly[i].y())
                     / (poly[j].y() - poly[i].y() + 1e-12f) + poly[i].x()))
            inside = not inside;
    return inside;
}

// Distance from q to the polygon boundary (0 if n < 3). Used to keep the robot's BODY inside, not just its
// centre — a pose 10 cm from a wall is not one a 0.6 m robot can occupy.
inline float distance_to_boundary(const Eigen::Vector2f& q, std::span<const Eigen::Vector2f> poly)
{
    const std::size_t n = poly.size();
    if (n < 3)
        return std::numeric_limits<float>::max();
    float best = std::numeric_limits<float>::max();
    for (std::size_t i = 0, j = n - 1; i < n; j = i++)
    {
        const Eigen::Vector2f a = poly[j], ab = poly[i] - poly[j];
        const float len2 = ab.squaredNorm();
        const float u = (len2 > 1e-12f) ? std::clamp((q - a).dot(ab) / len2, 0.0f, 1.0f) : 0.0f;
        best = std::min(best, (q - (a + u * ab)).norm());
    }
    return best;
}
}  // namespace detail

// ─── reachability: the room is a CONTAINER, not an obstacle ───────────────────────────────────────────────
//
// A viewpoint must be somewhere the robot can actually BE. Obstacles answer "can I stand inside this thing?";
// nothing answered "am I even in the room?", and for a wall-anchored object that is the question that bites:
// the far face of a door (or the back of a fridge pushed against a wall) is perfectly VISIBLE — the sightline
// reaches it straight through the aperture — and completely unreachable.
//
// Live 2026-08-07, door_concept: once the walls killed the two edge faces, +y and -y tied EXACTLY (expected
// gain 0.156 each — the raw information term is direction-blind, so the front and back of a door look
// identical to it) and the tie-break took +y, putting the affordance target at (−0.29, 7.27) in a room whose
// y never exceeds 4.63. The robot was being sent through the wall. Reachability breaks that tie the only way
// physics allows; making the gain itself direction-aware (anisotropic ray information) is the deeper fix and
// would break it on merit rather than on feasibility.
//
// This is a FEASIBILITY predicate like stands_inside, deliberately NOT folded into p_detect: an unreachable
// pose is not a badly-framed one, and letting the two trade off would let image quality buy a pose in a wall.
inline bool is_reachable(const Eigen::Vector2f& p, std::span<const Eigen::Vector2f> room_polygon,
                         float clearance_m = 0.0f)
{
    if (room_polygon.size() < 3)
        return true;                       // no room model ⇒ impose no constraint (refuse to guess)
    if (not detail::point_in_polygon(p, room_polygon))
        return false;
    return clearance_m <= 0.0f or detail::distance_to_boundary(p, room_polygon) >= clearance_m;
}

// Is this pose inside an obstacle, allowing `clearance_m` for the robot's own body? A FEASIBILITY question
// (you cannot stand inside a fridge), deliberately kept out of p_detect: it is the controller's domain, and
// conflating it with detectability would let a geometric impossibility trade off against image quality.
//
// ★Obstacles are passed as TRUE footprints and inflated HERE, only for this test. The line-of-sight test must
// see the real extents: an obstacle inflated by the robot radius casts a shadow ~0.6 m wider than the object
// does, so sharing one pre-inflated list (as the pre-port agents did) silently marks sightlines blocked that
// are wide open.
inline bool stands_inside(const Eigen::Vector2f& p, std::span<const Obstacle> obstacles, float clearance_m = 0.0f)
{
    const float pad = std::max(0.0f, clearance_m);
    for (const auto& o : obstacles)
    {
        const float c = std::cos(-o.yaw), s = std::sin(-o.yaw);
        const float dx = p.x() - o.cx, dy = p.y() - o.cy;
        if (std::abs(c * dx - s * dy) <= 0.5f * o.w + pad and std::abs(s * dx + c * dy) <= 0.5f * o.h + pad)
            return true;
    }
    return false;
}

// ─── projected fill ───────────────────────────────────────────────────────────────────────────────────────
//
// PINHOLE-EXACT, and computed the way the agents MEASURE roi_fill: project the box's CORNERS and take the
// bbox, exactly as table_projection.cpp does (col = cx_px + (X/Y)·fx, row = cy_px − (Z/Y)·fy), then divide by
// the image size — with W/fx = 2·tan(hfov/2) the pixel scale cancels and only the FoVs are needed.
//
// ★Per-corner, not a fronto-parallel segment at the centre distance. Perspective is the whole story at close
// range: the near corners of a 0.6 m box seen from 1.5 m project far wider than its support width, so the
// segment approximation UNDER-reads fill exactly where the "too close, mask truncates" shoulder lives — the
// failure this module exists to prevent. Cost is 8 corners.
//
// `heading` is the camera's optical axis in the horizontal plane; NaN ⇒ face the object centre (what every
// NBV candidate does). Camera frame follows the repo convention: X right, Y forward/depth, Z up.
//
// The "inside the object" sentinel is +infinity, deliberately NOT FLT_MAX: FLT_MAX is finite, so it slips past
// predicted_fill_sigma's isfinite() guard, and the marginalisation then averages p_detect over nodes that
// straddle back into the detectable range — handing a pose INSIDE the object a small non-zero P(detect).
// Returns {fill_max, fill_min} — BOTH projected axes, because the detector envelope's two shoulders are
// governed by different ones (see rc::detect::p_detect). The max alone cannot tell a tall sliver from a
// well-framed object, which is what sent the robot along a wall to look at a door edge-on.
inline std::pair<float, float> predicted_fill_axes(
        const Target& t, const Eigen::Vector2f& from, const Sensor& sensor,
        float heading = std::numeric_limits<float>::quiet_NaN())
{
    const Eigen::Vector2f to_obj = t.centre() - from;
    if (to_obj.norm() < 1e-3f)
        return {std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity()};   // standing on it: overflows by construction
    const float yaw = std::isfinite(heading) ? heading : std::atan2(to_obj.y(), to_obj.x());
    const Eigen::Vector2f fwd{std::cos(yaw), std::sin(yaw)};
    const Eigen::Vector2f right{std::sin(yaw), -std::cos(yaw)};

    const Eigen::Vector2f ex = t.axis_x(), ey = t.axis_y();
    const float hw = 0.5f * t.w, hh = 0.5f * t.h;
    const bool do_vert = sensor.has_vertical() and t.has_vertical();
    const std::array<float, 2> zs{t.z0, do_vert ? t.z1 : t.z0};

    float min_u = std::numeric_limits<float>::max(), max_u = -std::numeric_limits<float>::max();
    float min_v = std::numeric_limits<float>::max(), max_v = -std::numeric_limits<float>::max();
    for (int c = 0; c < 4; ++c)
    {
        const float sx = (c == 0 or c == 3) ? 1.0f : -1.0f;
        const float sy = (c < 2) ? 1.0f : -1.0f;
        const Eigen::Vector2f p = t.centre() + ex * (sx * hw) + ey * (sy * hh);
        const Eigen::Vector2f rel = p - from;
        const float depth = rel.dot(fwd);
        if (depth < 0.05f)
            return {std::numeric_limits<float>::infinity(),
                    std::numeric_limits<float>::infinity()};   // a corner at/behind the image plane
        const float u = rel.dot(right) / depth;
        min_u = std::min(min_u, u); max_u = std::max(max_u, u);
        for (const float z : zs)
        {
            const float v = (z - sensor.height_m) / depth;
            min_v = std::min(min_v, v); max_v = std::max(max_v, v);
        }
    }
    const float fill_h = (max_u - min_u) / (2.0f * sensor.tan_half_h());
    if (not do_vert)
        return {fill_h, fill_h};       // no vertical channel: the model knows only one axis
    const float fill_v = (max_v - min_v) / (2.0f * sensor.tan_half_v());
    return {std::max(fill_h, fill_v), std::min(fill_h, fill_v)};
}

// roi_fill as the agents MEASURE it: the max of the two axes. Thin wrapper over the pair.
inline float predicted_fill(const Target& t, const Eigen::Vector2f& from, const Sensor& sensor,
                            float heading = std::numeric_limits<float>::quiet_NaN())
{
    return predicted_fill_axes(t, from, sensor, heading).first;
}

// σ of the predicted fill, propagated from the belief's own σ (first order). Two independent routes:
//   · the EXTENT is uncertain      → ∂fill/∂support = 1/(d·tan)         , σ_support = ½·σ_extent
//   · the RANGE is uncertain       → ∂fill/∂d       = −fill/d
// This is what turns "how sure are we?" into "how much framing margin do we need?".
inline float predicted_fill_sigma(const Target& t, const Eigen::Vector2f& from, const Sensor& sensor)
{
    const float d = (t.centre() - from).norm();
    if (d < 1e-3f or (t.sigma_extent_m <= 0.0f and t.sigma_pos_m <= 0.0f))
        return 0.0f;
    const float fill = predicted_fill(t, from, sensor);
    if (not std::isfinite(fill))
        return 0.0f;
    // The binding axis carries the sensitivity; using the max axis' tangent keeps the two consistent.
    const float tan_half = (sensor.has_vertical() and t.has_vertical())
                         ? std::min(sensor.tan_half_h(), sensor.tan_half_v()) : sensor.tan_half_h();
    const float d_extent = 0.5f * t.sigma_extent_m / (d * tan_half);
    const float d_range  = fill * t.sigma_pos_m / d;
    return std::sqrt(d_extent * d_extent + d_range * d_range);
}

// ─── visibility ───────────────────────────────────────────────────────────────────────────────────────────
//
// Fraction of the object's silhouette that is inside the frustum AND unoccluded — the `visible_frac` the
// detector envelope multiplies by. Sampled around the footprint perimeter (and over the height when known)
// because a partially-clipped object is the normal case at close range, and the whole point of this module is
// that "partially" is a NUMBER. kPerim/kVert are quadrature resolution, not decision thresholds.
inline float visible_fraction(const Target& t, const Eigen::Vector2f& from, float yaw,
                              const Sensor& sensor, std::span<const Obstacle> obstacles)
{
    constexpr int kPerim = 16, kVert = 3;
    const float half_h = 0.5f * std::clamp(sensor.hfov_rad, 0.1f, 3.0f);
    const float half_v = 0.5f * std::clamp(sensor.vfov_rad, 0.1f, 3.0f);
    const bool  do_vert = sensor.has_vertical() and t.has_vertical();
    const Eigen::Vector2f ex = t.axis_x(), ey = t.axis_y();

    int seen = 0, total = 0;
    for (int i = 0; i < kPerim; ++i)
    {
        // Walk the footprint rectangle's perimeter: 4 sides × kPerim/4 samples.
        const float u = 4.0f * static_cast<float>(i) / kPerim;             // [0,4) — side index + fraction
        const int   side = static_cast<int>(u);
        const float f = 2.0f * (u - static_cast<float>(side)) - 1.0f;      // [-1,1) along the side
        float lx = 0.0f, ly = 0.0f;
        switch (side)
        {
            case 0:  lx =  0.5f * t.w; ly =  f * 0.5f * t.h; break;
            case 1:  lx = -0.5f * t.w; ly =  f * 0.5f * t.h; break;
            case 2:  lx =  f * 0.5f * t.w; ly =  0.5f * t.h; break;
            default: lx =  f * 0.5f * t.w; ly = -0.5f * t.h; break;
        }
        const Eigen::Vector2f p = t.centre() + lx * ex + ly * ey;
        const Eigen::Vector2f rel = p - from;
        const float range = rel.norm();
        if (range < 1e-3f)
            continue;

        // Horizontal containment: bearing offset from the heading.
        float bearing = std::atan2(rel.y(), rel.x()) - yaw;
        while (bearing >  std::numbers::pi_v<float>) bearing -= 2.0f * std::numbers::pi_v<float>;
        while (bearing < -std::numbers::pi_v<float>) bearing += 2.0f * std::numbers::pi_v<float>;
        const bool in_h = std::abs(bearing) <= half_h;

        // Occlusion is height-independent here (obstacles are floor footprints), so test it once per column.
        bool occluded = false;
        for (const auto& o : obstacles)
            if (detail::segment_hits_box(from, p, o)) { occluded = true; break; }

        for (int k = 0; k < (do_vert ? kVert : 1); ++k)
        {
            ++total;
            if (not in_h or occluded)
                continue;
            if (do_vert)
            {
                const float z = t.z0 + (t.z1 - t.z0) * static_cast<float>(k) / (kVert - 1);
                if (std::abs(std::atan2(z - sensor.height_m, range)) > half_v)
                    continue;
            }
            ++seen;
        }
    }
    return (total > 0) ? static_cast<float>(seen) / static_cast<float>(total) : 0.0f;
}

// ─── the detection probability, marginalised over the belief ──────────────────────────────────────────────
//
// E_belief[ P(detect | fill) ] — NOT P(detect | E[fill]). P(detect) is unimodal, so the two differ most near
// the shoulders, which is exactly where a framing decision is made. Integrating pushes the framing away from
// the cliff in proportion to how badly the object is known, which is the precision-derived replacement for a
// hand-picked stand-off margin.
// The pair is perturbed COHERENTLY: a range or extent error scales both projected axes together, so the
// quadrature walks a ray through (fill_max, fill_min) space rather than jittering the two independently.
inline float expected_p_detect(float fill_max, float fill_min, float fill_sigma, float visible_frac,
                               const rc::detect::DetectorEnvelope& env)
{
    // ★A NON-FINITE FILL IS A REAL INPUT, NOT AN ERROR: predicted_fill_axes returns +inf BY DESIGN when the
    // viewpoint is degenerate (camera essentially on top of the target — a just-born belief that has not yet
    // separated from the sensor). Physically that object overflows the frame completely, so it is NOT
    // detectable and the honest answer is 0. Without this the marginalisation below computes
    // ratio = fill_min/fill_max = inf/inf = NaN and returns NaN — and a NaN p_detect is not a wrong number,
    // it is a POISON: it flows into ExistenceBelief as p_vis, passes the `pv <= 0` guard (NaN compares
    // false), and pins L at NaN forever. Measured 2026-08-10 on the live bottle: L = nan from the first
    // logged cycle, unrecoverable by 50 subsequent good confirmations.
    if (not std::isfinite(fill_max) or not std::isfinite(fill_min))
        return 0.0f;
    if (not (fill_sigma > 1e-4f))
        return rc::detect::p_detect(fill_max, fill_min, visible_frac, env);
    const float ratio = (fill_max > 1e-6f) ? std::clamp(fill_min / fill_max, 0.0f, 1.0f) : 1.0f;
    float acc = 0.0f;
    for (std::size_t k = 0; k < detail::kGHx.size(); ++k)
    {
        const float fmax = std::max(0.0f, fill_max + detail::kGHx[k] * fill_sigma);
        acc += detail::kGHw[k] * rc::detect::p_detect(fmax, fmax * ratio, visible_frac, env);
    }
    return std::clamp(acc, 0.0f, 1.0f);
}

// Score ONE candidate pose. This is the tier-1 entry point: an agent whose candidate geometry is its own
// (bottle's hidden-face arc, human's occlusion-reducing orbit) still gets the weighting from one call.
inline Score score(const Candidate& c, const Target& t, const Sensor& sensor,
                   std::span<const Obstacle> obstacles = {}, float robot_radius_m = 0.0f)
{
    Score s;
    s.stands_inside = stands_inside(c.pos, obstacles, robot_radius_m);
    const auto [fmax, fmin] = predicted_fill_axes(t, c.pos, sensor, c.yaw);
    s.fill          = fmax;
    s.fill_min      = fmin;
    s.fill_sigma    = predicted_fill_sigma(t, c.pos, sensor);
    s.visible_frac  = visible_fraction(t, c.pos, c.yaw, sensor, obstacles);
    s.p_detect      = expected_p_detect(s.fill, s.fill_min, s.fill_sigma, s.visible_frac, sensor.env);
    s.expected_gain = s.p_detect * std::max(0.0f, c.raw_gain);
    return s;
}

// ─── stand-off from the model, not from a constant ────────────────────────────────────────────────────────
//
// Scans the SAME p_detect the score uses, along `dir`, so the answer inherits the vertical axis, the mount
// height and the belief σ — none of which an analytic horizontal inversion can see. `floor_m` is the caller's
// geometric limit (robot radius + object radius); below it the pose is unreachable, not merely bad.
//
// ★PER-FACE, and that is not a detail. A 2.4 × 0.9 m table presents 0.45 m of half-width from its narrow face
// and 1.2 m from its wide one, so the two want stand-offs a factor of ~2.7 apart. One shared stand-off (what
// every agent did, via the circumscribed radius) parks the wide face at a fill where the detector cannot fire
// and then reads the resulting ~0 score as "that face is useless" — when the truth is "stand further back for
// that face". Scan along the face normal the candidate actually uses.
// ★The scan is parameterised by the SAME origin the candidate is placed from — `origin + dir·d`. Getting this
// wrong is silent and costly: scanning ranges from the object CENTRE and then placing the robot at
// face_centre + normal·d puts it half an extent too far out, so every face lands off the peak it was supposed
// to sit on (measured: a table's narrow face at p_detect 0.881 instead of its 0.970 optimum). Callers using
// the four-face convention pass the FACE centre, which also makes `floor_m` mean what it says.
// `near_at_floor` marks a near end set by GEOMETRY (the robot's own radius) rather than by the detector — the
// caller may want to say "I cannot get closer" rather than "detectability stops here".
struct StandoffBand
{
    float best = 0.0f, near_m = 0.0f, far_m = 0.0f, peak_p = 0.0f;
    bool  near_at_floor = false;
};

inline StandoffBand standoff_band(const Target& t, const Sensor& sensor,
                                  const Eigen::Vector2f& origin, const Eigen::Vector2f& dir,
                                  float floor_m, float frac_of_peak = 0.5f, float max_m = 8.0f)
{
    constexpr int kSteps = 160;
    const float lo = std::max(0.05f, floor_m), hi = std::max(lo + 0.05f, max_m);
    std::array<float, kSteps + 1> p{};
    StandoffBand band;
    band.best = lo;
    for (int i = 0; i <= kSteps; ++i)
    {
        const float d = lo + (hi - lo) * static_cast<float>(i) / kSteps;
        const Eigen::Vector2f from = origin + dir * d;
        const auto [fmax, fmin] = predicted_fill_axes(t, from, sensor);
        p[i] = expected_p_detect(fmax, fmin, predicted_fill_sigma(t, from, sensor), 1.0f, sensor.env);
        if (p[i] > band.peak_p) { band.peak_p = p[i]; band.best = d; }
    }
    const float want = std::clamp(frac_of_peak, 0.0f, 0.99f) * band.peak_p;
    band.near_m = band.best;
    band.far_m  = band.best;
    for (int i = 0; i <= kSteps; ++i)
        if (p[i] >= want)
        {
            const float d = lo + (hi - lo) * static_cast<float>(i) / kSteps;
            band.near_m = std::min(band.near_m, d);
            band.far_m  = std::max(band.far_m, d);
        }
    band.near_at_floor = band.near_m <= lo + 1e-3f;

    // ★The far end must be the MODEL's answer, not the scan's edge. For a tall object the envelope is still
    // above `want` at max_m (a 1.7 m fridge is at p≈0.66 at 8 m), so the loop above would report 8.00 — a
    // number that looks like a limit and is really just where we stopped looking. Beyond the peak p falls
    // monotonically (fill → 0 ⇒ the too-few-pixels shoulder takes over), so march outward until it genuinely
    // drops below `want`, then bisect. The cap is a numerical backstop, not a modelling choice.
    if (band.far_m >= hi - 1e-3f)
    {
        constexpr float kHardCapM = 200.0f;
        const auto p_at = [&](float d)
        {
            const Eigen::Vector2f from = origin + dir * d;
            const auto [fmax, fmin] = predicted_fill_axes(t, from, sensor);
            return expected_p_detect(fmax, fmin, predicted_fill_sigma(t, from, sensor), 1.0f, sensor.env);
        };
        float in = band.far_m, out = band.far_m;
        while (out < kHardCapM and p_at(out) >= want)
        { in = out; out *= 1.5f; }
        if (p_at(out) >= want)
            band.far_m = out;               // still detectable at the cap: report the cap, honestly reached
        else
        {
            for (int k = 0; k < 24; ++k)    // bisect the crossing to ~1e-4 of the interval
            {
                const float mid = 0.5f * (in + out);
                (p_at(mid) >= want ? in : out) = mid;
            }
            band.far_m = in;
        }
    }
    return band;
}

// The framing the servo should drive to — the argmax of the detector envelope, i.e. the SAME target the
// stand-off realises. Publishing a different constant (every agent shipped 0.45) makes the servo undo the
// planner's choice on arrival.
inline float framing_fill(const Sensor& sensor) { return rc::detect::best_fill(sensor.env); }

// ─── the shared four-face plan ────────────────────────────────────────────────────────────────────────────
//
// Face order is the repo-wide [+x,-x,+y,-y] object-frame convention that the affordance protocol's ranked
// `face_gains` and the controller's resolver already agree on. Do not reorder.
struct Plan
{
    bool  valid       = false;
    int   best_face   = 0;
    float best_standoff_m = 0.0f;
    Eigen::Vector2f best_pos{0.0f, 0.0f};
    float best_yaw    = 0.0f;
    std::array<float, 4> face_gains{};      // EXPECTED (detection-weighted) nats — what selection ranks on
    std::array<float, 4> face_raw_gains{};  // as supplied by the agent's belief, for logs/diagnosis
    std::array<float, 4> face_p_detect{};
    std::array<float, 4> face_visible{};
    std::array<bool,  4> face_reachable{};  // pose inside the room (if one was supplied) — FEASIBILITY
    // ★Was ANY face actually usable (not inside an obstacle, and reachable)? When false, best_pos below is a
    // raw-argmax HINT — it is knowingly infeasible, and the raw argmax for a thin plate is an EDGE face whose
    // pose sits in the wall. A caller must NOT publish that as a standpoint: a producer's unroutable pose is
    // repaired downstream by nearest_reachable, which snaps it to the closest cell the robot can occupy —
    // i.e. the floor right at the object. That is how an unreachable viewpoint becomes "the robot drives
    // into the door". Refuse (hold the affordance) instead.
    bool  any_usable  = false;
    std::array<float, 4> face_standoff_m{}; // each face is framed at ITS OWN best range, not a shared one
    // The band PUBLISHED to the controller is the winning face's — the one it will actually realise. The
    // affordance protocol carries a single band; a controller that falls back to a lower-ranked face should
    // re-derive that face's band rather than reuse this one.
    float standoff_min_m = 0.0f, standoff_max_m = 0.0f, framing_fill = 0.0f;

    bool is_finite() const
    {
        return std::isfinite(best_pos.x()) and std::isfinite(best_pos.y()) and
               std::isfinite(best_yaw) and std::isfinite(face_gains[best_face]);
    }
};

// Enumerate the four vertical faces, place each at the model's stand-off, and rank them by DETECTION-WEIGHTED
// information. `raw_gain(face_idx, standoff_m)` is the agent's own D-optimal ΔH given a detection.
//
// A pose inside an obstacle is rejected (geometry, not preference); everything else — clipping, occlusion,
// bad framing — degrades the score continuously, so the ranking never loses a face it could still partly use.
template <class GainFn>
Plan plan_faces(const Target& t, const Sensor& sensor, float robot_radius_m,
                std::span<const Obstacle> obstacles, GainFn&& raw_gain,
                std::span<const Eigen::Vector2f> room_polygon = {})
{
    Plan plan;
    // ★THE choke point. Every four-face agent goes through here, so refusing on an incomplete camera model in
    // ONE place is what makes it impossible for an agent to forget and quietly publish a horizontal-only
    // viewpoint. `plan.valid` stays false and the callers' existing `if (not plan.valid) return {};` turns
    // that into "no proposal this cycle", which is the honest answer while the model is still arriving.
    if (not sensor.complete())
        return plan;
    if (not (t.w > 0.0f and t.h > 0.0f) or not sensor.env.valid())
        return plan;

    // Stand-offs here are FACE-RELATIVE (the repo-wide convention: viewpoint = face_centre + normal·d), so the
    // geometric floor is just the robot's radius — at d = robot_radius its disc exactly touches the face plane,
    // and a convex footprint lies entirely beyond that plane. The circumscribed radius belongs to a
    // CENTRE-relative stand-off; adding it here (as the pre-port code did, on a face-relative d) pushed every
    // viewpoint out by half an extent for no geometric reason.
    const float collision_floor = std::max(0.05f, robot_radius_m);
    plan.framing_fill = framing_fill(sensor);

    const Eigen::Vector2f ex = t.axis_x(), ey = t.axis_y();
    const float hw = 0.5f * t.w, hh = 0.5f * t.h;
    const std::array<Eigen::Vector2f, 4> normals{ ex, -ex, ey, -ey };
    const std::array<Eigen::Vector2f, 4> centres{ t.centre() + ex * hw, t.centre() - ex * hw,
                                                  t.centre() + ey * hh, t.centre() - ey * hh };

    std::array<StandoffBand, 4> bands{};
    float best_score = -std::numeric_limits<float>::max();
    bool  any = false;
    for (int i = 0; i < 4; ++i)
    {
        bands[i] = standoff_band(t, sensor, centres[i], normals[i], collision_floor);

        Candidate c;
        c.face     = i;
        c.pos      = centres[i] + normals[i] * bands[i].best;
        c.yaw      = std::atan2(t.cy - c.pos.y(), t.cx - c.pos.x());   // look back at the object centre
        c.raw_gain = static_cast<float>(raw_gain(i, bands[i].best));

        const Score s = score(c, t, sensor, obstacles, robot_radius_m);
        // …and can the robot BE there? The room is a container, not an obstacle (see is_reachable): the far
        // face of a wall-anchored object is fully visible through the aperture and completely unreachable.
        const bool reachable = is_reachable(c.pos, room_polygon, robot_radius_m);
        plan.face_raw_gains[i]  = c.raw_gain;
        plan.face_p_detect[i]   = s.p_detect;
        plan.face_visible[i]    = s.visible_frac;
        plan.face_standoff_m[i] = bands[i].best;
        plan.face_reachable[i]  = reachable;
        // A pose inside another object — or outside the room entirely — is not a viewpoint at all. These are
        // the two genuinely binary facts here; everything else degrades continuously.
        const bool usable       = not s.stands_inside and reachable;
        plan.face_gains[i]      = usable ? s.expected_gain : 0.0f;

        if (usable and std::isfinite(s.expected_gain) and s.expected_gain > best_score)
        {
            best_score = s.expected_gain;
            plan.best_face = i;
            plan.best_pos  = c.pos;
            plan.best_yaw  = c.yaw;
            any = true;
        }
    }
    // Every face unusable (boxed in): keep the raw argmax as a HINT so the caller still has a direction to
    // report, but leave the gains at their honest ~0 so nothing bids travel on it.
    plan.any_usable = any;
    if (not any)
    {
        const auto it = std::max_element(plan.face_raw_gains.begin(), plan.face_raw_gains.end());
        plan.best_face = static_cast<int>(std::distance(plan.face_raw_gains.begin(), it));
        plan.best_pos  = centres[plan.best_face] + normals[plan.best_face] * bands[plan.best_face].best;
        plan.best_yaw  = std::atan2(t.cy - plan.best_pos.y(), t.cx - plan.best_pos.x());
    }
    plan.best_standoff_m = bands[plan.best_face].best;
    plan.standoff_min_m  = bands[plan.best_face].near_m;
    plan.standoff_max_m  = bands[plan.best_face].far_m;
    plan.valid = plan.is_finite();
    return plan;
}

}  // namespace rc::nbv
