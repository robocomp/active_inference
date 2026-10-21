// doorway_corridor.h — the measured doorway as the local policy's field and trap logic see it (Task 8). Pure (Eigen + std):
// the controller (goal_adapter.h) and the Task-6 harness (tests/test_policy_scenarios.cpp) share it.
#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <optional>

namespace rc::policy
{
// The measured doorway gap in the ROOM frame. RelPose is the opening in the robot frame (x right, y forward): x.head<2>
// the centre, x.z() the wall's yaw psi; its normal (the direction of travel through it) is (-sin psi, cos psi) there
// (controller_anchor_servo.h anchor_cross_geometry).
struct RoomGap { Eigen::Vector2f centre = Eigen::Vector2f::Zero(), normal{0.f, 1.f}; float width = 0.f; };
// ★ THE DOORWAY'S CORRIDOR (where the cost-to-go may run through the opening). The planner's map is the CURRENT room plus
//   the door passage's corridor, which ends 2.0 m past the aperture (controller_door_passage.h: the cross standpoint 1.2 m
//   + 0.8 m) — the far room is not in it, so a goal line past the standoff would be unreachable on it. The opening AS
//   MEASURED supplies the rest: a strip along the normal from back_m before the gap centre to ahead_m past it, over the
//   lateral span the body's centre can occupy (|lateral| <= half the width - the inscribed radius; empty when the
//   opening is narrower than the body, so a too-narrow door stays unreachable). Contact with the jambs, the leaf and the
//   far room is the LiDAR's to judge (contact model), not this strip's.
inline bool in_doorway_corridor(const RoomGap& g, const Eigen::Vector2f& p, float back_m, float ahead_m, float inscribed_radius_m)
{
    const float half = 0.5f * g.width - inscribed_radius_m;
    if (not (half >= 0.f)) return false;
    const Eigen::Vector2f d = p - g.centre;
    const float s = d.dot(g.normal), lat = std::abs(d.dot(Eigen::Vector2f(-g.normal.y(), g.normal.x())));
    return s >= -back_m and s <= ahead_m and lat <= half;
}

// ★ Live fix 2 ruling 2: does a trap disc (centre c, radius r) MEET the measured doorway's corridor (the strip
//   in_doorway_corridor tests)? Only an opening at least the body's full width counts as evidence that the body fits;
//   a narrower one vetoes nothing.
inline bool disc_meets_doorway_corridor(const RoomGap& g, const Eigen::Vector2f& c, float r, float back_m, float ahead_m,
                                        float inscribed_radius_m, float body_width_m)
{
    if (not (g.width >= body_width_m)) return false;
    const float half = std::max(0.f, 0.5f * g.width - inscribed_radius_m);
    const Eigen::Vector2f d = c - g.centre, lat_dir(-g.normal.y(), g.normal.x());
    const float s = d.dot(g.normal), lat = d.dot(lat_dir);
    const float ds = s - std::clamp(s, -back_m, ahead_m), dl = lat - std::clamp(lat, -half, half);
    return std::hypot(ds, dl) <= r;
}

// ★ F4 (2026-10-21 doorway release) — THE MEASURED JAMBS IN THE ROUTE'S RISK. The cost-to-go's free test is the inscribed
//   radius (any heading fits), so without this the field is indifferent between 1 cm and 15 cm of clearance at a jamb and
//   an oblique approach cuts the corner (rtD1b: every doorway Stuck was 1-3 cm from a jamb, 6-12 cm off-centre). The body
//   crossing the opening ALONG ITS NORMAL at lateral offset e from the measured centre touches a jamb iff its left extent
//   e + h_l passes the left jamb at +W/2 or its right extent e - h_r passes the right one at -W/2. The opening's centre is
//   known to sigma_e (the gap's along-wall sigma and the contact model's shared offset sigma_s, independent) and both jambs
//   move WITH it (one measured line), so with one shared lateral error delta ~ N(0, sigma_e^2):
//     P(no contact | e) = Phi((e + c_r)/sigma_e) - Phi((e - c_l)/sigma_e),   c_l = W/2 - h_l, c_r = W/2 - h_r
//   and the crossing costs R(e) = -log P(no contact | e) nats. The WIDTH is measured too: each jamb edge carries the LiDAR's
//   range sigma independently, so the half-width W/2 is N(W/2, sigma_hw^2) with sigma_hw = sigma_r / sqrt 2 (doorway_gap.h
//   doorway_gap_meas_cov: centre and half-width are the edges' mean and half-difference), and P(no contact | e) above is
//   averaged over it — a gap measured a centimetre narrower than the body is then improbable, not impossible. The body is ON the jamb line while its centre's progress s
//   (along the normal, 0 = the line) is in [-front, back]; known to sigma_n (the gap's along-normal sigma, sigma_s):
//     q(s) = Phi((s + front)/sigma_n) - Phi((s - back)/sigma_n),   integral of q over s = front + back.
//   DENSITY (nats per metre, in the convention extra_fn and F1 share — GridPlanner::body_risk_at:
//   one independent body placement per `placement_m` of travel): R(e) q(s) placement_m / (front + back), so that a crossing
//   at offset e accrues exactly R(e) nats in F1's reading (F1's route-price prior is PARKED since F1 fix round 2) and
//   residual_risk_weight x R(e) x placement_m metres in the field.
//   Over the doorway's corridor only (in_doorway_corridor: the strip the measured opening adds; outside it the planner's map
//   and its distance field speak). No gate: R is a continuous function of e and W; P(no contact) = 0 to double precision (a
//   gap measured narrower than the body by many sigma) is +inf, which the field reads as impassable — what that measurement says.
struct DoorwayJambs
{
    RoomGap gap;
    float sigma_lat_m = 0.f;     // the opening's along-wall position sigma (anchor_cross_geometry sigma_lat)
    float sigma_n_m = 0.f;       // its along-normal position sigma
    float sigma_s_m = 0.f;       // the contact model's shared offset: sqrt(sigma_rel_xy^2 + body_geometry_sigma^2)
    float sigma_half_width_m = 0.f;   // the measured half-width's sigma: sigma_r / sqrt 2 (independent edges)
    float half_left_m = 0.f, half_right_m = 0.f;   // the hull's support toward +lat / -lat, heading along the normal
    float front_m = 0.f, back_m = 0.f;             // the hull's support along +normal / -normal
    float placement_m = 0.f;                       // one independent body placement (the circumscribed diameter; F1)
    float corridor_back_m = 1.f, corridor_ahead_m = 2.05f, inscribed_radius_m = 0.23f;   // in_doorway_corridor's strip
};

namespace detail
{
// Phi(a) - Phi(b), a >= b, accurate in both tails (the upper tail as Phi(-b) - Phi(-a))
inline double normal_interval(double a, double b)
{
    constexpr double k = 0.70710678118654752;
    if (b > 0.0) return 0.5 * (std::erfc(b * k) - std::erfc(a * k));
    if (a < 0.0) return 0.5 * (std::erfc(-a * k) - std::erfc(-b * k));
    return 1.0 - 0.5 * std::erfc(a * k) - 0.5 * std::erfc(-b * k);
}
}  // namespace detail

// R(e): the nats of contact risk of crossing the opening at lateral offset e (lat = left of the normal).
inline float doorway_jamb_crossing_nats(const DoorwayJambs& j, float e)
{
    const double se = std::sqrt(double(j.sigma_lat_m) * j.sigma_lat_m + double(j.sigma_s_m) * j.sigma_s_m);
    // P(no contact | e, half-width hw): the shared lateral error between the two jamb edges
    auto p_nc = [&](double hw) -> double
    {
        const double cl = hw - j.half_left_m, cr = hw - j.half_right_m;
        if (not (cl + cr > 0.0)) return 0.0;                                 // the body is wider than this opening
        if (not (se > 0.0)) return (e < cl and -e < cr) ? 1.0 : 0.0;         // a certain line: a step
        return detail::normal_interval((e + cr) / se, (e - cl) / se);
    };
    double p = 0.0;
    if (not (j.sigma_half_width_m > 0.f)) p = p_nc(0.5 * j.gap.width);
    else
    {
        // E over hw ~ N(W/2, sigma_hw^2): trapezoid over +-6 sigma (49 nodes), weights renormalised — exact to the
        // integrand's smoothness, which is set by se and sigma_hw alike
        constexpr int kN = 49; constexpr double kSpan = 6.0;
        double num = 0.0, den = 0.0;
        for (int k = 0; k < kN; ++k)
        {
            const double u = -kSpan + 2.0 * kSpan * k / (kN - 1), w = std::exp(-0.5 * u * u);
            num += w * p_nc(0.5 * j.gap.width + u * j.sigma_half_width_m);
            den += w;
        }
        p = num / den;
    }
    return p > 0.0 ? float(-std::log(p)) : std::numeric_limits<float>::infinity();
}

// The nats the body covers at p crossing this opening: R(e) x q(s) (q = P(the body is on the jamb line)); 0 outside the corridor.
inline float doorway_jamb_cell_nats(const DoorwayJambs& j, const Eigen::Vector2f& p)
{
    if (not in_doorway_corridor(j.gap, p, j.corridor_back_m, j.corridor_ahead_m, j.inscribed_radius_m)) return 0.f;
    const Eigen::Vector2f d = p - j.gap.centre, lat(-j.gap.normal.y(), j.gap.normal.x());
    const double s = d.dot(j.gap.normal), e = d.dot(lat);
    const double sn = std::sqrt(double(j.sigma_n_m) * j.sigma_n_m + double(j.sigma_s_m) * j.sigma_s_m);
    const double q = sn > 0.0 ? detail::normal_interval((s + j.front_m) / sn, (s - j.back_m) / sn)
                              : ((s >= -j.front_m and s <= j.back_m) ? 1.0 : 0.0);
    if (not (q > 0.0)) return 0.f;
    return float(q * doorway_jamb_crossing_nats(j, float(e)));
}
// The per-metre density at p (nats per metre, before residual_risk_weight); 0 outside the corridor.
inline float doorway_jamb_density(const DoorwayJambs& j, const Eigen::Vector2f& p)
{
    const float span = j.front_m + j.back_m;
    if (not (span > 0.f)) return 0.f;
    return doorway_jamb_cell_nats(j, p) * j.placement_m / span;
}

// ★ rtD1c leg 5 — WHICH OPENING IS THE DOOR is itself uncertain. The anchor (door_concept's door, a fitted model) and the
//   controller's [gap] measurement are two hypotheses about where the jambs are; the gap enters with the weight its own
//   score gives it (doorway_gap.h: the product of its location, width and fit agreements with the anchor, in [0, 1]),
//   p_gap = score — the [gap] detector locked onto an open LEAF at score 0.03-0.3 and reported 0.40-0.52 m against an
//   0.87 m door, and a 0.03-score phantom must not replace the anchor's geometry.
//   ⚑ The score is used AS P(the gap is the door). It is an UNCALIBRATED stand-in: a product of agreements with the anchor
//   itself (location under the anchor's covariance, width against the anchor's width, body fit), so it is benignly
//   circular — it can only say how much the measurement agrees with the hypothesis it competes with — and nothing has
//   checked that a score of 0.3 is right 30 % of the time.
//   Per place, the body is contact-free with probability p_gap P_gap(p) + (1 - p_gap) P_anchor(p). Under a hypothesis, a
//   cell inside its corridor has P = e^{-n} (n = the cell's nats, doorway_jamb_cell_nats); a cell OUTSIDE its corridor is
//   what the planner's map says it is — free floor (P = 1) where `map_free` admits it, and wall / leaf (P = 0) where it does
//   not (fix round 1, I-1: the corridor union also_free admits the phantom's strip; under the anchor that strip runs
//   through solid, so it must cost -log p_gap there, ~3.5 nats at score 0.03, not -log(1 - p_gap)). The density is that
//   mixture's -log: a phantom that closes the opening costs at most -log(1 - p_gap) over the anchor's, and a phantom's
//   strip through solid costs at least -log p_gap; a reliable gap (score ~1) is the geometry. No gap: the anchor's, exactly.
//   `map_free`: the planner's own free test WITHOUT the corridors (empty = every cell counts as map-free: the pre-fix
//   behaviour, kept for callers that have no map).
struct DoorwayRisk { DoorwayJambs anchor; std::optional<DoorwayJambs> gap; float p_gap = 0.f; };
inline float doorway_risk_density(const DoorwayRisk& r, const Eigen::Vector2f& p,
                                  const std::function<bool(const Eigen::Vector2f&)>& map_free = {})
{
    const float pg = std::clamp(r.p_gap, 0.f, 1.f);   // a weight: a probability by construction (score in [0,1])
    if (not r.gap.has_value() or not (pg > 0.f)) return doorway_jamb_density(r.anchor, p);
    const float span = r.anchor.front_m + r.anchor.back_m;   // the same body under both hypotheses
    if (not (span > 0.f)) return 0.f;
    const bool on_map = not map_free or map_free(p);
    // P(contact-free here | hypothesis): its corridor's jamb risk inside it; the map outside it
    const auto p_free = [&](const DoorwayJambs& j) -> double
    {
        if (in_doorway_corridor(j.gap, p, j.corridor_back_m, j.corridor_ahead_m, j.inscribed_radius_m))
            return std::exp(-double(doorway_jamb_cell_nats(j, p)));
        return on_map ? 1.0 : 0.0;
    };
    const double pnc = double(pg) * p_free(*r.gap) + (1.0 - pg) * p_free(r.anchor);
    return pnc > 0.0 ? float(-std::log(pnc) * r.anchor.placement_m / span) : std::numeric_limits<float>::infinity();
}
// The space the opening adds to the planner's map: either hypothesis's corridor (a mask, not a cost — the risk prices it).
inline bool in_doorway_risk_corridor(const DoorwayRisk& r, const Eigen::Vector2f& p)
{
    const auto in = [&](const DoorwayJambs& j) { return in_doorway_corridor(j.gap, p, j.corridor_back_m, j.corridor_ahead_m, j.inscribed_radius_m); };
    return in(r.anchor) or (r.gap.has_value() and r.p_gap > 0.f and in(*r.gap));
}

// The jambs of a measured opening for a body crossing it along its normal. sigma_range_m = the LiDAR range sigma each edge
// was measured with (Controller lidar_range_sigma_m). `bare` is the BARE hull (any type with
// support_radius_yaw(yaw, dir_world), inscribed_radius() and circumscribed_radius() — rc::RobotFootprint); back/ahead = the
// corridor strip the controller already joins the two rooms with (in_doorway_corridor).
template <class Body>
DoorwayJambs doorway_jambs(const RoomGap& g, float sigma_lat_m, float sigma_n_m, float sigma_s_m, float sigma_range_m,
                           const Body& bare, float corridor_back_m, float corridor_ahead_m)
{
    DoorwayJambs j;
    j.gap = g;
    j.sigma_lat_m = sigma_lat_m; j.sigma_n_m = sigma_n_m; j.sigma_s_m = sigma_s_m;
    j.sigma_half_width_m = sigma_range_m * 0.70710678f;
    const Eigen::Vector2f n = g.normal.normalized(), lat(-n.y(), n.x());
    const float yaw = std::atan2(n.y(), n.x());   // policy yaw: forward along the normal
    j.half_left_m = bare.support_radius_yaw(yaw, lat);
    j.half_right_m = bare.support_radius_yaw(yaw, Eigen::Vector2f(-lat));
    j.front_m = bare.support_radius_yaw(yaw, n);
    j.back_m = bare.support_radius_yaw(yaw, Eigen::Vector2f(-n));
    j.placement_m = 2.f * bare.circumscribed_radius();
    j.corridor_back_m = corridor_back_m; j.corridor_ahead_m = corridor_ahead_m; j.inscribed_radius_m = bare.inscribed_radius();
    return j;
}

}  // namespace rc::policy
