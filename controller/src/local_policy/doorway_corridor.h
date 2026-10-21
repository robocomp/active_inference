// doorway_corridor.h — the measured doorway as the local policy's field and trap logic see it (Task 8). Pure (Eigen + std):
// the controller (goal_adapter.h) and the Task-6 harness (tests/test_policy_scenarios.cpp) share it.
#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>

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
//   DENSITY (nats per metre, in the convention extra_fn and F1 share — GridPlanner::body_risk_at, Navigator::route_surprisal:
//   one independent body placement per `placement_m` of travel): R(e) q(s) placement_m / (front + back), so that a crossing
//   at offset e accrues exactly R(e) nats in F1's reading and residual_risk_weight x R(e) x placement_m metres in the field.
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

// The per-metre density at p (nats per metre, before residual_risk_weight); 0 outside the corridor.
inline float doorway_jamb_density(const DoorwayJambs& j, const Eigen::Vector2f& p)
{
    if (not in_doorway_corridor(j.gap, p, j.corridor_back_m, j.corridor_ahead_m, j.inscribed_radius_m)) return 0.f;
    const float span = j.front_m + j.back_m;
    if (not (span > 0.f)) return 0.f;
    const Eigen::Vector2f d = p - j.gap.centre, lat(-j.gap.normal.y(), j.gap.normal.x());
    const double s = d.dot(j.gap.normal), e = d.dot(lat);
    const double sn = std::sqrt(double(j.sigma_n_m) * j.sigma_n_m + double(j.sigma_s_m) * j.sigma_s_m);
    const double q = sn > 0.0 ? detail::normal_interval((s + j.front_m) / sn, (s - j.back_m) / sn)
                              : ((s >= -j.front_m and s <= j.back_m) ? 1.0 : 0.0);
    if (not (q > 0.0)) return 0.f;
    return float(q * doorway_jamb_crossing_nats(j, float(e)) * j.placement_m / span);
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
