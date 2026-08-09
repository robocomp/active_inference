/*
 * epistemic_planner.cpp
 *
 * AI2-native Σ-based D-optimal next-best-view for the chair-concept agent.
 *
 * Only the four vertical faces (±X, ±Y in chair frame) are considered because
 * a floor-navigating robot cannot observe the top face from above or the
 * bottom face at all.
 */

#include "epistemic_planner.h"
#include "chair_dof.h"          // kChairDofs: names/units (no σ* published for the chair)

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <print>
#include <vector>

// ─── EpistemicPlanner ────────────────────────────────────────────────────────


namespace rc {

namespace
{

// AI2 belief-state face sampler: synthetic samples on a vertical face plane (face f, floor → chair
// top), room frame. Dims come from the fixed template; only the pose (cx,cy,yaw) is a DOF.
// Room-frame bearing FROM THE CHAIR toward a viewpoint placed off face `face_idx`. This is the quantity
// ChairBelief bins its mode-evidence budget on (the observer's bearing as seen from the object), so the
// planner must compute it the same way or the novelty it reads back would be for the wrong bin. Face order
// is the repo-wide [+x,-x,+y,-y] object-frame convention, rotated into the room by the chair's yaw.
inline float face_view_azimuth(const ChairBelief& belief, int face_idx)
{
    const float yaw = belief.state().yaw;
    float local = 0.0f;                                   // outward normal of the face, object frame
    switch (face_idx)
    {
        case 0: local = 0.0f;                       break;   // +x
        case 1: local = static_cast<float>(M_PI);   break;   // -x
        case 2: local = static_cast<float>(M_PI_2); break;   // +y
        default: local = -static_cast<float>(M_PI_2);        // -y
    }
    const float a = yaw + local;
    return std::atan2(std::sin(a), std::cos(a));          // wrapped to (-pi, pi]
}

std::vector<Eigen::Vector3f> sample_face_surface(const ChairBelief& belief, int face_idx)
{
    constexpr int kTangent = 10, kVert = 6;
    const auto& s = belief.state();
    const float hw = belief.seat_w() * 0.5f, hh = belief.seat_d() * 0.5f;
    const float cyaw = std::cos(s.yaw), syaw = std::sin(s.yaw);
    std::vector<Eigen::Vector3f> pts; pts.reserve(kTangent * kVert);
    for (int m = 0; m < kTangent; ++m)
    {
        const float t = (kTangent > 1) ? (-1.0f + 2.0f * m / (kTangent - 1)) : 0.0f;
        float lx = 0.0f, ly = 0.0f;
        switch (face_idx)
        {
            case 0: lx =  hw;     ly =  t * hh; break;
            case 1: lx = -hw;     ly =  t * hh; break;
            case 2: lx =  t * hw; ly =  hh;     break;
            default: lx = t * hw; ly = -hh;     break;
        }
        const float rx = s.cx + cyaw * lx - syaw * ly;
        const float ry = s.cy + syaw * lx + cyaw * ly;
        const float top = belief.seat_h() + belief.back_h();
        for (int k = 0; k < kVert; ++k)
        {
            const float z = belief.cz() + top * ((kVert > 1) ? static_cast<float>(k) / (kVert - 1) : 1.0f);
            pts.emplace_back(rx, ry, z);
        }
    }
    return pts;
}

}  // namespace

EpistemicPlanner::EpistemicPlanner(float d_obs)
    : d_obs_(d_obs)
{}

// ── AI2-native Σ-based D-optimal next-best-view ──────────────────────────────────────────────
EpistemicProposal EpistemicPlanner::compute(const ChairBelief& belief, float lat_rate, float sigma_base,
                                            const rc::nbv::Sensor& sensor_in,
                                            const std::vector<rc::nbv::Obstacle>& obstacles,
                                            const std::vector<Eigen::Vector2f>& room_polygon) const
{
    // The camera geometry arrives from the caller (read fresh from the graph each cycle); the
    // detector envelope is ours. Merging here keeps the two concerns in their right owners.
    rc::nbv::Sensor sensor = sensor_in;
    sensor.env = det_env_;

    // REPORTED Σ: the yaw term folds in the discrete orientation-mode entropy, so a mode-ambiguous (side-on)
    // chair scores a backrest-revealing face highly and the orbit resolves the ambiguity. (Raw Σ would hide it.)
    const Eigen::Matrix<float, 3, 3> S = belief.covariance_reported();   // Σ over [cx,cy,yaw]
    const ChairBeliefState& s = belief.state();

    // The chair as the BELIEF holds it. The extent has NO uncertainty here — the chair model uses a fixed
    // template and only the pose (cx,cy,yaw) is a DOF — so sigma_extent_m stays 0 and only the range term
    // widens the framing. Honest by construction rather than by omission.
    rc::nbv::Target target;
    target.cx = s.cx; target.cy = s.cy; target.yaw = s.yaw;
    target.w  = belief.seat_w(); target.h = belief.seat_d();
    target.z0 = belief.cz();
    target.z1 = belief.cz() + belief.seat_h() + belief.back_h();   // the BACKREST is what the camera sees
    target.sigma_pos_m = std::sqrt(std::max({0.0f, S(0, 0), S(1, 1)}));

    const Eigen::Matrix<float, 3, 3> I3 = Eigen::Matrix<float, 3, 3>::Identity();
    const auto raw_gain_of = [&](int i, float standoff)
    {
        const float Ri = sigma_base * sigma_base + (lat_rate * standoff) * (lat_rate * standoff);
        const auto  dI = belief.predicted_information(sample_face_surface(belief, i), Ri);
        const float continuous = 0.5f * std::log(std::max(1e-9f, (I3 + S * dI).determinant()));

        // ★ADD THE DISCRETE TERM. Sigma_ carries only the WITHIN-mode yaw width (~1 deg), so a gain built
        // from it alone says a chair is nearly resolved while the belief still does not know which of four
        // ways it faces — the dominant uncertainty, and the one the reported covariance actually shows
        // (std_yaw_rep ~0.64 rad while std_yaw is 0.049). Resolving the mode is worth mode_entropy() nats.
        //
        // But only a look from an UNSPENT bearing can collect it: the accumulator charges each view-azimuth
        // bin a budget and weights new frames by novelty = remaining/(remaining+budget), so a frame from an
        // exhausted bearing contributes ~nothing however long the robot stares. Weighting the discrete term
        // by that SAME novelty is what breaks the loop where the planner kept re-proposing a bearing it had
        // already spent — a completed visit that resolved nothing, and a chair that was therefore
        // re-selected forever while every other object starved.
        //
        // The bearing this candidate would observe from is the face normal's OUTWARD direction, i.e. the
        // object-to-viewpoint bearing, which is what the accumulator bins on.
        const float az = face_view_azimuth(belief, i);
        const float discrete = belief.mode_entropy() * belief.view_novelty(az);
        return continuous + discrete;
    };

    const rc::nbv::Plan plan = rc::nbv::plan_faces(target, sensor, robot_radius_m_, obstacles, raw_gain_of,
                                                   room_polygon);
    if (not plan.valid)
        return {};
    // ★NO USABLE FACE ⇒ REFUSE, never publish the hint. plan.best_pos is then the raw argmax, which for a
    // wall-anchored object is a pose behind the wall. Publishing it hands the controller an unroutable
    // standpoint, and the controller REPAIRS rather than rejects: nearest_reachable is measured from the
    // robot, so the goal snaps to the nearest occupiable cell — the floor right at the object. An invalid
    // proposal makes the caller hold_offered() instead: "no viewpoint from here", which is the honest answer.
    if (not plan.any_usable)
    {
        static int shouted = 0;
        if (shouted++ < 5)
            std::print("chair_concept: [NBV] no usable face — every viewpoint is inside an obstacle or "
                       "outside the room. REFUSING (the raw argmax would stand in the wall).\n");
        return {};
    }
    const int   best_idx  = plan.best_face;
    const float best_gain = plan.face_gains[best_idx];   // DETECTION-WEIGHTED
    const auto& face_gain = plan.face_gains;
    if (not std::isfinite(best_gain))
        return {};

    static int dbg = 0;
    if (++dbg % 30 == 0)
    {
        constexpr float kRef = 0.10f;   // common scale: 10 cm / 0.1 rad
        static const char* fn[4] = {"+x", "-x", "+y", "-y"};
        int dom = 0; float best = -1.0f;
        for (int j = 0; j < S.rows(); ++j)
        {
            const float n = std::sqrt(std::max(0.0f, S(j, j))) / kRef;
            if (n > best) { best = n; dom = j; }
        }
        // `exp` vs `raw`: a large gap says the best-informing face is one the detector cannot fire from —
        // exactly the failure the weighting exists to catch, so both columns are printed.
        std::print("[epistemic-NBV] face={} gain={:.3f} | Σ dom-unc={} σ={:.3f}{} | d={:.2f}m fill*={:.2f} | "
                   "exp +x={:.2f} -x={:.2f} +y={:.2f} -y={:.2f} | raw {:.2f}/{:.2f}/{:.2f}/{:.2f} | "
                   "pdet {:.2f}/{:.2f}/{:.2f}/{:.2f}\n",
                   fn[best_idx], best_gain, kChairDofs[dom].name,
                   std::sqrt(std::max(0.0f, S(dom, dom))), kChairDofs[dom].unit,
                   plan.best_standoff_m, plan.framing_fill,
                   face_gain[0], face_gain[1], face_gain[2], face_gain[3],
                   plan.face_raw_gains[0], plan.face_raw_gains[1], plan.face_raw_gains[2], plan.face_raw_gains[3],
                   plan.face_p_detect[0], plan.face_p_detect[1], plan.face_p_detect[2], plan.face_p_detect[3]);
    }

    EpistemicProposal proposal{plan.best_pos.x(), plan.best_pos.y(), plan.best_yaw, best_gain, true};
    if (!proposal.is_finite())
        return {};
    return proposal;
}

}  // namespace rc
