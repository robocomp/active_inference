/*
 * epistemic_planner.cpp
 *
 * AI2-native Σ-based D-optimal next-best-view for the door-concept agent.
 *
 * Only the four vertical faces (±X, ±Y in door frame) are considered because
 * a floor-navigating robot cannot observe the top face from above or the
 * bottom face at all.
 */

#include "epistemic_planner.h"
#include "door_dof.h"          // kDoorDofs: names/units (no σ* published for the door)

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

// Face sampler: synthetic samples on one vertical face of the LEAF (floor → door top), room frame.
// Geometry comes from rc::door::LeafPose (door_geometry.h), so the samples sit on the panel where it
// PHYSICALLY IS. That matters the moment phi is fitted: sampling the wall plane instead would evaluate
// predicted_information at points that are no longer on any surface, and rank the faces against a panel
// that isn't there. At phi = 0 the leaf is flush in the aperture and these are the same points as before.
std::vector<Eigen::Vector3f> sample_face_surface(const door::LeafPose& L, float floor_z, int face_idx)
{
    constexpr int kTangent = 10, kVert = 6;
    const float hw = L.half_w, hh = L.half_t;
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
        const Eigen::Vector2f r = L.centre_xy + lx * L.ex + ly * L.ey;
        const float top = 2.0f * L.half_h;
        for (int k = 0; k < kVert; ++k)
        {
            const float z = floor_z + top * ((kVert > 1) ? static_cast<float>(k) / (kVert - 1) : 1.0f);
            pts.emplace_back(r.x(), r.y(), z);
        }
    }
    return pts;
}

}  // namespace

EpistemicPlanner::EpistemicPlanner(float d_obs)
    : d_obs_(d_obs)
{}

// ── AI2-native Σ-based D-optimal next-best-view ──────────────────────────────────────────────
EpistemicProposal EpistemicPlanner::compute(const DoorBelief& belief, float lat_rate, float sigma_base,
                                            const rc::nbv::Sensor& sensor_in,
                                            const std::vector<rc::nbv::Obstacle>& obstacles,
                                            const std::vector<Eigen::Vector2f>& room_polygon,
                                            rc::nbv::Plan* plan_out) const
{
    // The camera geometry arrives from the caller (read fresh from the graph each cycle); the
    // detector envelope is ours. Merging here keeps the two concerns in their right owners.
    rc::nbv::Sensor sensor = sensor_in;
    sensor.env = det_env_;

    // Σ over the wall-frame [s,w,h]: the D-optimal score ranks which viewpoint most reduces the panel's
    // along-wall offset + size uncertainty. (No orientation entropy — the wall fixes yaw.)
    const Eigen::Matrix<float, 3, 3> S = belief.covariance_reported();   // Σ over [s,w,h]
    // Faces of the LEAF, from its actual axes — NOT reconstructed from the wall tangent. The old code
    // took its normals from belief.yaw(), which is the wall's, so once a leaf swings it would send the
    // robot to stand square to the WALL rather than to the panel it is trying to observe.
    const door::LeafPose L = belief.leaf_pose();
    const Eigen::Vector2f ctr = L.centre_xy;
    // Face order, preserved exactly from the local enumeration this replaced:
    //   0 = +x free-edge side · 1 = -x hinge side · 2 = +y front panel · 3 = -y back panel
    // ── The shared detection-weighted plan (common/nbv) ───────────────────────────────────
    // The LEAF as the belief holds it. Its axes map 1:1 onto rc::nbv::Target's: LeafPose builds
    // ey = (-ex.y, ex.x) (door_geometry.h:133), which is exactly Target::axis_y(), so the [+x,-x,+y,-y] face
    // order is preserved. w = the leaf WIDTH, h = its THICKNESS (a few cm — the ±x edge faces are slivers,
    // and the D-optimal gain already demotes them because an edge view constrains [s,w,h] barely at all).
    rc::nbv::Target target;
    target.cx = ctr.x(); target.cy = ctr.y(); target.yaw = L.yaw();
    target.w  = 2.0f * L.half_w; target.h = 2.0f * L.half_t;
    target.z0 = belief.cz(); target.z1 = belief.cz() + 2.0f * L.half_h;
    target.sigma_pos_m    = std::sqrt(std::max(0.0f, S(0, 0)));   // s — the along-wall offset
    target.sigma_extent_m = std::sqrt(std::max({0.0f, S(1, 1), S(2, 2)}));   // w, h

    const Eigen::Matrix<float, 3, 3> I3 = Eigen::Matrix<float, 3, 3>::Identity();
    const auto raw_gain_of = [&](int i, float standoff)
    {
        const float Ri = sigma_base * sigma_base + (lat_rate * standoff) * (lat_rate * standoff);
        const auto  dI = belief.predicted_information(sample_face_surface(L, belief.cz(), i), Ri);
        return 0.5f * std::log(std::max(1e-9f, (I3 + S * dI).determinant()));
    };

    const rc::nbv::Plan plan = rc::nbv::plan_faces(target, sensor, robot_radius_m_, obstacles, raw_gain_of,
                                                   room_polygon);
    if (plan_out != nullptr)
        *plan_out = plan;          // monitoring: the caller logs the whole decision, not just its result
    if (not plan.valid)
        return {};
    // ★NO USABLE FACE ⇒ REFUSE, never publish the hint. plan.best_pos is then the raw argmax, and for a thin
    // panel that is an EDGE face whose pose lies IN the wall. Publishing it hands the controller an unroutable
    // standpoint, which its repair snaps to the nearest reachable cell — the floor at the door itself. The
    // robot then drives into the leaf it was supposed to photograph. An invalid proposal makes the caller
    // hold_offered() instead, which is the honest answer: "no viewpoint from here".
    if (not plan.any_usable)
    {
        static int shouted = 0;
        if (shouted++ < 5)
            std::print("door_concept: [NBV] no usable face — every viewpoint is inside an obstacle or outside "
                       "the room. REFUSING (the raw argmax would stand in the wall).\n");
        return {};
    }
    const int   best_idx  = plan.best_face;
    const float best_raw  = plan.face_raw_gains[best_idx];
    // ★BOUND THE GAIN BY THE ADEQUACY GAP — but only if a consumer has actually stated one.
    // Information beyond σ* is worthless to the consumer, so an adequate object must STOP attracting;
    // without this an agent asks to be visited forever and starves every other affordance (measured: the
    // chair, whose gain never fell because its dominant uncertainty is discrete). The guard matters as
    // much as the bound: adequacy_gap_nats() returns 0 for a DOF table with no σ* anywhere — an empty sum
    // — and 0 is exactly the value meaning "adequate", so applying it unguarded would force the gain to
    // zero and the object would never be selected at all. Ask first. Agents that DECLARE no consumer
    // demand (SIGMA-STAR: none) therefore stay unbounded, which is coherent: with no demand, "adequate"
    // is undefined. This activates by itself the moment a σ* is published.
    float best_gain = plan.face_gains[best_idx];   // DETECTION-WEIGHTED
    if (rc::any_sigma_star(rc::kDoorDofs))
    {
        const float adequacy_gap = rc::adequacy_gap_nats(rc::kDoorDofs, [&](std::size_t j) { return S(j, j); });
        const float useful_frac  = (best_raw > 1e-9f) ? std::min(best_raw, adequacy_gap) / best_raw : 0.0f;
        best_gain *= useful_frac;
    }
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
        // `exp` vs `raw`: a large gap says the best-informing face is one the detector cannot fire from.
        // `vis` is the column that shows the WALL doing its job: with the room polygon in the obstacle set the
        // two EDGE faces (+x/-x) must collapse toward 0 — their sightline runs along the wall — while the panel
        // faces (+y/-y) stay near 1 through the aperture. Before the walls were passed, all four read ~1 and
        // the winner was `face=-x`: a viewpoint 3.67 m along the wall, i.e. the affordance driving into it.
        std::print("[epistemic-NBV] face={} gain={:.3f} | Σ dom-unc={} σ={:.3f}{} | d={:.2f}m fill*={:.2f} | "
                   "exp {:.2f}/{:.2f}/{:.2f}/{:.2f} | raw {:.2f}/{:.2f}/{:.2f}/{:.2f} | "
                   "pdet {:.2f}/{:.2f}/{:.2f}/{:.2f} | vis {:.2f}/{:.2f}/{:.2f}/{:.2f}\n",
                   fn[best_idx], best_gain, kDoorDofs[dom].name,
                   std::sqrt(std::max(0.0f, S(dom, dom))), kDoorDofs[dom].unit,
                   plan.best_standoff_m, plan.framing_fill,
                   face_gain[0], face_gain[1], face_gain[2], face_gain[3],
                   plan.face_raw_gains[0], plan.face_raw_gains[1], plan.face_raw_gains[2], plan.face_raw_gains[3],
                   plan.face_p_detect[0], plan.face_p_detect[1], plan.face_p_detect[2], plan.face_p_detect[3],
                   plan.face_visible[0], plan.face_visible[1], plan.face_visible[2], plan.face_visible[3]);
    }

    EpistemicProposal proposal{plan.best_pos.x(), plan.best_pos.y(), plan.best_yaw, best_gain, true};
    if (!proposal.is_finite())
        return {};
    return proposal;
}

}  // namespace rc
