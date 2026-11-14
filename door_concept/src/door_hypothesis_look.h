// door_hypothesis_look.h — THE HYPOTHESIS LOOK: the pose that most reduces the uncertainty there is NOW (one-model spec §4). Header-only, DSR-free.
//
// What the robot does not know is not "does door X exist" one door at a time: it is WHICH of a few joint hypotheses holds (the leaf
// is the side-wall door's, or the front wall's, or nobody's; a remembered door is there or it is not). A hypothesis h says which
// apertures exist and the leaf state of each; a QUESTION is a set of mutually exclusive hypotheses with weights. A look from a pose
// z will report, per aperture that any hypothesis mentions, two things — does the camera report a door there, does the LiDAR read
// THROUGH the span — and the pose that most reduces the entropy of the weights, in expectation, is the one to go to:
//
//     EIG(pose) = H(w) − Σ_z p(z | pose) H(w | z)         (bf::expected_information_gain, over the joint outcomes z)
//
// So the best pose DEPENDS ON THE SITUATION: with two hypotheses that differ in where an opening is, it is the pose from which the
// two predict different readings (one sees through, the other sees wall); with one place {exists, not} it is the pose that sees the
// place best. There is no recipe (face the door, stand 1.5 m off): the walls, the furniture and the other hypotheses' leaves enter the
// prediction as occluders, so the pose falls out of the model.
//
// The outcome model is the SAME one the existence update scores the data with (so the look predicts what the sensors really report):
//   camera, aperture PRESENT   P(report) = front_end(pose, occluders of h) · ((1 − p_open) h_shut + p_open h_open)   [detector_llr.h with_leaf_state]
//   camera, aperture ABSENT    P(report) = false_alarm · front_end(pose, occluders of h)
//   LiDAR,  aperture PRESENT   P(through) = f_los · (p_open · through_open + (1 − p_open) · through_wall)
//   LiDAR,  aperture ABSENT    P(through) = f_los · through_wall                                    (etc/door_detectors_fitted.csv rows)
// front_end = rc::nbv::score's p_detect (detector envelope on the projected fill × the visible fraction), the walls (the room polygon,
// the apertures PRESENT under h punched out), the furniture and h's other leaves as occluders; f_los = the share of the span points
// the LiDAR has line of sight to (same occluders). p_open = 1 − cos φ, the share of the span a leaf at φ clears (door_opening_edges.h
// clear_fraction). The sensors are conditionally independent given h.
//
// Pure (Eigen + std + common headers): tests/test_door_hypothesis_look.cpp.
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numbers>
#include <set>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "core/bernoulli_filter.h"
#include "core/detector_llr.h"
#include "door_geometry.h"
#include "../../common/nbv/viewpoint_score.h"

namespace rc::door::hlook
{

/// One joint hypothesis: the apertures that exist under it (each with its leaf's state), and its weight. {} = "none of them" (the null).
struct Hyp
{
    std::vector<Aperture> apertures;
    std::vector<LeafState> leaves;
    double w = 0.0;
};

/// A cluster of mutually exclusive hypotheses; a single undecided place is {exists, not}.
struct Question
{
    std::string key;
    std::vector<Hyp> hyps;
};

struct Pose
{
    Eigen::Vector2f xy{0.f, 0.f};
    float yaw = 0.f;
};

/// Everything the sensors and the world supply. `polygon` / `furniture` must outlive the call.
struct Model
{
    rc::nbv::Sensor camera;                          // the ZED (intrinsics + mount height + detector envelope)
    std::span<const Eigen::Vector2f> polygon;        // the room outline
    std::span<const rc::nbv::Obstacle> furniture;    // graph objects
    core::DetectorModel zed;                         // hit_rate (shut), hit_rate_open, false_alarm
    core::LidarThroughModel lidar;                   // through_open, through_wall
    float wall_thickness_m = 0.05f;                  // the depth of material an aperture is cut through (nbv::wall_obstacles)
    float robot_radius_m = 0.30f;
};

struct Best
{
    Pose pose;
    double eig_nats = 0.0;
    bool valid = false;
};

/// The weights normalised to a distribution (a question whose weights are all 0 has none).
[[nodiscard]] inline std::vector<double> weights_of(const Question& q)
{
    std::vector<double> w;
    w.reserve(q.hyps.size());
    double z = 0.0;
    for (const auto& h : q.hyps) z += std::max(0.0, h.w);
    for (const auto& h : q.hyps) w.push_back(z > 0.0 ? std::max(0.0, h.w) / z : 0.0);
    return w;
}

/// A hypothesis weighing less than this is not one any decision the costs make would change for, and an EIG that leaves it out is off by at most
/// kWeightFloor·ln H nats. kept: a numerical floor (not a gate on the look): it lets a SETTLED question go (the gain falling to ~0 ranks it last
/// but a node is only withdrawn on a yes/no) and keeps the outcome table small (4^slots outcomes only for the slots the live hypotheses mention).
inline constexpr double kWeightFloor = 1e-3;

/// The question is SETTLED when at most one hypothesis keeps weight above kWeightFloor.
[[nodiscard]] inline bool settled(const Question& q)
{
    const auto w = weights_of(q);
    return std::ranges::count_if(w, [](double v) { return v > kWeightFloor; }) <= 1;
}

/// The question with the hypotheses below kWeightFloor left out (weights unchanged; weights_of renormalises).
[[nodiscard]] inline Question live_part(const Question& q)
{
    Question out{q.key, {}};
    const auto w = weights_of(q);
    for (std::size_t h = 0; h < q.hyps.size(); ++h)
        if (w[h] >= kWeightFloor) out.hyps.push_back(q.hyps[h]);
    return out;
}

[[nodiscard]] inline bool same_aperture(const Aperture& a, const Aperture& b)
{
    return a.wall_O == b.wall_O and a.wall_u == b.wall_u and a.s == b.s and a.w == b.w and a.depth == b.depth;
}

/// Reachable poses over the room floor: a grid with step = the robot's diameter and 8 headings (a sampling resolution, not a decision
/// threshold), kept if reachable (inside the polygon with the robot's body clear of the wall) and not standing inside an obstacle.
[[nodiscard]] inline std::vector<Pose> room_poses(std::span<const Eigen::Vector2f> polygon, std::span<const rc::nbv::Obstacle> obstacles,
                                                  float robot_radius_m)
{
    std::vector<Pose> out;
    if (polygon.size() < 3 or not (robot_radius_m > 0.f)) return out;
    Eigen::Vector2f lo = polygon[0], hi = polygon[0];
    for (const auto& p : polygon) { lo = lo.cwiseMin(p); hi = hi.cwiseMax(p); }
    const float step = 2.f * robot_radius_m;
    constexpr int kHeadings = 8;
    for (float y = lo.y() + 0.5f * step; y < hi.y(); y += step)
        for (float x = lo.x() + 0.5f * step; x < hi.x(); x += step)
        {
            const Eigen::Vector2f p(x, y);
            if (not rc::nbv::is_reachable(p, polygon, robot_radius_m) or rc::nbv::stands_inside(p, obstacles, robot_radius_m)) continue;
            for (int k = 0; k < kHeadings; ++k)
                out.push_back({p, 2.f * std::numbers::pi_v<float> * static_cast<float>(k) / kHeadings - std::numbers::pi_v<float>});
        }
    return out;
}

/// The occluder box of a leaf (its panel, standing at its angle) in room frame.
[[nodiscard]] inline rc::nbv::Obstacle leaf_obstacle(const Aperture& ap, const LeafState& ls)
{
    const LeafPose L = leaf_pose(ap, ls);
    return {L.centre_xy.x(), L.centre_xy.y(), 2.f * L.half_w, 2.f * L.half_t, L.yaw(), ap.floor_z, ap.floor_z + ap.h};
}

/// The share of the aperture's span (kSamples points along its wall line) the LiDAR at `from` has line of sight to.
/// The points sit in the gap the caller punched for the span, so they are not inside any wall box.
[[nodiscard]] inline float lidar_los(const Eigen::Vector2f& from, const Aperture& ap, std::span<const rc::nbv::Obstacle> occluders)
{
    constexpr int kSamples = 8;   // quadrature along the span, a sampling resolution
    int seen = 0;
    for (int k = 0; k < kSamples; ++k)
    {
        const Eigen::Vector2f p = ap.wall_O + (ap.s + (static_cast<float>(k) + 0.5f) / kSamples * ap.w) * ap.wall_u;
        bool blocked = false;
        for (const auto& o : occluders)
            if (rc::nbv::detail::segment_hits_box(from, p, o)) { blocked = true; break; }
        if (not blocked) ++seen;
    }
    return static_cast<float>(seen) / kSamples;
}

/// What a question needs from a pose, built once per question: the holes (the distinct apertures any hypothesis mentions), who holds
/// which, and the occluders each hypothesis sees each slot through.
struct Scene
{
    std::vector<Aperture> holes;
    std::vector<rc::nbv::Target> targets;                       // per slot: the aperture-sized leaf flush in its wall (as look::plan's)
    std::vector<std::vector<int>> present;                      // [h][slot] -> index into hyp.apertures, -1 absent
    std::vector<std::vector<float>> p_open;                     // [h][slot]
    std::vector<std::vector<std::vector<rc::nbv::Obstacle>>> occ;   // [h][slot] occluders (walls with present apertures + the slot's own punched out, furniture, h's other leaves)
};

[[nodiscard]] inline Scene make_scene(const Question& q, const Model& m)
{
    Scene sc;
    for (const auto& h : q.hyps)
        for (const auto& a : h.apertures)
            if (std::ranges::none_of(sc.holes, [&](const Aperture& s) { return same_aperture(s, a); })) sc.holes.push_back(a);
    const std::size_t H = q.hyps.size(), S = sc.holes.size();
    for (const auto& a : sc.holes)
    {
        rc::nbv::Target t;
        const Eigen::Vector2f c = a.centre_xy();
        t.cx = c.x(); t.cy = c.y(); t.yaw = a.yaw(); t.w = a.w; t.h = std::max(1e-3f, a.thickness);
        t.z0 = a.floor_z; t.z1 = a.floor_z + a.h;
        sc.targets.push_back(t);
    }
    sc.present.assign(H, std::vector<int>(S, -1));
    sc.p_open.assign(H, std::vector<float>(S, 0.f));
    sc.occ.assign(H, std::vector<std::vector<rc::nbv::Obstacle>>(S));
    for (std::size_t h = 0; h < H; ++h)
    {
        for (std::size_t s = 0; s < S; ++s)
            for (std::size_t k = 0; k < q.hyps[h].apertures.size(); ++k)
                if (same_aperture(sc.holes[s], q.hyps[h].apertures[k]))
                {
                    sc.present[h][s] = static_cast<int>(k);
                    sc.p_open[h][s] = std::clamp(1.f - std::cos(q.hyps[h].leaves[k].phi), 0.f, 1.f);   // clamp: a probability
                }
        for (std::size_t s = 0; s < S; ++s)
        {
            std::vector<rc::nbv::WallGap> gaps;
            for (std::size_t t = 0; t < S; ++t)
                if (t == s or sc.present[h][t] >= 0) gaps.push_back({sc.holes[t].centre_xy(), 0.5f * sc.holes[t].w});
            auto o = rc::nbv::wall_obstacles(m.polygon, std::max(1e-3f, m.wall_thickness_m), gaps);
            o.insert(o.end(), m.furniture.begin(), m.furniture.end());
            for (std::size_t t = 0; t < S; ++t)
                if (t != s and sc.present[h][t] >= 0)
                    o.push_back(leaf_obstacle(sc.holes[t], q.hyps[h].leaves[static_cast<std::size_t>(sc.present[h][t])]));
            sc.occ[h][s] = std::move(o);
        }
    }
    return sc;
}

namespace detail
{
/// The LiDAR's line of sight to every slot's span under every hypothesis, from a position (it does not depend on the heading: the LiDAR is 360 degrees).
inline std::vector<std::vector<float>> los_at(const Question& q, const Scene& sc, const Eigen::Vector2f& xy)
{
    std::vector<std::vector<float>> los(q.hyps.size(), std::vector<float>(sc.holes.size(), 0.f));
    for (std::size_t h = 0; h < q.hyps.size(); ++h)
        for (std::size_t s = 0; s < sc.holes.size(); ++s)
            los[h][s] = lidar_los(xy, sc.holes[s], sc.occ[h][s]);
    return los;
}

/// Can ANY point of the slot's footprint be inside the camera's horizontal field of view? False only when all four corners lie on one side beyond
/// the half-angle (the sight lines to the perimeter then all lie outside it, so the visible fraction is exactly 0 and the detector's p is 0).
inline bool in_view_possible(const rc::nbv::Target& t, const Pose& pose, const rc::nbv::Sensor& cam)
{
    const float half = 0.5f * std::clamp(cam.hfov_rad, 0.1f, 3.0f);
    int pos_side = 0, neg_side = 0;
    for (const float sx : {-0.5f, 0.5f})
        for (const float sy : {-0.5f, 0.5f})
        {
            const Eigen::Vector2f p = t.centre() + sx * t.w * t.axis_x() + sy * t.h * t.axis_y();
            const Eigen::Vector2f rel = p - pose.xy;
            float b = std::atan2(rel.y(), rel.x()) - pose.yaw;
            while (b > std::numbers::pi_v<float>) b -= 2.f * std::numbers::pi_v<float>;
            while (b < -std::numbers::pi_v<float>) b += 2.f * std::numbers::pi_v<float>;
            if (b > half) ++pos_side;
            else if (b < -half) ++neg_side;
        }
    return not (pos_side == 4 or neg_side == 4);
}

/// Per hypothesis and slot, the four outcome probabilities {neither, lidar only, camera only, both} (o = 2 camera_yes + lidar_yes). `los[h][s]` from los_at.
inline std::vector<std::vector<std::array<double, 4>>> slot_probs(const Question& q, const Scene& sc, const Model& m, const Pose& pose,
                                                                  const std::vector<std::vector<float>>& los)
{
    const std::size_t H = q.hyps.size(), S = sc.holes.size();
    std::vector<std::vector<std::array<double, 4>>> ps(H, std::vector<std::array<double, 4>>(S));
    for (std::size_t h = 0; h < H; ++h)
        for (std::size_t s = 0; s < S; ++s)
        {
            const bool present = sc.present[h][s] >= 0;
            const float po = sc.p_open[h][s];
            double front = 0.0;
            if (m.camera.complete() and in_view_possible(sc.targets[s], pose, m.camera))
                front = rc::nbv::score({pose.xy, pose.yaw, 1.f, -1}, sc.targets[s], m.camera, sc.occ[h][s], 0.f).p_detect;
            const double h_state = core::with_leaf_state(m.zed, present ? po : 0.f).hit_rate;
            const double pc = std::clamp(present ? front * h_state : front * m.zed.false_alarm, 0.0, 1.0);   // clamp: a probability
            const double through = present ? double(po) * m.lidar.through_open + (1.0 - double(po)) * m.lidar.through_wall : double(m.lidar.through_wall);
            const double pl = std::clamp(double(los[h][s]) * through, 0.0, 1.0);
            ps[h][s] = {(1 - pc) * (1 - pl), (1 - pc) * pl, pc * (1 - pl), pc * pl};
        }
    return ps;
}

/// The joint outcome table over the slots, p(z | h, pose), z = Σ_s o_s 4^s. EXACT: 4^slots columns.
inline std::vector<std::vector<double>> table(const Question& q, const Scene& sc, const Model& m, const Pose& pose,
                                              const std::vector<std::vector<float>>& los)
{
    const std::size_t H = q.hyps.size(), S = sc.holes.size();
    std::size_t Z = 1;
    for (std::size_t s = 0; s < S; ++s) Z *= 4;
    const auto ps = slot_probs(q, sc, m, pose, los);
    std::vector<std::vector<double>> out(H, std::vector<double>(Z, 0.0));
    for (std::size_t h = 0; h < H; ++h)
        for (std::size_t z = 0; z < Z; ++z)
        {
            double p = 1.0;
            std::size_t r = z;
            for (std::size_t s = 0; s < S; ++s, r /= 4) p *= ps[h][s][r % 4];
            out[h][z] = p;
        }
    return out;
}

/// Sampling resolution of the Monte-Carlo EIG: outcomes drawn per hypothesis. The same uniform stream serves every pose (common random numbers), so
/// the estimate is smooth across poses and the choice between them is not decided by sampling noise.
inline constexpr int kOutcomeSamples = 256;

/// EIG = Σ_h w_h E_{z ~ p(.|h)} [ln p(z|h) − ln p(z)] by sampling z; nothing is enumerated, so the cost is H·samples·H·S whatever 4^S is.
inline double eig_sampled(std::span<const double> w, const std::vector<std::vector<std::array<double, 4>>>& ps)
{
    const std::size_t H = ps.size(), S = H ? ps[0].size() : 0;
    double eig = 0.0;
    std::vector<double> lp(H);
    for (std::size_t h = 0; h < H; ++h)
    {
        if (not (w[h] > 0.0)) continue;
        std::uint64_t st = 0x9E3779B97F4A7C15ull * (h + 1);
        double acc = 0.0;
        for (int n = 0; n < kOutcomeSamples; ++n)
        {
            std::vector<int> o(S);
            for (std::size_t s = 0; s < S; ++s)
            {
                st ^= st << 13; st ^= st >> 7; st ^= st << 17;   // xorshift64
                const double u = static_cast<double>(st >> 11) * (1.0 / 9007199254740992.0);
                double c = 0.0;
                int k = 0;
                for (; k < 3; ++k) { c += ps[h][s][static_cast<std::size_t>(k)]; if (u < c) break; }
                o[s] = k;
            }
            double mx = -std::numeric_limits<double>::infinity();
            for (std::size_t g = 0; g < H; ++g)
            {
                double l = 0.0;
                for (std::size_t s = 0; s < S; ++s) l += std::log(ps[g][s][static_cast<std::size_t>(o[s])]);   // log 0 = -inf: an outcome g forbids
                lp[g] = l;
                if (w[g] > 0.0) mx = std::max(mx, l + std::log(w[g]));
            }
            double z = 0.0;
            for (std::size_t g = 0; g < H; ++g) z += w[g] > 0.0 ? std::exp(lp[g] + std::log(w[g]) - mx) : 0.0;
            acc += lp[h] - (mx + std::log(z));
        }
        eig += w[h] * acc / kOutcomeSamples;
    }
    return std::max(0.0, eig);
}
}   // namespace detail

/// p(z | h, pose) over the discrete outcomes {each slot: camera reports a door there or not} × {the LiDAR reads through its span or not}.
/// Rows are the question's hypotheses, columns the joint outcomes (see detail::table).
[[nodiscard]] inline std::vector<std::vector<double>> outcome_table(const Question& q, const Pose& pose, const Model& m)
{
    const Scene sc = make_scene(q, m);
    return detail::table(q, sc, m, pose, detail::los_at(q, sc, pose.xy));
}

/// resource budget, not a model term: the exact outcome tables of a question (poses x hypotheses x 4^slots doubles) are kept while they fit this many
/// cells (4 M = 32 MB); beyond it the EIG is estimated per pose from sampled outcomes and nothing is cached.
inline constexpr double kTableBudgetCells = 4.0e6;

/// The outcome tables of every pose of a question, memoised across cycles. The tables hold the GEOMETRY of the question (what each hypothesis
/// predicts from each pose); the weights, which move every cycle, enter only in the EIG. So while the hypotheses' geometry, the room and the
/// pose list are the same to a sampling resolution (1 cm and 0.02 rad for the geometry; the poses to the bit) the tables are not recomputed. Not a
/// gate on the look: a miss recomputes them exactly. Only exact tables within kTableBudgetCells are held.
struct LookCache
{
    std::vector<std::int32_t> sig;
    std::vector<std::vector<std::vector<double>>> tab;   // [pose][h][z]
    bool hit = false;                                    // the last call reused the tables
    bool sampled = false;                                // the last call was over budget: EIG by sampling, nothing held
};

namespace detail
{
[[nodiscard]] inline std::vector<std::int32_t> signature(const Question& q, std::span<const Pose> poses, const Model& m)
{
    std::vector<std::int32_t> g;
    const auto len = [&](float v) { g.push_back(static_cast<std::int32_t>(std::lround(v * 100.f))); };   // 1 cm
    const auto ang = [&](float v) { g.push_back(static_cast<std::int32_t>(std::lround(v * 50.f))); };    // 0.02 rad
    g.push_back(static_cast<std::int32_t>(q.hyps.size()));
    for (const auto& h : q.hyps)
    {
        g.push_back(static_cast<std::int32_t>(h.apertures.size()));
        for (std::size_t k = 0; k < h.apertures.size(); ++k)
        {
            const Aperture& a = h.apertures[k];
            len(a.wall_O.x()); len(a.wall_O.y()); ang(a.wall_u.x()); ang(a.wall_u.y()); len(a.s); len(a.w); len(a.h); len(a.depth); len(a.thickness);
            ang(h.leaves[k].phi); g.push_back(h.leaves[k].hinge == HingeSide::Near ? 0 : 1); g.push_back(h.leaves[k].swing > 0.f ? 1 : -1);
        }
    }
    g.push_back(static_cast<std::int32_t>(m.polygon.size()));
    for (const auto& v : m.polygon) { len(v.x()); len(v.y()); }
    g.push_back(static_cast<std::int32_t>(m.furniture.size()));
    for (const auto& o : m.furniture) { len(o.cx); len(o.cy); len(o.w); len(o.h); ang(o.yaw); }
    len(m.robot_radius_m); len(m.wall_thickness_m);
    g.push_back(static_cast<std::int32_t>(poses.size()));
    for (const auto& p : poses)   // the tables are index-aligned with the poses: every pose, to the bit
    {
        g.push_back(std::bit_cast<std::int32_t>(p.xy.x())); g.push_back(std::bit_cast<std::int32_t>(p.xy.y())); g.push_back(std::bit_cast<std::int32_t>(p.yaw));
    }
    return g;
}

/// The EIG of every pose (exact within the budget, sampled beyond it).
[[nodiscard]] inline std::vector<double> eigs(const Question& q, std::span<const Pose> poses, const Model& m, LookCache* cache)
{
    std::vector<double> out(poses.size(), 0.0);
    const auto w = weights_of(q);
    std::vector<const Aperture*> distinct;
    for (const auto& h : q.hyps)
        for (const auto& a : h.apertures)
            if (std::ranges::none_of(distinct, [&](const Aperture* d) { return same_aperture(*d, a); })) distinct.push_back(&a);
    std::size_t Z = 1;
    for (std::size_t s = 0; s < distinct.size(); ++s) Z *= 4;
    const bool exact = static_cast<double>(poses.size()) * static_cast<double>(q.hyps.size()) * static_cast<double>(Z) <= kTableBudgetCells;
    if (cache != nullptr) cache->sampled = not exact;
    if (not exact)
    {
        const Scene sc = make_scene(q, m);
        if (cache != nullptr) { cache->hit = false; cache->tab.clear(); cache->sig.clear(); }
        Eigen::Vector2f at(std::numeric_limits<float>::quiet_NaN(), 0.f);
        std::vector<std::vector<float>> los;
        for (std::size_t i = 0; i < poses.size(); ++i)
        {
            if (not (poses[i].xy == at)) { los = los_at(q, sc, poses[i].xy); at = poses[i].xy; }
            out[i] = eig_sampled(w, slot_probs(q, sc, m, poses[i], los));
        }
        return out;
    }
    std::vector<std::vector<std::vector<double>>> local;
    auto* tabs = &local;
    if (cache != nullptr)
    {
        auto sig = signature(q, poses, m);
        cache->hit = (sig == cache->sig and cache->tab.size() == poses.size());
        if (not cache->hit) { cache->sig = std::move(sig); cache->tab.clear(); }
        tabs = &cache->tab;
    }
    if (tabs->empty())
    {
        const Scene sc = make_scene(q, m);
        Eigen::Vector2f at(std::numeric_limits<float>::quiet_NaN(), 0.f);
        std::vector<std::vector<float>> los;
        tabs->reserve(poses.size());
        for (const auto& p : poses)
        {
            if (not (p.xy == at)) { los = los_at(q, sc, p.xy); at = p.xy; }
            tabs->push_back(table(q, sc, m, p, los));
        }
    }
    for (std::size_t i = 0; i < poses.size(); ++i) out[i] = core::bf::expected_information_gain(w, (*tabs)[i]);
    return out;
}
}   // namespace detail

/// The EIG of one pose, whichever way (exact table or sampled outcomes) fits the budget; for tests and callers with a single pose.
[[nodiscard]] inline double eig_at(const Question& question, const Pose& pose, const Model& m, bool force_sampled = false)
{
    const Question q = live_part(question);
    if (q.hyps.empty()) return 0.0;
    const Scene sc = make_scene(q, m);
    const auto los = detail::los_at(q, sc, pose.xy);
    return force_sampled ? detail::eig_sampled(weights_of(q), detail::slot_probs(q, sc, m, pose, los))
                         : core::bf::expected_information_gain(weights_of(q), detail::table(q, sc, m, pose, los));
}

/// The pose among `poses` with the largest expected information gain about the question (hypotheses below kWeightFloor left out; no top-K).
/// With a `standing` pose (the one last published, if it is still among `poses`) it is HELD: the result moves only when another pose's EIG, under
/// the current weights, exceeds the standing pose's EIG under the same weights; `Best::eig_nats` is then the standing pose's.
/// With a `cache` the exact tables are reused while the question's geometry is unchanged (see LookCache).
[[nodiscard]] inline Best best_look(const Question& question, std::span<const Pose> poses, const Model& m, LookCache* cache = nullptr,
                                    const Pose* standing = nullptr)
{
    Best best;
    const Question q = live_part(question);
    if (q.hyps.empty() or poses.empty()) return best;
    const auto e = detail::eigs(q, poses, m, cache);
    std::optional<std::size_t> hold;
    if (standing != nullptr)
        for (std::size_t i = 0; i < poses.size(); ++i)
            if (poses[i].xy == standing->xy and poses[i].yaw == standing->yaw) { hold = i; break; }
    std::size_t arg = 0;
    for (std::size_t i = 1; i < poses.size(); ++i)
        if (e[i] > e[arg]) arg = i;
    if (hold.has_value() and not (e[arg] > e[*hold])) arg = *hold;
    best.valid = true; best.pose = poses[arg]; best.eig_nats = e[arg];
    return best;
}

// ═══ THE QUESTIONS THE AGENT HAS (shared by the agent and the replay) ═══════════════════════════════════════════════════════════
// Production has no joint-hypothesis cluster object yet, so the questions are built from the entities that exist: a live door or a pending
// candidate is a MEMBER (aperture, leaf, existence log-odds L). A leaf-born candidate also carries the other frame-wall hypotheses of
// its mask (apertures_from_leaf's siblings) as ALTS, each with its log-weight relative to the member's own.

struct Alt { Aperture aperture; LeafState leaf; double dlogw = 0.0; };

struct Member
{
    std::string name;            // the door's node name / "place_<key>"
    Aperture aperture;
    LeafState leaf;
    double L = 0.0;              // existence log-odds
    bool leaf_born = false;      // a hypothesis that explains a side-on leaf mask: its competitors are the leaf's other frame walls
    bool restored_unseen = false;
    std::vector<Alt> alts;
};

[[nodiscard]] inline Hyp single_hyp(const Member& m, double w) { return {{m.aperture}, {m.leaf}, w}; }

/// A single undecided place: {exists with probability p, not}.
[[nodiscard]] inline Question place_question(std::string key, const Member& m, double p_exists)
{
    p_exists = std::clamp(p_exists, 0.0, 1.0);   // clamp: a probability
    return {std::move(key), {single_hyp(m, p_exists), Hyp{{}, {}, 1.0 - p_exists}}};
}

/// WHO COMPETES FOR A MASK, as a graph that lives only while it is true. Nodes: ('d', door id) or ('c', candidate key). A pair is recorded when
/// one mask puts non-negligible responsibility (rho >= kWeightFloor) on both. It lives while BOTH members are still undecided: any DECISION on a
/// member (its verdict, the code `refresh` is given, changing; or the member vanishing: removed, or a candidate born as a door) drops all that
/// member's pairs, and a pair re-forms only from a fresh mask. So clusters split as well as merge, with no time window.
struct Competition
{
    using NodeRef = std::pair<char, std::uint64_t>;
    std::set<std::pair<NodeRef, NodeRef>> pairs;
    std::map<NodeRef, int> code;   // each member's verdict code as of the last refresh

    void note(std::vector<NodeRef> nodes)
    {
        std::ranges::sort(nodes);
        nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
        for (std::size_t a = 0; a < nodes.size(); ++a)
            for (std::size_t b = a + 1; b < nodes.size(); ++b) pairs.emplace(nodes[a], nodes[b]);
    }
    /// `now`: every member that exists, with its verdict code (a change of code is a decision). Members absent from `now` are gone.
    void refresh(const std::map<NodeRef, int>& now)
    {
        std::set<NodeRef> decided;
        for (const auto& [n, c] : code)
            if (const auto it = now.find(n); it == now.end() or it->second != c) decided.insert(n);   // gone, or its verdict moved
        std::erase_if(pairs, [&](const auto& pr)
        { return decided.contains(pr.first) or decided.contains(pr.second) or not now.contains(pr.first) or not now.contains(pr.second); });
        code = now;
    }
};

/// The questions of a set of members. Two members COMPETE when some mask put non-negligible responsibility on both (door_mask_causes.h rho >= kWeightFloor); `competing` lists those pairs by member name. Connected components
/// of that graph are the clusters: members explaining one panel, of which ONE (or none) is the door (weights odds e^{L_i} against the null's 1). A
/// leaf-born member's alts join its cluster. A lone member is a single-member {exists, not} question while `undecided` says it is open. A cluster
/// stands until `settled`.
template <class Undecided>
[[nodiscard]] inline std::vector<Question> questions_from(std::span<const Member> members,
                                                          std::span<const std::pair<std::string, std::string>> competing, Undecided&& undecided)
{
    const std::size_t N = members.size();
    std::vector<std::size_t> comp(N);
    for (std::size_t i = 0; i < N; ++i) comp[i] = i;
    const auto find = [&](std::size_t i) { while (comp[i] != i) i = comp[i] = comp[comp[i]]; return i; };
    const auto index_of = [&](const std::string& n) -> std::optional<std::size_t>
    {
        for (std::size_t i = 0; i < N; ++i) if (members[i].name == n) return i;
        return std::nullopt;
    };
    for (const auto& [a, b] : competing)
        if (const auto ia = index_of(a), ib = index_of(b); ia.has_value() and ib.has_value()) comp[find(*ia)] = find(*ib);
    std::vector<Question> out;
    std::vector<bool> done(N, false);
    for (std::size_t r = 0; r < N; ++r)
    {
        if (done[r]) continue;
        std::vector<std::size_t> group;
        for (std::size_t i = 0; i < N; ++i)
            if (find(i) == find(r)) { group.push_back(i); done[i] = true; }
        const bool leafy = std::ranges::any_of(group, [&](std::size_t i) { return members[i].leaf_born; });
        if (group.size() == 1 and not undecided(members[group[0]])) continue;
        if (group.size() == 1 and not leafy)
        {
            const Member& m = members[group[0]];
            out.push_back(place_question("q_" + m.name, m, 1.0 / (1.0 + std::exp(-m.L))));
            continue;
        }
        Question q;
        std::string first = members[group[0]].name;
        for (std::size_t i : group) first = std::min(first, members[i].name);
        q.key = "c_" + first;
        q.hyps.push_back({{}, {}, 1.0});   // the null: none of them is the door
        for (std::size_t i : group)
        {
            const Member& m = members[i];
            q.hyps.push_back(single_hyp(m, std::exp(std::clamp(m.L, -30.0, 30.0))));   // clamp: exp's own range
        }
        for (std::size_t i : group)
        {
            const Member& m = members[i];
            for (const auto& a : m.alts)
            {
                // the same hole as a hypothesis already in: half a door width is the slot's own size (the seeds are one door width apart), not a tuning
                const bool dup = std::ranges::any_of(q.hyps, [&](const Hyp& h)
                {
                    return not h.apertures.empty() and (h.apertures[0].centre_xy() - a.aperture.centre_xy()).norm() < 0.5f * std::min(h.apertures[0].w, a.aperture.w);
                });
                if (dup) continue;
                q.hyps.push_back({{a.aperture}, {a.leaf}, std::exp(std::clamp(m.L + a.dlogw, -30.0, 30.0))});
            }
        }
        out.push_back(std::move(q));
    }
    return out;
}

}   // namespace rc::door::hlook
