#pragma once
/*
 * self_return_map.h — which body-frame places return the ROBOT ITSELF, learned from motion (Task 14 fix round 6).
 *
 * try79 (+0.252, +0.236) and try82 (+0.246, +0.242): a FIXED body-frame point at the front-right corner, outside both the
 * planning hull and the mesh ⊕ σ_r self footprint, sat in the contact band and closed every forward motion. Some part of
 * the robot (a mount, a bumper corner, a sensor) is in neither outline. No outline written by hand can be trusted to be
 * complete, so the self footprint is LEARNED from the one thing that tells the robot from the world:
 *
 *   a return that keeps its BODY-frame position while the robot moves through the WORLD is the robot;
 *   a return that keeps its WORLD position (and so slides through the body frame) is the world.
 *
 * MODEL. A body-frame grid of cells (size σ_r) over the domain the net cares about (the returns body grown by
 * margin + 2σ_r). Each cell carries the log-odds L that a return in it is the robot. Each sweep is compared with the
 * oldest sweep in a short history ring (kRing sweeps). T = the body motion between them (from the room poses). For a
 * current return q:
 *   self hypothesis  — it was at q then:              d_s = distance from q to the nearest return of the old sweep
 *   world hypothesis — it was at T⁻¹q then (it moved): d_w = distance from T⁻¹q to the nearest return of the old sweep
 *   ΔL = [ℓ(d_s) − ℓ(d_w)] / 2,   ℓ(d) = log(exp(−d²/2s²) + ε),  s² = 2σ_r² + σ_pose²
 * (two noisy fixes, plus the pose's own uncertainty). With no motion T⁻¹q = q and ΔL = 0 exactly: evidence is gained
 * ONLY as far as the motion separates the hypotheses — the likelihood term the ruling asks for, no motion gate. The ½: each
 * sweep enters exactly two comparisons (once as the current sweep, once as the reference), so its noise is shared by two.
 * EMPTINESS: a cell believed self (L > 0) with no return this sweep, while the motion would have moved a world point out of
 * it, is evidence against self: ΔL = m·log(1 − P_d)/2, m = 1 − exp(−δ²/2s²), δ = the motion's displacement there.
 * A real jamb the robot approaches is world-fixed: it slides through the body frame by exactly T, so d_w ≈ noise and
 * d_s grows with the motion — its cell's L goes DOWN, never up.
 *
 * BODY. Cells the evidence favours as self grow the contact law's body (body_with_learned): a body part is geometry.
 * QUERY. A return is self if any learned place within σ_r of it is (its own noise): the learned cells ⊕ σ_r.
 * USE. A kept return's contact weight is P(world) = 1 − P(self) of its cell, RENORMALISED at even odds:
 * min(1, 2·(1 − P(self))) — the same Φ/Φ(0) form as anchor_contact_weight: until the evidence FAVOURS self, a return is a
 * certain contact. Plain 1 − P(self) leaked P(self) of every ceiling (0.0035 m/s at the 0.01 prior, more on noise) and the
 * test robot CREPT through its own standoff into a jamb; only evidence that it is the robot may lower a contact
 * (rc::AnchorJamb::p, which every law already weighs).
 * SEED: the mesh ⊕ σ_r self footprint is certain self (the front face is self from the start) and is never learned over —
 * its returns are deleted upstream, so their absence is not evidence; everywhere else starts at the world prior.
 * Each return's evidence goes to its own cell (its scatter spreads one body point over the part ⊕ ~σ_r).
 * ⚑ DECISION CONSTANTS, flagged: the prior P0 (a new place is the world: 0.01), the cap |L| ≤ kLmax (keeps every cell
 *   re-learnable), the outlier floor ε, the ring length (one second at
 *   20 Hz). None is a gate on the data; each bounds or shapes a likelihood.
 * ⚠ A localisation JUMP looks like motion the world did not make (world returns stay put in the body frame) and would read
 *   as self evidence for every comparison spanning it. A pose step the BASE could not have made in that time
 *   (|Δp| > v_cap·dt + 3σ_pose or |Δθ| > w_cap·dt + 3σ_θ — the base's own capability, ⚑ 3σ flagged) is a discontinuity of
 *   the motion model: the history ring is restarted there, so no comparison ever spans it.
 * PERSISTED: etc/self_returns.csv ("i,j,logodds", classic locale, read with std::from_chars).
 */

#include "controller_anchor_servo.h"
#include "../../common/robot_footprint/mesh_hull.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <deque>
#include <fstream>
#include <locale>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>
#include <Eigen/Dense>

namespace rc
{

class SelfReturnMap
{
public:
    static constexpr int kRing = 20;          // ⚑ one second of sweeps at 20 Hz
    static constexpr float kP0 = 0.01f;       // ⚑ prior: a new place is the world
    static constexpr float kLmax = 6.f;       // ⚑ |log-odds| cap (P 0.9975): every cell stays re-learnable
    static constexpr float kEps = 0.01f;      // ⚑ outlier floor of the kernel likelihood (bounds one comparison's evidence)
    // P_d is MEASURED per cell: the rate at which sweeps put a return in it (an EMA over kRing sweeps). A part's returns
    // scatter over several cells, so a cell is hit on a fraction of sweeps — a constant P_d (0.5 was tried) charged every
    // miss as if the part had vanished and erased real parts faster than they were learned.

    // `self_fp`: the hand-written self footprint (mesh ⊕ σ_r) — certain self, never learned over (its returns are deleted
    // before they reach this map, so their absence must not be read as evidence).
    void configure(float sigma_r, std::span<const Eigen::Vector2f> returns_body, std::span<const Eigen::Vector2f> self_fp,
                   float margin)
    {
        sigma_ = std::max(sigma_r, 1e-3f);
        cell_ = sigma_;   // σ_r: a part's scatter falls in a few cells, not dozens
        body_.assign(returns_body.begin(), returns_body.end());
        self_fp_.assign(self_fp.begin(), self_fp.end());
        domain_ = margin + 2.f * sigma_;
    }
    // The base's capability (m/s, rad/s): a pose step beyond it in the elapsed time is a localisation jump, not motion.
    void set_capability(float v_cap, float w_cap) { v_cap_ = v_cap; w_cap_ = w_cap; }
    // Contact weight of a return there: P(world) renormalised so the prior is certain (see the header).
    [[nodiscard]] float p_world(const Eigen::Vector2f& q) const
    { return std::min(1.f, 2.f * (1.f - p_self(q))); }

    [[nodiscard]] bool in_domain(const Eigen::Vector2f& q) const
    { return body_.size() >= 3 and anchor_signed_distance(body_, q) < domain_; }

    [[nodiscard]] float logodds(const Eigen::Vector2f& q) const
    {
        const auto it = L_.find(key(ix(q.x()), ix(q.y())));
        return it != L_.end() ? it->second : prior();
    }
    // A return at q measures a body point within ~σ_r of it (its own range noise), so it is self if ANY place within σ_r of
    // it is: the learned self region is the learned cells ⊕ σ_r — exactly as the hand-written one is the mesh ⊕ σ_r. The
    // contact standoff (margin > σ_r from body_with_learned) keeps a world obstacle out of that band.
    [[nodiscard]] float p_self(const Eigen::Vector2f& q) const
    {
        if (self_fp_.size() >= 3 and anchor_point_in_convex(self_fp_, q)) return 1.f;
        if (not in_domain(q)) return kP0;
        float l = logodds(q);
        const int ci = ix(q.x()), cj = ix(q.y());
        for (int di = -1; di <= 1; ++di)
            for (int dj = -1; dj <= 1; ++dj)
                if ((centre(ci + di, cj + dj) - q).norm() <= sigma_)
                    if (const auto it = L_.find(key(ci + di, cj + dj)); it != L_.end()) l = std::max(l, it->second);
        return 1.f / (1.f + std::exp(-l));
    }

    // One sweep: the returns (robot frame, the kept ones), the robot's room pose (room_T_robot), its position σ.
    void update(std::span<const Eigen::Vector2f> returns_robot, const Eigen::Affine2f& room_T_robot, float sigma_pose,
                double t_s, float sigma_theta = 0.f)
    {
        // A pose step the base could not have made: restart the history (no comparison spans a discontinuity).
        if (not ring_.empty())
        {
            const Frame& last = ring_.back();
            const float dt = float(std::max(0.0, t_s - last.t));
            const Eigen::Affine2f D = last.T.inverse() * room_T_robot;
            const float dp = D.translation().norm();
            const float dth = std::abs(std::atan2(D.linear()(1, 0), D.linear()(0, 0)));
            if (dp > v_cap_ * dt + 3.f * sigma_pose or dth > w_cap_ * dt + 3.f * sigma_theta) { ring_.clear(); ++restarts_; }
        }
        Frame f;
        f.T = room_T_robot;
        f.t = t_s;
        for (const auto& q : returns_robot)
            if (q.allFinite() and in_domain(q)) f.pts.push_back(q);
        if (not ring_.empty())
        {
            const Frame& ref = ring_.front();
            // ref body → now body. A world point seen at r in the ref frame is at T·r now; a current q was at T⁻¹q then.
            const Eigen::Affine2f T = room_T_robot.inverse() * ref.T;
            const Eigen::Affine2f Tinv = T.inverse();
            const float s2 = 2.f * sigma_ * sigma_ + sigma_pose * sigma_pose;
            const auto ell = [&](float d2) { return std::log(std::exp(-d2 / (2.f * s2)) + kEps); };
            const auto nearest2 = [&](const Eigen::Vector2f& p)
            {
                float best = std::numeric_limits<float>::infinity();
                for (const auto& r : ref.pts) best = std::min(best, (r - p).squaredNorm());
                return best;
            };
            // Each return's evidence goes to ITS OWN cell. (Spreading it over the cells within 2σ_r was tried: the learned
            // place became the part ⊕ 2σ_r, a 6-cm blob that grew the contact body until an approaching jamb was inside it
            // and deleted — test 13 b2. The returns' own scatter already spreads the evidence over the part ⊕ σ_r.)
            std::unordered_map<std::int64_t, bool> seen;
            for (const auto& q : f.pts)
            {
                if (self_fp_.size() >= 3 and anchor_point_in_convex(self_fp_, q)) continue;
                const float dL = 0.5f * (ell(nearest2(q)) - ell(nearest2(Tinv * q)));
                const std::int64_t k = key(ix(q.x()), ix(q.y()));
                seen[k] = true;
                auto [it, fresh] = L_.try_emplace(k, prior());
                it->second = std::clamp(it->second + dL, -kLmax, kLmax);
            }
            // Emptiness: a believed-self cell with no return now, weighted by how far the motion moves a world point there,
            // against the cell's own measured hit rate h (a cell hit on 15% of sweeps misses on 85% of them as a matter of
            // course: log(1 − h) per miss).
            for (auto& [k, l] : L_)
            {
                const bool hit = seen.contains(k);
                float& h = hit_[k];
                h += ((hit ? 1.f : 0.f) - h) / float(kRing);
                if (l <= 0.f or hit) continue;
                const Eigen::Vector2f c = centre(int(k >> 32), int(std::int32_t(k & 0xffffffff)));
                const float m = 1.f - std::exp(-(Tinv * c - c).squaredNorm() / (2.f * s2));
                l = std::clamp(l + 0.5f * m * std::log(std::max(1e-3f, 1.f - h)), -kLmax, kLmax);
            }
        }
        ring_.push_back(std::move(f));
        while (int(ring_.size()) > kRing) ring_.pop_front();
    }

    // ★ A learned self place is BODY GEOMETRY: whatever returns from there is part of the robot, and the robot can hit
    //   things with it. So the contact law's body is the returns body GROWN by every cell the evidence favours as self
    //   (L > 0, even odds — the same point the contact weight renormalises at): conv(returns body ∪ those cells). Without
    //   it a world obstacle met AT the learned part is indistinguishable from the part itself in the last centimetres
    //   (measured, test 13 b2: the robot drove 2.6 cm into a jamb there); with it, the standoff is kept from the part.
    [[nodiscard]] std::vector<Eigen::Vector2f> body_with_learned(std::span<const Eigen::Vector2f> returns_body) const
    {
        std::vector<Eigen::Vector2f> pts(returns_body.begin(), returns_body.end());
        bool any = false;
        for (const auto& [k, l] : L_)
            if (l > 0.f) { pts.push_back(centre(int(k >> 32), int(std::int32_t(k & 0xffffffff)))); any = true; }
        if (not any) return {returns_body.begin(), returns_body.end()};
        return rc::mesh::convex_hull_2d(std::move(pts));
    }

    struct Census { int cells = 0, self_cells = 0, learned_self = 0; std::vector<Eigen::Vector3f> top; };
    // learned_self: cells above P 0.5 OUTSIDE `seed_fp` (what motion taught beyond the outline).
    [[nodiscard]] Census census(std::span<const Eigen::Vector2f> seed_fp, int n_top = 8) const
    {
        Census c;
        c.cells = int(L_.size());
        for (const auto& [k, l] : L_)
        {
            if (l <= 0.f) continue;
            ++c.self_cells;
            const Eigen::Vector2f p = centre(int(k >> 32), int(std::int32_t(k & 0xffffffff)));
            if (not anchor_point_in_convex(seed_fp, p)) { ++c.learned_self; c.top.emplace_back(p.x(), p.y(), l); }
        }
        std::ranges::sort(c.top, [](const auto& a, const auto& b) { return a.z() > b.z(); });
        if (int(c.top.size()) > n_top) c.top.resize(std::size_t(n_top));
        return c;
    }

    bool save(const std::string& path) const
    {
        std::ofstream f(path, std::ios::out | std::ios::trunc);
        if (not f) return false;
        f.imbue(std::locale::classic());
        f << "# self_return_map cell_m " << cell_ << "\n";
        for (const auto& [k, l] : L_)
            if (std::abs(l - prior()) > 1e-3f) f << int(k >> 32) << ',' << int(std::int32_t(k & 0xffffffff)) << ',' << l << '\n';
        return bool(f);
    }
    // Loads cells written with the SAME cell size (a different size is a different map: ignored, returns -1).
    int load(const std::string& path)
    {
        std::ifstream f(path);
        if (not f) return 0;
        std::string line;
        int n = 0;
        while (std::getline(f, line))
        {
            if (line.starts_with("#"))
            {
                const auto p = line.find("cell_m ");
                float c = 0.f;
                if (p != std::string::npos)
                {
                    const char* b = line.data() + p + 7;
                    std::from_chars(b, line.data() + line.size(), c);
                }
                if (std::abs(c - cell_) > 1e-6f) return -1;
                continue;
            }
            int i = 0, j = 0;
            float l = 0.f;
            const char* b = line.data();
            const char* e = b + line.size();
            auto r1 = std::from_chars(b, e, i);
            if (r1.ec != std::errc{} or r1.ptr == e or *r1.ptr != ',') continue;
            auto r2 = std::from_chars(r1.ptr + 1, e, j);
            if (r2.ec != std::errc{} or r2.ptr == e or *r2.ptr != ',') continue;
            if (std::from_chars(r2.ptr + 1, e, l).ec != std::errc{}) continue;
            L_[key(i, j)] = std::clamp(l, -kLmax, kLmax);
            ++n;
        }
        return n;
    }

    [[nodiscard]] float cell() const { return cell_; }
    [[nodiscard]] int restarts() const { return restarts_; }

private:
    struct Frame { Eigen::Affine2f T = Eigen::Affine2f::Identity(); double t = 0.0; std::vector<Eigen::Vector2f> pts; };
    [[nodiscard]] static float prior() { return std::log(kP0 / (1.f - kP0)); }
    [[nodiscard]] int ix(float v) const { return int(std::floor(v / cell_)); }
    [[nodiscard]] Eigen::Vector2f centre(int i, int j) const { return {(float(i) + 0.5f) * cell_, (float(j) + 0.5f) * cell_}; }
    [[nodiscard]] static std::int64_t key(int i, int j) { return (std::int64_t(i) << 32) | std::uint32_t(j); }

    float sigma_ = 0.03f, cell_ = 0.015f, domain_ = 0.11f, v_cap_ = 1.f, w_cap_ = 2.f;
    int restarts_ = 0;
    std::vector<Eigen::Vector2f> body_, self_fp_;
    std::unordered_map<std::int64_t, float> L_, hit_;
    std::deque<Frame> ring_;
};

}  // namespace rc
