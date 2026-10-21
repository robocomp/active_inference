// door_leaf.h — Task 3b (2026-10-21 doorway release, rtD1c leg 5): THE DOOR LEAF IS A KNOWN OBSTACLE, NOT UNEXPLAINED RESIDUAL.
// Pure (Eigen + std): the controller (goal_adapter.h converts the anchor's hypotheses) and the tests share it.
//
// ★ WHY. door_concept models the swung leaf (anchor_leaf.h: the hinge × swing hypotheses, each a centre line hinge→free edge
//   with its MAP angle φ, σ_φ and probability). residual_concept knows nothing of it: the leaf's returns land in its grid as
//   UNEXPLAINED occupancy, and a cell a live return corroborates is HARD in the planner (residual_risk.h harden_corroborated).
//   In rtD1c leg 5 the open leaf was hundreds of such cells (density up to 1240 m/m): inflated by the body's inscribed radius
//   they disconnected the cross goal's free mask for ~105 s, and the one object they describe was counted as hundreds of
//   independent obstacles. The model already holds a better explanation of those returns; this header uses it.
//
// ─── THE LEAF HYPOTHESIS IN THE ROOM FRAME ───────────────────────────────────────────────────────────────────────────────
//   A point r metres from the hinge along the centre line is q = A + R(ψ)·q_a (A, ψ: the anchor's pose; q_a its anchor-frame
//   coordinates). Its covariance is controller_anchor_servo.h anchor_leaf_points' — ONE rule for jambs and leaf:
//     Σ_q(r) = P_xy + σ_ψ² g gᵀ + r² σ_φ² t tᵀ,   g = J (q − A) = ∂q/∂ψ,   t = the leaf's unit normal
//   (a point r from the hinge moves r·dφ across the leaf when φ moves: σ_φ widens the SWEPT region toward the free edge).
//
// ─── 1. WHICH RESIDUAL CELLS THE LEAF EXPLAINS (leaf_explained) ─────────────────────────────────────────────────────────
//   A residual cell at q is occupancy with a cause: hypothesis k's leaf, or something the model does not know. The leaf's
//   predictive likelihood at q RELATIVE TO ITS MODE is w_k(q) = exp(−½ δᵀ S⁻¹ δ), δ = q − the nearest point of k's centre
//   line, S = Σ_q + (thickness: uniform ±half_thick ⇒ half_thick²/3 along t) + (the cell: uniform ⇒ cell²/12 I) — the share
//   of the cell k explains ("within its σ").
//   ⚑ It is NOT a calibrated responsibility: that needs the density of unknown occupancy near a door, which nothing measures;
//   the mode-relative kernel is that responsibility with the unknown density set equal to the leaf's own peak — the most the
//   leaf could claim. Flagged, not hidden (task-3b-report).
//   Under k the cell's price is the mixture over its cause: as k's leaf it adds nothing of its own (k's leaf is priced once,
//   as an object); as unknown occupancy it keeps residual's price r:
//     P_k(cell free) = w_k + (1 − w_k) e^{−r}   ⇒   r_k = −log(w_k + (1 − w_k) e^{−r})        (r = +inf, hard ⇒ r_k = −log w_k)
//
// ─── 2. ONE MIXTURE OVER THE HYPOTHESES — LEAF AND CELLS TOGETHER (DoorLeafEvidence, door_leaf_density) ────────────────
//   The world under hypothesis k is k's leaf PLUS the residual cells k does not explain. The body's centre at p (heading-free,
//   the inscribed disc r_b the cost-to-go's free test uses) is contact-free with probability
//     P(free at p) = Σ_k p_k · Φ((D_k − r_b − half_thick)/σ_D,k) · exp(−Σ_{cells c under the disc} r_k(c))
//   (D_k = the distance to k's centre line, σ_D,k² = n̂ᵀ Σ_q n̂ along the direction n̂ from it to p), and the density is
//   −log P(free at p), in nats per metre as GridPlanner::body_risk_at's.
//   ★ WHY TOGETHER (measured, test_policy_scenarios leaf_known_hinge_side): pricing the leaf mixture and the cells separately
//     makes a 90 %-certain leaf cost at most −log(1 − 0.9) = 2.3 nats/m to cross — "10 % it hangs from the other jamb" —
//     while the returns that say something IS there were handed to the leaf and forgotten: the field ran through the leaf
//     and the policy dithered at its face (1/20 reached against 20/20 with the leaf hard in the map). Under the other
//     hypotheses those same cells are UNEXPLAINED, so they still block: the coupling is the joint model, not a correction.
//   The cells the leaf can explain at all (some w_k > 0 in float) are therefore TAKEN from the planner's residual and priced
//   here, per hypothesis; every one keeps its exact price under every hypothesis that does not explain it (r_k = r when
//   w_k = 0), so the partition is arithmetic (where the sum is evaluated), not a model level. A hard cell no hypothesis
//   explains is +inf in every term, i.e. impassable exactly where the planner's free mask would have blocked it.
//   No gate anywhere; P = 0 (to double precision, in log space) is +inf, which the field reads as impassable.
//   ⚑ Two approximations, both stated: the cells are summed under the inscribed DISC (the planner's body_risk_at takes the
//   hull, minimised over 8 headings — a disc under-counts a hull), and summed cell nats inherit residual's convention of
//   independent cells (the correlated-cells over-count rtD1c exposed is residual's pricing, not changed here).
#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <vector>

namespace rc::policy
{
struct LeafHypothesis
{
    Eigen::Vector2f hinge = Eigen::Vector2f::Zero();   // the centre line's hinge end, room frame
    Eigen::Vector2f dir{1.f, 0.f};                     // unit, hinge → free edge, room frame
    float length = 0.f;                                // |free edge − hinge|
    float half_thick = 0.02f;                          // the leaf model's (anchor_leaf.h kLeafHalfThickM)
    float p = 1.f;                                     // the hypothesis's probability (Σ over a DoorLeaf = 1)
    Eigen::Vector2f pivot = Eigen::Vector2f::Zero();   // the anchor's centre (room frame): the point ψ rotates the leaf about
    Eigen::Matrix2f P_xy = Eigen::Matrix2f::Zero();    // the anchor's position covariance, room frame
    float var_psi = 0.f, var_phi = 0.f;                // the anchor's yaw variance, the leaf angle's
    [[nodiscard]] Eigen::Vector2f normal() const { return {-dir.y(), dir.x()}; }
    [[nodiscard]] Eigen::Vector2f tip() const { return hinge + length * dir; }
    // Σ_q at r metres from the hinge (header §: anchor_leaf_points' rule)
    [[nodiscard]] Eigen::Matrix2f point_cov(float r) const
    {
        const Eigen::Vector2f q = hinge + r * dir, d = q - pivot, g(-d.y(), d.x()), t = normal();
        return P_xy + var_psi * (g * g.transpose()) + (r * r * var_phi) * (t * t.transpose());
    }
    // the centre line's nearest point to x, as its distance r from the hinge (clamped to the segment: geometry, not a level)
    [[nodiscard]] float nearest_r(const Eigen::Vector2f& x) const { return std::clamp((x - hinge).dot(dir), 0.f, length); }
};
struct DoorLeaf
{
    std::vector<LeafHypothesis> hyps;
    [[nodiscard]] bool empty() const { return hyps.empty(); }
};

namespace detail
{
// log Φ(z), accurate in the far lower tail (erfc underflows near z = −38: the asymptotic series there)
inline double log_phi(double z)
{
    if (z > -30.0) return std::log(std::max(1e-300, 0.5 * std::erfc(-z * 0.70710678118654752)));
    const double z2 = z * z;
    return -0.5 * z2 - std::log(-z) - 0.91893853320467274 + std::log1p(-1.0 / z2 + 3.0 / (z2 * z2));
}
}  // namespace detail

// §1: the share w_k(q) ∈ [0,1] of a residual cell (centre q, side cell_m) hypothesis h explains.
[[nodiscard]] inline float leaf_explained(const LeafHypothesis& h, const Eigen::Vector2f& q, float cell_m)
{
    if (not (h.length > 0.f)) return 0.f;
    const float r = h.nearest_r(q);
    const Eigen::Vector2f d = q - (h.hinge + r * h.dir), t = h.normal();
    const Eigen::Matrix2d S = (h.point_cov(r) + (h.half_thick * h.half_thick / 3.f) * (t * t.transpose())
                               + (cell_m * cell_m / 12.f) * Eigen::Matrix2f::Identity()).cast<double>();
    if (not (S.determinant() > 0.0)) return 0.f;   // a degenerate covariance explains nothing
    const Eigen::Vector2d dd = d.cast<double>();
    return float(std::exp(-0.5 * dd.dot(S.ldlt().solve(dd))));
}

// §1: residual's price r of a cell under a hypothesis that explains the share w of it.
[[nodiscard]] inline float explained_risk_nats(float r, float w)
{
    const double pf = double(w) + (1.0 - double(w)) * (std::isfinite(r) ? std::exp(-double(r)) : 0.0);
    return pf > 0.0 ? float(-std::log(std::min(1.0, pf))) : std::numeric_limits<float>::infinity();
}

// §2: the door's leaf with the residual cells it can explain, each priced under every hypothesis.
struct DoorLeafEvidence
{
    DoorLeaf leaf;
    float cell_m = 0.f;
    std::vector<Eigen::Vector2f> cells;      // the cells taken from the planner's residual
    std::vector<float> r_k;                  // cells.size() x hyps.size(), row-major: r_k(cell) per hypothesis
    // a coarse bucket grid over the cells (a lookup structure; the bucket is one inscribed radius — any size gives the
    // same sums)
    Eigen::Vector2f b0 = Eigen::Vector2f::Zero(); float bsize = 0.25f; int bw = 0, bh = 0;
    std::vector<std::vector<int>> buckets;
    float mass = 0.f;                        // Σ over the taken cells of Σ_k p_k w_k: the occupancy the leaf explains (diagnostic)
    [[nodiscard]] bool empty() const { return leaf.empty(); }
};

// §2: take from residual's cells (centres + risk_nats; an EMPTY risk vector means "every cell hard") the ones the leaf can
// explain; `keep` receives the indices of the cells the planner keeps. body_r: the radius the density will sum under (the
// bucket size).
inline DoorLeafEvidence take_explained_cells(const DoorLeaf& L, std::span<const Eigen::Vector2f> centres, float cell_m,
                                             std::span<const float> risk, float body_r, std::vector<std::size_t>& keep)
{
    DoorLeafEvidence ev;
    ev.leaf = L;
    ev.cell_m = cell_m;
    keep.clear();
    keep.reserve(centres.size());
    const std::size_t K = L.hyps.size();
    const bool all_hard = risk.size() != centres.size();
    std::vector<float> w(K);
    double mass = 0.0;
    for (std::size_t i = 0; i < centres.size(); ++i)
    {
        bool any = false;
        for (std::size_t k = 0; k < K; ++k) { w[k] = leaf_explained(L.hyps[k], centres[i], cell_m); any = any or w[k] > 0.f; }
        if (not any) { keep.push_back(i); continue; }
        const float r = all_hard ? std::numeric_limits<float>::infinity() : risk[i];
        ev.cells.push_back(centres[i]);
        for (std::size_t k = 0; k < K; ++k) { ev.r_k.push_back(explained_risk_nats(r, w[k])); mass += double(L.hyps[k].p) * w[k]; }
    }
    ev.mass = float(mass);
    if (not ev.cells.empty())
    {
        ev.bsize = std::max(1e-3f, body_r + 0.5f * cell_m);
        Eigen::Vector2f lo = ev.cells.front(), hi = lo;
        for (const auto& c : ev.cells) { lo = lo.cwiseMin(c); hi = hi.cwiseMax(c); }
        ev.b0 = lo;
        ev.bw = int((hi.x() - lo.x()) / ev.bsize) + 1;
        ev.bh = int((hi.y() - lo.y()) / ev.bsize) + 1;
        ev.buckets.assign(std::size_t(ev.bw) * std::size_t(ev.bh), {});
        for (std::size_t i = 0; i < ev.cells.size(); ++i)
        {
            const int bx = int((ev.cells[i].x() - lo.x()) / ev.bsize), by = int((ev.cells[i].y() - lo.y()) / ev.bsize);
            ev.buckets[std::size_t(by) * std::size_t(ev.bw) + std::size_t(bx)].push_back(int(i));
        }
    }
    return ev;
}

// §2: −log P(the body's centre at p is contact-free) under the door's mixture (leaf + the cells it took). 0 with no leaf.
[[nodiscard]] inline float door_leaf_density(const DoorLeafEvidence& ev, const Eigen::Vector2f& p, float body_r)
{
    const auto& H = ev.leaf.hyps;
    if (H.empty()) return 0.f;
    const std::size_t K = H.size();
    // the cells under the disc (the free test's rule: |c − p| − cell/2 <= r_b), summed per hypothesis
    std::vector<double> R(K, 0.0);
    if (not ev.cells.empty())
    {
        const float reach = body_r + 0.5f * ev.cell_m;
        const int bx = int(std::floor((p.x() - ev.b0.x()) / ev.bsize)), by = int(std::floor((p.y() - ev.b0.y()) / ev.bsize));
        for (int y = by - 1; y <= by + 1; ++y)
            for (int x = bx - 1; x <= bx + 1; ++x)
            {
                if (x < 0 or y < 0 or x >= ev.bw or y >= ev.bh) continue;
                for (const int i : ev.buckets[std::size_t(y) * std::size_t(ev.bw) + std::size_t(x)])
                    if ((ev.cells[std::size_t(i)] - p).norm() <= reach)
                        for (std::size_t k = 0; k < K; ++k) R[k] += double(ev.r_k[std::size_t(i) * K + k]);
            }
    }
    // log Σ_k p_k Φ_k e^{−R_k}, in log space (log-sum-exp) so a large finite price stays finite
    double mass = 0.0, best = -std::numeric_limits<double>::infinity();
    std::vector<double> lt(K, -std::numeric_limits<double>::infinity());
    for (std::size_t k = 0; k < K; ++k)
    {
        const auto& h = H[k];
        if (not (h.p > 0.f)) continue;
        mass += h.p;
        if (not std::isfinite(R[k])) continue;
        const float r = h.nearest_r(p);
        const Eigen::Vector2f d = p - (h.hinge + r * h.dir);
        const float D = d.norm();
        const Eigen::Vector2f nh = D > 0.f ? Eigen::Vector2f(d / D) : h.normal();   // on the line: across the leaf
        const double sD = std::sqrt(std::max(0.0, double(nh.dot(h.point_cov(r) * nh))));
        const double slack = double(D) - double(body_r) - double(h.half_thick);
        const double lphi = sD > 0.0 ? detail::log_phi(slack / sD)
                                     : (slack > 0.0 ? 0.0 : -std::numeric_limits<double>::infinity());
        lt[k] = std::log(double(h.p)) + lphi - R[k];
        best = std::max(best, lt[k]);
    }
    if (not (mass > 0.0)) return 0.f;
    if (not std::isfinite(best)) return std::numeric_limits<float>::infinity();
    double acc = 0.0;
    for (const double v : lt) if (std::isfinite(v)) acc += std::exp(v - best);
    const double logp = best + std::log(acc) - std::log(mass);
    return float(std::max(0.0, -logp));   // −log of a probability: ≥ 0 (the max only absorbs rounding)
}

// The leaf alone (no residual cells: a leaf residual has not seen — e.g. within the LiDAR self-filter, anchor_leaf.h).
[[nodiscard]] inline float leaf_contact_nats(const DoorLeaf& L, const Eigen::Vector2f& p, float body_r)
{
    DoorLeafEvidence ev;
    ev.leaf = L;
    return door_leaf_density(ev, p, body_r);
}

}  // namespace rc::policy
