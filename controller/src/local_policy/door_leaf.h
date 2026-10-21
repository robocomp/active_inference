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
//     P_k(cell free) = w_k + (1 − w_k) e^{−r}   ⇒   r_k = −log(w_k + (1 − w_k) e^{−r})
//   ★ ONLY A BELIEF IS RE-EXPLAINED, NEVER OCCUPANCY (Task 3b fix round 1, I-1). A HARD cell is a live return inside the
//     body's height band THIS cycle (residual_risk.h harden_corroborated) — the cell is occupied, whatever caused it; with
//     no belief behind it, it is hard for the same conservative reason. Explanation re-assigns a cell's CAUSE and COUNT; it
//     cannot make an occupied place free. So a hard cell stays +inf under every hypothesis and stays in the planner's
//     distance map (a chair or a person inside the leaf's swept region is not "explained" into free space), and the
//     re-pricing applies to the SOFT cells only — where the smear of a leaf that moved and the correlated over-count live.
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
//   THE SPLIT, BY THE MODEL'S OWN WEIGHTS (fix round 1: no take rule). Every soft cell's price factors exactly into the
//   part ALL hypotheses agree on, m = min_k r_k, and the hypothesis-dependent excess e_k = r_k − m ≥ 0:
//     Σ_k p_k Φ_k exp(−Σ r_k) = exp(−Σ m) · Σ_k p_k Φ_k exp(−Σ e_k)
//   so the PLANNER keeps every cell, at price m (residual's belief minus what every hypothesis explains — hard cells
//   untouched), and the door's density carries the rest: −log Σ_k p_k Φ_k exp(−Σ_{cells under the disc} e_k). A cell no
//   hypothesis explains has e_k = 0 for all k and adds nothing here (an identity, not a cut-off); nothing leaves the map.
//   No gate anywhere; P = 0 (to double precision, in log space) is +inf, which the field reads as impassable.
//   ⚑ Two approximations, both stated: the EXCESS is summed under the inscribed DISC (the planner prices m under the hull,
//   minimised over 8 headings — a disc under-counts a hull), and summed cell nats inherit residual's convention of
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

// §2: the door's leaf with the hypothesis-dependent excess of the soft residual cells it explains.
// ★ Task 3f fix round 1 (I-1): IS THE BODY STILL WITHIN THE DOORWAY'S REACH? The opening (half its width about its centre) or
//   any leaf hypothesis's sweep (the disc its free edge describes about the hinge, plus the leaf's half thickness), grown by
//   the body's circumscribed radius: geometry, the set of centres from which the body can touch either. While it is, the
//   held leaf and the jambs stay priced in EVERY policy field — not only the cross's — so the goal that follows a passage
//   (an explore standpoint, a mission waypoint) still routes around the leaf the robot just went past.
[[nodiscard]] inline bool body_within_door_zone(const Eigen::Vector2f& p, float body_circ_r, const Eigen::Vector2f& opening_centre,
                                                float half_width, const DoorLeaf& leaf)
{
    if ((p - opening_centre).norm() < half_width + body_circ_r) return true;
    for (const auto& h : leaf.hyps)
        if ((p - h.hinge).norm() < h.length + h.half_thick + body_circ_r) return true;
    return false;
}

struct DoorLeafEvidence
{
    DoorLeaf leaf;
    float cell_m = 0.f;
    float body_r = 0.f;                      // the ONE inscribed radius the density sums under (the field's free test's, M-4)
    std::vector<Eigen::Vector2f> cells;      // the soft cells with a non-zero excess under some hypothesis
    std::vector<float> e_k;                  // cells.size() x hyps.size(), row-major: r_k − min_k r_k per hypothesis
    // a coarse bucket grid over the cells (a lookup structure; the bucket is the disc's reach — any size gives the same sums)
    Eigen::Vector2f b0 = Eigen::Vector2f::Zero(); float bsize = 0.25f; int bw = 0, bh = 0;
    std::vector<std::vector<int>> buckets;
    float mass = 0.f;                        // Σ over the soft cells of Σ_k p_k w_k: the belief the leaf explains (diagnostic)
    int hard_in_swept = 0;                   // hard cells the leaf could explain — kept occupied (diagnostic)
    [[nodiscard]] bool empty() const { return leaf.empty(); }
};

// §1–§2 over residual's cells (centres + risk_nats; an EMPTY risk vector means "every cell hard" and is left so). `risk` is
// rewritten IN PLACE to the planner's share m for every soft cell; hard cells are untouched. body_r: the bare body's
// inscribed radius — the same value the field's free test and door_leaf_density use (M-4).
inline DoorLeafEvidence explain_residual_by_leaf(const DoorLeaf& L, std::span<const Eigen::Vector2f> centres, float cell_m,
                                                 std::vector<float>& risk, float body_r)
{
    DoorLeafEvidence ev;
    ev.leaf = L;
    ev.cell_m = cell_m;
    ev.body_r = body_r;
    const std::size_t K = L.hyps.size();
    if (K == 0 or risk.size() != centres.size()) return ev;
    std::vector<float> w(K), rk(K);
    double mass = 0.0;
    for (std::size_t i = 0; i < centres.size(); ++i)
    {
        for (std::size_t k = 0; k < K; ++k) w[k] = leaf_explained(L.hyps[k], centres[i], cell_m);
        if (not std::isfinite(risk[i]))   // occupied: re-assigned cause, never freed (I-1)
        {
            if (std::ranges::any_of(w, [](float x) { return x > 0.f; })) ++ev.hard_in_swept;
            continue;
        }
        float m = std::numeric_limits<float>::infinity();
        for (std::size_t k = 0; k < K; ++k)
        {
            rk[k] = explained_risk_nats(risk[i], w[k]);
            if (L.hyps[k].p > 0.f) m = std::min(m, rk[k]);
            mass += double(L.hyps[k].p) * w[k];
        }
        if (not std::isfinite(m)) continue;   // no hypothesis carries probability: nothing to split
        risk[i] = m;
        if (std::ranges::all_of(rk, [m](float x) { return x == m; })) continue;   // no excess anywhere: an identity
        ev.cells.push_back(centres[i]);
        for (std::size_t k = 0; k < K; ++k) ev.e_k.push_back(rk[k] - m);
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

// §2: −log of the door's factor of P(the body's centre at p is contact-free): the leaf mixture with each hypothesis's excess
// of the cells under the disc (the common part m is the planner's). 0 with no leaf. Radius: ev.body_r (M-4).
[[nodiscard]] inline float door_leaf_density(const DoorLeafEvidence& ev, const Eigen::Vector2f& p)
{
    const float body_r = ev.body_r;
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
                        for (std::size_t k = 0; k < K; ++k) R[k] += double(ev.e_k[std::size_t(i) * K + k]);
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
    ev.body_r = body_r;
    return door_leaf_density(ev, p);
}

}  // namespace rc::policy
