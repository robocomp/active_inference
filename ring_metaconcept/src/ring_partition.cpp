#include "ring_partition.h"
#include "ring_belief.h"   // RingBelief::kSlotHypotheses — ONE list of admissible ring sizes, not two

#include <algorithm>
#include <cmath>
#include <numeric>
#include <print>
#include <ranges>

namespace rc
{
namespace
{

constexpr float kPi    = 3.14159265358979f;
constexpr float kTwoPi = 6.28318530717959f;

// log N(x; mu, var), guarded so a degenerate variance cannot produce a NaN that then silently
// poisons every comparison downstream.
inline float log_normal(float x, float mu, float var)
{
    const float v = std::max(1e-9f, var);
    const float d = x - mu;
    return -0.5f * (std::log(kTwoPi * v) + d * d / v);
}

inline float log_sum_exp(const std::vector<float>& terms)
{
    if (terms.empty())
        return -std::numeric_limits<float>::infinity();
    const float m = *std::ranges::max_element(terms);
    if (not std::isfinite(m))
        return m;
    float acc = 0.0f;
    for (const float t : terms) acc += std::exp(t - m);
    return m + std::log(std::max(acc, 1e-30f));
}

// 1/(1+e^-x), with the argument bounded so the exponential cannot overflow to inf/NaN. The bound is
// numerical hygiene, not modelling: at ±80 nats the sigmoid is already 0 or 1 to float precision.
inline float sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-std::clamp(x, -80.0f, 80.0f)));
}

// Minimal union-find. N is the number of chairs in a room — single digits — so nothing here needs
// to be clever.
struct UnionFind
{
    std::vector<std::size_t> parent;
    explicit UnionFind(std::size_t n) : parent(n) { std::iota(parent.begin(), parent.end(), 0u); }
    std::size_t find(std::size_t a) { while (parent[a] != a) { parent[a] = parent[parent[a]]; a = parent[a]; } return a; }
    void unite(std::size_t a, std::size_t b) { a = find(a); b = find(b); if (a != b) parent[b] = a; }
};

}  // namespace

// ─── The three kernels ───────────────────────────────────────────────────────

float log_pair_null(float d, const PartitionParams& p)
{
    // Two points drawn independently and uniformly over a region of area A have separation density
    // 2πd/A. Taking A from the actual room polygon (not a constant) is what keeps this comparison
    // from being scaled by an arbitrary number — exactly as RingBelief::null_nll does one level up.
    const float A = std::max(1e-3f, p.room_area_m2);
    return std::log(kTwoPi * std::max(d, 1e-3f) / A);
}

float log_radius_ring(float d, float var_pair, const PartitionParams& p)
{
    // The anchor is AT the centre, so its distance to a member IS the ring radius. The variance is
    // the radius prior, the two objects' own published position variances, and the slot slack —
    // every one of them already part of the arrangement model.
    const float var = p.prior_radius_std * p.prior_radius_std + var_pair
                    + p.sigma_slot_m * p.sigma_slot_m;
    return log_normal(d, p.prior_radius_m, var);
}

float log_pair_ring(float d, float var_pair, const PartitionParams& p)
{
    // Two members of one rig occupy two DISTINCT slots, so their separation is a chord subtending
    // 2πk/N. At link time neither N nor k is known, so both are marginalised out — uniformly over
    // kSlotHypotheses and over which pair of slots. That marginal is the honest statement of "these
    // two could belong to one ring" before any ring has been fitted.
    //
    // d(chord)/d(radius) = 2·sin(πk/N), so the radius prior propagates with that factor squared.
    std::vector<float> terms;
    terms.reserve(32);

    const float log_wN = -std::log(static_cast<float>(RingBelief::kSlotHypotheses.size()));
    const float var_r  = p.prior_radius_std * p.prior_radius_std;
    const float slack  = 2.0f * p.sigma_slot_m * p.sigma_slot_m;   // one slot slack per member

    for (const int n : RingBelief::kSlotHypotheses)
    {
        const float log_wk = -std::log(static_cast<float>(n - 1));
        for (int k = 1; k < n; ++k)
        {
            const float s   = 2.0f * std::sin(kPi * static_cast<float>(k) / static_cast<float>(n));
            const float mu  = s * p.prior_radius_m;
            const float var = s * s * var_r + var_pair + slack;
            terms.push_back(log_wN + log_wk + log_normal(d, mu, var));
        }
    }
    return log_sum_exp(terms);
}

// ─── Partition ───────────────────────────────────────────────────────────────

std::vector<Cluster> partition_members(const std::vector<PartitionMember>& ring,
                                       const std::vector<PartitionMember>& anchors,
                                       const PartitionParams& p)
{
    std::vector<Cluster> clusters;
    if (ring.empty())
        return clusters;   // no members ⇒ no rig, whatever the anchors are doing

    // ── 1. Single-link over RING members only ────────────────────────────────
    // Anchors are deliberately absent from the union-find: a table must never be able to bridge two
    // groups of chairs into one rig, and a table on its own must never become a cluster.
    UnionFind uf(ring.size());
    for (std::size_t i = 0; i < ring.size(); ++i)
        for (std::size_t j = i + 1; j < ring.size(); ++j)
        {
            const float d = (ring[i].xy - ring[j].xy).norm();
            const float lambda = log_pair_ring(d, ring[i].var + ring[j].var, p) - log_pair_null(d, p);
            if (lambda > 0.0f)   // the two hypotheses are equally likely at 0 — not a tuned cutoff
                uf.unite(i, j);
        }

    // ── 2. Collect components ────────────────────────────────────────────────
    std::vector<std::pair<std::size_t, std::size_t>> root_of;   // (root, index)
    root_of.reserve(ring.size());
    for (std::size_t i = 0; i < ring.size(); ++i)
        root_of.emplace_back(uf.find(i), i);

    std::vector<std::size_t> roots;
    for (const auto& [r, _] : root_of)
        if (std::ranges::find(roots, r) == roots.end())
            roots.push_back(r);

    for (const std::size_t r : roots)
    {
        Cluster c;
        for (const auto& [rr, idx] : root_of)
            if (rr == r)
            {
                c.member_index.push_back(idx);
                c.member_ids.push_back(ring[idx].id);
            }
        // ★Sorted member ids are the association key across cycles — an unstable order would make
        // the overlap match non-deterministic and churn the rig node.
        std::ranges::sort(c.member_ids);
        clusters.push_back(std::move(c));
    }
    // Deterministic cluster order, independent of how the graph enumerated the nodes.
    std::ranges::sort(clusters, [](const Cluster& a, const Cluster& b)
                                { return a.member_ids.front() < b.member_ids.front(); });

    // ── 3. Attach at most one anchor per cluster, by likelihood, greedy 1-to-1 ─
    // Score EVERY (anchor, cluster) pair; there is no distance rule and no accept/reject. A badly
    // scoring anchor is still attached — with a weight of ~0, which makes its centre prior inert.
    // That is the whole repair: selection and prior STRENGTH are the same quantity, so the ordering
    // of get_nodes_by_type() can no longer decide where the ring centre goes.
    struct Candidate { float lambda; std::size_t anchor; std::size_t cluster; };
    std::vector<Candidate> cands;
    cands.reserve(anchors.size() * clusters.size());

    for (std::size_t a = 0; a < anchors.size(); ++a)
        for (std::size_t c = 0; c < clusters.size(); ++c)
        {
            float lambda = 0.0f;
            for (const std::size_t mi : clusters[c].member_index)
            {
                const float d = (anchors[a].xy - ring[mi].xy).norm();
                lambda += log_radius_ring(d, anchors[a].var + ring[mi].var, p) - log_pair_null(d, p);
            }
            cands.push_back({lambda, a, c});
        }

    std::ranges::sort(cands, [&](const Candidate& x, const Candidate& y)
    {
        if (x.lambda != y.lambda) return x.lambda > y.lambda;
        if (anchors[x.anchor].id != anchors[y.anchor].id) return anchors[x.anchor].id < anchors[y.anchor].id;
        return clusters[x.cluster].member_ids.front() < clusters[y.cluster].member_ids.front();
    });

    std::vector<bool> anchor_taken(anchors.size(), false);
    std::vector<bool> cluster_taken(clusters.size(), false);
    for (const auto& cd : cands)
    {
        if (anchor_taken[cd.anchor] or cluster_taken[cd.cluster])
            continue;   // a ring has exactly ONE centre, and a table cannot centre two rings
        anchor_taken[cd.anchor]   = true;
        cluster_taken[cd.cluster] = true;
        Cluster& c       = clusters[cd.cluster];
        c.anchor_id      = anchors[cd.anchor].id;
        c.anchor_index   = cd.anchor;
        c.anchor_logodds = cd.lambda;
        c.anchor_weight  = sigmoid(cd.lambda);
    }
    return clusters;
}

// ─── Association ─────────────────────────────────────────────────────────────

Association associate_clusters(const std::vector<Cluster>& clusters,
                               const std::vector<std::pair<std::uint64_t, std::vector<std::uint64_t>>>& rigs)
{
    Association out;
    out.cluster_to_rig.assign(clusters.size(), -1);

    struct Match { std::size_t cluster; std::size_t rig; std::size_t inter; float jaccard; };
    std::vector<Match> matches;

    for (std::size_t c = 0; c < clusters.size(); ++c)
        for (std::size_t r = 0; r < rigs.size(); ++r)
        {
            std::vector<std::uint64_t> inter;
            std::ranges::set_intersection(clusters[c].member_ids, rigs[r].second, std::back_inserter(inter));
            if (inter.empty())
                continue;
            const std::size_t uni = clusters[c].member_ids.size() + rigs[r].second.size() - inter.size();
            matches.push_back({c, r, inter.size(),
                               static_cast<float>(inter.size()) / static_cast<float>(std::max<std::size_t>(1, uni))});
        }

    // Max-overlap first, Jaccard as tiebreak, rig key last so the result is deterministic. Overlap
    // before Jaccard on purpose: a rig that loses one of four chairs must keep its identity against
    // the survivor cluster, which is the case pure Jaccard gets wrong.
    std::ranges::sort(matches, [&](const Match& x, const Match& y)
    {
        if (x.inter != y.inter)     return x.inter > y.inter;
        if (x.jaccard != y.jaccard) return x.jaccard > y.jaccard;
        return rigs[x.rig].first < rigs[y.rig].first;
    });

    std::vector<bool> cluster_taken(clusters.size(), false);
    std::vector<bool> rig_taken(rigs.size(), false);
    for (const auto& m : matches)
    {
        if (cluster_taken[m.cluster] or rig_taken[m.rig])
            continue;
        cluster_taken[m.cluster] = true;
        rig_taken[m.rig]         = true;
        out.cluster_to_rig[m.cluster] = static_cast<std::int64_t>(rigs[m.rig].first);
    }
    for (std::size_t r = 0; r < rigs.size(); ++r)
        if (not rig_taken[r])
            out.unmatched_rigs.push_back(rigs[r].first);

    return out;
}

// ─── self_test ───────────────────────────────────────────────────────────────

bool partition_self_test()
{
    bool ok = true;
    const auto check = [&ok](bool cond, const char* what)
    {
        if (not cond) { std::print("  [RingPartition::self_test] FAIL: {}\n", what); ok = false; }
    };

    // ── 1. Two well-separated 4-chair rings must NOT merge ───────────────────
    {
        PartitionParams p;
        p.room_area_m2 = 40.0f;
        std::vector<PartitionMember> ring;
        std::uint64_t id = 1;
        for (const float cx : {0.0f, 6.0f})
            for (int k = 0; k < 4; ++k)
            {
                const float a = kPi * 0.5f * static_cast<float>(k);
                ring.push_back({id++, {cx + 0.8f * std::cos(a), 0.8f * std::sin(a)}, 0.02f});
            }
        const auto cl = partition_members(ring, {}, p);
        check(cl.size() == 2, "two rings 6 m apart should give 2 clusters");
        if (cl.size() == 2)
            check(cl[0].member_ids.size() == 4 and cl[1].member_ids.size() == 4,
                  "each of the two clusters should hold 4 members");
    }

    // ── 2. ★THE LIVE REGRESSION (measured 2026-08-11) ────────────────────────
    // Two chairs ringing table_1, and a second table 5.14 m away that the agent used to anchor on
    // purely because get_nodes_by_type() returned it first. The partition must pick table_1, and
    // table_2 must end up attached to nothing.
    {
        PartitionParams p;
        p.room_area_m2 = 40.0f;
        const std::vector<PartitionMember> ring{{101, {-2.58738f, -3.19748f}, 0.0856f},
                                                {102, {-1.79735f, -2.75656f}, 0.0879f}};
        const std::vector<PartitionMember> anchors{{201, { 2.08333f, -0.09413f}, 0.0954f},   // table_2 FIRST
                                                   {202, {-2.32618f, -2.74185f}, 0.0924f}};  // table_1
        const auto cl = partition_members(ring, anchors, p);
        check(cl.size() == 1, "the two live chairs must form ONE cluster");
        if (cl.size() == 1)
        {
            check(cl[0].member_ids.size() == 2, "that cluster must hold both chairs");
            check(cl[0].anchor_id == 202, "the attached anchor must be table_1, not the first-listed table_2");
            check(cl[0].anchor_weight > 0.9f, "table_1's centre prior must be near-full strength");
        }
    }

    // ── 3. A distant anchor must be INERT, not merely rejected ───────────────
    // Same scene with only the wrong table available. It is still attached (there is no accept/reject
    // step) but its weight must be ~0, so the centre prior it contributes cannot move the fit.
    {
        PartitionParams p;
        p.room_area_m2 = 40.0f;
        const std::vector<PartitionMember> ring{{101, {-2.58738f, -3.19748f}, 0.0856f},
                                                {102, {-1.79735f, -2.75656f}, 0.0879f}};
        const std::vector<PartitionMember> anchors{{201, {2.08333f, -0.09413f}, 0.0954f}};
        const auto cl = partition_members(ring, anchors, p);
        check(cl.size() == 1 and cl[0].anchor_weight < 1e-6f,
              "a table 5 m from the chairs must contribute a ~zero-weight centre prior");
    }

    // ── 4. Association survives a flickering member, and a split goes to the bigger half ──
    {
        const auto make = [](std::vector<std::uint64_t> ids)
        { Cluster c; c.member_ids = std::move(ids); return c; };

        const auto a1 = associate_clusters({make({1, 2})}, {{7, {1, 2, 3}}});
        check(a1.cluster_to_rig.size() == 1 and a1.cluster_to_rig[0] == 7,
              "a rig that loses one member must keep its identity, not re-birth");
        check(a1.unmatched_rigs.empty(), "that rig must not be reported unmatched");

        const auto a2 = associate_clusters({make({1, 2, 3, 4})}, {{7, {1, 2}}, {8, {3, 4}}});
        check(a2.cluster_to_rig.size() == 1 and (a2.cluster_to_rig[0] == 7 or a2.cluster_to_rig[0] == 8),
              "a merged cluster must adopt one of the two existing rigs");
        check(a2.unmatched_rigs.size() == 1, "the other rig must be reported unmatched");

        const auto a3 = associate_clusters({make({9})}, {{7, {1, 2}}});
        check(a3.cluster_to_rig.size() == 1 and a3.cluster_to_rig[0] == -1,
              "a cluster sharing no member must be a birth");
        check(a3.unmatched_rigs.size() == 1, "the old rig must then be unmatched");
    }

    // ── 5. Determinism: input order must not change the outcome ──────────────
    {
        PartitionParams p;
        p.room_area_m2 = 40.0f;
        std::vector<PartitionMember> ring{{101, {-2.58738f, -3.19748f}, 0.0856f},
                                          {102, {-1.79735f, -2.75656f}, 0.0879f}};
        std::vector<PartitionMember> anchors{{201, { 2.08333f, -0.09413f}, 0.0954f},
                                             {202, {-2.32618f, -2.74185f}, 0.0924f}};
        const auto before = partition_members(ring, anchors, p);
        std::ranges::reverse(ring);
        std::ranges::reverse(anchors);
        const auto after = partition_members(ring, anchors, p);
        check(before.size() == after.size(), "shuffling the input must not change the cluster count");
        if (before.size() == after.size() and not before.empty())
            check(before[0].member_ids == after[0].member_ids and before[0].anchor_id == after[0].anchor_id,
                  "shuffling the input must not change the members or the attached anchor");
    }

    std::print("[RingPartition::self_test] {}\n", ok ? "PASS" : "FAIL");
    return ok;
}

}  // namespace rc
