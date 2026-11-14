// bernoulli_filter.h — the labelled-Bernoulli arithmetic the door agent's one model is written in (spec §3.1/§3.2/§4).
// Header-only; Eigen and std ONLY (no door, DSR, Qt or OpenCV types): it is meant to move to common/ and be compared with the
// objects session's version. Four small pieces:
//   update_logodds          the exact Bernoulli existence update with clutter, marginalised over the masks' associations;
//   responsibilities        normalised weights of one mask's causes (log-sum-exp, stable);
//   k_best                  the k best joint assignments of masks to causes (exact enumeration, branch and bound);
//   expected_information_gain  discrete EIG of a look over hypotheses with weights w and observation model p(z | h).
#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <vector>

namespace rc::door::core::bf
{

/// Exact Bernoulli update in log-odds: L' = L + log(1 − p_d + p_d · S), S = Σ_m ρ_m · g_m / κ_m.
/// (Odds r'/(1−r') = r/(1−r) · (1 − p_d + p_d S): the denominator's S terms cancel, so the update is additive in L.)
/// p_d = 0 (not looked at, occluded) is the identity; S = 0 is a miss priced by p_d.
[[nodiscard]] inline double update_logodds(double L, double p_d, double S)
{
    return L + std::log(1.0 - p_d + p_d * S);
}

/// Normalised responsibilities from log weights (log-sum-exp, stable); sums to 1. −inf entries get 0.
[[nodiscard]] inline std::vector<double> responsibilities(std::span<const double> log_w)
{
    std::vector<double> w(log_w.size(), 0.0);
    double mx = -std::numeric_limits<double>::infinity();
    for (const double v : log_w)
        if (not std::isnan(v)) mx = std::max(mx, v);   // a NaN weight is no weight (explicit: it used to be dropped only by the order of std::max)
    if (not std::isfinite(mx))
        return w;   // nothing is possible: no responsibility anywhere (callers always include clutter, so this is a corner)
    double z = 0.0;
    for (std::size_t i = 0; i < log_w.size(); ++i)
        z += (w[i] = std::isnan(log_w[i]) ? 0.0 : std::exp(log_w[i] - mx));
    for (double& v : w)
        v /= z;
    return w;
}

struct Joint
{
    std::vector<int> cause_of_mask;
    double log_w;
};

/// k best joint assignments of masks to causes. log_w(m, c) = log weight of mask m under cause c (−inf = impossible).
/// `exclusive[c]` = cause c explains at most one mask (a door per sensor); non-exclusive (new door, clutter) any number.
/// Exact: depth-first over masks with an additive upper bound (the best remaining row maxima) against the k-th best so far.
/// Sorted by descending log_w. Fewer than k are returned when fewer assignments are possible.
[[nodiscard]] inline std::vector<Joint> k_best(const Eigen::MatrixXd& log_w, std::span<const bool> exclusive, int k)
{
    const int M = static_cast<int>(log_w.rows()), C = static_cast<int>(log_w.cols());
    std::vector<Joint> best;
    if (k <= 0)
        return best;
    if (M == 0)
        return {Joint{{}, 0.0}};
    // per mask: causes in descending weight; bound[m] = Σ_{i ≥ m} row max
    std::vector<std::vector<int>> order(M);
    std::vector<double> bound(M + 1, 0.0);
    for (int m = 0; m < M; ++m)
    {
        order[m].resize(C);
        for (int c = 0; c < C; ++c)
            order[m][c] = c;
        std::ranges::stable_sort(order[m], [&](int a, int b) { return log_w(m, a) > log_w(m, b); });
    }
    for (int m = M - 1; m >= 0; --m)
        bound[m] = bound[m + 1] + (C > 0 ? log_w(m, order[m][0]) : -std::numeric_limits<double>::infinity());
    std::vector<int> cur(M, -1);
    std::vector<char> used(C, 0);
    auto kth = [&]() { return static_cast<int>(best.size()) < k ? -std::numeric_limits<double>::infinity() : best.back().log_w; };
    auto rec = [&](auto&& self, int m, double acc) -> void
    {
        if (m == M)
        {
            Joint j{cur, acc};
            const auto at = std::ranges::upper_bound(best, acc, std::ranges::greater{}, &Joint::log_w);
            best.insert(at, std::move(j));
            if (static_cast<int>(best.size()) > k)
                best.pop_back();
            return;
        }
        for (const int c : order[m])
        {
            const double lw = log_w(m, c);
            if (not std::isfinite(lw))
                break;   // sorted descending: the rest is impossible too
            if (acc + lw + bound[m + 1] < kth())
                break;   // sorted descending: no later cause of this mask can reach the k-th best
            if (exclusive[c] and used[c])
                continue;
            cur[m] = c;
            if (exclusive[c])
                used[c] = 1;
            self(self, m + 1, acc + lw);
            if (exclusive[c])
                used[c] = 0;
        }
    };
    rec(rec, 0, 0.0);
    return best;
}

namespace detail
{
[[nodiscard]] inline double entropy(std::span<const double> p)
{
    double h = 0.0;
    for (const double v : p)
        if (v > 0.0)
            h -= v * std::log(v);
    return h;
}
}   // namespace detail

/// Discrete EIG: H(w) − Σ_z p(z) H(w | z), p_z_given_h[h][z].
[[nodiscard]] inline double expected_information_gain(std::span<const double> w, const std::vector<std::vector<double>>& p_z_given_h)
{
    const std::size_t H = w.size();
    if (H == 0 or p_z_given_h.size() != H)
        return 0.0;
    const std::size_t Z = p_z_given_h[0].size();
    double cond = 0.0;
    std::vector<double> post(H);
    for (std::size_t z = 0; z < Z; ++z)
    {
        double pz = 0.0;
        for (std::size_t h = 0; h < H; ++h)
            pz += w[h] * p_z_given_h[h][z];
        if (not (pz > 0.0))
            continue;   // an observation no hypothesis predicts carries no weight
        for (std::size_t h = 0; h < H; ++h)
            post[h] = w[h] * p_z_given_h[h][z] / pz;
        cond += pz * detail::entropy(post);
    }
    return detail::entropy(w) - cond;
}

}   // namespace rc::door::core::bf
