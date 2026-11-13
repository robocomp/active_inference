// room_door_count.h — A ROOM HAS DOORS (spec rt-build docs/superpowers/specs/2026-11-10-room-door-belief-design.md, Part A).
//
// The places door_concept holds for a room (wall seeds, mask candidates, outline doorways, born doors) each carry their own
// existence evidence. Read one by one they cannot say "this room has a door somewhere": a start room whose only mask is a
// phantom (clean27, clean34) has no place above the line, and the tour ended DONE. The count of doors couples them:
//
//     P(d_1..d_K | evidence) ∝ [ P(N = n) / C(K, n) ] · Π_i O_i^{d_i},     n = Σ d_i,   O_i = exp(L_i − base_i)
//
// Given n doors, which n of the K places hold them is a priori uniform: the configuration prior is P(N = n) / C(K, n).
// (Without the 1/C(K, n) the combinatorics swamp the count prior: 14 places with no evidence put the room's door count at
// the prior's ceiling — the first draft of this header did exactly that.) With no evidence, marg[i] = E[N] / K.
//
// O_i is the place's own odds with the per-wall base rate taken out (the count prior replaces it, so the door number is
// not counted twice); band, size, carried, crossing, the ledger, Sp and saturation stay with the place. P(N) has
// P(N = 0) = 0: a room that can be entered has a door (the spec's one structural assumption, flagged there §6.1).
//
// Each place's door mass is split by f_i ∈ [0,1], the weight with which it is a door the drive can already act on (a born
// door with an offered approach, the entry door: 1; Part B: the door's published P(exists)). The posterior gives
//   p_more  = P(Σ_i d_i (1 − f_i) ≥ 1)  — a door the drive cannot act on yet
//   marg[i] = P(d_i = 1)                 — the place's marginal under the count prior (the look planner's p'_i)
//
// Pure: no DSR, no Qt. Exact DP over places in log space, O(K · n_max); n_max = the count prior's support.
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <vector>

namespace rc::door::core
{

struct DoorCountPlace
{
    float log_odds = 0.f;   ///< ln O_i = L_i − base_i
    float found = 0.f;      ///< f_i ∈ [0,1]
};

struct DoorCountPosterior
{
    double p_more = 0.0;                ///< P(an unfound door exists)
    std::vector<double> marg;           ///< P(d_i = 1), per place
    std::vector<double> p_count;        ///< P(N = n | evidence), n = 0..n_max
};

/// The flagged initial count prior (spec §6.3): geometric on n ≥ 1 with mean `mean_n` (> 1), truncated at n_max and
/// renormalised; P(N = 0) = 0. Replaced by memory's counts once memory holds rooms (count_prior_from_rooms).
[[nodiscard]] inline std::vector<double> geometric_count_prior(double mean_n, int n_max)
{
    std::vector<double> p(static_cast<std::size_t>(std::max(n_max, 1)) + 1, 0.0);
    const double r = std::clamp(1.0 - 1.0 / std::max(mean_n, 1.0 + 1e-9), 0.0, 1.0 - 1e-12);   // P(N=n) ∝ r^(n-1), mean 1/(1-r)
    double s = 0.0;
    for (std::size_t n = 1; n < p.size(); ++n) { p[n] = std::pow(r, double(n - 1)); s += p[n]; }
    for (auto& x : p) x /= s;
    return p;
}

/// Empirical count prior from remembered rooms' door counts (each ≥ 1), smoothed by the geometric prior as a pseudo-count of
/// one room (so two rooms do not make a never-seen count impossible). Counts of 0 are ignored (a room is not remembered
/// doorless; it was remembered before its doors were found).
[[nodiscard]] inline std::vector<double> count_prior_from_rooms(std::span<const int> counts, double mean_n, int n_max)
{
    auto p = geometric_count_prior(mean_n, n_max);
    double total = 1.0;
    for (const int c : counts)
        if (c >= 1 and c <= n_max) { p[static_cast<std::size_t>(c)] += 1.0; total += 1.0; }
    for (auto& x : p) x /= total;
    return p;
}

namespace detail
{
inline double lse(double a, double b)
{
    if (a == -std::numeric_limits<double>::infinity()) return b;
    if (b == -std::numeric_limits<double>::infinity()) return a;
    const double m = std::max(a, b);
    return m + std::log(std::exp(a - m) + std::exp(b - m));
}
// A[k][n] = log Σ over configurations of places in `idx` with n doors in total, of Π O_i^{d_i}, where k ∈ {0,1} says
// whether at least one of those doors is UNFOUND (each door's mass split f / 1−f). Returned as two vectors (k = 0, 1).
inline void forward(std::span<const DoorCountPlace> pl, std::span<const std::size_t> idx, int n_max,
                    std::vector<double>& a0, std::vector<double>& a1)
{
    const double ninf = -std::numeric_limits<double>::infinity();
    a0.assign(static_cast<std::size_t>(n_max) + 1, ninf);
    a1.assign(static_cast<std::size_t>(n_max) + 1, ninf);
    a0[0] = 0.0;
    for (const std::size_t i : idx)
    {
        const double lo = double(pl[i].log_odds);
        const double f = std::clamp(double(pl[i].found), 0.0, 1.0);
        const double lf = f > 0.0 ? std::log(f) + lo : ninf;          // a door, found
        const double lu = f < 1.0 ? std::log1p(-f) + lo : ninf;       // a door, unfound
        std::vector<double> b0(a0.size(), ninf), b1(a1.size(), ninf);
        for (std::size_t n = 0; n < a0.size(); ++n)
        {
            b0[n] = lse(b0[n], a0[n]);                                // not a door
            b1[n] = lse(b1[n], a1[n]);
            if (n + 1 < a0.size())
            {
                b0[n + 1] = lse(b0[n + 1], a0[n] + lf);               // found door: the unfound flag is unchanged
                b1[n + 1] = lse(b1[n + 1], a1[n] + lf);
                b1[n + 1] = lse(b1[n + 1], lse(a0[n], a1[n]) + lu);   // unfound door: the flag becomes 1
            }
        }
        a0.swap(b0); a1.swap(b1);
    }
}
}   // namespace detail

/// The posterior. `prior` = P(N = n), n = 0..n_max (its size sets n_max; P(N > n_max) is taken as 0). Places with a
/// non-finite log-odds are clamped to the double range (a crossed door: +inf → certain). Exact; the marginals by leaving
/// each place out of the DP (O(K² · n_max), K ≤ ~40 in a room).
[[nodiscard]] inline DoorCountPosterior room_door_posterior(std::span<const DoorCountPlace> places, std::span<const double> prior)
{
    DoorCountPosterior out;
    const int n_max = static_cast<int>(prior.size()) - 1;
    out.marg.assign(places.size(), 0.0);
    out.p_count.assign(prior.size(), 0.0);
    if (n_max < 0) return out;
    std::vector<DoorCountPlace> pl(places.begin(), places.end());
    for (auto& p : pl)
        if (not std::isfinite(p.log_odds)) p.log_odds = std::isnan(p.log_odds) ? 0.f : (p.log_odds > 0 ? 700.f : -700.f);
    // log of the per-configuration prior P(N = n) / C(K, n); a count above K is impossible
    const double K = double(pl.size());
    std::vector<double> lp(prior.size());
    for (std::size_t n = 0; n < prior.size(); ++n)
        lp[n] = (prior[n] > 0.0 and double(n) <= K)
                    ? std::log(prior[n]) - (std::lgamma(K + 1.0) - std::lgamma(double(n) + 1.0) - std::lgamma(K - double(n) + 1.0))
                    : -std::numeric_limits<double>::infinity();

    std::vector<std::size_t> all(pl.size());
    for (std::size_t i = 0; i < all.size(); ++i) all[i] = i;
    std::vector<double> a0, a1;
    detail::forward(pl, all, n_max, a0, a1);
    double lz = -std::numeric_limits<double>::infinity(), lmore = lz;
    std::vector<double> lcount(prior.size(), lz);
    for (std::size_t n = 0; n < prior.size(); ++n)
    {
        lcount[n] = detail::lse(a0[n], a1[n]) + lp[n];
        lz = detail::lse(lz, lcount[n]);
        lmore = detail::lse(lmore, a1[n] + lp[n]);
    }
    if (not std::isfinite(lz)) return out;   // no configuration the prior allows (e.g. every place certainly absent): nothing to say
    out.p_more = std::exp(lmore - lz);
    for (std::size_t n = 0; n < prior.size(); ++n) out.p_count[n] = std::exp(lcount[n] - lz);
    // marginal of place i: configurations with d_i = 1 = (rest with n−1 doors) · O_i
    std::vector<std::size_t> rest; rest.reserve(pl.size());
    for (std::size_t i = 0; i < pl.size(); ++i)
    {
        rest.clear();
        for (std::size_t j = 0; j < pl.size(); ++j) if (j != i) rest.push_back(j);
        std::vector<double> r0, r1;
        detail::forward(pl, rest, n_max, r0, r1);
        double l1 = -std::numeric_limits<double>::infinity();
        for (std::size_t n = 1; n < prior.size(); ++n) l1 = detail::lse(l1, detail::lse(r0[n - 1], r1[n - 1]) + double(pl[i].log_odds) + lp[n]);
        out.marg[i] = std::exp(l1 - lz);
    }
    return out;
}

/// The value of a look, in seconds (spec §3.4): C_miss · P(this look finds a door the drive cannot act on yet). `marg` is
/// the place's posterior marginal, `found` its f, `hit` the camera's hit rate, `p_detect` the standpoint's envelope ×
/// visibility (nbv::score). Visibility enters once, inside p_detect.
[[nodiscard]] inline double look_value_s(double c_miss_s, double marg, double found, double hit, double p_detect)
{
    return c_miss_s * std::clamp(marg, 0.0, 1.0) * (1.0 - std::clamp(found, 0.0, 1.0)) * std::clamp(hit, 0.0, 1.0) * std::clamp(p_detect, 0.0, 1.0);
}

}   // namespace rc::door::core
