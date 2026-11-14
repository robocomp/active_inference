// wall_base_rate.h — the per-wall BASE RATE of a door place, as a function of the OTHER doors' beliefs (one-model, spec §3.4).
// Header-only, DSR-free. Replaces the integer "doors on this wall before me" (which depended on BIRTH ORDER) by a smooth term:
//   m = P(some other door on this wall exists) = 1 − Π(1 − r_j),   r_j = the other door's existence probability (last cycle's p);
//   log prior odds = (1 − m)·logit(E[first]) + m·logit(E[another]).
// A certain other door makes this one an "another" (≈ 0.01), no other door makes it a "first" (≈ 0.4), an uncertain one pays part.
// Nothing depends on order. The two rates are Beta beliefs (seeds below; Task 4 learns them from the remembered rooms).
#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <numeric>
#include <span>
#include <vector>
#include "room_door_count.h"

namespace rc::door::core
{

struct BetaBelief
{
    double a, b;
    [[nodiscard]] double mean() const { return a / (a + b); }
};

/// Seeds: Beta(2,3) mean 0.4 = a first door on a wall; Beta(0.05,4.95) mean 0.01 = a further one.
struct WallRates
{
    BetaBelief first{2.0, 3.0};
    BetaBelief another{0.05, 4.95};
};

/// How DISTINCT two places are: 1 - exp(-m^2/2), m^2 = squared separation over the summed position variances. Two labels on one hole
/// (m -> 0) are one door and charge each other nothing; two real doors (m >> 1) charge the full "another door" cost (S3: competition, no radius).
[[nodiscard]] inline float place_distinctness(double sq_dist, double var_sum)
{
    return static_cast<float>(-std::expm1(-0.5 * sq_dist / std::max(1e-9, var_sum)));
}

/// RESTORE (S10): a place is EXPLAINED by a live door when the door's cause takes the place's likelihood against "a new door at that
/// place" (the MAP comparison of the masks' responsibilities, same densities along the wall as Task 8): log p N(m; 0, var_sum) >= log b - log L,
/// m^2 the squared separation, var_sum the two position variances, `log_new` = log b - log L of a new place there. No radius.
[[nodiscard]] inline bool place_explained_by(double p_live, double sq_dist, double var_sum, double log_new)
{
    const double v = std::max(1e-9, var_sum);
    return std::log(std::max(1e-9, p_live)) - 0.5 * sq_dist / v - 0.5 * std::log(2.0 * std::numbers::pi * v) >= log_new;
}

/// Two places are ONE place when "one place" is the MAP reading of their separation: the same-place weight exp(-m^2/2) beats the
/// distinct weight 1 - exp(-m^2/2) (equal priors; place_distinctness < 1/2). Used where no wall (hence no log b - log L) is at hand.
[[nodiscard]] inline bool same_place(double sq_dist, double var_sum)
{
    return place_distinctness(sq_dist, var_sum) < 0.5f;
}

/// log prior odds of "this wall has this door" when the other doors on its wall exist with probabilities `r_others`.
[[nodiscard]] inline float wall_base_rate(const WallRates& rates, std::span<const float> r_others)
{
    double none = 1.0;   // Π(1 − r_j): no other door exists
    for (const float r : r_others)
        none *= 1.0 - static_cast<double>(r);
    const double m = 1.0 - none;
    auto logit = [](double p) { return std::log(p / (1.0 - p)); };
    return static_cast<float>((1.0 - m) * logit(rates.first.mean()) + m * logit(rates.another.mean()));
}


/// ONE-MODEL, THE EXACT PER-WALL CHARGE (final review 1). The members of one wall (live doors, pending candidates, wall seeds), each with
/// the log odds of its OWN evidence (ln O_i = L_i with the base rate taken out: band, size, extent, ledger, carried, crossing), and the slot each
/// belongs to: labels on ONE hole (same_place) are ONE slot, alternatives of each other, so a slot's odds are the sum of its members' (a hole
/// holds one door). The wall's configuration prior is the per-wall rates' own: no door 1, a first door odds P(first)/(1-P(first)), each further
/// door odds P(another)/(1-P(another)) (the flag-off rule, as a prior over which doors the wall holds). The posterior marginal of every slot is
/// the exact count DP of room_door_count.h (room_door_posterior), computed JOINTLY in one pass: nothing reads a previous cycle's p, there is no
/// iteration, nothing to damp or oscillate, and the answer does not depend on birth order. The returned value of a member is the base-rate
/// term that makes its L the logit of its own marginal: logit(marg_i) - ln O_i. One member alone gets logit(P(first)), as before.
///   `log_odds[i]`  ln O_i of member i;  `slot[i]`  its slot (0..n_slots-1, every slot used).
/// The returned vector has one base-rate log-odds per member.
[[nodiscard]] inline std::vector<double> wall_count_base_rates(const WallRates& rates, std::span<const double> log_odds, std::span<const int> slot)
{
    const double ninf = -std::numeric_limits<double>::infinity();
    auto logit = [](double p) { return std::log(p / (1.0 - p)); };
    const double first = logit(rates.first.mean());
    std::vector<double> out(log_odds.size(), first);   // the one-member answer, also what a non-finite result falls back to
    if (log_odds.empty()) return out;
    const int n_slots = 1 + *std::ranges::max_element(slot);
    std::vector<double> slot_lo(static_cast<std::size_t>(n_slots), ninf);   // ln of the slot's summed odds
    for (std::size_t i = 0; i < log_odds.size(); ++i)
        slot_lo[static_cast<std::size_t>(slot[i])] = detail::lse(slot_lo[static_cast<std::size_t>(slot[i])], log_odds[i]);
    std::vector<DoorCountPlace> places(slot_lo.size());
    for (std::size_t s = 0; s < places.size(); ++s)
        places[s] = {static_cast<float>(std::clamp(slot_lo[s], -700.0, 700.0)), 0.0f};   // clamp: the DP's own float/double range (room_door_posterior clamps the same way)
    // configuration prior P(config) ∝ w(n); room_door_posterior takes P(N = n) and divides by C(K, n): hand it w(n)·C(K, n)
    const std::size_t K = places.size();
    const double lw_first = first, lw_another = logit(rates.another.mean());
    std::vector<double> prior(K + 1, 0.0);
    double lw = 0.0;   // ln w(0)
    for (std::size_t n = 0; n <= K; ++n)
    {
        if (n == 1) lw = lw_first;
        else if (n > 1) lw += lw_another;
        const double lc = std::lgamma(double(K) + 1.0) - std::lgamma(double(n) + 1.0) - std::lgamma(double(K - n) + 1.0);
        prior[n] = std::exp(lw + lc);   // unnormalised (room_door_posterior normalises)
    }
    const double top = *std::ranges::max_element(prior);
    for (auto& x : prior) x /= top;
    const auto post = room_door_posterior(places, prior);
    for (std::size_t i = 0; i < log_odds.size(); ++i)
    {
        const std::size_t s = static_cast<std::size_t>(slot[i]);
        const double marg_slot = std::clamp(post.marg[s], std::numeric_limits<double>::min(), 1.0);   // numerical floor on a log
        const double lshare = log_odds[i] - slot_lo[s];                                              // ln(O_i / O_slot): this label's share of its hole
        const double marg_i = std::exp(std::log(marg_slot) + lshare);
        const double b = std::log(marg_slot) - slot_lo[s] - std::log1p(-std::min(marg_i, 1.0 - 1e-16));   // logit(marg_i) - ln O_i; 1e-16: numerical floor
        if (std::isfinite(b)) out[i] = b;
    }
    return out;
}

}   // namespace rc::door::core
