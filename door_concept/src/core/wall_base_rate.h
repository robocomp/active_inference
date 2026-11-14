// wall_base_rate.h — the per-wall BASE RATE of a door place, as a function of the OTHER doors' beliefs (one-model, spec §3.4).
// Header-only, DSR-free. Replaces the integer "doors on this wall before me" (which depended on BIRTH ORDER) by a smooth term:
//   m = P(some other door on this wall exists) = 1 − Π(1 − r_j),   r_j = the other door's existence probability (last cycle's p);
//   log prior odds = (1 − m)·logit(E[first]) + m·logit(E[another]).
// A certain other door makes this one an "another" (≈ 0.01), no other door makes it a "first" (≈ 0.4), an uncertain one pays part.
// Nothing depends on order. The two rates are Beta beliefs (seeds below; Task 4 learns them from the remembered rooms).
#pragma once
#include <algorithm>
#include <cmath>
#include <numbers>
#include <span>

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

}   // namespace rc::door::core
