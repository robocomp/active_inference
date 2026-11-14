// wall_base_rate.h — the per-wall BASE RATE of a door place, as a function of the OTHER doors' beliefs (one-model, spec §3.4).
// Header-only, DSR-free. Replaces the integer "doors on this wall before me" (which depended on BIRTH ORDER) by a smooth term:
//   m = P(some other door on this wall exists) = 1 − Π(1 − r_j),   r_j = the other door's existence probability (last cycle's p);
//   log prior odds = (1 − m)·logit(E[first]) + m·logit(E[another]).
// A certain other door makes this one an "another" (≈ 0.01), no other door makes it a "first" (≈ 0.4), an uncertain one pays part.
// Nothing depends on order. The two rates are Beta beliefs (seeds below; Task 4 learns them from the remembered rooms).
#pragma once
#include <cmath>
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
