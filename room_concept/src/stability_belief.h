#pragma once
// Stability BELIEF for the ordinary room-birth path (replaces "N CONSECUTIVE stable frames").
//
// A counter that resets to 0 on one unstable frame never births a room whose pose measure flaps around
// its bar (gold7: 0->10->1; gold10: 0-6 for 265 s), while a lucky run of N births it. Both are the same
// room. Here each frame adds the log-likelihood ratio its boolean stability measure carries, the sum
// forgets geometrically toward the prior (so ancient history cannot vouch for now), and a single flap
// costs one frame's evidence, never the whole accumulation.
//
// Model: a frame reads "stable" with probability p_s if the room is stable and p_u if it is not.
//   LLR(stable frame)   = ln(p_s/p_u)            > 0
//   LLR(unstable frame) = ln((1-p_s)/(1-p_u))    < 0
// Scale: the evidence is rescaled so that N consecutive stable frames from the prior reach the decision
// boundary, so STABLE_FRAMES_REQUIRED keeps its meaning as "how much clean evidence is enough".
// ONE DECISION BOUNDARY (flagged): posterior >= 0.99, i.e. a false birth costs 99x a late one. Births
// are expensive to undo (reparenting the robot), so this is a loss ratio, not a tuned cutoff.
// Forgetting horizon: 4N frames (lambda = 1 - 1/(4N)).
#include <algorithm>
#include <cmath>

namespace rc
{
class StabilityBelief
{
public:
    explicit StabilityBelief(int frames_required = 30, double p_stable = 0.8, double p_unstable = 0.4,
                             double posterior = 0.99)
    { configure(frames_required, p_stable, p_unstable, posterior); }

    void configure(int frames_required, double p_stable = 0.8, double p_unstable = 0.4, double posterior = 0.99)
    {
        const double n = std::max(1, frames_required);
        boundary_ = std::log(posterior / (1.0 - posterior));
        lambda_   = 1.0 - 1.0 / (4.0 * n);
        const double raw_pos = std::log(p_stable / p_unstable);
        const double raw_neg = std::log((1.0 - p_stable) / (1.0 - p_unstable));
        // n clean frames from the prior must land ON the boundary: g * sum_{k<n} lambda^k = boundary.
        const double geom = (1.0 - std::pow(lambda_, n)) / (1.0 - lambda_);
        const double g = boundary_ / geom;
        gain_pos_ = g;
        gain_neg_ = g * (raw_neg / raw_pos);
        log_odds_ = 0.0;
    }

    void reset() noexcept { log_odds_ = 0.0; }

    /// One frame's verdict. Returns true when the posterior is decisive.
    bool update(bool stable) noexcept
    {
        log_odds_ = lambda_ * log_odds_ + (stable ? gain_pos_ : gain_neg_);
        return decisive();
    }

    [[nodiscard]] bool   decisive() const noexcept { return log_odds_ >= boundary_ - 1e-9; }
    [[nodiscard]] double log_odds() const noexcept { return log_odds_; }
    [[nodiscard]] double posterior() const noexcept { return 1.0 / (1.0 + std::exp(-log_odds_)); }

private:
    double boundary_ = 4.595, lambda_ = 0.99, gain_pos_ = 0.15, gain_neg_ = -0.24, log_odds_ = 0.0;
};
}  // namespace rc
