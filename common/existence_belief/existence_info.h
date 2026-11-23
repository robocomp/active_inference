// common/existence_belief/existence_info.h — the information ONE look buys about an object's EXISTENCE (pure; std only).
//
// Existence E ~ Bernoulli(p_exists) is a variable of every object belief, like its pose or extent; one detector outcome
// Y ∈ {fires, silent} carries I(E;Y) nats about it. The value of looking is high while existence is open and ZERO once
// it is settled either way — no "verify me" rule is needed: the offer simply has nothing left to buy when a view has
// confirmed or removed the object. Bounded by H(E) ≤ ln 2 nats.
//
// Moved verbatim from door_concept/src/door_existence_info.h (which now forwards here) so every concept agent prices
// existence the same way. Owner ruling 2026-11-23: this replaces the 5-nat verification FLOOR on object epistemic
// offers (common/nbv/viewpoint_score.h adds it per candidate view, with P(detect) entering through the hit rate).
#pragma once
#include <algorithm>
#include <cmath>

namespace rc::exist
{
inline float binary_entropy_nats(float p)
{
    const auto h = [](float q) { return q > 0.f ? -q * std::log(q) : 0.f; };
    p = std::clamp(p, 0.f, 1.f);
    return h(p) + h(1.f - p);
}
/// I(E;Y) for E ~ Bernoulli(p_exists), Y = detector fires, P(Y|E=1) = p_hit, P(Y|E=0) = p_false.
inline float existence_mutual_info(float p_exists, float p_hit, float p_false)
{
    const float p  = std::clamp(p_exists, 0.f, 1.f);
    const float py = p * std::clamp(p_hit, 0.f, 1.f) + (1.f - p) * std::clamp(p_false, 0.f, 1.f);
    return std::max(0.f, binary_entropy_nats(py) - p * binary_entropy_nats(p_hit) - (1.f - p) * binary_entropy_nats(p_false));
}
}  // namespace rc::exist
