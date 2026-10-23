// door_existence_info.h — the information a ZED look buys about a door's EXISTENCE (pure; std only).
// Existence E ~ Bernoulli(p_exists) is a variable of the door's belief like its along-wall offset or width; one detector outcome
// Y carries I(E;Y) nats about it. It is part of the epistemic gain (EpistemicPlanner) and of the remembered-door look (door_look.h):
// the value of looking is high while existence is open (a 360-only birth sits at p ~ 0.7) and ZERO once it is settled either way -
// no rule 'ricoh-born => look' exists anywhere, the offer simply has nothing left to buy when the ZED has confirmed or removed it.
#pragma once
#include <algorithm>
#include <cmath>

namespace rc::door
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
}  // namespace rc::door
