#pragma once
/*
 * common/epistemic_step/offer_cache.h — compute an instance's offer once per belief change, and not during a cooldown.
 *
 * WHY (engine1 diagnosis item 3, 2026-11-24): the countable offer ran on EVERY compute tick for every instance — four
 * PersistentEngine::predicted_gain calls (eight when the best standpoint is blocked), ~6 ms each for a bed — and it ran during
 * the post-completion cooldown too, where epistemic::step zeroes the published gain AFTER compute. The bed published gain 0 on
 * 97 % of its rows and still paid 25 ms per tick for it.
 *
 * The offer is a function of the belief (θ, Σ and the latents) and of the world the standpoints are checked against. The
 * belief changes when the engine updates (its update serial bumps); a predict-only tick changes Σ only through Q. So:
 *   - recompute when the belief's update serial or the standing world's version changed, else hold the last proposal;
 *   - during a cooldown hold the last proposal whatever changed: its gain is suppressed (or floored) by epistemic::step anyway,
 *     and the look is not wanted yet. With nothing held yet there is nothing to hold, so it computes.
 * A compute that returns no proposal holds nothing (the next tick asks again): "no proposal" is not cached as an answer.
 *
 * Not a threshold: a cache key. What a held proposal does not see is the predict-only growth of Σ between updates (Q), which
 * moves the promise only by what Q adds; that is stated, not hidden.
 */

#include <cstdint>
#include <optional>
#include <utility>

namespace rc::epistemic
{

template <class Proposal>
struct OfferCache
{
    std::optional<Proposal> held;     // the last computed proposal
    long          serial = 0;         // the belief update serial it was computed at
    std::uint64_t world  = 0;         // the standing world's version it was computed at
    long          computes = 0;       // how many times compute() ran (diagnostic, tests)

    // `serial_now`: the belief's update serial; `world_now`: the version of what the standpoints are checked against;
    // `cooldown`: the instance is in its post-completion hold. `compute() -> std::optional<Proposal>`.
    template <class Compute>
    std::optional<Proposal> get(long serial_now, std::uint64_t world_now, bool cooldown, Compute&& compute)
    {
        if (held.has_value() and (cooldown or (serial_now == serial and world_now == world)))
            return held;
        ++computes;
        held = std::forward<Compute>(compute)();
        serial = serial_now;
        world  = world_now;
        return held;
    }
    void reset() { held.reset(); }
};

}  // namespace rc::epistemic
