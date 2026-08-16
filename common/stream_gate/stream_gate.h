/*
 * common/stream_gate/stream_gate.h — "is my primary input still arriving?", as arithmetic. SHARED, header-only.
 *
 * Every concept agent asks the same three questions about the stream it cannot work without — the retina
 * `masks` node for the object concepts, a LiDAR media plane for room_concept — and each had its own copy of
 * the answer:
 *
 *   ready    the producer is reachable at all                (admission, usable from Waiting)
 *   stalled  no NEW frame for longer than the timeout        (Operating: demote to a local emergency hold)
 *   live     a frame arrived RECENTLY                        (re-admission after a stall)
 *
 * `ready` stays with the agent: it is one call into whatever ingestor that agent owns, and the ingestor types
 * have nothing in common. What is shared is the part that was actually copied — the TIME arithmetic, which is
 * identical everywhere and has two traps in it:
 *
 * ★A NEGATIVE AGE IS NOT A STALL. Before the first frame ever arrives there is no "last frame", so the age
 * reads < 0. Treating that as a huge age makes every agent declare a stall the instant it reaches Operating
 * and demote itself while the producer is still starting up. The grace is measured from OPERATING ENTRY
 * instead, which is the only clock that means anything before the first frame.
 *
 * ★AND `live` IS NOT `not stalled`. A producer whose node persists but has stopped publishing would satisfy a
 * node-exists probe forever, so re-admission demands actual FRESHNESS — and, at cold start, deliberately
 * answers NO until the first frame is really in (age < 0 is not alive), so an agent cannot bounce straight
 * back out of the hold it just entered.
 *
 * Pure: no Qt, no DSR, no ingestor type — just the clock values the caller already has, so it can be reasoned
 * about (and tested) without standing up an agent.
 */

#pragma once

#include <cstdint>

namespace rc::stream
{

// `timeout_ms <= 0` disables the gate: every agent spells that as "the feature is off", so it is answered
// here once rather than re-tested at each call site.
inline bool gate_enabled(int timeout_ms) { return timeout_ms > 0; }

// No NEW frame for longer than the timeout. `age_ms < 0` means no frame has EVER arrived, in which case the
// grace runs from Operating entry (`operating_since_ms`, 0 = not yet entered ⇒ never stalled).
inline bool stalled(std::int64_t age_ms, int timeout_ms,
                    std::int64_t operating_since_ms, std::int64_t now_ms)
{
    if (not gate_enabled(timeout_ms))
        return false;
    if (age_ms < 0)
        return operating_since_ms > 0 and (now_ms - operating_since_ms) > timeout_ms;
    return age_ms > timeout_ms;
}

// A frame arrived recently enough to trust the producer again. Cold start (age < 0) is NOT live.
//
// ⚠Call this only with the gate ENABLED. With it disabled every agent falls back to its own node-exists
// probe (`stream_ready()`), and that probe is the one thing here that cannot be shared — the ingestor types
// have nothing in common. Folding the disabled case in would have quietly changed "gate off ⇒ ask the
// producer" into "gate off ⇒ demand a frame anyway", which is a different agent.
inline bool live(std::int64_t age_ms, int timeout_ms)
{
    return age_ms >= 0 and age_ms < timeout_ms;
}

}  // namespace rc::stream
