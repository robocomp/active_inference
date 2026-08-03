/*
 * birth_evidence.h — how much is ONE observation worth toward creating an instance? (shared, header-only)
 *
 * The CREATE counterpart of common/existence_belief/existence_belief.h. `InstanceTracker` decides WHEN a
 * candidate is promoted (streak ≥ birth_frames); this decides what each frame CONTRIBUTES to that streak.
 * Every concept agent must use it, so birth is one policy across the fleet rather than per-agent judgement —
 * see CONCEPT_AGENT_LIFECYCLE.md §7.1 ("common/instance_tracker for CREATE; nothing else may birth").
 *
 * Two rules, both structural — neither is a threshold, and neither has a value to tune:
 *
 * 1. AN OBSERVATION, NOT A CYCLE. Agents feed the tracker every compute cycle on purpose: a pending candidate
 *    that finds no matching detection EXPIRES, so skipping stale cycles would wipe every candidate and nothing
 *    would ever birth. But persisting a candidate and accruing evidence into it are different things, and
 *    conflating them made `birth_frames` count compute cycles. At ~10 Hz compute against a ~9.5 Hz mask stream
 *    one mask frame was counted several times, so "8 frames" was well under a second of a single unchanging
 *    view — which is how a YOLO false positive on a wall panel becomes furniture. Pass new_observation=false on
 *    a repeat frame_id: the candidate stays alive, its streak does not grow. This is the CREATE-side instance of
 *    "repetition is not independence" (see existence_belief.h for the REMOVE-side one).
 *
 * 2. BIRTH IS ADMITTED BY THE UPDATE RULE. ★A frame that may not MOVE an existing belief must not CREATE one.
 *    Agents already have a strict admissibility predicate for geometric updates — untruncated mask, robot
 *    still (or the mask well centred), the fit's own trunc/confirm-only gate. Birth must pass THAT SAME
 *    predicate, not a second, weaker set of conditions invented for this path. The caller evaluates it (only
 *    the agent knows its own fit gate) and passes `admissible`; false contributes 0. This is what stops
 *    furniture appearing in passing, from a smeared mask taken while the robot drives by.
 *
 * 3. AN ADMISSIBLE OBSERVATION IS STILL WORTH ONLY ITS RELIABILITY. Unit = ONE IDEAL observation. Two factors
 *    admissibility does NOT cover, both continuous, neither a cut-off:
 *      confidence — the detector's own P(class|mask). A 0.53 blob is worth half of a 0.90 one, so it needs
 *                   twice the observations. Admissibility says the FRAME is usable; it says nothing about
 *                   whether the detector believes the LABEL.
 *      range      — the SAME detectability curve the removal side uses for absence, (ref/range)^power. One
 *                   sensor model applied symmetrically to birth and to death, not a second notion of "too far"
 *                   invented here. A mask at 7 m subtends ~8% of the pixels it would at 2.5 m; it cannot
 *                   support CREATING an object, though it still refines a known one.
 *    Truncation and centredness are deliberately NOT weighted here — they are CONDITIONS in rule 2, and
 *    weighting them again would double-count the same physics.
 *
 * ★This scales BIRTH only. Association/refinement of an EXISTING instance keeps using every mask at full
 * weight, so a poor or distant view of a known object still updates it normally.
 *
 * Pure: no DSR, no Eigen, no mask type — the caller passes the four numbers its slice already carries, so this
 * stays unit-testable and agents with a different detector can feed it too.
 */

#pragma once

#include <algorithm>
#include <cmath>

namespace rc::birth
{

// The sensor's detectability model. Defaults MATCH the removal side's absence weighting on purpose: an object
// the sensor cannot resolve well enough to delete is one it cannot resolve well enough to create.
struct Detectability
{
    float periph_ref   = 0.50f;   // normalised centroid radius at which the off-axis penalty reaches ½
    float range_ref_m  = 2.5f;    // range below which a detection is trusted at full weight
    float range_power  = 2.0f;    // decay exponent (2 ≈ angular area ∝ 1/range²); 0 disables the range term
};

// The reliability fields of an ADMISSIBLE mask (rule 3). Truncation/centredness are absent on purpose — they
// are conditions in rule 2, checked by the agent's own update-admissibility predicate.
struct MaskQuality
{
    float confidence = 1.0f;   // detector P(class | mask)
    float range_m    = 0.0f;   // mean camera→mask depth (m); ≤0 ⇒ range term inactive
};

// What one admissible mask is worth toward a birth, in units of an ideal observation. ∈ [0, 1].
inline float observation_weight(const MaskQuality& q, const Detectability& d)
{
    const float w_conf  = std::clamp(q.confidence, 0.0f, 1.0f);
    const float w_range = (d.range_ref_m > 0.0f and d.range_power > 0.0f and q.range_m > 1e-3f)
                        ? std::min(1.0f, std::pow(d.range_ref_m / q.range_m, d.range_power))
                        : 1.0f;
    return w_conf * w_range;
}

// The full CREATE-side per-frame evidence. Zero unless this is a NEW observation (rule 1) that the agent's own
// UPDATE-admissibility predicate accepts (rule 2); otherwise the mask's reliability (rule 3). Multiply the
// agent's corroboration/plausibility/envelope factors into the result.
inline float evidence(const MaskQuality& q, const Detectability& d, bool new_observation, bool admissible)
{
    return (new_observation and admissible) ? observation_weight(q, d) : 0.0f;
}

}  // namespace rc::birth
