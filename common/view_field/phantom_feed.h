/*
 * phantom_feed.h — turn the phantom log into supervision for the p_FA field.
 *
 * THE SIGNAL, and why it needs no new sensing: a hallucination gives birth to an object; later the robot
 * comes back, looks closely, and the object is denied. That denial RETRO-LABELS the detections that gave
 * birth to it as clutter. The experiment is the robot's ordinary behaviour — all that was missing was
 * recording it and letting it update a belief.
 *
 * ★WHAT COUNTS, AND WHY IT IS A WEIGHT. A DEATH is evidence about the DETECTOR only if the killing look
 * could actually have resolved the object. A death at p_detect = 0 is our own removal bug — and we have a
 * measured example: hood_1 fell 0.959 -> 0.053 with ex_pdetect = 0.000 on every cycle of the decay, killed
 * by an undamped LiDAR carve while the camera was silenced. A field that swallowed that would learn "the
 * kitchen wall hallucinates hoods" from a bug in our own absence weighting.
 * So the weight is p_detect * in_fov_frac: how resolving the killing look was, times how much of the object
 * it could see. No cutoff, and the pathological events self-exclude at weight 0.
 *
 * ★AND THE OTHER SIDE MUST EXIST OR THE FIELD IS A RATCHET. Feeding only deaths lets alpha grow without
 * bound and p_FA can only rise — the field would slowly condemn the whole map. note_verified() is the
 * counterweight: an instance that has been confidently observed and holds a high existence is evidence that
 * detections of that label, from that direction, at that place, are GENUINE. Credited ONCE per instance
 * (the caller owns the one-shot flag) so the two event streams stay comparable — deaths are rare, and a
 * per-cycle confirm stream would swamp them and drive p_FA to zero everywhere, which is just the old
 * hard-coded 0.05 with extra steps.
 */

#pragma once

#include <string_view>

#include "view_field.h"
#include "../phantom_log/phantom_log.h"

namespace rc::field
{

// How much this event tells us about the detector. 0 ⇒ nothing (and observe() then does nothing at all).
inline float disconfirmation_weight(const rc::history::PhantomEvent& e)
{
    return std::clamp(e.p_detect, 0.0f, 1.0f) * std::clamp(e.in_fov_frac, 0.0f, 1.0f);
}

// A DEATH retro-labels the detections at (cell x bearing) as clutter, in proportion to how confident the
// disconfirmation was. Any other event type is ignored: only a denial carries this information.
inline void note_phantom_event(ViewField& p_fa, std::string_view label,
                               const rc::history::PhantomEvent& e)
{
    if (e.event != "DEATH")
        return;
    // ★★A VERIFIED INSTANCE THAT DIES IS NOT A FALSE ALARM. It is a real object that left, or one of our
    // own removal defects — either way the detections that birthed it were CORRECT. The confidence weight
    // cannot catch this: those deaths score 0.85-0.95 because the killing look really was good. Without
    // this gate the field learns the opposite of the truth from the very first run.
    if (e.ever_verified)
        return;
    p_fa.observe(label, e.x, e.y, e.view_bearing, /*positive=*/true, disconfirmation_weight(e));
}

// The counterweight: this instance was really there. Call ONCE per instance, when it has been confidently
// observed and its existence is high — the caller owns that one-shot, because only the caller knows the
// instance's identity and lifetime.
inline void note_verified(ViewField& p_fa, std::string_view label,
                          float x, float y, float view_bearing, float p_detect, float in_fov_frac)
{
    p_fa.observe(label, x, y, view_bearing, /*positive=*/false,
                 std::clamp(p_detect, 0.0f, 1.0f) * std::clamp(in_fov_frac, 0.0f, 1.0f));
}

}   // namespace rc::field
