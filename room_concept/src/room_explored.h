#pragma once
// ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
// P(this room is EXPLORED) — the belief behind a mission's `until explored(room)` (local-policy plan, Task 5,
// 2026-10-21). Published by room_concept on the room node as `room_explored_p` (cortex dsr_attr_name.h); read by the
// controller's mission predicate. ABSENT on the node = never assessed, which never satisfies an `until`.
//
// ★ IT IS THE EXPLORER'S OWN STOPPING QUESTION, ASKED AS A PROBABILITY. The layout explorer (layout_explorer.h,
//   plan_coverage) already prices a trip: a standpoint is worth visiting only if the best look from it reveals at
//   least `price` (the robot's footprint area of still-unseen space during the survey; the cells that decide one
//   refiner edit after promotion). Its `finished` is "no standpoint reveals that much". So the decision "keep
//   exploring vs stop" has two options valued in the SAME unit:
//      stop      — worth `price`        (what one trip costs; the time is better spent elsewhere)
//      continue  — worth `gain_left`    (the most any reachable look would still reveal)
//   and the probability of the stop option is the Luce choice between them:
//            p = price / (price + gain_left)
//   p = 1 when nothing is left to see, 0.5 EXACTLY at the explorer's own `finished` boundary, → 0 while a single look
//   would reveal many trips' worth. No new constant: `price` and `gain_left` are what the explorer already computes,
//   and the boundary a mission states (">= 0.9") is the person's preference, read as odds: stop when stopping is
//   preferred 9:1, i.e. when the best look left is worth at most 1/9 of a trip.
//
// ★ THE BOUNDARY SEEN, NOT THE IoU. gain_left is the unseen area a look would reveal inside (or bordering) the room's
//   outline — the live analogue of the bench's seen_frac (memory: iou-has-been-scoring-extrapolation): a room whose
//   far corner is only EXTRAPOLATED by the box cover still holds unknown cells there, so it is not explored however
//   good the outline looks.
//
// ★ A RECALLED ROOM STARTS FROM MEMORY. The room is still unexplored only if memory missed something AND this visit
//   has not seen it yet (noisy-OR):  p = 1 − (1 − p_mem)(1 − p_visit).  A look there is worth only what memory may
//   lack, (1 − p_mem)·gain, so a well-remembered room (p_mem = 1) is explored on entry and offers nothing, and a partly
//   remembered one explores only the rest. With p_mem = 0 every formula reduces to the fresh room's.
// ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
#include <algorithm>
#include <optional>
#include <string>
#include <unordered_map>

namespace rc::explored
{
inline float unit(float p) { return std::clamp(p, 0.f, 1.f); }   // a probability's domain, not a gate

/// Luce choice of STOP (worth `price`) against the best look left (worth `gain_left`). nullopt = no price to compare.
[[nodiscard]] inline std::optional<float> p_stop(float price, float gain_left)
{
    if (not (price > 0.f)) return std::nullopt;
    return price / (price + std::max(0.f, gain_left));
}

/// Unexplored only if memory missed it AND this visit has not seen it.
[[nodiscard]] inline float combine(float p_mem, float p_visit)
{
    return 1.f - (1.f - unit(p_mem)) * (1.f - unit(p_visit));
}

/// What one look is worth in a room memory already holds with belief `p_mem`: only what memory may lack.
[[nodiscard]] inline float recalled_gain(float p_mem, float gain) { return (1.f - unit(p_mem)) * gain; }

/// The belief from one explorer plan. `evaluated` = standpoints the explorer could assess (0 = it could assess
/// nothing: no belief, never "explored"); `best_gain` = the most any of them would reveal, ALREADY discounted by
/// recalled_gain (the explorer plans on that gain); `price` = the explorer's own price of a trip.
[[nodiscard]] inline std::optional<float> belief(int evaluated, float best_gain, float price, float p_mem = 0.f)
{
    if (evaluated <= 0) return std::nullopt;
    if (unit(p_mem) >= 1.f) return 1.f;   // memory holds all of it: nothing this visit could add
    const auto p_visit = p_stop(price, best_gain / (1.f - unit(p_mem)));   // undo the discount: what THIS visit has left
    if (not p_visit.has_value()) return std::nullopt;
    return combine(p_mem, *p_visit);
}

/// What memory restores for a recalled room. The recall node's own attribute when ltsm carried it; else this agent's
/// record of the room from earlier in the run; else the outline alone. A remembered outline with no record is the one
/// the survey PROMOTED, and promotion is the explorer's `finished`, which the recall path already treats as the
/// room's complete layout (adopted as given, never re-surveyed): p_mem = 1.
[[nodiscard]] inline float restored(std::optional<float> carried_by_recall, std::optional<float> own_record)
{
    if (carried_by_recall.has_value()) return unit(*carried_by_recall);
    if (own_record.has_value()) return unit(*own_record);
    return 1.f;
}

/// This agent's record, by room name, of the last belief it published — what a re-entry restores within a run.
class Memory
{
public:
    void remember(const std::string& room, float p) { by_room_[room] = unit(p); }
    [[nodiscard]] std::optional<float> recall(const std::string& room) const
    {
        if (const auto it = by_room_.find(room); it != by_room_.end()) return it->second;
        return std::nullopt;
    }
private:
    std::unordered_map<std::string, float> by_room_;
};
/// Which question the explorer's plan answers.
enum class Phase
{
    Survey,         ///< a proto room: how much of its boundary is still unseen
    Contested,      ///< promoted: can a look decide one refiner edit — NOT whether the boundary is seen
    RecalledRest,   ///< a partly remembered room: what memory may lack
};

/// The agent's glue (RoomSceneGraph calls exactly this): what is published, what is remembered, what is terminal.
class Tracker
{
public:
    /// A room born or entered with no memory of it.
    void reset() { recalled_ = false; p_mem_ = 0.f; }
    /// A re-entry: memory restores p_mem (the recall's own attribute, else this agent's record, else complete).
    void enter_recalled(const std::string& room, std::optional<float> carried)
    {
        recalled_ = true;
        p_mem_ = restored(carried, memory_.recall(room));
    }
    [[nodiscard]] float p_mem() const { return p_mem_; }
    [[nodiscard]] bool recalled_partial() const { return recalled_ and p_mem_ < 1.f; }
    /// One explorer plan in `room`. `finished` = the explorer's latched verdict (layout_finished_).
    [[nodiscard]] std::optional<float> plan(const std::string& room, Phase phase, int evaluated, float best_gain,
                                            float price, bool finished)
    {
        // ★ A recalled room's `finished` is the same terminal verdict as a survey's promotion: nothing reachable is
        //   left worth a trip, so the boundary question is closed (p = 1). Without it p stuck at 1/(2 − p_mem).
        if (phase == Phase::RecalledRest and finished) return settled(room);
        const auto p = belief(evaluated, best_gain, price, phase == Phase::RecalledRest ? p_mem_ : 0.f);
        // ★ ONLY THE BOUNDARY BELIEF IS REMEMBERED. The contested plan prices a refiner edit; a room is contested only
        //   once promoted, i.e. after its survey finished — so its memory is 1, whatever the refiner still asks.
        //   Remembering the refiner's value made the re-entry re-survey the whole remembered outline.
        if (phase == Phase::Contested) memory_.remember(room, 1.f);
        else if (p.has_value()) memory_.remember(room, *p);
        return p;
    }
    /// Nothing left for the explorer here (promoted and uncontested, a given layout, a complete or finished memory).
    float settled(const std::string& room)
    {
        memory_.remember(room, 1.f);
        return 1.f;
    }
    [[nodiscard]] const Memory& memory() const { return memory_; }
private:
    Memory memory_;
    bool recalled_ = false;
    float p_mem_ = 0.f;
};
}   // namespace rc::explored
