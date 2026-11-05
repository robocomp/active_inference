#pragma once
// entry_reloc.h — when the ENTRY RELOCALISATION (RoomSceneGraph::dsr_update_entry_relocalisation) hands afford_room back to the
// room's planners. Pure, tested in tools/entry_reloc_test.cpp.
//
// After a crossing, afford_room is the free standpoint nearest the new room's centroid: relocalise away from the walls before
// anything else. It ended ONLY when the robot came within half its diagonal of that point. explore_spike1 (2026-11-05): the
// controller handed the centre back ("no route … the body cannot stand there") and drove to repaired standpoints 0.65 m away,
// so the relocalisation never ended, the layout explorer stayed locked out for 750 s, its `finished` never came and the
// crossing-born room was never promoted. The consumer's own answer is the other way out: once it has COMPLETED the claim —
// satisfied, or handed back as refused / infeasible / unreachable / outside the room — it has done what it can with the centre,
// and a better standpoint is the planners' job. A protocol event, not a threshold.
namespace rc::entry
{
[[nodiscard]] constexpr bool still_relocalising(bool at_centre, bool claim_completed)
{
    return not at_centre and not claim_completed;
}
}  // namespace rc::entry
