# An undecided door candidate is unfinished business

Status: approved by the owner on 2026-11-09 ("write it up and build it"). Applies the principle of
`2026-11-06-unrecognised-doorway-hypothesis-design.md` §1: missing a door is a state of knowledge the planner can act on.

## 1. What failed

- **clean23** (start in room_2): `DONE` after 30 s, no door known. The robot's one candidate sat at L = -0.53 for 322
  logged frames, between the birth line (0) and the delete line (-1.08). It had **one view**, and the view ledger counts
  all frames of a view as one trial. Room promoted, no door found, so the drive concluded `Done`.
- **clean24** (start in room_3): the candidate near (-0.87, 2.61) sat undecided for at least 143 logged frames.

The drive reads only born doors, so a question door_concept had not answered was treated as no door.

## 2. What the data ruled out

The proposal was "go closer". clean24's masks do not support it: median YOLO score 0.63 at 4-5 m against 0.45 at 1-2 m.
clean23's masks are one frame repeated 331 times at 0.35. So the look asks for a **new view**, not a shorter range.

## 3. The model

**The question.** A pending candidate of the current room is OPEN while the costed decision has not refused it
(`not decide(L).remove`) and some standpoint would still add a view. Born candidates leave the list. No new level: the
delete line and the birth line are the ones already in `door_decision.h`.

**Which size judges the question.** The size term is the current frame's, so openness uses the best size term the
candidate ever gave (clean25: one mask of width 0.00 sank L to -11.9, the room read as settled and the tour ended in 15 s).
A clipped blob only understates a door's width, so a narrow frame says nothing against a place that once looked
door-sized, while a place that only ever looks narrow keeps its low term and is refused.

**The action.** Each open candidate of a promoted room gets `aff_door_look_place_<key>`, planned by the existing
`rc::door::look::plan`, valued by its EIG about existence, competing in the open contest like any look.

**What counts as new.** A standpoint is new if the candidate's own `ViewLedger` would not absorb a frame taken there
(`find_view == npos`). This is the ledger's own MAP rule (`k >= 1/2`), so a look that the belief would count as the same
trial is worth nothing and is not offered. A site that the controller finished or handed back goes into a second ledger
of tried sites, judged by the same rule, so a site the robot cannot reach is not offered again. The door's kernel width
leaves about four distinct views per candidate, which bounds the looks without a counter.

**When it ends.** A candidate stops being open when it is born, when it falls below the delete line, or when no new
standpoint remains. The third case leaves it not born and the tour is honest about it.

**The statement.** door_concept owns one node, `door_questions` (type metaconcept, `object_subtype` "door_questions"):
`parent` = the room it speaks about, `state` = "open" or "settled". While the room is provisional every undecided
candidate is open but no look is offered: the pre-promotion contest is unchanged.

**The drive.** `Return` and `Done` need the current room settled. The statement names its room, so a statement about
another room is no answer: this closes the race at promotion and at re-entry by construction. No statement at all (an
older door agent) does not hold the drive. The hold is bounded by `MaxRoomS`, the same limit as every other hold, and it
sits beside `remembered_uncrossed`. Crossing a found door is not held: leaving is not finishing, and the depth-first
return comes back to the room.

**A room with no door known.** (clean27: three tours promoted the start room within 25 s without one mask of a real door, and ended.) A promoted room with no known door is unfinished: door_concept seeds a place per door width along each wall (no detection needed), and the same belief answers. A good mask makes the door; an empty view closes the place. The seeds use the first-door base rate, as a fresh detection does: they add no evidence. Only the nearest open wall place holds a look node at a time; all of them count as open. A door becoming known retires the rest. A provisional room with no known door states "open", so the promotion cannot race the seeding.

## 4. Constants

None new. Flagged and inherited: `MaxRoomS` (decision 5 of the tour spec).

## 5. Known limits

- A real door that YOLO scores low from every view (clean23's 0.35) is treated as misses and can fall below the delete
  line. The look then ends the question honestly and the tour's pass check fails it.
- The confidence model is not refitted here. clean23 adds one distinct frame, which is not enough to change it.

## 6. The pass check

A tour passes only if it ends with `DONE` and the ground-truth trajectory goes through all three doors, which implies all
three rooms. A `DONE` alone counted clean23 as a pass.

## 7. Tests

`test_door_look`: a standpoint refused by the predicate is not chosen; every standpoint refused gives no plan.
`test_door_core_existence`: a candidate from one view is open; the view it has is not new, one at another bearing is; a
candidate refused by the delete line is not open. `test_exploration_drive`: Return and Done wait on an open room and go
at MaxRoomS; a found door does not wait; a statement about another room is no answer; no statement does not hold.
