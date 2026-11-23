# The "76 [recog] SWAP requested, 38 executed" of clean62..clean76

Log-only, 2026-11-23. Scratch: `SCR=auro_1122/scripts/review3/m3.py` (output `m3.json`).
Logs: `crossing/clean62..clean76/logs/ltsm_agent.log` (the swap lines are all in this log; `room_concept.log` has 0 `SWAP requested` lines, grep-verified in all 15 runs).

## 1. What the two numbers are

The recogniser prints TWO lines per request: `[recog] SWAP requested: room_id K (posterior ...)` (`ltsm_agent/src/room_recognizer.cpp:828`) and, from the executor, `[recog] SWAP requested for room_id K (executed by RoomSwap)`
(per-run counts: "SWAP requested:" = 38 in total, "SWAP requested for room_id" = 38 in total). So **76 lines = 38 requests**, not 76. "38 executed" = the 38 `[recog] swap to room_id K executed` lines (`room_recognizer.cpp:310`): all 38 requests were executed.
"Executed" means only that `RoomSwap::swap` posted a `proposal_room_K` node under the provisional room and staged memory (`ltsm_agent/src/room_swap.h:1-63`: "a swap PROPOSES; it changes nothing live ... room_concept adopts it in place or rejects the proposal").
Nothing is written to memory until the verdict. The room the robot is believed to be in changes ONLY when the verdict is `ACCEPTED` (`[swap] ACCEPTED: room_N adopted as r1_room_1` and `[eviction] room node ... now carries room_id 1 ... adopted in place`).
The promotions (43) are separate and all by passage lookup (`[promotion] PROMOTING`); the recogniser path is exercised only by these swaps.

## 2. Classification (38 requests, 14 of 15 runs; clean63 has no swap and no promotion)

All are the same kind of act: **re-identification of the provisional (new) room behind a freshly crossed door as a room already in memory**, i.e. loop closure. Not a correction of an adopted identity: no `UNDO` line appears in any of the 15 logs (grep), no `swap failed`.

| outcome (the `[swap]` verdict line that follows) | n | what it did to the believed room |
|---|---|---|
| `ACCEPTED ... adopted as r1_room_1` | **12** (12 runs) | **changed it**: the provisional room_N (4th: 11, 3rd: 1 [clean71]) became `r1_room_1` in place; 12 `adopted in place` lines match |
| `WITHDRAWN ... (REJECTED by room_concept)` | 21 | unchanged: room_concept's fit refused, "room_N stays a new room" |
| `WITHDRAWN ... (the robot left before a verdict)` | 3 | unchanged |
| no verdict (run ended) | 2 (clean73, clean74) | unchanged |

Proposals per provisional room: 2nd room 2 (both K=1), 3rd room 22 (K=1: 18, K=2: 4), 4th room 14 (K=1: 12, K=3: 2). Repeats are re-posts after a rejection (clean69: 10, clean76: 5 on the same provisional room).

## 3. Ground truth: the claim "P is K" against the true rooms

Truth of a provisional room P = column `truth` of `explore_rooms.csv` of the run (the GT polygon holding the robot at promotion) for rooms 1..3; for the 4th (never promoted in the explore tour) the polygon holding
the GT position 5 s after its `born_s`. Truth of K = the `truth` of `room_K`. A request is a **correct claim** if the two true rooms are equal. (An earlier attempt at the true room from the GT position at the log line failed:
the `ltsm_agent` column of `gt_pose.tsv` does not align to the saved log's line numbers, so line numbers were NOT used for time.)

| verdict | claim correct | claim wrong | total |
|---|---|---|---|
| ACCEPTED | **12** | **0** | 12 |
| REJECTED by room_concept | 1 (clean74, 4th room as r3_room_3) | 20 | 21 |
| left before a verdict | 0 | 3 | 3 |
| no verdict | 2 | 0 | 2 |
| total | 15 | 23 | 38 |

Believed room before / after an executed swap: for the 12 accepted ones, **before** the robot believed it was in a NEW room (a duplicate of a known room: wrong), **after** it believed `r1_room_1` = the true room (right), 12 of 12.
For the other 26 the belief did not change (new room before and after; for the 23 wrong claims that unchanged belief was the right one, for the 3 correct ones, a missed recognition: 1 rejected + 2 never decided).

## 4. Does "no room was ever taken for another" hold?

**Confirmed for what changes the robot's belief**: 0 of the 12 accepted swaps was wrong; 0 of the 23 wrong proposals was accepted (20 refused by room_concept, 3 withdrawn when the robot left, none adopted).
Qualifications the sentence needs: (a) the recogniser itself proposed wrong rooms 23 times in 38 (61 %, mostly the 3rd room offered as the start room: 18 of 24 proposals on the 3rd room), so the protection is room_concept's verdict
(the LiDAR fit), not the recogniser; the sentence "no room was taken for another" should be about adopted identities, not requests; (b) 1 correct claim was refused (clean74) and 2 got no verdict before the run ended;
(c) truth for the 4th provisional room comes from a position 5 s after its birth, and for 3rd/4th rooms in the ring the truth is a room-level label (3 rooms), so it cannot see an error between two rooms that share a label (none exist: 3 rooms);
(d) no `UNDO` ever fired, so no committed swap was reverted; whether any of the 12 adopted rooms was later mislabeled is not derivable from the ltsm log alone (the explore reports score the outlines, not the labels).
