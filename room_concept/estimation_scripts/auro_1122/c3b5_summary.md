# Item 13: out-of-view moved chair, 5 reps x 2 arms, 2026-11-22 19:33-21:28

- **World**: the owner's 19:26 world. CHAIR_3 is placed at its 11-21 home (-7.43, 8.51), then moved 1.09 m to (-6.35, 8.60) after leg 2.
  - room_2 now also holds CHAIR_4 (-2.17, 12.35), added by the owner. Results are not comparable to the 11-21 runs.
- **Builds**: main 565d9d4e (off_r5: fd0830ce HEAD, same binaries). Every run completed 4/4 legs.
- **Report**: `c3b5_report.csv`, from `c3b_report.py`.

## Grading
- **World-frame columns (w_*), new.** The moved chair is the room_2 chair whose pre-move belief, mapped to the world, is nearest CHAIR_3's home. All runs are within 0.08 m.
  - Mapping: gauge plus per-visit fit (`run_frames.py`).
  - Corrected: any chair belief within 0.25 m of the new world position, with std < 0.15 m.
- **Why the old criterion fails here.** c3b_timeline picks the room_2 chair with the most rows. With two chairs in room_2 it picked CHAIR_4 in on_r5, and it missed off_r1.

## Results (world frame)
| run | retina | first in view (s after return) | in view total (s) | corrected (s) | first seen → corrected (s) | by |
|---|---|---|---|---|---|---|
| on_r1 | ok | never | 0 | no | | |
| on_r2 | ok | never | 0 | no | | |
| on_r3 | ok | never | 0 | no | | |
| on_r4 | **segmenter failed: invalid** | 34.3 | 30.4 | no (0 chair births all run) | | |
| on_r5 | degraded (1 seg stage disabled, 5912 inference errors) | 42.2 | 24.5 | 74.0 | 31.8 | new node chair_180, not an offer |
| off_r1 | ok | 61.3 | 27.2 | 73.3 | 12.0 | new node |
| off_r2 | ok | 21.0 | 35.7 | no | | |
| off_r3 | ok | 59.3 | 18.2 | 62.1 | 2.8 | new node |
| off_r4 | ok | 25.6 | 5.2 | no | | |
| off_r5 | ok | never | 0 | no | | |

- **ON**: 1 of 4 valid runs corrected (on_r5, degraded retina). **OFF**: 2 of 5.
- In no run, on or off, did the offer correct the chair. Every correction came from a new node born on sight.
- No run kept a belief at the old spot after the return.

## Mechanism
1. **The offer has nothing to act on.**
   - Leaving room_2 deletes its chair nodes (chair_concept removes them; ltom logs "died").
   - On return, ltom publishes `ltom_recall_*` nodes. chair_concept uses those only to match a NEW detection (`common/recall_match`); it never re-creates a belief from a recall.
   - So the moved chair has no belief and no offer until it is seen again. The offer cannot send the robot to look at the old spot.
   - On 11-21 (c3b_on_hidden1, n=1) the chair was glimpsed from the entry 8 s after return. That glimpse re-bore chair_80 with σ 0.30, and only then did its own offer refine it. The 11-21 ON success depended on that glimpse.
2. **The ON arm spends its time on CHAIR_4.**
   - CHAIR_4's belief is right in position (within about 5 cm of truth) but its σ stays at 0.30 for 440–550 rows and never shrinks.
   - Its offer keeps the maximum price, 3.19 nats, and the controller claims it 3–4 times per run. It never pays off.
   - In ON runs the robot looked at CHAIR_4, not around the room, and in r1–r3 it never had the moved chair in view.
   - Why CHAIR_4's σ never shrinks is not yet examined.
3. **Seeing the chair is not enough.**
   - off_r2 had the new spot in view for 35.7 s (near for 53 s) and off_r4 for 5.2 s, with no correction.
   - The visibility rule ignores furniture and chair pose; CLOSET_0 is 1.7 m away. Unexamined.
4. **The retina failure in on_r4/r5** started at 20:53, a TensorRT engine deserialize error followed by CUDA inference errors.
   - main HEAD moved to fd0830ce (vygotskian_machine, "image baseline ... fits beside retina") during the series.
   - **Confirmed by vigotsky:** from about 20:50 to 21:30 it ran VLM image requests (gemma4:26b, about 13 GB of the 16 GB card), and several of those hit cudaMalloc OOM.
   - on_r4 and on_r5 fall inside that window.

## Design noted for later (owner ruling 2026-11-22: report as it stands)
- **The gap, shared by items 13 and 14.**
  - A recall (`ltom_recall_<uid>`) only renames a NEW tracker birth. This is by design: LIFECYCLE §1, "birth is tracker-only", which keeps memory from asserting presence in the graph.
  - Leaving a room deletes its instances, so a remembered object that is not seen right away has no belief and no offer.
  - Nothing sends the robot to look (item 13), and nothing finds the object missing (item 14).
- **Fix proposed, not built.**
  - The recall node carries its own epistemic offer ("look where I remember X"), valued from the memory's P(exists) and position σ.
  - If the object is seen, the normal birth takes its identity.
  - If the spot is in view and empty, that is absence evidence against the memory, lowering its P(exists).
  - No instance is created from memory, so §1 holds.
- **Open question:** where P(detect | viewpoint) lives for that absence update. It overlaps the permanence ledger (feat/ltom-permanence).
