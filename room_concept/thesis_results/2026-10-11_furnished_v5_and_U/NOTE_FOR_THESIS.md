# Room-layout estimation: furnished rooms with camera views, U-shaped rooms, and one edge case (2026-10-11)

This folder is
`/home/pbustos/robocomp/components/active_inference/room_concept/thesis_results/2026-10-11_furnished_v5_and_U/`.
It is meant for the Layout chapter. It adds to the package `2026-10-11_furnished_22d763b` and replaces nothing in it.

| Folder | What it contains |
|---|---|
| `furnished_v5/` | The ten furnished rooms run again, now with a camera snapshot taken after promotion. Table (`tab:estimation-synthetic-furnished-v5`), stats, layout panels, and `projection_mosaic.png` (ZED and Ricoh images with the promoted room drawn over them). |
| `u_rooms/` | Three furnished U-shaped rooms after the exploration fix. Table (`tab:estimation-synthetic-U`), layout panels, `projection_mosaic.png`, and every raw snapshot (`snapshots/room<k>_<camera>_<ts>.png`; the earlier timestamp of each pair is at promotion, the later at the end of the run). |
| `u_rooms_before_fix/` | The same three U rooms before the fix: the failure that motivated it. Table (`tab:estimation-synthetic-U-before`), layout panel, ground-truth rooms. |
| `edge_case_room8/` | The edge case to discuss: an early promotion in the furnished T room 8. Layout panel with the furniture, numbers. |

All IoU values are shape IoU (best rigid fit of the ground truth to the estimate), as in the earlier packages.
Layout panels: green = ground truth, black = promoted, red = refined, orange = path, blue dot = start, grey = furniture.

## 1. Builds
- **`furnished_v5/` and `edge_case_room8/`:** the room_concept behaviour of commit `22d763b9`, plus two things that do not touch estimation:
  - a snapshot hook in the camera viewer;
  - the camera mount self-calibration switched off (`mountApply = false`, `mountPublish = false`) with the calibration files restored to a clean state. The online mount calibration had drifted ("runaway") and made the projected walls miss the image walls; with it off the projections line up again. The cause of the drift is not yet diagnosed.
- **`u_rooms/`:** the same plus the exploration fix of section 3. This fix is **not yet committed** and has **not yet been re-run on the ten furnished rooms or the apartment**. Treat the U-room numbers as provisional until that regression run is done; I will send it.

## 2. Furnished rooms again, with camera views (`furnished_v5/`)
Same ten rooms, same starts and furniture as `2026-10-11_furnished_22d763b`. One run each, first attempt, nothing repeated.

- **Success:** 10/10 (Wilson 95% CI 0.72–1.00).
- **Refined IoU:** median **.994** [95% CI .991, .997], worst .990.
- **Promoted IoU:** median .984 [.971, .987], worst .954.
- **Refinement helps every time:** 10/10, median gain +.010, Wilcoxon p = .002.
- **Pose error:** median 1.2 cm.
- **Time to promotion:** median 25 s, range 0–53 s.

These agree with the previous furnished campaign (.994 median refined, 10/10). The two campaigns can be quoted as a repeat of each other; the table in the earlier package can stay as the main one, or be swapped for this one, since this one comes with the camera views.

**Camera mosaic (`furnished_v5/projection_mosaic.png`):** the promoted room drawn into the ZED (front camera) and the Ricoh (360° camera) images. It shows that the estimated layout, expressed in the robot's frame, lands on the real walls in the images. It is a qualitative figure: it adds no number.

## 3. U-shaped rooms: a failure, its cause, and the fix
### The failure (`u_rooms_before_fix/`)
Three furnished U rooms (8 vertices each, 34–47 m²). All three were promoted from the first 360° scan, without the robot moving:

| U room | Promoted IoU | Refined IoU |
|---|---|---|
| 0 | .744 | .995 |
| 1 | .407 | .478 |
| 2 | .764 | .996 |

- In rooms 0 and 2 the refiner later recovered the missing arm, because the robot happened to drive near it.
- In room 1 the room stayed half-estimated (.478): the far arm of the U was never visited.

### The cause
- The explorer decides where to go by counting, for each candidate standpoint, how much unseen floor it would see. A room is "finished" when no standpoint would see enough.
- It counted unseen floor only inside the current outline, or within 0.4 m beyond its open edges.
- From the start, the opening into the hidden arm is seen at a grazing angle. The first outline cut straight across it, so the arm lay outside the outline and counted for nothing. Every standpoint scored zero and the room was declared finished after one scan, although the free-space map plainly showed floor running into the arm.

### The fix
- Unseen floor also counts when it borders the floor already seen to be free, not only the simplified outline.
- Beyond an open edge it counts out to the range the explorer can see (8 m), not a 0.4 m band.
- An edge is judged open or walled section by section, not by majority, so a partly open edge still invites a look.
- Space behind a real wall still counts for nothing, so the robot is not drawn to look through walls.

### After the fix (`u_rooms/`)

| U room | Time to promotion | Path to promotion | Promoted IoU | Refined IoU | Vertices (GT 8) |
|---|---|---|---|---|---|
| 0 | 67 s | 8.2 m | .987 | .998 | 16 |
| 1 | 81 s | 8.0 m | .975 | .991 | 10 |
| 2 | 46 s | 6.8 m | .992 | .993 | 12 |

- In all three rooms the robot now goes round the inner block into the far arm before promoting (orange paths in `layouts_panel_furniture.png`).
- Room 1 goes from .407/.478 to .975/.991.
- **Remaining weakness: too many vertices** (10–16 against 8). As in the furnished rooms, these are small steps where furniture meets a wall, not misplaced walls; they cost little IoU.
- **Camera mosaic (`u_rooms/projection_mosaic.png`):** the end-of-run ZED and Ricoh images for each room. The Ricoh panoramas are the useful ones: the projected walls, including the inner block of the U, fall on the image walls, and the LiDAR corner markers sit on the image corners. The ZED views of rooms 1 and 2 face a nearby wall and show little; room 0's ZED view is a good one. The orange disc at the top-left of room 1's Ricoh image is a rendering artefact (a light), not part of the estimate.

Suggested reading for the chapter: present this as a failure found by the experiments, explained, and fixed, with before and after in one table, in the same way as the residual stall in the previous package.

## 4. Edge case for the discussion: early promotion in furnished T room 8 (`edge_case_room8/`)
This is the edge case the author chose for the thesis.

- **What happened:** in one run of furnished room 8 (a T), the room was promoted from the first scan as a single rectangle: one arm of the T, 4 vertices, promoted IoU **.513**. The robot had not moved.
- **Why:** from the start, a table stands in front of the opening into the stem of the T. The LiDAR sees its legs and top and the free-space map does not open up behind it, so the stem reads as closed and the first outline closes across it. With nothing unseen left inside that outline, promotion accepted it.
- **What saved it:** the refiner. As the robot moved on afterwards, the stem came into view and refinement grew the outline to the full T: refined IoU **.998** (panel title "IoU 0.513 -> 0.998").
- **It is intermittent:** the same room, same start, promoted the complete T in both full campaigns (`22d763b9` campaign: promoted .994, refined .999; `furnished_v5`: promoted .987, refined .999). Small differences in the first scan decide whether the table hides the opening.
- **Point for the discussion:** promotion is a decision taken on the evidence gathered so far. An obstacle in front of an opening can make an incomplete room look complete, and nothing in that evidence says otherwise. The design copes because promotion is not final: the refiner keeps editing the layout after promotion. The cost is that, for a while, the robot works with a smaller room than the real one.
- **Not yet known:** whether the exploration fix of section 3 also changes this case. It would only do so if some floor beyond the table were already seen as free. That run is still to be done.

## 5. Still to come
- The exploration fix re-run on the ten furnished rooms and on the apartment (regression check), and on room 8. Until then, the furnished and apartment numbers in the chapter come from the build without the fix, which is correct for them.
- The commit hash of the fix, once committed.
