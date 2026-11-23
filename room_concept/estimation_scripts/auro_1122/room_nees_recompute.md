# Room offer gain with its position covariance rescaled by the measured NEES (2026-11-23, log-only, no tours)

**Question** (plan item 2): the room is under-confident in robot position, with NEES/DOF 0.08 (median error 1.3 cm against σ 5.1 cm; nees_summary.md). If its x,y covariance is rescaled by 0.08, i.e. its precision is multiplied by 12.5, how many object-vs-room contests change at room scale 0.35 and 1.0?

## What the room advertises (room_scene_graph.cpp:3652, epistemic_planner.cpp:309-397)
gain = **marginal pose information** + **neglect drive**
- **Marginal pose information**: ½ log det(Y + I_view) − ½ log det(Y + I_here), clamped at 0.
  - Y = Σ_pose⁻¹, with belief forgetting (90 s) and a precision floor.
  - This is the only term the NEES rescale touches. A larger Y can only lower it, so a rescaled room gain is never higher.
- **Neglect drive**: 0.5 · log(1 + age/120 s) of the target cell.
  - It does not depend on Σ.
  - For a never-visited cell (sentinel age about 11 days) it is **4.49 nats**.

**Only the sum is logged**, so the pose part of each offer is not known exactly. What the logs do show: 16 of the 53 room gains in contested selections are 4.51 ± 0.02. That is the maximum neglect drive plus about 0.02 nats of pose information. Gains of 4.9–5.3 leave at most 0.4–0.8 nats for the pose part. When the target cell is unvisited, the pose part is at most g − 4.49.

## Contests
- **Runs**: step 1 (s1_rs035_3/4/5, s1_rs100_2/3) and the chair fix (s1g_*).
- **Definition**: a contest is every selection line where the room offer and at least one object offer were both eligible (Offered or Executing).
- **Flip condition**: selection score = gain × scale − 0.2·dist (+0.5 for the incumbent). A room win flips when the room's gain falls by at least (s_room − s_best object) / scale.
- **Only room wins can flip**, because the rescale only lowers the room's gain.

| scale | contests | object won | room won | room wins that flip if the room lost ALL its pose information (target unvisited, pose part = g − 4.49) | room wins that could flip at all (if the whole gain were pose information) |
|---|---|---|---|---|---|
| 0.35 | 86 | 63 | 23 | **1** | 19 |
| 1.0 | 51 | 21 | 30 | **0** | 28 |

The drop needed to flip a room win is 0.21–6.2 nats, with a median of about 2.6.

## Reading
- **When the room targets an unvisited cell, the rescale changes at most 1 contest at 0.35 and none at 1.0.** That is the usual case, and the clustering at 4.51 shows it. The room's offer is almost entirely its neglect drive, and the under-confident σ contributes a few tenths of a nat at most.
- **The open cases are the room offers aimed at already-visited cells.** These are gains of 0.06–1.34: 1 at 0.35 and 6 at 1.0. For them the split is unknown. 5 of them could flip if most of their gain is pose information (needed drops 0.21–1.20 against gains 0.25–1.34).
- **An exact count needs the two terms logged separately.** That is one line in room_concept at publish time (info, drive, Σ), plus the replay of one tour. It is not done here, because the item was specified as static.
- **Counting differs from aff_arbitration.py's "object beat room" numbers.** That script counts only the selection's winner against its runner-up. Here every selection with both kinds eligible counts.
