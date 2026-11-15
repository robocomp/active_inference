# objects_live6 — class factor + per-class box-fit error, live (2026-11-14)

Binary countable_concept 967e60fa. Same tour as objects_live4/5 (three_rooms_door0, 2 legs, LOOK_S 60). Legs OK
(room_2 70 s, room_3 25 s). n = 1: provisional. Data: `room_concept/estimation_scripts/crossing/objects_live6/`.

| check | objects_live5 | objects_live6 |
|---|---|---|
| blackboard as a bed | not born (birth p .167) | not born (birth p .167, 89 frames) ✅ |
| any bed node in room_1 | bed_8 (empty-volume phantom, 20 s) | **none** ✅ |
| bed | 5.5 cm, H .70 | bed_10 world (−5.22, 12.33), ~7 cm; 1.75 × 2.04, H .68; Λ median +3.47, p_cls min .80 ✅ |
| right nightstand | not born | nightstand_6 (−4.00, 13.18), ~4.5 cm ✅ |
| left nightstand | 5.5 cm | nightstand_7 (−6.46, 13.20), ~10 cm ✅ |
| toilet | 0.23 m, w 0.30, p_cls min .59 | toilet_6 (−0.96, 13.06), ~0.21 m; **w 0.58** (GT .58), 0.58 × 0.84 × 0.77, yaw 1.59; Λ min −0.30, p_cls min .75 ✅ |
| bed birth vs leg-1 arrival | +15.2 s | **−5.5 s** (objects_live4 −6.7) ✅ |
| rate | 10–15 c/s | 11.5–16 c/s ✅ |
| clean stop | ✅ | ✅ |

Notes: both nightstands were born this run, but the tracker's one-missed-cycle reset is unchanged (luck, not a fix).
The toilet's height read 0.77 vs GT 0.90 and its position is still ~0.2 m off (fit issue, unchanged). Λ min 0.00 for
the bed/nightstands = the newborn's first cycles (no size information yet), p_cls then = the prior 0.8.
