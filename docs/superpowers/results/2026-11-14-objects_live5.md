# objects_live5 — existence = occupancy × class, live (2026-11-14)

Binary: countable_concept at 15453e09 (rebuilt 19:47, no sources newer). Tour: warm_start_tour.sh, fixture
three_rooms_door0, N_LEGS=2, LOOK_S=60 — same command as objects_live4. Both legs OK (room_2 36 s, room_3 41 s).
Data: `room_concept/estimation_scripts/crossing/objects_live5/` (main checkout; logs/ and the three
`<cls>_ai2_log.csv`). n = 1 run: every number is provisional. `Concept.FilterLog` was on for the run only (reverted).

## Against the spec §8 pass list

| check | objects_live4 | objects_live5 | |
|---|---|---|---|
| blackboard as a bed (room_1) | bed_6 born, held p .98 ~13 s, on the graph ~72 s | **not born**: its detections (room (2.45, −4.37), 151 frames) scored birth p = 0.167 every frame | ✅ |
| any bed node in room_1 | bed_6 | **bed_8** born at room (2.35, −2.86), ~1.1 m in front of the blackboard, fitted bed-shaped (1.79 × 2.09, H 0.74, Λ ≈ +4.4, p_cls .96); LiDAR saw empty volume (occ 1 vs free 199) → removed by the OCCUPANCY channel after 20 s | ⚠️ partial |
| bed | bed_7 ~7 cm, H 0.65 → 0.80 close up | bed_9 world (−5.13, 12.40), **~5.5 cm** from GT (−5.18, 12.39); 1.75 × 2.06, **H 0.70**, yaw 0.00; p median .96, min .54 (clsL min +0.31) | ✅ |
| left nightstand | ~1 cm | nightstand_5 world (−6.31, 13.22), ~5.5 cm; 0.50 × 0.47 × 0.62; p min .92 | ✅ |
| right nightstand | ~6 cm (born) | **not born** — seen in 2 frames at room (0.90, 3.87), dropped (tracker one-missed-cycle reset, unchanged, next design) | ❌ (not this change) |
| toilet | tank north wall, ~0.25 m | toilet_5 world (−1.05, 13.02), ~0.23 m from GT; yaw 1.40 (bathroom frame); 0.30 × 0.89 × 0.87; p median .96, **min .59** | ✅ (fit as before) |
| born no later (time vs leg-1 arrival in the bedroom) | bed_7 −6.7 s, left nightstand +13.6 s | **bed_9 +15.2 s (≈22 s later)**, left nightstand +13.1 s (same) | ❌ bed later — cause not checked (path differs per run; birth score ×0.89–0.92 vs ×0.95–1.0 before; n = 1) |
| rate | ~10 Hz | nightstand 15, bed 12.6, toilet 10.0 cycles/s | ✅ |
| clean stop | ✅ | ✅ (SIGTERM; all three agent nodes deleted) | ✅ |

## What this says

- The birth score did its job: the blackboard never became a bed.
- A different false bed appeared: a "bed" mask placed ~1 m in front of the blackboard, fitted to a bed-sized box. Its
  shape AGREED with the bed prior (Λ > 0), so the class factor rightly had nothing against it; the occupancy factor
  removed it because LiDAR found the box empty. That is the two-factor design working as intended, but it means the
  remaining false-bed path is a mis-placed mask (depth/deprojection), not shape. Not investigated.
- The real toilet's class probability dipped to .59: its fit is 0.30 m wide against a 0.58 ± 0.08 prior (the fit, not
  the class factor, is off — the same fit problem as objects_live3/4). The class factor penalises a badly fitted real
  object; with a worse fit it could remove it. Watch.
- The bed's height stayed at 0.70 this run (objects_live4's 0.80 rise did not recur; the robot's path differs, so
  this says nothing yet about the footboard).
