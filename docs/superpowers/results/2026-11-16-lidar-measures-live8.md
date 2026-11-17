# lidar-measures live8 — flag on / off / shifted class means (2026-11-17)

Binary countable_concept aea3cbdc (cost-removal worktree), fleet from main (hybrid controller 38375eeb). Tour:
warm_start_tour.sh, three_rooms_door0, N_LEGS=2, LOOK_S=60 (same as objects_live6). All legs OK in all three runs.
n = 1 per condition: every number is provisional. Scorer: `countable_concept/tools/lidar_measures_score.py`
(Σ PASS = max std ≤ 0.15 m and ≥ 95 % of settled rows in [rms/3, 3·rms]; bathroom origin assumed ⇒ toilet centre approximate).
Data: `room_concept/estimation_scripts/crossing/objects_live8{,off,shift}/` (main checkout).

| | live8 (on) | live8off (off) | live8shift (on, nightstand means +1σ) | live6 (old) |
|---|---|---|---|---|
| bed centre err | 0.214 | 0.148 (85 cm step) | 0.201 | 0.060 |
| bed |H − top| | 0.022 | 0.080 | 0.144 | 0.032 |
| nightstands centre err | 0.131; phantom 0.97 + duplicate 0.48 | 0.094 / 0.112 | 0.105 / 0.149 | 0.109 / 0.071 |
| nightstand size vs truth (shift run) | — | — | ns_6 ≈ true (w +3.2, h −2.4, H +0.4 cm); ns_7 w +18.7, H −9.3 cm | — |
| largest std | ≤ 0.26 m | up to 2.7 m | ≤ 0.17 m | up to 1.6 m |
| Σ checks | H mostly PASS; cx/cy/h FAIL | all FAIL | mixed | all FAIL |
| fit rate | 10.7–11.9 Hz | 10.3–13 Hz | 10.2–12.8 Hz | 10–14 Hz |

Verdict: §8 does NOT pass. Σ is now bounded (was metres) and a size follows data rather than a shifted class mean
(ns_6), but position is not better than flag off and the bed drifts sideways in both flag-on runs.

Analysis (`.superpowers/sdd/2026-11-15-lidar-measures/live8-analysis.md`), ranked faults:
1. First-crossing rays give a return BEYOND the box's entry point to the entered face (never to H): with an
   over-tall belief, rays over the real mattress pushed the bed's robot-facing face inward (live8 cx +0.215,
   w −0.34 in two LiDAR-only bursts). The deferred ownership mixture (Tasks 7–8) / free-space-along-the-ray is the cure.
2. The once-counted class prior is centred on the birth SEED (spec 7c.1 stated limit): a bad seed (mask over the
   bed side, h 2.14) is no longer pulled back as the per-frame anchors did in live6 → the live8 phantom nightstand
   and its duplicate. Proposed: the class prior as a factor at the class mean, once; the seed only as the start.
3. A robot-attached slice (range 0.26–0.29 m, ~13k points, probably the tray) is assigned to the bed while parked
   in every run incl. live6; it lifts H 15–19 cm, and b_cam is shared by both slices of a cycle.
4. b_cam's H marginal is the mask-extent common mode (too broad); 5. the LiDAR select band (BirthHeightM + 0.1)
   sits below true tops; 6. cy +5–10 cm in all runs, flag on and off: the wall reference is ~10 cm off (polygon
   centreline vs inner face, or the scorer's origin) at a stated σ of 2 cm.
