# Step 1 results: existence information replaces the 5-nat floor, 2026-11-23

## Setup
- **Build**: worktree `active_inference.worktrees/existence-info` (branch feat/existence-info, base main 142b632d, uncommitted, 57 files).
  - Object agents (chair, table, bottle, hood, refrigerator, nightstand, bed, toilet) all come from the worktree, per `<run>/agents.txt`.
  - Controller, door, room and ltom come from main, unchanged.
- **Change**:
  - offer price = P(detect)·ΔH_position + I(E;Y), where I = h(p·d·P(detect) + (1−p)·f) − p·h(d·P(detect)) − (1−p)·h(f) ≤ ln 2;
  - the floor and its gate are removed;
  - nothing else changed.
- **Tours**: furnished three-room tours, owner's current world, cabinet out, LTOM empty, per-class logs in every run.
  - Room scale 0.35: s1_rs035_3/4/5.
  - Room scale 1.0: s1_rs100_2/3.
  - Reference: lam_0p2, the same world and agents on main, i.e. with the floor, n = 1.
- **Failed tours, not graded**: s1_rs035_1, s1_rs035_2 and s1_rs100_1 failed leg 1, the door crossing out of room_1 (approach_not_offered / cross_unreachable / door_wont_open). That step is single-entrant, so it doesn't involve object pricing. 3 of 8 tours failed, against 0 of 11 tours in this world on 11-22. Unexamined.
- **Scripts**: `scripts/aff_arbitration.py`, `scripts/c3a_objects.py`, and promised-vs-delivered (auro_1122/scripts/review/t3s1.py). "Delivered" is ½ ln det Σ_before/Σ_after over the claim window, plus the drop in existence entropy where ex_p is logged (not chair or bottle).

## Arbitration
| set | selections | contested (≥2 concepts) | object beat room / room beat object | margin median [range] | object winner price, median [IQR] | winners at 5.0 |
|---|---|---|---|---|---|---|
| step 1, scale 0.35 (3 tours) | 26 / 35 / 42 | 16 / 19 / 32 | 27 / 14 | 0.67, 0.39, 0.65 per run | 3.19 [1.37, 3.36] (n = 35) | 0 |
| step 1, scale 1.0 (2 tours) | 20 / 24 | 9 / 10 | 2 / 8 | 1.62, 2.08 | 3.19 [1.80, 3.23] (n = 5) | 0 |
| floor build, lam_0p2 (1 tour) | 38 | 27 | 16 / 1 | 0.59 | 5.0 [2.45, 5.0] (n = 17) | 7 of 17 |

- Contested winners at scale 0.35: chair 22, room 20, door 12, table 6, refrigerator 5, nightstand 1, bed 1.
- **The floor is gone**: no winner is priced at 5.0.
- **Object prices are still not small.** A chair is priced at about 3.19 nats, almost all of it position information from its σ ≈ 0.30 m birth/recall prior. Tables reach 2.7–8.8 and the refrigerator 3.8–5.4. The existence term adds at most 0.69.
- **The 0.35 room scale still decides object vs room.** With it, objects beat the room 27–14. Without it (1.0), the room wins 8–2.

## Promised vs delivered
| set | object claims | windows with a belief update | promised median [IQR] | delivered median [IQR] | Spearman promised vs delivered |
|---|---|---|---|---|---|
| step 1, 0.35 | 35 | 17 | 3.19 [1.48, 3.32] | 0.0 [0.0, 0.71] | +0.06 (p 0.73) |
| step 1, 1.0 | 5 | 3 | 3.19 | 0.54 [0, 1.91] | −0.41 (n = 5) |
| floor build lam_0p2 | 16 | 10 | 5.0 [2.97, 5.0] | 0.0 [−0.04, 0.10] | +0.18 (p 0.51) |

Per class at scale 0.35:
| class | n | promised | delivered median [IQR] |
|---|---|---|---|
| chair | 22 | 3.19 | 0 [0, 2.33] |
| table | 6 | 6.1 | 0 |
| refrigerator | 5 | 4.7 | 0 [−1.86, 2.44] |

- Existence-entropy drop delivered (13 claims with ex_p logged): median 0.
- Planar-error change over the window: median 0 m.
- **Promised and delivered are still uninformative once the floor is gone.** Prices now come from the geometric ΔH, and that is as poorly predicted as the floor was.
- **Caveat**: the window runs from the claim to the next selection, median 13 s. 18 of 35 windows contain no belief update at all, so for those "delivered" is unmeasured, not zero. Restricted to windows with updates (n = 17): delivered median 0.74 against promised 3.19, Spearman +0.02.

## C3a grade, for the record
| run | held of seen | phantoms | median planar error |
|---|---|---|---|
| s1_rs035_3 | 8 of 15 | 5 | 0.09 m |
| s1_rs035_4 | 13 of 15 | 1 | 0.16 m |
| s1_rs035_5 | 13 of 15 | 1 | 0.18 m |
| s1_rs100_2 | 11 of 15 | 0 | 0.13 m |
| s1_rs100_3 | 9 of 15 | 3 | 0.14 m |
| floor build lam_0p2 | 12 of 15 | 7 | 0.15 m |

## Reading, for the step-2 decision
- **By the owner's stated criterion, step 2 is indicated.** Promised vs delivered is still uninformative without the floor.
- **The prices that remain are the position ΔH**, driven by birth/recall priors (σ 0.30 m) and by the overconfident within-view shrinkage the NEES shows. That is the covariance recalibration of step 2.
- **The room scale is still load-bearing**: without it the room offer wins 80% of object-vs-room contests.
