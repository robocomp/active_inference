# Item 4: travel-cost sweep, λ = 0.1 / 0.2 / 0.4 nats/m, 2026-11-22 23:20-23:58

## Setup
- **Tour**: one furnished C3a tour per λ (n = 1 each), warm three-room fixture three_rooms_door0, legs R2 R3 R2 R1.
  - Run order 0.1 → 0.4 → 0.2.
  - LTOM empty at start; cabinet agent OUT.
- **λ**: controller `AffordanceLambdaCost`, confirmed in each controller log. The config is restored to 0.2.
- **World**: the owner's new world (19:26: closet, chair(4) and chair(5), bottle moved), with CHAIR_3 at the owner's (-7.97, 9.50).
- **Builds**:
  - controller: main binary of 2026-11-20 22:57;
  - nightstand, bed and toilet: cabinet-exit worktree e5310a2a, the same binaries as c3a_3 and c3a_cab1;
  - other concept agents and ltom: main binaries, HEAD 267db579 is scripts only.
- **Comparability**: λ = 0.2 here is NOT c3a_3 / c3a_cab1 (old world, cabinet IN). The reference for the sweep is lam_0p2.
- **Scripts**: `scripts/aff_arbitration.py` (selections) and `scripts/c3a_objects.py` (grade); visit order computed inline from `<run>/aff_arbitration.csv` + `c3a_instances.csv`.
- **Runs**: all 4/4 legs.

## Results
| λ | selections | contested (≥2 concepts eligible) | margin winner − runner-up, median [IQR] | object beat room / room beat object | distance (m), duration (min) | held of seen | phantoms | precision | median error (m) |
|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 37 | 25 | 0.76 [0.41, 2.52] | 15 / 2 | 72, 10.0 | 13 of 15 | 2 (2 bottle) | 0.88 | 0.11 |
| 0.2 | 38 | 27 | 0.59 [0.32, 2.23] | 16 / 1 | 83, 11.7 | 12 of 15 | 7 (3 table, 2 bottle, chair, nightstand) | 0.64 | 0.15 |
| 0.4 | 102 | 89 | 2.42 [0.86, 3.56] | 64 / 4 | 79, 13.0 | 11 of 15 | 4 (2 bottle, chair, toilet) | 0.65 | 0.14 |

- Margins are in score units: gain − λ·distance, nats.

**Winners by concept** (contested selections):
- 0.1: chair 6, room 6, nightstand 4, door 3, bottle 2, fridge 2, hood 1, table 1.
- 0.2: chair 6, room 6, nightstand 5, door 4, table 2, fridge 2, hood 1, bottle 1.
- 0.4: toilet 23, nightstand 22, room 13, chair 10, door 6, bottle 5, hood 4, fridge 4, table 2.

**Visit order** (first claim of each object, matched to Webots):
- 0.1: KitchenHood > chair(1) > bottle? > chair(3) > nightstand R > nightstand L > chair(2) > Fridge > RoundTable.
- 0.2: KitchenHood > chair(1) > table? > nightstand R > chair(3) > chair(5) > nightstand L > chair(2) > Fridge > bottle?.
- 0.4: bottle? > hood? > Fridge > chair(2) > Table1(1) > Chair > chair(4) > nightstand R > nightstand L > chair(3) > chair(5) > toilet? > Toilet > RoundTable.

**Order distances**:
| pair | edit distance of claim sequences | Kendall τ on first-claim order |
|---|---|---|
| 0.1 vs 0.2 | 13 (lengths 22 / 22) | +0.57 (8 common objects) |
| 0.1 vs 0.4 | 66 (22 / 79) | +0.14 (7) |
| 0.2 vs 0.4 | 66 (22 / 79) | −0.62 (7) |

## How to read it
- **0.1 and 0.2 behave alike.** The selection counts, the winners and the order agree broadly (τ +0.57).
- **0.4 changes the behaviour.**
  - Selections almost triple (102).
  - The robot alternates between nearby offers in the bedroom/bathroom: toilet 23 and nightstand 22 wins, and margins are much larger.
  - Distance and duration barely change, so the extra selections are switching among nearby targets, not extra travel.
  - The visit order is reordered; τ with 0.2 is negative.
- **The grade differences are within what one run per λ can show.** Phantoms vary 2–7, mostly bottle (the new-world bottle phantom) and tables.
- **Not established:** n = 1 per λ. Whether 0.4's switching is systematic needs repeats, at about 13 min per run.
