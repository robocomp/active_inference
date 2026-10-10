# Instructor notes

Every number on a slide carries a `source:` line: either a thesis file and line, or a lab exercise.
Before each session, re-check the thesis lines; the thesis is still moving.

## Day 1 — The challenge, and free energy (`slides/day1.pdf`, `labs/day1.ipynb`)

| time | block | frames |
|---|---|---|
| 0:00–0:20 | The challenge: the video, the cycle, every stage is a belief, toyworld | 1–6 |
| 0:20–0:30 | Behind a sensory boundary; where exact inference dies | 7–8 |
| 0:30–0:55 | VFE: rearrangement, identity, complexity − accuracy, Kalman worked example (do it on the board), Rényi aside | 9–14 |
| 0:55–1:10 | Laplace: Σ = H⁻¹, Gauss–Newton, the wall, honesty, Occam | 15–20 |
| 1:10–1:30 | Action: ∂F/∂a = 0, EFE, three names for one number, where to look, ln 2, travel, time scales | 21–29 |
| — | break | |
| 1:45–1:55 | What measurement said: explorer equivalence, F ≠ computation, stopping as an epistemic act, inherited errors | 30–34 |
| 1:55–3:00 | Lab 1 | 35–36 |

**Video.** The user supplies a Webots or real-apartment recording of the full cycle; drop it on
frame 2. If it is a file, `\usepackage{multimedia}` + `\movie` works in Okular / Acrobat only, so
it is safer to play it from a separate player.

**Kalman frame.** The derivation is the single most useful thing to do on the board: it shows
students that "free energy" is something they already know, and that the variance comes from a
different term from the mean.

**The key question of the lab** comes at the end of exercise 9: *what does ln 2 assume?* Don't
let students skip it. The answer (independent errors across looks) is the bridge to Day 2's
common-mode ceiling and to the thread of the whole course.

**Expected lab numbers** (seed 0; students' numbers differ slightly with other seeds):
- wall fit: α ≈ 89.9° ± 0.12°, d ≈ 4.01 ± 0.01 m, ρ(α, d) ≈ −0.97;
- honesty: uniform σ ratio ≈ 1.10 / 1.13; incidence σ ≈ 1.02–1.05;
- Occam: ΔF from +0.1 (s = 0.1°) to +5.2 nats (s = 30°); tilted wall −70 nats;
- gains: stay 0.693, under the wall 0.70, far west 0.95, behind the table 0.54, east corner 0.59;
- β switch ≈ 0.07 nats/m; predicted vs realised ΔH for far west: 0.949 vs 0.950.

**Post-thesis claim on a Day 1 slide/answer:** the layout explorer's precision term is inert
because the map keeps one point per voxel (memory note `efe-objective-misfit-carries-it`).
Marked as work in progress.

## Day 2 — Estimating the room, and finding oneself in it (`slides/day2.pdf`, `labs/day2.ipynb`)

| time | block | frames |
|---|---|---|
| 0:00–0:05 | Where we are | 2 |
| 0:05–0:15 | Two regimes; why they are split; the price | 3–5 |
| 0:15–0:40 | Acquisition: the state, the gauge, the derived band, one scan → four walls, MDL, why free space is needed | 6–12 |
| 0:40–1:00 | Localisation: the observation model (note the 1/N), the residual floor, honest prediction, the wrong-wall NEES, symmetry | 13–18 |
| 1:00–1:30 | Shared error: Woodbury ceiling (on the board), the toy ceiling, two case studies | 19–23 |
| — | break | |
| 1:45–3:00 | Lab 2 | 24–26 |

**Do the Woodbury step on the board.** Students who see σ²(JᵀJ)⁻¹ → 0 leaving Σ_δ behind remember it;
students who only see the result do not.

**The two case studies are post-thesis** (memory notes `association-dies-by-common-mode-latch`,
2026-09-18; `estimate-mode-needs-plain-slam`, 2026-10-09). Refresh both from tesis-12 before the
course: the layout result in particular is expected to land in the thesis, with final numbers.

**The NEES exercise (B) is the emotional centre of the lab**: one wall 1.5 cm out takes NEES from ~2
to ~90. Ask students *before* running it what they expect. Most guess "a bit worse".

**Expected lab numbers** (seed 0):
- θ₀ = −17.19° (truth −17.19°); band ±6.7 cm = 3.37σ; a point on the line brings 5.69 nats;
- walls fit to ≤5 mm / ≤0.16°; IoU 0.994, area 23.99 m²;
- L_edge = 7.56 nats (R = 12 m, c = 5 cm); the table face earns ≈ 82 nats;
- localisation rms ≈ 2 mm / 1 mm / 0.04°; mean NEES: true room ≈ 1.95, estimated ≈ 3.8, east wall +1.5 cm ≈ 90;
- symmetry: identical F at the true pose and its half-turn;
- ceiling: actual ≈ 8.8 → 6.6 mm (N = 1 → 32); independent 4.9 → 0.9 mm; Woodbury 9.3 → 7.6 mm.

**A trap I fell into while building the lab, worth telling:** the first scripted route drove the
robot through the east wall and out of the room. The localiser kept reporting a confident pose
against the living-room polygon, and the NEES came out at ~6. It looked like a subtle modelling
defect. It was a test that measured something other than what it claimed. The route now asserts
that the robot stays in the room.
