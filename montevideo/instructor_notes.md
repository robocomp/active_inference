# Instructor notes

Every number on a slide carries a `source:` line: either a thesis file and line, or a lab exercise.
Before each session, re-check the thesis lines; the thesis is still moving.

## Day 1 — The challenge, probability, and free energy (`slides/day1.pdf`, `labs/day1.ipynb`)

| time | block | frames |
|---|---|---|
| 0:00–0:15 | The challenge: the video, the cycle, every stage is a belief, toyworld | 1–6 |
| 0:15–0:50 | The grammar of uncertainty: variables, the joint table, marginalise, condition, product rule, the Gaussian picture and its two forms, chain rule, N! orderings, causal order, the robot's joint, Bayes as the three operations | 7–19 |
| 0:50–0:55 | Behind a sensory boundary; where exact inference dies (now a short recap: Bayes is already on the board) | 20–21 |
| 0:55–1:15 | VFE: rearrangement, identity, complexity − accuracy, Kalman worked example (do it on the board), Rényi aside | 22–27 |
| 1:15–1:30 | Laplace: Σ = H⁻¹, Gauss–Newton, the wall, honesty, Occam | 28–33 |
| 1:30–1:50 | Action: ∂F/∂a = 0, EFE, three names for one number, where to look, ln 2, travel, time scales | 34–42 |
| — | break | |
| 2:05–2:15 | What measurement said: explorer equivalence, F ≠ computation, stopping as an epistemic act, inherited errors | 43–47 |
| 2:15–3:00 | Lab 1: warm-up P1–P4 (~15 min), then exercises 1–7; 8–10 as homework if time runs out | 48–49 |

⚠ The probability block costs ~35 min, taken from the opening, VFE, Laplace, action and the lab.
The day is now tight; if it overruns, move "What measurement said" to the start of Day 2.

**Probability block.** Keep the door table (0.27 / 0.03 / 0.14 / 0.56) on the board for the whole
block: marginals in the margins, the gap column for conditioning, both products for the product
rule. The causal-order frames carry the argument the rest of the week depends on: every
ordering is exactly equal, so the data cannot choose one; physics chooses. Modularity is the
argument students remember (swap the sensor and only one causal factor changes). The precision-
form table pays off on Day 2 and later: forgetting an old pose is a Schur complement.

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
- warm-up: p(O=gap) = 0.41, p(open | gap) = 0.659; p(C | D, O) independent of O (0.95, 0.95),
  p(D=open | O, C=through) = 0.973 / 0.504; a new sensor changes p(O|C) and p(D|O,C) but not p(C);
  Gaussian conditional 2.517 ± 0.186 in both forms;
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
