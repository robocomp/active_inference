# Instructor notes

Every number on a slide carries a `source:` line: either a thesis file and line, or a lab exercise.
Before each session, re-check the thesis lines; the thesis is still moving.

## Day 1 — The challenge, probability, and free energy (`slides/day1.pdf`, `labs/day1.ipynb`)

Frame numbers are the slide footers (n/36).

| time | block | frames |
|---|---|---|
| 0:00–0:15 | The challenge: the video, the cycle, every stage is a belief, toyworld | 1–5 |
| 0:15–0:55 | The grammar of uncertainty: variables, the joint table, marginalise, condition, product rule, the Gaussian picture and its two forms, chain rule, N! orderings, causal order, perception / control / generative, the robot's joint, Bayes as the three operations | 6–19 |
| 0:55–1:00 | Behind a sensory boundary; where exact inference dies (a short recap: Bayes is already on the board) | 20–21 |
| 1:00–1:30 | VFE: rearrangement, identity, complexity − accuracy, Kalman worked example (do it on the board), F differentiated twice (belief and action, EFE in one line), Rényi aside | 22–27 |
| 1:30–1:50 | Laplace: Σ = H⁻¹, Gauss–Newton, the wall, honesty, Occam | 28–32 |
| — | break | |
| 2:00–2:10 | What measurement said: F ≠ computation, inherited errors | 33–34 |
| 2:10–3:00 | Lab 1: warm-up P1–P4 (~15 min), then exercises 1–6 | 35–36 |

**Moved to Day 3 (2026-10-17).** Day 1 had 44 frames. The action section (∂F/∂a = 0, EFE, three
names for one number, where to look, ln 2, travel, time scales) and the two measured results that
need it (explorer equivalence on 594 layouts; stopping as an epistemic act) now live in
`slides/day3_efe.tex`, to be `\input` by `day3.tex`. Day 1's lab exercises 7–10 moved with them:
`labs/src/day3.py` block B0, exercises W1–W4. Day 1 is now perception, plus one frame of action
(frame 26, added 2026-11-08): F is differentiated with respect to the belief and with respect to the
action, and G appears in one line with its two terms. It is a pointer; the derivation stays on Day 3.

**Graphics (2026-11-08).** Several frames lost bullets to pictures, so the spoken text now carries
them: the door table as a mosaic (area = probability, used for the joint, conditioning and the
product rule), the identity bars, complexity against inaccuracy, the Laplace bowls. The "how this
course is built" frame no longer lists the thread's four guises (common-mode ceiling, re-centred
prior, fused precision better than its best member, a metric scoring what the robot never saw): say
them. The averaging curve on it is a schematic, not data. The previous deck is
`slides/backup_day1_v35_2026-11-08.{tex,pdf}`.

**Probability block.** Keep the door table (0.27 / 0.03 / 0.14 / 0.56) on the board for the whole
block: marginals in the margins, the gap column for conditioning, both products for the product
rule. Frames 17–18 switch to the week's names S, O, A (action = where the LiDAR points, so
S→O←A, a collider; a chain S→O→A would be just as sparse reversed and lose the argument). The causal-order frames carry the argument the rest of the week depends on: every
ordering is exactly equal, so the data cannot choose one; physics chooses. Modularity is the
argument students remember (swap the sensor and only one causal factor changes). Frame 17 places
a plain controller or an RL policy in the picture: it is p(a|o), the anti-causal factor alone. It can
act well but cannot predict, be surprised, know what it does not know, or survive a sensor swap.
The precision-
form table pays off on Day 2 and later: forgetting an old pose is a Schur complement.

**Video.** The user supplies a Webots or real-apartment recording of the full cycle; drop it on
frame 2. If it is a file, `\usepackage{multimedia}` + `\movie` works in Okular / Acrobat only, so
it is safer to play it from a separate player.

**Kalman frame.** The derivation is the single most useful thing to do on the board: it shows
students that "free energy" is something they already know, and that the variance comes from a
different term from the mean.

**Expected lab numbers** (seed 0; students' numbers differ slightly with other seeds):
- warm-up: p(O=gap) = 0.41, p(open | gap) = 0.659; P3 (S, O, A collider S→O←A): p(A=towards | S) =
  0.5 for both S, p(S=open | gap, A) = 0.659 towards / 0.30 away; a new sensor changes all three
  anti-causal factors p(O), p(A|O), p(S|O,A), and only p(O|S,A) in the causal order;
  Gaussian conditional 2.517 ± 0.186 in both forms;
- wall fit: α ≈ 89.9° ± 0.12°, d ≈ 4.01 ± 0.01 m, ρ(α, d) ≈ −0.97;
- honesty: uniform σ ratio ≈ 1.10 / 1.13; incidence σ ≈ 1.02–1.05;
- Occam: ΔF from +0.1 (s = 0.1°) to +5.2 nats (s = 30°); tilted wall −70 nats;

**Day 3 material that came from Day 1** (block B0 of `labs/day3.ipynb`, frames in
`slides/day3_efe.tex`). The key question comes at the end of exercise W3: *what does ln 2 assume?*
Don't let students skip it. The answer (independent errors across looks) links back to Day 2's
common-mode ceiling and to the thread of the whole course. Expected numbers (seed 0): gains stay
0.693, under the wall 0.705, far west 0.949, behind the table 0.543, east corner 0.587; β switch ≈
0.07 nats/m; predicted vs realised ΔH for far west: 0.949 vs 0.951. Post-thesis claim in the W3
answer: the layout explorer's precision term is inert because the map keeps one point per voxel
(memory note `efe-objective-misfit-carries-it`), marked as work in progress.

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
