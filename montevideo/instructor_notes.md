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
| 2:10–4:00 | Lab 1: warm-up P1–P4 (~15 min), then exercises 1–6 (about 50 min of material: **an hour is spare, see below**) | 35–36 |

**Moved to Day 4 (2026-10-17).** Day 1 had 44 frames. The action section (∂F/∂a = 0, EFE, three
names for one number, where to look, ln 2, travel, time scales) and the two measured results that
need it (explorer equivalence on 594 layouts; stopping as an epistemic act) now live in
`slides/day4_efe.tex`, `\input` by `day4.tex`. Day 1's lab exercises 7–10 moved with them:
`labs/src/day4.py`, part A. Day 1 is now perception, plus one frame of action
(frame 26, added 2026-11-08): F is differentiated with respect to the belief and with respect to the
action, and G appears in one line with its two terms. It is a pointer; the derivation stays on Day 4.

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

**The session is 4 h, 9:00–13:00.** This table was timed for 3 h, and Lab 1 does not yet fill the hour left over.

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

## Day 2 — The CORTEX architecture: one belief, many owners (`slides/day2.pdf`, `labs/day2.ipynb`)

| time | block | frames |
|---|---|---|
| 0:00–0:15 | Who holds the belief; CORTEX, the whole picture (name the four colours and stop: the rest of the day is its centre circle); modules against a shared graph | 1–3 |
| 0:15–0:30 | The shared graph: RT edges, structural / non-structural, replicas, media plane, five memories | 4–8 |
| 0:30–1:10 | One free energy, many owners: the joint, three treatments, seven assumptions, one process per factor, F_i, the toy scene and its price, a mean is half a belief, two failures | 9–18 |
| 1:10–1:35 | Concept agents: definition, residual + Jacobian, what is published, residual_concept, the contract, the federation | 19–24 |
| 1:35–2:00 | No thresholds: gate against precision, the toy, bounded outliers, the gate that survives, the rule | 25–29 |
| 2:00–2:15 | break | |
| 2:15–4:00 | Lab 2 | 30–31 |

**Say it, don't read it.** Frame 12 is the argument of the day: every boundary between processes is a
conditional-independence assumption, and whatever crosses it is what survived a marginalisation.

**Frame 15 is the surprise.** Ask before showing it whether the agents' standard deviations are right.
Most say no. Each marginal is exact; only the correlation is lost. Relations between objects are what
the factorisation cannot hold.

**Frame 16: the price does not fall with more data.** Students expect it to. It is the information the
objects carry about each other, and no number of frames creates a place to keep it.

**The rule is not "never a gate".** Frame 28 proves one must stay, and frame 29 names the three that do
exist and their verdicts: admissibility justified, birth due to be replaced, the peer grace period a
recorded debt. Say explicitly that the thesis has not finished this.

**Expected lab numbers** (seed 0; r = (5 cm)², s² = (10 cm)²):
- NEES with the published covariance 1.09; ignoring it 5.57 (predicted (r+s²)/r = 5);
- sum of agents' F equals global F exactly at s² = 0;
- one object: exact sd 2.50 cm, agent 2.50 cm; the relation θ0−θ1: exact 1.58 cm, agents 3.54 cm;
- KL = F(agents) − F(exact) = 2.56 nats, the same at 10, 40 and 160 frames;
- price at s²/r = 4: 1.27 / 5.12 / 11.37 nats for 2 / 5 / 10 objects;
- near pass: best gate 2.0 m, rmse 1.01 cm, precision 0.86 cm; far pass: best gate 8.0 m, 4.79 cm,
  precision 4.22 cm; the near pass's gate on the far pass: 300 cm;
- biased frames: at n = 2000 the displacement is 3.00 / 3.00 / 2.99 / 0 cm for α = 1 / 0.3 / 0.05 / 0.

## Day 3 — Belief in time (`slides/day3.pdf`, `labs/day3.ipynb`)

Frame numbers are the slide footers (n/29).

| time | block | frames |
|---|---|---|
| 0:00–0:05 | Yesterday: who holds a belief; the life of an instance | 1–2 |
| 0:05–0:25 | How a belief begins: ΔF, the toy streak, what the robot computes instead | 3–5 |
| 0:25–0:45 | The room is the first belief born: description length, two regimes, free space, the derived band, promotion | 6–10 |
| 0:45–1:05 | How a belief ends: three verdicts, walls in the visibility factor, the weight after the ceiling, four rules | 11–14 |
| 1:05–1:25 | How a belief fades: constant Q, the floor, inferred volatility, the toy | 15–18 |
| 1:25–1:40 | Localisation: the room supplies the likelihood, the floor, NEES, symmetry | 19–22 |
| 1:40–1:50 | Whose evidence: association, the qualified likelihood | 23–24 |
| 1:50–2:00 | Shared error: ceiling (on the board), the toy, the latch | 25–27 |
| — | break (10 min) | |
| 2:10–4:00 | Lab 3 | 28–29 |

**The order follows the form.** The birth of objects and the birth of the room are the same equation
(F = inaccuracy + complexity, with complexity κ|V| for the room). Say so at frame 6: students should
feel the room section as a second instance of frame 3, not a new subject.

**The room section was rewritten on 2026-11-08 against the current chapter.** The layout chapter was
rewritten on 2026-10-11 around plain scan-to-map SLAM → Manhattan outline → promotion by description
length. The earlier deck described the replaced design (walls as landmarks, existence bins, 7.29 nats
an edge). The toy lab still fits four walls from one scan: it teaches the band and the price, and it
is deliberately simpler than the robot. Say that the robot does not do this any more.

**Withdrawn on purpose.** The old case study comparing pose error against the provisional layout
(0.22 / 3.13 / 1.55 m) is gone. The thesis keeps those numbers out because the build carried an open
association defect (comment at the head of `layout_acquisition.tex`). Frame 7 makes the argument from
structure instead.

**The volatility update differs from the thesis on purpose.** `agents_life_cycle.tex` l.1093–1097 writes
`½ e^{−ζ} d − k/2` with `d = δθᵀΣ⁻¹δθ`. As written that does not balance: d is already dimensionless, so
e^{−ζ}d is of order 1/Q. The lab and frame 17 use the normalised surprise of the innovation, d = ν²/S,
saturated as a Student-t so that one spurious frame moves ζ by a bounded amount, and subtract its
expectation under a calibrated model (0.786 for ν₀ = 4). With the unbounded surprise, one 40 cm
spurious mask raises ζ by about 18 in a single step, so the gain goes to nearly 1 on the next frame and
the accumulated belief is lost. The thesis marks this
construction as design only, and frame 17 says so.

**Frame 12 is the one to tell as a story**: the same bug in three agents, each copy consistent with
itself. **Do the floor on the board** (frame 16): the fixed-point equation takes two lines, and "elapsed
time does not appear" lands harder when they have just seen it vanish.

**The association case study is post-thesis** (memory note `association-dies-by-common-mode-latch`,
2026-09-18), on the earlier wall-landmark estimator. Keep it as a story about shared error, not as a
claim about the current pipeline.

**Numbers in frame 10 are simulation only.** The pipeline has not run on the physical robot. Re-check
l.1173–1177 and l.1455–1462 before the session; the thesis is still moving.

**Lab order.** Parts A–C (life of a belief) come first so they cannot be squeezed out; D–G are the
former Day 2 lab, renumbered (exercises 1–9 became 7–15). If time is short: A, B, C, E, G.

**Expected lab numbers** (seed 0):
- birth: ΔF after 1..10 detections: sharp 2.9 → −57.5; distant 3.0 → −14.6; reflections 3.0 → +57.8.
  Over 300 runs: 2 sharp views: streak 0%, ΔF<0 98%; 10 reflections: streak 94%, ΔF<0 0%;
- absence: a 2%-resolvable view drops L by 0.0079 (saturate, then weight) or 0.4000 (weight the count);
  tour: evidence removes the vase at frame 68; no walls adds the chair at 174; weight the count removes
  the vase at 13; the miss counter removes everything by frame 44;
- forgetting: predicted floor 10.0 mm. Constant Q / inferred Q: sd 9.4 / 6.5 mm; moved by the spurious
  mask 92 / 44 mm (17 / 19 mm ten frames on); sd after the 60 s gap 123 / 45 mm; frames to adapt to the
  real move 9 / 15. Biased stream: 5.4σ / 7.6σ out;
- room: θ₀ = −17.19° (truth −17.19°); band ±6.7 cm = 3.37σ; a point on the line brings 5.69 nats; walls fit
  to ≤5 mm / ≤0.16°; IoU 0.994, area 23.99 m²; L_edge = 7.56 nats; the table face earns ≈ 82 nats;
- localisation rms ≈ 2 mm / 1 mm / 0.04°; mean NEES: true room 1.95, estimated 3.81, east wall +1.5 cm 89.8;
- symmetry: F = −786.82 nats at the true pose and at its half-turn;
- ceiling: actual 8.8 → 6.6 mm (N = 1 → 32); independent 4.9 → 0.9 mm; Woodbury 9.3 → 7.6 mm.

**The NEES exercise (13) is still the emotional centre of the second half**: one wall 1.5 cm out takes
NEES from ~2 to ~90. Ask students *before* running it what they expect. Most guess "a bit worse".

**A trap met while building the lab, worth telling:** the first scripted route drove the robot through
the east wall and out of the room. The localiser kept reporting a confident pose against the
living-room polygon, and the NEES came out at ~6. It looked like a subtle modelling defect. It was a
test that measured something other than what it claimed. The route now asserts that the robot stays in
the room.

## Day 4 — Action (`slides/day4.pdf`, `labs/day4.ipynb`)

Frame numbers are the slide footers (n/31). The session is 4 h, 9:00–13:00.

| time | block | frames |
|---|---|---|
| 0:00–0:05 | The loop of the day: belief, offer, selection, route, look | 1 |
| 0:05–0:30 | Why action needs a second functional: flat ∂F/∂a, G derived, two terms, three names for one number, where to look, ln 2 and its assumption | 2–7 |
| 0:30–0:55 | Affordances: who computes which term, the split of knowledge, the price of a view, selecting, hand-off without a rule, the trap, what is not established | 8–14 |
| 0:55–1:25 | From a choice to a motion: obstacles in the geometry, routes as optimisation, Frenet, the law, pole assignment, decay in metres, no obstacle term, two depths | 15–22 |
| 1:25–2:00 | The same objective on itself: calibration as inference, gauges, observability, conditional vs marginal, injection, drift and re-centring, standing still | 23–29 |
| — | break | |
| 2:15–4:00 | Lab 4 | 30–31 |

**Do the pole derivation on the board** (frame 19): substitute the law into ė = vφ, φ̇ = ω − vκ and let the
characteristic polynomial factorise in front of them. Then change variable to arc length and let the v
cancel (frame 20). Those two lines are the whole reason L is a length and not a gain.

**Frame 14 is the honesty frame.** It says the derived explorer did not beat the hand-tuned one, and
that the explorer running now is a coverage rule. Do not soften it. The earlier version of this frame quoted
594 layouts and IoU 0.9794 / 0.9806; those numbers are no longer in the thesis, so the frame now follows
`active_inference.tex` l.876–897, which is qualitative. If tesis-12 restores them, put them back with a source.

**Frame 21's caveat is the thesis's own**: no comparison with the re-solved window disabled has been run
(`action.tex` l.1005–1007, l.1459–1464). The toy shows the division of labour, not that it wins.

**Frame 28, right panel.** The re-centring decision was taken on the argument. The thesis removed its one
live measurement because it was made on a build with a faulty bias estimate (`self_calibration.tex`
l.2800–2825). Say so; the panel shows what double counting does in the toy, not a robot result.

**Frame 11's gains are illustrative** (the thesis says so at `action.tex` l.42). λ = 0.2 nats/m and the 0.5 nat
margin are the deployed values.

**Material that came from Day 1** (block A of the lab, frames 2–7, `slides/day4_efe.tex`). The key question comes
at the end of exercise 3: *what does ln 2 assume?* Don't let students skip it. The answer (independent errors
across looks) links back to Day 3's common-mode ceiling and to the thread of the whole course. Post-thesis claim
in that answer: the layout explorer's precision term is inert because the map keeps one point per voxel
(memory note `efe-objective-misfit-carries-it`), marked as work in progress.

**The lab was reordered to follow the slides.** A (old block B0), B (new: offers), C (old B: explorer), D (new:
tracking), E (old A: body). Exercises are numbered straight through, 1–21. If short of time, skip 2, 16, 19, 20.
In part B the sensor sees only within ±45° of the object being served; with a 360° sensor one object learns
the other's worth on the way and the hand-offs disappear. That is worth a sentence.

**Part B, the margin.** I expected margin = 0 to cause dithering; it does not, in this toy (4 hand-offs either
way). What it changes is that the door overtakes the table's last small gain, so in most seeds the robot
leaves with the table's offer still open. The answer cell says this.

**Expected lab numbers** (seed 0; students' numbers differ slightly with other seeds):
- A: gains stay 0.693, under the wall 0.705, far west 0.949, behind the table 0.543, east corner 0.587; β switch ≈ 0.07 nats/m; predicted vs realised ΔH for far west 0.949 vs 0.951;
- B: holders B B B T B B B B T T T D D D D D D (17 cycles, 4 hand-offs); table error 0.7 cm, bottle 1.8 cm; both offers withdrawn at Σ*; at λ = 1 the robot stops after 2 cycles;
- C: first look at an unknown cell 0.369 nats, after two misses 0.051; honest explorer stops after 11 looks with 307 nats left, 220 of them inside the table; through matter: 15 looks and never stops, promised 5516 / collected 3764 against honest 3291 / 3737;
- D: at s = L, e/e₀ = 0.732 / 0.731 / 0.729 for v = 0.15 / 0.35 / 0.7 (predict 0.736); mis-aimed peak 3.66 cm at s/L = 1.00 (predict 3.68); no feedforward: e = −11.8 cm on a radius-2 m circle (predict −12.5); overshoot 0 / 0 / 0.5 / 255% at delay 0 / 0.42 / 0.7 / 1.0 s; gains frozen at 0.35 m/s: 0.651 and 0.819 against 0.736; clearance −39.6 → +23.1 cm; tracking rms 20.0 mm;
- E: NEES 8.30 → 1.57; s_v 0.9615 ± 0.0020 (truth 0.9615), s_ω 0.9686 ± 0.0038 (0.9709), b 0.0122 ± 0.0003 (0.0120); parked teaches b only (σ ratio 0.039); straight teaches s_v (0.113); spin: s_ω 0.435 marginal, 0.099 conditional; spin then park 0.142 / 0.139; recovery at n = 8: 0.591 against 0.585 from σ alone; 1000× slower drift gives 5.43× tighter; re-solving the same data takes the sd from 3.99 to 1.63 ×10⁻³ over 6 solves.
