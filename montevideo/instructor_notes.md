# Instructor notes

Every number on a slide carries a `source:` line: either a thesis file and line, or a lab exercise.
Before each session, re-check the thesis lines; the thesis is still moving.

## Day 1 — The challenge, probability, and free energy (`slides/day1.pdf`, `labs/day1.ipynb`)

Frame numbers are the slide footers (n/36).

| time | block | frames |
|---|---|---|
| 0:00–0:16 | The idea in 46 s (unnumbered slide after the title); the challenge: the video, the cycle, every stage is a belief, toyworld | 1–5 |
| 0:15–0:55 | The grammar of uncertainty: variables, the joint table, marginalise, condition, product rule, the Gaussian picture and its two forms, chain rule, N! orderings, causal order, perception / control / generative, the robot's joint, Bayes as the three operations | 6–19 |
| 0:55–1:00 | Behind a sensory boundary; where exact inference dies (a short recap: Bayes is already on the board) | 20–21 |
| 1:00–1:30 | VFE: rearrangement, identity, complexity − accuracy, Kalman worked example (do it on the board), F differentiated twice (belief and action, EFE in one line), Rényi aside | 22–27 |
| 1:30–1:50 | Laplace: Σ = H⁻¹, Gauss–Newton, the wall, honesty, Occam | 28–32 |
| — | break | |
| 2:00–2:10 | What measurement said: F ≠ computation, inherited errors | 33–34 |
| 2:10–4:00 | Lab 1: warm-up P1–P4 (~15 min), then exercises 1–6 (about 50 min of material: **an hour is spare, see below**) | 35–36 |

**The opening video (unnumbered full-screen slide after the title, `media/InferenciaActiva.mp4`, 46 s).** A robot in an orchard loses sight of a person behind leaves: it predicts, compares, feels the surprise, and has two ways out, change what it believes or move to see. Click once in pympress to start; it loops. Its captions are in **Spanish**: read the four steps aloud in English (believe, predict, compare, correct or act). Point at the second way out, acting to resolve doubt: that is the title of the course and the subject of Day 4. It is an illustration, as its corner label says, not a run of the robot.

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

## Day 2 — The CORTEX architecture: concepts and memories (`slides/day2.pdf`, `labs/day2.ipynb`)

| time | block | frames |
|---|---|---|
| 0:00–0:15 | Who holds the belief; CORTEX, the whole picture (name the four colours and stop: the rest of the day is its centre circle); modules against a shared graph | 1–3 |
| 0:15–0:30 | Concepts, the fabric of CORTEX: what a concept agent is, one chair across four dimensions, the class in the agent against the instances on the graph | 4–6 |
| 0:30–0:50 | The shared graph: the small graph then the real one, structural / non-structural, replicas, eventual consistency (frames 11–12), DDS domains and shared memory (frames 13–14) | 7–14 |
| 0:50–1:10 | Memories as concepts kept for longer: timescales, episodic, long-term spatial, long-term object, amortisation | 15–19 |
| 1:10–1:45 | One free energy, many owners: one big belief, what to report, cutting it into pieces, F_i as a balance, the toy scene and its price, a mean is half a belief, two failures | 20–28 |
| 1:45–2:00 | No thresholds: gate against precision, the toy, the gate that survives | 29–31 |
| 2:00–2:15 | break | |
| 2:15–4:00 | Lab 2 | 32–33 |

**CORTEX in action (the unnumbered full-screen slide after frame 2).** In pympress, click on it once to start; it loops (40.5 s), and the presenter window has the controls. pympress cannot autostart beamer's `\movie`. In another viewer the slide shows a still. It is a schematic: the order is illustrative, and the slide says so; on the robot everything runs at once. Point at the last line: every path went through the centre. The course copy of the drawing (`figures/cortex_overview.png`) has the thesis figure's `Root` node painted out, to match frame 7. The movie's route never goes through semantic or procedural memory (re-rendered 2026-11-09), but the drawing behind it still shows both circles, as the figure on frame 2 does. Frame 2 says they are not built.

**Eventual consistency (frames 11–12).** Walk frame 11 one agent at a time, from the left. The surprise to land: agent 2's write is simply lost, and nothing reports it; hence one owner per node. Students will ask why an arbitrary id may decide: frame 12 answers it. The id is arbitrary and so is the winner; what matters is that every copy applies the same ranking, so they agree. Swap the ids and they all agree on the other value. Agreement is what the CRDT promises; which value is right is what ownership is for. The rule shown is cortex's default CRDT mode (`core/crdt/delta_crdt.h`, `join_replace_conflict`): a change that has seen another replaces it, and otherwise the lower writer id wins. It is idealised: a comment in that file records that the causal test (`dotin`) carries a known workaround. The optional LWW mode, same-host only, orders by (timestamp, agent id), which is what the thesis sentence at `cortex.tex` l.257 describes.

**Shared memory (frame 14).** 85 GB/s is the whole DDR5 bus; one core copies well below that, so 33 µs is a lower bound. The driver still writes the frame once into the slot, so zero copy is about not paying again per reader, not about avoiding the first write. A story if time allows: for months the plane was silently NOT zero-copy, because FastDDS turns data-sharing off for keyed types and only logs it at INFO. Removing one `@key` took the bench from 494 µs at 1.4% CPU to 186 µs at 0.8% (memory note `zero-copy-was-never-on-keyed-types`, commit 48543c5; post-thesis). The configuration said on, and nothing measured it.

**Concepts first (2026-11-09).** Frame 5 is the map of the next three days: geometry, life cycle and aspect tomorrow, function on Day 4. The four dimensions are the course's framing; the thesis does not name them. Frame 6 is Day 1 again: the class is the prior, an instance the posterior. For a chair the belief is pose only and the shape is the class template; a table's shape is fitted.

**The graph from a run (frame 8).** The bottom layer is the simulator's apartment, not a belief. The `root` at the top of this picture is the viewer's World layer; frame 7 starts from the robot.

**Memories (frames 15–19): say what is not built.** Working, episodic and long-term spatial run. Long-term object memory (LTOM) is work in progress and is not in the thesis. Semantic and procedural are not built, although `cortex.tex` l.519–537 and l.549–570 describe them in the present tense; that text also says the spatial memory holds objects, which stopped being true with a2baaebe. There is no semantic memory beyond LTOM, which is per-instance today, and classes stay inside the concept agents.

**What not to claim about LTOM (state on 2026-11-09, re-check before the course).** It only offers: in `ltom_chair2` the chair was absent about 35 s after re-entry, until detected. Recall uses position only, so a moved chair is not established (offline atlas replay re-identified one 15/53 = 28%). Only chairs have done this live. Renaming, learned permanence and relations are designed, not built. The atlas exists only in `chair_concept` in shadow mode, and thumbnails are on a branch.

**Amortisation (frame 19) is a plan.** Missions run and their results are kept; nothing generalises from them yet.

**The main line is plain on purpose (2026-11-08).** The form admits cognitive-science students. The joint in full, the seven assumptions, the factorised q and the matrix C are backup frames after "Tomorrow", outside the frame count: show them if someone asks.

**Say it, don't read it.** Frame 22 is the argument of the day: every boundary between processes is a
conditional-independence assumption, and whatever crosses it is what survived a marginalisation.

**Frame 25 is the surprise.** Ask before showing it whether the agents' standard deviations are right.
Most say no. Each marginal is exact; only the correlation is lost. Relations between objects are what
the factorisation cannot hold.

**Frame 26: the price does not fall with more data.** Students expect it to. It is the information the
objects carry about each other, and no number of frames creates a place to keep it.

**The rule is not "never a gate".** Frame 31 proves one must stay, and the backup frame "The rule is" names the three that do
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

## Day 3 — Concepts: geometry, life cycle, aspect (`slides/day3.pdf`, `labs/day3.ipynb`)

Frame numbers are the slide footers (n/33).

| time | block | frames |
|---|---|---|
| 0:00–0:05 | Yesterday: a concept has dimensions, today three of them | 1 |
| 0:05–0:25 | What a concept agent is: two things it writes, what it publishes, the residual, the contract, owners alive | 2–6 |
| 0:25–0:40 | Geometry: the room is the first belief born, two regimes, free space | 7–9 |
| 0:40–0:50 | Geometry: localising, NEES against the room, symmetry | 10–12 |
| 0:50–1:00 | Geometry: association, shared error (on the board), the toy | 13–15 |
| 1:00–1:15 | Life cycle: the life of an instance, ΔF, the toy streak, what the robot computes | 16–19 |
| 1:15–1:30 | Life cycle: three verdicts, walls in the visibility factor, the weight after the ceiling, four rules | 20–23 |
| 1:30–1:40 | Life cycle: constant Q, inferred volatility, the toy | 24–26 |
| 1:40–2:00 | Aspect: class and instance, imagining, views per direction, second sight, toy and robot | 27–31 |
| — | break (10 min) | |
| 2:10–4:00 | Lab 3 | 32–33 |

The deck was reordered on 2026-11-09 to the course's dimensions. Moved out of the main line, kept as backup frames: the mask's label, the association latch case study, and the two plain frames on the floor (precision floor and residual floor).

**The main line is plain on purpose (2026-11-08).** As on Day 2, each formal frame (ΔF, the room's F(m), the derived band, promotion, the floor, ζ, the likelihood, the residual-floor arithmetic, association, Woodbury) is kept whole as a backup frame after "Tomorrow", outside the frame count.

**The room comes before births.** The birth of objects and the birth of the room are the same equation
(F = inaccuracy + complexity, with complexity κ|V| for the room). Frame 7 announces the trade and frame 17
pays it off: students should feel a birth as the room's corner over again, not a new subject.

**The room section was rewritten on 2026-11-08 against the current chapter.** The layout chapter was
rewritten on 2026-10-11 around plain scan-to-map SLAM → Manhattan outline → promotion by description
length. The earlier deck described the replaced design (walls as landmarks, existence bins, 7.29 nats
an edge). The toy lab still fits four walls from one scan: it teaches the band and the price, and it
is deliberately simpler than the robot. Say that the robot does not do this any more.

**Withdrawn on purpose.** The old case study comparing pose error against the provisional layout
(0.22 / 3.13 / 1.55 m) is gone. The thesis keeps those numbers out because the build carried an open
association defect (comment at the head of `layout_acquisition.tex`). Frame 8 makes the argument from
structure instead.

**The volatility update differs from the thesis on purpose.** `agents_life_cycle.tex` l.1093–1097 writes
`½ e^{−ζ} d − k/2` with `d = δθᵀΣ⁻¹δθ`. As written that does not balance: d is already dimensionless, so
e^{−ζ}d is of order 1/Q. The lab and backup frame "A constant becomes an inferred quantity" use the normalised surprise of the innovation, d = ν²/S,
saturated as a Student-t so that one spurious frame moves ζ by a bounded amount, and subtract its
expectation under a calibrated model (0.786 for ν₀ = 4). With the unbounded surprise, one 40 cm
spurious mask raises ζ by about 18 in a single step, so the gain goes to nearly 1 on the next frame and
the accumulated belief is lost. The thesis marks this
construction as design only, and frame 25 says so.

**Frame 21 is the one to tell as a story**: the same bug in three agents, each copy consistent with
itself. **If you have time, do the floor on the board** (between frames 24 and 25; picture and algebra are on backup frames): the fixed-point equation takes two lines, and "elapsed
time does not appear" lands harder when they have just seen it vanish.

**The association case study (now a backup frame) is post-thesis** (memory note `association-dies-by-common-mode-latch`,
2026-09-18), on the earlier wall-landmark estimator. Keep it as a story about shared error, not as a
claim about the current pipeline.

**Aspect (frames 27–31) is the part most likely to change before December.** The thesis barely covers it; the sources are the code and the runs. Re-check the status on frames 29 and 31 the week before: what is merged, what runs, which classes.

**Say the limits of frame 31 plainly.** The robot's acting recall is position only: the appearance term is zero in `common/recall_match/recall_match.h`, so the "+ look" columns are the toy, not the robot. π = 0.5 is a guess. Chairs only, in simulation (`ltom_chair2`, `rose_chair3`), and there is no recording of a chair coming back. The offline replay (15 of 53) was reported by the session building LTOM; it is not a thesis number.

**Frame 28 is honest about what imagining is.** `imagine_view` samples the hand-written card: shape and size, no colour, nothing learned. Your phrase for the class-level look ("what the language agent draws when asked how do you imagine a chair") is what students will remember; do not let it suggest the robot has learned what chairs look like.

**The toy's lesson that travels (frame 31, exercise 19):** at twice the assumed scatter, the look discards 58% of the chairs that never moved. It is the Day 2 lesson again: evidence with the wrong variance does harm. It is also the reason views are kept per direction.

**Numbers in the backup frame on promotion are simulation only.** The pipeline has not run on the physical robot. Re-check
l.1173–1177 and l.1455–1462 before the session; the thesis is still moving.

**Lab order.** The lab keeps its own order, which is not the deck's. Part I (aspect, exercises 16–19, about 25 min) is new and independent of the rest; expected output is in the solution notebook. Parts A–C (life of a belief) come first so they cannot be squeezed out; D–G are the
former Day 2 lab, renumbered (exercises 1–9 became 7–15). If time is short: A, B, E, G, I.

**Expected lab numbers** (seed 0):
- second sight (Part I), % correct, nearest / position / position + look: different looks: stayed 98.4 / 98.4 / 100.0, moved 32.8 / 6.4 / 100.0, new 0.0 / 80.8 / 99.7; one model of chair: stayed 98.6 / 98.6 / 98.6, moved 32.6 / 5.9 / 32.6, new 0.0 / 80.2 / 0.1. View noise ×1, 1.5, 2, 3 (stayed): 100.0, 91.0, 41.9, 1.9;
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
