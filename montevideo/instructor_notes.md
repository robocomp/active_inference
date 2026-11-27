# Instructor notes

Every number on a slide carries a `source:` line: either a thesis file and line, or a lab exercise.
Before each session, re-check the thesis lines; the thesis is still moving.

## Day 1 — The challenge and probability (`slides/day1.pdf`, `labs/day1.ipynb`, `handouts/cheatsheet.pdf`, `handouts/handout_day1.pdf`)

**Rewritten 2026-11-27.** The deck has 49 numbered frames plus one backup ("Bayes' rule with the action"). Frame numbers below are the slide footers (n/49). Hand out the two-page **cheatsheet** at the start (numpy broadcasting, the three operations, Gaussians, logs, derivatives, F in one box) and the **Day 1 handout** (every derivation written out, a guide for each exercise; no solution code). Start with a show of hands: who has seen Bayes' rule, a Kalman filter, numpy? It tells you which optional parts to drop.

| time | block | frames |
|---|---|---|
| 0:00–0:15 | The challenge: the apartment, the cycle, the toy version, how the course is built | 1–4 |
| 0:15–0:55 | The grammar of uncertainty: discrete beliefs, the joint, marginalise, condition, factorise; three videos (bars to densities, the three operations, Gaussian operations); continuous beliefs, the Gaussian, the covariance, the wall's covariance | 6–20 |
| 0:55–1:15 | Modelling: chain rule, two and three variables, how the world works, six orders, causal order (second table on the next click), model-free RL, injecting knowledge | 22–29 |
| 1:15–1:30 | Time: notation, Markov, the robot acts on what it sees, the robot's joint, filling in the model | 30–34 |
| 1:30–1:45 | Inference: its goal, why the posterior, Bayes as inversion, the boxed rule, why exact Bayes fails | 36–40 |
| 1:45–1:55 | Recursion: one step, the recursion, Gaussian and linear = Kalman, the effect on the belief | 42–45 |
| 1:55–2:10 | break | |
| 2:10–3:00 | Lab: Exercises 1, 3, 4, 5, 6, 7, 8 in class; 2, the N = 20 count of 5, and 9 are homework | 47 |

The lecture is long (about 115 min). If you are behind at 1:15, go faster through frames 24–29: the lab's Exercises 5–6 rebuild the orderings by hand.

**Questions in the bottom bar.** Seven frames end with a blue question bar instead of the lesson; the next click shows the answer. Ask, wait 30 s, take one or two answers, then click. Frames: 11 conditioning (p(open | wall) = 0.03/0.59 ≈ 0.05), 20 the wall's covariance (only Σ_oo changes, to 0.50), 31 Markov (a wheel bias must join s), 42 one step (perfect wheels: still ±0.5), 43 recursion (read again: ±0.14), 44 Kalman (equal noise: halfway, 1.40 m, σ/√2 = 0.38), 45 the effect (three drives: ±0.61). The macro is `\ask{question}{answer}` in `slides/preamble.tex`.

**The opening.** Frames 1–4 pose the whole problem. The "how this course is built" frame no longer lists the thread's four guises (common-mode ceiling, re-centred prior, fused precision better than its best member, a metric scoring what the robot never saw): say them. The averaging curve on it is a schematic, not data.

**Probability block.** Keep the door table (0.27 / 0.03 / 0.14 / 0.56) on the board for the whole block: marginals in the margins, the gap column for conditioning, both products for factorisation (0.30 × 0.9 and 0.41 × 0.66 both give 0.27). The marginalisation frame squeezes each column into one cell of the p(O) strip: p(O) is two numbers, p(o) is one of them. The Gaussian comes before it is used: the 1-D bell (μ, σ, 68 %), the covariance entry by entry, and the wall's Σ built from its model (Var s = 0.5², Var o = 0.25 + 0.2², Cov = 0.25 because the noise is independent of s). If asked why the *precision* form must pay when marginalising: Λ_ss is the curvature of −ln p(s,o) in s with o fixed (the conditional), so the marginal must add back what o was explaining (the Schur complement).

**Modelling block.** The argument the rest of the week depends on: every ordering gives exactly the same joint, so data cannot choose one; physics chooses. Modularity is what students remember (frame 27: a new policy rewrites one causal factor, all three in O → S → A). Frame 28 places a plain controller or an RL policy: it is p(a|o) alone; it can act well but cannot predict, be surprised, or survive a sensor swap. It starts discussions: park them until Day 4.

**Time and inference.** Frame 31 (Markov) carries the warning that returns in the lab (Exercise 9) and on Day 4: leave something that persists out of s, such as a wheel bias, and the filter becomes sure and wrong. Frame 36 states the goal of inference before Bayes appears: a belief about what is hidden. Frame 40 is the bridge to Day 2 (p(o) needs every s); Day 2 frame 6 points back to it. Bayes' rule with the action is now a backup: say only that the action, like the reading, is known and stays on the right of the bar.

**Kalman.** Frame 44 gives the equations and the precision form (precisions add), which Day 2 frame 24 reuses. Doing one step on the board with the frame-45 numbers (2.00 ± 0.50 → 1.50 ± 0.54 → 1.32 ± 0.19) is the most useful five minutes of the day.

**Expected lab numbers** (seed 0):
- Exercise 1: joint 0.27 / 0.03 / 0.14 / 0.56; p(O) = 0.41 / 0.59; p(open | gap) = 0.659.
- Exercise 3: p(o) = N(2.00, 0.29), gain k = 0.86, posterior 2.517 ± 0.186 (samples 2.521 ± 0.185).
- Exercise 4: area p(o = 2.6) = 0.398; posterior 2.517 ± 0.186 on the grid.
- Exercise 5: causal 6 numbers, anti-causal 7; N = 20: 1 048 575 against at most 160 (an upper bound; the exact count is 143).
- Exercise 6: p(A = towards | S) = 0.5 for both S; p(S = open | gap, A) = 0.659 / 0.30; a new sensor changes all three anti-causal factors and one causal factor.
- Exercise 7: p(look | S) = 0.627 / 0.218; given o₁ = gap, 0.685 for both (exact 0.625 / 0.218 / 0.683).
- Exercise 8: 1.50 ± 0.54, then 1.32 ± 0.19.
- Exercise 9 (homework): no bias, final error +0.03 m with σ 0.07 m; 30 % bias, +0.47 m with the same σ: 7σ off.

## Day 2 — Free energy: perception and action (`slides/day2.pdf`, `labs/day2.ipynb`, `handouts/handout_day2.pdf`)

**Rewritten 2026-11-27.** 34 numbered frames plus two backups (the spread check; F compares models). Gauss–Newton is not taught: the lab's section 6 uses `laplace()` as a tool. Expected Free Energy stays an idea on this day (its derivation is lab Exercise 3 and returns on Day 4).

| time | block | frames |
|---|---|---|
| 0:00–0:05 | Friston's definition; the amoeba video (a model of the pond grows inside it); the Spanish orchard video | 1 + videos |
| 0:05–0:15 | Active Inference is a Bayesian theory; tools: KL, expectation and entropy, −ln p in nats | 2–5 |
| 0:15–0:45 | Variational Bayes: replace the posterior, the KL, Bayes in logs, substitute, F, the boxed F, surprise + KL, springs (with a question), complexity − accuracy, two slopes | 6–15 |
| 0:45–1:05 | Laplace: the curved map, a Gaussian at the peak, the parabola; F under Laplace (four clicks), the boxed F_Laplace, why σ² = 1/ℋ | 16–21 |
| 1:05–1:20 | Perception and action: lowering F, perception gradient, Kalman as the bottom of F, action, the proportional controller | 22–26 |
| 1:20–1:30 | Planning: the timeline after now, two actions imagined, two reasons to act, how G is computed, the boxed G | 27–31 |
| 1:30–1:45 | break | |
| 1:45–2:50 | Lab, sections 1–4 (about 60 min); sections 5–6 are homework | 33 |
| 2:50–3:00 | Tomorrow | 34 |

**Questions in the bottom bar** (same mechanics as Day 1): 12 (q equals the posterior: F is the surprise), 13 springs (LiDAR ±0.4: the belief rests at 2.37), 14 (belief at the prior: complexity 0, accuracy poor), 16 (s = ±0.6 give the same range: two humps), 24 (prior ±0.5, LiDAR ±0.4: ±0.31), 25 (predicts 2.0, reads 2.6: drive towards the wall), 29 (wall known to 1 mm: nothing to learn), 30 (σ halved: ln 2 ≈ 0.69 nats).

**The derivation block (frames 7–11).** It is 25 minutes of algebra; the springs question on frame 13 is the first chance for the room to think. Write the four steps of frame 9 on the board as they appear. Neal & Hinton (1998) is the source of the rearrangement.

**Laplace (frames 18–21).** Frame 18 expands at the bottom of the bowl (no slope). Frame 19 expands around any μ, which has a slope b; it averages to zero under q because E_q[s − μ] = 0. Say this out loud: it is the step students miss. Frame 21 plots the two σ terms of F (drawn with ℋ = 4) and their sum; the minimum is at σ = 1/√ℋ.

**Perception and action.** Frame 24: Kalman reaches the bottom of F in one step; the gradient steps of frame 23 reach it in many (the lab's Exercise 4 shows both, and the step-size limit η < 2/ℋ). Frame 26: with ∂o/∂a = −1 (driving towards the wall shortens the range) the gain K_P is negative; the set point is the robot's own prediction, and a prior that holds a desire turns the prediction into a goal (Day 4).

**Expected Free Energy.** Frame 27 draws the timeline: real readings before now, imagined ones after. Frame 30 is how the robot computes G: imagine a state, imagine a reading, update as if it were real (Kalman), score how much narrower the belief gets and how far the reading is from the goal, average. The boxed G (frame 31) is that average written with beliefs. The minus sign of the learning term is a choice (Millidge, Tschantz & Buckley 2021): the derivation is in the handout and in lab Exercise 3.

**Expected lab numbers** (seed 0):
- Exercise 1: p(o) = 0.398, surprise 0.921 nats; F − KL = 0.921 for every belief tried (F = 6.935, 1.284, 0.921).
- Exercise 2: the grid minimum at μ = 2.515, σ = 0.185 (1/√ℋ = 0.186); F there = 0.921, the surprise.
- Exercise 3: learning term, imagined average 1.626 against the formula 1.629 nats (good sensor), 0.111 against 0.112 (poor sensor).
- Exercise 4: η = 0.01 and 0.06 settle at 2.517 (the Kalman mean); η = 0.08 diverges, since η(π_p + π_o) = 2.32 > 2.
- Exercises 5–6: 65 points; μ = 4.0015 m, ℋ = 162 500, σ_d = 2.48 mm both ways; 16 points: 5.00 mm (×2.02).
- Exercises 7–8 (homework): 400 scans; uniform σ: scatter 2.29 mm against 2.48 predicted (ratio 1.08); noise along the beam: 2.28 against 2.31 (1.01).
- Exercises 9–10 (homework): ΔF positive and growing with the prior width (fixed model wins at 90°); tilted 1.5°: −87.4 nats, free wins. The backup figure uses another scan (−70.2 nats); the verdicts agree.

## Day 3 — The CORTEX architecture: concepts and memories (`slides/day3.pdf`, `labs/day3.ipynb`, `handouts/handout_day3.pdf`)

**Rewritten 2026-11-27.** 42 numbered frames plus 7 backups (five more were parked in `slides/parked_day3_backups.tex`: the CRDT exchange and its tie rule, amortisation, "one belief asks four questions", the factorised belief). Frame numbers are the footers (n/42). Hand out the Day 3 handout; its glossary carries the day's new words (node, edge, RT edge, owner, CRDT, DDS, factor, cut, NEES).

| time | block | frames |
|---|---|---|
| 0:00–0:12 | Who holds the belief: Day 2's one q against a dozen processes; cognitive architectures (SOAR/ACT-R, layered robots, blackboards) and where CORTEX sits; CORTEX, the whole picture | 1–3 |
| 0:12–0:30 | Infrastructure: a shared graph instead of messages, copies merged by a CRDT, the protocol, sensor data beside the graphs; the CORTEX-in-action video | 4–8 |
| 0:30–0:50 | The working memory: the scene as a graph, a real one, the same graph in 3-D (a QR code and link to the interactive page: let students open it on their phones), two kinds of change, typed records, the RT tree, the ring of past poses | 9–16 |
| 0:50–1:05 | Concepts: four dimensions, the retina, one object from many fragments, class against instances | 17–20 |
| 1:05–1:20 | Memories: timescales, episodic (one episode drawn), long-term object (this chair, again; the agent; its graph; the Bayes match), long-term spatial; back to the whole architecture | 21–32 |
| 1:20–1:40 | One belief, cut among many agents: the product of factors, inside a factor, each agent's F, a pose crosses a cut with its covariance, the cost of cutting | 33–37 |
| 1:40–1:50 | No thresholds: precision against threshold, the tuned threshold that fails, the one threshold that stays | 38–40 |
| 1:50–2:05 | break | |
| 2:05–3:00 | Lab: Exercises 3, 4, 5, 6, 8 in class (about 55 min); 1, 2, 7, 9 at home | 41 |
| — | Tomorrow (Day 4's three blocks: the room, the chair's life, choosing a look) | 42 |

**Questions in the bottom bar** (ask, wait, click): 2 (in a layered robot, where is the belief about the room: spread over layers and messages; in CORTEX one node), 6 (two agents write one node: every copy ends with one of the writes; hence one owner per node), 15 (the robot's pose becomes less sure: Σ' grows, since room←Shadow is on the path), 20 (a third chair: one more node, not another agent), 36 (the pose becomes exact: the two scenes no longer differ), 37 (chair_1 placed 10 cm too far left: table_1 too, the same δ_t moves both), 38 (a chair at 5.1 m: the threshold drops it, the precision weighs it a little less than at 4.9 m).

**Frame 3.** The procedural memory is not running; the frame now says that CORTEX puts its weight on building the representation more than on planning and executing missions. The course copy of the drawing (`figures/cortex_overview.png`) has the thesis figure's `Root` node painted out.

**CORTEX in action (the full-screen slide after frame 8).** In pympress, click once to start; it loops (40.5 s). It is a schematic: the order is illustrative; on the robot everything runs at once. Point at the last line: every path went through the centre. The movie's route never goes through semantic or procedural memory.

**Copies and the CRDT (frames 5–6).** If asked why an arbitrary id may decide which write survives: the winner is arbitrary, but every copy applies the same ranking, so they agree. Agreement is what the CRDT promises; which value is right is what ownership is for. The rule is cortex's default CRDT mode (`core/crdt/delta_crdt.h`, `join_replace_conflict`): a change that has seen another replaces it, otherwise the lower writer id wins; a comment there records that the causal test (`dotin`) carries a known workaround. The optional LWW mode, same-host only, orders by (timestamp, agent id), as `cortex.tex` l.257 describes. The two frames that walked this through are parked.

**Sensor data (frame 8 and the backup "On one computer").** The bars are to scale. Zero copy means readers do not pay again per reader; the driver still writes the frame once. A story if time allows: for months the plane was silently NOT zero-copy, because FastDDS turns data-sharing off for keyed types and logs it only at INFO. Removing one `@key` took the bench from 494 µs at 1.4 % CPU to 186 µs at 0.8 % (commit 48543c5; post-thesis). The configuration said on, and nothing measured it.

**Working memory (frames 9–16).** Frame 14 now says that each node is a belief. Frames 14 and 15 both show the RT tree; 13 adds how a point's covariance is composed along the path. The `root` in frame 10's picture is the viewer's World layer; frame 9 starts from the robot.

**Concepts (frames 17–20).** Frame 17 is the map of the next days; the four dimensions are the course's framing, not the thesis's. Frame 20 is Day 1 again: the class is the prior, an instance the posterior. For a chair the belief is pose only and the shape is the class template; a table's shape is fitted.

**Memories (frames 21–32): say what is not built.** Working, episodic and long-term spatial run. Long-term object memory (LTOM, frame 25, moved from the backups 2026-11-27) is work in progress and not in the thesis. Its box: two chairs back with their own identities after room 1 → room 2 → room 1 (run `ltom_chair2`, belief 0.945 / 0.969); after a restart 14 remembered, 11 same identity, 2 new identities, 0 wrong (run `ltom_auro2`, 2026-11-21). The failure mode is a missed recall, which gives a duplicate identity, never a wrong one. The per-object "still there" belief (Beta permanence) is built on a branch but not yet discriminative: do not claim it. Semantic and procedural are not built, although `cortex.tex` l.519–537 and l.549–570 describe them in the present tense. Frame 23 draws one real episode (a 4-second record from a run on 2026-11-22). **LTOM details (frames 26–28, added 2026-11-27).** Frame 26: ltom_agent is a separate agent; it watches the working memory and keeps one entry per object on disk (uid, class, size, colour, lives and how each ended, a keyframe per room with pose and variance, the best thumbnail from each of 12 directions; `ltom_agent/src/repertory.h`). On entering a room it publishes a recall node (its own type, never `object`) for each remembered object that is not on the graph and whose last keyframe is in that room (`recall_select.h`). The language agent reads the repertory for "last seen in room_1, 15 h ago". Question 26: the memory may not resurrect the chair itself, because only the chair agent owns chairs and only a detection can confirm. Frame 27 is the user's capture of ltom's own graph (domain 5, `figures/d3_ltom_graph.png`): root, one node per room, the objects last seen in each, and `human_1` with a `has` edge and an `owns` edge to a chair (a relation told by the person). It is a one-way projection of the repertory, for the viewer and the language agent; recall itself happens on the working graph. Frame 28 is `common/recall_match/recall_match.h` + `view_atlas.h`: at a birth, the chair agent compares {each candidate, new}: p(x|rem) = (1−π)N(x; μ, Σ+R) + π/A, p(x|new) = 1/A, MAP with ties to new, π = `RecallPiMove` 0.5. The candidate's prior is the owner's last P(exists) when published, so a remembered phantom needs much stronger evidence. With equal priors π cancels from the decision: remembered wins iff N(x; μ, Σ+R) > 1/A (question 28). The plot is illustrative (Σ = 0.15² I, R = 0.10² I, A = 20 m²): 0.98 on the spot (the live runs gave 0.92–0.98), crossing at 0.55 m, 1/3 far away. Appearance is not used (the log ratio is 0). Each memory type opens with an unnumbered transition page. Frame 32 shows frame 3's drawing again to close the section; its question (where was chair_22 yesterday?) has two right halves: the object memory for where, the episodic memory for what the robot was doing. What not to claim about LTOM (re-check before the session): it only offers, the concept agent decides; recall uses position only (the look is the missing term); re-entry recall live has been shown mostly on chairs.

**One belief, cut (frames 33–37).** Frame 33 is the argument of the day: every boundary between processes is a conditional-independence assumption, and whatever crosses it is what survived a marginalisation. Frame 37 is the surprise: each object's spread is right; only the relation between objects is lost (lab Exercise 6: one object 2.50 cm both ways; the relation 1.58 cm exact, 3.54 cm from the agents). The price of that loss does not fall with more frames (Exercise 7): it is information the objects carry about each other, and the product of factors has no place to keep it. The joint in causal order, the seven assumptions and the matrix C are backups.

**No thresholds (frames 38–40).** The rule is not "never a threshold": frame 40 shows one must stay (a bias shared across frames), and the backup "The rule is" names the three that exist and their verdicts. Say that the thesis has not finished this.

**Expected lab numbers** (seed 0; r = (5 cm)², s² = (10 cm)²):
- Exercise 1: bottle_concept's write to table_1 is refused; the sweep removes bottle_1.
- Exercise 4 (2000 runs): NEES with the published covariance 1.07; ignoring it 5.01 (predicted (r + s²)/r = 5).
- Exercise 5: the sum of the agents' F equals the global F (416.839925 with this draw) when s² = 0.
- Exercise 6: one object, exact sd 2.50 cm and agent 2.50 cm; the relation θ₀ − θ₁, exact 1.58 cm and agents 3.54 cm.
- Exercise 7 (home): KL = F(agents) − F(exact) = 2.56 nats, the same at 10, 40 and 160 frames; at s²/r = 4: 1.27 / 5.12 / 11.37 nats for 2 / 5 / 10 objects.
- Exercise 8: near pass, best threshold 2.25 m, rmse 1.02 cm against 0.84 cm with the precision; far pass, best 8.25 m, 4.78 cm against 4.20 cm (the slide's figure, from another run, shows 2 m, 1.01, 0.86, 4.79, 4.22); the near threshold used on the far pass, 300 cm.
- Exercise 9 (home): at n = 2000 the displacement is 3.00 / 3.00 / 2.99 / 0 cm for α = 1 / 0.3 / 0.05 / 0.

## Day 4 — Lowering F, then choosing what to do (`slides/day4.pdf`, `labs/day4.ipynb`, `handouts/handout_day4.pdf`)

**Rewritten 2026-11-27.** 28 numbered frames plus 3 full-screen videos and 70 backups (three duplicate backups were parked in `slides/parked_day4_backups.tex`). Frame numbers are the footers (n/28); the full-screen videos carry no number. Hand out the Day 4 handout: it follows the deck, explains the three videos, and has a section per lab part. Everything shown today runs in simulation; say so on frame 1. Gauss–Newton is not taught: the slides say "a solver lowers F".

| time | block | frames |
|---|---|---|
| 0:00–0:15 | Opening: two examples, one F; perception moves the belief, action moves the robot (3.1 → 0.43 → 0); the two handles of F; the goal as a prior | 1–5 |
| 0:15–0:45 | Example 1, the room: three unseen rooms; the room supplies the likelihood (distance to a wall); localising lowers F over the pose; inside F(X), one cycle; the full loss over a window; **the pose-window video** (40 s); the simulation result | 6–11 |
| 0:45–1:25 | Example 2, the life of a chair: the overlay (mask + belief); the chair's F with the roles swapped; birth; **the birth video** (30 s); absence, the order of cap and weight; one tour; forgetting at one rate and as an inferred rate; the chair's life in simulation; the chair offers a look | 12–21 |
| 1:25–1:50 | Choosing what to do: score a reading not yet read (G); G = epistemic + pragmatic; the chair's two offers; the controller picks the lowest G; **the EFE video** (27 s, on frame 26) | 22–26 |
| 1:50–2:05 | break | |
| 2:05–3:00 | Lab: Exercises 1, 3, 4, 12, 16, 18 in class (about 55 min) | 27 |
| — | Tomorrow: language reads and changes the same beliefs | 28 |

**Questions in the bottom bar** (ask, wait, click): 9 (a chair between robot and wall: its returns pull little, because Huber bounds them; they are weighed, not thrown away), 10 (why corners from past poses but walls only from the newest scan: a corner is a few numbers per pose, a scan thousands of points), 13 (what the chair agent reads from the robot: its pose, with its covariance, as on Day 3), 16 (which order lets a poor view remove the chair: weight, then cap), 26 (why the robot stops detouring: its looks made the chair sharp, so every path teaches little and the goal decides). The other frames end with a lesson line.

**Opening (frames 2–5).** One scalar example carried through: prior N(1, 0.5²) on the distance to a wall, a reading 1.5 m. Perception lowers F to the surprise (0.43); only action, by changing the reading, lowers the surprise itself (to 0 after driving 0.5 m). Frame 3 derives the numbers from the two precisions (25 and 4). Frame 5 is the move students remember: put the goal in the prior and the robot drives until the prior comes true.

**Example 1, the room (frames 6–11).**
- Frame 6 is the Webots capture of the three rooms (`figures/three_rooms_webots.png`); the red circle marks the robot. Building the room's model is itself lowering F over the model; the whole procedure (hidden SLAM, pass/hit grid, outline, explorer, promotion by description length) is in the backups "Estimating a room", "Step 1"–"Step 5" and "The room as a probability model (1/4–4/4)". Use them only if asked.
- Frame 7 defines the distance function d(p): distance from a point to the nearest wall, computed once per room. The code calls it `sdf`, but on a polygon it is unsigned. No expected scan is drawn: the room turns a pose into prediction errors directly.
- Frame 9, one cycle (checked in the code, `room_concept/etc/config.toml`): returns are **Huber-weighted**, `HuberDelta = 0.15`, `ObsSigma = 0.05`, weight u = min(1, δ/|d|). It is **not** a mixture (the chair's is). A chair in front of a wall pulls a bounded amount.
- Frame 10, the full loss: a window of 5 poses (`WindowSize = 5`), a new pose every 10 cm or 0.15 rad; the wall term on the newest pose only (`SdfCurrentSlotOnly = true`); corners on all 5 poses (`corner_max_slots = 5`); wheel factors between consecutive poses; a prior on the oldest pose from the poses dropped (`BoundaryFejSchur = true`, an exact Schur complement, frozen at first estimates). The written-out formula is the backup "The full loss, written out". **Loops:** there are none, because the room is given; a corner seen from two poses does not connect them. Contrast with SLAM, where the map is a variable and loops are the point.
- **Pose-window video** (`media/pose_window.mp4`, 40 s, from `figures/make_window_video.py`): poses join every stride, wheel and corner factors link them, the newest pose's scan hits the walls, and the sixth pose pushes the oldest out, which becomes the prior. The stride is drawn sparser than the robot's.
- Frame 11 (`figures/d4_loc_error_sim.png`): median error 2.3 cm and 0.6°; the peaks come at door crossings.

**Example 2, the life of a chair (frames 12–21).**
- Frame 12 is a real simulated frame from the room session (run `chairfig_1127`, `chair_292`): the YOLO mask labelled chair, with the chair agent's belief drawn as a box. The mask is evidence in pixels; the belief is a box in the room with its doubt.
- Frame 13: the same F as the localiser with the roles swapped: the robot's pose is now given (read from the graph with its covariance), the chair's pose is unknown. The running agent fits **pose only** (x, y, heading); the shape is the class template. Its likelihood is a **mixture**, "chair or clutter" (mixture NLL in the chair fitter). That is the difference from the localiser's Huber.
- Frame 14, birth: one more object is worth adding when what it explains outweighs what it costs, both in nats. **Say the footnote plainly:** the running agent does not compute nats. It counts: a chair is born after 8 observations (`BirthFrames = 8`, `common/instance_tracker/birth_evidence.h`), each weighted by the detector's confidence and by a range factor, (2.5/r)² beyond 2.5 m. Same shape, not calibrated in nats; the backup "What the robot actually computes" says it is a threshold that should not be one.
- **Birth video** (`media/chair_birth.mp4`, 30 s, `figures/make_birth_video.py`, illustrative numbers). Blue bar, **accuracy**: the surprise a chair removes from the points, Σ[−ln p(z|clutter) + ln p(z|chair)]. Clutter is flat; the chair's depth spread grows as r² and equals the clutter spread at the range limit R, so one point is worth 2 ln(R/r) nats, scaled by 0.3 because points of one view share their error. Red bar, **complexity**: the KL of the fitted belief from the class prior, 3 + 3·½ ln(1 + 64n) nats for three unknowns after n points; it jumps on the first points and then grows only logarithmically (about 14 → 17 nats). The dashed line only marks the end of the red bar. Blue grows about linearly with the points, red logarithmically, so a real object is always born; birth at 15 s. Clutter's points are not explained by a chair, so its blue bar stays low.
- Frames 15–16, absence. A frame confirms (a return inside the volume), removes (predicted visible, nothing there) or holds (occluded, out of view, too far). Frame 16 (redrawn 2026-11-27) is the order of cap and weight: 40 beams, each 0.05 % likely to fire, give P(detect) ≈ 2 %. Cap then weight: 0.02 × min(40, 0.4) = 0.008. Weight then cap: min(0.02 × 40, 0.4) = 0.4, the full cap, from a view that could barely see. c = 1 per beam here; ℓ = 0.4 because the beams of one frame share their errors. These are the lab's numbers (Exercise 4).
- Frame 17, one tour (the lab's toy): only absence weighted by visibility, with the walls in the line of sight, removes just the vase. Without walls a chair in the next room votes for its own removal; this was a real defect in three agents (backup "A factor whose argument list was wrong"). Tell it as a story.
- Frames 18–19, forgetting. A constant process noise forgets at one rate; the inferred volatility lets each object infer its rate from its own surprise. **The update differs from the thesis on purpose**: `agents_life_cycle.tex` l.1093–1097 writes ½ e^{−ζ} d − k/2 with d = δθᵀΣ⁻¹δθ, which does not balance (d is already dimensionless). The lab uses the normalised innovation d = ν²/S, saturated as a Student-t so one spurious frame moves ζ a bounded amount, minus its expectation under a calibrated model (0.786 for ν₀ = 4). With the unbounded version one 40 cm spurious mask raises ζ by about 18 in one step and the belief is lost. The thesis marks this construction as design only. In the running chair agent `AI2AgeNominalDtS = 0`, so the covariance freezes when no mask arrives.
- Frame 20, one chair in simulation: born, sharpened, removed on leaving, recalled on return. Recall uses position only.
- Frame 21: the chair publishes an offer, "come and look at me", priced at what the look would teach (5 nats in the example).

**Choosing what to do (frames 22–26).**
- Frame 22: G is the free energy expected over readings the robot does not have yet; the goal is a prior over readings, p(o).
- Frame 23: G ≈ −I + E[−ln p(o)], with I = ½ ln(1 + σ_s²/σ_o²). The table: a look from 1 m, epistemic −0.99, pragmatic 0.58, G −0.41; from 2 m, −0.26, 3.22, 2.96. Several unknowns (the lab): I = ΔH = ½ ln(|Λ + JᵀR⁻¹J| / |Λ|).
- Frame 24: the chair prices what a look would teach it; where to drive is the controller's choice.
- Frame 25: the controller scores every offer, a* = argmin[−ΔH + pragmatic + λ·dist], λ = 0.2 nats/m, with a switching margin so it does not dither. **Say the lesson line:** this has not yet been shown to beat a hand-tuned heuristic (backup "What is not established"), and the covariances are not calibrated, so the prices are not yet fair (backup "Measured: …").
- Frame 26, **the EFE video** (`media/efe_paths.mp4`, 27 s, `figures/make_efe_video.py`, illustrative): a room from above, the door as the goal, a chair known only roughly (blue disc, 2 sd). Every half metre the robot imagines 11 arcs of 2.2 m and scores each with G = −I + λd (I: information its looks would carry about the chair, from range-dependent noise 0.3·r; d: the arc's end distance to the door; λ = 0.9 nats/m). It drives the best one, really looks, updates, and plans again. It bends towards the chair while it is uncertain, then turns for the door once I ≈ 0. **It is a toy rollout, not the robot's controller**: the robot does not sample arcs; it picks among offers (frame 25). The video is there so the idea sticks: epistemic value wins while it is large and dies by succeeding (backup "An offer loses value by succeeding").

**Lab 4 (frame 27).** Ten sections, marked in the notebook as class, home or optional.
- In class (about 55 min): Exercise 1 (birth as ΔF), 3–4 (absence, the order of cap and weight, the tour), 12 (localising), 16 and 18 (ΔH per candidate view, and the λ at which the choice flips).
- At home: 2, 5–6 (forgetting), 13 (NEES), 17, 19, 20–22 (priced offers on the graph).
- Optional: 7–11 (acquiring the room), 14 (symmetry), 15 (the common-mode ceiling). Section 8 is a reading. Route following and the occupancy-grid explorer are in `day4_extra.ipynb`.
- Section 5 falls back to the true room if a student skipped section 4. Pragmatic is a **cost** in the code (`+ aff.attrs["pragmatic"]`; the door's is −1 nat because the robot wants to end there): students meet this sign again in Exercise 21.
- **Exercise 13 is the emotional centre if anyone does it at home:** one wall 1.5 cm out takes NEES from about 2 to about 90. Ask what they expect first; most guess "a bit worse".
- A trap met while building the lab, worth telling: the first scripted route drove the robot through the east wall. The localiser kept reporting a confident pose against the polygon and NEES came out near 6. It looked like a subtle modelling defect; it was a test measuring something else. The route now asserts that the robot stays in the room.

**Expected lab numbers** (seed 0, from `labs/solutions/day4_solution.ipynb`):
- Exercise 1, ΔF after 1..10 detections: sharp 2.9 → −57.5; distant 3.0 → −14.6; reflections 3.0 → +57.8. Exercise 2: two sharp views, a streak of 3 accepts 0 %, ΔF < 0 accepts 98 %; ten reflections, streak 93.7 %, ΔF < 0 0 %.
- Exercise 3–4: a 2 %-resolvable view drops L by 0.0079 (cap, then weight) or 0.4000 (weight the count). Tour: evidence removes the vase at frame 68; without walls also the chair at 174; weighting the count removes the vase at 13; the miss counter removes everything by frame 44.
- Exercises 5–6: E[d̃] = 0.786; predicted floor 10.0 mm. Constant / inferred Q: sd 9.4 / 6.5 mm; moved by the spurious mask 92 / 44 mm (17 / 19 mm ten frames on); sd after the 60 s gap 123 / 45 mm; adapts to the real move in 9 / 15 frames. Biased stream: 5.4σ / 7.6σ out.
- Exercises 7–11 (optional): θ₀ = −17.19° (truth −17.19°); band ±6.7 cm = 3.37σ; a point on the line brings 5.69 nats; walls to ≤ 5 mm and ≤ 0.16°; IoU 0.994, area 23.99 m²; L_edge = 7.56 nats; the table face earns about 82 nats.
- Exercise 12: rms error 2.2 mm, 1.0 mm, 0.036°. Exercise 13: mean NEES 1.95 (true room), 3.81 (estimated room), 89.8 (east wall +1.5 cm); 3 is right.
- Exercise 14: F = −786.82 nats at the true pose and at its half-turn. Exercise 15: actual 8.8 → 6.6 mm (N = 1 → 32); independent 4.9 → 0.9 mm; Woodbury 9.3 → 7.6 mm.
- Exercise 16: ΔH stay 0.693, under the wall 0.705, far west 0.949, behind the table 0.543, east corner 0.587 nats. Exercise 18: far west wins up to λ = 0.06 nats/m; from 0.07 "stay here" wins. Exercise 19: predicted 0.949, realised 0.951 nats; α sd 0.1155° → 0.0568°.
- Exercises 20–22: who held the robot B B B T B B B B T T T D D D D D D; table error 0.7 cm, bottle 1.8 cm; both offers withdrawn at the end.

## Day 5 — Action (`slides/day5.pdf`, `labs/day5.ipynb`)

Frame numbers are the slide footers (n/31). The session is 3 h.

| time | block | frames |
|---|---|---|
| 0:00–0:05 | The loop of the day: belief, offer, selection, route, look | 1 |
| 0:05–0:25 | Why action needs a second functional: flat ∂F/∂a, G derived, two terms, three names for one number, where to look, ln 2 and its assumption | 2–7 |
| 0:25–0:45 | Affordances: who computes which term, the split of knowledge, the price of a view, selecting, hand-off without a rule, the trap, what is not established | 8–14 |
| 0:45–1:10 | From a choice to a motion: obstacles in the geometry, routes as optimisation, Frenet, the law, pole assignment, decay in metres, no obstacle term, two depths | 15–22 |
| 1:10–1:40 | Talking to the robot: plans not driving, questions as queries, imagining, claims backed by tools, inner voice (proposal, marked work in progress), what is measured | 23–29 |
| — | break | |
| 1:45–3:00 | Lab 5 (parts A–D; the body part was removed 2026-11-21) | 30–31 |

**The inner-voice frames (27–28) are a proposal from the Vygotskian Machine draft: nothing is built or run.** The
example on frame 28 is illustrative. Only the wake turn (`allowed_in_wake`) exists in code.

**Do the pole derivation on the board** (frame 19): substitute the law into ė = vφ, φ̇ = ω − vκ and let the
characteristic polynomial factorise in front of them. Then change variable to arc length and let the v
cancel (frame 20). Those two lines are the whole reason L is a length and not a gain.

**Frame 14 is the honesty frame.** It says the derived explorer did not beat the hand-tuned one, and
that the explorer running now is a coverage rule. Do not soften it. The earlier version of this frame quoted
594 layouts and IoU 0.9794 / 0.9806; those numbers are no longer in the thesis, so the frame now follows
`active_inference.tex` l.876–897, which is qualitative. If tesis-12 restores them, put them back with a source.

**Frame 21's caveat is the thesis's own**: no comparison with the re-solved window disabled has been run
(`action.tex` l.1005–1007, l.1459–1464). The toy shows the division of labour, not that it wins.

**Frame 11's gains are illustrative** (the thesis says so at `action.tex` l.42). λ = 0.2 nats/m and the 0.5 nat
margin are the deployed values.

**Material that came from Day 1** (block A of the lab, frames 2–7, `slides/day5_efe.tex`). The key question comes
at the end of exercise 3: *what does ln 2 assume?* Don't let students skip it. The answer (independent errors
across looks) links back to Day 4's common-mode ceiling and to the thread of the whole course. Post-thesis claim
in that answer: the layout explorer's precision term is inert because the map keeps one point per voxel
(memory note `efe-objective-misfit-carries-it`), marked as work in progress.

**The lab was reordered to follow the slides.** A (old block B0), B (new: offers), C (old B: explorer), D (new:
tracking). Part E (the body) was removed on 2026-11-21 to make room for the language section. Exercises
run 1–14. If short of time, skip 2 and 13.
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
