# Wall-SLAM adversarial review — 2026-09-02

Independent review of `src/wall_map.{h,cpp}`, `src/wall_segmenter.*`, `src/room_gn_solver.*`, the live
path in `src/room_concept.cpp` (`wall_slam_after_solve`) and `tools/wall_slam_selftest.cpp`, at commit
`6298895`. Produced by a fresh reviewer session with no shared context; every citation below was then
re-read and confirmed in the code before this file was written. **V** = verified at the cited lines;
**S** = mechanism verified, magnitude or frequency not measured.

Status column tracks what has been acted on. Findings are ranked most severe first.

| # | Sev | Status | Finding |
|---|-----|--------|---------|
| 1 | BLOCKER | fixed 09-02 (host id copied; spur host on a value copy) — bench bit-exact | use-after-free of `E` / `W` in `try_splice` / `try_spur_wraps` |
| 2 | BLOCKER | fixed 09-02 (collect ids, splice last-born first) — bit-exact; ascending order shifts seed 424242 0.942→0.821, see #18 | out-of-range index in the death sweep after `heal_order` |
| 3 | MAJOR | fixed 09-02 (agent runs re_derive on the bench cadence, cadence moved to WallMap::Params) | the live agent never calls `re_derive` — the bench grades a bench-only algorithm |
| 4 | MAJOR | open | the annealed Manhattan factor is inert by construction |
| 5 | MAJOR | open | carried wall information is damping, not a prior (re-centred every solve) |
| 6 | MAJOR | open | two θ₀ estimators; published polygon rotated w.r.t. the re-anchored frame |
| 7 | MAJOR | fixed 09-02 (immediate repair on the copy, loop to closure; publisher keeps last good, never raw) — bench unchanged | `manhattan_polygon()` repair inert on the copy; publisher flip-flops projected↔raw |
| 8 | MAJOR | fixed 09-02; strict net-of-seed purse landed with #10 stage 1 (walls inherit their candidate's support, so the real spur still pays) | spur-wrap purse paid with seeded (untested) bins |
| 9 | MAJOR | open | grid matter never forgets (`hits ≥ 3` is permanent) |
| 10 | MAJOR | stage 1 + MDL toll done 09-02 (one currency for LINE claims; grid stays the AREA currency; the toll is now a derived code length, 7.3 nats/edge) | three incommensurable "nats" currencies against one toll |
| 11 | MAJOR | open | corner explanation is size-blind; `corner_residue` orphaned and not re-anchored |
| 12 | MINOR | open | ≥40 unflagged literals in the body; key params not loadable |
| 13 | MINOR | open | θ₀ M-step weights by `points_seen`, not angular information |
| 14 | MINOR | open | `enforce_manhattan` public + desyncs `bins_s0`; ~2 MB grid copied per solve |
| 15 | MINOR | open | `try_down_jumps` O(N²)·bbox; grid fixed ±15 m around the FIRST pose |
| 16 | MINOR | open | bench PASS is checked on the BEST seed; min 0.942 < 0.95 bar hidden |
| 17 | NIT | open | `seg_to_wall` indices stale within the same `observe()` (display only) |
| 18 | MINOR | open (found by fix #2) | the death sweep is ORDER-DEPENDENT: which wall heal collapses depends on which death was spliced first |
| — | — | ok | thread safety of the live path: no defect found |

## 1. BLOCKER — use-after-free of `E` (`try_splice`) and `W` (`try_spur_wraps`) (V)
`wall_map.cpp:558` `WallLandmark* E = find(order[i]);` is a raw pointer into `walls`. The variant loop does
`:865 walls.push_back(w)` (may reallocate) and `:960 walls.resize(walls_before)` (shrinks, does not restore
the old buffer), then `:962` (debug) and `:971` `best_v = Best{…, E->id, …}` dereference `E`.
Same in `try_spur_wraps`: `:1041 W = find(...)`; each `(resume, tsign)` iteration does `:1153
walls.push_back(T); walls.push_back(M);` then `:1191/:1210 walls.resize(walls_before)`; the NEXT iteration
reads `W->phi`, `W->d` at `:1120-1124`.
Scenario: `walls.size()==capacity()` when a trial pushes; the next read is from freed memory. A stale
`uint64` usually still holds the old value, so it is silent until the allocator reuses the block — then a
wrong host id or a wall spliced at a random line. The hazard is already documented at `:2238` for
`re_derive`; these two were missed.
Fix: copy `E->id` / `W->phi,d` into locals before the trial loop (as `w_id` at `:1111`), or `reserve`.

## 2. BLOCKER — out-of-range index in the death sweep (V)
`wall_map.cpp:378-389` iterates `i` from `walls.size()-1` down and calls `splice_out(w.id)` with the comment
"index i is not reused after this". But `splice_out` (`:392-397`) calls `heal_order()`, which at `:433-435`
erases EVERY wall no longer referenced by `order` — at any index, including below `i`. Scenario: `i=5` dies,
heal collapses a now-adjacent near-parallel pair and drops index 3 ⇒ `walls.size()==4`; next iteration reads
`walls[4]`. Fix: collect dead ids, splice them out in a second loop.

## 3. MAJOR — `re_derive` is bench-only (V)
`grep re_derive src/*.cpp` hits only `wall_map.cpp`; `tools/wall_slam_selftest.cpp:621-625` calls it every 40
frames or after 30 rejections; `room_concept.cpp` never does. Contour re-derivation, grid-IoU adoption, the
surrender veto, the transplant — everything in `:2093-2606` — does not run in the agent. The 0.94-0.97 IoU
figures in the last commits grade an algorithm the agent does not execute. Either wire it into
`wall_slam_after_solve` on the bench cadence, or stop citing the bench as evidence for the live agent.

## 4. MAJOR — the annealed Manhattan factor is inert by construction (V mechanism, S magnitude)
`wall_map.cpp:155` `manhattan_var = max(σ², ε²)` is recomputed in `reclassify_all()`, which `observe()`
calls every frame (`:1818`). Stiffness `1/max(σ²,ε²)`: a wall 2× more tilted gets 4× less pull; the
gradient `ε/var = 1/ε` DECREASES with tilt — a redescending M-estimator whose scale is reset to the current
residual, i.e. a fixed point wherever the wall sits. Quantitatively (S): σ=2° ⇒ λ≈820; one slot's
`WallPointFactor` φ-curvature ≈30 (`room_gn_solver.cpp:495`), carried information after ~1000 slots
≈3·10⁴ — the factor is 1-2 orders of magnitude below what it argues with. "Soft in the loop + hard at the
output" is in practice "no Manhattan in the loop + hard at the output". Either a fixed σ with a proper
robust kernel / a 1-DOF parametrisation, or say plainly the in-loop factor only estimates θ₀.

## 5. MAJOR — carried information is damping, not a prior (V)
`room_gn_solver.cpp:900` `WallPriorFactor(o, w.phi, w.d, w.information)` uses the CURRENT `w.phi/d` as μ;
`:1135-1136` writes the solve back into `w.phi/d`; next solve μ is the new value. Dropped slots contribute
curvature but never pull the wall back to what they observed — they only slow it. The live window's
residual is re-applied every solve for the slot's lifetime — the double counting `wall_map.h:49-51` says it
avoids. Fix: per-wall information-form accumulators `(Λ, Λμ)`, prior mean `μ = Λ⁻¹(Λμ)`.

## 6. MAJOR — two θ₀ estimators (V mechanism, S magnitude)
Re-anchor uses the solver's θ₀ (`room_concept.cpp:2083`); `enforce_manhattan` (`wall_map.cpp:167-175`)
computes its own `points_seen`-weighted θ₀' on the copy. After the one-shot re-anchor the fleet expects
θ₀=0; the published edges lie at θ₀'−θ₀. The bench's "published tilt 0.00-0.06°" (`selftest:1286`)
measures against the live θ₀ and passes only because all ε<σ there. Fix: one θ₀ (run the M-step before
choosing `rot`), weighted by `information(0,0)` (see 13).

## 7. MAJOR — projection repair inert; publisher flip-flops projected↔raw (V mechanism, S frequency)
`wall_map.cpp:205-209` calls `proj.repair_if_crossing()` on a fresh copy; `:2614 if (++crossing_frames_ < 5)
return;` — the copy inherits the live counter (normally 0), so the copy is never repaired unless the live
map has been crossing for 4 frames. When the projection self-crosses or `heal_order` collapses same-class
adjacents (which the projection creates BY DESIGN, `:201-203`), `room_concept.cpp:2103` falls back to RAW.
Frame N: n vertices at 0°; frame N+1: n+2 vertices at 1.5°; frame N+2 back. `room_scene_graph.cpp:222`
treats the vertex-count change as a structure change and rewrites `delimiting_polygon_x/y`; door_concept
trusts wall yaw to (1°)² (`door_scene_graph.cpp:296`). Fix: unconditional repair on the copy; never fall
back to raw — publish the last good projected polygon.

## 8. MAJOR — spur-wrap purse paid with seed money (V)
`wall_map.cpp:320` `seed = clamp(exist_lodds, 0, birth_nats)`; `:326-329` bins created by extent growth are
filled with `seed`; `:355` an untested bin is left unchanged. Every confirmed wall grows new bins at exactly
`birth_nats`. `try_spur_wraps` `:1057-1061` counts `bin >= birth_nats` as SOLID and adds the whole value to
`bin_nats`; `:1204-1208` pays the 30-nat toll from it. Scenario: a segment spills its extent 1.6 m past the
corner into an unobserved doorway (`:1541-1543` grow `s_min/s_max` unconditionally); next frame 6.5
untested bins at 4.6 nats = 30 nats — the wrap fires with zero endpoint returns. The comment at
`:1198-1203` ("bins are beam-endpoint evidence") is false for seeded bins. The surrender rule already
subtracts the seed (`:2528`); the wrap does not. Fix: count only `b − birth_nats`.

## 9. MAJOR — matter never forgets (V mechanism, S live impact)
`wall_map.h:300-301` `is_occupied = hits ≥ 3 or lodds > 1.5`; `wall_map.cpp:1935` `hits` only increments;
`:1924-1925` a latched cell loses 0.02/pass but the `hits` half never clears. `jump_delta_nats`
`:2082-2085` charges `−max(l, 2)` ≥ 2 nats per matter cell for ever. A person crossing three scans, a door
leaf, a moved chair — permanent. The static-truth bench cannot see this. Fix: a return log-odds with the
same decay as free, or `last_hit_ms` + a freshness rule symmetric to `free_ms`.

## 10. MAJOR — three incommensurable "nats" against one toll (V)
`:994` `dnats = is_stub ? c.gain : jump_delta_nats(...)` — `c.gain` is a real log-likelihood ratio
(`wall_segmenter.cpp:46-52`); `jump_delta_nats` sums grid log-odds from ad-hoc increments (`:1928` −0.4/pass,
`:1935` +1.0/hit, `:1925` −0.02, clamp ±4). `:1329-1333` subtracts existence-bin nats (±1/frame,
incidence-weighted) from grid nats. `:1204` pays the same `order_jump_nats` toll in bin nats. The currency
was switched when one failed (`:988-993`, `:1198-1203`). `order_jump_nats=15` per +2 edges, sold as a
geometric prior on order (`wall_map.h:108-115`), implies p(n+2)/p(n)=e⁻¹⁵≈3·10⁻⁷; `:996-1000` concedes it is
"the noise floor of the evidence comparison". This is the core incoherence. Fix: one energy, one currency —
a calibrated inverse sensor model for the grid, every operator scored by ΔE of the same function.

## 11. MAJOR — corner explanation is size-blind; ledger orphaned and not re-anchored (V)
Gate: `:1453` span slack 0.2 m (along the segment only); `:1446` band 3×(map_σ_d+σ_d); `:1458-1460`
obliquity `sin(gate + 2·theta0_sigma())`; `:1461-1468` endpoints tested against INFINITE lines. No length or
point-count bound. A RANSAC tail chord has a derivable maximum length ≈ 2·band/sin α ≈ 0.5 m; a 7 m, 700-point
real diagonal across a modelled corner satisfies every clause and is silenced identically. Length is the
separating statistic the model comparison should use. `corner_residue` has zero readers (only the selftest
scan at `tools/...:976`); `reanchor()` `:2696-2722` transforms walls, candidates, θ₀ — NOT `corner_residue`,
so after the one-shot re-anchor prior entries are in the old frame and new chords are bucketed against them
(`:1476-1477`); the 64 cap (`:1485`) drops silently. Root cause upstream: the segmenter leaves 2-4σ tails
and re-fits them — absorb leftovers to the nearest extracted line, or MDL-refuse a hypothesis explicable by
existing lines.

## 12-17. MINOR / NIT
- **12** ≥40 unflagged literals in the body vs 11 in `Params`: 0.35 rad / 0.35-0.5 m twin tolerances
  (`:310-311, 546-548, 570, 1097-1098, 1243-1244, 1287-1288, 2252-2253, 2352, 2366`); `sin<5e-2` (`:420,
  1833`); ±1.0 m corner slack (`:910, 1183`); 500 pts confirmed (`:116, 2355`); 50 pts (`:1705, 2190`);
  0.2 m micro-edge (`:1702`); +0.02 IoU margin (`:2512`); weak info (30,15) (`:2194, 2236`); 5 crossing
  frames (`:2614`); snap/par/DP/bias/contour cells (`:2183, 2273-2277, 2133, 2214, 2128`); 0.6 m margin and
  `max(l,2)` (`:2067, 2085`); 0.4 m stub scan (`:742`); grid increments (`:1925, 1928, 1935`). None of
  `manhattan_*`, `order_*`, `adopt_surrender_nats`, `stub_*`, `max_candidates` is loadable
  (`room_config.cpp:100-125`).
- **13** θ₀ M-step (`:171`) weights by cumulative `points_seen`: never decays, double-counts a parked
  robot, summed on merge (`:1872`); a 20 cm jog with 5000 points has ~zero φ-information
  (`line_fit.h:93`) yet full weight. Use `information(0,0)`.
- **14** `enforce_manhattan()` is public and documented "call after each solve" (`wall_map.h:245-248`) yet
  `:189-193` shift `s_min/s_max` without `bins_s0` — a landmine for the first caller. The copy at `:199`
  carries `FreeGrid` (376² cells × 14 B ≈ 2 MB + `comp_cache_`) every solve, twice on the re-anchor frame
  (`room_concept.cpp:2076, 2091`); the projection needs walls/order/θ₀ only.
- **15** `try_down_jumps` `:1292-1298` all pairs × a grid bbox scan each; grid `:1894-1896` is a fixed
  ±15 m box around the FIRST pose — beams beyond are dropped by `in()` for good.
- **16** `selftest:1341` keeps the BEST seed as `R7`; `:1394/1402` check `IoU ≥ 0.95` on it. Min 0.942 is
  under the bar and "ALL PASS" hides it. Per-seed soft→projected deltas are +0.009 / +0.007 / −0.026; the
  "floor pays ~0.002" claim compared minima across seeds (unaligned measurements). Seed 424242's −0.026 is
  unexplained (S: heal collapsing a real pair on the copy).
- **17** `fr.seg_to_wall` (`:1524`) indexes `walls`, which `:1717, 1781, 1810, 1842` erase later in the same
  `observe()` — display only.

## Thread safety — clean (V)
All `wall_map_` mutation is on the localizer thread (`room_concept.cpp:213-217` → `:774/:585`, `:2674`,
`:2646`, `:4395`, `:2867`, `:2199`). The main thread reads `derived_polygon_` by value under
`wall_map_mutex_` (`room_concept.h:1086-1087`) and `res.wall_view` copies. `configure_room_estimate`
(`:1942`, unguarded `wall_map_ = {}`) runs in `initialize()` before `start()` — safe today, a race if ever
called at runtime. NaN write-back is guarded (`room_gn_solver.cpp:1091`).

## Design verdict
Not one generative model — five mechanisms with five scoring functions joined by rate limits ("one per
frame") and hysteresis. (i) The factor graph on (poses, walls, θ₀) is sound in form but both of its priors
are weakened to inertness (4, 5). (ii) The grid is an inverse-sensor occupancy map whose log-odds are spent
as likelihood nats (10). (iii) `observe()` runs nine discrete operators per frame — splice, spur-wrap,
down-jump, micro-prune, zero-evidence cull, Manhattan evict, ghost sweep, crossing repair, heal — each with
its own gate, currency and exemptions; the cap/mirror exemptions and counter-exemptions are the fingerprint
of operators fighting each other. (iv) The one global criterion (contour + grid IoU + surrender) is
bench-only (3). (v) The output projection is driven by a second θ₀ estimator (6).

The formulation that replaces the gates: one energy E(order, {φ,d}, θ₀ | points, grid) = point likelihood
(exists) + grid likelihood under a calibrated beam model (missing) + an MDL/geometric order prior (exists,
uncalibrated) + Manhattan prior (exists, disabled). Every operator becomes a proposal accepted iff ΔE<0 in
the same currency; the surrender rule and the cap/mirror exemptions vanish because the bins' likelihood is
already in E.

Genuinely good: the closed-cycle state space (junk has nowhere to live); per-bin existence with occlusion
holding; the splice grammar (notch / corner-cut / run-replacement / wrap-completion); the stub
discriminator's insight that two hypotheses differ over one observable region; honest gauge handling; the
refusal-counting instrumentation that found the author's own dead ends.

## Literature
- **S-Graphs+** (Bavle et al., arXiv 2212.11770): planes stay variables with all keyframe factors kept (or
  marginalised at a fixed linearisation); the room factor constrains the room centre from 4 planes. Nothing
  there resembles a re-centred prior (5) or an annealed room-plane variance (4).
- **Manhattan-frame estimation** (Straub et al., Mixture of Manhattan Frames, CVPR'14; real-time MF
  rotation, RSS'17): vMF mixture with EM — soft assignment, weights = concentration (information), not
  point counts. `enforce_manhattan` is a one-iteration hard-EM with the wrong weights (13); a proper E-step
  also resolves the ±45° class boundary that `classify()` hard-assigns.
- **Floorplan from occupancy**: Cabral & Furukawa, CVPR'14 — the Manhattan polygon as a globally optimal
  shortest path in a grid graph with a per-vertex penalty: the principled form of `re_derive` and of
  `order_jump_nats` (an MDL vertex cost with units). FloorNet (Liu, Wu, Furukawa '18) for the learned variant.
- **Polygon complexity**: MDL (Rissanen) derives the order prior (code length per vertex ≈ log #cells);
  reversible-jump MCMC (Dick, Torr, Cipolla IJCV'04) accepts dimension-changing proposals under ONE
  posterior — what splice / down-jump / adopt approximate without detailed balance.
- **Line landmarks** (PL-SLAM, orthonormal representation): the 2-D Hesse (φ,d) choice is fine; the defect
  is not the parametrisation but that its information is discarded each solve (5).

## 18. MINOR — the death sweep is order-dependent (V, measured 09-02)
Found while fixing #2. Splicing the same set of dead walls in ascending index order instead of the old
descending one moved seed 424242 from IoU 0.942 to 0.821 (deaths 27→48, walls 12→9) and seed 7 from 0.953
to 0.938; the descending order reproduces the old bench bit-exactly on all three seeds. The set of deaths
is identical; what differs is which near-parallel adjacency `heal_order` collapses after each splice —
so a wall's survival depends on the enumeration order of its neighbours' deaths, not on evidence. Kept the
descending order (validated behaviour). A principled sweep would splice all deaths first and heal once.

## Bench record for commit A (fixes #1, #2, #7, #8)
`wall_slam_selftest` at 931c67f (base) and after the four fixes: seed 7 / 1001 / 424242 IoU 0.953 / 0.967 /
0.942, published tilt 0.06° / 0.00° / 0.02°, bit-identical. The one FAILURE (Hausdorff 0.443 m ≥ 0.20 on the
best seed) is pre-existing at base. Bisect: #1 alone, #7 alone, and #2 (descending) alone are each
bit-exact; #2 ascending shifts (#18); #8 with a net-of-seed purse costs seed 1001 (0.932).

## Bench record for #3 (re_derive wired live)
Cadence (40 frames / 30 rejections) moved from bench literals into `WallMap::Params::rederive_*`; the agent
runs `re_derive` in `wall_slam_after_solve` after `merge_indistinguishable`, before the polygon is derived —
the bench's order. Bench identical (0.953 / 0.967 / 0.942, tilt 0.06° / 0.00° / 0.02°): the bench already
executed it; what changed is that the agent now does too. Live effect UNRUN.

## Record for #10 stage 1 — one currency for line claims (2026-09-02)
Design: every claim about matter ON A LINE is priced in the same nats — per extent bin, ≤1 nat per frame,
net of the birth seed, saturating at birth_nats: a candidate accrues `Candidate::bins` (the per-point gain now
decides birth only; it was 56,472 nats for one 12k-point candidate against a 15-nat toll), a wall born from a
candidate inherits them on top of its seed, a stub splice pays with them, a spur wrap pays with its overshoot
bins net of seed PLUS its grid term, a down-jump surrenders net bins, adoption's surrender veto was already
net. Area claims (boundary splices) keep paying in the grid's cell log-odds.
Measured (toll 15, IoU seed 7 / 1001 / 424242; base 0.953 / 0.967 / 0.942):
| variant | 7 | 1001 | 424242 | verdict |
|---|---|---|---|---|
| full single scalar (per-frame grid + line in/out + on-line exclusion + strict wrap + net rent) | 0.884 | 0.000 | 0.885 | seed 1001 dies: 133 adoptions, unclosed 4-wall map; toll-independent (same at 8/10/20) |
| … without per-frame grid | 0.904 | 0.913 | 0.852 | |
| … without on-line exclusion | 0.949 | 0.928 | 0.933 | 424242 churns 820 births / 811 deaths |
| … without net rent | 0.891 | 0.916 | 0.960 | |
| … exclusion restricted to MATTER on a line | 0.884 | 0.000 | 0.932 | 1001 identical ⇒ not the accounting |
| s1: candidate bins pay the stub; inheritance | 0.954 | 0.967 | 0.921 | KEPT |
| s1 + line support entering on boundary splices | 0.939 | 0.907 | 0.913 | double counts the grid's matter reward |
| s1 + support surrendered on boundary splices | 0.927 | 0.967 | 0.928 | |
| s1 + matter-on-line exclusion from the grid | 0.834 | 0.949 | 0.857 | removes the grid's only line reward |
| s1 + strict wrap purse + grid term (+ net rent) | 0.954 | 0.967 | 0.921 | bit-identical to s1 — KEPT |
Why the single scalar fails: `jump_delta_nats` already contains a line term for boundary claims — matter
released to the exterior IS "the edge sits on matter" — so adding candidate support double counts, and
excluding on-line matter removes the reward. The coherent split is by CLAIM TYPE (line vs area), not one
number per cell. Seed 424242's −0.021 is trajectory chaos: every proposed stub is still accepted (1 of 1, at
76 nats vs 30), the second stub of base's run is never proposed. Still open: the toll is a bench-calibrated
constant (an MDL vertex cost is the principled replacement); adoption still decides on grid IoU + surrender.
Bench-only: `WS_ORDER_JUMP_NATS=<v>` overrides the toll for sweeps.

## Record for #10 stage 2 — the toll as a code length (2026-09-02)
`order_jump_nats = 15`, `parity_jump_nats = 2` and the hand-tuned refund are replaced by
`WallMap::edge_code_nats()`: a two-part description length per edge (Rissanen 1978; Floor-SP's complexity
term) — ln 4 for the Manhattan class plus ln(2·sensor_range / grid cell) for the offset at the data's
resolution = 7.29 nats per edge at 15 m and 8 cm, i.e. 14.6 per +2 edges where the bench had settled on 15
(the literature doc's prediction: "if it lands near 15 the toll is vindicated"). An off-class edge names its
angle at the same resolution, ln(2π·range/cell) = 7.07 instead of ln 4 — the odd-jump surcharge becomes 5.7
nats instead of 2. What the code cannot derive is how many NEW lines a move names, so those are structural
counts swept on the bench: a replacement names one line (`replace_code_edges = 1`, the noise floor a free
replacement lacked), a spur wrap names its cap and mirror (`wrap_code_edges = 2`).
| replacement names | wrap names | seed 7 | 1001 | 424242 |
|---|---|---|---|---|
| 1 | 2 (kept) | 0.954 | 0.968 | 0.921 |
| 1 | 4 | 0.954 | 0.968 | 0.921 |
| 2 | 2 | 0.545 | 0.967 | 0.921 |
| 2 | 4 | 0.545 | 0.967 | 0.921 |
Charging a replacement two lines (14.6, the old 15) puts seed 7 on a knife edge: one replacement with
14.6 < Δ ≤ 15 nats is accepted and amputates. The refund fraction measured inert at 0.5 / 0.7 / 0.85 / 1.0,
so it is set to the principled 1.0 and documented as a switching cost. Bench-only overrides:
`WS_REPLACE_EDGES`, `WS_WRAP_EDGES`, `WS_KEEP`. Still open in #10: adoption decides on grid IoU + veto.
