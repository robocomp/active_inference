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
| 4 | MAJOR | measured 09-03, NOT inert: ×0.01 tilts walls 8-13°, ×100 breaks nothing — the factor is what holds in-loop tilt to 1-3°; no change | the annealed Manhattan factor is inert by construction |
| 5 | MAJOR | fixed 09-03 (information-form prior: Λ, μ from dropped slots only; birth precision kept for gating) — joint solve 0.031→0.023, pose RMSE down on all seeds, 39/40 checks | carried wall information is damping, not a prior (re-centred every solve) |
| 6 | MAJOR | CONFIRMED and quantified 09-03: the published polygon is internally EXACTLY rectilinear (0.00° on 3 seeds); its 0.02-0.41° against the live θ₀ IS θ₀'−θ₀. Still open | two θ₀ estimators; published polygon rotated w.r.t. the re-anchored frame |
| 7 | MAJOR | fixed 09-02 (immediate repair on the copy, loop to closure; publisher keeps last good, never raw) — bench unchanged | `manhattan_polygon()` repair inert on the copy; publisher flip-flops projected↔raw |
| 8 | MAJOR | fixed 09-02; strict net-of-seed purse landed with #10 stage 1 (walls inherit their candidate's support, so the real spur still pays) | spur-wrap purse paid with seeded (untested) bins |
| 9 | MAJOR | open | grid matter never forgets (`hits ≥ 3` is permanent) |
| 10 | MAJOR | stage 1 + MDL toll done 09-02 (one currency for LINE claims; grid stays the AREA currency; the toll is now a derived code length, 7.3 nats/edge) | three incommensurable "nats" currencies against one toll |
| 11 | MAJOR | open | corner explanation is size-blind; `corner_residue` orphaned and not re-anchored |
| 12 | MINOR | open | ≥40 unflagged literals in the body; key params not loadable |
| 13 | MINOR | open | θ₀ M-step weights by `points_seen`, not angular information |
| 14 | MINOR | open | `enforce_manhattan` public + desyncs `bins_s0`; ~2 MB grid copied per solve |
| 15 | MINOR | grid now re-anchored with the map 09-03 (was never transformed; live-only bug, 0.936→0.378 free-inside without it) | `try_down_jumps` O(N²)·bbox; grid fixed ±15 m around the FIRST pose |
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

## Record for the adoption judge and the forward-model referee (2026-09-03)
A forward-model referee (Thrun 2003 beam likelihood against the polygon; `Params::forward_referee`,
bench `WS_REFEREE=1`) logs every judged structure change next to its own verdict; the bench scores both
against the single-step truth (did the trial raise IoU with the real layout?). Findings: (1) the forward
model's Δ log-likelihoods are 10⁴-10⁵ nats (792k beams as independent, σ 3 cm) — its verdict is the bare
sign, hypersensitive to centimetres of wall offset and blind to area: it would accept 12 amputations of a
quarter of the flat where a real partial wall collects more hits than pass-throughs; (2) the incumbent
judges' errors are small (median 0.001 IoU); (3) contour adoption refused 178 cycles the single-step truth
preferred — 118 of them self-crossing and refused unjudged, the rest with negative grid terms from offset
noise. Acting on (3) was measured and REJECTED: judging adoption by the one energy (grid + support in −
surrender − code) gives 0.926/0.969/0.918; repairing self-crossings before judging gives 0.936/0.928/0.582
under the incumbent and 0.940/0.793/0.919 under the energy, with hundreds of births and deaths — cycles
the single-step truth likes and the map cannot live with. The single-step oracle is not the long-run
criterion; the 0.02 margin and the veto are hysteresis against exactly this. Both switches stay in Params
for the bench. Item #3's adoption criterion therefore stands as it was.

## Record for #5 (information-form wall prior) and #4 (Manhattan factor), 2026-09-03
#5: `WallLandmark::prior_info`/`prior_mu` accumulate what the dropped slots said (Λ' = Λ + H, Λ'μ' = Λμ +
H·x_lin − b, fused at merges, transformed at re-anchor); `WallPriorFactor` pulls toward μ instead of the
wall's current estimate. First attempt seeded the prior with the birth-time line-fit precision (points still
in the window): joint-solve loss 0.03 → 39, IoU 0.969 → 0.889 — double counting; the prior now starts at
zero and `information` keeps its gating / corner-sigma role. Result: joint solve 0.031 → 0.023 in 3 it (7),
pose RMSE 0.028/0.023/0.057 → 0.022/0.011/0.015, IoU 0.951/0.969/0.943 → 0.946/0.889/0.969, Hausdorff
0.73/0.35/0.30 → 0.30/2.43/0.20; 39/40 checks (Hausdorff ≤ 0.20 and IoU ≥ 0.95 both PASS on the best seed
for the first time). Seed 1001's 213 deaths are all zero-point jogs from 34 adoptions — adoption churn,
the next thing to look at, not the prior.
#4: bench-only `manhattan_gain` (WS_MANHATTAN_GAIN): ×1 tilt 8.4°/2.3°/2.7° (before projection), ×0.01
2.6°/8.5°/8.5° with IoU 0.935/0.951/0.932, ×100 2.3°/1.1°/1.1° with 0.956/0.934/0.960. The factor is not
inert; the review's stiffness comparison ignored that point data pull mostly along d. No change.

## Record for #6 (measured 2026-09-03) and level 2's rectilinearity invariant
A second bench line separates the two questions the old "published tilt" conflated: `internal-tilt` measures
every published edge against the polygon's OWN axis frame (the length-weighted circular mean of its edge
directions mod 90°), `published-tilt` against the live θ₀. Result on the three seeds: internal 0.00° / 0.00°
/ 0.00°, published 0.20° / 0.41° / 0.02°, frame gap θ₀'−θ₀ 0.20° / 0.41° / 0.02°. So the projection does
exactly what it claims — what leaves the agent is rectilinear by construction — and every degree of
"published tilt" is the frame gap of #6, not a leaning wall. The gap grew from 0.00-0.06° (39d087e) to
0.20-0.41° with the information-form prior, which is the same finding from the other side: two estimators
that used to agree by luck.
Level 2 was found to be the one thing that could break rectilinearity: a step flush with the end of its edge
has a zero-length side, and collapsing that vertex left a diagonal (seed 424242, internal tilt 0.51°). Steps
are now kept half a cell clear of both ends, and any trial that would leave an edge not parallel or
perpendicular to its host is refused outright. Internal tilt back to 0.00° on all three seeds, IoU and the
level-2 features unchanged (0.946/0.889/0.969; right pillar 0.18, alcove 0.28 on the best seed).

## Batch pass at saturation — measured on 12 seeds (2026-09-03)
Question: the between-seed spread is far larger than any algorithmic change; is that a DATA difference or a
PATH difference? Test: at the end of each run, re-derive the cycle from the final grid and grade it. Bench
harness `WS_SEEDS=n` (12 seeds, 42 s) and a bench-only `adopt_judge = 2` (adopt any closed cycle).
| | mean | median | min | max | std |
|---|---|---|---|---|---|
| online (12 seeds) | 0.914 | 0.923 | 0.805 | 0.969 | 0.041 |
| batch, forced, ≤12 iterations | 0.916 | 0.910 | 0.863 | 0.949 | 0.025 |
| oracle: best of the two per seed | 0.934 | — | 0.882 | 0.969 | — |
Findings. (1) The incumbent adoption judge adopted NOTHING at saturation on 12 of 12 seeds — the online
cycle is a fixed point of it, so simply re-running the re-derivation gains exactly zero. (2) With the
self-crossing repair the judge took one cycle, on the seed that had collapsed to 8 walls: 0.889 → 0.954,
Hausdorff 2.427 → 0.439 m. The repair that was measured HARMFUL online (churn) is valuable as a one-shot at
saturation. (3) Forced re-derivation neither helps nor hurts on average: it collapses the variance (std
0.041 → 0.025) by rescuing the failures (0.805 → 0.943, 0.917 → 0.949) and wrecking the successes (0.969 →
0.904, 0.946 → 0.904). The final grid carries a common ~0.91 quality regardless of path; the online model
sometimes beats it, because it integrates evidence the grid discards (existence bins, the candidate bank,
freshness). (4) So the value is in SELECTING, and the ceiling for a perfect selector is +0.020 mean and
+0.077 on the floor. (5) The forward beam model cannot be that selector as it stands: it chose the online
cycle 12 times out of 12 with Δ log-likelihood between −7·10⁴ and −2·10⁶ against a code length of ±100 — a
constant predictor. The reason is now clear: the grid contour's walls sit about one cell (8 cm) off the
returns while the online walls are line-fitted to a centimetre, and 792k beams at σ = 3 cm make that
sub-cell placement worth 10⁴× more than the whole topology. A structural comparison must first profile out
each candidate's edge offsets (re-fit its edges to the beams), which is the next experiment.

## Profiling out the edge offsets — the experiment, and what it found (2026-09-03, 12 seeds)
The saturation comparison was dominated by where edges SIT, not by structure, so each candidate polygon's
edges were slid along their own normals onto the returns assigned to them (per-edge nuisance offset,
median of the perpendicular residuals of beam endpoints within 25 cm; directions kept, vertices
re-intersected, so rectilinearity survives) and only then scored.
| | mean | median | min | max | std |
|---|---|---|---|---|---|
| online | 0.914 | 0.923 | 0.805 | 0.969 | 0.041 |
| + offset profiling | 0.944 | 0.954 | 0.825 | 0.987 | 0.047 |
| + judged selection online/batch | 0.956 | 0.966 | 0.908 | 0.987 | 0.045 |
THE FINDING IS NOT THE SELECTOR. Profiling improved IoU on 12 seeds of 12, mean +0.030, best result ever
measured 0.987 — because the correction is a BIAS, not noise: the signed median offset is negative on every
seed and essentially every edge, −0.020 to −0.085 m, mean ≈ −0.053, i.e. the published edges sit that far
inside the surface the beams return from.
**WHERE THE BIAS LIVES — measured next, and it is NOT the walls.** Per-wall median residual of the returns
each wall owns: every well-observed wall is on its data to 1-2 mm (+0.001 to +0.002 m at 12k-151k points and
1000+ frames). The offset comes from two other places. (a) LEVEL-2 DECORATIONS: the published polygon has 38
edges where the coarse cycle has 22, so 42% of it is level-2 steps, and those were built from 8 cm cells and
padded half a cell beyond them. (b) THINLY-OBSERVED WALLS: walls with 0-5000 points and 0-35 frames sit 4-11
cm inside (one at 15.8k points but spanning a recess reads −0.19, an artefact of the measurement's own
assignment band). So the earlier reading of this table — "every published wall sits 5 cm inside" — was
wrong, and the cure is not in the solver.
Secondary result: with the offsets profiled out, the forward beam model becomes usable as a selector for
the first time (9 of 12 against 7 of 12, and it can now say BATCH — it did so on the worst seed, correctly,
0.825 → 0.971). Its Δ log-likelihoods are still 10⁵-10⁶, so the sign is all that can be trusted.

## Level 2 fits its three degrees of freedom (2026-09-03)
Following the correction above: the cluster's cell bounding box now only LOCATES a feature, and the step's
three parameters — the two ends along the wall and the depth — are fitted to the returns. For an axis-aligned
step the three decouple into three one-dimensional robust fits (the median of the returns owned by each
face), so no joint optimisation is needed; `Params::level2_fit`, bench switch `WS_LEVEL2_FIT`.
| seed 7 / 1001 / 424242 | box only | fitted |
|---|---|---|
| published IoU | 0.946 / 0.889 / 0.969 | 0.953 / 0.898 / 0.978 |
| right pillar mis-explained | 0.38 / 0.38 / 0.18 | 0.09 / 0.31 / 0.11 |
| alcove mis-explained | 0.32 / 0.37 / 0.21 | 0.14 / 0.21 / 0.11 |
| left pillar mis-explained | 0.26 / 0.34 / 0.55 | 0.48 / 0.53 / 0.55 |
| per-edge offset remaining | −0.043 / −0.085 / −0.020 m | −0.001 / −0.018 / +0.013 m |
IoU improves on every seed and the offset that started this thread is gone, which confirms the diagnosis: it
was the decorations. Two features improve sharply; the LEFT PILLAR gets worse, and it is the one that sits
against a corner — its side-face band catches the perpendicular wall's returns. Features at a corner remain
level 2's known gap (a corner step is +2 edges and is not offered).

## What the local fits were losing, and what fixing them bought (2026-09-03)
Three defects in the level-2 fit, each found by asking what a face is allowed to look at:
1. **A band that reached the host wall.** The front face collected returns within 3 cells (24 cm) of its
   estimate, so a 20 cm deep column was measured against the wall's own returns at depth 0 and read
   shallower than it is. Bands are now `min(3 cells, 0.4 x the dimension being measured)`, so a face can
   never see the structure it is measured against.
2. **No incidence gate.** A beam nearly parallel to a face has its range error projected ALONG the face;
   it cannot place it. Returns with |n·d| < 0.2 are dropped.
3. **One pass.** The fit now runs twice — once wide to find the faces, once at one cell (about 4σ) to
   measure them — which removes what the wide band admitted.
And two guards were wrong rather than imprecise: the minimum feature size was two cells (16 cm) applied
AFTER the fit, so a spur measured correctly at 10-16 cm was thrown away; it is now physical, 7 cm. The
residual-cell clearance was tried at 1.0 cell and reverted to 1.5: at 1.0 a clean wall's own surface noise
forms clusters and three synthetic rooms grew spurious steps (notch Hausdorff 0.020 -> 0.141 m).
**Measured on 50 random rooms, identical population, the only difference being the fit:**
| | cell box only | fitted |
|---|---|---|
| wall column | 35 of 138 (25%) | 73 of 138 (53%) |
| alcove | 78 of 155 (50%) | 109 of 155 (70%) |
| corner column | 36 of 74 (49%) | 44 of 74 (59%) |
| spur | 9 of 100 (9%) | 12 of 100 (12%) |
| IoU mean / median | 0.901 / 0.936 | 0.910 / 0.946 |
Apartamento, same seeds: IoU 0.953 / 0.898 / 0.978 -> 0.955 / 0.902 / 0.980, and the left pillar goes from
0.48 / 0.53 to 0.17 / 0.20 mis-explained on the two seeds where it is not the graded one.
**A caution about the earlier spur headline.** The 2% -> 12% jump between the two population runs is NOT
attributable to the algorithm: the generator changed in the same step (L-shaped bases, and a reach cap that
replaced the old depth limit, so spurs are deeper). On identical rooms the 7 cm size floor is worth 2
points of spur rate and nothing else. Hausdorff median also rose with the fit (0.783 -> 1.033 m): more
features are found, and a found-but-mis-sized feature puts a vertex where none was before.
**L-shaped rooms are the harder half**: 19 of the 50 rooms have a reflex corner and average 0.859 IoU
against 0.942 for the 31 rectangles.

## The corner step (2026-09-03)
Level 2 could only step the middle of an edge, so a feature standing in a corner had no representation: it
was 41% of the population's corner columns and the apartamento's left pillar. A residual zone that reaches
an end of its edge now replaces the corner VERTEX with three instead — **+2 edges, not +4**, so a corner
feature is cheaper to describe because it reuses the corner, which is the right economy. Only matter at a
CONVEX corner is offered. The fit skips the face that IS the corner, since the returns there belong to the
neighbouring wall, and fits the other side and the depth as before.
One detail decided whether the operator existed at all: the reach test must be LOOSER than the clearance
that produced the footprint. A residual cell may not come closer than 1.5 cells to the perpendicular wall,
so a real corner column's cells stop 12 cm short of the corner; testing "reaches within 1.5 cells of the
end" therefore fired once in an entire run. At twice the clearance it fires 22 times. A mid-edge feature
near a corner is not mis-read as one for free: the step would claim the free gap between them as exterior
and the grid term charges for it.
**Measured on the 50-room population, identical rooms, the corner step the only difference:**
| | without | with |
|---|---|---|
| corner column | 44 of 74 (59%) | 54 of 74 (73%) |
| wall column | 73 of 138 (53%) | 75 of 138 (54%) |
| alcove | 109 of 155 (70%) | 109 of 155 (70%) |
| spur | 12 of 100 (12%) | 12 of 100 (12%) |
| IoU mean / median | 0.910 / 0.946 | 0.911 / 0.947 |
Apartamento: seed 7 0.955 -> 0.957, the other two unchanged, published polygon still internally
rectilinear at 0.00 deg. The left pillar of the real flat is still not explained on the graded seed; it is
0.6 m from its corner, so it is neither a clean mid-edge feature nor a corner one.

## The spur, diagnosed (2026-09-03)
Why the population finds one spur in eight, traced rather than guessed.
1. **The level-1 wrap is proposed constantly and refused on geometry.** Over three generated rooms the
   wrap machinery ran 1985 times and committed 3: of the refused trials, 1043 failed the interior test and
   508 failed to close. Some produce wild geometry (an edge 26 m long). The grammar is not the bottleneck
   in the sense of being absent — it is being offered and rejected.
2. **Level 2 does see spurs**, as thin deep zones — 6 of 48 proposals in those rooms — and the economy
   judges them correctly given what it is shown: one was accepted at gain 54 against cost 29, one refused
   at −7.6 because its claimed strip came out 18 matter cells against 19 free.
3. **The reason the strip is half free: one-sided visibility.** In 13 of 16 fits only ONE side face had
   returns; a thin protrusion is normally seen from one side. The unseen face then keeps the padded cell
   box, which is far wider than the wall.
4. **And the grid cannot fix that, because its cell is the size of the feature.** A 13 cm wall's latched
   matter cluster measures 0.317 m — an 8 cm grid blurs a thin wall to 2.5x its width. So a WIDTH test
   cannot separate a spur from a pillar: it either rejects every spur or accepts the real flat's 0.33 m
   pillars and shrinks them (measured, −0.008 IoU). The discriminator that works is ASPECT: a protrusion
   more than twice as deep as it is broad is a piece of wall, and the far face of a wall is one thickness
   away. With that gate the unseen face is placed by `stub_thickness` instead of by the cell grid.
Measured on the 50-room population, identical rooms: spur 12% -> 13%, **Hausdorff median 1.033 -> 0.894 m**,
IoU mean 0.911 -> 0.912, everything else unchanged; apartamento unchanged at 0.957 / 0.902 / 0.980.
The prior therefore improves how well a found spur is SIZED far more than how often one is FOUND. That
split is the real conclusion: **detection is grid-limited and measurement is beam-limited**. Raising the
spur rate needs evidence at beam resolution in the detector too — the existence bins along a wall line,
which is what the level-1 wrap already uses and what item 1 above says is being refused on geometry.

## Simulating the wall-to-ceiling line, and fusing it with the LiDAR (2026-09-03)
The bench now contains the thing that makes a ceiling line worth having, and the line itself.
**Furniture** (`WS_FURNITURE=1`): axis-aligned boxes standing against the walls. The LiDAR band cannot see
through them; the ceiling junction three metres up is never blocked by them. Structure is unchanged — the
truth polygon is the same — so the score measures exactly whether the hidden walls were recovered. Without
occluders the two sensors see the same walls and the camera can only add noise, so a bench without
furniture cannot answer the question at all.
**The ceiling line** (`WS_CEILING=1`): for each azimuth, the junction is seen at elevation
atan(dh / r) with dh the ceiling height above the camera; the camera measures that ANGLE with the scatter
we measured on the fitted contour (0.065 deg), and the range follows. Grazing directions are dropped.
**How the two fuse.** Both measure the same thing — a point on a wall — so they join ONE cloud and go
through one segmenter and one association. Only the precision differs, and that is what is carried: a
per-point weight equal to the ratio of variances. From r = dh / tan(alpha),
sigma_r = (dh² + r²)/dh · sigma_alpha, constant for the LiDAR and quadratic in range for the camera. At
2 m they are comparable, by 8 m the camera is 4 cm against the LiDAR's 2 cm, and the weight says so, which
is what stops a distant ceiling point from dragging a wall the LiDAR already knows well.
**Measured, 20 rooms, same seeds and same rooms in every arm:**
| arm | IoU mean | median |
|---|---|---|
| no furniture, LiDAR only | 0.941 | 0.963 |
| furniture, LiDAR only | 0.882 | 0.895 |
| furniture, LiDAR + ceiling line | 0.878 | 0.932 |
Furniture costs 0.068 of median IoU; the ceiling line gives back 0.037 of it, and helps in 16 rooms of 20,
is neutral in 3, and destroys 1 (room 1: 0.914 -> polygon never closed, 12 walls -> 9). The mean is dragged
below the LiDAR-only arm by that single collapse while the median rises — the distribution moved up and
grew a tail. That tail is the next thing to understand, and the likely mechanism is a conflict the bench
creates on purpose: the LiDAR sees a furniture FACE as a wall-like segment while the camera sees the true
wall behind it, so two incompatible lines compete for the same stretch of boundary. The model has no
furniture in it, so nothing explains the near line away.

## The tail was the two sensors arguing, and the argument is the evidence (2026-09-03)
The first fusion helped 16 rooms of 20 and destroyed 1, where the polygon stopped closing. The cause was
the conflict the bench creates on purpose: the LiDAR stops on a cupboard and hands the segmenter a
wall-like face, while the camera sees the true wall behind it, so two incompatible lines compete for one
stretch of boundary and nothing in the model explains the near one away.
The fix is not to arbitrate between them but to READ THE DISAGREEMENT. Both sensors are sampled on one
bearing grid. When the camera's range exceeds the LiDAR's on the SAME bearing by more than the two can
differ by chance — 3σ of their combined noise, where the camera's σ is the (dh²+r²)/dh·σ_α it already
carries — that is the signature of an occluder, and it says the near return is not on a wall. Those LiDAR
points are dropped from the wall cloud. No threshold beyond the 3σ, and the test costs nothing: it uses
the variance each sensor already reports.
| 20 rooms, identical, furniture in all | IoU mean | median | min | Hausdorff median |
|---|---|---|---|---|
| LiDAR only | 0.882 | 0.895 | 0.757 | 1.230 m |
| + ceiling line, sensors merged blindly | 0.878 | 0.932 | **0.000** | 1.275 m |
| + ceiling line, disagreement read as occlusion | **0.919** | **0.934** | 0.728 | 0.916 m |
Better in 15 rooms, neutral in 2, worse in 3, worst single loss 0.033, and the room that collapsed to 0.000
now scores 0.930. Furniture costs 0.068 of median IoU against a clean room; the fused pair gives back
0.039 of it. The tail is gone, the mean rose by 0.037, and the wall-SLAM bench is untouched at
0.957 / 0.902 / 0.980.

## The L-shaped rooms lose to COVERAGE, not to the model (2026-09-03)
L rooms average 0.861 against 0.942 for rectangles, and the cause was traced rather than guessed. A new
bench column separates the two ways a room can be lost, which no IoU can tell apart: `cover` is the
fraction of the truth's own interior the robot's grid ever learned anything about, and `excl` is the
fraction it learned about and still left outside the polygon.
| room | shape | IoU | cover | excl |
|---|---|---|---|---|
| 15 | L | 0.812 | 0.86 | 0.05 |
| 4 | L | 0.920 | 0.98 | 0.01 |
| 6 | L | 0.937 | 1.00 | 0.01 |
| 0 | rect | 0.984 | 1.00 | 0.01 |
The worst L room never saw 14% of itself; only 5% was seen and excluded. So the polygon is not wrong about
what it was shown — it was never shown a leg. Geometrically the missed area is one blob of 7.74 m² against
0.44 m² invented in total, i.e. the estimate is essentially a SUBSET of the truth.
More time makes it WORSE, not better: 900 / 1800 / 3000 frames give 0.812 / 0.768 / 0.764. The extra frames
go into churn, not into the leg, which rules out "the explorer is merely slow".
**The obvious cure was tried and reverted.** Sending every second coverage turn to the FARTHEST frontier
instead of the nearest (the nearest rule finishes the local region and never commits to the far leg, and
the mass score divides by distance so the far leg is discounted twice) moved the real flat's worst seed
0.902 -> 0.966 and its best 0.980 -> 0.966, with a second check failing; across 50 rooms the median fell
0.947 -> 0.941, the Hausdorff median rose 0.89 -> 1.14 m and spurs fell 13% -> 10%. A far frontier costs a
long drive whose frames come out of refinement. The coverage problem is real; a nearest-versus-farthest
heuristic is not its answer. What is, is an objective that prices the drive against what it would reveal —
mutual-information exploration, rank 6 of the literature review, still unimplemented.

## Information-gain exploration (literature rank 6), 2026-09-04
The L-room diagnosis said the loss is coverage, and that a nearest-versus-farthest heuristic is not the
answer; an objective that prices a drive against what it would reveal is. That objective is now in.
Every uncertain thing in the map contributes its OWN entropy, in nats, and the explorer maximises gain per
metre. A grid cell contributes H(p) from its log-odds. An existence bin IS a log-odds, so its entropy needs
no conversion. A candidate line contributes the entropy of its own accumulated evidence. A corner
contributes its differential-entropy excess, ln(sigma / publish bar), and nothing once it is inside the
bar. A cell the map is already sure about contributes nothing, so the drive stops on its own.
What that replaced: five hand-set target weights (1.0 unsolid bin, 2.0 candidate, 3.0 uncertain corner,
1.5 frontier, 2.5 weak matter), a 0.10-per-metre distance discount chosen by taste, and the every-fourth-
replan coverage turn that existed only because the old objective undervalued far regions. The remaining
constant is the distance the robot covers between replans, which makes the ratio nats per look.
| 50 rooms, identical | mean | median | max | Hausdorff median | alcove | corner col | spur |
|---|---|---|---|---|---|---|---|
| heuristic targets | 0.912 | 0.947 | 0.988 | 0.894 m | 70% | 73% | 13% |
| information gain | **0.920** | **0.954** | **0.993** | **0.774 m** | **74%** | **76%** | 9% |
The L rooms are where it lands: the worst went 0.812 -> 0.944 with coverage 0.86 -> 0.97, and another
0.920 -> 0.967. Spurs are the one regression, 13% -> 9%: the old list gave weak matter a 2.5 weight to
force dwelling on thin walls, and entropy disagrees — a wholly unknown cell (0.69 nats) outranks a
half-known one (0.58), which is correct as information and worse for spurs.
**The real flat is slightly worse**: 0.957 / 0.902 / 0.980 -> 0.908 / 0.975 / 0.945, still 39 of 40. Its
floor rises and its best falls. The 50-room population is fifty layouts against three seeds of one, so it
decides; and the coverage turn now removed was itself introduced to rescue a seed-7 failure on this very
layout, so the old objective was partly fitted to it. Default is now information gain
(`WS_NO_INFOGAIN=1` restores the heuristic).
