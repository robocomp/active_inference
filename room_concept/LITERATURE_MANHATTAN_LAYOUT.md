# Manhattan room-layout estimation: where this prototype sits in the literature

Written 2026-09-02 against `src/wall_map.{h,cpp}`, `WALL_SLAM_REVIEW_2026-09-02.md` and the
three-seed synthetic bench (`tools/wall_slam_selftest`). Every citation below was checked
against a primary source (publisher page, arXiv, or CVF open access).

## 1. What the prototype is, in the literature's vocabulary

It is an **online, single-pass, feature-based 2-D SLAM system whose map is a single closed
rectilinear polygon**. The landmarks are infinite lines in Hesse form (φ, d) — the 2-D analogue of
Kaess's infinite planes (ICRA 2015) and of the wall layer of S-Graphs+ (Bavle et al., RA-L 2023) —
estimated jointly with the poses in a sliding-window Gauss–Newton factor graph with point-to-line
factors, a soft Manhattan factor (φ − θ₀ − k·π/2, in the spirit of StructSLAM and ManhattanSLAM),
and priors carried from dropped slots. What is unusual is that the **map state is a cyclic order**,
not a landmark set: the model *is* a simple CCW polygon, so closure and non-self-intersection are
properties of the state space rather than post-hoc tests. Structure is therefore not born but
*jumped*: the polygon starts as the minimal Manhattan shape (a rectangle from the first scan's
oriented bounding box) and grows in even-order splices (+2 corner cut, +4 notch or spur wrap), each
paying a constant toll (15 nats per +2 edges) out of evidence newly explained where old and new
polygons disagree. Alongside runs a log-odds occupancy grid (Elfes 1989) supplying free space,
frontiers (Yamauchi 1997) for an epistemic explorer, and a periodic **global re-derivation** —
trace the free-space contour, snap it to the evidence lines, rectilinearise, adopt iff it explains
the grid better — a greedy online cousin of the offline floorplan-from-density-map line (Cabral &
Furukawa 2014 onwards). Wall extents carry per-0.25 m existence bins updated by beam
pass-throughs: negative evidence at sub-landmark resolution. The published polygon is projected to
exact Manhattan on a copy, a one-iteration hard-EM version of Straub et al.'s Manhattan-frame
estimation (CVPR 2014).
## 2. Comparison table

| Approach | Input / setting | Representation | How structure/topology change is decided | Online? | Reported accuracy | How the prototype differs |
|---|---|---|---|---|---|---|
| **This prototype** | mobile robot, 2-D LiDAR (720 rays, σ=2 cm) + noisy odometry, 1100 frames | one closed Manhattan polygon of (φ,d) wall landmarks + occupancy grid | greedy priced "order jumps" (+2/+4 edges) at a constant 15-nat toll, paid from grid/existence nats; periodic contour re-derivation with a surrender veto | **yes**, in the SLAM loop | synthetic 32-vertex flat: IoU 0.953/0.967/0.942, Hausdorff 0.44–2.2 m, pose RMSE 1–5 cm | — |
| Cabral & Furukawa, CVPR 2014 | registered images + SfM points, single room/floor | piecewise-planar Manhattan floorplan | globally optimal **shortest path** on a crafted graph; per-vertex penalty controls output complexity | no (batch) | qualitative + mesh compactness | prototype's re-derivation is a heuristic contour trace, not a global optimum; toll is not a code length |
| Chen, Liu, Wu & Furukawa (Floor-SP), ICCV 2019 | aligned RGB-D scans, 527 apartments | room-wise polygon graph | room-wise coordinate descent, each step a **shortest path**; objective = CNN data term + inter-room consistency + **model complexity term** | no | evaluated on 527 units, incl. non-Manhattan | prototype has the complexity term but not the global optimiser, and no learned data term |
| Liu, Wu & Furukawa (FloorNet), ECCV 2018 | RGB-D video (Tango), 155 homes | corner/edge heatmaps → floorplan | learned; IP post-process | no | own benchmark (155 homes) | prototype is model-based, no training data |
| Stekovic et al. (MonteFloor), ICCV 2021 | 3-D point cloud → top-view density | set of room polygons | **MCTS** over room proposals maximising a fitness objective | no | Structured3D + own | same "propose/accept structure" shape, but MCTS searches; prototype commits greedily, one jump per frame |
| Chen, Qian & Furukawa (HEAT), CVPR 2022; Yue et al. (RoomFormer), CVPR 2023 | point-density image | planar graph / variable-length polygon sequences | end-to-end learned (edge attention; two-level queries) | no | SOTA on Structured3D, SceneCAD | prototype has no learned prior at all; it also runs while the robot drives |
| Okorn et al., 3DPVT 2010 | static laser scans | 2-D line segments from Hough on a density histogram | thresholds on histogram peaks | no | qualitative | prototype's segmenter is comparable; the polygon/topology layer is what Okorn lacks |
| Mura et al., C&G 2014 | cluttered 3-D scans | per-room polygons from a wall-candidate space partition | diffusion on the cell complex | no | qualitative | prototype builds one room; Mura's cell decomposition is a natural global alternative to the contour trace |
| Ochmann et al., C&G 2016 / ISPRS J. 2019 | indoor point clouds | volumetric parametric building (walls with thickness) | **integer linear programming** over wall candidates (2019: exact) | no | qualitative/CAD-grade | same "select a subset of candidate lines" problem, solved exactly and offline vs greedily and online |
| Bormann et al., ICRA 2016 | 2-D grid maps of real buildings | room partition of an occupancy map | 4 surveyed segmentation algorithms + benchmark | offline over a finished map | comparative benchmark, public maps | prototype produces one polygon, not a room partition; Bormann's maps/metrics are directly reusable |
| Bavle et al., S-Graphs (RA-L 2022) / S-Graphs+ (RA-L 2023) | 3-D LiDAR, mobile robot | 4-layer factor graph: keyframes / walls / rooms / floors | rooms detected from free-space clusters, then a room factor over 4 wall planes; all kept in one optimisable graph | **yes** | pose + map accuracy on real and simulated sequences | prototype has one room and no room layer; its walls form a *closed cycle*, which S-Graphs+ does not enforce |
| Hughes, Chang & Carlone (Hydra), RSS 2022 | RGB-D, real-time | 3-D scene graph (objects/places/rooms/buildings) | room detection over the places graph; loop closure re-optimises layers | **yes** | batch-comparable accuracy online | Hydra's rooms are topological regions, not metric rectilinear polygons |
| Kaess, ICRA 2015; Yunus et al. (ManhattanSLAM), ICRA 2021 | RGB-D | infinite planes / mixture of Manhattan frames | landmarks added by data association; Manhattan constraint *in* the estimator | **yes** | trajectory error | prototype adds a topology (cycle) on top of the same landmark type |
| Wang et al. (Floorplan-SLAM), arXiv 2503.00397, 2025 | stereo camera | plane landmarks → incremental floorplan | continuously re-solved optimisation over plane landmarks and poses | **yes**, 25–45 FPS | 1000 m² scene in 9.44 min vs >10 h | closest in ambition; no explicit priced topology operators, no closed-polygon state space |
| Gao et al. (ERPoT), T-RO 2025 | 2-D scan vs prior polygon map | multi-polygon compact map | map given a priori; point-to-edge/vertex cost for tracking | tracking online, map prior | long-term pose tracking | shows a downstream use for the prototype's output; the prototype *builds* the polygon map |

## 3. Where the prototype is ahead or unusual

- **It changes topology inside the SLAM loop, while driving.** The accurate rectilinear-floorplan
  literature (Cabral & Furukawa; FloorNet; Floor-SP; MonteFloor; HEAT; RoomFormer; Ochmann; Mura) is
  offline over a finished, registered scan. The online structural-SLAM literature (S-Graphs+, Hydra,
  Floorplan-SLAM) is online but never commits to a simple closed rectilinear polygon whose *order*
  is a first-class state variable.
- **Topology as a state space, not a filter output** — the compared systems keep a landmark/plane
  set and derive structure afterwards.
- **Priced, dimension-changing operators with an instrumented refusal count.** Splice / spur-wrap /
  down-jump are birth–death–split–merge moves. Not new in kind — Green's reversible-jump MCMC
  (1995), its architectural use by Dick, Torr & Cipolla (2004) and Floor-SP's model-complexity term
  are the same idea — but doing it greedily, online, one jump per frame with every rejection
  counted, is unusual.
- **Per-extent-bin existence (the step-back operator).** Negative evidence with occlusion holding is
  standard for cells (Elfes 1989; Thrun 2003); applying it per 0.25 m bin *along a line landmark*,
  so a wall can lose a stretch without dying, is sharper than the per-landmark existence odds of
  feature SLAM (cf. segment endpoint management in Tardós et al. 2002; Nguyen et al. 2007). Honest
  caveat: evidence per sub-segment is old; the symmetric birth/death currency is the new part.
- **Output-stage Manhattan projection over a soft in-loop model**, costing ~0.002 IoU on the bench.
  The offline literature gets this for free by making the *search space* axis-aligned (Cabral &
  Furukawa's grid graph) — strictly stronger than projecting at the end.
- **The surrender veto** (an adoption may not erase observed existence support) has no analogue.
  That is partly a warning sign: it exists only because the global adoption criterion and the local
  existence evidence are in different units (§4).
## 4. Where it is behind

- **No quantitative comparison on any public dataset.** Every number comes from one synthetic
  32-vertex flat, three seeds, ray-cast from an SVG. The compared systems report on Structured3D,
  SceneCAD, 155–527 real scans, or real robot sequences. Nothing here is comparable to anything.
- **Three seeds is not a distribution**, and the bench's PASS is checked on the best seed
  (review #16); the IoU floor 0.942 is under the stated 0.95 bar.
- **Incommensurable evidence currencies against a constant toll** (review #10). `order_jump_nats =
  15` implies p(n+2)/p(n) ≈ 3·10⁻⁷ read as a geometric order prior, but is admitted to be the noise
  floor of a comparison whose inputs are a per-point log-likelihood ratio, an ad-hoc grid log-odds
  (−0.4/pass, +1.0/hit, clamped ±4) and per-frame existence bins. Floor-SP's complexity term and
  Rissanen's MDL supply the units this toll lacks.
- **The optimisation is greedy and order-dependent**: one jump per frame, hysteresis bands
  (`order_keep_fraction = 0.7`), a death sweep whose outcome depends on enumeration order
  (review #18: 0.942 → 0.821). Cabral & Furukawa and Floor-SP are globally optimal by construction;
  MonteFloor searches; RJMCMC would at least be a correct sampler.
- **No learned priors.** Every recent floorplan system uses a CNN/Transformer data term. Defensible
  for an active-inference agent, but a measured accuracy gap in the offline literature.
- **Manhattan-only and single-room.** Real apartments hold several Manhattan frames (Straub et al.
  2014) and chamfers, which are banked as candidates and never used. There is no room layer, room
  graph or door topology — all of which S-Graphs+, Hydra and Floor-SP have.
- **No loop closure or multi-session merging** (Floorplan-SLAM's main claim), and the grid is a
  fixed ±15 m box around the first pose (review #15).
## 5. What to borrow, concretely

1. **Replace `re_derive`'s contour-trace-and-snap with a shortest path on an axis-aligned graph over
   the evidence lines** — Cabral & Furukawa (CVPR 2014), refined by Floor-SP (ICCV 2019). Nodes are
   intersections of candidate Manhattan lines; edge weight is the negative grid log-likelihood along
   the segment; the **per-vertex penalty replaces `order_jump_nats`/`parity_jump_nats` outright**.
   The global escape hatch becomes optimal rather than heuristic, and the surrender veto dissolves —
   support that would be surrendered is already in the edge weights.
2. **Give the order prior units: a two-part MDL code length** (Rissanen 1978) — bits to name a
   vertex ≈ log₂(candidate lines × cells along one), plus the new wall's parameter cost. A derived
   number to put where the 15-nat toll is; if it lands near 15 the toll is vindicated, and if not,
   the discrepancy is the finding.
3. **Calibrate the grid with a real sensor model** — Elfes (1989) for the inverse form, Thrun (2003)
   for the forward form that removes the per-cell independence artefacts. This one change makes
   review #10 and #9 disappear together: grid nats become log-likelihoods in the same currency as
   the point factors, so the permanent `hits ≥ 3` latch and the ±4 clamp
   become derived quantities instead of constants.
4. **Adopt the S-Graphs+ treatment of wall landmarks** (RA-L 2023): keep the keyframe factors, or
   marginalise them at a fixed linearisation point into an information-form accumulator (Λ, Λμ),
   instead of re-centring the prior on the current estimate — review #5's fix, already standard
   there. Its rooms layer (a room factor over four wall planes) is the natural next level once the
   map must hold more than one room.
5. **Make the Manhattan step a proper E-step**: Straub et al. (2014) is a vMF mixture with soft
   assignment; `enforce_manhattan` is one-iteration hard-EM weighted by point counts. Weight by
   angular information and keep soft responsibilities: the ±45° hard boundary in `classify()` and
   the two-θ₀ inconsistency (review #6, #13) both go, and the model gains a path to the second
   Manhattan frame a real apartment usually has.
6. *(Cheap, optional)* **Write the jumps as reversible-jump proposals** (Green 1995): deriving each
   operator's acceptance ratio forces up- and down-jumps into one posterior and replaces the 0.7
   hysteresis band with detailed balance.
## 6. Public datasets / benchmarks that would make the numbers comparable

- **Structured3D** (Zheng et al., ECCV 2020) — 3.5k scenes with floorplan ground truth, the
  benchmark MonteFloor, HEAT and RoomFormer report on. A 2-D LiDAR trajectory can be ray-cast
  through it exactly as the bench ray-casts one SVG, giving directly comparable room-IoU and
  corner precision-recall over thousands of layouts.
- **HouseExpo + PseudoSLAM** (Li et al., 2019/2020) — 35,126 real-derived 2-D floor plans with a
  ray-casting simulator built for mobile robots; the lowest-friction way to turn three seeds into a
  distribution, and 2-D LiDAR native.
- **FloorNet's benchmark** (155 real Tango RGB-D scans) and **Floor-SP's** 527 apartment scans,
  which include non-Manhattan units and would expose the strict-Manhattan restriction honestly.
- **Bormann et al. (ICRA 2016)** room-segmentation maps, metrics and four reference
  implementations — the right benchmark the day the polygon becomes a room partition.
- **Real 2-D LiDAR logs**: the Cartographer public data (Hess et al., ICRA 2016, incl. the Deutsches
  Museum 2-D backpack sequences) and Radish (Howard & Roy, 2003). No polygon ground truth, but the
  standard stress test for a real sensor, real odometry and moving people — exactly where "matter
  never forgets" (review #9) will bite.
- **S-Graphs+ sequences** (`snt-arg/lidar_situational_graphs`) — real and simulated structured-indoor
  runs from the closest online system, so pose *and* wall accuracy can be graded against a published
  baseline rather than ground truth alone.

## 7. SOTA techniques worth adopting

Wider net than §5: anything current that could plug into one of the prototype's six mechanisms —
**the toll** (`order_jump_nats` / `parity_jump_nats` / `order_keep_fraction`), **the grid**
(`FreeGrid` increments, `hits`, `free_ms`), **the existence bins** (`exist_bins`, the step-back
operator), **the re-derivation** (`re_derive`), **the projection**
(`enforce_manhattan` / `manhattan_polygon`) and **the explorer** (frontiers, thin-wall dwell
targets). Ranked by expected payoff divided by integration cost.

### Tier 1 — high payoff, days of work

**Calibrated beam / inverse sensor model for the occupancy grid** (Elfes, *Computer* 1989; Thrun,
*Auton. Robots* 2003). Elfes gives the Bayesian per-cell update from a probabilistic range model;
Thrun's forward-model formulation solves the mapping problem in the joint space and removes the
per-cell-independence artefacts (thin structure carved away, conflicting evidence resolved by
whichever beam came last). Classical, no training, runs online at 2-D grid scale. *Replaces*:
**the grid**'s ad-hoc increments (−0.4 per pass, +1.0 per hit, −0.02, clamp ±4, the permanent
`hits ≥ 3` latch; a `grid_latch_discount` was tried on 2026-09-02 and reverted). The payoff is not accuracy but *commensurability* —
once cells hold log-likelihoods, `jump_delta_nats` and the point factors are in the same units and
review #10 evaporates.

**MDL code length in place of the toll** (Rissanen, *Automatica* 1978; the model-complexity term of
Floor-SP, Chen et al., ICCV 2019). A two-part code prices a polygon of order n as its parameters
plus the bits to name each vertex among the candidate intersections. *Replaces*: **the toll** — one
derived number instead of `order_jump_nats = 15`, `parity_jump_nats = 2` and the
`order_keep_fraction = 0.7` hysteresis band. Cheap: arithmetic over quantities the map already
tracks (candidate count, grid resolution, extent).

**Shortest path / global selection as the re-derivation** (Cabral & Furukawa, CVPR 2014; Floor-SP,
ICCV 2019; and, for the exact-optimum variant, the integer linear programme of Ochmann et al.,
*ISPRS J.* 2019). All three solve "choose a subset of candidate wall lines forming a valid layout"
globally rather than greedily — the first two by shortest path over a crafted graph, the third by
ILP. Offline as published, but a 20–40-line axis-aligned graph is small enough to solve at the
current 40-frame re-derivation cadence. *Replaces*: **the re-derivation**'s contour trace → snap →
rectilinearise → IoU-adopt heuristic, and with it the surrender veto (`adopt_surrender_nats`), since
support that would be surrendered is already priced in the edge weights.

**Soft Manhattan-frame inference, and Atlanta world as the escape hatch** (Straub et al., CVPR 2014;
Joo, Oh, Kweon & Bazin, CVPR 2018). Straub's mixture of Manhattan frames is a von Mises–Fisher
mixture with soft responsibilities and split/merge sampling; Joo et al. give a globally optimal
branch-and-bound estimate of an *Atlanta* frame — one vertical plus a set of horizontal directions
that need **not** be mutually orthogonal, i.e. exactly the apartment with a 45° chamfer the
prototype currently refuses. Both run at interactive rates on angles/normals. *Replaces*: **the
projection** — `enforce_manhattan`'s one-iteration hard-EM weighted by `points_seen`, and
`classify()`'s hard ±45° boundary — and eventually `manhattan_strict` itself, with the banked
`corner_residue` becoming a second frame rather than dead weight.

**Marginalise wall landmarks properly, and add a room factor** (Bavle et al., S-Graphs RA-L 2022;
S-Graphs+ RA-L 2023). S-Graphs+ keeps wall planes as variables with all keyframe factors (or
marginalised at a fixed linearisation point) and adds a rooms layer whose factor constrains a room
centre from four wall planes; it runs online on a LiDAR robot and is open source
(`snt-arg/lidar_situational_graphs`). *Replaces*: the re-centred `WallPriorFactor` (review #5) with
an information-form accumulator (Λ, Λμ). *Adds*: the layer above the polygon that the prototype will
need the moment a second room appears.

### Tier 2 — good payoff, a week or more

**Evidential / Dempster–Shafer occupancy grids with explicit conflict** (Moras & Cherfaoui, IEEE
Intelligent Vehicles Symposium 2011, *Moving objects detection by conflict analysis in evidential
grids*). Belief functions carry "free", "occupied" and "unknown" as distinct masses, and the
*conflict* mass generated when a new scan disagrees with the map is itself the moving-object
detector. Online, on vehicles. *Replaces / augments*: **the grid** — the principled cure for review
#9 ("matter never forgets"): a person crossing three scans raises conflict instead of latching
`hits ≥ 3` for ever. Also directly useful for doors, which are precisely cells that legitimately
change state.

**Dynamic occupancy grids via random finite sets** (Nuss et al., *IJRR* 2018). A particle-based
Dempster–Shafer grid carrying per-cell velocity, real-time on automotive hardware. Heavier than the
prototype needs, but the reference for how a grid should forget. *Augments*: **the grid**, and
indirectly **the existence bins** — the same "is this matter static?" question the step-back
operator asks per wall bin.

**Information-gain exploration instead of frontier heuristics** (Stachniss, Grisetti & Burgard,
RSS 2005; Julian, Karaman & Rus, *IJRR* 2014). Stachniss et al. trade action cost against expected
information gain over map *and* pose under an RBPF; Julian et al. prove that a mutual-information
reward computed from a beam model attracts the robot to unexplored space, and give a tractable form.
Both online. *Replaces*: **the explorer**'s hand-ranked target list (frontiers, untested existence
bins, homeless candidates, thin-wall dwell) with one MI objective over the layout posterior —
existence bins and candidate lines included — which also yields an honest "am I done?" test. This is
the item that best matches the agent's active-inference framing: expected information gain is the
epistemic term of expected free energy.

**Learned wall completion beyond the frontier** (Ericson & Jensfelt, *IEEE RA-L* 2024,
arXiv:2406.09160). Predicts unseen walls as a set of 2-D line segments from a partial occupancy grid
integrated along a 360° LiDAR trajectory, trained on real campus floor plans; autoregressive
attention model at robot scale. *Adds*: a prior over structure not yet observed — hypothesised lines
feed the candidate bank for **the re-derivation** and rank targets for **the explorer**. The cheapest
way to get a learned prior into this pipeline without touching the estimator, and the only learned
method listed here that is 2-D LiDAR native.

**Learned polygon decoders as an offline referee** (RoomFormer, Yue et al., CVPR 2023; SLIBO-Net,
Su et al., NeurIPS 2023; PolyRoom, Liu et al., ECCV 2024; FRI-Net, Xu et al., ECCV 2024). These take
a point-density image and emit room polygons directly: RoomFormer with two-level polygon/corner
queries, SLIBO-Net with a slicing-box representation plus a scale-independent metric, PolyRoom with
room-aware queries that explicitly suppress self-intersecting and topologically implausible output,
FRI-Net with a room-wise implicit field regularised toward regular layouts. None runs in a SLAM
loop. *Adds*: not an in-loop component but a **batch referee** — run one on the accumulated grid at
the end of a session and compare against the online polygon; the gap is the honest measure of what
the online commitment costs. SLIBO-Net's scale-independent metric is worth copying regardless of
whether the network is.

**Anytime search over structure proposals** (MonteFloor, Stekovic et al., ICCV 2021). MCTS over
room-proposal subsets maximising a fitness objective — the same accept/reject decision the prototype
makes, but searched rather than committed greedily one jump per frame. Offline as published, yet
MCTS is anytime by construction, so a bounded rollout budget per re-derivation cadence is plausible.
*Replaces*: the greedy commit in `try_splice` / `try_down_jumps`, and with it the order-dependence of
review #18. Costlier than Tier 1 because it needs the single energy function to exist first.

### Tier 3 — worth knowing, larger commitment

**Walls with thickness instead of the spur grammar** (Ochmann et al., *C&G* 2016 and *ISPRS J.*
2019). Their building model is volumetric: a wall is a slab with two faces and a thickness,
reconstructed by ILP over candidate wall planes. *Replaces*: the three-edge spur wrap
([near face, tip cap, far face]) plus `stub_thickness`, `stub_free_behind_min`, `mirror_backed` and
the cap/mirror rent exemptions — one primitive with a thickness parameter instead of a grammar and
four exemptions. A representation change, not a patch; it would also lift the constraint that the
8 cm cell size is dictated by the thinnest wall the model must keep.

**Doors and openings as first-class map elements** (Nikoohemat, Diakité, Zlatanova & Vosselman,
*Automation in Construction* 113, 2020 — door detection from the scanner trajectory; and the
architectural-graph matching of Shaheer et al., arXiv:2303.02076 / 2305.09295, which aligns an online
S-Graph to a building plan). The trajectory cue is directly available here: the robot drove through
the opening. *Adds*: a door channel where the polygon currently sees only an unexplained free-space
gap, plus a hand-off to the fleet's existing `door_concept` agent. Doors are also the main reason the
strict single-polygon model will eventually break.

**Prior-plan informed localisation and deviation detection** (Shaheer et al., iS-Graphs,
arXiv:2303.02076). Where a CAD plan exists, the map problem becomes alignment plus deviation
detection rather than construction. *Adds*: an initialisation for **the re-derivation** and a strong
prior on **the toll** (a jump that matches the plan is nearly free). Relevant only in buildings whose
plans are available.

**Polygon maps as the localisation substrate** (ERPoT, Gao et al., *IEEE T-RO* 2025). Point-to-edge
and point-to-vertex costs against a compact multi-polygon prior map give long-term pose tracking with
a tiny map footprint. *Adds*: the downstream consumer that justifies the polygon representation, and
a second, independent way to grade the published layout — a bad polygon shows up as bad tracking.

**LiDAR BEV structural detection as a segmenter replacement** (Li et al., arXiv:2603.19830, 2026):
wall/structure detection from 3-D LiDAR bird's-eye-view images with an oriented-box detector at
10 Hz on a single-board computer without GPU, benchmarked against Hough / RANSAC / LSD. *Replaces*:
`wall_segmenter`'s RANSAC line extraction, whose 2–4σ tails are the root cause of the corner-artefact
chords in review #11. Preprint — a lead, not a dependency.

### Ranking summary

| Rank | Technique | Mechanism it touches | Payoff | Integration cost |
|---|---|---|---|---|
| 1 | Calibrated inverse/forward sensor model (Elfes 1989; Thrun 2003) | the grid | very high — makes one currency possible | low |
| 2 | MDL code length (Rissanen 1978; Floor-SP 2019) | the toll | high — removes three decision constants | low |
| 3 | Shortest path / ILP global layout (Cabral & Furukawa 2014; Floor-SP 2019; Ochmann 2019) | the re-derivation | high — optimal, kills the surrender veto | medium |
| 4 | vMF Manhattan E-step + Atlanta frame (Straub 2014; Joo 2018) | the projection | high — fixes θ₀, opens chamfers | low–medium |
| 5 | Information-form wall priors + rooms layer (S-Graphs+ 2023) | solver priors, map layer | high — review #5, plus multi-room | medium |
| 6 | Mutual-information exploration (Stachniss 2005; Julian 2014) | the explorer | high — one objective, honest completion test | medium |
| 7 | Evidential grid with conflict (Moras & Cherfaoui 2011) | the grid, existence bins | medium–high — review #9, dynamic scenes | medium |
| 8 | Learned wall completion (Ericson & Jensfelt 2024) | re-derivation, explorer | medium–high — a real learned prior, 2-D native | medium |
| 9 | Learned polygon decoders as batch referee (RoomFormer / SLIBO-Net / PolyRoom / FRI-Net) | evaluation | medium — measures the cost of being online | low (offline) |
| 10 | Anytime MCTS over proposals (MonteFloor 2021) | splice / down-jump | medium — kills order-dependence | high (needs one energy first) |
| 11 | Volumetric walls with thickness (Ochmann 2016/2019) | spur grammar, grid resolution | medium — a parameter instead of a grammar | high |
| 12 | Doors/openings as map elements (Nikoohemat 2020; Shaheer 2023) | polygon topology | medium — needed for multi-room | high |
| 13 | Dynamic occupancy grid, RFS (Nuss et al. 2018) | the grid | medium — reference, likely overkill | high |
| 14 | Polygon-map pose tracking (ERPoT 2025) | downstream / evaluation | medium — independent grading signal | medium |
| 15 | BEV structural detection (Li et al. 2026, preprint) | wall_segmenter | uncertain — addresses review #11's root cause | medium |

## 8. References

1. Cabral, R., Furukawa, Y. *Piecewise Planar and Compact Floorplan Reconstruction from Images.* CVPR 2014, pp. 628–635. DOI 10.1109/CVPR.2014.546.
2. Liu, C., Wu, J., Furukawa, Y. *FloorNet: A Unified Framework for Floorplan Reconstruction from 3D Scans.* ECCV 2018, LNCS 11210. DOI 10.1007/978-3-030-01231-1_13.
3. Chen, J., Liu, C., Wu, J., Furukawa, Y. *Floor-SP: Inverse CAD for Floorplans by Sequential Room-wise Shortest Path.* ICCV 2019, pp. 2661–2670. arXiv:1908.06702.
4. Stekovic, S., Rad, M., Fraundorfer, F., Lepetit, V. *MonteFloor: Extending MCTS for Reconstructing Accurate Large-Scale Floor Plans.* ICCV 2021. arXiv:2103.11161.
5. Chen, J., Qian, Y., Furukawa, Y. *HEAT: Holistic Edge Attention Transformer for Structured Reconstruction.* CVPR 2022. arXiv:2111.15143.
6. Yue, Y., Kontogianni, T., Schindler, K., Engelmann, F. *Connecting the Dots: Floorplan Reconstruction Using Two-Level Queries (RoomFormer).* CVPR 2023. arXiv:2211.15658.
7. Okorn, B., Xiong, X., Akinci, B., Huber, D. *Toward Automated Modeling of Floor Plans.* 3DPVT 2010, Paris.
8. Mura, C., Mattausch, O., Jaspe Villanueva, A., Gobbetti, E., Pajarola, R. *Automatic room detection and reconstruction in cluttered indoor environments with complex room layouts.* Computers & Graphics 44 (2014) 20–32. DOI 10.1016/j.cag.2014.07.005.
9. Ochmann, S., Vock, R., Wessel, R., Klein, R. *Automatic reconstruction of parametric building models from indoor point clouds.* Computers & Graphics 54 (2016) 94–103.
10. Ochmann, S., Vock, R., Klein, R. *Automatic reconstruction of fully volumetric 3D building models from oriented point clouds.* ISPRS J. Photogrammetry and Remote Sensing 151 (2019) 251–262. arXiv:1907.00631.
11. Bormann, R., Jordan, F., Hampp, J., Hägele, M. *Room segmentation: Survey, implementation, and analysis.* ICRA 2016, pp. 1019–1026. DOI 10.1109/ICRA.2016.7487234.
12. Bavle, H., Sanchez-Lopez, J. L., Shaheer, M., Civera, J., Voos, H. *Situational Graphs for Robot Navigation in Structured Indoor Environments.* IEEE RA-L, 2022. arXiv:2202.12197.
13. Bavle, H., Sanchez-Lopez, J. L., Shaheer, M., Civera, J., Voos, H. *S-Graphs+: Real-time Localization and Mapping leveraging Hierarchical Representations.* IEEE RA-L 8(8):4927–4934, 2023. arXiv:2212.11770.
14. Hughes, N., Chang, Y., Carlone, L. *Hydra: A Real-time Spatial Perception System for 3D Scene Graph Construction and Optimization.* RSS 2022. arXiv:2201.13360.
15. Kaess, M. *Simultaneous Localization and Mapping with Infinite Planes.* ICRA 2015, pp. 4605–4611.
16. Yunus, R., Li, Y., Tombari, F. *ManhattanSLAM: Robust Planar Tracking and Mapping Leveraging Mixture of Manhattan Frames.* ICRA 2021, pp. 6687–6693. arXiv:2103.15068.
17. Zhou, H., Zou, D., Pei, L., Ying, R., Liu, P., Yu, W. *StructSLAM: Visual SLAM with Building Structure Lines.* IEEE Trans. Vehicular Technology 64(4):1364–1375, 2015.
18. Straub, J., Rosman, G., Freifeld, O., Leonard, J. J., Fisher III, J. W. *A Mixture of Manhattan Frames: Beyond the Manhattan World.* CVPR 2014, pp. 3770–3777.
19. Wang, H., Lv, Z., Wei, H., Zhu, H., Wu, Y. *Floorplan-SLAM: A Real-Time, High-Accuracy, and Long-Term Multi-Session Point-Plane SLAM for Efficient Floorplan Reconstruction.* arXiv:2503.00397, 2025 (preprint).
20. Gao, H. et al. *ERPoT: Effective and Reliable Pose Tracking for Mobile Robots Using Lightweight Polygon Maps.* IEEE Trans. Robotics, 2025. arXiv:2409.14723.
21. Elfes, A. *Using Occupancy Grids for Mobile Robot Perception and Navigation.* Computer 22(6):46–57, 1989. DOI 10.1109/2.30720.
22. Thrun, S. *Learning Occupancy Grid Maps with Forward Sensor Models.* Autonomous Robots 15:111–127, 2003. DOI 10.1023/A:1025584807625.
23. Rissanen, J. *Modeling by shortest data description.* Automatica 14(5):465–471, 1978.
24. Green, P. J. *Reversible jump Markov chain Monte Carlo computation and Bayesian model determination.* Biometrika 82(4):711–732, 1995.
25. Dick, A. R., Torr, P. H. S., Cipolla, R. *Modelling and Interpretation of Architecture from Several Images.* IJCV 60(2):111–134, 2004.
26. Tardós, J. D., Neira, J., Newman, P. M., Leonard, J. J. *Robust Mapping and Localization in Indoor Environments Using Sonar Data.* IJRR 21(4):311–330, 2002.
27. Nguyen, V., Gächter, S., Martinelli, A., Tomatis, N., Siegwart, R. *A comparison of line extraction algorithms using 2D range data for indoor mobile robotics.* Autonomous Robots 23:97–111, 2007.
28. Hess, W., Kohler, D., Rapp, H., Andor, D. *Real-Time Loop Closure in 2D LIDAR SLAM.* ICRA 2016, pp. 1271–1278. (Public 2-D/3-D backpack datasets, incl. Deutsches Museum.)
29. Yamauchi, B. *A frontier-based approach for autonomous exploration.* IEEE CIRA 1997, pp. 146–151.
30. Zheng, J., Zhang, J., Li, J., Tang, R., Gao, S., Zhou, Z. *Structured3D: A Large Photo-realistic Dataset for Structured 3D Modeling.* ECCV 2020. arXiv:1908.00222.
31. Li, T., Ho, D., Li, C., Zhu, D., Wang, C., Meng, M. Q.-H. *HouseExpo: A Large-scale 2D Indoor Layout Dataset for Learning-based Algorithms on Mobile Robots.* IROS/ICRA-era release, arXiv:1903.09845; IEEE 2020.
32. Howard, A., Roy, N. *The Robotics Data Set Repository (Radish).* 2003. http://radish.sourceforge.net/
33. Su, J.-W., Tung, K.-Y., Peng, C.-H., Wonka, P., Chu, H.-K. *SLIBO-Net: Floorplan Reconstruction via Slicing Box Representation with Local Geometry Regularization.* NeurIPS 2023.
34. Liu, Y., Zhu, L., Ma, X., Ye, H., Gao, X., Zheng, X., Shen, S. *PolyRoom: Room-aware Transformer for Floorplan Reconstruction.* ECCV 2024. arXiv:2407.10439.
35. Xu, H., Xu, J., Huang, Z., Xu, P., Huang, H., Hu, R. *FRI-Net: Floorplan Reconstruction via Room-wise Implicit Representation.* ECCV 2024. arXiv:2407.10687.
36. Chen, Z., Shi, Y., Nan, L., Xiong, Z., Zhu, X. X. *PolyGNN: Polyhedron-based graph neural network for 3D building reconstruction from point clouds.* ISPRS J. Photogrammetry and Remote Sensing, 2024. arXiv:2307.08636.
37. Joo, K., Oh, T.-H., Kweon, I. S., Bazin, J.-C. *Globally Optimal Inlier Set Maximization for Atlanta Frame Estimation.* CVPR 2018, pp. 5726–5734.
38. Moras, J., Cherfaoui, V. *Moving objects detection by conflict analysis in evidential grids.* IEEE Intelligent Vehicles Symposium (IV), 2011.
39. Nuss, D., Reuter, S., Thom, M., Yuan, T., Krehl, G., Maile, M., Gern, A., Dietmayer, K. *A random finite set approach for dynamic occupancy grid maps with real-time application.* IJRR 37(8), 2018. DOI 10.1177/0278364918775523.
40. Stachniss, C., Grisetti, G., Burgard, W. *Information Gain-based Exploration Using Rao-Blackwellized Particle Filters.* RSS 2005.
41. Julian, B. J., Karaman, S., Rus, D. *On mutual information-based control of range sensing robots for mapping applications.* IJRR 33(10):1375–1392, 2014.
42. Ericson, L., Jensfelt, P. *Beyond the Frontier: Predicting Unseen Walls from Occupancy Grids by Learning from Floor Plans.* IEEE RA-L, 2024. arXiv:2406.09160.
43. Nikoohemat, S., Diakité, A. A., Zlatanova, S., Vosselman, G. *Indoor 3D reconstruction from point clouds for optimal routing in complex buildings to support disaster management.* Automation in Construction 113 (2020) 103109.
44. Shaheer, M., Bavle, H., Sanchez-Lopez, J. L., Voos, H. *Graph-based Global Robot Localization Informing Situational Graphs with Architectural Graphs.* arXiv:2303.02076, 2023; and *Graph-based Global Robot SLAM using Architectural Plans*, arXiv:2305.09295, 2023.
45. Li, G., Espinosa-Angulo, P., Perez-Saura, D., Tapia-Fernandez, S. *Real-Time Structural Detection for Indoor Navigation from 3D LiDAR Using Bird's-Eye-View Images.* arXiv:2603.19830, 2026 (preprint).
