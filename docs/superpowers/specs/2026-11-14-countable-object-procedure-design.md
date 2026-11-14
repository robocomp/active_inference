# One procedure for countable objects: a class is a manifest of priors plus a mesh

Date: 2026-11-14. Status: APPROVED by the owner (11-14: four steps, no mesh-SDF fallback), revised after a Fable
review of the draft against the code. Nothing is built.

## 1. Intent (owner)

Adding a countable object (the next ones are the bathtub and the sink) must mean writing **one manifest of priors and
one mesh**, and nothing else. The manifest also carries the object's **geometric model**: its state variables and its
shape, composed from a shared library of shapes. Everything else is one common procedure, the same for every class.
Particularities live in priors.

**Why now.** nightstand_concept, bed_concept and toilet_concept (phase0: a5fbf362, 5071bd75, c8a2ab05) are copies of
refrigerator_concept, 29 files and ~10 k lines each. Every fix since had to be patched into each copy by script
(depth prior, wall/room-axis priors, `birth_on_priors`) or exists in one copy only (headboard; self-body term,
uncommitted in bed_concept). Phase 0 (`2026-11-10-countable-agents-phase0-design.md`) already moved room-following,
room filing and the epistemic planner to `common/`; this design finishes the job.

## 2. What the code already says (Fable review, 11-14; class names normalised)

- nightstand vs toilet: **6 differing lines** (comments, one texture attribute). Verified.
- nightstand vs bed: ~145 lines — ~55 the self-body term (already class-free, `namespace rc::self_body`), ~40 the
  headboard (`bed_belief.cpp:50-63`, `bed_fitter.cpp:888-908`, `bed_model.cpp:98-105`).
- refrigerator vs nightstand: ~750 lines, mostly comments; the real changes (gap variance in `flush_weight`, room-axis
  prior, `depth_mean()`, class-relative aspect) are class-agnostic.
- Already generic: the recursive-Laplace engine (`common/ai_belief/recursive_laplace.h`, templated on N, N = 3..7 in
  use); the LiDAR ray factor (needs only `sdf_prim`/`sdf_jacobian`/`n_prims`); the support bank (a 57-line adapter);
  the front-mode accumulator and clearance prior.
- **Differs in kind, not numbers:**
  1. *Quarter-turn relabel* (`bed_belief.cpp:714-727`, `apply_mode_rotation`): swapping w↔h is a symmetry of a bare
     box only. With a part on one face it is not. The allowed yaw modes must be **derived from the composed shape**.
  2. *Robust top* (`bed_fitter.cpp:888-908`): "box explains this point better than the headboard" generalises only
     if parts carry a **role** (body / appendage).
  3. *Door-seam front cue* (`bed_projection.cpp:42-614`): fridge-specific perception, off for bed. Becomes a **named
     optional cue** in the shared library, declared in the manifest.
- The six-ness lives in the belief class: `Matrix<float,6,…>` (53 occurrences in `bed_belief.*`), literal indices
  (`Id(5,5)`), `accumulate_lidar_rays<6>`, `std::array<DofSpec,N>`.
- Jacobians are central finite differences today (12 SDF evaluations per point per GN iteration, slope-clamped,
  `bed_belief.cpp:171-187`), not analytic.
- The manifest is **not** the source of values today: `verify_<obj>_manifest` only prints DIFFERS/MISSING; values
  live in a ~400-line `etc/config.toml`. Only the vertical span is adopted from the manifest
  (`concept_manifest.h:291,347`).
- The 3D viewer draws a mesh as a glyph keeping the node's w:d:h **ratios** under one uniform scale
  (`graph3d_scene_builder.cpp:340-364`), not a metric stretch.

## 3. Goals

- **G1. One generic program** (working name `countable_concept`), one binary for every class. It is given the class
  folder: `etc/config.toml` (agent plumbing: id, peers, domain), `<obj>.concept.toml` (the manifest), `meshes/<obj>.obj`.
  One process per class as today (one clone per object, owner 11-13), each with its own `[Agent] id`. **No thin
  per-class mains** — that is where per-class C++ would creep back.
- **G2. The common procedure**, identical for all classes: detection intake, self-body / room-envelope /
  foreign-claim birth evidence, the instance tracker, the birth snap on priors, the recursive-Laplace fit over the
  declared state, robust top and over-segmentation handling, existence, identity, affordances, graph publishing,
  history, cleanup.
- **G3. A small shape library, composed in the manifest.** Primitives grow only when a class needs one: first `box`,
  then `panel` (a thin box on a face), then box subtraction (bathtub, sink). Each part declares a role
  (`body` | `appendage`) and, if it is not centred, the face it attaches to (`back`, `front`, `left`, `right`, `top`).
  **No expression language:** each part parameter is a constant, a state variable, or a state variable times a
  constant. **No mesh-derived SDF fallback** (owner 11-14).
- **G4. Declared state.** Base `[cx, cy, H, w, h, yaw]` at fixed indices (so the existing priors keep addressing
  them), plus extra variables declared in the manifest, each with a mean and spread. A part size that varies between
  instances is an **estimated variable with a prior** (owner 11-14). Allowed yaw modes derive from the shape.
- **G5. Display.** `mesh_path` and the base dimensions on the node, as now. Extra variables go on the node as **one
  generic pair** of attributes (names + float vector), registered once in cortex — one owner reinstall, not one per
  class.
- **G6. The manifest becomes the source of class facts.** Physical facts (sizes, attachment, alignment, shape,
  confusable classes, detector labels) with provenance. Box-model tuning constants become **procedure defaults** in
  `common/`, not manifest entries.

## 4. Steps (owner 11-14: four steps)

**Step 1 — one binary, reproduction only.** Build `countable_concept` from the bed source (bed = nightstand + headboard
+ self-body; with no headboard declared it is the nightstand's code — planner's call, 11-14) with the class as a
runtime string. Run it for nightstand, toilet and bed. The headboard is a **fixed part** declared in the bed manifest
(no shape library yet, no new variable). The self-body term moves to `common/` (it is class-free) and is on for all
three. *Pass:* reproduces objects_live3 — bed and left nightstand within ~5 cm of ground truth, toilet tank to the
north wall — and no robot-plate bed. The three clone folders shrink to config + manifest + mesh.

**Step 2 — variable state.** The belief takes a run-time state size (Eigen max-size matrices with a fixed MaxN: no
per-point heap allocation; other agents untouched). Headboard height becomes the seventh variable with a prior.
Allowed yaw modes derive from the shape. *Pass:* bed at least as good as Step 1; the headboard height converges when
its top is seen and stays at its prior otherwise; fit time measured, agent still ~10 Hz.

**Step 3 — a new class by manifest only.** Box subtraction in the library; the bathtub (then the sink) added with a
manifest and a mesh and no C++. *Pass:* born and fitted live in the bathroom.

**Step 4 — other primitives only when a class needs them.**

**Tracker track (after Step 1, independent of 2–4).** The tracker rework from the owner's pasted LMB prompt (11-14)
lands here, once, in `common/instance_tracker`, opt-in per agent (default unchanged for the other 8 users):
per-candidate existence probability whose misses weigh by P(detect) instead of the one-missed-cycle reset
(`instance_tracker.h:345`), adaptive birth ∝ 1 − P(assigned), per-detection `xy_var` so candidates match by variance,
match update with clutter (r·p_D·g / (r·p_D·g + κ)), calibrated detector confidence, published expected count. *Pass:*
the right nightstand is born in the bedroom tour. Its own design detail goes in the plan.

## 5. Non-goals

- refrigerator, chair, cabinet, table, bottle, door, hood are not migrated now; later, one at a time, compared on a
  replay (old and new cannot run live together: they own the same nodes).
- No change to `common/instance_tracker` default behaviour.
- No analytic Jacobians, smooth union, capsules, cylinders in the first milestones — finite differences stay, so
  results are comparable.
- Not articulated objects, not uncountable classes (kitchen runs), not humans.

## 6. Anti-goals (failure even if it works)

- Per-class C++ anywhere outside a genuinely new shared primitive or a named optional cue.
- Thresholds or gates added to make one class work (CLAUDE.md modelling rule).
- A manifest that is a config dump of tuning knobs.
- Rate regression: the agent stays ~10 Hz; room/controller/retina at nominal rate.

## 7. Constraints

- CLAUDE.md rules (Queued DSR signals, type-attributed attributes, `std::from_chars`, and/or/not, -j8 capped builds,
  SIGTERM only, unique `[Agent] id`); the owner reinstalls cortex for new attributes.
- CONCEPT_AGENT_LIFECYCLE.md contract (`[Owns]`, birth tracker-only, existence channels); a conformance row for
  `countable_concept`.
- Agents talk only through the graph / memories. Meshes are the agent's own or CC0-like.
- Commits stay local on feat/countable-phase0; merge only when the owner asks.

## 8. Left to the builder

- Manifest TOML schema for parts, variables and cues; how config.toml shrinks.
- How the bed's role-tagged robust top generalises to N parts.
- Replay mechanism for comparisons (mask + pose replay vs several live tours per arm).
