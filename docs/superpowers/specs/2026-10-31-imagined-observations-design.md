# Imagined observations: sampling a concept and showing what the sensors would see — design

Date: 2026-10-31. Status: DRAFT, from a brainstorm in the chat_agent session. It builds on the perception cards
(`2026-10-31-perception-cards-design.md`, built on `feat/perception-cards`). Chair is the reference concept.

## 1. Intent

The person asks "show me how you imagine a chair", "under LiDAR or in the image", or "…if it were behind the
table". The robot should answer with what its **generative model** predicts:

1. **Sample a chair from the concept prior,** p(s | chair, context). This is a concrete instance: sizes, subtype,
   pose. It must come from what the robot believes chairs are like, and it must happen BEFORE anything reaches the
   sensors. (This is the user's point: you cannot submit "a chair" to a sensor field; you submit *a* chair.)
2. **Submit each sample to the sensor models,** p(o | s). This gives the predicted ZED mask and depth, and the
   predicted Helios/Bpearl returns, all through the real intrinsics, mounts and occluders of a world snapshot.
3. **Show a distribution, not one picture.** Several samples are overlaid. Their average is the expected
   observation, and the average P(detect) is the honest marginalisation over the concept's shape uncertainty.

Why:
- **Explainability for the person.** It shows how the robot sees.
- **A better `can_see` for hypothetical objects.** Today the card's single prior box is a point estimate.
- **A figure for the thesis's "self code" and active-inference chapters.** It is literally a sample from A and its
  prior.

## 2. The governing rule: the model that fits is the model that imagines

The prior and the shape used for imagination must be **exported by the concept agent from its own model**: the
model its fitter and its existence channel run. No second, hand-written chair may exist anywhere. This is the
perception cards' closed-term principle applied to shape: imagination must not drift from perception.

## 3. Chair today: what exists, and the inconsistency to settle first

**The shape model.** `chair_model.h` holds the compound SDF the fitter uses to split masks:
- the state is `ChairState{cx, cy, cz, yaw, seat_w, seat_d, seat_h, back_h}`, with `cz` pinned to the floor;
- the fixed constants are seat thickness (0.04 m in the model; `Tracker.BirthSeatThick` 0.075 in the config) and
  leg half-side 0.02 m;
- SDF = min(seat slab, backrest panel, 4 legs).

**Three sources disagree on the chair's size:**

| source | seat_w × seat_d | seat_h | back_h | top |
|---|---|---|---|---|
| `ChairState{}` defaults | 0.45 × 0.45 | 0.45 | 0.45 | 0.90 |
| `common/concept_manifest/chair.concept.toml` (`from = "nominal"`; says size is a FIXED template, pose-only state) | — | 0.45 | 0.45 | 0.90 |
| `etc/config.toml` `Tracker.BirthSeat*`, which the agent RUNS and the card now uses (Task 9) | 0.60 × 0.52 | 0.595 | 0.655 | 1.25 |

Plus a size spread `ChairModel.AI2PriorSizeStd = 0.15`. The manifest also still says removal uses
`zed_detectability()`, which Task 9 replaced.

**Decision (step 0). The user said "we need one source of truth"; the mechanism below is my proposal, pending a
yes.**
- **The concept manifest becomes the single source of the chair prior.** `common/concept_manifest/chair.concept.toml`
  holds "what a chair IS, world facts"; config is tuning.
- **The values it takes** are the ones the agent RUNS today:
  - seat 0.60 × 0.52, seat top 0.595, backrest 0.655 (top 1.25), seat thickness 0.075, size σ 0.15;
  - each with provenance;
  - the stale "fixed template" and `zed_detectability` text is corrected.
- **chair_concept reads the prior from the manifest at startup.**
  - `Tracker.BirthSeat*` and `ChairModel.AI2PriorSizeStd` are removed from `etc/config.toml`.
  - The birth template, the belief prior and the card's generative block all read the same struct.
- **`ChairState{}` defaults stop carrying chair facts.**
  - The initial state is built from the manifest.
  - An unreadable manifest refuses to start, the same rule as `rc::manifest::provenance_ok`; there is no silent
    0.45 fallback.
- **Test:** the generative block, the birth template and the manifest give identical numbers.
- **The same rule applies to each concept** as it gets a generative card.

## 4. The generative card

Extend the perception card (`common/perception_card/card.h`) with a `generative` block. The agent fills it from
its own model and config, the same way it fills the detection terms. The block has four parts.

**4.1 Shape program.** A closed vocabulary of primitives parameterised by state variables:
- primitives: `box`, `cylinder`, `slab`, `panel`, `leg_set` (n legs at the corners of a footprint);
- for chair: `slab(seat_w, seat_d, thick @ z = seat_h − thick)`, `panel(seat_w, thick, back_h @ the −y edge, from
  z = seat_h)`, `leg_set(4, half = 0.02, from 0 to seat_h − thick)`;
- the program is written by the agent from the same constants its SDF uses.
- Test: the primitives' union must match `ChairModel::sdf_point_at` at sampled points (the sign agrees, the
  distance agrees within the primitive approximation). This is the shape version of the term-identity test.

**4.2 State distribution.**
- One entry per state variable: `mean`, `sigma`, `lo`, `hi`.
  - Sizes are log-normal or truncated normal within physical bounds.
  - Means come from the birth template; sigmas from `AI2PriorSizeStd` (and per-variable values where the config
    has them).
- Categorical subtypes, where the concept has them: table round/square from `obj_shape`. Chair has none today; an
  armchair / stool split is out of scope.
- Correlations are optional, as a small covariance block. Seat height and back height are plausibly correlated.
  Default independent, and say so.

**4.3 Placement prior** p(pose | context):
- **Support.** `floor`: cz = 0, pinned, as chair_belief does.
- **Contexts.** A mixture over:
  - `free`: uniform over the room's free space;
  - `against_wall`: the backrest toward the nearest wall, within a gap distribution;
  - `at_table`: drawn from the ring arrangement around a table. The ring metaconcept already carries the
    arrangement's empirical distribution.
- The mixture weights are config, or learned (§4.4).
- Yaw follows the context: toward the table centre, or away from the wall. In `free` it is uniform.
- **Hard support constraints.** A sample must not interpenetrate walls or other objects. These are physics, not
  tuning: an imagined chair inside a wall is impossible, not unlikely. Reject and redraw, and report the
  acceptance rate in a note.

**4.4 Experience refinement (hierarchical).**
- The fitted sizes of real chairs met in this house are in LTSM/EM. Update the class prior with them:
  normal-inverse-gamma per size variable, with the card's prior as the hyperprior.
- So imagination drifts from "a generic chair" toward "one of our chairs".
- This is optional in v1. It is designed so the card's prior is the hyperprior and the refined prior is a
  separate, dated block.

## 5. Sampling

`std::vector<Sample> sample_concept(const Card&, const World&, const Context&, int n, std::uint64_t seed)`, in a new
pure Eigen header `common/perception_card/generative.h`. The steps:

1. Draw the subtype, then the sizes from §4.2.
2. Draw the pose:
   - **"Imagine a chair here":** the pose is fixed at the named place; yaw is drawn from the context.
   - **"Imagine a chair":** the context is chosen by the mixture, and the pose is drawn from it.
3. Reject samples that interpenetrate the snapshot world; redraw up to a budget.
4. Each sample carries: its state, its instantiated primitives in the room frame, and its weight (equal unless
   importance-sampled).

The seed is explicit, so a picture can be reproduced.

## 6. Imagined sensor views

Pure Eigen. They use the same world snapshot as `can_see` (`chat_agent/src/perception_world.h`) and the same
intrinsics, mounts and sensor models.

**6.1 ZED.** `ImaginedImage render_zed(const World&, const Viewpoint&, std::span<const Primitive>)`:
- Rasterise the primitives into a mask and a depth image at the ZED's resolution. It can be downsampled, e.g.
  320×180, for speed.
- Use the pinhole intrinsics and the mount pose, with a z-buffer against the snapshot's walls and obstacles.
- Output:
  - the predicted mask: visible pixels of the chair;
  - the occluded part, which is the mask with no z-buffer minus the visible mask, so it can be drawn shaded;
  - the fill (max-axis bounding-box fraction, the agent's definition);
  - P(detect) from the card's evaluator at the sample's exact geometry.

**6.2 LiDAR.** `ImaginedScan render_lidar(const World&, const Viewpoint&, const LidarRig&, std::span<const Primitive>)`:
- For each ring (from `SensorModel` ring elevations, through the mount) and each azimuth step, ray-cast against the
  primitives and the snapshot.
- Add range noise from `range_sigma_floor_m`. Apply the dead zone and max range, and q_return as a Bernoulli drop.
- Output the 3-D returns that hit the chair, the returns that hit the occluders, and λ.
- This turns the Poisson term into something visible: legs as sparse hits, the seat as an arc, a backrest that
  falls between rings simply missing.

**6.3 Agreement with the evaluator.** For each sample, the rendered fill and visible fraction must agree with
`evaluate()` at that sample's geometry, within the rasterisation error. This is a test.

## 7. Showing it

- **Tool:** `imagine_view(concept, where?, from?, sensor = zed|lidar|both, n = 8)` in chat_agent. It is read-only
  and goes in `allowed_in_wake`.
- **Display:** in the chat's side canvas via the existing `draw` path. The panels are:
  - a ZED panel: the n masks overlaid as a soft silhouette (the per-pixel frequency over samples), with the
    occluded parts hatched and the image border drawn;
  - a LiDAR panel: top-down and side-on views, the returns per sample in faint colour, the ring elevations drawn as
    lines in the side view so the gaps are visible;
  - a caption with the mean P(detect) ± its spread over samples, the limiting factor that is most frequent, the
    context mixture used, and the acceptance rate.
- **Answer text, e.g.:** "Here are 8 chairs as I imagine them by the table. My camera would see the seat and back in
  most of them (P 0.78 ± 0.06); my LiDAR's rings pass above the seat at 3 m, so it would mostly hit the legs and
  backrest."

## 8. Consequence for `can_see` and `imagine`

For a hypothetical object, a card's P(detect) becomes the Monte-Carlo average of `evaluate()` over prior samples,
for example with n = 32. That replaces the single prior box. `where_to_stand` can keep the point estimate for
speed, with a note. Existing instances keep using their measured geometry.

## 9. Goals, non-goals, anti-goals

**Goals:**
- Chair end to end: generative card, sampler, ZED and LiDAR renders, a canvas view, and the `imagine_view` tool.
- The shape-identity test against `ChairModel::sdf_point_at`.
- Evaluator agreement tests.
- The chair prior reconciled into one source of truth.

**Non-goals (v1):**
- Other concepts. Table, fridge and door follow the same pattern next; door's shape program is the leaf plus the
  frame it already has.
- Photorealistic rendering. Masks and depth are enough, because the detector sees masks.
- Learned priors. §4.4 is designed but optional.
- The ricoh.

**Anti-goals:**
- A second chair model written for imagination.
- Showing one "typical chair" as if it were the belief.
- Samples that interpenetrate the world.
- Pictures that disagree with the numbers the evaluator gives.

## 10. Constraints

- All CLAUDE.md rules: C++23 with and/or/not; from_chars; no thresholds where a model term works (the rejection
  constraint is physical, §4.3).
- The common code is header-only, Eigen and std, with no DSR.
- Ownership: the agent writes its own card only.
- Card JSON keeps a closed vocabulary (primitives, distributions, contexts), as the terms do.

## 11. Decided vs my calls vs left to the builder

**Decided with the user:**
- Sample from the concept prior first, then submit to the sensor fields.
- Start with chair.
- Show both the image and LiDAR views.

**My calls:**
- The generative block lives in the perception card.
- A closed primitive vocabulary; the shape-identity test against the agent's SDF.
- A placement mixture of free, against-wall and at-table; hard rejection for interpenetration.
- Monte-Carlo P(detect) for hypotheticals.
- `imagine_view` as a new tool, rather than an `imagine` question type.
- Downsampled rasterisation.

**Left to the builder:**
- Primitive JSON layout, rasteriser details, n and resolution defaults, colours, and canvas layout.

## 11b. Builder-check rulings (my calls; please check)

1. **`at_table` is not in v1.**
   - The ring metaconcept has a fitted ring-arrangement belief (ring_belief.h), not an exported empirical
     distribution. v1 contexts are `free` and `against_wall`.
   - `at_table` waits for a follow-up in which the ring agent publishes its slot poses and Σ to the graph. The card
     must not depend on ring code.
2. **Branches.**
   - `generative.h`, the sampler and the renderers go in `common/perception_card/` on `feat/perception-cards`.
   - The chair export goes in chair_concept on the same branch.
   - `imagine_view` goes in chat_agent on `feat/vlm-missions` after merging. This is the same pattern as `can_see`.
3. **Shape JSON uses fixed bindings, not an expression language.**
   - Each primitive dimension or offset is either a constant or `a·var + b`, with `var` one of the state variables.
   - The primitive types are closed: slab, panel, leg_set, box, cylinder. The parser stays tiny.
4. **Shape identity: the SDF is the truth.**
   - Test points are sampled around the chair. At every point farther than 1 cm from both the primitive surface and
     the SDF zero level, the inside/outside sign must agree.
   - Near the surface, |distance| must agree within 1 cm.
   - If they disagree, the primitives are fixed. The SDF is never patched to match.
   - The test runs on the config-derived state the agent actually births (§3), and also on the prior's ±2σ corners.
5. **P(detect) per sample.**
   - `evaluate()` takes the sample's bounding box, as today: the detection model is a box-fill model.
   - The authoritative fill is the agent's definition: the max-axis fraction of the projected bounding box.
   - The rendered mask's own bounding box must reproduce the evaluator's fill within one pixel's fraction.
   - The mask-based visible fraction must agree with the evaluator's within 0.05. That difference comes from
     compound versus box shape and is reported, not hidden.
6. **The rasteriser.**
   - It is the first image-space renderer in common/. It reuses `rc::nbv::Sensor`'s pinhole intrinsics and the
     camera mount.
   - Occluders are the snapshot's obstacles as boxes and the walls as vertical slabs from 0 to the room height. The
     room height comes from the graph if published; otherwise 2.5 m, with a note.
   - The floor and ceiling are not rendered, since they are not occluders of a chair.
7. **Rejection is physical only.** A sample is rejected when its footprint, at its z span, interpenetrates a wall
   piece or an obstacle. A chair standing in a doorway or in a door's swing zone is allowed: that is possible, so
   it is not rejected.
8. **Monte-Carlo P(detect) is used only in `imagine_view` and `can_see` for HYPOTHETICAL objects,** those with no
   graph instance, with n = 32. Existing instances keep their measured geometry. `where_to_stand` keeps the point
   estimate and says so in a note. Budget: under 200 ms per call, measured.
9. **Data path to the canvas.**
   - C++ computes everything and writes `imagined.json` into the canvas run directory. It holds the samples'
     primitives, the LiDAR returns per sample, the ring elevations, the stats and the captions.
   - It also writes a ZED panel PNG: the per-pixel frequency over samples, with occluded parts as a second channel.
   - `rcanvas.py` gains a reader plus `draw_imagined_zed` and `draw_imagined_lidar`.
   - No new tool-result schema: the tool returns the text answer and triggers the drawing, as `draw` does.
10. **Seed.** The tool picks a seed and echoes it in the caption. The person may pass one, and the same seed and
    snapshot give the same picture.
11. **Bounds have one source.** The sampler's lo/hi are exported from ChairModel's own size-bounds enforcer. The
    card carries them but never defines them, so no second chair definition exists.

## 12. Open question for the user

- ~~Which chair prior is the truth?~~ The user said "we need one source of truth". §3 proposes the manifest as that
  single source, with the values the agent runs. This awaits the user's yes.
