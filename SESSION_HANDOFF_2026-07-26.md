# Session handoff — chair_concept existence/pose/orientation + dining-set rig

Date: 2026-07-26. All changes **UNCOMMITTED** and **built green** unless noted. Primary agent:
`chair_concept` (+ one `room_concept` edit). Scene under test: **dining set = 1 table + N chairs**
(webots, beta config). Read this top-to-bottom to resume.

---

## 0. THE LIVE OPEN BUG (start here)

**Chairs around the table fit with WRONG YAW even from good, face-on views.** Root cause: the seat is
square (`sw==sd==0.45`), so yaw is 4-fold ambiguous and only the backrest breaks it; the backrest is a
thin plate with too few points to resolve reliably → the fit lands on a wrong mode and sticks
(`chair_1` observed stuck ~180°, backrest pointing *at* the table). No single view fixes it — the
information isn't in one chair's mask.

**Chosen solution: a dining-set / circular-rig meta-concept** that pushes precision-weighted priors
(chair yaw = face table center) down to the part agents. Full spec written this session:
**`DINING_SET_RIG_PLAN.md`** (repo root). Memory: `[[dining-set-rig-metaconcept]]`.

**To start**, the user must answer 4 decisions (bottom of the plan doc):
1. circular-first, or circular+rectangular?  (rec: circular-first)
2. anchor center on the `table` node, or chair-centroid only for v1?  (rec: table-anchor)
3. `KappaYawMax` cap — how hard the rig may ever push yaw (~1/(20°)²)?
4. separate `dining_set_concept` agent vs. fold in?  (plan assumes separate)

Then prototype **Phase 1 (yaw-only)** — that alone should snap `chair_1` to face the table.
⚠ Phase 0 registers new cortex attributes (`rig_yaw_prior`, `rig_yaw_kappa`, …) → needs a **cortex
reinstall by the user** before consumers compile.

---

## 1. What changed this session (chair_concept, all UNCOMMITTED, build green)

The whole existence/removal + be-still + ricoh-gate stack landed. Config is in `etc/config.toml`
(`[Existence]`, `[Tracker]`, `[ChairModel]`). Files: `src/specificworker.{cpp,h}`, `chair_fitter.{cpp,h}`,
`chair_instance.h`, `chair_belief.{cpp,h}` (read-only ref), `chair_config.{h,cpp}`.

1. **Existence log-odds removal** (`ChairInstance::exist_logodds`, `SpecificWorker::update_existence_beliefs`).
   Replaces the wall-clock stillbirth prune (which had age-immunity + binary-streak bugs). Two evidence
   channels, only while in the ZED frustum + on a fresh sensor frame:
   - WON a mask → sign by EXPLANATION = `(support/expected)·(1−clutter)`; sparse OR all-clutter → negative.
   - WON NOTHING → vacate negative, confidence RAMPS with `frames_since_detection` (anti-churn grace).
   Removed at `L < RemoveLogodds(−3)`, NO age immunity; real chair pinned at `MaxLogodds(+4)`.
   `[Existence]` cfg: Enabled, BirthLogodds 1.0, RemoveLogodds −3, MaxLogodds 4, EvidenceGain 0.15,
   ExpectedSupportC 2500, AdequacyRef 0.25, AdequacyCap 1.5, CalibAdapt 0, VacateConfidentFrames 45.
   History: v1 immortal → v2 churned (real chairs removed) → v3 (current, two-channel + ramp).
2. **Occlusion gate** (`ChairFitter::los_occluded`): a chair hidden behind a closer object (other chair /
   any detected mask this frame, via bearing cone) does NOT vacate. `[Existence]` OcclusionCheck=true,
   OcclusionMarginM 0.30. ⚠ can't see a non-masked wall/box occluder — add DSR/wall geometry if it bites.
3. **Room-containment pose prior**: `P(chair outside walls)≈0` from the trusted room `delimiting_polygon`
   (`ChairFitter::point_in_room`, `refresh_room_geometry`). Suppresses out-of-room BIRTHS + removes an
   out-of-room instance (strong negative BEFORE the frustum gate, so it reaches a chair behind a wall).
   `[Existence]` RoomPrior=true, RoomMarginM 0.40, OutOfRoomGain 1.5. Also: stale-room-node re-resolve.
   **Verified live this session: `roomprior_loaded=1`, real chairs `inroom=1 L=4`** — earlier failures
   were a STALE BINARY, not a missing polygon.
4. **Be-still-to-update as CONTINUOUS PRECISION** (AIF; replaced a hard gate the user challenged).
   `motion_magnitude()` = max(|motion_dotd|, ego_lin + AngLever·ego_ang) from the transform chain;
   `periphery_penalty()` = clamp((centroid_radius/PeriphRef)²). Pose common-mode
   `mot_{pos,yaw}_var = (motion_cm_*_gain·motion_mag)²·periph`. Still OR centred ⇒ full authority; moving
   AND peripheral ⇒ confirm-only as precision→0. Existence negative ×= `frame_reliability()`.
   `[ChairModel]` MotionCmPosGain 0.30, MotionCmYawGain 0.50, AI2AngLeverM 2.0, AI2PeriphRef 0.50,
   AI2MotionRefMps 0.60. OLD hard gate kept behind `AI2MotionConfirmOnly=false` (+AI2Still*) for A/B.
5. **ZED-only BIRTH gate** (`DetectionView.birthable = depth_var==0`): a ricoh LiDAR-depth slice
   (depth_var>0, unreliable) may associate/confirm but NOT birth → prevents ricoh phantoms (chair_3/4 at
   8.8 m outside room). `[Tracker]` RicohBirthEnabled=false, RicohBirthConf 0.60, RicohBirthMaxVar 0.005.
   Principle: ZED = sole birth+pose+shape+orientation authority; ricoh = bearing/confirm only.
6. **Diagnostics CSV** `etc/chair_existence_log.csv` (the user reads CSVs, not stdout): per instance every
   60 sensor frames — `cycle,node,L,cx,cy,inroom,roomprior_loaded,roi,won,since_det,occluded`. Also stdout
   `[existence]` + `[room-prior]` lines.

## 2. room_concept change (UNCOMMITTED, built green — needs restart, affects fleet)

`src/room_scene_graph.cpp` adopt branch: `delimiting_polygon` was written ONLY on room CREATE, never on
ADOPT → an adopted bare/persisted room node left ALL consumers' containment inert. Now idempotent: authors
the polygon from `nominal_room_polygon()` if the adopted node lacks a ≥3-vert one. Fixes cabinet too.
Latent hardening (not the cause of this session's issue). Logs `authored missing delimiting_polygon…`.

## 3. Pending / not done

- **Orientation range-weighting** (PROPOSED, not built): `resolve_orientation`'s discrete mode-evidence is
  weighted only by MOTION, not RANGE — so a far chair (7.4 m) accumulated noisy far-view votes into FALSE
  confidence (`std_yaw_rep` collapsed 1.68→0.20). Fix = add a range/observability factor to the mode
  evidence weight (mirror the obliquity/range yaw cap) so a far chair stays honestly uncertain. NOTE: the
  **dining-set rig supersedes this** for the dining case (the prior disambiguates); keep range-weighting
  for the general far-chair honesty. Decide whether still needed after the rig lands.
- **depth_var → R for ricoh ASSOCIATION** (FOLLOW-UP): chair does NOT add `depth_var` to `R`, so a ricoh
  slice that *associates* to a real chair still drives pose/shape at full weight (table already folds
  depth_var→R). Add it so ricoh association only weakly refines. Small, mirrors table.
- **Ricoh-visibility vacate** (DISCUSSED, deferred): vacate keys on the ZED frustum; a chair only
  ricoh-visible (out of ZED FoV) never accrues absence. table_concept made ricoh confirm-only on purpose
  (oblique 360 absence unreliable). Open design call.

## 4. How to verify after restart (all via CSVs the user reads)

- `etc/chair_existence_log.csv`: `roomprior_loaded` (1=polygon active), `inroom`, `L`, `won`, `since_det`.
- `etc/chair_events.csv`: BIRTH / MERGE / REMOVE / SUPPRESS (with reason: "outside room", "logodds …").
- `etc/ai2_log.csv`: per-cycle belief — cols incl energy(5), range(11), obliquity(12), clutter(13),
  cx(14) cy(15) yaw(16), seat_w(17) seat_d(18), std_yaw_rep(24).
- Out-of-room phantom should show `inroom=0, roomprior_loaded=1`, `L`→−3, then a `REMOVE` event.
- Wrong-yaw bug: chairs well-observed (obliquity ~0.99) but yaw wrong = the square-seat/backrest issue → the
  rig is the fix; NOT an observability problem.

## 5. Build / run

- `cbuild` = `cmake -B build && make -C build -j32` (nuke `build/CMakeCache.txt` after adding sources).
- Match libdsr Eigen alignment: NO `-march=native`, NO `EIGEN_MAX_ALIGN_BYTES=0`.
- chair `[Agent] id = 20`. dining_set agent (new) needs a UNIQUE id (propose 22; verify vs cabinet/others).
- Restart room_concept too (adopt-path change) — affects the shared room node / whole fleet.
- Diagnostics go to STDOUT (terminal) AND now to CSVs. User reads CSVs — prefer the CSV channel.

## 6. Key memory pointers (persist across sessions)

`[[existence-belief-removal-plan]]` (the long thread of this session), `[[dining-set-rig-metaconcept]]`
(+ `DINING_SET_RIG_PLAN.md`), `[[table-round-vs-square-model]]` (rig disambiguates it),
`[[ai2-concept-fit-principles]]` (read before touching any belief), `[[no-threshold-patches]]`,
`[[ask-before-changing-running-params]]`, `[[agent-id-collision-crash]]`.

## 7. Standing method notes (learned/confirmed this session)

- The user reads **CSVs, not stdout** — route diagnostics to files.
- Prefer the **AIF-continuous** (precision) form over hard gates; the user will challenge a threshold.
- A "still not working" report is often a **STALE BINARY** — check `bin/` mtime vs the CSV mtime first.
- ricoh depth is unreliable for pose/extent/orientation — ZED drives geometry, ricoh confirms.
