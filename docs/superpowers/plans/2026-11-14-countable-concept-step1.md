# countable_concept — Step 1 (one binary, reproduction only) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One `countable_concept` binary runs nightstand, bed and toilet, each from a class folder holding only
config + manifest pointer + mesh, and reproduces the objects_live3 results.

**Architecture:** `countable_concept/` is made from `bed_concept/src` (bed = nightstand + headboard + self-body; with no
headboard declared it is the nightstand's code, which differs from toilet's by 6 lines). `Bed*` identifiers become
`Countable*`; the class becomes a run-time string `[Concept] Class`. The headboard becomes a back panel declared in the
manifest (`[prior.headboard]`), absent for other classes. The self-body term moves to `common/self_body/`. Each class
folder keeps `etc/`, `meshes/` and a `bin/<obj>_concept` symlink to the shared binary, so process names (and
`concept_agents.sh`) stay per class.

**Tech Stack:** C++23, Qt6, DSR/cortex, Eigen, RoboComp `robocompdsl`-generated main, CMake; tests are plain CMake
executables under `<agent>/tests/` (no gtest).

**Spec:** `docs/superpowers/specs/2026-11-14-countable-object-procedure-design.md` (§4 Step 1).

## Global Constraints

- CLAUDE.md: DSR signals Queued (never `Qt::DirectConnection`); type-attributed attribute access only; parse numbers
  with `std::from_chars`; `and`/`or`/`not`; no new thresholds/gates; stop agents with SIGTERM only.
- Build: check `pgrep -a cc1plus` first; `systemd-run --user --scope -p MemoryHigh=40G -p MemoryMax=44G make -C <dir>/build -j8`.
  No `-march=native`, no `-DEIGEN_MAX_ALIGN_BYTES=0`. After adding sources, delete `build/CMakeCache.txt` and reconfigure.
- Tests that parse agent files run with `LANG=es_ES.UTF-8` (the card test already does).
- Agent ids unchanged: nightstand 29, bed 30, toilet 31; `[Owns]` unchanged.
- Step 1 is **reproduction only**: no shape library, no new state variable, no tracker change, no config value changes.
- Commits are local on `feat/countable-phase0`; end messages with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Webots is shared: claim it from the `chat` and `room` sessions before Task 6, send FREE after.

## Review Focus

- A class with **no** `[prior.headboard]` (nightstand, toilet) must get exactly the old plain-box SDF — pinned in Task 3.
- The class string must reach **every** former literal (`"bed"` label match, node prefix `bed_`, manifest path, card
  `concept_`, dashboard strip label, presence `.agent`, log prefixes) — pinned by the grep check in Task 2.
- Three processes of one binary must be individually started/stopped/checked by `concept_agents.sh` (`pgrep -x` on the
  symlink name) — pinned in Task 5.
- A config still using old section names (`[BedModel]`, `[NightstandConcept]`) must fail loudly at start, not run on
  defaults — pinned in Task 4.
- Agent CWD is the class folder, so `../common/concept_manifest/<class>.concept.toml` and `etc/*.csv` still resolve —
  pinned in Task 5.

---

### Task 0: Commit the bed self-body work as it stands

**Files:** the uncommitted bed_concept changes (`src/bed_self_body.h`, `tests/self_body_test.cpp`, `src/specificworker_lifecycle.cpp`, `src/specificworker.h`, `src/bed_config.{h,cpp}`, `src/CMakeLists.txt`, `tests/CMakeLists.txt`, `etc/config.toml`).

- [ ] **Step 1:** Build bed_concept and run `bed_concept/tests/build/{self_body_test,headboard_sdf_test}` and `LANG=es_ES.UTF-8 bed_card_test`. Expected: all PASS (self_body 15/15).
- [ ] **Step 2:** Commit: `bed_concept: self-body birth term — a mask on the robot's own top plate is not a bed (bed_5)`.

### Task 1: Move the self-body model to common/

**Files:**
- Create: `common/self_body/self_body.h` (content of `bed_concept/src/bed_self_body.h`, namespace `rc::self_body` unchanged)
- Create: `common/self_body/tests/self_body_test.cpp` + `CMakeLists.txt` (moved from bed_concept/tests)
- Modify: `bed_concept/src/specificworker_lifecycle.cpp` include → `"../../common/self_body/self_body.h"`; delete `bed_concept/src/bed_self_body.h`; drop the test from `bed_concept/tests/CMakeLists.txt`
- Modify: `common/run_tests.sh` to build/run the new test

**Interfaces:** Produces `rc::self_body::*` exactly as in `bed_self_body.h` today (no signature change).

- [ ] **Step 1:** Move files with `git mv`; fix includes.
- [ ] **Step 2:** Build the test via `common/run_tests.sh`; Expected: `self_body_test` 15/15 PASS.
- [ ] **Step 3:** Rebuild bed_concept; Expected: builds, card + headboard tests PASS.
- [ ] **Step 4:** Commit: `common/self_body: the robot self-body birth term, class-free`.

### Task 2: Create countable_concept from bed_concept, class as a run-time string

**Files:**
- Create: `countable_concept/` = copy of `bed_concept/{CMakeLists.txt,Makefile,generated,src,tests}` and `bed_concept.cdsl` → `countable_concept.cdsl` (no `etc/`, no `meshes/`, no `bin/`)
- Create: `tools/make_countable.py` — the one-off rename, kept for the record
- Rename: `src/bed_*.{h,cpp}` → `src/countable_*.{h,cpp}`; identifiers `Bed*`→`Countable*`, `bed_*`→`countable_*`, `BED`→`COUNTABLE`; executable/target `bed_concept`→`countable_concept`
- Modify: `src/countable_config.{h,cpp}` — new field `std::string concept_class` read from `Concept.Class` (required, no default); `kManifestPath` becomes `"../common/concept_manifest/" + concept_class + ".concept.toml"`; `rc::cfg::Reader` name and every `print` prefix use the `[Agent] name` string
- Modify: the class literals found by `grep -n '"bed' countable_concept/src/*` — label match (`specificworker_lifecycle.cpp` `sl.label == "bed"`, `countable_projection.cpp` `sl.label == "bed"`), node prefix in `countable_scene_graph.cpp`, `.agent` in `specificworker_presence.cpp`, `c.concept_` in `countable_card.h`, `.strip_label` in `specificworker_dashboard.cpp` — all from `cfg_.concept_class`

**Interfaces:**
- Produces: `rc::CountableConfig::concept_class : std::string`; `rc::CountableConfig::agent_name : std::string`;
  config sections `[Model]` (ex `[BedModel]`), `[Concept]` (ex `[BedConcept]`), `[Tracker]` unchanged; keys inside
  `[Concept]` lose the class word (`BedLogPeriodFrames`→`LogPeriodFrames`, `BedFilterEnabled`→`FilterEnabled`,
  `BedFilterLog`→`FilterLog`, `BedOwnWallSkipM`→`OwnWallSkipM`, …) — `tools/make_countable.py` emits the full
  old→new key map to `countable_concept/KEYMAP.md`, which Task 4 consumes.

- [ ] **Step 1:** Write `tools/make_countable.py` (copy + rename + section/key rename + KEYMAP.md); run it.
- [ ] **Step 2:** Replace the class literals listed above with `cfg_.concept_class` / `cfg_.agent_name`.
- [ ] **Step 3:** Check: `grep -rniE '"bed|bed_|BedModel|BedConcept' countable_concept/src` Expected: no hits except comments explaining history.
- [ ] **Step 4:** Clean configure + build. Expected: `countable_concept/bin/countable_concept` links.
- [ ] **Step 5:** Commit: `countable_concept: one binary for countable objects, class from [Concept] Class (from bed_concept)`.

### Task 3: The back panel comes from the manifest

**Files:**
- Modify: `common/concept_manifest/concept_manifest.h` — add `struct BackPanel { float height_m; float thickness_m; };` and `inline std::optional<BackPanel> load_back_panel(const std::string& path)` reading `[prior.headboard]` (`kind = "fixed_template"`, `face = "back"`; any other face/kind ⇒ print an error and return nullopt), numbers via `std::from_chars`
- Modify: `countable_concept/src/countable_config.cpp` — `ai2_headboard_h_m/t_m` set from `load_back_panel(manifest)`; the config keys `Model.HeadboardHeightM/ThickM` are removed (bed's values already live in `bed.concept.toml`: 1.25 / 0.10)
- Test: `countable_concept/tests/back_panel_test.cpp`

**Interfaces:** Consumes Task 2's `concept_class`. Produces `rc::manifest::load_back_panel`.

- [ ] **Step 1: Write the failing test** `back_panel_test`:
  - `load_back_panel(".../bed.concept.toml")` → `height_m == 1.25f`, `thickness_m == 0.10f`
  - `load_back_panel(".../nightstand.concept.toml")` and `toilet` → `nullopt`
  - with a CountableBelief built with params from each: for 200 random points in a 3×3×2 m cube around a state
    `cx=0 cy=0 H=0.6 w=0.5 h=0.5 yaw=0.3`, `sdf_prim` for nightstand/toilet params == `sdf_box` exactly (bitwise);
    for bed params == `min(sdf_box, sdf_headboard)`
  - run under `std::setlocale(LC_ALL, "")` with `LANG=es_ES.UTF-8`
- [ ] **Step 2:** Run it; Expected: FAIL (no `load_back_panel`).
- [ ] **Step 3:** Implement; wire into config.
- [ ] **Step 4:** Run `back_panel_test`, `headboard_sdf_test`, `furniture_prior_test` (moved from nightstand_concept/tests, run with nightstand params); Expected: all PASS.
- [ ] **Step 5:** Commit: `countable_concept: the back panel (headboard) is declared in the manifest`.

### Task 4: Class folders hold only config, mesh and a binary link

**Files:**
- Modify: `nightstand_concept/etc/config.toml`, `bed_concept/etc/config.toml`, `toilet_concept/etc/config.toml` — section/key rename from `KEYMAP.md`, plus `[Concept] Class = "<obj>"`; values unchanged (diff of values before/after must be empty — script check)
- Create: `<obj>_concept/bin/<obj>_concept` → symlink `../../countable_concept/bin/countable_concept`
- Delete (git rm): `<obj>_concept/{src,generated,tests,CMakeLists.txt,Makefile,*.cdsl}` for all three
- Modify: `countable_concept/tests/CMakeLists.txt` — card test parameterised: `countable_card_test <agent_dir>` built once, three `add_test` entries (nightstand, bed, toilet)
- Modify: `countable_concept/src/countable_config.cpp` — at start, if any section whose name ends in `Model` or `Concept` other than `[Model]`/`[Concept]` exists in the config, print `config uses pre-countable section [X] — see countable_concept/KEYMAP.md` and exit non-zero

- [ ] **Step 1: Write the failing test** `config_migration_test`: for each of the three migrated configs, every old key's value (from git `HEAD~` version) equals the new key's value via KEYMAP; and a config with `[BedModel]` makes `load_config` report the pre-countable error.
- [ ] **Step 2:** Run; Expected: FAIL.
- [ ] **Step 3:** Migrate configs (extend `tools/make_countable.py --migrate-config <dir>`), add the old-section check, create symlinks, remove clone sources.
- [ ] **Step 4:** Run `config_migration_test` and the three card tests (`LANG=es_ES.UTF-8`); Expected: PASS.
- [ ] **Step 5:** Commit: `nightstand/bed/toilet: class folders on countable_concept (config + mesh + link)`.

### Task 5: Launch, stop and register

**Files:**
- Modify: `room_concept/estimation_scripts/scripts/concept_agents.sh` — it already runs `$WT/$a/bin/$a`, which is now the symlink; keep executing the **link path, never its resolved target** (so `comm` = `<obj>_concept` and `pgrep -x` stays per class); CWD stays `$src/$a`; `agents.txt` additionally records the link target (`readlink -f`)
- Modify: `CONCEPT_AGENT_RECIPE.md` (new class = folder + manifest + mesh + id; no scaffold script), `CONCEPT_AGENT_LIFECYCLE.md` §6 (one row `countable_concept`, three classes), retire `tools/scaffold_from_refrigerator.sh` (git rm, note in RECIPE)

- [ ] **Step 1:** With Webots NOT running, start the three with `concept_agents.sh start <tmpdir>`; Expected: three processes, `pgrep -x nightstand_concept`, `bed_concept`, `toilet_concept` each one pid; each log shows `configuration loaded`, its own class manifest path and `[manifest] <obj> ✓`; they wait for peers.
- [ ] **Step 2:** `concept_agents.sh stop`; Expected: each `SIGTERM sent`, `all stopped`.
- [ ] **Step 3:** Commit: `countable_concept: launch per class through a bin/ link; docs`.

### Task 6: Live reproduction (acceptance)

- [ ] **Step 1:** Claim Webots (chat + room), then run the objects_live3 command with tag `objects_live4`:
  `cd room_concept/estimation_scripts && S=<phase0>/room_concept/estimation_scripts/scripts && N_LEGS=2 LOOK_S=60 ON_FLEET_UP="CONCEPT_AGENTS='nightstand_concept bed_concept toilet_concept' bash $S/concept_agents.sh start $PWD/crossing/objects_live4" bash $S/warm_start_tour.sh objects_live4 crossing/fixtures/three_rooms_door0`
- [ ] **Step 2:** Compare to objects_live3 with ground truth (bedroom room frame → world offset (−4.855, +9.285);
  bathroom via the stored layout `fixtures/three_rooms_door0/ltsm_history/*_layout.json`). Pass:
  left nightstand within 5 cm of (−6.36, 13.22); bed within 5 cm of (−5.18, 12.39), 1.75±0.1 × 2.10±0.1, mattress
  0.65±0.05, yaw back to the north wall; toilet tank to the north wall; **no bed birth on the robot plate**
  (no `bed_*` with cloud centroid z ≈ 0.78 within 0.3 m of the robot); each agent ~10 Hz (`Period =` lines ≈ 100 ms).
- [ ] **Step 3:** Send FREE; write results into the memory file `bath-bedroom-agents-plan-11-10.md`; commit any
  result notes.

---

## Tracker track (outline — its own plan after Step 1 passes)

In `common/instance_tracker`, opt-in per agent via `Tracker.CandidateExistence = true` (default false = unchanged for
the other 8 users): per-candidate existence r with miss update `r(1−p_D)/(1−r·p_D)` replacing the one-missed-cycle
drop (`instance_tracker.h:345`); adaptive birth ∝ 1 − P(assigned); per-detection `xy_var` passed by countable_concept;
match update `r·p_D·g/(r·p_D·g+κ)`; calibrated detector confidence; published expected count. Pass: the right
nightstand (≈(−4.00, 13.22)) is born in the bedroom tour.
