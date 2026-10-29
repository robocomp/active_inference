# Room-Transition Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Door crossings update the shared graph in one agreed order — room_concept switches the room, everyone else follows that fact; recognition proposes, room_concept adopts — so the three-room tour completes 6/6 with a consistent LTSM map.

**Architecture:** One owner and one graph fact per step (spec tables). The robot's room is decided ONLY by room_concept (the robot's live RT parent); ltsm and door_concept follow it. In the first-crossing case ltsm writes a `room_proposal` node instead of deleting/recalling rooms; room_concept accepts by renaming the provisional room in place (cortex rename) or rejects by marking the proposal.

**Tech Stack:** C++23, RoboComp DSR/cortex, Eigen; standalone test binaries (`EXCLUDE_FROM_ALL` CMake targets, or `g++` one-file tests in door_concept/tests).

**Spec:** `docs/superpowers/specs/2026-10-29-room-transition-protocol-design.md` (approved 10-29; read it — the case sequences are numbered there and referenced below as C1.n / C2.n).

## Global Constraints

- Worktree `/home/pbustos/robocomp/components/active_inference.worktrees/rt-build`, branch `feat/room-recognition`; commit by explicit pathspec; never push; end commit messages with the two attribution lines in the session reminder.
- CLAUDE.md rules: no thresholds/timers standing in for a fact (FLAG any unavoidable constant); `and`/`or`/`not`; C++23; QUEUED signals only; type-attributed DSR attrs only (no `runtime_checked_*`); `std::from_chars` for parsing; builds `make -C build -j8` after `pgrep -a cc1plus` (drop to `-j4` if another build runs); SIGTERM only.
- No new DSR attributes. Provisional = the existing `proto` self-edge (`rc::room::is_proto`). Proposal uses existing `room_id`, `recall_pose_prior`, `delimiting_polygon_x/y`, `state`, `object_subtype`, `parent`, `level`.
- Rule 1: no agent deletes or rebuilds a node it does not own (room_concept: rooms/walls/floor/robot RT; door_concept: doors; ltsm: passages, `recall_*`, `room_proposal`, memory).
- Rule 2: act on a fact, never on elapsed time.
- After any agent rebuild, confirm the binary contains the change (`strings bin/<agent> | grep -c "<new log text>"` ≥ 1) — a compile racing an edit left a stale binary on 10-29.

## Review Focus

1. **Proposal for a room that is no longer provisional** (P already promoted/adopted or deleted when the proposal lands): room_concept must mark it `rejected`, never adopt into a non-proto room. → Task 6 test `proposal_on_non_proto_rejected`.
2. **Two proposals in flight** (ltsm re-proposes before the first verdict): room_concept answers only the newest; ltsm deletes a superseded proposal itself. → Task 5 test `reproposal_replaces_pending`.
3. **Rename collides with a live node** (a `room_2` node already live): adoption is refused with `rejected`, no throw. → Task 6 test `rename_collision_rejected`.
4. **Robot crosses back before a verdict** (leaves P while the proposal is pending): ltsm withdraws the proposal (it owns it) and archives P as new. → Task 5 test `left_before_verdict_withdraws`.
5. **Wall with a door when the polygon loses that edge**: the wall is kept and moved to the nearest polygon edge, never deleted. → Task 7 test `wall_with_child_survives_shrink`.

---

### Task 1: cortex — allow renaming a node through `update_node`

**Files:**
- Modify: `/home/pbustos/robocomp/components/cortex/api/dsr_api.cpp:644-700` (the `"id and name must be unique"` guard in `DSRGraph::update_node`)
- Test: `/home/pbustos/robocomp/components/cortex/tests/` — new `rename_node_test.cpp` (follow the existing test layout there; if none fits, a standalone target in the cortex CMake)

**Interfaces:**
- Produces: `G->update_node(n)` where `n.name()` differs from the stored name, `n.id()` unchanged and the new name free ⇒ returns true; `get_node(new_name)` finds it, `get_node(old_name)` is nullopt, on THIS graph and on a second graph synced over DDS. A new name already used by another id ⇒ returns false (no throw).

- [ ] **Step 1:** Write `rename_node_test`: two `DSRGraph` instances on a private test domain (pick an unused domain, e.g. 77) joined like the existing cortex tests do; insert `room_4` on A, wait until B has it, rename to `room_2` on A, assert on both graphs: `get_node("room_2")->id()==id`, `not get_node("room_4")`, attributes and RT edges intact; then attempt renaming a second node to `room_2` ⇒ `false`.
- [ ] **Step 2:** Run it; expect FAIL (throw "id and name must be unique").
- [ ] **Step 3:** In `update_node`, replace the throw with: if the new name is used by a DIFFERENT id ⇒ return false; else let the LWW path (`update_node_local`, `dsr_lww_sync_engine.cpp:316-325`, already refreshes `name_map`/`id_map`) apply it. Check the remote-apply path also refreshes both maps; fix there if not.
- [ ] **Step 4:** Run the test; expect PASS. Run cortex's existing tests.
- [ ] **Step 5:** Commit in the cortex repo (local branch, no push). **Hand to the user to reinstall cortex**; Tasks 6 and 9 need it, Tasks 2–5 and 7–8 do not.

### Task 2: common — the robot's room by its live RT parent

**Files:**
- Modify: `common/room_resolve/room_resolve.h` (next to `current_room`)
- Test: `common/room_resolve/room_resolve_test.cpp` (create if absent; standalone g++ like door_concept/tests)

**Interfaces:**
- Produces: `std::optional<std::uint64_t> rc::room::robot_rt_room(DSR::DSRGraph& G)` — the room whose robot→room RT edge is live (`valid` absent or true); nullopt if none or more than one. Pure resolution over `RoomFact` is `std::optional<std::uint64_t> rc::room::resolve_robot_rt_room(std::span<const RoomFact>)` so it is testable without a graph.

- [ ] **Step 1:** Tests on `resolve_robot_rt_room`: one Live ⇒ it; Live + Retired ⇒ the Live one; two Live ⇒ nullopt; none ⇒ nullopt; proto Live ⇒ it (provisional rooms count: C2.3).
- [ ] **Step 2:** Run, expect FAIL (undefined).
- [ ] **Step 3:** Implement both (the RT scan already exists inside `current_room`; extract it).
- [ ] **Step 4:** Run, expect PASS. Commit.

### Task 3: ltsm — promotion follows room_concept's switch (C1.5, C2.5)

**Files:**
- Modify: `ltsm_agent/src/specificworker.cpp:539-640` (`step_promotion`), `ltsm_agent/src/room_eviction.cpp:1285-1350` (`promotion_evidence` stays for the log only)
- Test: `ltsm_agent/tools/promotion_trigger_test.cpp` + target in `ltsm_agent/generated/CMakeLists.txt`

**Interfaces:**
- Consumes: `rc::room::robot_rt_room` (Task 2).
- Produces: `bool ltsm::promotion_due(std::optional<std::uint64_t> robot_room, std::optional<std::uint64_t> current_edge_room)` = robot_room has a value and differs from current_edge_room. `step_promotion` promotes `robot_room` when due; the `p_inside < promotion_decision_prob` gate (line 569) and `Promotion.decision_prob` are removed (config key left harmless or deleted from etc/config.toml).

- [ ] **Step 1:** Test `promotion_due`: (room_3, room_1) ⇒ true; (room_1, room_1) ⇒ false; (nullopt, room_1) ⇒ false; (room_3, nullopt) ⇒ true.
- [ ] **Step 2:** Run, expect FAIL.
- [ ] **Step 3:** Implement; in `step_promotion` take the candidate room from `robot_rt_room(*G)` instead of the P(inside) loop; keep the "[promotion] … P(inside)" line as a log of `promotion_evidence` only.
- [ ] **Step 4:** Run the new test + `seam_evict_test room_chain_test room_swap_test room_recognizer_test passage_identity_test` (all must still pass). Build ltsm_agent. Commit.

### Task 4: door_concept — no entry mirror into an ANNOUNCED room; restore its doors (C1.4)

**Files:**
- Modify: `door_concept/src/specificworker.cpp` (`step_entry_mirrors` ~3220, `release_room` ~3373, `enter_room_memory` ~3533), `door_concept/src/door_identity_table.h` (`plan_room_entry`)
- Test: `door_concept/tests/test_door_room_swap.cpp` (extend)

**Interfaces:**
- Consumes: ltsm's `current` edge (unchanged trigger — it is now written only after room_concept's switch, Task 3) and the `recall_<room>` names door_concept already latches (`latch_recall_names`).
- Produces: entering a room whose name was announced by a recall ⇒ `plan_room_entry` returns the restore plan INCLUDING the crossed door's face (matched by place to the remembered face of the passage), and `step_entry_mirrors` makes no mirror for a room that is not `proto` or whose name was announced.

- [ ] **Step 1:** Tests: `announced_room_restores_crossed_face` (plan includes the crossed door's remembered face; no mirror requested); `fresh_room_still_mirrors` (unannounced proto room ⇒ mirror as today).
- [ ] **Step 2:** Run (`g++ -std=c++23 -O1 -I/usr/include/eigen3 tests/test_door_room_swap.cpp -o /tmp/t && /tmp/t`), expect FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run all door_concept tests touched by `plan_room_entry` (`test_door_room_swap`, `test_door_identity_table`); build door_concept. Commit.

### Task 5: ltsm — recognition PROPOSES (C2.6, C2.9)

**Files:**
- Create: `ltsm_agent/src/room_proposal.h/.cpp`
- Modify: `ltsm_agent/src/room_swap.cpp` (`swap` line 120: drop `post_late` and the `ev_.evict(discard)` of P; `undo` line 321; `on_room_left` 354), `ltsm_agent/src/recognition_cycle.h`, `ltsm_agent/src/room_recognizer.cpp` (rebase the filtered evidence on `rejected`)
- Test: `ltsm_agent/tools/room_proposal_test.cpp` + target

**Interfaces:**
- Produces (live graph, owned by ltsm): node `proposal_<K live name>` (e.g. `proposal_room_2`), type `metaconcept`, `object_subtype="room_proposal"`, `parent`=P id, RT edge P→proposal holding T(P←K), `room_id`=K's room_id, `delimiting_polygon_x/y`=K's outline, `recall_pose_prior`=packed robot-in-K modes (`pack_pose_prior`, as `post_late` uses), `state="pending"`.
- Produces: `class ltsm::RoomProposal { std::uint64_t post(std::uint64_t P, const std::string& K_mem, const Mat::RTMat& P_from_K, const std::vector<float>& prior); enum class Verdict { Pending, Accepted, Rejected, Gone }; Verdict verdict() const; void withdraw(); }`.
- Memory-side staging (`ev_.hold`, case-B twin `staged`, early carry-over) is unchanged; it is committed on `Accepted`, undone on `Rejected`/`Gone`.

- [ ] **Step 1:** Tests: `post_writes_pending_child_of_P`; `reproposal_replaces_pending` (Review Focus 2); `accepted_commits_staging` (verdict Accepted ⇒ carry-over present, hold released on leaving); `rejected_unstages_and_rebases` (no memory change survives; recogniser base reset); `left_before_verdict_withdraws` (Review Focus 4).
- [ ] **Step 2:** Run, expect FAIL.
- [ ] **Step 3:** Implement; remove the live deletion of P and the late recall from the swap path (rule 1). `recall_.step()` then posts `recall_room_1` under DOOR_1's room_2 face by itself once door_concept restores it (C2.9).
- [ ] **Step 4:** Run new + all five existing ltsm tests; build. Commit.

### Task 6: room_concept — adopt or reject a proposal (C2.7)

**Files:**
- Modify: `room_concept/src/room_scene_graph_late_recall.cpp` (`step_late_recall` 155, `find_late_recall` 25, `adopt_late_recall` 103 → read the proposal instead of a late recall), `room_concept/src/room_concept_reloc.cpp:262` (verdict), `room_concept/src/room_scene_graph.cpp` (frame change of own children)
- Test: `room_concept/tools/proposal_adoption_test.cpp` + target in `room_concept/src/CMakeLists.txt`

**Interfaces:**
- Consumes: the proposal node (Task 5); cortex rename (Task 1).
- Produces: on accept — P renamed to K's live name with `G->update_node` (same id), `room_id`=K's, `delimiting_polygon_x/y`=K's outline, P's own RT children (floor, robot) re-expressed by T(K←P), proto self-edge removed, proposal `state="accepted"`. On reject — proposal `state="rejected"`, nothing else changes. Decision = the existing late-adoption fit (`seed_reloc_prior` + `late_recall_verdict`), unchanged.

- [ ] **Step 1:** Tests: `accept_renames_in_place` (id kept, name/room_id/polygon/frame updated, proto edge gone); `reject_marks_only`; `proposal_on_non_proto_rejected` (RF1); `rename_collision_rejected` (RF3).
- [ ] **Step 2:** Run, expect FAIL.
- [ ] **Step 3:** Implement. Investigate the try32 rejection ("no pose explains this view", median |SDF| 0.16 m, `etc` logs in `crossing/recog_try32/logs/room_concept.log:731`) and record the cause in the ledger; fix only if it is in this code.
- [ ] **Step 4:** Run; build room_concept. Commit.

### Task 7: room_concept — walls move in place, never deleted under a door (rule 1)

**Files:**
- Modify: `room_concept/src/room_scene_graph.cpp` (`dsr_create_wall_nodes` 3805, `re_parent_walls_onto_floor` 3766)
- Test: `room_concept/tools/wall_update_test.cpp` + target

**Interfaces:**
- Produces: `void RoomSceneGraph::sync_walls_to_polygon(const std::vector<Eigen::Vector2f>& poly)` called whenever the published polygon changes (and on adoption, Task 6): edge i ↔ wall i by nearest edge; matched walls get a new RT (same node); unmatched walls with NO foreign children are deleted; unmatched walls WITH children are moved to their nearest edge; new edges get new walls.

- [ ] **Step 1:** Tests: `walls_follow_polygon` (RT of each wall = its edge midpoint/yaw); `wall_with_child_survives_shrink` (RF5); `childless_extra_wall_deleted`.
- [ ] **Step 2:** Run, expect FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run; build. Commit.

### Task 8: door_concept — follow an adoption (C2.8)

**Files:**
- Modify: `door_concept/src/specificworker.cpp` (`step_room_following` 3159, `enter_room_memory` 3533, `latch_recall_names` 3517)
- Test: `door_concept/tests/test_door_room_swap.cpp` (extend)

**Interfaces:**
- Consumes: proposal `state="accepted"` on `proposal_<K>` (Task 5/6) and the room's rename.
- Produces: on an accepted proposal for the current room — `plan_room_entry` takes the "recalled" branch for K, restores K's remembered doors, merges P's live doors into them BY PLACE (the crossed DOOR_2 face stays as a new face). Replaces the trigger on `recall_late` names.

- [ ] **Step 1:** Tests: `accepted_proposal_restores_and_merges`; `rejected_proposal_keeps_fresh`.
- [ ] **Step 2–4:** fail → implement → pass; build. Commit.

### Task 9: controller — read only protocol facts

**Files:**
- Modify: `controller/src/specificworker_mission.cpp`
- Test: `controller/tests/test_mission_door_relation.cpp`

**Interfaces:**
- Keep (they read protocol facts): `cross new` waits for `recall_<previous room>` (11936d55 — ltsm's C1.5/C2.5 fact); a relation's door deleted by door_concept while the passage is still on the near side re-resolves (71edf852 — door_concept's deletion fact).
- Remove: any remaining rule that infers order from timing (audit the 10-29 commits ac3581f0, 0d31c13f, 71edf852, 11936d55, 13be4e94; list in the commit message what was kept and why).

- [ ] **Step 1:** Audit; adjust tests; run `test_mission_door_relation`; build controller. Commit.

### Task 10: live verification + map check

**Files:**
- Create: `ltsm_agent/tools/ltsm_map_check.py` (from the 10-29 ad-hoc checks)

**Interfaces:**
- Produces: `python3 -I ltsm_map_check.py <ltsm_memory.json>` prints, per door face, its distance to the nearest wall of its room, and per passage loop the disagreement of the two paths; exit 1 if any face > 0.15 m from a wall or any loop disagrees by > 0.3 m / 10° (FLAGGED report limits, not runtime gates).

- [ ] **Step 1:** Write it; run on `crossing/recog_try32/ltsm_snapshots/ltsm_memory_final.json` — expect exit 1 (room_3 faces 1.14 m / 90°).
- [ ] **Step 2:** Rebuild all four agents, verify binaries (Global Constraints), run `recognition_tour.sh recog_proto1 240` from `room_concept/estimation_scripts` with `WT=…/rt-main WT_AGENTS="controller door_concept table_concept room_concept residual_concept ltsm_agent"`.
- [ ] **Step 3:** Expected: 6/6 legs, report C-checks pass, `ltsm_map_check.py crossing/recog_proto1/ltsm_memory.json` exit 0. Record in the ledger; commit the script.
