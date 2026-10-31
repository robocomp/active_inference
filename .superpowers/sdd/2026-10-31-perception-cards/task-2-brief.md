### Task 2: The closed term library, by MOVING the agents' private factors

**Files:**
- Create: `common/perception_card/terms.h`
- Modify:
  - `door_concept/src/door_existence.cpp:1260-1279`, so its factors call the moved functions;
  - `cabinet_concept/src/cabinet_existence.cpp:62-67,128`, for `absence_range_conf`.
- Test: `common/perception_card/tests/terms_test.cpp`

**Interfaces (produces):**
- **Consumes:** `kTermNames` from `card.h` (Task 1).
- One `inline float term_<name>(...)` per name (`term_coverage` = the visible fraction passed through;
  `term_motion` is added in Task 3). Each body is MOVED VERBATIM from the agent that owns it today,
  and that agent then calls it:
  - `term_envelope` wraps `rc::detect::p_detect` (both overloads);
  - `term_range_conf(range, ref_m, power)` is cabinet's `absence_range_conf`, `min(1, (ref/range)^power)`;
  - `term_leaf_obliquity`, `term_resolvability` and `term_kappa` come from door, with the signatures door uses
    today;
  - `term_in_fov` and `term_central` take the silhouette fractions the agents pass today.
- `float eval_term(const TermSpec&, const TermInputs&)` dispatches by name. `TermInputs` holds the intermediate
  quantities: `fill_max`, `fill_min`, `visible`, `in_fov`, `central`, `range`, `phi`, `obliquity`.

Steps:
- [ ] **Step 1: write the golden test.** Before moving anything, record each agent factor's output on 20 fixed
  input tuples per term, printed by a scratch harness that calls the CURRENT agent code. Paste them as
  constexpr tables into `terms_test.cpp`. The test asserts each `term_*` reproduces its table to 1e-6.
- [ ] **Step 2: run it and see it fail** (`terms.h` doesn't exist yet).
- [ ] **Step 3: move the functions into `terms.h`** and make door and cabinet call them. Leave no copies behind.
- [ ] **Step 4: run the test; it must pass.**
  - Build door_concept and cabinet_concept with `cbuild -j8`.
  - Run door's existing tests (`door_concept/tests/`) and the `door_replay` tool on a recorded tape
    (`door_concept/etc/door_inputs*.bin`, if any exists). Decisions must be byte-identical to before the move.
- [ ] **Step 5: commit** `perception_card: closed term library; door + cabinet factors moved, identical by construction`.

