# Task 2 report: closed term library

Status: DONE. Branch feat/perception-cards.

## Functions moved (from -> to, all in common/perception_card/terms.h, namespace rc::card)
- term_envelope (both overloads): wraps rc::detect::p_detect; door calls term_envelope(roi_fill, 1, env) (door_existence.cpp p_env).
- term_range_conf(range, ref_m, power): cabinet_existence.cpp absence_range_conf body; the lambda stays as a config binding that calls it.
- term_leaf_obliquity(rx, ry, nx, ny): door_view_obliquity arithmetic (door keeps the ai2_initialized / no-pose guards).
- term_resolvability(n_cells, n_detectable): DoorSilhouette::resolvability body; the member now calls it.
- term_kappa(var_b, range, half_diag): closed form from pose_overlap_factor; the covariance -> var_b/range measurement and spread_m stay in door.
- term_in_fov / term_central: IDENTITY on the fraction. Both agents multiply the silhouette fractions directly (door sil.in_fov_frac()/central_frac(), cabinet likewise) with no formula; the fraction is the agent's measurement. Agents now wrap them in the term calls.
- term_coverage: pass-through. term_motion: not added (Task 3).
- eval_term(TermSpec, TermInputs) + TermInputs (fill_max, fill_min, visible, in_fov, central, range, phi, obliquity, plus resolvability and kappa, see decisions).
Door P(detect) product order unchanged: ((((p_env*oblq)*res)*in_fov)*central)*kappa. Cabinet: range_conf*in_fov*central.

## Goldens
Harness (outside the repo): /tmp/claude-1000/-home-pbustos-robocomp-components-active-inference-chat-agent/a2f2272b-f33f-4143-8105-0ec4edc68e6a/scratchpad/terms_golden/golden.cpp, linked against the CURRENT door_existence.cpp (real door_view_obliquity, pose_overlap_factor, DoorSilhouette members) and rc::detect::p_detect, plus cabinet's lambda body verbatim (cabinet_existence.cpp is not linkable standalone). 20 tuples per term (envelope 4-arg, envelope 3-arg, range_conf, leaf_obliquity, resolvability, in_fov, central, kappa), printed with LC_ALL=C (under es_ES printf wrote commas) and pasted as constexpr tables into common/perception_card/tests/terms_test.cpp. Edge cases included: invalid envelopes, ref/power <= 0, range 0, coincident camera/leaf, grazing, n_detectable 0, var_b tiny.
Step 2 verified: before terms.h existed the test failed to compile (no such header).

## Tests
- common/perception_card: cmake -S . -B build && make -C build terms_test card_test; LANG=es_ES.UTF-8 build/terms_test -> "terms_test: ok"; card_test -> ALL OK.
- door_concept: cbuild -j8 OK; cabinet_concept: cbuild -j8 OK (re-run after the last header change).
- Door tests that include door_existence (8, built with the replay source list, run under es_ES): core_fit, open_leaf, body_wire, core_tracker, core_existence, ricoh_lifecycle, ricoh_look, existence_moving_phantom: all PASS.
  First run: test_door_core_tracker failed "core include scan ... hits == 0" because card.h pulls media_transport/flat_json.h into the framework-free core; baseline (stash) passed. Fixed by the split below; passes now.

## Replay comparison
Tape: /home/pbustos/robocomp/components/active_inference/door_concept/etc/door_inputs.bin (+ door_decisions.bin), 1.1 GB. door_replay built before (clean tree) and after, run with --csv DIR --trace '*' under es_ES. stdout identical except the "cost:" timing line; ai2_log.csv, ai2_log_phi.csv and door_existence_log.csv (contains p_detect, oblq, p_env, resolvability, central_frac per row) are byte-identical (cmp). (The replay's own verdict vs the live log, TRACKER/PHI/EXIST DIFFERENT, is the same before and after: the tape predates fixes; not our concern.) Cabinet has no replay tool; covered by identical-arithmetic goldens + build.

## Decisions / self-review
- TermSpec moved from card.h to new term_spec.h (card.h includes it), terms.h includes term_spec.h and detectability.h, NOT card.h. Reason above (core include scan). Deviates from "terms.h includes card.h"; TermSpec/kTermNames API unchanged for card users.
- term_leaf_obliquity takes plan-view (rx, ry, nx, ny) with plain floats, reproducing Eigen's norm/normalize/dot op order; the replay is byte-identical so it is bit-exact on the tape.
- term_kappa takes (var_b, range, half_diag) in double; guards (cov present, var_b > 0) stay in the agent.
- TermInputs gets extra fields resolvability and kappa because those two are measurements the evaluator cannot derive from the listed fields. eval_term: leaf_obliquity = clamp(|obliquity|); motion and unknown names return 1 (neutral) until Task 3; absent envelope params take DetectorEnvelope defaults, absent range_conf params disable it.
- Cabinet golden used a verbatim copy of the lambda body rather than a prior refactor to a free function (the agent file cannot be linked standalone; the refactor would add churn for the same arithmetic).
- No behaviour changed; no thresholds added.
