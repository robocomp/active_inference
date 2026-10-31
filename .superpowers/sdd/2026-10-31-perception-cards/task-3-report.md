# Task 3 report: camera evaluator

Status: DONE. Files: common/perception_card/{evaluator.h, terms.h, CMakeLists.txt, tests/evaluator_camera_test.cpp, tests/terms_test.cpp}.
Tests (run directly): card_test ALL OK, terms_test ok, evaluator_camera_test ok (a-f plus articulated edge-on and not-modelled).

## Carried from Task 2
- term_motion(sigma0, k, seconds_unseen) = sqrt(sigma0^2 + k*max(0,t)). eval_term "motion" returns the SPREAD (m), not a factor
  (sigma0 and k from the term's params; new TermInputs.seconds_unseen). The evaluator marginalises over it; it is never multiplied in.
- eval_term on a name outside kTermNames: hard std::abort with a message (stderr). Chosen over std::expected because it keeps the
  float signature (no caller churn; card_from_json already rejects bad names, so reaching it is a bug). terms_test: loops over all
  kTermNames (an unhandled one would abort the test) and forks a child to check an unknown name dies with SIGABRT.

## Decisions / ambiguities
- Limit = leave-one-out: the factor whose removal raises P the most (envelope, FOV via a 3 rad "wide" sensor, walls, objects, leaf_obliquity,
  other card terms). Sequential peeling was tried first: it labelled a target behind a wall "TooSmall". Infinite fill: TooClose if the
  centre is within the hfov, else OutOfView.
- ONE reporting threshold, flagged: limit is None when p >= 0.5 (or no single fix helps). P is never gated. Rationale: Limit explains a miss.
- Test envelope = the refrigerator agent's measured {0, 1.32, 0.26} (min_fill 0.10 gave p=0.445 at 3 m, contradicting test (a)).
- ChannelResult gained `notes` (brief's struct had none; needed for "walls unknown"). Result.notes aggregates in Task 4.
- Wall thickness 0.12 m (detail::kWallThicknessM). Walls carry no height (2D block). Mass level is treated like countable.
- Animate: 5x5 Gauss-Hermite on (x,y) using rc::nbv::detail::kGHx/kGHw; sigma0 is the motion term's "sigma0" (default 0). Heading fixed
  at the nominal centre. A missing motion term gives a note, no spread.
- Articulated: leaf normal = rotate local y by phi, obliquity vs the centre ray. phi nullopt -> closed + note. resolvability and kappa stay
  at neutral 1 (they need a measurement). range_conf applies to any level that lists it.
- Hypothetical Target has no sigma, so predicted_fill_sigma returns 0 (called for fidelity). Viewpoint inside a named obstacle: Inside, p=0.
- Channel "zed" absent / not modelled / no world.zed: NotModelled, p=0.
- evaluator.h includes media_transport.h for SensorModel (no DSR), but it pulls FastDDS IDL types, so the test links fastdds+fastcdr.
  Task 4 consumers of evaluator.h need that too (card.h alone does not).

## Fix round 1
- Mass level: p = visible_fraction x non-motion card terms, no envelope (also in the limit attribution). Test (iii).
- Limit: the p>=0.5 cut and the 1e-6 cutoff are gone. limit = argmax gain, `limit_gain` = P with that factor removed minus P; None only when all gains are 0.
  Occluder leave-one-out now runs whenever objects cost visibility, independent of the label. New Limit::Range for the other card terms.
- SensorModel moved UNCHANGED (no DDS-typed members) to common/media_transport/sensor_model.h; media_transport.h includes it. evaluator.h includes only that,
  and evaluator_camera_test no longer links fastdds/fastcdr. sensor_model_test and descriptor_roundtrip rebuilt (fresh build dir) and pass.
- NaN safety: non-finite p -> 0; standing on the target (no obstacle) -> p 0, TooClose. Inside gets limit_gain 1.
- Tests added: (i) two obstacles -> wide_box named; (ii) Inside p==0; (iii) mass; (iv) partly occluded (p 0.514, clear 0.561) names "post"
  (p is ~0.5, not 0.6: the envelope is the agent's real one); (v) "phi unknown" note; NaN case.
- (a): the envelope {0, 1.32, 0.26} IS the refrigerator agent's real config (refrigerator_concept/etc/config.toml:166-168). The test now asserts the
  computed p=0.570 (lower shoulder sig(~0.14/0.26)=0.63 still active because MinFill=0 is 'inactive', not a step) and limit TooSmall.
- Minor: sigma guard commented, <limits> removed.
- Run: LANG=es_ES.UTF-8 ./card_test (ALL OK), ./terms_test (ok), ./evaluator_camera_test (ok); media_transport: sensor_model_test ALL OK, descriptor_roundtrip ALL OK.
