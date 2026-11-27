# Blind referee packet: unmatched rows of the after-count

Source: active_inference commit cf65a471, tree hash `5c0029ab522691fb4747d2e7215493162f13f0b0` (branch feat/shared-engine, the end of K3).
Rows are the after-count rows whose key, reader or role is new relative to the labelled baseline. No label, no proposed label. The role column quotes the code's own comment or the loader text, not a reading.

## Definition (confirmed by the owner, applied literally)

A PATCH constant is any constant that is neither class knowledge nor a property of a sensor. That means any precision, gate, anchor, threshold, gain, margin, floor, clamp or reference/scale whose role is to correct the estimator's behaviour.

- CLASS knowledge: a prior on the class's size, shape, placement (e.g. stands against a wall), appearance, or persistence/permanence.
- SENSOR: a property of a sensor or its mount: noise σ, range, field of view, depth-noise model, detection probability of the detector, intrinsics.
- Owner's examples of PATCH (nightstand config): AI2DepthUnobsPrecision, AI2TopNoFloatPrecision / AI2TopNoFloatMargin, AI2TopOversegSigmaPerM, FeRef / FeScale, PlausHeightPriorGain.

## Exclusion rule

NOT counted at all: plumbing and engineering parameters with no estimation role. Examples: agent id, DDS domain, node names, file paths, log/dump/diagnostic switches (GnTrace, *Log*, *Dump*), period/Hz of the compute loop, thread counts, display/visualisation-only settings, GN iteration caps and numerical-solver tolerances. They are listed in each section with their names (label `excluded`) so a referee can audit them.


## Labels

`sensor` `class` `patch` `excluded`; `switch (patch)` = boolean that enables a patch mechanism; `switch (other)` = boolean with no patch mechanism behind it (publication, channel choice, history stage). A trailing `?` = ambiguous (counted in the "ambiguous" column only, never twice).

## Referee's rules (header of referee_labels_2026-11-27.tsv, verbatim)

```
# Rules applied to the hard cases
# 0. A row is labelled by the meaning of its key when the mechanism is on, not by its present value. A value of 0 that
#    only disables the mechanism does not change the label. A constant that no code reads at all is 'excluded' (dead).
# 1. Process noise. Class only where the code states an object fact with a value that follows from it (extent Q = 0:
#    'dimensions are constants of the object'). The per-frame pose Q (0.005 m, 0.01 rad, identical in every class) is
#    patch: the code gives it a filter role (tracks localisation jitter; rate at which evidence is discarded).
# 2. Sensor noise. Sensor when the code declares it as the measurement noise of a channel and gives no other purpose.
#    Patch when the code says it is enlarged on purpose, covers a model or pose error, or is a floor.
# 3. Sensor precision used as a weight. Patch when the device sigma exists as a separate constant and the precision is
#    combined with it or matched to another term (LidarPrecision). Sensor only when it is the device's 1/sigma^2 itself.
# 4. Prior stds. Class when they parameterise a prior over size, shape, placement, appearance or persistence in
#    physical units. Patch when the code says the prior only breaks a degeneracy or acts as a covariance ceiling,
#    or when a class fact is multiplied by a free gain (owner's example PlausHeightPriorGain).
# 5. Arbitration, planner, calibration. Gains, thresholds, margins, hysteresis and targets that shape decisions are
#    patch. Excluded: log-only scales, sample counts, numeric guards, write dedup, solver schedules, robot-body sizes.
# 6. Floors and clamps on an estimated state are patch even where the code calls them physical floors. Guards on pure
#    numerics (projection singularity, exact-zero cut-offs, float-precision culls) are excluded.
# 7. A constant in a class-size or class-shape slot is class even when its value was copied from another class
#    (low confidence).
# 8. Room-model and lexical knowledge (wall sigma, floor sigma, colour vocabulary) is neither class nor sensor; the
#    first sentence of the definition makes it patch (low confidence).
```

## Rows to label

Columns: id | key or constant | value | location (file:line at cf65a471) | role (quoted from the code)


### bed_concept/etc/config.toml

| A1901 | Model.AI2PriorDepthStd | -1.0f | NO toml key; code default at countable_concept/src/countable_config.cpp:141 | "loader: depth (local Y) prior std (m) when the class declares its own ([prior.depth] std_m); <=0 = the footprint std" |
| A1800 | Model.AI2CommonModePosStd | 0.03 | bed_concept/etc/config.toml:199 [Model]; read countable_concept/src/countable_config.cpp:304 | "Per-frame SHARED-ERROR common-mode stds — the information-saturation FLOORS (CLAUDE.md "encode the effect as a continuous covariance"): N correlated points from one frame cannot collapse Σ below these. LOAD-BEARING. They were previously read from these getters with the keys ABSENT from config (silent defaults) — now ex…" |
| A1801 | Model.AI2CommonModeSizeStd | 0.02 | bed_concept/etc/config.toml:200 [Model]; read countable_concept/src/countable_config.cpp:306 | "shared size (w,h,H) error floor (m) \| loader: shared size error w,h,H (m)" |
| A1802 | Model.AI2CommonModeYawStd | 0.03 | bed_concept/etc/config.toml:201 [Model]; read countable_concept/src/countable_config.cpp:308 | "shared yaw error floor (rad) \| loader: shared yaw error (rad)" |
| A1905 | Model.AI2RangeNoiseYawPerM | 0.03f | NO toml key; code default at countable_concept/src/countable_config.cpp:334 | "loader: yaw common-mode std growth (rad per m of range); FLAG-OFF path only (shared-engine Task 5)" |
| A1906 | Model.AI2RangeNoiseSizePerM | 0.08f | NO toml key; code default at countable_concept/src/countable_config.cpp:336 | "loader: SIZE (w,h,H) common-mode std growth (m per m of range); FLAG-OFF path only (shared-engine Task 5)" |
| A1908 | Concept.MeshPath | "" | NO toml key; code default at countable_concept/src/countable_config.cpp:409 | "loader: display mesh (relative to the meshes root); empty = <cls>_concept/meshes/<cls>.obj (the class-folder convention)" |
| A1909 | Concept.LookAttrs | "table" | NO toml key; code default at countable_concept/src/countable_config.cpp:411 | "loader: typed attribute family of the controller's look contract: table (table_roi_*, table_detection_*) or chair (chair_roi_*, chair_detection_*, chair_front_seen*)" |
| A1910 | Concept.AppearanceEnabled | false | NO toml key; code default at countable_concept/src/countable_config.cpp:417 | "loader: colour belief from the assigned ZED masks → mesh_color_rgb, color_name, color_prob (display + names; never the fit)" |
| A1911 | Concept.PublishExistProb | false | NO toml key; code default at countable_concept/src/countable_config.cpp:419 | "loader: publish obj_exist_prob = P(exists) = occupancy x class" |
| A1912 | Concept.ThumbnailsEnabled | false | NO toml key; code default at countable_concept/src/countable_config.cpp:420 | "loader: keep the best retina crop per viewing angle and publish them on the <class>_thumbs node (ltom)" |
| A1913 | Concept.ApproachEnabled | false | NO toml key; code default at countable_concept/src/countable_config.cpp:422 | "loader: offer the pragmatic approach affordance aff_<name>_approach" |
| A1914 | Concept.EpistemicOfferEnabled | true | NO toml key; code default at countable_concept/src/countable_config.cpp:423 | "loader: false = no epistemic viewpoint offer (AuRo C3(b) comparison arm); the belief still updates from what is seen" |
| A1915 | Concept.ChangedObjectEnabled | false | NO toml key; code default at countable_concept/src/countable_config.cpp:425 | "loader: changed object: ghost existence at remembered spots + the same_as identity link on objects born new" |
| A1916 | Tracker.LinkPiMove | 0.10f | NO toml key; code default at countable_concept/src/countable_config.cpp:427 | "loader: identity link: per-visit P(a remembered object was moved) (a GUESS)" |
| A1917 | Tracker.LinkPiGone | 0.10f | NO toml key; code default at countable_concept/src/countable_config.cpp:428 | "loader: identity link: per-visit P(a remembered object left the room) (a GUESS)" |
| A1918 | Tracker.LinkLambdaNew | 0.10f | NO toml key; code default at countable_concept/src/countable_config.cpp:429 | "loader: identity link: per-visit rate of a NEW object arriving (a GUESS)" |
| A1919 | Tracker.GhostEvidenceGain | 0.15f | NO toml key; code default at countable_concept/src/countable_config.cpp:430 | "loader: changed object: \|dL\| per look at a remembered spot's ghost (the legacy chair's Existence.EvidenceGain, ported unchanged)" |
| A1754 | Model.ReplayDump | false | bed_concept/etc/config.toml:78 [Model]; read countable_concept/src/countable_config.cpp:537 | "diagnostic only (shared-engine plan rev 2, Task R1): true writes each persistent belief's seed + every update's complete input to etc/replay/<node>_<run>_<serial>.bin (~0.6 MB per bed update) for countable_concept/tests/build/replay \| loader: DIAGNOSTIC only, no model effect: with LidarMeasuresOnGated write each persis…" |

### chair_concept/etc/countable.toml

| A0188 | Model.AI2PriorDepthStd | 0.026 | chair_concept/etc/countable.toml:102 [Model]; read countable_concept/src/countable_config.cpp:141 | "K2b chair (fix 1): (0.51 − 0.42)/√12, the same uniform (the depth's own std, K1 key) \| loader: depth (local Y) prior std (m) when the class declares its own ([prior.depth] std_m); <=0 = the footprint std" |
| A0223 | Model.AI2CommonModePosStd | 0.03 | chair_concept/etc/countable.toml:210 [Model]; read countable_concept/src/countable_config.cpp:304 | "Per-frame SHARED-ERROR common-mode stds — the information-saturation FLOORS (CLAUDE.md "encode the effect as a continuous covariance"): N correlated points from one frame cannot collapse Σ below these. LOAD-BEARING. They were previously read from these getters with the keys ABSENT from config (silent defaults) — now ex…" |
| A0224 | Model.AI2CommonModeSizeStd | 0.02 | chair_concept/etc/countable.toml:211 [Model]; read countable_concept/src/countable_config.cpp:306 | "shared size (w,h,H) error floor (m) \| loader: shared size error w,h,H (m)" |
| A0225 | Model.AI2CommonModeYawStd | 0.03 | chair_concept/etc/countable.toml:212 [Model]; read countable_concept/src/countable_config.cpp:308 | "shared yaw error floor (rad) \| loader: shared yaw error (rad)" |
| A0339 | Model.AI2RangeNoiseYawPerM | 0.03f | NO toml key; code default at countable_concept/src/countable_config.cpp:334 | "loader: yaw common-mode std growth (rad per m of range); FLAG-OFF path only (shared-engine Task 5)" |
| A0340 | Model.AI2RangeNoiseSizePerM | 0.08f | NO toml key; code default at countable_concept/src/countable_config.cpp:336 | "loader: SIZE (w,h,H) common-mode std growth (m per m of range); FLAG-OFF path only (shared-engine Task 5)" |
| A0281 | Concept.MeshPath | "chair_concept/meshes/chair_back_plus_y.obj" | chair_concept/etc/countable.toml:378 [Concept]; read countable_concept/src/countable_config.cpp:409 | "K2b chair: chair.obj turned 180° (backrest at +y, the countable convention) \| loader: display mesh (relative to the meshes root); empty = <cls>_concept/meshes/<cls>.obj (the class-folder convention)" |
| A0274 | Concept.LookAttrs | "chair" | chair_concept/etc/countable.toml:371 [Concept]; read countable_concept/src/countable_config.cpp:411 | "── K2b: the graph consumers' channels the legacy chair agent published (spike: .superpowers/sdd/2026-11-23-shared-engine/task-K2b-report.md) ── K2b chair: chair_roi_* / chair_detection_* / chair_front_seen(_frame) — the controller's chair look contract (affordance_protocol.h) and its look counter (controller_session.cp…" |
| A0275 | Concept.AppearanceEnabled | true | chair_concept/etc/countable.toml:372 [Concept]; read countable_concept/src/countable_config.cpp:417 | "K2b chair: colour belief → mesh_color_rgb (viewer tint), color_name / color_prob (language_agent, ltom, the identity link) \| loader: colour belief from the assigned ZED masks → mesh_color_rgb, color_name, color_prob (display + names; never the fit)" |
| A0276 | Concept.PublishExistProb | true | chair_concept/etc/countable.toml:373 [Concept]; read countable_concept/src/countable_config.cpp:419 | "K2b chair: obj_exist_prob = P(exists) — ltom's memory weight + recall prior, language_agent, controller missions \| loader: publish obj_exist_prob = P(exists) = occupancy x class" |
| A0277 | Concept.ThumbnailsEnabled | true | chair_concept/etc/countable.toml:374 [Concept]; read countable_concept/src/countable_config.cpp:420 | "K2b chair: chair_thumbs (best crop per viewing angle) for ltom \| loader: keep the best retina crop per viewing angle and publish them on the <class>_thumbs node (ltom)" |
| A0278 | Concept.ApproachEnabled | true | chair_concept/etc/countable.toml:375 [Concept]; read countable_concept/src/countable_config.cpp:422 | "K2b chair: aff_chair_N_approach — language_agent "approach the chair", the controller's pragmatic claim \| loader: offer the pragmatic approach affordance aff_<name>_approach" |
| A0279 | Concept.EpistemicOfferEnabled | true | chair_concept/etc/countable.toml:376 [Concept]; read countable_concept/src/countable_config.cpp:423 | "K2b chair: false = no viewpoint offer (the AuRo C3(b) comparison arm, as ChairConcept.EpistemicOfferEnabled was) \| loader: false = no epistemic viewpoint offer (AuRo C3(b) comparison arm); the belief still updates from what is seen" |
| A0280 | Concept.ChangedObjectEnabled | true | chair_concept/etc/countable.toml:377 [Concept]; read countable_concept/src/countable_config.cpp:425 | "K2b chair: ghosts at remembered spots + same_as_uid / same_as_prob / left_spot_prob (ltom joins the histories) \| loader: changed object: ghost existence at remembered spots + the same_as identity link on objects born new" |
| A0310 | Tracker.LinkPiMove | 0.10 | chair_concept/etc/countable.toml:459 [Tracker]; read countable_concept/src/countable_config.cpp:427 | "K2b chair: identity link: per-visit P(a remembered chair was moved). A GUESS (the legacy chair's value) \| loader: identity link: per-visit P(a remembered object was moved) (a GUESS)" |
| A0311 | Tracker.LinkPiGone | 0.10 | chair_concept/etc/countable.toml:460 [Tracker]; read countable_concept/src/countable_config.cpp:428 | "K2b chair: identity link: per-visit P(a remembered chair left the room). A GUESS (the legacy chair's value) \| loader: identity link: per-visit P(a remembered object left the room) (a GUESS)" |
| A0312 | Tracker.LinkLambdaNew | 0.10 | chair_concept/etc/countable.toml:461 [Tracker]; read countable_concept/src/countable_config.cpp:429 | "K2b chair: identity link: per-visit rate of a NEW chair arriving. A GUESS (the legacy chair's value) \| loader: identity link: per-visit rate of a NEW object arriving (a GUESS)" |
| A0313 | Tracker.GhostEvidenceGain | 0.15 | chair_concept/etc/countable.toml:462 [Tracker]; read countable_concept/src/countable_config.cpp:430 | "K2b chair: \|ΔL\| per look at a ghost's spot (the legacy chair's Existence.EvidenceGain, ported unchanged) \| loader: changed object: \|dL\| per look at a remembered spot's ghost (the legacy chair's Existence.EvidenceGain, ported unchanged)" |
| A0176 | Model.ReplayDump | false | chair_concept/etc/countable.toml:88 [Model]; read countable_concept/src/countable_config.cpp:537 | "diagnostic only (shared-engine plan rev 2, Task R1): true writes each persistent belief's seed + every update's complete input to etc/replay/<node>_<run>_<serial>.bin (~0.6 MB per chair update) for countable_concept/tests/build/replay \| loader: DIAGNOSTIC only, no model effect: with LidarMeasuresOnGated write each pers…" |

### common/ai_belief/beam_channel.h

| A2424 | i | 3 | common/ai_belief/beam_channel.h:72 | (no comment in the code) |
| A2425 | literal | 0.99, 5.0 | common/ai_belief/beam_channel.h:83 | (no comment in the code) |

### common/ai_belief/beam_ray.h

| A2426 | literal | 6 | common/ai_belief/beam_ray.h:842 | (no comment in the code) |
| A2427 | literal | 6 | common/ai_belief/beam_ray.h:847 | (no comment in the code) |
| A2428 | NT | 6 | common/ai_belief/beam_ray.h:903 | "Geometric breakpoints of part p on its descending branch (0, t*_p): region boundaries outside, face changes inside." |
| A2429 | NT | 6 | common/ai_belief/beam_ray.h:962 | "The term of ray i: E_i, its gradient and curvature (the ray-set sums them unscaled; κ ≡ 1, rev 3.3.11). false ⇔ the ray contributes exactly 0 (d > 8σ_s). / force_sweep: run the several-part panel sweep (switch search) even with one part present (the one-part check)." |
| A2430 | literal | 3 | common/ai_belief/beam_ray.h:1104 | (no comment in the code) |

### common/ai_belief/box_engine_model.h

| A2431 | N | 6 | common/ai_belief/box_engine_model.h:82 | (no comment in the code) |
| A2432 | kYaw | 5 | common/ai_belief/box_engine_model.h:83 | "θ = [cx, cy, H, w, h, yaw]" |
| A2433 | kHeight | 2 | common/ai_belief/box_engine_model.h:88 | "FLOOR-STANDING box whose TOP lies above the Helios and is grazed from below-and-beside, as the refrigerator's. The hood / HANGS above the sensor: its rings strike the underside (z = H − extent), not the top, so ρ_L on H is the wrong geometry / for it. Left as is for the belief port; the hood's fitter task (10d) must gi…" |

### common/ai_belief/foreign_box.h

| A2434 | j | 6 | common/ai_belief/foreign_box.h:51 | (no comment in the code) |
| A2435 | literal | 3 | common/ai_belief/foreign_box.h:82 | "∂d/∂w" |
| A2436 | literal | 4 | common/ai_belief/foreign_box.h:83 | "∂d/∂h" |
| A2437 | literal | 5 | common/ai_belief/foreign_box.h:84 | "∂p_local/∂yaw = (p_ly, −p_lx)" |
| A2438 | i | 3 | common/ai_belief/foreign_box.h:109 | (no comment in the code) |

### common/ai_belief/offer_view.h

| A2439 | rho_max | 6.0, 0.01 | common/ai_belief/offer_view.h:74 | "the staging's ρ_max and the model σ (BeamRays fields)" |
| A2440 | range_max | 6.0 | common/ai_belief/offer_view.h:75 | "the sensor's own range: a surface beyond it returns nothing" |
| A2441 | kGrid | 8 | common/ai_belief/offer_view.h:104 | "samples per face side (quadrature resolution)" |
| A2442 | half_h | 0.1f, 3.0f | common/ai_belief/offer_view.h:113 | "visible_fraction's field-of-view half angles, clamped as it clamps them" |
| A2443 | half_v | 0.1f, 3.0f | common/ai_belief/offer_view.h:114 | (no comment in the code) |
| A2444 | a | 3 | common/ai_belief/offer_view.h:123 | (no comment in the code) |
| A2445 | literal | 3, 3 | common/ai_belief/offer_view.h:129 | (no comment in the code) |
| A2446 | i | 3 | common/ai_belief/offer_view.h:210 | (no comment in the code) |
| A2447 | half_h | 0.1f, 3.0f, 0.1f, 3.0f | common/ai_belief/offer_view.h:266 | "the camera's label: nullopt outside its field of view from the standpoint (visible_surface's frustum test)" |

### common/ai_belief/parts_geometry.h

| A2448 | panel_t_m | 0.10f | common/ai_belief/parts_geometry.h:47 | "back panel thickness (m)" |
| A2449 | literal | 3, 5, 4, 3, 5, 4 | common/ai_belief/parts_geometry.h:133 | (no comment in the code) |
| A2450 | literal | 6 | common/ai_belief/parts_geometry.h:134 | (no comment in the code) |
| A2451 | literal | 4, 5 | common/ai_belief/parts_geometry.h:151 | (no comment in the code) |
| A2452 | literal | 4, 5 | common/ai_belief/parts_geometry.h:152 | (no comment in the code) |
| A2453 | literal | 3, 5, 4, 3, 5, 4 | common/ai_belief/parts_geometry.h:153 | (no comment in the code) |
| A2454 | literal | 3, 4, 5 | common/ai_belief/parts_geometry.h:175 | (no comment in the code) |
| A2455 | literal | 3, 4, 5 | common/ai_belief/parts_geometry.h:176 | (no comment in the code) |
| A2456 | literal | 3, 5 | common/ai_belief/parts_geometry.h:178 | (no comment in the code) |
| A2457 | literal | 4, 3 | common/ai_belief/parts_geometry.h:179 | (no comment in the code) |
| A2458 | literal | 5, 4 | common/ai_belief/parts_geometry.h:180 | (no comment in the code) |
| A2459 | literal | 6 | common/ai_belief/parts_geometry.h:181 | (no comment in the code) |
| A2460 | literal | 7 | common/ai_belief/parts_geometry.h:184 | (no comment in the code) |
| A2461 | literal | 6 | common/ai_belief/parts_geometry.h:189 | (no comment in the code) |

### common/ai_belief/persistent_engine.h

| A2462 | iters_max | 25 | common/ai_belief/persistent_engine.h:95 | "GN steps (accepted + rejected) per update" |
| A2463 | tol_sigma | 0.1 | common/ai_belief/persistent_engine.h:98 | "the stop rule: undamped step < tol_sigma·σ_k on every θ component" |
| A2464 | memory | 0.995 | common/ai_belief/persistent_engine.h:116 | "λ per measured update (~200 updates): long enough to earn stiffness, short to recant" |
| A2465 | literal | 9.0 | common/ai_belief/persistent_engine.h:286 | (no comment in the code) |
| A2466 | K | 3 + 6 * N + W + 3 * NP | common/ai_belief/persistent_engine.h:379 | "Per proportion THREE latents (fix round 1, I1): the object's q (shared by the sensors, ρ = 1) and, per sensor channel, a / per-view COMMON MODE of that channel on q — re-drawn with the channel's view ρ exactly as b_cam / b_L are for θ — so a / channel measures q + b_q,c: many frames from one view cannot pin q below tha…" |
| A2467 | kWall | 3 + 6 * N | common/ai_belief/persistent_engine.h:380 | "the wall latent's index inside b (when W = 1)" |
| A2468 | kProp | 3 + 6 * N + W | common/ai_belief/persistent_engine.h:381 | "the first proportion latent's index inside b (when NP > 0)" |
| A2469 | kPropCam | kProp + NP, kPropL = kProp + 2 * NP | common/ai_belief/persistent_engine.h:382 | "the camera's / LiDAR's common mode on q" |
| A2470 | kChain | 0, kCam = 3, kL = 3 + N, kFc = 3 + 2 * N, kFL = 3 + 3 * N, kSc = 3 + 4 * N, kSL  | common/ai_belief/persistent_engine.h:384 | (no comment in the code) |
| A2471 | literal | 3 | common/ai_belief/persistent_engine.h:524 | (no comment in the code) |
| A2472 | literal | 3 | common/ai_belief/persistent_engine.h:525 | (no comment in the code) |
| A2473 | literal | 12.0f | common/ai_belief/persistent_engine.h:566 | (no comment in the code) |
| A2474 | literal | 12.0 | common/ai_belief/persistent_engine.h:583 | (no comment in the code) |
| A2475 | literal | 12.0 | common/ai_belief/persistent_engine.h:585 | (no comment in the code) |
| A2476 | literal | 4.0 | common/ai_belief/persistent_engine.h:980 | "restart scheme; re-solve on the same linearisation" |
| A2477 | literal | 3.0 | common/ai_belief/persistent_engine.h:987 | (no comment in the code) |
| A2478 | literal | 4.0 | common/ai_belief/persistent_engine.h:990 | (no comment in the code) |
| A2479 | ch | 3 | common/ai_belief/persistent_engine.h:1067 | (no comment in the code) |
| A2480 | literal | 2e-3 | common/ai_belief/persistent_engine.h:1085 | (no comment in the code) |
| A2481 | literal | 2e-4 | common/ai_belief/persistent_engine.h:1086 | (no comment in the code) |
| A2482 | channel | 3 | common/ai_belief/persistent_engine.h:1098 | (no comment in the code) |
| A2483 | literal | 2e-3 | common/ai_belief/persistent_engine.h:1101 | (no comment in the code) |
| A2484 | literal | 2e-4 | common/ai_belief/persistent_engine.h:1102 | (no comment in the code) |
| A2485 | kOfferPathSteps | 10 | common/ai_belief/persistent_engine.h:1401 | "G5: chords of the straight path to a standpoint" |
| A2580 | Σ capped at the class prior on every predict | Σ_ii ≤ Σ₀,ii | common/ai_belief/persistent_engine.h:1514 | "code: for (int i = 0; i < N; ++i)" |

### common/ai_belief/viewpoint.h

| A2486 | i | 3 | common/ai_belief/viewpoint.h:149 | (no comment in the code) |
| A2487 | literal | 36 | common/ai_belief/viewpoint.h:184 | (no comment in the code) |
| A2488 | literal | 5 | common/ai_belief/viewpoint.h:185 | (no comment in the code) |
| A2489 | i | 3 | common/ai_belief/viewpoint.h:187 | (no comment in the code) |
| A2490 | j | 3 | common/ai_belief/viewpoint.h:188 | (no comment in the code) |
| A2491 | literal | 6 | common/ai_belief/viewpoint.h:189 | (no comment in the code) |
| A2492 | literal | 5 | common/ai_belief/viewpoint.h:223 | (no comment in the code) |

### common/concept_manifest/shape_parts.h

| A2493 | literal | 4 | common/concept_manifest/shape_parts.h:172 | (no comment in the code) |

### common/standpoint/residual_risk.h

| A2583 | no belief ⇒ hard | a residual cell with no usable Beta blocks | residual_risk.h / standpoint_oracle.h:14 | "(location not a single file:line in the doc)" |
| A2494 | kMaxIt | 300 | common/standpoint/residual_risk.h:60 | (no comment in the code) |
| A2495 | kEps | 1e-14, kFpMin = 1e-300 | common/standpoint/residual_risk.h:61 | (no comment in the code) |
| A2496 | kappa | 1e6 | common/standpoint/residual_risk.h:112 | "Var = 0: a point mass, as concentrated as a double can say" |
| A2582 | residual_risk Beta floor | α, β ≥ 0.5 | common/standpoint/residual_risk.h:114 | "code: alpha = std::max(0.5, p * kappa);" |
| A2581 | residual_risk occupancy boundary | Pr(θ_c < ½) | common/standpoint/residual_risk.h:126 | "code: const double r = -log_ibeta(a, b, 0.5);" |
| A2497 | literal | 32 | common/standpoint/residual_risk.h:173 | (no comment in the code) |
| A2498 | literal | 4 | common/standpoint/residual_risk.h:182 | (no comment in the code) |

### common/standpoint/standpoint_oracle.h

| A2499 | kHeadings | 8 | common/standpoint/standpoint_oracle.h:95 | "GridPlanner's heading buckets: one per 45°" |
| A2500 | xmin | 0.1f | common/standpoint/standpoint_oracle.h:138 | (no comment in the code) |
| A2501 | xmin | 0.1f | common/standpoint/standpoint_oracle.h:154 | (no comment in the code) |
| A2502 | literal | 3 | common/standpoint/standpoint_oracle.h:179 | (no comment in the code) |
| A2503 | literal | 3 | common/standpoint/standpoint_oracle.h:192 | (no comment in the code) |
| A2504 | literal | 0.95 | common/standpoint/standpoint_oracle.h:219 | "the robot never moving again. Why kept for now: it is the controller's degraded-mode guard and K3 must not change the / controller's decisions (golden parity); the model-level replacement would be a belief over the room polygon's validity / (room_concept's own fit quality), which no consumer has yet. Note it now applie…" |
| A2505 | literal | 3 | common/standpoint/standpoint_oracle.h:224 | (no comment in the code) |
| A2506 | k | 1469598103934665603ull | common/standpoint/standpoint_oracle.h:374 | (no comment in the code) |
| A2507 | literal | 1099511628211ull | common/standpoint/standpoint_oracle.h:375 | (no comment in the code) |
| A2508 | literal | 4 | common/standpoint/standpoint_oracle.h:425 | (no comment in the code) |
| A2509 | literal | 3 | common/standpoint/standpoint_oracle.h:437 | (no comment in the code) |
| A2510 | i | 3 | common/standpoint/standpoint_oracle.h:438 | (no comment in the code) |

### countable_concept/src/countable_belief.cpp

| A2511 | i | 6 | countable_concept/src/countable_belief.cpp:294 | "q = 0 (no motion at all for that DOF) ⇒ the variance never doubles: τ = +inf, not 0 (F1 round 6)" |
| A2512 | k | 6 | countable_concept/src/countable_belief.cpp:401 | (no comment in the code) |
| A2513 | literal | 0.1f | countable_concept/src/countable_belief.cpp:416 | (no comment in the code) |
| A2514 | literal | 0.99f | countable_concept/src/countable_belief.cpp:747 | (no comment in the code) |
| A2515 | kEmIters | 10 | countable_concept/src/countable_belief.cpp:813 | "EM iterations (a solver constant)" |
| A2516 | literal | 0.95 | countable_concept/src/countable_belief.cpp:829 | (no comment in the code) |
| A2517 | literal | 86400.0 | countable_concept/src/countable_belief.cpp:919 | (no comment in the code) |
| A2518 | literal | 6 | countable_concept/src/countable_belief.cpp:929 | (no comment in the code) |
| A2519 | literal | 3 | countable_concept/src/countable_belief.cpp:941 | (no comment in the code) |
| A2520 | literal | 4 | countable_concept/src/countable_belief.cpp:942 | (no comment in the code) |
| A2521 | literal | 5 | countable_concept/src/countable_belief.cpp:946 | (no comment in the code) |
| A2522 | literal | 4, 3, 5 | countable_concept/src/countable_belief.cpp:1239 | "the persistent joint state relabels with θ: w, h of θ and of every 6-sized latent block (b_cam, b_L, u_cam, u_L, s_cam, / s_L); e and ℓ are room-frame, untouched (a no-op before the joint is seeded)" |

### countable_concept/src/countable_belief.h

| A2523 | predict_period_s | 0.1f | countable_concept/src/countable_belief.h:292 | "the agent's predict period (Model.AI2AgeNominalDtS, the mask stream's nominal period)" |
| A2579 | F1r9 cap switch: gn_tol_sigma, which now decides absorbed (Laplace Σ) or not absorbed (P̄) | 0.1 | countable_concept/src/countable_belief.h:332 | "code: double gn_tol_sigma = 0.1;" |
| A2524 | N | 6 | countable_concept/src/countable_belief.h:440 | (no comment in the code) |
| A2525 | kYaw | 5 | countable_concept/src/countable_belief.h:441 | "θ = [cx, cy, H, w, h, yaw]" |
| A2526 | kHeight | 2 | countable_concept/src/countable_belief.h:442 | "H: the DOF the LiDAR's top-grazing ring decorrelates (ρ_L height)" |
| A2527 | kWallLatents | 1 | countable_concept/src/countable_belief.h:480 | "the wall latent (F1.4): one per (room, edge), and the wall priors it carries, entered once at s = θ + B_w b" |
| A2528 | kProportions | 1 | countable_concept/src/countable_belief.h:491 | "K2a-L: the class shape proportions (persistent_engine.h kProp). ONE slot for every countable class (the engine's latent / layout is fixed at compile time); a class declaring none keeps it at Δ = 0 (exactly zero, zero variance)." |
| A2529 | kProportions | 0 | countable_concept/src/countable_belief.h:523 | (no comment in the code) |

### countable_concept/src/countable_birth_seed.h

| A2530 | literal | 10 | countable_concept/src/countable_birth_seed.h:44 | (no comment in the code) |

### countable_concept/src/countable_class_evidence.h

| A2531 | literal | 0.97 | countable_concept/src/countable_class_evidence.h:133 | (no comment in the code) |

### countable_concept/src/countable_config.cpp

| A2532 | literal | 6 | countable_concept/src/countable_config.cpp:207 | (no comment in the code) |
| A2533 | literal | 0.10f | countable_concept/src/countable_config.cpp:427 | (no comment in the code) |
| A2534 | literal | 0.10f | countable_concept/src/countable_config.cpp:428 | (no comment in the code) |
| A2535 | literal | 0.10f | countable_concept/src/countable_config.cpp:429 | (no comment in the code) |
| A2536 | literal | 0.15f | countable_concept/src/countable_config.cpp:430 | (no comment in the code) |

### countable_concept/src/countable_config.h

| A2537 | link_pi_move | 0.10f | countable_concept/src/countable_config.h:454 | "Tracker.LinkPiMove — per-visit P(a remembered object was moved)" |
| A2538 | link_pi_gone | 0.10f | countable_concept/src/countable_config.h:455 | "Tracker.LinkPiGone — per-visit P(a remembered object left the room)" |
| A2539 | link_lambda_new | 0.10f | countable_concept/src/countable_config.h:456 | "Tracker.LinkLambdaNew — per-visit rate of a NEW object arriving" |
| A2540 | ghost_evidence_gain | 0.15f | countable_concept/src/countable_config.h:457 | "Tracker.GhostEvidenceGain — \|ΔL\| per look at a ghost's spot (the legacy chair's Existence.EvidenceGain, ported unchanged)" |

### countable_concept/src/countable_fitter.cpp

| A2541 | literal | 180.0 | countable_concept/src/countable_fitter.cpp:194 | (no comment in the code) |
| A2542 | literal | 3, 36 | countable_concept/src/countable_fitter.cpp:395 | (no comment in the code) |
| A2543 | literal | 3 | countable_concept/src/countable_fitter.cpp:397 | (no comment in the code) |
| A2544 | literal | 36, 36, 36 | countable_concept/src/countable_fitter.cpp:406 | (no comment in the code) |
| A2545 | literal | 36, 36 | countable_concept/src/countable_fitter.cpp:407 | (no comment in the code) |
| A2546 | literal | 3, 3 | countable_concept/src/countable_fitter.cpp:414 | (no comment in the code) |
| A2547 | literal | 3 | countable_concept/src/countable_fitter.cpp:415 | (no comment in the code) |
| A2548 | literal | 3 | countable_concept/src/countable_fitter.cpp:416 | (no comment in the code) |
| A2549 | literal | 3 | countable_concept/src/countable_fitter.cpp:851 | (no comment in the code) |
| A2550 | literal | 3.0f | countable_concept/src/countable_fitter.cpp:1181 | (no comment in the code) |
| A2551 | literal | 10 | countable_concept/src/countable_fitter.cpp:2076 | (no comment in the code) |
| A2552 | literal | 180.0 | countable_concept/src/countable_fitter.cpp:2132 | (no comment in the code) |

### countable_concept/src/countable_gain_query.h

| A2553 | literal | 5 | countable_concept/src/countable_gain_query.h:57 | (no comment in the code) |

### countable_concept/src/countable_latent_inputs.h

| A2554 | kZedYoloBoundaryStdM | 0.0289f | countable_concept/src/countable_latent_inputs.h:38 | "objects_step0/zed: 0.0289 m (mean −0.0031 m; mean \|offset\| 0.0131 m, the step-0 range-bin table's fridge rows). An upper bound: / the residual layer mismatch of the fridge box is inside it. A shared offset of every boundary pixel moves the face it lies on, / so it is the shared error of the centre (one face seen) and o…" |

### countable_concept/src/countable_neighbour_size.h

| A2555 | literal | 6 | countable_concept/src/countable_neighbour_size.h:46 | (no comment in the code) |
| A2556 | literal | 6 | countable_concept/src/countable_neighbour_size.h:47 | (no comment in the code) |
| A2557 | k | 12.0f | countable_concept/src/countable_neighbour_size.h:48 | (no comment in the code) |

### countable_concept/src/countable_replay.h

| A2558 | kVersion | 1 | countable_concept/src/countable_replay.h:53 | "Bump kVersion when a field's MEANING or UNIT changes (or an event's semantics do). Adding or removing a field needs no / bump: the by-name reader handles it (missing / unknown). A reader refuses a dump newer than the kVersion it was built with." |
| A2559 | literal | 64, 16, 40 | countable_concept/src/countable_replay.h:409 | (no comment in the code) |

### countable_concept/src/countable_scene_graph.cpp

| A2560 | literal | 3.0f | countable_concept/src/countable_scene_graph.cpp:254 | (no comment in the code) |
| A2561 | literal | 0.25f | countable_concept/src/countable_scene_graph.cpp:328 | (no comment in the code) |

### countable_concept/src/countable_size_std.h

| A2562 | literal | 3, 4 | countable_concept/src/countable_size_std.h:36 | (no comment in the code) |
| A2563 | literal | 3 | countable_concept/src/countable_size_std.h:52 | (no comment in the code) |

### countable_concept/src/specificworker.cpp

| A2564 | literal | 1000003ull | countable_concept/src/specificworker.cpp:1034 | (no comment in the code) |

### countable_concept/src/specificworker.h

| A2565 | belief | 4.0f | countable_concept/src/specificworker.h:250 | (no comment in the code) |

### countable_concept/src/specificworker_consumers.cpp

| A2566 | literal | 3, 3 | countable_concept/src/specificworker_consumers.cpp:40 | (no comment in the code) |
| A2567 | kPublishPeriod | std::chrono::seconds(1) | countable_concept/src/specificworker_consumers.cpp:96 | "machine guard: ltom polls at 2 Hz" |
| A2568 | literal | 3, 3 | countable_concept/src/specificworker_consumers.cpp:160 | (no comment in the code) |

### hood_concept/etc/countable.toml

| A1320 | Model.AI2PriorDepthStd | 0.15 | hood_concept/etc/countable.toml:70 [Model]; read countable_concept/src/countable_config.cpp:141 | "K1 hood: the depth's own std (manifest [prior.depth] std_m) — a hood's depth spreads less than its width \| loader: depth (local Y) prior std (m) when the class declares its own ([prior.depth] std_m); <=0 = the footprint std" |
| A1382 | Model.AI2CommonModePosStd | 0.03 | hood_concept/etc/countable.toml:213 [Model]; read countable_concept/src/countable_config.cpp:304 | "Per-frame SHARED-ERROR common-mode stds — the information-saturation FLOORS (CLAUDE.md "encode the effect as a continuous covariance"): N correlated points from one frame cannot collapse Σ below these. LOAD-BEARING. They were previously read from these getters with the keys ABSENT from config (silent defaults) — now ex…" |
| A1383 | Model.AI2CommonModeSizeStd | 0.02 | hood_concept/etc/countable.toml:214 [Model]; read countable_concept/src/countable_config.cpp:306 | "shared size (w,h,H) error floor (m) \| loader: shared size error w,h,H (m)" |
| A1384 | Model.AI2CommonModeYawStd | 0.03 | hood_concept/etc/countable.toml:215 [Model]; read countable_concept/src/countable_config.cpp:308 | "shared yaw error floor (rad) \| loader: shared yaw error (rad)" |
| A1486 | Model.AI2RangeNoiseYawPerM | 0.03f | NO toml key; code default at countable_concept/src/countable_config.cpp:334 | "loader: yaw common-mode std growth (rad per m of range); FLAG-OFF path only (shared-engine Task 5)" |
| A1487 | Model.AI2RangeNoiseSizePerM | 0.08f | NO toml key; code default at countable_concept/src/countable_config.cpp:336 | "loader: SIZE (w,h,H) common-mode std growth (m per m of range); FLAG-OFF path only (shared-engine Task 5)" |
| A1489 | Concept.MeshPath | "" | NO toml key; code default at countable_concept/src/countable_config.cpp:409 | "loader: display mesh (relative to the meshes root); empty = <cls>_concept/meshes/<cls>.obj (the class-folder convention)" |
| A1490 | Concept.LookAttrs | "table" | NO toml key; code default at countable_concept/src/countable_config.cpp:411 | "loader: typed attribute family of the controller's look contract: table (table_roi_*, table_detection_*) or chair (chair_roi_*, chair_detection_*, chair_front_seen*)" |
| A1491 | Concept.AppearanceEnabled | false | NO toml key; code default at countable_concept/src/countable_config.cpp:417 | "loader: colour belief from the assigned ZED masks → mesh_color_rgb, color_name, color_prob (display + names; never the fit)" |
| A1492 | Concept.PublishExistProb | false | NO toml key; code default at countable_concept/src/countable_config.cpp:419 | "loader: publish obj_exist_prob = P(exists) = occupancy x class" |
| A1493 | Concept.ThumbnailsEnabled | false | NO toml key; code default at countable_concept/src/countable_config.cpp:420 | "loader: keep the best retina crop per viewing angle and publish them on the <class>_thumbs node (ltom)" |
| A1494 | Concept.ApproachEnabled | false | NO toml key; code default at countable_concept/src/countable_config.cpp:422 | "loader: offer the pragmatic approach affordance aff_<name>_approach" |
| A1495 | Concept.EpistemicOfferEnabled | true | NO toml key; code default at countable_concept/src/countable_config.cpp:423 | "loader: false = no epistemic viewpoint offer (AuRo C3(b) comparison arm); the belief still updates from what is seen" |
| A1496 | Concept.ChangedObjectEnabled | false | NO toml key; code default at countable_concept/src/countable_config.cpp:425 | "loader: changed object: ghost existence at remembered spots + the same_as identity link on objects born new" |
| A1497 | Tracker.LinkPiMove | 0.10f | NO toml key; code default at countable_concept/src/countable_config.cpp:427 | "loader: identity link: per-visit P(a remembered object was moved) (a GUESS)" |
| A1498 | Tracker.LinkPiGone | 0.10f | NO toml key; code default at countable_concept/src/countable_config.cpp:428 | "loader: identity link: per-visit P(a remembered object left the room) (a GUESS)" |
| A1499 | Tracker.LinkLambdaNew | 0.10f | NO toml key; code default at countable_concept/src/countable_config.cpp:429 | "loader: identity link: per-visit rate of a NEW object arriving (a GUESS)" |
| A1500 | Tracker.GhostEvidenceGain | 0.15f | NO toml key; code default at countable_concept/src/countable_config.cpp:430 | "loader: changed object: \|dL\| per look at a remembered spot's ghost (the legacy chair's Existence.EvidenceGain, ported unchanged)" |
| A1336 | Model.ReplayDump | false | hood_concept/etc/countable.toml:92 [Model]; read countable_concept/src/countable_config.cpp:537 | "diagnostic only (shared-engine plan rev 2, Task R1): true writes each persistent belief's seed + every update's complete input to etc/replay/<node>_<run>_<serial>.bin (~0.6 MB per hood update) for countable_concept/tests/build/replay \| loader: DIAGNOSTIC only, no model effect: with LidarMeasuresOnGated write each persi…" |

### hood_concept/src/hood_belief.cpp

| A2569 | literal | 0.99f | hood_concept/src/hood_belief.cpp:683 | (no comment in the code) |
| A2570 | literal | 3, 5, 4, 3, 5, 4 | hood_concept/src/hood_belief.cpp:725 | (no comment in the code) |
| A2571 | literal | 3 | hood_concept/src/hood_belief.cpp:735 | (no comment in the code) |
| A2572 | literal | 4 | hood_concept/src/hood_belief.cpp:736 | (no comment in the code) |
| A2573 | literal | 4, 3, 5 | hood_concept/src/hood_belief.cpp:895 | "the persistent joint state relabels with θ: w, h of θ and of every 6-sized latent block; e and ℓ are room-frame / (a no-op before the joint is seeded, and on the legacy path)" |

### nightstand_concept/etc/config.toml

| A1689 | Model.AI2PriorDepthStd | -1.0f | NO toml key; code default at countable_concept/src/countable_config.cpp:141 | "loader: depth (local Y) prior std (m) when the class declares its own ([prior.depth] std_m); <=0 = the footprint std" |
| A1589 | Model.AI2CommonModePosStd | 0.03 | nightstand_concept/etc/config.toml:198 [Model]; read countable_concept/src/countable_config.cpp:304 | "Per-frame SHARED-ERROR common-mode stds — the information-saturation FLOORS (CLAUDE.md "encode the effect as a continuous covariance"): N correlated points from one frame cannot collapse Σ below these. LOAD-BEARING. They were previously read from these getters with the keys ABSENT from config (silent defaults) — now ex…" |
| A1590 | Model.AI2CommonModeSizeStd | 0.02 | nightstand_concept/etc/config.toml:199 [Model]; read countable_concept/src/countable_config.cpp:306 | "shared size (w,h,H) error floor (m) \| loader: shared size error w,h,H (m)" |
| A1591 | Model.AI2CommonModeYawStd | 0.03 | nightstand_concept/etc/config.toml:200 [Model]; read countable_concept/src/countable_config.cpp:308 | "shared yaw error floor (rad) \| loader: shared yaw error (rad)" |
| A1694 | Model.AI2RangeNoiseYawPerM | 0.03f | NO toml key; code default at countable_concept/src/countable_config.cpp:334 | "loader: yaw common-mode std growth (rad per m of range); FLAG-OFF path only (shared-engine Task 5)" |
| A1695 | Model.AI2RangeNoiseSizePerM | 0.08f | NO toml key; code default at countable_concept/src/countable_config.cpp:336 | "loader: SIZE (w,h,H) common-mode std growth (m per m of range); FLAG-OFF path only (shared-engine Task 5)" |
| A1697 | Concept.MeshPath | "" | NO toml key; code default at countable_concept/src/countable_config.cpp:409 | "loader: display mesh (relative to the meshes root); empty = <cls>_concept/meshes/<cls>.obj (the class-folder convention)" |
| A1698 | Concept.LookAttrs | "table" | NO toml key; code default at countable_concept/src/countable_config.cpp:411 | "loader: typed attribute family of the controller's look contract: table (table_roi_*, table_detection_*) or chair (chair_roi_*, chair_detection_*, chair_front_seen*)" |
| A1699 | Concept.AppearanceEnabled | false | NO toml key; code default at countable_concept/src/countable_config.cpp:417 | "loader: colour belief from the assigned ZED masks → mesh_color_rgb, color_name, color_prob (display + names; never the fit)" |
| A1700 | Concept.PublishExistProb | false | NO toml key; code default at countable_concept/src/countable_config.cpp:419 | "loader: publish obj_exist_prob = P(exists) = occupancy x class" |
| A1701 | Concept.ThumbnailsEnabled | false | NO toml key; code default at countable_concept/src/countable_config.cpp:420 | "loader: keep the best retina crop per viewing angle and publish them on the <class>_thumbs node (ltom)" |
| A1702 | Concept.ApproachEnabled | false | NO toml key; code default at countable_concept/src/countable_config.cpp:422 | "loader: offer the pragmatic approach affordance aff_<name>_approach" |
| A1703 | Concept.EpistemicOfferEnabled | true | NO toml key; code default at countable_concept/src/countable_config.cpp:423 | "loader: false = no epistemic viewpoint offer (AuRo C3(b) comparison arm); the belief still updates from what is seen" |
| A1704 | Concept.ChangedObjectEnabled | false | NO toml key; code default at countable_concept/src/countable_config.cpp:425 | "loader: changed object: ghost existence at remembered spots + the same_as identity link on objects born new" |
| A1705 | Tracker.LinkPiMove | 0.10f | NO toml key; code default at countable_concept/src/countable_config.cpp:427 | "loader: identity link: per-visit P(a remembered object was moved) (a GUESS)" |
| A1706 | Tracker.LinkPiGone | 0.10f | NO toml key; code default at countable_concept/src/countable_config.cpp:428 | "loader: identity link: per-visit P(a remembered object left the room) (a GUESS)" |
| A1707 | Tracker.LinkLambdaNew | 0.10f | NO toml key; code default at countable_concept/src/countable_config.cpp:429 | "loader: identity link: per-visit rate of a NEW object arriving (a GUESS)" |
| A1708 | Tracker.GhostEvidenceGain | 0.15f | NO toml key; code default at countable_concept/src/countable_config.cpp:430 | "loader: changed object: \|dL\| per look at a remembered spot's ghost (the legacy chair's Existence.EvidenceGain, ported unchanged)" |
| A1544 | Model.ReplayDump | false | nightstand_concept/etc/config.toml:78 [Model]; read countable_concept/src/countable_config.cpp:537 | "diagnostic only (shared-engine plan rev 2, Task R1): true writes each persistent belief's seed + every update's complete input to etc/replay/<node>_<run>_<serial>.bin (~0.6 MB per bed update) for countable_concept/tests/build/replay \| loader: DIAGNOSTIC only, no model effect: with LidarMeasuresOnGated write each persis…" |

### refrigerator_concept/etc/countable.toml

| A1091 | Model.AI2PriorDepthStd | -1.0f | NO toml key; code default at countable_concept/src/countable_config.cpp:141 | "loader: depth (local Y) prior std (m) when the class declares its own ([prior.depth] std_m); <=0 = the footprint std" |
| A0990 | Model.AI2CommonModePosStd | 0.03 | refrigerator_concept/etc/countable.toml:213 [Model]; read countable_concept/src/countable_config.cpp:304 | "Per-frame SHARED-ERROR common-mode stds — the information-saturation FLOORS (CLAUDE.md "encode the effect as a continuous covariance"): N correlated points from one frame cannot collapse Σ below these. LOAD-BEARING. They were previously read from these getters with the keys ABSENT from config (silent defaults) — now ex…" |
| A0991 | Model.AI2CommonModeSizeStd | 0.02 | refrigerator_concept/etc/countable.toml:214 [Model]; read countable_concept/src/countable_config.cpp:306 | "shared size (w,h,H) error floor (m) \| loader: shared size error w,h,H (m)" |
| A0992 | Model.AI2CommonModeYawStd | 0.03 | refrigerator_concept/etc/countable.toml:215 [Model]; read countable_concept/src/countable_config.cpp:308 | "shared yaw error floor (rad) \| loader: shared yaw error (rad)" |
| A1092 | Model.AI2RangeNoiseYawPerM | 0.03f | NO toml key; code default at countable_concept/src/countable_config.cpp:334 | "loader: yaw common-mode std growth (rad per m of range); FLAG-OFF path only (shared-engine Task 5)" |
| A1093 | Model.AI2RangeNoiseSizePerM | 0.08f | NO toml key; code default at countable_concept/src/countable_config.cpp:336 | "loader: SIZE (w,h,H) common-mode std growth (m per m of range); FLAG-OFF path only (shared-engine Task 5)" |
| A1095 | Concept.MeshPath | "" | NO toml key; code default at countable_concept/src/countable_config.cpp:409 | "loader: display mesh (relative to the meshes root); empty = <cls>_concept/meshes/<cls>.obj (the class-folder convention)" |
| A1096 | Concept.LookAttrs | "table" | NO toml key; code default at countable_concept/src/countable_config.cpp:411 | "loader: typed attribute family of the controller's look contract: table (table_roi_*, table_detection_*) or chair (chair_roi_*, chair_detection_*, chair_front_seen*)" |
| A1097 | Concept.AppearanceEnabled | false | NO toml key; code default at countable_concept/src/countable_config.cpp:417 | "loader: colour belief from the assigned ZED masks → mesh_color_rgb, color_name, color_prob (display + names; never the fit)" |
| A1098 | Concept.PublishExistProb | false | NO toml key; code default at countable_concept/src/countable_config.cpp:419 | "loader: publish obj_exist_prob = P(exists) = occupancy x class" |
| A1099 | Concept.ThumbnailsEnabled | false | NO toml key; code default at countable_concept/src/countable_config.cpp:420 | "loader: keep the best retina crop per viewing angle and publish them on the <class>_thumbs node (ltom)" |
| A1100 | Concept.ApproachEnabled | false | NO toml key; code default at countable_concept/src/countable_config.cpp:422 | "loader: offer the pragmatic approach affordance aff_<name>_approach" |
| A1101 | Concept.EpistemicOfferEnabled | true | NO toml key; code default at countable_concept/src/countable_config.cpp:423 | "loader: false = no epistemic viewpoint offer (AuRo C3(b) comparison arm); the belief still updates from what is seen" |
| A1102 | Concept.ChangedObjectEnabled | false | NO toml key; code default at countable_concept/src/countable_config.cpp:425 | "loader: changed object: ghost existence at remembered spots + the same_as identity link on objects born new" |
| A1103 | Tracker.LinkPiMove | 0.10f | NO toml key; code default at countable_concept/src/countable_config.cpp:427 | "loader: identity link: per-visit P(a remembered object was moved) (a GUESS)" |
| A1104 | Tracker.LinkPiGone | 0.10f | NO toml key; code default at countable_concept/src/countable_config.cpp:428 | "loader: identity link: per-visit P(a remembered object left the room) (a GUESS)" |
| A1105 | Tracker.LinkLambdaNew | 0.10f | NO toml key; code default at countable_concept/src/countable_config.cpp:429 | "loader: identity link: per-visit rate of a NEW object arriving (a GUESS)" |
| A1106 | Tracker.GhostEvidenceGain | 0.15f | NO toml key; code default at countable_concept/src/countable_config.cpp:430 | "loader: changed object: \|dL\| per look at a remembered spot's ghost (the legacy chair's Existence.EvidenceGain, ported unchanged)" |
| A0944 | Model.ReplayDump | false | refrigerator_concept/etc/countable.toml:92 [Model]; read countable_concept/src/countable_config.cpp:537 | "diagnostic only (shared-engine plan rev 2, Task R1): true writes each persistent belief's seed + every update's complete input to etc/replay/<node>_<run>_<serial>.bin (~0.6 MB per refrigerator update) for countable_concept/tests/build/replay \| loader: DIAGNOSTIC only, no model effect: with LidarMeasuresOnGated write ea…" |

### refrigerator_concept/src/refrigerator_belief.cpp

| A2574 | literal | 0.99f | refrigerator_concept/src/refrigerator_belief.cpp:624 | (no comment in the code) |
| A2575 | literal | 3, 5, 4, 3, 5, 4, 6 | refrigerator_concept/src/refrigerator_belief.cpp:664 | (no comment in the code) |
| A2576 | literal | 3 | refrigerator_concept/src/refrigerator_belief.cpp:674 | (no comment in the code) |
| A2577 | literal | 4 | refrigerator_concept/src/refrigerator_belief.cpp:675 | (no comment in the code) |
| A2578 | literal | 4, 3, 5 | refrigerator_concept/src/refrigerator_belief.cpp:834 | "the persistent joint state relabels with θ: w, h of θ and of every 6-sized latent block; e and ℓ are room-frame / (a no-op before the joint is seeded, and on the legacy path)" |

### table_concept/etc/countable.toml

| A0559 | Model.AI2PriorDepthStd | 0.572 | table_concept/etc/countable.toml:107 [Model]; read countable_concept/src/countable_config.cpp:141 | "K2c table: = AI2PriorFootprintStd (symmetry c2v) \| loader: depth (local Y) prior std (m) when the class declares its own ([prior.depth] std_m); <=0 = the footprint std" |
| A0594 | Model.AI2CommonModePosStd | 0.03 | table_concept/etc/countable.toml:215 [Model]; read countable_concept/src/countable_config.cpp:304 | "Per-frame SHARED-ERROR common-mode stds — the information-saturation FLOORS (CLAUDE.md "encode the effect as a continuous covariance"): N correlated points from one frame cannot collapse Σ below these. LOAD-BEARING. They were previously read from these getters with the keys ABSENT from config (silent defaults) — now ex…" |
| A0595 | Model.AI2CommonModeSizeStd | 0.02 | table_concept/etc/countable.toml:216 [Model]; read countable_concept/src/countable_config.cpp:306 | "shared size (w,h,H) error floor (m) \| loader: shared size error w,h,H (m)" |
| A0596 | Model.AI2CommonModeYawStd | 0.03 | table_concept/etc/countable.toml:217 [Model]; read countable_concept/src/countable_config.cpp:308 | "shared yaw error floor (rad) \| loader: shared yaw error (rad)" |
| A0705 | Model.AI2RangeNoiseYawPerM | 0.03f | NO toml key; code default at countable_concept/src/countable_config.cpp:334 | "loader: yaw common-mode std growth (rad per m of range); FLAG-OFF path only (shared-engine Task 5)" |
| A0706 | Model.AI2RangeNoiseSizePerM | 0.08f | NO toml key; code default at countable_concept/src/countable_config.cpp:336 | "loader: SIZE (w,h,H) common-mode std growth (m per m of range); FLAG-OFF path only (shared-engine Task 5)" |
| A0708 | Concept.MeshPath | "" | NO toml key; code default at countable_concept/src/countable_config.cpp:409 | "loader: display mesh (relative to the meshes root); empty = <cls>_concept/meshes/<cls>.obj (the class-folder convention)" |
| A0645 | Concept.LookAttrs | "table" | table_concept/etc/countable.toml:376 [Concept]; read countable_concept/src/countable_config.cpp:411 | "── K2c: the graph consumers' channels the legacy table agent published (spike: .superpowers/sdd/2026-11-23-shared-engine/task-K2c-report.md) ── K2c table: table_roi_* / table_detection_* / table_frames_since_detection — the family the legacy table wrote (controller look contract; room_concept's object anchor reads tabl…" |
| A0646 | Concept.AppearanceEnabled | true | table_concept/etc/countable.toml:377 [Concept]; read countable_concept/src/countable_config.cpp:417 | "K2c table: colour belief → mesh_color_rgb (viewer tint, as the legacy table published); ALSO color_name / color_prob, which the legacy table did not publish (additive) \| loader: colour belief from the assigned ZED masks → mesh_color_rgb, color_name, color_prob (display + names; never the fit)" |
| A0647 | Concept.PublishExistProb | false | table_concept/etc/countable.toml:378 [Concept]; read countable_concept/src/countable_config.cpp:419 | "K2c table: the legacy table did not publish obj_exist_prob \| loader: publish obj_exist_prob = P(exists) = occupancy x class" |
| A0648 | Concept.ThumbnailsEnabled | false | table_concept/etc/countable.toml:379 [Concept]; read countable_concept/src/countable_config.cpp:420 | "K2c table: the legacy table had no thumbnails; REQUIRED off by symmetry c2v (binned on the yaw, a representative) \| loader: keep the best retina crop per viewing angle and publish them on the <class>_thumbs node (ltom)" |
| A0649 | Concept.ApproachEnabled | false | table_concept/etc/countable.toml:380 [Concept]; read countable_concept/src/countable_config.cpp:422 | "K2c table: the legacy table had no approach affordance; REQUIRED off by symmetry c2v (fronted) \| loader: offer the pragmatic approach affordance aff_<name>_approach" |
| A0650 | Concept.EpistemicOfferEnabled | true | table_concept/etc/countable.toml:381 [Concept]; read countable_concept/src/countable_config.cpp:423 | "K2c table: the viewpoint offer (epistemic_* on the node, aff_table_N), as the legacy table published \| loader: false = no epistemic viewpoint offer (AuRo C3(b) comparison arm); the belief still updates from what is seen" |
| A0651 | Concept.ChangedObjectEnabled | false | table_concept/etc/countable.toml:382 [Concept]; read countable_concept/src/countable_config.cpp:425 | "K2c table: the legacy table had no ghosts / same_as_* link \| loader: changed object: ghost existence at remembered spots + the same_as identity link on objects born new" |
| A0709 | Tracker.LinkPiMove | 0.10f | NO toml key; code default at countable_concept/src/countable_config.cpp:427 | "loader: identity link: per-visit P(a remembered object was moved) (a GUESS)" |
| A0710 | Tracker.LinkPiGone | 0.10f | NO toml key; code default at countable_concept/src/countable_config.cpp:428 | "loader: identity link: per-visit P(a remembered object left the room) (a GUESS)" |
| A0711 | Tracker.LinkLambdaNew | 0.10f | NO toml key; code default at countable_concept/src/countable_config.cpp:429 | "loader: identity link: per-visit rate of a NEW object arriving (a GUESS)" |
| A0712 | Tracker.GhostEvidenceGain | 0.15f | NO toml key; code default at countable_concept/src/countable_config.cpp:430 | "loader: changed object: \|dL\| per look at a remembered spot's ghost (the legacy chair's Existence.EvidenceGain, ported unchanged)" |
| A0547 | Model.ReplayDump | false | table_concept/etc/countable.toml:93 [Model]; read countable_concept/src/countable_config.cpp:537 | "diagnostic only (shared-engine plan rev 2, Task R1): true writes each persistent belief's seed + every update's complete input to etc/replay/<node>_<run>_<serial>.bin (~0.6 MB per table update) for countable_concept/tests/build/replay \| loader: DIAGNOSTIC only, no model effect: with LidarMeasuresOnGated write each pers…" |

### toilet_concept/etc/config.toml

| A2110 | Model.AI2PriorDepthStd | -1.0f | NO toml key; code default at countable_concept/src/countable_config.cpp:141 | "loader: depth (local Y) prior std (m) when the class declares its own ([prior.depth] std_m); <=0 = the footprint std" |
| A2010 | Model.AI2CommonModePosStd | 0.03 | toilet_concept/etc/config.toml:199 [Model]; read countable_concept/src/countable_config.cpp:304 | "Per-frame SHARED-ERROR common-mode stds — the information-saturation FLOORS (CLAUDE.md "encode the effect as a continuous covariance"): N correlated points from one frame cannot collapse Σ below these. LOAD-BEARING. They were previously read from these getters with the keys ABSENT from config (silent defaults) — now ex…" |
| A2011 | Model.AI2CommonModeSizeStd | 0.02 | toilet_concept/etc/config.toml:200 [Model]; read countable_concept/src/countable_config.cpp:306 | "shared size (w,h,H) error floor (m) \| loader: shared size error w,h,H (m)" |
| A2012 | Model.AI2CommonModeYawStd | 0.03 | toilet_concept/etc/config.toml:201 [Model]; read countable_concept/src/countable_config.cpp:308 | "shared yaw error floor (rad) \| loader: shared yaw error (rad)" |
| A2114 | Model.AI2RangeNoiseYawPerM | 0.03f | NO toml key; code default at countable_concept/src/countable_config.cpp:334 | "loader: yaw common-mode std growth (rad per m of range); FLAG-OFF path only (shared-engine Task 5)" |
| A2115 | Model.AI2RangeNoiseSizePerM | 0.08f | NO toml key; code default at countable_concept/src/countable_config.cpp:336 | "loader: SIZE (w,h,H) common-mode std growth (m per m of range); FLAG-OFF path only (shared-engine Task 5)" |
| A2117 | Concept.MeshPath | "" | NO toml key; code default at countable_concept/src/countable_config.cpp:409 | "loader: display mesh (relative to the meshes root); empty = <cls>_concept/meshes/<cls>.obj (the class-folder convention)" |
| A2118 | Concept.LookAttrs | "table" | NO toml key; code default at countable_concept/src/countable_config.cpp:411 | "loader: typed attribute family of the controller's look contract: table (table_roi_*, table_detection_*) or chair (chair_roi_*, chair_detection_*, chair_front_seen*)" |
| A2119 | Concept.AppearanceEnabled | false | NO toml key; code default at countable_concept/src/countable_config.cpp:417 | "loader: colour belief from the assigned ZED masks → mesh_color_rgb, color_name, color_prob (display + names; never the fit)" |
| A2120 | Concept.PublishExistProb | false | NO toml key; code default at countable_concept/src/countable_config.cpp:419 | "loader: publish obj_exist_prob = P(exists) = occupancy x class" |
| A2121 | Concept.ThumbnailsEnabled | false | NO toml key; code default at countable_concept/src/countable_config.cpp:420 | "loader: keep the best retina crop per viewing angle and publish them on the <class>_thumbs node (ltom)" |
| A2122 | Concept.ApproachEnabled | false | NO toml key; code default at countable_concept/src/countable_config.cpp:422 | "loader: offer the pragmatic approach affordance aff_<name>_approach" |
| A2123 | Concept.EpistemicOfferEnabled | true | NO toml key; code default at countable_concept/src/countable_config.cpp:423 | "loader: false = no epistemic viewpoint offer (AuRo C3(b) comparison arm); the belief still updates from what is seen" |
| A2124 | Concept.ChangedObjectEnabled | false | NO toml key; code default at countable_concept/src/countable_config.cpp:425 | "loader: changed object: ghost existence at remembered spots + the same_as identity link on objects born new" |
| A2125 | Tracker.LinkPiMove | 0.10f | NO toml key; code default at countable_concept/src/countable_config.cpp:427 | "loader: identity link: per-visit P(a remembered object was moved) (a GUESS)" |
| A2126 | Tracker.LinkPiGone | 0.10f | NO toml key; code default at countable_concept/src/countable_config.cpp:428 | "loader: identity link: per-visit P(a remembered object left the room) (a GUESS)" |
| A2127 | Tracker.LinkLambdaNew | 0.10f | NO toml key; code default at countable_concept/src/countable_config.cpp:429 | "loader: identity link: per-visit rate of a NEW object arriving (a GUESS)" |
| A2128 | Tracker.GhostEvidenceGain | 0.15f | NO toml key; code default at countable_concept/src/countable_config.cpp:430 | "loader: changed object: \|dL\| per look at a remembered spot's ghost (the legacy chair's Existence.EvidenceGain, ported unchanged)" |
| A1964 | Model.ReplayDump | false | toilet_concept/etc/config.toml:78 [Model]; read countable_concept/src/countable_config.cpp:537 | "diagnostic only (shared-engine plan rev 2, Task R1): true writes each persistent belief's seed + every update's complete input to etc/replay/<node>_<run>_<serial>.bin (~0.6 MB per bed update) for countable_concept/tests/build/replay \| loader: DIAGNOSTIC only, no model effect: with LidarMeasuresOnGated write each persis…" |
