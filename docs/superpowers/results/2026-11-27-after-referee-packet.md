# Blind referee packet: unmatched rows of the after-count

Source: active_inference commit cf65a471, tree hash `5c0029ab522691fb4747d2e7215493162f13f0b0` (branch feat/shared-engine, the end of K3).
Rows are the after-count rows whose key, reader or role is new relative to the labelled baseline, rows whose enclosing function changed ("context changed"), and baseline rows that were not found by name or value ("moved or removed?"). No label, no proposed label. The role column quotes the code's own comment or the loader text, not a reading.

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

| A1754 | Model.ReplayDump | false | bed_concept/etc/config.toml:78 [Model]; read countable_concept/src/countable_config.cpp:537 | "diagnostic only (shared-engine plan rev 2, Task R1): true writes each persistent belief's seed + every update's complete input to etc/replay/<node>_<run>_<serial>.bin (~0.6 MB per bed update) for countable_concept/tests/build/replay \| loader: DIAGNOSTIC only, no model effect: with LidarMeasuresOnGated write each persis…" |
| A1901 | Model.AI2PriorDepthStd | -1.0f | NO toml key; code default at countable_concept/src/countable_config.cpp:141 | "loader: depth (local Y) prior std (m) when the class declares its own ([prior.depth] std_m); <=0 = the footprint std" |
| A1800 | Model.AI2CommonModePosStd | 0.03 | bed_concept/etc/config.toml:199 [Model]; read countable_concept/src/countable_config.cpp:304 | "Per-frame SHARED-ERROR common-mode stds — the information-saturation FLOORS (CLAUDE.md "encode the effect as a continuous covariance"): N correlated points from one frame cannot collapse Σ below these. LOAD-BEARING. They were previously read from these getters with the keys ABSENT from config (silent defaults) — now ex…" |
| A1801 | Model.AI2CommonModeSizeStd | 0.02 | bed_concept/etc/config.toml:200 [Model]; read countable_concept/src/countable_config.cpp:306 | "shared size (w,h,H) error floor (m) \| loader: shared size error w,h,H (m)" |
| A1802 | Model.AI2CommonModeYawStd | 0.03 | bed_concept/etc/config.toml:201 [Model]; read countable_concept/src/countable_config.cpp:308 | "shared yaw error floor (rad) \| loader: shared yaw error (rad)" |
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

### chair_concept/etc/countable.toml

| A0176 | Model.ReplayDump | false | chair_concept/etc/countable.toml:88 [Model]; read countable_concept/src/countable_config.cpp:537 | "diagnostic only (shared-engine plan rev 2, Task R1): true writes each persistent belief's seed + every update's complete input to etc/replay/<node>_<run>_<serial>.bin (~0.6 MB per chair update) for countable_concept/tests/build/replay \| loader: DIAGNOSTIC only, no model effect: with LidarMeasuresOnGated write each pers…" |
| A0188 | Model.AI2PriorDepthStd | 0.026 | chair_concept/etc/countable.toml:102 [Model]; read countable_concept/src/countable_config.cpp:141 | "K2b chair (fix 1): (0.51 − 0.42)/√12, the same uniform (the depth's own std, K1 key) \| loader: depth (local Y) prior std (m) when the class declares its own ([prior.depth] std_m); <=0 = the footprint std" |
| A0223 | Model.AI2CommonModePosStd | 0.03 | chair_concept/etc/countable.toml:210 [Model]; read countable_concept/src/countable_config.cpp:304 | "Per-frame SHARED-ERROR common-mode stds — the information-saturation FLOORS (CLAUDE.md "encode the effect as a continuous covariance"): N correlated points from one frame cannot collapse Σ below these. LOAD-BEARING. They were previously read from these getters with the keys ABSENT from config (silent defaults) — now ex…" |
| A0224 | Model.AI2CommonModeSizeStd | 0.02 | chair_concept/etc/countable.toml:211 [Model]; read countable_concept/src/countable_config.cpp:306 | "shared size (w,h,H) error floor (m) \| loader: shared size error w,h,H (m)" |
| A0225 | Model.AI2CommonModeYawStd | 0.03 | chair_concept/etc/countable.toml:212 [Model]; read countable_concept/src/countable_config.cpp:308 | "shared yaw error floor (rad) \| loader: shared yaw error (rad)" |
| A0274 | Concept.LookAttrs | "chair" | chair_concept/etc/countable.toml:371 [Concept]; read countable_concept/src/countable_config.cpp:411 | "── K2b: the graph consumers' channels the legacy chair agent published (spike: .superpowers/sdd/2026-11-23-shared-engine/task-K2b-report.md) ── K2b chair: chair_roi_* / chair_detection_* / chair_front_seen(_frame) — the controller's chair look contract (affordance_protocol.h) and its look counter (controller_session.cp…" |
| A0275 | Concept.AppearanceEnabled | true | chair_concept/etc/countable.toml:372 [Concept]; read countable_concept/src/countable_config.cpp:417 | "K2b chair: colour belief → mesh_color_rgb (viewer tint), color_name / color_prob (language_agent, ltom, the identity link) \| loader: colour belief from the assigned ZED masks → mesh_color_rgb, color_name, color_prob (display + names; never the fit)" |
| A0276 | Concept.PublishExistProb | true | chair_concept/etc/countable.toml:373 [Concept]; read countable_concept/src/countable_config.cpp:419 | "K2b chair: obj_exist_prob = P(exists) — ltom's memory weight + recall prior, language_agent, controller missions \| loader: publish obj_exist_prob = P(exists) = occupancy x class" |
| A0277 | Concept.ThumbnailsEnabled | true | chair_concept/etc/countable.toml:374 [Concept]; read countable_concept/src/countable_config.cpp:420 | "K2b chair: chair_thumbs (best crop per viewing angle) for ltom \| loader: keep the best retina crop per viewing angle and publish them on the <class>_thumbs node (ltom)" |
| A0278 | Concept.ApproachEnabled | true | chair_concept/etc/countable.toml:375 [Concept]; read countable_concept/src/countable_config.cpp:422 | "K2b chair: aff_chair_N_approach — language_agent "approach the chair", the controller's pragmatic claim \| loader: offer the pragmatic approach affordance aff_<name>_approach" |
| A0279 | Concept.EpistemicOfferEnabled | true | chair_concept/etc/countable.toml:376 [Concept]; read countable_concept/src/countable_config.cpp:423 | "K2b chair: false = no viewpoint offer (the AuRo C3(b) comparison arm, as ChairConcept.EpistemicOfferEnabled was) \| loader: false = no epistemic viewpoint offer (AuRo C3(b) comparison arm); the belief still updates from what is seen" |
| A0280 | Concept.ChangedObjectEnabled | true | chair_concept/etc/countable.toml:377 [Concept]; read countable_concept/src/countable_config.cpp:425 | "K2b chair: ghosts at remembered spots + same_as_uid / same_as_prob / left_spot_prob (ltom joins the histories) \| loader: changed object: ghost existence at remembered spots + the same_as identity link on objects born new" |
| A0281 | Concept.MeshPath | "chair_concept/meshes/chair_back_plus_y.obj" | chair_concept/etc/countable.toml:378 [Concept]; read countable_concept/src/countable_config.cpp:409 | "K2b chair: chair.obj turned 180° (backrest at +y, the countable convention) \| loader: display mesh (relative to the meshes root); empty = <cls>_concept/meshes/<cls>.obj (the class-folder convention)" |
| A0310 | Tracker.LinkPiMove | 0.10 | chair_concept/etc/countable.toml:459 [Tracker]; read countable_concept/src/countable_config.cpp:427 | "K2b chair: identity link: per-visit P(a remembered chair was moved). A GUESS (the legacy chair's value) \| loader: identity link: per-visit P(a remembered object was moved) (a GUESS)" |
| A0311 | Tracker.LinkPiGone | 0.10 | chair_concept/etc/countable.toml:460 [Tracker]; read countable_concept/src/countable_config.cpp:428 | "K2b chair: identity link: per-visit P(a remembered chair left the room). A GUESS (the legacy chair's value) \| loader: identity link: per-visit P(a remembered object left the room) (a GUESS)" |
| A0312 | Tracker.LinkLambdaNew | 0.10 | chair_concept/etc/countable.toml:461 [Tracker]; read countable_concept/src/countable_config.cpp:429 | "K2b chair: identity link: per-visit rate of a NEW chair arriving. A GUESS (the legacy chair's value) \| loader: identity link: per-visit rate of a NEW object arriving (a GUESS)" |
| A0313 | Tracker.GhostEvidenceGain | 0.15 | chair_concept/etc/countable.toml:462 [Tracker]; read countable_concept/src/countable_config.cpp:430 | "K2b chair: \|ΔL\| per look at a ghost's spot (the legacy chair's Existence.EvidenceGain, ported unchanged) \| loader: changed object: \|dL\| per look at a remembered spot's ghost (the legacy chair's Existence.EvidenceGain, ported unchanged)" |

### chair_concept/src/chair_belief.cpp

| A2584 | flip evidence clamps (uses) | +-kFlipClamp | chair_concept/src/chair_belief.cpp:257 [BASELINE tree 535d57ec, not found by name or value] | **moved or removed?** baseline location chair_concept/src/chair_belief.cpp:257; baseline code: "flip_acc_[k] += frame_w * (mixture_nll(pts, rot, R) - e0); // >0 current better, <0 mode k better ; flip_acc_[k] = std::clamp(flip_acc_[k], -kFlipClamp, kFlipClamp); ; }"; after tree 5c0029ab522691fb4747d2e7215493162f13f0b0 |

### common/affordance_protocol/affordance_protocol.h

| A2586 | advance roi_fill | table 0.45, hood 0.30 | common/affordance_protocol/affordance_protocol.h:134 [BASELINE tree 535d57ec, not found by name or value] | **moved or removed?** baseline location common/affordance_protocol/affordance_protocol.h:134; baseline code: "// .center ('table_roi_offset') // err_vec → base yaw/side centring ; // .advance('table_roi_fill', 0.45f) // scalar → advance to stand-off ; // .valid ('table_roi_valid') // gate"; comment: "write_contract(G, node, Contract::servo() / .center ("table_roi_offset")             // err_vec  → base yaw/side centring / .advance("table_roi_fill", 0.45f)        // scalar   → advance to stand-off"; after tree 5c0029ab522691fb4747d2e7215493162f13f0b0 |

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

### common/appearance_belief/colour_terms.h

| A2587 | colour prototypes (12) | RGB table | common/appearance_belief/colour_terms.h:19 [BASELINE tree 535d57ec, not found by name or value] | **moved or removed?** baseline location common/appearance_belief/colour_terms.h:19; baseline code: "inline constexpr std::array<ColourTerm, 12> kPalette{{ ; {'red', 0.80f, 0.10f, 0.10f}, {'orange', 0.95f, 0.55f, 0.10f}, {'yellow', 0.95f, 0.90f, 0.15f}, ; {'green', 0.15f, 0.60f, 0.20f}, {'blue', 0.15f, 0.30f, 0.80f}, {'purple', 0.45f, 0.20f, 0.65f},"; after tree 5c0029ab522691fb4747d2e7215493162f13f0b0 |

### common/concept_manifest/shape_parts.h

| A2493 | literal | 4 | common/concept_manifest/shape_parts.h:172 | (no comment in the code) |

### common/existence_belief/existence_belief.h

| A2585 | erfc / tanh structure | 0.7071, tanh | common/existence_belief/existence_belief.h:122 [BASELINE tree 535d57ec, not found by name or value] | **moved or removed?** baseline location common/existence_belief/existence_belief.h:122; baseline code: "const float cyaw = std::cos(yaw), syaw = std::sin(yaw); ; const auto phi = [](float x) { return 0.5f * std::erfc(-x * 0.70710678f); }; // Φ(x) = ½·erfc(−x/√2) ; const float sigma_surf = std::sqrt(p.sensor_sigma_m * p.sensor_sigma_m + surface_sigma_m * surface_"; comment: "Φ(x) = ½·erfc(−x/√2)"; after tree 5c0029ab522691fb4747d2e7215493162f13f0b0 |

### common/graph3d/graph3d_scene_builder.cpp

| A2588 | scene colours, radii, gaps | ~45 literals | common/graph3d/graph3d_scene_builder.cpp:23 [BASELINE tree 535d57ec, not found by name or value] | **moved or removed?** baseline location common/graph3d/graph3d_scene_builder.cpp:23; baseline code: "constexpr Rgb kRobot {0.92f, 0.94f, 0.98f}; // bottom of the ladder: bright, it anchors everything ; constexpr Rgb kSlate {0.42f, 0.47f, 0.54f}; // the room: the container, deliberately quiet"; comment: "bottom of the ladder: bright, it anchors everything"; after tree 5c0029ab522691fb4747d2e7215493162f13f0b0 |

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
| A2269 | front_acc re-baseline clamp | +-6.0 | countable_concept/src/countable_belief.cpp:1221 | **context changed** (enclosing function differs between 535d57ec and cf65a471; before/after text below). Baseline name: front_acc re-baseline clamp. Code comment: (none) |
| A2522 | literal | 4, 3, 5 | countable_concept/src/countable_belief.cpp:1239 | "the persistent joint state relabels with θ: w, h of θ and of every 6-sized latent block (b_cam, b_L, u_cam, u_L, s_cam, / s_L); e and ℓ are room-frame, untouched (a no-op before the joint is seeded)" |
| A2268 | kFrontClamp | 6.0 | countable_concept/src/countable_belief.cpp:1380 | **context changed** (enclosing function differs between 535d57ec and cf65a471; before/after text below). Baseline name: kFrontClamp. Code comment: "Clamp each accumulated mode-evidence to ±kFrontClamp so a long confident run cannot drive the accumulator / arbitrarily deep and make the belief unable to RECANT when the bed is physically turned. ≈6 nats ≈ a / 400:1 odds — decisive but recoverable. Mirrors ChairBelief's ±6 flip clamp." |

<details><summary>A2269: context changed, function `bool CountableBelief::resolve_front_geometric()` (16 lines before, 17 after)</summary>

Evidence line (before, countable_concept/src/countable_belief.cpp:1549): `front_acc_[m] = std::clamp(old[(m + kbest) % 4] - old[kbest], -6.0f, 6.0f);`; found unchanged at countable_concept/src/countable_belief.cpp:1221.

```
BEFORE (from line 1538)

AFTER (from line 1209)
    if (params_.symmetric) return false;   // K2c: every mode is the same body — nothing to resolve
```
</details>


<details><summary>A2268: context changed, function `bool CountableBelief::resolve_front(const FrontCue& cue, float evidence_weight)` (93 lines before, 94 after)</summary>

Evidence line (before, countable_concept/src/countable_belief.cpp:1717): `constexpr float kFrontClamp = 6.0f;`; found unchanged at countable_concept/src/countable_belief.cpp:1380.

```
BEFORE (from line 1713)

AFTER (from line 1375)
    if (params_.symmetric) return false;   // K2c: no front to find — every mode is the same body
```
</details>


### countable_concept/src/countable_belief.h

| A2523 | predict_period_s | 0.1f | countable_concept/src/countable_belief.h:292 | "the agent's predict period (Model.AI2AgeNominalDtS, the mask stream's nominal period)" |
| A2579 | F1r9 cap switch: gn_tol_sigma, which now decides absorbed (Laplace Σ) or not absorbed (P̄) | 0.1 | countable_concept/src/countable_belief.h:332 | "code: double gn_tol_sigma = 0.1;" |
| A2524 | N | 6 | countable_concept/src/countable_belief.h:440 | (no comment in the code) |
| A2525 | kYaw | 5 | countable_concept/src/countable_belief.h:441 | "θ = [cx, cy, H, w, h, yaw]" |
| A2526 | kHeight | 2 | countable_concept/src/countable_belief.h:442 | "H: the DOF the LiDAR's top-grazing ring decorrelates (ρ_L height)" |
| A2527 | kWallLatents | 1 | countable_concept/src/countable_belief.h:480 | "the wall latent (F1.4): one per (room, edge), and the wall priors it carries, entered once at s = θ + B_w b" |
| A2528 | kProportions | 1 | countable_concept/src/countable_belief.h:491 | "K2a-L: the class shape proportions (persistent_engine.h kProp). ONE slot for every countable class (the engine's latent / layout is fixed at compile time); a class declaring none keeps it at Δ = 0 (exactly zero, zero variance)." |
| A2529 | kProportions | 0 | countable_concept/src/countable_belief.h:523 | (no comment in the code) |

### countable_concept/src/countable_birth_footprint.h

| A2279 | birth footprint percentiles | 0.05 / 0.95 / 0.9 | countable_concept/src/countable_birth_footprint.h:49 | **context changed** (enclosing function differs between 535d57ec and cf65a471; before/after text below). Baseline name: birth footprint percentiles. Code comment: (none) |

<details><summary>A2279: context changed, function `inline BirthFootprint birth_footprint(std::span<const Eigen::Vector2f> xy, const Eigen::Vector2f& wall_p, const Eigen::Vector2f& wall_n, float class_w, float cl` (32 lines before, 32 after)</summary>

Evidence line (before, countable_concept/src/countable_birth_footprint.h:44): `const float a0 = pct(a, 0.05f), a1 = pct(a, 0.95f), b0 = pct(b, 0.05f), b1 = pct(b, 0.95f);`; found unchanged at countable_concept/src/countable_birth_footprint.h:49.

```
BEFORE (from line 51)
    const float ex_t = std::max(0.0f, out.extent_t - class_w), ex_n = std::max(0.0f, out.extent_n - class_d);
AFTER (from line 56)
    const float ex_t = out.extent_t - class_w, ex_n = std::max(0.0f, out.extent_n - class_d);
```
</details>


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
| A2283 | graph-attr fallback seeds | 0.60,0.60,1.70 | countable_concept/src/countable_fitter.cpp:894 | **context changed** (enclosing function differs between 535d57ec and cf65a471; before/after text below). Baseline name: graph-attr fallback seeds. Code comment: "standard bed footprint ≈ 0.60×0.60" |
| A2275 | wall fit sigma | 0.02 | countable_concept/src/countable_fitter.cpp:1069 | **context changed** (enclosing function differs between 535d57ec and cf65a471; before/after text below). Baseline name: wall fit sigma. Code comment: (none) |
| A2287 | birth top quantiles | 0.95 / 0.97 | countable_concept/src/countable_fitter.cpp:1147 | **context changed** (enclosing function differs between 535d57ec and cf65a471; before/after text below). Baseline name: birth top quantiles. Code comment: (none) |
| A2288 | birth z band | 3 sigma | countable_concept/src/countable_fitter.cpp:1180 | **context changed** (enclosing function differs between 535d57ec and cf65a471; before/after text below). Baseline name: birth z band. Code comment: "tails, which are cloud ARTEFACTS (not bed surface) and would bias the footprint outward. / The band's base is the body's underside: the floor for a floor-anchored class, H − e for a hanging one (a / point under a hood is the hob or the worktop, not the hood)." |
| A2550 | literal | 3.0f | countable_concept/src/countable_fitter.cpp:1181 | (no comment in the code) |
| A2290 | birth size floor | 0.10 | countable_concept/src/countable_fitter.cpp:1185 | **context changed** (enclosing function differs between 535d57ec and cf65a471; before/after text below). Baseline name: birth size floor. Code comment: "major axis = the seen face's width (reliable)" |
| A2289 | birth anisotropy test | 0.10 | countable_concept/src/countable_fitter.cpp:1204 | **context changed** (enclosing function differs between 535d57ec and cf65a471; before/after text below). Baseline name: birth anisotropy test. Code comment: (none) |
| A2284 | kObliquityYawGain | 0.05 | countable_concept/src/countable_fitter.cpp:1686 | **context changed** (enclosing function differs between 535d57ec and cf65a471; before/after text below). Baseline name: kObliquityYawGain. Code comment: "FootprintResidual == true branch there is currently NO obliquity/tilt yaw cap wired in (chain_cov_yaw / carries only the white range term) — that branch is experimental. Do not delete kObliquityYawGain until / tilt_yaw_common_mode is called here and validated. See countable_belief.cpp::tilt_yaw_common_mode. / ~30° σ_yaw at cos=0.09 → per-point GN holds yaw at grazing" |
| A2285 | obliquity cos clamp | [0.05,1] | countable_concept/src/countable_fitter.cpp:1687 | **context changed** (enclosing function differs between 535d57ec and cf65a471; before/after text below). Baseline name: obliquity cos clamp. Code comment: (none) |
| A2291 | kYawJumpLogRad | 0.087 | countable_concept/src/countable_fitter.cpp:1804 | **context changed** (enclosing function differs between 535d57ec and cf65a471; before/after text below). Baseline name: kYawJumpLogRad. Code comment: "≈5°: diagnostic PRINT trigger only (no fit effect)" |
| A2551 | literal | 10 | countable_concept/src/countable_fitter.cpp:2076 | (no comment in the code) |
| A2552 | literal | 180.0 | countable_concept/src/countable_fitter.cpp:2132 | (no comment in the code) |

<details><summary>A2283: context changed, function `bool CountableFitter::ensure_instance(const DSR::Node& node, std::uint64_t room_id)` (86 lines before, 98 after)</summary>

Evidence line (before, countable_concept/src/countable_fitter.cpp:772): `fix(init_state.w, 0.60f, "w"); // standard bed footprint ≈ 0.60×0.60`; found unchanged at countable_concept/src/countable_fitter.cpp:894.

```
BEFORE (from line 720)

AFTER (from line 835)
    // A hanging class's height_m is its body's EXTENT, not its top: start from the class's top prior; the RT origin (the
    // underside) refines it below.
    if (rc::countable::VerticalSpan{cfg_.vertical_extent_m}.hangs()) init_state.countable_height = cfg_.ai2_prior_height_m;
```
```
BEFORE (from line 730)

AFTER (from line 848)
                // A hanging class's node carries its BODY: origin = the underside, height_m = the extent, so its top
                // H = z + height_m (floor-anchored: origin z = 0 and height_m IS H, unchanged).
                if (const rc::countable::VerticalSpan span{cfg_.vertical_extent_m}; span.hangs())
                    init_state.countable_height = (tvec.size() >= 3 ? tvec[2] : 0.0f) + span.extent_m;
```
```
BEFORE (from line 785)

AFTER (from line 907)
    // K2b: the pragmatic approach (Concept.ApproachEnabled; the legacy chair's). The Schmitt band and debounce are
    // PragmaticAffordance's defaults; Fronted: the class has a front (its −y face), named once the yaw is surely known.
    if (cfg_.approach_enabled)
        inst.approach.init(G_, node.id(), node.name(), rc::pragmatic::PragmaticAffordance::Policy{},
                           rc::pragmatic::ObjectApproach::Sides::Fronted);
```
</details>


<details><summary>A2275: context changed, function `WallRef CountableFitter::nearest_wall(const Eigen::Vector2f& q) const` (29 lines before, 30 after)</summary>

Evidence line (before, countable_concept/src/countable_fitter.cpp:975): `w.ok = true; w.p = best_foot; w.n = nrm; w.sigma_m = 0.02f;`; found unchanged at countable_concept/src/countable_fitter.cpp:1069.

```
BEFORE (from line 976)

AFTER (from line 1070)
    w.room = room_id_; w.edge = static_cast<int>(best_i);   // the edge's identity as room_concept publishes the outline (F1.4)
```
</details>


<details><summary>A2287: context changed, function `float CountableFitter::run_inference(CountableInstance& inst, const CountableObservation& observation)` (685 lines before, 757 after)</summary>

Evidence line (before, countable_concept/src/countable_fitter.cpp:1048): `const std::size_t k = static_cast<std::size_t>(0.95f * (zs.size() - 1));`; found unchanged at countable_concept/src/countable_fitter.cpp:1147.

```
BEFORE (from line 1040)

AFTER (from line 1135)
        std::vector<Eigen::Vector3f> birth_cloud;   // F1 round 6.2: the seed H is re-derived from it once the belief knows its shape
```
```
BEFORE (from line 1042)

AFTER (from line 1138)
            birth_cloud.reserve(static_cast<std::size_t>(npts));
            birth_cloud.insert(birth_cloud.end(), observation.candidate_pts.begin(), observation.candidate_pts.end());
            birth_cloud.insert(birth_cloud.end(), observation.residual_pts.begin(), observation.residual_pts.end());
```
```
BEFORE (from line 1066)
            // Skip on a TRIMMED mask (extent is only a lower bound → would birth too small); wait for a full mask.
            if (cfg_.footprint_moment_precision > 0.0f and inst.last_trunc_frac <= 1e-3f)
AFTER (from line 1165)
            // Legacy path: skip on a TRIMMED mask (extent is only a lower bound → would birth too small); wait for a full mask.
            // Persistent path (plan rev 2 F1.3): no switch. The engine enters the class prior at the class mean, so the seed is
            // only the GN start, and a clipped mask enters as what it is — a seen extent shorter than the class's — through
            // birth_on_priors' position covariance, (class extent − seen extent)²/12 along the wall (countable_birth_footprint.h).
            // engine1's bed: trunc 0.23 % skipped the whole seed and GN started 0.8 m short of the bed's centre.
            if (cfg_.footprint_moment_precision > 0.0f and (cfg_.lidar_measures_on_gated or inst.last_trunc_frac <= 1e-3f))
```
```
BEFORE (from line 1075)

AFTER (from line 1178)
                // The band's base is the body's underside: the floor for a floor-anchored class, H − e for a hanging one (a
                // point under a hood is the hob or the worktop, not the hood).
```
```
BEFORE (from line 1076)
                const float z_lo = 3.0f * cfg_.ai2_sigma_base_m;
AFTER (from line 1181)
                const float z_lo = rc::countable::VerticalSpan{cfg_.vertical_extent_m}.z0(s0.H) + 3.0f * cfg_.ai2_sigma_base_m;
```
```
BEFORE (from line 1101)
                    // ★BORN ON THE FURNITURE PRIORS (owner 2026-11-14): room-axis yaw, back on the nearest wall.
                    if (have_room_geometry_) birth_pos_cov = birth_on_priors(s0, pts, z_lo, z_hi, inst.node_name);
AFTER (from line 1206)
                    // ★BORN ON THE FURNITURE PRIORS (owner 2026-11-14): room-axis yaw, back on the nearest wall — for a class that
                    // declares a back-face attachment only (F1 round 9 a, rc::birth_on_wall); any other keeps this cold-start seed.
                    birth_pos_cov = rc::seed_birth(s0, pts, z_lo, z_hi, cfg_, have_room_geometry_,
                                                   [this](const Eigen::Vector2f& q) { return nearest_wall(q); }, inst.node_name);
```
```
BEFORE (from line 1108)

AFTER (from line 1215)
        p.range_noise_lat_per_m = cfg_.ai2_range_noise_lat_per_m;
```
```
BEFORE (from line 1113)

AFTER (from line 1221)
        p.prior_depth_std     = cfg_.ai2_prior_depth_std;
```
```
BEFORE (from line 1114)

AFTER (from line 1223)
        p.vertical_extent_m   = cfg_.vertical_extent_m;   // the manifest's support: floor-anchored or hanging (K1)
```
```
BEFORE (from line 1115)

AFTER (from line 1225)
        p.symmetric           = cfg_.shape_symmetric;   // K2c: [shape] symmetry = "c2v" (a table: the four yaw modes are one body)
```
```
BEFORE (from line 1118)

AFTER (from line 1229)
        p.shape_parts         = cfg_.shape_parts;   // the solid: one table for the SDF, the beam parts and the carve (K2a)
        p.solid_parts         = cfg_.solid_parts;
        p.shape_proportions   = cfg_.shape_proportions;   // the class proportions, ρ = 1 latents (K2a-L)
```
```
BEFORE (from line 1129)

AFTER (from line 1243)
        p.wall_standoff_m         = cfg_.wall_standoff_m;       // [prior.attachment] standoff (F1.4; the persistent path's flush prior)
        p.wall_standoff_std_m     = cfg_.wall_standoff_std_m;
        p.move_rate_per_day       = cfg_.move_rate_per_day;     // [prior.permanence] (F1 round 6): the volatility's hyperprior
        p.move_scale_m            = cfg_.move_scale_m;
        p.move_scale_rad          = cfg_.move_scale_rad;
        if (cfg_.ai2_age_nominal_dt_s > 0.0f) p.predict_period_s = cfg_.ai2_age_nominal_dt_s;
```
```
BEFORE (from line 1170)

AFTER (from line 1290)
        p.layer_spread_m = cfg_.layer_spread_m;        // the class's layer spread Δ (manifest; the persistent path's shape latent)
```
```
BEFORE (from line 1172)

AFTER (from line 1293)
        if (p.persistent and not birth_cloud.empty())
        {   // F1 round 6.2: the seed H from the model the update uses (class-prior MAP ⊕ the top of the points the BOX explains,
            // box-vs-headboard shares at the born pose), not the p95 of a cloud that holds the headboard (bed: 1.02 vs 0.65).
            const float H_p95 = s0.H;
            s0.H = inst.ai2_belief.seed_height(birth_cloud, rc::kZedYoloBoundaryStdM);
            inst.ai2_belief = CountableBelief(s0, p);
            std::print("[{}] birth seed H: p95 of the cloud {:.3f} → box-explained top ⊕ class prior {:.3f}\n", inst.node_name, H_p95, s0.H);
        }
```
```
BEFORE (from line 1173)

AFTER (from line 1302)
        inst.replay.reset();                                                           // a new birth starts a new stream
        inst.offer_cache.reset();                                                      // a new belief: its update serial restarts
        if (cfg_.replay_dump and p.persistent) open_replay(inst, s0, p, birth_pos_cov);
```
```
BEFORE (from line 1210)

AFTER (from line 1342)
            if (inst.replay) inst.replay->inflate(dt, cfg_.ai2_age_nominal_dt_s);
```
```
BEFORE (from line 1262)

AFTER (from line 1395)
    // GATE AUDIT (plan rev 2 F1.8, persistent path; .superpowers/sdd/2026-11-23-shared-engine/task-F1-report.md): both gates
    // are KEPT, each with its reason, neither replaced on hope.
    //  - truncation (AI2TruncGateFrac 0.10): fired on 0 of 1034 bed and 0 of 2448 nightstand rows in objects_replay1. Under the
    //    point-to-surface likelihood a clipped mask gives fewer, not biased, surface points; what IS biased is z_top_obs (the
    //    p97 top of a cloud clipped at the top border) and the birth extent (F1.3 handles birth). Model replacement: z_top_obs
    //    as a censored observation when the clip is at the top. Not done: no recorded frame exercises it.
    //  - confirm_only (moving AND off-centre): 959 of 1034 bed rows gated, 324 of them with a mask (LiDAR-only instead). The
    //    fresh latent u_cam models the same ego-motion smear, so the two count it twice. Model replacement: u_cam alone.
    //    Not done: the replay cannot see it (a gated cycle records only its LiDAR-only frame, not the camera frame it skipped)
    //    and u_cam's MotionCm gains are unmeasured (τ, T5 ruling) — removing the gate would hand the smear to an unmeasured
    //    term. Next: record the skipped camera frame in the replay tap, and measure τ_eff (Fable review 2.8).
```
```
BEFORE (from line 1304)
    // The latents' marginals (§7c.2): b_cam = the config floors + range lateral/size + the obliquity yaw cap (today's camera
    // fill WITHOUT chain and ego-motion); b_L = the config floors. e's marginal is Σ_pose, staged by stage_persistent (§7c.7).
    // The ego-motion terms are FRESH every frame (ρ = 0): the belief's per-frame latents u_cam / u_L (spec 7c.8 rev 3.3.13).
    const auto latent_inputs = [&](float obliquity_yaw_var)
AFTER (from line 1448)
    // The latents' marginals (§7c.2; shared-engine G1, controller ruling for Task 5): b_L = the constant metric common-mode stds
    // squared (AI2CommonMode{Pos,Size,Yaw}Std); b_cam = the measured ZED YOLO-seg boundary std on position and size (plan rev 2
    // F1.6, rc::kZedYoloBoundaryStdM) and the configured yaw std. The range growth (AI2RangeNoise{Yaw,Size}PerM, and the lat
    // term on b_cam) and the obliquity yaw term (kObliquityYawGain with its cos clamp) are NOT in them: the mask boundary
    // error is shape-dominated and neither pixel- nor metre-constant (step 0, docs/superpowers/results/2026-11-24-step0.md),
    // so no measured sensor row replaces them yet and they are dropped from the persistent path rather than kept as patch.
    // The pose-chain share is e (marginal Σ_pose, staged by stage_persistent, §7c.7), as before. The ego-motion terms
    // (MotionCm* · motion_magnitude · periphery, unchanged) are FRESH every frame (ρ = 0): u_cam / u_L (rev 3.3.13).
    // The computation is rc::persistent_latent_inputs (countable_latent_inputs.h), tested by latent_inputs_test.
    const auto latent_inputs = [&]
```
```
BEFORE (from line 1309)
        const float p2 = cfg_.ai2_common_mode_pos_std  * cfg_.ai2_common_mode_pos_std;
        const float s2 = cfg_.ai2_common_mode_size_std * cfg_.ai2_common_mode_size_std;
        const float y2 = cfg_.ai2_common_mode_yaw_std  * cfg_.ai2_common_mode_yaw_std;
        LatentInputs li;
        li.sig2_cam = {p2 + range_lat_var, p2 + range_lat_var, s2 + range_size_var, s2 + range_size_var,
                       s2 + range_size_var, y2 + range_yaw_var + obliquity_yaw_var};
        li.sig2_L   = {p2, p2, s2, s2, s2, y2};
        li.fresh    = {mot_pos_var, mot_pos_var, mot_size_var, mot_size_var, mot_size_var, mot_yaw_var};
        return li;
AFTER (from line 1459)
        rc::LatentView v;
        v.range_m = range; v.obliquity_cos = inst.dbg_obliquity_cos;
        v.chain_var_xx = inst.chain_cov_xx; v.chain_var_yy = inst.chain_cov_yy;
        v.mot_pos_var = mot_pos_var; v.mot_size_var = mot_size_var; v.mot_yaw_var = mot_yaw_var;
        return rc::persistent_latent_inputs({cfg_.ai2_common_mode_pos_std, cfg_.ai2_common_mode_size_std,
                                             cfg_.ai2_common_mode_yaw_std}, v);
```
```
BEFORE (from line 1344)

AFTER (from line 1491)
        log_delivered(inst);                       // G5: what this accepted update delivered
```
```
BEFORE (from line 1352)

AFTER (from line 1500)
            if (inst.replay) inst.replay->room_interior(room_interior_);
```
```
BEFORE (from line 1354)
            frame.wall = nearest_wall(inst.ai2_belief.back_centre(bs_now));
AFTER (from line 1503)
            // K2c: a symmetric class has no back face, so its wall is the one nearest its CENTRE (invariant under the c2v group;
            // the back centre moves to another face in another sheet). The staged wall feeds the explain-away component and δ.
            frame.wall = nearest_wall(inst.ai2_belief.params().symmetric ? Eigen::Vector2f(bs_now.cx, bs_now.cy)
                                                                         : inst.ai2_belief.back_centre(bs_now));
```
```
BEFORE (from line 1372)

AFTER (from line 1524)
        if (inst.replay) inst.replay->update(inst.processed_cycles, gated, frame);
```
(remaining changed hunks of this function trimmed)
</details>


<details><summary>A2288: context changed, function `float CountableFitter::run_inference(CountableInstance& inst, const CountableObservation& observation)` (685 lines before, 757 after)</summary>

Evidence line (before, countable_concept/src/countable_fitter.cpp:1075): `const float z_hi = s0.H + 3.0f * cfg_.ai2_sigma_base_m;`; found unchanged at countable_concept/src/countable_fitter.cpp:1180.

```
BEFORE (from line 1040)

AFTER (from line 1135)
        std::vector<Eigen::Vector3f> birth_cloud;   // F1 round 6.2: the seed H is re-derived from it once the belief knows its shape
```
```
BEFORE (from line 1042)

AFTER (from line 1138)
            birth_cloud.reserve(static_cast<std::size_t>(npts));
            birth_cloud.insert(birth_cloud.end(), observation.candidate_pts.begin(), observation.candidate_pts.end());
            birth_cloud.insert(birth_cloud.end(), observation.residual_pts.begin(), observation.residual_pts.end());
```
```
BEFORE (from line 1066)
            // Skip on a TRIMMED mask (extent is only a lower bound → would birth too small); wait for a full mask.
            if (cfg_.footprint_moment_precision > 0.0f and inst.last_trunc_frac <= 1e-3f)
AFTER (from line 1165)
            // Legacy path: skip on a TRIMMED mask (extent is only a lower bound → would birth too small); wait for a full mask.
            // Persistent path (plan rev 2 F1.3): no switch. The engine enters the class prior at the class mean, so the seed is
            // only the GN start, and a clipped mask enters as what it is — a seen extent shorter than the class's — through
            // birth_on_priors' position covariance, (class extent − seen extent)²/12 along the wall (countable_birth_footprint.h).
            // engine1's bed: trunc 0.23 % skipped the whole seed and GN started 0.8 m short of the bed's centre.
            if (cfg_.footprint_moment_precision > 0.0f and (cfg_.lidar_measures_on_gated or inst.last_trunc_frac <= 1e-3f))
```
```
BEFORE (from line 1075)

AFTER (from line 1178)
                // The band's base is the body's underside: the floor for a floor-anchored class, H − e for a hanging one (a
                // point under a hood is the hob or the worktop, not the hood).
```
```
BEFORE (from line 1076)
                const float z_lo = 3.0f * cfg_.ai2_sigma_base_m;
AFTER (from line 1181)
                const float z_lo = rc::countable::VerticalSpan{cfg_.vertical_extent_m}.z0(s0.H) + 3.0f * cfg_.ai2_sigma_base_m;
```
```
BEFORE (from line 1101)
                    // ★BORN ON THE FURNITURE PRIORS (owner 2026-11-14): room-axis yaw, back on the nearest wall.
                    if (have_room_geometry_) birth_pos_cov = birth_on_priors(s0, pts, z_lo, z_hi, inst.node_name);
AFTER (from line 1206)
                    // ★BORN ON THE FURNITURE PRIORS (owner 2026-11-14): room-axis yaw, back on the nearest wall — for a class that
                    // declares a back-face attachment only (F1 round 9 a, rc::birth_on_wall); any other keeps this cold-start seed.
                    birth_pos_cov = rc::seed_birth(s0, pts, z_lo, z_hi, cfg_, have_room_geometry_,
                                                   [this](const Eigen::Vector2f& q) { return nearest_wall(q); }, inst.node_name);
```
```
BEFORE (from line 1108)

AFTER (from line 1215)
        p.range_noise_lat_per_m = cfg_.ai2_range_noise_lat_per_m;
```
```
BEFORE (from line 1113)

AFTER (from line 1221)
        p.prior_depth_std     = cfg_.ai2_prior_depth_std;
```
```
BEFORE (from line 1114)

AFTER (from line 1223)
        p.vertical_extent_m   = cfg_.vertical_extent_m;   // the manifest's support: floor-anchored or hanging (K1)
```
```
BEFORE (from line 1115)

AFTER (from line 1225)
        p.symmetric           = cfg_.shape_symmetric;   // K2c: [shape] symmetry = "c2v" (a table: the four yaw modes are one body)
```
```
BEFORE (from line 1118)

AFTER (from line 1229)
        p.shape_parts         = cfg_.shape_parts;   // the solid: one table for the SDF, the beam parts and the carve (K2a)
        p.solid_parts         = cfg_.solid_parts;
        p.shape_proportions   = cfg_.shape_proportions;   // the class proportions, ρ = 1 latents (K2a-L)
```
```
BEFORE (from line 1129)

AFTER (from line 1243)
        p.wall_standoff_m         = cfg_.wall_standoff_m;       // [prior.attachment] standoff (F1.4; the persistent path's flush prior)
        p.wall_standoff_std_m     = cfg_.wall_standoff_std_m;
        p.move_rate_per_day       = cfg_.move_rate_per_day;     // [prior.permanence] (F1 round 6): the volatility's hyperprior
        p.move_scale_m            = cfg_.move_scale_m;
        p.move_scale_rad          = cfg_.move_scale_rad;
        if (cfg_.ai2_age_nominal_dt_s > 0.0f) p.predict_period_s = cfg_.ai2_age_nominal_dt_s;
```
```
BEFORE (from line 1170)

AFTER (from line 1290)
        p.layer_spread_m = cfg_.layer_spread_m;        // the class's layer spread Δ (manifest; the persistent path's shape latent)
```
```
BEFORE (from line 1172)

AFTER (from line 1293)
        if (p.persistent and not birth_cloud.empty())
        {   // F1 round 6.2: the seed H from the model the update uses (class-prior MAP ⊕ the top of the points the BOX explains,
            // box-vs-headboard shares at the born pose), not the p95 of a cloud that holds the headboard (bed: 1.02 vs 0.65).
            const float H_p95 = s0.H;
            s0.H = inst.ai2_belief.seed_height(birth_cloud, rc::kZedYoloBoundaryStdM);
            inst.ai2_belief = CountableBelief(s0, p);
            std::print("[{}] birth seed H: p95 of the cloud {:.3f} → box-explained top ⊕ class prior {:.3f}\n", inst.node_name, H_p95, s0.H);
        }
```
```
BEFORE (from line 1173)

AFTER (from line 1302)
        inst.replay.reset();                                                           // a new birth starts a new stream
        inst.offer_cache.reset();                                                      // a new belief: its update serial restarts
        if (cfg_.replay_dump and p.persistent) open_replay(inst, s0, p, birth_pos_cov);
```
```
BEFORE (from line 1210)

AFTER (from line 1342)
            if (inst.replay) inst.replay->inflate(dt, cfg_.ai2_age_nominal_dt_s);
```
```
BEFORE (from line 1262)

AFTER (from line 1395)
    // GATE AUDIT (plan rev 2 F1.8, persistent path; .superpowers/sdd/2026-11-23-shared-engine/task-F1-report.md): both gates
    // are KEPT, each with its reason, neither replaced on hope.
    //  - truncation (AI2TruncGateFrac 0.10): fired on 0 of 1034 bed and 0 of 2448 nightstand rows in objects_replay1. Under the
    //    point-to-surface likelihood a clipped mask gives fewer, not biased, surface points; what IS biased is z_top_obs (the
    //    p97 top of a cloud clipped at the top border) and the birth extent (F1.3 handles birth). Model replacement: z_top_obs
    //    as a censored observation when the clip is at the top. Not done: no recorded frame exercises it.
    //  - confirm_only (moving AND off-centre): 959 of 1034 bed rows gated, 324 of them with a mask (LiDAR-only instead). The
    //    fresh latent u_cam models the same ego-motion smear, so the two count it twice. Model replacement: u_cam alone.
    //    Not done: the replay cannot see it (a gated cycle records only its LiDAR-only frame, not the camera frame it skipped)
    //    and u_cam's MotionCm gains are unmeasured (τ, T5 ruling) — removing the gate would hand the smear to an unmeasured
    //    term. Next: record the skipped camera frame in the replay tap, and measure τ_eff (Fable review 2.8).
```
```
BEFORE (from line 1304)
    // The latents' marginals (§7c.2): b_cam = the config floors + range lateral/size + the obliquity yaw cap (today's camera
    // fill WITHOUT chain and ego-motion); b_L = the config floors. e's marginal is Σ_pose, staged by stage_persistent (§7c.7).
    // The ego-motion terms are FRESH every frame (ρ = 0): the belief's per-frame latents u_cam / u_L (spec 7c.8 rev 3.3.13).
    const auto latent_inputs = [&](float obliquity_yaw_var)
AFTER (from line 1448)
    // The latents' marginals (§7c.2; shared-engine G1, controller ruling for Task 5): b_L = the constant metric common-mode stds
    // squared (AI2CommonMode{Pos,Size,Yaw}Std); b_cam = the measured ZED YOLO-seg boundary std on position and size (plan rev 2
    // F1.6, rc::kZedYoloBoundaryStdM) and the configured yaw std. The range growth (AI2RangeNoise{Yaw,Size}PerM, and the lat
    // term on b_cam) and the obliquity yaw term (kObliquityYawGain with its cos clamp) are NOT in them: the mask boundary
    // error is shape-dominated and neither pixel- nor metre-constant (step 0, docs/superpowers/results/2026-11-24-step0.md),
    // so no measured sensor row replaces them yet and they are dropped from the persistent path rather than kept as patch.
    // The pose-chain share is e (marginal Σ_pose, staged by stage_persistent, §7c.7), as before. The ego-motion terms
    // (MotionCm* · motion_magnitude · periphery, unchanged) are FRESH every frame (ρ = 0): u_cam / u_L (rev 3.3.13).
    // The computation is rc::persistent_latent_inputs (countable_latent_inputs.h), tested by latent_inputs_test.
    const auto latent_inputs = [&]
```
```
BEFORE (from line 1309)
        const float p2 = cfg_.ai2_common_mode_pos_std  * cfg_.ai2_common_mode_pos_std;
        const float s2 = cfg_.ai2_common_mode_size_std * cfg_.ai2_common_mode_size_std;
        const float y2 = cfg_.ai2_common_mode_yaw_std  * cfg_.ai2_common_mode_yaw_std;
        LatentInputs li;
        li.sig2_cam = {p2 + range_lat_var, p2 + range_lat_var, s2 + range_size_var, s2 + range_size_var,
                       s2 + range_size_var, y2 + range_yaw_var + obliquity_yaw_var};
        li.sig2_L   = {p2, p2, s2, s2, s2, y2};
        li.fresh    = {mot_pos_var, mot_pos_var, mot_size_var, mot_size_var, mot_size_var, mot_yaw_var};
        return li;
AFTER (from line 1459)
        rc::LatentView v;
        v.range_m = range; v.obliquity_cos = inst.dbg_obliquity_cos;
        v.chain_var_xx = inst.chain_cov_xx; v.chain_var_yy = inst.chain_cov_yy;
        v.mot_pos_var = mot_pos_var; v.mot_size_var = mot_size_var; v.mot_yaw_var = mot_yaw_var;
        return rc::persistent_latent_inputs({cfg_.ai2_common_mode_pos_std, cfg_.ai2_common_mode_size_std,
                                             cfg_.ai2_common_mode_yaw_std}, v);
```
```
BEFORE (from line 1344)

AFTER (from line 1491)
        log_delivered(inst);                       // G5: what this accepted update delivered
```
```
BEFORE (from line 1352)

AFTER (from line 1500)
            if (inst.replay) inst.replay->room_interior(room_interior_);
```
```
BEFORE (from line 1354)
            frame.wall = nearest_wall(inst.ai2_belief.back_centre(bs_now));
AFTER (from line 1503)
            // K2c: a symmetric class has no back face, so its wall is the one nearest its CENTRE (invariant under the c2v group;
            // the back centre moves to another face in another sheet). The staged wall feeds the explain-away component and δ.
            frame.wall = nearest_wall(inst.ai2_belief.params().symmetric ? Eigen::Vector2f(bs_now.cx, bs_now.cy)
                                                                         : inst.ai2_belief.back_centre(bs_now));
```
```
BEFORE (from line 1372)

AFTER (from line 1524)
        if (inst.replay) inst.replay->update(inst.processed_cycles, gated, frame);
```
(remaining changed hunks of this function trimmed)
</details>


<details><summary>A2290: context changed, function `float CountableFitter::run_inference(CountableInstance& inst, const CountableObservation& observation)` (685 lines before, 757 after)</summary>

Evidence line (before, countable_concept/src/countable_fitter.cpp:1080): `s0.w = std::max(0.10f, mom.ext_major); // major axis = the seen face's width (reliable)`; found unchanged at countable_concept/src/countable_fitter.cpp:1185.

```
BEFORE (from line 1040)

AFTER (from line 1135)
        std::vector<Eigen::Vector3f> birth_cloud;   // F1 round 6.2: the seed H is re-derived from it once the belief knows its shape
```
```
BEFORE (from line 1042)

AFTER (from line 1138)
            birth_cloud.reserve(static_cast<std::size_t>(npts));
            birth_cloud.insert(birth_cloud.end(), observation.candidate_pts.begin(), observation.candidate_pts.end());
            birth_cloud.insert(birth_cloud.end(), observation.residual_pts.begin(), observation.residual_pts.end());
```
```
BEFORE (from line 1066)
            // Skip on a TRIMMED mask (extent is only a lower bound → would birth too small); wait for a full mask.
            if (cfg_.footprint_moment_precision > 0.0f and inst.last_trunc_frac <= 1e-3f)
AFTER (from line 1165)
            // Legacy path: skip on a TRIMMED mask (extent is only a lower bound → would birth too small); wait for a full mask.
            // Persistent path (plan rev 2 F1.3): no switch. The engine enters the class prior at the class mean, so the seed is
            // only the GN start, and a clipped mask enters as what it is — a seen extent shorter than the class's — through
            // birth_on_priors' position covariance, (class extent − seen extent)²/12 along the wall (countable_birth_footprint.h).
            // engine1's bed: trunc 0.23 % skipped the whole seed and GN started 0.8 m short of the bed's centre.
            if (cfg_.footprint_moment_precision > 0.0f and (cfg_.lidar_measures_on_gated or inst.last_trunc_frac <= 1e-3f))
```
```
BEFORE (from line 1075)

AFTER (from line 1178)
                // The band's base is the body's underside: the floor for a floor-anchored class, H − e for a hanging one (a
                // point under a hood is the hob or the worktop, not the hood).
```
```
BEFORE (from line 1076)
                const float z_lo = 3.0f * cfg_.ai2_sigma_base_m;
AFTER (from line 1181)
                const float z_lo = rc::countable::VerticalSpan{cfg_.vertical_extent_m}.z0(s0.H) + 3.0f * cfg_.ai2_sigma_base_m;
```
```
BEFORE (from line 1101)
                    // ★BORN ON THE FURNITURE PRIORS (owner 2026-11-14): room-axis yaw, back on the nearest wall.
                    if (have_room_geometry_) birth_pos_cov = birth_on_priors(s0, pts, z_lo, z_hi, inst.node_name);
AFTER (from line 1206)
                    // ★BORN ON THE FURNITURE PRIORS (owner 2026-11-14): room-axis yaw, back on the nearest wall — for a class that
                    // declares a back-face attachment only (F1 round 9 a, rc::birth_on_wall); any other keeps this cold-start seed.
                    birth_pos_cov = rc::seed_birth(s0, pts, z_lo, z_hi, cfg_, have_room_geometry_,
                                                   [this](const Eigen::Vector2f& q) { return nearest_wall(q); }, inst.node_name);
```
```
BEFORE (from line 1108)

AFTER (from line 1215)
        p.range_noise_lat_per_m = cfg_.ai2_range_noise_lat_per_m;
```
```
BEFORE (from line 1113)

AFTER (from line 1221)
        p.prior_depth_std     = cfg_.ai2_prior_depth_std;
```
```
BEFORE (from line 1114)

AFTER (from line 1223)
        p.vertical_extent_m   = cfg_.vertical_extent_m;   // the manifest's support: floor-anchored or hanging (K1)
```
```
BEFORE (from line 1115)

AFTER (from line 1225)
        p.symmetric           = cfg_.shape_symmetric;   // K2c: [shape] symmetry = "c2v" (a table: the four yaw modes are one body)
```
```
BEFORE (from line 1118)

AFTER (from line 1229)
        p.shape_parts         = cfg_.shape_parts;   // the solid: one table for the SDF, the beam parts and the carve (K2a)
        p.solid_parts         = cfg_.solid_parts;
        p.shape_proportions   = cfg_.shape_proportions;   // the class proportions, ρ = 1 latents (K2a-L)
```
```
BEFORE (from line 1129)

AFTER (from line 1243)
        p.wall_standoff_m         = cfg_.wall_standoff_m;       // [prior.attachment] standoff (F1.4; the persistent path's flush prior)
        p.wall_standoff_std_m     = cfg_.wall_standoff_std_m;
        p.move_rate_per_day       = cfg_.move_rate_per_day;     // [prior.permanence] (F1 round 6): the volatility's hyperprior
        p.move_scale_m            = cfg_.move_scale_m;
        p.move_scale_rad          = cfg_.move_scale_rad;
        if (cfg_.ai2_age_nominal_dt_s > 0.0f) p.predict_period_s = cfg_.ai2_age_nominal_dt_s;
```
```
BEFORE (from line 1170)

AFTER (from line 1290)
        p.layer_spread_m = cfg_.layer_spread_m;        // the class's layer spread Δ (manifest; the persistent path's shape latent)
```
```
BEFORE (from line 1172)

AFTER (from line 1293)
        if (p.persistent and not birth_cloud.empty())
        {   // F1 round 6.2: the seed H from the model the update uses (class-prior MAP ⊕ the top of the points the BOX explains,
            // box-vs-headboard shares at the born pose), not the p95 of a cloud that holds the headboard (bed: 1.02 vs 0.65).
            const float H_p95 = s0.H;
            s0.H = inst.ai2_belief.seed_height(birth_cloud, rc::kZedYoloBoundaryStdM);
            inst.ai2_belief = CountableBelief(s0, p);
            std::print("[{}] birth seed H: p95 of the cloud {:.3f} → box-explained top ⊕ class prior {:.3f}\n", inst.node_name, H_p95, s0.H);
        }
```
```
BEFORE (from line 1173)

AFTER (from line 1302)
        inst.replay.reset();                                                           // a new birth starts a new stream
        inst.offer_cache.reset();                                                      // a new belief: its update serial restarts
        if (cfg_.replay_dump and p.persistent) open_replay(inst, s0, p, birth_pos_cov);
```
```
BEFORE (from line 1210)

AFTER (from line 1342)
            if (inst.replay) inst.replay->inflate(dt, cfg_.ai2_age_nominal_dt_s);
```
```
BEFORE (from line 1262)

AFTER (from line 1395)
    // GATE AUDIT (plan rev 2 F1.8, persistent path; .superpowers/sdd/2026-11-23-shared-engine/task-F1-report.md): both gates
    // are KEPT, each with its reason, neither replaced on hope.
    //  - truncation (AI2TruncGateFrac 0.10): fired on 0 of 1034 bed and 0 of 2448 nightstand rows in objects_replay1. Under the
    //    point-to-surface likelihood a clipped mask gives fewer, not biased, surface points; what IS biased is z_top_obs (the
    //    p97 top of a cloud clipped at the top border) and the birth extent (F1.3 handles birth). Model replacement: z_top_obs
    //    as a censored observation when the clip is at the top. Not done: no recorded frame exercises it.
    //  - confirm_only (moving AND off-centre): 959 of 1034 bed rows gated, 324 of them with a mask (LiDAR-only instead). The
    //    fresh latent u_cam models the same ego-motion smear, so the two count it twice. Model replacement: u_cam alone.
    //    Not done: the replay cannot see it (a gated cycle records only its LiDAR-only frame, not the camera frame it skipped)
    //    and u_cam's MotionCm gains are unmeasured (τ, T5 ruling) — removing the gate would hand the smear to an unmeasured
    //    term. Next: record the skipped camera frame in the replay tap, and measure τ_eff (Fable review 2.8).
```
```
BEFORE (from line 1304)
    // The latents' marginals (§7c.2): b_cam = the config floors + range lateral/size + the obliquity yaw cap (today's camera
    // fill WITHOUT chain and ego-motion); b_L = the config floors. e's marginal is Σ_pose, staged by stage_persistent (§7c.7).
    // The ego-motion terms are FRESH every frame (ρ = 0): the belief's per-frame latents u_cam / u_L (spec 7c.8 rev 3.3.13).
    const auto latent_inputs = [&](float obliquity_yaw_var)
AFTER (from line 1448)
    // The latents' marginals (§7c.2; shared-engine G1, controller ruling for Task 5): b_L = the constant metric common-mode stds
    // squared (AI2CommonMode{Pos,Size,Yaw}Std); b_cam = the measured ZED YOLO-seg boundary std on position and size (plan rev 2
    // F1.6, rc::kZedYoloBoundaryStdM) and the configured yaw std. The range growth (AI2RangeNoise{Yaw,Size}PerM, and the lat
    // term on b_cam) and the obliquity yaw term (kObliquityYawGain with its cos clamp) are NOT in them: the mask boundary
    // error is shape-dominated and neither pixel- nor metre-constant (step 0, docs/superpowers/results/2026-11-24-step0.md),
    // so no measured sensor row replaces them yet and they are dropped from the persistent path rather than kept as patch.
    // The pose-chain share is e (marginal Σ_pose, staged by stage_persistent, §7c.7), as before. The ego-motion terms
    // (MotionCm* · motion_magnitude · periphery, unchanged) are FRESH every frame (ρ = 0): u_cam / u_L (rev 3.3.13).
    // The computation is rc::persistent_latent_inputs (countable_latent_inputs.h), tested by latent_inputs_test.
    const auto latent_inputs = [&]
```
```
BEFORE (from line 1309)
        const float p2 = cfg_.ai2_common_mode_pos_std  * cfg_.ai2_common_mode_pos_std;
        const float s2 = cfg_.ai2_common_mode_size_std * cfg_.ai2_common_mode_size_std;
        const float y2 = cfg_.ai2_common_mode_yaw_std  * cfg_.ai2_common_mode_yaw_std;
        LatentInputs li;
        li.sig2_cam = {p2 + range_lat_var, p2 + range_lat_var, s2 + range_size_var, s2 + range_size_var,
                       s2 + range_size_var, y2 + range_yaw_var + obliquity_yaw_var};
        li.sig2_L   = {p2, p2, s2, s2, s2, y2};
        li.fresh    = {mot_pos_var, mot_pos_var, mot_size_var, mot_size_var, mot_size_var, mot_yaw_var};
        return li;
AFTER (from line 1459)
        rc::LatentView v;
        v.range_m = range; v.obliquity_cos = inst.dbg_obliquity_cos;
        v.chain_var_xx = inst.chain_cov_xx; v.chain_var_yy = inst.chain_cov_yy;
        v.mot_pos_var = mot_pos_var; v.mot_size_var = mot_size_var; v.mot_yaw_var = mot_yaw_var;
        return rc::persistent_latent_inputs({cfg_.ai2_common_mode_pos_std, cfg_.ai2_common_mode_size_std,
                                             cfg_.ai2_common_mode_yaw_std}, v);
```
```
BEFORE (from line 1344)

AFTER (from line 1491)
        log_delivered(inst);                       // G5: what this accepted update delivered
```
```
BEFORE (from line 1352)

AFTER (from line 1500)
            if (inst.replay) inst.replay->room_interior(room_interior_);
```
```
BEFORE (from line 1354)
            frame.wall = nearest_wall(inst.ai2_belief.back_centre(bs_now));
AFTER (from line 1503)
            // K2c: a symmetric class has no back face, so its wall is the one nearest its CENTRE (invariant under the c2v group;
            // the back centre moves to another face in another sheet). The staged wall feeds the explain-away component and δ.
            frame.wall = nearest_wall(inst.ai2_belief.params().symmetric ? Eigen::Vector2f(bs_now.cx, bs_now.cy)
                                                                         : inst.ai2_belief.back_centre(bs_now));
```
```
BEFORE (from line 1372)

AFTER (from line 1524)
        if (inst.replay) inst.replay->update(inst.processed_cycles, gated, frame);
```
(remaining changed hunks of this function trimmed)
</details>


<details><summary>A2289: context changed, function `float CountableFitter::run_inference(CountableInstance& inst, const CountableObservation& observation)` (685 lines before, 757 after)</summary>

Evidence line (before, countable_concept/src/countable_fitter.cpp:1099): `if (sum > 1e-4f and (mom.ext_major - mom.ext_minor) / sum > 0.10f)`; found unchanged at countable_concept/src/countable_fitter.cpp:1204.

```
BEFORE (from line 1040)

AFTER (from line 1135)
        std::vector<Eigen::Vector3f> birth_cloud;   // F1 round 6.2: the seed H is re-derived from it once the belief knows its shape
```
```
BEFORE (from line 1042)

AFTER (from line 1138)
            birth_cloud.reserve(static_cast<std::size_t>(npts));
            birth_cloud.insert(birth_cloud.end(), observation.candidate_pts.begin(), observation.candidate_pts.end());
            birth_cloud.insert(birth_cloud.end(), observation.residual_pts.begin(), observation.residual_pts.end());
```
```
BEFORE (from line 1066)
            // Skip on a TRIMMED mask (extent is only a lower bound → would birth too small); wait for a full mask.
            if (cfg_.footprint_moment_precision > 0.0f and inst.last_trunc_frac <= 1e-3f)
AFTER (from line 1165)
            // Legacy path: skip on a TRIMMED mask (extent is only a lower bound → would birth too small); wait for a full mask.
            // Persistent path (plan rev 2 F1.3): no switch. The engine enters the class prior at the class mean, so the seed is
            // only the GN start, and a clipped mask enters as what it is — a seen extent shorter than the class's — through
            // birth_on_priors' position covariance, (class extent − seen extent)²/12 along the wall (countable_birth_footprint.h).
            // engine1's bed: trunc 0.23 % skipped the whole seed and GN started 0.8 m short of the bed's centre.
            if (cfg_.footprint_moment_precision > 0.0f and (cfg_.lidar_measures_on_gated or inst.last_trunc_frac <= 1e-3f))
```
```
BEFORE (from line 1075)

AFTER (from line 1178)
                // The band's base is the body's underside: the floor for a floor-anchored class, H − e for a hanging one (a
                // point under a hood is the hob or the worktop, not the hood).
```
```
BEFORE (from line 1076)
                const float z_lo = 3.0f * cfg_.ai2_sigma_base_m;
AFTER (from line 1181)
                const float z_lo = rc::countable::VerticalSpan{cfg_.vertical_extent_m}.z0(s0.H) + 3.0f * cfg_.ai2_sigma_base_m;
```
```
BEFORE (from line 1101)
                    // ★BORN ON THE FURNITURE PRIORS (owner 2026-11-14): room-axis yaw, back on the nearest wall.
                    if (have_room_geometry_) birth_pos_cov = birth_on_priors(s0, pts, z_lo, z_hi, inst.node_name);
AFTER (from line 1206)
                    // ★BORN ON THE FURNITURE PRIORS (owner 2026-11-14): room-axis yaw, back on the nearest wall — for a class that
                    // declares a back-face attachment only (F1 round 9 a, rc::birth_on_wall); any other keeps this cold-start seed.
                    birth_pos_cov = rc::seed_birth(s0, pts, z_lo, z_hi, cfg_, have_room_geometry_,
                                                   [this](const Eigen::Vector2f& q) { return nearest_wall(q); }, inst.node_name);
```
```
BEFORE (from line 1108)

AFTER (from line 1215)
        p.range_noise_lat_per_m = cfg_.ai2_range_noise_lat_per_m;
```
```
BEFORE (from line 1113)

AFTER (from line 1221)
        p.prior_depth_std     = cfg_.ai2_prior_depth_std;
```
```
BEFORE (from line 1114)

AFTER (from line 1223)
        p.vertical_extent_m   = cfg_.vertical_extent_m;   // the manifest's support: floor-anchored or hanging (K1)
```
```
BEFORE (from line 1115)

AFTER (from line 1225)
        p.symmetric           = cfg_.shape_symmetric;   // K2c: [shape] symmetry = "c2v" (a table: the four yaw modes are one body)
```
```
BEFORE (from line 1118)

AFTER (from line 1229)
        p.shape_parts         = cfg_.shape_parts;   // the solid: one table for the SDF, the beam parts and the carve (K2a)
        p.solid_parts         = cfg_.solid_parts;
        p.shape_proportions   = cfg_.shape_proportions;   // the class proportions, ρ = 1 latents (K2a-L)
```
```
BEFORE (from line 1129)

AFTER (from line 1243)
        p.wall_standoff_m         = cfg_.wall_standoff_m;       // [prior.attachment] standoff (F1.4; the persistent path's flush prior)
        p.wall_standoff_std_m     = cfg_.wall_standoff_std_m;
        p.move_rate_per_day       = cfg_.move_rate_per_day;     // [prior.permanence] (F1 round 6): the volatility's hyperprior
        p.move_scale_m            = cfg_.move_scale_m;
        p.move_scale_rad          = cfg_.move_scale_rad;
        if (cfg_.ai2_age_nominal_dt_s > 0.0f) p.predict_period_s = cfg_.ai2_age_nominal_dt_s;
```
```
BEFORE (from line 1170)

AFTER (from line 1290)
        p.layer_spread_m = cfg_.layer_spread_m;        // the class's layer spread Δ (manifest; the persistent path's shape latent)
```
```
BEFORE (from line 1172)

AFTER (from line 1293)
        if (p.persistent and not birth_cloud.empty())
        {   // F1 round 6.2: the seed H from the model the update uses (class-prior MAP ⊕ the top of the points the BOX explains,
            // box-vs-headboard shares at the born pose), not the p95 of a cloud that holds the headboard (bed: 1.02 vs 0.65).
            const float H_p95 = s0.H;
            s0.H = inst.ai2_belief.seed_height(birth_cloud, rc::kZedYoloBoundaryStdM);
            inst.ai2_belief = CountableBelief(s0, p);
            std::print("[{}] birth seed H: p95 of the cloud {:.3f} → box-explained top ⊕ class prior {:.3f}\n", inst.node_name, H_p95, s0.H);
        }
```
```
BEFORE (from line 1173)

AFTER (from line 1302)
        inst.replay.reset();                                                           // a new birth starts a new stream
        inst.offer_cache.reset();                                                      // a new belief: its update serial restarts
        if (cfg_.replay_dump and p.persistent) open_replay(inst, s0, p, birth_pos_cov);
```
```
BEFORE (from line 1210)

AFTER (from line 1342)
            if (inst.replay) inst.replay->inflate(dt, cfg_.ai2_age_nominal_dt_s);
```
```
BEFORE (from line 1262)

AFTER (from line 1395)
    // GATE AUDIT (plan rev 2 F1.8, persistent path; .superpowers/sdd/2026-11-23-shared-engine/task-F1-report.md): both gates
    // are KEPT, each with its reason, neither replaced on hope.
    //  - truncation (AI2TruncGateFrac 0.10): fired on 0 of 1034 bed and 0 of 2448 nightstand rows in objects_replay1. Under the
    //    point-to-surface likelihood a clipped mask gives fewer, not biased, surface points; what IS biased is z_top_obs (the
    //    p97 top of a cloud clipped at the top border) and the birth extent (F1.3 handles birth). Model replacement: z_top_obs
    //    as a censored observation when the clip is at the top. Not done: no recorded frame exercises it.
    //  - confirm_only (moving AND off-centre): 959 of 1034 bed rows gated, 324 of them with a mask (LiDAR-only instead). The
    //    fresh latent u_cam models the same ego-motion smear, so the two count it twice. Model replacement: u_cam alone.
    //    Not done: the replay cannot see it (a gated cycle records only its LiDAR-only frame, not the camera frame it skipped)
    //    and u_cam's MotionCm gains are unmeasured (τ, T5 ruling) — removing the gate would hand the smear to an unmeasured
    //    term. Next: record the skipped camera frame in the replay tap, and measure τ_eff (Fable review 2.8).
```
```
BEFORE (from line 1304)
    // The latents' marginals (§7c.2): b_cam = the config floors + range lateral/size + the obliquity yaw cap (today's camera
    // fill WITHOUT chain and ego-motion); b_L = the config floors. e's marginal is Σ_pose, staged by stage_persistent (§7c.7).
    // The ego-motion terms are FRESH every frame (ρ = 0): the belief's per-frame latents u_cam / u_L (spec 7c.8 rev 3.3.13).
    const auto latent_inputs = [&](float obliquity_yaw_var)
AFTER (from line 1448)
    // The latents' marginals (§7c.2; shared-engine G1, controller ruling for Task 5): b_L = the constant metric common-mode stds
    // squared (AI2CommonMode{Pos,Size,Yaw}Std); b_cam = the measured ZED YOLO-seg boundary std on position and size (plan rev 2
    // F1.6, rc::kZedYoloBoundaryStdM) and the configured yaw std. The range growth (AI2RangeNoise{Yaw,Size}PerM, and the lat
    // term on b_cam) and the obliquity yaw term (kObliquityYawGain with its cos clamp) are NOT in them: the mask boundary
    // error is shape-dominated and neither pixel- nor metre-constant (step 0, docs/superpowers/results/2026-11-24-step0.md),
    // so no measured sensor row replaces them yet and they are dropped from the persistent path rather than kept as patch.
    // The pose-chain share is e (marginal Σ_pose, staged by stage_persistent, §7c.7), as before. The ego-motion terms
    // (MotionCm* · motion_magnitude · periphery, unchanged) are FRESH every frame (ρ = 0): u_cam / u_L (rev 3.3.13).
    // The computation is rc::persistent_latent_inputs (countable_latent_inputs.h), tested by latent_inputs_test.
    const auto latent_inputs = [&]
```
```
BEFORE (from line 1309)
        const float p2 = cfg_.ai2_common_mode_pos_std  * cfg_.ai2_common_mode_pos_std;
        const float s2 = cfg_.ai2_common_mode_size_std * cfg_.ai2_common_mode_size_std;
        const float y2 = cfg_.ai2_common_mode_yaw_std  * cfg_.ai2_common_mode_yaw_std;
        LatentInputs li;
        li.sig2_cam = {p2 + range_lat_var, p2 + range_lat_var, s2 + range_size_var, s2 + range_size_var,
                       s2 + range_size_var, y2 + range_yaw_var + obliquity_yaw_var};
        li.sig2_L   = {p2, p2, s2, s2, s2, y2};
        li.fresh    = {mot_pos_var, mot_pos_var, mot_size_var, mot_size_var, mot_size_var, mot_yaw_var};
        return li;
AFTER (from line 1459)
        rc::LatentView v;
        v.range_m = range; v.obliquity_cos = inst.dbg_obliquity_cos;
        v.chain_var_xx = inst.chain_cov_xx; v.chain_var_yy = inst.chain_cov_yy;
        v.mot_pos_var = mot_pos_var; v.mot_size_var = mot_size_var; v.mot_yaw_var = mot_yaw_var;
        return rc::persistent_latent_inputs({cfg_.ai2_common_mode_pos_std, cfg_.ai2_common_mode_size_std,
                                             cfg_.ai2_common_mode_yaw_std}, v);
```
```
BEFORE (from line 1344)

AFTER (from line 1491)
        log_delivered(inst);                       // G5: what this accepted update delivered
```
```
BEFORE (from line 1352)

AFTER (from line 1500)
            if (inst.replay) inst.replay->room_interior(room_interior_);
```
```
BEFORE (from line 1354)
            frame.wall = nearest_wall(inst.ai2_belief.back_centre(bs_now));
AFTER (from line 1503)
            // K2c: a symmetric class has no back face, so its wall is the one nearest its CENTRE (invariant under the c2v group;
            // the back centre moves to another face in another sheet). The staged wall feeds the explain-away component and δ.
            frame.wall = nearest_wall(inst.ai2_belief.params().symmetric ? Eigen::Vector2f(bs_now.cx, bs_now.cy)
                                                                         : inst.ai2_belief.back_centre(bs_now));
```
```
BEFORE (from line 1372)

AFTER (from line 1524)
        if (inst.replay) inst.replay->update(inst.processed_cycles, gated, frame);
```
(remaining changed hunks of this function trimmed)
</details>


<details><summary>A2284: context changed, function `float CountableFitter::run_inference(CountableInstance& inst, const CountableObservation& observation)` (685 lines before, 757 after)</summary>

Evidence line (before, countable_concept/src/countable_fitter.cpp:1518): `constexpr float kObliquityYawGain = 0.05f; // ~30° σ_yaw at cos=0.09 → per-point GN holds yaw at grazing`; found unchanged at countable_concept/src/countable_fitter.cpp:1686.

```
BEFORE (from line 1040)

AFTER (from line 1135)
        std::vector<Eigen::Vector3f> birth_cloud;   // F1 round 6.2: the seed H is re-derived from it once the belief knows its shape
```
```
BEFORE (from line 1042)

AFTER (from line 1138)
            birth_cloud.reserve(static_cast<std::size_t>(npts));
            birth_cloud.insert(birth_cloud.end(), observation.candidate_pts.begin(), observation.candidate_pts.end());
            birth_cloud.insert(birth_cloud.end(), observation.residual_pts.begin(), observation.residual_pts.end());
```
```
BEFORE (from line 1066)
            // Skip on a TRIMMED mask (extent is only a lower bound → would birth too small); wait for a full mask.
            if (cfg_.footprint_moment_precision > 0.0f and inst.last_trunc_frac <= 1e-3f)
AFTER (from line 1165)
            // Legacy path: skip on a TRIMMED mask (extent is only a lower bound → would birth too small); wait for a full mask.
            // Persistent path (plan rev 2 F1.3): no switch. The engine enters the class prior at the class mean, so the seed is
            // only the GN start, and a clipped mask enters as what it is — a seen extent shorter than the class's — through
            // birth_on_priors' position covariance, (class extent − seen extent)²/12 along the wall (countable_birth_footprint.h).
            // engine1's bed: trunc 0.23 % skipped the whole seed and GN started 0.8 m short of the bed's centre.
            if (cfg_.footprint_moment_precision > 0.0f and (cfg_.lidar_measures_on_gated or inst.last_trunc_frac <= 1e-3f))
```
```
BEFORE (from line 1075)

AFTER (from line 1178)
                // The band's base is the body's underside: the floor for a floor-anchored class, H − e for a hanging one (a
                // point under a hood is the hob or the worktop, not the hood).
```
```
BEFORE (from line 1076)
                const float z_lo = 3.0f * cfg_.ai2_sigma_base_m;
AFTER (from line 1181)
                const float z_lo = rc::countable::VerticalSpan{cfg_.vertical_extent_m}.z0(s0.H) + 3.0f * cfg_.ai2_sigma_base_m;
```
```
BEFORE (from line 1101)
                    // ★BORN ON THE FURNITURE PRIORS (owner 2026-11-14): room-axis yaw, back on the nearest wall.
                    if (have_room_geometry_) birth_pos_cov = birth_on_priors(s0, pts, z_lo, z_hi, inst.node_name);
AFTER (from line 1206)
                    // ★BORN ON THE FURNITURE PRIORS (owner 2026-11-14): room-axis yaw, back on the nearest wall — for a class that
                    // declares a back-face attachment only (F1 round 9 a, rc::birth_on_wall); any other keeps this cold-start seed.
                    birth_pos_cov = rc::seed_birth(s0, pts, z_lo, z_hi, cfg_, have_room_geometry_,
                                                   [this](const Eigen::Vector2f& q) { return nearest_wall(q); }, inst.node_name);
```
```
BEFORE (from line 1108)

AFTER (from line 1215)
        p.range_noise_lat_per_m = cfg_.ai2_range_noise_lat_per_m;
```
```
BEFORE (from line 1113)

AFTER (from line 1221)
        p.prior_depth_std     = cfg_.ai2_prior_depth_std;
```
```
BEFORE (from line 1114)

AFTER (from line 1223)
        p.vertical_extent_m   = cfg_.vertical_extent_m;   // the manifest's support: floor-anchored or hanging (K1)
```
```
BEFORE (from line 1115)

AFTER (from line 1225)
        p.symmetric           = cfg_.shape_symmetric;   // K2c: [shape] symmetry = "c2v" (a table: the four yaw modes are one body)
```
```
BEFORE (from line 1118)

AFTER (from line 1229)
        p.shape_parts         = cfg_.shape_parts;   // the solid: one table for the SDF, the beam parts and the carve (K2a)
        p.solid_parts         = cfg_.solid_parts;
        p.shape_proportions   = cfg_.shape_proportions;   // the class proportions, ρ = 1 latents (K2a-L)
```
```
BEFORE (from line 1129)

AFTER (from line 1243)
        p.wall_standoff_m         = cfg_.wall_standoff_m;       // [prior.attachment] standoff (F1.4; the persistent path's flush prior)
        p.wall_standoff_std_m     = cfg_.wall_standoff_std_m;
        p.move_rate_per_day       = cfg_.move_rate_per_day;     // [prior.permanence] (F1 round 6): the volatility's hyperprior
        p.move_scale_m            = cfg_.move_scale_m;
        p.move_scale_rad          = cfg_.move_scale_rad;
        if (cfg_.ai2_age_nominal_dt_s > 0.0f) p.predict_period_s = cfg_.ai2_age_nominal_dt_s;
```
```
BEFORE (from line 1170)

AFTER (from line 1290)
        p.layer_spread_m = cfg_.layer_spread_m;        // the class's layer spread Δ (manifest; the persistent path's shape latent)
```
```
BEFORE (from line 1172)

AFTER (from line 1293)
        if (p.persistent and not birth_cloud.empty())
        {   // F1 round 6.2: the seed H from the model the update uses (class-prior MAP ⊕ the top of the points the BOX explains,
            // box-vs-headboard shares at the born pose), not the p95 of a cloud that holds the headboard (bed: 1.02 vs 0.65).
            const float H_p95 = s0.H;
            s0.H = inst.ai2_belief.seed_height(birth_cloud, rc::kZedYoloBoundaryStdM);
            inst.ai2_belief = CountableBelief(s0, p);
            std::print("[{}] birth seed H: p95 of the cloud {:.3f} → box-explained top ⊕ class prior {:.3f}\n", inst.node_name, H_p95, s0.H);
        }
```
```
BEFORE (from line 1173)

AFTER (from line 1302)
        inst.replay.reset();                                                           // a new birth starts a new stream
        inst.offer_cache.reset();                                                      // a new belief: its update serial restarts
        if (cfg_.replay_dump and p.persistent) open_replay(inst, s0, p, birth_pos_cov);
```
```
BEFORE (from line 1210)

AFTER (from line 1342)
            if (inst.replay) inst.replay->inflate(dt, cfg_.ai2_age_nominal_dt_s);
```
```
BEFORE (from line 1262)

AFTER (from line 1395)
    // GATE AUDIT (plan rev 2 F1.8, persistent path; .superpowers/sdd/2026-11-23-shared-engine/task-F1-report.md): both gates
    // are KEPT, each with its reason, neither replaced on hope.
    //  - truncation (AI2TruncGateFrac 0.10): fired on 0 of 1034 bed and 0 of 2448 nightstand rows in objects_replay1. Under the
    //    point-to-surface likelihood a clipped mask gives fewer, not biased, surface points; what IS biased is z_top_obs (the
    //    p97 top of a cloud clipped at the top border) and the birth extent (F1.3 handles birth). Model replacement: z_top_obs
    //    as a censored observation when the clip is at the top. Not done: no recorded frame exercises it.
    //  - confirm_only (moving AND off-centre): 959 of 1034 bed rows gated, 324 of them with a mask (LiDAR-only instead). The
    //    fresh latent u_cam models the same ego-motion smear, so the two count it twice. Model replacement: u_cam alone.
    //    Not done: the replay cannot see it (a gated cycle records only its LiDAR-only frame, not the camera frame it skipped)
    //    and u_cam's MotionCm gains are unmeasured (τ, T5 ruling) — removing the gate would hand the smear to an unmeasured
    //    term. Next: record the skipped camera frame in the replay tap, and measure τ_eff (Fable review 2.8).
```
```
BEFORE (from line 1304)
    // The latents' marginals (§7c.2): b_cam = the config floors + range lateral/size + the obliquity yaw cap (today's camera
    // fill WITHOUT chain and ego-motion); b_L = the config floors. e's marginal is Σ_pose, staged by stage_persistent (§7c.7).
    // The ego-motion terms are FRESH every frame (ρ = 0): the belief's per-frame latents u_cam / u_L (spec 7c.8 rev 3.3.13).
    const auto latent_inputs = [&](float obliquity_yaw_var)
AFTER (from line 1448)
    // The latents' marginals (§7c.2; shared-engine G1, controller ruling for Task 5): b_L = the constant metric common-mode stds
    // squared (AI2CommonMode{Pos,Size,Yaw}Std); b_cam = the measured ZED YOLO-seg boundary std on position and size (plan rev 2
    // F1.6, rc::kZedYoloBoundaryStdM) and the configured yaw std. The range growth (AI2RangeNoise{Yaw,Size}PerM, and the lat
    // term on b_cam) and the obliquity yaw term (kObliquityYawGain with its cos clamp) are NOT in them: the mask boundary
    // error is shape-dominated and neither pixel- nor metre-constant (step 0, docs/superpowers/results/2026-11-24-step0.md),
    // so no measured sensor row replaces them yet and they are dropped from the persistent path rather than kept as patch.
    // The pose-chain share is e (marginal Σ_pose, staged by stage_persistent, §7c.7), as before. The ego-motion terms
    // (MotionCm* · motion_magnitude · periphery, unchanged) are FRESH every frame (ρ = 0): u_cam / u_L (rev 3.3.13).
    // The computation is rc::persistent_latent_inputs (countable_latent_inputs.h), tested by latent_inputs_test.
    const auto latent_inputs = [&]
```
```
BEFORE (from line 1309)
        const float p2 = cfg_.ai2_common_mode_pos_std  * cfg_.ai2_common_mode_pos_std;
        const float s2 = cfg_.ai2_common_mode_size_std * cfg_.ai2_common_mode_size_std;
        const float y2 = cfg_.ai2_common_mode_yaw_std  * cfg_.ai2_common_mode_yaw_std;
        LatentInputs li;
        li.sig2_cam = {p2 + range_lat_var, p2 + range_lat_var, s2 + range_size_var, s2 + range_size_var,
                       s2 + range_size_var, y2 + range_yaw_var + obliquity_yaw_var};
        li.sig2_L   = {p2, p2, s2, s2, s2, y2};
        li.fresh    = {mot_pos_var, mot_pos_var, mot_size_var, mot_size_var, mot_size_var, mot_yaw_var};
        return li;
AFTER (from line 1459)
        rc::LatentView v;
        v.range_m = range; v.obliquity_cos = inst.dbg_obliquity_cos;
        v.chain_var_xx = inst.chain_cov_xx; v.chain_var_yy = inst.chain_cov_yy;
        v.mot_pos_var = mot_pos_var; v.mot_size_var = mot_size_var; v.mot_yaw_var = mot_yaw_var;
        return rc::persistent_latent_inputs({cfg_.ai2_common_mode_pos_std, cfg_.ai2_common_mode_size_std,
                                             cfg_.ai2_common_mode_yaw_std}, v);
```
```
BEFORE (from line 1344)

AFTER (from line 1491)
        log_delivered(inst);                       // G5: what this accepted update delivered
```
```
BEFORE (from line 1352)

AFTER (from line 1500)
            if (inst.replay) inst.replay->room_interior(room_interior_);
```
```
BEFORE (from line 1354)
            frame.wall = nearest_wall(inst.ai2_belief.back_centre(bs_now));
AFTER (from line 1503)
            // K2c: a symmetric class has no back face, so its wall is the one nearest its CENTRE (invariant under the c2v group;
            // the back centre moves to another face in another sheet). The staged wall feeds the explain-away component and δ.
            frame.wall = nearest_wall(inst.ai2_belief.params().symmetric ? Eigen::Vector2f(bs_now.cx, bs_now.cy)
                                                                         : inst.ai2_belief.back_centre(bs_now));
```
```
BEFORE (from line 1372)

AFTER (from line 1524)
        if (inst.replay) inst.replay->update(inst.processed_cycles, gated, frame);
```
(remaining changed hunks of this function trimmed)
</details>


<details><summary>A2285: context changed, function `float CountableFitter::run_inference(CountableInstance& inst, const CountableObservation& observation)` (685 lines before, 757 after)</summary>

Evidence line (before, countable_concept/src/countable_fitter.cpp:1519): `const float oblq_cos = std::clamp(inst.dbg_obliquity_cos, 0.05f, 1.0f);`; found unchanged at countable_concept/src/countable_fitter.cpp:1687.

```
BEFORE (from line 1040)

AFTER (from line 1135)
        std::vector<Eigen::Vector3f> birth_cloud;   // F1 round 6.2: the seed H is re-derived from it once the belief knows its shape
```
```
BEFORE (from line 1042)

AFTER (from line 1138)
            birth_cloud.reserve(static_cast<std::size_t>(npts));
            birth_cloud.insert(birth_cloud.end(), observation.candidate_pts.begin(), observation.candidate_pts.end());
            birth_cloud.insert(birth_cloud.end(), observation.residual_pts.begin(), observation.residual_pts.end());
```
```
BEFORE (from line 1066)
            // Skip on a TRIMMED mask (extent is only a lower bound → would birth too small); wait for a full mask.
            if (cfg_.footprint_moment_precision > 0.0f and inst.last_trunc_frac <= 1e-3f)
AFTER (from line 1165)
            // Legacy path: skip on a TRIMMED mask (extent is only a lower bound → would birth too small); wait for a full mask.
            // Persistent path (plan rev 2 F1.3): no switch. The engine enters the class prior at the class mean, so the seed is
            // only the GN start, and a clipped mask enters as what it is — a seen extent shorter than the class's — through
            // birth_on_priors' position covariance, (class extent − seen extent)²/12 along the wall (countable_birth_footprint.h).
            // engine1's bed: trunc 0.23 % skipped the whole seed and GN started 0.8 m short of the bed's centre.
            if (cfg_.footprint_moment_precision > 0.0f and (cfg_.lidar_measures_on_gated or inst.last_trunc_frac <= 1e-3f))
```
```
BEFORE (from line 1075)

AFTER (from line 1178)
                // The band's base is the body's underside: the floor for a floor-anchored class, H − e for a hanging one (a
                // point under a hood is the hob or the worktop, not the hood).
```
```
BEFORE (from line 1076)
                const float z_lo = 3.0f * cfg_.ai2_sigma_base_m;
AFTER (from line 1181)
                const float z_lo = rc::countable::VerticalSpan{cfg_.vertical_extent_m}.z0(s0.H) + 3.0f * cfg_.ai2_sigma_base_m;
```
```
BEFORE (from line 1101)
                    // ★BORN ON THE FURNITURE PRIORS (owner 2026-11-14): room-axis yaw, back on the nearest wall.
                    if (have_room_geometry_) birth_pos_cov = birth_on_priors(s0, pts, z_lo, z_hi, inst.node_name);
AFTER (from line 1206)
                    // ★BORN ON THE FURNITURE PRIORS (owner 2026-11-14): room-axis yaw, back on the nearest wall — for a class that
                    // declares a back-face attachment only (F1 round 9 a, rc::birth_on_wall); any other keeps this cold-start seed.
                    birth_pos_cov = rc::seed_birth(s0, pts, z_lo, z_hi, cfg_, have_room_geometry_,
                                                   [this](const Eigen::Vector2f& q) { return nearest_wall(q); }, inst.node_name);
```
```
BEFORE (from line 1108)

AFTER (from line 1215)
        p.range_noise_lat_per_m = cfg_.ai2_range_noise_lat_per_m;
```
```
BEFORE (from line 1113)

AFTER (from line 1221)
        p.prior_depth_std     = cfg_.ai2_prior_depth_std;
```
```
BEFORE (from line 1114)

AFTER (from line 1223)
        p.vertical_extent_m   = cfg_.vertical_extent_m;   // the manifest's support: floor-anchored or hanging (K1)
```
```
BEFORE (from line 1115)

AFTER (from line 1225)
        p.symmetric           = cfg_.shape_symmetric;   // K2c: [shape] symmetry = "c2v" (a table: the four yaw modes are one body)
```
```
BEFORE (from line 1118)

AFTER (from line 1229)
        p.shape_parts         = cfg_.shape_parts;   // the solid: one table for the SDF, the beam parts and the carve (K2a)
        p.solid_parts         = cfg_.solid_parts;
        p.shape_proportions   = cfg_.shape_proportions;   // the class proportions, ρ = 1 latents (K2a-L)
```
```
BEFORE (from line 1129)

AFTER (from line 1243)
        p.wall_standoff_m         = cfg_.wall_standoff_m;       // [prior.attachment] standoff (F1.4; the persistent path's flush prior)
        p.wall_standoff_std_m     = cfg_.wall_standoff_std_m;
        p.move_rate_per_day       = cfg_.move_rate_per_day;     // [prior.permanence] (F1 round 6): the volatility's hyperprior
        p.move_scale_m            = cfg_.move_scale_m;
        p.move_scale_rad          = cfg_.move_scale_rad;
        if (cfg_.ai2_age_nominal_dt_s > 0.0f) p.predict_period_s = cfg_.ai2_age_nominal_dt_s;
```
```
BEFORE (from line 1170)

AFTER (from line 1290)
        p.layer_spread_m = cfg_.layer_spread_m;        // the class's layer spread Δ (manifest; the persistent path's shape latent)
```
```
BEFORE (from line 1172)

AFTER (from line 1293)
        if (p.persistent and not birth_cloud.empty())
        {   // F1 round 6.2: the seed H from the model the update uses (class-prior MAP ⊕ the top of the points the BOX explains,
            // box-vs-headboard shares at the born pose), not the p95 of a cloud that holds the headboard (bed: 1.02 vs 0.65).
            const float H_p95 = s0.H;
            s0.H = inst.ai2_belief.seed_height(birth_cloud, rc::kZedYoloBoundaryStdM);
            inst.ai2_belief = CountableBelief(s0, p);
            std::print("[{}] birth seed H: p95 of the cloud {:.3f} → box-explained top ⊕ class prior {:.3f}\n", inst.node_name, H_p95, s0.H);
        }
```
```
BEFORE (from line 1173)

AFTER (from line 1302)
        inst.replay.reset();                                                           // a new birth starts a new stream
        inst.offer_cache.reset();                                                      // a new belief: its update serial restarts
        if (cfg_.replay_dump and p.persistent) open_replay(inst, s0, p, birth_pos_cov);
```
```
BEFORE (from line 1210)

AFTER (from line 1342)
            if (inst.replay) inst.replay->inflate(dt, cfg_.ai2_age_nominal_dt_s);
```
```
BEFORE (from line 1262)

AFTER (from line 1395)
    // GATE AUDIT (plan rev 2 F1.8, persistent path; .superpowers/sdd/2026-11-23-shared-engine/task-F1-report.md): both gates
    // are KEPT, each with its reason, neither replaced on hope.
    //  - truncation (AI2TruncGateFrac 0.10): fired on 0 of 1034 bed and 0 of 2448 nightstand rows in objects_replay1. Under the
    //    point-to-surface likelihood a clipped mask gives fewer, not biased, surface points; what IS biased is z_top_obs (the
    //    p97 top of a cloud clipped at the top border) and the birth extent (F1.3 handles birth). Model replacement: z_top_obs
    //    as a censored observation when the clip is at the top. Not done: no recorded frame exercises it.
    //  - confirm_only (moving AND off-centre): 959 of 1034 bed rows gated, 324 of them with a mask (LiDAR-only instead). The
    //    fresh latent u_cam models the same ego-motion smear, so the two count it twice. Model replacement: u_cam alone.
    //    Not done: the replay cannot see it (a gated cycle records only its LiDAR-only frame, not the camera frame it skipped)
    //    and u_cam's MotionCm gains are unmeasured (τ, T5 ruling) — removing the gate would hand the smear to an unmeasured
    //    term. Next: record the skipped camera frame in the replay tap, and measure τ_eff (Fable review 2.8).
```
```
BEFORE (from line 1304)
    // The latents' marginals (§7c.2): b_cam = the config floors + range lateral/size + the obliquity yaw cap (today's camera
    // fill WITHOUT chain and ego-motion); b_L = the config floors. e's marginal is Σ_pose, staged by stage_persistent (§7c.7).
    // The ego-motion terms are FRESH every frame (ρ = 0): the belief's per-frame latents u_cam / u_L (spec 7c.8 rev 3.3.13).
    const auto latent_inputs = [&](float obliquity_yaw_var)
AFTER (from line 1448)
    // The latents' marginals (§7c.2; shared-engine G1, controller ruling for Task 5): b_L = the constant metric common-mode stds
    // squared (AI2CommonMode{Pos,Size,Yaw}Std); b_cam = the measured ZED YOLO-seg boundary std on position and size (plan rev 2
    // F1.6, rc::kZedYoloBoundaryStdM) and the configured yaw std. The range growth (AI2RangeNoise{Yaw,Size}PerM, and the lat
    // term on b_cam) and the obliquity yaw term (kObliquityYawGain with its cos clamp) are NOT in them: the mask boundary
    // error is shape-dominated and neither pixel- nor metre-constant (step 0, docs/superpowers/results/2026-11-24-step0.md),
    // so no measured sensor row replaces them yet and they are dropped from the persistent path rather than kept as patch.
    // The pose-chain share is e (marginal Σ_pose, staged by stage_persistent, §7c.7), as before. The ego-motion terms
    // (MotionCm* · motion_magnitude · periphery, unchanged) are FRESH every frame (ρ = 0): u_cam / u_L (rev 3.3.13).
    // The computation is rc::persistent_latent_inputs (countable_latent_inputs.h), tested by latent_inputs_test.
    const auto latent_inputs = [&]
```
```
BEFORE (from line 1309)
        const float p2 = cfg_.ai2_common_mode_pos_std  * cfg_.ai2_common_mode_pos_std;
        const float s2 = cfg_.ai2_common_mode_size_std * cfg_.ai2_common_mode_size_std;
        const float y2 = cfg_.ai2_common_mode_yaw_std  * cfg_.ai2_common_mode_yaw_std;
        LatentInputs li;
        li.sig2_cam = {p2 + range_lat_var, p2 + range_lat_var, s2 + range_size_var, s2 + range_size_var,
                       s2 + range_size_var, y2 + range_yaw_var + obliquity_yaw_var};
        li.sig2_L   = {p2, p2, s2, s2, s2, y2};
        li.fresh    = {mot_pos_var, mot_pos_var, mot_size_var, mot_size_var, mot_size_var, mot_yaw_var};
        return li;
AFTER (from line 1459)
        rc::LatentView v;
        v.range_m = range; v.obliquity_cos = inst.dbg_obliquity_cos;
        v.chain_var_xx = inst.chain_cov_xx; v.chain_var_yy = inst.chain_cov_yy;
        v.mot_pos_var = mot_pos_var; v.mot_size_var = mot_size_var; v.mot_yaw_var = mot_yaw_var;
        return rc::persistent_latent_inputs({cfg_.ai2_common_mode_pos_std, cfg_.ai2_common_mode_size_std,
                                             cfg_.ai2_common_mode_yaw_std}, v);
```
```
BEFORE (from line 1344)

AFTER (from line 1491)
        log_delivered(inst);                       // G5: what this accepted update delivered
```
```
BEFORE (from line 1352)

AFTER (from line 1500)
            if (inst.replay) inst.replay->room_interior(room_interior_);
```
```
BEFORE (from line 1354)
            frame.wall = nearest_wall(inst.ai2_belief.back_centre(bs_now));
AFTER (from line 1503)
            // K2c: a symmetric class has no back face, so its wall is the one nearest its CENTRE (invariant under the c2v group;
            // the back centre moves to another face in another sheet). The staged wall feeds the explain-away component and δ.
            frame.wall = nearest_wall(inst.ai2_belief.params().symmetric ? Eigen::Vector2f(bs_now.cx, bs_now.cy)
                                                                         : inst.ai2_belief.back_centre(bs_now));
```
```
BEFORE (from line 1372)

AFTER (from line 1524)
        if (inst.replay) inst.replay->update(inst.processed_cycles, gated, frame);
```
(remaining changed hunks of this function trimmed)
</details>


<details><summary>A2291: context changed, function `float CountableFitter::run_inference(CountableInstance& inst, const CountableObservation& observation)` (685 lines before, 757 after)</summary>

Evidence line (before, countable_concept/src/countable_fitter.cpp:1637): `constexpr float kYawJumpLogRad = 0.087f; // ≈5°: diagnostic PRINT trigger only (no fit effect)`; found unchanged at countable_concept/src/countable_fitter.cpp:1804.

```
BEFORE (from line 1040)

AFTER (from line 1135)
        std::vector<Eigen::Vector3f> birth_cloud;   // F1 round 6.2: the seed H is re-derived from it once the belief knows its shape
```
```
BEFORE (from line 1042)

AFTER (from line 1138)
            birth_cloud.reserve(static_cast<std::size_t>(npts));
            birth_cloud.insert(birth_cloud.end(), observation.candidate_pts.begin(), observation.candidate_pts.end());
            birth_cloud.insert(birth_cloud.end(), observation.residual_pts.begin(), observation.residual_pts.end());
```
```
BEFORE (from line 1066)
            // Skip on a TRIMMED mask (extent is only a lower bound → would birth too small); wait for a full mask.
            if (cfg_.footprint_moment_precision > 0.0f and inst.last_trunc_frac <= 1e-3f)
AFTER (from line 1165)
            // Legacy path: skip on a TRIMMED mask (extent is only a lower bound → would birth too small); wait for a full mask.
            // Persistent path (plan rev 2 F1.3): no switch. The engine enters the class prior at the class mean, so the seed is
            // only the GN start, and a clipped mask enters as what it is — a seen extent shorter than the class's — through
            // birth_on_priors' position covariance, (class extent − seen extent)²/12 along the wall (countable_birth_footprint.h).
            // engine1's bed: trunc 0.23 % skipped the whole seed and GN started 0.8 m short of the bed's centre.
            if (cfg_.footprint_moment_precision > 0.0f and (cfg_.lidar_measures_on_gated or inst.last_trunc_frac <= 1e-3f))
```
```
BEFORE (from line 1075)

AFTER (from line 1178)
                // The band's base is the body's underside: the floor for a floor-anchored class, H − e for a hanging one (a
                // point under a hood is the hob or the worktop, not the hood).
```
```
BEFORE (from line 1076)
                const float z_lo = 3.0f * cfg_.ai2_sigma_base_m;
AFTER (from line 1181)
                const float z_lo = rc::countable::VerticalSpan{cfg_.vertical_extent_m}.z0(s0.H) + 3.0f * cfg_.ai2_sigma_base_m;
```
```
BEFORE (from line 1101)
                    // ★BORN ON THE FURNITURE PRIORS (owner 2026-11-14): room-axis yaw, back on the nearest wall.
                    if (have_room_geometry_) birth_pos_cov = birth_on_priors(s0, pts, z_lo, z_hi, inst.node_name);
AFTER (from line 1206)
                    // ★BORN ON THE FURNITURE PRIORS (owner 2026-11-14): room-axis yaw, back on the nearest wall — for a class that
                    // declares a back-face attachment only (F1 round 9 a, rc::birth_on_wall); any other keeps this cold-start seed.
                    birth_pos_cov = rc::seed_birth(s0, pts, z_lo, z_hi, cfg_, have_room_geometry_,
                                                   [this](const Eigen::Vector2f& q) { return nearest_wall(q); }, inst.node_name);
```
```
BEFORE (from line 1108)

AFTER (from line 1215)
        p.range_noise_lat_per_m = cfg_.ai2_range_noise_lat_per_m;
```
```
BEFORE (from line 1113)

AFTER (from line 1221)
        p.prior_depth_std     = cfg_.ai2_prior_depth_std;
```
```
BEFORE (from line 1114)

AFTER (from line 1223)
        p.vertical_extent_m   = cfg_.vertical_extent_m;   // the manifest's support: floor-anchored or hanging (K1)
```
```
BEFORE (from line 1115)

AFTER (from line 1225)
        p.symmetric           = cfg_.shape_symmetric;   // K2c: [shape] symmetry = "c2v" (a table: the four yaw modes are one body)
```
```
BEFORE (from line 1118)

AFTER (from line 1229)
        p.shape_parts         = cfg_.shape_parts;   // the solid: one table for the SDF, the beam parts and the carve (K2a)
        p.solid_parts         = cfg_.solid_parts;
        p.shape_proportions   = cfg_.shape_proportions;   // the class proportions, ρ = 1 latents (K2a-L)
```
```
BEFORE (from line 1129)

AFTER (from line 1243)
        p.wall_standoff_m         = cfg_.wall_standoff_m;       // [prior.attachment] standoff (F1.4; the persistent path's flush prior)
        p.wall_standoff_std_m     = cfg_.wall_standoff_std_m;
        p.move_rate_per_day       = cfg_.move_rate_per_day;     // [prior.permanence] (F1 round 6): the volatility's hyperprior
        p.move_scale_m            = cfg_.move_scale_m;
        p.move_scale_rad          = cfg_.move_scale_rad;
        if (cfg_.ai2_age_nominal_dt_s > 0.0f) p.predict_period_s = cfg_.ai2_age_nominal_dt_s;
```
```
BEFORE (from line 1170)

AFTER (from line 1290)
        p.layer_spread_m = cfg_.layer_spread_m;        // the class's layer spread Δ (manifest; the persistent path's shape latent)
```
```
BEFORE (from line 1172)

AFTER (from line 1293)
        if (p.persistent and not birth_cloud.empty())
        {   // F1 round 6.2: the seed H from the model the update uses (class-prior MAP ⊕ the top of the points the BOX explains,
            // box-vs-headboard shares at the born pose), not the p95 of a cloud that holds the headboard (bed: 1.02 vs 0.65).
            const float H_p95 = s0.H;
            s0.H = inst.ai2_belief.seed_height(birth_cloud, rc::kZedYoloBoundaryStdM);
            inst.ai2_belief = CountableBelief(s0, p);
            std::print("[{}] birth seed H: p95 of the cloud {:.3f} → box-explained top ⊕ class prior {:.3f}\n", inst.node_name, H_p95, s0.H);
        }
```
```
BEFORE (from line 1173)

AFTER (from line 1302)
        inst.replay.reset();                                                           // a new birth starts a new stream
        inst.offer_cache.reset();                                                      // a new belief: its update serial restarts
        if (cfg_.replay_dump and p.persistent) open_replay(inst, s0, p, birth_pos_cov);
```
```
BEFORE (from line 1210)

AFTER (from line 1342)
            if (inst.replay) inst.replay->inflate(dt, cfg_.ai2_age_nominal_dt_s);
```
```
BEFORE (from line 1262)

AFTER (from line 1395)
    // GATE AUDIT (plan rev 2 F1.8, persistent path; .superpowers/sdd/2026-11-23-shared-engine/task-F1-report.md): both gates
    // are KEPT, each with its reason, neither replaced on hope.
    //  - truncation (AI2TruncGateFrac 0.10): fired on 0 of 1034 bed and 0 of 2448 nightstand rows in objects_replay1. Under the
    //    point-to-surface likelihood a clipped mask gives fewer, not biased, surface points; what IS biased is z_top_obs (the
    //    p97 top of a cloud clipped at the top border) and the birth extent (F1.3 handles birth). Model replacement: z_top_obs
    //    as a censored observation when the clip is at the top. Not done: no recorded frame exercises it.
    //  - confirm_only (moving AND off-centre): 959 of 1034 bed rows gated, 324 of them with a mask (LiDAR-only instead). The
    //    fresh latent u_cam models the same ego-motion smear, so the two count it twice. Model replacement: u_cam alone.
    //    Not done: the replay cannot see it (a gated cycle records only its LiDAR-only frame, not the camera frame it skipped)
    //    and u_cam's MotionCm gains are unmeasured (τ, T5 ruling) — removing the gate would hand the smear to an unmeasured
    //    term. Next: record the skipped camera frame in the replay tap, and measure τ_eff (Fable review 2.8).
```
```
BEFORE (from line 1304)
    // The latents' marginals (§7c.2): b_cam = the config floors + range lateral/size + the obliquity yaw cap (today's camera
    // fill WITHOUT chain and ego-motion); b_L = the config floors. e's marginal is Σ_pose, staged by stage_persistent (§7c.7).
    // The ego-motion terms are FRESH every frame (ρ = 0): the belief's per-frame latents u_cam / u_L (spec 7c.8 rev 3.3.13).
    const auto latent_inputs = [&](float obliquity_yaw_var)
AFTER (from line 1448)
    // The latents' marginals (§7c.2; shared-engine G1, controller ruling for Task 5): b_L = the constant metric common-mode stds
    // squared (AI2CommonMode{Pos,Size,Yaw}Std); b_cam = the measured ZED YOLO-seg boundary std on position and size (plan rev 2
    // F1.6, rc::kZedYoloBoundaryStdM) and the configured yaw std. The range growth (AI2RangeNoise{Yaw,Size}PerM, and the lat
    // term on b_cam) and the obliquity yaw term (kObliquityYawGain with its cos clamp) are NOT in them: the mask boundary
    // error is shape-dominated and neither pixel- nor metre-constant (step 0, docs/superpowers/results/2026-11-24-step0.md),
    // so no measured sensor row replaces them yet and they are dropped from the persistent path rather than kept as patch.
    // The pose-chain share is e (marginal Σ_pose, staged by stage_persistent, §7c.7), as before. The ego-motion terms
    // (MotionCm* · motion_magnitude · periphery, unchanged) are FRESH every frame (ρ = 0): u_cam / u_L (rev 3.3.13).
    // The computation is rc::persistent_latent_inputs (countable_latent_inputs.h), tested by latent_inputs_test.
    const auto latent_inputs = [&]
```
```
BEFORE (from line 1309)
        const float p2 = cfg_.ai2_common_mode_pos_std  * cfg_.ai2_common_mode_pos_std;
        const float s2 = cfg_.ai2_common_mode_size_std * cfg_.ai2_common_mode_size_std;
        const float y2 = cfg_.ai2_common_mode_yaw_std  * cfg_.ai2_common_mode_yaw_std;
        LatentInputs li;
        li.sig2_cam = {p2 + range_lat_var, p2 + range_lat_var, s2 + range_size_var, s2 + range_size_var,
                       s2 + range_size_var, y2 + range_yaw_var + obliquity_yaw_var};
        li.sig2_L   = {p2, p2, s2, s2, s2, y2};
        li.fresh    = {mot_pos_var, mot_pos_var, mot_size_var, mot_size_var, mot_size_var, mot_yaw_var};
        return li;
AFTER (from line 1459)
        rc::LatentView v;
        v.range_m = range; v.obliquity_cos = inst.dbg_obliquity_cos;
        v.chain_var_xx = inst.chain_cov_xx; v.chain_var_yy = inst.chain_cov_yy;
        v.mot_pos_var = mot_pos_var; v.mot_size_var = mot_size_var; v.mot_yaw_var = mot_yaw_var;
        return rc::persistent_latent_inputs({cfg_.ai2_common_mode_pos_std, cfg_.ai2_common_mode_size_std,
                                             cfg_.ai2_common_mode_yaw_std}, v);
```
```
BEFORE (from line 1344)

AFTER (from line 1491)
        log_delivered(inst);                       // G5: what this accepted update delivered
```
```
BEFORE (from line 1352)

AFTER (from line 1500)
            if (inst.replay) inst.replay->room_interior(room_interior_);
```
```
BEFORE (from line 1354)
            frame.wall = nearest_wall(inst.ai2_belief.back_centre(bs_now));
AFTER (from line 1503)
            // K2c: a symmetric class has no back face, so its wall is the one nearest its CENTRE (invariant under the c2v group;
            // the back centre moves to another face in another sheet). The staged wall feeds the explain-away component and δ.
            frame.wall = nearest_wall(inst.ai2_belief.params().symmetric ? Eigen::Vector2f(bs_now.cx, bs_now.cy)
                                                                         : inst.ai2_belief.back_centre(bs_now));
```
```
BEFORE (from line 1372)

AFTER (from line 1524)
        if (inst.replay) inst.replay->update(inst.processed_cycles, gated, frame);
```
(remaining changed hunks of this function trimmed)
</details>


### countable_concept/src/countable_gain_query.h

| A2553 | literal | 5 | countable_concept/src/countable_gain_query.h:57 | (no comment in the code) |

### countable_concept/src/countable_latent_inputs.h

| A2554 | kZedYoloBoundaryStdM | 0.0289f | countable_concept/src/countable_latent_inputs.h:38 | "objects_step0/zed: 0.0289 m (mean −0.0031 m; mean \|offset\| 0.0131 m, the step-0 range-bin table's fridge rows). An upper bound: / the residual layer mismatch of the fridge box is inside it. A shared offset of every boundary pixel moves the face it lies on, / so it is the shared error of the centre (one face seen) and o…" |

### countable_concept/src/countable_neighbour_size.h

| A2555 | literal | 6 | countable_concept/src/countable_neighbour_size.h:46 | (no comment in the code) |
| A2556 | literal | 6 | countable_concept/src/countable_neighbour_size.h:47 | (no comment in the code) |
| A2557 | k | 12.0f | countable_concept/src/countable_neighbour_size.h:48 | (no comment in the code) |

### countable_concept/src/countable_projection.cpp

| A2293 | near-plane guard | Y<=0.20 | countable_concept/src/countable_projection.cpp:217 | **context changed** (enclosing function differs between 535d57ec and cf65a471; before/after text below). Baseline name: near-plane guard. Code comment: "skip corners at/near the image plane: X/Y explodes there" |
| A2292 | roi sanity gate | \\\|off\\\|<3, fill<4 | countable_concept/src/countable_projection.cpp:242 | **context changed** (enclosing function differs between 535d57ec and cf65a471; before/after text below). Baseline name: roi sanity gate. Code comment: "bbox explodes to absurd offsets). Beyond a sane bound the ROI is unusable for centring: / mark invalid (the controller then keeps sweeping / treats framing as unknown) and clamp the / stored values so consumers/logs never see garbage." |
| A2294 | spatial hash CELL | 6.0 px | countable_concept/src/countable_projection.cpp:281 | **context changed** (enclosing function differs between 535d57ec and cf65a471; before/after text below). Baseline name: spatial hash CELL. Code comment: "spatial-hash cell size (px). A discretization/method constant, not a belief" |

<details><summary>A2293: context changed, function `void CountableProjection::compute_projected_roi(CountableInstance& inst)` (68 lines before, 68 after)</summary>

Evidence line (before, countable_concept/src/countable_projection.cpp:216): `if (Y <= 0.20) continue; // skip corners at/near the image plane: X/Y explodes there`; found unchanged at countable_concept/src/countable_projection.cpp:217.

```
BEFORE (from line 210)
            for (const float z : {0.0f, s.countable_height})
AFTER (from line 211)
            for (const float z : {span_.z0(s.countable_height), s.countable_height})
```
</details>


<details><summary>A2292: context changed, function `void CountableProjection::compute_projected_roi(CountableInstance& inst)` (68 lines before, 68 after)</summary>

Evidence line (before, countable_concept/src/countable_projection.cpp:241): `const bool sane = std::isfinite(off_x) and std::isfinite(off_y) and std::isfinite(fill) and std::abs(off_x) < 3.0f and std::abs(off_y) < 3.0f and fill < 4.0f;`; found unchanged at countable_concept/src/countable_projection.cpp:242.

```
BEFORE (from line 210)
            for (const float z : {0.0f, s.countable_height})
AFTER (from line 211)
            for (const float z : {span_.z0(s.countable_height), s.countable_height})
```
</details>


<details><summary>A2294: context changed, function `SilhouetteExistence CountableProjection::compute_silhouette_existence(const CountableInstance& inst)` (169 lines before, 175 after)</summary>

Evidence line (before, countable_concept/src/countable_projection.cpp:280): `constexpr float CELL = 6.0f; // spatial-hash cell size (px). A discretization/method constant, not a belief`; found unchanged at countable_concept/src/countable_projection.cpp:281.

```
BEFORE (from line 398)
    // spans the full box height [0, H].
AFTER (from line 399)
    // spans the full box height [z0, H] (z0 = 0 floor-anchored, H − e hanging).
```
```
BEFORE (from line 408)
                classify(lx, ly, s.H * (iv + 0.5f) / NV);
AFTER (from line 409)
                classify(lx, ly, span_.z0(s.H) + span_.height(s.H) * (iv + 0.5f) / NV);
```
```
BEFORE (from line 420)

AFTER (from line 421)
    // BOTTOM face only when the camera is genuinely below it: a HANGING body (a hood above the ZED) shows its underside. A
    // floor-anchored one's underside is the floor (z0 = 0), never above the camera, so this adds nothing there.
    if (span_.hangs() and cam_lz < span_.z0(s.H))
        for (int iu = 0; iu < NU; ++iu)
            for (int iv = 0; iv < NV; ++iv)
                classify((-1.0f + 2.0f * (iu + 0.5f) / NU) * hw, (-1.0f + 2.0f * (iv + 0.5f) / NV) * hd, span_.z0(s.H));
```
</details>


### countable_concept/src/countable_replay.h

| A2558 | kVersion | 1 | countable_concept/src/countable_replay.h:53 | "Bump kVersion when a field's MEANING or UNIT changes (or an event's semantics do). Adding or removing a field needs no / bump: the by-name reader handles it (missing / unknown). A reader refuses a dump newer than the kVersion it was built with." |
| A2559 | literal | 64, 16, 40 | countable_concept/src/countable_replay.h:409 | (no comment in the code) |

### countable_concept/src/countable_scene_graph.cpp

| A2298 | kPosEps / kYawEps | 0.003 / 0.005 | countable_concept/src/countable_scene_graph.cpp:154 | **context changed** (enclosing function differs between 535d57ec and cf65a471; before/after text below). Baseline name: kPosEps / kYawEps. Code comment: "dead-banded RT pose), rebuilt here every cycle from the raw fit — so writing it from the / sub-cm oscillating state each frame makes the viewer bed JITTER even though room→bed is / static. Freezing the mesh once settled removes the jitter. Mirrors bottle_concept's pub gate. / 3 mm" |
| A2560 | literal | 3.0f | countable_concept/src/countable_scene_graph.cpp:254 | (no comment in the code) |
| A2561 | literal | 0.25f | countable_concept/src/countable_scene_graph.cpp:328 | (no comment in the code) |

<details><summary>A2298: context changed, function `void CountableSceneGraph::step_write_model(CountableInstance& inst, DSR::Node& node, std::uint64_t room_id, float free_energy)` (101 lines before, 158 after)</summary>

Evidence line (before, countable_concept/src/countable_scene_graph.cpp:148): `constexpr float kPosEps = 0.003f; // 3 mm`; found unchanged at countable_concept/src/countable_scene_graph.cpp:154.

```
BEFORE (from line 163)
        G_->add_or_modify_attrib_local<height_m_att>(node, s.countable_height);
AFTER (from line 169)
        // the body's extent above its origin: H floor-anchored, the class extent e hanging (origin = the underside)
        G_->add_or_modify_attrib_local<height_m_att>(node, rc::countable::VerticalSpan{cfg_.vertical_extent_m}.height(s.countable_height));
```
```
BEFORE (from line 216)

AFTER (from line 223)
    // K2b: the family is the class's (Concept.LookAttrs). The controller's chair contract waits on chair_front_seen +
    // chair_detection_confidence (affordance_protocol.h) and counts looks by chair_front_seen_frame (controller_session.cpp),
    // the attributes the legacy chair agent wrote; every other class keeps table_*.
    if (cfg_.look_attrs == "chair")
    {
        G_->add_or_modify_attrib_local<chair_roi_offset_att>(node, std::vector<float>{inst.roi_offset_x, inst.roi_offset_y});
        G_->add_or_modify_attrib_local<chair_roi_fill_att>(node, inst.roi_fill);
        G_->add_or_modify_attrib_local<chair_roi_valid_att>(node, inst.roi_valid);
        G_->add_or_modify_attrib_local<chair_detection_alive_att>(node, inst.detection_alive);
        G_->add_or_modify_attrib_local<chair_detection_confidence_att>(node, inst.last_mask_confidence);
        G_->add_or_modify_attrib_local<chair_frames_since_detection_att>(node, inst.frames_since_detection);
        G_->add_or_modify_attrib_local<chair_front_seen_att>(node, inst.front_seen);
        G_->add_or_modify_attrib_local<chair_front_seen_frame_att>(node, inst.front_seen_frame);
    }
    else
    {
```
```
BEFORE (from line 223)

AFTER (from line 246)
    }

    // Colour (K2b, Concept.AppearanceEnabled; from the legacy chair). The display tint as CHROMATICITY pre-scaled by the belief's
    // confidence toward neutral grey (it fades in as evidence accumulates), and the palette name with its probability (what the
    // chat says). Absent until something was measured. Reads the appearance belief only; nothing geometric depends on it.
    if (cfg_.appearance_enabled and inst.appearance.initialized())
    {
        const float           c       = inst.appearance.confidence();
        const Eigen::Vector3f neutral = Eigen::Vector3f::Constant(1.0f / 3.0f);
        const Eigen::Vector3f tint    = neutral + c * (inst.appearance.map() - neutral);
        G_->add_or_modify_attrib_local<mesh_color_rgb_att>(node, std::vector<float>{tint.x(), tint.y(), tint.z()});
    }
    if (cfg_.appearance_enabled)
        if (const auto cn = rc::appearance::name_colour(inst.appearance); not cn.name.empty())
        {
            G_->add_or_modify_attrib_local<color_name_att>(node, cn.name);
            G_->add_or_modify_attrib_local<color_prob_att>(node, cn.prob);
        }
    // P(this object is really there) = occupancy x class (K2b, Concept.PublishExistProb): ltom keeps the last value as what the
    // memory is worth, and a recalled candidate's prior is this number (recall_match.h).
    if (cfg_.publish_exist_prob and inst.ai2_initialized)
        if (const float p = inst.p_exists(); std::isfinite(p))
            G_->add_or_modify_attrib_local<obj_exist_prob_att>(node, p);
    // The identity link (changed object, K2b, Concept.ChangedObjectEnabled): the node's name never changes; ltom joins the
    // histories. Absent = no link.
    if (cfg_.changed_object_enabled)
    {
        if (inst.link_uid != 0)
        {
            G_->add_or_modify_attrib_local<same_as_uid_att>(node, inst.link_uid);
            G_->add_or_modify_attrib_local<same_as_prob_att>(node, inst.link_prob);
            G_->add_or_modify_attrib_local<left_spot_prob_att>(node, inst.link_left_spot_prob);
        }
        else
        {
            G_->remove_attrib_local<same_as_uid_att>(node);
            G_->remove_attrib_local<same_as_prob_att>(node);
            G_->remove_attrib_local<left_spot_prob_att>(node);
        }
    }
```
</details>


### countable_concept/src/countable_size_std.h

| A2562 | literal | 3, 4 | countable_concept/src/countable_size_std.h:36 | (no comment in the code) |
| A2563 | literal | 3 | countable_concept/src/countable_size_std.h:52 | (no comment in the code) |

### countable_concept/src/countable_support_bank.h

| A2295 | support select below_m | 0.05 | countable_concept/src/countable_support_bank.h:42 | **context changed** (enclosing function differs between 535d57ec and cf65a471; before/after text below). Baseline name: support select below_m. Code comment: (none) |

<details><summary>A2295: context changed, function `inline std::pair<Extent, Params> extent_of(const CountableInstance& inst, const CountableConfig& cfg)` (18 lines before, 19 after)</summary>

Evidence line (before, countable_concept/src/countable_support_bank.h:40): `prm.below_m = 0.05f;`; found unchanged at countable_concept/src/countable_support_bank.h:42.

```
BEFORE (from line 35)
    // HARD bound — the floor — needing only the sensor-noise allowance, not model slack.
AFTER (from line 36)
    // HARD bound — the floor — needing only the sensor-noise allowance, not model slack. A hanging class (hood) spans
    // [H − e, H]: its underside is the crisp edge over the hob, the same sensor-noise allowance below it.
```
```
BEFORE (from line 38)
    e.z0_m = 0.0f;
AFTER (from line 40)
    e.z0_m = rc::countable::VerticalSpan{cfg.vertical_extent_m}.z0(s.countable_height);
```
</details>


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
| A1336 | Model.ReplayDump | false | hood_concept/etc/countable.toml:92 [Model]; read countable_concept/src/countable_config.cpp:537 | "diagnostic only (shared-engine plan rev 2, Task R1): true writes each persistent belief's seed + every update's complete input to etc/replay/<node>_<run>_<serial>.bin (~0.6 MB per hood update) for countable_concept/tests/build/replay \| loader: DIAGNOSTIC only, no model effect: with LidarMeasuresOnGated write each persi…" |
| A1382 | Model.AI2CommonModePosStd | 0.03 | hood_concept/etc/countable.toml:213 [Model]; read countable_concept/src/countable_config.cpp:304 | "Per-frame SHARED-ERROR common-mode stds — the information-saturation FLOORS (CLAUDE.md "encode the effect as a continuous covariance"): N correlated points from one frame cannot collapse Σ below these. LOAD-BEARING. They were previously read from these getters with the keys ABSENT from config (silent defaults) — now ex…" |
| A1383 | Model.AI2CommonModeSizeStd | 0.02 | hood_concept/etc/countable.toml:214 [Model]; read countable_concept/src/countable_config.cpp:306 | "shared size (w,h,H) error floor (m) \| loader: shared size error w,h,H (m)" |
| A1384 | Model.AI2CommonModeYawStd | 0.03 | hood_concept/etc/countable.toml:215 [Model]; read countable_concept/src/countable_config.cpp:308 | "shared yaw error floor (rad) \| loader: shared yaw error (rad)" |
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

### hood_concept/src/hood_belief.cpp

| A2239 | wall segment min length | 0.1 | hood_concept/src/hood_belief.cpp:467 | **context changed** (enclosing function differs between 535d57ec and cf65a471; before/after text below). Baseline name: wall segment min length. Code comment: "earns the alignment. A genuinely free-standing object has wgt → 0 and keeps its data-driven yaw. / Both symmetric solutions (yaw and yaw+180°) satisfy this term, so it does not fight the / front/back resolution — that stays with the appearance cue." |
| A2569 | literal | 0.99f | hood_concept/src/hood_belief.cpp:683 | (no comment in the code) |
| A2570 | literal | 3, 5, 4, 3, 5, 4 | hood_concept/src/hood_belief.cpp:725 | (no comment in the code) |
| A2571 | literal | 3 | hood_concept/src/hood_belief.cpp:735 | (no comment in the code) |
| A2572 | literal | 4 | hood_concept/src/hood_belief.cpp:736 | (no comment in the code) |
| A2573 | literal | 4, 3, 5 | hood_concept/src/hood_belief.cpp:895 | "the persistent joint state relabels with θ: w, h of θ and of every 6-sized latent block; e and ℓ are room-frame / (a no-op before the joint is seeded, and on the legacy path)" |

<details><summary>A2239: context changed, function `void HoodBelief::accumulate_wall(const HoodBeliefState& s, const HoodFrame& f, Eigen::Matrix<float, 6, 6>& Id, Eigen::Matrix<float, 6, 1>& bd, bool base, bool b` (82 lines before, 85 after)</summary>

Evidence line (before, hood_concept/src/hood_belief.cpp:465): `const float sigma_theta = (f.wall.length_m > 0.1f) ? f.wall.sigma_m / f.wall.length_m : 0.0f;`; found unchanged at hood_concept/src/hood_belief.cpp:467.

```
BEFORE (from line 397)
                                         Eigen::Matrix<float, 6, 6>& Id, Eigen::Matrix<float, 6, 1>& bd) const
AFTER (from line 397)
                                         Eigen::Matrix<float, 6, 6>& Id, Eigen::Matrix<float, 6, 1>& bd,
                                         bool base, bool boost, bool parallel, double* E) const
```
```
BEFORE (from line 427)
    const float lam_flush = wgt * (lam_base + params_.depth_unobs_precision * depth_unobs);
AFTER (from line 428)
    const float lam_flush = wgt * ((base ? lam_base : 0.0f) + (boost ? params_.depth_unobs_precision * depth_unobs : 0.0f));
```
```
BEFORE (from line 445)

AFTER (from line 446)
    if (E) *E += 0.5 * double(lam_flush) * double(gap) * double(gap);
```
```
BEFORE (from line 446)
    if (params_.wall_parallel_precision > 0.0f)
AFTER (from line 448)
    if (parallel and params_.wall_parallel_precision > 0.0f)
```
```
BEFORE (from line 476)

AFTER (from line 478)
        if (E) *E += 0.5 * double(lam_par) * double(rp) * double(rp);
```
</details>


### nightstand_concept/etc/config.toml

| A1544 | Model.ReplayDump | false | nightstand_concept/etc/config.toml:78 [Model]; read countable_concept/src/countable_config.cpp:537 | "diagnostic only (shared-engine plan rev 2, Task R1): true writes each persistent belief's seed + every update's complete input to etc/replay/<node>_<run>_<serial>.bin (~0.6 MB per bed update) for countable_concept/tests/build/replay \| loader: DIAGNOSTIC only, no model effect: with LidarMeasuresOnGated write each persis…" |
| A1689 | Model.AI2PriorDepthStd | -1.0f | NO toml key; code default at countable_concept/src/countable_config.cpp:141 | "loader: depth (local Y) prior std (m) when the class declares its own ([prior.depth] std_m); <=0 = the footprint std" |
| A1589 | Model.AI2CommonModePosStd | 0.03 | nightstand_concept/etc/config.toml:198 [Model]; read countable_concept/src/countable_config.cpp:304 | "Per-frame SHARED-ERROR common-mode stds — the information-saturation FLOORS (CLAUDE.md "encode the effect as a continuous covariance"): N correlated points from one frame cannot collapse Σ below these. LOAD-BEARING. They were previously read from these getters with the keys ABSENT from config (silent defaults) — now ex…" |
| A1590 | Model.AI2CommonModeSizeStd | 0.02 | nightstand_concept/etc/config.toml:199 [Model]; read countable_concept/src/countable_config.cpp:306 | "shared size (w,h,H) error floor (m) \| loader: shared size error w,h,H (m)" |
| A1591 | Model.AI2CommonModeYawStd | 0.03 | nightstand_concept/etc/config.toml:200 [Model]; read countable_concept/src/countable_config.cpp:308 | "shared yaw error floor (rad) \| loader: shared yaw error (rad)" |
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

### refrigerator_concept/etc/countable.toml

| A0944 | Model.ReplayDump | false | refrigerator_concept/etc/countable.toml:92 [Model]; read countable_concept/src/countable_config.cpp:537 | "diagnostic only (shared-engine plan rev 2, Task R1): true writes each persistent belief's seed + every update's complete input to etc/replay/<node>_<run>_<serial>.bin (~0.6 MB per refrigerator update) for countable_concept/tests/build/replay \| loader: DIAGNOSTIC only, no model effect: with LidarMeasuresOnGated write ea…" |
| A1091 | Model.AI2PriorDepthStd | -1.0f | NO toml key; code default at countable_concept/src/countable_config.cpp:141 | "loader: depth (local Y) prior std (m) when the class declares its own ([prior.depth] std_m); <=0 = the footprint std" |
| A0990 | Model.AI2CommonModePosStd | 0.03 | refrigerator_concept/etc/countable.toml:213 [Model]; read countable_concept/src/countable_config.cpp:304 | "Per-frame SHARED-ERROR common-mode stds — the information-saturation FLOORS (CLAUDE.md "encode the effect as a continuous covariance"): N correlated points from one frame cannot collapse Σ below these. LOAD-BEARING. They were previously read from these getters with the keys ABSENT from config (silent defaults) — now ex…" |
| A0991 | Model.AI2CommonModeSizeStd | 0.02 | refrigerator_concept/etc/countable.toml:214 [Model]; read countable_concept/src/countable_config.cpp:306 | "shared size (w,h,H) error floor (m) \| loader: shared size error w,h,H (m)" |
| A0992 | Model.AI2CommonModeYawStd | 0.03 | refrigerator_concept/etc/countable.toml:215 [Model]; read countable_concept/src/countable_config.cpp:308 | "shared yaw error floor (rad) \| loader: shared yaw error (rad)" |
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

### refrigerator_concept/src/refrigerator_belief.cpp

| A2203 | wall segment min length | 0.1 | refrigerator_concept/src/refrigerator_belief.cpp:437 | **context changed** (enclosing function differs between 535d57ec and cf65a471; before/after text below). Baseline name: wall segment min length. Code comment: "by the flush posterior, which is the point: ADJACENCY is what earns the alignment, and a / free-standing object (a mid-room fridge, a kitchen island) has wgt -> 0 and keeps its data-driven / yaw. Both 180-degree solutions satisfy it, so the front/back resolution is untouched." |
| A2574 | literal | 0.99f | refrigerator_concept/src/refrigerator_belief.cpp:624 | (no comment in the code) |
| A2575 | literal | 3, 5, 4, 3, 5, 4, 6 | refrigerator_concept/src/refrigerator_belief.cpp:664 | (no comment in the code) |
| A2576 | literal | 3 | refrigerator_concept/src/refrigerator_belief.cpp:674 | (no comment in the code) |
| A2577 | literal | 4 | refrigerator_concept/src/refrigerator_belief.cpp:675 | (no comment in the code) |
| A2578 | literal | 4, 3, 5 | refrigerator_concept/src/refrigerator_belief.cpp:834 | "the persistent joint state relabels with θ: w, h of θ and of every 6-sized latent block; e and ℓ are room-frame / (a no-op before the joint is seeded, and on the legacy path)" |

<details><summary>A2203: context changed, function `void RefrigeratorBelief::accumulate_wall(const RefrigeratorBeliefState& s, const RefrigeratorFrame& f, Eigen::Matrix<float, 6, 6>& Id, Eigen::Matrix<float, 6, 1` (66 lines before, 69 after)</summary>

Evidence line (before, refrigerator_concept/src/refrigerator_belief.cpp:435): `const float sigma_theta = (f.wall.length_m > 0.1f) ? f.wall.sigma_m / f.wall.length_m : 0.0f;`; found unchanged at refrigerator_concept/src/refrigerator_belief.cpp:437.

```
BEFORE (from line 379)
                                         Eigen::Matrix<float, 6, 6>& Id, Eigen::Matrix<float, 6, 1>& bd) const
AFTER (from line 379)
                                         Eigen::Matrix<float, 6, 6>& Id, Eigen::Matrix<float, 6, 1>& bd,
                                         bool base, bool boost, bool parallel, double* E) const
```
```
BEFORE (from line 401)
    const float lam_flush = wgt * (lam_base + params_.depth_unobs_precision * depth_unobs);
AFTER (from line 402)
    const float lam_flush = wgt * ((base ? lam_base : 0.0f) + (boost ? params_.depth_unobs_precision * depth_unobs : 0.0f));
```
```
BEFORE (from line 419)

AFTER (from line 420)
    if (E) *E += 0.5 * double(lam_flush) * double(gap) * double(gap);
```
```
BEFORE (from line 420)
    if (params_.wall_parallel_precision > 0.0f)
AFTER (from line 422)
    if (parallel and params_.wall_parallel_precision > 0.0f)
```
```
BEFORE (from line 442)

AFTER (from line 444)
        if (E) *E += 0.5 * double(lam_par) * double(rp) * double(rp);
```
</details>


### toilet_concept/etc/config.toml

| A1964 | Model.ReplayDump | false | toilet_concept/etc/config.toml:78 [Model]; read countable_concept/src/countable_config.cpp:537 | "diagnostic only (shared-engine plan rev 2, Task R1): true writes each persistent belief's seed + every update's complete input to etc/replay/<node>_<run>_<serial>.bin (~0.6 MB per bed update) for countable_concept/tests/build/replay \| loader: DIAGNOSTIC only, no model effect: with LidarMeasuresOnGated write each persis…" |
| A2110 | Model.AI2PriorDepthStd | -1.0f | NO toml key; code default at countable_concept/src/countable_config.cpp:141 | "loader: depth (local Y) prior std (m) when the class declares its own ([prior.depth] std_m); <=0 = the footprint std" |
| A2010 | Model.AI2CommonModePosStd | 0.03 | toilet_concept/etc/config.toml:199 [Model]; read countable_concept/src/countable_config.cpp:304 | "Per-frame SHARED-ERROR common-mode stds — the information-saturation FLOORS (CLAUDE.md "encode the effect as a continuous covariance"): N correlated points from one frame cannot collapse Σ below these. LOAD-BEARING. They were previously read from these getters with the keys ABSENT from config (silent defaults) — now ex…" |
| A2011 | Model.AI2CommonModeSizeStd | 0.02 | toilet_concept/etc/config.toml:200 [Model]; read countable_concept/src/countable_config.cpp:306 | "shared size (w,h,H) error floor (m) \| loader: shared size error w,h,H (m)" |
| A2012 | Model.AI2CommonModeYawStd | 0.03 | toilet_concept/etc/config.toml:201 [Model]; read countable_concept/src/countable_config.cpp:308 | "shared yaw error floor (rad) \| loader: shared yaw error (rad)" |
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
