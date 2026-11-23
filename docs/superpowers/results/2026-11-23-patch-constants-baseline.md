# Patch-constant baseline before the shared-engine work (AuRo test c)

- Commit: `752e93a89e54655100a1b0c1cbd1e46f40ccf95e` (branch feat/shared-engine, at main 752e93a8)
- Date: 2026-11-23
- Scope: the seven class files (chair, table, refrigerator, hood, nightstand, bed, toilet), the countable engine they share (nightstand, bed, toilet), and the shared engine `common/ai_belief/*.h`.
- Nothing in code or config was changed. This file is the only output.

## Definition (confirmed by the owner, applied literally)

A PATCH constant is any constant that is neither class knowledge nor a property of a sensor. That means any precision, gate, anchor, threshold, gain, margin, floor, clamp or reference/scale whose role is to correct the estimator's behaviour.

- CLASS knowledge: a prior on the class's size, shape, placement (e.g. stands against a wall), appearance, or persistence/permanence.
- SENSOR: a property of a sensor or its mount: noise σ, range, field of view, depth-noise model, detection probability of the detector, intrinsics.
- Owner's examples of PATCH (nightstand config): AI2DepthUnobsPrecision, AI2TopNoFloatPrecision / AI2TopNoFloatMargin, AI2TopOversegSigmaPerM, FeRef / FeScale, PlausHeightPriorGain.

## Exclusion rule

NOT counted at all: plumbing and engineering parameters with no estimation role. Examples: agent id, DDS domain, node names, file paths, log/dump/diagnostic switches (GnTrace, *Log*, *Dump*), period/Hz of the compute loop, thread counts, display/visualisation-only settings, GN iteration caps and numerical-solver tolerances. They are listed in each section with their names (label `excluded`) so a referee can audit them.

Ambiguous items carry the best label plus `?` and one line of why. Boolean feature switches are listed separately as switches. A switch that turns a patch mechanism on counts as a patch in the summary row (column "patch total"); the table says which switches those are.

## Labels

`sensor` `class` `patch` `excluded`; `switch (patch)` = boolean that enables a patch mechanism; `switch (other)` = boolean with no patch mechanism behind it (publication, channel choice, history stage). A trailing `?` = ambiguous (counted in the "ambiguous" column only, never twice).

## How the inventory was made (reproducible)

1. Config keys. Every `key = value` of each `etc/config.toml` is a row (awk over the file: section, key, value). The reader line is found by grepping the quoted `Section.Key` in the class config loader (`*_config.cpp`; `countable_config.cpp` for nightstand, bed, toilet).
2. Hidden config constants. The loaders read some keys with a code default although the toml has no such key (the value in force is then the default). They are listed as rows marked `NO toml key`. Found by diffing the keys read in `*_config.cpp` against the toml.
3. Hard-coded constants in src. Two passes: (a) every `constexpr` / `static const` / named `k...` constant; (b) a literal scan: lines with a numeric literal other than 0, 0.5, 1, 1.5, 2 and `1e-N` guards, after dropping comments, prints/log formats, `check(` and the self-tests. Each hit was read in context. Trivia (unit conversions, array sizes, `max(1e-9f, x)` division guards, self-test fixtures, structure defaults that duplicate a config key) are not listed.
4. Struct defaults in `*_belief.h` / `*_config.h` that mirror a toml key are the same constant as the key and are not counted twice.
5. Class-specific branches in the countable engine: grep for the class name, `concept_class` comparisons and manifest-selected paths.

Limits, stated plainly:
- Only the files named in the task are scanned. Constants in other shared libraries the agents call (`common/existence_belief`, `common/detectability`, `common/birth`, `common/instance_tracker`, `common/contour_edge`, `common/exclusion`, `common/occlusion`, `common/concept_manifest`) are NOT in this table. They carry patch constants too; a second pass over them would raise every row.
- The class manifests (`common/concept_manifest/<class>.concept.toml`: size prior, headboard, π_cls, fit errors) are class knowledge by definition and are not counted.
- The literal scan is manual in the last step. A literal inside a long expression could be missed; the config rows are exact, the src rows are a careful sample of the model math, not a proof of completeness.
- `chair` and `table` have their own belief code; `refrigerator` and `hood` are near copies of each other (their src rows repeat the same constants on purpose: a constant copied into a second file is a second constant to retune). nightstand, bed and toilet run one binary: their src is counted once, in the countable-engine row.
- Labels are the best reading of role and comment. 100% of config rows have a label; the `?` rows are the judgement calls.

## Summary table

Row counts. "patch" = patch constants (without switches). "switches (patch)" = booleans that turn a patch mechanism on. "patch total" = patch + switches (patch). "ambiguous" = rows labelled with `?`, split by best label in the last column. Each row is counted in exactly one of: class, sensor, patch, switches (patch), switches (other), ambiguous, excluded.

| row | class | sensor | patch | switches (patch) | ambiguous (?) | excluded | switches (other) | patch total | ambiguous split (best label) | total rows |
|---|---|---|---|---|---|---|---|---|---|---|
| chair | 5 | 3 | 82 | 9 | 26 | 35 | 6 | 91 | class 3, patch 11, sensor 4, excluded 8 | 166 |
| table | 5 | 10 | 82 | 13 | 35 | 42 | 3 | 95 | class 7, patch 11, sensor 12, excluded 5 | 190 |
| refrigerator | 9 | 10 | 96 | 8 | 49 | 44 | 4 | 104 | class 11, patch 21, sensor 12, excluded 5 | 220 |
| hood | 11 | 7 | 95 | 8 | 49 | 44 | 4 | 103 | class 11, patch 21, sensor 12, excluded 5 | 218 |
| nightstand (config) | 10 | 11 | 84 | 11 | 32 | 44 | 4 | 95 | class 10, patch 6, sensor 14, excluded 2 | 196 |
| bed (config) | 10 | 11 | 84 | 11 | 32 | 44 | 4 | 95 | class 10, patch 6, sensor 14, excluded 2 | 196 |
| toilet (config) | 10 | 11 | 84 | 11 | 32 | 44 | 4 | 95 | class 10, patch 6, sensor 14, excluded 2 | 196 |
| countable engine (src) | 3 | 3 | 10 | 0 | 19 | 7 | 0 | 10 | class 2, patch 15, excluded 2 | 42 |
| shared engine (common/ai_belief) | 0 | 0 | 1 | 0 | 6 | 8 | 0 | 1 | patch 2, sensor 3, excluded 1 | 15 |

Reading the table:
- A class's true patch load is its own row plus the engine row it runs on: chair = chair row + shared engine; table, refrigerator, hood likewise; nightstand, bed, toilet = their config row + countable engine + shared engine. The three countable classes repeat the same keys with different values; the engine row is counted once.
- If every `?` were patch, the upper bound is patch total + (ambiguous rows labelled patch). If every `?` were class or sensor, the lower bound is patch total. The sensitivity is in the "ambiguous split" column.
- Patch-enabling switches in the table above are the booleans labelled `switch (patch)` in each section; the rest are `switch (other)`.

## chair

### Config keys and hidden defaults

| name | value | where | role | label |
|---|---|---|---|---|
| Agent.id | 20 | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Agent.name | "chair_concept" | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Agent.configFile | "" | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Agent.domain | 0 | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Agent.tree | false | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Agent.graph | false | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Agent.2d | false | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Agent.3d | false | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Presence.required_agent_names | ["robot_concept", "room_concept"] | etc/config.toml [Presence] | presence protocol plumbing | excluded |
| Presence.config_dir | "etc" | etc/config.toml [Presence] | presence protocol plumbing | excluded |
| Presence.monitor_period_ms | 500 | etc/config.toml [Presence] | presence protocol plumbing | excluded |
| Presence.heartbeat_timeout_ms | 3000 | etc/config.toml [Presence] | presence protocol plumbing | excluded |
| Presence.rejoin_grace_ms | 1500 | etc/config.toml [Presence] | presence protocol plumbing | excluded |
| Presence.stale_grace_ms | 120000 | etc/config.toml [Presence] | presence protocol plumbing | excluded |
| Presence.agent_info_period_ms | 1000 | etc/config.toml [Presence] | presence protocol plumbing | excluded |
| Owns.nodes | ["chair_concept 20", "chair_*"] | etc/config.toml [Owns] | owned graph nodes (plumbing) | excluded |
| Period.Compute | 100 | etc/config.toml [Period] | loop period | excluded |
| Period.Emergency | 500 | etc/config.toml [Period] | loop period | excluded |
| Component.Debug.Verbose | false | etc/config.toml [Component.Debug] | log switch | excluded |
| Media.MasksStallTimeoutMs | 3000 | etc/config.toml [Media] ; read chair_config.cpp:73 | watchdog timeout on the mask stream (stops robot on stale data) | excluded |
| ChairModel.SigmaObs | 0.05 | etc/config.toml [ChairModel] ; read chair_config.cpp:85 | band splitting top/leg (candidate vs residual) membership | patch |
| ChairModel.SdfThresholdForStorage | 0.08 | etc/config.toml [ChairModel] ; read chair_config.cpp:87 | on-surface membership threshold for candidate/residual split | patch |
| ChairModel.ColourCsvPath | "etc/colour_trace.csv" | etc/config.toml [ChairModel] ; read chair_config.cpp:191 | colour log path | excluded |
| ChairModel.AI2CsvPath | "etc/ai2_log.csv" | etc/config.toml [ChairModel] ; read chair_config.cpp:189 | per-cycle CSV path | excluded |
| ChairModel.AI2SigmaBaseM | 0.03 | etc/config.toml [ChairModel] ; read chair_config.cpp:91 | base on-surface observation noise (m); larger than depth noise, includes mask-edge error | sensor? |
| ChairModel.AI2ClutterFrac | 0.10 | etc/config.toml [ChairModel] ; read chair_config.cpp:99 | prior weight of the uniform clutter mixture (mask error) | patch |
| ChairModel.AI2ClutterScaleM | 0.12 | etc/config.toml [ChairModel] ; read chair_config.cpp:101 | distance beyond which a point is explained by clutter | patch |
| ChairModel.AI2ClutterStructureGain | 1.0 | etc/config.toml [ChairModel] ; read chair_config.cpp:103 | shrink clutter prior for seat-coplanar points (stop escape valve) | patch |
| ChairModel.AI2ProcessStdM | 0.005 | etc/config.toml [ChairModel] ; read chair_config.cpp:105 | process noise on pose = how much furniture moves per frame (permanence); LIFECYCLE says no constant Q | class? |
| ChairModel.AI2ProcessStdYaw | 0.01 | etc/config.toml [ChairModel] ; read chair_config.cpp:107 | process noise on yaw (permanence); LIFECYCLE says no constant Q | class? |
| ChairModel.AI2ProcessStdSize | 0.0005 | etc/config.toml [ChairModel] ; read chair_config.cpp:109 | process noise on rigid size DOFs (permanence) | class? |
| ChairModel.AI2AgeNominalDtS | 0.0 | etc/config.toml [ChairModel] ; read chair_config.cpp:111 | nominal mask-stream period (10 Hz); 0 = aging off | sensor? |
| ChairModel.AI2FloorZ | 0.0 | etc/config.toml [ChairModel] ; read chair_config.cpp:113 | room-frame floor height datum (world convention) | excluded? |
| ChairModel.AI2FloorStd | 0.03 | etc/config.toml [ChairModel] ; read chair_config.cpp:115 | floor-height uncertainty into common-mode z | patch |
| ChairModel.AI2SeatAnchorStd | 0.04 | etc/config.toml [ChairModel] ; read chair_config.cpp:117 | seat-layer height anchor noise (gauge fix) | patch |
| ChairModel.AI2SeatAnchorBand | 0.12 | etc/config.toml [ChairModel] ; read chair_config.cpp:119 | seat-layer mean-shift bandwidth for the anchor | patch |
| ChairModel.AI2SeatExtentStd | 0.02 | etc/config.toml [ChairModel] ; read chair_config.cpp:121 | seat footprint span anchor noise (fixes seat_d collapse) | patch |
| ChairModel.AI2CommonModePosStd | 0.03 | etc/config.toml [ChairModel] ; read chair_config.cpp:123 | shared per-frame position error floor | patch |
| ChairModel.AI2CommonModeSizeStd | 0.02 | etc/config.toml [ChairModel] ; read chair_config.cpp:125 | shared per-frame size error floor | patch |
| ChairModel.AI2CommonModeYawStd | 0.03 | etc/config.toml [ChairModel] ; read chair_config.cpp:127 | shared per-frame yaw error floor | patch |
| ChairModel.AI2RangeNoiseLatPerM | 0.02 | etc/config.toml [ChairModel] ; read chair_config.cpp:129 | lateral deprojection std growth per metre of range (intrinsics-like) feeding R and position common-mode | sensor? |
| ChairModel.AI2RangeNoiseYawPerM | 0.03 | etc/config.toml [ChairModel] ; read chair_config.cpp:131 | yaw common-mode growth with range (anti far-view rotation) | patch |
| ChairModel.MotionCmPosGain | 0.30 | etc/config.toml [ChairModel] ; read chair_config.cpp:133 | ego-motion shared position error per m/s (anti-drift) | patch |
| ChairModel.MotionCmYawGain | 0.50 | etc/config.toml [ChairModel] ; read chair_config.cpp:135 | ego-motion shared yaw error per m/s (anti-rotate) | patch |
| ChairModel.AI2AngLeverM | 2.0 | etc/config.toml [ChairModel] ; read chair_config.cpp:137 | rad/s to m/s lever for ego angular motion | patch |
| ChairModel.AI2PeriphRef | 0.50 | etc/config.toml [ChairModel] ; read chair_config.cpp:139 | centroid radius where periphery penalty saturates | patch |
| ChairModel.AI2MotionRefMps | 0.60 | etc/config.toml [ChairModel] ; read chair_config.cpp:141 | motion at which a peripheral frame is unreliable | patch |
| ChairModel.FixationMinPts | 150 | etc/config.toml [ChairModel] ; read chair_config.cpp:147 | fixation gate: minimum mask points | patch |
| ChairModel.FixationMaxClutter | 0.35 | etc/config.toml [ChairModel] ; read chair_config.cpp:149 | fixation gate: maximum clutter fraction | patch |
| ChairModel.FixationRangeM | 0.0 | etc/config.toml [ChairModel] ; read chair_config.cpp:151 | fixation gate: range cut (retired, 0 = off) | patch |
| ChairModel.FixationCentreFrac | 0.60 | etc/config.toml [ChairModel] ; read chair_config.cpp:153 | fixation gate: fovea radius | patch |
| ChairModel.FixationPeriphFloor | 0.05 | etc/config.toml [ChairModel] ; read chair_config.cpp:157 | floor of the peripheral precision factor | patch |
| ChairModel.FixationStillDotd | 0.05 | etc/config.toml [ChairModel] ; read chair_config.cpp:159 | fixation gate: still, mask smear | patch |
| ChairModel.FixationStillLinMps | 0.05 | etc/config.toml [ChairModel] ; read chair_config.cpp:161 | fixation gate: still, linear speed | patch |
| ChairModel.FixationStillAngRadps | 0.10 | etc/config.toml [ChairModel] ; read chair_config.cpp:163 | fixation gate: still, angular speed | patch |
| ChairModel.AI2StillLinMps | 0.05 | etc/config.toml [ChairModel] ; read chair_config.cpp:165 | still threshold: camera linear speed | patch |
| ChairModel.AI2StillAngRadps | 0.10 | etc/config.toml [ChairModel] ; read chair_config.cpp:167 | still threshold: camera angular speed | patch |
| ChairModel.AI2StillDotd | 0.05 | etc/config.toml [ChairModel] ; read chair_config.cpp:169 | still threshold: mask ego-motion smear | patch |
| ChairModel.AI2MovingUpdateCenterRadius | 0.35 | etc/config.toml [ChairModel] ; read chair_config.cpp:171 | centroid radius below which a moving update is allowed | patch |
| ChairModel.AI2TruncGateFrac | 0.10 | etc/config.toml [ChairModel] ; read chair_config.cpp:183 | skip update if mask border fraction above this (truncation gate) | patch |
| ChairModel.AI2GnIters | 4 | etc/config.toml [ChairModel] ; read chair_config.cpp:185 | GN iterations per frame (iteration cap) | excluded |
| ChairModel.AI2ExtentStd | 0.05 | etc/config.toml [ChairModel] ; read chair_config.cpp:187 | extent-observation noise of coverage likelihood | patch |
| ChairModel.AI2ModeSatBackPts | 60.0 | etc/config.toml [ChairModel] ; read chair_config.cpp:268 | backrest mass at which a frame is half-informative for yaw mode | patch |
| ChairModel.AI2ViewBudget | 3.0 | etc/config.toml [ChairModel] ; read chair_config.cpp:270 | max yaw-mode evidence per bearing bin | patch |
| ChairConcept.StateEps | 0.04 | etc/config.toml [ChairConcept] ; read chair_config.cpp:57 | convergence status tolerance (m+rad); no effect on the estimate | excluded |
| ChairConcept.KStable | 30 | etc/config.toml [ChairConcept] ; read chair_config.cpp:59 | frames before model_stable status flag | excluded |
| ChairConcept.DetectionAliveMaxFrames | 40 | etc/config.toml [ChairConcept] ; read chair_config.cpp:61 | frames without a mask before detection_alive=false (status flag; check whether existence reads it) | excluded? |
| ChairConcept.ObsDistance | 1.8 | etc/config.toml [ChairConcept] ; read chair_config.cpp:63 | planner stand-off preference (viewpoint planner, not the estimator) | excluded? |
| ChairConcept.MinStandOffM | 2.0 | etc/config.toml [ChairConcept] ; read chair_config.cpp:65 | planner minimum stand-off (viewpoint planner, not the estimator) | excluded? |
| ChairConcept.EpistemicCooldownCycles | 200 | etc/config.toml [ChairConcept] ; read chair_config.cpp:67 | planner anti-chatter cooldown (viewpoint planner) | excluded? |
| ChairConcept.ChairLogPeriodFrames | 0 | etc/config.toml [ChairConcept] ; read chair_config.cpp:71 | log throttle | excluded |
| ChairConcept.SupportBankMaxPoints | 4000 | etc/config.toml [ChairConcept] ; read chair_config.cpp:75 | memory cap of the support bank | excluded |
| ChairConcept.SupportBankQuantizationM | 0.02 | etc/config.toml [ChairConcept] ; read chair_config.cpp:77 | dedup grid of the support bank (storage) | excluded |
| ChairConcept.SupportSelectRadiusMarginM | 0.50 | etc/config.toml [ChairConcept] ; read chair_config.cpp:79 | XY ownership margin of support points | patch |
| ChairConcept.SupportSelectHeightMarginM | 0.25 | etc/config.toml [ChairConcept] ; read chair_config.cpp:81 | Z margin of support point selection | patch |
| ChairConcept.RtCovScale | 1.0 | etc/config.toml [ChairConcept] ; read chair_config.cpp:196 | scale on published RT covariance (calibrate toward NEES 1) | patch |
| Tracker.GateMahalanobis | 9.0 | etc/config.toml [Tracker] ; read chair_config.cpp:201 | association gate (chi2) | patch |
| Tracker.GateFallbackM | 0.40 | etc/config.toml [Tracker] ; read chair_config.cpp:203 | association metric gate before covariance exists | patch |
| Tracker.DetectionNoiseM | 0.20 | etc/config.toml [Tracker] ; read chair_config.cpp:205 | association noise added to cover centroid-vs-fit offset | patch |
| Tracker.RecallPiMove | 0.5 | etc/config.toml [Tracker] ; read chair_config.cpp:207 | permanence: P(remembered object was moved) | class |
| Tracker.RecallLogPriorNew | 0.0 | etc/config.toml [Tracker] ; read chair_config.cpp:209 | permanence: log prior of a new object | class |
| Tracker.LinkPiMove | 0.10 | etc/config.toml [Tracker] ; read chair_config.cpp:211 | permanence: P(remembered moved) | class |
| Tracker.LinkPiGone | 0.10 | etc/config.toml [Tracker] ; read chair_config.cpp:212 | permanence: P(remembered left room) | class |
| Tracker.LinkLambdaNew | 0.10 | etc/config.toml [Tracker] ; read chair_config.cpp:213 | permanence: rate of new arrivals | class |
| Tracker.BirthFrames | 8 | etc/config.toml [Tracker] ; read chair_config.cpp:214 | frames a mask stays unexplained before birth (debounce) | patch |
| Tracker.BirthMinSepM | 0.70 | etc/config.toml [Tracker] ; read chair_config.cpp:216 | minimum separation of a birth from existing objects | patch |
| Tracker.MergeOverlap | 0.20 | etc/config.toml [Tracker] ; read chair_config.cpp:222 | footprint overlap that merges two instances | patch |
| Tracker.RicohBirthConf | 0.60 | etc/config.toml [Tracker] ; read chair_config.cpp:276 | min confidence for a Ricoh birth | patch |
| Tracker.RicohBirthMaxVar | 0.005 | etc/config.toml [Tracker] ; read chair_config.cpp:278 | max depth variance for a Ricoh birth | patch |
| Existence.BirthLogodds | 1.0 | etc/config.toml [Existence] ; read chair_config.cpp:234 | log-odds seeded at birth | patch |
| Existence.RemoveLogodds | -3.0 | etc/config.toml [Existence] ; read chair_config.cpp:236 | log-odds below which an instance is removed | patch |
| Existence.MaxLogodds | 4.0 | etc/config.toml [Existence] ; read chair_config.cpp:240 | saturation cap of log-odds | patch |
| Existence.EvidenceGain | 0.15 | etc/config.toml [Existence] ; read chair_config.cpp:242 | per-frame log-odds step scale | patch |
| Existence.ExpectedSupportC | 2500.0 | etc/config.toml [Existence] ; read chair_config.cpp:244 | expected mask support points = C/range^2 (detector/sensor density scale) | sensor? |
| Existence.AdequacyRef | 0.25 | etc/config.toml [Existence] ; read chair_config.cpp:246 | support ratio below which evidence is negative | patch |
| Existence.AdequacyCap | 1.5 | etc/config.toml [Existence] ; read chair_config.cpp:248 | clamp of adequacy | patch |
| Existence.CalibAdapt | 0.0 | etc/config.toml [Existence] ; read chair_config.cpp:250 | EWMA rate adapting C (0 = fixed) | patch |
| Existence.VacateConfidentFrames | 45 | etc/config.toml [Existence] ; read chair_config.cpp:252 | frames until full vacate negative | patch |
| Existence.OcclusionMarginM | 0.30 | etc/config.toml [Existence] ; read chair_config.cpp:256 | occluder must be this much closer | patch |
| Existence.RoomMarginM | 0.40 | etc/config.toml [Existence] ; read chair_config.cpp:262 | tolerance of centre outside walls | patch |
| Existence.OutOfRoomGain | 1.5 | etc/config.toml [Existence] ; read chair_config.cpp:264 | log-odds step per frame outside room | patch |
| Ice.Warn.Connections | "0" | etc/config.toml [Ice] | Ice middleware setting | excluded |
| Ice.Trace.Network | "0" | etc/config.toml [Ice] | Ice middleware setting | excluded |
| Ice.Trace.Protocol | "0" | etc/config.toml [Ice] | Ice middleware setting | excluded |
| Ice.MessageSizeMax | "20004800" | etc/config.toml [Ice] | Ice middleware setting | excluded |
| RigPrior.YawKappaMax | 1.5 | etc/config.toml [RigPrior] ; read chair_config.cpp:230 | cap of the rig yaw prior strength | patch |
| RigPrior.StaleMs | 5000 | etc/config.toml [RigPrior] ; read chair_config.cpp:232 | freshness window of the rig message (watchdog) | excluded? |
| ChairModel.DetectMinFill | 0.10 | NO toml key; code default at chair_config.cpp:93 | detector envelope lower shoulder | sensor |
| ChairModel.DetectMaxFill | 0.60 | NO toml key; code default at chair_config.cpp:95 | detector envelope upper shoulder | sensor |
| ChairModel.DetectSoft | 0.06 | NO toml key; code default at chair_config.cpp:97 | detector envelope softness | sensor |
| ChairModel.AI2ObliquityYawGain | 0.05 | NO toml key; code default at chair_config.cpp:173 | grazing-view yaw cap (hard-coded kObliquityYawGain in table/fridge/hood/countable) | patch |
| ChairModel.AI2OrientationMotionRef | 0.50 | NO toml key; code default at chair_config.cpp:175 | motion at which yaw-mode vote is halved | patch |
| ChairModel.AI2FeBaselineAdaptDown | 0.05 | NO toml key; code default at chair_config.cpp:177 | free-energy baseline EMA, fast down | patch |
| ChairModel.AI2FeBaselineAdaptUp | 0.005 | NO toml key; code default at chair_config.cpp:179 | free-energy baseline EMA, slow up | patch |
| ChairModel.AI2FeSurpriseSmooth | 0.10 | NO toml key; code default at chair_config.cpp:181 | smoothing of positive FE gap (surprise) | patch |
| Tracker.BirthFragmentFrac | 0.25 | NO toml key; code default at chair_config.cpp:218 | fraction below which a candidate is a fragment of a larger mask | patch |
| Tracker.BirthFragmentReachM | 1.00 | NO toml key; code default at chair_config.cpp:220 | reach for fragment parent search | patch |
| Existence.RemoveFrames | 15 | NO toml key; code default at chair_config.cpp:238 | removal debounce frames | patch |
| Bearing.ConfirmGateRad | 0.17 | NO toml key; code default at chair_config.cpp:282 | bearing hypothesis: bearing explained gate | patch |
| Bearing.BirthFrames | 8 | NO toml key; code default at chair_config.cpp:284 | frames a mask stays unexplained before birth (debounce) | patch |
| Bearing.MatchRad | 0.17 | NO toml key; code default at chair_config.cpp:286 | bearing hypothesis: match tolerance | patch |
| Bearing.MaxMiss | 4 | NO toml key; code default at chair_config.cpp:288 | bearing hypothesis: streak gap tolerance | patch |
| Bearing.NominalRangeM | 2.0 | NO toml key; code default at chair_config.cpp:290 | bearing hypothesis: nominal range | patch |
| Bearing.AlongStdM | 3.0 | NO toml key; code default at chair_config.cpp:292 | bearing hypothesis birth: along-ray sigma | patch |
| Bearing.AcrossStdM | 0.30 | NO toml key; code default at chair_config.cpp:294 | bearing hypothesis birth: across-ray sigma | patch |
| Bearing.YawStdRad | 3.14 | NO toml key; code default at chair_config.cpp:296 | bearing hypothesis: yaw sigma | patch |

### Hard-coded in src

| name | value | where | role | label |
|---|---|---|---|---|
| kFlipClamp | 6.0 | chair_concept/src/chair_belief.h:266 | clamp of yaw-mode evidence ("THRESHOLD, flagged, physical") | patch |
| kRigKappaMax | 0.25*kFlipClamp=1.5 | chair_concept/src/chair_belief.h:267 | hard cap of the rig yaw prior (mirrors RigPrior.YawKappaMax) | patch |
| flip evidence clamps (uses) | +-kFlipClamp | chair_concept/src/chair_belief.cpp:257 | same clamp applied at accumulation | patch |
| prior_pos_std | 0.30 | chair_concept/src/chair_belief.h:66 | broad pose prior std (ceiling of Sigma when clamped to prior) | patch? |
| prior_yaw_std | 0.60 | chair_concept/src/chair_belief.h:67 | broad yaw prior std | patch? |
| kViewBins | 24 | chair_concept/src/chair_belief.h:113 | 15 deg bins of the per-view evidence budget | patch? |
| fd_eps | 1e-3 | chair_concept/src/chair_belief.h:117 | finite-difference step of SDF Jacobian (numerics) | excluded |
| z-band floor | 0.03 | chair_concept/src/chair_belief.cpp:102 | floor of leg/seat/back soft-gate bandwidth | patch |
| seed seat-top quantile | 0.55 | chair_concept/src/chair_fitter.cpp:365 | quantile taken as seat-top at birth (yaw seed) | patch? |
| seed seat/back band | +-0.10 | chair_concept/src/chair_fitter.cpp:373 | vertical bands separating seat and back points in the yaw seed | patch? |
| seed min backrest offset | 0.05 | chair_concept/src/chair_fitter.cpp:387 | backrest offset below which the yaw seed is not used | patch? |
| seat_h floor | 0.10 | chair_concept/src/chair_fitter.cpp:127 | floor of seat height read from graph | patch? |
| obliquity cos clamp | [0.05,1] | chair_concept/src/chair_fitter.cpp:502 | clamp so 1/cos stays finite ("THRESHOLD, flagged, physical") | patch |
| roi sanity gate | \\|off\\|<3, fill<4 | chair_concept/src/chair_fitter.cpp:1083 | reject implausible projected ROI (gate); clamps +-3 follow | patch |
| near-plane guard | Y<=0.20 | chair_concept/src/chair_fitter.cpp:1064 | skip points at/behind the image plane (projection numerics) | excluded? |
| range floor in existence | 0.5 | chair_concept/src/chair_existence.cpp:145 | floor of range in expected-support C/range^2 | patch |
| existence initial/max L | 0.0 / 4.0 | chair_concept/src/chair_instance.h:118 | second copy of the +4 log-odds cap (not read from Existence.MaxLogodds) | patch? |
| birth range ref / power | 2.5 / 2.0 | chair_concept/src/specificworker.cpp:566 | range reference and exponent of birth observation quality (table reads these from config) | patch |
| peripheral track radius | 0.30 | chair_concept/src/specificworker.cpp:961 | radius of other tracks in peripheral attention | patch? |
| support select below_m | 0.05 | chair_concept/src/chair_support_bank.h:40 | margin below the floor when selecting owned points | patch |
| SDF_SMOOTH_K | 0.02 | chair_concept/src/chair_model.h:63 | smoothing width of the compound SDF | patch? |
| SDF_INSIDE_SCALE | 0.3 | chair_concept/src/chair_model.h:64 | scale of the inside-SDF term | patch? |
| kPosEps / kYawEps / min write | 0.003 / 0.005 / 0.05 m | chair_concept/src/chair_scene_graph.cpp:139 | publish-only change dedup (no estimation role) | excluded |
| planner kRef / kTangent / robot radius | 0.10 / 10,6 / 0.30 | chair_concept/src/epistemic_planner.cpp:163 | viewpoint planner constants (planner, not estimator) | excluded? |
| contour raster kPad / kShift | 4.0 / 4 | chair_concept/src/chair_contour.h:117 | rasterisation padding and sub-pixel shift (rendering) | excluded |

### Switches

| name | value | where | what it switches | kind |
|---|---|---|---|---|
| ChairModel.AI2MotionConfirmOnly | false | etc/config.toml [ChairModel] ; read chair_config.cpp:143 | hard confirm-only gate when moving & off-centre | patch mechanism |
| ChairModel.FixationEnabled | true | etc/config.toml [ChairModel] ; read chair_config.cpp:145 | fixation gate (attention) on/off | patch mechanism |
| ChairModel.FixationCentrePrecision | false | etc/config.toml [ChairModel] ; read chair_config.cpp:155 | A/B: centred as precision instead of gate | patch mechanism |
| ChairModel.AI2ModeObsWeighting | true | etc/config.toml [ChairModel] ; read chair_config.cpp:266 | information-weighted yaw-mode votes | patch mechanism |
| ChairConcept.EpistemicOfferEnabled | true | etc/config.toml [ChairConcept] ; read chair_config.cpp:69 | publish epistemic viewpoint offer (planner; comparison arm) | other |
| ChairConcept.RtCovUpload | true | etc/config.toml [ChairConcept] ; read chair_config.cpp:194 | publish pose covariance on RT edge | other |
| ChairConcept.RtCovAddChain | true | etc/config.toml [ChairConcept] ; read chair_config.cpp:198 | add localisation chain covariance to published RT cov | other |
| Tracker.NllCost | false | etc/config.toml [Tracker] ; read chair_config.cpp:272 | association cost choice (NLL vs distance) | other |
| Tracker.RicohBirthEnabled | false | etc/config.toml [Tracker] ; read chair_config.cpp:274 | allow Ricoh slice to birth | patch mechanism |
| Existence.Enabled | true | etc/config.toml [Existence] ; read chair_config.cpp:224 | master switch | patch mechanism |
| Existence.OcclusionCheck | true | etc/config.toml [Existence] ; read chair_config.cpp:254 | suppress vacate negative when occluded | patch mechanism |
| Existence.ContourCheck | true | etc/config.toml [Existence] ; read chair_config.cpp:258 | classifier-free contour existence channel | other |
| Existence.RoomPrior | true | etc/config.toml [Existence] ; read chair_config.cpp:260 | room-containment pose prior (world fact: a centre outside the walls ~ 0) | other |
| RigPrior.Enabled | true | etc/config.toml [RigPrior] ; read chair_config.cpp:228 | master switch | patch mechanism |
| Bearing.BirthEnabled | false | NO toml key; code default at chair_config.cpp:280 | bearing-only hypothesis birth (off) | patch mechanism |

### Excluded, by file

- `etc/config.toml`: 38: `Agent.id`, `Agent.name`, `Agent.configFile`, `Agent.domain`, `Agent.tree`, `Agent.graph`, `Agent.2d`, `Agent.3d`, `Presence.required_agent_names`, `Presence.config_dir`, `Presence.monitor_period_ms`, `Presence.heartbeat_timeout_ms`, `Presence.rejoin_grace_ms`, `Presence.stale_grace_ms`, `Presence.agent_info_period_ms`, `Owns.nodes`, `Period.Compute`, `Period.Emergency`, `Component.Debug.Verbose`, `Media.MasksStallTimeoutMs`, `ChairModel.ColourCsvPath`, `ChairModel.AI2CsvPath`, `ChairModel.AI2FloorZ?`, `ChairModel.AI2GnIters`, `ChairConcept.StateEps`, `ChairConcept.KStable`, `ChairConcept.DetectionAliveMaxFrames?`, `ChairConcept.ObsDistance?`, `ChairConcept.MinStandOffM?`, `ChairConcept.EpistemicCooldownCycles?`, `ChairConcept.ChairLogPeriodFrames`, `ChairConcept.SupportBankMaxPoints`, `ChairConcept.SupportBankQuantizationM`, `Ice.Warn.Connections`, `Ice.Trace.Network`, `Ice.Trace.Protocol`, `Ice.MessageSizeMax`, `RigPrior.StaleMs?`
- `chair_concept/src/chair_belief.h`: 1: `fd_eps`
- `chair_concept/src/chair_fitter.cpp`: 1: `near-plane guard?`
- `chair_concept/src/chair_scene_graph.cpp`: 1: `kPosEps / kYawEps / min write`
- `chair_concept/src/epistemic_planner.cpp`: 1: `planner kRef / kTangent / robot radius?`
- `chair_concept/src/chair_contour.h`: 1: `contour raster kPad / kShift`

## table

### Config keys and hidden defaults

| name | value | where | role | label |
|---|---|---|---|---|
| Agent.id | 7 | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Agent.name | "table_concept" | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Agent.configFile | "" | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Agent.domain | 0 | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Agent.tree | false | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Agent.graph | false | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Agent.2d | false | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Agent.3d | false | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Presence.required_agent_names | ["robot_concept", "room_concept"] | etc/config.toml [Presence] | presence protocol plumbing | excluded |
| Presence.config_dir | "etc" | etc/config.toml [Presence] | presence protocol plumbing | excluded |
| Presence.monitor_period_ms | 500 | etc/config.toml [Presence] | presence protocol plumbing | excluded |
| Presence.heartbeat_timeout_ms | 3000 | etc/config.toml [Presence] | presence protocol plumbing | excluded |
| Presence.rejoin_grace_ms | 1500 | etc/config.toml [Presence] | presence protocol plumbing | excluded |
| Presence.stale_grace_ms | 120000 | etc/config.toml [Presence] | presence protocol plumbing | excluded |
| Presence.agent_info_period_ms | 1000 | etc/config.toml [Presence] | presence protocol plumbing | excluded |
| Owns.nodes | ["table_concept 7", "table*"] | etc/config.toml [Owns] | owned graph nodes (plumbing) | excluded |
| Period.Compute | 100 | etc/config.toml [Period] | loop period | excluded |
| Period.Emergency | 500 | etc/config.toml [Period] | loop period | excluded |
| Component.Debug.Verbose | false | etc/config.toml [Component.Debug] | log switch | excluded |
| Media.MasksStallTimeoutMs | 3000 | etc/config.toml [Media] ; read table_config.cpp:89 | watchdog timeout on the mask stream (stops robot on stale data) | excluded |
| TableModel.SigmaObs | 0.05 | etc/config.toml [TableModel] ; read table_config.cpp:111 | band splitting top/leg (candidate vs residual) membership | patch |
| TableModel.SdfThresholdForStorage | 0.08 | etc/config.toml [TableModel] ; read table_config.cpp:113 | on-surface membership threshold for candidate/residual split | patch |
| TableModel.AI2CsvPath | "etc/ai2_log.csv" | etc/config.toml [TableModel] ; read table_config.cpp:183 | per-cycle CSV path | excluded |
| TableModel.BirthSurpriseProbe | true | etc/config.toml [TableModel] ; read table_config.cpp:187 | read-only diagnostic probe of the residual grid | excluded |
| TableModel.AI2SigmaBaseM | 0.03 | etc/config.toml [TableModel] ; read table_config.cpp:117 | base on-surface observation noise (m); larger than depth noise, includes mask-edge error | sensor? |
| TableModel.PixelSigmaOverF | 0.0015 | etc/config.toml [TableModel] ; read table_config.cpp:189 | pixel noise over focal (intrinsics) | sensor |
| TableModel.DepthSigma0M | 0.006 | etc/config.toml [TableModel] ; read table_config.cpp:191 | ZED depth std floor | sensor |
| TableModel.DepthSigmaRangeCoef | 0.004 | etc/config.toml [TableModel] ; read table_config.cpp:193 | ZED depth std growth with range^2 | sensor |
| TableModel.ModelSigmaM | 0.010 | etc/config.toml [TableModel] ; read table_config.cpp:195 | residual model/surface sigma floor | patch |
| TableModel.DepthTiltStd | 0.020 | etc/config.toml [TableModel] ; read table_config.cpp:201 | ZED shared depth-tilt prior std; consumed only by FootprintResidual (off) | sensor? |
| TableModel.DepthBiasStd | 0.015 | etc/config.toml [TableModel] ; read table_config.cpp:203 | ZED shared depth-bias prior std; consumed only by FootprintResidual (off) | sensor? |
| TableModel.DepthScaleStd | 0.010 | etc/config.toml [TableModel] ; read table_config.cpp:205 | ZED shared depth-scale prior std; consumed only by FootprintResidual (off) | sensor? |
| TableModel.AI2ClutterFrac | 0.10 | etc/config.toml [TableModel] ; read table_config.cpp:125 | prior weight of the uniform clutter mixture (mask error) | patch |
| TableModel.AI2ClutterScaleM | 0.12 | etc/config.toml [TableModel] ; read table_config.cpp:127 | distance beyond which a point is explained by clutter | patch |
| TableModel.AI2PriorSizeStd | 0.30 | etc/config.toml [TableModel] ; read table_config.cpp:129 | broad size prior std | class |
| TableModel.AI2ProcessStdM | 0.005 | etc/config.toml [TableModel] ; read table_config.cpp:131 | process noise on pose = how much furniture moves per frame (permanence); LIFECYCLE says no constant Q | class? |
| TableModel.AI2ProcessStdYaw | 0.01 | etc/config.toml [TableModel] ; read table_config.cpp:133 | process noise on yaw (permanence); LIFECYCLE says no constant Q | class? |
| TableModel.AI2ProcessStdExtentM | 0.0 | etc/config.toml [TableModel] ; read table_config.cpp:135 | process noise on extent (0 = dimensions are constants) | class? |
| TableModel.AI2AgeNominalDtS | 0.1 | etc/config.toml [TableModel] ; read table_config.cpp:139 | nominal mask-stream period (10 Hz); 0 = aging off | sensor? |
| TableModel.AI2TruncGateFrac | 0.10 | etc/config.toml [TableModel] ; read table_config.cpp:167 | skip update if mask border fraction above this (truncation gate) | patch |
| TableModel.FixationMinPts | 300 | etc/config.toml [TableModel] ; read table_config.cpp:171 | fixation gate: minimum mask points | patch |
| TableModel.FixationMaxTrunc | 0.10 | etc/config.toml [TableModel] ; read table_config.cpp:173 | fixation gate: maximum truncation | patch |
| TableModel.FixationRangeM | 0.0 | etc/config.toml [TableModel] ; read table_config.cpp:175 | fixation gate: range cut (retired, 0 = off) | patch |
| TableModel.FixationCentreFrac | 0.60 | etc/config.toml [TableModel] ; read table_config.cpp:177 | fixation gate: fovea radius | patch |
| TableModel.FixationStillDotd | 0.05 | etc/config.toml [TableModel] ; read table_config.cpp:179 | fixation gate: still, mask smear | patch |
| TableModel.AI2RangeNoiseLatPerM | 0.02 | etc/config.toml [TableModel] ; read table_config.cpp:153 | lateral deprojection std growth per metre of range (intrinsics-like) feeding R and position common-mode | sensor? |
| TableModel.AI2RangeNoiseYawPerM | 0.03 | etc/config.toml [TableModel] ; read table_config.cpp:155 | yaw common-mode growth with range (anti far-view rotation) | patch |
| TableModel.AI2RangeNoiseSizePerM | 0.08 | etc/config.toml [TableModel] ; read table_config.cpp:157 | size common-mode growth with range (freeze geometry afar) | patch |
| TableModel.AI2CoverageSizeGain | 0.30 | etc/config.toml [TableModel] ; read table_config.cpp:159 | coverage common-mode: size | patch |
| TableModel.AI2CoverageYawGain | 0.15 | etc/config.toml [TableModel] ; read table_config.cpp:161 | coverage common-mode: yaw | patch |
| TableModel.AI2CoveragePosGain | 0.05 | etc/config.toml [TableModel] ; read table_config.cpp:163 | coverage common-mode: position | patch |
| TableModel.AI2CoverageMin | 0.02 | etc/config.toml [TableModel] ; read table_config.cpp:165 | floor of coverage ratio c | patch |
| TableModel.AI2GnIters | 4 | etc/config.toml [TableModel] ; read table_config.cpp:181 | GN iterations per frame (iteration cap) | excluded |
| TableModel.AI2CommonModePosStd | 0.03 | etc/config.toml [TableModel] ; read table_config.cpp:141 | shared per-frame position error floor | patch |
| TableModel.AI2CommonModeSizeStd | 0.02 | etc/config.toml [TableModel] ; read table_config.cpp:143 | shared per-frame size error floor | patch |
| TableModel.AI2CommonModeYawStd | 0.03 | etc/config.toml [TableModel] ; read table_config.cpp:145 | shared per-frame yaw error floor | patch |
| TableModel.LidarPrecision | 2500.0 | etc/config.toml [TableModel] ; read table_config.cpp:256 | per-ray LiDAR range precision (1/sigma^2 at 2 cm); set as a weight to fix mask under-size | sensor? |
| TableModel.LidarBpearlPrecision | 0.0 | etc/config.toml [TableModel] ; read table_config.cpp:258 | per-ray precision of the low LiDAR (0 = off) | sensor? |
| TableModel.LidarRobustCM | 0.05 | etc/config.toml [TableModel] ; read table_config.cpp:260 | Cauchy scale of LiDAR residual | patch |
| TableModel.LidarSelectMarginM | 0.10 | etc/config.toml [TableModel] ; read table_config.cpp:262 | margin of LiDAR return pre-selection box | patch |
| TableModel.LidarCoverageN0 | 60.0 | etc/config.toml [TableModel] ; read table_config.cpp:264 | ray count for full LiDAR weight | patch |
| TableModel.LidarCoverageAngPower | 1.0 | etc/config.toml [TableModel] ; read table_config.cpp:266 | angular-coverage down-weight exponent | patch |
| TableModel.MaxStepM | 1.0 | etc/config.toml [TableModel] ; read table_config.cpp:268 | reject a GN centre step larger than this (divergence guard) | patch |
| TableModel.CoveragePrecision | 0.0 | etc/config.toml [TableModel] ; read table_config.cpp:270 | grow-only pull of clutter-ceded points (0 = off) | patch |
| TableModel.CoverageRobustCM | 0.15 | etc/config.toml [TableModel] ; read table_config.cpp:272 | Cauchy scale of coverage pull | patch |
| TableModel.FreeSpacePrecision | 500.0 | etc/config.toml [TableModel] ; read table_config.cpp:274 | shrink-only free-space pull per through-beam | patch |
| TableModel.FootprintMomentPrecision | 2000.0 | etc/config.toml [TableModel] ; read table_config.cpp:276 | footprint 2nd-moment factor precision (w,h,yaw) | patch |
| TableModel.FootprintMomentRangePerM | 0.03 | etc/config.toml [TableModel] ; read table_config.cpp:278 | moment variance growth with range | patch |
| TableModel.FeBaselineAdaptDown | 0.05 | etc/config.toml [TableModel] ; read table_config.cpp:280 | free-energy baseline EMA, fast down | patch |
| TableModel.FeBaselineAdaptUp | 0.005 | etc/config.toml [TableModel] ; read table_config.cpp:282 | free-energy baseline EMA, slow up | patch |
| TableModel.FeSurpriseSmooth | 0.10 | etc/config.toml [TableModel] ; read table_config.cpp:284 | smoothing of positive FE gap (surprise) | patch |
| TableModel.FootprintMomentMotionGain | 0.30 | etc/config.toml [TableModel] ; read table_config.cpp:286 | moment variance growth with motion | patch |
| TableModel.OrientationMotionRef | 0.50 | etc/config.toml [TableModel] ; read table_config.cpp:288 | motion at which yaw-mode vote is halved | patch |
| TableModel.FootprintMomentCompletenessGain | 0.5 | etc/config.toml [TableModel] ; read table_config.cpp:292 | moment variance growth with partial view | patch |
| TableModel.FootprintMomentMinCompleteness | 0.02 | etc/config.toml [TableModel] ; read table_config.cpp:294 | clamp floor of 1/completeness | patch |
| TableModel.ObliquityMomentGain | 0.05 | etc/config.toml [TableModel] ; read table_config.cpp:290 | grazing-view cap on moment channel | patch |
| TableModel.ExistenceRemovalProb | 0.12 | etc/config.toml [TableModel] ; read table_config.cpp:300 | removal probability threshold | patch |
| TableModel.ExistenceOutOfRoomGain | 1.5 | etc/config.toml [TableModel] ; read table_config.cpp:302 | log-odds step per cycle outside the room | patch |
| TableModel.ExistenceLogoddsMax | 4.0 | etc/config.toml [TableModel] ; read table_config.cpp:304 | clamp of \|L\| | patch |
| TableModel.ExistenceFrameCorrelation | 0.327 | etc/config.toml [TableModel] ; read table_config.cpp:306 | measured frame-to-frame miss correlation of the detector (rho) | sensor? |
| TableModel.ShapeEvidenceEmaAlpha | 0.15 | etc/config.toml [TableModel] ; read table_config.cpp:105 | EMA weight of round-vs-square evidence | patch |
| TableModel.ExistenceDetectionProb | 0.85 | etc/config.toml [TableModel] ; read table_config.cpp:308 | P(beam returns from occupied footprint) (sensor hit rate) | sensor |
| TableModel.ExistenceClutterProb | 0.05 | etc/config.toml [TableModel] ; read table_config.cpp:310 | P(beam returns from an empty footprint): spurious-return rate of the LiDAR | sensor? |
| TableModel.ExistenceSensorSigmaM | 0.03 | etc/config.toml [TableModel] ; read table_config.cpp:312 | LiDAR range sigma | sensor |
| TableModel.ExistenceRemoveFrames | 15 | etc/config.toml [TableModel] ; read table_config.cpp:314 | removal debounce frames | patch |
| TableModel.ExistenceAbsenceRangeRefM | 2.5 | etc/config.toml [TableModel] ; read table_config.cpp:316 | range below which absence is trusted fully | patch |
| TableModel.ExistenceAbsenceRangePower | 2.0 | etc/config.toml [TableModel] ; read table_config.cpp:318 | decay exponent of absence with range | patch |
| TableModel.ExistenceLosMarginM | 0.25 | etc/config.toml [TableModel] ; read table_config.cpp:320 | line-of-sight margin beyond first LiDAR hit | patch |
| TableModel.ExistenceLosAzimBins | 0 | etc/config.toml [TableModel] ; read table_config.cpp:322 | LiDAR azimuth oracle bins (0 = off) | patch |
| TableModel.ExistenceVerifySurprise | 20.0 | etc/config.toml [TableModel] ; read table_config.cpp:324 | surprise above which object wants verification | patch |
| TableModel.ExistenceVerifyGain | 5.0 | etc/config.toml [TableModel] ; read table_config.cpp:328 | epistemic gain of a verification request | patch |
| TableConcept.ShapeEvalPeriod | 30 | etc/config.toml [TableConcept] ; read table_config.cpp:95 | cycles between shape evaluations (cadence) | excluded |
| TableConcept.ShapeEvalMinPoints | 300 | etc/config.toml [TableConcept] ; read table_config.cpp:97 | points needed before the shape test runs | patch |
| TableConcept.ShapeFitIters | 20 | etc/config.toml [TableConcept] ; read table_config.cpp:101 | iterations per shape hypothesis (iteration cap) | excluded |
| TableConcept.ShapeFitMaxPoints | 1000 | etc/config.toml [TableConcept] ; read table_config.cpp:103 | subsample size for the shape test (cost cap) | excluded |
| TableConcept.StateEps | 0.04 | etc/config.toml [TableConcept] ; read table_config.cpp:63 | convergence status tolerance (m+rad); no effect on the estimate | excluded |
| TableConcept.KStable | 30 | etc/config.toml [TableConcept] ; read table_config.cpp:65 | frames before model_stable status flag | excluded |
| TableConcept.DetectionAliveMaxFrames | 40 | etc/config.toml [TableConcept] ; read table_config.cpp:67 | frames without a mask before detection_alive=false (status flag; check whether existence reads it) | excluded? |
| TableConcept.MatchedFramesBeforeAging | 5 | etc/config.toml [TableConcept] ; read table_config.cpp:69 | matched frames before a stale cycle may age the belief | patch |
| TableConcept.CentralRegionFrac | 0.25 | etc/config.toml [TableConcept] ; read table_config.cpp:71 | central image box where the detector is reliable; part of p_detect | sensor? |
| TableConcept.TableLogPeriodFrames | 0 | etc/config.toml [TableConcept] ; read table_config.cpp:77 | log throttle | excluded |
| TableConcept.ShowDashboard | true | etc/config.toml [TableConcept] ; read table_config.cpp:93 | GUI switch | excluded |
| TableConcept.ObsDistance | 1.8 | etc/config.toml [TableConcept] ; read table_config.cpp:73 | planner stand-off preference (viewpoint planner, not the estimator) | excluded? |
| TableConcept.EpistemicCooldownCycles | 200 | etc/config.toml [TableConcept] ; read table_config.cpp:75 | planner anti-chatter cooldown (viewpoint planner) | excluded? |
| TableConcept.SupportBankMaxPoints | 4000 | etc/config.toml [TableConcept] ; read table_config.cpp:79 | memory cap of the support bank | excluded |
| TableConcept.SupportBankQuantizationM | 0.02 | etc/config.toml [TableConcept] ; read table_config.cpp:81 | dedup grid of the support bank (storage) | excluded |
| TableConcept.SupportSelectRadiusMarginM | 0.50 | etc/config.toml [TableConcept] ; read table_config.cpp:83 | XY ownership margin of support points | patch |
| TableConcept.SupportSelectHeightMarginM | 0.25 | etc/config.toml [TableConcept] ; read table_config.cpp:85 | Z margin of support point selection | patch |
| TableConcept.DumpCloudPath | "/tmp/round_table.xyz" | etc/config.toml [TableConcept] ; read table_config.cpp:107 | diagnostic cloud dump path | excluded |
| TableConcept.RtCovScale | 1.0 | etc/config.toml [TableConcept] ; read table_config.cpp:209 | scale on published RT covariance (calibrate toward NEES 1) | patch |
| TableConcept.ObjectObsFrame | "body" | etc/config.toml [TableConcept] ; read table_config.cpp:213 | frame name of the published observation | excluded |
| Tracker.GateMahalanobis | 9.0 | etc/config.toml [Tracker] ; read table_config.cpp:217 | association gate (chi2) | patch |
| Tracker.GateFallbackM | 0.50 | etc/config.toml [Tracker] ; read table_config.cpp:219 | association metric gate before covariance exists | patch |
| Tracker.DetectionNoiseM | 0.35 | etc/config.toml [Tracker] ; read table_config.cpp:221 | association noise added to cover centroid-vs-fit offset | patch |
| Tracker.RecallPiMove | 0.5 | etc/config.toml [Tracker] ; read table_config.cpp:223 | permanence: P(remembered object was moved) | class |
| Tracker.RecallLogPriorNew | 0.0 | etc/config.toml [Tracker] ; read table_config.cpp:225 | permanence: log prior of a new object | class |
| Tracker.BirthFrames | 8 | etc/config.toml [Tracker] ; read table_config.cpp:227 | frames a mask stays unexplained before birth (debounce) | patch |
| Tracker.BirthFusionGain | 6.0 | etc/config.toml [Tracker] ; read table_config.cpp:231 | extra birth evidence at full corroboration | patch |
| Tracker.BirthFusionMassRef | 8.0 | etc/config.toml [Tracker] ; read table_config.cpp:233 | residual mass at half saturation | patch |
| Tracker.BirthFusionRadiusM | 0.50 | etc/config.toml [Tracker] ; read table_config.cpp:235 | window of residual-mass sample | patch |
| Tracker.BirthMinSepM | 0.60 | etc/config.toml [Tracker] ; read table_config.cpp:237 | minimum separation of a birth from existing objects | patch |
| Tracker.MergeOverlap | 0.05 | etc/config.toml [Tracker] ; read table_config.cpp:239 | footprint overlap that merges two instances | patch |
| Tracker.BirthWidthM | 1.0 | etc/config.toml [Tracker] ; read table_config.cpp:241 | default size of an instance born from a mask (class seed; fit refines it) | class? |
| Tracker.BirthDepthM | 0.6 | etc/config.toml [Tracker] ; read table_config.cpp:243 | default size of an instance born from a mask (class seed; fit refines it) | class? |
| Tracker.BirthHeightM | 0.75 | etc/config.toml [Tracker] ; read table_config.cpp:245 | default size of an instance born from a mask (class seed; fit refines it) | class? |
| Tracker.RicohAttentionConf | 0.60 | etc/config.toml [Tracker] ; read table_config.cpp:247 | min confidence for Ricoh attention | patch |
| Tracker.RicohAttentionAngleMargin | 0.05 | etc/config.toml [Tracker] ; read table_config.cpp:249 | angular margin of Ricoh attention match | patch |
| Tracker.RicohAttentionRangeBandM | 1.0 | etc/config.toml [Tracker] ; read table_config.cpp:251 | Ricoh range roughness band used as a match margin | sensor? |
| Ice.Warn.Connections | "0" | etc/config.toml [Ice] | Ice middleware setting | excluded |
| Ice.Trace.Network | "0" | etc/config.toml [Ice] | Ice middleware setting | excluded |
| Ice.Trace.Protocol | "0" | etc/config.toml [Ice] | Ice middleware setting | excluded |
| Ice.MessageSizeMax | "20004800" | etc/config.toml [Ice] | Ice middleware setting | excluded |
| TableConcept.ShapeEvidenceClamp | 8.0 | NO toml key; code default at table_config.cpp:99 | clamp of shape evidence | patch |
| TableModel.DetectMinFill | 0.10 | NO toml key; code default at table_config.cpp:119 | detector envelope lower shoulder | sensor |
| TableModel.DetectMaxFill | 0.60 | NO toml key; code default at table_config.cpp:121 | detector envelope upper shoulder | sensor |
| TableModel.DetectSoft | 0.06 | NO toml key; code default at table_config.cpp:123 | detector envelope softness | sensor |
| TableModel.MotionCmPosGain | 0.10 | NO toml key; code default at table_config.cpp:147 | ego-motion shared position error per m/s (anti-drift) | patch |
| TableModel.MotionCmSizeGain | 0.20 | NO toml key; code default at table_config.cpp:149 | ego-motion shared size error per m/s (anti-reshape) | patch |
| TableModel.MotionCmYawGain | 0.12 | NO toml key; code default at table_config.cpp:151 | ego-motion shared yaw error per m/s (anti-rotate) | patch |
| TableConcept.DetectProbeCsvPath | "etc/detect_probe.csv" | NO toml key; code default at table_config.cpp:185 | probe CSV path | excluded |
| TableModel.VerifySurpriseSmooth | 0.10 | NO toml key; code default at table_config.cpp:326 | EMA weight of go-verify surprise | patch |
| TableModel.ExistenceScoreCutoff | 0.25 | NO toml key; code default at table_config.cpp:340 | score cutoff of existence grading | patch |
| TableModel.ExistenceScoreViewSigma | 0.5 | NO toml key; code default at table_config.cpp:342 | view sigma of existence grading | patch |
| TableConcept.RicohConfirmDetectionProb | 0.60 | NO toml key; code default at table_config.cpp:353 | Ricoh detection probability (confirm path, off) | sensor |
| TableConcept.RicohConfirmClutterProb | 0.15 | NO toml key; code default at table_config.cpp:355 | Ricoh clutter probability (confirm path, off) | sensor |

### Hard-coded in src

| name | value | where | role | label |
|---|---|---|---|---|
| kDrop | 1e12 | table_concept/src/table_belief.cpp:699 | grow-only guard: a moment that would shrink w/h gets infinite variance (comment: "the file's one hard switch") | patch |
| kYawScale2 | 0.25 rad^2 | table_concept/src/table_belief.cpp:757 | scale of the yaw-continuity cost of the fold | patch |
| flip evidence clamp | +-6.0 | table_concept/src/table_belief.cpp:816 | clamp of w-h mode evidence (same as chair kFlipClamp) | patch |
| size floors | w,h>=0.10, H>=top+0.05 | table_concept/src/table_belief.cpp:163 | "physical floors, not tuned gates" (comment) | patch? |
| on-plane skip | 0.05 | table_concept/src/table_belief.cpp:354 | coverage pull ignores points with top-plane weight below this | patch? |
| top band half-width | 3 sigma | table_concept/src/table_belief.cpp:462 | half-width of the slab z-band in sigma_base units (three sites) | patch? |
| prior std pose | 0.30 m, 0.60 rad | table_concept/src/table_belief.cpp:200 | broad prior covariance of cx,cy,yaw (ceiling when clamped to prior) | patch? |
| kObliquityYawGain | 0.05 | table_concept/src/table_fitter.cpp:648 | grazing-view yaw cap | patch |
| obliquity cos clamp | [0.05,1] | table_concept/src/table_fitter.cpp:649 | clamp so 1/cos stays finite | patch |
| range floor | 0.1 | table_concept/src/table_fitter.cpp:125 | floor of range in frame weight | patch |
| graph-attr fallback seeds | w 1.0, h 0.6, H 0.75 | table_concept/src/table_fitter.cpp:233 | size used when the graph node lacks a size attribute (class seed) | class? |
| birth top quantile | 0.95 | table_concept/src/table_fitter.cpp:406 | quantile taken as the tabletop at birth | patch? |
| birth anisotropy test | 0.10 | table_concept/src/table_fitter.cpp:433 | anisotropy above which the moment yaw seeds the box | patch? |
| birth size floor | 0.10 | table_concept/src/table_fitter.cpp:426 | floor of seeded w,h | patch? |
| round seed quantile/band/scale | 0.90 / 0.08 / 1.3, 1.2, 0.8 | table_concept/src/table_fitter.cpp:849 | round-vs-square seed from mean radius | patch? |
| round seed radius clamp | [0.2,1.5] and [0.2,3.0] | table_concept/src/table_fitter.cpp:853 | clamps of seeded radius / extents | patch |
| kEqualVolumeDilation | 1.4142 | table_concept/src/table_existence.cpp:380 | shell around the box with equal volume (clutter contrast) | patch? |
| roi sanity gate | \\|off\\|<3, fill<4 | table_concept/src/table_projection.cpp:186 | reject implausible projected ROI; clamps +-3 and [0,4] follow | patch |
| near-plane guard | Y<=0.20 | table_concept/src/table_projection.cpp:161 | skip points at/behind the image plane (three sites) | excluded? |
| spatial hash CELL | 6.0 px | table_concept/src/table_projection.cpp:239 | cell size of a pixel spatial hash | excluded |
| support select below_m | 0.05 | table_concept/src/table_support_bank.h:42 | margin below the floor when selecting owned points | patch |
| birth periphery ref (literal) | 0.50 | table_concept/src/specificworker_lifecycle.cpp:142 | periphery reference of birth quality, hard-coded (chair/fridge/hood read AI2PeriphRef) | patch |
| TOP_THICKNESS / LEG_RADIUS | 0.03 / 0.025 | table_concept/src/table_model.h:59 | template top thickness and leg radius | class |
| round table ped_radius / ring_frac | 0.06 / 0.80 | table_concept/src/round_table_belief.h:71 | round-table template shape | class |
| SDF_SMOOTH_K | 0.02 | table_concept/src/table_model.h:61 | smoothing width of the compound SDF | patch? |
| SDF_INSIDE_SCALE | 0.3 | table_concept/src/table_model.h:62 | scale of the inside-SDF term | patch? |
| kYawJumpLogRad | 0.087 | table_concept/src/table_fitter.cpp:781 | diagnostic print trigger only | excluded |
| kPosEps / kYawEps | 0.003 / 0.005 | table_concept/src/table_scene_graph.cpp:145 | publish-only change dedup | excluded |
| planner kRef / kTangent | 0.10 / 10,6 | table_concept/src/epistemic_planner.cpp:150 | viewpoint planner constants | excluded? |

### Switches

| name | value | where | what it switches | kind |
|---|---|---|---|---|
| Masks.legacy_room_frame | false | etc/config.toml [Masks] ; read table_config.cpp:91 | A/B scaffolding: legacy room-frame convention for masks | other |
| TableModel.FootprintResidual | false | etc/config.toml [TableModel] ; read table_config.cpp:197 | footprint residual + depth nuisance (grazing-yaw fix) | patch mechanism |
| TableModel.QuotientChart | false | etc/config.toml [TableModel] ; read table_config.cpp:199 | symmetry-quotient chart (yaw snaps) | patch mechanism |
| TableModel.AI2ClampSigmaToPrior | true | etc/config.toml [TableModel] ; read table_config.cpp:137 | cap Sigma at the prior on predict | patch mechanism |
| TableModel.FixationEnabled | true | etc/config.toml [TableModel] ; read table_config.cpp:169 | fixation gate (attention) on/off | patch mechanism |
| TableModel.ExistenceRemovalEnabled | true | etc/config.toml [TableModel] ; read table_config.cpp:298 | enable log-odds removal | patch mechanism |
| TableModel.ExistenceLidarConfirms | false | etc/config.toml [TableModel] ; read table_config.cpp:334 | LiDAR votes on existence | patch mechanism |
| TableModel.ExistenceLocalClutter | true | etc/config.toml [TableModel] ; read table_config.cpp:332 | measure clutter prior by local shell contrast | patch mechanism |
| TableModel.ExistenceLegOccupancy | true | etc/config.toml [TableModel] ; read table_config.cpp:330 | carve leg volumes for occupancy | patch mechanism |
| TableConcept.PublishObjectObs | true | etc/config.toml [TableConcept] ; read table_config.cpp:211 | publish per-frame observation to room anchor factor | other |
| Tracker.BirthFusion | true | etc/config.toml [Tracker] ; read table_config.cpp:229 | fused birth evidence from residual grid | patch mechanism |
| TableModel.ExistenceAbsenceEnvelope | true | NO toml key; code default at table_config.cpp:336 | switch of a patch mechanism | patch mechanism |
| TableModel.ExistenceScoreGrade | true | NO toml key; code default at table_config.cpp:338 | switch of a patch mechanism | patch mechanism |
| TableModel.ExistenceContourCheck | true | NO toml key; code default at table_config.cpp:344 | classifier-free contour existence channel | other |
| TableModel.ExistenceBlindSpotMap | true | NO toml key; code default at table_config.cpp:346 | switch of a patch mechanism | patch mechanism |
| TableConcept.RicohConfirmEnabled | false | NO toml key; code default at table_config.cpp:351 | switch of a patch mechanism | patch mechanism |

### Excluded, by file

- `etc/config.toml`: 42: `Agent.id`, `Agent.name`, `Agent.configFile`, `Agent.domain`, `Agent.tree`, `Agent.graph`, `Agent.2d`, `Agent.3d`, `Presence.required_agent_names`, `Presence.config_dir`, `Presence.monitor_period_ms`, `Presence.heartbeat_timeout_ms`, `Presence.rejoin_grace_ms`, `Presence.stale_grace_ms`, `Presence.agent_info_period_ms`, `Owns.nodes`, `Period.Compute`, `Period.Emergency`, `Component.Debug.Verbose`, `Media.MasksStallTimeoutMs`, `TableModel.AI2CsvPath`, `TableModel.BirthSurpriseProbe`, `TableModel.AI2GnIters`, `TableConcept.ShapeEvalPeriod`, `TableConcept.ShapeFitIters`, `TableConcept.ShapeFitMaxPoints`, `TableConcept.StateEps`, `TableConcept.KStable`, `TableConcept.DetectionAliveMaxFrames?`, `TableConcept.TableLogPeriodFrames`, `TableConcept.ShowDashboard`, `TableConcept.ObsDistance?`, `TableConcept.EpistemicCooldownCycles?`, `TableConcept.SupportBankMaxPoints`, `TableConcept.SupportBankQuantizationM`, `TableConcept.DumpCloudPath`, `TableConcept.ObjectObsFrame`, `Ice.Warn.Connections`, `Ice.Trace.Network`, `Ice.Trace.Protocol`, `Ice.MessageSizeMax`, `TableConcept.DetectProbeCsvPath`
- `table_concept/src/table_projection.cpp`: 2: `near-plane guard?`, `spatial hash CELL`
- `table_concept/src/table_fitter.cpp`: 1: `kYawJumpLogRad`
- `table_concept/src/table_scene_graph.cpp`: 1: `kPosEps / kYawEps`
- `table_concept/src/epistemic_planner.cpp`: 1: `planner kRef / kTangent?`

## refrigerator

### Config keys and hidden defaults

| name | value | where | role | label |
|---|---|---|---|---|
| Agent.id | 22 | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Agent.name | "refrigerator_concept" | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Agent.configFile | "" | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Agent.domain | 0 | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Agent.tree | false | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Agent.graph | false | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Agent.2d | false | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Agent.3d | false | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Presence.required_agent_names | ["robot_concept", "room_concept"] | etc/config.toml [Presence] | presence protocol plumbing | excluded |
| Presence.config_dir | "etc" | etc/config.toml [Presence] | presence protocol plumbing | excluded |
| Presence.monitor_period_ms | 500 | etc/config.toml [Presence] | presence protocol plumbing | excluded |
| Presence.heartbeat_timeout_ms | 3000 | etc/config.toml [Presence] | presence protocol plumbing | excluded |
| Presence.rejoin_grace_ms | 1500 | etc/config.toml [Presence] | presence protocol plumbing | excluded |
| Presence.stale_grace_ms | 120000 | etc/config.toml [Presence] | presence protocol plumbing | excluded |
| Presence.agent_info_period_ms | 1000 | etc/config.toml [Presence] | presence protocol plumbing | excluded |
| Owns.nodes | ["refrigerator_concept 22", "refrigerator_*"] | etc/config.toml [Owns] | owned graph nodes (plumbing) | excluded |
| Period.Compute | 100 | etc/config.toml [Period] | loop period | excluded |
| Period.Emergency | 500 | etc/config.toml [Period] | loop period | excluded |
| Component.Debug.Verbose | false | etc/config.toml [Component.Debug] | log switch | excluded |
| Media.MasksStallTimeoutMs | 3000 | etc/config.toml [Media] ; read refrigerator_config.cpp:79 | watchdog timeout on the mask stream (stops robot on stale data) | excluded |
| RefrigeratorModel.SigmaObs | 0.05 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:93 | band splitting top/leg (candidate vs residual) membership | patch |
| RefrigeratorModel.SdfThresholdForStorage | 0.08 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:95 | on-surface membership threshold for candidate/residual split | patch |
| RefrigeratorModel.AI2CsvPath | "etc/ai2_log.csv" | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:200 | per-cycle CSV path | excluded |
| RefrigeratorModel.BirthSurpriseProbe | true | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:204 | read-only diagnostic probe of the residual grid | excluded |
| RefrigeratorModel.AI2SigmaBaseM | 0.03 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:99 | base on-surface observation noise (m); larger than depth noise, includes mask-edge error | sensor? |
| RefrigeratorModel.PixelSigmaOverF | 0.0015 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:206 | pixel noise over focal (intrinsics) | sensor |
| RefrigeratorModel.DepthSigma0M | 0.006 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:208 | ZED depth std floor | sensor |
| RefrigeratorModel.DepthSigmaRangeCoef | 0.004 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:210 | ZED depth std growth with range^2 | sensor |
| RefrigeratorModel.ModelSigmaM | 0.010 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:212 | residual model/surface sigma floor | patch |
| RefrigeratorModel.DepthTiltStd | 0.020 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:218 | ZED shared depth-tilt prior std; consumed only by FootprintResidual (off) | sensor? |
| RefrigeratorModel.DepthBiasStd | 0.015 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:220 | ZED shared depth-bias prior std; consumed only by FootprintResidual (off) | sensor? |
| RefrigeratorModel.DepthScaleStd | 0.010 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:222 | ZED shared depth-scale prior std; consumed only by FootprintResidual (off) | sensor? |
| RefrigeratorModel.AI2ClutterFrac | 0.10 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:101 | prior weight of the uniform clutter mixture (mask error) | patch |
| RefrigeratorModel.AI2ClutterScaleM | 0.12 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:103 | distance beyond which a point is explained by clutter | patch |
| RefrigeratorModel.AI2PriorSizeStd | 0.30 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:105 | broad size prior std | class |
| RefrigeratorModel.AI2PriorFootprintM | 0.60 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:107 | class footprint prior mean | class |
| RefrigeratorModel.AI2PriorFootprintStd | 0.08 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:109 | class footprint prior std | class |
| RefrigeratorModel.AI2PriorHeightM | 1.90 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:114 | class height prior mean | class |
| RefrigeratorModel.AI2PriorHeightStd | 0.30 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:116 | class height prior std | class |
| RefrigeratorModel.AI2DepthUnobsPrecision | 20000.0 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:118 | extra depth precision when depth is unobserved (single face) | patch |
| RefrigeratorModel.AI2DepthObsBandM | 0.10 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:120 | spread below which cloud is a single depth face | patch |
| RefrigeratorModel.AI2TopNoFloatPrecision | 10000.0 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:122 | two-sided anchor pinning H to observed top | patch |
| RefrigeratorModel.AI2TopNoFloatMarginM | 0.02 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:124 | upward allowance of the top anchor | patch |
| RefrigeratorModel.AI2TopOversegSigmaPerM | 2.0 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:126 | R inflation above the robust top (over-segmentation tail) | patch |
| RefrigeratorModel.AspectScale | 0.15 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:228 | class aspect scale of the plausibility filter | class? |
| RefrigeratorModel.SizeScale | 0.15 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:230 | class footprint scale of the plausibility filter | class? |
| RefrigeratorModel.AltSizeScale | 0.60 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:232 | "other furniture" alternative size std of class evidence | class? |
| RefrigeratorModel.HeightPlausibleMin | 1.20 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:234 | class height below which the object is implausible | class? |
| RefrigeratorModel.HeightSoft | 0.15 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:236 | softness of the class height logistic | class? |
| RefrigeratorModel.FeRef | 2.0 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:238 | reference FE of a healthy fit (plausibility) | patch |
| RefrigeratorModel.FeScale | 1.0 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:240 | scale of FE excess in plausibility | patch |
| RefrigeratorModel.PlausClamp | 8.0 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:242 | clamp of plausibility evidence | patch |
| RefrigeratorModel.PlausHeightPriorGain | 2000.0 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:244 | precision of short-height prior | patch |
| RefrigeratorModel.PlausToExistenceGain | 1.5 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:246 | gain from plausibility to existence class term | patch |
| RefrigeratorModel.SingletonInhibition | 1.0 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:248 | mutual inhibition of neighbours in class existence | patch |
| RefrigeratorModel.AI2WallPrecision | 400.0 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:128 | precision of wall-flush factor at zero gap (class fact is WallFlushPrior) | patch? |
| RefrigeratorModel.AI2WallParallelPrecision | 200.0 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:130 | precision of back-face-parallel-to-wall factor | patch? |
| RefrigeratorModel.AI2WallReachM | 0.15 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:132 | gap scale over which flush weight decays | patch? |
| RefrigeratorModel.AI2DoorClearanceGain | 3.0 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:134 | nats of preference for door facing into the room; class fact (door side) scaled by a gain | patch? |
| RefrigeratorModel.DetectMinFill | 0.0 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:136 | detector envelope lower shoulder | sensor |
| RefrigeratorModel.DetectMaxFill | 1.32 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:138 | detector envelope upper shoulder | sensor |
| RefrigeratorModel.DetectSoft | 0.26 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:140 | detector envelope softness | sensor |
| RefrigeratorModel.AI2VolatilityLr | 0.02 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:144 | learning rate of inferred volatility | patch |
| RefrigeratorModel.AI2VolatilitySigma | 2.0 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:146 | hyperprior width of inferred volatility | patch |
| RefrigeratorModel.AI2WallExplainFrac | 0.25 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:148 | weight of "point belongs to a room wall" mixture component | patch |
| RefrigeratorModel.AI2WallExplainSigmaM | 0.05 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:150 | wall plane slack | patch |
| RefrigeratorModel.WallNoCrossPrecision | 2000.0 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:152 | one-sided precision against wall penetration | patch |
| RefrigeratorModel.WallNoCrossMarginM | 0.0 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:154 | margin of wall no-cross constraint | patch |
| RefrigeratorModel.AI2ProcessStdM | 0.005 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:156 | process noise on pose = how much furniture moves per frame (permanence); LIFECYCLE says no constant Q | class? |
| RefrigeratorModel.AI2ProcessStdYaw | 0.01 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:158 | process noise on yaw (permanence); LIFECYCLE says no constant Q | class? |
| RefrigeratorModel.AI2AgeNominalDtS | 0.1 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:160 | nominal mask-stream period (10 Hz); 0 = aging off | sensor? |
| RefrigeratorModel.AI2TruncGateFrac | 0.10 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:196 | skip update if mask border fraction above this (truncation gate) | patch |
| RefrigeratorModel.AI2RangeNoiseLatPerM | 0.02 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:190 | lateral deprojection std growth per metre of range (intrinsics-like) feeding R and position common-mode | sensor? |
| RefrigeratorModel.AI2RangeNoiseYawPerM | 0.03 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:192 | yaw common-mode growth with range (anti far-view rotation) | patch |
| RefrigeratorModel.AI2RangeNoiseSizePerM | 0.08 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:194 | size common-mode growth with range (freeze geometry afar) | patch |
| RefrigeratorModel.AI2GnIters | 4 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:198 | GN iterations per frame (iteration cap) | excluded |
| RefrigeratorModel.AI2CommonModePosStd | 0.03 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:162 | shared per-frame position error floor | patch |
| RefrigeratorModel.AI2CommonModeSizeStd | 0.02 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:164 | shared per-frame size error floor | patch |
| RefrigeratorModel.AI2CommonModeYawStd | 0.03 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:166 | shared per-frame yaw error floor | patch |
| RefrigeratorModel.MotionCmPosGain | 0.10 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:168 | ego-motion shared position error per m/s (anti-drift) | patch |
| RefrigeratorModel.MotionCmSizeGain | 0.20 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:170 | ego-motion shared size error per m/s (anti-reshape) | patch |
| RefrigeratorModel.MotionCmYawGain | 0.12 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:172 | ego-motion shared yaw error per m/s (anti-rotate) | patch |
| RefrigeratorModel.AI2AngLeverM | 2.0 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:174 | rad/s to m/s lever for ego angular motion | patch |
| RefrigeratorModel.AI2PeriphRef | 0.50 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:176 | centroid radius where periphery penalty saturates | patch |
| RefrigeratorModel.AI2MotionRefMps | 0.60 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:178 | motion at which a peripheral frame is unreliable | patch |
| RefrigeratorModel.AI2StillLinMps | 0.05 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:182 | still threshold: camera linear speed | patch |
| RefrigeratorModel.AI2StillAngRadps | 0.10 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:184 | still threshold: camera angular speed | patch |
| RefrigeratorModel.AI2StillDotd | 0.05 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:186 | still threshold: mask ego-motion smear | patch |
| RefrigeratorModel.AI2MovingUpdateCenterRadius | 0.35 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:188 | centroid radius below which a moving update is allowed | patch |
| RefrigeratorModel.LidarPrecision | 2500.0 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:313 | per-ray LiDAR range precision (1/sigma^2 at 2 cm); set as a weight to fix mask under-size | sensor? |
| RefrigeratorModel.LidarBpearlPrecision | 0.0 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:315 | per-ray precision of the low LiDAR (0 = off) | sensor? |
| RefrigeratorModel.LidarRobustCM | 0.05 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:317 | Cauchy scale of LiDAR residual | patch |
| RefrigeratorModel.LidarSelectMarginM | 0.10 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:319 | margin of LiDAR return pre-selection box | patch |
| RefrigeratorModel.LidarCoverageN0 | 60.0 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:321 | ray count for full LiDAR weight | patch |
| RefrigeratorModel.LidarCoverageAngPower | 1.0 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:323 | angular-coverage down-weight exponent | patch |
| RefrigeratorModel.MaxStepM | 1.0 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:325 | reject a GN centre step larger than this (divergence guard) | patch |
| RefrigeratorModel.CoveragePrecision | 0.0 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:327 | grow-only pull of clutter-ceded points (0 = off) | patch |
| RefrigeratorModel.CoverageRobustCM | 0.15 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:329 | Cauchy scale of coverage pull | patch |
| RefrigeratorModel.FreeSpacePrecision | 500.0 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:331 | shrink-only free-space pull per through-beam | patch |
| RefrigeratorModel.FootprintMomentPrecision | 2000.0 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:333 | footprint 2nd-moment factor precision (w,h,yaw) | patch |
| RefrigeratorModel.FootprintMomentRangePerM | 0.03 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:335 | moment variance growth with range | patch |
| RefrigeratorModel.FeBaselineAdaptDown | 0.05 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:337 | free-energy baseline EMA, fast down | patch |
| RefrigeratorModel.FeBaselineAdaptUp | 0.005 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:339 | free-energy baseline EMA, slow up | patch |
| RefrigeratorModel.FeSurpriseSmooth | 0.10 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:341 | smoothing of positive FE gap (surprise) | patch |
| RefrigeratorModel.FootprintMomentMotionGain | 0.30 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:343 | moment variance growth with motion | patch |
| RefrigeratorModel.OrientationMotionRef | 0.50 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:345 | motion at which yaw-mode vote is halved | patch |
| RefrigeratorModel.FootprintMomentCompletenessGain | 0.5 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:361 | moment variance growth with partial view | patch |
| RefrigeratorModel.FootprintMomentMinCompleteness | 0.02 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:363 | clamp floor of 1/completeness | patch |
| RefrigeratorModel.ObliquityMomentGain | 0.05 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:359 | grazing-view cap on moment channel | patch |
| RefrigeratorModel.ExistenceRemovalProb | 0.12 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:369 | removal probability threshold | patch |
| RefrigeratorModel.ExistenceOutOfRoomGain | 1.5 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:371 | log-odds step per cycle outside the room | patch |
| RefrigeratorModel.ExistenceLogoddsMax | 4.0 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:375 | clamp of \|L\| | patch |
| RefrigeratorModel.ExistenceDetectionProb | 0.85 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:377 | P(beam returns from occupied footprint) (sensor hit rate) | sensor |
| RefrigeratorModel.ExistenceClutterProb | 0.05 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:379 | P(beam returns from an empty footprint): spurious-return rate of the LiDAR | sensor? |
| RefrigeratorModel.ExistenceSensorSigmaM | 0.03 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:381 | LiDAR range sigma | sensor |
| RefrigeratorModel.ExistenceRemoveFrames | 15 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:383 | removal debounce frames | patch |
| RefrigeratorModel.ExistenceAbsenceRangeRefM | 2.5 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:385 | range below which absence is trusted fully | patch |
| RefrigeratorModel.ExistenceAbsenceRangePower | 2.0 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:387 | decay exponent of absence with range | patch |
| RefrigeratorModel.ExistenceVerifySurprise | 20.0 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:389 | surprise above which object wants verification | patch |
| RefrigeratorModel.ExistenceVerifyGain | 5.0 | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:393 | epistemic gain of a verification request | patch |
| RefrigeratorConcept.StateEps | 0.04 | etc/config.toml [RefrigeratorConcept] ; read refrigerator_config.cpp:55 | convergence status tolerance (m+rad); no effect on the estimate | excluded |
| RefrigeratorConcept.KStable | 30 | etc/config.toml [RefrigeratorConcept] ; read refrigerator_config.cpp:57 | frames before model_stable status flag | excluded |
| RefrigeratorConcept.DetectionAliveMaxFrames | 40 | etc/config.toml [RefrigeratorConcept] ; read refrigerator_config.cpp:59 | frames without a mask before detection_alive=false (status flag; check whether existence reads it) | excluded? |
| RefrigeratorConcept.MatchedFramesBeforeAging | 5 | etc/config.toml [RefrigeratorConcept] ; read refrigerator_config.cpp:61 | matched frames before a stale cycle may age the belief | patch |
| RefrigeratorConcept.CentralRegionFrac | 0.25 | etc/config.toml [RefrigeratorConcept] ; read refrigerator_config.cpp:63 | central image box where the detector is reliable; part of p_detect | sensor? |
| RefrigeratorConcept.FridgeFilterLog | false | etc/config.toml [RefrigeratorConcept] ; read refrigerator_config.cpp:250 | diagnostic log switch | excluded |
| RefrigeratorConcept.FrontMinFaceAreaPx | 900.0 | etc/config.toml [RefrigeratorConcept] ; read refrigerator_config.cpp:353 | min projected face area to score door-ness | patch |
| RefrigeratorConcept.FrontMinConfidence | 0.10 | etc/config.toml [RefrigeratorConcept] ; read refrigerator_config.cpp:355 | min door-ness margin to emit a cue | patch |
| RefrigeratorConcept.FrontLog | true | etc/config.toml [RefrigeratorConcept] ; read refrigerator_config.cpp:357 | diagnostic log switch | excluded |
| RefrigeratorConcept.RefrigeratorLogPeriodFrames | 30 | etc/config.toml [RefrigeratorConcept] ; read refrigerator_config.cpp:67 | log throttle | excluded |
| RefrigeratorConcept.ShowDashboard | true | etc/config.toml [RefrigeratorConcept] ; read refrigerator_config.cpp:81 | GUI switch | excluded |
| RefrigeratorConcept.EpistemicCooldownCycles | 200 | etc/config.toml [RefrigeratorConcept] ; read refrigerator_config.cpp:65 | planner anti-chatter cooldown (viewpoint planner) | excluded? |
| RefrigeratorConcept.SupportBankMaxPoints | 4000 | etc/config.toml [RefrigeratorConcept] ; read refrigerator_config.cpp:69 | memory cap of the support bank | excluded |
| RefrigeratorConcept.SupportBankQuantizationM | 0.02 | etc/config.toml [RefrigeratorConcept] ; read refrigerator_config.cpp:71 | dedup grid of the support bank (storage) | excluded |
| RefrigeratorConcept.SupportSelectRadiusMarginM | 0.50 | etc/config.toml [RefrigeratorConcept] ; read refrigerator_config.cpp:73 | XY ownership margin of support points | patch |
| RefrigeratorConcept.SupportSelectHeightMarginM | 0.25 | etc/config.toml [RefrigeratorConcept] ; read refrigerator_config.cpp:75 | Z margin of support point selection | patch |
| RefrigeratorConcept.DumpCloudPath | "/tmp/round_refrigerator.xyz" | etc/config.toml [RefrigeratorConcept] ; read refrigerator_config.cpp:89 | diagnostic cloud dump path | excluded |
| RefrigeratorConcept.RtCovScale | 1.0 | etc/config.toml [RefrigeratorConcept] ; read refrigerator_config.cpp:254 | scale on published RT covariance (calibrate toward NEES 1) | patch |
| RefrigeratorConcept.ObjectObsFrame | "body" | etc/config.toml [RefrigeratorConcept] ; read refrigerator_config.cpp:258 | frame name of the published observation | excluded |
| Tracker.GateMahalanobis | 9.0 | etc/config.toml [Tracker] ; read refrigerator_config.cpp:262 | association gate (chi2) | patch |
| Tracker.GateFallbackM | 0.50 | etc/config.toml [Tracker] ; read refrigerator_config.cpp:264 | association metric gate before covariance exists | patch |
| Tracker.DetectionNoiseM | 0.35 | etc/config.toml [Tracker] ; read refrigerator_config.cpp:266 | association noise added to cover centroid-vs-fit offset | patch |
| Tracker.RecallPiMove | 0.05 | etc/config.toml [Tracker] ; read refrigerator_config.cpp:268 | permanence: P(remembered object was moved) | class |
| Tracker.RecallLogPriorNew | 0.0 | etc/config.toml [Tracker] ; read refrigerator_config.cpp:270 | permanence: log prior of a new object | class |
| Tracker.BirthFrames | 8 | etc/config.toml [Tracker] ; read refrigerator_config.cpp:272 | frames a mask stays unexplained before birth (debounce) | patch |
| Tracker.BirthFusionGain | 6.0 | etc/config.toml [Tracker] ; read refrigerator_config.cpp:276 | extra birth evidence at full corroboration | patch |
| Tracker.BirthFusionMassRef | 8.0 | etc/config.toml [Tracker] ; read refrigerator_config.cpp:278 | residual mass at half saturation | patch |
| Tracker.BirthFusionRadiusM | 0.50 | etc/config.toml [Tracker] ; read refrigerator_config.cpp:280 | window of residual-mass sample | patch |
| Tracker.BirthMinSepM | 0.60 | etc/config.toml [Tracker] ; read refrigerator_config.cpp:282 | minimum separation of a birth from existing objects | patch |
| Tracker.MergeOverlap | 0.05 | etc/config.toml [Tracker] ; read refrigerator_config.cpp:284 | footprint overlap that merges two instances | patch |
| Tracker.BirthWidthM | 1.0 | etc/config.toml [Tracker] ; read refrigerator_config.cpp:286 | default size of an instance born from a mask (class seed; fit refines it) | class? |
| Tracker.BirthDepthM | 0.6 | etc/config.toml [Tracker] ; read refrigerator_config.cpp:288 | default size of an instance born from a mask (class seed; fit refines it) | class? |
| Tracker.BirthHeightM | 0.75 | etc/config.toml [Tracker] ; read refrigerator_config.cpp:290 | default size of an instance born from a mask (class seed; fit refines it) | class? |
| Tracker.RicohAttentionConf | 0.60 | etc/config.toml [Tracker] ; read refrigerator_config.cpp:304 | min confidence for Ricoh attention | patch |
| Tracker.RicohAttentionAngleMargin | 0.05 | etc/config.toml [Tracker] ; read refrigerator_config.cpp:306 | angular margin of Ricoh attention match | patch |
| Tracker.RicohAttentionRangeBandM | 1.0 | etc/config.toml [Tracker] ; read refrigerator_config.cpp:308 | Ricoh range roughness band used as a match margin | sensor? |
| Ice.Warn.Connections | "0" | etc/config.toml [Ice] | Ice middleware setting | excluded |
| Ice.Trace.Network | "0" | etc/config.toml [Ice] | Ice middleware setting | excluded |
| Ice.Trace.Protocol | "0" | etc/config.toml [Ice] | Ice middleware setting | excluded |
| Ice.MessageSizeMax | "20004800" | etc/config.toml [Ice] | Ice middleware setting | excluded |
| RefrigeratorConcept.ShapeEvalPeriod | 30 | NO toml key; code default at refrigerator_config.cpp:83 | cycles between shape evaluations (cadence) | excluded |
| RefrigeratorConcept.ShapeEvalMinPoints | 300 | NO toml key; code default at refrigerator_config.cpp:85 | points needed before the shape test runs | patch |
| RefrigeratorConcept.ShapeEvidenceClamp | 8.0 | NO toml key; code default at refrigerator_config.cpp:87 | clamp of shape evidence | patch |
| RefrigeratorConcept.DetectProbeCsvPath | "etc/detect_probe.csv" | NO toml key; code default at refrigerator_config.cpp:202 | probe CSV path | excluded |
| Tracker.BirthFragmentVoxelM | 0.03 | NO toml key; code default at refrigerator_config.cpp:295 | dedup grid of birth burst | patch |
| Tracker.BirthFragmentMaxPts | 20000 | NO toml key; code default at refrigerator_config.cpp:297 | hard point cap of the birth burst (memory) | excluded |
| Tracker.BirthFragmentDeltaMs | 4000 | NO toml key; code default at refrigerator_config.cpp:300 | time window of the birth burst kept (ms) | patch? |
| Tracker.BirthAdmitPlausibility | 0.35 | NO toml key; code default at refrigerator_config.cpp:302 | birth admission plausibility (0 = off) | patch |
| RefrigeratorModel.ExistenceFrameCorrelation | 0.0 | NO toml key; code default at refrigerator_config.cpp:373 | measured frame-to-frame miss correlation of the detector (rho) | sensor? |
| RefrigeratorModel.VerifySurpriseSmooth | 0.10 | NO toml key; code default at refrigerator_config.cpp:391 | EMA weight of go-verify surprise | patch |
| RefrigeratorConcept.RicohConfirmDetectionProb | 0.60 | NO toml key; code default at refrigerator_config.cpp:400 | Ricoh detection probability (confirm path, off) | sensor |
| RefrigeratorConcept.RicohConfirmClutterProb | 0.15 | NO toml key; code default at refrigerator_config.cpp:402 | Ricoh clutter probability (confirm path, off) | sensor |

### Hard-coded in src

| name | value | where | role | label |
|---|---|---|---|---|
| kLambda (volatility memory) | 0.995 | refrigerator_concept/src/refrigerator_belief.cpp:264 | forgetting factor of inferred-volatility sufficient statistics (~200 observations) | patch? |
| omega upper clamp | 4.0 | refrigerator_concept/src/refrigerator_belief.cpp:274 | clamp of log process noise omega | patch |
| volatility_omega_min | -18.0 | refrigerator_concept/src/refrigerator_belief.h:192 | numerical floor of omega ("not a tuning knob") | excluded |
| jac_slope_clamp | 5.0 | refrigerator_concept/src/refrigerator_belief.h:209 | clamp per-DOF SDF slope (guards seam-point information spike) | patch |
| kFrontClamp | 6.0 | refrigerator_concept/src/refrigerator_belief.cpp:694 | clamp of door-mode evidence, +-6 nats (also line of the re-baseline clamp) | patch |
| front_acc re-baseline clamp | +-6.0 | refrigerator_concept/src/refrigerator_belief.cpp:668 | second use of the same +-6 clamp | patch |
| kRectAspect | 0.10 | refrigerator_concept/src/refrigerator_belief.h:485 | \|w-h\|/(w+h) above which footprint counts as rectangular | patch? |
| front_view_budget_ | 3.0 | refrigerator_concept/src/refrigerator_belief.h:493 | mode-evidence budget per observer-bearing bin (mirrors chair view_budget) | patch |
| kFrontViewBins | 24 | refrigerator_concept/src/refrigerator_belief.h:491 | 15 deg bins of the view budget | patch? |
| size floors w,h,H | 0.10 | refrigerator_concept/src/refrigerator_belief.cpp:182 | clamp: minimum w,h,H of the state | patch? |
| wall segment min length | 0.1 | refrigerator_concept/src/refrigerator_belief.cpp:435 | wall segment shorter than this gives no parallel-wall precision | patch? |
| wall fit sigma (belief.h) | 0.02 | refrigerator_concept/src/refrigerator_belief.h:52 | sigma of the fitted wall line | patch? |
| wall fit sigma (fitter) | 0.02 | refrigerator_concept/src/refrigerator_fitter.cpp:519 | same wall sigma set in the fitter | patch? |
| prior_pos_std | 0.30 | refrigerator_concept/src/refrigerator_belief.h:102 | broad pose prior std (also the ceiling when Sigma is clamped to prior) | patch? |
| prior_yaw_std | 0.60 | refrigerator_concept/src/refrigerator_belief.h:104 | broad yaw prior std | patch? |
| kObliquityYawGain | 0.05 | refrigerator_concept/src/refrigerator_fitter.cpp:873 | grazing-view yaw cap: sigma_yaw = gain*(1/cos-1) | patch |
| obliquity cos clamp | [0.05,1] | refrigerator_concept/src/refrigerator_fitter.cpp:874 | clamp so 1/cos stays finite | patch |
| range floor | 0.1 | refrigerator_concept/src/refrigerator_fitter.cpp:261 | floor of range used in the frame weight | patch |
| birth top quantile | 0.95 | refrigerator_concept/src/refrigerator_fitter.cpp:590 | quantile taken as the robust top at birth | patch? |
| birth top quantile (2nd) | 0.97 | refrigerator_concept/src/refrigerator_fitter.cpp:825 | quantile taken as the robust top in the top-anchor | patch? |
| birth z band | 3 sigma | refrigerator_concept/src/refrigerator_fitter.cpp:617 | half-width (in sigma_base) of the top band used to seed w,h | patch? |
| birth anisotropy test | 0.10 | refrigerator_concept/src/refrigerator_fitter.cpp:639 | footprint anisotropy above which the moment yaw seeds the box | patch? |
| birth size floor | 0.10 | refrigerator_concept/src/refrigerator_fitter.cpp:622 | floor of seeded w | patch? |
| kYawJumpLogRad | 0.087 | refrigerator_concept/src/refrigerator_fitter.cpp:1011 | diagnostic print trigger only (no fit effect) | excluded |
| roi sanity gate | \\|off\\|<3, fill<4 | refrigerator_concept/src/refrigerator_projection.cpp:226 | reject implausible projected ROI (gate); clamps +-3 and [0,4] follow | patch |
| near-plane guard | Y<=0.20 | refrigerator_concept/src/refrigerator_projection.cpp:201 | skip points at/behind the image plane (projection numerics) | excluded? |
| spatial hash CELL | 6.0 px | refrigerator_concept/src/refrigerator_projection.cpp:271 | cell size of a pixel spatial hash (method constant) | excluded |
| support select below_m | 0.05 | refrigerator_concept/src/refrigerator_support_bank.h:40 | margin below the floor when selecting owned points | patch |
| SDF_SMOOTH_K | 0.02 | refrigerator_concept/src/refrigerator_model.h:61 | smoothing width of the compound SDF | patch? |
| SDF_INSIDE_SCALE | 0.3 | refrigerator_concept/src/refrigerator_model.h:62 | scale of the inside-SDF term | patch? |
| TOP_THICKNESS / LEG_RADIUS | 0.03 / 0.025 | refrigerator_concept/src/refrigerator_model.h:59 | template top thickness and leg radius (shape geometry, copied from table) | class? |
| kPosEps / kYawEps | 0.003 / 0.005 | refrigerator_concept/src/refrigerator_scene_graph.cpp:144 | publish-only change dedup (no estimation role) | excluded |
| kRobotRadiusM | 0.30 | refrigerator_concept/src/epistemic_planner.cpp:78 | viewpoint planner robot radius (planner) | excluded? |
| planner kRef | 0.10 | refrigerator_concept/src/epistemic_planner.cpp:180 | planner common scale 10 cm / 0.1 rad | excluded? |
| nominal size w,h,H (model.h) | 0.60,0.60,1.70 | refrigerator_concept/src/refrigerator_model.h:26 | default template size of a fridge | class |
| graph-attr fallback seeds | 0.60,0.60,1.70 | refrigerator_concept/src/refrigerator_fitter.cpp:372 | size used when the graph node lacks a size attribute | class |

### Switches

| name | value | where | what it switches | kind |
|---|---|---|---|---|
| RefrigeratorModel.FootprintResidual | false | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:214 | footprint residual + depth nuisance (grazing-yaw fix) | patch mechanism |
| RefrigeratorModel.QuotientChart | false | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:216 | symmetry-quotient chart (yaw snaps) | patch mechanism |
| RefrigeratorModel.AI2VolatilityInfer | true | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:142 | infer per-DOF process noise (HISTORY stage; replaces constant Q) | other |
| RefrigeratorModel.AI2MotionConfirmOnly | true | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:180 | hard confirm-only gate when moving & off-centre | patch mechanism |
| RefrigeratorModel.ExistenceRemovalEnabled | true | etc/config.toml [RefrigeratorModel] ; read refrigerator_config.cpp:367 | enable log-odds removal | patch mechanism |
| RefrigeratorConcept.FridgeFilterEnabled | true | etc/config.toml [RefrigeratorConcept] ; read refrigerator_config.cpp:226 | "is this really X?" plausibility filter | patch mechanism |
| RefrigeratorConcept.FrontDetectEnabled | true | etc/config.toml [RefrigeratorConcept] ; read refrigerator_config.cpp:349 | appearance-based front/door-mode detector | other |
| RefrigeratorConcept.ContourCheckEnabled | true | etc/config.toml [RefrigeratorConcept] ; read refrigerator_config.cpp:351 | classifier-free contour existence channel | other |
| RefrigeratorConcept.PublishObjectObs | true | etc/config.toml [RefrigeratorConcept] ; read refrigerator_config.cpp:256 | publish per-frame observation to room anchor factor | other |
| Tracker.BirthFusion | true | etc/config.toml [Tracker] ; read refrigerator_config.cpp:274 | fused birth evidence from residual grid | patch mechanism |
| Tracker.BirthFragment | true | NO toml key; code default at refrigerator_config.cpp:293 | switch of a patch mechanism | patch mechanism |
| RefrigeratorConcept.RicohConfirmEnabled | false | NO toml key; code default at refrigerator_config.cpp:398 | switch of a patch mechanism | patch mechanism |

### Excluded, by file

- `etc/config.toml`: 42: `Agent.id`, `Agent.name`, `Agent.configFile`, `Agent.domain`, `Agent.tree`, `Agent.graph`, `Agent.2d`, `Agent.3d`, `Presence.required_agent_names`, `Presence.config_dir`, `Presence.monitor_period_ms`, `Presence.heartbeat_timeout_ms`, `Presence.rejoin_grace_ms`, `Presence.stale_grace_ms`, `Presence.agent_info_period_ms`, `Owns.nodes`, `Period.Compute`, `Period.Emergency`, `Component.Debug.Verbose`, `Media.MasksStallTimeoutMs`, `RefrigeratorModel.AI2CsvPath`, `RefrigeratorModel.BirthSurpriseProbe`, `RefrigeratorModel.AI2GnIters`, `RefrigeratorConcept.StateEps`, `RefrigeratorConcept.KStable`, `RefrigeratorConcept.DetectionAliveMaxFrames?`, `RefrigeratorConcept.FridgeFilterLog`, `RefrigeratorConcept.FrontLog`, `RefrigeratorConcept.RefrigeratorLogPeriodFrames`, `RefrigeratorConcept.ShowDashboard`, `RefrigeratorConcept.EpistemicCooldownCycles?`, `RefrigeratorConcept.SupportBankMaxPoints`, `RefrigeratorConcept.SupportBankQuantizationM`, `RefrigeratorConcept.DumpCloudPath`, `RefrigeratorConcept.ObjectObsFrame`, `Ice.Warn.Connections`, `Ice.Trace.Network`, `Ice.Trace.Protocol`, `Ice.MessageSizeMax`, `RefrigeratorConcept.ShapeEvalPeriod`, `RefrigeratorConcept.DetectProbeCsvPath`, `Tracker.BirthFragmentMaxPts`
- `refrigerator_concept/src/refrigerator_belief.h`: 1: `volatility_omega_min`
- `refrigerator_concept/src/refrigerator_fitter.cpp`: 1: `kYawJumpLogRad`
- `refrigerator_concept/src/refrigerator_projection.cpp`: 2: `near-plane guard?`, `spatial hash CELL`
- `refrigerator_concept/src/refrigerator_scene_graph.cpp`: 1: `kPosEps / kYawEps`
- `refrigerator_concept/src/epistemic_planner.cpp`: 2: `kRobotRadiusM?`, `planner kRef?`

## hood

### Config keys and hidden defaults

| name | value | where | role | label |
|---|---|---|---|---|
| Agent.id | 17 | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Agent.name | "hood_concept" | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Agent.configFile | "" | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Agent.domain | 0 | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Agent.tree | false | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Agent.graph | false | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Agent.2d | false | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Agent.3d | false | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Presence.required_agent_names | ["robot_concept", "room_concept"] | etc/config.toml [Presence] | presence protocol plumbing | excluded |
| Presence.config_dir | "etc" | etc/config.toml [Presence] | presence protocol plumbing | excluded |
| Presence.monitor_period_ms | 500 | etc/config.toml [Presence] | presence protocol plumbing | excluded |
| Presence.heartbeat_timeout_ms | 3000 | etc/config.toml [Presence] | presence protocol plumbing | excluded |
| Presence.rejoin_grace_ms | 1500 | etc/config.toml [Presence] | presence protocol plumbing | excluded |
| Presence.stale_grace_ms | 120000 | etc/config.toml [Presence] | presence protocol plumbing | excluded |
| Presence.agent_info_period_ms | 1000 | etc/config.toml [Presence] | presence protocol plumbing | excluded |
| Owns.nodes | ["hood_concept 21", "hood_*"] | etc/config.toml [Owns] | owned graph nodes (plumbing) | excluded |
| Period.Compute | 100 | etc/config.toml [Period] | loop period | excluded |
| Period.Emergency | 500 | etc/config.toml [Period] | loop period | excluded |
| Component.Debug.Verbose | false | etc/config.toml [Component.Debug] | log switch | excluded |
| Media.MasksStallTimeoutMs | 3000 | etc/config.toml [Media] ; read hood_config.cpp:85 | watchdog timeout on the mask stream (stops robot on stale data) | excluded |
| HoodModel.SigmaObs | 0.05 | etc/config.toml [HoodModel] ; read hood_config.cpp:99 | band splitting top/leg (candidate vs residual) membership | patch |
| HoodModel.SdfThresholdForStorage | 0.08 | etc/config.toml [HoodModel] ; read hood_config.cpp:101 | on-surface membership threshold for candidate/residual split | patch |
| HoodModel.AI2CsvPath | "etc/ai2_log.csv" | etc/config.toml [HoodModel] ; read hood_config.cpp:303 | per-cycle CSV path | excluded |
| HoodModel.BirthSurpriseProbe | true | etc/config.toml [HoodModel] ; read hood_config.cpp:307 | read-only diagnostic probe of the residual grid | excluded |
| HoodModel.AI2SigmaBaseM | 0.03 | etc/config.toml [HoodModel] ; read hood_config.cpp:105 | base on-surface observation noise (m); larger than depth noise, includes mask-edge error | sensor? |
| HoodModel.PixelSigmaOverF | 0.0015 | etc/config.toml [HoodModel] ; read hood_config.cpp:309 | pixel noise over focal (intrinsics) | sensor |
| HoodModel.DepthSigma0M | 0.006 | etc/config.toml [HoodModel] ; read hood_config.cpp:311 | ZED depth std floor | sensor |
| HoodModel.DepthSigmaRangeCoef | 0.004 | etc/config.toml [HoodModel] ; read hood_config.cpp:313 | ZED depth std growth with range^2 | sensor |
| HoodModel.ModelSigmaM | 0.010 | etc/config.toml [HoodModel] ; read hood_config.cpp:315 | residual model/surface sigma floor | patch |
| HoodModel.DepthTiltStd | 0.020 | etc/config.toml [HoodModel] ; read hood_config.cpp:321 | ZED shared depth-tilt prior std; consumed only by FootprintResidual (off) | sensor? |
| HoodModel.DepthBiasStd | 0.015 | etc/config.toml [HoodModel] ; read hood_config.cpp:323 | ZED shared depth-bias prior std; consumed only by FootprintResidual (off) | sensor? |
| HoodModel.DepthScaleStd | 0.010 | etc/config.toml [HoodModel] ; read hood_config.cpp:325 | ZED shared depth-scale prior std; consumed only by FootprintResidual (off) | sensor? |
| HoodModel.AI2ClutterFrac | 0.10 | etc/config.toml [HoodModel] ; read hood_config.cpp:107 | prior weight of the uniform clutter mixture (mask error) | patch |
| HoodModel.AI2ClutterScaleM | 0.12 | etc/config.toml [HoodModel] ; read hood_config.cpp:109 | distance beyond which a point is explained by clutter | patch |
| HoodModel.AI2PriorSizeStd | 0.30 | etc/config.toml [HoodModel] ; read hood_config.cpp:111 | broad size prior std | class |
| HoodModel.AI2PriorFootprintM | 0.90 | etc/config.toml [HoodModel] | class footprint prior mean | class |
| HoodModel.AI2PriorDepthM | 0.50 | etc/config.toml [HoodModel] | class depth prior mean | class |
| HoodModel.AI2PriorFootprintStd | 0.08 | etc/config.toml [HoodModel] | class footprint prior std | class |
| HoodModel.AI2PriorHeightM | 2.05 | etc/config.toml [HoodModel] | class height prior mean | class |
| HoodModel.AI2PriorHeightStd | 0.30 | etc/config.toml [HoodModel] ; read hood_config.cpp:214 | class height prior std | class |
| HoodModel.AI2DepthUnobsPrecision | 20000.0 | etc/config.toml [HoodModel] ; read hood_config.cpp:216 | extra depth precision when depth is unobserved (single face) | patch |
| HoodModel.AI2DepthObsBandM | 0.10 | etc/config.toml [HoodModel] ; read hood_config.cpp:218 | spread below which cloud is a single depth face | patch |
| HoodModel.AI2TopNoFloatPrecision | 10000.0 | etc/config.toml [HoodModel] ; read hood_config.cpp:220 | two-sided anchor pinning H to observed top | patch |
| HoodModel.AI2TopNoFloatMarginM | 0.02 | etc/config.toml [HoodModel] ; read hood_config.cpp:222 | upward allowance of the top anchor | patch |
| HoodModel.AI2TopOversegSigmaPerM | 2.0 | etc/config.toml [HoodModel] ; read hood_config.cpp:224 | R inflation above the robust top (over-segmentation tail) | patch |
| HoodModel.AspectScale | 0.15 | etc/config.toml [HoodModel] ; read hood_config.cpp:331 | class aspect scale of the plausibility filter | class? |
| HoodModel.SizeScale | 0.15 | etc/config.toml [HoodModel] ; read hood_config.cpp:333 | class footprint scale of the plausibility filter | class? |
| HoodModel.AltSizeScale | 0.60 | etc/config.toml [HoodModel] ; read hood_config.cpp:335 | "other furniture" alternative size std of class evidence | class? |
| HoodModel.HeightPlausibleMin | 1.20 | etc/config.toml [HoodModel] ; read hood_config.cpp:337 | class height below which the object is implausible | class? |
| HoodModel.HeightSoft | 0.15 | etc/config.toml [HoodModel] ; read hood_config.cpp:339 | softness of the class height logistic | class? |
| HoodModel.FeRef | 2.0 | etc/config.toml [HoodModel] ; read hood_config.cpp:341 | reference FE of a healthy fit (plausibility) | patch |
| HoodModel.FeScale | 1.0 | etc/config.toml [HoodModel] ; read hood_config.cpp:343 | scale of FE excess in plausibility | patch |
| HoodModel.PlausClamp | 8.0 | etc/config.toml [HoodModel] ; read hood_config.cpp:345 | clamp of plausibility evidence | patch |
| HoodModel.PlausHeightPriorGain | 2000.0 | etc/config.toml [HoodModel] ; read hood_config.cpp:347 | precision of short-height prior | patch |
| HoodModel.PlausToExistenceGain | 0.0 | etc/config.toml [HoodModel] ; read hood_config.cpp:349 | gain from plausibility to existence class term | patch |
| HoodModel.SingletonInhibition | 1.0 | etc/config.toml [HoodModel] ; read hood_config.cpp:351 | mutual inhibition of neighbours in class existence | patch |
| HoodModel.AI2WallPrecision | 400.0 | etc/config.toml [HoodModel] ; read hood_config.cpp:226 | precision of wall-flush factor at zero gap (class fact is WallFlushPrior) | patch? |
| HoodModel.AI2WallParallelPrecision | 200.0 | etc/config.toml [HoodModel] ; read hood_config.cpp:228 | precision of back-face-parallel-to-wall factor | patch? |
| HoodModel.AI2WallReachM | 0.15 | etc/config.toml [HoodModel] ; read hood_config.cpp:230 | gap scale over which flush weight decays | patch? |
| HoodModel.AI2DoorClearanceGain | 3.0 | etc/config.toml [HoodModel] ; read hood_config.cpp:232 | nats of preference for door facing into the room; class fact (door side) scaled by a gain | patch? |
| HoodModel.AI2VolatilityLr | 0.02 | etc/config.toml [HoodModel] ; read hood_config.cpp:245 | learning rate of inferred volatility | patch |
| HoodModel.AI2VolatilitySigma | 2.0 | etc/config.toml [HoodModel] ; read hood_config.cpp:247 | hyperprior width of inferred volatility | patch |
| HoodModel.AI2WallExplainFrac | 0.25 | etc/config.toml [HoodModel] ; read hood_config.cpp:249 | weight of "point belongs to a room wall" mixture component | patch |
| HoodModel.AI2WallExplainSigmaM | 0.05 | etc/config.toml [HoodModel] ; read hood_config.cpp:251 | wall plane slack | patch |
| HoodModel.WallNoCrossPrecision | 2000.0 | etc/config.toml [HoodModel] ; read hood_config.cpp:255 | one-sided precision against wall penetration | patch |
| HoodModel.WallNoCrossMarginM | 0.0 | etc/config.toml [HoodModel] ; read hood_config.cpp:257 | margin of wall no-cross constraint | patch |
| HoodModel.AI2ProcessStdM | 0.005 | etc/config.toml [HoodModel] ; read hood_config.cpp:259 | process noise on pose = how much furniture moves per frame (permanence); LIFECYCLE says no constant Q | class? |
| HoodModel.AI2ProcessStdYaw | 0.01 | etc/config.toml [HoodModel] ; read hood_config.cpp:261 | process noise on yaw (permanence); LIFECYCLE says no constant Q | class? |
| HoodModel.AI2AgeNominalDtS | 0.1 | etc/config.toml [HoodModel] ; read hood_config.cpp:263 | nominal mask-stream period (10 Hz); 0 = aging off | sensor? |
| HoodModel.AI2TruncGateFrac | 0.10 | etc/config.toml [HoodModel] ; read hood_config.cpp:299 | skip update if mask border fraction above this (truncation gate) | patch |
| HoodModel.AI2RangeNoiseLatPerM | 0.02 | etc/config.toml [HoodModel] ; read hood_config.cpp:293 | lateral deprojection std growth per metre of range (intrinsics-like) feeding R and position common-mode | sensor? |
| HoodModel.AI2RangeNoiseYawPerM | 0.03 | etc/config.toml [HoodModel] ; read hood_config.cpp:295 | yaw common-mode growth with range (anti far-view rotation) | patch |
| HoodModel.AI2RangeNoiseSizePerM | 0.08 | etc/config.toml [HoodModel] ; read hood_config.cpp:297 | size common-mode growth with range (freeze geometry afar) | patch |
| HoodModel.AI2GnIters | 4 | etc/config.toml [HoodModel] ; read hood_config.cpp:301 | GN iterations per frame (iteration cap) | excluded |
| HoodModel.AI2CommonModePosStd | 0.03 | etc/config.toml [HoodModel] ; read hood_config.cpp:265 | shared per-frame position error floor | patch |
| HoodModel.AI2CommonModeSizeStd | 0.02 | etc/config.toml [HoodModel] ; read hood_config.cpp:267 | shared per-frame size error floor | patch |
| HoodModel.AI2CommonModeYawStd | 0.03 | etc/config.toml [HoodModel] ; read hood_config.cpp:269 | shared per-frame yaw error floor | patch |
| HoodModel.MotionCmPosGain | 0.10 | etc/config.toml [HoodModel] ; read hood_config.cpp:271 | ego-motion shared position error per m/s (anti-drift) | patch |
| HoodModel.MotionCmSizeGain | 0.20 | etc/config.toml [HoodModel] ; read hood_config.cpp:273 | ego-motion shared size error per m/s (anti-reshape) | patch |
| HoodModel.MotionCmYawGain | 0.12 | etc/config.toml [HoodModel] ; read hood_config.cpp:275 | ego-motion shared yaw error per m/s (anti-rotate) | patch |
| HoodModel.AI2AngLeverM | 2.0 | etc/config.toml [HoodModel] ; read hood_config.cpp:277 | rad/s to m/s lever for ego angular motion | patch |
| HoodModel.AI2PeriphRef | 0.50 | etc/config.toml [HoodModel] ; read hood_config.cpp:279 | centroid radius where periphery penalty saturates | patch |
| HoodModel.AI2MotionRefMps | 0.60 | etc/config.toml [HoodModel] ; read hood_config.cpp:281 | motion at which a peripheral frame is unreliable | patch |
| HoodModel.AI2StillLinMps | 0.05 | etc/config.toml [HoodModel] ; read hood_config.cpp:285 | still threshold: camera linear speed | patch |
| HoodModel.AI2StillAngRadps | 0.10 | etc/config.toml [HoodModel] ; read hood_config.cpp:287 | still threshold: camera angular speed | patch |
| HoodModel.AI2StillDotd | 0.05 | etc/config.toml [HoodModel] ; read hood_config.cpp:289 | still threshold: mask ego-motion smear | patch |
| HoodModel.AI2MovingUpdateCenterRadius | 0.35 | etc/config.toml [HoodModel] ; read hood_config.cpp:291 | centroid radius below which a moving update is allowed | patch |
| HoodModel.LidarPrecision | 2500.0 | etc/config.toml [HoodModel] ; read hood_config.cpp:416 | per-ray LiDAR range precision (1/sigma^2 at 2 cm); set as a weight to fix mask under-size | sensor? |
| HoodModel.LidarBpearlPrecision | 0.0 | etc/config.toml [HoodModel] ; read hood_config.cpp:418 | per-ray precision of the low LiDAR (0 = off) | sensor? |
| HoodModel.LidarRobustCM | 0.05 | etc/config.toml [HoodModel] ; read hood_config.cpp:420 | Cauchy scale of LiDAR residual | patch |
| HoodModel.LidarSelectMarginM | 0.10 | etc/config.toml [HoodModel] ; read hood_config.cpp:422 | margin of LiDAR return pre-selection box | patch |
| HoodModel.LidarCoverageN0 | 60.0 | etc/config.toml [HoodModel] ; read hood_config.cpp:424 | ray count for full LiDAR weight | patch |
| HoodModel.LidarCoverageAngPower | 1.0 | etc/config.toml [HoodModel] ; read hood_config.cpp:426 | angular-coverage down-weight exponent | patch |
| HoodModel.MaxStepM | 1.0 | etc/config.toml [HoodModel] ; read hood_config.cpp:428 | reject a GN centre step larger than this (divergence guard) | patch |
| HoodModel.CoveragePrecision | 0.0 | etc/config.toml [HoodModel] ; read hood_config.cpp:430 | grow-only pull of clutter-ceded points (0 = off) | patch |
| HoodModel.CoverageRobustCM | 0.15 | etc/config.toml [HoodModel] ; read hood_config.cpp:432 | Cauchy scale of coverage pull | patch |
| HoodModel.FreeSpacePrecision | 500.0 | etc/config.toml [HoodModel] ; read hood_config.cpp:434 | shrink-only free-space pull per through-beam | patch |
| HoodModel.FootprintMomentPrecision | 2000.0 | etc/config.toml [HoodModel] ; read hood_config.cpp:436 | footprint 2nd-moment factor precision (w,h,yaw) | patch |
| HoodModel.FootprintMomentRangePerM | 0.03 | etc/config.toml [HoodModel] ; read hood_config.cpp:438 | moment variance growth with range | patch |
| HoodModel.FeBaselineAdaptDown | 0.05 | etc/config.toml [HoodModel] ; read hood_config.cpp:440 | free-energy baseline EMA, fast down | patch |
| HoodModel.FeBaselineAdaptUp | 0.005 | etc/config.toml [HoodModel] ; read hood_config.cpp:442 | free-energy baseline EMA, slow up | patch |
| HoodModel.FeSurpriseSmooth | 0.10 | etc/config.toml [HoodModel] ; read hood_config.cpp:444 | smoothing of positive FE gap (surprise) | patch |
| HoodModel.FootprintMomentMotionGain | 0.30 | etc/config.toml [HoodModel] ; read hood_config.cpp:446 | moment variance growth with motion | patch |
| HoodModel.OrientationMotionRef | 0.50 | etc/config.toml [HoodModel] ; read hood_config.cpp:448 | motion at which yaw-mode vote is halved | patch |
| HoodModel.FootprintMomentCompletenessGain | 0.5 | etc/config.toml [HoodModel] ; read hood_config.cpp:464 | moment variance growth with partial view | patch |
| HoodModel.FootprintMomentMinCompleteness | 0.02 | etc/config.toml [HoodModel] ; read hood_config.cpp:466 | clamp floor of 1/completeness | patch |
| HoodModel.ObliquityMomentGain | 0.05 | etc/config.toml [HoodModel] ; read hood_config.cpp:462 | grazing-view cap on moment channel | patch |
| HoodModel.ExistenceRemovalProb | 0.12 | etc/config.toml [HoodModel] ; read hood_config.cpp:472 | removal probability threshold | patch |
| HoodModel.ExistenceOutOfRoomGain | 1.5 | etc/config.toml [HoodModel] ; read hood_config.cpp:474 | log-odds step per cycle outside the room | patch |
| HoodModel.ExistenceLogoddsMax | 4.0 | etc/config.toml [HoodModel] ; read hood_config.cpp:478 | clamp of \|L\| | patch |
| HoodModel.ExistenceDetectionProb | 0.55 | etc/config.toml [HoodModel] | P(beam returns from occupied footprint) (sensor hit rate) | sensor |
| HoodModel.ExistenceClutterProb | 0.05 | etc/config.toml [HoodModel] ; read hood_config.cpp:484 | P(beam returns from an empty footprint): spurious-return rate of the LiDAR | sensor? |
| HoodModel.ExistenceFrameCorrelation | 0.477 | etc/config.toml [HoodModel] ; read hood_config.cpp:476 | measured frame-to-frame miss correlation of the detector (rho) | sensor? |
| HoodModel.ExistenceSensorSigmaM | 0.03 | etc/config.toml [HoodModel] ; read hood_config.cpp:486 | LiDAR range sigma | sensor |
| HoodModel.ExistenceRemoveFrames | 15 | etc/config.toml [HoodModel] ; read hood_config.cpp:488 | removal debounce frames | patch |
| HoodModel.ExistenceAbsenceRangeRefM | 2.5 | etc/config.toml [HoodModel] ; read hood_config.cpp:490 | range below which absence is trusted fully | patch |
| HoodModel.ExistenceAbsenceRangePower | 2.0 | etc/config.toml [HoodModel] ; read hood_config.cpp:492 | decay exponent of absence with range | patch |
| HoodModel.ExistenceVerifySurprise | 20.0 | etc/config.toml [HoodModel] ; read hood_config.cpp:494 | surprise above which object wants verification | patch |
| HoodModel.ExistenceVerifyGain | 5.0 | etc/config.toml [HoodModel] ; read hood_config.cpp:498 | epistemic gain of a verification request | patch |
| HoodConcept.StateEps | 0.04 | etc/config.toml [HoodConcept] ; read hood_config.cpp:61 | convergence status tolerance (m+rad); no effect on the estimate | excluded |
| HoodConcept.KStable | 30 | etc/config.toml [HoodConcept] ; read hood_config.cpp:63 | frames before model_stable status flag | excluded |
| HoodConcept.DetectionAliveMaxFrames | 40 | etc/config.toml [HoodConcept] ; read hood_config.cpp:65 | frames without a mask before detection_alive=false (status flag; check whether existence reads it) | excluded? |
| HoodConcept.MatchedFramesBeforeAging | 5 | etc/config.toml [HoodConcept] ; read hood_config.cpp:67 | matched frames before a stale cycle may age the belief | patch |
| HoodConcept.CentralRegionFrac | 0.25 | etc/config.toml [HoodConcept] ; read hood_config.cpp:69 | central image box where the detector is reliable; part of p_detect | sensor? |
| HoodConcept.FridgeFilterLog | false | etc/config.toml [HoodConcept] ; read hood_config.cpp:353 | diagnostic log switch | excluded |
| HoodConcept.FrontMinFaceAreaPx | 900.0 | etc/config.toml [HoodConcept] ; read hood_config.cpp:456 | min projected face area to score door-ness | patch |
| HoodConcept.FrontMinConfidence | 0.10 | etc/config.toml [HoodConcept] ; read hood_config.cpp:458 | min door-ness margin to emit a cue | patch |
| HoodConcept.FrontLog | true | etc/config.toml [HoodConcept] ; read hood_config.cpp:460 | diagnostic log switch | excluded |
| HoodConcept.HoodLogPeriodFrames | 30 | etc/config.toml [HoodConcept] ; read hood_config.cpp:73 | log throttle | excluded |
| HoodConcept.ShowDashboard | true | etc/config.toml [HoodConcept] ; read hood_config.cpp:87 | GUI switch | excluded |
| HoodConcept.EpistemicCooldownCycles | 200 | etc/config.toml [HoodConcept] ; read hood_config.cpp:71 | planner anti-chatter cooldown (viewpoint planner) | excluded? |
| HoodConcept.SupportBankMaxPoints | 4000 | etc/config.toml [HoodConcept] ; read hood_config.cpp:75 | memory cap of the support bank | excluded |
| HoodConcept.SupportBankQuantizationM | 0.02 | etc/config.toml [HoodConcept] ; read hood_config.cpp:77 | dedup grid of the support bank (storage) | excluded |
| HoodConcept.SupportSelectRadiusMarginM | 0.50 | etc/config.toml [HoodConcept] ; read hood_config.cpp:79 | XY ownership margin of support points | patch |
| HoodConcept.SupportSelectHeightMarginM | 0.25 | etc/config.toml [HoodConcept] ; read hood_config.cpp:81 | Z margin of support point selection | patch |
| HoodConcept.DumpCloudPath | "/tmp/round_hood.xyz" | etc/config.toml [HoodConcept] ; read hood_config.cpp:95 | diagnostic cloud dump path | excluded |
| HoodConcept.RtCovScale | 1.0 | etc/config.toml [HoodConcept] ; read hood_config.cpp:357 | scale on published RT covariance (calibrate toward NEES 1) | patch |
| HoodConcept.ObjectObsFrame | "body" | etc/config.toml [HoodConcept] ; read hood_config.cpp:361 | frame name of the published observation | excluded |
| Tracker.GateMahalanobis | 9.0 | etc/config.toml [Tracker] ; read hood_config.cpp:365 | association gate (chi2) | patch |
| Tracker.GateFallbackM | 0.50 | etc/config.toml [Tracker] ; read hood_config.cpp:367 | association metric gate before covariance exists | patch |
| Tracker.DetectionNoiseM | 0.35 | etc/config.toml [Tracker] ; read hood_config.cpp:369 | association noise added to cover centroid-vs-fit offset | patch |
| Tracker.RecallPiMove | 0.05 | etc/config.toml [Tracker] ; read hood_config.cpp:371 | permanence: P(remembered object was moved) | class |
| Tracker.RecallLogPriorNew | 0.0 | etc/config.toml [Tracker] ; read hood_config.cpp:373 | permanence: log prior of a new object | class |
| Tracker.BirthFrames | 8 | etc/config.toml [Tracker] ; read hood_config.cpp:375 | frames a mask stays unexplained before birth (debounce) | patch |
| Tracker.BirthFusionGain | 6.0 | etc/config.toml [Tracker] ; read hood_config.cpp:379 | extra birth evidence at full corroboration | patch |
| Tracker.BirthFusionMassRef | 8.0 | etc/config.toml [Tracker] ; read hood_config.cpp:381 | residual mass at half saturation | patch |
| Tracker.BirthFusionRadiusM | 0.50 | etc/config.toml [Tracker] ; read hood_config.cpp:383 | window of residual-mass sample | patch |
| Tracker.BirthMinSepM | 0.60 | etc/config.toml [Tracker] ; read hood_config.cpp:385 | minimum separation of a birth from existing objects | patch |
| Tracker.MergeOverlap | 0.05 | etc/config.toml [Tracker] ; read hood_config.cpp:387 | footprint overlap that merges two instances | patch |
| Tracker.BirthWidthM | 1.0 | etc/config.toml [Tracker] ; read hood_config.cpp:389 | default size of an instance born from a mask (class seed; fit refines it) | class? |
| Tracker.BirthDepthM | 0.6 | etc/config.toml [Tracker] ; read hood_config.cpp:391 | default size of an instance born from a mask (class seed; fit refines it) | class? |
| Tracker.BirthHeightM | 0.75 | etc/config.toml [Tracker] ; read hood_config.cpp:393 | default size of an instance born from a mask (class seed; fit refines it) | class? |
| Tracker.RicohAttentionConf | 0.60 | etc/config.toml [Tracker] ; read hood_config.cpp:407 | min confidence for Ricoh attention | patch |
| Tracker.RicohAttentionAngleMargin | 0.05 | etc/config.toml [Tracker] ; read hood_config.cpp:409 | angular margin of Ricoh attention match | patch |
| Tracker.RicohAttentionRangeBandM | 1.0 | etc/config.toml [Tracker] ; read hood_config.cpp:411 | Ricoh range roughness band used as a match margin | sensor? |
| Ice.Warn.Connections | "0" | etc/config.toml [Ice] | Ice middleware setting | excluded |
| Ice.Trace.Network | "0" | etc/config.toml [Ice] | Ice middleware setting | excluded |
| Ice.Trace.Protocol | "0" | etc/config.toml [Ice] | Ice middleware setting | excluded |
| Ice.MessageSizeMax | "20004800" | etc/config.toml [Ice] | Ice middleware setting | excluded |
| HoodConcept.ShapeEvalPeriod | 30 | NO toml key; code default at hood_config.cpp:89 | cycles between shape evaluations (cadence) | excluded |
| HoodConcept.ShapeEvalMinPoints | 300 | NO toml key; code default at hood_config.cpp:91 | points needed before the shape test runs | patch |
| HoodConcept.ShapeEvidenceClamp | 8.0 | NO toml key; code default at hood_config.cpp:93 | clamp of shape evidence | patch |
| HoodModel.VerticalExtentM | 0.50 | NO toml key; code default at hood_config.cpp:127 | class vertical extent | class |
| HoodConcept.DetectProbeCsvPath | "etc/detect_probe.csv" | NO toml key; code default at hood_config.cpp:305 | probe CSV path | excluded |
| Tracker.BirthFragmentVoxelM | 0.03 | NO toml key; code default at hood_config.cpp:398 | dedup grid of birth burst | patch |
| Tracker.BirthFragmentMaxPts | 20000 | NO toml key; code default at hood_config.cpp:400 | hard point cap of the birth burst (memory) | excluded |
| Tracker.BirthFragmentDeltaMs | 4000 | NO toml key; code default at hood_config.cpp:403 | time window of the birth burst kept (ms) | patch? |
| Tracker.BirthAdmitPlausibility | 0.35 | NO toml key; code default at hood_config.cpp:405 | birth admission plausibility (0 = off) | patch |
| HoodModel.VerifySurpriseSmooth | 0.10 | NO toml key; code default at hood_config.cpp:496 | EMA weight of go-verify surprise | patch |
| HoodConcept.RicohConfirmDetectionProb | 0.60 | NO toml key; code default at hood_config.cpp:505 | Ricoh detection probability (confirm path, off) | sensor |
| HoodConcept.RicohConfirmClutterProb | 0.15 | NO toml key; code default at hood_config.cpp:507 | Ricoh clutter probability (confirm path, off) | sensor |

### Hard-coded in src

| name | value | where | role | label |
|---|---|---|---|---|
| kLambda (volatility memory) | 0.995 | hood_concept/src/hood_belief.cpp:267 | forgetting factor of inferred-volatility sufficient statistics (~200 observations) | patch? |
| omega upper clamp | 4.0 | hood_concept/src/hood_belief.cpp:277 | clamp of log process noise omega | patch |
| volatility_omega_min | -18.0 | hood_concept/src/hood_belief.h:218 | numerical floor of omega ("not a tuning knob") | excluded |
| jac_slope_clamp | 5.0 | hood_concept/src/hood_belief.h:235 | clamp per-DOF SDF slope (guards seam-point information spike) | patch |
| kFrontClamp | 6.0 | hood_concept/src/hood_belief.cpp:753 | clamp of door-mode evidence, +-6 nats (also line of the re-baseline clamp) | patch |
| front_acc re-baseline clamp | +-6.0 | hood_concept/src/hood_belief.cpp:727 | second use of the same +-6 clamp | patch |
| kRectAspect | 0.10 | hood_concept/src/hood_belief.h:516 | \|w-h\|/(w+h) above which footprint counts as rectangular | patch? |
| front_view_budget_ | 3.0 | hood_concept/src/hood_belief.h:524 | mode-evidence budget per observer-bearing bin (mirrors chair view_budget) | patch |
| kFrontViewBins | 24 | hood_concept/src/hood_belief.h:522 | 15 deg bins of the view budget | patch? |
| size floors w,h,H | 0.10 | hood_concept/src/hood_belief.cpp:185 | clamp: minimum w,h,H of the state | patch? |
| wall segment min length | 0.1 | hood_concept/src/hood_belief.cpp:465 | wall segment shorter than this gives no parallel-wall precision | patch? |
| wall fit sigma (belief.h) | 0.02 | hood_concept/src/hood_belief.h:64 | sigma of the fitted wall line | patch? |
| wall fit sigma (fitter) | 0.02 | hood_concept/src/hood_fitter.cpp:528 | same wall sigma set in the fitter | patch? |
| prior_pos_std | 0.30 | hood_concept/src/hood_belief.h:116 | broad pose prior std (also the ceiling when Sigma is clamped to prior) | patch? |
| prior_yaw_std | 0.60 | hood_concept/src/hood_belief.h:118 | broad yaw prior std | patch? |
| kObliquityYawGain | 0.05 | hood_concept/src/hood_fitter.cpp:884 | grazing-view yaw cap: sigma_yaw = gain*(1/cos-1) | patch |
| obliquity cos clamp | [0.05,1] | hood_concept/src/hood_fitter.cpp:885 | clamp so 1/cos stays finite | patch |
| range floor | 0.1 | hood_concept/src/hood_fitter.cpp:262 | floor of range used in the frame weight | patch |
| birth top quantile | 0.95 | hood_concept/src/hood_fitter.cpp:599 | quantile taken as the robust top at birth | patch? |
| birth top quantile (2nd) | 0.97 | hood_concept/src/hood_fitter.cpp:836 | quantile taken as the robust top in the top-anchor | patch? |
| birth z band | 3 sigma | hood_concept/src/hood_fitter.cpp:626 | half-width (in sigma_base) of the top band used to seed w,h | patch? |
| birth anisotropy test | 0.10 | hood_concept/src/hood_fitter.cpp:648 | footprint anisotropy above which the moment yaw seeds the box | patch? |
| birth size floor | 0.10 | hood_concept/src/hood_fitter.cpp:631 | floor of seeded w | patch? |
| kYawJumpLogRad | 0.087 | hood_concept/src/hood_fitter.cpp:1022 | diagnostic print trigger only (no fit effect) | excluded |
| roi sanity gate | \\|off\\|<3, fill<4 | hood_concept/src/hood_projection.cpp:226 | reject implausible projected ROI (gate); clamps +-3 and [0,4] follow | patch |
| near-plane guard | Y<=0.20 | hood_concept/src/hood_projection.cpp:201 | skip points at/behind the image plane (projection numerics) | excluded? |
| spatial hash CELL | 6.0 px | hood_concept/src/hood_projection.cpp:322 | cell size of a pixel spatial hash (method constant) | excluded |
| SDF_SMOOTH_K | 0.02 | hood_concept/src/hood_model.h:87 | smoothing width of the compound SDF | patch? |
| SDF_INSIDE_SCALE | 0.3 | hood_concept/src/hood_model.h:88 | scale of the inside-SDF term | patch? |
| TOP_THICKNESS / LEG_RADIUS | 0.03 / 0.025 | hood_concept/src/hood_model.h:85 | template top thickness and leg radius (shape geometry, copied from table) | class? |
| kPosEps / kYawEps | 0.003 / 0.005 | hood_concept/src/hood_scene_graph.cpp:152 | publish-only change dedup (no estimation role) | excluded |
| kRobotRadiusM | 0.30 | hood_concept/src/epistemic_planner.cpp:81 | viewpoint planner robot radius (planner) | excluded? |
| planner kRef | 0.10 | hood_concept/src/epistemic_planner.cpp:190 | planner common scale 10 cm / 0.1 rad | excluded? |
| prior_height_m default | 2.05 | hood_concept/src/hood_belief.h:134 | hood top above the floor | class |
| nominal size w,h / z_top / extent | 0.90,0.50 / 2.05 / 0.50 | hood_concept/src/hood_model.h:48 | default template of a hood | class |

### Switches

| name | value | where | what it switches | kind |
|---|---|---|---|---|
| HoodModel.FootprintResidual | false | etc/config.toml [HoodModel] ; read hood_config.cpp:317 | footprint residual + depth nuisance (grazing-yaw fix) | patch mechanism |
| HoodModel.QuotientChart | false | etc/config.toml [HoodModel] ; read hood_config.cpp:319 | symmetry-quotient chart (yaw snaps) | patch mechanism |
| HoodModel.AI2VolatilityInfer | true | etc/config.toml [HoodModel] ; read hood_config.cpp:243 | infer per-DOF process noise (HISTORY stage; replaces constant Q) | other |
| HoodModel.AI2MotionConfirmOnly | true | etc/config.toml [HoodModel] ; read hood_config.cpp:283 | hard confirm-only gate when moving & off-centre | patch mechanism |
| HoodModel.ExistenceRemovalEnabled | true | etc/config.toml [HoodModel] ; read hood_config.cpp:470 | enable log-odds removal | patch mechanism |
| HoodConcept.FridgeFilterEnabled | true | etc/config.toml [HoodConcept] ; read hood_config.cpp:329 | "is this really X?" plausibility filter | patch mechanism |
| HoodConcept.ContourCheckEnabled | true | etc/config.toml [HoodConcept] ; read hood_config.cpp:454 | classifier-free contour existence channel | other |
| HoodConcept.FrontDetectEnabled | true | etc/config.toml [HoodConcept] ; read hood_config.cpp:452 | appearance-based front/door-mode detector | other |
| HoodConcept.PublishObjectObs | true | etc/config.toml [HoodConcept] ; read hood_config.cpp:359 | publish per-frame observation to room anchor factor | other |
| Tracker.BirthFusion | true | etc/config.toml [Tracker] ; read hood_config.cpp:377 | fused birth evidence from residual grid | patch mechanism |
| Tracker.BirthFragment | true | NO toml key; code default at hood_config.cpp:396 | switch of a patch mechanism | patch mechanism |
| HoodConcept.RicohConfirmEnabled | false | NO toml key; code default at hood_config.cpp:503 | switch of a patch mechanism | patch mechanism |

### Excluded, by file

- `etc/config.toml`: 42: `Agent.id`, `Agent.name`, `Agent.configFile`, `Agent.domain`, `Agent.tree`, `Agent.graph`, `Agent.2d`, `Agent.3d`, `Presence.required_agent_names`, `Presence.config_dir`, `Presence.monitor_period_ms`, `Presence.heartbeat_timeout_ms`, `Presence.rejoin_grace_ms`, `Presence.stale_grace_ms`, `Presence.agent_info_period_ms`, `Owns.nodes`, `Period.Compute`, `Period.Emergency`, `Component.Debug.Verbose`, `Media.MasksStallTimeoutMs`, `HoodModel.AI2CsvPath`, `HoodModel.BirthSurpriseProbe`, `HoodModel.AI2GnIters`, `HoodConcept.StateEps`, `HoodConcept.KStable`, `HoodConcept.DetectionAliveMaxFrames?`, `HoodConcept.FridgeFilterLog`, `HoodConcept.FrontLog`, `HoodConcept.HoodLogPeriodFrames`, `HoodConcept.ShowDashboard`, `HoodConcept.EpistemicCooldownCycles?`, `HoodConcept.SupportBankMaxPoints`, `HoodConcept.SupportBankQuantizationM`, `HoodConcept.DumpCloudPath`, `HoodConcept.ObjectObsFrame`, `Ice.Warn.Connections`, `Ice.Trace.Network`, `Ice.Trace.Protocol`, `Ice.MessageSizeMax`, `HoodConcept.ShapeEvalPeriod`, `HoodConcept.DetectProbeCsvPath`, `Tracker.BirthFragmentMaxPts`
- `hood_concept/src/hood_belief.h`: 1: `volatility_omega_min`
- `hood_concept/src/hood_fitter.cpp`: 1: `kYawJumpLogRad`
- `hood_concept/src/hood_projection.cpp`: 2: `near-plane guard?`, `spatial hash CELL`
- `hood_concept/src/hood_scene_graph.cpp`: 1: `kPosEps / kYawEps`
- `hood_concept/src/epistemic_planner.cpp`: 2: `kRobotRadiusM?`, `planner kRef?`

## nightstand (config)

### Config keys and hidden defaults

| name | value | where | role | label |
|---|---|---|---|---|
| Agent.id | 29 | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Agent.name | "nightstand_concept" | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Agent.configFile | "" | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Agent.domain | 0 | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Agent.tree | false | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Agent.graph | false | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Agent.2d | false | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Agent.3d | false | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Presence.required_agent_names | ["robot_concept", "room_concept"] | etc/config.toml [Presence] | presence protocol plumbing | excluded |
| Presence.config_dir | "etc" | etc/config.toml [Presence] | presence protocol plumbing | excluded |
| Presence.monitor_period_ms | 500 | etc/config.toml [Presence] | presence protocol plumbing | excluded |
| Presence.heartbeat_timeout_ms | 3000 | etc/config.toml [Presence] | presence protocol plumbing | excluded |
| Presence.rejoin_grace_ms | 1500 | etc/config.toml [Presence] | presence protocol plumbing | excluded |
| Presence.stale_grace_ms | 120000 | etc/config.toml [Presence] | presence protocol plumbing | excluded |
| Presence.agent_info_period_ms | 1000 | etc/config.toml [Presence] | presence protocol plumbing | excluded |
| Owns.nodes | ["nightstand_concept 29", "nightstand_*"] | etc/config.toml [Owns] | owned graph nodes (plumbing) | excluded |
| Period.Compute | 100 | etc/config.toml [Period] | loop period | excluded |
| Period.Emergency | 500 | etc/config.toml [Period] | loop period | excluded |
| Component.Debug.Verbose | false | etc/config.toml [Component.Debug] | log switch | excluded |
| Media.MasksStallTimeoutMs | 3000 | etc/config.toml [Media] ; read countable_config.cpp:106 | watchdog timeout on the mask stream (stops robot on stale data) | excluded |
| Model.SigmaObs | 0.05 | etc/config.toml [Model] ; read countable_config.cpp:120 | band splitting top/leg (candidate vs residual) membership | patch |
| Model.SdfThresholdForStorage | 0.08 | etc/config.toml [Model] ; read countable_config.cpp:122 | on-surface membership threshold for candidate/residual split | patch |
| Model.AI2CsvPath | "etc/ai2_log.csv" | etc/config.toml [Model] ; read countable_config.cpp:278 | per-cycle CSV path | excluded |
| Model.BirthSurpriseProbe | true | etc/config.toml [Model] ; read countable_config.cpp:282 | read-only diagnostic probe of the residual grid | excluded |
| Model.AI2SigmaBaseM | 0.03 | etc/config.toml [Model] ; read countable_config.cpp:126 | base on-surface observation noise (m); larger than depth noise, includes mask-edge error | sensor? |
| Model.PixelSigmaOverF | 0.0015 | etc/config.toml [Model] ; read countable_config.cpp:284 | pixel noise over focal (intrinsics) | sensor |
| Model.DepthSigma0M | 0.006 | etc/config.toml [Model] ; read countable_config.cpp:286 | ZED depth std floor | sensor |
| Model.DepthSigmaRangeCoef | 0.004 | etc/config.toml [Model] ; read countable_config.cpp:288 | ZED depth std growth with range^2 | sensor |
| Model.ModelSigmaM | 0.010 | etc/config.toml [Model] ; read countable_config.cpp:290 | residual model/surface sigma floor | patch |
| Model.LabelDumpEnabled | false | etc/config.toml [Model] ; read countable_config.cpp:449 | measurement-only dump switch | excluded |
| Model.LabelDumpMarginM | 0.6 | etc/config.toml [Model] ; read countable_config.cpp:451 | measurement-only dump region | excluded |
| Model.MaskPixelErrorRate | 0.094 | etc/config.toml [Model] | camera mask pixel-error rate (measured detector property) | sensor |
| Model.FloorSigmaM | 0.005 | etc/config.toml [Model] | sigma of the floor as a known surface in the beam foreign density | patch? |
| Model.DepthTiltStd | 0.020 | etc/config.toml [Model] ; read countable_config.cpp:296 | ZED shared depth-tilt prior std; consumed only by FootprintResidual (off) | sensor? |
| Model.DepthBiasStd | 0.015 | etc/config.toml [Model] ; read countable_config.cpp:298 | ZED shared depth-bias prior std; consumed only by FootprintResidual (off) | sensor? |
| Model.DepthScaleStd | 0.010 | etc/config.toml [Model] ; read countable_config.cpp:300 | ZED shared depth-scale prior std; consumed only by FootprintResidual (off) | sensor? |
| Model.AI2ClutterFrac | 0.10 | etc/config.toml [Model] ; read countable_config.cpp:128 | prior weight of the uniform clutter mixture (mask error) | patch |
| Model.AI2ClutterScaleM | 0.12 | etc/config.toml [Model] ; read countable_config.cpp:130 | distance beyond which a point is explained by clutter | patch |
| Model.AI2PriorSizeStd | 0.30 | etc/config.toml [Model] ; read countable_config.cpp:132 | broad size prior std | class |
| Model.AI2PriorFootprintM | 0.51 | etc/config.toml [Model] ; read countable_config.cpp:134 | class footprint prior mean | class |
| Model.AI2PriorFootprintStd | 0.06 | etc/config.toml [Model] ; read countable_config.cpp:136 | class footprint prior std | class |
| Model.AI2PriorHeightM | 0.62 | etc/config.toml [Model] ; read countable_config.cpp:192 | class height prior mean | class |
| Model.AI2PriorHeightStd | 0.1 | etc/config.toml [Model] ; read countable_config.cpp:194 | class height prior std | class |
| Model.AI2DepthUnobsPrecision | 20000.0 | etc/config.toml [Model] ; read countable_config.cpp:196 | extra depth precision when depth is unobserved (single face) | patch |
| Model.AI2DepthObsBandM | 0.10 | etc/config.toml [Model] ; read countable_config.cpp:198 | spread below which cloud is a single depth face | patch |
| Model.AI2TopNoFloatPrecision | 10000.0 | etc/config.toml [Model] ; read countable_config.cpp:200 | two-sided anchor pinning H to observed top | patch |
| Model.AI2TopNoFloatMarginM | 0.02 | etc/config.toml [Model] ; read countable_config.cpp:202 | upward allowance of the top anchor | patch |
| Model.AI2TopOversegSigmaPerM | 2.0 | etc/config.toml [Model] ; read countable_config.cpp:204 | R inflation above the robust top (over-segmentation tail) | patch |
| Model.AspectScale | 0.15 | etc/config.toml [Model] ; read countable_config.cpp:306 | class aspect scale of the plausibility filter | class? |
| Model.SizeScale | 0.1 | etc/config.toml [Model] ; read countable_config.cpp:308 | class footprint scale of the plausibility filter | class? |
| Model.HeightPlausibleMin | 0.4 | etc/config.toml [Model] ; read countable_config.cpp:310 | class height below which the object is implausible | class? |
| Model.HeightSoft | 0.08 | etc/config.toml [Model] ; read countable_config.cpp:312 | softness of the class height logistic | class? |
| Model.FeRef | 2.0 | etc/config.toml [Model] ; read countable_config.cpp:314 | reference FE of a healthy fit (plausibility) | patch |
| Model.FeScale | 1.0 | etc/config.toml [Model] ; read countable_config.cpp:316 | scale of FE excess in plausibility | patch |
| Model.PlausHeightPriorGain | 2000.0 | etc/config.toml [Model] ; read countable_config.cpp:318 | precision of short-height prior | patch |
| Model.SingletonInhibition | 1.0 | etc/config.toml [Model] ; read countable_config.cpp:320 | mutual inhibition of neighbours in class existence | patch |
| Model.AI2WallPrecision | 400.0 | etc/config.toml [Model] ; read countable_config.cpp:206 | precision of wall-flush factor at zero gap (class fact is WallFlushPrior) | patch? |
| Model.AI2WallParallelPrecision | 200.0 | etc/config.toml [Model] ; read countable_config.cpp:208 | precision of back-face-parallel-to-wall factor | patch? |
| Model.AI2WallReachM | 0.15 | etc/config.toml [Model] ; read countable_config.cpp:210 | gap scale over which flush weight decays | patch? |
| Model.WallFlushPrior | 0.90 | etc/config.toml [Model] ; read countable_config.cpp:183 | P(stands against a wall) | class |
| Model.RoomAxisStdDeg | 3.0 | etc/config.toml [Model] ; read countable_config.cpp:185 | aligned with room axes (std deg) | class |
| Model.AI2DoorClearanceGain | 3.0 | etc/config.toml [Model] ; read countable_config.cpp:212 | nats of preference for door facing into the room; class fact (door side) scaled by a gain | patch? |
| Model.AI2VolatilityLr | 0.02 | etc/config.toml [Model] ; read countable_config.cpp:222 | learning rate of inferred volatility | patch |
| Model.AI2VolatilitySigma | 2.0 | etc/config.toml [Model] ; read countable_config.cpp:224 | hyperprior width of inferred volatility | patch |
| Model.AI2WallExplainFrac | 0.25 | etc/config.toml [Model] ; read countable_config.cpp:226 | weight of "point belongs to a room wall" mixture component | patch |
| Model.AI2WallExplainSigmaM | 0.05 | etc/config.toml [Model] ; read countable_config.cpp:228 | wall plane slack | patch |
| Model.WallNoCrossPrecision | 2000.0 | etc/config.toml [Model] ; read countable_config.cpp:230 | one-sided precision against wall penetration | patch |
| Model.WallNoCrossMarginM | 0.0 | etc/config.toml [Model] ; read countable_config.cpp:232 | margin of wall no-cross constraint | patch |
| Model.AI2ProcessStdM | 0.005 | etc/config.toml [Model] ; read countable_config.cpp:234 | process noise on pose = how much furniture moves per frame (permanence); LIFECYCLE says no constant Q | class? |
| Model.AI2ProcessStdYaw | 0.01 | etc/config.toml [Model] ; read countable_config.cpp:236 | process noise on yaw (permanence); LIFECYCLE says no constant Q | class? |
| Model.AI2AgeNominalDtS | 0.1 | etc/config.toml [Model] ; read countable_config.cpp:238 | nominal mask-stream period (10 Hz); 0 = aging off | sensor? |
| Model.AI2TruncGateFrac | 0.10 | etc/config.toml [Model] ; read countable_config.cpp:274 | skip update if mask border fraction above this (truncation gate) | patch |
| Model.AI2RangeNoiseLatPerM | 0.02 | etc/config.toml [Model] ; read countable_config.cpp:268 | lateral deprojection std growth per metre of range (intrinsics-like) feeding R and position common-mode | sensor? |
| Model.AI2RangeNoiseYawPerM | 0.03 | etc/config.toml [Model] ; read countable_config.cpp:270 | yaw common-mode growth with range (anti far-view rotation) | patch |
| Model.AI2RangeNoiseSizePerM | 0.08 | etc/config.toml [Model] ; read countable_config.cpp:272 | size common-mode growth with range (freeze geometry afar) | patch |
| Model.AI2GnIters | 4 | etc/config.toml [Model] ; read countable_config.cpp:276 | GN iterations per frame (iteration cap) | excluded |
| Model.AI2CommonModePosStd | 0.03 | etc/config.toml [Model] ; read countable_config.cpp:240 | shared per-frame position error floor | patch |
| Model.AI2CommonModeSizeStd | 0.02 | etc/config.toml [Model] ; read countable_config.cpp:242 | shared per-frame size error floor | patch |
| Model.AI2CommonModeYawStd | 0.03 | etc/config.toml [Model] ; read countable_config.cpp:244 | shared per-frame yaw error floor | patch |
| Model.MotionCmPosGain | 0.10 | etc/config.toml [Model] ; read countable_config.cpp:246 | ego-motion shared position error per m/s (anti-drift) | patch |
| Model.MotionCmSizeGain | 0.20 | etc/config.toml [Model] ; read countable_config.cpp:248 | ego-motion shared size error per m/s (anti-reshape) | patch |
| Model.MotionCmYawGain | 0.12 | etc/config.toml [Model] ; read countable_config.cpp:250 | ego-motion shared yaw error per m/s (anti-rotate) | patch |
| Model.AI2AngLeverM | 2.0 | etc/config.toml [Model] ; read countable_config.cpp:252 | rad/s to m/s lever for ego angular motion | patch |
| Model.AI2PeriphRef | 0.50 | etc/config.toml [Model] ; read countable_config.cpp:254 | centroid radius where periphery penalty saturates | patch |
| Model.ClassAltStdM | 1.0 | etc/config.toml [Model] ; read countable_config.cpp:177 | "other furniture" alternative size std of class evidence | class? |
| Model.BirthHeightNoiseM | 0.05 | etc/config.toml [Model] ; read countable_config.cpp:179 | sigma of the height measured from the mask cloud (detector/depth property) | sensor? |
| Model.BirthHeightOutlierFrac | 0.05 | etc/config.toml [Model] ; read countable_config.cpp:181 | share of masks bleeding onto wall/person (detector property) | sensor? |
| Model.AI2MotionRefMps | 0.60 | etc/config.toml [Model] ; read countable_config.cpp:256 | motion at which a peripheral frame is unreliable | patch |
| Model.AI2StillLinMps | 0.05 | etc/config.toml [Model] ; read countable_config.cpp:260 | still threshold: camera linear speed | patch |
| Model.AI2StillAngRadps | 0.10 | etc/config.toml [Model] ; read countable_config.cpp:262 | still threshold: camera angular speed | patch |
| Model.AI2StillDotd | 0.05 | etc/config.toml [Model] ; read countable_config.cpp:264 | still threshold: mask ego-motion smear | patch |
| Model.AI2MovingUpdateCenterRadius | 0.35 | etc/config.toml [Model] ; read countable_config.cpp:266 | centroid radius below which a moving update is allowed | patch |
| Model.LidarPrecision | 2500.0 | etc/config.toml [Model] ; read countable_config.cpp:387 | per-ray LiDAR range precision (1/sigma^2 at 2 cm); set as a weight to fix mask under-size | sensor? |
| Model.LidarBpearlPrecision | 0.0 | etc/config.toml [Model] ; read countable_config.cpp:389 | per-ray precision of the low LiDAR (0 = off) | sensor? |
| Model.LidarRobustCM | 0.05 | etc/config.toml [Model] ; read countable_config.cpp:391 | Cauchy scale of LiDAR residual | patch |
| Model.LidarSelectMarginM | 0.10 | etc/config.toml [Model] ; read countable_config.cpp:393 | margin of LiDAR return pre-selection box | patch |
| Model.LidarCoverageN0 | 60.0 | etc/config.toml [Model] ; read countable_config.cpp:395 | ray count for full LiDAR weight | patch |
| Model.LidarCoverageAngPower | 1.0 | etc/config.toml [Model] ; read countable_config.cpp:397 | angular-coverage down-weight exponent | patch |
| Model.MaxStepM | 1.0 | etc/config.toml [Model] ; read countable_config.cpp:399 | reject a GN centre step larger than this (divergence guard) | patch |
| Model.CoveragePrecision | 0.0 | etc/config.toml [Model] ; read countable_config.cpp:401 | grow-only pull of clutter-ceded points (0 = off) | patch |
| Model.CoverageRobustCM | 0.15 | etc/config.toml [Model] ; read countable_config.cpp:403 | Cauchy scale of coverage pull | patch |
| Model.FreeSpacePrecision | 500.0 | etc/config.toml [Model] ; read countable_config.cpp:405 | shrink-only free-space pull per through-beam | patch |
| Model.FootprintMomentPrecision | 2000.0 | etc/config.toml [Model] ; read countable_config.cpp:407 | footprint 2nd-moment factor precision (w,h,yaw) | patch |
| Model.FootprintMomentRangePerM | 0.03 | etc/config.toml [Model] ; read countable_config.cpp:409 | moment variance growth with range | patch |
| Model.FeBaselineAdaptDown | 0.05 | etc/config.toml [Model] ; read countable_config.cpp:411 | free-energy baseline EMA, fast down | patch |
| Model.FeBaselineAdaptUp | 0.005 | etc/config.toml [Model] ; read countable_config.cpp:413 | free-energy baseline EMA, slow up | patch |
| Model.FeSurpriseSmooth | 0.10 | etc/config.toml [Model] ; read countable_config.cpp:415 | smoothing of positive FE gap (surprise) | patch |
| Model.FootprintMomentMotionGain | 0.30 | etc/config.toml [Model] ; read countable_config.cpp:417 | moment variance growth with motion | patch |
| Model.OrientationMotionRef | 0.50 | etc/config.toml [Model] ; read countable_config.cpp:419 | motion at which yaw-mode vote is halved | patch |
| Model.FootprintMomentCompletenessGain | 0.5 | etc/config.toml [Model] ; read countable_config.cpp:435 | moment variance growth with partial view | patch |
| Model.FootprintMomentMinCompleteness | 0.02 | etc/config.toml [Model] ; read countable_config.cpp:437 | clamp floor of 1/completeness | patch |
| Model.ObliquityMomentGain | 0.05 | etc/config.toml [Model] ; read countable_config.cpp:433 | grazing-view cap on moment channel | patch |
| Model.ExistenceRemovalProb | 0.12 | etc/config.toml [Model] ; read countable_config.cpp:457 | removal probability threshold | patch |
| Model.ExistenceOutOfRoomGain | 1.5 | etc/config.toml [Model] ; read countable_config.cpp:459 | log-odds step per cycle outside the room | patch |
| Model.ExistenceLogoddsMax | 4.0 | etc/config.toml [Model] ; read countable_config.cpp:463 | clamp of \|L\| | patch |
| Model.ExistenceDetectionProb | 0.85 | etc/config.toml [Model] ; read countable_config.cpp:465 | P(beam returns from occupied footprint) (sensor hit rate) | sensor |
| Model.ExistenceClutterProb | 0.05 | etc/config.toml [Model] ; read countable_config.cpp:467 | P(beam returns from an empty footprint): spurious-return rate of the LiDAR | sensor? |
| Model.ExistenceSensorSigmaM | 0.03 | etc/config.toml [Model] ; read countable_config.cpp:469 | LiDAR range sigma | sensor |
| Model.ExistenceRemoveFrames | 15 | etc/config.toml [Model] ; read countable_config.cpp:471 | removal debounce frames | patch |
| Model.ExistenceAbsenceRangeRefM | 2.5 | etc/config.toml [Model] ; read countable_config.cpp:473 | range below which absence is trusted fully | patch |
| Model.ExistenceAbsenceRangePower | 2.0 | etc/config.toml [Model] ; read countable_config.cpp:475 | decay exponent of absence with range | patch |
| Model.ExistenceVerifySurprise | 20.0 | etc/config.toml [Model] ; read countable_config.cpp:477 | surprise above which object wants verification | patch |
| Model.ExistenceVerifyGain | 5.0 | etc/config.toml [Model] ; read countable_config.cpp:481 | epistemic gain of a verification request | patch |
| Concept.Class | "nightstand" | etc/config.toml [Concept] ; read countable_config.cpp:52 | which class this binary runs as (selects manifest, label, node prefix) | excluded |
| Concept.StateEps | 0.04 | etc/config.toml [Concept] ; read countable_config.cpp:82 | convergence status tolerance (m+rad); no effect on the estimate | excluded |
| Concept.KStable | 30 | etc/config.toml [Concept] ; read countable_config.cpp:84 | frames before model_stable status flag | excluded |
| Concept.DetectionAliveMaxFrames | 40 | etc/config.toml [Concept] ; read countable_config.cpp:86 | frames without a mask before detection_alive=false (status flag; check whether existence reads it) | excluded? |
| Concept.MatchedFramesBeforeAging | 5 | etc/config.toml [Concept] ; read countable_config.cpp:88 | matched frames before a stale cycle may age the belief | patch |
| Concept.CentralRegionFrac | 0.25 | etc/config.toml [Concept] ; read countable_config.cpp:90 | central image box where the detector is reliable; part of p_detect | sensor? |
| Concept.FilterLog | false | etc/config.toml [Concept] ; read countable_config.cpp:322 | diagnostic log switch | excluded |
| Concept.FrontMinFaceAreaPx | 900.0 | etc/config.toml [Concept] ; read countable_config.cpp:427 | min projected face area to score door-ness | patch |
| Concept.FrontMinConfidence | 0.10 | etc/config.toml [Concept] ; read countable_config.cpp:429 | min door-ness margin to emit a cue | patch |
| Concept.FrontLog | true | etc/config.toml [Concept] ; read countable_config.cpp:431 | diagnostic log switch | excluded |
| Concept.LogPeriodFrames | 30 | etc/config.toml [Concept] ; read countable_config.cpp:94 | log throttle | excluded |
| Concept.ShowDashboard | true | etc/config.toml [Concept] ; read countable_config.cpp:108 | GUI switch | excluded |
| Concept.EpistemicCooldownCycles | 200 | etc/config.toml [Concept] ; read countable_config.cpp:92 | planner anti-chatter cooldown (viewpoint planner) | excluded? |
| Concept.SupportBankMaxPoints | 4000 | etc/config.toml [Concept] ; read countable_config.cpp:96 | memory cap of the support bank | excluded |
| Concept.SupportBankQuantizationM | 0.02 | etc/config.toml [Concept] ; read countable_config.cpp:98 | dedup grid of the support bank (storage) | excluded |
| Concept.SupportSelectRadiusMarginM | 0.50 | etc/config.toml [Concept] ; read countable_config.cpp:100 | XY ownership margin of support points | patch |
| Concept.SupportSelectHeightMarginM | 0.25 | etc/config.toml [Concept] ; read countable_config.cpp:102 | Z margin of support point selection | patch |
| Concept.DumpCloudPath | "/tmp/round_nightstand.xyz" | etc/config.toml [Concept] ; read countable_config.cpp:116 | diagnostic cloud dump path | excluded |
| Concept.RtCovScale | 1.0 | etc/config.toml [Concept] ; read countable_config.cpp:326 | scale on published RT covariance (calibrate toward NEES 1) | patch |
| Concept.ObjectObsFrame | "body" | etc/config.toml [Concept] ; read countable_config.cpp:330 | frame name of the published observation | excluded |
| Tracker.GateMahalanobis | 9.0 | etc/config.toml [Tracker] ; read countable_config.cpp:334 | association gate (chi2) | patch |
| Tracker.GateFallbackM | 0.50 | etc/config.toml [Tracker] ; read countable_config.cpp:336 | association metric gate before covariance exists | patch |
| Tracker.DetectionNoiseM | 0.35 | etc/config.toml [Tracker] ; read countable_config.cpp:338 | association noise added to cover centroid-vs-fit offset | patch |
| Tracker.RecallPiMove | 0.05 | etc/config.toml [Tracker] ; read countable_config.cpp:340 | permanence: P(remembered object was moved) | class |
| Tracker.RecallLogPriorNew | 0.0 | etc/config.toml [Tracker] ; read countable_config.cpp:342 | permanence: log prior of a new object | class |
| Tracker.BirthFrames | 8 | etc/config.toml [Tracker] ; read countable_config.cpp:344 | frames a mask stays unexplained before birth (debounce) | patch |
| Tracker.BirthAdmitPlausibility | 0.0 | etc/config.toml [Tracker] ; read countable_config.cpp:376 | birth admission plausibility (0 = off) | patch |
| Tracker.BirthFusionGain | 6.0 | etc/config.toml [Tracker] ; read countable_config.cpp:348 | extra birth evidence at full corroboration | patch |
| Tracker.BirthFusionMassRef | 8.0 | etc/config.toml [Tracker] ; read countable_config.cpp:350 | residual mass at half saturation | patch |
| Tracker.BirthFusionRadiusM | 0.50 | etc/config.toml [Tracker] ; read countable_config.cpp:352 | window of residual-mass sample | patch |
| Tracker.BirthMinSepM | 0.4 | etc/config.toml [Tracker] ; read countable_config.cpp:354 | minimum separation of a birth from existing objects | patch |
| Tracker.MergeOverlap | 0.05 | etc/config.toml [Tracker] ; read countable_config.cpp:358 | footprint overlap that merges two instances | patch |
| Tracker.BirthWidthM | 0.5 | etc/config.toml [Tracker] ; read countable_config.cpp:360 | default size of an instance born from a mask (class seed; fit refines it) | class? |
| Tracker.BirthDepthM | 0.5 | etc/config.toml [Tracker] ; read countable_config.cpp:362 | default size of an instance born from a mask (class seed; fit refines it) | class? |
| Tracker.BirthHeightM | 0.45 | etc/config.toml [Tracker] ; read countable_config.cpp:364 | default size of an instance born from a mask (class seed; fit refines it) | class? |
| Tracker.RicohAttentionConf | 0.60 | etc/config.toml [Tracker] ; read countable_config.cpp:378 | min confidence for Ricoh attention | patch |
| Tracker.RicohAttentionAngleMargin | 0.05 | etc/config.toml [Tracker] ; read countable_config.cpp:380 | angular margin of Ricoh attention match | patch |
| Tracker.RicohAttentionRangeBandM | 1.0 | etc/config.toml [Tracker] ; read countable_config.cpp:382 | Ricoh range roughness band used as a match margin | sensor? |
| Ice.Warn.Connections | "0" | etc/config.toml [Ice] | Ice middleware setting | excluded |
| Ice.Trace.Network | "0" | etc/config.toml [Ice] | Ice middleware setting | excluded |
| Ice.Trace.Protocol | "0" | etc/config.toml [Ice] | Ice middleware setting | excluded |
| Ice.MessageSizeMax | "20004800" | etc/config.toml [Ice] | Ice middleware setting | excluded |
| Concept.ShapeEvalPeriod | 30 | NO toml key; code default at countable_config.cpp:110 | cycles between shape evaluations (cadence) | excluded |
| Concept.ShapeEvalMinPoints | 300 | NO toml key; code default at countable_config.cpp:112 | points needed before the shape test runs | patch |
| Concept.ShapeEvidenceClamp | 8.0 | NO toml key; code default at countable_config.cpp:114 | clamp of shape evidence | patch |
| Model.AI2PriorDepthM | -1.0 | NO toml key; code default at countable_config.cpp:187 | class depth prior mean | class |
| Model.DetectMinFill | 0.10 | NO toml key; code default at countable_config.cpp:214 | detector envelope lower shoulder | sensor |
| Model.DetectMaxFill | 0.60 | NO toml key; code default at countable_config.cpp:216 | detector envelope upper shoulder | sensor |
| Model.DetectSoft | 0.06 | NO toml key; code default at countable_config.cpp:218 | detector envelope softness | sensor |
| Concept.DetectProbeCsvPath | "etc/detect_probe.csv" | NO toml key; code default at countable_config.cpp:280 | probe CSV path | excluded |
| Tracker.BirthFragmentVoxelM | 0.03 | NO toml key; code default at countable_config.cpp:369 | dedup grid of birth burst | patch |
| Tracker.BirthFragmentMaxPts | 20000 | NO toml key; code default at countable_config.cpp:371 | hard point cap of the birth burst (memory) | excluded |
| Tracker.BirthFragmentDeltaMs | 4000 | NO toml key; code default at countable_config.cpp:374 | time window of the birth burst kept (ms) | patch? |
| Model.GnTrace | 0 | NO toml key; code default at countable_config.cpp:447 | GN trace switch | excluded |
| Model.ExistenceFrameCorrelation | 0.0 | NO toml key; code default at countable_config.cpp:461 | measured frame-to-frame miss correlation of the detector (rho) | sensor? |
| Model.VerifySurpriseSmooth | 0.10 | NO toml key; code default at countable_config.cpp:479 | EMA weight of go-verify surprise | patch |
| Concept.RicohConfirmDetectionProb | 0.60 | NO toml key; code default at countable_config.cpp:488 | Ricoh detection probability (confirm path, off) | sensor |
| Concept.RicohConfirmClutterProb | 0.15 | NO toml key; code default at countable_config.cpp:490 | Ricoh clutter probability (confirm path, off) | sensor |

### Switches

| name | value | where | what it switches | kind |
|---|---|---|---|---|
| Model.FootprintResidual | false | etc/config.toml [Model] ; read countable_config.cpp:292 | footprint residual + depth nuisance (grazing-yaw fix) | patch mechanism |
| Model.QuotientChart | false | etc/config.toml [Model] ; read countable_config.cpp:294 | symmetry-quotient chart (yaw snaps) | patch mechanism |
| Model.LidarMeasuresOnGated | true | etc/config.toml [Model] ; read countable_config.cpp:445 | camera attributes, LiDAR measures (persistent shared errors) | patch mechanism |
| Model.AI2VolatilityInfer | true | etc/config.toml [Model] ; read countable_config.cpp:220 | infer per-DOF process noise (HISTORY stage; replaces constant Q) | other |
| Model.AI2MotionConfirmOnly | true | etc/config.toml [Model] ; read countable_config.cpp:258 | hard confirm-only gate when moving & off-centre | patch mechanism |
| Model.ExistenceRemovalEnabled | true | etc/config.toml [Model] ; read countable_config.cpp:441 | enable log-odds removal | patch mechanism |
| Model.ExistenceCarveParts | true | etc/config.toml [Model] ; read countable_config.cpp:443 | carve composed parts, not solid box | patch mechanism |
| Concept.FilterEnabled | true | etc/config.toml [Concept] ; read countable_config.cpp:304 | "is this really X?" plausibility filter | patch mechanism |
| Concept.FrontDetectEnabled | true | etc/config.toml [Concept] ; read countable_config.cpp:423 | appearance-based front/door-mode detector | other |
| Concept.ContourCheckEnabled | true | etc/config.toml [Concept] ; read countable_config.cpp:425 | classifier-free contour existence channel | other |
| Concept.PublishObjectObs | true | etc/config.toml [Concept] ; read countable_config.cpp:328 | publish per-frame observation to room anchor factor | other |
| Tracker.BirthFusion | true | etc/config.toml [Tracker] ; read countable_config.cpp:346 | fused birth evidence from residual grid | patch mechanism |
| Tracker.BirthSelfBody | true | NO toml key; code default at countable_config.cpp:356 | birth exclusion of the robot own body | patch mechanism |
| Tracker.BirthFragment | true | NO toml key; code default at countable_config.cpp:367 | switch of a patch mechanism | patch mechanism |
| Concept.RicohConfirmEnabled | false | NO toml key; code default at countable_config.cpp:486 | switch of a patch mechanism | patch mechanism |

### Excluded, by file

- `etc/config.toml`: 46: `Agent.id`, `Agent.name`, `Agent.configFile`, `Agent.domain`, `Agent.tree`, `Agent.graph`, `Agent.2d`, `Agent.3d`, `Presence.required_agent_names`, `Presence.config_dir`, `Presence.monitor_period_ms`, `Presence.heartbeat_timeout_ms`, `Presence.rejoin_grace_ms`, `Presence.stale_grace_ms`, `Presence.agent_info_period_ms`, `Owns.nodes`, `Period.Compute`, `Period.Emergency`, `Component.Debug.Verbose`, `Media.MasksStallTimeoutMs`, `Model.AI2CsvPath`, `Model.BirthSurpriseProbe`, `Model.LabelDumpEnabled`, `Model.LabelDumpMarginM`, `Model.AI2GnIters`, `Concept.Class`, `Concept.StateEps`, `Concept.KStable`, `Concept.DetectionAliveMaxFrames?`, `Concept.FilterLog`, `Concept.FrontLog`, `Concept.LogPeriodFrames`, `Concept.ShowDashboard`, `Concept.EpistemicCooldownCycles?`, `Concept.SupportBankMaxPoints`, `Concept.SupportBankQuantizationM`, `Concept.DumpCloudPath`, `Concept.ObjectObsFrame`, `Ice.Warn.Connections`, `Ice.Trace.Network`, `Ice.Trace.Protocol`, `Ice.MessageSizeMax`, `Concept.ShapeEvalPeriod`, `Concept.DetectProbeCsvPath`, `Tracker.BirthFragmentMaxPts`, `Model.GnTrace`

## bed (config)

### Config keys and hidden defaults

| name | value | where | role | label |
|---|---|---|---|---|
| Agent.id | 30 | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Agent.name | "bed_concept" | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Agent.configFile | "" | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Agent.domain | 0 | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Agent.tree | false | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Agent.graph | false | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Agent.2d | false | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Agent.3d | false | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Presence.required_agent_names | ["robot_concept", "room_concept"] | etc/config.toml [Presence] | presence protocol plumbing | excluded |
| Presence.config_dir | "etc" | etc/config.toml [Presence] | presence protocol plumbing | excluded |
| Presence.monitor_period_ms | 500 | etc/config.toml [Presence] | presence protocol plumbing | excluded |
| Presence.heartbeat_timeout_ms | 3000 | etc/config.toml [Presence] | presence protocol plumbing | excluded |
| Presence.rejoin_grace_ms | 1500 | etc/config.toml [Presence] | presence protocol plumbing | excluded |
| Presence.stale_grace_ms | 120000 | etc/config.toml [Presence] | presence protocol plumbing | excluded |
| Presence.agent_info_period_ms | 1000 | etc/config.toml [Presence] | presence protocol plumbing | excluded |
| Owns.nodes | ["bed_concept 30", "bed_*"] | etc/config.toml [Owns] | owned graph nodes (plumbing) | excluded |
| Period.Compute | 100 | etc/config.toml [Period] | loop period | excluded |
| Period.Emergency | 500 | etc/config.toml [Period] | loop period | excluded |
| Component.Debug.Verbose | false | etc/config.toml [Component.Debug] | log switch | excluded |
| Media.MasksStallTimeoutMs | 3000 | etc/config.toml [Media] ; read countable_config.cpp:106 | watchdog timeout on the mask stream (stops robot on stale data) | excluded |
| Model.SigmaObs | 0.05 | etc/config.toml [Model] ; read countable_config.cpp:120 | band splitting top/leg (candidate vs residual) membership | patch |
| Model.SdfThresholdForStorage | 0.08 | etc/config.toml [Model] ; read countable_config.cpp:122 | on-surface membership threshold for candidate/residual split | patch |
| Model.AI2CsvPath | "etc/ai2_log.csv" | etc/config.toml [Model] ; read countable_config.cpp:278 | per-cycle CSV path | excluded |
| Model.BirthSurpriseProbe | true | etc/config.toml [Model] ; read countable_config.cpp:282 | read-only diagnostic probe of the residual grid | excluded |
| Model.AI2SigmaBaseM | 0.03 | etc/config.toml [Model] ; read countable_config.cpp:126 | base on-surface observation noise (m); larger than depth noise, includes mask-edge error | sensor? |
| Model.PixelSigmaOverF | 0.0015 | etc/config.toml [Model] ; read countable_config.cpp:284 | pixel noise over focal (intrinsics) | sensor |
| Model.DepthSigma0M | 0.006 | etc/config.toml [Model] ; read countable_config.cpp:286 | ZED depth std floor | sensor |
| Model.DepthSigmaRangeCoef | 0.004 | etc/config.toml [Model] ; read countable_config.cpp:288 | ZED depth std growth with range^2 | sensor |
| Model.ModelSigmaM | 0.010 | etc/config.toml [Model] ; read countable_config.cpp:290 | residual model/surface sigma floor | patch |
| Model.LabelDumpEnabled | false | etc/config.toml [Model] ; read countable_config.cpp:449 | measurement-only dump switch | excluded |
| Model.LabelDumpMarginM | 0.6 | etc/config.toml [Model] ; read countable_config.cpp:451 | measurement-only dump region | excluded |
| Model.MaskPixelErrorRate | 0.085 | etc/config.toml [Model] | camera mask pixel-error rate (measured detector property) | sensor |
| Model.FloorSigmaM | 0.005 | etc/config.toml [Model] | sigma of the floor as a known surface in the beam foreign density | patch? |
| Model.DepthTiltStd | 0.020 | etc/config.toml [Model] ; read countable_config.cpp:296 | ZED shared depth-tilt prior std; consumed only by FootprintResidual (off) | sensor? |
| Model.DepthBiasStd | 0.015 | etc/config.toml [Model] ; read countable_config.cpp:298 | ZED shared depth-bias prior std; consumed only by FootprintResidual (off) | sensor? |
| Model.DepthScaleStd | 0.010 | etc/config.toml [Model] ; read countable_config.cpp:300 | ZED shared depth-scale prior std; consumed only by FootprintResidual (off) | sensor? |
| Model.AI2ClutterFrac | 0.10 | etc/config.toml [Model] ; read countable_config.cpp:128 | prior weight of the uniform clutter mixture (mask error) | patch |
| Model.AI2ClutterScaleM | 0.12 | etc/config.toml [Model] ; read countable_config.cpp:130 | distance beyond which a point is explained by clutter | patch |
| Model.AI2PriorSizeStd | 0.30 | etc/config.toml [Model] ; read countable_config.cpp:132 | broad size prior std | class |
| Model.AI2PriorFootprintM | 1.75 | etc/config.toml [Model] ; read countable_config.cpp:134 | class footprint prior mean | class |
| Model.AI2PriorFootprintStd | 0.2 | etc/config.toml [Model] ; read countable_config.cpp:136 | class footprint prior std | class |
| Model.AI2PriorDepthM | 2.1 | etc/config.toml [Model] ; read countable_config.cpp:187 | class depth prior mean | class |
| Model.AI2PriorHeightM | 0.65 | etc/config.toml [Model] ; read countable_config.cpp:192 | class height prior mean | class |
| Model.AI2PriorHeightStd | 0.10 | etc/config.toml [Model] ; read countable_config.cpp:194 | class height prior std | class |
| Model.AI2DepthUnobsPrecision | 20000.0 | etc/config.toml [Model] ; read countable_config.cpp:196 | extra depth precision when depth is unobserved (single face) | patch |
| Model.AI2DepthObsBandM | 0.10 | etc/config.toml [Model] ; read countable_config.cpp:198 | spread below which cloud is a single depth face | patch |
| Model.AI2TopNoFloatPrecision | 10000.0 | etc/config.toml [Model] ; read countable_config.cpp:200 | two-sided anchor pinning H to observed top | patch |
| Model.AI2TopNoFloatMarginM | 0.02 | etc/config.toml [Model] ; read countable_config.cpp:202 | upward allowance of the top anchor | patch |
| Model.AI2TopOversegSigmaPerM | 2.0 | etc/config.toml [Model] ; read countable_config.cpp:204 | R inflation above the robust top (over-segmentation tail) | patch |
| Model.AspectScale | 0.15 | etc/config.toml [Model] ; read countable_config.cpp:306 | class aspect scale of the plausibility filter | class? |
| Model.SizeScale | 0.3 | etc/config.toml [Model] ; read countable_config.cpp:308 | class footprint scale of the plausibility filter | class? |
| Model.HeightPlausibleMin | 0.35 | etc/config.toml [Model] ; read countable_config.cpp:310 | class height below which the object is implausible | class? |
| Model.HeightSoft | 0.1 | etc/config.toml [Model] ; read countable_config.cpp:312 | softness of the class height logistic | class? |
| Model.FeRef | 2.0 | etc/config.toml [Model] ; read countable_config.cpp:314 | reference FE of a healthy fit (plausibility) | patch |
| Model.FeScale | 1.0 | etc/config.toml [Model] ; read countable_config.cpp:316 | scale of FE excess in plausibility | patch |
| Model.PlausHeightPriorGain | 2000.0 | etc/config.toml [Model] ; read countable_config.cpp:318 | precision of short-height prior | patch |
| Model.SingletonInhibition | 1.0 | etc/config.toml [Model] ; read countable_config.cpp:320 | mutual inhibition of neighbours in class existence | patch |
| Model.AI2WallPrecision | 400.0 | etc/config.toml [Model] ; read countable_config.cpp:206 | precision of wall-flush factor at zero gap (class fact is WallFlushPrior) | patch? |
| Model.AI2WallParallelPrecision | 200.0 | etc/config.toml [Model] ; read countable_config.cpp:208 | precision of back-face-parallel-to-wall factor | patch? |
| Model.AI2WallReachM | 0.15 | etc/config.toml [Model] ; read countable_config.cpp:210 | gap scale over which flush weight decays | patch? |
| Model.WallFlushPrior | 0.90 | etc/config.toml [Model] ; read countable_config.cpp:183 | P(stands against a wall) | class |
| Model.RoomAxisStdDeg | 3.0 | etc/config.toml [Model] ; read countable_config.cpp:185 | aligned with room axes (std deg) | class |
| Model.AI2DoorClearanceGain | 3.0 | etc/config.toml [Model] ; read countable_config.cpp:212 | nats of preference for door facing into the room; class fact (door side) scaled by a gain | patch? |
| Model.AI2VolatilityLr | 0.02 | etc/config.toml [Model] ; read countable_config.cpp:222 | learning rate of inferred volatility | patch |
| Model.AI2VolatilitySigma | 2.0 | etc/config.toml [Model] ; read countable_config.cpp:224 | hyperprior width of inferred volatility | patch |
| Model.AI2WallExplainFrac | 0.25 | etc/config.toml [Model] ; read countable_config.cpp:226 | weight of "point belongs to a room wall" mixture component | patch |
| Model.AI2WallExplainSigmaM | 0.05 | etc/config.toml [Model] ; read countable_config.cpp:228 | wall plane slack | patch |
| Model.WallNoCrossPrecision | 2000.0 | etc/config.toml [Model] ; read countable_config.cpp:230 | one-sided precision against wall penetration | patch |
| Model.WallNoCrossMarginM | 0.0 | etc/config.toml [Model] ; read countable_config.cpp:232 | margin of wall no-cross constraint | patch |
| Model.AI2ProcessStdM | 0.005 | etc/config.toml [Model] ; read countable_config.cpp:234 | process noise on pose = how much furniture moves per frame (permanence); LIFECYCLE says no constant Q | class? |
| Model.AI2ProcessStdYaw | 0.01 | etc/config.toml [Model] ; read countable_config.cpp:236 | process noise on yaw (permanence); LIFECYCLE says no constant Q | class? |
| Model.AI2AgeNominalDtS | 0.1 | etc/config.toml [Model] ; read countable_config.cpp:238 | nominal mask-stream period (10 Hz); 0 = aging off | sensor? |
| Model.AI2TruncGateFrac | 0.10 | etc/config.toml [Model] ; read countable_config.cpp:274 | skip update if mask border fraction above this (truncation gate) | patch |
| Model.AI2RangeNoiseLatPerM | 0.02 | etc/config.toml [Model] ; read countable_config.cpp:268 | lateral deprojection std growth per metre of range (intrinsics-like) feeding R and position common-mode | sensor? |
| Model.AI2RangeNoiseYawPerM | 0.03 | etc/config.toml [Model] ; read countable_config.cpp:270 | yaw common-mode growth with range (anti far-view rotation) | patch |
| Model.AI2RangeNoiseSizePerM | 0.08 | etc/config.toml [Model] ; read countable_config.cpp:272 | size common-mode growth with range (freeze geometry afar) | patch |
| Model.AI2GnIters | 4 | etc/config.toml [Model] ; read countable_config.cpp:276 | GN iterations per frame (iteration cap) | excluded |
| Model.AI2CommonModePosStd | 0.03 | etc/config.toml [Model] ; read countable_config.cpp:240 | shared per-frame position error floor | patch |
| Model.AI2CommonModeSizeStd | 0.02 | etc/config.toml [Model] ; read countable_config.cpp:242 | shared per-frame size error floor | patch |
| Model.AI2CommonModeYawStd | 0.03 | etc/config.toml [Model] ; read countable_config.cpp:244 | shared per-frame yaw error floor | patch |
| Model.MotionCmPosGain | 0.10 | etc/config.toml [Model] ; read countable_config.cpp:246 | ego-motion shared position error per m/s (anti-drift) | patch |
| Model.MotionCmSizeGain | 0.20 | etc/config.toml [Model] ; read countable_config.cpp:248 | ego-motion shared size error per m/s (anti-reshape) | patch |
| Model.MotionCmYawGain | 0.12 | etc/config.toml [Model] ; read countable_config.cpp:250 | ego-motion shared yaw error per m/s (anti-rotate) | patch |
| Model.AI2AngLeverM | 2.0 | etc/config.toml [Model] ; read countable_config.cpp:252 | rad/s to m/s lever for ego angular motion | patch |
| Model.AI2PeriphRef | 0.50 | etc/config.toml [Model] ; read countable_config.cpp:254 | centroid radius where periphery penalty saturates | patch |
| Model.ClassAltStdM | 1.0 | etc/config.toml [Model] ; read countable_config.cpp:177 | "other furniture" alternative size std of class evidence | class? |
| Model.BirthHeightNoiseM | 0.05 | etc/config.toml [Model] ; read countable_config.cpp:179 | sigma of the height measured from the mask cloud (detector/depth property) | sensor? |
| Model.BirthHeightOutlierFrac | 0.05 | etc/config.toml [Model] ; read countable_config.cpp:181 | share of masks bleeding onto wall/person (detector property) | sensor? |
| Model.AI2MotionRefMps | 0.60 | etc/config.toml [Model] ; read countable_config.cpp:256 | motion at which a peripheral frame is unreliable | patch |
| Model.AI2StillLinMps | 0.05 | etc/config.toml [Model] ; read countable_config.cpp:260 | still threshold: camera linear speed | patch |
| Model.AI2StillAngRadps | 0.10 | etc/config.toml [Model] ; read countable_config.cpp:262 | still threshold: camera angular speed | patch |
| Model.AI2StillDotd | 0.05 | etc/config.toml [Model] ; read countable_config.cpp:264 | still threshold: mask ego-motion smear | patch |
| Model.AI2MovingUpdateCenterRadius | 0.35 | etc/config.toml [Model] ; read countable_config.cpp:266 | centroid radius below which a moving update is allowed | patch |
| Model.LidarPrecision | 2500.0 | etc/config.toml [Model] ; read countable_config.cpp:387 | per-ray LiDAR range precision (1/sigma^2 at 2 cm); set as a weight to fix mask under-size | sensor? |
| Model.LidarBpearlPrecision | 0.0 | etc/config.toml [Model] ; read countable_config.cpp:389 | per-ray precision of the low LiDAR (0 = off) | sensor? |
| Model.LidarRobustCM | 0.05 | etc/config.toml [Model] ; read countable_config.cpp:391 | Cauchy scale of LiDAR residual | patch |
| Model.LidarSelectMarginM | 0.10 | etc/config.toml [Model] ; read countable_config.cpp:393 | margin of LiDAR return pre-selection box | patch |
| Model.LidarCoverageN0 | 60.0 | etc/config.toml [Model] ; read countable_config.cpp:395 | ray count for full LiDAR weight | patch |
| Model.LidarCoverageAngPower | 1.0 | etc/config.toml [Model] ; read countable_config.cpp:397 | angular-coverage down-weight exponent | patch |
| Model.MaxStepM | 1.0 | etc/config.toml [Model] ; read countable_config.cpp:399 | reject a GN centre step larger than this (divergence guard) | patch |
| Model.CoveragePrecision | 0.0 | etc/config.toml [Model] ; read countable_config.cpp:401 | grow-only pull of clutter-ceded points (0 = off) | patch |
| Model.CoverageRobustCM | 0.15 | etc/config.toml [Model] ; read countable_config.cpp:403 | Cauchy scale of coverage pull | patch |
| Model.FreeSpacePrecision | 500.0 | etc/config.toml [Model] ; read countable_config.cpp:405 | shrink-only free-space pull per through-beam | patch |
| Model.FootprintMomentPrecision | 2000.0 | etc/config.toml [Model] ; read countable_config.cpp:407 | footprint 2nd-moment factor precision (w,h,yaw) | patch |
| Model.FootprintMomentRangePerM | 0.03 | etc/config.toml [Model] ; read countable_config.cpp:409 | moment variance growth with range | patch |
| Model.FeBaselineAdaptDown | 0.05 | etc/config.toml [Model] ; read countable_config.cpp:411 | free-energy baseline EMA, fast down | patch |
| Model.FeBaselineAdaptUp | 0.005 | etc/config.toml [Model] ; read countable_config.cpp:413 | free-energy baseline EMA, slow up | patch |
| Model.FeSurpriseSmooth | 0.10 | etc/config.toml [Model] ; read countable_config.cpp:415 | smoothing of positive FE gap (surprise) | patch |
| Model.FootprintMomentMotionGain | 0.30 | etc/config.toml [Model] ; read countable_config.cpp:417 | moment variance growth with motion | patch |
| Model.OrientationMotionRef | 0.50 | etc/config.toml [Model] ; read countable_config.cpp:419 | motion at which yaw-mode vote is halved | patch |
| Model.FootprintMomentCompletenessGain | 0.5 | etc/config.toml [Model] ; read countable_config.cpp:435 | moment variance growth with partial view | patch |
| Model.FootprintMomentMinCompleteness | 0.02 | etc/config.toml [Model] ; read countable_config.cpp:437 | clamp floor of 1/completeness | patch |
| Model.ObliquityMomentGain | 0.05 | etc/config.toml [Model] ; read countable_config.cpp:433 | grazing-view cap on moment channel | patch |
| Model.ExistenceRemovalProb | 0.12 | etc/config.toml [Model] ; read countable_config.cpp:457 | removal probability threshold | patch |
| Model.ExistenceOutOfRoomGain | 1.5 | etc/config.toml [Model] ; read countable_config.cpp:459 | log-odds step per cycle outside the room | patch |
| Model.ExistenceLogoddsMax | 4.0 | etc/config.toml [Model] ; read countable_config.cpp:463 | clamp of \|L\| | patch |
| Model.ExistenceDetectionProb | 0.85 | etc/config.toml [Model] ; read countable_config.cpp:465 | P(beam returns from occupied footprint) (sensor hit rate) | sensor |
| Model.ExistenceClutterProb | 0.05 | etc/config.toml [Model] ; read countable_config.cpp:467 | P(beam returns from an empty footprint): spurious-return rate of the LiDAR | sensor? |
| Model.ExistenceSensorSigmaM | 0.03 | etc/config.toml [Model] ; read countable_config.cpp:469 | LiDAR range sigma | sensor |
| Model.ExistenceRemoveFrames | 15 | etc/config.toml [Model] ; read countable_config.cpp:471 | removal debounce frames | patch |
| Model.ExistenceAbsenceRangeRefM | 2.5 | etc/config.toml [Model] ; read countable_config.cpp:473 | range below which absence is trusted fully | patch |
| Model.ExistenceAbsenceRangePower | 2.0 | etc/config.toml [Model] ; read countable_config.cpp:475 | decay exponent of absence with range | patch |
| Model.ExistenceVerifySurprise | 20.0 | etc/config.toml [Model] ; read countable_config.cpp:477 | surprise above which object wants verification | patch |
| Model.ExistenceVerifyGain | 5.0 | etc/config.toml [Model] ; read countable_config.cpp:481 | epistemic gain of a verification request | patch |
| Concept.Class | "bed" | etc/config.toml [Concept] ; read countable_config.cpp:52 | which class this binary runs as (selects manifest, label, node prefix) | excluded |
| Concept.StateEps | 0.04 | etc/config.toml [Concept] ; read countable_config.cpp:82 | convergence status tolerance (m+rad); no effect on the estimate | excluded |
| Concept.KStable | 30 | etc/config.toml [Concept] ; read countable_config.cpp:84 | frames before model_stable status flag | excluded |
| Concept.DetectionAliveMaxFrames | 40 | etc/config.toml [Concept] ; read countable_config.cpp:86 | frames without a mask before detection_alive=false (status flag; check whether existence reads it) | excluded? |
| Concept.MatchedFramesBeforeAging | 5 | etc/config.toml [Concept] ; read countable_config.cpp:88 | matched frames before a stale cycle may age the belief | patch |
| Concept.CentralRegionFrac | 0.25 | etc/config.toml [Concept] ; read countable_config.cpp:90 | central image box where the detector is reliable; part of p_detect | sensor? |
| Concept.FilterLog | false | etc/config.toml [Concept] ; read countable_config.cpp:322 | diagnostic log switch | excluded |
| Concept.FrontMinFaceAreaPx | 900.0 | etc/config.toml [Concept] ; read countable_config.cpp:427 | min projected face area to score door-ness | patch |
| Concept.FrontMinConfidence | 0.10 | etc/config.toml [Concept] ; read countable_config.cpp:429 | min door-ness margin to emit a cue | patch |
| Concept.FrontLog | true | etc/config.toml [Concept] ; read countable_config.cpp:431 | diagnostic log switch | excluded |
| Concept.LogPeriodFrames | 30 | etc/config.toml [Concept] ; read countable_config.cpp:94 | log throttle | excluded |
| Concept.ShowDashboard | true | etc/config.toml [Concept] ; read countable_config.cpp:108 | GUI switch | excluded |
| Concept.EpistemicCooldownCycles | 200 | etc/config.toml [Concept] ; read countable_config.cpp:92 | planner anti-chatter cooldown (viewpoint planner) | excluded? |
| Concept.SupportBankMaxPoints | 4000 | etc/config.toml [Concept] ; read countable_config.cpp:96 | memory cap of the support bank | excluded |
| Concept.SupportBankQuantizationM | 0.02 | etc/config.toml [Concept] ; read countable_config.cpp:98 | dedup grid of the support bank (storage) | excluded |
| Concept.SupportSelectRadiusMarginM | 0.50 | etc/config.toml [Concept] ; read countable_config.cpp:100 | XY ownership margin of support points | patch |
| Concept.SupportSelectHeightMarginM | 0.25 | etc/config.toml [Concept] ; read countable_config.cpp:102 | Z margin of support point selection | patch |
| Concept.DumpCloudPath | "/tmp/round_bed.xyz" | etc/config.toml [Concept] ; read countable_config.cpp:116 | diagnostic cloud dump path | excluded |
| Concept.RtCovScale | 1.0 | etc/config.toml [Concept] ; read countable_config.cpp:326 | scale on published RT covariance (calibrate toward NEES 1) | patch |
| Concept.ObjectObsFrame | "body" | etc/config.toml [Concept] ; read countable_config.cpp:330 | frame name of the published observation | excluded |
| Tracker.GateMahalanobis | 9.0 | etc/config.toml [Tracker] ; read countable_config.cpp:334 | association gate (chi2) | patch |
| Tracker.GateFallbackM | 1 | etc/config.toml [Tracker] ; read countable_config.cpp:336 | association metric gate before covariance exists | patch |
| Tracker.DetectionNoiseM | 0.6 | etc/config.toml [Tracker] ; read countable_config.cpp:338 | association noise added to cover centroid-vs-fit offset | patch |
| Tracker.RecallPiMove | 0.05 | etc/config.toml [Tracker] ; read countable_config.cpp:340 | permanence: P(remembered object was moved) | class |
| Tracker.RecallLogPriorNew | 0.0 | etc/config.toml [Tracker] ; read countable_config.cpp:342 | permanence: log prior of a new object | class |
| Tracker.BirthFrames | 8 | etc/config.toml [Tracker] ; read countable_config.cpp:344 | frames a mask stays unexplained before birth (debounce) | patch |
| Tracker.BirthAdmitPlausibility | 0.0 | etc/config.toml [Tracker] ; read countable_config.cpp:376 | birth admission plausibility (0 = off) | patch |
| Tracker.BirthFusionGain | 6.0 | etc/config.toml [Tracker] ; read countable_config.cpp:348 | extra birth evidence at full corroboration | patch |
| Tracker.BirthFusionMassRef | 8.0 | etc/config.toml [Tracker] ; read countable_config.cpp:350 | residual mass at half saturation | patch |
| Tracker.BirthFusionRadiusM | 1 | etc/config.toml [Tracker] ; read countable_config.cpp:352 | window of residual-mass sample | patch |
| Tracker.BirthMinSepM | 1.2 | etc/config.toml [Tracker] ; read countable_config.cpp:354 | minimum separation of a birth from existing objects | patch |
| Tracker.MergeOverlap | 0.05 | etc/config.toml [Tracker] ; read countable_config.cpp:358 | footprint overlap that merges two instances | patch |
| Tracker.BirthWidthM | 1.6 | etc/config.toml [Tracker] ; read countable_config.cpp:360 | default size of an instance born from a mask (class seed; fit refines it) | class? |
| Tracker.BirthDepthM | 1.9 | etc/config.toml [Tracker] ; read countable_config.cpp:362 | default size of an instance born from a mask (class seed; fit refines it) | class? |
| Tracker.BirthHeightM | 0.5 | etc/config.toml [Tracker] ; read countable_config.cpp:364 | default size of an instance born from a mask (class seed; fit refines it) | class? |
| Tracker.RicohAttentionConf | 0.60 | etc/config.toml [Tracker] ; read countable_config.cpp:378 | min confidence for Ricoh attention | patch |
| Tracker.RicohAttentionAngleMargin | 0.05 | etc/config.toml [Tracker] ; read countable_config.cpp:380 | angular margin of Ricoh attention match | patch |
| Tracker.RicohAttentionRangeBandM | 1.0 | etc/config.toml [Tracker] ; read countable_config.cpp:382 | Ricoh range roughness band used as a match margin | sensor? |
| Ice.Warn.Connections | "0" | etc/config.toml [Ice] | Ice middleware setting | excluded |
| Ice.Trace.Network | "0" | etc/config.toml [Ice] | Ice middleware setting | excluded |
| Ice.Trace.Protocol | "0" | etc/config.toml [Ice] | Ice middleware setting | excluded |
| Ice.MessageSizeMax | "20004800" | etc/config.toml [Ice] | Ice middleware setting | excluded |
| Concept.ShapeEvalPeriod | 30 | NO toml key; code default at countable_config.cpp:110 | cycles between shape evaluations (cadence) | excluded |
| Concept.ShapeEvalMinPoints | 300 | NO toml key; code default at countable_config.cpp:112 | points needed before the shape test runs | patch |
| Concept.ShapeEvidenceClamp | 8.0 | NO toml key; code default at countable_config.cpp:114 | clamp of shape evidence | patch |
| Model.DetectMinFill | 0.10 | NO toml key; code default at countable_config.cpp:214 | detector envelope lower shoulder | sensor |
| Model.DetectMaxFill | 0.60 | NO toml key; code default at countable_config.cpp:216 | detector envelope upper shoulder | sensor |
| Model.DetectSoft | 0.06 | NO toml key; code default at countable_config.cpp:218 | detector envelope softness | sensor |
| Concept.DetectProbeCsvPath | "etc/detect_probe.csv" | NO toml key; code default at countable_config.cpp:280 | probe CSV path | excluded |
| Tracker.BirthFragmentVoxelM | 0.03 | NO toml key; code default at countable_config.cpp:369 | dedup grid of birth burst | patch |
| Tracker.BirthFragmentMaxPts | 20000 | NO toml key; code default at countable_config.cpp:371 | hard point cap of the birth burst (memory) | excluded |
| Tracker.BirthFragmentDeltaMs | 4000 | NO toml key; code default at countable_config.cpp:374 | time window of the birth burst kept (ms) | patch? |
| Model.GnTrace | 0 | NO toml key; code default at countable_config.cpp:447 | GN trace switch | excluded |
| Model.ExistenceFrameCorrelation | 0.0 | NO toml key; code default at countable_config.cpp:461 | measured frame-to-frame miss correlation of the detector (rho) | sensor? |
| Model.VerifySurpriseSmooth | 0.10 | NO toml key; code default at countable_config.cpp:479 | EMA weight of go-verify surprise | patch |
| Concept.RicohConfirmDetectionProb | 0.60 | NO toml key; code default at countable_config.cpp:488 | Ricoh detection probability (confirm path, off) | sensor |
| Concept.RicohConfirmClutterProb | 0.15 | NO toml key; code default at countable_config.cpp:490 | Ricoh clutter probability (confirm path, off) | sensor |

### Switches

| name | value | where | what it switches | kind |
|---|---|---|---|---|
| Model.FootprintResidual | false | etc/config.toml [Model] ; read countable_config.cpp:292 | footprint residual + depth nuisance (grazing-yaw fix) | patch mechanism |
| Model.QuotientChart | false | etc/config.toml [Model] ; read countable_config.cpp:294 | symmetry-quotient chart (yaw snaps) | patch mechanism |
| Model.LidarMeasuresOnGated | true | etc/config.toml [Model] ; read countable_config.cpp:445 | camera attributes, LiDAR measures (persistent shared errors) | patch mechanism |
| Model.AI2VolatilityInfer | true | etc/config.toml [Model] ; read countable_config.cpp:220 | infer per-DOF process noise (HISTORY stage; replaces constant Q) | other |
| Model.AI2MotionConfirmOnly | true | etc/config.toml [Model] ; read countable_config.cpp:258 | hard confirm-only gate when moving & off-centre | patch mechanism |
| Model.ExistenceRemovalEnabled | true | etc/config.toml [Model] ; read countable_config.cpp:441 | enable log-odds removal | patch mechanism |
| Model.ExistenceCarveParts | true | etc/config.toml [Model] ; read countable_config.cpp:443 | carve composed parts, not solid box | patch mechanism |
| Concept.FilterEnabled | true | etc/config.toml [Concept] ; read countable_config.cpp:304 | "is this really X?" plausibility filter | patch mechanism |
| Concept.FrontDetectEnabled | false | etc/config.toml [Concept] ; read countable_config.cpp:423 | appearance-based front/door-mode detector | other |
| Concept.ContourCheckEnabled | true | etc/config.toml [Concept] ; read countable_config.cpp:425 | classifier-free contour existence channel | other |
| Concept.PublishObjectObs | true | etc/config.toml [Concept] ; read countable_config.cpp:328 | publish per-frame observation to room anchor factor | other |
| Tracker.BirthFusion | true | etc/config.toml [Tracker] ; read countable_config.cpp:346 | fused birth evidence from residual grid | patch mechanism |
| Tracker.BirthSelfBody | true | etc/config.toml [Tracker] ; read countable_config.cpp:356 | birth exclusion of the robot own body | patch mechanism |
| Tracker.BirthFragment | true | NO toml key; code default at countable_config.cpp:367 | switch of a patch mechanism | patch mechanism |
| Concept.RicohConfirmEnabled | false | NO toml key; code default at countable_config.cpp:486 | switch of a patch mechanism | patch mechanism |

### Excluded, by file

- `etc/config.toml`: 46: `Agent.id`, `Agent.name`, `Agent.configFile`, `Agent.domain`, `Agent.tree`, `Agent.graph`, `Agent.2d`, `Agent.3d`, `Presence.required_agent_names`, `Presence.config_dir`, `Presence.monitor_period_ms`, `Presence.heartbeat_timeout_ms`, `Presence.rejoin_grace_ms`, `Presence.stale_grace_ms`, `Presence.agent_info_period_ms`, `Owns.nodes`, `Period.Compute`, `Period.Emergency`, `Component.Debug.Verbose`, `Media.MasksStallTimeoutMs`, `Model.AI2CsvPath`, `Model.BirthSurpriseProbe`, `Model.LabelDumpEnabled`, `Model.LabelDumpMarginM`, `Model.AI2GnIters`, `Concept.Class`, `Concept.StateEps`, `Concept.KStable`, `Concept.DetectionAliveMaxFrames?`, `Concept.FilterLog`, `Concept.FrontLog`, `Concept.LogPeriodFrames`, `Concept.ShowDashboard`, `Concept.EpistemicCooldownCycles?`, `Concept.SupportBankMaxPoints`, `Concept.SupportBankQuantizationM`, `Concept.DumpCloudPath`, `Concept.ObjectObsFrame`, `Ice.Warn.Connections`, `Ice.Trace.Network`, `Ice.Trace.Protocol`, `Ice.MessageSizeMax`, `Concept.ShapeEvalPeriod`, `Concept.DetectProbeCsvPath`, `Tracker.BirthFragmentMaxPts`, `Model.GnTrace`

## toilet (config)

### Config keys and hidden defaults

| name | value | where | role | label |
|---|---|---|---|---|
| Agent.id | 31 | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Agent.name | "toilet_concept" | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Agent.configFile | "" | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Agent.domain | 0 | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Agent.tree | false | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Agent.graph | false | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Agent.2d | false | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Agent.3d | false | etc/config.toml [Agent] | agent/DDS plumbing | excluded |
| Presence.required_agent_names | ["robot_concept", "room_concept"] | etc/config.toml [Presence] | presence protocol plumbing | excluded |
| Presence.config_dir | "etc" | etc/config.toml [Presence] | presence protocol plumbing | excluded |
| Presence.monitor_period_ms | 500 | etc/config.toml [Presence] | presence protocol plumbing | excluded |
| Presence.heartbeat_timeout_ms | 3000 | etc/config.toml [Presence] | presence protocol plumbing | excluded |
| Presence.rejoin_grace_ms | 1500 | etc/config.toml [Presence] | presence protocol plumbing | excluded |
| Presence.stale_grace_ms | 120000 | etc/config.toml [Presence] | presence protocol plumbing | excluded |
| Presence.agent_info_period_ms | 1000 | etc/config.toml [Presence] | presence protocol plumbing | excluded |
| Owns.nodes | ["toilet_concept 31", "toilet_*"] | etc/config.toml [Owns] | owned graph nodes (plumbing) | excluded |
| Period.Compute | 100 | etc/config.toml [Period] | loop period | excluded |
| Period.Emergency | 500 | etc/config.toml [Period] | loop period | excluded |
| Component.Debug.Verbose | false | etc/config.toml [Component.Debug] | log switch | excluded |
| Media.MasksStallTimeoutMs | 3000 | etc/config.toml [Media] ; read countable_config.cpp:106 | watchdog timeout on the mask stream (stops robot on stale data) | excluded |
| Model.SigmaObs | 0.05 | etc/config.toml [Model] ; read countable_config.cpp:120 | band splitting top/leg (candidate vs residual) membership | patch |
| Model.SdfThresholdForStorage | 0.08 | etc/config.toml [Model] ; read countable_config.cpp:122 | on-surface membership threshold for candidate/residual split | patch |
| Model.AI2CsvPath | "etc/ai2_log.csv" | etc/config.toml [Model] ; read countable_config.cpp:278 | per-cycle CSV path | excluded |
| Model.BirthSurpriseProbe | true | etc/config.toml [Model] ; read countable_config.cpp:282 | read-only diagnostic probe of the residual grid | excluded |
| Model.AI2SigmaBaseM | 0.03 | etc/config.toml [Model] ; read countable_config.cpp:126 | base on-surface observation noise (m); larger than depth noise, includes mask-edge error | sensor? |
| Model.PixelSigmaOverF | 0.0015 | etc/config.toml [Model] ; read countable_config.cpp:284 | pixel noise over focal (intrinsics) | sensor |
| Model.DepthSigma0M | 0.006 | etc/config.toml [Model] ; read countable_config.cpp:286 | ZED depth std floor | sensor |
| Model.DepthSigmaRangeCoef | 0.004 | etc/config.toml [Model] ; read countable_config.cpp:288 | ZED depth std growth with range^2 | sensor |
| Model.ModelSigmaM | 0.010 | etc/config.toml [Model] ; read countable_config.cpp:290 | residual model/surface sigma floor | patch |
| Model.LabelDumpEnabled | false | etc/config.toml [Model] ; read countable_config.cpp:449 | measurement-only dump switch | excluded |
| Model.LabelDumpMarginM | 0.6 | etc/config.toml [Model] ; read countable_config.cpp:451 | measurement-only dump region | excluded |
| Model.MaskPixelErrorRate | 0.001 | etc/config.toml [Model] | camera mask pixel-error rate (measured detector property) | sensor |
| Model.FloorSigmaM | 0.005 | etc/config.toml [Model] | sigma of the floor as a known surface in the beam foreign density | patch? |
| Model.DepthTiltStd | 0.020 | etc/config.toml [Model] ; read countable_config.cpp:296 | ZED shared depth-tilt prior std; consumed only by FootprintResidual (off) | sensor? |
| Model.DepthBiasStd | 0.015 | etc/config.toml [Model] ; read countable_config.cpp:298 | ZED shared depth-bias prior std; consumed only by FootprintResidual (off) | sensor? |
| Model.DepthScaleStd | 0.010 | etc/config.toml [Model] ; read countable_config.cpp:300 | ZED shared depth-scale prior std; consumed only by FootprintResidual (off) | sensor? |
| Model.AI2ClutterFrac | 0.10 | etc/config.toml [Model] ; read countable_config.cpp:128 | prior weight of the uniform clutter mixture (mask error) | patch |
| Model.AI2ClutterScaleM | 0.12 | etc/config.toml [Model] ; read countable_config.cpp:130 | distance beyond which a point is explained by clutter | patch |
| Model.AI2PriorSizeStd | 0.30 | etc/config.toml [Model] ; read countable_config.cpp:132 | broad size prior std | class |
| Model.AI2PriorFootprintM | 0.58 | etc/config.toml [Model] ; read countable_config.cpp:134 | class footprint prior mean | class |
| Model.AI2PriorFootprintStd | 0.08 | etc/config.toml [Model] ; read countable_config.cpp:136 | class footprint prior std | class |
| Model.AI2PriorDepthM | 0.73 | etc/config.toml [Model] ; read countable_config.cpp:187 | class depth prior mean | class |
| Model.AI2PriorHeightM | 0.9 | etc/config.toml [Model] ; read countable_config.cpp:192 | class height prior mean | class |
| Model.AI2PriorHeightStd | 0.15 | etc/config.toml [Model] ; read countable_config.cpp:194 | class height prior std | class |
| Model.AI2DepthUnobsPrecision | 20000.0 | etc/config.toml [Model] ; read countable_config.cpp:196 | extra depth precision when depth is unobserved (single face) | patch |
| Model.AI2DepthObsBandM | 0.10 | etc/config.toml [Model] ; read countable_config.cpp:198 | spread below which cloud is a single depth face | patch |
| Model.AI2TopNoFloatPrecision | 10000.0 | etc/config.toml [Model] ; read countable_config.cpp:200 | two-sided anchor pinning H to observed top | patch |
| Model.AI2TopNoFloatMarginM | 0.02 | etc/config.toml [Model] ; read countable_config.cpp:202 | upward allowance of the top anchor | patch |
| Model.AI2TopOversegSigmaPerM | 2.0 | etc/config.toml [Model] ; read countable_config.cpp:204 | R inflation above the robust top (over-segmentation tail) | patch |
| Model.AspectScale | 0.15 | etc/config.toml [Model] ; read countable_config.cpp:306 | class aspect scale of the plausibility filter | class? |
| Model.SizeScale | 0.12 | etc/config.toml [Model] ; read countable_config.cpp:308 | class footprint scale of the plausibility filter | class? |
| Model.HeightPlausibleMin | 0.5 | etc/config.toml [Model] ; read countable_config.cpp:310 | class height below which the object is implausible | class? |
| Model.HeightSoft | 0.08 | etc/config.toml [Model] ; read countable_config.cpp:312 | softness of the class height logistic | class? |
| Model.FeRef | 2.0 | etc/config.toml [Model] ; read countable_config.cpp:314 | reference FE of a healthy fit (plausibility) | patch |
| Model.FeScale | 1.0 | etc/config.toml [Model] ; read countable_config.cpp:316 | scale of FE excess in plausibility | patch |
| Model.PlausHeightPriorGain | 2000.0 | etc/config.toml [Model] ; read countable_config.cpp:318 | precision of short-height prior | patch |
| Model.SingletonInhibition | 1.0 | etc/config.toml [Model] ; read countable_config.cpp:320 | mutual inhibition of neighbours in class existence | patch |
| Model.AI2WallPrecision | 400.0 | etc/config.toml [Model] ; read countable_config.cpp:206 | precision of wall-flush factor at zero gap (class fact is WallFlushPrior) | patch? |
| Model.AI2WallParallelPrecision | 200.0 | etc/config.toml [Model] ; read countable_config.cpp:208 | precision of back-face-parallel-to-wall factor | patch? |
| Model.AI2WallReachM | 0.25 | etc/config.toml [Model] ; read countable_config.cpp:210 | gap scale over which flush weight decays | patch? |
| Model.WallFlushPrior | 0.95 | etc/config.toml [Model] ; read countable_config.cpp:183 | P(stands against a wall) | class |
| Model.RoomAxisStdDeg | 3.0 | etc/config.toml [Model] ; read countable_config.cpp:185 | aligned with room axes (std deg) | class |
| Model.AI2DoorClearanceGain | 3.0 | etc/config.toml [Model] ; read countable_config.cpp:212 | nats of preference for door facing into the room; class fact (door side) scaled by a gain | patch? |
| Model.AI2VolatilityLr | 0.02 | etc/config.toml [Model] ; read countable_config.cpp:222 | learning rate of inferred volatility | patch |
| Model.AI2VolatilitySigma | 2.0 | etc/config.toml [Model] ; read countable_config.cpp:224 | hyperprior width of inferred volatility | patch |
| Model.AI2WallExplainFrac | 0.25 | etc/config.toml [Model] ; read countable_config.cpp:226 | weight of "point belongs to a room wall" mixture component | patch |
| Model.AI2WallExplainSigmaM | 0.05 | etc/config.toml [Model] ; read countable_config.cpp:228 | wall plane slack | patch |
| Model.WallNoCrossPrecision | 2000.0 | etc/config.toml [Model] ; read countable_config.cpp:230 | one-sided precision against wall penetration | patch |
| Model.WallNoCrossMarginM | 0.0 | etc/config.toml [Model] ; read countable_config.cpp:232 | margin of wall no-cross constraint | patch |
| Model.AI2ProcessStdM | 0.005 | etc/config.toml [Model] ; read countable_config.cpp:234 | process noise on pose = how much furniture moves per frame (permanence); LIFECYCLE says no constant Q | class? |
| Model.AI2ProcessStdYaw | 0.01 | etc/config.toml [Model] ; read countable_config.cpp:236 | process noise on yaw (permanence); LIFECYCLE says no constant Q | class? |
| Model.AI2AgeNominalDtS | 0.1 | etc/config.toml [Model] ; read countable_config.cpp:238 | nominal mask-stream period (10 Hz); 0 = aging off | sensor? |
| Model.AI2TruncGateFrac | 0.10 | etc/config.toml [Model] ; read countable_config.cpp:274 | skip update if mask border fraction above this (truncation gate) | patch |
| Model.AI2RangeNoiseLatPerM | 0.02 | etc/config.toml [Model] ; read countable_config.cpp:268 | lateral deprojection std growth per metre of range (intrinsics-like) feeding R and position common-mode | sensor? |
| Model.AI2RangeNoiseYawPerM | 0.03 | etc/config.toml [Model] ; read countable_config.cpp:270 | yaw common-mode growth with range (anti far-view rotation) | patch |
| Model.AI2RangeNoiseSizePerM | 0.08 | etc/config.toml [Model] ; read countable_config.cpp:272 | size common-mode growth with range (freeze geometry afar) | patch |
| Model.AI2GnIters | 4 | etc/config.toml [Model] ; read countable_config.cpp:276 | GN iterations per frame (iteration cap) | excluded |
| Model.AI2CommonModePosStd | 0.03 | etc/config.toml [Model] ; read countable_config.cpp:240 | shared per-frame position error floor | patch |
| Model.AI2CommonModeSizeStd | 0.02 | etc/config.toml [Model] ; read countable_config.cpp:242 | shared per-frame size error floor | patch |
| Model.AI2CommonModeYawStd | 0.03 | etc/config.toml [Model] ; read countable_config.cpp:244 | shared per-frame yaw error floor | patch |
| Model.MotionCmPosGain | 0.10 | etc/config.toml [Model] ; read countable_config.cpp:246 | ego-motion shared position error per m/s (anti-drift) | patch |
| Model.MotionCmSizeGain | 0.20 | etc/config.toml [Model] ; read countable_config.cpp:248 | ego-motion shared size error per m/s (anti-reshape) | patch |
| Model.MotionCmYawGain | 0.12 | etc/config.toml [Model] ; read countable_config.cpp:250 | ego-motion shared yaw error per m/s (anti-rotate) | patch |
| Model.AI2AngLeverM | 2.0 | etc/config.toml [Model] ; read countable_config.cpp:252 | rad/s to m/s lever for ego angular motion | patch |
| Model.AI2PeriphRef | 0.50 | etc/config.toml [Model] ; read countable_config.cpp:254 | centroid radius where periphery penalty saturates | patch |
| Model.ClassAltStdM | 1.0 | etc/config.toml [Model] ; read countable_config.cpp:177 | "other furniture" alternative size std of class evidence | class? |
| Model.BirthHeightNoiseM | 0.05 | etc/config.toml [Model] ; read countable_config.cpp:179 | sigma of the height measured from the mask cloud (detector/depth property) | sensor? |
| Model.BirthHeightOutlierFrac | 0.05 | etc/config.toml [Model] ; read countable_config.cpp:181 | share of masks bleeding onto wall/person (detector property) | sensor? |
| Model.AI2MotionRefMps | 0.60 | etc/config.toml [Model] ; read countable_config.cpp:256 | motion at which a peripheral frame is unreliable | patch |
| Model.AI2StillLinMps | 0.05 | etc/config.toml [Model] ; read countable_config.cpp:260 | still threshold: camera linear speed | patch |
| Model.AI2StillAngRadps | 0.10 | etc/config.toml [Model] ; read countable_config.cpp:262 | still threshold: camera angular speed | patch |
| Model.AI2StillDotd | 0.05 | etc/config.toml [Model] ; read countable_config.cpp:264 | still threshold: mask ego-motion smear | patch |
| Model.AI2MovingUpdateCenterRadius | 0.35 | etc/config.toml [Model] ; read countable_config.cpp:266 | centroid radius below which a moving update is allowed | patch |
| Model.LidarPrecision | 2500.0 | etc/config.toml [Model] ; read countable_config.cpp:387 | per-ray LiDAR range precision (1/sigma^2 at 2 cm); set as a weight to fix mask under-size | sensor? |
| Model.LidarBpearlPrecision | 0.0 | etc/config.toml [Model] ; read countable_config.cpp:389 | per-ray precision of the low LiDAR (0 = off) | sensor? |
| Model.LidarRobustCM | 0.05 | etc/config.toml [Model] ; read countable_config.cpp:391 | Cauchy scale of LiDAR residual | patch |
| Model.LidarSelectMarginM | 0.10 | etc/config.toml [Model] ; read countable_config.cpp:393 | margin of LiDAR return pre-selection box | patch |
| Model.LidarCoverageN0 | 60.0 | etc/config.toml [Model] ; read countable_config.cpp:395 | ray count for full LiDAR weight | patch |
| Model.LidarCoverageAngPower | 1.0 | etc/config.toml [Model] ; read countable_config.cpp:397 | angular-coverage down-weight exponent | patch |
| Model.MaxStepM | 1.0 | etc/config.toml [Model] ; read countable_config.cpp:399 | reject a GN centre step larger than this (divergence guard) | patch |
| Model.CoveragePrecision | 0.0 | etc/config.toml [Model] ; read countable_config.cpp:401 | grow-only pull of clutter-ceded points (0 = off) | patch |
| Model.CoverageRobustCM | 0.15 | etc/config.toml [Model] ; read countable_config.cpp:403 | Cauchy scale of coverage pull | patch |
| Model.FreeSpacePrecision | 500.0 | etc/config.toml [Model] ; read countable_config.cpp:405 | shrink-only free-space pull per through-beam | patch |
| Model.FootprintMomentPrecision | 2000.0 | etc/config.toml [Model] ; read countable_config.cpp:407 | footprint 2nd-moment factor precision (w,h,yaw) | patch |
| Model.FootprintMomentRangePerM | 0.03 | etc/config.toml [Model] ; read countable_config.cpp:409 | moment variance growth with range | patch |
| Model.FeBaselineAdaptDown | 0.05 | etc/config.toml [Model] ; read countable_config.cpp:411 | free-energy baseline EMA, fast down | patch |
| Model.FeBaselineAdaptUp | 0.005 | etc/config.toml [Model] ; read countable_config.cpp:413 | free-energy baseline EMA, slow up | patch |
| Model.FeSurpriseSmooth | 0.10 | etc/config.toml [Model] ; read countable_config.cpp:415 | smoothing of positive FE gap (surprise) | patch |
| Model.FootprintMomentMotionGain | 0.30 | etc/config.toml [Model] ; read countable_config.cpp:417 | moment variance growth with motion | patch |
| Model.OrientationMotionRef | 0.50 | etc/config.toml [Model] ; read countable_config.cpp:419 | motion at which yaw-mode vote is halved | patch |
| Model.FootprintMomentCompletenessGain | 0.5 | etc/config.toml [Model] ; read countable_config.cpp:435 | moment variance growth with partial view | patch |
| Model.FootprintMomentMinCompleteness | 0.02 | etc/config.toml [Model] ; read countable_config.cpp:437 | clamp floor of 1/completeness | patch |
| Model.ObliquityMomentGain | 0.05 | etc/config.toml [Model] ; read countable_config.cpp:433 | grazing-view cap on moment channel | patch |
| Model.ExistenceRemovalProb | 0.12 | etc/config.toml [Model] ; read countable_config.cpp:457 | removal probability threshold | patch |
| Model.ExistenceOutOfRoomGain | 1.5 | etc/config.toml [Model] ; read countable_config.cpp:459 | log-odds step per cycle outside the room | patch |
| Model.ExistenceLogoddsMax | 4.0 | etc/config.toml [Model] ; read countable_config.cpp:463 | clamp of \|L\| | patch |
| Model.ExistenceDetectionProb | 0.85 | etc/config.toml [Model] ; read countable_config.cpp:465 | P(beam returns from occupied footprint) (sensor hit rate) | sensor |
| Model.ExistenceClutterProb | 0.05 | etc/config.toml [Model] ; read countable_config.cpp:467 | P(beam returns from an empty footprint): spurious-return rate of the LiDAR | sensor? |
| Model.ExistenceSensorSigmaM | 0.03 | etc/config.toml [Model] ; read countable_config.cpp:469 | LiDAR range sigma | sensor |
| Model.ExistenceRemoveFrames | 15 | etc/config.toml [Model] ; read countable_config.cpp:471 | removal debounce frames | patch |
| Model.ExistenceAbsenceRangeRefM | 2.5 | etc/config.toml [Model] ; read countable_config.cpp:473 | range below which absence is trusted fully | patch |
| Model.ExistenceAbsenceRangePower | 2.0 | etc/config.toml [Model] ; read countable_config.cpp:475 | decay exponent of absence with range | patch |
| Model.ExistenceVerifySurprise | 20.0 | etc/config.toml [Model] ; read countable_config.cpp:477 | surprise above which object wants verification | patch |
| Model.ExistenceVerifyGain | 5.0 | etc/config.toml [Model] ; read countable_config.cpp:481 | epistemic gain of a verification request | patch |
| Concept.Class | "toilet" | etc/config.toml [Concept] ; read countable_config.cpp:52 | which class this binary runs as (selects manifest, label, node prefix) | excluded |
| Concept.StateEps | 0.04 | etc/config.toml [Concept] ; read countable_config.cpp:82 | convergence status tolerance (m+rad); no effect on the estimate | excluded |
| Concept.KStable | 30 | etc/config.toml [Concept] ; read countable_config.cpp:84 | frames before model_stable status flag | excluded |
| Concept.DetectionAliveMaxFrames | 40 | etc/config.toml [Concept] ; read countable_config.cpp:86 | frames without a mask before detection_alive=false (status flag; check whether existence reads it) | excluded? |
| Concept.MatchedFramesBeforeAging | 5 | etc/config.toml [Concept] ; read countable_config.cpp:88 | matched frames before a stale cycle may age the belief | patch |
| Concept.CentralRegionFrac | 0.25 | etc/config.toml [Concept] ; read countable_config.cpp:90 | central image box where the detector is reliable; part of p_detect | sensor? |
| Concept.FilterLog | false | etc/config.toml [Concept] ; read countable_config.cpp:322 | diagnostic log switch | excluded |
| Concept.FrontMinFaceAreaPx | 900.0 | etc/config.toml [Concept] ; read countable_config.cpp:427 | min projected face area to score door-ness | patch |
| Concept.FrontMinConfidence | 0.10 | etc/config.toml [Concept] ; read countable_config.cpp:429 | min door-ness margin to emit a cue | patch |
| Concept.FrontLog | true | etc/config.toml [Concept] ; read countable_config.cpp:431 | diagnostic log switch | excluded |
| Concept.LogPeriodFrames | 30 | etc/config.toml [Concept] ; read countable_config.cpp:94 | log throttle | excluded |
| Concept.ShowDashboard | true | etc/config.toml [Concept] ; read countable_config.cpp:108 | GUI switch | excluded |
| Concept.EpistemicCooldownCycles | 200 | etc/config.toml [Concept] ; read countable_config.cpp:92 | planner anti-chatter cooldown (viewpoint planner) | excluded? |
| Concept.SupportBankMaxPoints | 4000 | etc/config.toml [Concept] ; read countable_config.cpp:96 | memory cap of the support bank | excluded |
| Concept.SupportBankQuantizationM | 0.02 | etc/config.toml [Concept] ; read countable_config.cpp:98 | dedup grid of the support bank (storage) | excluded |
| Concept.SupportSelectRadiusMarginM | 0.50 | etc/config.toml [Concept] ; read countable_config.cpp:100 | XY ownership margin of support points | patch |
| Concept.SupportSelectHeightMarginM | 0.25 | etc/config.toml [Concept] ; read countable_config.cpp:102 | Z margin of support point selection | patch |
| Concept.DumpCloudPath | "/tmp/round_toilet.xyz" | etc/config.toml [Concept] ; read countable_config.cpp:116 | diagnostic cloud dump path | excluded |
| Concept.RtCovScale | 1.0 | etc/config.toml [Concept] ; read countable_config.cpp:326 | scale on published RT covariance (calibrate toward NEES 1) | patch |
| Concept.ObjectObsFrame | "body" | etc/config.toml [Concept] ; read countable_config.cpp:330 | frame name of the published observation | excluded |
| Tracker.GateMahalanobis | 9.0 | etc/config.toml [Tracker] ; read countable_config.cpp:334 | association gate (chi2) | patch |
| Tracker.GateFallbackM | 0.50 | etc/config.toml [Tracker] ; read countable_config.cpp:336 | association metric gate before covariance exists | patch |
| Tracker.DetectionNoiseM | 0.35 | etc/config.toml [Tracker] ; read countable_config.cpp:338 | association noise added to cover centroid-vs-fit offset | patch |
| Tracker.RecallPiMove | 0.05 | etc/config.toml [Tracker] ; read countable_config.cpp:340 | permanence: P(remembered object was moved) | class |
| Tracker.RecallLogPriorNew | 0.0 | etc/config.toml [Tracker] ; read countable_config.cpp:342 | permanence: log prior of a new object | class |
| Tracker.BirthFrames | 8 | etc/config.toml [Tracker] ; read countable_config.cpp:344 | frames a mask stays unexplained before birth (debounce) | patch |
| Tracker.BirthAdmitPlausibility | 0.0 | etc/config.toml [Tracker] ; read countable_config.cpp:376 | birth admission plausibility (0 = off) | patch |
| Tracker.BirthFusionGain | 6.0 | etc/config.toml [Tracker] ; read countable_config.cpp:348 | extra birth evidence at full corroboration | patch |
| Tracker.BirthFusionMassRef | 8.0 | etc/config.toml [Tracker] ; read countable_config.cpp:350 | residual mass at half saturation | patch |
| Tracker.BirthFusionRadiusM | 0.50 | etc/config.toml [Tracker] ; read countable_config.cpp:352 | window of residual-mass sample | patch |
| Tracker.BirthMinSepM | 0.5 | etc/config.toml [Tracker] ; read countable_config.cpp:354 | minimum separation of a birth from existing objects | patch |
| Tracker.MergeOverlap | 0.05 | etc/config.toml [Tracker] ; read countable_config.cpp:358 | footprint overlap that merges two instances | patch |
| Tracker.BirthWidthM | 0.55 | etc/config.toml [Tracker] ; read countable_config.cpp:360 | default size of an instance born from a mask (class seed; fit refines it) | class? |
| Tracker.BirthDepthM | 0.65 | etc/config.toml [Tracker] ; read countable_config.cpp:362 | default size of an instance born from a mask (class seed; fit refines it) | class? |
| Tracker.BirthHeightM | 0.55 | etc/config.toml [Tracker] ; read countable_config.cpp:364 | default size of an instance born from a mask (class seed; fit refines it) | class? |
| Tracker.RicohAttentionConf | 0.60 | etc/config.toml [Tracker] ; read countable_config.cpp:378 | min confidence for Ricoh attention | patch |
| Tracker.RicohAttentionAngleMargin | 0.05 | etc/config.toml [Tracker] ; read countable_config.cpp:380 | angular margin of Ricoh attention match | patch |
| Tracker.RicohAttentionRangeBandM | 1.0 | etc/config.toml [Tracker] ; read countable_config.cpp:382 | Ricoh range roughness band used as a match margin | sensor? |
| Ice.Warn.Connections | "0" | etc/config.toml [Ice] | Ice middleware setting | excluded |
| Ice.Trace.Network | "0" | etc/config.toml [Ice] | Ice middleware setting | excluded |
| Ice.Trace.Protocol | "0" | etc/config.toml [Ice] | Ice middleware setting | excluded |
| Ice.MessageSizeMax | "20004800" | etc/config.toml [Ice] | Ice middleware setting | excluded |
| Concept.ShapeEvalPeriod | 30 | NO toml key; code default at countable_config.cpp:110 | cycles between shape evaluations (cadence) | excluded |
| Concept.ShapeEvalMinPoints | 300 | NO toml key; code default at countable_config.cpp:112 | points needed before the shape test runs | patch |
| Concept.ShapeEvidenceClamp | 8.0 | NO toml key; code default at countable_config.cpp:114 | clamp of shape evidence | patch |
| Model.DetectMinFill | 0.10 | NO toml key; code default at countable_config.cpp:214 | detector envelope lower shoulder | sensor |
| Model.DetectMaxFill | 0.60 | NO toml key; code default at countable_config.cpp:216 | detector envelope upper shoulder | sensor |
| Model.DetectSoft | 0.06 | NO toml key; code default at countable_config.cpp:218 | detector envelope softness | sensor |
| Concept.DetectProbeCsvPath | "etc/detect_probe.csv" | NO toml key; code default at countable_config.cpp:280 | probe CSV path | excluded |
| Tracker.BirthFragmentVoxelM | 0.03 | NO toml key; code default at countable_config.cpp:369 | dedup grid of birth burst | patch |
| Tracker.BirthFragmentMaxPts | 20000 | NO toml key; code default at countable_config.cpp:371 | hard point cap of the birth burst (memory) | excluded |
| Tracker.BirthFragmentDeltaMs | 4000 | NO toml key; code default at countable_config.cpp:374 | time window of the birth burst kept (ms) | patch? |
| Model.GnTrace | 0 | NO toml key; code default at countable_config.cpp:447 | GN trace switch | excluded |
| Model.ExistenceFrameCorrelation | 0.0 | NO toml key; code default at countable_config.cpp:461 | measured frame-to-frame miss correlation of the detector (rho) | sensor? |
| Model.VerifySurpriseSmooth | 0.10 | NO toml key; code default at countable_config.cpp:479 | EMA weight of go-verify surprise | patch |
| Concept.RicohConfirmDetectionProb | 0.60 | NO toml key; code default at countable_config.cpp:488 | Ricoh detection probability (confirm path, off) | sensor |
| Concept.RicohConfirmClutterProb | 0.15 | NO toml key; code default at countable_config.cpp:490 | Ricoh clutter probability (confirm path, off) | sensor |

### Switches

| name | value | where | what it switches | kind |
|---|---|---|---|---|
| Model.FootprintResidual | false | etc/config.toml [Model] ; read countable_config.cpp:292 | footprint residual + depth nuisance (grazing-yaw fix) | patch mechanism |
| Model.QuotientChart | false | etc/config.toml [Model] ; read countable_config.cpp:294 | symmetry-quotient chart (yaw snaps) | patch mechanism |
| Model.LidarMeasuresOnGated | true | etc/config.toml [Model] ; read countable_config.cpp:445 | camera attributes, LiDAR measures (persistent shared errors) | patch mechanism |
| Model.AI2VolatilityInfer | true | etc/config.toml [Model] ; read countable_config.cpp:220 | infer per-DOF process noise (HISTORY stage; replaces constant Q) | other |
| Model.AI2MotionConfirmOnly | true | etc/config.toml [Model] ; read countable_config.cpp:258 | hard confirm-only gate when moving & off-centre | patch mechanism |
| Model.ExistenceRemovalEnabled | true | etc/config.toml [Model] ; read countable_config.cpp:441 | enable log-odds removal | patch mechanism |
| Model.ExistenceCarveParts | true | etc/config.toml [Model] ; read countable_config.cpp:443 | carve composed parts, not solid box | patch mechanism |
| Concept.FilterEnabled | true | etc/config.toml [Concept] ; read countable_config.cpp:304 | "is this really X?" plausibility filter | patch mechanism |
| Concept.FrontDetectEnabled | false | etc/config.toml [Concept] ; read countable_config.cpp:423 | appearance-based front/door-mode detector | other |
| Concept.ContourCheckEnabled | true | etc/config.toml [Concept] ; read countable_config.cpp:425 | classifier-free contour existence channel | other |
| Concept.PublishObjectObs | true | etc/config.toml [Concept] ; read countable_config.cpp:328 | publish per-frame observation to room anchor factor | other |
| Tracker.BirthFusion | true | etc/config.toml [Tracker] ; read countable_config.cpp:346 | fused birth evidence from residual grid | patch mechanism |
| Tracker.BirthSelfBody | true | NO toml key; code default at countable_config.cpp:356 | birth exclusion of the robot own body | patch mechanism |
| Tracker.BirthFragment | true | NO toml key; code default at countable_config.cpp:367 | switch of a patch mechanism | patch mechanism |
| Concept.RicohConfirmEnabled | false | NO toml key; code default at countable_config.cpp:486 | switch of a patch mechanism | patch mechanism |

### Excluded, by file

- `etc/config.toml`: 46: `Agent.id`, `Agent.name`, `Agent.configFile`, `Agent.domain`, `Agent.tree`, `Agent.graph`, `Agent.2d`, `Agent.3d`, `Presence.required_agent_names`, `Presence.config_dir`, `Presence.monitor_period_ms`, `Presence.heartbeat_timeout_ms`, `Presence.rejoin_grace_ms`, `Presence.stale_grace_ms`, `Presence.agent_info_period_ms`, `Owns.nodes`, `Period.Compute`, `Period.Emergency`, `Component.Debug.Verbose`, `Media.MasksStallTimeoutMs`, `Model.AI2CsvPath`, `Model.BirthSurpriseProbe`, `Model.LabelDumpEnabled`, `Model.LabelDumpMarginM`, `Model.AI2GnIters`, `Concept.Class`, `Concept.StateEps`, `Concept.KStable`, `Concept.DetectionAliveMaxFrames?`, `Concept.FilterLog`, `Concept.FrontLog`, `Concept.LogPeriodFrames`, `Concept.ShowDashboard`, `Concept.EpistemicCooldownCycles?`, `Concept.SupportBankMaxPoints`, `Concept.SupportBankQuantizationM`, `Concept.DumpCloudPath`, `Concept.ObjectObsFrame`, `Ice.Warn.Connections`, `Ice.Trace.Network`, `Ice.Trace.Protocol`, `Ice.MessageSizeMax`, `Concept.ShapeEvalPeriod`, `Concept.DetectProbeCsvPath`, `Tracker.BirthFragmentMaxPts`, `Model.GnTrace`

## countable engine (src)

Hard-coded constants and class branches in `countable_concept/src` (shared by nightstand, bed, toilet). Config keys of these classes are in their own sections.

Class-specific branches: there is no check on the class name or id. The only class-specific code path is the headboard (bed), selected by manifest data (`[prior.headboard]` gives `headboard_h_m > 0`), 12 sites listed in the last row below. The class name is used only as a label/prefix (YOLO label, node names, manifest path). Other class differences are config values (see the nightstand, bed, toilet sections) and the manifest.

### Hard-coded in src

| name | value | where | role | label |
|---|---|---|---|---|
| kLambda (volatility memory) | 0.995 | countable_concept/src/countable_belief.cpp:279 | forgetting factor of inferred-volatility statistics | patch? |
| omega upper clamp | 4.0 | countable_concept/src/countable_belief.cpp:289 | clamp of log process noise omega | patch |
| volatility_omega_min | -18.0 | countable_concept/src/countable_belief.h:236 | numerical floor of omega | excluded |
| gn_tol_sigma | 0.1 | countable_concept/src/countable_belief.h:277 | GN convergence tolerance | excluded |
| LM damping x4 | 4.0 | countable_concept/src/countable_belief.cpp:1308 | Levenberg-Marquardt damping update (solver) | excluded |
| finite-difference steps | 2e-3 / 2e-4 | countable_concept/src/countable_belief.cpp:1370 | FD step of diagnostics gradient | excluded |
| jac_slope_clamp | 5.0 | countable_concept/src/countable_belief.h:284 | clamp per-DOF SDF slope | patch |
| kFrontClamp | 6.0 | countable_concept/src/countable_belief.cpp:1717 | clamp of door-mode evidence, +-6 nats | patch |
| front_acc re-baseline clamp | +-6.0 | countable_concept/src/countable_belief.cpp:1549 | second use of the same clamp | patch |
| kRectAspect | 0.10 | countable_concept/src/countable_belief.h:812 | \|w-h\|/(w+h) above which footprint counts as rectangular | patch? |
| front_view_budget_ | 3.0 | countable_concept/src/countable_belief.h:820 | mode-evidence budget per observer-bearing bin | patch |
| size floors w,h,H | 0.10 | countable_concept/src/countable_belief.cpp:197 | clamp: minimum w,h,H | patch? |
| model size floors | 0.1 / 0.05 | countable_concept/src/countable_model.cpp:115 | clamp of w,h and leg length in the model | patch? |
| wall segment min length | 0.1 | countable_concept/src/countable_belief.cpp:473 | wall segment shorter than this gives no parallel-wall precision | patch? |
| wall-hit count margin | 5 c | countable_concept/src/countable_belief.cpp:1013 | a surface hit this many scales before the return counts as blocking | patch? |
| prior_pos_std / prior_yaw_std | 0.30 / 0.60 | countable_concept/src/countable_belief.h:108 | broad pose prior std | patch? |
| wall fit sigma | 0.02 | countable_concept/src/countable_fitter.cpp:975 | sigma of the fitted wall line | patch? |
| class_p_label_true default | 0.8 | countable_concept/src/countable_belief.h:149 | P(label true); manifest [prior.class] overrides | class |
| headboard_t_m default | 0.10 | countable_concept/src/countable_belief.h:142 | headboard thickness; manifest [prior.headboard] overrides | class |
| robust z-extent quantiles | 0.97 / 0.03 | countable_concept/src/countable_class_evidence.h:121 | robust height of the cloud used by class evidence | patch? |
| birth footprint percentiles | 0.05 / 0.95 / 0.9 | countable_concept/src/countable_birth_footprint.h:44 | robust extent of the birth cloud along the wall | patch? |
| kHeliosRangeSigmaM | 0.02 | countable_concept/src/countable_beam_config.h:27 | Helios LiDAR range sigma | sensor |
| beam staging rho_max / sigma_s | 6.0 / 0.01 | countable_concept/src/countable_beam_staging.h:79 | default max range and range sigma of staged beams | sensor |
| kZedMountY | -0.075 | countable_concept/src/countable_fitter.cpp:126 | ZED mount offset on the body frame (ROBOT_GEOMETRY.md) | sensor |
| graph-attr fallback seeds | 0.60,0.60,1.70 | countable_concept/src/countable_fitter.cpp:772 | size used when the graph node lacks a size attribute: FRIDGE numbers shared by every class (comment says "standard bed") | class? |
| kObliquityYawGain | 0.05 | countable_concept/src/countable_fitter.cpp:1518 | grazing-view yaw cap | patch |
| obliquity cos clamp | [0.05,1] | countable_concept/src/countable_fitter.cpp:1519 | clamp so 1/cos stays finite | patch |
| range floor | 0.1 | countable_concept/src/countable_fitter.cpp:661 | floor of range in frame weight | patch |
| birth top quantiles | 0.95 / 0.97 | countable_concept/src/countable_fitter.cpp:1048 | quantiles taken as the robust top at birth and in the top anchor | patch? |
| birth z band | 3 sigma | countable_concept/src/countable_fitter.cpp:1075 | half-width of the top band that seeds w,h | patch? |
| birth anisotropy test | 0.10 | countable_concept/src/countable_fitter.cpp:1099 | anisotropy above which moment yaw seeds the box | patch? |
| birth size floor | 0.10 | countable_concept/src/countable_fitter.cpp:1080 | floor of seeded w | patch? |
| kYawJumpLogRad | 0.087 | countable_concept/src/countable_fitter.cpp:1637 | diagnostic print trigger only | excluded |
| roi sanity gate | \\|off\\|<3, fill<4 | countable_concept/src/countable_projection.cpp:241 | reject implausible projected ROI; clamps follow | patch |
| near-plane guard | Y<=0.20 | countable_concept/src/countable_projection.cpp:216 | skip points at/behind image plane (several sites) | excluded? |
| spatial hash CELL | 6.0 px | countable_concept/src/countable_projection.cpp:280 | cell size of a pixel spatial hash | excluded |
| support select below_m | 0.05 | countable_concept/src/countable_support_bank.h:40 | margin below the floor when selecting owned points | patch |
| SDF_SMOOTH_K / INSIDE_SCALE | 0.02 / 0.3 | countable_concept/src/countable_model.h:64 | smoothing of the compound SDF | patch? |
| TOP_THICKNESS / LEG_RADIUS | 0.03 / 0.025 | countable_concept/src/countable_model.h:62 | template top thickness and leg radius (table numbers shared by all classes) | class? |
| kPosEps / kYawEps | 0.003 / 0.005 | countable_concept/src/countable_scene_graph.cpp:148 | publish-only change dedup | excluded |
| planner kRobotRadiusM / kRef | 0.30 / 0.10 | countable_concept/src/epistemic_planner.cpp:78 | viewpoint planner constants | excluded? |
| headboard branch (bed only) | headboard_h_m > 0 | countable_concept/src/countable_belief.cpp:54 | class-specific code path: SDF = min(box, panel), 2 beam parts. Selected by manifest data, not by class name. 11 sites (belief.h:162,465; belief.cpp:54,691,794,913; model.cpp:99; fitter.cpp:1466,1482; existence.cpp:366; lifecycle.cpp:209,563) | class |

### Excluded, by file

- `countable_concept/src/countable_belief.h`: 2: `volatility_omega_min`, `gn_tol_sigma`
- `countable_concept/src/countable_belief.cpp`: 2: `LM damping x4`, `finite-difference steps`
- `countable_concept/src/countable_fitter.cpp`: 1: `kYawJumpLogRad`
- `countable_concept/src/countable_projection.cpp`: 2: `near-plane guard?`, `spatial hash CELL`
- `countable_concept/src/countable_scene_graph.cpp`: 1: `kPosEps / kYawEps`
- `countable_concept/src/epistemic_planner.cpp`: 1: `planner kRobotRadiusM / kRef?`

## shared engine (common/ai_belief)

Hard-coded numeric constants in `common/ai_belief/*.h`. `recursive_laplace.h`, `persistent_update.h`, `beam_room.h` and `dof_spec.h` have no model constants beyond division guards. `beam_ray.h` and `lidar_ray_factor.h` carry the rest. No config key reaches this layer except through the class loaders (for example Model.LidarRobustCM -> robust_c_m).

### Hard-coded in src

| name | value | where | role | label |
|---|---|---|---|---|
| LidarRayFactor.robust_c_m | 0.05 | common/ai_belief/lidar_ray_factor.h:56 | default Cauchy scale / range sigma (config LidarRobustCM overrides) | patch? |
| LidarRayFactor.surf_eps_m | 2e-3 | common/ai_belief/lidar_ray_factor.h:59 | sphere-trace hit tolerance + FD step | excluded |
| LidarRayFactor.max_range_m | 6.0 | common/ai_belief/lidar_ray_factor.h:60 | stop marching at scene bound (a range limit of the sensor) | sensor? |
| LidarRayFactor.max_steps | 64 | common/ai_belief/lidar_ray_factor.h:61 | sphere-trace iteration cap | excluded |
| LidarRayFactor w_u | 0.2 | common/ai_belief/lidar_ray_factor.h:139 | share of returns the model cannot explain (uniform floor) | patch |
| LidarRayFactor tail extent | 5 sigma | common/ai_belief/lidar_ray_factor.h:212 | ray integration end beyond the observed return (two sites) | patch? |
| BeamRays.rho_max | 6.0 | common/ai_belief/beam_ray.h:451 | default max range of a ray set | sensor? |
| BeamRays.sigma_s | 0.01 | common/ai_belief/beam_ray.h:452 | default range sigma of a ray set | sensor? |
| kBeamN | 4 | common/ai_belief/beam_ray.h:424 | Gauss-Legendre nodes per subinterval (quadrature) | excluded |
| kBeamZMax / kBeamZStep | 3.0 / 0.6 | common/ai_belief/beam_ray.h:425 | extent and spacing of quantile levels (quadrature) | excluded |
| kBeamKm | 8 | common/ai_belief/beam_ray.h:427 | delta-quantile nodes of the curvature moments (quadrature) | excluded |
| kBeamFarSigmas | 8.0 | common/ai_belief/beam_ray.h:428 | a ray/part farther than 8 sigma_s adds exactly 0 ("numerical floor, not a model gate") | excluded? |
| bracket search | +-100 | common/ai_belief/beam_ray.h:478 | root bracket of a numerical solve | excluded |
| root tolerance | 64 eps | common/ai_belief/beam_ray.h:675 | floating-point tolerance | excluded |
| RecursiveLaplace dt / weight eps | 1e-6 | common/ai_belief/recursive_laplace.h:106 | division guards | excluded |

### Excluded, by file

- `common/ai_belief/lidar_ray_factor.h`: 2: `LidarRayFactor.surf_eps_m`, `LidarRayFactor.max_steps`
- `common/ai_belief/beam_ray.h`: 6: `kBeamN`, `kBeamZMax / kBeamZStep`, `kBeamKm`, `kBeamFarSigmas?`, `bracket search`, `root tolerance`
- `common/ai_belief/recursive_laplace.h`: 1: `RecursiveLaplace dt / weight eps`

## The 10 patch constants that look most entangled

Each line: the constant, the other patches that move with it, and the modelling gap they all compensate. "Entangled" = removing one makes the others need retuning, which the live-test comments in the files confirm (the comments say which fix replaced or capped which).

1. `AI2CommonModeYawStd` / `AI2CommonModePosStd` / `AI2CommonModeSizeStd`: with `AI2RangeNoiseYawPerM`, `AI2RangeNoiseSizePerM`, `MotionCmYawGain`, `MotionCmPosGain`, `MotionCmSizeGain`, `AI2PeriphRef`, `AI2AngLeverM`, `AI2MotionRefMps`. gap: shared error per frame. N correlated points from one frame must not shrink Σ; the floor, the range growth and the motion growth all stand in for one missing frame-level error term.
2. `FixationEnabled` and its gates `FixationMinPts`, `FixationMaxClutter`, `FixationCentreFrac`, `FixationStill*`: with `AI2MotionConfirmOnly`, `AI2StillLinMps`/`AngRadps`/`Dotd`, `AI2MovingUpdateCenterRadius`, `FixationPeriphFloor`, `FixationCentrePrecision`. gap: shared error per frame. The config comment says the graded terms saturate at a non-zero asymptote, so a hard gate was added on top; two gates (fixation, confirm-only) and one A/B sit side by side.
3. `AI2CoverageSizeGain` / `AI2CoverageYawGain` / `AI2CoveragePosGain` / `AI2CoverageMin` (table) and `FootprintMomentCompletenessGain` / `FootprintMomentMinCompleteness`: with `ObliquityMomentGain`, `kObliquityYawGain` (+ cos clamp 0.05), `FootprintMomentPrecision`, `FootprintMomentRangePerM`, `FootprintMomentMotionGain`. gap: shape misfit from partial or grazing views. A foreshortened footprint is explained by stretching or rotating the box; six separate inflations try to stop that.
4. `AI2DepthUnobsPrecision` + `AI2DepthObsBandM`: with `AI2PriorFootprintStd` (class), `FreeSpacePrecision`, `CoveragePrecision`, `CoverageRobustCM`, `LidarCoverageAngPower`. gap: unobserved faces / free space. A one-face view lets depth collapse; the depth precision, the free-space shrink and the coverage grow-only pull bound each other (coverage is off, free space on).
5. `AI2TopNoFloatPrecision` / `AI2TopNoFloatMarginM`: with `AI2TopOversegSigmaPerM`, `AI2PriorHeightStd` (class), `PlausHeightPriorGain`, `HeightSoft`, `HeightPlausibleMin`. gap: mask error. The box top is a free boundary, so the over-segmented mask tail ratchets H up; an anchor, an R inflation and a short-height prior all push H back down.
6. `AI2WallPrecision` / `AI2WallParallelPrecision` / `AI2WallReachM`: with `WallNoCrossPrecision`/`MarginM`, `AI2WallExplainFrac`/`SigmaM`, `AI2DoorClearanceGain`, `WallFlushPrior` (class). gap: unobserved faces. The back face is hidden against a wall; flush, parallel, no-cross, wall-as-explanation and door-clearance each cover a different symptom of the same hidden face.
7. `AI2ClutterFrac` / `AI2ClutterScaleM` (+ `AI2ClutterStructureGain` in chair): with `SigmaObs`, `SdfThresholdForStorage`, `ModelSigmaM`, `AI2SigmaBaseM`, `LidarRobustCM`, `CoverageRobustCM`. gap: mask error. The uniform clutter component, the membership band and the σ floors all decide how much of a mask is trusted; the comments record the clutter term silently absorbing a wrong template size.
8. `kFlipClamp` / `kFrontClamp` / `ShapeEvidenceClamp` and `AI2ViewBudget` / `front_view_budget_`: with `AI2ModeSatBackPts`, `OrientationMotionRef`, `ShapeEvidenceEmaAlpha`, `kRigKappaMax`/`YawKappaMax`. gap: other (dwell time counted as information). A static view repeated every frame adds evidence; a clamp, a per-view budget, a saturation and an EMA all limit the same accumulation.
9. `LidarPrecision` (set as a weight, not just σ): with `LidarCoverageN0`, `LidarCoverageAngPower`, `FootprintMomentPrecision`, `MaxStepM`, `FreeSpacePrecision`. gap: mask error (mask erosion under-size, about 6 cm). The LiDAR factor, its coverage weights and the free-space pull are tuned together against the same under-size.
10. `FeRef` / `FeScale`: with `PlausToExistenceGain`, `SingletonInhibition`, `FeBaselineAdaptDown`/`Up`, `FeSurpriseSmooth`, `ExistenceVerifySurprise`/`Gain`. gap: shape misfit. A poor fit is turned into lower plausibility, lower existence and a verify request through four stacked gains with no common unit; FeRef is marked "needs live tuning" in the config.

## Things that could not be classified cleanly, and findings a referee should know

- Process noise (`AI2ProcessStdM`, `AI2ProcessStdYaw`, `AI2ProcessStdSize`, `AI2ProcessStdExtentM`) is labelled class? (permanence of static furniture). CONCEPT_AGENT_LIFECYCLE.md forbids constant process noise and the agents carry `AI2VolatilityInfer` for that; with inference on, the constant is only the hyperprior centre. Read it as patch if the owner rules the constant is a behaviour fix.
- `AI2PriorFootprintStd` and `AI2PriorHeightStd` are labelled class (spread of the size prior), but the config comments say the tight value exists so a partial view cannot float the unobserved depth. That is a patch role carried by a class number.
- `AI2WallPrecision`, `AI2WallParallelPrecision`, `AI2WallReachM`, `AI2DoorClearanceGain` are labelled patch?: the class fact is `WallFlushPrior` (class) and the door side; the precisions are the strength of the pull.
- `LidarPrecision` and `LidarBpearlPrecision` are labelled sensor?: 1/σ² of the 2 cm Helios range noise, but the config says it is "load-bearing" to correct the mask extent, which is a patch role.
- `AI2SigmaBaseM` (0.03 m) is labelled sensor?: the depth noise is `DepthSigma0M` (6 mm); 3 cm contains mask-edge error.
- `DepthTiltStd`, `DepthBiasStd`, `DepthScaleStd` are labelled sensor? and are read only when `FootprintResidual` is on (off everywhere).
- `CentralRegionFrac`, `ExpectedSupportC`, `ExistenceFrameCorrelation`, `ExistenceClutterProb`, `AI2RangeNoiseLatPerM`, `AI2AgeNominalDtS`, `BirthHeightNoiseM`, `BirthHeightOutlierFrac`, `RicohAttentionRangeBandM` are labelled sensor?: they describe the detector or sensor but are used as correction scales.
- `FloorSigmaM` (nightstand config, countable) is labelled patch?: the spec calls it the sigma of the floor as a known surface in the beam foreign density; it is neither class nor sensor.
- `AspectScale`, `SizeScale`, `HeightPlausibleMin`, `HeightSoft`, `AltSizeScale`, `ClassAltStdM`, `BirthWidthM`/`DepthM`/`HeightM` are labelled class?: size/shape scales of the plausibility (class evidence) filter and birth seeds. The scene comment in nightstand/bed/toilet configs says these were re-fitted to the Webots threerooms-piso world, which makes them look like world fits, not class knowledge.
- `ObsDistance`, `MinStandOffM`, `EpistemicCooldownCycles`, `StaleMs`, `DetectionAliveMaxFrames` and the planner constants (`kRef`, robot radius 0.30) are excluded with `?`: viewpoint planner and watchdogs, not the estimator. `DetectionAliveMaxFrames` should be checked for a reader in the existence path.
- The countable engine seeds a node that lacks a size attribute with 0.60 x 0.60 x 1.70 (fridge numbers, comment "standard bed footprint") for every class, and its template `TOP_THICKNESS`/`LEG_RADIUS`/SDF smoothing are table numbers. Same for hood, which copies the fridge constants (SDF, clamps, kLambda). These are inherited constants, labelled class?/patch?.
- chair keeps a second copy of the +4 log-odds cap in `chair_instance.h` (`existence{0.0f, 4.0f}`) that is not read from `Existence.MaxLogodds`; table builds the birth periphery reference as a literal 0.50 while chair, refrigerator, hood and countable read `AI2PeriphRef`; chair uses literal 2.5/2.0 for the birth range reference while table reads them from config.
- chair has 20 constants that exist only as code defaults (the `Bearing.*` hypothesis birth, `AI2ObliquityYawGain`, `AI2OrientationMotionRef`, `AI2FeBaseline*`, `DetectMinFill/MaxFill/Soft`, `Existence.RemoveFrames`, `Tracker.BirthFragment*`). The toml does not show them. The same holds for table (motion gains 0.10/0.20/0.12 defaults, `Existence*` score keys), fridge, hood and the countable engine.
- Zero-valued entries (`CoveragePrecision = 0`, `LidarBpearlPrecision = 0`, `BirthAdmitPlausibility = 0`, `FootprintMomentCompletenessGain` default 0 in the engine) are counted: the mechanism exists in code and is switched off by its value. After the refactor a row that is removed, not just zeroed, should be marked as such.
- bed and toilet configs have the same key set as nightstand with different values (size priors, wall prior, birth seeds). Their rows are listed separately, but the engine they share is counted once.
- `Masks.legacy_room_frame` (table only) is a coordinate-convention A/B switch; it is listed as switch (other).
- The "where" column for config rows gives the loader line (`*_config.cpp:N`), which is where the key is read; the consumer (fitter, belief) was followed only when the role was unclear from the comment.
