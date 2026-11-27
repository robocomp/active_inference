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

---

## Addendum 1: object-path libraries (scope extension)

- Counted on tree `6e93ab69c852dd8524e8896ad003c146dff8b8dd` (HEAD `5823d6349245f2878c306e60b4a79222baf94f9f`, the commit that holds the baseline above). Date 2026-11-23.
- Same definition, labels, exclusion rule and method as above. No existing row was relabelled; the `?` rows above are untouched.
- Libraries: `ls common/` has `existence_belief`, `detectability`, `instance_tracker`, `contour_edge`, `exclusion`, `occlusion`. There is NO `common/birth`. The `rc::birth` namespace lives in `common/instance_tracker/birth_evidence.h`, so the "birth" row = `instance_tracker/birth_evidence.h` + the two closest directories `common/birth_surprise` and `common/birth_fragment`. `instance_tracker.h` is the instance_tracker row.
- `common/ai_belief`: all six files were already scanned in the baseline; there is nothing new. (`persistent_update.h`, `beam_room.h`, `dof_spec.h`, `recursive_laplace.h` have no model constants; see the shared-engine section.)
- Scan: same two passes (named constants; literal scan read in context). Defaults in library structs that a class config overrides are listed as library constants (they are the value in force when the class does not set the key). Test fixtures, log formats and `check(` lines are skipped. `exclusion/*.h` and `birth_surprise/residual_field_reader.h` and `birth_fragment.h` (non-test) hold no numeric model constants; `exclusion` therefore has 0 rows. `common/perception_card` (the p_detect composition) is linked by the chair and was not in the requested list; it is not scanned.
- Limit as before: manual last step; the rows are a careful sample, not a proof.

### Addendum summary table (libraries)

Same columns as the main summary.

| library | class | sensor | patch | switches (patch) | ambiguous (?) | excluded | switches (other) | patch total | ambiguous split | total rows |
|---|---|---|---|---|---|---|---|---|---|---|
| common/existence_belief | 0 | 2 | 6 | 0 | 1 | 3 | 0 | 6 | sensor 1 | 12 |
| common/detectability | 0 | 3 | 0 | 0 | 4 | 1 | 0 | 0 | patch 4 | 8 |
| common/birth (closest: instance_tracker/birth_evidence.h + birth_surprise + birth_fragment) | 0 | 0 | 3 | 0 | 0 | 1 | 0 | 3 | - | 4 |
| common/instance_tracker | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 6 | - | 6 |
| common/contour_edge | 0 | 0 | 0 | 0 | 7 | 1 | 0 | 0 | patch 3, sensor 4 | 8 |
| common/exclusion | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 |
| common/occlusion | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 2 | - | 2 |

### common/existence_belief

| name | value | where | role | label |
|---|---|---|---|---|
| CarveParams.sensor_sigma_m | 0.03 | common/existence_belief/existence_belief.h:52 | LiDAR range sigma (default; config ExistenceSensorSigmaM overrides) | sensor |
| CarveParams.detection_prob | 0.85 | common/existence_belief/existence_belief.h:53 | P(beam returns / occupied), sensor hit rate (default) | sensor |
| CarveParams.clutter_prob | 0.05 | common/existence_belief/existence_belief.h:54 | P(beam returns / empty) (default) | sensor? |
| ExistenceBelief L_max default | 4.0 | common/existence_belief/existence_belief.h:483 | /L/ cap of the log-odds | patch |
| frame-correlation clamp | 0.95 | common/existence_belief/existence_belief.h:513 | upper clamp of rho | patch |
| stationarity clamp | 60 | common/existence_belief/existence_belief.h:559 | argument clamp of exp(-u2) (numerics) | excluded |
| RemovalPolicy.logodds_max | 4.0 | common/existence_belief/existence_belief.h:754 | /L/ clamp, also the recantation budget | patch |
| RemovalPolicy.removal_prob | 0.12 | common/existence_belief/existence_belief.h:755 | remove below P(exists) | patch |
| RemovalPolicy.frame_correlation | 0.0 (off) | common/existence_belief/existence_belief.h:756 | rho for correlated misses; 0 = independent | patch |
| RemovalPolicy.remove_frames | 15 | common/existence_belief/existence_belief.h:757 | removal debounce in ideal observations | patch |
| RemovalPolicy.stall_warn_cycles | 300 | common/existence_belief/existence_belief.h:758 | report period of a starved instance (logging) | excluded |
| erfc / tanh structure | 0.7071, tanh | common/existence_belief/existence_belief.h:122 | Gaussian CDF; tanh common-mode saturation is structural (no constant) | excluded |

### common/detectability

| name | value | where | role | label |
|---|---|---|---|---|
| DetectorEnvelope.min_fill | 0.10 | common/detectability/detectability.h:49 | fleet-prior lower shoulder of detector envelope | sensor |
| DetectorEnvelope.max_fill | 0.60 | common/detectability/detectability.h:50 | fleet-prior upper shoulder | sensor |
| DetectorEnvelope.soft | 0.06 | common/detectability/detectability.h:51 | shoulder roll-off width | sensor |
| best_fill fallback | 0.35 | common/detectability/detectability.h:117 | framing returned when the envelope is invalid | patch? |
| best_fill scan grid / frac clamp | 200 / 0.99 | common/detectability/detectability.h:121 | scan resolution and clamp of peak fraction (numerics) | excluded |
| BlindSpotParams.ell_az/ell_el/ell_lr | 0.30 / 0.12 / 0.25 | common/detectability/blind_spot_map.h:39 | kernel length scales of the learned blind-spot correction | patch? |
| BlindSpotParams.kappa0 | 1.0 | common/detectability/blind_spot_map.h:40 | geometric prior strength (observations) | patch? |
| visit-close distance | 0.25 (dist^2) | common/detectability/blind_spot_map.h:80 | a move of half a kernel length closes a visit | patch? |

### common/birth (closest: instance_tracker/birth_evidence.h + birth_surprise + birth_fragment)

| name | value | where | role | label |
|---|---|---|---|---|
| Detectability.periph_ref | 0.50 | common/instance_tracker/birth_evidence.h:58 | periphery reference of birth quality (default) | patch |
| Detectability.range_ref_m | 2.5 | common/instance_tracker/birth_evidence.h:59 | range reference of birth quality (default) | patch |
| BirthSurpriseProbe.cover_margin_m | 0.30 | common/birth_surprise/birth_surprise_probe.h:75 | margin of "region covered by a believed box" | patch |
| surprise log radii | 0.50 / 0.30 | common/birth_surprise/birth_surprise_log.h:100 | radii of the residual-mass sample (diagnostic log, read-only probe) | excluded |

### common/instance_tracker

| name | value | where | role | label |
|---|---|---|---|---|
| TrackerParams.gate_mahalanobis | 9.0 | common/instance_tracker/instance_tracker.h:41 | association chi2 gate (default; class config overrides) | patch |
| TrackerParams.gate_fallback_m | 0.40 | common/instance_tracker/instance_tracker.h:42 | metric gate before covariance exists | patch |
| TrackerParams.detection_noise_m | 0.05 | common/instance_tracker/instance_tracker.h:48 | association noise | patch |
| TrackerParams.birth_frames | 5.0 | common/instance_tracker/instance_tracker.h:55 | birth debounce (default) | patch |
| TrackerParams.birth_min_sep_m | 0.25 | common/instance_tracker/instance_tracker.h:57 | birth separation (default) | patch |
| TrackerParams.birth_match_m | 0.20 | common/instance_tracker/instance_tracker.h:59 | candidate match radius of a pending birth | patch |

### common/contour_edge

| name | value | where | role | label |
|---|---|---|---|---|
| DepthCheck.sigma_depth_m | 0.10 | common/contour_edge/contour_depth_check.h:102 | depth sigma of the contour depth channel | sensor? |
| DepthCheck.sigma_depth_rel | 0.02 | common/contour_edge/contour_depth_check.h:103 | relative depth sigma | sensor? |
| DepthCheck.sigma_step_m | 0.08 | common/contour_edge/contour_depth_check.h:107 | depth step sigma | sensor? |
| kMinValidDepth | 0.05 | common/contour_edge/contour_depth_check.cpp:13 | minimum valid depth (ZED) | sensor? |
| kControlOffsets | -1.6,-1,1,1.6 | common/contour_edge/contour_edge_project.h:95 | control offsets for the slid-sideways comparison | patch? |
| control offsets (cpp copy) | -1.6,-1,1,1.6 | common/contour_edge/contour_edge_check.cpp:86 | second copy of the same offsets | patch? |
| edge min length | 4.0 px | common/contour_edge/contour_edge_check.cpp:52 | skip edges shorter than this (also in depth_check) | patch? |
| sampling density | 8..160 steps, len/3 | common/contour_edge/contour_edge_check.cpp:56 | samples along an edge (cost/resolution) | excluded |

### common/exclusion

No numeric model constants found in `exclusion.h` / `claims.h` (geometry and set logic only).

### common/occlusion

| name | value | where | role | label |
|---|---|---|---|---|
| kOwnWallSkipM | 0.15 | common/occlusion/occlusion.h:84 | a sample within this of a wall is not occluded by it | patch |
| occluder angular floors | 0.05 / 0.2 | common/occlusion/occlusion.h:51 | floors of occluder half-size and range | patch |

### Libraries linked per class

From `#include` in each `*_concept/src` (grep of `common/...` includes) and the CMake source lists. `*` = chair reaches `detectability` only through `common/perception_card/terms.h` (chair_detect.h), not by a direct include. Files of `common/ai_belief` are listed by file, so a class is charged only for the engine headers it includes.

| class | ai_belief files linked | libraries linked |
|---|---|---|
| chair | dof_spec.h, recursive_laplace.h | existence_belief, instance_tracker, contour_edge, exclusion, occlusion, detectability*, birth_evidence |
| table | dof_spec.h, recursive_laplace.h, lidar_ray_factor.h | existence_belief, instance_tracker, contour_edge, exclusion, occlusion, detectability, birth_evidence, birth_surprise |
| refrigerator | dof_spec.h, recursive_laplace.h, lidar_ray_factor.h | existence_belief, instance_tracker, contour_edge, exclusion, occlusion, detectability, birth_evidence, birth_surprise, birth_fragment |
| hood | dof_spec.h, recursive_laplace.h, lidar_ray_factor.h | existence_belief, instance_tracker, contour_edge, exclusion, occlusion, detectability, birth_evidence, birth_surprise, birth_fragment |
| nightstand, bed, toilet (countable) | beam_ray.h, beam_room.h, dof_spec.h, lidar_ray_factor.h, persistent_update.h, recursive_laplace.h | existence_belief, instance_tracker, contour_edge, exclusion, occlusion, detectability, birth_evidence, birth_surprise, birth_fragment |

### Combined totals per class (class row + engine row it runs on + libraries it links)

Class row = the baseline row (config + class src). Engine = for nightstand/bed/toilet the countable-engine row (src) plus the linked ai_belief files; for the others, the linked ai_belief files. Each constant counted once per class. Cells are counts of rows.

| class | class | sensor | patch | switches (patch) | ambiguous (?) | excluded | switches (other) | patch total | ambiguous split | total rows |
|---|---|---|---|---|---|---|---|---|---|---|
| chair | 5 | 8 | 98 | 9 | 38 | 41 | 6 | 107 | class 3, patch 18, sensor 9, excluded 8 | 205 |
| table | 5 | 15 | 100 | 13 | 50 | 51 | 3 | 113 | class 7, patch 20, sensor 18, excluded 5 | 237 |
| refrigerator | 9 | 15 | 114 | 8 | 64 | 53 | 4 | 122 | class 11, patch 30, sensor 18, excluded 5 | 267 |
| hood | 11 | 12 | 113 | 8 | 64 | 53 | 4 | 121 | class 11, patch 30, sensor 18, excluded 5 | 265 |
| nightstand | 13 | 19 | 112 | 11 | 69 | 65 | 4 | 123 | class 12, patch 30, sensor 22, excluded 5 | 293 |
| bed | 13 | 19 | 112 | 11 | 69 | 65 | 4 | 123 | class 12, patch 30, sensor 22, excluded 5 | 293 |
| toilet | 13 | 19 | 112 | 11 | 69 | 65 | 4 | 123 | class 12, patch 30, sensor 22, excluded 5 | 293 |

Parts of each combined total (patch total, ambiguous), so the growth is traceable:

| class | own row | engine src | ai_belief linked | libraries linked | combined patch total | combined ambiguous |
|---|---|---|---|---|---|---|
| chair | 91 / 26 | 0 / 0 | 0 / 0 | 16 / 12 | 107 | 38 |
| table | 95 / 35 | 0 / 0 | 1 / 3 | 17 / 12 | 113 | 50 |
| refrigerator | 104 / 49 | 0 / 0 | 1 / 3 | 17 / 12 | 122 | 64 |
| hood | 103 / 49 | 0 / 0 | 1 / 3 | 17 / 12 | 121 | 64 |
| nightstand | 95 / 32 | 10 / 19 | 1 / 6 | 17 / 12 | 123 | 69 |
| bed | 95 / 32 | 10 / 19 | 1 / 6 | 17 / 12 | 123 | 69 |
| toilet | 95 / 32 | 10 / 19 | 1 / 6 | 17 / 12 | 123 | 69 |

(Cells show patch total / ambiguous.) The three countable classes share the engine and library rows; their combined totals differ only in their own config row.


---

## Addendum 2: scope from the build

Rule (referee's ruling, verbatim):

> "The scope is every source file each class binary compiles or includes from common/ and from its own directory, taken from the build (the compile commands and the compiler's dependency output), not from a hand list. The after-count uses the same rule on the same build targets."

- Counted on tree `45e5f5280f48f85a2e222900aaedce39a869583b` (HEAD `535d57ec6a77c18a430848e6dad4da9a61bb647e`, which holds the baseline and addendum 1). Date 2026-11-23. Same definition, labels and exclusion rule; no existing row was relabelled and the `?` rows are untouched.
- Build targets: `chair_concept`, `table_concept`, `refrigerator_concept`, `hood_concept`, `countable_concept` (the binary nightstand, bed and toilet run).
- Method: `cmake -S <agent> -B <scratch>/deps/<agent> -DCMAKE_EXPORT_COMPILE_COMMANDS=ON` for each target (nothing was built). Every translation unit in `compile_commands.json` was rerun with `-o`/`-c` removed and `-MM -MG` added (8 at a time). The dependency lists were merged and filtered to files that exist under the worktree `common/` and under the agent's own directory. System, Qt, Eigen, `/usr/local` (cortex), and generated files were dropped.
- Notes: (1) `-MG` was not needed: no listed header was missing. (2) 13 of the translation units per target are Qt `*_autogen/mocs_compilation.cpp` files that do not exist before a build, so `-MM` returned nothing for them; the headers they would pull (the Q_OBJECT classes) are reached by the normal translation units, so the lists are not shortened by this. All other translation units gave a dependency list. (3) Dropped as generated: `<agent>_concept/generated/*` (RoboComp boilerplate, 4 files: 3 reached per target) and `common/media_transport/generated/idl/*` (Fast DDS IDL output, 35 files reached). They are tracked but machine generated, and they hold no model constants. (4) Class manifests (`*.concept.toml`) are class knowledge and stay uncounted, as before; the manifest code (`common/concept_manifest/*.h`) is in scope and is listed.
- Method for new files: the same literal scan and per-library rows. Plumbing-only files are listed with 0 rows. Pure display (`dashboard`, `graph3d`) rows are labelled excluded.
- Own-directory files were already scanned in the baseline, except a few not in my earlier lists (`*_dof.h` target sigmas and `*_model.cpp` floors); they are added below as rows of the class concerned.

### Files pulled in per target

| target | translation units | own-dir files | common/ files | total files (after dropping generated) | generated dropped | common/ files already counted (baseline + addendum 1) | common/ files newly counted |
|---|---|---|---|---|---|---|---|
| chair_concept | 68 | 27 | 138 | 165 | 38 | 16 | 122 |
| table_concept | 71 | 30 | 141 | 171 | 38 | 22 | 119 |
| refrigerator_concept | 71 | 28 | 139 | 167 | 38 | 21 | 118 |
| hood_concept | 71 | 28 | 141 | 169 | 38 | 21 | 120 |
| countable_concept | 73 | 43 | 145 | 188 | 38 | 23 | 122 |

Union of common/ files newly counted: 132 files in 53 directories (list below under "Newly counted files").

### Per-target file lists

<details><summary>chair_concept: 165 files</summary>

**chair_concept/src/** (27)

```
chair_belief.cpp
chair_belief.h
chair_card.h
chair_config.cpp
chair_config.h
chair_contour.h
chair_detect.h
chair_dof.h
chair_existence.cpp
chair_existence.h
chair_fitter.cpp
chair_fitter.h
chair_ghost.h
chair_instance.h
chair_mesh.h
chair_model.cpp
chair_model.h
chair_prior.h
chair_scene_graph.cpp
chair_scene_graph.h
chair_support_bank.h
epistemic_planner.cpp
epistemic_planner.h
specificworker.cpp
specificworker.h
specificworker_dashboard.cpp
specificworker_presence.cpp
```

**common/affordance_manager/** (4)

```
affordance_manager.cpp
affordance_manager.h
affordance_room_scope.h
update_if_live.h
```

**common/affordance_protocol/** (2)

```
affordance_goal_parse.h
affordance_protocol.h
```

**common/agent_exit/** (1)

```
terminal_exit.h
```

**common/agent_presence_coordinator/** (2)

```
agent_presence_coordinator.cpp
agent_presence_coordinator.h
```

**common/agent_presence_monitor/** (2)

```
agent_presence_monitor.cpp
agent_presence_monitor.h
```

**common/agent_state_publisher/** (4)

```
agent_state_publisher.h
agent_status.h
agent_status_overlay.cpp
agent_status_overlay.h
```

**common/ai_belief/** (2)

```
dof_spec.h
recursive_laplace.h
```

**common/appearance_belief/** (4)

```
appearance_belief.h
colour_name.h
colour_terms.h
colour_trace.h
```

**common/birth_fragment/** (1)

```
birth_fragment.h
```

**common/concept_manifest/** (2)

```
back_panel.h
concept_manifest.h
```

**common/concept_presence/** (1)

```
concept_presence.h
```

**common/config_report/** (2)

```
config_read.h
config_report.h
```

**common/contour_edge/** (6)

```
contour_depth_check.cpp
contour_depth_check.h
contour_edge_check.cpp
contour_edge_check.h
contour_edge_project.cpp
contour_edge_project.h
```

**common/dashboard/** (13)

```
belief_card_fill.h
belief_certainty.h
belief_inspector.h
belief_series.h
belief_strip.h
concept_dashboard.h
custom_widget.h
evidence_monitor.h
series_pruner.cpp
series_pruner.h
timeseries_plot.cpp
timeseries_plot.h
window_geometry.h
```

**common/detectability/** (1)

```
detectability.h
```

**common/diag_log/** (1)

```
rotating_csv.h
```

**common/epistemic_step/** (1)

```
epistemic_step.h
```

**common/exclusion/** (2)

```
claims.h
exclusion.h
```

**common/existence_belief/** (1)

```
existence_belief.h
```

**common/footprint/** (1)

```
footprint.h
```

**common/graph3d/** (3)

```
graph3d_scene.h
graph3d_scene_builder.cpp
graph3d_scene_builder.h
```

**common/graph_layout/** (1)

```
graph_layout.h
```

**common/graph_provenance/** (1)

```
creation_stamp.h
```

**common/instance_identity/** (1)

```
instance_identity.h
```

**common/instance_tracker/** (2)

```
birth_evidence.h
instance_tracker.h
```

**common/interaction_anchor/** (3)

```
anchor_leaf.h
interaction_anchor.h
relative_pose.h
```

**common/lidar_ingestor/** (2)

```
concept_lidar_ingestor.cpp
concept_lidar_ingestor.h
```

**common/mask_ingestor/** (4)

```
mask_azimuth_windows.h
mask_ingestor.cpp
mask_ingestor.h
mask_nodes.h
```

**common/media_transport/** (6)

```
flat_json.h
lidar_plane_reader.cpp
lidar_plane_reader.h
media_transport.cpp
media_transport.h
sensor_model.h
```

**common/nbv/** (4)

```
countable_planner.h
graph_obstacles.h
identified_obstacle.h
viewpoint_score.h
```

**common/obj/** (3)

```
convergence.h
obj_loader.cpp
obj_loader.h
```

**common/object_affordance/** (2)

```
object_affordance.cpp
object_affordance.h
```

**common/occlusion/** (1)

```
occlusion.h
```

**common/owned_nodes/** (1)

```
owned_nodes.h
```

**common/perception_card/** (4)

```
card.h
card_io.h
term_spec.h
terms.h
```

**common/peripheral_channel/** (1)

```
peripheral_channel.h
```

**common/phantom_log/** (2)

```
observer_pose.h
phantom_log.h
```

**common/pragmatic_affordance/** (5)

```
object_approach.cpp
object_approach.h
object_approach_geometry.h
pragmatic_affordance.cpp
pragmatic_affordance.h
```

**common/recall_match/** (3)

```
identity_link.h
recall_candidates.h
recall_match.h
```

**common/rgb_ingestor/** (4)

```
depth_ingestor.cpp
depth_ingestor.h
rgb_ingestor.cpp
rgb_ingestor.h
```

**common/robot_capability/** (2)

```
robot_capability.cpp
robot_capability.h
```

**common/robot_footprint/** (4)

```
mesh_hull.cpp
mesh_hull.h
robot_footprint.cpp
robot_footprint.h
```

**common/robust_metrics/** (2)

```
robust_metrics.cpp
robust_metrics.h
```

**common/room_resolve/** (4)

```
room_containment.h
room_follow.h
room_resolve.h
room_resolve_rule.h
```

**common/room_transition/** (1)

```
exit_fact.h
```

**common/rt_covariance/** (1)

```
rt_covariance.h
```

**common/rt_query_probe/** (1)

```
rt_query_probe.h
```

**common/rt_twist/** (1)

```
rt_twist.h
```

**common/sensor_calib/** (6)

```
proposal_publisher.cpp
proposal_publisher.h
scan_registration.cpp
scan_registration.h
trajectory_harvester.cpp
trajectory_harvester.h
```

**common/status_stream/** (3)

```
status_sink.h
status_stream.cpp
status_stream.h
```

**common/stream_gate/** (1)

```
stream_gate.h
```

**common/support_bank/** (1)

```
support_bank.h
```

**common/thumbs/** (2)

```
thumb_select.h
thumb_wire.h
```

**common/track/** (2)

```
merge_instances.h
merge_sweep.h
```

**common/view_atlas/** (1)

```
view_atlas.h
```

</details>

<details><summary>table_concept: 171 files</summary>

**common/affordance_manager/** (4)

```
affordance_manager.cpp
affordance_manager.h
affordance_room_scope.h
update_if_live.h
```

**common/affordance_protocol/** (2)

```
affordance_goal_parse.h
affordance_protocol.h
```

**common/agent_exit/** (1)

```
terminal_exit.h
```

**common/agent_presence_coordinator/** (2)

```
agent_presence_coordinator.cpp
agent_presence_coordinator.h
```

**common/agent_presence_monitor/** (2)

```
agent_presence_monitor.cpp
agent_presence_monitor.h
```

**common/agent_state_publisher/** (4)

```
agent_state_publisher.h
agent_status.h
agent_status_overlay.cpp
agent_status_overlay.h
```

**common/ai_belief/** (4)

```
beam_ray.h
dof_spec.h
lidar_ray_factor.h
recursive_laplace.h
```

**common/appearance_belief/** (1)

```
appearance_belief.h
```

**common/birth_fragment/** (1)

```
birth_fragment.h
```

**common/birth_surprise/** (3)

```
birth_surprise_log.h
birth_surprise_probe.h
residual_field_reader.h
```

**common/concept_manifest/** (2)

```
back_panel.h
concept_manifest.h
```

**common/concept_presence/** (1)

```
concept_presence.h
```

**common/config_report/** (2)

```
config_read.h
config_report.h
```

**common/contour_edge/** (6)

```
contour_depth_check.cpp
contour_depth_check.h
contour_edge_check.cpp
contour_edge_check.h
contour_edge_project.cpp
contour_edge_project.h
```

**common/dashboard/** (13)

```
belief_card_fill.h
belief_certainty.h
belief_inspector.h
belief_series.h
belief_strip.h
concept_dashboard.h
custom_widget.h
evidence_monitor.h
series_pruner.cpp
series_pruner.h
timeseries_plot.cpp
timeseries_plot.h
window_geometry.h
```

**common/detect_probe/** (1)

```
detect_probe.h
```

**common/detectability/** (2)

```
blind_spot_map.h
detectability.h
```

**common/diag_log/** (1)

```
rotating_csv.h
```

**common/epistemic_step/** (1)

```
epistemic_step.h
```

**common/exclusion/** (2)

```
claims.h
exclusion.h
```

**common/existence_belief/** (1)

```
existence_belief.h
```

**common/footprint/** (1)

```
footprint.h
```

**common/graph3d/** (3)

```
graph3d_scene.h
graph3d_scene_builder.cpp
graph3d_scene_builder.h
```

**common/graph_layout/** (1)

```
graph_layout.h
```

**common/graph_provenance/** (1)

```
creation_stamp.h
```

**common/instance_identity/** (1)

```
instance_identity.h
```

**common/instance_tracker/** (2)

```
birth_evidence.h
instance_tracker.h
```

**common/interaction_anchor/** (3)

```
anchor_leaf.h
interaction_anchor.h
relative_pose.h
```

**common/lidar_ingestor/** (2)

```
concept_lidar_ingestor.cpp
concept_lidar_ingestor.h
```

**common/lidar_select/** (1)

```
lidar_select.h
```

**common/mask_ingestor/** (4)

```
mask_azimuth_windows.h
mask_ingestor.cpp
mask_ingestor.h
mask_nodes.h
```

**common/media_transport/** (6)

```
flat_json.h
lidar_plane_reader.cpp
lidar_plane_reader.h
media_transport.cpp
media_transport.h
sensor_model.h
```

**common/nbv/** (4)

```
countable_planner.h
graph_obstacles.h
identified_obstacle.h
viewpoint_score.h
```

**common/obj/** (3)

```
convergence.h
obj_loader.cpp
obj_loader.h
```

**common/object_affordance/** (2)

```
object_affordance.cpp
object_affordance.h
```

**common/object_anchor/** (2)

```
object_anchor_contract.h
ray_anisotropic_cov.h
```

**common/occlusion/** (1)

```
occlusion.h
```

**common/owned_nodes/** (1)

```
owned_nodes.h
```

**common/perception_card/** (3)

```
card.h
card_io.h
term_spec.h
```

**common/peripheral_channel/** (1)

```
peripheral_channel.h
```

**common/phantom_log/** (2)

```
observer_pose.h
phantom_log.h
```

**common/pragmatic_affordance/** (5)

```
object_approach.cpp
object_approach.h
object_approach_geometry.h
pragmatic_affordance.cpp
pragmatic_affordance.h
```

**common/recall_match/** (2)

```
recall_candidates.h
recall_match.h
```

**common/rgb_ingestor/** (4)

```
depth_ingestor.cpp
depth_ingestor.h
rgb_ingestor.cpp
rgb_ingestor.h
```

**common/robot_capability/** (2)

```
robot_capability.cpp
robot_capability.h
```

**common/robot_footprint/** (4)

```
mesh_hull.cpp
mesh_hull.h
robot_footprint.cpp
robot_footprint.h
```

**common/robust_metrics/** (2)

```
robust_metrics.cpp
robust_metrics.h
```

**common/room_resolve/** (4)

```
room_containment.h
room_follow.h
room_resolve.h
room_resolve_rule.h
```

**common/room_transition/** (1)

```
exit_fact.h
```

**common/rt_covariance/** (1)

```
rt_covariance.h
```

**common/rt_query_probe/** (1)

```
rt_query_probe.h
```

**common/rt_twist/** (1)

```
rt_twist.h
```

**common/sensor_calib/** (6)

```
proposal_publisher.cpp
proposal_publisher.h
scan_registration.cpp
scan_registration.h
trajectory_harvester.cpp
trajectory_harvester.h
```

**common/status_stream/** (3)

```
status_sink.h
status_stream.cpp
status_stream.h
```

**common/stream_gate/** (1)

```
stream_gate.h
```

**common/support_bank/** (1)

```
support_bank.h
```

**common/track/** (2)

```
merge_instances.h
merge_sweep.h
```

**common/view_atlas/** (1)

```
view_atlas.h
```

**table_concept/src/** (30)

```
epistemic_planner.cpp
epistemic_planner.h
round_table_belief.h
specificworker.cpp
specificworker.h
specificworker_dashboard.cpp
specificworker_lifecycle.cpp
specificworker_presence.cpp
table_belief.cpp
table_belief.h
table_card.h
table_config.cpp
table_config.h
table_contour.h
table_dof.h
table_existence.cpp
table_existence.h
table_fitter.cpp
table_fitter.h
table_geometry.h
table_instance.h
table_lidar_range_channel.cpp
table_lidar_range_channel.h
table_model.cpp
table_model.h
table_projection.cpp
table_projection.h
table_scene_graph.cpp
table_scene_graph.h
table_support_bank.h
```

</details>

<details><summary>refrigerator_concept: 167 files</summary>

**common/affordance_manager/** (4)

```
affordance_manager.cpp
affordance_manager.h
affordance_room_scope.h
update_if_live.h
```

**common/affordance_protocol/** (2)

```
affordance_goal_parse.h
affordance_protocol.h
```

**common/agent_exit/** (1)

```
terminal_exit.h
```

**common/agent_presence_coordinator/** (2)

```
agent_presence_coordinator.cpp
agent_presence_coordinator.h
```

**common/agent_presence_monitor/** (2)

```
agent_presence_monitor.cpp
agent_presence_monitor.h
```

**common/agent_state_publisher/** (4)

```
agent_state_publisher.h
agent_status.h
agent_status_overlay.cpp
agent_status_overlay.h
```

**common/ai_belief/** (4)

```
beam_ray.h
dof_spec.h
lidar_ray_factor.h
recursive_laplace.h
```

**common/birth_fragment/** (1)

```
birth_fragment.h
```

**common/birth_surprise/** (3)

```
birth_surprise_log.h
birth_surprise_probe.h
residual_field_reader.h
```

**common/concept_manifest/** (2)

```
back_panel.h
concept_manifest.h
```

**common/concept_presence/** (1)

```
concept_presence.h
```

**common/config_report/** (2)

```
config_read.h
config_report.h
```

**common/contour_edge/** (6)

```
contour_depth_check.cpp
contour_depth_check.h
contour_edge_check.cpp
contour_edge_check.h
contour_edge_project.cpp
contour_edge_project.h
```

**common/dashboard/** (13)

```
belief_card_fill.h
belief_certainty.h
belief_inspector.h
belief_series.h
belief_strip.h
concept_dashboard.h
custom_widget.h
evidence_monitor.h
series_pruner.cpp
series_pruner.h
timeseries_plot.cpp
timeseries_plot.h
window_geometry.h
```

**common/detect_probe/** (1)

```
detect_probe.h
```

**common/detectability/** (1)

```
detectability.h
```

**common/diag_log/** (1)

```
rotating_csv.h
```

**common/epistemic_step/** (1)

```
epistemic_step.h
```

**common/exclusion/** (2)

```
claims.h
exclusion.h
```

**common/existence_belief/** (1)

```
existence_belief.h
```

**common/footprint/** (1)

```
footprint.h
```

**common/graph3d/** (3)

```
graph3d_scene.h
graph3d_scene_builder.cpp
graph3d_scene_builder.h
```

**common/graph_layout/** (1)

```
graph_layout.h
```

**common/graph_provenance/** (1)

```
creation_stamp.h
```

**common/instance_identity/** (1)

```
instance_identity.h
```

**common/instance_tracker/** (2)

```
birth_evidence.h
instance_tracker.h
```

**common/interaction_anchor/** (3)

```
anchor_leaf.h
interaction_anchor.h
relative_pose.h
```

**common/lidar_ingestor/** (2)

```
concept_lidar_ingestor.cpp
concept_lidar_ingestor.h
```

**common/lidar_select/** (1)

```
lidar_select.h
```

**common/mask_ingestor/** (4)

```
mask_azimuth_windows.h
mask_ingestor.cpp
mask_ingestor.h
mask_nodes.h
```

**common/media_transport/** (6)

```
flat_json.h
lidar_plane_reader.cpp
lidar_plane_reader.h
media_transport.cpp
media_transport.h
sensor_model.h
```

**common/nbv/** (4)

```
countable_planner.h
graph_obstacles.h
identified_obstacle.h
viewpoint_score.h
```

**common/obj/** (3)

```
convergence.h
obj_loader.cpp
obj_loader.h
```

**common/object_affordance/** (2)

```
object_affordance.cpp
object_affordance.h
```

**common/object_anchor/** (2)

```
object_anchor_contract.h
ray_anisotropic_cov.h
```

**common/occlusion/** (1)

```
occlusion.h
```

**common/owned_nodes/** (1)

```
owned_nodes.h
```

**common/perception_card/** (3)

```
card.h
card_io.h
term_spec.h
```

**common/peripheral_channel/** (1)

```
peripheral_channel.h
```

**common/phantom_log/** (2)

```
observer_pose.h
phantom_log.h
```

**common/pragmatic_affordance/** (5)

```
object_approach.cpp
object_approach.h
object_approach_geometry.h
pragmatic_affordance.cpp
pragmatic_affordance.h
```

**common/recall_match/** (2)

```
recall_candidates.h
recall_match.h
```

**common/rgb_ingestor/** (4)

```
depth_ingestor.cpp
depth_ingestor.h
rgb_ingestor.cpp
rgb_ingestor.h
```

**common/robot_capability/** (2)

```
robot_capability.cpp
robot_capability.h
```

**common/robot_footprint/** (4)

```
mesh_hull.cpp
mesh_hull.h
robot_footprint.cpp
robot_footprint.h
```

**common/robust_metrics/** (2)

```
robust_metrics.cpp
robust_metrics.h
```

**common/room_resolve/** (4)

```
room_containment.h
room_follow.h
room_resolve.h
room_resolve_rule.h
```

**common/room_transition/** (1)

```
exit_fact.h
```

**common/rt_covariance/** (1)

```
rt_covariance.h
```

**common/rt_query_probe/** (1)

```
rt_query_probe.h
```

**common/rt_twist/** (1)

```
rt_twist.h
```

**common/sensor_calib/** (6)

```
proposal_publisher.cpp
proposal_publisher.h
scan_registration.cpp
scan_registration.h
trajectory_harvester.cpp
trajectory_harvester.h
```

**common/status_stream/** (3)

```
status_sink.h
status_stream.cpp
status_stream.h
```

**common/stream_gate/** (1)

```
stream_gate.h
```

**common/support_bank/** (1)

```
support_bank.h
```

**common/track/** (2)

```
merge_instances.h
merge_sweep.h
```

**common/view_atlas/** (1)

```
view_atlas.h
```

**refrigerator_concept/src/** (28)

```
epistemic_planner.cpp
epistemic_planner.h
refrigerator_belief.cpp
refrigerator_belief.h
refrigerator_card.h
refrigerator_config.cpp
refrigerator_config.h
refrigerator_dof.h
refrigerator_existence.cpp
refrigerator_existence.h
refrigerator_fitter.cpp
refrigerator_fitter.h
refrigerator_geometry.h
refrigerator_instance.h
refrigerator_lidar_range_channel.cpp
refrigerator_lidar_range_channel.h
refrigerator_model.cpp
refrigerator_model.h
refrigerator_projection.cpp
refrigerator_projection.h
refrigerator_scene_graph.cpp
refrigerator_scene_graph.h
refrigerator_support_bank.h
specificworker.cpp
specificworker.h
specificworker_dashboard.cpp
specificworker_lifecycle.cpp
specificworker_presence.cpp
```

</details>

<details><summary>hood_concept: 169 files</summary>

**common/affordance_manager/** (4)

```
affordance_manager.cpp
affordance_manager.h
affordance_room_scope.h
update_if_live.h
```

**common/affordance_protocol/** (2)

```
affordance_goal_parse.h
affordance_protocol.h
```

**common/agent_exit/** (1)

```
terminal_exit.h
```

**common/agent_presence_coordinator/** (2)

```
agent_presence_coordinator.cpp
agent_presence_coordinator.h
```

**common/agent_presence_monitor/** (2)

```
agent_presence_monitor.cpp
agent_presence_monitor.h
```

**common/agent_state_publisher/** (4)

```
agent_state_publisher.h
agent_status.h
agent_status_overlay.cpp
agent_status_overlay.h
```

**common/ai_belief/** (4)

```
beam_ray.h
dof_spec.h
lidar_ray_factor.h
recursive_laplace.h
```

**common/birth_fragment/** (1)

```
birth_fragment.h
```

**common/birth_surprise/** (3)

```
birth_surprise_log.h
birth_surprise_probe.h
residual_field_reader.h
```

**common/concept_manifest/** (2)

```
back_panel.h
concept_manifest.h
```

**common/concept_presence/** (1)

```
concept_presence.h
```

**common/config_report/** (2)

```
config_read.h
config_report.h
```

**common/contour_edge/** (6)

```
contour_depth_check.cpp
contour_depth_check.h
contour_edge_check.cpp
contour_edge_check.h
contour_edge_project.cpp
contour_edge_project.h
```

**common/dashboard/** (13)

```
belief_card_fill.h
belief_certainty.h
belief_inspector.h
belief_series.h
belief_strip.h
concept_dashboard.h
custom_widget.h
evidence_monitor.h
series_pruner.cpp
series_pruner.h
timeseries_plot.cpp
timeseries_plot.h
window_geometry.h
```

**common/detect_probe/** (1)

```
detect_probe.h
```

**common/detectability/** (1)

```
detectability.h
```

**common/diag_log/** (1)

```
rotating_csv.h
```

**common/epistemic_step/** (1)

```
epistemic_step.h
```

**common/exclusion/** (2)

```
claims.h
exclusion.h
```

**common/existence_belief/** (1)

```
existence_belief.h
```

**common/footprint/** (1)

```
footprint.h
```

**common/graph3d/** (3)

```
graph3d_scene.h
graph3d_scene_builder.cpp
graph3d_scene_builder.h
```

**common/graph_layout/** (1)

```
graph_layout.h
```

**common/graph_provenance/** (1)

```
creation_stamp.h
```

**common/instance_identity/** (1)

```
instance_identity.h
```

**common/instance_tracker/** (2)

```
birth_evidence.h
instance_tracker.h
```

**common/interaction_anchor/** (3)

```
anchor_leaf.h
interaction_anchor.h
relative_pose.h
```

**common/lidar_ingestor/** (2)

```
concept_lidar_ingestor.cpp
concept_lidar_ingestor.h
```

**common/lidar_select/** (1)

```
lidar_select.h
```

**common/mask_ingestor/** (4)

```
mask_azimuth_windows.h
mask_ingestor.cpp
mask_ingestor.h
mask_nodes.h
```

**common/media_transport/** (6)

```
flat_json.h
lidar_plane_reader.cpp
lidar_plane_reader.h
media_transport.cpp
media_transport.h
sensor_model.h
```

**common/nbv/** (4)

```
countable_planner.h
graph_obstacles.h
identified_obstacle.h
viewpoint_score.h
```

**common/obj/** (3)

```
convergence.h
obj_loader.cpp
obj_loader.h
```

**common/object_affordance/** (2)

```
object_affordance.cpp
object_affordance.h
```

**common/object_anchor/** (2)

```
object_anchor_contract.h
ray_anisotropic_cov.h
```

**common/occlusion/** (1)

```
occlusion.h
```

**common/owned_nodes/** (1)

```
owned_nodes.h
```

**common/perception_card/** (3)

```
card.h
card_io.h
term_spec.h
```

**common/peripheral_channel/** (1)

```
peripheral_channel.h
```

**common/phantom_log/** (2)

```
observer_pose.h
phantom_log.h
```

**common/pragmatic_affordance/** (5)

```
object_approach.cpp
object_approach.h
object_approach_geometry.h
pragmatic_affordance.cpp
pragmatic_affordance.h
```

**common/recall_match/** (2)

```
recall_candidates.h
recall_match.h
```

**common/rgb_ingestor/** (4)

```
depth_ingestor.cpp
depth_ingestor.h
rgb_ingestor.cpp
rgb_ingestor.h
```

**common/robot_capability/** (2)

```
robot_capability.cpp
robot_capability.h
```

**common/robot_footprint/** (4)

```
mesh_hull.cpp
mesh_hull.h
robot_footprint.cpp
robot_footprint.h
```

**common/robust_metrics/** (2)

```
robust_metrics.cpp
robust_metrics.h
```

**common/room_resolve/** (4)

```
room_containment.h
room_follow.h
room_resolve.h
room_resolve_rule.h
```

**common/room_transition/** (1)

```
exit_fact.h
```

**common/rt_covariance/** (1)

```
rt_covariance.h
```

**common/rt_query_probe/** (1)

```
rt_query_probe.h
```

**common/rt_twist/** (1)

```
rt_twist.h
```

**common/sensor_calib/** (6)

```
proposal_publisher.cpp
proposal_publisher.h
scan_registration.cpp
scan_registration.h
trajectory_harvester.cpp
trajectory_harvester.h
```

**common/status_stream/** (3)

```
status_sink.h
status_stream.cpp
status_stream.h
```

**common/stream_gate/** (1)

```
stream_gate.h
```

**common/support_bank/** (1)

```
support_bank.h
```

**common/track/** (2)

```
merge_instances.h
merge_sweep.h
```

**common/view_atlas/** (1)

```
view_atlas.h
```

**common/view_field/** (2)

```
phantom_feed.h
view_field.h
```

**hood_concept/src/** (28)

```
epistemic_planner.cpp
epistemic_planner.h
hood_belief.cpp
hood_belief.h
hood_card.h
hood_config.cpp
hood_config.h
hood_dof.h
hood_existence.cpp
hood_existence.h
hood_fitter.cpp
hood_fitter.h
hood_geometry.h
hood_instance.h
hood_lidar_range_channel.cpp
hood_lidar_range_channel.h
hood_model.cpp
hood_model.h
hood_projection.cpp
hood_projection.h
hood_scene_graph.cpp
hood_scene_graph.h
hood_support_bank.h
specificworker.cpp
specificworker.h
specificworker_dashboard.cpp
specificworker_lifecycle.cpp
specificworker_presence.cpp
```

</details>

<details><summary>countable_concept: 188 files</summary>

**common/affordance_manager/** (4)

```
affordance_manager.cpp
affordance_manager.h
affordance_room_scope.h
update_if_live.h
```

**common/affordance_protocol/** (2)

```
affordance_goal_parse.h
affordance_protocol.h
```

**common/agent_exit/** (1)

```
terminal_exit.h
```

**common/agent_presence_coordinator/** (2)

```
agent_presence_coordinator.cpp
agent_presence_coordinator.h
```

**common/agent_presence_monitor/** (2)

```
agent_presence_monitor.cpp
agent_presence_monitor.h
```

**common/agent_state_publisher/** (4)

```
agent_state_publisher.h
agent_status.h
agent_status_overlay.cpp
agent_status_overlay.h
```

**common/ai_belief/** (6)

```
beam_ray.h
beam_room.h
dof_spec.h
lidar_ray_factor.h
persistent_update.h
recursive_laplace.h
```

**common/birth_fragment/** (1)

```
birth_fragment.h
```

**common/birth_surprise/** (3)

```
birth_surprise_log.h
birth_surprise_probe.h
residual_field_reader.h
```

**common/concept_manifest/** (4)

```
back_panel.h
class_prior.h
concept_manifest.h
shape_parts.h
```

**common/concept_presence/** (1)

```
concept_presence.h
```

**common/config_report/** (2)

```
config_read.h
config_report.h
```

**common/contour_edge/** (6)

```
contour_depth_check.cpp
contour_depth_check.h
contour_edge_check.cpp
contour_edge_check.h
contour_edge_project.cpp
contour_edge_project.h
```

**common/dashboard/** (13)

```
belief_card_fill.h
belief_certainty.h
belief_inspector.h
belief_series.h
belief_strip.h
concept_dashboard.h
custom_widget.h
evidence_monitor.h
series_pruner.cpp
series_pruner.h
timeseries_plot.cpp
timeseries_plot.h
window_geometry.h
```

**common/detect_probe/** (1)

```
detect_probe.h
```

**common/detectability/** (1)

```
detectability.h
```

**common/diag_log/** (1)

```
rotating_csv.h
```

**common/epistemic_step/** (1)

```
epistemic_step.h
```

**common/exclusion/** (2)

```
claims.h
exclusion.h
```

**common/existence_belief/** (1)

```
existence_belief.h
```

**common/footprint/** (1)

```
footprint.h
```

**common/graph3d/** (3)

```
graph3d_scene.h
graph3d_scene_builder.cpp
graph3d_scene_builder.h
```

**common/graph_layout/** (1)

```
graph_layout.h
```

**common/graph_provenance/** (1)

```
creation_stamp.h
```

**common/instance_identity/** (1)

```
instance_identity.h
```

**common/instance_tracker/** (2)

```
birth_evidence.h
instance_tracker.h
```

**common/interaction_anchor/** (3)

```
anchor_leaf.h
interaction_anchor.h
relative_pose.h
```

**common/lidar_ingestor/** (2)

```
concept_lidar_ingestor.cpp
concept_lidar_ingestor.h
```

**common/lidar_select/** (1)

```
lidar_select.h
```

**common/mask_ingestor/** (4)

```
mask_azimuth_windows.h
mask_ingestor.cpp
mask_ingestor.h
mask_nodes.h
```

**common/media_transport/** (6)

```
flat_json.h
lidar_plane_reader.cpp
lidar_plane_reader.h
media_transport.cpp
media_transport.h
sensor_model.h
```

**common/nbv/** (4)

```
countable_planner.h
graph_obstacles.h
identified_obstacle.h
viewpoint_score.h
```

**common/obj/** (3)

```
convergence.h
obj_loader.cpp
obj_loader.h
```

**common/object_affordance/** (2)

```
object_affordance.cpp
object_affordance.h
```

**common/object_anchor/** (2)

```
object_anchor_contract.h
ray_anisotropic_cov.h
```

**common/occlusion/** (1)

```
occlusion.h
```

**common/owned_nodes/** (1)

```
owned_nodes.h
```

**common/perception_card/** (3)

```
card.h
card_io.h
term_spec.h
```

**common/peripheral_channel/** (1)

```
peripheral_channel.h
```

**common/phantom_log/** (2)

```
observer_pose.h
phantom_log.h
```

**common/place_memory/** (1)

```
place_map.h
```

**common/pragmatic_affordance/** (5)

```
object_approach.cpp
object_approach.h
object_approach_geometry.h
pragmatic_affordance.cpp
pragmatic_affordance.h
```

**common/recall_match/** (2)

```
recall_candidates.h
recall_match.h
```

**common/rgb_ingestor/** (4)

```
depth_ingestor.cpp
depth_ingestor.h
rgb_ingestor.cpp
rgb_ingestor.h
```

**common/robot_capability/** (2)

```
robot_capability.cpp
robot_capability.h
```

**common/robot_footprint/** (4)

```
mesh_hull.cpp
mesh_hull.h
robot_footprint.cpp
robot_footprint.h
```

**common/robust_metrics/** (2)

```
robust_metrics.cpp
robust_metrics.h
```

**common/room_resolve/** (4)

```
room_containment.h
room_follow.h
room_resolve.h
room_resolve_rule.h
```

**common/room_transition/** (1)

```
exit_fact.h
```

**common/rt_covariance/** (1)

```
rt_covariance.h
```

**common/rt_query_probe/** (1)

```
rt_query_probe.h
```

**common/rt_twist/** (1)

```
rt_twist.h
```

**common/self_body/** (1)

```
self_body.h
```

**common/sensor_calib/** (6)

```
proposal_publisher.cpp
proposal_publisher.h
scan_registration.cpp
scan_registration.h
trajectory_harvester.cpp
trajectory_harvester.h
```

**common/status_stream/** (3)

```
status_sink.h
status_stream.cpp
status_stream.h
```

**common/stream_gate/** (1)

```
stream_gate.h
```

**common/support_bank/** (1)

```
support_bank.h
```

**common/track/** (2)

```
merge_instances.h
merge_sweep.h
```

**common/view_atlas/** (1)

```
view_atlas.h
```

**countable_concept/src/** (43)

```
countable_assets.h
countable_beam_config.h
countable_beam_staging.h
countable_belief.cpp
countable_belief.h
countable_birth_footprint.h
countable_card.h
countable_class_evidence.h
countable_config.cpp
countable_config.h
countable_config_error.h
countable_dof.h
countable_existence.cpp
countable_existence.h
countable_fitter.cpp
countable_fitter.h
countable_geometry.h
countable_instance.h
countable_lidar_range_channel.cpp
countable_lidar_range_channel.h
countable_mask_membership.cpp
countable_mask_membership.h
countable_model.cpp
countable_model.h
countable_projection.cpp
countable_projection.h
countable_scene_graph.cpp
countable_scene_graph.h
countable_sections.h
countable_shape_parts.h
countable_support_bank.h
countable_viewpoint.cpp
countable_viewpoint.h
epistemic_planner.cpp
epistemic_planner.h
rt_memo.h
specificworker.cpp
specificworker.h
specificworker_dashboard.cpp
specificworker_lifecycle.cpp
specificworker_presence.cpp
tick_memo.h
tick_timing.h
```

</details>

### Newly counted files (not scanned in the baseline or addendum 1)

| library (common/) | files | pulled by | rows |
|---|---|---|---|
| affordance_manager | affordance_manager.cpp, affordance_manager.h, affordance_room_scope.h, update_if_live.h | all five | 4 |
| affordance_protocol | affordance_goal_parse.h, affordance_protocol.h | all five | 4 |
| agent_exit | terminal_exit.h | all five | 0 |
| agent_presence_coordinator | agent_presence_coordinator.cpp, agent_presence_coordinator.h | all five | 0 |
| agent_presence_monitor | agent_presence_monitor.cpp, agent_presence_monitor.h | all five | 0 |
| agent_state_publisher | agent_state_publisher.h, agent_status.h, agent_status_overlay.cpp, agent_status_overlay.h | all five | 0 |
| appearance_belief | appearance_belief.h, colour_name.h, colour_terms.h, colour_trace.h | chair, table | 6 |
| concept_manifest | back_panel.h, class_prior.h, concept_manifest.h, shape_parts.h | all five | 1 |
| concept_presence | concept_presence.h | all five | 0 |
| config_report | config_read.h, config_report.h | all five | 0 |
| dashboard | belief_card_fill.h, belief_certainty.h, belief_inspector.h, belief_series.h, belief_strip.h, concept_dashboard.h, custom_widget.h, evidence_monitor.h, series_pruner.cpp, series_pruner.h, timeseries_plot.cpp, timeseries_plot.h, window_geometry.h | all five | 3 |
| detect_probe | detect_probe.h | table, refrigerator, hood, nightstand/bed/toilet | 0 |
| diag_log | rotating_csv.h | all five | 0 |
| epistemic_step | epistemic_step.h | all five | 0 |
| footprint | footprint.h | all five | 0 |
| graph3d | graph3d_scene.h, graph3d_scene_builder.cpp, graph3d_scene_builder.h | all five | 1 |
| graph_layout | graph_layout.h | all five | 0 |
| graph_provenance | creation_stamp.h | all five | 0 |
| instance_identity | instance_identity.h | all five | 0 |
| interaction_anchor | anchor_leaf.h, interaction_anchor.h, relative_pose.h | all five | 3 |
| lidar_ingestor | concept_lidar_ingestor.cpp, concept_lidar_ingestor.h | all five | 0 |
| lidar_select | lidar_select.h | table, refrigerator, hood, nightstand/bed/toilet | 2 |
| mask_ingestor | mask_azimuth_windows.h, mask_ingestor.cpp, mask_ingestor.h, mask_nodes.h | all five | 2 |
| media_transport | flat_json.h, lidar_plane_reader.cpp, lidar_plane_reader.h, media_transport.cpp, media_transport.h, sensor_model.h | all five | 1 |
| nbv | countable_planner.h, graph_obstacles.h, identified_obstacle.h, viewpoint_score.h | all five | 4 |
| obj | convergence.h, obj_loader.cpp, obj_loader.h | all five | 0 |
| object_affordance | object_affordance.cpp, object_affordance.h | all five | 2 |
| object_anchor | object_anchor_contract.h, ray_anisotropic_cov.h | table, refrigerator, hood, nightstand/bed/toilet | 5 |
| owned_nodes | owned_nodes.h | all five | 0 |
| perception_card | card.h, card_io.h, term_spec.h, terms.h | all five | 1 |
| peripheral_channel | peripheral_channel.h | all five | 1 |
| phantom_log | observer_pose.h, phantom_log.h | all five | 0 |
| place_memory | place_map.h | nightstand/bed/toilet | 4 |
| pragmatic_affordance | object_approach.cpp, object_approach.h, object_approach_geometry.h, pragmatic_affordance.cpp, pragmatic_affordance.h | all five | 3 |
| recall_match | identity_link.h, recall_candidates.h, recall_match.h | all five | 1 |
| rgb_ingestor | depth_ingestor.cpp, depth_ingestor.h, rgb_ingestor.cpp, rgb_ingestor.h | all five | 0 |
| robot_capability | robot_capability.cpp, robot_capability.h | all five | 0 |
| robot_footprint | mesh_hull.cpp, mesh_hull.h, robot_footprint.cpp, robot_footprint.h | all five | 2 |
| robust_metrics | robust_metrics.cpp, robust_metrics.h | all five | 0 |
| room_resolve | room_containment.h, room_follow.h, room_resolve.h, room_resolve_rule.h | all five | 0 |
| room_transition | exit_fact.h | all five | 0 |
| rt_covariance | rt_covariance.h | all five | 2 |
| rt_query_probe | rt_query_probe.h | all five | 0 |
| rt_twist | rt_twist.h | all five | 0 |
| self_body | self_body.h | nightstand/bed/toilet | 1 |
| sensor_calib | proposal_publisher.cpp, proposal_publisher.h, scan_registration.cpp, scan_registration.h, trajectory_harvester.cpp, trajectory_harvester.h | all five | 5 |
| status_stream | status_sink.h, status_stream.cpp, status_stream.h | all five | 0 |
| stream_gate | stream_gate.h | all five | 0 |
| support_bank | support_bank.h | all five | 3 |
| thumbs | thumb_select.h, thumb_wire.h | chair | 0 |
| track | merge_instances.h, merge_sweep.h | all five | 0 |
| view_atlas | view_atlas.h | all five | 5 |
| view_field | phantom_feed.h, view_field.h | hood | 2 |

### Rows of the newly counted files

Same columns as the main summary; per library.

| library | class | sensor | patch | switches (patch) | ambiguous (?) | excluded | switches (other) | patch total | ambiguous split | total rows |
|---|---|---|---|---|---|---|---|---|---|---|
| common/affordance_manager | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | excluded 3 | 4 |
| common/affordance_protocol | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | excluded 3 | 4 |
| common/agent_exit | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 |
| common/agent_presence_coordinator | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 |
| common/agent_presence_monitor | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 |
| common/agent_state_publisher | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 |
| common/appearance_belief | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | class 3, patch 2, sensor 1 | 6 |
| common/concept_manifest | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | excluded 1 | 1 |
| common/concept_presence | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 |
| common/config_report | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 |
| common/dashboard | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | - | 3 |
| common/detect_probe | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 |
| common/diag_log | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 |
| common/epistemic_step | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 |
| common/footprint | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 |
| common/graph3d | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | - | 1 |
| common/graph_layout | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 |
| common/graph_provenance | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 |
| common/instance_identity | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 |
| common/interaction_anchor | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | class 1, patch 2 | 3 |
| common/lidar_ingestor | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 |
| common/lidar_select | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 2 | - | 2 |
| common/mask_ingestor | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | patch 1 | 2 |
| common/media_transport | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | - | 1 |
| common/nbv | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | sensor 1, excluded 2 | 4 |
| common/obj | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 |
| common/object_affordance | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | excluded 1 | 2 |
| common/object_anchor | 0 | 0 | 2 | 0 | 3 | 0 | 0 | 2 | patch 1, sensor 2 | 5 |
| common/owned_nodes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 |
| common/perception_card | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | - | 1 |
| common/peripheral_channel | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | - | 1 |
| common/phantom_log | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 |
| common/place_memory | 0 | 0 | 1 | 0 | 3 | 0 | 0 | 1 | class 1, patch 2 | 4 |
| common/pragmatic_affordance | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | excluded 3 | 3 |
| common/recall_match | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | 1 |
| common/rgb_ingestor | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 |
| common/robot_capability | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 |
| common/robot_footprint | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | excluded 1 | 2 |
| common/robust_metrics | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 |
| common/room_resolve | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 |
| common/room_transition | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 |
| common/rt_covariance | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | patch 1 | 2 |
| common/rt_query_probe | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 |
| common/rt_twist | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 |
| common/self_body | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | excluded 1 | 1 |
| common/sensor_calib | 0 | 1 | 0 | 0 | 3 | 1 | 0 | 0 | patch 2, sensor 1 | 5 |
| common/status_stream | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 |
| common/stream_gate | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 |
| common/support_bank | 0 | 0 | 2 | 0 | 0 | 1 | 0 | 2 | - | 3 |
| common/thumbs | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 |
| common/track | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | - | 0 |
| common/view_atlas | 0 | 0 | 0 | 0 | 4 | 1 | 0 | 0 | class 1, patch 2, sensor 1 | 5 |
| common/view_field | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | patch 1, excluded 1 | 2 |

#### common/affordance_manager

| name | value | where | role | label |
|---|---|---|---|---|
| kPosEps / kYawEps | 0.03 / 0.0872665 | common/affordance_manager/affordance_manager.cpp:443 | affordance proposal change dedup (affordance layer, not the estimator) | excluded? |
| kSameSpotM / kSameYawRad | 0.30 / 0.05 | common/affordance_manager/affordance_manager.cpp:815 | "same spot" test of an affordance claim | excluded? |
| kVerdictQuantM | 0.05 | common/affordance_manager/affordance_manager.cpp:1461 | quantisation of the logged verdict | excluded |
| select_lambda_cost_ | 0.2 | common/affordance_manager/affordance_manager.h:398 | cost weight of affordance selection (planner) | excluded? |

#### common/affordance_protocol

| name | value | where | role | label |
|---|---|---|---|---|
| detection_confidence GE | 0.20 | common/affordance_protocol/affordance_protocol.h:228 | contract gate on detector confidence, same literal in 8 contracts | excluded? |
| still(a,b) | 0.10/0.15, 0.12/0.20 | common/affordance_protocol/affordance_protocol.h:229 | contract "robot still" thresholds per class contract | excluded? |
| advance roi_fill | table 0.45, hood 0.30 | common/affordance_protocol/affordance_protocol.h:134 | contract advance threshold on ROI fill | excluded? |
| timeout_ms default | 10000 | common/affordance_protocol/affordance_protocol.h:82 | contract timeout | excluded |

#### common/agent_exit

0 rows (plumbing or numeric guards only; files: terminal_exit.h).

#### common/agent_presence_coordinator

0 rows (plumbing or numeric guards only; files: agent_presence_coordinator.cpp, agent_presence_coordinator.h).

#### common/agent_presence_monitor

0 rows (plumbing or numeric guards only; files: agent_presence_monitor.cpp, agent_presence_monitor.h).

#### common/agent_state_publisher

0 rows (plumbing or numeric guards only; files: agent_state_publisher.h, agent_status.h, agent_status_overlay.cpp, agent_status_overlay.h).

#### common/appearance_belief

| name | value | where | role | label |
|---|---|---|---|---|
| prior_sigma | 0.29 | common/appearance_belief/appearance_belief.h:48 | colour prior std (uniform on [0,1]) | class? |
| drift_sigma_per_s | 0.02 | common/appearance_belief/appearance_belief.h:53 | colour process noise (appearance persistence) | class? |
| max_cells_per_frame | 16 | common/appearance_belief/appearance_belief.h:60 | cap of colour evidence per frame (shared-error style cap) | patch? |
| p_foreign_mask | 0.05 | common/appearance_belief/appearance_belief.h:67 | share of mask pixels from another object (detector error) | sensor? |
| chroma_sigma / lum_sigma | 0.04 / 0.15 | common/appearance_belief/colour_name.h:45 | width of colour-name likelihood | patch? |
| colour prototypes (12) | RGB table | common/appearance_belief/colour_terms.h:19 | RGB prototypes of the colour vocabulary | class? |

#### common/concept_manifest

| name | value | where | role | label |
|---|---|---|---|---|
| band-contains-body fraction | 0.999 | common/concept_manifest/concept_manifest.h:408 | startup validation of a LiDAR band against the body | excluded? |

#### common/concept_presence

0 rows (plumbing or numeric guards only; files: concept_presence.h).

#### common/config_report

0 rows (plumbing or numeric guards only; files: config_read.h, config_report.h).

#### common/dashboard

| name | value | where | role | label |
|---|---|---|---|---|
| window / linger seconds | 60 / 6 / 5 (clamp) | common/dashboard/belief_strip.h:492 | display time windows (display only) | excluded |
| visible_window_s | 60 | common/dashboard/concept_dashboard.h:59 | display window (display only) | excluded |
| plot padding | 0.05, 0.1 | common/dashboard/timeseries_plot.cpp:218 | plot axis padding (display only) | excluded |

#### common/detect_probe

0 rows (plumbing or numeric guards only; files: detect_probe.h).

#### common/diag_log

0 rows (plumbing or numeric guards only; files: rotating_csv.h).

#### common/epistemic_step

0 rows (plumbing or numeric guards only; files: epistemic_step.h).

#### common/footprint

0 rows (plumbing or numeric guards only; files: footprint.h).

#### common/graph3d

| name | value | where | role | label |
|---|---|---|---|---|
| scene colours, radii, gaps | ~45 literals | common/graph3d/graph3d_scene_builder.cpp:23 | viewer colours and stratum geometry (visualisation only) | excluded |

#### common/graph_layout

0 rows (plumbing or numeric guards only; files: graph_layout.h).

#### common/graph_provenance

0 rows (plumbing or numeric guards only; files: creation_stamp.h).

#### common/instance_identity

0 rows (plumbing or numeric guards only; files: instance_identity.h).

#### common/interaction_anchor

| name | value | where | role | label |
|---|---|---|---|---|
| kLeafHalfThickM | 0.02 | common/interaction_anchor/anchor_leaf.h:41 | half thickness of the door leaf slab (shape) | class? |
| kRotScaleSigma | 0.09 | common/interaction_anchor/relative_pose.h:74 | sigma scale of relative rotation | patch? |
| kUnseenSigmaXYM / kUnseenSigmaYawRad | 0.08 / 0.05 | common/interaction_anchor/relative_pose.h:119 | sigma assigned to an unseen anchor | patch? |

#### common/lidar_ingestor

0 rows (plumbing or numeric guards only; files: concept_lidar_ingestor.cpp, concept_lidar_ingestor.h).

#### common/lidar_select

| name | value | where | role | label |
|---|---|---|---|---|
| coverage_n0 default | 60 | common/lidar_select/lidar_select.h:58 | ray count for full weight (same as LidarCoverageN0) | patch |
| robust_c_m default | 0.10 | common/lidar_select/lidar_select.h:60 | Cauchy scale default (class config LidarRobustCM is 0.05) | patch |

#### common/mask_ingestor

| name | value | where | role | label |
|---|---|---|---|---|
| semantic class_id threshold | 1000 | common/mask_ingestor/mask_ingestor.h:136 | id convention for semantic masks | excluded |
| pose_extrap_max_dt_s | 0.2 | common/mask_ingestor/mask_ingestor.h:278 | max extrapolation of the camera pose to the mask stamp | patch? |

#### common/media_transport

| name | value | where | role | label |
|---|---|---|---|---|
| deg2rad | pi/180 | common/media_transport/sensor_model.h:100 | unit conversion | excluded |

#### common/nbv

| name | value | where | role | label |
|---|---|---|---|---|
| default hfov | 70 deg | common/nbv/viewpoint_score.h:78 | default camera field of view (overwritten from the sensor model) | sensor? |
| fov validity bounds / clamps | 0.05, 3.0, 0.1 | common/nbv/viewpoint_score.h:83 | validity guards on FOV (planner) | excluded? |
| Gauss-Hermite nodes and weights | 5 nodes | common/nbv/viewpoint_score.h:253 | quadrature | excluded |
| planner floors and caps | 0.05 / 200 / 8.0 | common/nbv/viewpoint_score.h:677 | depth floor, search caps, collision floor of the viewpoint planner | excluded? |

#### common/obj

0 rows (plumbing or numeric guards only; files: convergence.h, obj_loader.cpp, obj_loader.h).

#### common/object_affordance

| name | value | where | role | label |
|---|---|---|---|---|
| contract gate | 0.20 / 0.15 | common/object_affordance/object_affordance.cpp:339 | default contract gate and still threshold of the affordance | excluded? |
| gain change dedup | 0.1 | common/object_affordance/object_affordance.cpp:296 | republish only if gain changed more than this | excluded |

#### common/object_anchor

| name | value | where | role | label |
|---|---|---|---|---|
| sigma_perp | 0.04 | common/object_anchor/ray_anisotropic_cov.h:29 | across-ray sigma of the object anchor | sensor? |
| c_bias | 0.7 | common/object_anchor/ray_anisotropic_cov.h:30 | bias factor of the along-ray sigma | patch? |
| depth_alpha | 0.01 | common/object_anchor/ray_anisotropic_cov.h:31 | depth sigma growth | sensor? |
| obliquity_floor | 0.15 | common/object_anchor/ray_anisotropic_cov.h:32 | floor of obliquity | patch |
| sigma_along_min | 0.06 | common/object_anchor/ray_anisotropic_cov.h:33 | floor of along-ray sigma | patch |

#### common/owned_nodes

0 rows (plumbing or numeric guards only; files: owned_nodes.h).

#### common/perception_card

| name | value | where | role | label |
|---|---|---|---|---|
| division guards | 1e-3 / 1e-6 | common/perception_card/terms.h:57 | numerical guards; terms take their constants (ref_m, power, sigma0, k) from each class card, not from this library | excluded |

#### common/peripheral_channel

| name | value | where | role | label |
|---|---|---|---|---|
| angular_margin_rad | 0.10 | common/peripheral_channel/peripheral_channel.h:95 | angular margin of a peripheral match | patch |

#### common/phantom_log

0 rows (plumbing or numeric guards only; files: observer_pose.h, phantom_log.h).

#### common/place_memory

| name | value | where | role | label |
|---|---|---|---|---|
| pool_p | 3.0 | common/place_memory/place_map.h:173 | pooling exponent of place scores | patch? |
| tau / decay_a / decay_b / decay_r_m | 0.0552 / 0.770 / 0.190 / 0.580 | common/place_memory/place_map.h:235 | fitted memory decay of a place (persistence) | class? |
| scene_depth_m | 4.0 | common/place_memory/place_map.h:240 | nominal scene depth | patch? |
| max_pos_sigma_m | 6.0 | common/place_memory/place_map.h:253 | position sigma above which a place is not used (gate) | patch |

#### common/pragmatic_affordance

| name | value | where | role | label |
|---|---|---|---|---|
| turn_radius_m / value / arrival_m | 0.398 / 0.3 / 0.25 | common/pragmatic_affordance/object_approach.h:56 | approach affordance parameters (planner/affordance) | excluded? |
| clearance_m | 0.398 | common/pragmatic_affordance/object_approach_geometry.h:56 | approach geometry clearance (robot property) | excluded? |
| offer_prob / withdraw_prob | 0.60 / 0.35 | common/pragmatic_affordance/pragmatic_affordance.h:123 | hysteresis of offering an affordance | excluded? |

#### common/recall_match

| name | value | where | role | label |
|---|---|---|---|---|
| LinkParams pi_move / pi_gone / lambda_new | 0.10 each | common/recall_match/identity_link.h:51 | permanence rates of the identity link (same as Tracker.Link* config) | class |

#### common/rgb_ingestor

0 rows (plumbing or numeric guards only; files: depth_ingestor.cpp, depth_ingestor.h, rgb_ingestor.cpp, rgb_ingestor.h).

#### common/robot_capability

0 rows (plumbing or numeric guards only; files: robot_capability.cpp, robot_capability.h).

#### common/robot_footprint

| name | value | where | role | label |
|---|---|---|---|---|
| base polygon | 0.284/0.10/0.22/0.23 m | common/robot_footprint/robot_footprint.cpp:26 | robot body outline (robot property; not the object estimator) | excluded? |
| footprint validation bounds | 0.15..1.00, 0.05, 0.10 | common/robot_footprint/robot_footprint.cpp:122 | sanity checks on a loaded footprint | excluded |

#### common/robust_metrics

0 rows (plumbing or numeric guards only; files: robust_metrics.cpp, robust_metrics.h).

#### common/room_resolve

0 rows (plumbing or numeric guards only; files: room_containment.h, room_follow.h, room_resolve.h, room_resolve_rule.h).

#### common/room_transition

0 rows (plumbing or numeric guards only; files: exit_fact.h).

#### common/rt_covariance

| name | value | where | role | label |
|---|---|---|---|---|
| kFlatRollPitchVar | 5e-4 | common/rt_covariance/rt_covariance.h:45 | variance of roll/pitch of a floor-standing object | patch? |
| change_frac / min_move_m | 0.05 / 0.05 | common/rt_covariance/rt_covariance.h:87 | publication dedup | excluded |

#### common/rt_query_probe

0 rows (plumbing or numeric guards only; files: rt_query_probe.h).

#### common/rt_twist

0 rows (plumbing or numeric guards only; files: rt_twist.h).

#### common/self_body

| name | value | where | role | label |
|---|---|---|---|---|
| far cull | 6 sigma | common/self_body/self_body.h:94 | distance beyond which self-body term is skipped (cost cull) | excluded? |

#### common/sensor_calib

| name | value | where | role | label |
|---|---|---|---|---|
| move_sigma | 0.1 | common/sensor_calib/proposal_publisher.h:75 | sigma of a published calibration proposal | patch? |
| scan_registration shrink | 0.7 / 0.25 | common/sensor_calib/scan_registration.cpp:157 | sigma shrink and weight floor in the registration loop | patch? |
| ScanReg sigma_m / pi_inlier0 | 0.01 / 0.8 | common/sensor_calib/scan_registration.h:22 | LiDAR registration range sigma / inlier prior | sensor? |
| trajectory harvester sigmas | 0.05,0.0175,0.01,0.002,0.005,0.2,0.01,0.01,0.002 | common/sensor_calib/trajectory_harvester.h:85 | odometry, wheel, gyro and registration noise of the self-calibration | sensor |
| ring_span_s / history_s | 30 / 90 | common/sensor_calib/trajectory_harvester.h:96 | history windows (engineering) | excluded |

#### common/status_stream

0 rows (plumbing or numeric guards only; files: status_sink.h, status_stream.cpp, status_stream.h).

#### common/stream_gate

0 rows (plumbing or numeric guards only; files: stream_gate.h).

#### common/support_bank

| name | value | where | role | label |
|---|---|---|---|---|
| radius_margin_m / height_margin_m | 0.50 / 0.25 | common/support_bank/support_bank.h:65 | ownership margins (class config SupportSelect*Margin overrides) | patch |
| below_m | 0.05 | common/support_bank/support_bank.h:74 | margin below the floor | patch |
| quantization_m | 0.02 | common/support_bank/support_bank.h:75 | dedup grid | excluded |

#### common/thumbs

0 rows (plumbing or numeric guards only; files: thumb_select.h, thumb_wire.h).

#### common/track

0 rows (plumbing or numeric guards only; files: merge_instances.h, merge_sweep.h).

#### common/view_atlas

| name | value | where | role | label |
|---|---|---|---|---|
| ell_rad / repeat_ell_rad / repeat_ell_logr | 0.90 / 0.10 / 0.20 | common/view_atlas/view_atlas.h:41 | kernel length scales of the view atlas | patch? |
| mu_own / mu_bg / cov_own / cov_bg / cov_n | 0.96,0.97 / 0.87,0.50 / 9e-4 ... | common/view_atlas/view_atlas.h:46 | measured appearance statistics of own vs background | sensor? |
| lambda_nats | 0.05 | common/view_atlas/view_atlas.h:51 | weight of the atlas term | patch? |
| max_rate_hz | 4.0 | common/view_atlas/view_atlas.h:52 | update rate cap | excluded |
| pi_move | 0.2 | common/view_atlas/view_atlas.h:53 | P(object moved), permanence | class? |

#### common/view_field

| name | value | where | role | label |
|---|---|---|---|---|
| cell_m | 0.25 | common/view_field/view_field.h:71 | grid resolution | excluded? |
| prior_strength / forget | 20 / 0.995 | common/view_field/view_field.h:78 | prior pseudo-counts and forgetting of the view field | patch? |

#### Own-directory rows added (files not in the earlier lists)

| class | name | value | where | role | label |
|---|---|---|---|---|---|
| chair | DOF sigma_star | 0.07 m,0.07 m,0.07 rad | chair_concept/src/chair_dof.h:42 | target posterior sigma per DOF: when an object counts as resolved (epistemic demand) | patch? |
| table | DOF sigma_star | 0.02 m (cx,cy,H,w,h), 0.05 rad (yaw) | table_concept/src/table_dof.h:25 | target posterior sigma per DOF (epistemic demand) | patch? |
| table | model size floors | w,h>=0.1; H>=0.05+top; leg>=0.05 | table_concept/src/table_model.cpp:139 | clamps of the state in the model | patch? |
| refrigerator | DOF sigma_star | 0.02 m, 0.05 rad (copied from table) | refrigerator_concept/src/refrigerator_dof.h:21 | target posterior sigma per DOF (epistemic demand) | patch? |
| refrigerator | model size floors | w,h>=0.1; H>=0.05+top; leg>=0.05 | refrigerator_concept/src/refrigerator_model.cpp:108 | clamps of the state in the model | patch? |
| hood | model size floors | w,h>=0.1; extent>=0.05 | hood_concept/src/hood_model.cpp:113 | clamps of the state in the model | patch? |

### Updated combined totals per class under the build rule

Class load = own row (baseline) + own-directory rows added above + engine row (nightstand/bed/toilet: countable-engine src) + every common/ file the target pulls (ai_belief, the seven libraries and the newly counted files), counted per file. Cells are counts of rows. `countable` is shared by nightstand, bed and toilet; each adds its own config row.

| class | class | sensor | patch | switches (patch) | ambiguous (?) | excluded | switches (other) | patch total | ambiguous split | total rows |
|---|---|---|---|---|---|---|---|---|---|---|
| chair | 6 | 9 | 101 | 9 | 69 | 57 | 6 | 110 | class 8, patch 26, sensor 13, excluded 22 | 257 |
| table | 5 | 16 | 107 | 13 | 89 | 71 | 3 | 120 | class 11, patch 32, sensor 26, excluded 20 | 304 |
| refrigerator | 9 | 16 | 121 | 8 | 96 | 73 | 4 | 129 | class 13, patch 38, sensor 25, excluded 20 | 327 |
| hood | 11 | 13 | 120 | 8 | 97 | 73 | 4 | 128 | class 13, patch 38, sensor 25, excluded 21 | 326 |
| nightstand | 13 | 20 | 120 | 11 | 100 | 80 | 4 | 131 | class 15, patch 38, sensor 27, excluded 20 | 348 |
| bed | 13 | 20 | 120 | 11 | 100 | 80 | 4 | 131 | class 15, patch 38, sensor 27, excluded 20 | 348 |
| toilet | 13 | 20 | 120 | 11 | 100 | 80 | 4 | 131 | class 15, patch 38, sensor 27, excluded 20 | 348 |

Breakdown (patch total / ambiguous):

| class | own row (+own dir) | countable engine src | common/ (all pulled files) | combined patch total | combined ambiguous | baseline row alone | addendum 1 combined |
|---|---|---|---|---|---|---|---|
| chair | 91 / 27 | 0 / 0 | 19 / 42 | 110 | 69 | 91 / 26 | 107 / 38 |
| table | 95 / 37 | 0 / 0 | 25 / 52 | 120 | 89 | 95 / 35 | 113 / 50 |
| refrigerator | 104 / 51 | 0 / 0 | 25 / 45 | 129 | 96 | 104 / 49 | 122 / 64 |
| hood | 103 / 50 | 0 / 0 | 25 / 47 | 128 | 97 | 103 / 49 | 121 / 64 |
| nightstand | 95 / 32 | 10 / 19 | 26 / 49 | 131 | 100 | 95 / 32 | 123 / 69 |
| bed | 95 / 32 | 10 / 19 | 26 / 49 | 131 | 100 | 95 / 32 | 123 / 69 |
| toilet | 95 / 32 | 10 / 19 | 26 / 49 | 131 | 100 | 95 / 32 | 123 / 69 |

The growth from addendum 1 comes from the newly counted common/ files. Most are labelled `excluded` or `?` (affordance, planner, display, calibration code on the object path); the patch total moves little. Which of the `?` rows are patch is left to the blind referee.



---

## Addendum 3: the blind referee's labels replace the `?` rows (2026-11-27)

- Labels file: `registry/referee_labels_2026-11-27.tsv` in the AuRo paper folder (read-only), sha256
  `495b5e83e54a1bc86c3c334890bbd6c09ed5a41917384c50806326a09a64cb09`. 340 rows R001–R340, labelled by a blind referee who
  read the code at 80738627 and nothing under docs/superpowers. Its labels: 177 patch, 70 class, 67 sensor, 26 excluded.
- Row map: `registry/counter_labels_2026-11-27.tsv`, sha256 `f846ff04a0cb838035e758115e95a8555b4dca7c4eede87e43faa1b15c409c8a`
  (id, section, key, this file's best label).
- Method, mechanical, with no relabelling: `docs/superpowers/results/2026-11-27-addendum3-recount.py <this file> <counter
  labels> <referee labels> <out.json>`. It reads every `?` row of this file in document order up to "After part 1". All
  340 rows align with the counter_labels ids on both section and best label (0 mismatches). Each row is charged to a
  class with the addendum 2 rule: own sections and own-directory rows to that class; countable-engine rows to
  nightstand, bed and toilet; a common/ row to every target whose build file list (addendum 2, "Per-target file
  lists") contains the file in its `where` column.
- Check: with the best labels, the script reproduces addendum 2's "ambiguous split" column exactly for all seven
  classes. Every other column is unchanged from addendum 2; only the ambiguous column is redistributed, so the total
  rows are the same.

### Combined totals per class (build rule, referee labels in place of `?`)

| class | class | sensor | patch | switches (patch) | excluded | switches (other) | patch total | addendum 2 patch total (`?` not counted) | referee split of the `?` rows | total rows |
|---|---|---|---|---|---|---|---|---|---|---|
| chair | 12 | 18 | 142 | 9 | 70 | 6 | 151 | 110 | class 6, patch 41, sensor 9, excluded 13 | 257 |
| table | 15 | 34 | 156 | 13 | 83 | 3 | 169 | 120 | class 10, patch 49, sensor 18, excluded 12 | 304 |
| refrigerator | 25 | 33 | 172 | 8 | 85 | 4 | 180 | 129 | class 16, patch 51, sensor 17, excluded 12 | 327 |
| hood | 27 | 30 | 172 | 8 | 85 | 4 | 180 | 128 | class 16, patch 52, sensor 17, excluded 12 | 326 |
| nightstand | 30 | 40 | 171 | 11 | 92 | 4 | 182 | 131 | class 17, patch 51, sensor 20, excluded 12 | 348 |
| bed | 30 | 40 | 171 | 11 | 92 | 4 | 182 | 131 | class 17, patch 51, sensor 20, excluded 12 | 348 |
| toilet | 30 | 40 | 171 | 11 | 92 | 4 | 182 | 131 | class 17, patch 51, sensor 20, excluded 12 | 348 |

### Counting rules for the after-count (owner, 2026-11-27)

The referee's rules 0–8 (the header of the labels file) are now counting rules, alongside the definition, for the
after-count:
- label a row by the meaning of its key, not its present value;
- a dead constant (one no code reads) is excluded;
- the pose process noise is patch;
- inflated sensor noise and floors are patch;
- a class prior times a free gain is patch;
- a composite row keeps its id and its dominant label.

The "After part 1" section below was counted before these labels existed. Its `?` rows still need the same treatment
before before/after totals are compared.


---

## After part 1 (shared engine, Task G1x, 2026-11-26)

### How this count was made

- **Tree.** Counted on the tree of `cf65a471` (`5c0029ab522691fb4747d2e7215493162f13f0b0`), branch feat/shared-engine, the end
  of K3. That is the code the batch builds (see `2026-11-26-part1-grading.md`).
- **Rule.** Addendum 2's rule, verbatim, on the **same five targets**: `chair_concept`, `table_concept`,
  `refrigerator_concept`, `hood_concept` and `countable_concept`.
  - For each target: `cmake -S <agent> -B <scratch>/deps/<agent> -DCMAKE_EXPORT_COMPILE_COMMANDS=ON` (nothing built).
  - Every translation unit was rerun with `-MM -MG`, 8 at a time. The lists were filtered to the worktree's `common/` and
    the agent's own directory. Generated files were dropped (38 per target, as before), and so were the 13 `mocs_compilation`
    units per target that return nothing.
- **Labels and definition.** Unchanged. An existing row is relabelled only where its role in the code changed; each such row
  is listed below as "moved".
- **Incremental, not a re-inventory.** Files that are byte-identical since `535d57ec` (the tree addendum 2 counted) keep their
  rows. Files that changed or entered the scope were rescanned on their diff `535d57ec..cf65a471`:
  - added lines: every new named constant, and every numeric literal outside comments, logs and tests, read in context;
  - removed lines: checked for baseline rows that disappeared.

  Config files were diffed key by key against the baseline rows, by name. The loader's key set was diffed against the toml
  (hidden defaults).
- **Limit.** As before, the last step is manual, so the src rows are a careful sample, not a proof. The config rows are
  exact.

### Files pulled in per target

| target | TU | own-dir files (was) | common/ files (was) | added since addendum 2 | dropped |
|---|---|---|---|---|---|
| chair_concept | 68 | 27 (27) | 141 (138) | `concept_manifest/manifest_toml.h`, `standpoint/residual_risk.h`, `standpoint/standpoint_oracle.h` | — |
| table_concept | 71 | 30 (30) | 144 (141) | the same three | — |
| refrigerator_concept | 71 | 28 (28) | 151 (139) | the same three, plus `ai_belief/{beam_channel,beam_room,box_engine_model,foreign_box,offer_view,persistent_engine,persistent_update,viewpoint}.h` and `place_memory/place_map.h` (T10a's legacy-side adapter) | — |
| hood_concept | 71 | 28 (28) | 153 (141) | the same as refrigerator | — |
| countable_concept | 73 | 50 (43) | 164 (145) | the three, plus `ai_belief/{beam_channel,foreign_box,offer_view,parts_geometry,persistent_engine,viewpoint}.h`, `appearance_belief/{appearance_belief,colour_name,colour_terms}.h`, `concept_manifest/{layer_spread,permanence,wall_standoff}.h`, `epistemic_step/offer_cache.h`, `recall_match/identity_link.h`, `thumbs/{thumb_select,thumb_wire}.h`; own: `countable_{birth_seed,gain_query,latent_inputs,neighbour_size,replay,size_std,vertical_span}.h`, `specificworker_consumers.cpp` | `countable_viewpoint.cpp` (moved into `common/ai_belief/viewpoint.h`) |

Two of the three files every target gained are not new constants. `common/standpoint/residual_risk.h` and the GridPlanner
kernels in `standpoint_oracle.h` were moved by K3 from `controller/`, which is outside every concept target's scope. They are
in the count now because they sit in `common/`.

### Which target each class runs (this changed)

| class | baseline binary | binary now (as launched by `concept_agents.sh`) |
|---|---|---|
| nightstand, bed, toilet | countable_concept | countable_concept (own `etc/config.toml`) |
| refrigerator, hood | refrigerator_concept, hood_concept | **countable_concept** (`<cls>/countable/<cls>` + `etc/countable.toml`, K1). The legacy binary is still built and selectable (`LEGACY_AGENTS`) |
| chair | chair_concept | **countable_concept** (`chair_concept/countable/chair_concept` + `etc/countable.toml`, K2b). Legacy still built |
| table | table_concept | **cut from part 1** (owner 11-26). The batch runs the legacy `table_concept`. `concept_agents.sh` would otherwise launch the K2c countable link, so the batch must set `LEGACY_AGENTS="table_concept"` |

So for six classes the "after" row is a countable row, whichever target the class started on.

### Disposition of the registered-prediction rows (spec "Registered prediction for test (c)")

There are 51 registered rows on the countable path: 47 config rows of the nightstand / bed / toilet configs (the K1/K2b
configs carry the same keys), and 4 countable src rows. The same 51 apply to refrigerator, hood and chair, which now run
countable_concept.

| row(s) | spec item | disposition on countable_concept (6 classes) | evidence |
|---|---|---|---|
| `AI2CommonModePosStd`, `AI2CommonModeSizeStd`, `AI2CommonModeYawStd` | G1 | **moved**, patch → sensor? (counted ambiguous, best label sensor) | On the persistent path the camera's shared-error marginal on cx, cy, H, w, h is the MEASURED `kZedYoloBoundaryStdM` 0.0289 m (`countable_latent_inputs.h:38`, F1.6). The three keys remain the LiDAR latent's marginals and the camera's yaw marginal, still at the old tuned values (0.03 / 0.02 / 0.03), unmeasured (`:51-53`). The role is the sensor's; the value is not measured. |
| `AI2RangeNoiseYawPerM`, `AI2RangeNoiseSizePerM` | G1 | **kept** (toml key deleted in T5, loader default 0.03 / 0.08 still read, `countable_config.cpp:334-337`; used on the flag-off path only, `countable_fitter.cpp:1362-1363`) | Under the build rule a hidden default is a row. It is inert on the graded path, but it is not removed. |
| `MotionCmPosGain`, `MotionCmSizeGain`, `MotionCmYawGain` | G1 | **kept** (fresh ego-motion share) | T5 ruling: τ was not measured (progress.md:33, :48) |
| `AI2PeriphRef`, `AI2AngLeverM`, `AI2MotionRefMps` | G1 | **kept** | Taken out of the persistent sig2_cam in T5, but still read by `motion_magnitude` / `periphery_penalty` (`countable_fitter.cpp:601, :608`), which feed the fresh share and the confirm-only gate |
| `AI2MotionConfirmOnly`, `AI2StillLinMps`, `AI2StillAngRadps`, `AI2StillDotd`, `AI2MovingUpdateCenterRadius` | G1 | **kept** | The F1.8 gate audit kept the confirm-only gate, counted as a patch (`countable_fitter.cpp:1395-1404`; progress.md:112) |
| `kFrontClamp`, `front_view_budget_`, `ShapeEvidenceClamp` | G1 | **kept** | `countable_belief.cpp:1380`, `countable_belief.h:1019` |
| `AI2DepthUnobsPrecision`, `AI2DepthObsBandM` | G2 | **kept** | F1.5: the wall boost was dropped on the persistent path, but the footprint share was KEPT on replay evidence (`countable_belief.cpp:757-773`). See the closed-list check, V2. |
| `FreeSpacePrecision`, `CoveragePrecision`, `CoverageRobustCM`, `LidarCoverageAngPower`, `LidarCoverageN0` | G2 | **kept** | still read: `countable_lidar_range_channel.cpp:79, 109, 172`; `countable_fitter.cpp:1272-1274` |
| `FootprintMomentCompletenessGain`, `FootprintMomentMinCompleteness` | G2 | **kept** | `countable_fitter.cpp:1276-1277` |
| `WallNoCrossPrecision`, `WallNoCrossMarginM` | G2 | **kept** (in scope, read by the legacy path) | The persistent no-cross was deleted by the F1 spec amendment. The containment knowledge moved into the stand-off prior and the wall latent, which are class and world rows. The keys stay, and so do the rows. |
| `AI2WallExplainFrac`, `AI2WallExplainSigmaM` | G2 | **kept** | the wall component of the persistent point mixture (`countable_belief.cpp:744-754`) |
| `AI2WallPrecision`, `AI2WallParallelPrecision`, `AI2WallReachM` | G2 (to sensor or class) | **kept** (label patch? unchanged) | On the persistent path, with a stand-off declared (every countable class except the c2v table), `AI2WallPrecision` is only the on/off switch: λ = 1/standoff_std² from the manifest (`countable_belief.cpp:860-866`). `AI2WallParallelPrecision` is only a fallback: λ = 1/(σ_wall/L)² from the room's published covariance (`:416-417`). The legacy path uses both values as before, so the rows stay. |
| `AI2TopNoFloatPrecision`, `AI2TopNoFloatMarginM`, `AI2TopOversegSigmaPerM` | G3 | **kept** | `countable_belief.cpp:609` and every camera frame (closed-list V1) |
| `AI2ClutterFrac`, `AI2ClutterScaleM` | G3 | **kept** | `countable_belief.cpp:747-752` |
| `AI2SigmaBaseM` | G3 | **kept** (sensor?) | still the per-point R base. The measured boundary error went into the camera LATENT, not into R. |
| `SdfThresholdForStorage` | G3 | **kept** | `countable_fitter.cpp:986` |
| `LidarPrecision` | G3 (to sensor σ) | **kept** (sensor?) | still sets the beam density's p0 (`countable_fitter.cpp:78`). σ is the sensor's `kHeliosRangeSigmaM`, but the weight remains. |
| `ObliquityMomentGain`, `FootprintMomentPrecision`, `FootprintMomentRangePerM`, `FootprintMomentMotionGain` | G4 | **kept** | `countable_fitter.cpp:1700-1714`; whether the graded path reads them was not re-verified |
| `kObliquityYawGain`, obliquity cos clamp | G4 | **kept** (flag-off path only since T5) | `countable_fitter.cpp:1686-1688` |
| `PlausHeightPriorGain` | G4 | **kept** | the short-height prior, at fixed precision, on every persistent update (closed-list V3) |
| `EpistemicCooldownCycles` | G5 | **kept** (excluded?) | — |
| G6 "no constant beyond the robot footprint" | G6 | **fails by the letter** | K3 brought into every target's scope: the 95 % room-mask guard (a flagged threshold); residual_risk's ½ occupancy boundary and its Beta floor; the no-belief ⇒ hard convention (see "Rows added"). The margin and lattice cell are the caller's (countable: margin 0, residual's cell). |

**Totals on countable_concept: removed 0, moved 3, kept 48 of 51.**
- Of the 48 kept, 6 are inert on the graded path: the two RangeNoise defaults, the two WallNoCross keys, kObliquityYawGain
  and its cos clamp.
- 2 more changed role on the graded path without leaving the scope: AI2WallPrecision and AI2WallParallelPrecision.
- Nothing is truly removed. The nearest case is not a registered row: the literal `kRobotRadiusM` 0.30
  (`epistemic_planner.cpp:78`) is now read from the robot's footprint (`rc::RobotFootprint::shadow()`,
  `epistemic_planner.cpp:83`). Its baseline row "planner kRobotRadiusM / kRef" (excluded?) stays, because `kRef` 0.10 remains
  (`:218`).

**Legacy targets** (`chair_concept`, `table_concept`, and `refrigerator_concept` / `hood_concept` behind `LEGACY_AGENTS`):
their own src and config are unchanged since the baseline (T10a's fridge and hood adapters are bit-exact behind a switch),
so every registered row there is **kept**.

**Chair:** the chair's own legacy rows left the chair's graded scope when it moved to countable_concept. These are:
- the `Fixation*` gates (10 rows);
- `kFlipClamp`, `AI2ViewBudget`, `AI2ModeSatBackPts`, `AI2ClutterStructureGain`;
- `ChairModel.AI2ObliquityYawGain`.

But the chair now carries the countable rows instead, including some it never had:
- `AI2DepthUnobs*`, `AI2TopNoFloat*`, `WallNoCross*`, `PlausHeightPriorGain`, `kFrontClamp`;
- `AI2MotionConfirmOnly = true` (`chair_concept/etc/countable.toml:230`; it was `false` on the legacy chair).

The net effect is in the totals below.

### Rows added by part 1 (by honest category)

**Flagged thresholds**, each counted as patch:
| row | value | where | note |
|---|---|---|---|
| F1r9 cap switch: `gn_tol_sigma`, which now decides absorbed (Laplace Σ) or not absorbed (P̄) | 0.1 | `countable_belief.h:332` → `persistent_engine.h:98, :1039` | **moved**, excluded → patch: the baseline row "gn_tol_sigma, GN convergence tolerance, excluded" got an estimation role in F1 round 9. Flagged ⚠ THRESHOLD, justified in the ledger (progress.md:245). `iters_max` 25 (`persistent_engine.h:95`), which makes an update cap, stays excluded. |
| room-mask guard | ≥ 95 % of the lattice outside ⇒ polygon discarded | `common/standpoint/standpoint_oracle.h:219` | **moved into scope** by K3 (it was `GridPlanner::set_world` in controller/). Flagged ⚠ THRESHOLD, not justified by a model term (`:213`). Counted in all five targets. |

**Unflagged threshold found by this count** (patch, NEW):
| row | value | where | note |
|---|---|---|---|
| wall-identity gate | χ² d² ≤ 9.0 on (offset, angle) | `common/ai_belief/persistent_engine.h:286` (`same_wall`, F1 round 6 "geometric wall identity") | A hard 3σ gate that decides whether a staged wall is the same physical wall: same latent and priors not re-entered, or a new source. It is not flagged in the code. **It should be**: flag it, or replace it by a model term. |

**Other new or newly counted patch rows:**
| row | value | where | label |
|---|---|---|---|
| Σ capped at the class prior on every predict | Σ_ii ≤ Σ₀,ii | `persistent_engine.h:1514-1520` | patch: pre-existing (moved from `countable_belief.cpp:856`); the baseline literal scan missed it; **first counted now**. Closed-list V5. |
| `Tracker.GhostEvidenceGain` | 0.15 | `countable_config.cpp:430` (all countable classes; toml in chair) | patch: the legacy chair's `Existence.EvidenceGain`, ported unchanged (K2b) |
| ghost `ExistenceBelief` L_max copy | 4.0 | `countable_concept/src/specificworker.h:250` | patch: a second literal copy of the log-odds cap (K2b) |
| engine volatility memory | 0.995 | `persistent_engine.h:116` | patch?: a second copy of `kLambda` (the countable copy stays, `countable_belief.cpp:268`) |
| birth seed-height quantile | 0.95 of the box-explained cloud | `countable_belief.cpp:829` (F1 round 6) | patch?: an uncommented quantile (F1 minor, progress.md:127) |
| residual_risk occupancy boundary | Pr(θ_c < ½) | `common/standpoint/residual_risk.h:126` | patch?: moved into scope by K3; "free" is defined by a ½ cut on the occupancy rate |
| residual_risk Beta floor | α, β ≥ 0.5 | `residual_risk.h:114-115` | patch?: moved into scope by K3 |
| no belief ⇒ hard | a residual cell with no usable Beta blocks | `residual_risk.h` / `standpoint_oracle.h:14` | patch?: the inherited conservative convention (spec K3 amendment) |
| beam gap-ray count | P(inside) > 0.99, 5c behind | `common/ai_belief/beam_channel.h:83` | patch?: **moved** from `countable_belief.cpp:1013` (baseline row "wall-hit count margin"); not new |

**Measured sensor rows:**
| row | value | where | note |
|---|---|---|---|
| ZED YOLO-seg boundary error, metric | 0.0289 m | `countable_latent_inputs.h:38` | NEW, sensor, measured (step 0, refrigerator, n = 66). It replaces AI2CommonModePos/Size on the camera latent (see "moved"). |
| `Model.MaskPixelErrorRate` (ε_px) | ns 0.094, bed 0.085, toilet 0.001, fridge 0.066, hood 0.066, chair 0.066, table-countable 0.118 | each class config, `:79`–`:94` | NOT a new row (it existed as sensor). Measured for fridge (0.066) and table (0.118, an upper bound). **Borrowed** from the fridge for hood and chair, which were never measured: those rows are labelled sensor here, but their values are not measurements. The toilet keeps the owner-decided 0.001 against a measured 0.009 (T6 ruling). |
| offer staging defaults | ρ_max 6.0, σ_s 0.01, range_max 6.0 | `common/ai_belief/offer_view.h:74-75` | sensor? (copies of the BeamRays defaults), 2 rows |

**Class rows:**
| row | value | where |
|---|---|---|
| `Model.AI2PriorDepthStd` | −1 (off) or the class's depth std | `countable_config.cpp:141` (K1) |
| `Tracker.LinkPiMove` / `LinkPiGone` / `LinkLambdaNew` | 0.10 each | `countable_config.cpp:427-429` (K2b; the loader's own text: "a GUESS"). Labelled class to match the baseline's `identity_link.h` row. |
| `recall_match/identity_link.h` LinkParams | 0.10 each | newly pulled by countable (the baseline's row) |
| parts panel thickness default | 0.10 m | `common/ai_belief/parts_geometry.h:47` (a copy of headboard_t_m's default) |

**Cited and uncited class constants in the manifests.** The baseline rule leaves manifests uncounted, as class knowledge. They
are listed here so the referee can see what part 1 added:
- **wall stand-offs:** `standoff_m` / `standoff_std_m` for nightstand, bed, toilet, refrigerator, hood and chair: 12 numbers.
  - cited: fridge (Bosch, Liebherr); toilet (weak: TOTO, forum reports);
  - uncited convention: nightstand, bed, hood, chair.
- **`[prior.permanence]`** (move rate, move scale m / rad): 7 classes × 3 numbers. Every one is an uncited convention.
- **layer spreads Δ (G4):** in each manifest's `[shape]` (toilet Δ_H = 0; hood Δ from a cited canopy).
- **proportion ranges:** toilet bowl_top [0.447, 0.675] (ABA §604.4, Gerber/Kohler); chair seat_top [0.377, 0.667]; table
  top_under [0.883, 0.975]. All cited.
- **K2b/K2c class priors** from cited uniform ranges: chair w, h, H; table w, h, H. Some of the K1 hood priors are marked
  uncited.

**Excluded rows added:**
- engine: GN cap, restart tolerances, Nielsen and LM constants, offer path chords, finite-difference steps
  (`persistent_engine.h`);
- guards: FOV clamp, degenerate-geometry guards (`offer_view.h`, `viewpoint.h:39`);
- residual_risk numerics (κ = 1e6, continued-fraction caps);
- the oracle's default cell 0.1 (the caller passes it);
- `kEmIters` 10, `kPublishPeriod` 1 s;
- `Model.ReplayDump`, `Concept.LookAttrs`, `Concept.MeshPath`.

In all, excluded +15: these 16 rows, less the `gn_tol_sigma` row moved to patch.

**Switches (other):** `Concept.AppearanceEnabled`, `PublishExistProb`, `ThumbnailsEnabled`, `ApproachEnabled`,
`EpistemicOfferEnabled`, `ChangedObjectEnabled` (K2b consumer channels, read for every countable class).

**Newly pulled library rows:** `appearance_belief`, 6 rows (class? 3, patch? 2, sensor? 1).

### Combined totals per class after part 1 (build rule)

Each row is the addendum-2 row of the target the class now runs, plus the deltas above:

| target | class | sensor | patch | switches (patch) | ? | excluded | switches (other) | total |
|---|---|---|---|---|---|---|---|---|
| countable | +6 | +1 | +3 | 0 | +17 | +15 | +6 | +48 |
| legacy table and chair | 0 | 0 | +1 | 0 | +3 | +3 | 0 | +7 |

The countable patch delta of +3 is made of:
- +6 new or newly counted: the room guard, the tol_sigma relabel, the wall χ² gate, the Σ cap, GhostEvidenceGain, the ghost
  L_max;
- −3 for the AI2CommonMode* rows moved to sensor?.

| class (binary it runs) | class | sensor | patch | switches (patch) | ambiguous (?) | excluded | switches (other) | **patch total** | total rows | baseline patch total (addendum 2) | change |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nightstand (countable) | 19 | 21 | 123 | 11 | 117 | 95 | 10 | **134** | 396 | 131 | **+3** |
| bed (countable) | 19 | 21 | 123 | 11 | 117 | 95 | 10 | **134** | 396 | 131 | **+3** |
| toilet (countable) | 19 | 21 | 123 | 11 | 117 | 95 | 10 | **134** | 396 | 131 | **+3** |
| refrigerator (countable, K1) | 19 | 21 | 123 | 11 | 117 | 95 | 10 | **134** | 396 | 129 | **+5** |
| hood (countable, K1) | 19 | 21 | 123 | 11 | 117 | 95 | 10 | **134** | 396 | 128 | **+6** |
| chair (countable, K2b) | 19 | 21 | 123 | 11 | 117 | 95 | 10 | **134** | 396 | 110 | **+24** |
| table (legacy, cut) | 5 | 16 | 108 | 13 | 92 | 74 | 3 | **121** | 311 | 120 | **+1** |

Notes:
- **The six countable classes now have identical rows.** Their configs share one key set (`ReplayDump` added; the two
  RangeNoise keys are now hidden defaults, not gone; K1/K2b added nothing the loader does not read for every class). The
  class differences are values and manifests.
- **Not one class went down.** The chair went up most (+24): it traded its own legacy rows (patch total 110) for the
  countable rows (131 at baseline).
- **Ambiguous rows went up by 17.** 9 of them are code moved into scope from the controller or newly pulled libraries; 3 are
  the AI2CommonMode* rows; 5 are new.
- **The legacy refrigerator and hood binaries** (selectable, not graded) gained the engine files through T10a's adapter, and
  `place_map.h`. Their patch totals by the same deltas are 129 → 134 (fridge) and 128 → 133 (hood). This is approximate: the
  adapter's guard literals are counted excluded.

### Registered predictions against the result

- **G1, G2, G3, G4, G5 — FAIL.**
  - The spec registered about 40 rows for removal from the countable classes (G1–G4) and the offer-side gains for G5.
  - None was removed under the build rule. Three were moved (AI2CommonMode*, patch → sensor?).
  - Six became inert on the graded path while staying in scope. Under the rule ("every source file each class binary
    compiles"), inert rows are counted, because the flag-off path is compiled and selectable (the spec keeps it for the
    arXiv v1 rebuild and C3 non-regression until grading).
  - The terms that do act on the graded path are KEPT, each with a recorded reason or none:
    - the top anchor (none);
    - the depth-unobs footprint share (replay evidence);
    - the short-height prior (none);
    - the confirm-only gate (F1.8 audit);
    - MotionCm (τ unmeasured).
- **G6 — FAIL by the letter.**
  - G6 was to add no constant beyond the robot footprint.
  - The shared oracle brought the 95 % room-mask guard (flagged), the ½ occupancy boundary, the Beta floor and the no-belief
    convention into every target's scope. They are pre-existing in the controller, not new to the system, but new to the
    class count.
  - The countable side runs with margin 0, a caller's choice (owner item: publish it as a robot property).
- **"Not expected to move"** (existence and plausibility, process noise, birth seeds, the room's 0.35): did not move.
  - As expected, except that process noise moved from a constant to the class permanence prior (owner ruling, F1 round 6).
    This is new manifest class knowledge. The old `AI2ProcessStd*` rows remain, read by the legacy path.

**Headline (countable classes):**
- registered rows: removed 0, moved 3, kept 48;
- new rows +48, of which patch +6, two of them flagged thresholds plus one unflagged threshold found here;
- patch total 131 → 134 for nightstand, bed and toilet, and up for every class that moved onto the binary.

**The count went up, not down. Test (c) fails as registered.**
