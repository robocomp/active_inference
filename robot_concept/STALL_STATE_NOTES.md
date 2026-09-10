# Stall State Notes

This file stores the design discussion about handling critical runtime stalls through the global worker state machine.

## Current Global State Machine

From `generated/genericworker.h` and `generated/genericworker.cpp`:

- `Initialize -> Compute` happens automatically on enter.
- `Compute -> Emergency` happens only when `goToEmergency()` is emitted.
- `Emergency -> Restore` happens only when `goToRestore()` is emitted.
- `Restore -> Compute` happens automatically on enter.

Implication:

- `Restore` is currently a bridge back to `Compute`, not a holding state.
- If we want to wait until inputs are healthy again, that waiting must happen in `Emergency` with the current generated machine.

## Why Centralize Stalls In Emergency

Main benefit discussed:

- centralize critical runtime situations in one global state
- publish a topic from `Emergency` so other agents know this component is stalled
- expose a single authoritative reason for degradation or loss of service
- stop scattering critical control decisions across acquisition threads and compute paths

Potential reason codes:

- `rgbd_missing`
- `lidar_missing`
- `both_missing`
- `transform_missing`
- `fps_collapse`
- `proxy_failures`

## What Should Trigger A Stall

Input loss is only one possible trigger. Candidate stall conditions discussed:

- no valid RGBD frames for longer than a timeout
- no valid LiDAR frames for longer than a timeout
- stream cut during execution
- sustained drastic reduction in compute FPS
- sustained drastic reduction in acquisition FPS
- repeated proxy failures in a time window
- missing critical transforms or graph dependencies for too long
- processing backlog causing stale data age beyond a limit

Key idea:

- Emergency should represent failure of service guarantees, not one isolated symptom.

## Detection And Transition Strategy

Recommended separation of responsibilities:

- low-level detectors stay near the source of truth: acquisition threads, watchdogs, timestamp age checks, FPS monitors
- these detectors update shared health state only
- one central policy point decides when the component is officially stalled
- that policy point emits `goToEmergency()`

Why:

- avoids scattered `goToEmergency()` emissions from many unrelated code paths
- keeps policy coherent and easier to tune

## Emergency Semantics

Suggested semantics if this is implemented:

- `Emergency` is the holding state for stalled operation
- `Emergency` periodically publishes a stall topic/status for the rest of the system
- `Emergency` includes structured state such as reason, time since stall, and current health metrics
- `goToRestore()` is emitted only after recovery conditions are met
- `Restore` performs any one-shot cleanup or reset and then returns to `Compute`

## Recovery Policy

Recovery should not happen on the first good sample. Use hysteresis and dwell time.

Suggested principles:

- require valid input to remain healthy for a minimum time before leaving `Emergency`
- use different enter/exit thresholds for FPS-based conditions
- avoid state flapping on brief transport hiccups

Examples:

- enter stall if metric stays below threshold for `T_enter`
- exit stall only if metric stays above recovery threshold for `T_exit`
- choose `T_exit > 0` and `threshold_exit > threshold_enter` when relevant

## Architectural Caveat

Open question discussed:

- should `Emergency` mean truly unsafe/critical, or just unavailable/degraded?

If the system semantics require a hard global signal to other agents, using `Emergency` is justified.
If not, a dedicated `Stalled` or `Degraded` state may be architecturally cleaner.

## No-Code Decision At This Point

Implementation was intentionally deferred. The agreed immediate direction was:

- keep this debate recorded for later
- fix more urgent issues first
- revisit the state-machine integration afterward

## Next Design Questions Before Coding

- which conditions should trigger `Emergency` versus simple local logging
- what exact timeouts define stream loss or stream stall
- whether LiDAR and RGBD should have identical policies
- whether transform loss should also count as a stall
- what payload the published stall topic should contain
- what recovery dwell time is required before returning to `Compute`