# C4 control: gyro bias +0.002 rad/s, calibrator ON vs OFF, 2026-11-23 (stopped after one leg, owner ruling)

## Setup
- **World**: Webots piso.wbt, the world of arm 7.
- **Bias**: the shadow-bridge-dds bridge with `SensorNoise.GyroBias = 0.002`, confirmed in its log: `bias=+0,002 rad/s`.
- **Route**: controller route `calib turns`, 14 laps, 98 waypoints, started in Path mode.
- **Fleet**: room_concept, residual_concept and controller from main (binaries as on disk on 11-23); robot_concept as running.
- **Cold start**: the calibration state was wiped before the leg.
- **Report**: `room_concept/tools/c4_report.py`, plus `tools/calib_benefit.py` for the within-run counterfactual.

## Result (leg c4_on_r1, calibrator ON, mask -1)

| | c4_on_r1 (today's build) | arm7 gyro-all (Oct 4 build) |
|---|---|---|
| distance, duration | 226 m, 14.1 min | 180 m, 14.3 min |
| pose-optimiser solves (`iters>0`) | **17** | 127 |
| calibration episodes | **10** | 66 |
| b_omega at end | **−0.00001 ± 0.00050 (its prior; never informed, never applied)** | +0.00065 ± 0.00025 |
| heading drift while moving, pred − truth, deg/s, by thirds | +0.202 / +0.255 / +0.198 | +0.064 / +0.045 / +0.030 |
| GT heading error, deg per rad turned | 1.43 | 0.66 |

- The drift definition (`c4_report.py`) is signed (predicted − true heading increment) over moving rows, by thirds of distance. On arm7 it reproduces the published falling drift, with a slightly different definition than the §13 table.

## Reading
- **On today's build the calibrator is starved.**
  - An episode closes only after an actual pose correction by the optimiser (`motion_calibration.h:473-505`).
  - The optimiser ran 17 times in 226 m, against 127 times in 180 m in October.
  - So b_ω never became informed, was never applied, and the ON arm behaved like OFF: the drift stays at about +0.2 deg/s for the whole leg.
- **Live ON-vs-OFF legs on this build were stopped** (owner ruling 2026-11-23): they would compare two arms that both learn nothing.
- **Why the optimiser now runs so rarely is not investigated.** It is separate work, likely the localisation path (polish / early exit) changed since October.
- **The control that does exist is the within-run counterfactual** (`calib_benefit.py`, EXPERIMENT.md §13 addendum). It removes the acting calibration from the same run's prediction:

| run | GT signed drift, no calibration | GT signed drift, as run |
|---|---|---|
| arm7 gyro-all | −0.100 deg/s | −0.052 deg/s |
| arm7 native-all | −0.067 deg/s | −0.035 deg/s |
| c4_on_r1 (16 anchored stretches only) | −0.012 deg/s | −0.007 deg/s |

- **What the paper's C4 numbers stand for**: the October build (arm7, commit 6bab57e era). They should be labelled as such.
- **On the current build** the calibrator's input rate is the open problem.

## Restored after the leg
- Bridge GyroBias back to 0.0 (from the backup). The bridge needs a restart to read it.
- room_concept config restored byte for byte.
- `etc/motion_calib_state.csv` put back from the backup.
- The leg folders are `crossing/c4_pilot`, `c4_on_r1_1lap` (latch bug, 1 lap), `c4_on_r1` (the leg) and `c4_off_r1` (aborted before the route).

## Controller bug found on the way (not fixed)
- A first Run with laps = 1 latches a one-lap route in the policy driver (`controller_policy_drive.cpp:711-729`).
- Stop does not clear the latch, so a later 14-lap Run finishes after one lap.
