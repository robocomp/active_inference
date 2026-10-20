# Measuring the local policy's inputs (Task 7)

Four numbers the local-motion policy needs that nothing measured before. Two come from existing logs (done
2026-10-20, offline); two need the simulator running. Every provisional value in the tree is marked
`PROVISIONAL — replace with the measured value (controller/tools/measure_policy_inputs.md)`.

| input | where it lives | status | tool |
|---|---|---|---|
| base deadband `v_min`, `w_min` | `robot_concept/etc/config_shadow.toml` `[Agent] base_min_linear_speed / base_min_rot_speed` → robot node `robot_min_*` → `rc::BaseCapability::min_linear_speed / min_rot_speed` | **PROVISIONAL** 0.04 m/s, 0.10 rad/s | `measure_deadband.py` (LIVE) |
| contact band floor | `controller/etc/config.toml` `[Lidar] ContactMinHeightM` | **PROVISIONAL** 0.08 m | `floor_band.py` (LIVE) |
| `sigma_rel_xy`, `sigma_rel_yaw` | `controller/etc/config.toml` `[Policy] SigmaRelXyM / SigmaRelYawRad` | fitted 0.011 m / 0.013 rad | `pose_error_stats.py` (logs) |
| `pose_drift_sigma_m`, `pose_drift_time_s` | `controller/etc/config.toml` `[Policy] PoseDriftSigmaM / PoseDriftTimeS` | fitted 0.064 m / 115 s | `pose_error_stats.py` (logs) |

Paths below: `B=/home/pbustos/robocomp/components/active_inference` (main checkout, where the fleet runs and the
data files land), `WT=/home/pbustos/robocomp/components/active_inference.worktrees/interaction-anchor` (this branch).

## 0. Before the live session

* Webots running with the world loaded, robot **parked on OPEN floor** — at least 0.7 m free ahead and behind and
  1 m all round with no furniture leg nearby (the floor percentile is what is being measured). `robot_concept` running.
* Binaries of this branch built: `cd $WT/controller && bash -ic 'cbuild -j8'` (check `pgrep -a cc1plus` first).
* The deadband channel on the graph additionally needs cortex reinstalled with `robot_min_linear_speed` /
  `robot_min_rot_speed` (already added to `cortex/core/include/dsr/core/types/type_checking/dsr_attr_name.h`, NOT
  installed). Measuring does not need it; PUBLISHING the result does. After the reinstall, re-run `cbuild -j8` in
  `controller` and `robot_concept` — CMake re-detects the attributes (`RC_HAVE_ROBOT_DEADBAND_ATTRS`) by itself and
  the controller's `[capability]` block then prints a `deadband` line.

## 1. Fleet up, controller STOPPED, robot where it is

```bash
cd $B
D=$B/room_concept/estimation_scripts/crossing/policy_inputs_$(date +%m%d_%H%M)
FLEET="retina room_concept residual_concept controller door_concept ltsm_agent table_concept" \
WT=$WT WT_AGENTS="controller door_concept table_concept" \
  $B/room_concept/estimation_scripts/scripts/run_crossing.sh $D here
```

(chair_concept excluded.) Wait until room_concept has localised (its log reports the room; ~30-60 s) — the probe
refuses to start without a pose and a map.

## 2. Floor band → `Lidar.ContactMinHeightM`

```bash
python3 $WT/controller/tools/floor_band.py --sweeps 20 | tee $D/floor_band.txt
cp $B/controller/lidar_z_dump.csv $D/
```

Sends `zdump 20` through `$B/controller/remote_cmd.txt`; the controller writes the next 20 raw sweeps (robot frame,
pre band, with a self flag) to `$B/controller/lidar_z_dump.csv`. The script prints the floor-z histogram, per-plane
and per-range percentiles, and the line to paste into `$WT/controller/etc/config.toml` (`[Lidar] ContactMinHeightM`,
value = floor p99.9 + sigma_r, with the run id). If it warns that the value reaches 0.10 m, read the per-range table
before pasting anything: the 0.10 m test expectation (`low_returns_reach_policy`) would be contradicted by the sensor.

## 3. Deadband → `base_min_linear_speed / base_min_rot_speed`

```bash
python3 $WT/controller/tools/measure_deadband.py --out $D/deadband.csv | tee $D/deadband.txt
```

Sends `probe <v> <w> 2.0` for v in 0.005..0.10 m/s and w in 0.01..0.25 rad/s, each sign in alternation (net motion
zero), plus one zero command for the noise floor; reads the motion from `$B/room_concept/tmp/sdf_localizer/gt_error.csv`
(GT) or, failing that, from the pose rows the controller writes to `$B/controller/base_probe.csv`. About 2.5 min.
Paste the two printed lines into `$WT/robot_concept/etc/config_shadow.toml` `[Agent]` (replacing the PROVISIONAL
ones) and restart robot_concept. The script sends `stop` at the end.

The probe is refused (logged as `[probe] refused: ...` in `$D/logs/controller.log`) if the predicted sweep is not
free on the planner's grid — move the robot to more open floor.

## 4. Stop the fleet — SIGTERM only

```bash
$B/room_concept/estimation_scripts/scripts/run_crossing.sh $D collect
```

(`collect` sends `stop`, copies the logs incl. `gt_error.csv` into `$D`, then `pkill -TERM`s the fleet. Never `kill -9`.)

## 5. Pose-error statistics (offline, any time)

```bash
python3 $WT/controller/tools/pose_error_stats.py --csv $D/pose_error_runs.csv \
    "$B/room_concept/estimation_scripts/crossing/*/gt_error.csv" $B/room_concept/tmp/sdf_localizer/gt_error.csv
```

Prints the per-run table, the per-run spread (median [p10, p90]) and the pooled fits; update `[Policy]` in
`$WT/controller/etc/config.toml` if a new set of runs moves them. The live copy under `tmp/sdf_localizer/` is
de-duplicated against its archived crossing directory by first timestamp.
