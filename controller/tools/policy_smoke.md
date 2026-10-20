# Live smoke test of the local motion policy (Task 8, step 5) — PENDING (simulator not running on 2026-10-20)

The policy drives every motion when `controller/etc/config.toml [Policy] Enabled = true` (the default on this branch).
This is the first live run: one door crossing out of room_1 through door_2 and back, in the three-room world.

Paths: `B=/home/pbustos/robocomp/components/active_inference` (main checkout — the fleet runs from there, data files and
`controller/remote_cmd.txt` live there), `WT=/home/pbustos/robocomp/components/active_inference.worktrees/interaction-anchor`
(this branch).

## 0. Before

* Webots running with the THREE-ROOM world loaded (the world `three_rooms_tour.sh` uses), robot in room_1.
* This branch built: `cd $WT/controller && pgrep -a cc1plus; bash -ic 'cbuild -j8'` (also `door_concept` /
  `table_concept` of this branch if they changed since their last build).
* Stop agents only with SIGTERM (`run_crossing.sh <dir> collect` does that). Never `kill -9`.

## 1. Fleet up WITHOUT chair_concept, controller STOPPED, robot where it is

```bash
cd $B
D=$B/room_concept/estimation_scripts/crossing/policy_smoke_$(date +%m%d_%H%M)
FLEET="retina room_concept residual_concept controller door_concept ltsm_agent table_concept" \
WT=$WT WT_AGENTS="controller door_concept table_concept" \
  $B/room_concept/estimation_scripts/scripts/run_crossing.sh $D here
```

Wait until room_concept has localised (its log names the room; ~30-60 s). Check the controller's start-up lines:

```bash
grep -E "\[capability\]|Policy\.|\[policy\]" $D/logs/controller.log | head -20
```

`Policy.Enabled` must read `true`. The `[capability]` block shows the deadband line only after cortex is reinstalled
(`robot_min_*`); without it the policy uses the PROVISIONAL `[Policy] MinLinearSpeedMps / MinRotSpeedRps` (0.04 / 0.10).

## 2. Arm, cross door_2, then back

```bash
echo run > $B/controller/remote_cmd.txt; sleep 2
echo "cross door_2" > $B/controller/remote_cmd.txt
# wait for the passage to end: "[passage]" ... done / failed in the controller log
grep -E "\[passage\]|\[anchor\] cross|\[gap\]|\[policy\]" $D/logs/controller.log | tail -40
# then back the other way (the door is door_2 from room_2 as well — check the room's door list in the log if not)
echo "cross door_2" > $B/controller/remote_cmd.txt
```

(A relative move must not re-arm the base: after `echo stop > $B/controller/remote_cmd.txt`, `move 0.3` must be
REFUSED with "the base is disarmed (Stop) — press Run first".)

## 3. Collect and stop

```bash
cp $B/controller/policy_decisions.csv $D/
$B/room_concept/estimation_scripts/scripts/run_crossing.sh $D collect
```

## 4. What to look for

`policy_decisions.csv` (one row per control cycle; classic-locale CSV): `t_ms, who, goal_kind, mode, v, w, log_p_safe,
log_p_cont, progress_m, eligible, replanned, plan_remaining_p, prediction_failed, trap, unreachable, world_applied, x, y,
yaw, v_meas, w_meas, odo_valid, field_m, n_returns, n_unseen, returns_age_s, latency_s, step_ms, why`.

```bash
python3 - $D/policy_decisions.csv <<'EOF'
import csv, sys, statistics as st
rows = list(csv.DictReader(open(sys.argv[1])))
t = [int(r['t_ms']) * 1e-3 for r in rows]; m = [r['mode'] for r in rows]
changes = [t[i] for i in range(1, len(m)) if m[i] != m[i - 1]]
secs = {}
for x in changes: secs[int(x)] = secs.get(int(x), 0) + 1
span = range(int(t[0]), int(t[-1]) + 1)
per_s = [secs.get(s, 0) for s in span]
print(f"rows {len(rows)}  span {t[-1]-t[0]:.0f} s  mode changes {len(changes)}  median/s {st.median(per_s):.2f}  max/s {max(per_s)}")
for k in ('Drive', 'Release', 'Arrived', 'Stuck', 'Blind'): print(f"  {k:8s} {m.count(k)}")
print("  traps", sum(r['trap'] == '1' for r in rows), " failed predictions", sum(r['prediction_failed'] == '1' for r in rows),
      " unreachable", sum(r['unreachable'] == '1' for r in rows))
print("  step_ms median %.2f p90 %.2f" % (st.median(float(r['step_ms']) for r in rows),
      sorted(float(r['step_ms']) for r in rows)[int(0.9 * len(rows))]))
print("  returns_age_s max %.2f  latency_s median %.3f" % (max(float(r['returns_age_s']) for r in rows),
      st.median(float(r['latency_s']) for r in rows)))
EOF
```

PASS criteria (the brief's): both crossings complete (`[anchor] cross: … — through` and the passage's done line);
**mode flapping < 1 change/s median**; no `Stuck`; no contact (Webots shows none; `v_meas` never stuck at 0 while `v`
asks for motion for more than a plan duration).

**MUST-PASS GATES (review I5, M5, M6) — stop the run if any fails:**

* **Odometry turn sign** — no `[policy] … Blind — odometry turn sign contradicts the pose` line in the controller log
  (`grep "turn sign contradicts" $D/logs/controller.log` must be empty), and in Drive rows with `|w_meas|` above the
  deadband the sign of `w_meas` matches the sign of the change of `yaw` between consecutive rows.
* **Field rebuild time** — every `[policy] goal for … field built in X ms` line has X <= 100 (two control periods; the unit
  gate in test_grid_planner_curtain (6) measured 24 ms for a 20 x 12 m map with a soft layer):
  `grep -o "field built in [0-9.]* ms" $D/logs/controller.log | awk '{if ($4 > 100) bad++} END {print bad+0, "over 100 ms"}'`
* **Odometry fresh** — no `odometry stale` line while the base is driving (`grep "odometry stale" $D/logs/controller.log`).

Also check, because they are first-time live facts:

* **Rot sign**: in Drive rows, `w` and `w_meas` have the SAME sign (the session sends rot = +w; robot_current_angular_speed
  is CCW-positive). Opposite signs = stop the run: the base turns the wrong way.
* **Doorway completion** is the protocol's (`[anchor] cross: … — through`); the policy never reports Arrived on a
  Doorway goal (its goal line is 2 tol_xy beyond the standoff).
* **The doorway field**: the `[policy] goal for cross door_2 …` line prints `field … m from the robot` — it must be FINITE
  (the far side is joined by the measured-gap corridor, goal_adapter.h `in_doorway_corridor`). `inf` = the goal line is
  not connected on the map.
* **WATCH (Task 1b ledger)**: try78-80 saw self-returns at y ≈ +0.26 in sim; the sim body front is +0.230. If the first
  Drive rows show `log_p_safe` ≈ 0 with nothing ahead, or `Stuck "returns inside the body"`, dump the contact buffer and
  look for returns at the front face (`zdump 5` via remote_cmd.txt, then `tools/floor_band.py --file`).
* **Floor**: with the helios far floor cut by source (lidar_height_band.h), `n_returns` on open floor should not grow
  with the distance to the walls beyond ~3.9 m.
* **returns_age_s** stays below 0.5 s (above it the policy is Blind and holds).
* **unseen contact**: `n_unseen` > 0 only after a real block; frequent hypotheses in open floor = the odometry sign/scale
  or the command latency (`[Policy] CommandLatencyS`, PROVISIONAL 0.06) is wrong.
