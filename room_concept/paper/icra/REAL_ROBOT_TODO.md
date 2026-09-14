# Real-robot run — TODO

Shadow · real apartment · ~25 min of robot time. Self-contained: you should not need any other file.
All commands are run from `room_concept/`.

Analysis is fixed in advance by `paper/icra/PREREGISTRATION_REAL_TOUR.md`. Do not change what is
reported after seeing the numbers.

---

## 1 · Bench, no robot (5 min)

- [ ] `etc/config.toml` → `LearnMotionModel = false`
- [ ] `etc/config.toml` → `PoseClampVMax` / `PoseClampWMax` = the **controller's real limits** (not 0.7 / 0.8)
- [ ] copy the MRPT SVG into `layouts/`
- [ ] `etc/config.toml` `[Scenario.apartamento]` → `RoomLayoutSvg = "<the MRPT svg>"`
- [ ] `make -C build gn_selftest && ../bin/gn_selftest` → all pass
- [ ] `make -C build motion_calib && ../bin/motion_calib --selftest` → all pass

**Do not run anything below if a selftest fails.**

---

## 2 · Phase 1 — parked, motors ENABLED, 5 min · GO / NO-GO

Park in the room, motors on, zero command, hands off. Log 5 minutes.

- [ ] `../bin/motion_calib <debug_log.csv>`
- [ ] read the parked spread of the RAW reported velocity

| result | action |
|---|---|
| σ_v within 2× of **0.0202 m/s** AND σ_ω within 2× of **0.0323 rad/s** | constants transfer → **go to step 5** |
| either outside 2× | → do steps 3 and 4 first, then step 5 |

- [ ] save the debug log either way — these frames are also a paper measurement

---

## 3 · Phase 2 — closing pivot, 2 min · *only if NO-GO*

- [ ] ~4 full turns in place, return to the start heading
- [ ] `../bin/motion_calib <log>` → new `PreintOdomScaleOmega` into `[Platform.Shadow]`

## 4 · Phase 3 — straight leg, 2 min · *only if NO-GO*

- [ ] drive a straight leg and **tape-measure it**
- [ ] `../bin/motion_calib <log>` → new `PreintOdomScaleV` into `[Platform.Shadow]`

---

## 5 · The tour (~15 min)

- [ ] `given` mode, MRPT layout as the map
- [ ] ≥ **100 m** travelled
- [ ] ≥ **3 stops of ≥ 30 s**, robot completely still
      ⚠ the repeat/scatter split is measured ONLY over stationary stretches. No stops → no result.
- [ ] rotation at **several different rates**, slow and fast — not one constant-speed loop
- [ ] every wall seen, including ones only visible from far away

---

## 6 · Save — STOP THE AGENT FIRST, then copy

- [ ] **Stop the agent** (SIGTERM / Ctrl-C — never `kill -9`; it owns nodes in the shared graph)
- [ ] confirm it is stopped: `pgrep -x room_concept` returns nothing
- [ ] only then copy

⚠ **Two ways to lose the run, and we hit both in one afternoon.**

1. **Copying while it still runs gives you a partial tour.** Done on 2026-09-14: the copy caught the
   first third, and every number moved when the full log was analysed — 148k candidates became 300k,
   12 stationary stretches became 34, the ratio went 9.9 to 8.7. One claim reversed outright and had
   to be withdrawn from the paper. A partial log looks completely normal; nothing warns you.
2. **Starting the agent again destroys the previous run.** `room_concept.h` opens the probe with
   `std::ios::trunc`. This is how the paper's original tour was lost, along with the numbers that had
   already been published from it.

```
D=datasets/real_apartment/$(date +%m%d_%H%M); mkdir -p $D
cp tmp/corner_probe.csv  $D/corner_probe.csv
cp tmp/layout_trace.csv  $D/layout_trace.csv
cp layouts/<the MRPT svg> $D/
```

- [ ] both CSVs copied
- [ ] MRPT SVG copied, with its stated precision (≤ 2–3 cm) written down
- [ ] `motion_calib` debug log copied
- [ ] one line of notes: date · robot · room · tour length · were constants re-derived?

⚠ **Both CSVs are required.** `corner_probe.csv` carries no pose, so without `layout_trace.csv` there
are no rotation bands and no stationary stretches — i.e. no headline number, and the run is wasted.

---

## 7 · Analyse

```
python3 analysis/corner_channel/split_and_nis.py $D/corner_probe.csv $D/layout_trace.csv
```

- [ ] paste the output into the run notes verbatim
- [ ] anything quoted in the paper comes from that output and nowhere else

---

## Stop conditions

- A selftest in step 1 fails → fix it; do not "try it anyway".
- Tour had no stationary stops → redo the tour; it cannot support the main number.
- Nothing analysed by **Tue 15 Sep, 12:00 CEST** → submit the paper as it stands; the tour becomes a
  journal-version result.

## For the paper

Anonymous: "a differential-drive robot with a 3D LiDAR, in an apartment-scale indoor space".
No robot name, laboratory, institution or building.
