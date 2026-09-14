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
- [ ] **see the same corners from several different places** — the viewpoint test needs corners
      observed in ≥ 3 separate stops, so vary where you stop, not just how you drive between stops
- [ ] rotation at several rates, slow and fast — not one constant-speed loop
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
cp tmp/corner_probe.csv $D/     # ESSENTIAL — every number in the paper comes from this one
cp tmp/corner_nis.csv   $D/     # per-frame stats
cp etc/pose_trace.csv   $D/     # the only log carrying the ROBOT's pose; gives the tour length
cp layouts/<the MRPT svg> $D/
```

- [ ] the three CSVs copied
- [ ] MRPT SVG copied, with its stated precision (≤ 2–3 cm) written down
- [ ] `motion_calib` debug log copied
- [ ] one line of notes: date · robot · room · tour length · were constants re-derived?

⚠ **Do not look for `tmp/layout_trace.csv`.** This build does not write it (it was 0 bytes after the
2026-09-14 tour). Nothing needs it: stillness is recovered from the detections themselves, so
`corner_probe.csv` alone produces the headline number. Its pose column was the ROOM MODEL's anyway,
not the robot's, which is what put a wrong rotation figure into a paper figure.

---

## 7 · Analyse

```
python3 analysis/corner_channel/split_and_nis.py $D/corner_probe.csv | tee $D/RESULTS.txt
```

One argument. The script needs no pose file and no reference layout.

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
