# Builds and dates of the runs the AuRo paper uses, 2026-11-23

Sources: `<run>/agents.txt` (name, binary, commit, real binary; written by `concept_agents.sh`; it records the HEAD of the checkout at fleet start, it does not prove the binary was built from that commit),
`<run>/logs/*.log` (config path of each agent, which names the checkout it ran from), `gt_pose.tsv` first/last timestamps (local time, UTC+1) for the run window, `git log`. A table of the extraction is in
`auro_1122/scripts/review2/runinfo.tsv` (script inline in this session). Folder mtimes are NOT run dates: many folders were rewritten by the 11-22 regrade (e.g. `c3a_1` mtime 11-22 22:12, run 11-21 14:46).
No run prints a build id of its own; `agents.txt` exists only for the furnished runs and lists the concept agents + ltom, not room_concept/ltsm/controller/door/retina.

## Frozen clean62-76 (C1 three-room tours): the build id question
**Answer: c63d56d9 is right; f74d1b41 is the build one commit earlier and is wrong for these runs.**
- Git: `f74d1b41` (2026-11-13 00:35, "doors: a phantom can be refuted...") is an ancestor of `c63d56d9` (2026-11-13 14:42, "doors: a wall seen through is soft, a leaf steps before its mask, a deleted door keeps its height"), 1 commit apart. Tag `auro-freeze-11-13` points to `c63d56d9` (`git tag -l`). `c63d56d9` is an ancestor of the rt-build HEAD (`1444c335`, 11-13 23:20).
- Timeline of the runs: clean62 starts 14:43 (first `ltsm_history` file 14:42:42, `gt_pose.tsv` first row 14:43), i.e. after the 14:42:00 commit. clean62-76 span 14:43 - 19:37 on 2026-11-13, all on `active_inference.worktrees/rt-build` (config paths in each run's `logs/`).
  Series runs: clean62 14:43, 63 14:49, 64 17:45, 65 17:54, 66 18:00, 67 18:09, 68 18:17, 69 18:25, 70 18:33, 71 18:41, 72 18:49, 73 18:58, 74 19:07, 75 19:21, 76 19:30 (clean64-65 had been renamed `*_diskfull` mid-series and resumed at 17:44).
- The memory note `resume-11-13-auro-series.md` has the same sequence: f74d1b41 was the freeze when clean60/clean61 failed (DOOR_0 aperture dragged; door never born); the fixes in c63d56d9 were written for those failures, the owner approved the commit, the tag was moved to it, and only then "Series RUNNING clean62-76". Its final line reads "series = c63d56d9 + pre-11-14 library".
- Where the wrong id sits: `crossing/aff_arbitration_summary.md:57` ("clean62-76 = the frozen f74d1b41 series"); the paper's `sections/I-evidence-tables.tex:17` already says c63d56d9.
- Not provable from the runs: no clean62-76 log prints the commit, and the rt-build `door_concept` binary now on disk is dated 2026-11-13 22:10 (rebuilt after the series), so the binaries of the series cannot be checked against the commit. The inference rests on the commit time, the run times and the owner-approved tag move.
- Library: the series ran on the libdsr_api.so installed before 2026-11-14 00:06 (memory note, 11-14 ruling). Not re-checked here.

## Table
Commit column: `agents.txt` HEAD(s) for the concept agents; "core" = room/ltsm/controller/door/retina/robot, which come from the checkout named in the logs and whose commit is NOT recorded: where the runs say `active_inference` the checkout is main, whose HEAD at the time is given in the last column (from `git log`, with the product-source diff between consecutive set commits in the notes below).

| set | run ids | run window (local) | concept-agent commit (`agents.txt`) | core agents from | main HEAD at that time |
|---|---|---|---|---|---|
| bench rooms (100 random rooms, orders 4-12) | `data/newroom_bench/results.csv` in the paper tree; no run folder | files dated 2026-11-06 20:17 | not recorded | room_concept bench (`tools/newroom_bench.h`) | 79fc14b1 (2026-11-06 20:15), inferred only |
| apartment campaign, live 10 starts | `estimation_campaign_v3` (`/home/pbustos/claude_scratch_2026-10-10/`) | 2026-10-10/11 (folder mtime 10-11 11:54) | not recorded | main | PROGRESS.md names `f92a016` (2026-10-11 02:01) |
| apartment (synthetic, thesis table) | `room_concept/estimation_scripts/apartment/campaign_v5_curious`, `campaign_v4` | mtime 2026-10-11 21:31 / 12:33 | not recorded | main | 8da07f06 "campaign v4 on 22d763b9" (2026-10-11 13:57) |
| **frozen C1** | clean62-76 (13/15 passed; fails clean63, clean71) | 2026-11-13 14:43 - 19:37 | none (no furnished agents) | rt-build | **c63d56d9** (see above) |
| explorer rerun, 594 plans | `explorer_rerun_1122/` | 2026-11-22 11:00-12:43 (file mtimes) | n/a | `bin/wall_slam_selftest` built 2026-11-22 11:00 from main 640298ef + an uncommitted edit of `tools/wall_slam_selftest.cpp` (`SUMMARY.md`) | 640298ef |
| C2 explore | c2_x1 (failed, DOOR_0 shut), c2_x1b, c2_x2, c2_x3 | 2026-11-21 21:32 - 23:22 | no `agents.txt` (no concept agents) | main | 4ff4684b (2026-11-21 20:31), per memory `auro-c2-room-identity-11-21` |
| C2 warm | c2_w1 (failed start-up), c2_w2, c2_w3 | 2026-11-21 22:57 - 23:37 | **3df281a0** (11-21 22:55) | main | 3df281a0 |
| C2 recog runs | recog_try1/2/3/10/32, recog_proto1/5/8/11-15 | 2026-10-24 to 2026-10-30 (and recog_proto1/try1-3 re-copied 11-13 20:20) | no `agents.txt` | `active_inference.worktrees/rt-main` | not recorded (rt-main is not a git repo now) |
| C3a, old world | c3a_1, c3a_2 | 2026-11-21 14:46 - 15:21 | **6f8a850b** (11-21 13:56) | main | 6f8a850b |
| C3a, old world, cabinet fix | c3a_cab1, c3a_3 | 2026-11-21 16:30 - 16:54 | 6f8a850b + **e5310a2a** (cabinet, nightstand, bed, toilet from the cabinet-exit worktree, 11-21 16:28) | main | 6f8a850b |
| C3a, new world | c3a_w2_1..3 | 2026-11-22 22:14 - 22:52 | e5310a2a (cabinet/nightstand/bed/toilet) + **fd0830ce** (rest, main HEAD) | main | fd0830ce |
| C3b pair 1 (no LTOM) | c3b_on1, c3b_off1 | 2026-11-21 10:47 - 11:09 | **9af4002a** (11-20 22:56) | main (controller binary of 11-20 22:57) | 9af4002a |
| C3b pair, LTOM | c3b_on_ltom1, c3b_off_ltom1 | 2026-11-21 11:11 - 11:33 | 9af4002a | main | 9af4002a |
| C3b hidden | c3b_on_hidden1 / c3b_off_hidden1 | 2026-11-21 13:37 - 14:06 | **e3f67218** (on) / edd267dc (off) | main | the two commits differ only in paper files (`git diff --stat`) |
| C3b x5 | c3b5_on_r1..r5, c3b5_off_r1..r4 | 2026-11-22 19:33 - 21:04 | **565d9d4e** (11-22 18:19) | main | 565d9d4e |
| | c3b5_off_r5 | 2026-11-22 21:17 - 21:28 | fd0830ce (same binaries, `c3b5_summary.md`) | main | fd0830ce |
| C14 removal | c14_r1, c14_r2, c14_r3 | 2026-11-22 21:32 - 22:07 | **fd0830ce** (11-22 21:10) | main | fd0830ce |
| lambda sweep | lam_0p1, lam_0p4, lam_0p2 (run order) | 2026-11-22 23:21 - 23:58 | e5310a2a (nightstand, bed, toilet) + 267db579 (rest, scripts only) | controller: main binary of 2026-11-20 22:57 (`lambda_summary.md`) | 267db579 (11-22 23:18) |
| NEES grading | 12 runs: c3a_1, c3a_2, c3a_3, c3a_cab1, c3b_on1, c3b_off1, c3b_on_hidden1, c3b_off_hidden1, c3b_on_ltom1, c3b_off_ltom1, c2_w2, c2_w3 | graded 2026-11-22 (no new run) | per run as above | per run as above | |
| C4 base arms (sim) | arm4: `etc/runs/arm4_off.csv`, `arm4_on.csv`; arm7: `arm7_gyro-all.csv`, `arm7_native-all.csv`, `arm7_native-k.csv` | arm4 2026-08-31 22:14 / 22:30; arm7 2026-10-04 14:02 - 18:41 | not recorded | room_concept main | arm4: HEAD 2026-08-31 22:09 was afbb642e, the run note says an uncommitted `MapMode` edit was in the tree (`EXPERIMENT.md:269`); arm7 gyro-all re-run after the two calibrator fixes of **6bab57ee** (2026-10-04 18:22), csv 18:41 |
| C4 today (in progress) | c4_pilot (10:20 - 10:48, calibrator state in `tmp/heading/heading_2026-11-23_10-20-24.csv`, 27.6 min), c4_on_r1_1lap (10:48 start, 7.1 min), c4_on_r1 (resumed 10:55, 12.5 min, heading in `tmp/heading/heading_2026-11-23_10-55-45.csv`), c4_prev (old copy) | 2026-11-23 10:20 - 11:09 | no `agents.txt` | main (config path is relative `etc/config.toml`; fleet = room_concept residual_concept controller) | main HEAD 142b632d (2026-11-23 10:19, paper-only); conditions `cond=on MotionCalibApply=true mask=-1 bridge_GyroBias=+0.002` (`c4_conditions.txt`) |

Notes
- Between the set commits the product source did not change except as follows (`git diff --name-only`, excluding paper/montevideo/vygotskian/scripts/tests): 9af4002a -> e3f67218 -> edd267dc -> 6f8a850b nothing; 6f8a850b -> e5310a2a cabinet_concept (4 files); e5310a2a -> 3df281a0 chair_concept (config.toml, chair_config.{h,cpp}, chair_fitter.cpp, specificworker.cpp); 3df281a0 -> 565d9d4e -> fd0830ce -> 267db579 nothing.
  So the C2 warm runs (3df281a0) are the only furnished runs with the changed chair_concept; the C3a/C3b/C14/lambda runs share one chair, room, ltsm, controller and door code, except the cabinet-exit worktree agents (e5310a2a) noted above.
- "Uncommitted" state is not recorded per run, so an `agents.txt` commit is an upper bound on the code, not a guarantee. Known uncommitted pieces at the time: scripts (c3a_objects.py, nees_agents.py, ...), `chair_concept EpistemicOfferEnabled` flag and `ai2_log` t_ms column (memory `auro-c3b-change-test-11-21`, "UNCOMMITTED in main" on 11-21).
- `agents.txt` of c3a_1 lists 6f8a850b but c3a_3's countable logs were copies of c3a_2's until recovered (`nees_summary.md`), so check per-agent logs before quoting a commit for the countable agents of c3a_3.
- recog runs: their dates are from `gt_pose.tsv` or log mtimes (`*` in `runinfo.tsv` marks mtime-based dates); the commits are not derivable.
- Bench commit and apartment-campaign commit are inferred from the file dates and PROGRESS.md; not derivable from a record in the run.
