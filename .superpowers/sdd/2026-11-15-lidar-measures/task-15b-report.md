# Task 15b report: label dump + eps_px measurement (spec 7c.8 a0)

## What
Part 1 (C++, config-gated `Model.LabelDumpEnabled`, default false; needs `Model.LidarMeasuresOnGated`):
- countable_config.h/.cpp: key `label_dump_enabled`.
- countable_mask_membership.h/.cpp: NEW sibling `make_membership_components` -> `MembershipComponents{in_image,s_px,sigma_px,pi_mask,q}`; make_membership untouched.
- countable_lidar_range_channel.h/.cpp: `helios_returns_in_select_box(anchor_cloud)` (same box as select_into, same centroid, now via shared `cloud_centroid`, same summation order so feed() is bit-identical).
- countable_fitter.h/.cpp: build_membership also fills `label_components_` when the flag is on (same projector/sigma_chain/mot_pos_var); `dump_labels()` called just before each `lidar_channel_.feed` (gated LiDAR-only path: gated=1; fresh path: gated=0) when membership is non-empty. File `label_dump.csv` in the directory of Model.AI2CsvPath, via rc::diag::open_rotating (like timing_log.csv), classic locale, flushed per cycle. Row: pkt_ts (mask stamp),node,cls,gated,ex,ey,ez (room frame),in_image,s_px,sigma_px,pi_mask,q.
- etc/config.toml of nightstand, bed, toilet: `LabelDumpEnabled = true` (controller turns off afterwards).
- tests/config_migration_test.cpp: the new key added to the `produced` allow-list (otherwise the test fails "came from no old key"). Remember to remove it together with the config line.
Part 2: countable_concept/tools/eps_px_measure.py.

## Tests
- Built countable_concept (single binary shared by the three agent dirs via symlink; there is no per-agent CMake) with make -j8: OK.
- tests: cmake -B build && make -j8 && ctest: 23/23 pass incl. lidar_golden_test (bit-exact, flag off), mask_membership_test, beam_ray_test (other subagent's, was present and passing).
- Script smoke-tested on a synthetic 4-row dump (on/foreign/in_image=0 classification right); bstar: eps .01 -> 0.022, .02 -> 0.035, .05 -> 0.051, <=.005 -> none (neutral at b=0).
- The C++ dump itself is NOT run live (no agent run, per instructions).

## Run the script
    cd countable_concept/tools
    python3 -I eps_px_measure.py RUN_DIR [--scene SCENE_CSV] [--room bedroom=x,y,rot --room bathroom=x,y,rot]
RUN_DIR needs <cls>_label_dump.csv, <cls>_ai2_log.csv, gt_error.csv + proto_room.csv (room->world fit as in lidar_measures_score) and gt_pose.tsv. Prints per class: counts, eps_out, eps_in, eps_px, not-in-image share, b*(eps_px).

## Concerns
- Room frame fit comes from fit_room_frames (gt_error.csv + proto_room.csv); a room whose fit fails is skipped unless --room is given. Fit rms/accepted threshold is not enforced here (an 'error' fit with rms>5 cm is also skipped).
- Instance->GT association uses the last ai2_log centre (falls back to median return position).
- Printed b* is "none" when eps/(1-eps)*St(0)/p_pass + G(0) <= 1 (i.e. eps <~ 0.5-0.6%), also printed for eps=1 degenerate cases.
- "G(-b)" interpreted as the standard normal density phi(b).
- Dump is stamped with the mask timestamp (last_mask_timestamp_ms), not the sweep stamp.
