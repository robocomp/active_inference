# lidar_measures_score.py report

Script: `countable_concept/tools/lidar_measures_score.py RUN_DIR [OFF_DIR] [--ref LIVE6_DIR] [--shift] [--pin] [--room NAME=x,y,rot]`.
The header of every output restates the definitions.

## What the numbers mean
- look: maximal run of fresh rows (gated==0 and lidar_only!=1) of one instance, no pkt_ts gap > 2 s.
- settled window: all rows from the first fresh row with range <= 2.5 m onward. Per DOF (cx, cy, H, h): rms of the true error, min/max logged std.
  PASS iff max std <= 0.15 m AND >= 95 % of settled rows have rms/3 <= std <= 3*rms. Also prints the fraction below rms/3 (overconfidence, owner question).
- NEES: (ex/std_cx)^2+(ey/std_cy)^2+(eH/std_H)^2 per row (diagonal Sigma), median; expected about 2.4.
- drift: last fitted centre of each look (world); step between consecutive looks projected on the direction to the nearest wall box and nearest other GT object; PASS iff largest <= 10 cm.
- rate: rows/(last-first pkt_ts) per instance, PASS >= 8 Hz (median-dt rate also shown).
- lidar_only share, medians of rho_cam/rho_L/rho_chain, and |dphi_cam|,|dz_cam| medians over parked rows (gt_pose speed < 0.02 m/s, yaw rate < 0.02 rad/s); n/a when the columns are absent.
- --shift: final w, h, H minus the TRUE size, PASS iff within 0.03 (w,h) / 0.05 (H), i.e. closer to the truth than to the shifted prior (+0.06/+0.10). h/w are matched to the GT footprint by the box yaw (swap when |sin yaw| > 0.707).
- Instances are matched to the nearest GT of their class; > 0.75 m from all = PHANTOM.

## GT and transform
GT: `countable_concept/tools/scene_threerooms.csv` (world). bed = union of mattress/footboard/headboard (centre -5.182,12.391; top = mattress 0.65); nightstands = body+legs (top 0.62); toilet = Webots node centre (-0.874,12.865) as the live6 comparison used, top 0.90 as in the live6 results doc (the tank top is 1.018).
The ai2_log is in the ROOM frame. The room frame origin is the door the robot entered by: bedroom = DOOR_1 (-4.769, 9.3796), rot 0; bathroom = DOOR_2 (-1.705, 10.77), rot -pi/2 (rotations from carve_replay_rays.py room_transforms). This is independent of the fits (carve_replay instead fitted the translation to the fits themselves). Check on live6: bed 6 cm, nightstands 11 and 7 cm. The bathroom origin is an ASSUMPTION: the toilet centre error under it is 0.35 m (live6 doc quoted ~0.21 m with a fitted placement); do not read the toilet centre error as exact. `--pin` fits each room translation to the class's own GT (placement not independent; then the toilet centre error is 0 by construction), `--room bathroom=x,y,rot` overrides.

## Caveats found on live6
- Gated rows grow std (prediction only), so the two-sided check FAILs on live6 for most DOFs (max std 0.4-1.6 m while gated at close range): expected for the flag-off build, this is what the flag-on run must cure.
- nightstand_7 never has a fresh-mask row within 2.5 m (min range 1.93 m but all gated): DOF checks not scored (FAIL line).
- Flag-off era logs lack all 16 new columns; handled (n/a). `--shift` on live6 is meaningless, only a smoke test.

## live6 output (no flags)
```
lidar_measures_score  (spec 2026-11-15 §8)
look = fresh-mask run, gap <= 2000 ms; settled = from the first fresh row with range <= 2.5 m
Sigma PASS = max std <= 0.15 m AND >= 95 % of settled rows with rms/3 <= std <= 3*rms (per DOF, rms of the true error)
drift PASS = largest step toward nearest wall / neighbour between looks <= 0.10 m;  rate PASS >= 8 Hz
room->world: bedroom origin (-4.769, 9.380) rot 0.0000; bathroom origin (-1.705, 10.770) rot -1.5708
GT: bed (-5.182, 12.391) 1.75x2.12 top 0.65; nightstand_0 (-6.361, 13.216) 0.50x0.51 top 0.62; nightstand_1 (-4.003, 13.216) 0.50x0.51 top 0.62; toilet (-0.874, 12.865) 0.58x0.73 top 0.90
====================================================================================================
RUN RUN /home/pbustos/robocomp/components/active_inference/room_concept/estimation_scripts/crossing/objects_live6
  note: bed: columns absent (flag-off era): lidar_only,lidar_msum,rho_cam...
  note: nightstand: columns absent (flag-off era): lidar_only,lidar_msum,rho_cam...
  note: toilet: columns absent (flag-off era): lidar_only,lidar_msum,rho_cam...
----------------------------------------------------------------------------------------------------
bed_10 -> GT bed (-5.182, 12.391, top 0.65)  rows 951  looks 5  settled rows 904
  final centre error      0.060 m  (live6 ref: bed 0.055, nightstands 0.02-0.06, toilet 0.23)  [info]
  final |H - true top|    0.032 m  (H 0.682, top 0.65)  [info; compare with the live6 column]
  cx rms 0.0825  std min 0.0242 max 0.4280  inside[rms/3,3rms]  66.2% (below   0.9%)   FAIL
  cy rms 0.0628  std min 0.0361 max 1.1227  inside[rms/3,3rms]  12.6% (below   0.0%)   FAIL
  H  rms 0.0397  std min 0.0422 max 0.5389  inside[rms/3,3rms]  11.8% (below   0.0%)   FAIL
  h  rms 0.0846  std min 0.0636 max 1.5966  inside[rms/3,3rms]  11.7% (below   0.0%)   FAIL
  median NEES (3 dof: cx,cy,H; expected ~2.4)  0.13  [info]
  largest centre step toward wall between looks 0.048 m (largest step 0.155 m)   PASS
  fit rate 12.1 Hz (median-dt 9.9 Hz)   PASS
  lidar_only / rho / dphi / dz: n/a (columns absent)  [info]
----------------------------------------------------------------------------------------------------
nightstand_6 -> GT nightstand_1 (-4.003, 13.216, top 0.62)  rows 852  looks 4  settled rows 847
  final centre error      0.109 m  (live6 ref: bed 0.055, nightstands 0.02-0.06, toilet 0.23)  [info]
  final |H - true top|    0.003 m  (H 0.623, top 0.62)  [info; compare with the live6 column]
  cx rms 0.1639  std min 0.0176 max 0.0757  inside[rms/3,3rms]  29.8% (below  70.2%)   FAIL
  cy rms 0.0540  std min 0.0531 max 1.2273  inside[rms/3,3rms]  21.0% (below   0.0%)   FAIL
  H  rms 0.0481  std min 0.0289 max 0.0816  inside[rms/3,3rms] 100.0% (below   0.0%)   PASS
  h  rms 0.0228  std min 0.0976 max 1.6521  inside[rms/3,3rms]   0.0% (below   0.0%)   FAIL
  median NEES (3 dof: cx,cy,H; expected ~2.4)  11.33  [info]
  largest centre step toward bed between looks 0.050 m (largest step 0.081 m)   PASS
  fit rate 10.6 Hz (median-dt 9.9 Hz)   PASS
  lidar_only / rho / dphi / dz: n/a (columns absent)  [info]
----------------------------------------------------------------------------------------------------
nightstand_7 -> GT nightstand_0 (-6.361, 13.216, top 0.62)  rows 740  looks 3  settled rows 0
  final centre error      0.071 m  (live6 ref: bed 0.055, nightstands 0.02-0.06, toilet 0.23)  [info]
  final |H - true top|    0.010 m  (H 0.630, top 0.62)  [info; compare with the live6 column]
  no fresh-mask row from within 2.5 m: DOF checks not scored   FAIL
  largest centre step toward bed between looks 0.018 m (largest step 0.074 m)   PASS
  fit rate 10.3 Hz (median-dt 9.9 Hz)   PASS
  lidar_only / rho / dphi / dz: n/a (columns absent)  [info]
----------------------------------------------------------------------------------------------------
toilet_6 -> GT toilet (-0.874, 12.865, top 0.90)  rows 580  looks 3  settled rows 580
  final centre error      0.351 m  (live6 ref: bed 0.055, nightstands 0.02-0.06, toilet 0.23)  [info]
  final |H - true top|    0.134 m  (H 0.766, top 0.90)  [info; compare with the live6 column]
  cx rms 0.2088  std min 0.0185 max 0.1244  inside[rms/3,3rms]  29.8% (below  70.2%)   FAIL
  cy rms 0.2471  std min 0.0186 max 0.1357  inside[rms/3,3rms]  10.9% (below  89.1%)   FAIL
  H  rms 0.3051  std min 0.0615 max 0.8175  inside[rms/3,3rms]  84.0% (below  16.0%)   FAIL
  h  rms 0.1286  std min 0.0219 max 0.0768  inside[rms/3,3rms]   7.2% (below  92.8%)   FAIL
  median NEES (3 dof: cx,cy,H; expected ~2.4)  32.36  [info]
  largest centre step toward wall between looks 0.044 m (largest step 0.074 m)   PASS
  fit rate 14.3 Hz (median-dt 10.0 Hz)   PASS
  lidar_only / rho / dphi / dz: n/a (columns absent)  [info]
====================================================================================================
SUMMARY (RUN): 4 object instance(s), 11 DOF std-check FAIL line(s), 0 drift FAIL, 0 rate FAIL
```
