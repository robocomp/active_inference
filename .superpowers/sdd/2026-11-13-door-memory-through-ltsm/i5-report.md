# I5 report
Changed: ltsm_agent/src/room_eviction.{h,cpp}, tools/remembered_doors_test.cpp.
- twin_attrs() free helper = the one attribute filter, used by twin() and the refresh.
- RoomEviction::refresh_door_twin(): write-on-change (DSR::Attribute operator==; pose compared as the rt_attrs an add would write); node update + RT edge (memory room -> twin), only when something differs.
- sync_found_doors(): known door = synced_doors_ name, else place match (previously `continue`); refreshed unless held(), staged, or robot under >1 room. "Another hole" branch untouched. Doc comment updated.
- Pose is refreshed only when the twin's parent is the memory room itself (a twin hung from a wall keeps its wall-frame pose; avoids a second RT parent). Attributes still refresh.
Tests: remembered_doors_test ALL OK (0 failures), incl. 4 new (attrs, write-on-change, pose, held). Agent build `make -C build -j8`: OK.

## Fix round 1
Pose refresh now also covers a wall-hung twin: parent kept, parent->twin edge = T(room<-parent)^-1 * T(room<-live door), planar; T(room<-parent) via pose_in_room(*mem_, ...) (composes memory RT edges, no inner-eigen cache). Unresolved parent => pose skipped, attrs still refreshed. Edge compared exactly, written only on difference; only an existing parent edge is overwritten (never a second parent). New test sync_refreshes_pose_of_wall_hung_twin passes; remembered_doors_test ALL OK; agent build OK.

## Fix round 2
The pose refresh now starts from the EXISTING edge and overrides only x, y, yaw (z, roll, pitch, rt_covariance and every other attr kept); compared on x/y/yaw only. Tests added: sync_refresh_keeps_edge_covariance, staged guard, two-rooms guard; sync_refreshes_pose also asserts y and yaw. remembered_doors_test ALL OK; agent build OK.

## Fix round 3
refresh_door_twin decodes the stored rotation as R = Rz(rz)Ry(ry)Rx(rx) (rt_attrs' convention), takes planar yaw by atan2, and writes R_new = Rz(dyaw)*R_old re-encoded with eulerAngles(2,1,0).reverse() (tilt kept). pose_in_room already extracts yaw by atan2 (se2_of); no change needed there. The yaw comparison uses a float-precision tolerance (8*FLT_EPSILON on the wrapped difference), NOT a modelling threshold: a flipped triple decodes only to ~1e-7, so an exact test would rewrite on every sync. x,y compared exactly in float. Test sync_refresh_handles_flipped_euler (room-hung and wall-hung) passes with repeat sync == 0; remembered_doors_test ALL OK; agent build OK.
