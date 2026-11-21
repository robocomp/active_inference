# cabinet_concept does not release its instances on a room change — diagnosis

Run: `room_concept/estimation_scripts/crossing/c3a_1/` (main checkout). Code: branch `fix/cabinet-exit-release`
off main `6f8a850b`.

## What the run shows

`graph_trace.csv`, the crossing room_1 → room_2 (ltsm moves `current` at t=…939.303):

```
939.303 edge_insert  ltsm(19)  Shadow -> room_2  current
939.359 attr         cabinet(21) cabinet_w13_base  parent = room_2   (node id 30813670392987669, born under room_1 at 778.787)
939.360 edge_insert  cabinet(21) room_2 -> cabinet_w13_base  RT
939.375 …            cabinet_w14_base  parent = room_2, RT from room_2
939.383 …            cabinet_w14_up    parent = room_2, RT from room_2
939.389 …            cabinet_peninsula parent = room_2, RT from room_2
```

The same node ids, not new births: the four kitchen runs of room_1 are MOVED under room_2 within 60 ms of the
crossing. At the next crossing (room_3 current at 181.304) `cabinet_w13_base` is moved again, under room_3 (181.377).
In the same window every other object agent (hood, bed, refrigerator, chair, …) deletes its room_1 nodes.

`logs/cabinet_concept.log`: the `[kitchen] built 64 cells over 32 walls` line appears ONCE (room_1, t=777.990); at the
crossing only `[room] room 30813938811666437 outline 8 verts` is printed — the polygon changed, the cell table did not.
No `[room] '…' left behind in another room, dropped` line is ever printed.

`logs/room_concept.log:741` / `:845`: `archived room … not reaped yet: a foreign node holds it: cabinet_w13_base
(object) under room_1` (then `under room_2`). room_concept cannot reap an archived room that still has a child, so
ltsm never sees the room behind the door as archived ("door_24 gets no recall: the room behind ('room_2') is live"),
the controller fails with `door_unresolved`, and the tour stops at 2/4 legs (the control run c3a_2 without cabinet:
4/4).

## Why (code)

cabinet runs in **kitchen mode** (`etc/config.toml: KitchenModel = true`). `SpecificWorker::compute()`:

```cpp
room_node_id_ = rc::room::follow_current_room(room_node_id_, rc::room::current_room(*G));   // follows the robot
…
if (cfg_.kitchen_model) { run_kitchen_model(); }
else {
    run_instance_tracker();
    …
    rc::room::drop_left_behind(*G, "cabinet", room_node_id_, …);   // ← the release lives ONLY here
    …
}
```

1. **The release (fix 1) is in the wrong branch.** Phase 0 (2026-11-13) added the shared release
   `rc::room::drop_left_behind` to cabinet, but inside the CLASSIC pipeline's `else`. The mode the agent actually runs
   in never calls it. (The same file already records this exact defect for the Ricoh channel: "it sat inside the
   classic-only branch, so the mode this agent actually runs in never called it once".) The conformance row says
   "room follows robot ✅"; true for the classic path only.

2. **The re-hanging (fix 2) is the kitchen model being a model of ONE room that is never re-bound.**
   `run_kitchen_model()` builds `kitchen_mgr_` (the (wall,tier) cell table) once, `if (not kitchen_mgr_.built())`,
   over `fitter_->kitchen_walls()` — room_1's 32 walls. Its cells' wall charts and run beliefs are in room_1's frame.
   Nothing ever unbuilds it. After the crossing, `publish_kitchen_boxes()` writes every active cell's box with
   `room_node_id_` — which now follows the robot to room_2:

   ```cpp
   auto room_opt = G->get_node(room_node_id_);                       // room_2 now
   …
   rt_api_->insert_or_assign_edge_RT(room_opt.value(), it->second, {b.cx, b.cy, b.z0}, …);   // existing node
   ```

   `insert_or_assign_edge_RT` on an existing node with a new parent moves the node: it rewrites `parent` and hangs the
   RT edge from room_2 (the trace's `attr parent = room_2` + `edge_insert room_2 -> cabinet_*`, both by agent 21). So
   room_1's runs, with room_1-frame coordinates, are re-parented under a room where they were never observed. The
   runs then keep receiving room_2's LiDAR as absence evidence and are refuted one by one (`RETIRE cell` at 940.9–941.1),
   but `cabinet_w13_base` survived long enough to be carried into room_3 as well.

   Making `room_node_id_` follow the robot (Phase 0) was correct; it is the kitchen model that still silently assumed
   a latched room. The working room changed under a model that was built for the previous one.

## The fix, mirroring the convention

- Hoist the shared `rc::room::drop_left_behind(*G, "cabinet", room_node_id_, forget)` out of the classic branch so it
  runs in both modes, its `forget` also dropping the kitchen node bookkeeping. (Classic: unchanged behaviour.
  Kitchen: the generic backstop for any `cabinet*` node whose parent names another room.)
- Bind the kitchen model to the room it was built over (`kitchen_room_`, set when the cell table is built). When that
  room is left behind (`rc::room::left_behind`, the same rule), release it: delete every kitchen run node, forget the
  cell → node map and covariance gates, and reset `kitchen_mgr_` so it rebuilds over the NEW room's walls once that
  room has a usable outline (births WAIT until then, as everywhere else).
- `publish_kitchen_boxes()` parents to the room the model was built over, and publishes nothing while that is not the
  working room. The model can no longer write a run under a room it was never observed in, whatever the order of
  events within a cycle.

Kitchen runs carry no `aff_*` affordances (the trace shows none; affordances belong to the classic instances, whose
`forget` already removes them). Kitchen nodes carry no `object_uid`, so ltom does not recall them: on return to the
kitchen the runs are re-learnt from the masks (the cell table is rebuilt), which is what the agent does at startup.
Graceful-exit cleanup (`cleanup_owned_nodes`) and the startup stale-sweep (`remove_owned_cabinet_nodes`, initialize +
first Operating) are untouched.
