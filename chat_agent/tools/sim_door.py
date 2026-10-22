#!/usr/bin/env python3
"""sim_door.py open|close X_M Y_M MAX_M [ENDPOINT] — chat_agent's sim_door tool: ask the simulator's DoorControl
(webots-bridge) to open or close the door nearest (X_M, Y_M), a point in the WEBOTS WORLD frame in metres.

chat_agent computes that point (src/sim_door.h: the door's centre in the room frame, through the sim ground-truth
robot pose). This script matches it against the doors the provider ADVERTISES (getCapabilities, mm), refuses when the
nearest is more than MAX_M away (it would open some other door), and quotes the chosen id in requestDoor — the same
"nearest by place" rule as door_concept's actuator. Prints ONE JSON line:
  {"ok": bool, "id": "DOOR_1", "dist_m": 0.12, "runner_up_m": 4.3, "refusal": "UnknownDoor", "error": "..."}
Exit code 0 whenever the JSON was printed (ok false included)."""
import json
import math
import sys
from pathlib import Path

out = {"ok": False}
try:
    import Ice
    Ice.loadSlice(str(Path(__file__).resolve().parent / "DoorControl.ice"))
    import RoboCompDoorControl as D

    action, x, y, max_m = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    endpoint = sys.argv[5] if len(sys.argv) > 5 else "doorcontrol:tcp -h localhost -p 10017"
    if action not in ("open", "close"):
        raise ValueError(f"action must be open or close, not {action!r}")
    with Ice.initialize(["sim_door", "--Ice.Default.Timeout=2000"]) as ic:
        p = D.DoorControlPrx.checkedCast(ic.stringToProxy(endpoint))
        if p is None:
            raise RuntimeError(f"no DoorControl at {endpoint}")
        caps = p.getCapabilities()
        if not caps.canActuate or not caps.doors:
            raise RuntimeError("the simulator advertises no door it can move")
        by_dist = sorted((math.hypot(d.pose.x / 1000.0 - x, d.pose.y / 1000.0 - y), d) for d in caps.doors)
        dist, door = by_dist[0][0], by_dist[0][1]
        out.update(id=door.id, dist_m=round(dist, 3), runner_up_m=round(by_dist[1][0], 3) if len(by_dist) > 1 else -1)
        if dist > max_m:
            out["error"] = (f"the nearest simulator door ({door.id}) is {dist:.2f} m from where this door is believed "
                            f"to be (> {max_m:.2f} m): not moving some other door")
        else:
            req = D.DoorRequest(doorId=door.id, door=door,
                                action=D.DoorAction.OpenDoor if action == "open" else D.DoorAction.CloseDoor,
                                apertureHint=0.0, purpose="chat_agent sim_door")
            ack = p.requestDoor(req)
            out["ok"] = bool(ack.accepted)
            if not ack.accepted:
                out["refusal"] = str(ack.refusal)
except Exception as e:   # one JSON line, always: the chat reports it in words
    out["error"] = f"{type(e).__name__}: {e}"
print(json.dumps(out))
