#!/usr/bin/env python3
"""Open/close a Webots door through the bridge's DoorControl (port 10017).  door.py open|close [DOOR_1] [open_deg]
   open_deg: how far an OPEN swings the leaf (the bridge's apertureHint); absent = the bridge's default (90 deg)."""
import sys, Ice
Ice.loadSlice('-I/home/pbustos/robocomp/components/webots-shadow/components/shadow-bridge-dds/generated --all '
              '/home/pbustos/robocomp/components/webots-shadow/components/shadow-bridge-dds/generated/DoorControl.ice')
import RoboCompDoorControl as D
import math
action, door = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else 'DOOR_1')
hint = -math.radians(float(sys.argv[3])) if len(sys.argv) > 3 and action == 'open' else 0.0
with Ice.initialize(sys.argv[:1]) as ic:
    p = D.DoorControlPrx.checkedCast(ic.stringToProxy('doorcontrol:tcp -h localhost -p 10017'))
    req = D.DoorRequest(doorId=door, door=D.DoorRef(id=door, pose=D.Pose2D(0, 0, 0), width=1.0),
                        action=D.DoorAction.OpenDoor if action == "open" else D.DoorAction.CloseDoor,
                        apertureHint=hint, purpose='crossing test reset')
    ack = p.requestDoor(req)
    print(door, action, 'accepted' if ack.accepted else f'refused {ack.refusal}')
