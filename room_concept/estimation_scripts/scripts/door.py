#!/usr/bin/env python3
"""Open/close a Webots door through the bridge's DoorControl (port 10017).  door.py open|close [DOOR_1]"""
import sys, Ice
Ice.loadSlice('-I/home/pbustos/robocomp/components/webots-shadow/components/shadow-bridge-dds/generated --all '
              '/home/pbustos/robocomp/components/webots-shadow/components/shadow-bridge-dds/generated/DoorControl.ice')
import RoboCompDoorControl as D
action, door = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else 'DOOR_1')
with Ice.initialize(sys.argv[:1]) as ic:
    p = D.DoorControlPrx.checkedCast(ic.stringToProxy('doorcontrol:tcp -h localhost -p 10017'))
    req = D.DoorRequest(doorId=door, door=D.DoorRef(id=door, pose=D.Pose2D(0, 0, 0), width=1.0),
                        action=D.DoorAction.OpenDoor if action == "open" else D.DoorAction.CloseDoor,
                        apertureHint=0.0, purpose='crossing test reset')
    ack = p.requestDoor(req)
    print(door, action, 'accepted' if ack.accepted else f'refused {ack.refusal}')
