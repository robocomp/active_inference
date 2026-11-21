#!/usr/bin/env python3
"""Read or move a Webots object by DEF through the bridge's Webots2Robocomp (port 10006).
   webots_object.py get CHAIR_3              -> prints "x y z qx qy qz qw" (metres, world frame)
   webots_object.py set CHAIR_3 x y [yaw_rad] -> moves it (z and, if no yaw is given, orientation unchanged)
The bridge works in MILLIMETRES with a unit quaternion (shadow-bridge-dds setObjectPose/getObjectPose)."""
import math, sys, Ice
G = '/home/pbustos/robocomp/components/webots-shadow/components/shadow-bridge-dds/generated'
Ice.loadSlice(f'-I{G} --all {G}/Webots2Robocomp.ice')
import RoboCompWebots2Robocomp as W

with Ice.initialize(sys.argv[:1]) as ic:
    p = W.Webots2RobocompPrx.checkedCast(ic.stringToProxy('webots2robocomp:tcp -h localhost -p 10006'))
    cmd, dfn = sys.argv[1], sys.argv[2]
    cur = p.getObjectPose(dfn)
    if cmd == 'get':
        q = cur.orientation
        print(f'{cur.position.x / 1000:.4f} {cur.position.y / 1000:.4f} {cur.position.z / 1000:.4f} '
              f'{q.x:.6f} {q.y:.6f} {q.z:.6f} {q.w:.6f}')
    elif cmd == 'set':
        x, y = float(sys.argv[3]) * 1000, float(sys.argv[4]) * 1000
        q = cur.orientation
        if len(sys.argv) > 5:
            yaw = float(sys.argv[5])
            q = W.Quaternion(0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
        p.setObjectPose(dfn, W.ObjectPose(W.Vector3(x, y, cur.position.z), q))
        n = p.getObjectPose(dfn)
        print(f'{dfn} -> {n.position.x / 1000:.4f} {n.position.y / 1000:.4f}')
