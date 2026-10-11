#!/usr/bin/env python3
"""Teleport the Webots robot through webots-bridge.  tp.py get | tp.py set x y theta   (world metres / rad, z kept)"""
import sys, math, Ice
Ice.loadSlice('-I/home/pbustos/robocomp/components/webots-bridge/generated --all /home/pbustos/robocomp/components/webots-bridge/generated/Webots2Robocomp.ice')
import RoboCompWebots2Robocomp as W
with Ice.initialize(sys.argv[:1]) as ic:
    p = W.Webots2RobocompPrx.checkedCast(ic.stringToProxy('webots2robocomp:tcp -h localhost -p 10006'))
    cur = p.getObjectPose('shadow')
    q = cur.orientation
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    if sys.argv[1] == 'get':
        print(f'{cur.position.x/1000:.4f} {cur.position.y/1000:.4f} {cur.position.z/1000:.4f} yaw {yaw:.4f}  q=({q.x:.4f},{q.y:.4f},{q.z:.4f},{q.w:.4f})')
    else:
        x, y, th = map(float, sys.argv[2:5])
        pose = W.ObjectPose(W.Vector3(x * 1000, y * 1000, cur.position.z + 20.0),   # 2 cm up: never spawn into the floor
                            W.Quaternion(0.0, 0.0, math.sin(th / 2), math.cos(th / 2)))
        p.setObjectPose('shadow', pose)
        print('sent', x, y, th)
