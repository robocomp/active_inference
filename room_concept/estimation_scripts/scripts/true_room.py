#!/usr/bin/env python3
"""The robot's TRUE room from the ground-truth track (gt_pose.tsv: t_ms x y yaw ...), threerooms-piso.wbt: start in room_1;
crossing DOOR_1's line (y = 9.3796, |x - -4.769| < 0.7) toggles room_1 <-> room_2; crossing DOOR_2's line (x = -1.705,
|y - 10.77| < 0.7) toggles room_2 <-> room_3. Rooms are not convex boxes in world coordinates, so a region test misgrades
(warm8 leg 2: the robot in room_3 at (-2.39, 8.47)); the doors are the only places the room can change.
usage: true_room.py gt_pose.tsv"""
import csv, sys
D1Y, D1X, D2X, D2Y, HALF = 9.3796, -4.769, -1.705, 10.77, 0.7
room, prev = 'room_1', None
for r in csv.DictReader(open(sys.argv[1]), delimiter='\t'):
    try: x, y = float(r['x']), float(r['y'])
    except (KeyError, ValueError): continue
    if prev is not None:
        px, py = prev
        if (py - D1Y) * (y - D1Y) < 0 and abs(0.5 * (x + px) - D1X) < HALF:
            room = {'room_1': 'room_2', 'room_2': 'room_1'}.get(room, room)
        if (px - D2X) * (x - D2X) < 0 and abs(0.5 * (y + py) - D2Y) < HALF:
            room = {'room_2': 'room_3', 'room_3': 'room_2'}.get(room, room)
    prev = (x, y)
print(room)
