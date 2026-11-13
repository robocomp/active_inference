#!/usr/bin/env python3
"""Which of the world's doors the robot has crossed, from the ground-truth track (gt_pose.tsv: t_ms x y yaw ...),
threerooms-piso.wbt — the same door lines as true_room.py: DOOR_1 y = 9.3796 near x = -4.769, DOOR_2 x = -1.705 near
y = 10.77, DOOR_0 y = 7.2256 near x = -3.593; a crossing counts within the door's half-width (0.7 m) of its centre.
The exploration tour's completion criterion (owner 2026-11-11): every door crossed at least once.
usage: doors_crossed.py gt_pose.tsv   -> "k/3 DOOR_a,DOOR_b"   (k crossed of the 3)"""
import csv, sys
DOORS = {'DOOR_1': ('y', 9.3796, -4.769), 'DOOR_2': ('x', -1.705, 10.77), 'DOOR_0': ('y', 7.2256, -3.593)}
HALF = 0.7
crossed, prev = set(), None
try:
    for r in csv.DictReader(open(sys.argv[1]), delimiter='\t'):
        try: x, y = float(r['x']), float(r['y'])
        except (KeyError, ValueError): continue
        if prev is not None:
            px, py = prev
            for name, (axis, line, along) in DOORS.items():
                a0, a1, b = (py, y, 0.5 * (x + px)) if axis == 'y' else (px, x, 0.5 * (y + py))
                if (a0 - line) * (a1 - line) < 0 and abs(b - along) < HALF: crossed.add(name)
        prev = (x, y)
except FileNotFoundError:
    pass
print(f"{len(crossed)}/{len(DOORS)} {','.join(sorted(crossed))}")
