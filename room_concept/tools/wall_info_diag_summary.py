#!/usr/bin/env python3
"""
wall_info_diag_summary.py — read tmp/wall_info_diag.csv (RoomConcept::log_wall_information_diag).

Question it answers: does the publish decision fail ONLY because it reads the walls' CARRIED information,
which excludes the live window? If so, over a parked stretch:
  - worst_corner_sigma_carried stays at the rectangle-prior value (~1.02 m), publishable_carried = 0;
  - worst_corner_sigma_combined drops below publish_corner_sigma within the first frames, publishable_combined = 1;
  - the window never fills past its first slot (window_slots stays low) because stride_replace overwrites it;
  - combined sigma does NOT keep shrinking frame after frame while parked (no double counting of repeats).

  python3 tools/wall_info_diag_summary.py [tmp/wall_info_diag.csv]
"""
import csv
import sys
from collections import OrderedDict

path = sys.argv[1] if len(sys.argv) > 1 else 'tmp/wall_info_diag.csv'
frames = OrderedDict()
with open(path) as f:
    for r in csv.DictReader(f):
        frames.setdefault(int(r['ts_ms']), []).append(r)
if not frames:
    sys.exit(f'no rows in {path}')

t0 = next(iter(frames))
print(f'{len(frames)} frames, {(next(reversed(frames)) - t0) / 1000:.1f} s\n')
print(f'{"t":>6} {"slots":>5} {"assoc":>5} | {"worst σ carried":>15} {"pub":>3} | {"worst σ combined":>16} {"pub":>3} | '
      f'per-wall σ_d carried -> combined (m)')
step = max(1, len(frames) // 40)
first_pub_combined = None
first_pub_carried = None
for i, (ts, rows) in enumerate(frames.items()):
    r0 = rows[0]
    if first_pub_combined is None and r0['publishable_combined'] == '1':
        first_pub_combined = ts
    if first_pub_carried is None and r0['publishable_carried'] == '1':
        first_pub_carried = ts
    if i % step and i != len(frames) - 1:
        continue
    walls = ' '.join(f'{float(r["carried_sigma_d"]):.3f}->{float(r["combined_sigma_d"]):.4f}' for r in rows)
    print(f'{(ts - t0) / 1000:6.1f} {r0["window_slots"]:>5} {r0["slots_with_assoc"]:>5} | '
          f'{float(r0["worst_corner_sigma_carried"]):15.3f} {r0["publishable_carried"]:>3} | '
          f'{float(r0["worst_corner_sigma_combined"]):16.4f} {r0["publishable_combined"]:>3} | {walls}')

bar = float(next(iter(frames.values()))[0]['publish_corner_sigma'])
print(f'\npublish bar: {bar} m')
print(f'first publishable using CARRIED information : '
      + ('never' if first_pub_carried is None else f'{(first_pub_carried - t0) / 1000:.2f} s'))
print(f'first publishable using CARRIED + WINDOW    : '
      + ('never' if first_pub_combined is None else f'{(first_pub_combined - t0) / 1000:.2f} s'))
