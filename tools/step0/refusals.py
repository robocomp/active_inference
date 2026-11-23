#!/usr/bin/env python3
"""Step 0, Step 3: why does the controller refuse an object standpoint?  refusals.py <crossing_dir> <run> [<run> ...]
For every `[why] STOPPED TARGET REFUSED` line in <run>/logs/controller.log:
  * the line number is turned into a time through gt_pose.tsv (column `controller` = controller.log line count at each
    1 Hz sample);
  * the robot pose at that time comes from proto_room.csv (x, y internal) and the logged gauge (internal -> published);
  * the offered standpoint is the `at (x,y)` in the line (room frame = published frame).
Output: per refusal the reason text class, the distance robot -> standpoint, and whether it is inside the footprint's
circumscribed circle (R_BODY m).
"""
import bisect, csv, math, os, re, sys
R_BODY = 0.3249   # ROBOT_GEOMETRY.md: footprint maximum support radius, -52 deg (0.3249) / +51 deg (0.3245)
PAT = re.compile(r"STOPPED TARGET REFUSED — '([^']+)' at \(([-0-9.]+),([-0-9.]+)\): (.*)")


def load(run):
    gp = [r for r in csv.DictReader(open(f'{run}/gt_pose.tsv'), delimiter='\t')]
    lines = [(int(r['controller']), int(r['t_ms'])) for r in gp]
    pr = []
    for r in csv.DictReader(open(f'{run}/proto_room.csv')):
        try: pr.append((int(r['ts_ms']), float(r['x']), float(r['y']), float(r['gauge_x']), float(r['gauge_y']), float(r['gauge_theta'])))
        except (KeyError, ValueError): pass
    pr = sorted(p for p in pr if p[0] > 1_000_000_000_000 and all(math.isfinite(v) for v in p[1:]))   # drop ts 0 / nan rows
    return lines, pr


def main(base, runs):
    tot = {}
    for run in runs:
        d = os.path.join(base, run)
        lines, pr = load(d)
        cn = [l[0] for l in lines]; ts = [p[0] for p in pr]
        for k, line in enumerate(open(f'{d}/logs/controller.log', errors='replace'), 1):
            m = PAT.search(line)
            if not m: continue
            name, sx, sy, why = m.group(1), float(m.group(2)), float(m.group(3)), m.group(4)
            cls = 'occupancy belief, object standpoint' if why.startswith('occupancy belief') else 'no route (afford_room)' if why.startswith('no route') else 'other'
            i = min(max(bisect.bisect_left(cn, k), 0), len(lines) - 1)
            t = lines[i][1]
            j = min(max(bisect.bisect_left(ts, t), 0), len(pr) - 1)
            _, x, y, g0, g1, gth = pr[j]
            px = math.cos(gth) * x - math.sin(gth) * y + g0
            py = math.sin(gth) * x + math.cos(gth) * y + g1
            dist = math.hypot(px - sx, py - sy)
            tot.setdefault(cls, []).append(dist)
            print(f'{run}\tline {k}\tt {t}\t{name}\tstandpoint ({sx:.2f},{sy:.2f})\trobot ({px:.2f},{py:.2f})\tdist {dist:.2f} m\t{cls}')
    print()
    for cls, ds in tot.items():
        ds.sort()
        print(f'{cls}: n {len(ds)}, inside footprint circle (<= {R_BODY} m): {sum(d <= R_BODY for d in ds)}, '
              f'dist min/median/max {ds[0]:.2f}/{ds[len(ds)//2]:.2f}/{ds[-1]:.2f} m')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
