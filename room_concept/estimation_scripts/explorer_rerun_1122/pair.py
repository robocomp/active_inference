#!/usr/bin/env python3
"""Pair the current coverage explorer (newroom harness) with the earlier run_explore arms on the 594 MatterportLayout plans.
usage: pair.py   (run in this directory). Writes per_plan.csv, prints paired differences with 95% CIs.
IoU compared at 3 decimals (run_explore's log prints 3); ties = equal at 3 decimals. Failures (IoU 0.000 sentinel) dropped
pairwise and counted."""
import csv, re, statistics as st, math

pat = re.compile(r'room\s+(\d+)\s+(\S+)\s+corners\s+(\d+)\s+IoU\s+([\d.]+)\s+cover\s+([\d.]+)')
def old(path):
    return {int(m.group(1)): (m.group(2), int(m.group(3)), float(m.group(4)), float(m.group(5)))
            for m in map(pat.search, open(path, errors='ignore')) if m}
def new(path):
    return {int(r['room_id']): (round(float(r['iou']), 3), float(r['seen_frac']), int(r['frames_run']), r['stop'])
            for r in csv.DictReader(open(path))}

E, H, N9, N8 = old('old_eig/run.log'), old('old_heur/run.log'), new('newroom_matterport594_f900.csv'), new('newroom_matterport594_f8000.csv')
with open('per_plan.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['room', 'plan', 'corners', 'iou_current_f900', 'seen_current_f900', 'frames_current_f900', 'stop_current_f900',
                'iou_current_f8000', 'seen_current_f8000', 'iou_old_eig', 'cover_old_eig', 'iou_old_heur', 'cover_old_heur'])
    for i in sorted(N9):
        e, h, n8 = E.get(i), H.get(i), N8.get(i)
        w.writerow([i, e[0] if e else '', e[1] if e else '', *N9[i], n8[0] if n8 else '', n8[1] if n8 else '',
                    e[2] if e else '', e[3] if e else '', h[2] if h else '', h[3] if h else ''])

def pair(name, a, b, idx):
    ks = [k for k in a if k in b]
    fail = sum(1 for k in ks if a[k][idx] == 0 or b[k][idx] == 0)
    d = [a[k][idx] - b[k][idx] for k in ks if a[k][idx] > 0 and b[k][idx] > 0]
    n = len(d); m = st.mean(d); se = st.stdev(d) / math.sqrt(n)
    wins = sum(x > 0 for x in d); loss = sum(x < 0 for x in d)
    print(f'{name}: n={n} (failures dropped {fail})  mean diff {m:+.4f}  95% CI [{m-1.96*se:+.4f}, {m+1.96*se:+.4f}]  '
          f'median {st.median(d):+.4f}  wins/ties/losses {wins}/{n-wins-loss}/{loss}')

cur9 = {k: (v[0],) for k, v in N9.items()}; cur8 = {k: (v[0],) for k, v in N8.items()}
e_ = {k: (v[2],) for k, v in E.items()}; h_ = {k: (v[2],) for k, v in H.items()}
for nm, dd in [('current f900', N9), ('current f8000', N8)]:
    iou = [v[0] for v in dd.values()]; seen = [v[1] for v in dd.values()]; fr = [v[2] for v in dd.values()]
    print(f'{nm}: n={len(dd)} IoU mean {st.mean(iou):.4f} median {st.median(iou):.4f} min {min(iou):.3f}; seen mean {st.mean(seen):.3f}; '
          f'frames median {st.median(fr)} max {max(fr)}; stopped itself {sum(v[3] != "budget" for v in dd.values())}')
for nm, dd in [('old EIG', E), ('old heuristic', H)]:
    iou = [v[2] for v in dd.values() if v[2] > 0]
    print(f'{nm}: n={len(dd)} IoU mean {st.mean(iou):.4f} median {st.median(iou):.4f} failures {sum(v[2] == 0 for v in dd.values())}; '
          f'cover mean {st.mean(v[3] for v in dd.values()):.3f}')
pair('current(f900) - old EIG', cur9, e_, 0)
pair('current(f900) - old heuristic', cur9, h_, 0)
pair('old EIG - old heuristic (reproduces 09-14)', e_, h_, 0)
pair('current f8000 - f900', cur8, cur9, 0)
