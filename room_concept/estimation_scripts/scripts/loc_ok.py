#!/usr/bin/env python3
"""Is room_concept's localisation good enough to act on? A TOUR-HARNESS check (user 2026-11-01, warm12: the pose drifted
0.2-0.4 m in room_1 while driving, the door was misplaced and the crossing failed; "check confidence is good enough to proceed").
Reads the newest tmp/agent_events_*.jsonl and takes the last `window` s of `state` records. Measured on warm11/warm12 against
ground truth: the scan's median residual sdf_med tracks the pose error (0.01 m at 0.09 m error; 0.10-0.24 m at 0.2-0.4 m).
Prints one line; exit 0 = OK.   usage: loc_ok.py <room_concept_dir> [max_sdf_med=0.03] [window_s=5]"""
import glob, json, os, statistics, sys
d = sys.argv[1]; max_sdf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.03; win = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
files = sorted(glob.glob(os.path.join(d, 'tmp', 'agent_events_*.jsonl')), key=os.path.getmtime)
if not files: print('loc: no agent_events file'); sys.exit(2)
S = []
with open(files[-1], errors='ignore') as f:
    for line in f:
        if '"kind":"state"' not in line: continue
        try: r = json.loads(line)
        except ValueError: continue
        l = r.get('loc', {})
        if l.get('have') and l.get('sdf_med') is not None: S.append((r['t'], l))
if not S: print('loc: no pose yet'); sys.exit(2)
t1 = S[-1][0]; W = [l for t, l in S if t1 - t <= win * 1000]
sdf = statistics.median(l['sdf_med'] for l in W); sx = max(l['sx'] for l in W); sth = max(l['sth'] for l in W)
ok = all(l.get('ok') and not l.get('diverged') for l in W) and sdf <= max_sdf
print(f"loc {'OK' if ok else 'NOT OK'}: median sdf_med {sdf:.3f} m (max {max_sdf}) over {len(W)} records, sx {sx:.3f} m, sth {sth:.3f} rad, mode {W[-1].get('mode')}")
sys.exit(0 if ok else 1)
