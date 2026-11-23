#!/usr/bin/env python3
"""AuRo item 13 (2026-11-22): per-run report of the out-of-view moved-chair test.   c3b_report.py <run_dir>... [--csv out.csv]

Builds on c3b_timeline.py (which picks the moved chair's node, its pre-move belief and the correction moment; the criterion
is a chair belief in room_2 at 1.0 +- 0.25 m from the pre-move belief with std_cx and std_cy < 0.15 m). Adds, from the Webots
truth:
  FIRST SEEN   first moment after the return (leg 3 end) that the moved chair's NEW position is in view: within 4.0 m of the
               robot, within +-55 deg of its heading (Webots yaw 0 faces +y) and no wall between (c3a_objects.py's rule).
  DWELL        seconds in view after the return, (a) up to the correction and (b) until the end of the run.
  NEAR         seconds after the return with the robot within 2.0 m of the chair's new position.
  BY OFFER     whether the moved chair's own offer (aff_<node>) was claimed by the controller at the correction moment.
  CLAIMS       the controller's selections after the return ([aff-select] -> lines), in order, room offers included.
"""
import csv, math, os, re, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wbt_rooms as W
from run_frames import make_to_world

RANGE_M, HFOV_DEG, NEAR_M = 4.0, 110.0, 2.0
walls, doors, meshes = W.parse_wbt(W.WBT)
occ = W.rasterise(walls, {k: dict(v, width=0.0) for k, v in doors.items()}, meshes)
x0, _, y0, _ = W.BOX


def blocked(a, b):
    n = int(math.hypot(b[0] - a[0], b[1] - a[1]) / 0.02)
    for i in range(2, n - 2):
        x = a[0] + (b[0] - a[0]) * i / n; y = a[1] + (b[1] - a[1]) * i / n
        if occ[int((y - y0) / W.RES), int((x - x0) / W.RES)]: return True
    return False


def one(run):
    run = run.rstrip('/'); name = os.path.basename(run)
    out = subprocess.run([sys.executable, f'{HERE}/c3b_timeline.py', run], capture_output=True, text=True).stdout
    legs = {int(r['leg']): r for r in csv.DictReader(open(f'{run}/legs.tsv'), delimiter='\t')}
    mv = open(f'{run}/chair_move.txt').read()
    arm = 'off' if re.search(r'EpistemicOfferEnabled = false', mv) else 'on'
    got = [l for l in mv.splitlines() if re.match(r'^-?\d+\.\d+ -?\d+\.\d+ ', l)]
    cx, cy = map(float, got[-1].split()[:2])                       # chair's new world position (read back from Webots)
    t_ret = int(legs[3]['t1_ms']) if 3 in legs and legs[3]['status'] == 'OK' else None
    m = re.search(r'moved chair = (chair_\d+): pre-move belief \(([-\d.]+),([-\d.]+)\) room frame, std \(([\d.]+),([\d.]+)\)', out)
    node, sig_pre = (m.group(1), max(float(m.group(4)), float(m.group(5)))) if m else (None, None)
    c = re.search(r'corrected \[any chair belief\]: (\S+) at ([-\d.]+) s after return, displacement ([\d.]+) m, std \(([\d.]+),([\d.]+)\); claim then: \'([^\']*)\'', out)
    t_corr = t_ret + float(c.group(2)) * 1000 if c and t_ret else None
    gt = [(int(r['t_ms']), float(r['x']), float(r['y']), float(r['yaw'])) for r in csv.DictReader(open(f'{run}/gt_pose.tsv'), delimiter='\t')]
    first = None; dwell_corr = dwell_all = near = 0.0
    if t_ret:
        for a, b in zip(gt, gt[1:]):
            if b[0] < t_ret: continue
            n = max(1, int((b[0] - a[0]) / 100)); dyaw = math.remainder(b[3] - a[3], 2 * math.pi)
            for k in range(n):
                f = k / n; t = a[0] + f * (b[0] - a[0]); dt = (b[0] - a[0]) / n / 1000
                if t < t_ret or dt > 2.0: continue
                x, y, yaw = a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2]), a[3] + f * dyaw
                d = math.hypot(cx - x, cy - y)
                if d <= NEAR_M: near += dt
                if d > RANGE_M or abs(math.remainder(math.atan2(cy - y, cx - x) - yaw - math.pi / 2, 2 * math.pi)) > math.radians(HFOV_DEG / 2) \
                        or blocked((x, y), (cx, cy)):
                    continue
                if first is None: first = t
                dwell_all += dt
                if t_corr is None or t <= t_corr: dwell_corr += dt
    claims = []
    if t_ret:
        tl = {}
        for r in csv.DictReader(open(f'{run}/gt_pose.tsv'), delimiter='\t'):
            if r.get('controller', '').isdigit(): tl[int(r['controller'])] = int(r['t_ms'])
        pts = sorted(tl.items())
        for i, l in enumerate(open(f'{run}/logs/controller.log', errors='replace'), 1):
            s = re.search(r"\[aff-select\] -> '([^']+)'", l)
            if not s: continue
            t = next((tm for ln, tm in pts if ln >= i), None)
            if t and t >= t_ret: claims.append(((t - t_ret) / 1000, s.group(1)))
    by_offer = bool(c and node and f'aff_{c.group(1)}' in c.group(6).split('|'))
    # WORLD-FRAME grade (11-22): room_2 holds TWO chairs since the owner's 19:26 world, so c3b_timeline's "most belief rows
    # in room_2" can pick the wrong node. Here the moved chair is the room_2 chair whose pre-move belief, mapped to the
    # world (run_frames: gauge + per-visit fit), is nearest CHAIR_3's home; corrected = the first row after the return of
    # ANY chair belief within TOL_W of the chair's new world position with std_cx, std_cy < 0.15 (c3b_timeline's STD_MAX).
    TOL_W, STD_MAX = 0.25, 0.15
    hm = re.search(r'CHAIR_3 -> ([-\d.]+) ([-\d.]+)', mv); home = (float(hm.group(1)), float(hm.group(2))) if hm else None
    wnode = werr_pre = wcorr = wcorr_node = werr_corr = wsig = ''; old_held_s = ''
    seg_dead = os.path.exists(f'{run}/logs/retina.log') and 'SegStage] disabled' in open(f'{run}/logs/retina.log', errors='replace').read()
    if os.path.exists(f'{run}/chair_ai2_log.csv') and home and t_ret:
        tw = make_to_world(run)
        rows = []
        for r in csv.DictReader(open(f'{run}/chair_ai2_log.csv')):
            try: rows.append((int(r['t_ms']), r['node'], float(r['cx']), float(r['cy']), max(float(r['std_cx']), float(r['std_cy']))))
            except (KeyError, ValueError): pass
        l1, l2 = legs[1], legs[2]
        pre = {}
        for t, n, x, y, sg in rows:
            if int(l1['t1_ms']) <= t <= int(l2['t0_ms']):
                w = tw(x, y, t)
                if w: pre.setdefault(n, []).append(w)
        if pre:
            def dist_home(n):
                ws = pre[n][-50:]; mx = sorted(w[0] for w in ws)[len(ws) // 2]; my = sorted(w[1] for w in ws)[len(ws) // 2]
                return math.hypot(mx - home[0], my - home[1])
            wnode = min(pre, key=dist_home); werr_pre = round(dist_home(wnode), 2)
        held = []
        for t, n, x, y, sg in rows:
            if t < t_ret: continue
            w = tw(x, y, t)
            if not w: continue
            if math.hypot(w[0] - home[0], w[1] - home[1]) <= TOL_W: held.append(t)
            if wcorr == '' and sg < STD_MAX and math.hypot(w[0] - cx, w[1] - cy) <= TOL_W:
                wcorr = round((t - t_ret) / 1000, 1); wcorr_node = n; werr_corr = round(math.hypot(w[0] - cx, w[1] - cy), 2); wsig = round(sg, 3)
        old_held_s = round((max(held) - t_ret) / 1000, 1) if held else 'none'
    agents = open(f'{run}/agents.txt').read().split('\n')[0].split('\t') if os.path.exists(f'{run}/agents.txt') else ['', '', '']
    return dict(run=name, arm=arm, legs_ok=sum(1 for r in legs.values() if r['status'] == 'OK'), moved_node=node or '',
                chair_new_xy=f'{cx:.2f},{cy:.2f}', returned=int(t_ret is not None),
                first_seen_s='' if first is None or not t_ret else round((first - t_ret) / 1000, 1),
                corrected=int(c is not None), corrected_at_s=c.group(2) if c else '',
                first_seen_to_corr_s=round((t_corr - first) / 1000, 1) if c and first else '',
                corr_node=c.group(1) if c else '', by_own_offer=int(by_offer), claim_at_corr=c.group(6) if c else '',
                sigma_pre=sig_pre if sig_pre is not None else '', sigma_at_corr=max(float(c.group(4)), float(c.group(5))) if c else '',
                dwell_in_view_to_corr_s=round(dwell_corr, 1), dwell_in_view_total_s=round(dwell_all, 1), near_2m_s=round(near, 1),
                seg_dead=int(seg_dead), w_moved_node=wnode, w_pre_err_m=werr_pre, w_corrected=int(wcorr != ''), w_corrected_at_s=wcorr,
                w_first_seen_to_corr_s=round(wcorr - (first - t_ret) / 1000, 1) if wcorr != '' and first else '',
                w_corr_node=wcorr_node, w_corr_err_m=werr_corr, w_sigma_at_corr=wsig, old_spot_belief_last_s=old_held_s,
                claims_after_return=' '.join(f'{t:.0f}s:{a}' for t, a in claims[:15]), build=agents[2] if len(agents) > 2 else '')


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    csv_out = sys.argv[sys.argv.index('--csv') + 1] if '--csv' in sys.argv else None
    if csv_out: args = [a for a in args if a != csv_out]
    rows = []
    for r in args:
        try: rows.append(one(r))
        except Exception as e: print(f'{r}: {e}')
    for d in rows: print(' | '.join(f'{k}={v}' for k, v in d.items()))
    if csv_out and rows:
        with open(csv_out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
