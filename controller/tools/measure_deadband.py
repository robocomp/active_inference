#!/usr/bin/env python3
"""
measure_deadband.py — the base's DEADBAND: the smallest |v| / |w| it actually executes (Task 7, local-motion policy).

Steps the base through small open-loop commands with the controller's `probe <v> <w> <s>` (controller_base_probe.h),
sent through the controller's remote channel (remote_cmd.txt in the directory the controller RUNS from), and reads the
motion back:
  * GT     room_concept's sdf_localizer gt_error.csv (Webots supervisor pose), over [t_start, t_end + settle] — preferred;
  * POSE   the localised pose the controller logs in base_probe.csv at the probe's start and end — the fallback.
Each command is probed forward and reverse (or left and right) in alternation, so the robot ends where it started.
A zero command ("probe 0 0 <s>") gives the noise floor of the motion read-back.

For each probe: executed speed = displacement (or |dtheta|) / hold; ratio = executed / commanded.
The deadband for a sign is the smallest commanded magnitude from which EVERY larger one has ratio >= --ratio (0.5:
"the base executes at least half of what it was told" — a reporting criterion for this measurement, not a control
gate). The value to configure (robot_concept etc/config_shadow.toml Agent.base_min_linear_speed / base_min_rot_speed)
is the LARGER of the two signs.

PRECONDITIONS (measure_policy_inputs.md): fleet up (run_crossing.sh <dir> here), robot parked on OPEN floor with
>= 0.7 m free ahead and behind (the controller refuses a probe whose predicted sweep is not free), no mission running,
and the base ARMED by the operator (`run_crossing.sh <dir> run`): a probe never re-arms a stopped base.

Usage: measure_deadband.py [--controller-dir DIR] [--gt FILE] [--hold 2.0] [--out results.csv]
"""
import argparse
import math
import os
import sys
import time

import pandas as pd

B = '/home/pbustos/robocomp/components/active_inference'
V_LIST = [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
W_LIST = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.13, 0.16, 0.20, 0.25]


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def send(ctrl_dir, line, timeout_s=5.0):
    """Drop one command into remote_cmd.txt atomically and wait until the controller consumed (deleted) it."""
    path = os.path.join(ctrl_dir, 'remote_cmd.txt')
    t0 = time.time()
    while os.path.exists(path):                 # a previous command not consumed yet
        if time.time() - t0 > timeout_s:
            sys.exit(f'remote_cmd.txt not consumed in {timeout_s} s — is the controller running from {ctrl_dir}?')
        time.sleep(0.1)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        f.write(line + '\n')
    os.replace(tmp, path)
    t0 = time.time()
    while os.path.exists(path):
        if time.time() - t0 > timeout_s:
            sys.exit(f'remote_cmd.txt not consumed in {timeout_s} s — is the controller running from {ctrl_dir}?')
        time.sleep(0.05)


def probe_rows(ctrl_dir):
    p = os.path.join(ctrl_dir, 'base_probe.csv')
    if not os.path.exists(p):
        return pd.DataFrame()
    return pd.read_csv(p)


def gt_motion(gt_path, t0, t1):
    """(displacement m, |dtheta| rad) between the GT poses nearest t0 and t1 (ms), or None."""
    if not gt_path or not os.path.exists(gt_path):
        return None
    # tail of a possibly huge, growing file: read only the last ~20 MB
    size = os.path.getsize(gt_path)
    with open(gt_path, 'rb') as f:
        head_b = f.readline()
        head = head_b.decode()
        start = max(len(head_b), size - 20_000_000)
        f.seek(start)
        if start > len(head_b):
            f.readline()                        # drop the partial line
        body = f.read().decode(errors='replace')
    import io
    d = pd.read_csv(io.StringIO(head + body), usecols=['ts_ms', 'gt_x', 'gt_y', 'gt_theta'])
    d = d[(d.ts_ms >= t0 - 500) & (d.ts_ms <= t1 + 500)]
    if len(d) < 2:
        return None
    a = d.iloc[(d.ts_ms - t0).abs().argmin()]
    b = d.iloc[(d.ts_ms - t1).abs().argmin()]
    return math.hypot(b.gt_x - a.gt_x, b.gt_y - a.gt_y), abs(wrap(b.gt_theta - a.gt_theta))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--controller-dir', default=f'{B}/controller', help='the directory the controller runs FROM')
    ap.add_argument('--gt', default=f'{B}/room_concept/tmp/sdf_localizer/gt_error.csv')
    ap.add_argument('--hold', type=float, default=2.0)
    ap.add_argument('--settle', type=float, default=1.5, help='s after the stop before the next probe (coast + log)')
    ap.add_argument('--ratio', type=float, default=0.5)
    ap.add_argument('--only', choices=['v', 'w'], help='probe only linear or only rotational speeds')
    ap.add_argument('--out', default='deadband_results.csv')
    a = ap.parse_args()

    plan = [(0.0, 0.0)]
    if a.only != 'w':
        for v in V_LIST:
            plan += [(v, 0.0), (-v, 0.0)]
    if a.only != 'v':
        for w in W_LIST:
            plan += [(0.0, w), (0.0, -w)]

    n_before = len(probe_rows(a.controller_dir))
    results = []
    for v, w in plan:
        send(a.controller_dir, f'probe {v:.4f} {w:.4f} {a.hold:.2f}')
        time.sleep(a.hold + a.settle)
        rows = probe_rows(a.controller_dir)
        if len(rows) <= n_before:
            print(f'probe v {v:+.3f} w {w:+.3f}: NO ROW (refused? disarmed => send `run` first; see the controller log: [probe] ...)', file=sys.stderr)
            continue
        r = rows.iloc[-1]
        n_before = len(rows)
        settle_ms = int(a.settle * 1000) - 300
        g = gt_motion(a.gt, int(r.t_start_ms), int(r.t_end_ms) + settle_ms)
        if g is not None:
            disp, dth, src = g[0], g[1], 'gt'
        else:
            disp = math.hypot(r.x1 - r.x0, r.y1 - r.y0) if pd.notna(r.x1) else float('nan')
            dth = abs(wrap(r.theta1 - r.theta0)) if pd.notna(r.theta1) else float('nan')
            src = 'pose'
        cmd = abs(v) if v != 0 else abs(w)
        executed = (disp if v != 0 else dth) / a.hold
        ratio = executed / cmd if cmd > 0 else float('nan')
        results.append(dict(v=v, w=w, hold_s=a.hold, disp_m=disp, dtheta_rad=dth, executed=executed, ratio=ratio, src=src,
                            t_start_ms=int(r.t_start_ms), t_end_ms=int(r.t_end_ms)))
        print(f'probe v {v:+.3f} w {w:+.3f}: moved {disp:.4f} m, turned {dth:.4f} rad ({src}) -> executed {executed:.4f}'
              f' ratio {ratio:.2f}')

    df = pd.DataFrame(results)
    df.to_csv(a.out, index=False)
    print(f'\nwrote {a.out}')
    base = df[(df.v == 0) & (df.w == 0)]
    if len(base):
        print(f'noise floor (zero command): {base.disp_m.iloc[0]:.4f} m, {base.dtheta_rad.iloc[0]:.4f} rad over {a.hold} s')

    def deadband(sub, key):
        sub = sub.assign(mag=sub[key].abs()).sort_values('mag')
        ok = (sub.ratio >= a.ratio).values
        for i in range(len(sub)):
            if ok[i:].all():
                return float(sub.mag.values[i])
        return float('nan')

    out = {}
    for name, key, sel in (('linear', 'v', df.v != 0), ('rot', 'w', df.w != 0)):
        sub = df[sel]
        if not len(sub):
            continue
        pos, neg = deadband(sub[sub[key] > 0], key), deadband(sub[sub[key] < 0], key)
        out[name] = max(pos, neg)
        unit = 'm/s' if key == 'v' else 'rad/s'
        print(f'deadband {name}: +{pos:.3f} / -{neg:.3f} {unit}  =>  configure {out[name]:.3f} {unit}')
    run_id = time.strftime('%Y-%m-%d %H:%M')
    if 'linear' in out:
        print(f'\nrobot_concept etc/config_shadow.toml [Agent]:\n  base_min_linear_speed = {out["linear"]:.3f}'
              f'   # measured {run_id}, measure_deadband.py ({a.out})')
    if 'rot' in out:
        print(f'  base_min_rot_speed    = {out["rot"]:.3f}   # measured {run_id}, measure_deadband.py ({a.out})')
    send(a.controller_dir, 'stop')


if __name__ == '__main__':
    main()
