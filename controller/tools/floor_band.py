#!/usr/bin/env python3
"""
floor_band.py — where the FLOOR ends in the LiDAR cloud: the value of Lidar.ContactMinHeightM (Task 7).

Asks the controller (remote `zdump <n>`, controller_obstacle_tracker.cpp) for the next n raw sweeps in the ROBOT frame
(z = height above the floor; pre z-band, pre self-filter, with a self flag) — or reads an existing lidar_z_dump.csv
with --file — and histograms the z of the FLOOR returns of a robot PARKED ON OPEN FLOOR.

Which returns are floor: not self (the body), within --max-range of the robot (the policy's working range; the
helios's grazing far-floor returns beyond 3.88 m read 13-17 cm high and are not floor the body can reach), z below
--floor-zmax, and in a (2 deg azimuth x 0.10 m range) cell that holds NO return above --struct-z (a wall or a leg makes
returns at every height in its cell; the floor does not). That last test is what keeps the foot of a wall out of the
floor percentile, which p99.9 would otherwise be.

  ContactMinHeightM = p99.9(z of floor returns) + sigma_r   (sigma_r = Controller.LidarRangeSigmaM, 0.03)

Prints the histogram, per-plane and per-range-band percentiles, and the line to paste into controller/etc/config.toml
with the run id.

Usage: floor_band.py [--controller-dir DIR] [--sweeps 20] [--file lidar_z_dump.csv] [--max-range 3.0] [--sigma-r 0.03]
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import pandas as pd

B = '/home/pbustos/robocomp/components/active_inference'


def send(ctrl_dir, line, timeout_s=5.0):
    path = os.path.join(ctrl_dir, 'remote_cmd.txt')
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        f.write(line + '\n')
    os.replace(tmp, path)
    t0 = time.time()
    while os.path.exists(path):
        if time.time() - t0 > timeout_s:
            sys.exit(f'remote_cmd.txt not consumed in {timeout_s} s — is the controller running from {ctrl_dir}?')
        time.sleep(0.05)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--controller-dir', default=f'{B}/controller')
    ap.add_argument('--sweeps', type=int, default=20)
    ap.add_argument('--file', help='analyse this dump instead of asking the controller for a new one')
    ap.add_argument('--max-range', type=float, default=3.0)
    ap.add_argument('--floor-zmax', type=float, default=0.25)
    ap.add_argument('--struct-z', type=float, default=0.35)
    ap.add_argument('--sigma-r', type=float, default=0.03)
    a = ap.parse_args()

    path = a.file
    if not path:
        path = os.path.join(a.controller_dir, 'lidar_z_dump.csv')
        t_req = time.time()
        send(a.controller_dir, f'zdump {a.sweeps}')
        # the dump is complete when the file holds the last sweep index and has not grown for 1 s
        last, stable = -1, 0.0
        while True:
            time.sleep(0.5)
            if not os.path.exists(path) or os.path.getmtime(path) < t_req:
                if time.time() - t_req > 30:
                    sys.exit('no lidar_z_dump.csv after 30 s — is the LiDAR stream up?')
                continue
            with open(path, 'rb') as f:
                f.seek(max(0, os.path.getsize(path) - 200))
                tail = f.read().decode(errors='replace').strip().splitlines()
            try:
                last = int(tail[-1].split(',')[0])
            except (ValueError, IndexError):
                continue
            if last >= a.sweeps - 1:
                stable += 0.5
                if stable >= 1.0:
                    break
    d = pd.read_csv(path, comment='#')
    run_id = time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(path)))
    d['r'] = np.hypot(d.x, d.y)
    d['az'] = np.degrees(np.arctan2(d.x, d.y))
    print(f'{path}: {len(d)} returns, {d.scan.nunique()} sweeps, self {int(d.self.sum())} (dropped)')
    d = d[d.self == 0]

    # cells holding vertical structure (a wall / a leg): any return above struct_z
    d['cell'] = (np.floor((d.az + 180) / 2.0).astype(int) * 1000 + np.floor(d.r / 0.10).astype(int))
    struct_cells = set(d.loc[d.z > a.struct_z, 'cell'])
    near = d[d.r <= a.max_range]
    floor = near[(near.z < a.floor_zmax) & ~near.cell.isin(struct_cells)]
    if len(floor) < 100:
        sys.exit(f'only {len(floor)} floor returns within {a.max_range} m — park the robot on open floor')

    print(f'\nfloor returns within {a.max_range} m (structure cells excluded): {len(floor)}')
    edges = np.arange(-0.10, a.floor_zmax + 0.0001, 0.01)
    h, _ = np.histogram(floor.z, bins=edges)
    for lo, n in zip(edges[:-1], h):
        if n:
            print(f'  z [{lo:+.2f},{lo + 0.01:+.2f}) {n:7d} ' + '#' * int(60 * n / h.max()))
    for plane, g in floor.groupby('plane'):
        print(f'  plane {plane}: n {len(g):7d}  p50 {g.z.quantile(.5):+.4f}  p99 {g.z.quantile(.99):+.4f}  '
              f'p99.9 {g.z.quantile(.999):+.4f}  max {g.z.max():+.4f}')
    print('  by range (all low returns, structure cells excluded, ALL ranges — shows where the floor reads high):')
    allfloor = d[(d.z < a.floor_zmax) & ~d.cell.isin(struct_cells)]
    for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 99)):
        g = allfloor[(allfloor.r >= lo) & (allfloor.r < hi)]
        if len(g):
            print(f'    r [{lo},{hi}) m: n {len(g):7d}  p50 {g.z.quantile(.5):+.4f}  p99.9 {g.z.quantile(.999):+.4f}')
    p999 = float(floor.z.quantile(0.999))
    value = p999 + a.sigma_r
    print(f'\nfloor p99.9 = {p999:+.4f} m, sigma_r = {a.sigma_r:.3f}  =>  ContactMinHeightM = {value:.3f}')
    print(f'\ncontroller/etc/config.toml [Lidar]:\n  ContactMinHeightM = {value:.3f}   # measured {run_id}: floor p99.9 '
          f'{p999:+.4f} + sigma_r {a.sigma_r:.2f} (floor_band.py, {len(floor)} floor returns <= {a.max_range} m, '
          f'{d.scan.nunique()} sweeps)')
    if value >= 0.10:
        print('\n⚠ the floor band reaches 0.10 m: a 0.10 m return would NOT be contact for the policy — the test '
              'low_returns_reach_policy encodes the opposite expectation; look at the per-plane/per-range table first.')


if __name__ == '__main__':
    main()
