#!/usr/bin/env python3
"""Flag diagnostic columns that are never actually computed, before anyone reasons from them.

WHY. On 2026-08-19 six separate wrong conclusions were drawn from fields whose NAME promised
something the field did not carry. A constant column is indistinguishable, at a glance, from a
measurement that happens to be steady — and it reads as evidence:

  n_collisions = 0        read as "no collisions"        -> never written (MPPI is not the live tracker)
  path_kappa   = -999     read as a curvature            -> "not computed" sentinel
  gate_min_esdf = -1      read as "no clearance"         -> the gate did not run
  yaw_err_deg  = 0        read as "no heading error"     -> not applicable to a Reach target
  min_esdf vs clear_now   compared directly              -> different reference points (0.273 m apart)
  file "mppi_diag.csv"    read as MPPI telemetry         -> the live tracker is PLAIN

Run this on any diagnostic file BEFORE using a column as evidence. A column that never varies is not
evidence; it is either a constant of the configuration or a field nobody fills in.

usage: dead_columns.py [file.csv ...]        (defaults to the controller's usual outputs)
"""
import csv, sys, os

SENTINELS = {-999.0, -1.0, -9999.0}

def scan(path):
    if not os.path.exists(path):
        return
    rows = [l for l in open(path) if not l.lstrip().startswith("#")]
    rd = list(csv.reader(rows))
    if len(rd) < 20:
        print(f"=== {path}: too few rows ({len(rd)}) to judge")
        return
    hdr = rd[0]
    data = [r for r in rd[1:] if len(r) == len(hdr)]
    torn = len(rd) - 1 - len(data)
    print(f"=== {path}  ({len(data)} rows, {len(hdr)} cols"
          + (f", ⚠ {torn} malformed" if torn else "") + ")")
    dead, sent, live = [], [], 0
    for i, c in enumerate(hdr):
        nums = []
        for r in data:
            try:
                v = float(r[i])
                if v == v:
                    nums.append(v)
            except ValueError:
                pass
        if not nums:
            continue
        u = set(nums)
        if len(u) == 1:
            (sent if nums[0] in SENTINELS else dead).append((c, nums[0]))
        elif u <= SENTINELS:
            sent.append((c, sorted(u)))
        else:
            live += 1
    for c, v in sent:
        print(f"   ✗ {c:26s} SENTINEL ONLY  {v}   <- not computed; do not reason from it")
    for c, v in dead:
        print(f"   ⚠ {c:26s} CONSTANT       {v}   <- a config constant, or nobody fills it in")
    print(f"   ✓ {live} columns carry varying data\n")

def main():
    args = sys.argv[1:] or ["tracker_diag.csv", "mppi_diag.csv", "approach_diag.csv",
                            "band_diag.csv", "proximity_obstacles.csv", "stall_events.csv"]
    for a in args:
        scan(a)

if __name__ == "__main__":
    main()
