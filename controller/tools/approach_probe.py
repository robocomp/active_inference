#!/usr/bin/env python3
"""Diagnose the affordance arrival from approach_diag.jsonl (self-describing, one object per cycle).

WHY JSONL AND NOT THE CSV. On 2026-08-19 four wrong conclusions were drawn from approach_diag.csv,
every one of them a column meaning something other than its name, or a position that had shifted:

  yaw_err_deg     0 on 100% of rows — written `has_value() ? ... : 0.f`, and a Reach target never sets
                  the optional. 0 means NOT APPLICABLE, not "no error".
  min_esdf_m      robot-centre-referenced; clear_now_m is footprint-referenced. Constant 0.273 m apart
                  (the body half-extent). Comparing them manufactured a fake 370x disagreement.
  rob_facing_deg  remainder(theta + pi/2, 2pi); sign convention does not match cmd_rot, so correlating
                  them gives -0.44 and the false claim that the robot turns backwards.
  d_arrival_m /   read off the ControlOutput and off an accessor AFTER compute() returned — not
  goal_thr        necessarily the operands the arrival gate actually compared. Hence arr_dist/arr_thr.

Plus two structural failures of that file: its header is built from adjacent C++ string literals, so an
inserted literal lands inside the header line invisibly (twice in one day), and a torn final row is
indistinguishable from a short one. This reader found a 13-field row among 143 valid ones.

In JSONL every value carries its key. A shifted or truncated line cannot be silently misread — it fails
to parse and is reported.
"""
import sys, json
from collections import Counter

def load(path):
    good, bad = [], 0
    for n, line in enumerate(open(path), 1):
        line = line.strip()
        if not line:
            continue
        try:
            good.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1          # a torn line is DETECTED, never silently misread
    print(f"[load] {path}: {len(good)} records, {bad} unparseable (torn/partial lines are expected at the tail)")
    if not good:
        sys.exit("[load] nothing to analyse")
    keys = Counter(k for r in good for k in r)
    missing = {k: len(good) - v for k, v in keys.items() if v != len(good)}
    if missing:
        print(f"[load] ⚠ fields absent from some records: {missing}")
    return good

def main(path="approach_diag.jsonl"):
    rows = load(path)
    print(f"[phase] {dict(Counter(r['phase'] for r in rows))}")
    reached = sum(1 for r in rows if r["phase"] == "reached")
    print(f"[phase] reached={reached}  ALIGN={sum(1 for r in rows if r['phase']=='ALIGN')}"
          "   <- the number that has been 0 all day\n")

    if "arr_gate" not in rows[0]:
        print("[arrival] no arr_* probe fields; nothing further to say.")
        return

    # Does the gate's own view differ from what the file otherwise reports?
    dd = sum(1 for r in rows if abs(r["arr_dist"] - r["d_arrival_m"]) > 1e-6)
    dt = sum(1 for r in rows if abs(r["arr_thr"] - r["goal_thr"]) > 1e-6)
    print(f"[arrival] arr_dist != d_arrival_m on {dd} rows;  arr_thr != goal_thr on {dt} rows")
    if dd or dt:
        print("[arrival] ★ THE LOGGED COLUMNS ARE NOT THE GATE'S OPERANDS — that alone explains the")
        print("[arrival]   contradiction, and everything inferred from d_arrival_m/goal_thr is void.")

    inside = [r for r in rows if r["arr_dist"] < r["arr_thr"]]
    print(f"\n[arrival] rows with arr_dist < arr_thr (the gate's own numbers): {len(inside)}")
    for r in inside[:10]:
        print(f"    dist={r['arr_dist']:.4f} thr={r['arr_thr']:.4f} | gate={r['arr_gate']} "
              f"end={r['arr_end']} active={r['arr_active']} facing={r['arr_facing']} "
              f"passed={r['arr_passed']} recede={r['arr_recede']} path_n={r['arr_path_n']} "
              f"| phase={r['phase']}")

    contra = [r for r in inside if r["arr_gate"] == 0 and r["arr_end"] == 1]
    opened = [r for r in rows if r["arr_gate"] == 1]
    print(f"\n[arrival] gate opened on {len(opened)} row(s)")
    if contra:
        print(f"[arrival] ✗ {len(contra)} row(s): end=1 and dist<thr yet gate=0 — arithmetically")
        print("[arrival]   impossible with operands captured at the comparison. Look elsewhere than")
        print("[arrival]   the gate's inputs.")
    if opened and reached == 0:
        print("[arrival] ★ the gate OPENED and nothing ever reported `reached`: the arrival is lost")
        print("[arrival]   AFTER the gate — check arr_facing (align branch) and the caller's handling.")
    if not opened and not inside:
        print("[arrival] the robot never got inside the band this run — not an arrival bug, a driving one.")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "approach_diag.jsonl")
