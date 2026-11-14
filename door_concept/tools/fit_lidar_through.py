#!/usr/bin/env -S python3 -I
"""Fit the LiDAR "through" probabilities (`lidar,through_open` and `lidar,through_wall` of etc/door_detectors.csv) from lidar_bins.csv
files written by tools/door_truth_scan.cpp (per low-LiDAR sweep and true Webots door: the wall-line bins OBSERVED, and how many read
THROUGH, door_opening_edges.h's through_bin_counts on the registered sweep).
  through_open = P(an observed bin reads through | it lies in a true OPEN aperture, 5 cm inside the jambs)
  through_wall = P(an observed bin reads through | it lies on solid wall): the wall stretches beside the doors, plus the aperture of a
                 door that is still SHUT (its leaf is wall to the LiDAR).
Both are the POSTERIOR MEAN of the pooled bin ratio under a Beta prior centred on the compiled default with a pseudo-count of PRIOR_N = 20
bins, (through + 20 d) / (observed + 20): a probability of exactly 0 or 1 is not a likelihood anyone can multiply, and on the apartment
tapes the wall bins have almost no through readings in ~10^5. `n` is the number of SWEEPS
that contributed (the bins of one sweep are correlated, so the bin count would overstate the evidence); the bin counts are printed.
--skip FILE:DOOR:STATE drops one (tape, door, state) whose "shut" label the data itself contradicts: tape_dm's DOOR_0 reads through 28 % of
its aperture bins in the 853 sweeps BEFORE its first recorded OPEN request (cycle 1991) — the door was already ajar, by a request the tape
does not carry — so its "shut" bins are not solid wall.
Usage: python3 -I tools/fit_lidar_through.py BINS.csv [BINS.csv ...] [--skip FILE:DOOR:STATE] [--dry-run] [--out etc/door_detectors.csv] [--source TAG]
Writes the two `lidar` rows only without --dry-run; every other row of the output file is carried over. No bins for a row: the default, n = 0.
"""
import argparse, csv, datetime, os, sys

DEFAULT = {"through_open": 0.75, "through_wall": 0.01}
PRIOR_N = 20   # pseudo-bins of the Beta prior centred on the compiled default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bins", nargs="+"); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--skip", action="append", default=[])
    ap.add_argument("--out", default="etc/door_detectors.csv"); ap.add_argument("--source", default=None)
    a = ap.parse_args()
    acc = {"open_ap": [0, 0, 0], "shut_ap": [0, 0, 0], "wall": [0, 0, 0]}   # observed bins, through bins, sweeps
    skip = {tuple(x.rsplit(":", 2)) for x in a.skip}
    for p in a.bins:
        for r in csv.DictReader(open(p, newline="")):
            o, t = int(r["obs"]), int(r["thr"])
            if (p, r["door"], r["state"]) in skip:
                continue
            if o == 0:
                continue
            k = "wall" if r["region"] == "wall" else ("open_ap" if r["state"] == "open" else "shut_ap")
            acc[k][0] += o; acc[k][1] += t; acc[k][2] += 1
    for k, (o, t, n) in acc.items():
        print(f"{k:8s} sweeps {n:6d} bins {o:7d} through {t:6d}  raw {t/o if o else float('nan'):.4f}")
    rows = {}
    o, t, n = acc["open_ap"]
    rows["through_open"] = ((t + PRIOR_N * DEFAULT["through_open"]) / (o + PRIOR_N), n)
    o2, t2, n2 = (acc["wall"][i] + acc["shut_ap"][i] for i in range(3))
    rows["through_wall"] = ((t2 + PRIOR_N * DEFAULT["through_wall"]) / (o2 + PRIOR_N), n2)
    for k, (v, nn) in rows.items():
        print(f"lidar,{k}={v:.5f} (n={nn} sweeps)")
    if a.dry_run:
        return 0
    src = a.source or f"fit_lidar_through.py {datetime.date.today()}"
    kept = [ln for ln in open(a.out) if not ln.startswith("lidar,")] if os.path.exists(a.out) else ["camera,param,value,n,source\n"]
    with open(a.out, "w", newline="") as o_:
        o_.writelines(kept)
        for k, (v, nn) in rows.items():
            o_.write(f"lidar,{k},{v:.6g},{nn},{src if nn > 0 else 'compiled_defaults'}\n")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
