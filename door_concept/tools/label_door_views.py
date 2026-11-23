#!/usr/bin/env -S python3 -I
"""Label the doors of a replayed tape with the Webots world's true doors (threerooms-piso.wbt: DOOR_0 / DOOR_1 / DOOR_2).

Inputs, one run directory RUN/ holding what `door_replay --csv RUN` and `door_truth_scan` wrote:
  door_views_log.csv      per-frame detector rows  (cycle,door,camera,view_id,q,fill_h,fill_v,vis,kappa,aspect,hit,...)
  door_existence_log.csv  per-cycle per-door rows  (cycle,node,cx,cy,... geo_x,geo_mu0,geo_s,...) in the CURRENT ROOM's frame
  stdout_diag.log         the replay's core stdout lines per cycle (candidate places: `[candidate] place_N at (x,y)`)
  room_pose.csv           world<-current-room SE(2) per cycle (tools/door_truth_scan.cpp)
Door = node name (door_N, or place_N for a candidate place). Its world position is the median over its existence rows of room_pose(cycle) applied to (cx, cy); the label is 1
when that position lies within MATCH_M of a true door centre (a true door's aperture is 1.0 m wide: a hypothesis whose centre is within
0.75 m overlaps the real hole by more than its own half-width), else 0. A door whose rows span more than SPREAD_M in the world (carried
across rooms, so its room frame is not the current room's) has no trustworthy position and is dropped from both outputs.
Outputs: RUN/door_views_labelled.csv (door_views_log.csv + label, gt) and RUN/door_geo_pairs.csv (cycle,door,label,geo_x,geo_mu0,geo_s).
Usage: python3 -I tools/label_door_views.py RUN [RUN ...]
"""
import csv, math, os, re, sys
import numpy as np

GT = {"DOOR_0": (-3.583, 7.166), "DOOR_1": (-4.771, 9.240), "DOOR_2": (-1.705, 10.77)}   # world centres (threerooms-piso.wbt, door_apertures.json)
MATCH_M = 0.75
AMBIG_M = 1.5   # kept: 0.75 < e <= 1.5 m is neither the true door (a hole 1.0 m wide: half-width 0.5 + SLAM error) nor clearly another place — a mis-placed
                # estimate of the true door or a phantom beside it; such a hypothesis is EXCLUDED from both classes rather than guessed. No model term
                # does this job: it is the labelling of the evaluation data, not a term of the likelihood.
SPREAD_M = 0.75


def label_run(run):
    pose = {}
    for r in csv.DictReader(open(os.path.join(run, "room_pose.csv"), newline="")):
        pose[int(r["cycle"])] = (float(r["tx"]), float(r["ty"]), float(r["yaw"]))
    pos = {}
    ex_rows = list(csv.DictReader(open(os.path.join(run, "door_existence_log.csv"), newline="")))
    for r in ex_rows:
        p = pose.get(int(r["cycle"]))
        if p is None:
            continue
        c, s = math.cos(p[2]), math.sin(p[2])
        x, y = float(r["cx"]), float(r["cy"])
        pos.setdefault(r["node"], []).append((c * x - s * y + p[0], s * x + c * y + p[1]))
    # candidate places (place_N): the replay's stdout lines `[candidate] place_N at (x,y)` stamped with their cycle (door_replay --csv's
    # stdout_diag.log), in the same current-room frame
    sd = os.path.join(run, "stdout_diag.log")
    if os.path.exists(sd):
        for ln in open(sd):
            m = re.match(r"(\d+)\t.*\[candidate\] (place_\d+) at \((-?[\d.]+),(-?[\d.]+)\)", ln)
            p = pose.get(int(m.group(1))) if m else None
            if p is None:
                continue
            c, s = math.cos(p[2]), math.sin(p[2])
            x, y = float(m.group(3)), float(m.group(4))
            pos.setdefault(m.group(2), []).append((c * x - s * y + p[0], s * x + c * y + p[1]))
    lab, gtn, drop, ambig = {}, {}, [], []
    for d, ps in pos.items():
        a = np.array(ps)
        m = np.median(a, axis=0)
        if float(np.max(np.hypot(*(a - m).T))) > SPREAD_M and float(np.percentile(np.hypot(*(a - m).T), 90)) > SPREAD_M:
            drop.append(d)
            continue
        best = min(GT, key=lambda g: math.hypot(m[0] - GT[g][0], m[1] - GT[g][1]))
        e = math.hypot(m[0] - GT[best][0], m[1] - GT[best][1])
        if MATCH_M < e <= AMBIG_M:
            print(f"  {os.path.basename(run)} {d}: world ({m[0]:.2f},{m[1]:.2f}) nearest {best} at {e:.2f} m -> ambiguous, excluded")
            ambig.append(d)
            continue
        lab[d] = 1 if e <= MATCH_M else 0
        gtn[d] = best if lab[d] else ""
        print(f"  {os.path.basename(run)} {d}: world ({m[0]:.2f},{m[1]:.2f}) nearest {best} at {e:.2f} m -> label {lab[d]}")
    for d in drop:
        print(f"  {os.path.basename(run)} {d}: dropped (rows spread over more than {SPREAD_M} m in the world)")
    vr = list(csv.DictReader(open(os.path.join(run, "door_views_log.csv"), newline="")))
    n_unlab = 0
    with open(os.path.join(run, "door_views_labelled.csv"), "w", newline="") as o:
        w = csv.writer(o)
        w.writerow(list(vr[0].keys()) + ["label", "gt"])
        for r in vr:
            if r["door"] not in lab:
                n_unlab += 1
                continue
            w.writerow(list(r.values()) + [lab[r["door"]], gtn[r["door"]]])
    ng = 0
    with open(os.path.join(run, "door_geo_pairs.csv"), "w", newline="") as o:
        w = csv.writer(o)
        w.writerow(["cycle", "door", "label", "geo_x", "geo_mu0", "geo_s"])
        for r in ex_rows:
            if r["node"] not in lab or r["geo_x"] in ("nan", ""):
                continue
            w.writerow([r["cycle"], r["node"], lab[r["node"]], r["geo_x"], r["geo_mu0"], r["geo_s"]])
            ng += 1
    n1 = sum(1 for v in lab.values() if v == 1)
    print(f"{os.path.basename(run)}: {len(lab)} doors ({n1} true, {len(lab)-n1} not), {len(drop)} dropped, {len(ambig)} ambiguous, "
          f"{len(vr)-n_unlab} view rows labelled ({n_unlab} dropped), {ng} geo rows")


if __name__ == "__main__":
    for r in sys.argv[1:]:
        label_run(r)
