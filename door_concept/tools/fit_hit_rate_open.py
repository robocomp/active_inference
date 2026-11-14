#!/usr/bin/env -S python3 -I
"""Fit the detector's hit rate SPLIT BY THE LEAF'S TRUE STATE (Task 8b): h_shut and h_open per camera, from the truth-labelled view rows
of door_replay runs (tools/label_door_views.py -> door_views_labelled.csv; tools/door_truth_scan.cpp -> lidar_bins.csv, whose `state` per
true door is shut, or open once an OPEN request is >= 10 s old; the 10 s of swing are skipped, "none" before any sweep).
One trial per (door, camera, view_id): outcome y = the mean hit weight of its frames, front end fe = mean q / h_used (the replay's own
hit_rate), so P(report) = fe * h. Only label-1 views (a true door) with fe > 0 and state shut|open count. The rate is the posterior mean
under Beta(PRIOR_N d, PRIOR_N (1 - d)) centred on the table's shut hit rate d (Task 5's rule), the open views' outcomes read on the table's scale by the shut views: h_open = (PRIOR_N d + k sum y) / (PRIOR_N + sum fe); n = open views.
Usage: python3 -I tools/fit_hit_rate_open.py RUN [RUN ...] [--used zed=0.938,ricoh=0.706] [--default zed=0.9,ricoh=0.5]
Prints camera,hit_rate_open,h,n,... rows for etc/door_detectors_fitted.csv."""
import csv, os, sys, collections

PRIOR_N = 20
def parse_kv(s): return {k: float(v) for k, v in (x.split("=") for x in s.split(","))}
args = [a for a in sys.argv[1:] if not a.startswith("--")]
opt = {a.split("=", 1)[0]: a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--") and "=" in a}
used = parse_kv(opt.get("--used", "zed=0.938295,ricoh=0.705911"))
default = parse_kv(opt.get("--default", "zed=0.9,ricoh=0.5"))
acc = collections.defaultdict(lambda: [0.0, 0.0, 0])   # (camera, state) -> sum y, sum fe, views
for run in args:
    st = collections.defaultdict(dict)
    for r in csv.DictReader(open(os.path.join(run, "lidar_bins.csv"))):
        st[r["door"]][int(r["cycle"])] = r["state"]
    def state(door, c):
        ks = [k for k in st.get(door, {}) if k <= c]
        return st[door][max(ks)] if ks else "none"
    views = collections.defaultdict(list)
    for r in csv.DictReader(open(os.path.join(run, "door_views_labelled.csv"))):
        if r["label"] != "1" or r["camera"] not in ("zed", "ricoh") or r["view_id"] == "-1": continue
        views[(r["door"], r["camera"], r["view_id"])].append(r)
    for (door, cam, _), rows in views.items():
        fe = sum(float(r["q"]) for r in rows) / len(rows) / used[cam]
        if not fe > 0: continue
        s = state(rows[0]["gt"], int(sorted(int(r["cycle"]) for r in rows)[len(rows) // 2]))
        if s not in ("shut", "open"): continue
        a = acc[(cam, s)]
        a[0] += sum(float(r["hit"]) for r in rows) / len(rows); a[1] += fe; a[2] += 1
for cam in ("zed", "ricoh"):
    ys, fes, ns = acc[(cam, "shut")]; yo, feo, no = acc[(cam, "open")]
    # The hit weight is the report's reliability-weighted outcome and the front end only an envelope, so the raw ratio y/fe is not on the
    # table's scale (it exceeds 1 on shut views). Calibrate on the SHUT views: k maps their ratio onto the table's hit rate, and the
    # open views are read through the same k. The prior is the table's own hit rate (no leaf effect) worth PRIOR_N views.
    k = used[cam] / (ys / fes) if ys > 0 and fes > 0 else 1.0
    h = (PRIOR_N * used[cam] + k * yo) / (PRIOR_N + feo)
    print(f"{cam},hit_rate_open,{h:.6f},{no},leaf_state_labels shut_views={ns} k={k:.3f} sum_y_open={yo:.2f} sum_fe_open={feo:.2f}")
