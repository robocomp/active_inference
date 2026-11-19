#!/usr/bin/env python3
"""Summarise countable_concept's timing_log.csv: per stage mean / median / p95 ms and share of the tick,
for all ticks and split by "tick had camera/ray data" vs "no data". Usage: timing_summary.py timing_log.csv"""
import csv, statistics, sys

STAGES = ["mask_refresh", "tracker", "foreign_claims", "lidar_pump", "rgb_stage", "ego_motion", "inst_total",
          "observe", "membership", "lidar_feed", "infer", "publish", "ricoh", "existence", "class_factor",
          "dashboard", "tick"]
# Sub-stages (CSV columns after "tick"), printed as indented rows right under their parent; not in "(unaccounted)".
SUB = {"ego_motion": ["ego_rt_room_body", "ego_rt_body_zed", "ego_robot_node", "ego_twist", "ego_sensor_model"],
       "existence": ["ex_silhouette", "ex_contour", "ex_lidar_carve", "ex_decide", "ex_removal"]}
NESTED = {"observe": "inst_total", "infer": "inst_total", "publish": "inst_total",
          "membership": "infer", "lidar_feed": "infer"}   # stage -> the stage that already contains it


def p95(v):
    v = sorted(v)
    return v[min(len(v) - 1, int(0.95 * len(v)))]


def report(title, rows):
    print(f"\n== {title}: {len(rows)} ticks ==")
    if not rows:
        return
    tick_mean = statistics.fmean(r["tick"] for r in rows)
    print(f"{'stage':<20}{'mean':>9}{'median':>9}{'p95':>9}{'share%':>8}")
    def line(name, v, note=""):
        print(f"{name:<20}{statistics.fmean(v):9.3f}{statistics.median(v):9.3f}{p95(v):9.3f}"
              f"{100 * statistics.fmean(v) / tick_mean if tick_mean else 0:8.1f}{note}")
    for s in STAGES:
        line(s, [r[s] for r in rows], f"  (inside {NESTED[s]})" if s in NESTED else "")
        for c in SUB.get(s, []):
            if c in rows[0]:
                line("   " + c, [r[c] for r in rows], f"  (inside {s})")
    if "rt_calls" in rows[0]:
        v = [r["rt_calls"] for r in rows]
        print(f"{'rt_calls/tick':<20}{statistics.fmean(v):9.2f}{statistics.median(v):9.0f}{p95(v):9.0f}"
              "   (get_transformation_matrix calls)")
    other = tick_mean - sum(statistics.fmean(r[s] for r in rows) for s in
                            ["mask_refresh", "tracker", "foreign_claims", "lidar_pump", "rgb_stage", "ego_motion",
                             "inst_total", "ricoh", "existence", "class_factor", "dashboard"])
    print(f"{'(unaccounted)':<20}{other:9.3f}{'':18}{100 * other / tick_mean if tick_mean else 0:8.1f}")


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    with open(sys.argv[1], newline="") as f:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    report("all ticks", rows)
    data = [r for r in rows if r["n_cam"] + r["n_ray"] > 0]
    report("ticks WITH camera/ray data", data)
    report("ticks with NO data", [r for r in rows if r["n_cam"] + r["n_ray"] == 0])
    print(f"\ninstances/tick mean {statistics.fmean(r['n_inst'] for r in rows):.2f}  cam {statistics.fmean(r['n_cam'] for r in rows):.2f}"
          f"  ray-only {statistics.fmean(r['n_ray'] for r in rows):.2f}  none {statistics.fmean(r['n_none'] for r in rows):.2f}")


main()
