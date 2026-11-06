#!/usr/bin/env python3
"""Summarise tmp/newroom_bench/results.csv (written by WS_NEWROOM) into summary.md.

usage: python3 -I tools/newroom_summary.py results.csv summary.md [title]
Statistics are over ALL runs and over VALID runs only (left_room == 0: the executed pose never left the TRUE polygon).
"""
import csv
import statistics as st
import sys


def q(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def block(rows, name):
    n = len(rows)
    if n == 0:
        return f"| {name} | 0 |" + " - |" * 12
    iou = [float(r["iou"]) for r in rows]
    rot4 = [float(r["iou_rot4"]) for r in rows]
    prom = [r for r in rows if r["promoted"] == "1"]
    fp = [int(r["frames_to_promotion"]) for r in prom]
    pe = [float(r["pose_err_med_m"]) for r in rows]
    ex = sum(r["exact_vertices"] == "1" for r in rows)
    return (f"| {name} | {n} | {st.median(iou):.3f} | {q(iou, .25):.3f}-{q(iou, .75):.3f} | {st.mean(iou):.3f} | {min(iou):.3f} | "
            f"{sum(x >= .90 for x in iou)} | {sum(x >= .95 for x in iou)} | {ex} | "
            f"{len(prom)}/{n} | {(st.median(fp) if fp else float('nan')):.0f} | {st.median(pe):.3f} | {st.median(rot4):.3f} |")


def main():
    src, dst = sys.argv[1], sys.argv[2]
    title = sys.argv[3] if len(sys.argv) > 3 else "new-room bench"
    rows = list(csv.DictReader(open(src, newline="")))
    valid = [r for r in rows if r["left_room"] == "0"]
    out = [f"# {title}", "", f"{len(rows)} runs, {len(valid)} valid (executed pose never outside the true polygon), "
           f"{len(rows) - len(valid)} flagged left_room.", "",
           "| set | n | IoU median | IoU IQR | IoU mean | IoU worst | >=0.90 | >=0.95 | exact vertices | promoted | median frames to promotion | pose err median (m) | IoU rot4 median |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
           block(rows, "all"), block(valid, "valid only"), ""]
    unp = [r for r in rows if r["promoted"] != "1"]
    out.append(f"Never promoted: {len(unp)} -> rooms {[r['room_id'] for r in unp]}")
    stops = {}
    for r in rows:
        stops[r["stop"]] = stops.get(r["stop"], 0) + 1
    out.append(f"Stop reasons: {stops}")
    empt = [r["room_id"] for r in rows if int(r["est_verts"]) == 0]
    out.append(f"Empty outline (0 vertices): {empt}")
    worst = sorted(rows, key=lambda r: float(r["iou"]))[:8]
    out += ["", "Worst 8 by IoU: " + "; ".join(
        f"room {r['room_id']} (ord {r['order']}) IoU {float(r['iou']):.3f} verts {r['est_verts']}/{r['true_verts']} "
        f"promoted={r['promoted']} out={r['frames_outside']} seen={float(r['seen_frac']):.2f}" for r in worst)]
    for lo, hi in ((4, 6), (7, 9), (10, 12)):
        sub = [float(r["iou"]) for r in rows if lo <= int(r["order"]) <= hi]
        if sub:
            out.append(f"- orders {lo}-{hi}: n={len(sub)} IoU median {st.median(sub):.3f} worst {min(sub):.3f}")
    seen = [float(r["seen_frac"]) for r in rows]
    ip = [float(r["iou_promo"]) for r in rows if r["promoted"] == "1"]
    if ip:
        out.append(f"- OUTLINE STAGE alone (adopted outline at promotion, before any refiner edit), {len(ip)} promoted runs: "
                   f"IoU median {st.median(ip):.3f} IQR {q(ip, .25):.3f}-{q(ip, .75):.3f} mean {st.mean(ip):.3f} worst {min(ip):.3f}; "
                   f">=0.90 {sum(x >= .90 for x in ip)}, >=0.95 {sum(x >= .95 for x in ip)}")
        d = [float(r["iou"]) - float(r["iou_promo"]) for r in rows if r["promoted"] == "1"]
        out.append(f"- refiner effect (final IoU - promotion IoU): median {st.median(d):+.3f}, improved {sum(x > 1e-4 for x in d)}, "
                   f"worsened {sum(x < -1e-4 for x in d)}, unchanged {sum(abs(x) <= 1e-4 for x in d)}; worst {min(d):+.3f}")
    stuck = [r["room_id"] for r in rows if int(r["frames_blocked"]) > 0.5 * int(r["frames_run"])]
    out.append(f"- rooms where the robot was wedged by the bumper for most of the run (start inside body_radius of a wall): {stuck}")
    out.append(f"- seen fraction of true boundary: median {st.median(seen):.2f}, min {min(seen):.2f}")
    wall = [float(r["wall_s"]) for r in rows]
    out.append(f"- wall time per room: median {st.median(wall):.1f} s, total {sum(wall) / 60:.1f} min")
    open(dst, "w").write("\n".join(out) + "\n")
    print("\n".join(out))


main()
