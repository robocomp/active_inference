#!/usr/bin/env python3
"""Grade an exploration tour run (spec docs/superpowers/specs/2026-11-05-exploration-tour-design.md, Goals).

usage: grade_explore_tour.py <run_dir>     -> <run_dir>/explore_report.txt + explore_rooms.csv (also printed)

Per estimated room: born (s), promoted (s), how the robot left it (exploration_drive.csv), doors found in it, and — when
Webots ground truth has a polygon for the room the robot was truly in at promotion — the IoU of the learnt outline after a
rigid alignment (wbt_rooms.align_and_score). The ground-truth room is chosen by PLACE (the polygon containing the robot's
true position at promotion), not by name: true_room.py and wbt_rooms.py name the room behind DOOR_1 differently.
wbt_rooms.py has no polygon for the start room (room_1 is closed by a mesh partition): its IoU is reported as n/a.
"""
import csv, glob, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# wbt_rooms.py (Webots ground-truth polygons) lives untracked in the main checkout's scripts dir: look there too
sys.path.append("/home/pbustos/robocomp/components/active_inference/room_concept/estimation_scripts/scripts")


def proto_events(run):
    """{room_index_in_order: {'born': s, 'promoted': s, 'promoted_ms': ms}} from proto_room.csv, in birth order."""
    path = os.path.join(run, "proto_room.csv")
    rows = list(csv.DictReader(open(path))) if os.path.exists(path) else []
    if not rows:
        return [], None
    t0 = int(rows[0]["ts_ms"])
    rooms = []
    for r in rows:
        ev = r.get("event", "")
        if ev.startswith("startup_born") or "crossing_born" in ev:
            rooms.append({"born": (int(r["ts_ms"]) - t0) / 1000.0, "promoted": None, "promoted_ms": None,
                          "area": None, "how": "crossing" if "crossing_born" in ev else "start", "door": r.get("door", "")})
        elif ev == "promoted" and rooms and rooms[-1]["promoted"] is None:
            rooms[-1].update(promoted=(int(r["ts_ms"]) - t0) / 1000.0, promoted_ms=int(r["ts_ms"]),
                             area=float(r["fs_area"]) if r.get("fs_area") not in (None, "", "nan") else None)
    return rooms, t0


def drive_rows(run):
    path = os.path.join(run, "exploration_drive.csv")
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def gt_at(run, t_ms):
    path = os.path.join(run, "gt_pose.tsv")
    if not os.path.exists(path) or t_ms is None:
        return None
    best = None
    for r in csv.DictReader(open(path), delimiter="\t"):
        try:
            t, x, y = int(r["t_ms"]), float(r["x"]), float(r["y"])
        except (KeyError, ValueError):
            continue
        if best is None or abs(t - t_ms) < abs(best[0] - t_ms):
            best = (t, x, y)
    return best[1:] if best else None


def learnt_polygons(run):
    """[(name, [(x, y)...])] for every learnt layout SVG, newest last per name; x un-mirrored (stored negated)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(run, "*.svg")), key=os.path.getmtime):
        base = os.path.basename(f)
        m = re.match(r"(room_\d+)-", base)
        name = m.group(1) if m else "start"
        pts = re.search(r'points="([^"]+)"', open(f).read())
        if pts:
            out[name] = [(-float(a), float(b)) for a, b in (p.split(",") for p in pts.group(1).split())]
    return out


def room_names(run, rooms):
    """The estimated name of each room of `rooms` (birth order): 'start' for the start room (its SVG carries no room_ prefix),
    and for a crossing-born room the name ltsm_agent promoted through the door it was born by ([promotion] PROMOTING <room>
    ... crossed door <door> ...). Never by list position: SVGs may be missing (an older run kept only the newest 3), and
    index pairing grades each one against another room. A door name may be recycled and a crossing into a RECALLED room
    is promoted as well, so every crossing of proto_room.csv consumes its promotion in order. A room never promoted has
    no name."""
    path = os.path.join(run, "logs", "ltsm_agent.log")
    promos = re.findall(r"\[promotion\] PROMOTING (room_\d+) .*? crossed door (\S+) ",
                        open(path, errors="replace").read()) if os.path.exists(path) else []
    pos = 0                                        # the next promotion not yet consumed (crossings and promotions are in order)
    by_row = {}                                    # proto_room.csv row index of a crossing -> the name promoted for it
    rows = list(csv.DictReader(open(os.path.join(run, "proto_room.csv"))))
    for n, r in enumerate(rows):
        if "born" not in r.get("event", "") or r["event"].startswith("startup_born"):
            continue
        j = next((k for k in range(pos, len(promos)) if promos[k][1] == r.get("door", "")), None)
        if j is not None:
            by_row[n] = promos[j][0]
            pos = j + 1
    crossing_born_rows = [n for n, r in enumerate(rows) if "crossing_born" in r.get("event", "")]
    names, c = [], 0
    for i, r in enumerate(rooms):
        if r["how"] == "start":
            names.append("start")
            continue
        names.append(by_row.get(crossing_born_rows[c], f"#{i}") if c < len(crossing_born_rows) else f"#{i}")
        c += 1
    return names


def point_in(poly, x, y):
    inside, n = False, len(poly)
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def main(run):
    rooms, _ = proto_events(run)
    drive = drive_rows(run)
    learnt = learnt_polygons(run)
    try:
        import wbt_rooms
        gt, _doors, _walls = wbt_rooms.rooms()   # {name: {'poly':..., 'area':...}}
    except Exception as e:   # the grader still reports everything else
        gt, gt_err = {}, str(e)
    else:
        gt_err = ""
    # estimated room names, by the name ltsm_agent gave each room (never by list position); the start room is room_1
    names = room_names(run, rooms)
    # a room RECOGNISED as a remembered one is not promoted again: it is restored from memory (ltsm's swap ACCEPTED)
    recognised = {}
    try:
        for ln in open(os.path.join(run, "logs", "ltsm_agent.log"), errors="replace"):
            m = re.search(r"\[swap\] ACCEPTED: (room_\d+) adopted as (\S+?);", ln)
            if m: recognised[m.group(1)] = m.group(2)
    except OSError:
        pass
    lines, table = [], []
    for i, r in enumerate(rooms):
        est = names[i]
        pos = gt_at(run, r["promoted_ms"])
        truth, iou = None, "n/a"
        if pos is not None:
            for gname, g in gt.items():
                if point_in([tuple(p) for p in g["poly"]], *pos):
                    truth = gname
        if truth is not None and est in learnt:
            sc = wbt_rooms.align_and_score(gt[truth]["poly"], learnt[est])
            iou = f"{sc['iou']:.3f}"
        elif truth is not None:
            iou = f"n/a (no SVG for {est} in this run)"
        elif truth is None:
            iou = "n/a (no GT polygon contains the robot at promotion" + (" — start room" if r["how"] == "start" else "") + ")"
        leaves = [d for d in drive if (d["room"] == ("room_1" if est == "start" else est))]
        causes = ",".join(f"{d['cause']}->{d['door'] or '-'}" for d in leaves) or "-"
        found = sorted({c.split(":")[0] for d in leaves for c in d["candidates"].split(";") if c and c.split(":")[1] == "1"})
        row = {"room": "room_1" if est == "start" else est, "how": r["how"], "born_s": r["born"], "promoted_s": r["promoted"],
               "area_m2": r["area"], "truth": truth or "-", "iou": iou, "decisions": causes, "doors_found": " ".join(found),
               "recognised_as": recognised.get(est, "")}
        if r["promoted"] is None and est in recognised:
            iou = f"— recognised as {recognised[est]} (restored from memory, not re-estimated)"
            row["iou"] = iou
        table.append(row)
        lines.append(f"{row['room']:7} {r['how']:8} born {r['born']:7.1f} s  promoted "
                     f"{('%.1f s' % r['promoted']) if r['promoted'] is not None else ('RECOGNISED' if est in recognised else 'NEVER'):>9}  area "
                     f"{('%.1f' % r['area']) if r['area'] else '-':>5} m2  truth {row['truth']:7} IoU {iou}\n"
                     f"        decisions: {causes}\n        doors found: {row['doors_found'] or '-'}")
    max_time = sum(1 for d in drive if d["cause"] == "max-time")
    done = any(d["cause"] == "done" for d in drive)
    head = (f"run {run}: {len(rooms)} room(s), {sum(1 for r in rooms if r['promoted'] is not None)} promoted, "
            f"{sum(1 for d in drive if d['door'] and d['is_return'] == '0')} crossing decision(s), "
            f"{sum(1 for d in drive if d['is_return'] == '1')} return(s), {max_time} max-time leave(s), "
            f"{'DONE' if done else 'not done'}" + (f"  [ground truth unavailable: {gt_err}]" if gt_err else ""))
    report = head + "\n" + "\n".join(lines) + "\n"
    open(os.path.join(run, "explore_report.txt"), "w").write(report)
    with open(os.path.join(run, "explore_rooms.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()) if table else ["room"])
        w.writeheader()
        w.writerows(table)
    print(report, end="")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
