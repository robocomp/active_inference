#!/usr/bin/env python3
"""live_calib.py — sweep a Webots door through leaf angles and ask the VLM about it from known robot poses.

Fills the bins the offline set cannot (ajar 15-45 deg, half 45-75 deg): Webots doors in the saved runs are only ever
shut (0) or opened without a hint (90). Writes one CSV row per (pose, door angle, camera, prompt variant):

    ts_ms, door, phi_true_deg, phi_bin, clear_w_m, passable, pose, robot_x, robot_y, robot_yaw, camera, dist_m, side,
    oblique_deg, bearing_deg, variant, state, visible, latency_s, raw, crop

and the crops it showed the model (so every answer can be audited by eye).

PRECONDITIONS (nothing here starts or stops anything):
  * Webots running with the shadow bridge (DoorControl on :10017, Webots2Robocomp on :10006), ollama with
    qwen3-vl:8b-instruct.
  * A frame source, one of
      --source snapshot  (default) room_concept running with its camera viewer: the script touches
                         room_concept/tmp/camera_snapshot and picks up tmp/snapshots/{zed,ricoh}_<ts>.png. ZED AND Ricoh,
                         but the frames carry room_concept's DEBUG OVERLAYS (the offline overlay set measures that).
      --source zeddump   retina's [ZedDump] with min_move_m = 0 and min_turn_deg = 0 (else a PARKED robot saves nothing;
                         ~15 frames/s x 100 KB while it runs — switch it back afterwards). ZED only, NO overlays, and the
                         Webots ground-truth pose comes with each frame. Cleanest; prefer it for the ZED table.
  * Nothing else may move the doors meanwhile: controller STOPPED, and door_concept either not running or with no
    `open` affordance being executed (it calls the same DoorControl). Not during a crossing run.
  * The robot PARKED. Either drive it to each pose yourself (one --pose per invocation, or --here) or pass --teleport
    to have the script put it there (Webots supervisor; room_concept will see a kidnap and relocalise — fine for this
    measurement, NOT during an estimation run).

USAGE
  # 1. dry run: checks connectivity READ-ONLY (door list, robot pose, ollama), prints the plan, moves nothing
  python3 tools/vlm_door_calib/live_calib.py --dry-run
  # 2. geometry check at the current pose: grabs ONE frame, writes check_<cam>.png with the projected doorway box,
  #    moves nothing. Look at it: the red box must sit on the doorway (Ricoh: the door must be centred).
  python3 tools/vlm_door_calib/live_calib.py --check --here --doors DOOR_1
  # 3. the sweep, at the current pose
  python3 tools/vlm_door_calib/live_calib.py --here --doors DOOR_1 --final open
  # 4. the sweep over the pose table (data/live_poses.csv), teleporting
  python3 tools/vlm_door_calib/live_calib.py --teleport --doors DOOR_1 --final open --source zeddump
Then:  python3 tools/vlm_door_calib/live_calib.py --report data/live_<stamp>/live.csv

Door angles: DoorControl.requestDoor(OpenDoor, apertureHint = -phi) holds any angle in [-pi, 0] (the bridge clamps;
caps.continuousAngle = true); CloseDoor drives it to 0. The leaf swings at the bridge's DoorControl.SwingRate
(0.5 rad/s); we wait for RequestState.Delivered (= the leaf arrived), then a settle time, then a FRESH frame.
--final {open,close,leave} says what to leave the door at; it is required unless --dry-run/--check.
"""
import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import door_vlm as dv  # noqa: E402

HERE = Path(__file__).resolve().parent
AI = HERE.parent.parent                                        # active_inference/
SNAP_TRIGGER = AI / "room_concept/tmp/camera_snapshot"
SNAP_DIR = AI / "room_concept/tmp/snapshots"
ZEDDUMP_DIR = AI / "retina/etc/zed_frames"
BRIDGE_GEN = "/home/pbustos/robocomp/components/webots-shadow/components/shadow-bridge-dds/generated"
SWING_RATE = 0.5                                               # rad/s, bridge [DoorControl] SwingRate


def ice_proxies():
    import Ice
    Ice.loadSlice(f"-I{BRIDGE_GEN} --all {BRIDGE_GEN}/DoorControl.ice")
    Ice.loadSlice(f"-I{BRIDGE_GEN} --all {BRIDGE_GEN}/Webots2Robocomp.ice")
    import RoboCompDoorControl as D
    import RoboCompWebots2Robocomp as W
    ic = Ice.initialize(sys.argv[:1])
    door = D.DoorControlPrx.checkedCast(ic.stringToProxy("doorcontrol:tcp -h localhost -p 10017"))
    web = W.Webots2RobocompPrx.checkedCast(ic.stringToProxy("webots2robocomp:tcp -h localhost -p 10006"))
    return ic, door, web, D, W


def robot_pose(web):
    """World metres + HEADING. Shadow drives along its local +y, so heading = webots_yaw + pi/2 = -robot_gt_angle
    (checked live 2026-10-12: quaternion yaw 85.9 deg while robot_gt_angle = -3.064 rad; and the crossing start pose
    `tp.py set -5.0 6.8 0` faces DOOR_1, i.e. +y)."""
    p = web.getObjectPose("shadow")
    q = p.orientation
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    heading = math.atan2(math.sin(yaw + math.pi / 2), math.cos(yaw + math.pi / 2))
    return p.position.x / 1000.0, p.position.y / 1000.0, heading, p


def teleport(web, W, x, y, heading, cur):
    th = heading - math.pi / 2                                 # Webots yaw (see robot_pose)
    pose = W.ObjectPose(W.Vector3(x * 1000, y * 1000, cur.position.z + 20.0),
                        W.Quaternion(0.0, 0.0, math.sin(th / 2), math.cos(th / 2)))
    web.setObjectPose("shadow", pose)
    time.sleep(4.0)                                            # land + let the localiser and the viewer catch up


def set_door(door, D, door_id, phi_rad, prev_phi, purpose="vlm door calibration"):
    req = D.DoorRequest(doorId=door_id, door=D.DoorRef(id=door_id, pose=D.Pose2D(0, 0, 0), width=1.0),
                        action=D.DoorAction.OpenDoor if phi_rad > 1e-3 else D.DoorAction.CloseDoor,
                        apertureHint=-phi_rad if phi_rad > 1e-3 else 0.0, purpose=purpose)
    ack = door.requestDoor(req)
    if not ack.accepted:
        raise RuntimeError(f"{door_id}: request refused ({ack.refusal})")
    deadline = time.time() + abs(phi_rad - (prev_phi if prev_phi is not None else math.pi)) / SWING_RATE + 4.0
    while time.time() < deadline:
        st = door.getRequestStatus(ack.requestId)
        if st.state == D.RequestState.Delivered:
            break
        if st.state == D.RequestState.Aborted:
            raise RuntimeError(f"{door_id}: request aborted")
        time.sleep(0.2)
    else:
        print(f"  [warn] {door_id} not Delivered within the expected swing time — continuing")
    time.sleep(0.8)                                            # settle (renderer + one camera period)


def grab(source, cameras, timeout=6.0):
    """Fresh frames AFTER now. -> {camera: (bgr, ts_ms, gt_pose|None)}"""
    t0_ms = int(time.time() * 1000)
    got = {}
    if source == "snapshot":
        SNAP_TRIGGER.parent.mkdir(parents=True, exist_ok=True)
        before = set(SNAP_DIR.glob("*.png")) if SNAP_DIR.exists() else set()
        SNAP_TRIGGER.touch()
        os.utime(SNAP_TRIGGER, None)
        end = time.time() + timeout
        while time.time() < end and len(got) < len(cameras):
            time.sleep(0.25)
            if not SNAP_DIR.exists():
                continue
            for f in sorted(set(SNAP_DIR.glob("*.png")) - before):
                cam = f.name.split("_")[0]
                if cam in cameras and cam not in got:
                    time.sleep(0.2)                            # let the writer finish
                    img = cv2.imread(str(f))
                    if img is not None:
                        got[cam] = (img, int(f.stem.split("_")[-1]), None)
    else:   # zeddump
        csvp = ZEDDUMP_DIR / "poses.csv"
        end = time.time() + timeout
        while time.time() < end and "zed" not in got:
            time.sleep(0.3)
            with open(csvp, "rb") as fh:                       # last line only
                fh.seek(max(0, os.path.getsize(csvp) - 4096))
                last = fh.read().decode(errors="ignore").strip().splitlines()[-1].split(",")
            if last[0].isdigit() and int(last[0]) > t0_ms + 300:
                img = cv2.imread(str(ZEDDUMP_DIR / last[1]))
                if img is not None and last[14] == "1":
                    got["zed"] = (img, int(last[0]), (float(last[10]), float(last[11]), -float(last[12])))
    return got


def crops_for(cam, img, pose_xyh, door_xyr):
    """-> list of (crop_bgr, boxed_bgr|None, geometry dict) for one camera frame."""
    x, y, h = pose_xyh
    if cam == "zed":
        box = dv.zed_door_box((x, y), h, door_xyr)
        if not box["in_view"]:
            return []
        (x0, y0, _), crop = dv.square_crop(img, box)
        c, _, n = dv.door_frame(*door_xyr)
        rel = c - np.array([x, y])
        brg = math.degrees(math.atan2(rel[1], rel[0]) - h)
        return [(crop, dv.draw_box(crop, box, (x0, y0)), dict(dist=box["dist"], side=box["side"],
                                                             oblique=box["oblique"],
                                                             bearing=(brg + 180) % 360 - 180))]
    view, g = dv.ricoh_door_view(img, (x, y), h, door_xyr)
    c, _, n = dv.door_frame(*door_xyr)
    C = np.array([x, y])
    los = (c - C) / np.linalg.norm(c - C)
    return [(view, None, dict(dist=g["dist"], side=int(np.sign((C - c) @ n)),
                              oblique=math.degrees(math.acos(abs(los @ n))), bearing=g["bearing_deg"]))]


def load_poses(path, doors):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(l for l in f if not l.startswith("#")):
            if r["door"] in doors:
                rows.append(dict(label=r["label"], door=r["door"], x=float(r["x"]), y=float(r["y"]),
                                 th=float(r["theta"])))
    return rows


def report(path):
    import pandas as pd
    d = pd.read_csv(path)
    d = d[d.state != "?"]
    order = ["closed(0)", "ajar(15-45)", "half(45-75)", "open(75-90)"]
    for (v, cam), g in d.groupby(["variant", "camera"]):
        print(f"\n### {v} / {cam}: n={len(g)}, latency median {g.latency_s.median():.2f}s")
        m = pd.crosstab(g.phi_bin, g.state, normalize="index").reindex(order).round(3)
        m["n"] = g.groupby("phi_bin").size().reindex(order)
        print(m.to_string())
        print("P(answer | passable):")
        print(pd.crosstab(g.passable, g.state, normalize="index").round(3).to_string())
        g = g.assign(dbin=pd.cut(g.dist_m, [0, 2.5, 4, 8]).astype(str))
        print("P(open | phi_bin, distance):")
        print(g.assign(o=g.state == "open").pivot_table(index="phi_bin", columns="dbin", values="o",
                                                        aggfunc="mean").reindex(order).round(2).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="read-only checks + the plan; moves nothing")
    ap.add_argument("--check", action="store_true", help="grab one frame here, write the projected box; moves nothing")
    ap.add_argument("--report", metavar="CSV", help="summarise a finished live.csv")
    ap.add_argument("--doors", default="DOOR_1", help="comma list of DoorControl ids (DOOR_0,DOOR_1)")
    ap.add_argument("--phis", default="0,10,20,30,40,50,60,70,80,90", help="leaf angles, degrees")
    ap.add_argument("--poses", default=str(HERE / "data/live_poses.csv"))
    ap.add_argument("--here", action="store_true", help="use the robot's current pose only (no pose table)")
    ap.add_argument("--teleport", action="store_true", help="teleport to each pose of the table (Webots supervisor)")
    ap.add_argument("--source", choices=["snapshot", "zeddump"], default="snapshot")
    ap.add_argument("--cameras", default="zed,ricoh")
    ap.add_argument("--variants", default="plain,passable,json_box")
    ap.add_argument("--repeats", type=int, default=2, help="fresh frames per (pose, angle)")
    ap.add_argument("--shuffle", action="store_true", help="visit the angles in random order (no hysteresis)")
    ap.add_argument("--final", choices=["open", "close", "leave"], help="leave each door at this state at the end")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.report:
        return report(a.report)

    cams = [c for c in a.cameras.split(",") if c]
    if a.source == "zeddump":
        cams = [c for c in cams if c == "zed"]
    doors = a.doors.split(",")
    phis = [float(p) for p in a.phis.split(",")]

    ic, door, web, D, W = ice_proxies()
    try:
        caps = door.getCapabilities()                          # read-only
        dpose = {d.id: (d.pose.x / 1000.0, d.pose.y / 1000.0, d.pose.angle) for d in caps.doors}
        print(f"DoorControl: canActuate={caps.canActuate} continuousAngle={caps.continuousAngle} doors="
              + ", ".join(f"{k}@({v[0]:.2f},{v[1]:.2f},{math.degrees(v[2]):.0f}deg)" for k, v in dpose.items()))
        missing = [d for d in doors if d not in dpose]
        if missing:
            sys.exit(f"doors not advertised by the bridge: {missing}")
        x, y, yaw, cur = robot_pose(web)
        print(f"robot (Webots GT): ({x:.2f}, {y:.2f}) heading {math.degrees(yaw):.1f} deg")
        poses = ([dict(label="here", door=d, x=x, y=y, th=yaw) for d in doors] if (a.here or a.check)
                 else load_poses(a.poses, doors))
        n_q = len(poses) * len(phis) * a.repeats * len(cams) * len(a.variants.split(","))
        print(f"plan: {len(poses)} pose(s) x {len(phis)} angles x {a.repeats} frames x {len(cams)} camera(s) x "
              f"{len(a.variants.split(','))} variant(s) = {n_q} queries (~{n_q * 1.2 / 60:.0f} min of VLM) + "
              f"{len(poses) * len(phis) * (3 + 1.0 * a.repeats):.0f} s of swings/settling; source={a.source}")
        for p in poses:
            b = dv.zed_door_box((p["x"], p["y"]), p["th"], dpose[p["door"]])
            print(f"  {p['label']:14s} {p['door']} robot ({p['x']:.2f},{p['y']:.2f},{math.degrees(p['th']):.0f}deg)"
                  f"  dist {b['dist']:.2f} m side {b['side']:+d} oblique {b['oblique']:.0f}deg"
                  f"  in ZED view: {b['in_view']}")
        try:
            import urllib.request
            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3).read()
            print("ollama: reachable")
        except Exception as e:  # noqa: BLE001
            print(f"ollama: NOT reachable ({e})")
        if a.dry_run:
            print("dry run: nothing moved.")
            return 0

        out = Path(a.out) if a.out else HERE / f"data/live_{time.strftime('%Y%m%d_%H%M%S')}"
        out.mkdir(parents=True, exist_ok=True)
        if a.check:
            fr = grab(a.source, cams)
            for cam, (img, ts, gt) in fr.items():
                ph = gt if gt else (x, y, yaw)
                for d in doors:
                    for k, (crop, boxed, g) in enumerate(crops_for(cam, img, ph, dpose[d])):
                        cv2.imwrite(str(out / f"check_{cam}_{d}.png"), boxed if boxed is not None else crop)
                        print(f"check: {out}/check_{cam}_{d}.png  {g}")
                    if cam == "zed":
                        b = dv.zed_door_box(ph[:2], ph[2], dpose[d])
                        full = img.copy()
                        cv2.rectangle(full, (int(b["u0"]), int(b["v0"])), (int(b["u1"]), int(b["v1"])), (0, 0, 255), 3)
                        cv2.imwrite(str(out / f"check_full_zed_{d}.png"), full)
            if not fr:
                print("check: no fresh frame (is the source running?)")
            return 0
        if a.final is None:
            sys.exit("--final {open,close,leave} is required for a sweep")

        f = open(out / "live.csv", "w", newline="")
        w = csv.writer(f)
        w.writerow(["ts_ms", "door", "phi_true_deg", "phi_bin", "clear_w_m", "passable", "pose", "robot_x", "robot_y",
                    "robot_yaw", "camera", "dist_m", "side", "oblique_deg", "bearing_deg", "variant", "state",
                    "visible", "latency_s", "raw", "crop"])
        rng = np.random.default_rng(0)
        last_pose = None
        prev_phi = {d: None for d in doors}
        for p in poses:
            if a.teleport and p["label"] != "here" and last_pose != p["label"]:
                _, _, _, cur = robot_pose(web)
                print(f"teleport -> {p['label']} ({p['x']:.2f},{p['y']:.2f},{math.degrees(p['th']):.0f}deg)")
                teleport(web, W, p["x"], p["y"], p["th"], cur)
            last_pose = p["label"]
            order = list(rng.permutation(phis)) if a.shuffle else phis
            for phi_deg in order:
                phi = math.radians(phi_deg)
                print(f"[{p['label']}] {p['door']} -> {phi_deg:.0f} deg")
                set_door(door, D, p["door"], phi, prev_phi[p["door"]])
                prev_phi[p["door"]] = phi
                for rep in range(a.repeats):
                    fr = grab(a.source, cams)
                    if not fr:
                        print("  [warn] no fresh frame")
                    for cam, (img, ts, gt) in fr.items():
                        rx, ry, ryaw, _ = robot_pose(web)
                        ph = gt if gt else (rx, ry, ryaw)
                        for k, (crop, boxed, g) in enumerate(crops_for(cam, img, ph, dpose[p["door"]])):
                            cp = out / f"{p['label']}_{p['door']}_{int(phi_deg):03d}_{cam}_{rep}.jpg"
                            cv2.imwrite(str(cp), crop)
                            for v in a.variants.split(","):
                                if dv.VARIANTS[v]["box"] and boxed is None:
                                    continue
                                r = dv.ask(boxed if dv.VARIANTS[v]["box"] else crop, v)
                                w.writerow([ts, p["door"], phi_deg, dv.phi_bin(phi_deg),
                                            round(dv.clear_width(phi), 3), int(dv.passable(phi)), p["label"],
                                            round(ph[0], 3), round(ph[1], 3), round(ph[2], 4), cam, round(g["dist"], 2),
                                            g["side"], round(g["oblique"], 1), round(g["bearing"], 1), v, r["state"],
                                            r["visible"], r["latency_s"], r["raw"], cp.name])
                                f.flush()
                                print(f"  {cam:5s} {v:9s} -> {r['state']:7s} ({r['latency_s']:.2f}s)")
        f.close()
        if a.final != "leave":
            for d in doors:
                set_door(door, D, d, math.pi / 2 if a.final == "open" else 0.0, prev_phi[d])
                print(f"{d} left {a.final}")
        print(f"done: {out}/live.csv  ->  python3 {Path(__file__).name} --report {out}/live.csv")
    finally:
        ic.destroy()
    return 0


if __name__ == "__main__":
    sys.exit(main())
