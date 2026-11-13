#!/usr/bin/env python3
"""det_census.py <out.csv> [seconds] — WHAT DOES RETINA DETECT? A passive DSR observer (agent id 959, never writes).

Reads retina's front-camera mask node (`masks_zed`) every 100 ms and appends one row per new mask slice:
  t_s, frame, label, conf, range_m (camera-frame centroid norm), pose (the robot's TRUE Webots pose, from a side file the
  runner writes: crossing/<run>/pose.txt, "x y yaw" — optional).
Used to check a new concept agent's class BEFORE cloning the agent: is the label detected at all on the Webots model, from
where, how confidently. Stops after `seconds` (default: until killed with SIGTERM/SIGINT).
"""
import csv, math, os, signal, sys, time
from PySide6.QtCore import QCoreApplication, QTimer
import pydsr

OUT = sys.argv[1]
SECONDS = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
POSE = os.path.join(os.path.dirname(os.path.abspath(OUT)), "pose.txt")

app = QCoreApplication(sys.argv)
G = pydsr.DSRGraph(0, "det_census", 959)
f = open(OUT, "w", newline="")
w = csv.writer(f)
w.writerow(["t_s", "frame", "label", "conf", "range_m", "pose"])
t0 = time.time()
last_frame = None


def attr(node, name):
    try:
        return node.attrs[name].value if name in node.attrs else None
    except Exception:
        return None


def labels_of(raw, n):
    if raw is None:
        return ["?"] * n
    if isinstance(raw, str):
        for sep in ("|", "\n", ";", ","):
            if sep in raw:
                return [s.strip() for s in raw.split(sep)][:n]
        return [raw] * n if n == 1 else [raw]
    return [str(x) for x in raw][:n]


def poll():
    global last_frame
    if SECONDS and time.time() - t0 > SECONDS:
        app.quit(); return
    node = G.get_node("masks_zed")
    if node is None:
        return
    stamp = attr(node, "mask_timestamp_ms")
    if stamp is None or stamp == last_frame:
        return
    last_frame = stamp
    seq = lambda v: [] if v is None else list(v)   # pydsr returns numpy arrays: `or []` is ambiguous on them
    count = attr(node, "mask_count")
    n = int(count) if count is not None else 0
    labels = labels_of(attr(node, "mask_labels"), n)
    confs = seq(attr(node, "mask_confidences"))
    cents = seq(attr(node, "mask_centroids_xyz"))
    pose = open(POSE).read().strip() if os.path.exists(POSE) else ""
    for i in range(n):
        lab = labels[i] if i < len(labels) else "?"
        conf = confs[i] if i < len(confs) else float("nan")
        rng = math.sqrt(sum(c * c for c in cents[3 * i:3 * i + 3])) if len(cents) >= 3 * i + 3 else float("nan")
        w.writerow([f"{time.time() - t0:.2f}", stamp, lab, f"{conf:.3f}", f"{rng:.2f}", pose])
    f.flush()


signal.signal(signal.SIGTERM, lambda *_: app.quit())
signal.signal(signal.SIGINT, lambda *_: app.quit())
timer = QTimer(); timer.timeout.connect(poll); timer.start(100)
app.exec()
f.close()
