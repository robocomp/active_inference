#!/usr/bin/env python3
"""pose_lag_anatomy — how old is the newest pose, and WHY?

★READ THIS BEFORE ACTING ON THE NUMBER. The stamp on an RT block is the LIDAR CAPTURE TIME, not the
time the pose was written (room_concept.cpp: res.timestamp_ms = lidar.second). So "the newest pose is
aged 68 ms" does NOT mean room_concept took 68 ms. It means the scan that pose describes was captured
68 ms ago, and most of that was gone before room_concept saw the data at all.

Measured on Shadow 2026-09-10, decomposing the same 68 ms:
    ~37 ms   the lidar scan is ALREADY THIS OLD when any consumer receives it (controller overlay
             CSV, t_ms - lidar_ts). Sensor + transport, before our code runs.
    ~16 ms   half a publish period at 31 Hz -- what a consumer asking "now" pays for free.
    ~15 ms   everything else: queueing, DDS, and the pose computation itself, which the author
             measures at under 3 ms when the pose is predicted rather than optimised.
An earlier version of this tool called that whole residual "delay inside the producer" and concluded
the producer was slow. It cannot tell the difference between a slow producer and data that was
already old, and the difference is the entire conclusion. It now reports the split and refuses to
name a cause.

So the two candidate cures are NOT "publish faster" vs "optimise the producer". They are:
  · shorten the LIDAR delivery path (the 37 ms), which dominates; and before touching anything,
    check whether the scan stamp is the START or the END of a sweep -- a 10 Hz spinning sensor
    accumulates for 100 ms, and that convention alone can account for tens of milliseconds;
  · publish more often, worth ~16 ms at best.

⚠ pose_age_ms in the controller's overlay CSV is NOT this quantity. It is time since the pose VALUE
last moved by more than an epsilon, so it reads large whenever the robot is nearly still. Comparing
the two as if they measured the same thing is how this file's first reading came out 2x off.

  usage:  python3 pose_lag_anatomy.py
"""
import sys, time, statistics
from PySide6.QtCore import QCoreApplication, QTimer
import pydsr
app = QCoreApplication(sys.argv); G = pydsr.DSRGraph(0, "lag_anatomy", 957)
fresh, spacing = [], []
def attr(o,n):
    try: return list(o.attrs[n].value) if n in o.attrs else None
    except Exception: return None
def grab():
    r = G.get_nodes_by_type("robot"); m = G.get_nodes_by_type("room")
    if not r or not m: return
    e = G.get_edge(r[0].id, m[0].id, "RT")
    if e is None: return
    ts = attr(e, "rt_timestamps")
    if not ts: return
    now = time.time()*1000.0
    good = sorted(t for t in ts if t != 0)
    fresh.append(now - good[-1])
    d = [good[i+1]-good[i] for i in range(len(good)-1)]
    spacing.extend(x for x in d if 0 < x < 1000)
def report():
    if not fresh: print("no data"); app.quit(); return
    f = sorted(fresh); s = sorted(spacing)
    print("pose publishing, measured on the live edge:")
    print("  new pose every       p50 %5.1f ms  p95 %5.1f ms   => %.1f Hz" % (s[len(s)//2], s[int(len(s)*.95)], 1000.0/s[len(s)//2]))
    print("  newest pose is aged  p50 %5.1f ms  p95 %5.1f ms" % (f[len(f)//2], f[int(len(f)*.95)]))
    half = s[len(s)//2]/2.0
    extra = f[len(f)//2] - half
    print()
    print("  half a publish period (unavoidable at this rate): %5.1f ms" % half)
    print("  everything else (data age on arrival + queue + compute): %5.1f ms" % extra)
    if extra > half:
        print("  -> publishing faster would recover at most the %.1f ms above." % half)
        print("     The rest is NOT necessarily producer slowness: the stamp is the SENSOR CAPTURE")
        print("     time, so most of it is usually the age of the data on arrival. Split it with")
        print("     the controller overlay CSV (t_ms - lidar_ts) before blaming any component.")
    else:
        print("  -> the publish RATE dominates. Publishing faster is the lever.")
    app.quit()
QTimer.singleShot(800, lambda: [QTimer.singleShot(250*i, grab) for i in range(60)])
QTimer.singleShot(17000, report)
sys.exit(app.exec())
