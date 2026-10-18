#!/usr/bin/env python3
"""Offline check of the canvas stock scripts: each one runs through bootstrap.py (as chat_agent runs it) against a
synthetic snapshot — an L-shaped room, two doors, three objects, a fake ZED frame — and must write out.png.
A model-style script that tries the network must fail. Pass --keep DIR to keep the PNGs for a look."""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CANVAS = os.path.join(HERE, "..", "canvas")

SNAPSHOT = {
    "version": 1, "time": "12:00:00", "canvas": {"w": 640, "h": 480},
    "room": {"name": "room_1", "polygon": [[-3, -2], [3, -2], [3, 1], [0, 1], [0, 3], [-3, 3]]},
    "robot": {"x": -1.0, "y": 0.0, "yaw": 0.0, "heading": 1.5708, "radius": 0.35},
    "doors": [{"name": "door_1", "x": 1.5, "y": -2, "yaw": 0, "width": 0.9, "p_open": 0.9, "a": [1.05, -2], "b": [1.95, -2]},
              {"name": "door_2", "x": -3, "y": 2, "yaw": 1.57, "width": 0.8, "p_open": 0.1, "a": [-3, 1.6], "b": [-3, 2.4]}],
    "objects": [{"name": "table_1", "type": "table", "x": 1.5, "y": 0, "yaw": 0.3, "width": 1.2, "depth": 0.7,
                 "height": 0.75, "dist": 2.5, "bearing_deg": -90},
                {"name": "chair_1", "type": "chair", "x": 0.5, "y": 0.2, "yaw": 0, "width": 0.45, "depth": 0.5,
                 "height": 0.9, "dist": 1.51, "bearing_deg": -82},
                {"name": "bottle_1", "type": "bottle", "x": 1.4, "y": 0.1, "yaw": None, "width": None, "depth": None,
                 "height": 0.25, "dist": 2.4, "bearing_deg": -88}],
    "target": {"x": 1.5, "y": -1.5, "affordance": "aff_door_1_approach"},
    "mission": None,
}

BAD_NET = "import urllib.request\nurllib.request.urlopen('http://example.com', timeout=2)\n"


def run(rundir, script, *args):
    if os.path.exists(os.path.join(rundir, "out.png")):
        os.remove(os.path.join(rundir, "out.png"))
    p = subprocess.run([sys.executable, "-I", os.path.join(CANVAS, "bootstrap.py"), script, *args],
                       cwd=rundir, capture_output=True, text=True, timeout=30)
    return p.returncode, p.stderr, os.path.exists(os.path.join(rundir, "out.png"))


def main():
    keep = sys.argv[sys.argv.index("--keep") + 1] if "--keep" in sys.argv else None
    rundir = tempfile.mkdtemp()
    fails = 0
    try:
        with open(os.path.join(rundir, "snapshot.json"), "w") as f:
            json.dump(SNAPSHOT, f)
        from PIL import Image
        Image.new("RGB", (1280, 720), (90, 120, 160)).save(os.path.join(rundir, "zed.jpg"))
        cases = [("layout", [os.path.join(CANVAS, "stock", "layout.py")]),
                 ("camera_zed", [os.path.join(CANVAS, "stock", "camera.py"), "zed"]),
                 ("camera_ricoh_missing", [os.path.join(CANVAS, "stock", "camera.py"), "ricoh"])]
        for name, argv in cases:
            rc, err, png = run(rundir, *argv)
            ok = rc == 0 and png
            print(f"{'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"\n{err}"))
            fails += not ok
            if ok and keep:
                os.makedirs(keep, exist_ok=True)
                shutil.copy(os.path.join(rundir, "out.png"), os.path.join(keep, name + ".png"))
        with open(os.path.join(rundir, "net.py"), "w") as f:
            f.write(BAD_NET)
        rc, err, _ = run(rundir, os.path.join(rundir, "net.py"))
        ok = rc != 0 and "no network" in err
        print(f"{'PASS' if ok else 'FAIL'}  network blocked")
        fails += not ok
    finally:
        shutil.rmtree(rundir)
    print("all canvas stock checks passed" if fails == 0 else f"{fails} canvas check(s) FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
