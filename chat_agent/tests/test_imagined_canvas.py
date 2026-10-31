#!/usr/bin/env python3
"""Smoke test of rcanvas.imagined() and the two imagined draw functions, headless (Agg). Writes a tiny imagined.json
and imagined_zed.png in a temp run dir (the shape imagined_io.h writes) and draws both views."""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "canvas"))

IMAGINED = {
    "version": 1, "caption": "a chair", "seed": "7", "drawn": 3, "rejected": 0, "outside_polygon": 0, "notes": [],
    "mc": {"p_notice_mean": 0.8, "p_notice_sd": 0.1, "p_recognise_mean": 0.5, "p_recognise_sd": 0.1, "modal_limit": 0, "n": 2},
    "image": {"w": 8, "h": 6},
    "samples": [{"x": 1, "y": 0, "yaw": 0, "weight": 1, "state": {}, "parts": [{"cx": 1, "cy": 0, "w": .5, "h": .5, "yaw": 0, "z0": 0, "z1": .9}]},
                {"x": 1.2, "y": 0, "yaw": 0.1, "weight": 1, "state": {}, "parts": []}],
    "lidar": [[{"on_target": [[1, 0, .5], [1.1, 0.1, .6]], "on_other": [[3, 0, 1]], "ring_elev_room_deg": [-10, 5],
                "lambda0": 2, "modelled": True, "notes": []},
               {"on_target": [], "on_other": [], "ring_elev_room_deg": [], "lambda0": 0, "modelled": False, "notes": []}], []],
}


def main():
    import numpy as np
    from PIL import Image
    d = tempfile.mkdtemp()
    os.chdir(d)
    import rcanvas
    fails = 0

    def check(name, ok):
        nonlocal fails
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        fails += not ok

    check("imagined() without files is None", rcanvas.imagined() is None)
    with open("imagined.json", "w") as f:
        json.dump(IMAGINED, f)
    img = np.zeros((6, 8, 3), np.uint8)
    img[1:4, 2:5, 0] = 200
    img[3:5, 4:7, 1] = 120
    Image.fromarray(img).save("imagined_zed.png")
    im = rcanvas.imagined()
    check("imagined() parses", im is not None and im.caption == "a chair" and len(im.samples) == 2 and im.mc.n == 2)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    try:
        rcanvas.draw_imagined_zed(ax)
        fig.savefig("zed.png")
        check("draw_imagined_zed runs headless", os.path.getsize("zed.png") > 0)
    except Exception as e:
        check(f"draw_imagined_zed ({e!r})", False)
    fig, (a, b) = plt.subplots(1, 2)
    try:
        rcanvas.draw_imagined_lidar(a, b)
        fig.savefig("lidar.png")
        check("draw_imagined_lidar runs headless", os.path.getsize("lidar.png") > 0)
    except Exception as e:
        check(f"draw_imagined_lidar ({e!r})", False)
    print("all imagined canvas checks passed" if fails == 0 else f"{fails} imagined canvas check(s) FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
