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
                "lambda0": 2, "modelled": True, "notes": [], "mount_xyz": [0, 0, 1.0], "bearing_rad": 0.0,
                "n_total_target": 2, "n_total_other": 1},
               {"on_target": [], "on_other": [], "ring_elev_room_deg": [], "lambda0": 0, "modelled": False, "notes": [],
                "mount_xyz": [0, 0, 1.0], "bearing_rad": 0.0, "n_total_target": 0, "n_total_other": 0}],
              [{"on_target": [], "on_other": [], "ring_elev_room_deg": [0, 20], "lambda0": 0, "modelled": True, "notes": [],
                "mount_xyz": [0, 0, 0.6], "bearing_rad": 0.0, "n_total_target": 0, "n_total_other": 0}]],
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
    fig, (a, b, c) = plt.subplots(1, 3)
    try:
        rcanvas.draw_imagined_lidar(a, [b, c])
        fig.savefig("lidar.png")
        check("draw_imagined_lidar runs headless", os.path.getsize("lidar.png") > 0)
    except Exception as e:
        check(f"draw_imagined_lidar ({e!r})", False)
    # one side panel per LiDAR, each with its own ring lines from its own mount, reaching that panel's x-limit
    check("lidar 0 panel: 2 ring lines from z 1.0", sorted(round(l.get_ydata()[0], 2) for l in b.get_lines()) == [1.0, 1.0])
    check("lidar 1 panel: 2 ring lines from z 0.6", sorted(round(l.get_ydata()[0], 2) for l in c.get_lines()) == [0.6, 0.6])
    check("ring lines reach the final x-limit", all(abs(l.get_xdata()[-1] - ax.get_xlim()[1]) < 1e-6
                                                    for ax in (b, c) for l in ax.get_lines()))
    # an unmodelled LiDAR: annotated empty panel
    with open("imagined.json", "w") as f:
        d = json.loads(json.dumps(IMAGINED))
        d["lidar"][0][0]["modelled"] = False
        d["lidar"][0][0]["on_target"] = []
        json.dump(d, f)
    fig, (a2, b2, c2) = plt.subplots(1, 3)
    rcanvas.draw_imagined_lidar(a2, [b2, c2])
    check("unmodelled LiDAR panel is annotated and empty", len(b2.get_lines()) == 0 and any("not modelled" in t.get_text() for t in b2.texts))
    with open("imagined.json", "w") as f:
        json.dump(IMAGINED, f)
    # a stale PNG with no image in the JSON is not drawn
    with open("imagined.json", "w") as f:
        json.dump(dict(IMAGINED, image={"w": 0, "h": 0}), f)
    fig, ax = plt.subplots()
    rcanvas.draw_imagined_zed(ax)
    check("stale PNG is not drawn", len(ax.images) == 0)
    # a long caption is wrapped, so the two panels' titles do not run into each other
    with open("imagined.json", "w") as f:
        json.dump(dict(IMAGINED, caption="8 imagined chairs wherever such things usually stand: P(notice) 0.00 ± 0.00"), f)
    fig, (za, ta) = plt.subplots(1, 2)
    rcanvas.draw_imagined_zed(za)
    check("long ZED title is wrapped", "\n" in za.get_title() and all(len(x) <= 44 for x in za.get_title().split("\n")))
    with open("imagined.json", "w") as f:
        json.dump(IMAGINED, f)
    with open("imagined.json", "w") as f:
        f.write('{"version": 1, "cap')
    check("truncated imagined.json is None", rcanvas.imagined() is None)
    print("all imagined canvas checks passed" if fails == 0 else f"{fails} imagined canvas check(s) FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
