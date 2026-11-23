#!/usr/bin/env python3
# 3D viewer for residual_nvblox: occupied voxels + ESDF 2D on the floor, from /dev/shm/residual_nvblox_3d.bin.
#   needs Nvblox.Dump3D = true;  DISPLAY=:1 ~/software/pip_venv/bin/python tools/nvblox_rerun_view.py [hz]
import os
import sys
import time

import numpy as np
import rerun as rr

PATH = "/dev/shm/residual_nvblox_3d.bin"
HZ = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
RAMP = np.array([[40, 60, 220], [40, 210, 230], [250, 230, 40], [230, 40, 30]], np.float32)


def ramp(t):
    s = np.clip(t, 0, 1) * 3
    i = np.minimum(2, s.astype(int))
    u = (s - i)[:, None]
    return (RAMP[i] + u * (RAMP[i + 1] - RAMP[i])).astype(np.uint8)


def read():
    b = open(PATH, "rb").read()
    w, h, n = np.frombuffer(b, np.int32, 3)
    xmin, ymin, cell, bin_h, rx, ry, emax = np.frombuffer(b, np.float32, 7, 12)
    o = 40
    esdf = np.frombuffer(b, np.float32, w * h, o).reshape(h, w); o += 4 * w * h
    state = np.frombuffer(b, np.uint8, w * h, o).reshape(h, w); o += w * h
    vox = np.frombuffer(b, np.float32, 4 * n, o).reshape(n, 4)
    return w, h, xmin, ymin, cell, bin_h, rx, ry, emax, esdf, state, vox


def esdf_texture(esdf, emax, cell):
    tex = np.full(esdf.shape + (3,), 70, np.uint8)
    seen = ~np.isnan(esdf)
    d = np.where(seen, esdf, 0)
    g = (30 + 200 * np.minimum(1, d / emax)).astype(np.uint8)
    tex[seen] = np.stack([np.zeros_like(g), g // 2, g], -1)[seen]
    tex[seen & (np.fmod(d, 0.5) < cell * 0.7)] = 230
    tex[seen & (d <= 0)] = (230, 40, 30)
    return tex


rr.init("residual_nvblox_3d")
rr.spawn(port=9881, memory_limit="1GB", executable_path=os.path.join(os.path.dirname(sys.executable), "rerun"))
rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
last = 0
while True:
    time.sleep(1 / HZ)
    try:
        m = os.stat(PATH).st_mtime_ns
        if m == last:
            continue
        last = m
        w, h, xmin, ymin, cell, bin_h, rx, ry, emax, esdf, state, vox = read()
    except (OSError, ValueError):
        continue
    rr.set_time("t", timestamp=m * 1e-9)
    p = 1 / (1 + np.exp(-vox[:, 3]))
    col = ramp(vox[:, 2] / 1.8)
    rr.log("world/voxels", rr.Boxes3D(centers=vox[:, :3], half_sizes=[cell / 2, cell / 2, bin_h / 2],
                                      colors=np.c_[col, (255 * np.clip(2 * p - 0.6, 0.3, 1)).astype(np.uint8)],
                                      fill_mode="solid"))
    x1, y1 = xmin + w * cell, ymin + h * cell
    rr.log("world/esdf", rr.Mesh3D(vertex_positions=[[xmin, ymin, 0], [x1, ymin, 0], [x1, y1, 0], [xmin, y1, 0]],
                                   triangle_indices=[[0, 1, 2], [0, 2, 3]],
                                   vertex_texcoords=[[0, 0], [1, 0], [1, 1], [0, 1]],
                                   albedo_texture=esdf_texture(esdf, emax, cell)))
    iy, ix = np.nonzero(state == 2)
    rr.log("world/residual", rr.Boxes3D(centers=np.c_[xmin + (ix + 0.5) * cell, ymin + (iy + 0.5) * cell, np.full(ix.size, 0.01)],
                                        half_sizes=[cell / 2, cell / 2, 0.005], colors=[255, 0, 255], fill_mode="solid"))
    rr.log("world/robot", rr.Points3D([[rx, ry, 0.05]], radii=0.12, colors=[255, 255, 255]))
