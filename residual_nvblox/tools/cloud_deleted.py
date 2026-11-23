#!/usr/bin/env python3
# LiDAR clouds before (pre) and as integrated by nvblox (post), from the bridge's /dev/shm/nvblox_cloud_{pre,post}_<id>.bin.
# Each raw point is: kept (grey), pushed under the floor along its own ray by FloorPush (blue), or DELETED (red:
# self-mask or FloorPush's ambiguous band). Virtual FloorFill returns are post-only and not drawn.
#   needs Nvblox.DumpClouds = true;  ~/software/pip_venv/bin/python tools/cloud_deleted.py [--dir /dev/shm] [--radius 3] [--out cloud_deleted.png]
import argparse

import matplotlib
import numpy as np

NAMES = {0: "helios", 1: "bpearl"}


def load(path):
    b = open(path, "rb").read()
    T = np.frombuffer(b, np.float32, 12).reshape(3, 4)
    n = np.frombuffer(b, np.int32, 1, 48)[0]
    return T, np.frombuffer(b, np.float32, 3 * n, 52).reshape(-1, 3)


def classify(pre, post):
    exact = {p.tobytes() for p in post}
    ray = lambda P: np.round(P / np.linalg.norm(P, axis=1, keepdims=True), 4)
    post_rays = {r.tobytes() for r in ray(post)}
    kept = np.array([p.tobytes() in exact for p in pre], bool)
    pushed = ~kept & np.array([r.tobytes() in post_rays for r in ray(pre)], bool)
    return kept, pushed, ~kept & ~pushed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/dev/shm")
    ap.add_argument("--radius", type=float, default=3.0, help="half-size of the view around the sensor (m)")
    ap.add_argument("--out", default="cloud_deleted.png")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    if not a.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(16, 13), gridspec_kw={"height_ratios": [3, 1.2]})
    for col, (sid, name) in enumerate(NAMES.items()):
        T, pre = load(f"{a.dir}/nvblox_cloud_pre_{sid}.bin")
        _, post = load(f"{a.dir}/nvblox_cloud_post_{sid}.bin")
        kept, pushed, deleted = classify(pre, post)
        Q = pre @ T[:, :3].T + T[:, 3]
        o = T[:, 3]
        d = Q[:, :2] - o[:2]
        r = np.hypot(d[:, 0], d[:, 1])
        near = r < a.radius
        top, side = ax[0, col], ax[1, col]
        for m, c, s, lab in ((kept, "0.6", 1, "conservados"), (pushed, "tab:blue", 1, "empujados bajo el suelo"),
                             (deleted, "red", 4, "borrados")):
            m = m & near
            top.scatter(d[m, 0], d[m, 1], s=s, c=c, lw=0, label=f"{lab} ({m.sum()})")
            side.scatter(r[m], Q[m, 2], s=s, c=c, lw=0)
        top.plot(0, 0, "k+", ms=12)
        top.set_title(f"{name}: {len(pre)} crudos → {len(post)} integrados (incl. relleno de suelo)")
        top.set_aspect("equal")
        top.set_xlim(-a.radius, a.radius)
        top.set_ylim(-a.radius, a.radius)
        top.set_xlabel("x habitación − sensor (m)")
        top.set_ylabel("y (m)")
        top.legend(loc="upper right", markerscale=4)
        top.grid(alpha=0.3)
        side.set_xlim(0, a.radius)
        side.set_ylim(-0.3, 1.5)
        side.set_xlabel("distancia horizontal al sensor (m)")
        side.set_ylabel("z habitación (m)")
        side.axhline(0, c="k", lw=0.5)
        side.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print("saved", a.out)
    if a.show:
        plt.show()


if __name__ == "__main__":
    main()
