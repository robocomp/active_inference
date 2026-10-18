"""rcanvas — what a canvas script may read and how it draws (chat_agent's side canvas).

A canvas script never touches the robot's graph. chat_agent writes a SNAPSHOT into the run dir (the cwd) before
each run, and the script draws from it:

    snapshot.json   the room, the robot, the doors, the objects, the target, the mission (canvas_snapshot.cpp)
    zed.jpg         front camera (ZED), when the script mentions image()
    ricoh.jpg       360-degree camera (Ricoh), likewise

Frames and angles (all ROOM frame, metres, radians):
    robot.x, robot.y        position of the robot's base (`body`)
    robot.heading           direction the robot FACES, counter-clockwise from the room's +x axis
    door.a, door.b          the two ends of the door opening; door.p_open = P(open) or None
    obj.x, obj.y, obj.yaw   centre and orientation; obj.width runs along the object's yaw, obj.depth across it

The API:
    w = world()                     the snapshot (attribute access; a missing part is None, a missing list is [])
    img = image("zed" | "ricoh")    numpy uint8 RGB array, or None when that camera has no frame
    fig = figure()                  a matplotlib figure sized to the canvas
    ax = layout_axes(fig)           an axis with equal aspect, metres, y up
    draw_room(ax, w); draw_doors(ax, w); draw_objects(ax, w); draw_robot(ax, w); draw_target(ax, w)
    draw_layout(ax, w)              all five, then the view fitted to the room
    save(fig)                       REQUIRED last call: writes out.png, which the canvas shows
    plt, np, math                   also exported by `from rcanvas import *` (matplotlib on Agg)
    save_image(img)                 the same for a numpy/PIL image (no matplotlib, faster)
"""
import json
import math
import os
from types import SimpleNamespace

OUT = "out.png"

__all__ = ["world", "image", "figure", "save", "save_image", "layout_axes", "draw_room", "draw_doors", "draw_objects",
           "draw_robot", "draw_target", "draw_layout", "fit_view", "plt", "np", "math"]


def __getattr__(name):
    """`plt` and `np` come with `from rcanvas import *` (a model-written script often uses them unimported), but are
    imported only when asked for, so a PIL-only script (stock/camera.py) does not pay for matplotlib."""
    if name == "plt":
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    if name == "np":
        import numpy as np
        return np
    raise AttributeError(name)


def _ns(v):
    if isinstance(v, dict):
        return SimpleNamespace(**{k: _ns(x) for k, x in v.items()})
    if isinstance(v, list):
        return [_ns(x) for x in v]
    return v


_EMPTY = {"room": None, "robot": None, "doors": [], "objects": [], "target": None, "mission": None,
          "motion": None, "canvas": {"w": 800, "h": 600}, "time": ""}


# Every documented field exists (None when unknown), so a script can test `is None` instead of catching errors.
_OBJECT = dict.fromkeys(["name", "type", "x", "y", "yaw", "width", "depth", "height", "dist", "bearing_deg"])
_DOOR = dict.fromkeys(["name", "x", "y", "yaw", "width", "a", "b", "p_open"])


def world():
    try:
        with open("snapshot.json", encoding="utf-8") as f:
            d = json.load(f)
    except FileNotFoundError:
        d = {}
    d = {**_EMPTY, **d}
    d["objects"] = [{**_OBJECT, **o} for o in d.get("objects") or []]
    d["doors"] = [{**_DOOR, **o} for o in d.get("doors") or []]
    return _ns(d)


def image(name="zed"):
    path = {"zed": "zed.jpg", "front": "zed.jpg", "ricoh": "ricoh.jpg", "360": "ricoh.jpg"}.get(name, name)
    if not os.path.exists(path):
        return None
    import numpy as np
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"))


def _canvas_px():
    c = world().canvas
    return max(200, int(c.w)), max(150, int(c.h))


def figure(dpi=100):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w, h = _canvas_px()
    return plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)


def save(fig=None):
    import matplotlib.pyplot as plt
    (fig or plt.gcf()).savefig(OUT)


def save_image(img):
    from PIL import Image
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    img.save(OUT)


def layout_axes(fig):
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    return ax


def draw_room(ax, w):
    if w.room is None or not w.room.polygon:
        return
    xs = [p[0] for p in w.room.polygon] + [w.room.polygon[0][0]]
    ys = [p[1] for p in w.room.polygon] + [w.room.polygon[0][1]]
    ax.fill(xs, ys, color="#eceff1", zorder=0)
    ax.plot(xs, ys, color="#37474f", linewidth=2.5, zorder=1)


def draw_doors(ax, w):
    for d in w.doors:
        if d.a is None or d.b is None:
            continue
        p = d.p_open
        color = "#9e9e9e" if p is None else ("#2e7d32" if p >= 0.5 else "#c62828")
        ax.plot([d.a[0], d.b[0]], [d.a[1], d.b[1]], color=color, linewidth=6, solid_capstyle="butt", zorder=3)
        label = d.name if p is None else f"{d.name}  P(open) {p:.2f}"
        ax.annotate(label, ((d.a[0] + d.b[0]) / 2, (d.a[1] + d.b[1]) / 2), fontsize=8, color=color,
                    xytext=(4, 4), textcoords="offset points", zorder=6)


def _box(cx, cy, yaw, w, d):
    c, s = math.cos(yaw), math.sin(yaw)
    pts = [(-w / 2, -d / 2), (w / 2, -d / 2), (w / 2, d / 2), (-w / 2, d / 2), (-w / 2, -d / 2)]
    return [cx + c * x - s * y for x, y in pts], [cy + s * x + c * y for x, y in pts]


def draw_objects(ax, w):
    for o in w.objects:
        if o.x is None or o.y is None:
            continue
        ow, od = o.width or 0.3, o.depth or 0.3
        xs, ys = _box(o.x, o.y, o.yaw or 0.0, ow, od)
        ax.fill(xs, ys, color="#ffcc80", alpha=0.8, zorder=2)
        ax.plot(xs, ys, color="#e65100", linewidth=1, zorder=2)
        ax.annotate(o.name, (o.x, o.y), fontsize=8, ha="center", va="center", zorder=6)


def draw_robot(ax, w):
    r = w.robot
    if r is None:
        return
    import matplotlib.patches as mp
    ax.add_patch(mp.Circle((r.x, r.y), r.radius, color="#1565c0", alpha=0.25, zorder=4))
    L = r.radius * 1.4
    h = r.heading
    tip = (r.x + L * math.cos(h), r.y + L * math.sin(h))
    left = (r.x + 0.5 * r.radius * math.cos(h + 2.4), r.y + 0.5 * r.radius * math.sin(h + 2.4))
    right = (r.x + 0.5 * r.radius * math.cos(h - 2.4), r.y + 0.5 * r.radius * math.sin(h - 2.4))
    ax.add_patch(mp.Polygon([tip, left, right], color="#1565c0", zorder=5))


def draw_target(ax, w):
    t = w.target
    if t is None:
        return
    ax.plot([t.x], [t.y], marker="x", markersize=12, markeredgewidth=3, color="#6a1b9a", zorder=6)
    if w.robot is not None:
        ax.plot([w.robot.x, t.x], [w.robot.y, t.y], color="#6a1b9a", linestyle="--", linewidth=1, zorder=4)


def fit_view(ax, w, margin=0.5):
    xs, ys = [], []
    if w.room is not None and w.room.polygon:
        xs += [p[0] for p in w.room.polygon]
        ys += [p[1] for p in w.room.polygon]
    if w.robot is not None:
        xs.append(w.robot.x)
        ys.append(w.robot.y)
    if not xs:
        return
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)


def draw_layout(ax, w):
    draw_room(ax, w)
    draw_doors(ax, w)
    draw_objects(ax, w)
    draw_target(ax, w)
    draw_robot(ax, w)
    fit_view(ax, w)
