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
    g = graph()                     the robot's working memory (its DSR graph), when the script mentions graph():
                                    g.nodes [n.id, n.name, n.type, n.agent], g.edges [e.src, e.dst, e.type] (node ids)
    img = draw_graph(focus=None)    that graph laid out by Graphviz as a PIL image (RT edges = the tree, others
                                    dashed); focus = a node name or id: it, its RT ancestors and subtree, its neighbours
"""
import json
import math
import os
from types import SimpleNamespace

OUT = "out.png"

__all__ = ["world", "image", "figure", "save", "save_image", "layout_axes", "draw_room", "draw_doors", "draw_objects",
           "draw_robot", "draw_target", "draw_layout", "fit_view", "graph", "draw_graph", "plt", "np", "math"]


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
    ax.set_aspect("equal", adjustable="box")
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


# ─── the working-memory graph ────────────────────────────────────────────────────────────────────────────────────

def graph():
    try:
        with open("graph.json", encoding="utf-8") as f:
            d = json.load(f)
    except FileNotFoundError:
        d = {"nodes": [], "edges": []}
    nodes = [SimpleNamespace(id=n["id"], name=n["name"], type=n["type"], agent=n.get("agent")) for n in d["nodes"]]
    edges = [SimpleNamespace(src=e["from"], dst=e["to"], type=e["type"]) for e in d["edges"]]
    return SimpleNamespace(nodes=nodes, edges=edges)


_TYPE_COLOUR = {"world": "#cfd8dc", "root": "#cfd8dc", "room": "#c8e6c9", "robot": "#bbdefb", "object": "#ffe0b2",
                "person": "#f8bbd0", "affordance": "#e1bee7", "metaconcept": "#d1c4e9", "agent": "#fff9c4",
                "obstacle": "#ffccbc", "rgbd": "#b2ebf2", "lidar": "#b2ebf2", "imu": "#b2ebf2", "camera": "#b2ebf2"}


def _subgraph(g, focus):
    by_id = {n.id: n for n in g.nodes}
    root = next((n for n in g.nodes if n.name == focus or n.id == str(focus)), None)
    if root is None:
        raise ValueError(f"no node named '{focus}' in the graph")
    rt_children, rt_parent = {}, {}
    for e in g.edges:
        if e.type == "RT":
            rt_children.setdefault(e.src, []).append(e.dst)
            rt_parent[e.dst] = e.src
    keep = {root.id}
    up = root.id
    while up in rt_parent:                       # the RT chain to the top
        up = rt_parent[up]
        keep.add(up)
    stack = [root.id]
    while stack:                                 # the RT subtree
        for c in rt_children.get(stack.pop(), []):
            if c not in keep:
                keep.add(c)
                stack.append(c)
    for e in g.edges:                            # direct neighbours of the focus, any edge
        if root.id in (e.src, e.dst):
            keep.update((e.src, e.dst))
    return SimpleNamespace(nodes=[by_id[i] for i in keep if i in by_id],
                           edges=[e for e in g.edges if e.src in keep and e.dst in keep])


def _dot_esc(s):
    return str(s).replace("\\", "\\\\").replace('"', '\\"')


def _dot_quote(s):
    return '"' + _dot_esc(s) + '"'


def draw_graph(focus=None, g=None):
    """The graph as a PIL image, fitted to the canvas: Graphviz `dot`, RT edges as the tree (solid), every other edge
    dashed and labelled. Falls back to a networkx spring layout when `dot` is not installed."""
    import shutil
    import subprocess
    from io import BytesIO
    from PIL import Image
    g = g or graph()
    if focus:
        g = _subgraph(g, focus)
    w, h = _canvas_px()
    if not g.nodes:
        img = Image.new("RGB", (w, h), (255, 255, 255))
        from PIL import ImageDraw
        ImageDraw.Draw(img).text((20, h // 2), "the graph is empty (or not available)", fill=(90, 90, 90))
        return img
    if shutil.which("dot") is None:
        return _draw_graph_networkx(g, w, h)
    lines = ["digraph G {", "rankdir=TB; nodesep=0.15; ranksep=0.35; bgcolor=white;",
             'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.06,0.03"];',
             'edge [fontname="Helvetica", fontsize=8, arrowsize=0.6];']
    for n in g.nodes:
        colour = _TYPE_COLOUR.get(n.type, "#eeeeee")
        label = '"' + _dot_esc(n.name) + "\\n" + _dot_esc(n.type) + '"'   # \\n = a line break inside a DOT label
        lines.append(f"{_dot_quote(n.id)} [label={label}, fillcolor={_dot_quote(colour)}"
                     + (", penwidth=2.5" if focus and (n.name == focus or n.id == str(focus)) else "") + "];")
    for e in g.edges:
        if e.type == "RT":
            lines.append(f"{_dot_quote(e.src)} -> {_dot_quote(e.dst)} [color=\"#455a64\"];")
        else:
            lines.append(f"{_dot_quote(e.src)} -> {_dot_quote(e.dst)} [style=dashed, color=\"#9e9e9e\", "
                         f"fontcolor=\"#757575\", label={_dot_quote(e.type)}, constraint=false];")
    lines.append("}")
    p = subprocess.run(["dot", "-Tpng", "-Gdpi=96", f"-Gsize={w / 96:.2f},{h / 96:.2f}"],
                       input="\n".join(lines).encode(), capture_output=True, timeout=20)
    if p.returncode != 0:
        raise RuntimeError("graphviz failed: " + p.stderr.decode(errors="replace")[-500:])
    return Image.open(BytesIO(p.stdout)).convert("RGB")


def _draw_graph_networkx(g, w, h):
    import networkx as nx
    from io import BytesIO
    from PIL import Image
    G = nx.DiGraph()
    for n in g.nodes:
        G.add_node(n.id)
    for e in g.edges:
        G.add_edge(e.src, e.dst)
    pos = nx.spring_layout(G, seed=1)
    fig = figure()
    ax = fig.add_subplot(111)
    ax.axis("off")
    names = {n.id: n.name for n in g.nodes}
    nx.draw_networkx(G, pos, ax=ax, labels=names, node_size=300, font_size=7,
                     node_color=[_TYPE_COLOUR.get(n.type, "#eeeeee") for n in g.nodes])
    buf = BytesIO()
    fig.savefig(buf, format="png")
    return Image.open(buf).convert("RGB")
