import csv, matplotlib
matplotlib.use("Agg")
# ⚠ TYPE 42 (TrueType), NOT matplotlib's DEFAULT TYPE 3. IEEE/PaperCept forbid Type 3 fonts outright
# and the automated checker rejects on them; every figure in this paper was Type 3 until 2026-09-14.
# It is one line here and a resubmission if it is missed.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPoly, Circle

SRC = "datasets/webots_piso/runs/room2_parked_then_rotated_1357.csv"
BAR = 0.06          # publish gate, metres — the same constant the agent gates on
TRUTH_W, TRUTH_H = 6.000, 4.000

rows = {}
with open(SRC, encoding="utf-8") as f:
    for line in f:
        if line.startswith("#"): continue
        p = line.rstrip("\n").split(";")
        if len(p) < 7: continue
        rows[int(p[0])] = p

def parse(p):
    verts = [tuple(map(float, v.split(","))) for v in p[3].split()]
    csig  = [float(x) for x in p[4].split()]
    esig  = [float(x) for x in p[5].split()]
    return verts, csig, esig

WANT = [(12999, "parked, 681 s"), (13478, "after one rotation in place")]
# ⚠ GENERATED AT THE SIZE IT IS PLACED AT. This figure sits at \columnwidth (~3.5 in); drawn at
# 7.16 in it was scaled to 49% and every label rendered at half its nominal size, which is why the
# text could not be read. Matplotlib points are only points if the figure is not resized afterwards.
# ── BOTH PANELS ARE DRAWN IN THE SAME ORIENTATION, AND THIS IS NOT COSMETIC ─────────────────────
# The map frame is gauge-free: its origin is where the robot started and its orientation is whatever
# theta0 currently is, and the re-anchor rotates it between these two frames — the room comes out with
# its long axis along x in one panel and along y in the other. Drawn as stored, the pair reads as a
# room that CHANGED SHAPE, which is the opposite of what it shows. Each panel is therefore rotated so
# its own longest edge is horizontal: nothing measured is altered, because every quantity here (edge
# lengths, corner sigma, the truth comparison) is invariant to that rotation. What the pair then shows
# is the only thing that actually differs between the two frames — the uncertainty.
import math as _mm
def canonical(V):
    best, bi = -1.0, 0
    for i in range(len(V)):
        a, b = V[i], V[(i + 1) % len(V)]
        d = _mm.hypot(b[0] - a[0], b[1] - a[1])
        if d > best: best, bi = d, i
    a, b = V[bi], V[(bi + 1) % len(V)]
    th = -_mm.atan2(b[1] - a[1], b[0] - a[0])
    cx = sum(p[0] for p in V) / len(V); cy = sum(p[1] for p in V) / len(V)
    c, s_ = _mm.cos(th), _mm.sin(th)
    return lambda p: (c * (p[0] - cx) - s_ * (p[1] - cy), s_ * (p[0] - cx) + c * (p[1] - cy))

fig, axes = plt.subplots(1, 2, figsize=(3.46, 2.35))
for ax, (fr, title) in zip(axes, WANT):
    V, C, E = parse(rows[fr])
    V = [canonical(V)(p) for p in V]
    ax.add_patch(MPoly(V, closed=True, facecolor="#f6f2ea", edgecolor="none", zorder=0))
    # per-edge band: half-width is that wall's own offset sigma, drawn to scale
    for i in range(len(V)):
        a, b = V[i], V[(i + 1) % len(V)]
        sd = E[i] if i < len(E) else 0.0
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#0e7490",
                lw=max(0.8, 2.0 * sd * 6.0), alpha=0.30, solid_capstyle="butt", zorder=1)
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#1f2328", lw=1.3, zorder=3)
    # per-corner sigma disc, radius = the sigma itself; green under the gate, orange over it
    for (x, y), s in zip(V, C):
        ok = s <= BAR
        ax.add_patch(Circle((x, y), s, facecolor="#177245" if ok else "#c2410c",
                            alpha=0.22 if ok else 0.16, edgecolor="none", zorder=2))
        ax.add_patch(Circle((x, y), 0.05, facecolor="#177245" if ok else "#c2410c",
                            edgecolor="none", zorder=4))
    # ── THE ERROR, AS A NUMBER, BECAUSE IT CANNOT BE A PICTURE ──────────────────────────────────
    # 5 mm on a 6 m wall is 0.08 percent: at any scale that fits the room it is far thinner than the
    # line used to draw it. Showing the estimate inside the truth band says "they agree"; it cannot
    # say BY HOW MUCH, and the panel is claiming a specific quantity. So each dimension is labelled
    # with what was measured and how far that is from truth, and the reader watches those numbers
    # barely move while sigma collapses beside them — which is the whole point of the pair.
    import math as _m
    edges = sorted(((_m.hypot(V[(i+1) % len(V)][0]-V[i][0], V[(i+1) % len(V)][1]-V[i][1]), i)
                    for i in range(len(V))), reverse=True)
    for length, i in (edges[0], edges[2] if len(edges) > 2 else edges[-1]):
        a, b = V[i], V[(i+1) % len(V)]
        truth_len = TRUTH_W if length > 5.0 else TRUTH_H
        err_mm = (length - truth_len) * 1000.0
        mx_, my_ = (a[0]+b[0])/2.0, (a[1]+b[1])/2.0
        horiz = abs(b[0]-a[0]) > abs(b[1]-a[1])
        ax.text(mx_, my_, f"{length:.3f} m  ({err_mm:+.0f} mm)", fontsize=4.6, color="#31465c",
                ha="center", va="center", rotation=0 if horiz else 90, zorder=8,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))

    worst = max(C)
    ax.set_title(f"{title}\nworst corner $\\sigma$ = {worst:.3f} m"
                 + ("  (publishable)" if worst <= BAR else "  (withheld)"),
                 fontsize=6.0, color="#1f2328", pad=3)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-4.3, 4.3); ax.set_ylim(-3.4, 3.4)

# ── TRUTH IS DRAWN IN EACH PANEL'S OWN FRAME, AND THAT IS NOT A FUDGE ────────────────────────────
# The map frame is GAUGE-FREE while the layout is being estimated: its origin is wherever the robot
# started and its orientation is whatever theta0 currently is, and the re-anchor rotates the whole map
# between these two frames — measured here, the long axis lies along x in the left panel and along y
# in the right. A single fixed truth rectangle therefore crosses the estimate at 90 degrees in one of
# the two panels, which is a defect of the DRAWING, not of the estimate.
# What the paper claims is frame-invariant — edge lengths and corner angles — so the truth rectangle
# is placed at each panel's own centroid and orientation, taken from the estimate's longest edge. The
# reader then compares shape and size, which is exactly the quantity being claimed, instead of an
# absolute placement that carries no meaning before the frame is anchored.
import math
def truth_in_panel_frame(V, w, h):
    best, bi = -1.0, 0
    for i in range(len(V)):
        a, b = V[i], V[(i + 1) % len(V)]
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        if d > best: best, bi = d, i
    a, b = V[bi], V[(bi + 1) % len(V)]
    th = math.atan2(b[1] - a[1], b[0] - a[0])
    cx = sum(p[0] for p in V) / len(V)
    cy = sum(p[1] for p in V) / len(V)
    c, s2 = math.cos(th), math.sin(th)
    return [(cx + c * dx - s2 * dy, cy + s2 * dx + c * dy)
            for dx, dy in ((-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2))]

for ax, (fr, _t) in zip(axes, WANT):
    V, _C, _E = parse(rows[fr])
    V = [canonical(V)(p) for p in V]
    T = truth_in_panel_frame(V, TRUTH_W, TRUTH_H)
    # A WIDE PALE BAND, UNDER the estimate, not a hairline on top of it. Truth and estimate differ by
    # 5 and 12 mm here — far less than the estimate's own stroke width at this scale — so a dashed
    # line drawn over it is simply invisible, which read as "the ground truth is missing". Drawn as a
    # band, the estimate is seen lying INSIDE the truth, and the fact that it fits within the band's
    # width is exactly the result the panel exists to show.
    ax.add_patch(MPoly(T, closed=True, facecolor="none", edgecolor="#8aa4c8", lw=4.0,
                       alpha=0.75, zorder=0.5, joinstyle="miter"))
    ax.add_patch(MPoly(T, closed=True, facecolor="none", edgecolor="#31465c", lw=0.7,
                       ls=(0, (3, 2.4)), zorder=6))
fig.subplots_adjust(left=0.005, right=0.995, top=0.83, bottom=0.115, wspace=0.02)
# ── THE LEGEND MUST FIT INSIDE THE FIGURE, AND ONE LINE DID NOT ───────────────────────
# At 5.2 pt this legend was ~3.6 in of text inside a 3.46 in figure, so it ran off BOTH sides and
# lost a character at each end ("ale band … horizonta"). matplotlib neither clips nor warns: text
# placed with fig.text is simply drawn outside the canvas and cropped by the PDF media box, so it
# looks correct in every check except the one that matters. Two lines fit with margin to spare, and
# the assertion before savefig now fails the build rather than shipping a clipped legend again.
_LEG = ["pale band: ground truth   ·   discs: per-corner $\\sigma$, drawn to scale",
        "both panels: the same room, the same scale, long axis horizontal"]
for _i, _s in enumerate(_LEG):
    fig.text(0.5, 0.058 - 0.046 * _i, _s, ha="center", fontsize=5.0, color="#31465c")

fig.canvas.draw()
_W, _H = fig.get_size_inches() * fig.dpi
for _t in fig.texts:
    _bb = _t.get_window_extent(fig.canvas.get_renderer())
    assert _bb.x0 >= 2 and _bb.x1 <= _W - 2, (
        f"figure text runs off the canvas: {_t.get_text()!r} spans "
        f"{_bb.x0:.0f}..{_bb.x1:.0f} of 0..{_W:.0f} px")
fig.savefig("paper/icra/fig/teaser.pdf")
print("wrote paper/icra/fig/teaser.pdf")
for fr, t in WANT:
    V, C, E = parse(rows[fr]); print(f"  {t}: worst sigma {max(C):.4f}  worst edge {max(E):.4f}")
