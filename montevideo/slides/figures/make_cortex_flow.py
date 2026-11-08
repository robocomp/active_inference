"""Animation for day2.tex: one schematic pass of data through CORTEX, on the thesis' own drawing.

Every route is an arrow of the drawing, and every agent-to-agent path goes through the working
memory. The ORDER is illustrative: on the robot these run concurrently, at different rates.

Run from slides/:  python figures/make_cortex_flow.py [--stills]
Writes media/cortex_flow.mp4 and figures/cortex_flow_poster.png.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent
W, H, FPS = 1920, 1080, 30
PANEL_X = 1050
VIEW_W = 1400.0                                 # coordinates below are in a 1400 px wide view of the drawing
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
COOL, ACCENT, GOOD, INK, MUTED = (46, 109, 180), (192, 57, 43), (46, 139, 87), (35, 55, 59), (130, 130, 130)

art = Image.open(HERE / "cortex_overview.png").convert("RGB")
scale = H / art.height
art = art.resize((round(art.width * scale), H), Image.LANCZOS)
S = art.width / VIEW_W


def P(x, y):
    return x * S, y * S


# --- places on the drawing, (x0, y0, x1, y1), view coordinates ---------------------------------------
BOX = dict(
    camera=(358, 1382, 482, 1448), lidar=(502, 1382, 628, 1448), imu=(778, 1382, 902, 1448),
    video=(362, 1215, 478, 1297), fusion=(615, 1218, 731, 1297), imu2=(782, 1218, 898, 1297),
    perceptive=(487, 902, 663, 1032), self=(700, 902, 886, 1032), body=(968, 902, 1136, 1032),
    table=(948, 43, 1140, 150), room=(730, 63, 885, 143), cup=(1170, 265, 1362, 372),
    ep_agent=(494, 64, 649, 128), mission=(415, 526, 570, 605), sem_agent=(465, 722, 640, 852),
    ltsm_agent=(1166, 801, 1299, 931), base=(1171, 1220, 1345, 1307), motors=(1197, 1383, 1320, 1448),
)
CIRCLE = dict(wm=(843, 540, 218), episodic=(525, 328, 138), procedural=(205, 300, 190),
              semantic=(220, 722, 197), ltsm=(1222, 570, 152))
NODE = dict(robot=(810, 638), lidar=(810, 465), camera=(742, 492), table=(947, 550), afford=(947, 457),
            cup=(1000, 640))

# --- routes: each is a polyline along an arrow of the drawing -----------------------------------------
CENTRE = (843, 540)
R = dict(
    cam=[(420, 1382), (420, 1105), (575, 1105), (575, 1032)],
    lid=[(565, 1382), (565, 1335), (652, 1335), (672, 1218), (672, 1105), (790, 1105), (790, 1032)],
    imu=[(840, 1382), (840, 1032)],
    self_wm=[(795, 902), (820, 755), NODE["robot"]],
    perc_wm=[(612, 900), (735, 725), NODE["camera"]],
    wm_table=[CENTRE, (925, 340), (1010, 150)],
    table_wm=[(1010, 150), (925, 340), NODE["table"]],
    wm_room=[CENTRE, (828, 325), (812, 145)],
    room_wm=[(812, 145), (828, 325), CENTRE],
    cup_wm=[(1170, 345), (1025, 425), NODE["cup"]],
    wm_cup=[CENTRE, (1025, 425), (1170, 345)],
    wm_sem=[CENTRE, (680, 680), (632, 720)],
    sem_mem=[(465, 770), (415, 765)],
    proc_mis=[(330, 440), (420, 528)],
    mis_wm=[(570, 562), (628, 558), CENTRE],
    table_aff=[(1010, 150), (925, 340), NODE["afford"]],
    wm_body=[NODE["afford"], (935, 735), (1025, 900)],
    body_base=[(1092, 1032), (1092, 1125), (1258, 1125), (1258, 1220)],
    base_mot=[(1258, 1307), (1258, 1383)],
    wm_ep=[CENTRE, (735, 350), (610, 130)],
    ep_mem=[(562, 128), (548, 195)],
    wm_ltsm=[CENTRE, (1010, 680), (1165, 808)],
    ltsm_mem=[(1228, 801), (1228, 722)],
)

# --- the story: (title, one line, lit places, [(route, kind, t0, t1)], [node pulses]) -----------------
SENSE, BELIEF, ACT = "sense", "belief", "act"
STEPS = [
    ("Sensors fire", "Camera, LiDAR and IMU reach the agents that touch the body.",
     ["camera", "lidar", "imu", "video", "fusion", "imu2", "perceptive", "self"],
     [("cam", SENSE, 0.0, 0.75), ("lid", SENSE, 0.05, 0.85), ("imu", SENSE, 0.1, 0.7)], []),
    ("Two facts land on the graph", "Where the robot is, and what the camera segmented.",
     ["perceptive", "self"], [("self_wm", BELIEF, 0.0, 0.6), ("perc_wm", BELIEF, 0.3, 0.9)], [("robot", 0.55), ("camera", 0.85)]),
    ("Concept agents read it", "Nobody is told. Each takes what it needs from the graph.",
     ["table", "room"], [("wm_table", BELIEF, 0.0, 0.8), ("wm_room", BELIEF, 0.15, 0.9)], []),
    ("They revise their beliefs", "The table's numbers change. The structure does not.",
     ["table", "room"], [("table_wm", BELIEF, 0.0, 0.75), ("room_wm", BELIEF, 0.15, 0.85)], [("table", 0.72)]),
    ("Something new is born", "A cup node appears: a structural change. Semantic memory checks it makes sense.",
     ["cup", "sem_agent", "semantic"], [("cup_wm", BELIEF, 0.0, 0.42), ("wm_sem", BELIEF, 0.48, 0.8), ("sem_mem", BELIEF, 0.8, 0.98)], [("cup", 0.42)]),
    ("A mission is supplied", "Procedural memory holds what the robot can do, as chains of actions.",
     ["procedural", "mission"], [("proc_mis", ACT, 0.0, 0.4), ("mis_wm", ACT, 0.4, 0.9)], []),
    ("An object makes an offer", "The table publishes what looking at it is worth, in nats.",
     ["table"], [("table_aff", ACT, 0.0, 0.8)], [("afford", 0.78)]),
    ("The controller decides", "It compares the offers by expected free energy, and moves.",
     ["body", "base", "motors"], [("wm_body", ACT, 0.0, 0.5), ("body_base", ACT, 0.5, 0.86), ("base_mot", ACT, 0.86, 0.98)], []),
    ("Episodic memory keeps the record", "What was undertaken, and what came of it.",
     ["ep_agent", "episodic"], [("wm_ep", BELIEF, 0.0, 0.75), ("ep_mem", BELIEF, 0.75, 0.98)], []),
    ("The robot leaves the room", "The room moves to long-term memory, and comes back on return.",
     ["ltsm_agent", "ltsm"], [("wm_ltsm", BELIEF, 0.0, 0.7), ("ltsm_mem", BELIEF, 0.7, 0.98)], []),
]
STEP_S, END_S = 4.0, 4.5
TOTAL = STEP_S * len(STEPS) + END_S


def polyline(pts):
    """Arc-length parametrised polyline in output pixels."""
    p = np.array([P(*q) for q in pts])
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    return p, d / d[-1]


ROUTE = {k: polyline(v) for k, v in R.items()}


def at(route, s):
    p, u = ROUTE[route]
    return np.interp(s, u, p[:, 0]), np.interp(s, u, p[:, 1])


# --- sprites, supersampled so the marks are antialiased ----------------------------------------------
def sprite(kind, radius, alpha=255):
    k, n = 4, int(radius * 2 + 8)
    im = Image.new("RGBA", (n * k, n * k), (0, 0, 0, 0))
    d, c, r = ImageDraw.Draw(im), n * k / 2, radius * k
    col = dict(sense=ACCENT, belief=COOL, act=GOOD)[kind] + (alpha,)
    outline = (255, 255, 255, alpha)
    w = max(2, int(0.28 * radius * k / 2))
    if kind == SENSE:
        d.rectangle([c - r, c - r, c + r, c + r], fill=col, outline=outline, width=w)
    elif kind == ACT:
        d.polygon([(c, c - r * 1.3), (c + r * 1.3, c), (c, c + r * 1.3), (c - r * 1.3, c)], fill=col, outline=outline, width=w)
    else:
        d.ellipse([c - r, c - r, c + r, c + r], fill=col, outline=outline, width=w)
    return im.resize((n, n), Image.LANCZOS)


HEADS = {k: sprite(k, 10) for k in (SENSE, BELIEF, ACT)}
TAILS = {k: [sprite(k, 7 - i * 0.9, 190 - i * 34) for i in range(5)] for k in (SENSE, BELIEF, ACT)}


def ring(radius, alpha):
    k, n = 4, int(radius * 2 + 10)
    im = Image.new("RGBA", (n * k, n * k), (0, 0, 0, 0))
    c, r = n * k / 2, radius * k
    ImageDraw.Draw(im).ellipse([c - r, c - r, c + r, c + r], outline=ACCENT + (alpha,), width=4 * k)
    return im.resize((n, n), Image.LANCZOS)


def stamp(frame, spr, x, y):
    frame.paste(spr, (int(x - spr.width / 2), int(y - spr.height / 2)), spr)


# --- lighting: the drawing is dimmed and the places in play are lit -----------------------------------
base = np.asarray(art, dtype=np.float32)
dim = base * 0.34 + 255.0 * 0.66


def mask_for(names):
    m = Image.new("L", art.size, 0)
    d = ImageDraw.Draw(m)
    for n in list(names) + ["wm"]:
        if n in CIRCLE:
            x, y, r = CIRCLE[n]
            d.ellipse([*P(x - r, y - r), *P(x + r, y + r)], fill=255)
        else:
            x0, y0, x1, y1 = BOX[n]
            d.rounded_rectangle([*P(x0 - 6, y0 - 6), *P(x1 + 6, y1 + 6)], radius=12, fill=255)
    return np.asarray(m.filter(ImageFilter.GaussianBlur(3)), dtype=np.float32) / 255.0


MASKS = [mask_for(s[2]) for s in STEPS] + [mask_for([])]
Fs = lambda n, b=False: ImageFont.truetype(BOLD if b else FONT, n)


def wrap(draw, text, font, width):
    lines, cur = [], ""
    for w in text.split():
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=font) > width:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    return lines + [cur]


def panel(frame, step, t_in_step):
    d = ImageDraw.Draw(frame)
    x, right = PANEL_X + 25, W - 30
    d.text((x, 34), "CORTEX in action", font=Fs(46, True), fill=INK)
    y = 112
    for i, s in enumerate(STEPS):
        now, past = i == step, i < step
        if now:
            d.rounded_rectangle([x - 12, y - 8, right, y + 112], radius=12, fill=(232, 240, 249))
            d.text((x + 4, y), f"{i + 1}. {s[0]}", font=Fs(30, True), fill=INK)
            for j, line in enumerate(wrap(d, s[1], Fs(23), right - x - 26)[:2]):
                d.text((x + 4, y + 44 + j * 30), line, font=Fs(23), fill=INK)
            y += 128
        else:
            d.text((x + 4, y), f"{i + 1}. {s[0]}", font=Fs(23), fill=(90, 90, 90) if past else (165, 165, 165))
            y += 42
    if step >= len(STEPS):
        pass
    # legend, with shape as well as colour
    ly = H - 96
    for j, (k, lab) in enumerate(((SENSE, "sensor data"), (BELIEF, "beliefs"), (ACT, "offers, actions"))):
        lx = x + 12 + j * 240
        stamp(frame, HEADS[k], lx, ly + 13)
        d.text((lx + 22, ly), lab, font=Fs(22), fill=INK)
    d.text((x, H - 50), "Schematic. On the robot all of this runs at once, at different rates.", font=Fs(20), fill=MUTED)


def render(t):
    step = int(t // STEP_S)
    in_end = step >= len(STEPS)
    u = (t - STEP_S * step) / STEP_S if not in_end else 0.0
    fade = min(1.0, (t - STEP_S * min(step, len(STEPS))) / 0.3)
    prev = MASKS[max(0, min(step, len(STEPS)) - 1)] if step > 0 else MASKS[0]
    cur = MASKS[min(step, len(STEPS))]
    m = (prev * (1 - fade) + cur * fade)[..., None]
    out = np.full((H, W, 3), 255.0, dtype=np.float32)
    x0 = (PANEL_X - art.width) // 2
    out[:, x0:x0 + art.width] = dim * (1 - m) + base * m
    frame = Image.fromarray(out.astype(np.uint8))

    def px(x, y):
        return x + x0, y

    # the cup node exists from its birth onwards
    t_birth = STEP_S * (4 + 0.42)
    if t >= t_birth:
        d = ImageDraw.Draw(frame)
        a, b = px(*P(*NODE["table"])), px(*P(*NODE["cup"]))
        d.line([a, b], fill=(40, 40, 40), width=2)
        grow = min(1.0, (t - t_birth) / 0.3)
        k, n = 4, 40
        im = Image.new("RGBA", (n * k, n * k), (0, 0, 0, 0))
        c, r = n * k / 2, 10 * k * grow
        ImageDraw.Draw(im).ellipse([c - r, c - r, c + r, c + r], fill=ACCENT + (255,), outline=(20, 20, 20, 255), width=k)
        stamp(frame, im.resize((n, n), Image.LANCZOS), *b)
        d.text((b[0] + 16, b[1] - 11), "Cup", font=Fs(18, True), fill=(30, 30, 30))

    if not in_end:
        s = STEPS[step]
        for route, kind, t0, t1 in s[3]:
            for lag in (0.0, 0.09, 0.18):
                v = (u - t0) / (t1 - t0) - lag * (1.0 / (t1 - t0)) * 0.5
                if not 0.0 <= v <= 1.0:
                    continue
                for i, tail in enumerate(TAILS[kind]):
                    tv = max(0.0, v - (i + 1) * 0.022)
                    stamp(frame, tail, *px(*at(route, tv)))
                stamp(frame, HEADS[kind], *px(*at(route, v)))
        for node, t0 in s[4]:
            age = (u - t0) * STEP_S
            if 0.0 <= age <= 0.9:
                stamp(frame, ring(10 + 34 * age, int(255 * (1 - age / 0.9))), *px(*P(*NODE[node])))
    panel(frame, step, u)
    if in_end:
        d = ImageDraw.Draw(frame)
        x, right = PANEL_X + 25, W - 30
        lines = wrap(d, "Every path went through the centre. No agent ever spoke to another.", Fs(31, True), right - x)
        y = 610
        d.rounded_rectangle([x - 12, y - 16, right, y + 40 * len(lines) + 10], radius=12, fill=(250, 230, 227))
        for j, line in enumerate(lines):
            d.text((x + 4, y + j * 40), line, font=Fs(31, True), fill=(150, 40, 30))
    return frame


if "--stills" in sys.argv:
    out = Path(sys.argv[sys.argv.index("--stills") + 1])
    for i, t in enumerate((1.5, 5.0, 9.5, 18.6, 21.5, 26.5, 30.0, 34.0, 38.5, 42.5)):
        render(t).save(out / f"still_{i:02d}.png")
    sys.exit(0)

(HERE.parent / "media").mkdir(exist_ok=True)
render(STEP_S * 4 + 0.6 * STEP_S).save(HERE / "cortex_flow_poster.png")
ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                       "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", "-preset", "medium",
                       "-movflags", "+faststart", str(HERE.parent / "media" / "cortex_flow.mp4")], stdin=subprocess.PIPE)
for n in range(int(TOTAL * FPS)):
    ff.stdin.write(render(n / FPS).tobytes())
ff.stdin.close()
ff.wait()
print("cortex_flow.mp4 written,", TOTAL, "s")
