#!/usr/bin/env python3
"""door_vlm.py — shared pieces of the VLM door-state calibration (offline_eval.py, live_calib.py).

    * the question: a door crop -> {"state": "open"|"closed"|"ajar", "visible": true|false}, asked of the local
      ollama model qwen3-vl:8b-instruct (the NON-thinking tag; plain `qwen3-vl:8b` thinks and burns hundreds of
      tokens before answering) at temperature 0 with a tiny num_predict;
    * the prompt VARIANTS compared offline (plain one-word / JSON schema / JSON + drawn box / JSON + "ignore
      overlays");
    * the geometry that turns a door pose + the robot's ground-truth pose into a crop of the ZED image
      (pinhole, fx=fy=448, 1280x720, mount 7.5 cm behind the base centre at z=0.9775 m — robot_concept/shadow.json
      + ROBOT_GEOMETRY.md) or of the Ricoh equirectangular panorama (gnomonic re-projection);
    * the operational truth: clear_width(phi) = w * (1 - max(0, cos phi)) and passable = clear_width >= 0.70 m.

Conventions (verified on 93k ZedDump frames + live): the robot drives along world angle h = -robot_gt_angle (the
published, sign-inverted ground truth); in Webots terms h = webots_yaw + pi/2 (Shadow's forward is its local +y). A Webots
`Door` PROTO's translation is the doorway centre; its leaf/width runs along (-sin rot, cos rot) and the wall normal
is (cos rot, sin rot).
"""
import base64
import json
import math
import re
import time
import urllib.request

import cv2
import numpy as np

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3-vl:8b-instruct"
NUM_CTX = 4096          # match what chat_agent loads: a different num_ctx makes ollama RELOAD the model
NUM_PREDICT = 40
MAX_SIDE = 512          # crops are small; ollama would rescale anyway (~1 Mpx cap)
JPEG_Q = 90

# ── operational truth ────────────────────────────────────────────────────────────────────────────────
DOOR_W = 1.0            # Webots Door PROTO leaf width (size 0.1 x 1 x 2.1)
BODY_W = 0.70           # door_concept's passable threshold (clear span vs 0.70 m of body)


def clear_width(phi_rad, w=DOOR_W):
    return w * (1.0 - max(0.0, math.cos(phi_rad)))


def passable(phi_rad, w=DOOR_W):
    return clear_width(phi_rad, w) >= BODY_W


def phi_bin(phi_deg):
    if phi_deg < 7.5:
        return "closed(0)"
    if phi_deg < 45:
        return "ajar(15-45)"
    if phi_deg < 75:
        return "half(45-75)"
    return "open(75-90)"


# ── prompts ──────────────────────────────────────────────────────────────────────────────────────────
DEFS = ("Definitions: 'closed' = the door leaf fills the doorway, nothing behind it is visible through the doorway. "
        "'ajar' = the leaf is partly open, leaving only a narrow gap (narrower than a person). "
        "'open' = the leaf is swung well back and the doorway is clear enough for a person to walk through.")
OVERLAY = (" Coloured lines, dots, translucent purple/blue fills, grids and text labels are DEBUG OVERLAYS drawn on top "
           "of the camera image: ignore them and judge only the real scene.")
SCHEMA = {"type": "object",
          "properties": {"state": {"type": "string", "enum": ["open", "closed", "ajar"]},
                         "visible": {"type": "boolean"}},
          "required": ["state", "visible"]}

VARIANTS = {
    # one word, free text, no schema
    "plain": dict(json=False, box=False, overlay=False,
                  system="You are the vision of a mobile robot. Answer with exactly one lower-case word.",
                  user="Look at the door in the centre of this picture. Is it open, closed or ajar? "
                       "If there is no door in the centre of the picture, answer none."),
    # JSON schema through ollama's `format`
    "json": dict(json=True, box=False, overlay=False,
                 system="You are the vision of a mobile robot. You report the state of a door. " + DEFS,
                 user='The picture is a crop centred on where a door should be. Report the state of the door in the '
                      'centre as JSON {"state": "open"|"closed"|"ajar", "visible": true|false}. visible=false if no '
                      'door (leaf or door frame) is visible in the centre of the picture, e.g. a wall hides it.'),
    # same + a red box drawn around the projected doorway
    "json_box": dict(json=True, box=True, overlay=False,
                     system="You are the vision of a mobile robot. You report the state of a door. " + DEFS,
                     user='A red rectangle marks where a doorway should be. Report the state of the door in the red '
                          'rectangle as JSON {"state": "open"|"closed"|"ajar", "visible": true|false}. visible=false '
                          'if no door (leaf or door frame) is visible inside the red rectangle, e.g. a wall hides it. '
                          'Ignore any other door.'),
    # json + the "ignore overlays" sentence
    "json_ovl": dict(json=True, box=False, overlay=True,
                     system="You are the vision of a mobile robot. You report the state of a door. " + DEFS + OVERLAY,
                     user='The picture is a crop centred on where a door should be. Report the state of the door in '
                          'the centre as JSON {"state": "open"|"closed"|"ajar", "visible": true|false}. visible=false '
                          'if no door (leaf or door frame) is visible in the centre of the picture, e.g. a wall hides '
                          'it.'),
    # the operational question asked directly (yes -> "open", no -> "closed"): can the robot pass?
    "passable": dict(json=False, box=False, overlay=False,
                     system="You are the vision of a mobile robot. Answer with exactly one lower-case word.",
                     user="Look at the doorway in the centre of this picture. Is the door open wide enough for a "
                          "person to walk through it? Answer yes or no. If there is no doorway in the centre of the "
                          "picture, answer none."),
}


def encode(bgr, max_side=MAX_SIDE, q=JPEG_Q):
    h, w = bgr.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        bgr = cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
    assert ok
    return base64.b64encode(buf.tobytes()).decode()


def parse(variant, text):
    """-> (state in {open, closed, ajar, none, ?}, visible bool|None)"""
    t = text.strip().lower()
    if VARIANTS[variant]["json"]:
        try:
            j = json.loads(t)
            vis = bool(j.get("visible", True))
            st = str(j.get("state", "?"))
            return (st if vis else "none"), vis
        except (json.JSONDecodeError, AttributeError):
            pass
    m = re.search(r"\b(open|closed|ajar|none|shut|yes|no)\b", t)
    if not m:
        return "?", None
    st = {"shut": "closed", "yes": "open", "no": "closed"}.get(m.group(1), m.group(1))
    return st, st != "none"


def ask(bgr, variant, model=MODEL, num_predict=NUM_PREDICT, timeout=120):
    """One query. Returns dict(state, visible, raw, latency_s, prompt_tokens, eval_tokens)."""
    v = VARIANTS[variant]
    body = {"model": model, "stream": False, "think": False,
            "options": {"temperature": 0, "seed": 0, "num_ctx": NUM_CTX, "num_predict": num_predict},
            "messages": [{"role": "system", "content": v["system"]},
                         {"role": "user", "content": v["user"], "images": [encode(bgr)]}]}
    if v["json"]:
        body["format"] = SCHEMA
    req = urllib.request.Request(OLLAMA_URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    dt = time.monotonic() - t0
    raw = out["message"].get("content", "")
    st, vis = parse(variant, raw)
    return dict(state=st, visible=vis, raw=raw.replace("\n", " "), latency_s=round(dt, 3),
                prompt_tokens=out.get("prompt_eval_count"), eval_tokens=out.get("eval_count"))


# ── geometry ─────────────────────────────────────────────────────────────────────────────────────────
ZED = dict(fx=448.0, fy=448.0, cx=640.0, cy=360.0, W=1280, H=720, z=0.9775, back=0.075)
RICOH = dict(z=0.0325 + 1.275, back=0.170)


def door_frame(x, y, rot):
    """centre c, width axis a, wall normal n (world)."""
    return np.array([x, y]), np.array([-math.sin(rot), math.cos(rot)]), np.array([math.cos(rot), math.sin(rot)])


def doorway_corners(x, y, rot, half=0.55, top=2.15):
    c, a, _ = door_frame(x, y, rot)
    return np.array([[*(c + s * a), z] for s in (-half, half) for z in (0.0, top)])


def cam_pose(rx, ry, heading, back):
    f = np.array([math.cos(heading), math.sin(heading)])
    return np.array([rx, ry]) - back * f, heading


def project_zed(C, h, P):
    f = np.array([math.cos(h), math.sin(h)]); r = np.array([math.sin(h), -math.cos(h)])
    rel = P[:, :2] - C
    d = rel @ f; x = rel @ r
    return ZED["cx"] + ZED["fx"] * x / d, ZED["cy"] - ZED["fy"] * (P[:, 2] - ZED["z"]) / d, d


def zed_door_box(robot_xy, heading, door):
    """door = (x, y, rot) world. Returns dict(u0,u1,v0,v1,dist,side,oblique,in_view)."""
    C, h = cam_pose(robot_xy[0], robot_xy[1], heading, ZED["back"])
    P = doorway_corners(*door)
    u, v, d = project_zed(C, h, P)
    c, _, n = door_frame(*door)
    dist = float(np.linalg.norm(c - C))
    los = (c - C) / dist
    in_view = bool(d.min() > 0.3 and u.min() >= 0 and u.max() <= ZED["W"])
    return dict(u0=float(u.min()), u1=float(u.max()), v0=float(max(0, v.min())), v1=float(min(ZED["H"], v.max())),
                dist=dist, side=int(np.sign((C - c) @ n)), oblique=float(math.degrees(math.acos(abs(los @ n)))),
                in_view=in_view)


def square_crop(bgr, box, k=1.35, wk=2.6):
    """Square crop around a doorway box: 1.35x its height, at least 2.6x its width (so a swung leaf fits)."""
    H, W = bgr.shape[:2]
    cu = (box["u0"] + box["u1"]) / 2; cv_ = (box["v0"] + box["v1"]) / 2
    side = min(max(k * (box["v1"] - box["v0"]), wk * (box["u1"] - box["u0"])), H)
    x0 = int(np.clip(cu - side / 2, 0, W - side)); y0 = int(np.clip(cv_ - side / 2, 0, H - side))
    return (x0, y0, int(side)), bgr[y0:y0 + int(side), x0:x0 + int(side)].copy()


def draw_box(crop, box, origin, colour=(0, 0, 255), thick=3):
    x0, y0 = origin
    out = crop.copy()
    cv2.rectangle(out, (int(box["u0"] - x0), int(box["v0"] - y0)), (int(box["u1"] - x0), int(box["v1"] - y0)),
                  colour, thick)
    return out


def ricoh_door_view(pano_bgr, robot_xy, heading, door, out=448, fov_margin=2.2, bearing_sign=+1):
    """Gnomonic (pinhole) view out of the equirectangular Ricoh panorama, aimed at the doorway centre.
    Assumes the panorama centre column is the robot's forward direction (as in the room_concept snapshots) and that
    image-left is robot-left; `bearing_sign` flips that if live_calib.py --check shows the box on the wrong side."""
    C, h = cam_pose(robot_xy[0], robot_xy[1], heading, RICOH["back"])
    c, _, _ = door_frame(*door)
    rel = c - C
    dist = float(np.linalg.norm(rel))
    bearing = math.atan2(rel[1], rel[0]) - h                       # + = left of forward
    elev = math.atan2(1.05 - RICOH["z"], dist)                     # aim at mid-door height
    fov = min(math.radians(100), fov_margin * 2 * math.atan2(1.1, dist))
    f = (out / 2) / math.tan(fov / 2)
    Hp, Wp = pano_bgr.shape[:2]
    jj, ii = np.meshgrid(np.arange(out), np.arange(out))
    x = (jj - out / 2) / f; y = -(ii - out / 2) / f                 # camera: x right, y up, z forward
    dirs = np.stack([x, y, np.ones_like(x)], -1)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    ce, se = math.cos(elev), math.sin(elev)                         # pitch by elev
    yv = dirs[..., 1] * ce + dirs[..., 2] * se
    zv = -dirs[..., 1] * se + dirs[..., 2] * ce
    xv = dirs[..., 0]
    az = bearing_sign * bearing - np.arctan2(xv, zv)                # + = left
    el = np.arcsin(np.clip(yv, -1, 1))
    mapx = ((0.5 - az / (2 * math.pi)) % 1.0) * Wp
    mapy = (0.5 - el / math.pi) * Hp
    view = cv2.remap(pano_bgr, mapx.astype(np.float32), mapy.astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_WRAP)
    return view, dict(dist=dist, bearing_deg=math.degrees(bearing), fov_deg=math.degrees(fov))
