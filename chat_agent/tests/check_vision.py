#!/usr/bin/env python3
"""Live vision check (needs ollama + a vision model): saved camera frames through the same preparation chat_agent's
look() uses (longest side <= MAX_SIDE, JPEG quality Q, base64), asked a few questions; prints answers, token counts
and latency. A crude keyword grader marks each answer; read the answers, the grader is a smoke test, not a judge.

    python3 tests/check_vision.py [model] [max_side] [jpeg_quality] [num_predict]

Frames are the room-estimation snapshots under room_concept/estimation_scripts (overlays drawn on them — the prompt
tells the model to ignore them).
"""
import base64
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
SNAP = ROOT.parent / "room_concept/estimation_scripts"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3-vl:8b-instruct"
MAX_SIDE = int(sys.argv[2]) if len(sys.argv) > 2 else 768
QUALITY = int(sys.argv[3]) if len(sys.argv) > 3 else 80
NUM_PREDICT = int(sys.argv[4]) if len(sys.argv) > 4 else 300
URL = "http://localhost:11434/api/chat"

SYSTEM = ("You are the voice of Shadow, a mobile robot. You are shown an image from one of your cameras. "
          "Coloured lines, dots, grids, text labels (wall_N) and a legend box are DEBUG OVERLAYS drawn on top of the "
          "picture: ignore them, describe only the real scene. The light-grey curved shape at the bottom of the image "
          "is your own body. Answer in one to three short sentences. Left/right are from the camera's point of view.")

FRAMES = {
    "kitchen_corridor": "apartment/campaign_v5_curious/run_0/snapshots/zed_1791745901549.png",
    "door_ahead": "crossing/try7/snapshots/zed_1791753485762.png",
    "tables_room": "synthetic/campaign_v5_furn/run_7/snapshots/zed_1791723369265.png",
    "kitchen_far": "apartment/campaign_v5_curious/run_5/snapshots/zed_1791746529276.png",
    "pano_360": "crossing/try7/snapshots/ricoh_1791753485929.png",
}

# (frame, question, grader on the lower-cased answer)
CASES = [
    ("kitchen_corridor", "What do you see?",
     lambda a: re.search(r"kitchen|cabinet|counter|hood", a) is not None),
    ("kitchen_corridor", "Is there a door? Is it open or closed?",
     lambda a: "door" in a and re.search(r"closed|shut", a) is not None),
    ("kitchen_corridor", "Is the kitchen on your left or on your right?",
     lambda a: "left" in a and "right" not in a.split("left")[0][-20:]),
    ("door_ahead", "Is there a door in front of you? Is it open?",
     lambda a: "door" in a and re.search(r"closed|shut|not open", a) is not None),
    ("door_ahead", "What colour is the door?",
     lambda a: "green" in a),
    ("tables_room", "How many tables do you see?",
     lambda a: re.search(r"\b(2|two)\b", a) is not None),
    ("tables_room", "How many chairs do you see?",
     lambda a: re.search(r"\b(0|no|none|zero)\b|not see any|don't see any|do not see any", a) is not None),
    ("kitchen_far", "Is there a refrigerator? Is it near or far?",
     lambda a: re.search(r"fridge|refrigerator", a) is not None and "far" in a),
    ("pano_360", "This is your 360-degree camera (the panorama wraps around). Is there a door, and is it open?",
     lambda a: "door" in a and re.search(r"closed|shut", a) is not None),
    ("pano_360", "Are there any chairs? What colour are they?",
     lambda a: "chair" in a and re.search(r"yellow|pink|magenta|purple", a) is not None),
]


def prep(path):
    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    s = min(1.0, MAX_SIDE / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
    assert ok
    return base64.b64encode(buf.tobytes()).decode(), img.shape[1], img.shape[0], len(buf)


def ask(question, b64):
    body = {"model": MODEL, "stream": False, "think": False, "keep_alive": "30m",
            "options": {"temperature": 0.2, "num_ctx": 8192, "num_predict": NUM_PREDICT},
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": question, "images": [b64]}]}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.loads(r.read())
    return out, time.monotonic() - t0


def main():
    cache = {k: prep(SNAP / v) for k, v in FRAMES.items()}
    passed = 0
    for frame, question, ok in CASES:
        b64, w, h, nbytes = cache[frame]
        out, dt = ask(question, b64)
        ans = out["message"].get("content", "").strip()
        think = out["message"].get("thinking") or ""
        good = ok(ans.lower())
        passed += good
        print(f"{'PASS' if good else 'FAIL'} [{frame} {w}x{h} {nbytes // 1024} KiB] {question}\n"
              f"     -> {ans!r}\n     prompt={out.get('prompt_eval_count')} tok, out={out.get('eval_count')} tok, "
              f"{dt:.1f} s, done={out.get('done_reason')}{', THINKING ' + str(len(think)) + ' chars' if think else ''}")
    print(f"\n{passed}/{len(CASES)} keyword checks passed (model {MODEL}, side {MAX_SIDE}, q {QUALITY}, "
          f"num_predict {NUM_PREDICT})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
