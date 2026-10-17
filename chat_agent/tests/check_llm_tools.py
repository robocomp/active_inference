#!/usr/bin/env python3
"""Live check (needs ollama): does the model map motion / object questions to the right tool calls?

Builds the same request chat_agent builds — a system message of etc/system_prompt.md + the interface rules extracted
from src/chat_session.cpp (so they cannot drift), the history, the question, and LAST a system message with a
SYNTHETIC STATE — and the same tool schema, then asks a few questions and grades the first tool call. Also reports
prompt_eval_count = the real token size of system+tools+STATE.

    python3 tests/check_llm_tools.py [model] [repeats]
"""
import json
import re
import statistics
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3:8b"
REPEATS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
URL = "http://localhost:11434/api/chat"


def interface_rules(default_step="0.3"):
    src = (ROOT / "src/chat_session.cpp").read_text()
    m = re.search(r'kInterfaceRules = QStringLiteral\((.*?")\);', src, re.S)
    lits = re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))
    text = "".join(lits).encode().decode("unicode_escape").encode("latin-1").decode("utf-8")
    return text.replace("%1", default_step)


STATE = """time: 18:42:10
controller (the agent that drives the robot): running
current room: room_1
other rooms in the live map: none
robot pose in room_1: x=0.42 m, y=-1.10 m, yaw=88 deg
doors of the current room (2):
  - door_0: 3.1 m away, bearing +62 deg (ahead-left), P(open)=0.91, affordances: approach=offered open=absent cross=offered [nearest]
  - door_1: 4.4 m away, bearing -120 deg (behind-right), P(open)=0.12, affordances: approach=offered open=offered cross=absent
remembered rooms (not in the live map, known from memory): none
objects perceived in the current room (only this room is perceived): 1 bottle, 2 chair, 1 table
  nearest 4:
  - chair_1 2.1 m, +35 deg ahead-left, 0.45x0.50x0.90 m
  - table_1 2.8 m, +10 deg ahead, 1.20x0.80x0.75 m
  - bottle_1 2.9 m, +12 deg ahead, 0.07x0.07x0.25 m
  - chair_2 3.4 m, -100 deg to the right, 0.45x0.50x0.90 m
last command sent from this chat: none
"""


def tools():
    def t(name, desc, props, req):
        p = {"type": "object", "properties": props}
        if req:
            p["required"] = req
        return {"type": "function", "function": {"name": name, "description": desc, "parameters": p}}
    return [
        t("cross_door", "Take the robot through a door of the CURRENT room: approach it, open it if it is shut, and go "
          "through. Shows the person a confirmation card; nothing moves until they confirm.",
          {"door": {"type": "string", "enum": ["door_0", "door_1", "nearest"],
                    "description": "a door of the current room, or 'nearest'"}}, ["door"]),
        t("stop", "Stop the robot immediately and abandon what it is doing.", {}, []),
        t("cancel_passage", "Abandon the door crossing in progress (the robot stays where it is).", {}, []),
        t("move_relative", "Move the robot a short distance relative to where it is now: first turn in place by "
          "turn_deg, then drive straight distance_m. Small moves start at once; long ones ask the person to confirm. "
          "The robot refuses if a wall or obstacle is in the way.",
          {"distance_m": {"type": "number", "description": "metres to drive after turning: positive = forward, "
                                                           "negative = backward, 0 = do not drive"},
           "turn_deg": {"type": "number", "description": "degrees to turn in place first: positive = left, "
                                                         "negative = right, 180 = turn around, 0 = do not turn"}},
          ["distance_m", "turn_deg"]),
        t("find_objects", "List the objects the robot has perceived in the CURRENT room, nearest first, with "
          "distance, bearing and size. Read-only.",
          {"subtype": {"type": "string", "description": "a class to filter by, in English singular (chair, table, "
                                                        "bottle, cabinet, refrigerator, hood, person); empty = all"}}, []),
        t("where_is", "Where is an object: give its name (e.g. chair_1) or its class (e.g. refrigerator). Returns its "
          "room and, if it is in the current room, its distance and bearing; or that it has not been perceived. "
          "Read-only.", {"name": {"type": "string", "description": "an object name or class"}}, ["name"]),
        t("look", "Look through a camera now: the current picture is shown to you in the next message. Read-only.",
          {"camera": {"type": "string", "enum": ["front", "360"],
                      "description": "'front' = the camera facing ahead (default); '360' = the all-round panorama"}}, []),
    ]


def system():
    desc = (ROOT / "etc/system_prompt.md").read_text().strip()
    return desc + "\n" + interface_rules()


def state(last_move="none"):
    expl = {"move 0.30": "move 0.30 (move forward 0.30 m)", "move -0.30": "move -0.30 (move backward 0.30 m)"}
    return "STATE:\n" + STATE + f"last move/turn sent from this chat: {expl.get(last_move, last_move)}\n"


def ask(messages):
    body = {"model": MODEL, "messages": messages, "tools": tools(), "stream": False, "think": False,
            "keep_alive": "30m", "options": {"temperature": 0.2, "num_ctx": 16384}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())


def first_call(resp):
    calls = resp.get("message", {}).get("tool_calls") or []
    if not calls:
        return None, {}
    fn = calls[0]["function"]
    args = fn.get("arguments") or {}
    if isinstance(args, str):
        args = json.loads(args)
    return fn["name"], args


def num(a, k):
    try:
        return float(a.get(k, 0) or 0)
    except (TypeError, ValueError):
        return float("nan")


def history_after_move(user_text, dist, turn, cmd):
    return [{"role": "user", "content": user_text},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "move_relative",
                                                                              "arguments": {"distance_m": dist, "turn_deg": turn}}}]},
            {"role": "tool", "tool_name": "move_relative", "content": f"OK: '{cmd}' sent to the controller."},
            {"role": "assistant", "content": "Done, I moved."},
            {"role": "tool", "tool_name": "robot_status", "content": f"The command '{cmd}' ended: completed. done: {cmd}"}]


CASES = [
    # (label, history, last_move, user text, grader(name, args, content) -> bool)
    ("advance 1 meter", [], "none", "advance 1 meter",
     lambda n, a, c: n == "move_relative" and abs(num(a, "distance_m") - 1.0) < 1e-6 and num(a, "turn_deg") == 0),
    ("advance a bit", [], "none", "advance a bit",
     lambda n, a, c: n == "move_relative" and abs(num(a, "distance_m") - 0.3) < 1e-6 and num(a, "turn_deg") == 0),
    ("a bit more (after forward)", history_after_move("advance a bit", 0.3, 0, "move 0.30"), "move 0.30",
     "advance a bit more",
     lambda n, a, c: n == "move_relative" and abs(num(a, "distance_m") - 0.3) < 1e-6 and num(a, "turn_deg") == 0),
    ("a bit more (after backward)", history_after_move("back up a little", -0.3, 0, "move -0.30"), "move -0.30",
     "a bit more",
     lambda n, a, c: n == "move_relative" and abs(num(a, "distance_m") + 0.3) < 1e-6 and num(a, "turn_deg") == 0),
    ("turn around", [], "none", "turn around",
     lambda n, a, c: n == "move_relative" and abs(abs(num(a, "turn_deg")) - 180) < 1e-6 and num(a, "distance_m") == 0),
    ("turn right", [], "none", "turn right",
     lambda n, a, c: n == "move_relative" and num(a, "turn_deg") == -90 and num(a, "distance_m") == 0),
    ("avanza un poco (es)", [], "none", "avanza un poco",
     lambda n, a, c: n == "move_relative" and abs(num(a, "distance_m") - 0.3) < 1e-6),
    ("how many chairs", [], "none", "how many chairs are there?",
     lambda n, a, c: (n == "find_objects" and "chair" in str(a.get("subtype", "")).lower())
     or (n is None and re.search(r"\b(2|two)\b", c.lower()) is not None)),
    ("where is the fridge", [], "none", "where is the fridge?",
     lambda n, a, c: (n == "where_is" and re.search("fridge|refrigerator", str(a.get("name", "")).lower()) is not None)
     or (n is None and re.search(r"not (yet )?(perceived|seen)|haven't (perceived|seen)|have not (perceived|seen)", c.lower()) is not None)),
    # look(): appearance / "what do you see" goes to the camera; counts stay with the STATE (the chairs case above).
    ("what do you see", [], "none", "what do you see?",
     lambda n, a, c: n == "look" and str(a.get("camera", "front")) in ("front", "")),
    ("colour of the chair", [], "none", "what colour is the chair?",
     lambda n, a, c: n == "look"),
    ("look behind you", [], "none", "what do you see behind you?",
     lambda n, a, c: n == "look" and str(a.get("camera", "")) == "360"),
]


def main():
    sys_tokens = None
    results = []
    lat = []
    for label, hist, last_move, text, ok in CASES:
        for k in range(REPEATS):
            msgs = [{"role": "system", "content": system()}] + hist + [{"role": "user", "content": text},
                                                                       {"role": "system", "content": state(last_move)}]
            t0 = time.monotonic()
            r = ask(msgs)
            lat.append(time.monotonic() - t0)
            if sys_tokens is None and not hist:
                sys_tokens = r.get("prompt_eval_count")
            n, a = first_call(r)
            content = r.get("message", {}).get("content", "")
            good = ok(n, a, content)
            results.append(good)
            shown = f"{n}({json.dumps(a)})" if n else f"text: {content.strip()[:110]!r}"
            print(f"{'PASS' if good else 'FAIL'}  {label:<30} -> {shown}")
    sm = system()
    print(f"\nsystem message: {len(sm)} chars; STATE: {len(STATE)} chars; tools JSON: {len(json.dumps(tools()))} chars")
    print(f"prompt_eval_count (system + tools + one user line + STATE): {sys_tokens} tokens")
    ls = sorted(lat)
    print(f"latency: median {statistics.median(ls):.2f} s, mean {statistics.mean(ls):.2f} s, "
          f"max {ls[-1]:.2f} s (first call includes any cold load: {lat[0]:.2f} s)")
    print(f"{sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
