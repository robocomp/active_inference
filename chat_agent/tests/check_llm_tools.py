#!/usr/bin/env python3
"""Live check (needs ollama): does the model map motion / object questions to the right tool calls?

Builds the same request chat_agent builds — a system message of etc/system_prompt.md + the interface rules extracted
from src/chat_session.cpp (so they cannot drift), the history, the question, and LAST a system message with a
SYNTHETIC STATE — and the same tool schema, then asks a few questions and grades the first tool call. Also reports
prompt_eval_count = the real token size of system+tools+STATE.

    python3 tests/check_llm_tools.py [model] [repeats]
"""
import json
import os
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
can move: yes
current room: room_1
other rooms in the live map: none
robot pose in room_1: x=0.42 m, y=-1.10 m, yaw=88 deg
robot motion now (odometry, 80 ms old): forward 0.00 m/s, sideways 0.00 m/s, not turning
controller target now: none (it is executing no affordance)
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
today: 2 missions (1 succeeded); last: cross door_0 → room_1
"""


def tools():
    def t(name, desc, props, req):
        p = {"type": "object", "properties": props}
        if req:
            p["required"] = req
        return {"type": "function", "function": {"name": name, "description": desc, "parameters": p}}
    return [
        t("cross_door", "Take the robot through a door of the CURRENT room: approach it, open it if it is shut, and go "
          "through. Starts at once.",
          {"door": {"type": "string", "enum": ["door_0", "door_1", "nearest"],
                    "description": "a door of the current room, or 'nearest'"}}, ["door"]),
        t("stop", "Stop the robot immediately and abandon what it is doing.", {}, []),
        t("cancel_passage", "Abandon the door crossing in progress (the robot stays where it is).", {}, []),
        t("move_relative", "Move the robot a short distance relative to where it is now: first turn in place by "
          "turn_deg, then drive straight distance_m. Starts at once. The robot refuses if a wall or obstacle "
          "is in the way. Not for going to an object (approach_object) or through a door (cross_door).",
          {"distance_m": {"type": "number", "description": "metres to drive after turning: positive = forward, "
                                                           "negative = backward, 0 = do not drive"},
           "turn_deg": {"type": "number", "description": "degrees to turn in place first: positive = left, "
                                                         "negative = right, 180 = turn around, 0 = do not turn"}},
          ["distance_m", "turn_deg"]),
        t("approach_object", "Drive to an object of the CURRENT room and stop in front of it, facing it: the "
          "controller plans the path around obstacles. Give its name (chair_1) or its class (table: the nearest one). "
          "Starts at once.", {"object": {"type": "string", "description": "an object name or class"}}, ["object"]),
        t("find_objects", "List the objects the robot has perceived in the CURRENT room, nearest first, with "
          "distance, bearing and size. Read-only.",
          {"subtype": {"type": "string", "description": "a class to filter by, in English singular (chair, table, "
                                                        "bottle, cabinet, refrigerator, hood, person); empty = all"}}, []),
        t("where_is", "Where is an object: give its name (e.g. chair_1) or its class (e.g. refrigerator). Returns its "
          "room and, if it is in the current room, its distance and bearing; or that it has not been perceived. "
          "Read-only.", {"name": {"type": "string", "description": "an object name or class"}}, ["name"]),
        t("agents", "The robot's software agents (programs such as controller, retina, room_concept, door_concept): "
          "which are running, their state (Operating / Degraded / Waiting), uptime, CPU and memory of each. "
          "Read-only.", {}, []),
        t("sensors", "The robot's sensors (cameras, LiDAR, IMU): their streams, the frame rate MEASURED now, "
          "resolution, age of the last frame and bandwidth. Read-only.",
          {"name": {"type": "string", "description": "a sensor or camera to filter by (e.g. ricoh, 360, zed, front, "
                                                     "lidar); empty = all"}}, []),
        t("graph_summary", "What the robot's working memory (its graph) holds: node and edge counts by type and by the "
          "agent that created them. Read-only.", {}, []),
        t("inspect", "Every attribute and edge of ONE node of the robot's graph, by name (e.g. door_1, Shadow, "
          "controller 7) or id. Read-only.", {"node": {"type": "string", "description": "the node's name or id"}},
          ["node"]),
        # episodic memory (chat_session.cpp tools_json; answers in src/em_answers.h)
        t("episodes", "What the robot did TODAY, from its episodic memory, with start and end time and outcome, oldest "
          "first. With no kind and no target it lists only the MISSIONS; pass kind (e.g. cross, open) for the steps and "
          "claims of that kind, target for everything aimed at one node, or call episode(id) for one mission's steps "
          "and claims. All filters optional. Read-only.",
          {"since": {"type": "string", "description": "only episodes that started at or after this time: HH:MM (today, "
                                                      "local time) or epoch milliseconds"},
           "until": {"type": "string", "description": "only episodes that started before this time: HH:MM (today, local "
                                                      "time) or epoch milliseconds"},
           "author": {"type": "string", "enum": ["human", "self"],
                      "description": "human = asked by a person, self = on the robot's own"},
           "kind": {"type": "string", "description": "mission, or a step/claim kind such as cross, open, approach"},
           "target": {"type": "string", "description": "a door, room or object name, e.g. door_27"}}, []),
        t("episode", "One episode from today's episodic memory in detail, by id (as listed by episodes): its steps and "
          "claims, the beliefs read before and after (e.g. P(open) of a door) and what happened meanwhile. Read-only.",
          {"id": {"type": "string", "description": "the episode id, e.g. 20261017-3"}}, ["id"]),
        t("outcome_stats", "How often something succeeded, failed or was lost over ALL TIME (today included), its mean "
          "duration and information gain, from the episodic memory. Steps and claims of the same kind are separate "
          "rows; missions are split into asked-by-a-person and on-its-own. Read-only.",
          {"kind": {"type": "string", "description": "mission, or a step/claim kind such as cross, open, approach"},
           "target": {"type": "string", "description": "optional: a door/room/object name, e.g. door_27; empty = all "
                                                       "of that kind"}}, ["kind"]),
        t("last_episode", "The most recent episode today about one door, room or object (aimed at it or that read its "
          "beliefs). Read-only.",
          {"target": {"type": "string", "description": "a door, room or object name, e.g. door_27"}}, ["target"]),
        t("look", "Look through a camera now: the current picture is shown to you in the next message. Read-only.",
          {"camera": {"type": "string", "enum": ["front", "360"],
                      "description": "'front' = the camera facing ahead (default); '360' = the all-round panorama"}}, []),
    ]


def system():
    desc = (ROOT / "etc/system_prompt.md").read_text().strip()
    return desc + "\n" + interface_rules()


def state(last_move="none", extra=""):
    expl = {"move 0.30": "move 0.30 (move forward 0.30 m)", "move -0.30": "move -0.30 (move backward 0.30 m)"}
    base = STATE.replace("last command sent from this chat: none\n", "") if "last command sent" in extra else STATE
    for key in ("robot motion now", "controller target now"):   # the case's own line replaces the default one
        if key in extra:
            base = "\n".join(l for l in base.split("\n") if not l.startswith(key))
    return "STATE:\n" + base + extra + f"last move/turn sent from this chat: {expl.get(last_move, last_move)}\n"


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
    # shut doors, live 2026-10-18: the model invented "confirmation from the door controller" and looped on "yes"
    # without a call. cross_door opens a shut door itself; a yes to an offered action is that action's call.
    ("cross the shut door", [], "none", "cross door_1 into the next room",
     lambda n, a, c: n == "cross_door" and str(a.get("door", "")) == "door_1"),
    ("cross it (door is shut)", [{"role": "user", "content": "see door_1?"},
                                 {"role": "assistant", "content": "Yes, I can see door_1. It is 4.4 meters away, behind "
                                  "my right side, and it appears to be shut (P(open)=0.12)."}], "none",
     "cross it into the next room",
     lambda n, a, c: n == "cross_door" and str(a.get("door", "")) == "door_1"),
    ("yes after an offer", [{"role": "user", "content": "cross door_1"},
                            {"role": "assistant", "content": "door_1 is shut (P(open)=0.12). To cross it, I must first "
                             "open it, but I cannot do that without confirmation from the door controller. Would you "
                             "like me to attempt to open it?"}], "none",
     "yes",
     lambda n, a, c: n == "cross_door" and str(a.get("door", "")) == "door_1"),
    # live 2026-10-18: the person deleted door_3 during its crossing; the model went on describing it from its own
    # earlier reply. The STATE now says the door is gone (doors_gone / command_target_gone, command_gate.h).
    ("deleted door", [{"role": "user", "content": "do you see a door in front of you?"},
                      {"role": "assistant", "content": "Yes, door_3 is 1.8 m away at -26 deg (ahead-right). It is "
                       "shut (P(open)=0.00)."},
                      {"role": "user", "content": "go there and cross it"},
                      {"role": "assistant", "content": "I am approaching door_3 to cross it."}], "none",
     "check now the door",
     lambda n, a, c: n == "cancel_passage" or n is None and (re.search(r"door_3", c) is None
                                    or re.search(r"no longer|gone|deleted|removed|not there|isn't there", c) is not None),
     "doors gone since the last message: door_3 (deleted from the map: it is not there any more, whatever earlier "
     "messages said)\nlast command sent from this chat: \"cross door_3\", status: in_progress, step: open, WARNING: "
     "door_3 is NO LONGER in the map (deleted), so this crossing cannot finish; call cancel_passage() yourself to end it\n"),
    # live 2026-10-18: "you ARE moving" — the controller drove on its own, the chat had sent nothing, and the model
    # said it was not moving. The STATE now carries the measured motion (world_digest.cpp, motion_line).
    ("where are you going (moving)", [], "none", "where are you going?",
     lambda n, a, c: re.search(r"not (currently )?moving|standing still|stationary|not going anywhere|no active",
                               c.lower()) is None,
     "robot motion now (odometry, 60 ms old): forward 0.38 m/s, sideways 0.00 m/s, turning left 9 deg/s\n"),
    # where it is going: the controller's own target (world_digest.cpp; controller_facts.h), not an invention.
    ("where to (affordance)", [], "none", "where are you going?",
     lambda n, a, c: "door_0" in c and "table" not in c.lower(),
     "robot motion now (odometry, 60 ms old): forward 0.38 m/s, sideways 0.00 m/s, turning left 9 deg/s\ncontroller target now: cross door_0 (aff_door_0_cross), its goal 2.9 m away, ahead-left\n"),
    ("where to (mission)", [], "none", "what are you doing?",
     lambda n, a, c: "table_1" in c,
     "robot motion now (odometry, 60 ms old): forward 0.38 m/s, sideways 0.00 m/s, turning left 9 deg/s\ncontroller mission now: \"approach table_1\" (from chat), running: go_to pose(room_1 0.95 -0.25)\n"),
    ("where to (exploring)", [], "none", "where are you going?",
     lambda n, a, c: re.search(r"explor", c.lower()) is not None,
     "robot motion now (odometry, 60 ms old): forward 0.38 m/s, sideways 0.00 m/s, turning left 9 deg/s\ncontroller target now: explore the room (afford_room), its goal 1.8 m away, to the left\n"),
    ("are you moving (moving)", [], "none", "are you moving?",
     lambda n, a, c: re.search(r"\byes\b|i am moving|i'm moving|moving forward", c.lower()) is not None,
     "robot motion now (odometry, 60 ms old): forward 0.38 m/s, sideways 0.00 m/s, turning left 9 deg/s\n"),
    ("are you moving (still)", [], "none", "are you moving?",
     lambda n, a, c: re.search(r"\bno\b|not moving|standing still|stationary", c.lower()) is not None),
    # live 2026-10-18: "approach the table" / "approach chair 1" — no tool existed: a straight move at the chair (refused,
    # the chair blocks it) or a claim of moving with no call at all. approach_object sends a go_to mission.
    ("approach the table", [], "none", "approach the table",
     lambda n, a, c: n == "approach_object" and "table" in str(a.get("object", "")).lower()),
    ("approach chair 1", [], "none", "approach chair 1",
     lambda n, a, c: n == "approach_object" and str(a.get("object", "")).lower().replace(" ", "_") in ("chair_1",)),
    ("go to the bottle", [], "none", "go to the bottle",
     lambda n, a, c: n == "approach_object" and "bottle" in str(a.get("object", "")).lower()),
    ("acércate a la mesa (es)", [], "none", "acércate a la mesa",
     lambda n, a, c: n == "approach_object" and re.search("table|mesa", str(a.get("object", "")).lower()) is not None),
    # system questions: the STATE's agents/cameras lines answer the gist, the tools the details.
    ("how many agents", [], "none", "how many agents are running?",
     lambda n, a, c: n == "agents" or (n is None and re.search(r"\b(14|fourteen)\b", c.lower()) is not None)),
    ("cuántos agentes (es)", [], "none", "¿cuántos agentes están funcionando?",
     lambda n, a, c: n == "agents" or (n is None and re.search(r"\b(14|catorce)\b", c.lower()) is not None)),
    # live 2026-10-18: "agents ... in the WM" was answered as "I have no access to agents" (no tool call). The WM IS
    # the graph, so graph_summary is a fair pick: its result carries the agents census too (system_facts.h).
    ("agents alive in the WM", [], "none", "how many agents are alive in the WM?",
     lambda n, a, c: n in ("agents", "graph_summary")),
    ("agents alive", [], "none", "how many agents are alive?",
     lambda n, a, c: n == "agents"),
    ("agents in working memory", [], "none", "which agents are in the working memory?",
     lambda n, a, c: n in ("agents", "graph_summary")),
    # paraphrases: at temperature 0.2 a phrasing gets the same tool every repeat, so the spread is across phrasings.
    ("agents in the WM", [], "none", "how many agents are in the WM?",
     lambda n, a, c: n in ("agents", "graph_summary")),
    ("all agents up", [], "none", "are all the agents up?",
     lambda n, a, c: n == "agents"),
    ("list running agents", [], "none", "list the running agents",
     lambda n, a, c: n == "agents"),
    ("is retina running", [], "none", "is retina running?",
     lambda n, a, c: n == "agents" or (n == "inspect" and "retina" in str(a.get("node", "")).lower())),
    ("which programs run", [], "none", "which programs are running right now?",
     lambda n, a, c: n == "agents"),
    ("controller alive", [], "none", "is the controller alive?",
     lambda n, a, c: n == "agents" or (n == "inspect" and "controller" in str(a.get("node", "")).lower())),
    ("agentes vivos WM (es)", [], "none", "¿cuántos agentes hay vivos en la WM?",
     lambda n, a, c: n in ("agents", "graph_summary")),
    ("room_concept cpu", [], "none", "what is the CPU load of room_concept?",
     lambda n, a, c: n == "agents" or (n == "inspect" and "room_concept" in str(a.get("node", "")).lower())),
    ("which agent is degraded", [], "none", "is any agent in trouble?",
     lambda n, a, c: n == "agents" or (n is None and "door_concept" in c)),
    ("agent memory use", [], "none", "how much memory is the controller using?",
     lambda n, a, c: n == "agents" or (n == "inspect" and "controller" in str(a.get("node", "")).lower())),
    ("ricoh frame rate", [], "none", "what is the frame rate of the Ricoh camera?",
     lambda n, a, c: (n == "sensors" and re.search("ricoh|360", str(a.get("name", "")).lower()) is not None)
     or (n is None and "15" in c)),
    ("ricoh resolution", [], "none", "what resolution does the 360 camera stream at?",
     lambda n, a, c: n == "sensors" and re.search("ricoh|360|^$", str(a.get("name", "")).lower()) is not None),
    ("fps de la cámara (es)", [], "none", "¿a cuántos fps va la cámara frontal?",
     lambda n, a, c: (n == "sensors" and re.search("zed|front|frontal|^$", str(a.get("name", "")).lower()) is not None)
     or (n is None and re.search(r"29[.,]9|30", c) is not None)),
    ("working memory", [], "none", "how many nodes are in your working memory?",
     lambda n, a, c: n == "graph_summary"),
    ("inspect a node", [], "none", "show me all the attributes of door_1",
     lambda n, a, c: n == "inspect" and str(a.get("node", "")) == "door_1"),
    ("look behind you", [], "none", "what do you see behind you?",
     lambda n, a, c: n == "look" and str(a.get("camera", "")) == "360"),
    # episodic memory: what happened earlier goes to the memory tools, never to the live STATE or a guess.
    ("what did you do this morning", [], "none", "what did you do this morning?",
     lambda n, a, c: n == "episodes"),
    ("what did you do today", [], "none", "what have you done today?",
     lambda n, a, c: n == "episodes" or (n is None and re.search(r"\b(2|two)\b", c.lower()) is not None)),
    ("since 10:00", [], "none", "what missions did I give you since 10:00?",
     lambda n, a, c: n == "episodes" and str(a.get("since", "")).startswith("10:00")),
    ("qué hiciste hoy (es)", [], "none", "¿qué has hecho hoy?",
     lambda n, a, c: n == "episodes" or (n is None and re.search(r"\b(2|dos)\b", c.lower()) is not None)),
    ("how often door shut", [], "none", "how often was door_1 shut?",
     lambda n, a, c: n == "outcome_stats" and str(a.get("target", "")) == "door_1"),
    ("how often crossing failed", [], "none", "how often have door crossings failed?",
     lambda n, a, c: n == "outcome_stats" and "cross" in str(a.get("kind", "")).lower()),
    ("last time at door_0", [], "none", "when did you last go through door_0?",
     lambda n, a, c: n == "last_episode" and str(a.get("target", "")) == "door_0"
     or (n == "episodes" and str(a.get("target", "")) == "door_0")),
    ("details of an episode", [], "none", "tell me more about episode 20261017-3",
     lambda n, a, c: n == "episode" and str(a.get("id", "")) == "20261017-3"),
]


def main():
    sys_tokens = None
    results = []
    lat = []
    only = re.compile(os.environ.get("ONLY", ""))   # ONLY=regex runs just the cases whose label matches
    for label, hist, last_move, text, ok, *extra in (c for c in CASES if only.search(c[0])):
        for k in range(REPEATS):
            msgs = [{"role": "system", "content": system()}] + hist + [{"role": "user", "content": text},
                                                                       {"role": "system", "content": state(last_move, *extra)}]
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
