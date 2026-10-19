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
# the cap chat_agent sends (etc/config.toml [LLM] num_predict): a call longer than it is cut and lost, live and here
NUM_PREDICT = int(re.search(r"^num_predict\s*=\s*(-?\d+)", (ROOT / "etc/config.toml")
                            .read_text(), re.M).group(1))
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3-vl:8b-instruct"   # the model in etc/config.toml
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
agents alive now: chair_concept, chat_agent, controller, door_concept (running, but Degraded), ltsm_agent, residual_concept, retina, robot_concept, room_concept, table_concept (10; agents() for health, CPU, memory)
doors of the current room (2):
  - door_0: 3.1 m away, bearing +62 deg (ahead-left), P(open)=0.91, affordances: approach=offered open=absent cross=offered [nearest]
  - door_1: 4.4 m away, bearing -120 deg (behind-right), P(open)=0.12, affordances: approach=offered open=offered cross=absent
remembered rooms (not in the live map, known from memory): none
objects perceived in the current room (only this room is perceived): 1 bottle, 2 chair, 1 table
  nearest 4:
  - chair_1 2.1 m, +35 deg ahead-left, 0.45x0.50x0.90 m, approach sides: front (its own front; far side as seen from here), left (its own left; left side as seen from here), right (its own right; right side as seen from here)
  - table_1 2.8 m, +10 deg ahead, 1.20x0.80x0.75 m
  - bottle_1 2.9 m, +12 deg ahead, 0.07x0.07x0.25 m
  - chair_2 3.4 m, -100 deg to the right, 0.45x0.50x0.90 m
last command sent from this chat: none
today: 2 missions (1 succeeded); last: cross door_0 → room_1
"""


# the fleet down (live 2026-10-19): no room, no controller, nothing perceived
EMPTY_STATE = """time: 12:17:35
can move: NO (the motion controller is not running)
current room: unknown
other rooms in the live map: none
robot pose: unknown (no current room)
robot motion now (odometry, 29 ms old): forward 0.00 m/s, sideways 0.00 m/s, turning right 0 deg/s
controller target now: none (it is executing no affordance)
doors of the current room (0):
  none known
remembered rooms (not in the live map, known from memory): none
objects perceived in the current room (only this room is perceived): none
last command sent from this chat: none
today: 1 mission (0 succeeded); last: leg 3 → room_1
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
        t("approach_object", "Drive to an object of the CURRENT room and stop in front of it, facing it: the object "
          "itself says where to stand and from which sides it can be approached (the STATE lists them); the controller "
          "plans the path around obstacles. Give its name (chair_1) or its class (table: the nearest one), and "
          "optionally the side in the person's own words. Starts at once.",
          {"object": {"type": "string", "description": "an object name or class"},
           "side": {"type": "string", "description": "ONLY when the person names a side: the side to approach from, as they said it "
                                                     "(right, the far side, behind it, por la izquierda) or a side name "
                                                     "from the STATE; empty = the object's default"}}, ["object"]),
        t("find_objects", "List the objects the robot has perceived in the CURRENT room, nearest first, with "
          "distance, bearing and size. Read-only.",
          {"subtype": {"type": "string", "description": "a class to filter by, in English singular (chair, table, "
                                                        "bottle, cabinet, refrigerator, hood, person); empty = all"}}, []),
        t("where_is", "Where is an object: give its name (e.g. chair_1) or its class (e.g. refrigerator). Returns its "
          "room and, if it is in the current room, its distance and bearing; or that it has not been perceived. "
          "Read-only.", {"name": {"type": "string", "description": "an object name or class"}}, ["name"]),
        t("agent_control", "Start, stop or restart one of the robot's software agents (room_concept, residual_concept, "
          "retina, door_concept, table_concept, chair_concept, bottle_concept, cabinet_concept, hood_concept, "
          "human_concept, refrigerator_concept, controller, ltsm_agent, episodic_agent). When the person asks to start, "
          "stop or restart an agent, call it with the name they give, also one not listed: the tool says why not. "
          "Never to answer a question about them (that is agents). Stopping or restarting the controller or "
          "retina needs the person's yes: the tool says so, you ask, and you call it again with the same arguments "
          "only after they answer yes.",
          {"action": {"type": "string", "enum": ["start", "stop", "restart"]},
           "agent": {"type": "string", "description": "the agent's name, e.g. chair_concept"}}, ["action", "agent"]),
        t("agents", "The robot's software agents: which are running, their state (Operating / Degraded / Waiting), "
          "uptime, CPU and memory of each. Read-only.", {}, []),
        t("sensors", "The robot's sensors (cameras, LiDAR, IMU): their streams, the frame rate MEASURED now, "
          "resolution, age of the last frame and bandwidth: numbers about the streams, not what they measure. To let "
          "the person SEE what the LiDARs or cameras measure use show (lidar / camera). Read-only.",
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
        # the side canvas (chat_session.cpp tools_json; canvas_runner.h)
        t("show", "Show the PERSON a ready-made picture on the canvas beside the chat (you do not see it: look is for "
          "you). camera = the live camera picture (args 'zed' = front, 'ricoh' = 360); layout = the current room from "
          "above with the robot, doors and objects; plan = the mission the robot runs; graph = the robot's working "
          "memory as a graph of nodes and edges, the DSR graph: what it knows and how it is connected — 'the graph', 'your "
          "memory graph' (args: a node name to zoom in on it, e.g. 'room_1'; empty = all); "
          "model = the 3-D model of the robot or of an object, which the person can turn with the mouse (args: "
          "'robot', an object name like chair_1, or a class like table); lidar = the live 3-D LiDAR point clouds "
          "in the robot's frame (args: leave EMPTY for both, the usual answer; 'helios' or 'bpearl' only when the person names one); "
          "other names are drawings "
          "saved earlier. Only when one of these is exactly what was asked; any other chart, plot or picture: draw.",
          {"name": {"type": "string", "enum": ["camera", "graph", "layout", "lidar", "model", "plan"], "description": "which drawing"},
           "args": {"type": "string", "description": "camera: 'zed' (front, default) or 'ricoh' (360)"},
           "live": {"type": "boolean", "description": "true = keep redrawing it (default for camera and "
                                                      "layout); false = draw once"}}, ["name"]),
        t("draw", "Draw a NEW picture for the PERSON on the canvas beside the chat by writing a Python script (NOT for "
          "the camera, the room map, the plan, the working-memory graph or a 3-D model: show has those, even when the "
          "person says 'draw'; NOT for a table: table): a chart, a plot, a map with something marked. Python 3 with matplotlib/numpy/PIL, no network. "
          "`from rcanvas import *` gives: w = world() (w.room.name, w.room.polygon [[x,y],...]; w.robot.x/.y/"
          ".heading (rad, the direction it faces)/.radius; w.doors [d.name, d.a, d.b (ends [x,y]), d.p_open]; "
          "w.objects [o.name, o.type, o.x, o.y, o.yaw, o.width, o.depth, o.height, o.dist]; w.target.x/.y; any of "
          "these may be None), image('zed'|'ricoh') -> RGB numpy array or None, fig = figure(), ax = "
          "layout_axes(fig) (equal-aspect map axes), draw_layout(ax, w) (room+doors+objects+robot+target), "
          "draw_room/draw_doors/draw_objects/draw_robot/draw_target(ax, w); g = graph() (the working-memory graph: "
          "g.nodes [n.id, n.name, n.type, n.agent], g.edges [e.src, e.dst, e.type] by node id), draw_graph(focus=None) "
          "-> a PIL image of it. Metres, room frame. The script MUST end "
          "with save(fig) (or save_image(img)). If it fails you get the error back: fix it and call draw again.",
          {"script": {"type": "string", "description": "the Python source"},
           "title": {"type": "string", "description": "a short caption, e.g. 'distance to each object'"},
           "live": {"type": "boolean", "description": "true = keep redrawing it with fresh data; false "
                                                      "(default) = draw once"}}, ["script", "title"]),
        t("table", "Show the PERSON a table on the canvas beside the chat, from rows you already have: your previous "
          "answer, the STATE or a tool result (a list of agents, of objects and distances, of what you can do). No "
          "script; a table is never a draw. Short rows, as many as the content has.",
          {"title": {"type": "string", "description": "a short caption"},
           "header": {"type": "array", "items": {"type": "string"}, "description": "the column names"},
           "rows": {"type": "array", "items": {"type": "array", "items": {"type": "string"}},
                    "description": "one list of cells per row, as many cells as the header"}}, ["title", "rows"]),
        t("hide_canvas", "Close the drawing canvas beside the chat.", {}, []),
        t("look", "Look through a camera now: the current picture is shown to YOU (not to the person) in the next message, so you "
        "can answer about it. To show a picture to the person use show or draw. Read-only.",
          {"camera": {"type": "string", "enum": ["front", "360"],
                      "description": "'front' = the camera facing ahead (default); '360' = the all-round panorama"}}, []),
    ]


def system():
    desc = (ROOT / "etc/system_prompt.md").read_text().strip()
    return desc + "\n" + interface_rules()


def state(last_move="none", extra=""):
    if extra.startswith("time:"):   # a whole STATE of its own (EMPTY_STATE)
        return "STATE:\n" + extra + f"last move/turn sent from this chat: {last_move}\n"
    expl = {"move 0.30": "move 0.30 (move forward 0.30 m)", "move -0.30": "move -0.30 (move backward 0.30 m)"}
    base = STATE.replace("last command sent from this chat: none\n", "") if "last command sent" in extra else STATE
    for key in ("robot motion now", "controller target now"):   # the case's own line replaces the default one
        if key in extra:
            base = "\n".join(l for l in base.split("\n") if not l.startswith(key))
    return "STATE:\n" + base + extra + f"last move/turn sent from this chat: {expl.get(last_move, last_move)}\n"


def ask(messages):
    body = {"model": MODEL, "messages": messages, "tools": tools(), "stream": False, "think": False,
            "keep_alive": "30m", "options": {"temperature": 0.2, "num_ctx": 16384, "num_predict": NUM_PREDICT}}
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
            {"role": "tool", "tool_name": "robot_status", "content": f"The command '{cmd}' ended: completed. done: {cmd}"}]


def draw_script_runs(a):
    """The model's draw(script=...) actually draws: run it through canvas/bootstrap.py (as chat_agent does) against
    test_canvas_stock.py's synthetic snapshot."""
    import subprocess
    import tempfile
    sys.path.insert(0, str(ROOT / "tests"))
    from test_canvas_stock import SNAPSHOT
    script = a.get("script") or ""
    if "save" not in script:
        return False
    with tempfile.TemporaryDirectory() as d:
        Path(d, "snapshot.json").write_text(json.dumps(SNAPSHOT))
        Path(d, "script.py").write_text(script)
        p = subprocess.run([sys.executable, "-I", str(ROOT / "canvas/bootstrap.py"), str(Path(d, "script.py"))],
                           cwd=d, capture_output=True, text=True, timeout=30)
        if p.returncode != 0:
            print("      script failed:", p.stderr.strip().splitlines()[-1:] if p.stderr else p.returncode)
        return p.returncode == 0 and Path(d, "out.png").exists()


def draws_table(n, a, must=()):
    """The table tool with rows holding every word in `must`. A draw() script that makes its own table does not count:
    its font is whatever the model wrote, too big for the canvas (live 2026-10-19); the table tool fits it."""
    rows = a.get("rows") if n == "table" else None
    return isinstance(rows, list) and len(rows) > 0 and all(w in json.dumps(rows) for w in must)


def lists_agents(c):
    """A text answer that names the alive agents from the STATE's line (all ten, not a made-up few)."""
    names = ["chair_concept", "controller", "door_concept", "ltsm_agent", "residual_concept", "retina",
             "robot_concept", "room_concept", "table_concept"]
    return all(n in c for n in names)


WHAT_I_CAN_DO = [{"role": "user", "content": "tell me what you can do"},
                 {"role": "assistant", "content": "I can explore the current room, recognize doors and objects, cross "
                  "doors, move short distances, and approach objects. I cannot fetch things, open doors manually, or "
                  "follow people. I can also show camera views, 3-D models, or room layouts if requested."}]

# live 2026-10-19, as chat_session keeps it: the call and its result, not the follow-up said after it. With the
# follow-up ("I am approaching table_7 … I will stop in front of it") in the history, the next approach got that
# sentence, for table_4, and no call (0/5 replayed); without it 5/5 called approach_object.
APPROACHED = [{"role": "user", "content": "approach the table in front"},
              {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "approach_object",
                                                                              "arguments": {"object": "table_1"}}}]},
              {"role": "tool", "tool_name": "approach_object", "content": "OK: mission 'approach table_1' sent: the "
               "controller drives to a spot 0.75 m from its centre, facing it."},
              {"role": "tool", "tool_name": "robot_status", "content": "The command 'approach table_1' ended: completed."},
              {"role": "user", "content": "advance 0.5 meters"},
              {"role": "assistant", "content": "Moving forward 0.50 m."}]

AGENTS_LISTED = [{"role": "user", "content": "make a list with all the active agents"},
                 {"role": "assistant", "content": "The active agents are: chair_concept, chat_agent, controller, "
                  "door_concept, episodic_agent, residual_concept, retina, robot_concept, room_concept, and table_concept."}]

CASES = [
    # the side canvas (2026-10-18): the person's pictures go through draw(); stock first, a script only when none fits
    ("show where the doors are", [], "none", "can you draw me a map of the room with the doors?",
     lambda n, a, c: (n == "show" and a.get("name") == "layout") or (n == "draw" and draw_script_runs(a))),
    ("show the front camera", [], "none", "put the front camera on the screen for me",
     lambda n, a, c: n == "show" and a.get("name") == "camera" and str(a.get("args", "zed")) in ("zed", "front", "")),
    ("draw the graph", [], "none", "can you draw the graph of your memory for me?",
     lambda n, a, c: n == "show" and a.get("name") == "graph"),
    ("graph around a door", [], "none", "show me the part of the graph around door_1",
     lambda n, a, c: n == "show" and a.get("name") == "graph" and "door_1" in str(a.get("args", ""))),
    ("3d model of the chair", [], "none", "let me see what the chair looks like in 3D",
     lambda n, a, c: n == "show" and a.get("name") == "model" and "chair" in str(a.get("args", ""))),
    ("what do you look like", [], "none", "what do you look like? show me",
     lambda n, a, c: n == "show" and a.get("name") == "model" and str(a.get("args", "robot")) in ("robot", "Shadow", "")),
    ("lidar points", [], "none", "let me see the lidar points",
     lambda n, a, c: n == "show" and a.get("name") == "lidar"),
    ("bar chart of distances", [], "none", "plot a bar chart of how far each object is from you",
     lambda n, a, c: n == "draw" and draw_script_runs(a)),
    # live 2026-10-19, fleet down: "show what you can do in a table" got "I cannot show anything on the canvas because I
    # am not in a known room" three times. A table of words needs no room: draw it.
    ("abilities table, fleet down", WHAT_I_CAN_DO, "none", "show all you can do in a table",
     lambda n, a, c: draws_table(n, a), EMPTY_STATE),
    ("abilities table on canvas", WHAT_I_CAN_DO, "none", "show it on a table on the canvas",
     lambda n, a, c: draws_table(n, a), EMPTY_STATE),
    # live 2026-10-19: "show the list in the canvas" after the agents list showed a stock table of abilities instead
    ("agents list on canvas", AGENTS_LISTED, "none", "show the list in the canvas",
     lambda n, a, c: draws_table(n, a, ("retina", "controller", "room_concept"))),
    ("table of objects", [], "none", "make a table of the objects you see and how far they are",
     lambda n, a, c: draws_table(n, a, ("chair_1", "table_1", "bottle_1"))),
    ("abilities table, room known", WHAT_I_CAN_DO, "none", "show what you can do in a table",
     lambda n, a, c: draws_table(n, a)),
    # live 2026-10-19: "draw a table with your abilities" went to draw, a script with font size 10 that overflowed
    ("draw a table of abilities", WHAT_I_CAN_DO, "none", "draw a table with your abilities",
     lambda n, a, c: draws_table(n, a)),
    ("approach again, by a side", APPROACHED, "none", "approach the chair by the side the robot is on",
     lambda n, a, c: n == "approach_object" and "chair" in str(a.get("object", ""))),
    ("approach again", APPROACHED, "none", "now approach the chair",
     lambda n, a, c: n == "approach_object" and "chair" in str(a.get("object", ""))),
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
    # 2026-10-18: the object proposes its sides (aff_<object>_approach options, STATE "approach sides"); the model passes
    # the person's words as `side` and the chat resolves them (approach.h resolve_side).
    ("approach the chair from the right", [], "none", "approach the chair from the right",
     lambda n, a, c: n == "approach_object" and "chair" in str(a.get("object", "")).lower()
     and "right" in str(a.get("side", "")).lower()),
    ("go to chair 1 from the front", [], "none", "go to chair 1 from the front",
     lambda n, a, c: n == "approach_object" and "chair" in str(a.get("object", "")).lower()
     and "front" in str(a.get("side", "")).lower()),
    ("acércate a la silla por la izquierda (es)", [], "none", "acércate a la silla por la izquierda",
     lambda n, a, c: n == "approach_object" and re.search("chair|silla", str(a.get("object", "")).lower()) is not None
     and re.search("left|izquierda", str(a.get("side", "")).lower()) is not None),
    ("approach the chair from behind (not offered: the tool says so)", [], "none", "approach the chair from behind",
     lambda n, a, c: n == "approach_object" and "chair" in str(a.get("object", "")).lower()),
    # no side asked: none, or the object's DEFAULT (the first listed: chair_1's front) — the same standpoint. The model
    # names the first listed side 3/3 even when told to pass one only when the person names it; harmless, so allowed.
    ("approach chair 1 (no side asked)", [], "none", "approach chair 1",
     lambda n, a, c: n == "approach_object" and str(a.get("side", "")).strip().lower() in ("", "front")),
    # system questions: the STATE's agents/cameras lines answer the gist, the tools the details.
    ("how many agents", [], "none", "how many agents are running?",
     lambda n, a, c: n == "agents" or (n is None and re.search(r"\b(10|ten)\b", c.lower()) is not None)),
    ("cuántos agentes (es)", [], "none", "¿cuántos agentes están funcionando?",
     lambda n, a, c: n == "agents" or (n is None and re.search(r"\b(10|diez)\b", c.lower()) is not None)),
    # live 2026-10-18: "agents ... in the WM" was answered as "I have no access to agents" (no tool call). The WM IS
    # the graph, so graph_summary is a fair pick: its result carries the agents census too (system_facts.h).
    ("agents alive in the WM", [], "none", "how many agents are alive in the WM?",
     lambda n, a, c: n in ("agents", "graph_summary") or (n is None and (lists_agents(c) or re.search(r"\b(10|ten|diez)\b", c.lower()) is not None))),
    ("agents alive", [], "none", "how many agents are alive?",
     lambda n, a, c: n in ("agents", "graph_summary") or (n is None and (lists_agents(c) or re.search(r"\b(10|ten|diez)\b", c.lower()) is not None))),
    ("agents in working memory", [], "none", "which agents are in the working memory?",
     lambda n, a, c: n in ("agents", "graph_summary") or (n is None and (lists_agents(c) or re.search(r"\b(10|ten|diez)\b", c.lower()) is not None))),
    # paraphrases: at temperature 0.2 a phrasing gets the same tool every repeat, so the spread is across phrasings.
    ("agents in the WM", [], "none", "how many agents are in the WM?",
     lambda n, a, c: n in ("agents", "graph_summary") or (n is None and (lists_agents(c) or re.search(r"\b(10|ten|diez)\b", c.lower()) is not None))),
    # agent_control (2026-10-19): the person starts / stops / restarts agents from the chat; questions stay on agents
    ("restart the chair agent", [], "none", "restart the chair agent",
     lambda n, a, c: n == "agent_control" and a.get("action") == "restart" and "chair" in str(a.get("agent", ""))),
    ("stop retina", [], "none", "stop retina",
     lambda n, a, c: n == "agent_control" and a.get("action") == "stop" and "retina" in str(a.get("agent", ""))),
    ("arranca el agente de las mesas (es)", [], "none", "arranca el agente de las mesas",
     lambda n, a, c: n == "agent_control" and a.get("action") == "start" and "table" in str(a.get("agent", ""))),
    ("is the chair agent running (no control)", [], "none", "is the chair agent running?",
     lambda n, a, c: n != "agent_control"),
    # live 2026-10-19: "can you start agents?" got "I cannot start agents directly", and "start refrigerator_concept"
    # (then missing from the list) got the same excuse without a call
    ("can you start agents (yes)", [], "none", "can you start agents?",
     lambda n, a, c: n is None and re.search(r"\b(cannot|can't|can not|unable)\b", c.lower()) is None
     and "start" in c.lower()),
    ("start refrigerator_concept", [], "none", "start refrigerator_concept",
     lambda n, a, c: n == "agent_control" and a.get("action") == "start" and "refrigerator" in str(a.get("agent", ""))),
    ("start an agent not in the list", [], "none", "start kitchen_metaconcept",
     lambda n, a, c: n == "agent_control" and "kitchen" in str(a.get("agent", ""))),
    # a plain "yes" to agent_control's question never reaches the model: the session carries it out (fast path,
    # chat_session.cpp 1a) — the model answered it by claiming "stopped" without calling the tool (eval 2026-10-19)
    ("no to stopping the controller",
     [{"role": "user", "content": "stop the controller agent"},
      {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "agent_control",
                                                                         "arguments": {"action": "stop", "agent": "controller"}}}]},
      {"role": "tool", "tool_name": "agent_control", "content": "NOT DONE: ask the person \"Stop controller? The robot "
       "stops moving until it is back.\" and call agent_control again with the same arguments only if they answer yes."},
      {"role": "assistant", "content": "Stop controller? The robot stops moving until it is back."}], "none",
     "no, leave it",
     lambda n, a, c: n != "agent_control"),
    ("all agents up", [], "none", "are all the agents up?",
     lambda n, a, c: n == "agents" or (n is None and "door_concept" in c and re.search(r"degraded", c.lower()) is not None)),
    ("list running agents", [], "none", "list the running agents",
     lambda n, a, c: n in ("agents", "graph_summary") or (n is None and (lists_agents(c) or re.search(r"\b(10|ten|diez)\b", c.lower()) is not None))),
    ("is retina running", [], "none", "is retina running?",
     lambda n, a, c: n == "agents" or (n == "inspect" and "retina" in str(a.get("node", "")).lower())
     or (n is None and re.search(r"\byes\b|is running|is alive", c.lower()) is not None)),
    ("which programs run", [], "none", "which programs are running right now?",
     lambda n, a, c: n in ("agents", "graph_summary") or (n is None and (lists_agents(c) or re.search(r"\b(10|ten|diez)\b", c.lower()) is not None))),
    ("controller alive", [], "none", "is the controller alive?",
     lambda n, a, c: n == "agents" or (n == "inspect" and "controller" in str(a.get("node", "")).lower())
     or (n is None and re.search(r"\byes\b|is running|is alive", c.lower()) is not None)),
    ("agentes vivos WM (es)", [], "none", "¿cuántos agentes hay vivos en la WM?",
     lambda n, a, c: n in ("agents", "graph_summary") or (n is None and (lists_agents(c) or re.search(r"\b(10|ten|diez)\b", c.lower()) is not None))),
    ("room_concept cpu", [], "none", "what is the CPU load of room_concept?",
     lambda n, a, c: n == "agents" or (n == "inspect" and "room_concept" in str(a.get("node", "")).lower())),
    # live 2026-10-18: answered with the four EXAMPLE names of the old agents() description, no call.
    ("what agents are alive", [], "none", "what agents are alive?",
     lambda n, a, c: n == "agents" or (n is None and lists_agents(c))),
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
