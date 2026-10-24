#!/usr/bin/env python3
"""Live check (needs ollama): does the model WRITE a runnable, correct mission from a request?

The planner request chat_agent will send: the three skills (etc/skills/missions.md + doors.md + exploring.md, front
matter stripped) as the system prompt, the same synthetic STATE as check_llm_tools.py, then the person's request.
No tools, temperature 0, think false, num_predict from etc/config.toml. The reply is piped through
tests/mission_check (parse_planned: the DSL, or JSON if it starts with `{`; then validate) and the case's checklist is
run over the printed step lines. `ask:` is the right answer to an impossible/unclear request and a failure otherwise.

    python3 tests/check_llm_missions.py <model> [repeats=3] [--form text|json]
    OLLAMA=http://192.168.1.34:11434 python3 tests/check_llm_missions.py qwen3.8 3 --form text

--form json adds Ollama's `format` (a JSON schema mirroring common/mission/mission_json.h) and one system line that
asks for the same mission as JSON (the skills teach the text form).
"""
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_args = [a for a in sys.argv[1:]]
FORM = "text"
if "--form" in _args:
    i = _args.index("--form")
    FORM = _args[i + 1]
    del _args[i:i + 2]
MODEL = _args[0] if _args else "qwen3-vl:8b-instruct"
REPEATS = int(_args[1]) if len(_args) > 1 else 3
URL = os.environ.get("OLLAMA", "http://localhost:11434") + "/api/chat"

# reuse STATE / state() of check_llm_tools.py (exec the file up to its __main__ guard, as it is written for)
_tools_path = ROOT / "tests/check_llm_tools.py"
_src = _tools_path.read_text().split('if __name__ == "__main__":')[0]
_ns = {"__file__": str(_tools_path), "__name__": "check_llm_tools"}
_saved = sys.argv
sys.argv = ["check_llm_tools.py"]
exec(compile(_src, str(_tools_path), "exec"), _ns)
sys.argv = _saved
NUM_PREDICT = _ns["NUM_PREDICT"]
# These cases (names_room_2) assume NO remembered rooms: room_2 is a name the STATE does not hold. The tools eval's STATE
# gained a long-term-memory line (room_2, room_3) for known_rooms/route_to; drop it here so the premise stays true.
STATE_TEXT = "\n".join(l for l in _ns["state"]().split("\n") if not l.startswith("rooms in long-term memory"))


def skills():
    parts = []
    for n in ("missions", "doors", "exploring"):
        t = (ROOT / f"etc/skills/{n}.md").read_text()
        t = re.sub(r"\A(<!--.*?-->\s*)?---\n.*?\n---\n", "", t, flags=re.S)   # an optional leading comment, then the front matter
        parts.append(t.strip())
    return "\n\n".join(parts)


JSON_LINE = ('Output the mission as JSON instead of text, in this schema: {"v":1,"title":"…","steps":[{"id":"1","op":…, '
             'args…}]}. Step ids are "1","2",…; a step inside a body is "2.1". A reference is an object: '
             '{"node":"door_1"} · {"anchor":"start"} · {"var":"next"} · {"pose":{"room":…,"x":…,"y":…}}; a predicate is '
             '{"p":"explored","room":{"node":"room_1"},"ge":0.9}. on_fail is "skip" or {"retry":2}; bind is a string on cross. '
             'Example: {"v":1,"title":"look at the next room and return","steps":[{"id":"1","op":"anchor","name":"start"},'
             '{"id":"2","op":"cross","door":{"node":"door_1"},"bind":"next","on_fail":{"retry":2}},'
             '{"id":"3","op":"explore","room":{"var":"next"},"budget_s":60},{"id":"4","op":"report","text":"the objects here"},'
             '{"id":"5","op":"go_to","to":{"anchor":"start"}}]}. '
             'If the request is unclear or impossible, output {"v":1,"ask":"<question>","steps":[]}.')

SCHEMA = {
    "type": "object",
    "properties": {
        "v": {"type": "integer"}, "title": {"type": "string"}, "ask": {"type": "string"},
        "steps": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "op": {"type": "string", "enum": ["seq", "repeat", "cross", "open", "go_to", "approach", "explore",
                                                  "wait", "say", "report", "anchor", "move", "stop"]},
                "door": {"type": "object"}, "to": {"type": "object"}, "room": {"type": "object"},
                "object": {"type": "object"}, "until": {"type": "object"},
                "expect": {"type": "string"}, "side": {"type": "string"}, "text": {"type": "string"},
                "name": {"type": "string"}, "bind": {"type": "string"},
                "budget_s": {"type": "number"}, "s": {"type": "number"}, "distance_m": {"type": "number"},
                "turn_deg": {"type": "number"}, "standoff_m": {"type": "number"}, "times": {"type": "integer"},
                "on_fail": {"anyOf": [{"type": "string"}, {"type": "object"}]},
                "steps": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["id", "op"]}},
    },
    "required": ["v", "steps"],
}


# ── the checker binary (parse_planned), compiled once ─────────────────────────────────────────────────────────────
def build_checker():
    out = Path(os.environ.get("MISSION_CHECK_BIN", "/tmp/mission_check_" + str(os.getuid())))
    src = ROOT / "tests/mission_check.cpp"
    deps = [src, *(ROOT / "src").glob("mission_plan.h"), *(ROOT.parent / "common/mission").glob("*.h")]
    if not out.exists() or out.stat().st_mtime < max(p.stat().st_mtime for p in deps):
        cfl = subprocess.run(["pkg-config", "--cflags", "Qt6Core"], capture_output=True, text=True).stdout.strip().replace("-I", "-isystem ")
        lib = subprocess.run(["pkg-config", "--libs", "Qt6Core"], capture_output=True, text=True).stdout.strip()
        cmd = f"g++ -std=c++23 -O1 -fPIC -Wno-deprecated-literal-operator -o {out} {src} {cfl} {lib}"
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if r.returncode:
            sys.exit("mission_check build failed:\n" + r.stderr[-2000:])
    return str(out)


CHECKER = build_checker()


def check(reply):
    r = subprocess.run([CHECKER], input=reply, capture_output=True, text=True, timeout=20)
    lines = r.stdout.strip().split("\n")
    if lines[0] != "OK":
        return None, lines[0]
    steps = []
    for ln in lines[1:]:
        depth, _, ln = ln.partition(" ")   # mission_check prints the nesting depth first
        op, _, rest = ln.partition(" ")
        steps.append((op, {"_d": depth, **dict(re.findall(r'(\w+)=("[^"]*"|\S+)', rest))}))
    return steps, ""


FENCED = [0]
FENCED_PROSE = [0]   # fenced AND text outside the fence
UNFENCED_PROSE = [0]   # no fence, but the reply does not start with mission / ask: / {


def unfence(reply):
    """models copy the skill's ``` examples: take the block's inside (chat_agent must do the same; counted, not hidden)."""
    m = re.search(r"```[a-z]*\n(.*?)(```|\Z)", reply, re.S)
    if m:
        FENCED[0] += 1
        if (reply[:m.start()] + reply[m.end():]).strip():
            FENCED_PROSE[0] += 1
        return m.group(1).strip()
    if not re.match(r"\s*(mission\b|ask:|\{)", reply, re.I):
        UNFENCED_PROSE[0] += 1
    return reply.strip()


def is_ask(reply):
    t = reply.strip()
    if t.lower().startswith("ask:"):
        return True
    if t.startswith("{"):
        try:
            return bool(json.loads(t).get("ask"))
        except Exception:
            return False
    return False


# ── checklist helpers over the step lines [(op, {key: value})] ──────────────────────────────────────────────────────
def of(L, op):
    return [(i, a) for i, (o, a) in enumerate(L) if o == op]


def crosses(L, door=None):
    return [(i, a) for i, a in of(L, "cross") if door is None or a.get("door") == f"node:{door}"]


def bound(L):
    return [(i, a["bind"]) for i, a in crosses(L) if "bind" in a]


def explores_bound(L):
    b = {n for _, n in bound(L)}
    return [(i, a) for i, a in of(L, "explore") if a.get("room", "").startswith("var:") and a["room"][4:] in b]


def returns_after(L, k):
    """a way back after step k: any later cross, or a later go_to anchor:."""
    return any(o == "cross" or (o == "go_to" and a.get("to", "").startswith("anchor:"))
               for o, a in L[k + 1:])


def visit_room(L):
    """cross with bind, then explore $bind, then a report, then a way back."""
    b, e = bound(L), explores_bound(L)
    if not b or not e:
        return False
    rep = [i for i, _ in of(L, "report") if i > e[0][0]]
    return bool(rep) and returns_after(L, rep[0])


def names_room_2(L):
    """room_2 is not in the STATE (no other rooms, none remembered): naming it in any form is an invented name."""
    return any(re.search(r"(node|pose):room_2\b|passage:[^ ]*@room_2\b", v) for _, a in L for v in a.values())


def visit_named_room(L):
    """ask:, or cross with bind= then explore $var, then a way back; never `explore room_2`."""
    ex = explores_bound(L)
    return not names_room_2(L) and bool(ex) and returns_after(L, ex[0][0])


def wait_in_new_room(L):
    """a cross with bind=, THEN a 30 s wait after it, then a way back; never room_2 by name."""
    b = bound(L)
    w = [i for i, a in of(L, "wait") if num(a, "s") == 30 and b and i > b[0][0]]
    return not names_room_2(L) and bool(w) and returns_after(L, w[0])


def twice(L):
    """exactly two crossings of door_1: two top-level crosses, or `repeat 2` whose whole body is ONE cross of door_1."""
    if len(L) == 2 and all(o == "cross" and a.get("door") == "node:door_1" and a["_d"] == "0" for o, a in L):
        return True
    for i, (o, a) in enumerate(L):
        if o == "repeat" and num(a, "times") == 2:
            d = int(a["_d"])
            body = []
            for o2, a2 in L[i + 1:]:
                if int(a2["_d"]) <= d:
                    break
                body.append((o2, a2))
            return len(body) == 1 and body[0][0] == "cross" and body[0][1].get("door") == "node:door_1" and len(L) == 2
    return False


def single(L, op, **want):
    return len(L) == 1 and L[0][0] == op and all(L[0][1].get(k) == v for k, v in want.items())


def num(a, k):
    try:
        return float(a[k])
    except Exception:
        return None


CASES = [
    # (label, request, checker(L) -> bool, expect_ask)
    ("spec 5.1 (EN)", "Approach door_1, cross to the next room, tell me what objects are there, and come back to room_1.",
     lambda L: bool(crosses(L, "door_1")) and visit_room(L) and any(a.get("door") == "node:door_1" for _, a in crosses(L)[:1]), False),
    ("paraphrase EN", "Go through door_1, look around the room on the other side, report what is in it, then return here.",
     lambda L: bool(crosses(L, "door_1")) and visit_room(L), False),
    ("paraphrase EN 2", "Please check what objects are in the room behind door_1 and then get back to where you started.",
     lambda L: bool(crosses(L, "door_1")) and visit_room(L), False),
    ("paraphrase ES", "Aborda door_1, pasa a la siguiente habitación, dime qué objetos hay y vuelve a room_1.",
     lambda L: bool(crosses(L, "door_1")) and visit_room(L), False),
    ("paraphrase ES 2", "Cruza door_1, mira qué hay en la otra sala y regresa.",
     lambda L: bool(crosses(L, "door_1")) and visit_room(L), False),
    ("explore room_2 and back (EN)", "Explore room_2 and come back.", visit_named_room, "either"),
    ("explore room_2 and back (ES)", "Explora room_2 y vuelve.", visit_named_room, "either"),
    ("cross door_1 twice", "Cross door_1 twice.",
     twice, False),
    ("go to room_2, wait 30s, back", "Go to room_2, wait 30 seconds, and come back.", wait_in_new_room, "either"),
    ("cross door_0 and report", "Cross door_0 and tell me what you see on the other side.",
     lambda L: bool(crosses(L, "door_0")) and any(i > crosses(L, "door_0")[0][0] for i, _ in of(L, "report")), False),
    ("report in current room", "Tell me what is in this room.",
     lambda L: bool(of(L, "report")) and not crosses(L), False),
    ("nonexistent door (EN)", "Cross door_7 and tell me what is there.", None, True),
    ("nonexistent door (ES)", "Pasa por door_9 y explora la sala.", None, True),
    ("unclear request", "Do something useful.", None, True),
    ("single: cross", "Cross door_0.", lambda L: len(L) == 1 and bool(crosses(L, "door_0")), False),
    ("single: open", "Open door_1.", lambda L: single(L, "open", door="node:door_1"), False),
    ("single: approach", "Approach the table.", lambda L: single(L, "approach", object="node:table_1"), False),
    ("single: turn around", "Turn around.", lambda L: len(L) == 1 and L[0][0] == "move" and abs(num(L[0][1], "turn_deg") or 0) == 180, False),
    ("single: wait", "Wait 10 seconds.", lambda L: len(L) == 1 and L[0][0] == "wait" and num(L[0][1], "s") == 10, False),
    ("single: explore here", "Explore this room for one minute.",
     lambda L: len(L) == 1 and L[0][0] == "explore" and L[0][1].get("room") == "node:room_1" and num(L[0][1], "budget_s") == 60, False),
    ("chair then table", "Go to a chair and then to the table.",
     lambda L: [a.get("object") for o, a in L if o == "approach"][-1:] == ["node:table_1"]
     and [a.get("object", "") for o, a in L if o == "approach"][:1] in (["node:chair_1"], ["node:chair_2"]), False),
    ("move then turn", "Move forward 2 meters and then turn 90 degrees.",
     lambda L: any(num(a, "distance_m") == 2 for _, a in of(L, "move")) and any(abs(num(a, "turn_deg") or 0) == 90 for _, a in of(L, "move")), False),
    ("explore until explored", "Explore this room until it is 90% explored, for at most 5 minutes.",
     lambda L: len(L) >= 1 and L[0][0] == "explore" and "explored" in L[0][1].get("until", "") and num(L[0][1], "budget_s") == 300, False),
    ("cross and come back to start", "Cross door_0 and then come back to where you started.",
     lambda L: bool(crosses(L, "door_0")) and returns_after(L, crosses(L, "door_0")[0][0]), False),
]


def ask_model(system, user):
    msgs = [{"role": "system", "content": system}, {"role": "system", "content": STATE_TEXT}, {"role": "user", "content": user}]
    body = {"model": MODEL, "messages": msgs, "stream": False, "think": False, "keep_alive": "30m",
            "options": {"temperature": 0, "num_ctx": 16384, "num_predict": NUM_PREDICT}}
    if FORM == "json":
        body["format"] = SCHEMA
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read()).get("message", {}).get("content", "")
    except Exception as e:
        return f"<HTTP error: {e}>"


def main():
    system = skills()
    if FORM == "json":
        system += "\n\n" + JSON_LINE
    results, lat = [], []
    for label, request, chk, want_ask in CASES:
        for rep in range(REPEATS):
            t0 = time.monotonic()
            reply = unfence(ask_model(system, request))
            lat.append(time.monotonic() - t0)
            if want_ask == "either" and is_ask(reply):
                good, why = True, ""
            elif want_ask is True:
                good, why = is_ask(reply), "no ask: " + reply.strip().replace("\n", " | ")[:120]
            elif is_ask(reply):
                good, why = False, "asked instead: " + reply.strip()[:100]
            else:
                steps, err = check(reply)
                if steps is None:
                    good, why = False, f"mission_check: {err} :: " + reply.strip().replace("\n", " | ")[:140]
                else:
                    good = bool(chk(steps))
                    why = "checklist: " + "; ".join(o + (" " + " ".join(f"{k}={v}" for k, v in a.items())) for o, a in steps)[:200]
            results.append(good)
            print(f"{'PASS' if good else 'FAIL'}  {label:<30} #{rep + 1}" + ("" if good else f"  -> {why}"), flush=True)
    ls = sorted(lat)
    print(f"\nmodel {MODEL} form {FORM}; system {len(system)} chars; STATE {len(STATE_TEXT)} chars")
    print(f"latency: median {statistics.median(ls):.2f} s, mean {statistics.mean(ls):.2f} s, max {ls[-1]:.2f} s")
    print(f"replies wrapped in a ``` fence: {FENCED[0]}/{len(results)} (of those, with prose outside it: {FENCED_PROSE[0]}); "
          f"unfenced with prose first: {UNFENCED_PROSE[0]}/{len(results)}")
    print(f"{sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
