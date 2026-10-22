#!/usr/bin/env python3
"""Offline check of scripts/em_transitions.py on a hand-written archive: python3 tests/em_transitions_test.py"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parent / "scripts" / "em_transitions.py"


def rec(id, level, kind, target, t0, t1, outcome, pre, post, scene_pre, scene_post, room="room_1", room_after="room_1",
        parent="20261019-1", events=(), alts=""):
    return {"id": id, "parent": parent, "level": level, "author": "self", "kind": kind, "target": target,
            "target_ltsm": "", "t0": t0, "t1": t1, "room": room, "room_after": room_after, "outcome": outcome,
            "pre": pre, "post": post, "events": list(events), "alternatives_json": alts, "settled": True,
            "scene_pre": scene_pre, "scene_post": scene_post}


def door(name, p):
    return [{"node": name, "type": "object", "values": {"door_open_prob": p, "x": 1.0, "cov_xx": 0.1}}]


def main():
    mission = {"id": "20261019-1", "parent": "", "level": "mission", "author": "self", "kind": "mission", "t0": 0,
               "t1": 9000, "events": []}
    # chair_7 flickers within the mission: 5 births ⇒ never an effect
    flicker = [{"dt_ms": 10 * i, "kind": "born", "node": "chair_7"} for i in range(5)]
    open_door = rec("20261019-2", "claim", "open", "door_5", 1000, 3000, "satisfied", door("door_5", 0.0),
                    door("door_5", 0.92),
                    {"things": ["door_5", "door_8", "room_1"], "offered": ["aff_door_5_open", "aff_door_8_open"]},
                    {"things": ["chair_7", "door_5", "door_8", "room_1"],
                     "offered": ["aff_door_5_cross", "aff_door_8_open"]},
                    events=flicker,
                    alts='[{"node_name":"aff_door_5_open","gain":0.3},{"node_name":"aff_door_8_open","gain":0.1}]')
    cross = rec("20261019-3", "step", "cross", "door_5", 3500, 8000, "satisfied", door("door_5", 0.92),
                door("door_5", 0.95),
                {"things": ["door_5", "room_1"], "offered": ["aff_door_5_cross"]},
                {"things": ["door_5", "room_1", "room_2"], "offered": ["afford_room"]}, room_after="room_2")
    lost = rec("20261019-4", "claim", "open", "door_8", 8100, 8200, "lost", [], [], {"things": [], "offered": []},
               {"things": [], "offered": []})
    old = rec("20261019-5", "claim", "open", "door_8", 8300, 8400, "satisfied", [], [], None, None)
    del old["scene_pre"], old["scene_post"]
    with tempfile.TemporaryDirectory() as d:
        arch = Path(d) / "20261019.jsonl"
        with arch.open("w") as f:
            for r in (mission, open_door, cross, lost, old):
                f.write(json.dumps(r) + "\n")
        out = Path(d) / "t.jsonl"
        p = subprocess.run([sys.executable, str(SCRIPT), d, "-o", str(out), "--table"], capture_output=True, text=True)
        assert p.returncode == 0, p.stderr
        rows = [json.loads(l) for l in out.open()]
        assert "skipped 1 EM stop; skipped 1 no scene" in p.stderr, p.stderr
        assert [r["id"] for r in rows] == ["20261019-2", "20261019-3"], rows

        o = rows[0]
        e = o["effects"]
        assert o["action"] == "open" and o["target_type"] == "door" and o["dur_s"] == 2.0
        assert e["beliefs"] == {"door_open_prob": 0.92}, e
        assert e["became_possible"] == ["cross <target>"] and e["no_longer_possible"] == ["open <target>"], e
        assert e["appeared"] == [] and not e["room_changed"] and not e["target_gone"], e   # chair_7 flickers
        assert o["before"]["things"] == {"<door>": 1, "<room>": 1, "<target>": 1}, o["before"]
        assert o["before"]["offered"] == ["open <door>", "open <target>"], o["before"]
        assert o["before"]["target"] == {"door_open_prob": 0.0}          # beliefs only, no pose
        assert o["chosen_among"] == ["open <door>", "open <target>"]
        assert o["effects_instances"]["became_possible"] == ["aff_door_5_cross"]

        c = rows[1]["effects"]
        assert c["room_changed"] and c["appeared"] == ["<room>"] and c["beliefs"] == {}, c   # +0.03 is below MIN_CHANGE
        assert c["became_possible"] == ["explore the room"] and c["no_longer_possible"] == ["cross <target>"], c

        assert "open         door" in p.stdout and "satisfied 100%" in p.stdout and "+cross <target> 100%" in p.stdout, p.stdout
        assert "room changed 100%" in p.stdout, p.stdout
    print("em_transitions_test: ok")


if __name__ == "__main__":
    main()
