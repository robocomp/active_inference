#!/usr/bin/env -S python3 -I
"""test_door_costs_tool.py - tools/door_costs_from_logs.py (re-review N3, ruling): C_delete samples are re-births CROSSED later, the
tour samples are kept alongside --events, and a result with C_keep >= C_delete is REFUSED (exit 2, the old file untouched).
Run (from door_concept/):  python3 -I tests/test_door_costs_tool.py"""
import csv, json, pathlib, subprocess, sys, tempfile

TOOL = pathlib.Path(__file__).resolve().parent.parent / "tools" / "door_costs_from_logs.py"
HEADER = "kind,t_wall_ms,room,place,x,y,seconds,door\n"
fails = 0


def check(c, what):
    global fails
    print(("PASS " if c else "FAIL ") + what)
    fails += 0 if c else 1


def run(tmp, tours, events, out):
    args = [sys.executable, "-I", str(TOOL), "--tours", str(tours), "--out", str(out)]
    if events:
        args += ["--events"] + [str(e) for e in events]
    return subprocess.run(args, capture_output=True, text=True)


def rows(out):
    return list(csv.DictReader(open(out)))


with tempfile.TemporaryDirectory() as t:
    tmp = pathlib.Path(t)
    empty = tmp / "no_tours"; empty.mkdir()
    ev = tmp / "events.csv"
    ev.write_text(HEADER
                  + "".join(f"keep_episode,{1000 * i},room_1,door_1,nan,nan,10.0,door_1\n" for i in range(3))
                  + "".join(f"rebirth,{10000 + i},room_1,door_{i},1,1,50.0,door_{10 + i}\n" for i in range(3))
                  + "".join(f"crossed,{20000 + i},room_1,door_{10 + i},1,1,0,door_{10 + i}\n" for i in range(3))
                  + "rebirth,30000,room_1,door_7,1,1,0.3,door_20\n")          # a flicker: never crossed
    out = tmp / "costs.csv"
    r = run(tmp, empty, [ev], out)
    check(r.returncode == 0 and out.exists(), f"valid events: written (rc {r.returncode})")
    d = [x for x in rows(out) if x["kind"] == "delete_ordinary"]
    check(len(d) == 3 and all(float(x["seconds"]) == 50.0 and "crossed-later" in x["source"] for x in d),
          "C_delete samples = the 3 re-births crossed later; the 0.3 s flicker never crossed is not one")
    check(sum(x["kind"] == "keep" for x in rows(out)) == 3, "C_keep = the 3 keep episodes")

    # equal fallbacks (nothing at all): REFUSED, the old file left as it was
    before = out.read_bytes()
    r = run(tmp, empty, [], out)
    check(r.returncode == 2 and "REFUSED" in r.stderr and out.read_bytes() == before, "no samples (keep = delete = 30 s): refused, file untouched")
    # keep > delete: refused
    ev2 = tmp / "events2.csv"
    ev2.write_text(HEADER + "".join(f"keep_episode,{i},room_1,door_1,nan,nan,90.0,door_1\n" for i in range(3))
                   + "".join(f"rebirth,{100 + i},room_1,door_{i},1,1,20.0,door_{10 + i}\ncrossed,{200 + i},room_1,door_{10 + i},1,1,0,door_{10 + i}\n"
                             for i in range(3)))
    r = run(tmp, empty, [ev2], out)
    check(r.returncode == 2 and out.read_bytes() == before, "C_keep 90 s >= C_delete 20 s: refused, file untouched")
    # an old 7-column file confirms no C_delete sample
    ev3 = tmp / "events3.csv"
    ev3.write_text("kind,t_wall_ms,room,place,x,y,seconds\n" + "".join(f"rebirth,{i},room_1,door_{i},1,1,50.0\n" for i in range(3)))
    r = run(tmp, empty, [ev3], out)
    check(r.returncode == 2 and out.read_bytes() == before, "old file without the door column: no C_delete sample (fallbacks equal: refused)")

    # the tour samples are KEPT alongside --events: one fake tour with one failed passage (5 s, C_keep)
    tours = tmp / "tours"; tour = tours / "explore_tour10"; (tour / "logs").mkdir(parents=True)
    (tour / "exploration_drive.csv").write_text("t_ms,door\n1000,door_1\n6000,\n")
    (tour / "logs" / "controller.log").write_text("[explore] cross 'door_1' -> DOOR PASSAGE\n")
    (tour / "events.jsonl").write_text(json.dumps({"kind": "log", "msg": "nothing", "t": 0}) + "\n")
    out2 = tmp / "costs2.csv"
    r = run(tmp, tours, [ev], out2)
    srcs = [x["source"] for x in rows(out2) if x["kind"] == "keep"] if out2.exists() else []
    check(r.returncode == 0 and any("explore_tour10:passage-no-crossing door_1" in s for s in srcs) and len(srcs) == 4,
          "with --events the tour's C_keep sample is kept (4 = 1 tour + 3 events)")

print("ALL PASS" if fails == 0 else f"{fails} FAILED")
sys.exit(1 if fails else 0)
