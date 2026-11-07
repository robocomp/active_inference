#!/usr/bin/env -S python3 -I
"""door_costs_from_logs.py — Task 6: initial C_keep / C_delete(ordinary) from the recorded exploration tours.

  Keep           = duration of a door-passage attempt that did NOT end in a crossing.
  DeleteOrdinary = seconds from a door's existence REMOVE to its re-creation at the matched place (REACQUIRE row).

Sources (per tour dir explore_tour<N>/, all wall-clock-stamped except where said):
  exploration_drive.csv  t_ms of every "cross door_X" decision (a passage attempt starts there; the next row ends it)
  logs/controller.log    "[explore] cross 'door_X' -> DOOR PASSAGE" (which decisions really started a passage; no clock)
  events.jsonl           room_concept's log: "crossed door_X" (a crossing happened), "door_X left the graph" (the REMOVE,
                         t = steady ms) and "aperture resolved for door_X" (the re-born door's RT edge is back)
  door_events.csv        REMOVE / REACQUIRE / RELEASE rows (seq only, NO clock) + logs/door_concept.log (REMOVE id -> name)
NOT used: door_inputs.bin and door_decisions.bin (stale copies; the latter carries other door ids and no REMOVE rows).

Judgement calls (stated, not hidden):
  * "same place" is judged by door_concept's own REACQUIRE (within the identity radius, same hole, same name). A pair is dropped
    when a RELEASE row lies between its REMOVE and REACQUIRE: the room frame changed, so the place is not comparable.
  * The k-th "left the graph" of a name is the k-th REMOVE of that name (a tour is dropped when the counts differ).
  * A failed attempt needs a clock at both ends. The end is the NEXT drive decision; a failure on the last drive row ends only
    when the run was stopped, which is the experiment's end, not the agent's: it is skipped (listed, not invented).
  * Fewer than 3 samples of a kind => that kind falls back to the spec's reading (Keep = 30 s, DeleteOrdinary = 30 s),
    written as ONE row whose source says so.
  --events FILE [FILE...]   the agent's own etc/door_cost_events.csv (kind,t_wall_ms,room,place,x,y,seconds,door), written live by
                         door_concept, ADDED to the tour samples (never instead of them): keep_episode rows are C_keep samples (seconds).
                         ★A C_delete sample is a rebirth whose door (the `door` column, the reborn door's live name) is CROSSED later in the
                         same file (re-review N3, ruling): the crossing is the only in-data ground truth that the deleted place was a door.
                         A rebirth never crossed is a flicker or unconfirmed and is NOT a sample (listed). An old file without the `door`
                         column confirms nothing.
  --tours DIR             the tour root (default the crossing dir); --out FILE  the output (default etc/door_costs.csv).
Output: etc/door_costs.csv  (kind,seconds,source), medians and n printed. The agent never writes this file.
★REFUSAL (ruling "never equal costs", spec §3.6): if the resulting median C_keep >= median C_delete (equal fallbacks included), the tool
  writes NOTHING, leaves the old file as it was and exits 2. The agent's loader refuses such a file too (door_decision.h).
"""
import csv, json, re, statistics, sys, pathlib

TOURS = pathlib.Path("/home/pbustos/robocomp/components/active_inference/room_concept/estimation_scripts/crossing")
OUT = pathlib.Path(__file__).resolve().parent.parent / "etc" / "door_costs.csv"
FALLBACK_S = 30.0   # the spec's stated reading: the order of one failed approach
MIN_N = 3


def tour_events(d):
    cross, left, ap = [], {}, {}
    for line in open(d / "events.jsonl", errors="replace"):
        j = json.loads(line)
        if j.get("kind") != "log":
            continue
        m, t = j["msg"], j["t"]
        if "begin_room_estimation" in m and (c := re.search(r"crossed (door_\d+)", m)):
            cross.append(t)
        elif (c := re.search(r"\[room\]\[doors\] (door_\d+) left the graph", m)):
            left.setdefault(c.group(1), []).append(t)
        elif (c := re.search(r"\[room\]\[doors\] aperture resolved for (door_\d+)", m)):
            ap.setdefault(c.group(1), []).append(t)
    return cross, left, ap


def keep_samples(d, tour, notes):
    out = []
    try:
        rows = list(csv.DictReader(open(d / "exploration_drive.csv")))
    except OSError:
        return out
    cross, _, _ = tour_events(d)
    started = {}
    for x in open(d / "logs/controller.log", errors="replace"):
        if (c := re.search(r"\[explore\] cross '(door_\d+)' -> DOOR PASSAGE", x)):
            started[c.group(1)] = started.get(c.group(1), 0) + 1
    for i, r in enumerate(rows):
        if not r["door"] or started.get(r["door"], 0) == 0:
            continue
        started[r["door"]] -= 1
        t0 = int(r["t_ms"])
        t1 = int(rows[i + 1]["t_ms"]) if i + 1 < len(rows) else None
        crossed = any(t0 <= c and (t1 is None or c < t1) for c in cross)
        if crossed:
            continue
        if t1 is None:
            notes.append(f"{tour}: passage to {r['door']} failed on the last drive row, no agent-side end: skipped")
            continue
        out.append(((t1 - t0) / 1000.0, f"{tour}:passage-no-crossing {r['door']}"))
    return out


def delete_samples(d, tour, notes):
    out = []
    try:
        ev = list(csv.DictReader(open(d / "door_events.csv")))
    except OSError:
        return out
    ev = [e for e in ev if e["event"] in ("REMOVE", "REACQUIRE", "RELEASE")]
    if not any(e["event"] == "REACQUIRE" for e in ev):
        return out
    _, left, ap = tour_events(d)
    log = open(d / "logs/door_concept.log", errors="replace").read().splitlines()
    name_of = {}
    for i, x in enumerate(log):
        if (c := re.search(r"\[existence\] REMOVE id=(\d+)", x)):
            for y in log[i + 1:i + 6]:
                if (n := re.search(r"\[identity\] (door_\d+) removed for absence|\[affordance\] removed node for '(door_\d+)'", y)):
                    name_of[c.group(1)] = n.group(1) or n.group(2)
                    break
    removes = [e for e in ev if e["event"] == "REMOVE"]
    if sum(len(v) for v in left.values()) != len(removes) or any(e["id"] not in name_of for e in removes):
        notes.append(f"{tour}: REMOVE rows ({len(removes)}) do not match the clocked 'left the graph' lines: tour skipped")
        return out
    kth, t_of = {}, {}
    for e in removes:
        n = name_of[e["id"]]
        k = kth.get(n, 0); kth[n] = k + 1
        t_of[e["seq"]] = (n, left[n][k])
    for j, e in enumerate(ev):
        if e["event"] != "REACQUIRE":
            continue
        name = e["note"].strip()
        prev = [p for p in ev[:j] if p["event"] == "REMOVE" and t_of[p["seq"]][0] == name]
        if not prev:
            continue
        rm = prev[-1]
        between = ev[ev.index(rm) + 1:j]
        if any(b["event"] == "RELEASE" for b in between):
            notes.append(f"{tour}: {name} REMOVE seq {rm['seq']} -> REACQUIRE straddles a RELEASE: skipped")
            continue
        t_rm = t_of[rm["seq"]][1]
        later = [t for t in ap.get(name, []) if t >= t_rm]
        if not later:
            continue
        out.append(((later[0] - t_rm) / 1000.0, f"{tour}:REMOVE-REACQUIRE {name} seq{rm['seq']}"))
    return out


def event_samples(path, notes):
    keep, dele, rebirths, crossed = [], [], [], {}
    rows = list(csv.DictReader(open(path, errors="replace")))
    for row in rows:
        try:
            sec, t = float(row["seconds"]), int(row["t_wall_ms"])
        except (KeyError, ValueError, TypeError):
            continue
        door = (row.get("door") or "").strip()
        tag = f"{pathlib.Path(path).name}:{row['room']}/{row['place']}@{row['t_wall_ms']}"
        if row["kind"] == "keep_episode" and sec > 0:
            keep.append((sec, tag))
        elif row["kind"] == "rebirth" and sec > 0:
            rebirths.append((t, sec, door, tag))
        elif row["kind"] == "crossed" and door:
            crossed.setdefault(door, []).append(t)
    for t, sec, door, tag in rebirths:
        if door and any(tc >= t for tc in crossed.get(door, [])):
            dele.append((sec, f"{tag}:crossed-later {door}"))
        else:
            notes.append(f"{tag}: rebirth of {door or '(no door column)'} after {sec:.1f} s never crossed later: not a C_delete sample")
    return keep, dele


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main():
    keep, dele, notes = [], [], []
    tours = pathlib.Path(arg("--tours", str(TOURS)))
    out = pathlib.Path(arg("--out", str(OUT)))
    for n in range(10, 23):   # the tour samples are always kept (ruling N3)
        d = tours / f"explore_tour{n}"
        if not d.is_dir():
            continue
        keep += keep_samples(d, d.name, notes)
        dele += delete_samples(d, d.name, notes)
    if "--events" in sys.argv:
        files = []
        for f in sys.argv[sys.argv.index("--events") + 1:]:
            if f.startswith("--"):
                break
            files.append(f)
        for f in files:
            k, d = event_samples(f, notes)
            keep += k; dele += d
    return write_rows(keep, dele, notes, out)


def write_rows(keep, dele, notes, out=OUT):
    rows = []
    for kind, samples in (("keep", keep), ("delete_ordinary", dele)):
        print(f"{kind}: n={len(samples)}", [f"{s:.1f}" for s, _ in samples])
        if len(samples) < MIN_N:
            print(f"  fewer than {MIN_N} samples: falling back to {FALLBACK_S:.0f} s (spec reading), flagged in the source column")
            rows.append((kind, FALLBACK_S, f"fallback_spec_{FALLBACK_S:.0f}s_n={len(samples)}<{MIN_N}"))
        else:
            print(f"  median {statistics.median(s for s, _ in samples):.2f} s")
            rows += [(kind, s, src) for s, src in samples]
    for x in notes:
        print("  note:", x)
    med = {k: statistics.median(s for kk, s, _ in rows if kk == k) for k in ("keep", "delete_ordinary")}
    if not med["delete_ordinary"] > med["keep"]:
        print(f"REFUSED: median C_keep {med['keep']:.3f} s >= median C_delete {med['delete_ordinary']:.3f} s - deletion must cost more "
              f"(spec 3.6, never equal costs). {out} left as it was.", file=sys.stderr)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["kind", "seconds", "source"])
        for k, s, src in rows:
            w.writerow([k, f"{s:.3f}", src])
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
