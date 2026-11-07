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
Output: etc/door_costs.csv  (kind,seconds,source), medians and n printed.
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


def main():
    keep, dele, notes = [], [], []
    for n in range(10, 23):
        d = TOURS / f"explore_tour{n}"
        if not d.is_dir():
            continue
        keep += keep_samples(d, d.name, notes)
        dele += delete_samples(d, d.name, notes)
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
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["kind", "seconds", "source"])
        for k, s, src in rows:
            w.writerow([k, f"{s:.3f}", src])
    print("wrote", OUT)


if __name__ == "__main__":
    sys.exit(main())
