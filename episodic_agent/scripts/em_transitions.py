#!/usr/bin/env python3
"""
em_transitions.py — the episodic memory as a world-model data set (2026-10-18).

Reads EM archives (etc/episodes/YYYYMMDD.jsonl; the last line per id wins) and writes ONE LINE PER APPLIED ACTION
(a closed step or claim): the state before, the action, the state after once its consequences settled, and what
changed. Every name is given twice: as the instance ("door_5") and abstracted to its type, with the action's own
target as "<target>" ("open <target>" made "cross <target>" possible; "<door>" is some other door).

    before / after   room, the target's beliefs (kFactAttrs only, no pose), the scene (things per type, actions on offer)
    effects          room_changed, belief deltas >= MIN_CHANGE, became_possible / no_longer_possible (actions),
                     appeared / dropped (things), target_gone
    chosen_among     the selector's candidates (claims)

Scene differences come from the record's scene_pre / scene_post snapshots, so flicker BETWEEN them cannot leak in; a
name that flickers within its mission (born or offered more than FLICKER times — a detector losing a door edge-on)
is still never an effect. Records without snapshots (older than 2026-10-18) are skipped and counted.

    em_transitions.py etc/episodes                  # every archive in the folder -> transitions.jsonl
    em_transitions.py etc/episodes/20261019.jsonl -o day.jsonl
    em_transitions.py etc/episodes --table          # also print the counting baseline

The counting baseline: per (action, target type, precondition), how often each outcome and each abstract effect
happened. With a handful of action and target types it is the model a learned one has to beat.
"""

import argparse
import collections
import json
import re
import sys
from pathlib import Path

FACT_ATTRS = ("door_open_prob", "door_reach_prob", "door_crossing_progress", "obj_exist_prob", "passage_passable")
MIN_CHANGE = 0.05                   # a belief delta below this is no consequence (episode_story.h kMinChange)
FLICKER = 4                         # born/offered more often than this within a mission = flicker (kFlicker)
EM_OUTCOMES = {"lost", "interrupted_by_em_shutdown"}   # EM's own stops, not the world's answer
VERBS = ("open", "cross", "approach", "inspect", "close", "reach", "look", "pass")


def stem(name):
    """'door_5' -> 'door'; a name without a trailing _<digits> is its own type."""
    m = re.fullmatch(r"(.+?)_\d+", name)
    return m.group(1) if m else name


def thing(name, target):
    return "<target>" if name and name == target else f"<{stem(name)}>"


def action(aff, target):
    """An affordance node name as an action: 'aff_door_5_open' -> 'open <target>' / 'open <door>'."""
    if aff == "afford_room":
        return "explore the room"
    if not aff.startswith("aff_"):
        return aff
    rest = aff[4:]
    obj, _, verb = rest.rpartition("_")
    if obj and verb in VERBS:
        return f"{verb} {thing(obj, target)}"
    return f"look at {thing(rest, target)}"


def load(paths):
    records = {}
    for p in paths:
        for n, line in enumerate(p.open(encoding="utf-8"), 1):
            try:
                r = json.loads(line)
                records[r["id"]] = r
            except (json.JSONDecodeError, KeyError):
                print(f"{p}:{n} skipped (invalid line)", file=sys.stderr)
    return records


def mission_of(r, records):
    seen = set()
    while r.get("parent") and r["parent"] in records and r["id"] not in seen:
        seen.add(r["id"])
        r = records[r["parent"]]
    return r


def flickering(missions, records):
    """Per mission id: the names born / offered more than FLICKER times in its tree (events deduplicated by time)."""
    trees = collections.defaultdict(set)
    for r in records.values():
        m = mission_of(r, records)
        for e in r.get("events", []):
            if e["kind"] in ("born", "offered"):
                trees[m["id"]].add((r["t0"] + e["dt_ms"], e["kind"], e["node"]))
    out = {}
    for mid in missions:
        c = collections.Counter(node for _, _, node in trees.get(mid, ()))
        out[mid] = {n for n, k in c.items() if k > FLICKER}
    return out


def beliefs(facts):
    if not facts:
        return {}
    v = facts[0].get("values", {})
    return {k: round(v[k], 3) for k in FACT_ATTRS if k in v}


def scene(sc, target):
    return {"things": dict(sorted(collections.Counter(thing(t, target) for t in sc["things"]).items())),
            "offered": sorted({action(a, target) for a in sc["offered"]})}


def transition(r, mission, noisy):
    target = r.get("target", "")
    pre, post = r["scene_pre"], r["scene_post"]
    b0, b1 = beliefs(r.get("pre")), beliefs(r.get("post"))
    deltas = {k: round(b1[k] - b0[k], 3) for k in b0 if k in b1 and abs(b1[k] - b0[k]) >= MIN_CHANGE}

    def diff(a, b):
        return sorted(set(a) - set(b) - noisy)

    inst = {"became_possible": diff(post["offered"], pre["offered"]),
            "no_longer_possible": diff(pre["offered"], post["offered"]),
            "appeared": diff(post["things"], pre["things"]),
            "dropped": diff(pre["things"], post["things"])}
    effects = {"room_changed": bool(r.get("room_after")) and r.get("room_after") != r.get("room"),
               "beliefs": deltas,
               "became_possible": sorted({action(a, target) for a in inst["became_possible"]}),
               "no_longer_possible": sorted({action(a, target) for a in inst["no_longer_possible"]}),
               "appeared": sorted({thing(t, target) for t in inst["appeared"]}),
               "dropped": sorted({thing(t, target) for t in inst["dropped"]}),
               "target_gone": bool(target) and bool(r.get("pre")) and not r.get("post")}
    alts = []
    if r.get("alternatives_json"):
        try:
            alts = sorted({action(a.get("node_name", ""), target) for a in json.loads(r["alternatives_json"])})
        except (json.JSONDecodeError, AttributeError):
            pass
    return {"id": r["id"], "mission": mission["id"], "author": mission.get("author"), "level": r["level"],
            "t0": r["t0"], "dur_s": round((r["t1"] - r["t0"]) / 1000.0, 2), "settled": bool(r.get("settled")),
            "action": r["kind"], "target": target, "target_type": stem(target) if target else "",
            "target_ltsm": r.get("target_ltsm", ""),
            "before": {"room": r.get("room", ""), "target": b0, **scene(pre, target)},
            "after": {"room": r.get("room_after", ""), "target": b1, **scene(post, target)},
            "outcome": r.get("outcome", ""), "effects": effects, "effects_instances": inst,
            "promised": r.get("promised"), "delivered": r.get("delivered"), "chosen_among": alts}


def bucket(p):
    return "low" if p < 0.3 else "high" if p > 0.7 else "mid"


def table(rows):
    groups = collections.defaultdict(list)
    for t in rows:
        pre = ",".join(f"{k.removeprefix('door_')}={bucket(v)}" for k, v in sorted(t["before"]["target"].items()))
        groups[(t["action"], t["target_type"] or "-", pre or "-")].append(t)
    print(f"{'action':<12} {'target':<8} {'precondition':<40} {'n':>3}  outcomes / effects (fraction of n)")
    for (act, typ, pre), ts in sorted(groups.items()):
        n = len(ts)
        outc = collections.Counter(t["outcome"] or "?" for t in ts)
        fx = collections.Counter()
        for t in ts:
            e = t["effects"]
            if e["room_changed"]:
                fx["room changed"] += 1
            for k, d in e["beliefs"].items():
                fx[f"{k} {'up' if d > 0 else 'down'}"] += 1
            fx.update(f"+{a}" for a in e["became_possible"])
            fx.update(f"-{a}" for a in e["no_longer_possible"])
            if e["target_gone"]:
                fx["target gone"] += 1
        words = [f"{o} {c / n:.0%}" for o, c in outc.most_common()] + ["|"] + \
                [f"{f} {c / n:.0%}" for f, c in fx.most_common(6)]
        print(f"{act:<12} {typ:<8} {pre:<40} {n:>3}  {'  '.join(words)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path, help="archive .jsonl files or folders of them")
    ap.add_argument("-o", "--out", type=Path, default=Path("transitions.jsonl"))
    ap.add_argument("--all", action="store_true", help="keep EM's own stops (lost, interrupted_by_em_shutdown)")
    ap.add_argument("--table", action="store_true", help="print the counting baseline")
    a = ap.parse_args()

    paths = sorted(p for i in a.inputs for p in ([i] if i.is_file() else i.glob("[0-9]" * 8 + ".jsonl")))
    records = load(paths)
    actions = [r for r in records.values() if r.get("level") in ("step", "claim") and r.get("t1") is not None]
    missions = {mission_of(r, records)["id"] for r in actions}
    noisy = flickering(missions, records)
    rows, skipped = [], collections.Counter()
    for r in sorted(actions, key=lambda r: r["t0"]):
        if r.get("outcome") in EM_OUTCOMES and not a.all:
            skipped["EM stop"] += 1
        elif "scene_pre" not in r or "scene_post" not in r:
            skipped["no scene (older record)"] += 1
        else:
            m = mission_of(r, records)
            rows.append(transition(r, m, noisy.get(m["id"], set())))
    with a.out.open("w", encoding="utf-8") as f:
        for t in rows:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"{len(paths)} archives, {len(actions)} closed actions -> {len(rows)} transitions in {a.out}"
          + "".join(f"; skipped {n} {why}" for why, n in skipped.items()), file=sys.stderr)
    if a.table and rows:
        table(rows)


if __name__ == "__main__":
    main()
