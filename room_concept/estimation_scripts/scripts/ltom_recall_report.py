#!/usr/bin/env python3
"""ltom_recall_report.py <crossing/NAME> — grade an LTOM recall test (ltom_recall_test.sh).

Pass 1 stores; pass 2 should recall. Per object class:
  stored      objects in the repertory after pass 1 (each has a uid)
  p2 births   tracker births in pass 2 (agent logs: "[tracker] birth at … — inside" plus RESURRECT lines)
  recalled    pass-2 births that RESURRECTED a pass-1 uid (same identity came back as the prior)
  wrong uid   RESURRECT of a uid NOT stored in pass 1 (should be 0)
  new         pass-2 births that did not resurrect (a stored object re-born as new is a MISSED recall)
  missed      stored objects never resurrected in pass 2 (not seen again, or not matched)
Reads only text/JSON the runs left behind; no graph access.
"""
import json, os, re, sys
from collections import defaultdict

def repertory(path):
    if not os.path.exists(path):
        return {}
    d = json.load(open(path))
    return {e["uid"]: e for e in d.get("entries", [])}

RESURRECT = re.compile(r"RESURRECT (\w+?)_(\d+) uid=(\d+) p=([0-9.]+)")
BIRTH_IN = re.compile(r"\[tracker\] birth at \(.*\) room \d+ — inside")

def main():
    root = sys.argv[1]
    name = os.path.basename(root.rstrip("/"))
    stored = repertory(os.path.join(root, "repertory_after_p1.json"))
    by_cls = defaultdict(set)
    for uid, e in stored.items():
        by_cls[e.get("cls", "?")].add(uid)
    logs = os.path.join(os.path.dirname(root.rstrip("/")), f"{name}_p2", "logs")
    rows = []
    classes = sorted(set(by_cls) | {f[:-len("_concept.log")] for f in (os.listdir(logs) if os.path.isdir(logs) else [])
                                    if f.endswith("_concept.log")})
    for cls in classes:
        log = os.path.join(logs, f"{cls}_concept.log")
        births = res_ok = res_bad = 0
        recalled = set()
        if os.path.exists(log):
            for line in open(log, errors="replace"):
                if BIRTH_IN.search(line):
                    births += 1
                m = RESURRECT.search(line)
                if m:
                    uid = m.group(3)
                    if uid in by_cls.get(cls, set()):
                        res_ok += 1; recalled.add(uid)
                    else:
                        res_bad += 1
        stored_n = len(by_cls.get(cls, set()))
        rows.append((cls, stored_n, births, res_ok, res_bad, max(0, births - res_ok - res_bad), stored_n - len(recalled)))
    print(f"LTOM recall test {name}: {len(stored)} objects stored after pass 1")
    print(f"{'class':14}{'stored':>8}{'p2 births':>11}{'recalled':>10}{'wrong uid':>11}{'new':>6}{'missed':>8}")
    for r in rows:
        if r[1] == 0 and r[2] == 0 and r[3] == 0 and r[4] == 0:
            continue   # an agent that stores no objects and births none in pass 2 (door, residual, room): nothing to grade
        print(f"{r[0]:14}{r[1]:>8}{r[2]:>11}{r[3]:>10}{r[4]:>11}{r[5]:>6}{r[6]:>8}")
    tot_stored = sum(r[1] for r in rows)
    recalled_uids = sum(r[1] - r[6] for r in rows)
    print(f"recall rate: {recalled_uids}/{tot_stored} stored objects came back with their pass-1 uid")
    if not os.path.isdir(logs):
        print(f"(no pass-2 logs at {logs})")
    ltom_checks(root, name)


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None

LINE_T = re.compile(r"^(\d+\.\d+) ")
OFFER = re.compile(r"\[ltom\] recall offered ltom_recall_(\d+) \((\w+)\) in room (\S+)")
WITHDRAW = re.compile(r"\[ltom\] recall withdrawn ltom_recall_(\d+)")
BORN = re.compile(r"\[ltom\] born (\S+) uid=(\d+)")
GONE = re.compile(r"\[ltom\] (?:died|owner_exit) (\S+) uid=(\d+)")

def ltom_checks(root, name):
    """The ltom owner's checks (chat session, 2026-11-13): (a) humans unchanged, (b) owned things keep their object,
    (c) recall offered/withdrawn per room with timing, (d) two live nodes on one uid."""
    reps = {k: load(os.path.join(root, f"repertory_{k}.json")) for k in ("before", "after_p1", "after_p2")}
    print("\n(a) humans unchanged across the passes:")
    base = (reps["before"] or {}).get("humans")
    for k in ("after_p1", "after_p2"):
        h = (reps[k] or {}).get("humans") if reps[k] else None
        print(f"    {k}: {'MISSING FILE' if reps[k] is None else ('identical' if h == base else 'CHANGED')}")
    print("(b) owned things — the object's uid is still in memory, and was resurrected in pass 2:")
    p2_res = set()
    logs2 = os.path.join(os.path.dirname(root.rstrip("/")), f"{name}_p2", "logs")
    for f in (os.listdir(logs2) if os.path.isdir(logs2) else []):
        if f.endswith("_concept.log"):
            for line in open(os.path.join(logs2, f), errors="replace"):
                m = RESURRECT.search(line)
                if m: p2_res.add(m.group(3))
    stored2 = {e["uid"] for e in ((reps["after_p2"] or {}).get("entries", []))}
    for hm in (base or []):
        for t in hm.get("things", []):
            u = t.get("object_uid")
            print(f"    {hm.get('id')} owns {t.get('name_when_said')} uid={u}: in memory after p2: {u in stored2}; "
                  f"resurrected in p2: {u in p2_res}; still listed after p2: "
                  f"{any(x.get('object_uid') == u for h2 in ((reps['after_p2'] or {}).get('humans') or []) for x in h2.get('things', []))}")
    print("    (the live `owns` EDGE in the LTOM graph is not in any file: check it in the viewer while pass 2 runs)")
    for p in ("p1", "p2"):
        log = os.path.join(os.path.dirname(root.rstrip("/")), f"{name}_{p}", "logs", "ltom_agent.log")
        if not os.path.exists(log):
            print(f"(c,d) {p}: no ltom_agent log"); continue
        t0 = None; offers = defaultdict(list); withdrawn = defaultdict(list); room_of = {}
        live = {}; dups = []
        for line in open(log, errors="replace"):
            mt = LINE_T.match(line); t = float(mt.group(1)) if mt else None
            if t is not None and t0 is None: t0 = t
            rel = (t - t0) if (t is not None and t0 is not None) else float("nan")
            if m := OFFER.search(line):
                offers[m.group(3)].append(rel); room_of[m.group(1)] = m.group(3)
            elif m := WITHDRAW.search(line):
                withdrawn[room_of.get(m.group(1), "?")].append(rel)
            elif m := BORN.search(line):
                uid = m.group(2)
                if uid in live and live[uid] != m.group(1):
                    dups.append((rel, uid, live[uid], m.group(1)))
                live[uid] = m.group(1)
            elif m := GONE.search(line):
                if live.get(m.group(2)) == m.group(1): live.pop(m.group(2))
        print(f"(c) {p} recall nodes per room (seconds from ltom start):")
        for room in sorted(set(offers) | set(withdrawn)):
            o, w = offers.get(room, []), withdrawn.get(room, [])
            span = lambda v: (lambda f: f"{min(f):.0f}-{max(f):.0f} s" if f else "- (no timestamps)")([x for x in v if x == x])
            print(f"    {room:10} offered {len(o):3} ({span(o)})  withdrawn {len(w):3} ({span(w)})")
        print(f"(d) {p} duplicates (a uid born again while its first node is live): {len(dups)}")
        for rel, uid, a, b in dups[:10]:
            print(f"    t={rel:.0f} s uid={uid}: {a} still live when {b} was born")

if __name__ == "__main__":
    main()
