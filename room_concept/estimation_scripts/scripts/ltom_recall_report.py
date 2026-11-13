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

if __name__ == "__main__":
    main()
