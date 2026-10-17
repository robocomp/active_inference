#!/usr/bin/env python3
"""Acceptance report over an episodic-memory archive day file (<archive_dir>/<YYYYMMDD>.jsonl).

  python3 -I tools/em_report.py etc/episodes/20261017.jsonl [--also prev.jsonl ...]
          [--expect-closed] [--json] [--controller-log FILE]

One JSON object per line; the same id may appear several times (open line, then closing line):
the LAST line per id wins. Unparseable (truncated) lines are skipped with a warning.
Exit status 1 on any structural failure.
"""
import argparse, json, statistics, sys
from collections import Counter, defaultdict

CLOSED_BY_EM = ("lost", "interrupted_by_em_shutdown")
WORDS = ("lost", "interrupted_by_em_shutdown", "room_changed", "superseded")


def load(path, warnings):
    recs = {}
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
                if not isinstance(o, dict) or not isinstance(o.get("id"), str):
                    raise ValueError("not a record")
            except ValueError as e:
                warnings.append(f"{path}:{n}: skipped unparseable line ({e})")
                continue
            recs[o["id"]] = o  # last line wins
    return recs


def analyze(recs, also_ids, expect_closed):
    fails = []
    for id, r in sorted(recs.items()):
        lvl, par = r.get("level"), r.get("parent", "")
        if lvl == "mission":
            if par:
                fails.append((id, f"mission has parent '{par}'"))
        elif not par:
            fails.append((id, f"{lvl} has no parent"))
        elif par not in recs and par not in also_ids:
            fails.append((id, f"parent '{par}' not found"))
        t0, t1 = r.get("t0"), r.get("t1")
        if t1 is None:
            if expect_closed and r.get("outcome") not in CLOSED_BY_EM:
                fails.append((id, "open (no t1) with --expect-closed"))
        elif t0 is not None and t1 < t0:
            fails.append((id, f"t1 {t1} < t0 {t0}"))
    return fails


def summarize(recs, fails):
    c = lambda key: dict(sorted(Counter(r.get(key, "") or "" for r in recs.values()).items()))
    dur = defaultdict(list)
    for r in recs.values():
        if r.get("t1") is not None and r.get("t0") is not None and r["t1"] >= r["t0"]:
            dur[r.get("kind", "")].append((r["t1"] - r["t0"]) / 1000.0)
    outcomes = Counter(r.get("outcome", "") for r in recs.values())
    lost_lv = Counter(r.get("level") for r in recs.values() if r.get("outcome") == "lost")
    return {
        "records": len(recs), "by_level": c("level"), "by_author": c("author"), "by_outcome": c("outcome"),
        "median_duration_s_by_kind": {k: round(statistics.median(v), 3) for k, v in sorted(dur.items())},
        "outcome_words": {w: outcomes.get(w, 0) for w in WORDS},
        "lost": {"total": outcomes.get("lost", 0), "by_level": dict(sorted(lost_lv.items()))},
        "failures": [{"id": i, "why": w} for i, w in fails],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archive")
    ap.add_argument("--also", action="append", default=[], help="neighbouring day file whose ids may be parents")
    ap.add_argument("--expect-closed", action="store_true", help="open records are failures (clean-stop check)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--controller-log", help="accepted for the run recipe; not parsed")
    a = ap.parse_args(argv)
    warnings = []
    recs = load(a.archive, warnings)
    also = set()
    for p in a.also:
        also |= set(load(p, warnings))
    fails = analyze(recs, also, a.expect_closed)
    s = summarize(recs, fails)
    if a.json:
        s["warnings"] = warnings
        print(json.dumps(s, indent=1))
    else:
        for w in warnings:
            print("warning:", w)
        print(f"records: {s['records']}")
        for k in ("by_level", "by_author", "by_outcome"):
            print(f"{k}: {s[k]}")
        print(f"median duration s by kind: {s['median_duration_s_by_kind']}")
        print(f"outcome words: {s['outcome_words']}")
        print(f"lost: {s['lost']['total']} by level {s['lost']['by_level']}")
        print(f"structural failures: {len(fails)}")
        for i, w in fails:
            print(f"  FAIL {i}: {w}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
