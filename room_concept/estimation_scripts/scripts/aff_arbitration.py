#!/usr/bin/env python3
"""Offer arbitration from existing controller logs (AuRo C0, 2026-11-22).

Reads every "[aff-select] -> '<winner>' ... [candidates: '<n>'(<state>) gain=g score=s | ...]" line of
<run>/logs/controller.log (one line per change of selected affordance, affordance_manager.cpp:50/1007).
Score = gain' - lambda*nav_dist (+ switch margin for the incumbent), gain' = room gain x RoomGainScale while an
object offer is eligible (affordance_manager.cpp neg_efe). Eligible = Offered or Executing.
Time: the controller line count in <run>/gt_pose.tsv (1 Hz) -> seconds since the first row.
Writes <run>/aff_arbitration.csv; prints a per-run summary.
usage: aff_arbitration.py <run_dir> [<run_dir> ...]
"""
import csv, re, statistics, sys
from pathlib import Path

SEL = re.compile(r"\[aff-select\] -> '([^']+)' \((\w+)\) gain=([-\d.e]+) nats.*?\[candidates:(.*)\]")
CAND = re.compile(r"'([^']+)'\((\w+)\) gain=([-\d.einf]+) score=([-\d.einf]+)")
REF = re.compile(r"STOPPED TARGET REFUSED — '([^']+)'")
PRAG = re.compile(r"^aff_door_\d+_(open|close|cross|approach)$")


def producer(name):
    """(concept, offer kind) of an affordance name."""
    if name == "afford_room": return "room", "room"
    if name.startswith("aff_door_look_place_"): return "door", "door_look"
    if PRAG.match(name): return "door", "door_" + PRAG.match(name).group(1)
    if re.match(r"^aff_door_\d+$", name): return "door", "door_view"
    m = re.match(r"^aff_([a-z]+)_\d+$", name)
    return (m.group(1), m.group(1)) if m else ("other", name)


def line_times(run):
    """controller.log line number -> seconds since run start, from gt_pose.tsv's 1 Hz line counts."""
    rows = list(csv.DictReader(open(run / "gt_pose.tsv"), delimiter="\t"))
    t0 = int(rows[0]["t_ms"])
    pts = [(int(r["controller"]), (int(r["t_ms"]) - t0) / 1000) for r in rows if r.get("controller", "").isdigit()]
    def at(n):
        for c, t in pts:
            if c >= n: return t
        return pts[-1][1] if pts else float("nan")
    return at


def mine(run):
    run = Path(run); at = line_times(run); out = []
    refused = set()   # names whose latest event is "STOPPED TARGET REFUSED" (skipped by the refused-moments-ago rule)
    for i, ln in enumerate(open(run / "logs/controller.log", errors="replace"), 1):
        r = REF.search(ln)
        if r: refused.add(r.group(1)); continue
        m = SEL.search(ln)
        if not m: continue
        win = m.group(1)
        cands = [(n, st, float(g), float(s)) for n, st, g, s in CAND.findall(m.group(4))]
        elig = [c for c in cands if c[1] in ("Offered", "Executing")]
        wsc = next((c[3] for c in cands if c[0] == win), float("nan"))
        others = sorted((c for c in elig if c[0] != win and c[0] not in refused), key=lambda c: -c[3])
        skipped = [c[0] for c in elig if c[0] != win and c[0] in refused and c[3] > wsc]
        refused.discard(win)
        ru = others[0] if others else None
        concepts = sorted({producer(c[0])[0] for c in elig} | {producer(win)[0]})
        out.append(dict(t_s=round(at(i), 1), log_line=i, winner=win, winner_kind=producer(win)[1],
                        winner_concept=producer(win)[0], winner_gain=float(m.group(3)), winner_score=wsc,
                        runner_up=ru[0] if ru else "", runner_up_concept=producer(ru[0])[0] if ru else "",
                        runner_up_score=ru[3] if ru else "", margin=round(wsc - ru[3], 4) if ru else "",
                        skipped_refused=";".join(skipped), n_eligible=len(elig), concepts=";".join(concepts),
                        pragmatic_in_open_contest=int(any(PRAG.match(c[0]) for c in elig) and len(elig) > 1),
                        candidates=" | ".join(f"{c[0]}({c[1]}) g={c[2]:.3g} s={c[3]:.3g}" for c in cands)))
    if out:
        with open(run / "aff_arbitration.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    return out


def summary(run, rows):
    from collections import Counter
    print(f"== {Path(run).name}: {len(rows)} selections")
    print("  by offer kind:", dict(Counter(r["winner_kind"] for r in rows).most_common()))
    multi = [r for r in rows if len(r["concepts"].split(";")) >= 2]
    print(f"  eligible set held >=2 concepts: {len(multi)}/{len(rows)}; winners there:",
          dict(Counter(r["winner_concept"] for r in multi).most_common()))
    ms = [r["margin"] for r in multi if r["margin"] != ""]
    if ms:
        print(f"  margin winner - runner-up (multi-concept): median {statistics.median(ms):.2f}, range "
              f"[{min(ms):.2f}, {max(ms):.2f}] (score units, nats - 0.2/m)")
    obj_room = [r for r in multi if r["winner_concept"] not in ("room", "door") and "room" in r["concepts"]]
    room_obj = [r for r in multi if r["winner_concept"] == "room" and set(r["concepts"].split(";")) - {"room", "door"}]
    print(f"  object beat room: {len(obj_room)}; room beat object: {len(room_obj)};"
          f" higher-scoring offer skipped as just refused: {sum(bool(r['skipped_refused']) for r in rows)};"
          f" pragmatic offer inside an open multi-entrant contest: {sum(r['pragmatic_in_open_contest'] for r in rows)}")


if __name__ == "__main__":
    for r in sys.argv[1:]:
        summary(r, mine(r))
