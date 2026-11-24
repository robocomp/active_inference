#!/usr/bin/env python3
"""Promised vs delivered nats per object visit (spec 2026-11-23 shared-engine G5, AuRo test b).

    promise_delivery.py <run>      <run> = a run folder, or its name under ../crossing/

Inputs in the run folder (harvested by concept_agents.sh): <class>_offers.csv (t_ms, node, standpoint_x, standpoint_y,
standpoint_yaw, promised_nats; one row per published offer) and <class>_delivered.csv (t_ms, node, delivered_nats; one row
per accepted persistent update, the theta-marginal entropy drop of the published Sigma), plus logs/controller.log and
gt_pose.tsv for the claims.

Claim windows: the rule of auro_1122/claim_windows.md (auro_1122/scripts/review5/cw.py), unchanged.
  * A claim is a controller `[aff-select] -> 'aff_<class>_<id>'` line; rooms and doors are not object offers.
  * all claims:  [claim, next [aff-select]] (the run's end for the last one).
  * per visit:   ['aff_X reached ->', own end], own end = the first of `dwell over: '<node>' mask seen`, `TARGET REFUSED —
                 'aff_X'`, `NONE ELIGIBLE` after the claim, capped by the next selection. Only reached claims have one.
  * Times: controller line -> seconds from gt_pose.tsv's 1 Hz controller line counts (aff_arbitration.line_times), so a
    window shorter than ~1 s reads as 0 s. The agents' rows carry wall-clock epoch ms, the clock of gt_pose.tsv's t_ms.
Promised = the node's last offer at or before the claim. Delivered = the sum of the node's delivered_nats rows inside the
window (an update's drop, summed over the visit's updates; no row = 0, a true "no update").

Prints per class (and ALL): n, promised median [IQR], the spread of promises (min-max, std, distinct values), delivered
median [IQR], and Spearman(promised, delivered) where there are >= 3 claims and the promises are not all equal.
"""
import bisect
import collections
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aff_arbitration import SEL, producer, line_times   # noqa: E402

CROSSING = Path(__file__).resolve().parent.parent / "crossing"


def spearman(x, y):
    """(rho, p) — scipy when available, else rho from average ranks and p = nan."""
    try:
        from scipy.stats import spearmanr
        r = spearmanr(x, y)
        return float(r.statistic), float(r.pvalue)
    except ImportError:
        def ranks(v):
            order = sorted(range(len(v)), key=lambda i: v[i])
            rk = [0.0] * len(v)
            i = 0
            while i < len(order):
                j = i
                while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                    j += 1
                for k in range(i, j + 1):
                    rk[order[k]] = (i + j) / 2.0
                i = j + 1
            return rk
        rx, ry = ranks(x), ranks(y)
        mx, my = statistics.fmean(rx), statistics.fmean(ry)
        num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
        den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
        return (num / den if den > 0 else float("nan")), float("nan")


def read_rows(path, value):
    """node -> sorted [(t_ms, value)] from one of the G5 logs; {} if the class wrote none."""
    d = collections.defaultdict(list)
    if not path.exists():
        return d
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                d[r["node"]].append((int(r["t_ms"]), float(r[value])))
            except (KeyError, ValueError):
                continue
    for k in d:
        d[k].sort()
    return d


def quart(v):
    if not v:
        return "-"
    if len(v) == 1:
        return f"{v[0]:.3f}"
    q = statistics.quantiles(v, n=4, method="inclusive")
    return f"{statistics.median(v):.3f} [{q[0]:.3f}-{q[2]:.3f}]"


def spread(v):
    if not v:
        return "-"
    sd = statistics.pstdev(v) if len(v) > 1 else 0.0
    return f"{min(v):.3f}-{max(v):.3f} sd {sd:.3f} ({len(set(round(x, 6) for x in v))} distinct)"


def claims(run):
    """Object claims with their all-claims and per-visit windows (epoch ms), as cw.py builds them."""
    at = line_times(run)
    gt = list(csv.DictReader(open(run / "gt_pose.tsv"), delimiter="\t"))
    gt0, gt_end = int(gt[0]["t_ms"]), int(gt[-1]["t_ms"])
    lines = open(run / "logs" / "controller.log", errors="replace").read().splitlines()
    sels = [(i, m.group(1)) for i, ln in enumerate(lines, 1) if (m := SEL.search(ln))]
    t_of = lambda n: gt0 + at(n) * 1000.0
    out = []
    for k, (i, win) in enumerate(sels):
        cls, _ = producer(win)
        if cls in ("room", "door"):
            continue
        node = win[4:]
        j = sels[k + 1][0] if k + 1 < len(sels) else len(lines) + 1
        reach, own = None, j
        for n in range(i + 1, j):
            ln = lines[n - 1]
            if reach is None and f"{win} reached ->" in ln:
                reach = n
            if (f"dwell over: '{node}' mask seen" in ln or f"TARGET REFUSED — '{win}'" in ln or "NONE ELIGIBLE" in ln):
                own = min(own, n)
        t_c = t_of(i)
        t_j = t_of(j) if k + 1 < len(sels) else gt_end
        t_own = min(t_of(own) if own <= len(lines) else t_j, t_j)
        out.append(dict(cls=cls, node=node, t_c=t_c, t_j=t_j,
                        t_reach=t_of(reach) if reach is not None else None, t_own=t_own))
    return out


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    run = Path(sys.argv[1])
    if not run.is_dir():
        run = CROSSING / sys.argv[1]
    cl = claims(run)
    offers, delivered = {}, {}
    rows = []
    for c in cl:
        if c["cls"] not in offers:
            offers[c["cls"]] = read_rows(run / f"{c['cls']}_offers.csv", "promised_nats")
            delivered[c["cls"]] = read_rows(run / f"{c['cls']}_delivered.csv", "delivered_nats")
        of = offers[c["cls"]].get(c["node"], [])
        k = bisect.bisect_right([t for t, _ in of], c["t_c"]) - 1
        if k < 0:
            rows.append(dict(c, promised=None))
            continue
        dl = delivered[c["cls"]].get(c["node"], [])
        tot = lambda t0, t1: sum(v for t, v in dl if t0 < t <= t1)
        rows.append(dict(c, promised=of[k][1], d_all=tot(c["t_c"], c["t_j"]),
                         d_visit=tot(c["t_reach"], c["t_own"]) if c["t_reach"] is not None else None))

    print(f"run {run}: {len(cl)} object claims, {sum(r['promised'] is None for r in rows)} with no offer row before the claim")
    by = collections.defaultdict(list)
    for r in rows:
        if r["promised"] is not None:
            by[r["cls"]].append(r)
            by["ALL"].append(r)
    for window, key in (("all claims", "d_all"), ("arrival->end", "d_visit")):
        print(f"\n== {window} ==")
        print(f"{'class':<13} {'n':>3}  {'promised':<24} {'spread of promises':<38} {'delivered':<24} spearman")
        for cls in sorted(by, key=lambda c: (c == "ALL", c)):
            rs = [r for r in by[cls] if r[key] is not None]
            p = [r["promised"] for r in rs]
            d = [r[key] for r in rs]
            sp = "n too small" if len(rs) < 3 else ("promises constant" if len(set(p)) < 2 else
                                                    "rho {:.2f}, p {:.2g}".format(*spearman(p, d)))
            print(f"{cls:<13} {len(rs):>3}  {quart(p):<24} {spread(p):<38} {quart(d):<24} {sp}")


if __name__ == "__main__":
    main()
