#!/usr/bin/env python3
"""E1 (paper §5.2), main test: Brier score of the elicited pi_cf against class presence in a room corpus.

For every corpus room of function f and every mapped class c, a source predicts P(c present | f) and is scored
(pi − z)², z = 1 if c is in the room. Sources:
    vlm       pi_cf from tools/e1_out/level4_priors.csv
    constant  0.75 for every (c, f)   ("likely", the paper's constant prior)
    uniform   0.5  for every (c, f)   (the mean of a uniform Beta(1,1) prior)
and two references fitted ON the corpus (so not competitors, only bounds):
    pooled    frequency of c over all rooms, ignoring f   (what knowing nothing about function buys)
    oracle    frequency of c in rooms of f                 (the irreducible Brier score)
Differences vlm − baseline get a 95% bootstrap interval over houses (rooms of one house are resampled together).

    python3 tools/brier_e1.py [--rooms tools/corpora/procthor_rooms.csv] [--exclude "kitchen:sink,bedroom:dining table"]
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCES = ("vlm", "constant", "uniform", "pooled", "oracle")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rooms", type=Path, default=HERE / "corpora/procthor_rooms.csv")
    ap.add_argument("--priors", type=Path, default=HERE / "e1_out/level4_priors.csv")
    ap.add_argument("--exclude", default="", help='"func:class,..." pairs left out (e.g. generator artifacts)')
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=None, help="per (function, class) CSV; default next to --priors")
    args = ap.parse_args()

    rooms = list(csv.DictReader(args.rooms.open()))
    classes = list(rooms[0])[4:]
    vlm = {(r["cls"], r["func"]): float(r["pi_cf"]) for r in csv.DictReader(args.priors.open())}
    excluded = {tuple(x.split(":")[::-1]) for x in args.exclude.split(",") if x}   # (class, func)
    funcs = sorted({r["function"] for r in rooms})
    pairs = [(c, f) for f in funcs for c in classes if (c, f) in vlm and (c, f) not in excluded]

    n_f = defaultdict(int)
    k_cf = defaultdict(int)
    k_c = defaultdict(int)
    for r in rooms:
        n_f[r["function"]] += 1
        for c in classes:
            z = int(r[c])
            k_cf[(c, r["function"])] += z
            k_c[c] += z
    n_all = len(rooms)
    pred = {}
    for c, f in pairs:
        pred[(c, f)] = dict(vlm=vlm[(c, f)], constant=0.75, uniform=0.5, pooled=k_c[c] / n_all,
                            oracle=k_cf[(c, f)] / n_f[f])

    # Per-house sums of squared error per source, and per-(c, f) sums for the breakdown.
    house_se = defaultdict(lambda: [0.0] * len(SOURCES))
    house_n = defaultdict(int)
    pair_se = defaultdict(lambda: [0.0] * len(SOURCES))
    by_func = {f: set(c for c, ff in pairs if ff == f) for f in funcs}
    for r in rooms:
        f, h = r["function"], r["house"]
        for c in by_func[f]:
            z = int(r[c])
            p = pred[(c, f)]
            se = [(p[s] - z) ** 2 for s in SOURCES]
            hs, ps = house_se[h], pair_se[(c, f)]
            for i, e in enumerate(se):
                hs[i] += e
                ps[i] += e
            house_n[h] += 1

    houses = list(house_se)
    tot_n = sum(house_n.values())
    brier = [sum(house_se[h][i] for h in houses) / tot_n for i in range(len(SOURCES))]

    rng = random.Random(0)
    diffs = {s: [] for s in ("constant", "uniform", "pooled")}
    for _ in range(args.boot):
        sample = rng.choices(houses, k=len(houses))
        n = sum(house_n[h] for h in sample)
        tot = [sum(house_se[h][i] for h in sample) / n for i in range(3)] + \
              [sum(house_se[h][3] for h in sample) / n]
        for s, i in (("constant", 1), ("uniform", 2), ("pooled", 3)):
            diffs[s].append(tot[0] - tot[i])

    def ci(xs):
        xs = sorted(xs)
        return xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]

    print(f"corpus {args.rooms.name}: {n_all} rooms in {len(houses)} houses; {len(pairs)} (class, function) pairs, "
          f"{tot_n} predictions" + (f"; excluded {sorted(excluded)}" if excluded else ""))
    print("\nBrier score (lower is better)")
    for s, b in zip(SOURCES, brier):
        note = {"pooled": "  (fitted on corpus, ignores function)", "oracle": "  (fitted on corpus: the floor)"}.get(s, "")
        print(f"  {s:<9} {b:.4f}{note}")
    print("\nvlm − baseline (negative = VLM better), 95% bootstrap over houses")
    for s, xs in diffs.items():
        lo, hi = ci(xs)
        print(f"  vs {s:<9} {brier[0] - brier[SOURCES.index(s)]:+.4f}  [{lo:+.4f}, {hi:+.4f}]")

    print("\nper function            vlm  constant  uniform   pooled   oracle")
    for f in funcs:
        cs = by_func[f]
        n = n_f[f] * len(cs)
        b = [sum(pair_se[(c, f)][i] for c in cs) / n for i in range(len(SOURCES))]
        print(f"  {f:<18} " + "  ".join(f"{x:7.4f}" for x in b))

    rows = []
    for c, f in pairs:
        n = n_f[f]
        rows.append(dict(func=f, cls=c, corpus_freq=round(k_cf[(c, f)] / n, 4), vlm=pred[(c, f)]["vlm"],
                         gap=round(pred[(c, f)]["vlm"] - k_cf[(c, f)] / n, 4),
                         brier_vlm=round(pair_se[(c, f)][0] / n, 4), brier_constant=round(pair_se[(c, f)][1] / n, 4)))
    rows.sort(key=lambda r: -abs(r["gap"]))
    print("\nlargest gaps  (vlm − corpus frequency)")
    for r in rows[:10]:
        print(f"  {r['func']:<12} {r['cls']:<18} vlm {r['vlm']:.2f}  corpus {r['corpus_freq']:.2f}  gap {r['gap']:+.2f}")
    out = args.out or args.priors.parent / f"brier_{args.rooms.stem}.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    sys.exit(main())
