#!/usr/bin/env python3
"""E1 (paper §5.2), second test: replay logged kitchen membership decisions with the elicited class log-odds.

kitchen_metaconcept computes, per member m with class log-odds λ (kitchen_belief.cpp, membership_prob):
    w   = sigmoid(logit(1 − ClutterFrac) + λ)          class prior weight
    r   = w·E / (w·E + 1 − w)                          membership, E = p_frame(geometry) / p_null(geometry)
and the per-cycle frame evidence (instant_log_odds) is  Σ_m log(w·E + 1 − w), low-passed by an EMA and clamped.

The log (etc/kitchen_fit.csv) holds r for every member and cycle under the AUTHORED λ, so E is recovered
exactly, E = r(1 − w) / (w(1 − r)), and with the elicited λ':
    r' = sigmoid(logit(r) + λ' − λ)
The frame evidence is replayed the same way, through the same EMA and clamp.

What this holds fixed: the fitted frame (axis, worktop, depth). In the agent the frame is itself fitted
with membership weights, so a changed λ would also move it slightly; this replay is first order in that.
A decision is "member" when r > 0.5. Ground truth is given per node in --truth (default: every logged
member is a real kitchen unit, which is the case for the logged Webots apartment run).

    python3 tools/replay_kitchen.py [--log ../kitchen_metaconcept/etc/kitchen_fit.csv] [--elicited tools/e1_out/level3_kitchen.csv]
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # active_inference/


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def weight(clutter: float, lam: float) -> float:
    return min(max(sigmoid(logit(1.0 - clutter) + lam), 1e-4), 1.0 - 1e-4)


def replay(rows, authored, elicited, clutter, alpha, clamp, truth):
    """Per-cycle replay. Returns per-node decision counts and the two frame-evidence traces."""
    by_cycle = defaultdict(list)
    for r in rows:
        by_cycle[int(r["cycle"])].append(r)
    stats = defaultdict(lambda: dict(n=0, a_in=0, e_in=0, flips_in=0, flips_out=0, a_ok=0, e_ok=0,
                                     saturated=0))
    lo_a = lo_e = 0.0
    trace = []
    for c in sorted(by_cycle):
        inst_a = inst_e = 0.0
        for r in by_cycle[c]:
            cls, node = r["class"], r["node"]
            la, le = authored[cls], elicited.get(cls, authored[cls])
            ra = float(r["membership"])
            if ra <= 1e-6 or ra >= 1.0 - 1e-6:
                stats[node]["saturated"] += 1
            re_ = sigmoid(logit(ra) + le - la)
            wa, we = weight(clutter, la), weight(clutter, le)
            E = math.exp(logit(ra)) * (1.0 - wa) / wa
            inst_a += math.log(wa * E + 1.0 - wa)
            inst_e += math.log(we * E + 1.0 - we)
            s = stats[node]
            s["n"] += 1
            ia, ie = ra > 0.5, re_ > 0.5
            s["a_in"] += ia
            s["e_in"] += ie
            s["flips_in"] += (ie and not ia)
            s["flips_out"] += (ia and not ie)
            t = truth.get(node, True)
            s["a_ok"] += (ia == t)
            s["e_ok"] += (ie == t)
        lo_a = max(-clamp, min(clamp, lo_a + alpha * (inst_a - lo_a)))
        lo_e = max(-clamp, min(clamp, lo_e + alpha * (inst_e - lo_e)))
        trace.append((c, float(by_cycle[c][0]["log_odds"]), lo_a, lo_e))
    return stats, trace


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", type=Path, default=ROOT / "kitchen_metaconcept/etc/kitchen_fit.csv")
    ap.add_argument("--config", type=Path, default=ROOT / "kitchen_metaconcept/etc/config.toml")
    ap.add_argument("--elicited", type=Path, default=HERE / "e1_out/level3_kitchen.csv")
    ap.add_argument("--truth", type=Path, default=None,
                    help="CSV node,is_kitchen_unit (1/0); nodes not listed count as kitchen units")
    ap.add_argument("--out", type=Path, default=HERE / "e1_out/replay_kitchen.csv")
    args = ap.parse_args()

    km = tomllib.loads(args.config.read_text())["KitchenMetaconcept"]
    authored = dict(zip(km["MemberClasses"], km["MemberClassLogOdds"]))
    clutter, alpha = km["ClutterFrac"], km["EvidenceEmaAlpha"]
    clamp = 8.0  # KitchenParams::logodds_clamp (kitchen_belief.h), not in the config
    elicited = {r["cls"]: float(r["elicited"]) for r in csv.DictReader(args.elicited.open())}
    truth = {}
    if args.truth:
        truth = {r["node"]: r["is_kitchen_unit"] == "1" for r in csv.DictReader(args.truth.open())}

    rows = list(csv.DictReader(args.log.open()))
    stats, trace = replay(rows, authored, elicited, clutter, alpha, clamp, truth)

    # Sanity: the authored replay of the frame evidence must reproduce the logged EMA.
    err = max(abs(logged - a) for _, logged, a, _ in trace)

    print(f"log {args.log} : {len(trace)} cycles, {len(rows)} member-cycles")
    print(f"authored replay of the frame log-odds vs logged: max |diff| = {err:.3g} nats")
    print(f"\n{'node':<20} {'class':<13} {'λ auth':>6} {'λ elic':>6} {'cycles':>7} "
          f"{'in(auth)':>9} {'in(elic)':>9} {'→in':>6} {'→out':>6} {'correct auth':>12} {'correct elic':>12}")
    cls_of = {r["node"]: r["class"] for r in rows}
    out = []
    for node, s in sorted(stats.items()):
        c = cls_of[node]
        print(f"{node:<20} {c:<13} {authored[c]:>6.2f} {elicited.get(c, authored[c]):>6.2f} {s['n']:>7} "
              f"{s['a_in']:>9} {s['e_in']:>9} {s['flips_in']:>6} {s['flips_out']:>6} {s['a_ok']:>12} {s['e_ok']:>12}")
        out.append(dict(node=node, cls=c, authored=authored[c], elicited=elicited.get(c, authored[c]),
                        is_kitchen_unit=int(truth.get(node, True)), **s))
    tot = {k: sum(s[k] for s in stats.values()) for k in ("n", "flips_in", "flips_out", "a_ok", "e_ok")}
    print(f"\nmember-cycle decisions: {tot['n']}; flipped in {tot['flips_in']}, out {tot['flips_out']}; "
          f"correct authored {tot['a_ok']}, elicited {tot['e_ok']} (net {tot['e_ok'] - tot['a_ok']:+d})")
    fa = [a for _, _, a, _ in trace]
    fe = [e for _, _, _, e in trace]
    first = lambda xs: next((i + 1 for i, x in enumerate(xs) if x >= 0.9 * clamp), None)
    print(f"frame log-odds: authored final {fa[-1]:.2f}, elicited final {fe[-1]:.2f}; "
          f"cycles to 90% of clamp: authored {first(fa)}, elicited {first(fe)}")
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
