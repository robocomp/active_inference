#!/usr/bin/env python3
"""List and compare the dated long-term-memory generations ltsm_agent writes ([Memory] history_dir).

Each generation is ltsm_<YYYYMMDD_HHMMSS_mmm>.json (the memory graph) plus, once a layout has been solved for it,
ltsm_<stamp>_layout.json (the MAP layout: room poses + covariance, seam residuals, P(inlier), energies).

    tools/ltsm_history.py [DIR]                 list every generation, oldest first
    tools/ltsm_history.py [DIR] --diff A B      room poses of two generations' layouts side by side (A, B = stamps,
                                                indices from the listing, or -1 / -2 for the newest ones)
DIR defaults to etc/ltsm_history (run from the component root, like the agent).
"""
import argparse
import json
import math
import pathlib
import sys


def generations(d: pathlib.Path):
    return sorted(p for p in d.glob("ltsm_*.json") if not p.name.endswith("_layout.json"))


def stamp(p: pathlib.Path) -> str:
    return p.stem[len("ltsm_"):]


def layout_of(p: pathlib.Path):
    lp = p.with_name(p.stem + "_layout.json")
    if not lp.exists():
        return None
    with lp.open() as f:
        return json.load(f)


def graph_summary(p: pathlib.Path):
    with p.open() as f:
        g = json.load(f)
    syms = g.get("DSRModel", {}).get("symbols", {}).values()
    rooms = sorted(s["name"] for s in syms if s.get("type") == "room")
    passages = sorted(s["name"] for s in syms
                      if s.get("attribute", {}).get("object_subtype", {}).get("value") == "passage")
    return len(list(g.get("DSRModel", {}).get("symbols", {}))), rooms, passages


def pick(gens, key: str) -> pathlib.Path:
    try:
        return gens[int(key)]
    except (ValueError, IndexError):
        pass
    for p in gens:
        if stamp(p).startswith(key):
            return p
    sys.exit(f"no generation matches {key!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", nargs="?", default="etc/ltsm_history")
    ap.add_argument("--diff", nargs=2, metavar=("A", "B"))
    a = ap.parse_args()
    d = pathlib.Path(a.dir)
    gens = generations(d)
    if not gens:
        sys.exit(f"no generations in {d}")

    if a.diff:
        A, B = (pick(gens, k) for k in a.diff)
        la, lb = layout_of(A), layout_of(B)
        if la is None or lb is None:
            sys.exit("both generations need a _layout.json sidecar")
        ra = {r["name"]: r for r in la["rooms"]}
        rb = {r["name"]: r for r in lb["rooms"]}
        print(f"{'room':16s} {stamp(A):>30s} {stamp(B):>30s} {'moved m':>8s} {'turned°':>8s}")
        for name in sorted(set(ra) | set(rb)):
            pa, pb = ra.get(name), rb.get(name)
            fmt = lambda r: f"({r['x']:7.3f}, {r['y']:7.3f}, {math.degrees(r['yaw']):7.2f})" if r else f"{'-':>27s}"
            moved = turned = ""
            if pa and pb:
                moved = f"{math.hypot(pb['x'] - pa['x'], pb['y'] - pa['y']):8.3f}"
                turned = f"{math.degrees(math.remainder(pb['yaw'] - pa['yaw'], 2 * math.pi)):8.2f}"
            print(f"{name:16s} {fmt(pa):>30s} {fmt(pb):>30s} {moved:>8s} {turned:>8s}")
        return

    print(f"{'#':>3s}  {'stamp':19s} {'nodes':>5s} {'rooms':>5s} {'pass':>4s}  {'F before':>9s} {'F after':>9s}  disowned seams")
    for i, p in enumerate(gens):
        n, rooms, passages = graph_summary(p)
        lay = layout_of(p)
        if lay:
            fb, fa = lay["energy_before"]["total"], lay["energy_after"]["total"]
            dis = [q["name"] for q in lay.get("passages", []) if 0 <= q.get("p_inlier", -1) < 0.5]
            tail = f"{fb:9.3f} {fa:9.3f}  {', '.join(dis) or '-'}"
        else:
            tail = f"{'-':>9s} {'-':>9s}  (no layout)"
        print(f"{i:3d}  {stamp(p):19s} {n:5d} {len(rooms):5d} {len(passages):4d}  {tail}")


if __name__ == "__main__":
    main()
