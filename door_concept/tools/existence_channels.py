#!/usr/bin/env -S python3 -I
"""Where did one door's existence log-odds go?  Over a cycle range, for one door of door_existence_log.csv (and, if given,
door_views_log.csv): the change of L split by channel (the ledger's per-update deltas zed_dl / ric_dl / geo_dl / sup_dl / act_dl),
the change of each prior term (band, base, size, extent), what the ledger moved by that no channel delta accounts for (a view's
aggregate re-scored by the frames that joined it shows up in that frame's own delta, so the remainder is the change of the
carried / crossing / vision terms), and the views that contributed (camera, mean q, hits).
  python3 -I tools/existence_channels.py existence.csv DOOR FROM TO [--views door_views_log.csv]"""
import csv, math, sys

CH = ["zed_dl", "ric_dl", "geo_dl", "sup_dl", "act_dl"]
PR = ["prior_band", "prior_base", "prior_size", "prior_extent"]


def num(s):
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def split(rows, door, c0, c1):
    """-> dict with 'dL', per-channel cumulative deltas, per-prior change, 'rest', 'n' for the door's rows in [c0, c1]."""
    sel = [r for r in rows if r["node"] == door and c0 <= int(r["cycle"]) <= c1]
    if len(sel) < 2:
        return None
    out = {"n": len(sel), "first": int(sel[0]["cycle"]), "last": int(sel[-1]["cycle"])}
    out["dL"] = float(sel[-1]["L"]) - float(sel[0]["L"])
    for k in CH:   # the first row's delta is the change INTO the first row: not part of the span
        out[k] = sum(num(r[k]) or 0.0 for r in sel[1:])
    for k in PR:
        out[k] = float(sel[-1][k]) - float(sel[0][k])
    out["rest"] = out["dL"] - sum(out[k] for k in CH) - sum(out[k] for k in PR)
    out["L_min"] = min(float(r["L"]) for r in sel)
    out["L_min_cycle"] = next(int(r["cycle"]) for r in sel if float(r["L"]) == out["L_min"])
    return out


def views(vrows, door, c0, c1):
    """-> {(camera): [n, mean q, mean hit]} over the door's view rows in the range."""
    acc = {}
    for r in vrows:
        if r["door"] != door or not (c0 <= int(r["cycle"]) <= c1):
            continue
        a = acc.setdefault(r["camera"], [0, 0.0, 0.0, 0, 0])
        q, h = num(r["q"]), num(r["hit"])
        a[0] += 1
        if q is not None: a[1] += q; a[3] += 1
        if h is not None: a[2] += h; a[4] += 1
    return {k: (a[0], a[1] / a[3] if a[3] else float("nan"), a[2] / a[4] if a[4] else float("nan")) for k, a in acc.items()}


def main(argv):
    if len(argv) < 5:
        print(__doc__); return 2
    rows = list(csv.DictReader(open(argv[1])))
    door, c0, c1 = argv[2], int(argv[3]), int(argv[4])
    s = split(rows, door, c0, c1)
    if s is None:
        print(f"{door}: fewer than two rows in cycles {c0}..{c1}"); return 1
    print(f"{door} cycles {s['first']}..{s['last']} ({s['n']} rows): dL {s['dL']:+.3f}  (L min {s['L_min']:+.3f} at {s['L_min_cycle']})")
    for k in CH + PR + ["rest"]:
        print(f"  {k:12s} {s[k]:+.3f}")
    if "--views" in argv:
        vrows = list(csv.DictReader(open(argv[argv.index("--views") + 1])))
        for cam, (n, q, h) in sorted(views(vrows, door, c0, c1).items()):
            print(f"  view {cam:8s} rows {n:4d}  mean q {q:.3f}  mean hit {h:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
