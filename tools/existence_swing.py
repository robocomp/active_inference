"""Per door: sign changes of geo_dl between consecutive rows, median |geo_dl|, and the largest |d geo_dl| rows with their geometry.

usage: python3 -I tools/existence_swing.py <door_existence_log.csv> [--node NAME] [--top N]
"""
import csv
import statistics
import sys

SHOW = ("geo_x", "geo_mu0", "geo_mu1", "geo_s", "sup_x", "cx", "cy")


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    node = argv[argv.index("--node") + 1] if "--node" in argv else None
    top = int(argv[argv.index("--top") + 1]) if "--top" in argv else 10
    path = args[0]
    if node is not None:
        args = [a for a in args if a != node]
        path = args[0]
    per = {}
    with open(path, newline="") as f:
        for i, r in enumerate(csv.DictReader(f)):
            if node is not None and r["node"] != node:
                continue
            per.setdefault(r["node"], []).append((i, r))
    for name, rows in sorted(per.items()):
        g = [0.0 if v != v else v for v in (float(r["geo_dl"]) for _, r in rows)]   # nan: no geometry this cycle
        nz = [(k, v) for k, v in enumerate(g) if v != 0.0]
        sign_changes = sum(1 for (_, a), (_, b) in zip(nz, nz[1:]) if (a > 0) != (b > 0))
        med = statistics.median(abs(v) for v in g) if g else 0.0
        print(f"{name}: rows={len(g)} sign_changes={sign_changes} median_abs_geo_dl={med:.4g}")
        jumps = sorted(((abs(g[k] - g[k - 1]), k) for k in range(1, len(g))), reverse=True)[:top]
        for d, k in jumps:
            i, r = rows[k]
            vals = " ".join(f"{c}={r[c]}" for c in SHOW)
            print(f"  row={i + 2} cycle={r['cycle']} geo_dl={g[k]:+.4g} prev={g[k - 1]:+.4g} |d|={d:.4g} {vals}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
