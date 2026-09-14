#!/usr/bin/env python3
"""A/B the epistemic explorer against the heuristic it superseded, on the same 594 layouts.

Run the two arms first (same binary, same corpus, same frame cap):

    OUT=<dir>/c594      WS_NO7=1                  WS_ROOMS=594 \
        WS_LAYOUTS=datasets/matterport_layout/corpus594.txt \
        WS_POLY_ROOM=$OUT/poly bin/wall_slam_selftest > $OUT/run.log 2>&1
    OUT=<dir>/c594_heur WS_NO7=1 WS_NO_INFOGAIN=1 WS_ROOMS=594 ... (same)

then:  python3 analysis/explorer/explorer_ab.py <dir>/c594 <dir>/c594_heur

Two endpoints, both PAIRED by room, because the room-to-room spread (0.95 to 1.00) is an order of
magnitude larger than any explorer effect and an unpaired test would drown in it:

  1. final IoU          -- parsed from run.log
  2. frames to the first PUBLISHABLE map -- parsed from the per-room poly CSV

CAVEAT that decides how to read the result: failed rooms write the sentinels IoU 0.000 and
Hausdorff 1e9. They are filtered here and counted separately; a mean over the raw column is
meaningless (the first aggregate ever taken over it reported a mean Hausdorff of 1.7e6 m).

CAVEAT on the CSV: `closed,publishable` is field 6, NOT the last field -- two status strings follow
it. Reading p[-1] silently yields "no room ever became publishable", which is what it did first.
"""
import glob, os, re, sys, random, statistics as st


def final_iou(run_log):
    """room -> (corners, IoU, Hausdorff) from the per-room summary lines."""
    rows = {}
    pat = re.compile(r'room\s+(\d+)\s+(\S+)\s+corners\s+(\d+)\s+IoU\s+([\d.]+).*?Hausdorff\s+([\d.e+]+)\s*m')
    for line in open(run_log, errors='ignore'):
        m = pat.search(line)
        if m:
            rows[int(m.group(1))] = (int(m.group(3)), float(m.group(4)), float(m.group(5)))
    return rows


def first_publishable(poly_dir):
    """room -> frame index of the first row whose publishable flag is 1, else None."""
    out = {}
    for f in glob.glob(os.path.join(poly_dir, 'poly_*.csv')):
        room = int(os.path.basename(f)[5:-4])
        out[room] = None
        for line in open(f, errors='ignore'):
            if line.startswith('#'):
                continue
            p = line.split(';')
            if len(p) < 8 or ',' not in p[6]:
                continue
            if p[6].strip().split(',')[1] == '1':
                out[room] = int(p[0])
                break
    return out


def ci(diffs, B=2000, seed=11):
    random.seed(seed)
    bs = sorted(st.mean(random.choices(diffs, k=len(diffs))) for _ in range(B))
    return bs[int(.025 * B)], bs[int(.975 * B) - 1]


def failed(rec):
    return rec[1] <= 0.0 or rec[2] > 1e6


def main(dir_eig, dir_heur):
    E, H = final_iou(os.path.join(dir_eig, 'run.log')), final_iou(os.path.join(dir_heur, 'run.log'))
    common = sorted(set(E) & set(H))
    fE = [r for r in common if failed(E[r])]
    fH = [r for r in common if failed(H[r])]
    ok = [r for r in common if r not in set(fE) and r not in set(fH)]
    print(f"rooms paired {len(common)} | failures: EIG {len(fE)}  HEUR {len(fH)} | scored {len(ok)}")

    for name, rows in (('EIG', E), ('HEUR', H)):
        v = sorted(rows[r][1] for r in ok)
        print(f"  {name:5s} IoU mean {st.mean(v):.4f} med {st.median(v):.4f} "
              f"p10 {v[len(v)//10]:.4f} >=.95 {100*sum(i>=.95 for i in v)/len(v):.1f}%")

    d = [E[r][1] - H[r][1] for r in ok]
    lo, hi = ci(d)
    print(f"\nPAIRED dIoU (EIG-HEUR): mean {st.mean(d):+.4f}  95% CI [{lo:+.4f},{hi:+.4f}]")
    print(f"  EIG better {sum(x>1e-4 for x in d)} | HEUR better {sum(x<-1e-4 for x in d)} | tie {sum(abs(x)<=1e-4 for x in d)}")

    pE, pH = first_publishable(dir_eig), first_publishable(dir_heur)
    both = [r for r in common if pE.get(r) is not None and pH.get(r) is not None]
    print(f"\nnever publishable: EIG {sum(pE.get(r) is None for r in common)} "
          f"HEUR {sum(pH.get(r) is None for r in common)} | both reached it: {len(both)}")
    if both:
        dp = [pE[r] - pH[r] for r in both]
        lo, hi = ci(dp)
        print(f"  frames to publishable: EIG med {st.median([pE[r] for r in both]):.0f} "
              f"HEUR med {st.median([pH[r] for r in both]):.0f}")
        print(f"  PAIRED difference: mean {st.mean(dp):+.1f} frames  95% CI [{lo:+.1f},{hi:+.1f}]")

    print("\nEXPLORATORY, not pre-registered -- by ground-truth corner count:")
    for lo_c, hi_c, lab in ((0, 6, '6'), (7, 8, '8'), (9, 10, '10'), (11, 99, '12+')):
        g = [r for r in ok if lo_c <= E[r][0] <= hi_c]
        if len(g) < 10:
            continue
        dg = [E[r][1] - H[r][1] for r in g]
        lo, hi = ci(dg, seed=3)
        print(f"  {lab:>4} corners, {len(g):>3} rooms: dIoU {st.mean(dg):+.4f}  [{lo:+.4f},{hi:+.4f}]")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
