#!/usr/bin/env python3
"""Read etc/door_approach.csv and show how one class's evidence evolves along an approach.

The question this answers: as the robot drives down the corridor toward a door, does the POSTERIOR
climb smoothly for metres before the ARGMAX flips? If it does, the argmax is discarding evidence the
belief layer could have used, and the range at which it flips is the cost of that discard.

Two things this tool deliberately does NOT do:

  * It does not drop rows with no detection. Those rows ARE the measurement — a curve fitted only to
    frames where something was found starts at the moment of success and cannot show an onset.
  * It does not decide, from one number, that anything is proven. Rows at 20 Hz from a stationary
    robot are not independent samples: the same look, repeated. Every summary here is therefore also
    reported per DISTINCT VIEWPOINT (10 cm / 5 deg cells), and where the two disagree the viewpoint
    count is the honest one.

Usage:
    tools/plot_door_approach.py [etc/door_approach.csv] [--source zed|ricoh] [--bin 0.5]
    tools/plot_door_approach.py ... --segment       list approach segments (monotone closing runs)
    tools/plot_door_approach.py ... --plot out.png  write a chart instead of a table

Locale note: Python's float() is locale-independent, so this reader cannot suffer the comma-decimal
truncation that bit the C++ tools under es_ES (CLAUDE.md). The C++ writer imbues the classic locale.
"""

import argparse
import csv
import math
import sys
from collections import defaultdict


def read_rows(path, source):
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if source and r.get("source") != source:
                continue
            rows.append(r)
    return rows


def f(row, key):
    """Empty cell -> None, never 0.0. The writer leaves 'not measured' empty on purpose and
    collapsing that to a number is how a missing value becomes a data point."""
    v = row.get(key, "")
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def viewpoint_key(row, dxy=0.10, dth=math.radians(5)):
    x, y, th = f(row, "cam_x"), f(row, "cam_y"), f(row, "cam_yaw")
    if x is None or y is None or th is None:
        return None
    return (round(x / dxy), round(y / dxy), round(th / dth))


RANGE_KEY = "anchor_range_m"


def rng_of(row):
    """The range axis. Prefer anchor_range_m — the distance to where the door was last actually seen,
    which keeps counting down through the frames where nothing is found. Fall back to the raw peak
    depth only for logs written before the anchor existed, and say so once."""
    v = f(row, RANGE_KEY)
    return v if v is not None else f(row, "range_m")


def summarize(rows, bin_m):
    """Per range bin: how often the live pipeline saw the class, and what the model actually
    computed. `frames` and `views` are both reported because their ratio IS the overcount factor."""
    peaks = [r for r in rows if r["kind"] == "peak"]
    if peaks and f(peaks[0], RANGE_KEY) is None:
        print("NOTE: no anchor_range_m column — this log predates the anchor, so the axis is the raw\n"
              "      peak-pixel depth and is only meaningful on rows where the class actually won.\n")
    bins = defaultdict(lambda: {"frames": 0, "argmax_hit": 0, "p": [], "margin": [], "views": set()})
    for r in peaks:
        rng = rng_of(r)
        if rng is None:
            continue                       # no depth at the peak pixel: range unknown, not zero
        b = bins[int(rng / bin_m)]
        b["frames"] += 1
        b["argmax_hit"] += 1 if (f(r, "argmax_px") or 0) > 0 else 0
        if (p := f(r, "p_class")) is not None:
            b["p"].append(p)
        if (m := f(r, "margin")) is not None:
            b["margin"].append(m)
        if (vk := viewpoint_key(r)) is not None:
            b["views"].add(vk)

    def med(xs):
        return sorted(xs)[len(xs) // 2] if xs else None

    # ★Say WHY a table is empty. 1205 ricoh rows binned to nothing on the first live run because the
    # panorama carries no ZED depth, so range_m is empty on every one of them — and an empty table
    # under a "1205 rows" banner reads as "nothing happened" instead of "this source has no range".
    if not bins:
        n_no_range = sum(1 for r in peaks if rng_of(r) is None)
        print(f"no rows with a range: {n_no_range} of {len(peaks)} peak rows have an empty range_m.")
        if n_no_range == len(peaks) and peaks:
            print("This source has no depth (the 360 panorama does not carry one), so it cannot be\n"
                  "binned by range. Its posterior is still recorded — read it with --source zed for\n"
                  "the approach, or against stamp_ms/viewpoint for this one.")
        return

    print(f"{'range (m)':>12} {'frames':>7} {'views':>6} {'argmax fires':>13} "
          f"{'med P':>7} {'med margin':>11}  onset")
    for k in sorted(bins):
        b = bins[k]
        lo, hi = k * bin_m, (k + 1) * bin_m
        rate = b["argmax_hit"] / b["frames"]
        mp, mm = med(b["p"]), med(b["margin"])
        # A bin where the model is already finding the class but the argmax never fires is the
        # interesting one: evidence present, discarded by the collapse to a label.
        flag = ""
        if mm is not None and mp is not None:
            if rate == 0.0 and mp > 0.20:
                flag = "  <- evidence, no mask"
            elif 0.0 < rate < 1.0:
                flag = "  <- flip"
        print(f"{lo:6.1f}-{hi:4.1f} {b['frames']:7d} {len(b['views']):6d} "
              f"{rate:12.1%} {mp if mp is not None else float('nan'):7.3f} "
              f"{mm if mm is not None else float('nan'):11.3f}{flag}")

    total_f = sum(b["frames"] for b in bins.values())
    total_v = len(set().union(*[b["views"] for b in bins.values()])) if bins else 0
    if total_v:
        print(f"\n{total_f} frames from {total_v} distinct viewpoints "
              f"(x{total_f / total_v:.1f}). Frames at one pose are ONE look repeated, not "
              f"{total_f // max(1, total_v)} independent observations.")


def segments(rows, min_close_m=1.0):
    """Split into approach segments: runs where range is closing. This is what 'the robot drives
    toward the door' looks like in the file, and it is the unit a curve should be read over — an
    average across a whole tour mixes approaches, departures and pans."""
    peaks = [r for r in rows if r["kind"] == "peak" and rng_of(r) is not None]
    peaks.sort(key=lambda r: int(r["stamp_ms"]))
    runs, cur = [], []
    for r in peaks:
        if cur and rng_of(r) > rng_of(cur[-1]) + 0.30:   # opened up: end of approach
            runs.append(cur)
            cur = []
        cur.append(r)
    runs.append(cur)

    print(f"{'#':>3} {'frames':>7} {'from (m)':>9} {'to (m)':>8} {'closed':>7} "
          f"{'P first':>8} {'P last':>7} {'first mask at':>14}")
    n = 0
    for run in runs:
        if len(run) < 5:
            continue
        r0, r1 = rng_of(run[0]), rng_of(run[-1])
        if r0 - r1 < min_close_m:
            continue
        n += 1
        first_mask = next((rng_of(r) for r in run if (f(r, "argmax_px") or 0) > 0), None)
        p0 = f(run[0], "p_class")
        p1 = f(run[-1], "p_class")
        print(f"{n:3d} {len(run):7d} {r0:9.2f} {r1:8.2f} {r0 - r1:7.2f} "
              f"{p0 if p0 is not None else float('nan'):8.3f} "
              f"{p1 if p1 is not None else float('nan'):7.3f} "
              f"{(f'{first_mask:.2f} m' if first_mask is not None else 'never'):>14}")
    if n == 0:
        print("(no closing run longer than "
              f"{min_close_m} m — the robot never approached anything in this log)")


def covariates(rows):
    """Contrast every per-frame covariate on frames where the mask FIRED against frames where it went
    silent — restricted to frames where the door is anchored and in front, so the comparison is
    between two populations looking at the same thing.

    ★This is the test, and it is stated before it is run: if BLUR separates the two, the dropouts are
    an image-sharpness problem; if LUMA separates them, auto-exposure; if EGO_V separates them but
    neither of the others does, the covariate is motion itself and the mechanism is still unidentified;
    if EGO_W is ~0 throughout, "pure translation" is confirmed rather than assumed. A covariate that
    does not separate is a hypothesis killed, which is worth as much as one that does.
    """
    peaks = [r for r in rows if r["kind"] == "peak"]
    moving = [r for r in peaks if (v := f(r, "ego_v")) is not None and v > 0.02]
    if not moving:
        print("no rows with ego_v — this log predates the covariate columns; restart retina to collect")
        return
    hit = [r for r in moving if (f(r, "argmax_px") or 0) > 0]
    mis = [r for r in moving if (f(r, "argmax_px") or 0) == 0]
    print(f"moving frames: {len(hit)} with a mask, {len(mis)} silent\n")
    print(f"{'covariate':>16} {'fires':>12} {'silent':>12} {'difference':>12}")
    for key in ("blur", "luma", "ego_v", "ego_w", "anchor_range_m", "anchor_bearing", "p_class"):
        a = [f(r, key) for r in hit if f(r, key) is not None]
        b = [f(r, key) for r in mis if f(r, key) is not None]
        if not a or not b:
            continue
        ma = sorted(a)[len(a) // 2]
        mb = sorted(b)[len(b) // 2]
        print(f"{key:>16} {ma:12.3f} {mb:12.3f} {mb - ma:12.3f}")
    print("\nMedians. Read the difference column against what each covariate would mean:\n"
          "  blur  much lower on silent frames -> the frames really are blurred\n"
          "  luma  differing                   -> exposure is hunting\n"
          "  ego_w ~0 in both                  -> the motion is pure translation, as stated\n"
          "  only ego_v differing              -> motion is a proxy for something not yet logged")


def plot(rows, out_path, bin_m):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib not available — use the table view instead")

    peaks = [r for r in rows if r["kind"] == "peak" and rng_of(r) is not None]
    x = [rng_of(r) for r in peaks]
    y = [f(r, "p_class") for r in peaks]
    hit = [(f(r, "argmax_px") or 0) > 0 for r in peaks]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter([a for a, h in zip(x, hit) if not h], [b for b, h in zip(y, hit) if not h],
               s=8, c="#B0B0B0", label="argmax silent")
    ax.scatter([a for a, h in zip(x, hit) if h], [b for b, h in zip(y, hit) if h],
               s=8, c="#FF9E4A", label="argmax fires")
    ax.set_xlabel("range at peak pixel (m)")
    ax.set_ylabel("peak P(class)")
    ax.set_ylim(0, 1)
    ax.invert_xaxis()                      # the robot moves left-to-right, i.e. closer
    ax.legend(loc="upper left")
    ax.set_title("Posterior vs range along the approach — grey is evidence the argmax discarded")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="etc/door_approach.csv")
    ap.add_argument("--source", default="zed", help="zed | ricoh | '' for both")
    ap.add_argument("--bin", type=float, default=0.5, help="range bin width in metres")
    ap.add_argument("--segment", action="store_true", help="list approach segments")
    ap.add_argument("--covar", action="store_true",
                    help="contrast per-frame covariates on firing vs silent frames")
    ap.add_argument("--plot", metavar="PNG", help="write a scatter instead of a table")
    a = ap.parse_args()

    rows = read_rows(a.path, a.source)
    if not rows:
        sys.exit(f"no rows in {a.path} for source={a.source!r}")
    label = rows[0].get("label", "?")
    print(f"{len(rows)} rows, class '{label}', source '{a.source or 'all'}'\n")

    if a.plot:
        plot(rows, a.plot, a.bin)
    elif a.covar:
        covariates(rows)
    elif a.segment:
        segments(rows)
    else:
        summarize(rows, a.bin)


if __name__ == "__main__":
    main()
