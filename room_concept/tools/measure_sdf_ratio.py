#!/usr/bin/env python3
"""Measure the median/mean ratio of the absolute SDF residual, from a localiser log.

WHY THIS EXISTS
---------------
`UpdateResult::sdf_mse` became a MEDIAN absolute residual on 2026-08-13 (it had been a MEAN on the
early-exit path, which is >98% of frames).  Every threshold calibrated against the old mean had to be
multiplied by median/mean to keep its meaning, and that factor was initially taken to be

    kMedianOverMeanAbs = 0.674 / 0.798 = 0.845,

which is the ratio for a HALF-NORMAL residual.  The residual is not half-normal: roughly two thirds
of the floor is systematic map mismatch — furniture, doorway gaps, wall thickness the polygon does
not model — rather than zero-mean noise.  A systematic component pushes the mean and the median
TOWARDS each other (in the limit of a pure offset the ratio is 1), so 0.845 is expected to be an
OVER-correction, and the thresholds derived with it are expected to be slightly too tight.

Rather than assume the distribution, measure it.  On an early-exit frame the log carries both
statistics of the SAME residuals at the SAME pose:

    sdf_mse            median |sdf| at the predicted pose
    early_exit_metric  mean   |sdf| at the predicted pose

so their ratio, frame by frame, IS the correction factor — no new instrumentation, and no assumption
about the shape of the distribution.

USAGE
-----
    python3 tools/measure_sdf_ratio.py tmp/sdf_localizer/log_*.csv

Use a log produced AFTER 2026-08-13.  On an older log `sdf_mse` is itself the mean on early-exit
frames, so the ratio comes out at 1.000 and the script says so instead of reporting a result.

NOTE ON READING THESE LOGS
--------------------------
Rows can be ragged and the header has historically drifted out of step with the writer, so columns
are resolved by NAME and short rows are skipped and counted.  Only early-exit frames are used, since
those are the ones on which the two columns describe the same residual set.
"""

import sys
import glob
import statistics as st

# Thresholds that were rescaled by the provisional factor, with their ORIGINAL (pre-2026-08-13)
# values.  The script reprints them at the measured ratio so the two can be compared directly.
ORIGINAL_THRESHOLDS = [
    ("DSR.StableSdfMseMax",                       0.09),
    ("RoomConcept.SymmetryGoodFitMse",            0.12),
    ("RoomConcept.BoundaryHessianQualityThreshold", 0.06),
    ("RoomConcept.BoundaryMuQualityThreshold",    0.10),
    ("EpistemicController.SdfSafe",               0.06),
    ("EpistemicController.SdfDanger",             0.10),
]
PROVISIONAL = 0.674 / 0.798


def read_pairs(path):
    """Yield (median, mean) for every early-exit frame with both columns finite and positive."""
    pairs, ragged, skipped = [], 0, 0
    with open(path, newline="") as fh:
        header = fh.readline().rstrip("\n").split(",")
        try:
            i_med = header.index("sdf_mse")
            i_mean = header.index("early_exit_metric")
            i_ee = header.index("early_exit")
        except ValueError as exc:
            raise SystemExit(f"{path}: missing column ({exc}). Is this a localiser log?")
        width = len(header)
        for line in fh:
            row = line.rstrip("\n").split(",")
            if len(row) < width:
                ragged += 1
                continue
            try:
                if row[i_ee].strip() not in ("1", "true", "True"):
                    skipped += 1
                    continue
                med, mean = float(row[i_med]), float(row[i_mean])
            except ValueError:
                ragged += 1
                continue
            if med > 0 and mean > 0 and med == med and mean == mean:
                pairs.append((med, mean))
            else:
                skipped += 1
    return pairs, ragged, skipped


def main(paths):
    pairs, ragged, skipped = [], 0, 0
    for p in paths:
        a, b, c = read_pairs(p)
        pairs += a
        ragged += b
        skipped += c

    if len(pairs) < 200:
        raise SystemExit(f"only {len(pairs)} usable early-exit frames — too few to trust. "
                         "Use a longer run, and one with the robot actually moving.")

    ratios = [m / n for m, n in pairs]
    ratios.sort()
    r_med = st.median(ratios)
    q05 = ratios[int(0.05 * len(ratios))]
    q95 = ratios[int(0.95 * len(ratios))]

    print(f"frames used          {len(pairs)}   (ragged {ragged}, skipped {skipped})")
    print(f"median|sdf| / mean|sdf|")
    print(f"  median of ratio    {r_med:.4f}")
    print(f"  5th–95th pct       {q05:.4f} – {q95:.4f}")
    print(f"  provisional value  {PROVISIONAL:.4f}   (half-normal assumption)")

    if r_med > 0.995:
        print("\nThe ratio is ~1: this log predates the 2026-08-13 change, so both columns hold the "
              "mean.\nRe-run on a newer log — this result says nothing about the distribution.")
        return

    print(f"\nkMedianOverMeanAbs should be {r_med:.3f} (room_concept.cpp).")
    print("Thresholds re-derived from their ORIGINAL mean-calibrated values:\n")
    print(f"  {'key':<46}{'original':>10}{'provisional':>13}{'measured':>11}")
    for key, original in ORIGINAL_THRESHOLDS:
        print(f"  {key:<46}{original:>10.3f}{original*PROVISIONAL:>13.3f}{original*r_med:>11.3f}")
    print("\nThe spread above is the frame-to-frame spread of the ratio, not the uncertainty of its "
          "median;\nwith this many frames the median is far better determined than that range "
          "suggests.")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    files = [f for a in args for f in glob.glob(a)] or args
    main(files)
