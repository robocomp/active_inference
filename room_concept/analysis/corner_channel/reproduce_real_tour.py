#!/usr/bin/env python3
"""Re-derive the paper's physical-robot numbers (§V-E, Table V row "physical robot") from RAW sessions.

    python3 analysis/corner_channel/reproduce_real_tour.py <session_dir> [<session_dir> ...] \
            [--combined <robot's combined corner_probe.csv>]

Session dirs IN TOUR ORDER, each holding the corner_probe.csv (and ideally pose_trace.csv) copied out
of that session. The numbers in the paper were computed ON THE ROBOT, from a combined file this machine
never saw. This script repeats every step here, independently, so the table does not rest on a merge
nobody else ran:

  1. MERGE. Each session's probe starts at frame 0, and split_and_nis.py keys on frame, so frames are
     offset per session (offset = running max + 1). If the robot's combined file is given, the merged
     rows are compared with it as a multiset — a different offset convention is tolerated, a different
     row set is not.
  2. ANALYSE with the committed split_and_nis.py, unmodified, on the merged file.
  3. BOUNDARY CHECK. split_and_nis.py builds stretches over CONSECUTIVE frames in sorted order, so a
     session's last frame and the next session's first frame are compared as if adjacent. If the robot
     was parked across a restart, a stretch could span two sessions. The merged stretch count must equal
     the sum of per-session counts; otherwise a stretch crossed a boundary and the pairs are suspect.
  4. INDEX NOTE. model_index is an index into the loaded layout's vertex list (corner_detector.h), so it
     means the same corner across sessions ONLY if every session loaded the same layout SVG. No index
     statistic can prove that; per-session corner sets are printed and the SVG must be checked by hand.
  Distance is printed raw AND with physically impossible steps removed; see path_length().
  5. COMPARE each quoted quantity with the paper, to its printed precision.
"""
import os, re, subprocess, sys, math
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.join(HERE, "split_and_nis.py")

# What main.tex quotes for the physical robot (lines ~582-617). Value, and the tolerance its printed
# precision implies (half the last printed digit).
PAPER = {
    "nis_accepted":  (0.836, 0.0005),
    "stretches":     (6, 0),
    "pairs":         (46, 0),
    "repeat_m":      (0.0624, 0.00005),
    "scatter_m":     (0.0046, 0.00005),
    "ratio":         (13.4, 0.05),
    "bias_gt_pct":   (98, 0.5),
    "vp_corners":    (10, 0),
    "within_m":      (0.0046, 0.00005),
    "between_m":     (0.0459, 0.00005),
    "vp_ratio":      (9.9, 0.05),
    "distance_m":    (295.10, 0.005),
}

PATTERNS = {
    "nis_accepted": r"NIS/dof overall:\s*([\d.]+)",
    "stretches":    r"2\. stationary stretches .*?:\s*(\d+) of",
    "pairs":        r"(\d+) corner-stretch pairs",
    "repeat_m":     r"repeats \(bias\)\s+median ([\d.]+)",
    "scatter_m":    r"scatters \(noise\) median ([\d.]+)",
    "ratio":        r"ratio ([\d.]+)x",
    "bias_gt_pct":  r"bias exceeds scatter in (\d+)%",
    "vp_corners":   r"(\d+) corners seen in >=3 stretches",
    "within_m":     r"within-stretch scatter\s+([\d.]+)",
    "between_m":    r"between-stretch spread\s+([\d.]+)",
    "vp_ratio":     r"\(([\d.]+)x within\)",
}


def run_analysis(path):
    out = subprocess.run([sys.executable, ANALYSIS, path], capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"split_and_nis.py failed on {path}:\n{out.stderr}")
    return out.stdout


def parse(text):
    got = {}
    for k, pat in PATTERNS.items():
        m = re.search(pat, text)
        if m:
            got[k] = float(m.group(1))
    return got


def read_probe(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        header = f.readline().rstrip("\n")
        ncol = len(header.split(","))
        rows = [l.rstrip("\n").split(",") for l in f]
    return header, [r for r in rows if len(r) == ncol]   # drop a truncated last line, as the analysis does


def path_length(pose_trace):
    """Distance from the robot pose trace, two ways, because they disagree.

    RAW sums |dxy| over consecutive corrected poses (type 0). It counts every relocalisation snap as
    travel. Measured 2026-09-14 on a 6-min Webots run: raw 310 m, of which 282 m came from 76 steps
    faster than 5 m/s (one 21 m; a run of 4.47 m every 50 ms) — actual driving was ~22 m. A raw sum of
    that kind is what the paper's tour length must be checked against. FEASIBLE drops each step whose speed,
    over its valid_ts interval, exceeds V_FEASIBLE — twice the controller's configured 0.5 m/s, so real
    driving is never cut. That cutoff is a diagnostic of the LOG, not a model term; both are printed.
    """
    V_FEASIBLE = 1.0
    raw = feas = 0.0; jmax = 0.0; prev = None
    with open(pose_trace, encoding="utf-8", errors="ignore") as f:
        head = f.readline().rstrip("\n").split(",")
        it, its, ix, iy = head.index("type"), head.index("valid_ts_ms"), head.index("x"), head.index("y")
        for line in f:
            p = line.rstrip("\n").split(",")
            if len(p) != len(head) or p[it] != "0":
                continue
            t, x, y = int(p[its]), float(p[ix]), float(p[iy])
            if prev is not None:
                d = math.hypot(x - prev[1], y - prev[2]); dt = (t - prev[0]) / 1000.0
                raw += d; jmax = max(jmax, d)
                if dt > 0 and d / dt <= V_FEASIBLE:
                    feas += d
            prev = (t, x, y)
    return raw, feas, jmax


def main(argv):
    combined = None
    if "--combined" in argv:
        i = argv.index("--combined")
        combined = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    sessions = argv
    if not sessions:
        sys.exit(__doc__)

    # ── 1 · merge ─────────────────────────────────────────────────────────────────────────────
    header0, merged, offset = None, [], 0
    per_session_stretches, per_session_idx, distance, dist_feas, dist_known = [], [], 0.0, 0.0, True
    print("== sessions ==")
    for s in sessions:
        probe = os.path.join(s, "corner_probe.csv")
        header, rows = read_probe(probe)
        if header0 is None:
            header0 = header
        elif header != header0:
            sys.exit(f"header of {probe} differs from the first session's: cannot merge")
        cols = header.split(",")
        F, M = cols.index("frame"), cols.index("model_index")
        frames = [int(r[F]) for r in rows]
        per_session_idx.append({int(r[M]) for r in rows})
        for r in rows:
            r = list(r); r[F] = str(int(r[F]) + offset); merged.append(r)
        stats = parse(run_analysis(probe))
        per_session_stretches.append(int(stats.get("stretches", 0)))
        pt = os.path.join(s, "pose_trace.csv")
        if os.path.exists(pt):
            d, dfeas, jmax = path_length(pt)
            distance += d; dist_feas += dfeas
            dist_txt = f"raw {d:.2f} m, feasible {dfeas:.2f} m, largest step {jmax:.2f} m"
        else:
            dist_known, dist_txt = False, "no pose_trace.csv"
        print(f"  {s}: {len(rows)} rows, frames {min(frames)}..{max(frames)} -> offset {offset}, "
              f"stretches {per_session_stretches[-1]}, corners {len(per_session_idx[-1])} (max index {max(per_session_idx[-1])}), {dist_txt}")
        offset += max(frames) + 1

    out_path = os.path.join(os.path.dirname(os.path.abspath(sessions[0].rstrip("/"))),
                            "merged_here_corner_probe.csv")
    with open(out_path, "w") as f:
        f.write(header0 + "\n")
        f.writelines(",".join(r) + "\n" for r in merged)
    print(f"  merged: {len(merged)} rows, {len({r[0] for r in merged})} frames -> {out_path}")

    # ── 1b · against the robot's combined file ───────────────────────────────────────────────
    if combined:
        h, crow = read_probe(combined)
        ours = Counter(",".join(r[1:]) for r in merged)      # frame numbering excluded: offsets may differ
        theirs = Counter(",".join(r[1:]) for r in crow)
        same_frames = sorted({r[0] for r in merged}, key=int) == sorted({r[0] for r in crow}, key=int)
        print(f"\n== robot's combined file == {len(crow)} rows; header {'same' if h == header0 else 'DIFFERENT'}; "
              f"row multiset {'IDENTICAL' if ours == theirs else 'DIFFERS'}; "
              f"frame numbering {'identical' if same_frames else 'differs (offset convention)'}")
        if ours != theirs:
            print(f"   only here: {sum((ours - theirs).values())} rows, only on robot: "
                  f"{sum((theirs - ours).values())} rows")

    # ── 2 · analyse ───────────────────────────────────────────────────────────────────────────
    text = run_analysis(out_path)
    print("\n== split_and_nis.py on the merge ==\n" + text)
    got = parse(text)
    if dist_known:
        got["distance_m"] = distance
        print(f"  distance: raw {distance:.2f} m, feasible-steps-only {dist_feas:.2f} m "
              f"({100 * (distance - dist_feas) / max(distance, 1e-9):.0f}% of the raw sum is impossible steps)")

    # ── 3/4 · checks ──────────────────────────────────────────────────────────────────────────
    print("== checks ==")
    ok_b = int(got.get("stretches", -1)) == sum(per_session_stretches)
    print(f"  boundary: merged stretches {int(got.get('stretches', -1))} vs per-session sum "
          f"{sum(per_session_stretches)} -> {'OK' if ok_b else 'A STRETCH CROSSES A SESSION BOUNDARY'}")
    # A session that never sees a corner has a lower max index, so no index statistic can prove the
    # sessions shared a layout; only the SVG each loaded can. Printed as information, not a verdict.
    shared = set.intersection(*per_session_idx) if per_session_idx else set()
    print(f"  layout index (information only): corners per session {[len(x) for x in per_session_idx]}, "
          f"seen in every session {len(shared)} — confirm every session loaded the SAME layout SVG")

    # ── 5 · against the paper ─────────────────────────────────────────────────────────────────
    print("\n== against main.tex ==")
    bad = 0
    for k, (want, tol) in PAPER.items():
        if k not in got:
            print(f"  {k:>13}: paper {want}  here —  (not produced)"); bad += 1; continue
        ok = abs(got[k] - want) <= tol + 1e-12
        bad += not ok
        print(f"  {k:>13}: paper {want:<8} here {got[k]:<10.4f} {'ok' if ok else 'MISMATCH'}")
    print(f"\n{'ALL QUOTED NUMBERS REPRODUCE' if bad == 0 and ok_b else f'{bad} MISMATCH(ES) — do not submit these numbers unexamined'}")
    return 0 if bad == 0 and ok_b else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
