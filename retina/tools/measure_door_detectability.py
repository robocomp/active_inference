#!/usr/bin/env python3
"""
measure_door_detectability.py - does the perception stack know about the apartment's doors?

THE QUESTION. door_concept's only removal channel is the ZED silhouette, and the only thing that
can hold a door alive is a mask labelled "door". "door" is not a COCO class, so it comes solely
from the ADE20K semantic path - whose export collapses a 150-way softmax to a uint8 argmax before
onnxruntime ever sees it. A door that loses NARROWLY to "wall" (ADE20K class 0, the most frequent
class in the set) is then indistinguishable from a door the model never saw.

So there are two candidate cures and they call for opposite work:

  (a) NARROW LOSS  -> the posterior is the fix. Restore it (tools/expose_semantic_logits.py) and
                      let the door channel consume P(door|pixel) as a continuous weight.
  (b) GENUINELY BLIND -> no belief change helps, and the answer is a different model. One already
                      exists: beta_robotica_class_private/door_learner, YOLO26x fine-tuned on three
                      merged door datasets, mAP50 0.913 on ITS OWN held-out split - which is not
                      this apartment.

This tool measures both on the SAME frames, so the comparison is like-for-like.

MODE "semantic": replicates what SemanticMaskStage actually does (morph open+close k=5, connected
components, drop below mask_min_area_frac of the pixels LOOKED AT - not of the canvas) and then
re-runs the identical pipeline on a P(door) threshold sweep. The headline number is the frames that
currently yield NO door mask but would yield one from the posterior.

MODE "detector": runs the fine-tuned door model over the same frames. Handles both export layouts -
yolo26 end2end [1,300,6] (the layout retina's parse_detections_end2end already reads) and yolo11
[1,7,8400] (raw, needs NMS).

★A THRESHOLD SWEEP IS NOT A THRESHOLD PROPOSAL. The sweep exists to show the SHAPE of the
distribution. If recovery only appears in a narrow band of t, that is evidence of (b) dressed as
(a), and the honest read is "the signal is not there". Nothing here is meant to become a
configured cutoff; the design intent downstream is a continuous weight.

Usage:
    python3 tools/measure_door_detectability.py semantic [--limit N] [--stride N]
    python3 tools/measure_door_detectability.py detector [--model yolo26|yolo11] [--limit N]
"""

import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort

FRAMES = "etc/depth_frames/*.jpg"
SEM_PROBS = "models/yolo26/yolo26l-sem-ade20k-probs.onnx"
DOOR_MODELS = {
    "yolo26": "/home/pbustos/robocomp/components/beta_robotica_class_private/door_learner/"
              "runs/train/doors_yolo26/weights/best.onnx",
    "yolo11": "/home/pbustos/robocomp/components/beta_robotica_class_private/door_learner/"
              "runs/train/doors_yolo11/weights/best.onnx",
    # YOLO11s re-exported with nms=True -> [1,300,6], the layout parse_detections_end2end reads.
    # Verified same semantics as the yolo26x export: xyxy in 640-letterbox px, conf-sorted, 300 rows.
    "yolo11s_e2e": "models/doors/doors_yolo11s_end2end.onnx",
}
DOOR_NAMES = ["open_door", "closed_door", "semi_door"]

# Channel order written by expose_semantic_logits.py. Keep in step with its --classes default.
PROB_CLASSES = ["wall", "cabinet", "door", "shelf", "hood"]
ADE_DOOR, ADE_WALL = 14, 0

# Mirrors [Semantic] in etc/config.toml.
MIN_AREA_FRAC = 0.003
MORPH_KERNEL = 5
INPUT = 640


def letterbox(im, s=INPUT):
    h, w = im.shape[:2]
    sc = min(s / h, s / w)
    nh, nw = int(round(h * sc)), int(round(w * sc))
    out = np.full((s, s, 3), 114, np.uint8)
    pt, pl = (s - nh) // 2, (s - nw) // 2
    out[pt:pt + nh, pl:pl + nw] = cv2.resize(im, (nw, nh))
    return out, sc, pl, pt, nw, nh


def to_input(lb):
    return cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None] / 255.0


def components(binary, min_area):
    """SemanticMaskStage's pipeline: denoise, label, keep components at or above min_area."""
    if MORPH_KERNEL > 1:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_KERNEL, MORPH_KERNEL))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    return [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)
            if stats[i, cv2.CC_STAT_AREA] >= min_area]


def frame_list(args):
    files = sorted(glob.glob(FRAMES))
    if args.stride > 1:
        files = files[::args.stride]
    if args.limit:
        files = files[:args.limit]
    return files


def run_semantic(args):
    if not os.path.exists(SEM_PROBS):
        sys.exit(f"missing {SEM_PROBS} - run tools/expose_semantic_logits.py first")
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(SEM_PROBS, so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    dc = PROB_CLASSES.index("door")
    wc = PROB_CLASSES.index("wall")

    thresholds = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    files = frame_list(args)
    print(f"[semantic] {len(files)} frames, model={SEM_PROBS}")

    n_argmax = 0
    n_thr = {t: 0 for t in thresholds}
    recovered = {t: 0 for t in thresholds}
    lost_p, lost_wall = [], []          # P(door) / P(wall) on argmax-wall pixels, frames w/o a door mask
    per_frame = []
    t0 = time.time()

    for i, f in enumerate(files):
        img = cv2.imread(f)
        if img is None:
            continue
        lb, _, pl, pt, nw, nh = letterbox(img)
        out = sess.run(None, {inp: to_input(lb)})
        labels = out[0][0]                                    # [640,640] uint8, letterbox frame
        probs = out[1][0].astype(np.float32)                  # [5,80,80]

        # THE DENOMINATOR IS WHAT WAS LOOKED AT (semantic_mask_stage.cpp:41), not the 640x640 canvas.
        inferred = nw * nh
        min_area = max(1, int(MIN_AREA_FRAC * inferred))

        # Restrict to the active (non-padded) region so padding cannot contribute components.
        active = np.zeros_like(labels, dtype=bool)
        active[pt:pt + nh, pl:pl + nw] = True

        door_argmax = ((labels == ADE_DOOR) & active).astype(np.uint8)
        has_argmax = len(components(door_argmax, min_area)) > 0
        n_argmax += has_argmax

        # Upsample probabilities the way yolo_semantic.cpp does for scores: INTER_LINEAR.
        pd = cv2.resize(probs[dc], (INPUT, INPUT), interpolation=cv2.INTER_LINEAR)
        pw = cv2.resize(probs[wc], (INPUT, INPUT), interpolation=cv2.INTER_LINEAR)

        row = {"file": os.path.basename(f), "argmax": bool(has_argmax),
               "pd_max": float(pd[active].max())}
        for t in thresholds:
            hit = len(components(((pd >= t) & active).astype(np.uint8), min_area)) > 0
            n_thr[t] += hit
            if hit and not has_argmax:
                recovered[t] += 1
            row[f"t{t}"] = bool(hit)
        per_frame.append(row)

        if not has_argmax:
            m = (labels == ADE_WALL) & active
            if m.any():
                # Where the model said "wall", how much door probability was actually there?
                # ★PAIRED, not two independent maxima. max(P(door)) and max(P(wall)) over the same
                # REGION land on DIFFERENT pixels, and quoting them side by side invents a margin
                # that no pixel has. Take the argmax-door pixel, then read P(wall) AT THAT PIXEL.
                idx = np.argmax(np.where(m, pd, -1.0))
                lost_p.append(float(pd.flat[idx]))
                lost_wall.append(float(pw.flat[idx]))

        if (i + 1) % 100 == 0:
            print(f"  ... {i+1}/{len(files)}  ({(time.time()-t0)/(i+1):.2f}s/frame)")

    n = len(per_frame)
    print(f"\n[semantic] {n} frames in {time.time()-t0:.0f}s\n")
    print(f"  CURRENT (argmax == door, the live behaviour):")
    print(f"    frames yielding at least one door mask : {n_argmax}/{n}  ({100*n_argmax/max(n,1):.1f}%)\n")
    print(f"  COUNTERFACTUAL (P(door) >= t, same morph + min_area):")
    print(f"    {'t':>6} {'frames with a door mask':>26} {'of which argmax had NONE':>26}")
    for t in thresholds:
        print(f"    {t:>6.2f} {n_thr[t]:>10d} ({100*n_thr[t]/max(n,1):>5.1f}%)      "
              f"{recovered[t]:>10d} ({100*recovered[t]/max(n,1):>5.1f}%)")

    if lost_p:
        a = np.array(lost_p)
        w = np.array(lost_wall)
        print(f"\n  ON FRAMES WITH NO DOOR MASK - peak P(door) among argmax-'wall' pixels (n={len(a)}):")
        for q in (10, 25, 50, 75, 90, 99):
            print(f"    p{q:<3d} {np.percentile(a, q):.4f}")
        print(f"    max  {a.max():.4f}   mean {a.mean():.4f}")
        print(f"    fraction of those frames with peak P(door) > 0.10 : {100*(a>0.10).mean():.1f}%")
        print(f"    fraction                            > 0.25 : {100*(a>0.25).mean():.1f}%")
        print(f"\n  AT THAT SAME PIXEL, how narrowly did door lose to wall?")
        print(f"    median P(wall)          : {np.median(w):.4f}")
        print(f"    median margin P(w)-P(d) : {np.median(w-a):.4f}")
        print(f"    frames where door was RUNNER-UP-CLOSE (margin < 0.10) : "
              f"{100*((w-a)<0.10).mean():.1f}%")
        print(f"    frames where wall won by a landslide  (margin > 0.50) : "
              f"{100*((w-a)>0.50).mean():.1f}%")

    out = "etc/door_detectability_semantic.json"
    with open(out, "w") as fh:
        json.dump({"n": n, "argmax_frames": n_argmax,
                   "threshold_frames": {str(k): v for k, v in n_thr.items()},
                   "recovered": {str(k): v for k, v in recovered.items()},
                   "lost_pdoor": [float(v) for v in lost_p],
                   "lost_pwall_same_pixel": [float(v) for v in lost_wall],
                   "per_frame": per_frame}, fh)
    print(f"\n  wrote {out}")


def run_detector(args):
    path = DOOR_MODELS[args.model]
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    files = frame_list(args)
    print(f"[detector] {len(files)} frames, model={args.model} ({os.path.getsize(path)//2**20} MB)")

    confs, per_frame = [], []
    n_hit = 0
    by_class = {c: 0 for c in DOOR_NAMES}
    t0 = time.time()

    for i, f in enumerate(files):
        img = cv2.imread(f)
        if img is None:
            continue
        lb, sc, pl, pt, _, _ = letterbox(img)
        raw = sess.run(None, {inp: to_input(lb)})[0]

        dets = []
        if raw.shape[1] == 300 and raw.shape[2] == 6:          # end2end: [1,300,6]
            for r in raw[0]:
                if r[4] >= args.conf:
                    dets.append((float(r[4]), int(r[5])))
        else:                                                   # yolo11 raw: [1,4+nc,8400]
            d = raw[0].T                                        # [8400, 7]
            scores = d[:, 4:]
            best = scores.max(1)
            keep = best >= args.conf
            for s, c in zip(best[keep], scores[keep].argmax(1)):
                dets.append((float(s), int(c)))

        hit = len(dets) > 0
        n_hit += hit
        for s, c in dets:
            confs.append(s)
            if 0 <= c < len(DOOR_NAMES):
                by_class[DOOR_NAMES[c]] += 1
        per_frame.append({"file": os.path.basename(f), "n": len(dets),
                          "best": max([s for s, _ in dets], default=0.0)})
        if (i + 1) % 100 == 0:
            print(f"  ... {i+1}/{len(files)}  ({(time.time()-t0)/(i+1):.2f}s/frame)")

    n = len(per_frame)
    print(f"\n[detector] {n} frames in {time.time()-t0:.0f}s  (conf >= {args.conf})\n")
    print(f"  frames with at least one door detection : {n_hit}/{n}  ({100*n_hit/max(n,1):.1f}%)")
    print(f"  total detections                        : {len(confs)}")
    print(f"  by class                                : {by_class}")
    if confs:
        a = np.array(confs)
        print(f"\n  confidence distribution (n={len(a)}):")
        for q in (10, 25, 50, 75, 90):
            print(f"    p{q:<3d} {np.percentile(a, q):.3f}")
        print(f"    max  {a.max():.3f}   mean {a.mean():.3f}")

    out = f"etc/door_detectability_detector_{args.model}.json"
    with open(out, "w") as fh:
        json.dump({"n": n, "hit_frames": n_hit, "conf": confs,
                   "by_class": by_class, "per_frame": per_frame}, fh)
    print(f"\n  wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["semantic", "detector"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--model", default="yolo26", choices=list(DOOR_MODELS))
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()
    (run_semantic if args.mode == "semantic" else run_detector)(args)


if __name__ == "__main__":
    main()
