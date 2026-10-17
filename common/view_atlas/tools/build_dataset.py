#!/usr/bin/env python3
"""Labelled views dataset for the view atlas: YOLOE masks -> ground-truth instance by casting the mask's
bottom-contact pixel onto the floor into the Webots furniture footprints (labeller from the 2026-10-17
spike, colour-verified), then the SAME crop + DINOv3 + colour histogram the C++ module computes
(common/view_atlas/view_crop.h). DINOv3 runs through onnxruntime on the exported retina/models/dinov3/
dinov3_vits16_crop224.onnx (the model the C++ encoder loads, parity 1.000000), not transformers, so the
fitted constants describe exactly the agent's descriptor. YOLOE runs on the GPU via ultralytics. Two sources:
  zed_frames : retina ZedDump of an apartment tour (gt_* = robot, ZED 0.075 m behind along forward)
  zed_orbits : orbit recorder (same CSV convention; lap 1 moves the yellow chair)
usage (from active_inference/):
  python3 common/view_atlas/tools/build_dataset.py --out retina/etc/view_atlas_dataset
"""
import argparse, csv, importlib.util, json, math, os, sys
import numpy as np, pandas as pd, cv2
import onnxruntime as ort
from shapely.geometry import shape, Point

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RETINA = os.path.join(ROOT, "retina")
F, CX, CY, CAM_Z, MOUNT_Y = 448.0, 640.0, 360.0, 0.9775, -0.075
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
OK = {"RoundTable": {"table", "dining table", "coffee table", "desk"}, "Table": {"table", "dining table", "coffee table", "desk"},
      "Desk": {"desk", "table", "workbench"}, "desk": {"desk", "table", "workbench"},
      "Chair": {"chair", "stool", "armchair"}, "chair": {"chair", "stool", "armchair"}, "Fridge": {"refrigerator"},
      "Worktop": {"countertop", "kitchen island", "cabinet", "shelf", "sink", "stove", "oven", "dishwasher"},
      "BookshelfCubes": {"bookshelf", "shelf", "cube shelf", "cabinet"}, "Pillar": {"column"}, "Heater": {"radiator"}}
# Object yaw (world) per lap: only the chairs' yaw matters (the only object that moves and turns);
# static objects keep yaw 0 because only bearing DIFFERENCES within one object are used for them.
YAW = {("Chair", 0): 1.0472, ("Chair", 1): math.pi / 2, ("chair(1)", 0): -2.8797953, ("chair(1)", 1): -2.8797953}

def ok_labels(n): return next((v for k, v in OK.items() if n.startswith(k)), set())
def cls_of(n):
    for k, c in [("RoundTable", "table"), ("Table", "table"), ("Desk", "desk"), ("desk", "desk"), ("Chair", "chair"),
                 ("chair", "chair"), ("Fridge", "refrigerator"), ("Worktop", "worktop"), ("Bookshelf", "bookshelf"),
                 ("Pillar", "pillar"), ("Heater", "heater")]:
        if n.startswith(k): return c
    return n.lower()

def make_crop(bgr, mask):
    """Mirror of view_crop.h::make_crop. mask: bool HxW."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0: return None
    H, W = mask.shape
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    border = np.count_nonzero((xs == 0) | (ys == 0) | (xs == W - 1) | (ys == H - 1))
    cx, cy = 0.5 * (x0 + x1 + 1), 0.5 * (y0 + y1 + 1)
    s = 1.15 * max(x1 - x0 + 1, y1 - y0 + 1)
    X1, Y1 = int(max(cx - s / 2, 0.0)), int(max(cy - s / 2, 0.0))
    X2, Y2 = int(min(cx + s / 2, float(W))), int(min(cy + s / 2, float(H)))
    X2, Y2 = max(X2, X1 + 1), max(Y2, Y1 + 1)
    crop = bgr[Y1:Y2, X1:X2].copy(); crop[~mask[Y1:Y2, X1:X2]] = (104, 116, 124)
    rgb = cv2.cvtColor(cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    box, mbox = bgr[y0:y1 + 1, x0:x1 + 1], mask[y0:y1 + 1, x0:x1 + 1]
    return rgb, box, mbox, len(xs), border / len(xs)

def colour_hist(box, mbox):
    """Mirror of view_crop.h::colour_hist."""
    hsv = cv2.cvtColor(box, cv2.COLOR_BGR2HSV)[mbox]
    h = np.zeros(64, np.float32)
    if len(hsv): np.add.at(h, (hsv[:, 0].astype(int) * 16 // 180) * 4 + hsv[:, 1].astype(int) * 4 // 256, 1.0)
    h = np.sqrt(h); n = np.linalg.norm(h)
    return h / n if n > 0 else h

def sources(a):
    yield "frames", os.path.join(RETINA, "etc", "zed_frames"), {0: os.path.join(ROOT, "room_concept/estimation_scripts/apartment/furniture.json")}
    o = os.path.join(RETINA, "etc", "zed_orbits")
    yield "orbits", o, {0: os.path.join(o, "furniture_lap0.json"), 1: os.path.join(o, "furniture_lap1.json")}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    spec = importlib.util.spec_from_file_location("exp", os.path.join(RETINA, "tools", "export_yoloe_household.py"))
    exp = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp); VOCAB = exp.VOCAB
    from ultralytics import YOLOE
    cwd = os.getcwd(); os.chdir(os.path.join(RETINA, "models/yolo26/_yoloe_work"))
    yolo = YOLOE("yoloe-26x-seg.pt"); yolo.set_classes(VOCAB); os.chdir(cwd)
    dino = ort.InferenceSession(os.path.join(RETINA, "models", "dinov3", "dinov3_vits16_crop224.onnx"), providers=["CPUExecutionProvider"])

    rows, S, C = [], [], []
    for ds, d, furn_files in sources(a):
        poses = pd.read_csv(os.path.join(d, "poses.csv"))
        poses = poses[[os.path.exists(os.path.join(d, f)) for f in poses.file]].reset_index(drop=True)
        if "lap" not in poses: poses["lap"] = 0
        if a.limit: poses = poses.iloc[:a.limit]
        furn = {lap: [f for f in json.load(open(p)) if ok_labels(f["name"])] for lap, p in furn_files.items()}
        names = [f["name"] for f in furn[0]]
        polys = {lap: [shape(f["geometry"]) for f in fs] for lap, fs in furn.items()}
        for fi, r in poses.iterrows():
            img = cv2.imread(os.path.join(d, r.file)); lap = int(r.lap)
            h = -r.gt_angle; fwd = np.array([math.cos(h), math.sin(h)]); right = np.array([math.sin(h), -math.cos(h)])
            cam = np.array([r.gt_x, r.gt_y]) + MOUNT_Y * fwd
            res = yolo.predict(img, conf=0.25, verbose=False, retina_masks=True)[0]
            if res.masks is None: continue
            for k, m in enumerate(res.masks.data.cpu().numpy() > 0.5):
                lab = VOCAB[int(res.boxes.cls[k])]; ys, xs = np.nonzero(m)
                if len(ys) < 50: continue
                vb = int(np.percentile(ys, 99))
                if vb >= img.shape[0] - 4: continue                          # bottom cut: contact pixel unseen
                ub = float(np.median(xs[ys >= vb - 6])); dz = -(vb - CY) / F
                if dz >= -1e-3: continue                                     # bottom above the horizon
                p = cam + (CAM_Z / -dz) * ((ub - CX) / F * right + fwd)
                cand = [j for j, P in enumerate(polys[lap]) if lab in ok_labels(names[j]) and P.distance(Point(p)) < 0.25]
                if len(cand) != 1: continue
                j = cand[0]; nm = names[j]
                cr = make_crop(img, m)
                if cr is None: continue
                rgb, box, mbox, npx, trunc = cr
                x = (((rgb.astype(np.float32) / 255) - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32).copy()
                v = dino.run(["embedding"], {"pixel_values": x})[0][0]      # already L2-normalised
                ctr = np.array(polys[lap][j].centroid.coords[0])
                # inst = the physical object name: both recordings are the SAME world, so one name = one object (dataset column = origin)
                rows.append(dict(view_id=len(rows), dataset=ds, stamp_ms=int(r.stamp_ms), lap=lap, inst=nm,
                                 inst_name=nm, cls=cls_of(nm), cam_x=cam[0], cam_y=cam[1], obj_x=ctr[0], obj_y=ctr[1],
                                 obj_yaw=YAW.get((nm, lap), 0.0), range_m=float(np.linalg.norm(ctr - cam)), mask_px=npx, trunc_frac=trunc))
                S.append(v.astype("<f4")); C.append(colour_hist(box, mbox).astype("<f4"))
            if fi % 200 == 0: print(f"{ds} {fi}/{len(poses)} views={len(rows)}", flush=True)
    with open(os.path.join(a.out, "views.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    np.stack(S).tofile(os.path.join(a.out, "shape.f32")); np.stack(C).tofile(os.path.join(a.out, "colour.f32"))
    print("views:", len(rows)); print(pd.DataFrame(rows).groupby(["dataset", "inst"]).size().to_string())

if __name__ == "__main__":
    main()
