#!/usr/bin/env python3
"""Golden crop fixture: 20 real YOLOE masks from the orbit recording, with the PYTHON mirror's crop and
colour histogram (build_dataset.py::make_crop / colour_hist). crop_golden.cpp checks the C++ crop against
them and checks decimated masks (retina's stride-2 sampling) against dense ones.
usage (from active_inference/): python3 common/view_atlas/tools/export_crop_golden.py"""
import importlib.util, os, sys
sys.modules["torchaudio"] = None
import numpy as np, cv2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
spec = importlib.util.spec_from_file_location("bd", os.path.join(ROOT, "common/view_atlas/tools/build_dataset.py"))
bd = importlib.util.module_from_spec(spec); spec.loader.exec_module(bd)
spec = importlib.util.spec_from_file_location("exp", os.path.join(ROOT, "retina/tools/export_yoloe_household.py"))
exp = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)

def main():
    frames = os.path.join(ROOT, "retina/etc/zed_orbits"); out = os.path.join(ROOT, "common/view_atlas/tools/crop_golden")
    os.makedirs(out, exist_ok=True)
    from ultralytics import YOLOE
    cwd = os.getcwd(); os.chdir(os.path.join(ROOT, "retina/models/yolo26/_yoloe_work"))
    yolo = YOLOE("yoloe-26x-seg.pt"); yolo.set_classes(exp.VOCAB); os.chdir(cwd)
    files = sorted(f for f in os.listdir(frames) if f.endswith(".jpg") and "_depth" not in f)[::40]
    rows, cols = [], []
    for f in files:
        img = cv2.imread(os.path.join(frames, f))
        r = yolo.predict(img, conf=0.25, verbose=False, retina_masks=True)[0]
        if r.masks is None: continue
        for m in r.masks.data.cpu().numpy() > 0.5:
            if m.sum() < 2000 or len(rows) >= 20: continue
            k = len(rows)
            rgb, box, mbox, npx, trunc = bd.make_crop(img, m)
            cv2.imwrite(os.path.join(out, f"mask_{k:02d}.png"), m.astype(np.uint8) * 255)
            cv2.imwrite(os.path.join(out, f"py_crop_{k:02d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cols.append(bd.colour_hist(box, mbox).astype("<f4")); rows.append(f)
        if len(rows) >= 20: break
    with open(os.path.join(out, "masks.txt"), "w") as fh:
        for k, f in enumerate(rows): fh.write(f"{k} {f}\n")
    np.stack(cols).tofile(os.path.join(out, "py_colour.f32"))
    print("golden masks:", len(rows))

if __name__ == "__main__":
    main()
