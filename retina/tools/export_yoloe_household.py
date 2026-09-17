#!/usr/bin/env python3
"""Export YOLOE-26 (open-vocabulary seg) with retina's household vocabulary to ONNX.

The vocabulary is FROZEN into the exported weights (text embeddings are re-parameterised into the head),
so changing a name means re-running this script. The class names are written to the ONNX metadata
(`names`), which YoloSegDetector reads — no class table to keep in sync in C++.

Output signature is identical to yolo26x-seg.onnx: output0 [1,300,38] (x1,y1,x2,y2,conf,cls + 32 mask
coefficients, NMS-free) and output1 [1,32,160,160] prototypes.

    python tools/export_yoloe_household.py                 # -> models/yolo26/yoloe-26x-seg-household.onnx
    python tools/export_yoloe_household.py --size l

Bake-off 2026-09-17 on sim panoramas (memory: household-detector-bakeoff): vs COCO yolo26x-seg this finds
radiators, doors, cabinets, range hoods and kitchen islands COCO has no class for; refrigerator/microwave recall
is ~75-80% of COCO's (some of COCO's extra hits were false: brown cube shelves labelled refrigerator).
"""
import argparse
import os
import shutil
import sys

sys.modules["torchaudio"] = None  # a torchaudio built for another torch breaks the transformers import chain; unused

# Names consumed downstream are spelled exactly as consumed after YoloProcessor::normalize_yolo_label
# (table, chair, bottle, refrigerator, microwave, tv->monitor, bookshelf, radiator, cabinet, chest of drawers,
# door, range hood->hood). The rest are DISTRACTORS and must stay: an open-vocabulary head gives a region the
# closest name OFFERED, so without "workbench"/"desk" the lab benches become "table", and without
# "dining table"/"coffee table" the round table becomes "desk" (measured: table 90 crops -> 0).
VOCAB = [
    "person",
    "table", "dining table", "coffee table", "desk", "workbench", "chair", "stool", "armchair", "sofa", "bed", "bench",
    "bookshelf", "shelf", "cube shelf", "cabinet", "chest of drawers", "wardrobe",
    "kitchen island", "countertop", "sink", "refrigerator", "microwave", "oven", "stove", "range hood",
    "dishwasher", "washing machine", "kettle", "coffee machine", "trash can",
    "bottle", "cup", "mug", "glass", "bowl", "plate", "vase", "potted plant", "box", "basket",
    "door", "door handle", "window", "curtain", "radiator", "air conditioner", "light switch", "lamp", "column",
    "tv", "monitor", "laptop", "keyboard", "whiteboard", "poster", "picture frame", "clock",
    "toilet", "bathtub", "towel", "robot",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="x", choices=list("nsmlx"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from ultralytics import YOLO

    here = os.path.dirname(os.path.abspath(__file__))
    models = os.path.normpath(os.path.join(here, "..", "models", "yolo26"))
    out = args.out or os.path.join(models, f"yoloe-26{args.size}-seg-household.onnx")

    work = os.path.join(models, "_yoloe_work")          # the .pt, text encoder and export land here
    os.makedirs(work, exist_ok=True)
    os.chdir(work)
    model = YOLO(f"yoloe-26{args.size}-seg.pt")
    model.set_classes(VOCAB)
    path = model.export(format="onnx", imgsz=640, dynamic=False, simplify=True)
    shutil.move(path, out)
    print(f"exported {len(VOCAB)} classes -> {out}")


if __name__ == "__main__":
    main()
