#!/usr/bin/env python3
"""
expose_semantic_logits.py - recover the per-class posterior the ADE20K export throws away.

WHY. yolo26*-sem-ade20k.onnx has ONE output: output0, uint8 [1,640,640]. Walking the graph back
from it:

    Conv classifier.1 -> [1,150,80,80] logits -> Resize -> [1,150,640,640] -> ArgMax -> Cast -> uint8

The 150-way softmax is collapsed INSIDE the graph. So a door pixel at P(door)=0.45 losing to
P(wall)=0.47 is emitted as "wall", indistinguishable from P(door)=0.001 - and "wall" is ADE20K
class 0, the most frequent class in the dataset. That is not a thresholding choice we can tune
from the outside; the information is destroyed before onnxruntime returns.

Downstream this is visible as SemanticMap::scores == 1.0 for every pixel
(retina/src/yolo_semantic.cpp, the dense-int branch), which is why "confidence" carries ZERO
information for every semantic-path agent (door, hood, cabinet, shelf), and why
[Semantic].conf_thresh - documented as a "per-pixel argmax-softmax floor" - is an inoperative
knob: there is no softmax left to floor.

WHAT THIS DOES. Pure graph surgery on the existing .onnx with the onnx package - no .pt (which is
not on disk), no ultralytics, no retraining. It appends

    Softmax(axis=1) -> Gather(axis=1, accepted ids) -> Cast(fp16) -> new output "class_probs"

on the LOGIT tensor, and leaves output0 byte-identical. Softmax must precede the Gather or the
values are not posteriors (normalisation is over all 150 classes).

WHY IT IS CHEAP. The tap is at the classifier's native 80x80, BEFORE the Resize to 640x640. A full
[1,150,640,640] float volume would be 245 MB/frame and is not an option; K=5 classes at 80x80 in
fp16 is 5*80*80*2 = 64 kB/frame. The consumer resizes, exactly as it already does for the labels.

Usage:
    python3 tools/expose_semantic_logits.py [--model PATH] [--out PATH] [--classes wall,door,...]
"""

import argparse
import sys

import onnx
from onnx import TensorProto, helper

# ADE20K SceneParse150, model class-id order. Mirrors
# retina/src/yolo_semantic.cpp::default_class_names() - keep the two in step.
ADE20K = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed", "windowpane", "grass",
    "cabinet", "sidewalk", "person", "earth", "door", "table", "mountain", "plant", "curtain", "chair",
    "car", "water", "painting", "sofa", "shelf", "house", "sea", "mirror", "rug", "field",
    "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp", "bathtub", "railing", "cushion",
    "base", "box", "column", "signboard", "chest of drawers", "counter", "sand", "sink", "skyscraper", "fireplace",
    "refrigerator", "grandstand", "path", "stairs", "runway", "case", "pool table", "pillow", "screen door", "stairway",
    "river", "bridge", "bookcase", "blind", "coffee table", "toilet", "flower", "book", "hill", "bench",
    "countertop", "stove", "palm", "kitchen island", "computer", "swivel chair", "boat", "bar", "arcade machine", "hovel",
    "bus", "towel", "light", "truck", "tower", "chandelier", "awning", "streetlight", "booth", "television receiver",
    "airplane", "dirt track", "apparel", "pole", "land", "bannister", "escalator", "ottoman", "bottle", "buffet",
    "poster", "stage", "van", "ship", "fountain", "conveyer belt", "canopy", "washer", "plaything", "swimming pool",
    "stool", "barrel", "basket", "waterfall", "tent", "bag", "minibike", "cradle", "oven", "ball",
    "food", "step", "tank", "trade name", "microwave", "pot", "animal", "bicycle", "lake", "dishwasher",
    "screen", "blanket", "sculpture", "hood", "sconce", "vase", "traffic light", "tray", "ashcan", "fan",
    "pier", "crt screen", "plate", "monitor", "bulletin board", "shower", "radiator", "glass", "clock", "flag",
]

# [Semantic].accepted_labels (retina_params.h) plus "wall". Wall is the competitor that BEATS a
# door; without its probability a low P(door) cannot be read as "narrowly lost" vs "not seen".
DEFAULT_CLASSES = "wall,cabinet,door,shelf,hood"

OUTPUT_NAME = "class_probs"
TOP_NAME = "top_prob"


def find_logit_tensor(graph):
    """Locate the pre-argmax logit tensor by walking back from the graph output.

    Done structurally rather than by hard-coding "/model.17/classifier/classifier.1/Conv_output_0",
    so a re-exported model with different node names still works. The signature we look for is the
    ArgMax that produces the label map: its input is the logit volume.
    """
    producer = {o: n for n in graph.node for o in n.output}
    cur = graph.output[0].name
    for _ in range(12):
        node = producer.get(cur)
        if node is None:
            break
        if node.op_type == "ArgMax":
            # The Resize between the classifier and the ArgMax is a pure spatial upsample, so we
            # tap ABOVE it: same posteriors, 64x fewer elements.
            src = node.input[0]
            up = producer.get(src)
            if up is not None and up.op_type == "Resize":
                return up.input[0], src
            return src, src
        cur = node.input[0]
    return None, None


def tensor_shape(graph, name):
    for v in list(graph.value_info) + list(graph.output) + list(graph.input):
        if v.name == name:
            return [d.dim_value or d.dim_param for d in v.type.tensor_type.shape.dim]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/yolo26/yolo26l-sem-ade20k.onnx")
    ap.add_argument("--out", default="models/yolo26/yolo26l-sem-ade20k-probs.onnx")
    ap.add_argument("--classes", default=DEFAULT_CLASSES,
                    help="comma-separated ADE20K names to expose (order defines the channel order)")
    ap.add_argument("--fp32", action="store_true", help="emit float32 instead of float16")
    args = ap.parse_args()

    names = [c.strip() for c in args.classes.split(",") if c.strip()]
    try:
        ids = [ADE20K.index(n) for n in names]
    except ValueError as e:
        sys.exit(f"unknown ADE20K class: {e}")

    model = onnx.load(args.model)
    graph = model.graph

    logit, resized = find_logit_tensor(graph)
    if logit is None:
        sys.exit("could not locate the ArgMax / logit tensor - is this a *-sem-* export?")

    shape = tensor_shape(graph, logit)
    print(f"[surgery] model      : {args.model}")
    print(f"[surgery] logit tap  : {logit}  shape={shape}")
    if resized != logit:
        print(f"[surgery]              (tapped ABOVE the Resize to {tensor_shape(graph, resized)})")
    print(f"[surgery] classes    : " + ", ".join(f"{n}={i}" for n, i in zip(names, ids)))

    # Softmax over the CLASS axis, over all 150 - the gather comes after, or these are not posteriors.
    dtype = TensorProto.FLOAT if args.fp32 else TensorProto.FLOAT16
    nodes = [
        helper.make_node("Softmax", [logit], ["probs_all_150"], name="expose_softmax", axis=1),
        helper.make_node("Gather", ["probs_all_150", "expose_class_ids"], ["probs_selected"],
                         name="expose_gather", axis=1),
        helper.make_node("Cast", ["probs_selected"], [OUTPUT_NAME], name="expose_cast", to=dtype),
        # ★TOP_PROB: max over ALL 150 classes = the argmax's own confidence. Without it `scores`
        # can only be honest for the gathered subset, and the dense-int path's 1.0-everywhere lie
        # would survive for every other class. One extra channel, ~13 kB.
        helper.make_node("ReduceMax", ["probs_all_150", "expose_axis1"], ["top_selected"],
                         name="expose_reducemax", keepdims=1),
        helper.make_node("Cast", ["top_selected"], [TOP_NAME], name="expose_top_cast", to=dtype),
    ]

    graph.initializer.append(
        helper.make_tensor("expose_class_ids", TensorProto.INT64, [len(ids)], ids))
    graph.initializer.append(
        helper.make_tensor("expose_axis1", TensorProto.INT64, [1], [1]))
    graph.node.extend(nodes)

    hw = shape[2:] if shape and len(shape) == 4 else ["h", "w"]
    out_shape = [1, len(ids)] + hw
    graph.output.append(helper.make_tensor_value_info(OUTPUT_NAME, dtype, out_shape))
    graph.output.append(helper.make_tensor_value_info(TOP_NAME, dtype, [1, 1] + hw))

    # ★THE MODEL DECLARES ITS OWN CHANNEL MAP. Putting the class ids in a config key would let the
    # two drift silently - a reordered --classes and a stale .toml would mean the consumer reads
    # P(door) out of the cabinet channel, with nothing to catch it. Read back with
    # Ort::ModelMetadata::LookupCustomMetadataMapAllocated("prob_class_ids").
    for k, v in (("prob_class_ids", ",".join(str(i) for i in ids)),
                 ("prob_class_names", ",".join(names))):
        e = model.metadata_props.add(); e.key, e.value = k, v

    onnx.checker.check_model(model, full_check=False)
    onnx.save(model, args.out)

    print(f"[surgery] new outputs: {OUTPUT_NAME} {out_shape} + {TOP_NAME} {[1,1]+hw} "
          f"({'float32' if args.fp32 else 'float16'})")
    print(f"[surgery] declared   : prob_class_ids={','.join(str(i) for i in ids)}")
    print(f"[surgery] wrote      : {args.out}")
    print(f"[surgery] output0 is untouched; the class order above IS the channel order.")


if __name__ == "__main__":
    main()
